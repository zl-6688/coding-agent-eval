"""agent loop —— 项目的常量核心：LLM → 工具调用 → 结果 → 再 LLM，直到收尾。

4 个接缝埋 span（agent.run / agent.turn / llm.call / tool.*），loop 逻辑本身不变——
埋点只在接缝，不侵入业务。这正是"loop 是常量，机制是变量"的体现。

eval 专属参数统一收进 EvalHooks 传入；不传时核心退化成纯线性 loop，无任何 eval 分支。
"""

import copy
import logging
import time
import uuid
from dataclasses import dataclass

import anthropic

from obs.trace import JsonlSink, SpanKind, set_sink, span

from . import config, llm, tools
from .context.attachments import request_attachment_messages
from .context import compact
from .context.project_instructions import ProjectInstructionsLoader
from .context.request_view import build_request_view
from .context.system_prompt import SystemState, build_system
from .mcp.source import load_stdio_mcp_tool_source
from .mcp.session_cache import McpSessionLease
from .runtime.hooks import HookBus, NoOpHookBus
from .runtime.memory_integration import (
    extract_session_memory_after_tools,
    maybe_inject_auto_memory_recall,
    write_auto_memory,
)
from .runtime.messages import (
    ensure_message_uuids,
    new_assistant_message,
    new_user_message,
)
from .runtime.permissions import PermissionEngine
from .runtime.request_context import (
    budget_system as _budget_system,
    invalidate_memory_index_context,
    memory_index_context_message,
    request_context_messages,
)
from .runtime.run_context import RunContext, RunState
from .runtime.run_hooks import (
    append_hook_result_messages as _append_hook_result_messages,
    run_stop_failure_hook,
    run_stop_hook,
    run_user_prompt_submit_hook,
    should_continue_after_stop as _should_continue_after_stop,
    stop_feedback_message as _stop_feedback_message,
)
from .runtime.settings import (
    MemoryRuntimeSettings,
    load_merged_settings,
    memory_runtime_settings_from_settings,
)
from .runtime.tool_messages import (
    close_dangling_tool_uses,
    split_tool_runtime_messages as _split_tool_runtime_messages,
)
from .skills import (
    SkillCatalog,
    discover_skill_catalog,
    reset_invoked_skills,
    restore_invoked_skills_from_messages,
    skill_listing_context_message,
)
from .tasks import drain_task_notifications
from .tools.deferred import DeferredToolPolicy, DeferredToolState, reset_deferred_tool_states
from .tools.pool import ToolPoolContext, assemble_tool_pool
from .tools.request import build_tool_request_view
from .tools.runtime import ToolExecutionRuntime

_log = logging.getLogger(__name__)


# context-overflow 信号词：deepseek/anthropic 撞窗时返 400(BadRequestError),message 含这些词之一。
# WHY 只在 400 处理分支内匹配:正常一轮里的 400 几乎只可能是输入超窗;匹配不中(罕见的请求构造类
#   400,如 tool schema 非法)就 re-raise,让真 bug 浮出来、别被误吞成 context_overflow。
_OVERFLOW_HINTS = ("context", "too long", "too large", "length", "exceed", "maximum", "window")


def _is_context_overflow(exc: Exception) -> bool:
    """判断一个 LLM 异常是否为「输入上下文超窗」。

    full 臂（不压）够长会撞窗 → 把模型层 400 变成可判结果 outcome=context_overflow，别崩 CLI /
    污染整批跑。这是 eval 工程兜底（防 full 崩时丢数据点），非 enablement 头条信号——头条靠
    退化曲线（远早于 crash），不靠 crash 这个硬事件。"""
    msg = str(getattr(exc, "message", "") or exc).lower()
    return any(k in msg for k in _OVERFLOW_HINTS)



def _anthropic_api_error_types() -> tuple[type[BaseException], ...]:
    classes: list[type[BaseException]] = []
    for name in ("APIError", "BadRequestError"):
        cls = getattr(anthropic, name, None)
        if not isinstance(cls, type) or not issubclass(cls, BaseException):
            continue
        if any(issubclass(cls, existing) for existing in classes):
            continue
        classes.append(cls)
    return tuple(classes)

@dataclass
class EvalHooks:
    """eval 专属注入参数，run_task 只有拿到此对象才启用 eval 行为。

    不传（eval_hooks=None）时等价于全默认——核心退化成纯线性 loop，行为不变。

    compact_strategy: 选哪个压缩策略做 A/B 对比（归因各策略各自的净收益）
      none     — 对照组（不压）
      micro    — 只 microcompact（清旧 tool_result，免费、0 模型调用）
      full     — 只 full_compact（LLM 摘要，每次触发 1 次模型调用 + 击穿 cache）
      pipeline — micro→full（忠实 CC 完整流程）
      truncate — 保两端硬截断（compact_naive）= 压缩 eval 的地板对照臂；
                 与 pipeline 把上下文恒定在预算内，差别只在「中段承重」摘要 vs 硬丢，
                 pipeline 赢过它护城河才成立
    compact_window:    eval regime 缩放：override CompactConfig.context_window
    compact_threshold: eval：显式 override 触发线（不给则 auto 计算）
    stop_at_context:   历史生成模式——上下文越过此值就快照 messages 并早退，
                       交给上层做多臂分叉（full/compressed/truncated）
    """
    compact_strategy: str = "none"
    compact_window: int | None = None
    compact_threshold: int | None = None
    stop_at_context: int | None = None
    # Pin agent temperature for A/B eval: keeps memory open/closed as sole variable.
    # None = provider default.  Set to 0.0 for reproducibility in eval runs.
    agent_temperature: float | None = None
    # Optional system identity override for prompt A/Bs. None keeps the product default.
    identity: str | None = None
    # Mechanism-isolation evals may exclude machine-local skill summaries/tools.
    # Product behavior remains enabled by default.
    skills_enabled: bool = True

# SYSTEM 常量已迁移至 agent/context/system_prompt.py（DEFAULT_IDENTITY + build_system）。
# run_task 内按状态动态构建（每 run 一次，build_system 内部每段独立缓存）。


def _final_text(resp) -> str:
    return "".join(
        getattr(b, "text", "") for b in resp.content
        if getattr(b, "type", None) == "text"
    )


def _executor_host_cwd(executor) -> str | None:
    """Return the host project path used for filesystem-only project metadata."""
    host_cwd = getattr(executor, "host_cwd", None)
    if host_cwd is not None:
        return str(host_cwd)
    if getattr(executor, "kind", None) == "docker":
        return None
    return str(executor.cwd)


def _durable_message_target(request_target: int, budget_system: str,
                            clear_at_least: int = 0) -> int:
    """Convert a total request target into a target for durable messages."""
    return max(0, request_target - compact.estimate([], budget_system) - clear_at_least)


def _apply_compaction(messages, eh: "EvalHooks", cfg, trigger, budget_system,
                      session_memory, idle_seconds, skill_agent_id: str | None = None,
                      deferred_tool_state: DeferredToolState | None = None,
                      executor=None, post_compact_sink=None, read_state=None):
    """压缩接缝：按 eval_hooks 里的策略单选一种，发模型前执行。

    逻辑与旧 loop 内的 5-arm if/elif 完全一致，只是提取成独立函数让 turn loop 保持可读。
    返回（可能已压缩的）messages；调用方负责累加 n_compactions。
    """
    strat = eh.compact_strategy
    if strat == "micro":
        return compact.microcompact(
            messages,
            cfg,
            target_tokens=_durable_message_target(
                trigger,
                budget_system,
                clear_at_least=cfg.microcompact_clear_at_least,
            ),
        )
    elif strat == "full":
        return compact.full_compact(
            messages,
            budget_system,
            cfg,
            skill_agent_id=skill_agent_id,
            deferred_tool_state=deferred_tool_state,
            executor=executor,
            post_compact_sink=post_compact_sink,
            read_state=read_state,
            auto_thr=trigger,
        )
    elif strat == "pipeline":
        return compact.compact_pipeline(messages, system=budget_system, cfg=cfg, target_tokens=trigger,
                                        session_memory=session_memory, idle_seconds=idle_seconds,
                                        skill_agent_id=skill_agent_id,
                                        deferred_tool_state=deferred_tool_state,
                                        executor=executor,
                                        post_compact_sink=post_compact_sink,
                                        read_state=read_state)
    elif strat == "truncate":
        # 地板对照臂：保两端硬截断到 trigger 预算（保头承重 + 保尾近端、丢中段）。
        # 与 pipeline 同把上下文恒定在 trigger 内，差别只在「中段承重」摘要 vs 硬丢——
        # 曲线分叉 pipeline−truncate 即「智能压缩 > 蠢截断」的护城河。
        return compact.compact_naive(messages, target_tokens=trigger, system=budget_system)
    return messages


def _compact_boundary_identities(messages: list) -> set[str | None]:
    return {
        compact.message_identity(message)
        for message in messages
        if compact.is_compact_boundary_message(message)
    }


def _has_new_compact_boundary(
    before_identities: set[str | None],
    after: list,
) -> bool:
    """True only when a successful full/SM compact created a new boundary."""

    return any(
        compact.is_compact_boundary_message(message)
        and compact.message_identity(message) not in before_identities
        for message in after
    )



def _append_background_task_notifications(messages: list, run_id: str) -> int:
    notifications = drain_task_notifications(run_id)
    if not notifications:
        return 0
    _append_user_text(messages, "\n\n".join(notifications))
    return len(notifications)


def _append_user_text(messages: list, text: str) -> None:
    block = {"type": "text", "text": text}
    last = messages[-1] if messages else None
    if last is not None and last.get("role") == "user":
        content = last.get("content")
        if isinstance(content, str):
            last["content"] = [
                {"type": "text", "text": content},
                block,
            ]
        elif isinstance(content, list):
            content.append(block)
        else:
            last["content"] = [block]
        return
    messages.append(new_user_message([block]))


def run_task(task: str, max_turns: int = 20, trace: bool = True,
             meta: dict | None = None,
             initial_messages: list | None = None, session_memory=None,
             auto_memory=None,
             return_messages: bool = False, eval_hooks: "EvalHooks | None" = None,
             hook_bus: HookBus | NoOpHookBus | None = None,
             enable_deferred_tools: bool = False,
             enable_mcp: bool = False,
             mcp_config_path: str | None = None,
             mcp_session: McpSessionLease | None = None,
             permission_engine: "PermissionEngine | None" = None,
             memory_settings: "MemoryRuntimeSettings | None" = None,
             tool_result_callback=None):
    """跑一个 coding 任务，返回最终文本总结。

    trace=True 时，本次运行写成一条独立的 trace（.traces/run_<id>.jsonl）。

    ── 可续作扩展（resume / 未来记忆模块用；不传时行为不变）──
      - initial_messages：用它接续而非从 [{user: task}] 重新开始（resume-from-compressed 入口）。
      - return_messages  ：返回 (最终文本 or None, messages 快照) 而非只返回文本。
      - session_memory   ：opt-in 会话笔记（周期性 forked-LLM 写 SM 文件）。
      - auto_memory      ：opt-in 跨会话经验记忆（query loop 终止时调一次 write）。
      - enable_deferred_tools：opt-in 本地 deferred schema selection；开启后 main loop
        首轮只暴露 non-deferred tools + local ToolSearch，选中后下一轮再发送 schema。
      - permission_engine：opt-in 工具权限引擎；REPL 从 ~/.ace / .ace/settings.json 构建后传入；
        不传时退化为空引擎（eval / scripts 默认路径不变）。
    这些参数是"loop 可续作/记忆/动态工具"的最小接缝：不传时退化成原行为。

    eval 专属参数（压缩策略/窗口/触发线/早退点）统一由 eval_hooks 传入；
    不传时等价于全默认（不压缩，纯线性 loop）。
    """
    eh = eval_hooks or EvalHooks()
    run_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    if trace:
        set_sink(JsonlSink(config.TRACES_DIR / f"run_{run_id}.jsonl"))
    # 身份 metadata：调用方（eval 层）传 instance_id/version/mode，loop 不关心内容、只透传上 span。
    # run_id 让每次执行（含重试）在 Phoenix 里可区分。session.id 留给将来的记忆模块（多轮会话）。
    run_meta = {"run_id": run_id, **(meta or {})}
    active_hook_bus = hook_bus or NoOpHookBus()
    run_state = RunState(messages=[])
    run_context = RunContext(
        task=task,
        run_id=run_id,
        run_meta=run_meta,
        return_messages=return_messages,
        state=run_state,
        hook_bus=active_hook_bus,
    )

    # CC 严格预算：窗口给定 → 触发线 auto = (window−20K)−13K（200K→167K）；compact_threshold 可显式 override。
    cfg = compact.CompactConfig(context_window=eh.compact_window) if eh.compact_window else compact.DEFAULT
    trigger = eh.compact_threshold or (compact.auto_threshold(cfg) if eh.compact_strategy != "none" else None)
    on = eh.compact_strategy != "none" and trigger
    tools.reset_todos()
    tools.reset_bash_history()
    tools.reset_file_read_state()
    compact.reset_state()   # 清 _recent_files 等，防跨 run 残留（post-compact 文件恢复用）
    reset_invoked_skills()
    reset_deferred_tool_states()
    # 分段式 system prompt：每 run 构建一次；build_system 内部每段独立缓存，输入不变不重算。
    # workdir 从当前 executor.cwd 取（local=config.WORKDIR，docker=容器内路径）；
    # P0-D correction：system 只收稳定 policy；index 是 AutoMemory 实例持有的
    # conversation snapshot，compact 只让下一 user query 失效。Selector 为显式实验模式。
    executor = tools.get_executor()
    workdir = executor.cwd
    project_workdir = _executor_host_cwd(executor)
    settings_workdir = project_workdir if project_workdir is not None else config.WORKDIR
    resolved_memory_settings = memory_settings or memory_runtime_settings_from_settings(
        load_merged_settings(settings_workdir)
    )
    memory_enabled = bool(auto_memory is not None and resolved_memory_settings.enabled)
    active_auto_memory = auto_memory if memory_enabled else None
    _am_mem_dir = (
        getattr(active_auto_memory, "memory_dir", None)
        if active_auto_memory is not None
        else None
    )
    memory_dir = str(_am_mem_dir) if _am_mem_dir is not None else None
    run_context.memory_dir = memory_dir
    selector_auto_memory = (
        active_auto_memory
        if resolved_memory_settings.recall_mode == "selector"
        else None
    )
    run_context.workdir = workdir
    run_context.project_workdir = project_workdir
    project_profile = (
        ProjectInstructionsLoader().load(project_workdir)
        if project_workdir is not None
        else None
    )
    project_context_message = project_profile.to_context_message() if project_profile is not None else None
    run_context.project_profile = project_profile
    run_context.project_context_message = project_context_message
    if project_profile is not None:
        compact.exclude_post_compact_file(project_profile.path)
    skill_workdir = project_workdir if project_workdir is not None else workdir
    skill_catalog = (
        discover_skill_catalog(skill_workdir)
        if eh.skills_enabled
        else SkillCatalog()
    )
    skill_context_message = skill_listing_context_message(skill_catalog)
    deferred_policy = DeferredToolPolicy(enabled=enable_deferred_tools)
    deferred_state = DeferredToolState.for_agent(run_id)
    run_context.skill_catalog = skill_catalog
    run_context.skill_context_message = skill_context_message
    run_context.deferred_policy = deferred_policy
    run_context.deferred_state = deferred_state
    mcp_tool_definitions = ()
    mcp_session_lease = mcp_session

    def _close_mcp_tool_source() -> None:
        run_context.close_mcp_tool_source(owned=run_context.mcp_source_owned)

    if mcp_session_lease is not None:
        run_context.mcp_tool_source = mcp_session_lease.source
        mcp_tool_definitions = mcp_session_lease.definitions
        run_context.mcp_source_owned = not mcp_session_lease.borrowed
    elif enable_mcp or mcp_config_path is not None:
        run_context.mcp_tool_source = load_stdio_mcp_tool_source(
            workdir=str(skill_workdir),
            config_path=mcp_config_path,
        )
        if run_context.mcp_tool_source is not None:
            try:
                mcp_tool_definitions = run_context.mcp_tool_source.list_tool_definitions()
            except BaseException:
                _close_mcp_tool_source()
                raise
    run_context.mcp_tool_definitions = mcp_tool_definitions
    try:
        permission_engine = permission_engine or PermissionEngine()
        run_context.permission_engine = permission_engine
        tool_pool = assemble_tool_pool(
            ToolPoolContext(
                workdir=str(skill_workdir),
                metadata={"skill_catalog": skill_catalog},
                mcp_tool_definitions=mcp_tool_definitions,
                permission_engine=permission_engine,
                enable_deferred_tools=deferred_policy.enabled,
            )
        )
        run_context.tool_pool = tool_pool
        # Build durable messages before the request view so deferred markers can be restored.
        messages = copy.deepcopy(initial_messages) if initial_messages else [new_user_message(task)]
        ensure_message_uuids(messages)
        run_state.replace_messages(messages)
        if initial_messages:
            restore_invoked_skills_from_messages(messages, agent_id=run_id)
        deferred_state.restore_from_messages(messages)
        initial_tool_request_view = build_tool_request_view(
            tool_pool,
            policy=deferred_policy,
            state=deferred_state,
            messages=messages,
        )
        context_messages = request_context_messages(
            project_context_message,
            skill_context_message,
            initial_tool_request_view.deferred_index_context_message,
            memory_index_context_message(
                active_auto_memory,
                enabled=memory_enabled,
                recall_mode=resolved_memory_settings.recall_mode,
            ),
        )
        run_context.context_messages = context_messages
        file_executor = executor
        if memory_dir is not None:
            try:
                file_executor = tools.bind_memory_file_access(executor, memory_dir)
            except ValueError as exc:
                # Optional memory must fail closed without taking down the main
                # coding task.  Invalid roots receive no extra file capability.
                _log.warning("memory root capability rejected: %s", exc)
        tool_runtime = ToolExecutionRuntime.from_tool_pool(
            tool_pool,
            hook_bus=active_hook_bus,
            run_id=run_id,
            cwd=str(workdir),
            project_context_message=project_context_message,
            agent_id=run_id,
            agent_type="main",
            is_subagent=False,
            executor=file_executor,
            permission_engine=permission_engine,
            tool_result_callback=tool_result_callback,
        )
        run_context.tool_runtime = tool_runtime
        system_state = SystemState(
            tools=initial_tool_request_view.prompt_tools,
            workdir=workdir,
            memory_dir=memory_dir,
            memory_enabled=memory_enabled,
            memory_recall_mode=resolved_memory_settings.recall_mode,
        )
        system = (
            build_system(system_state, identity=eh.identity)
            if eh.identity is not None
            else build_system(system_state)
        )
        run_context.system = system

        budget_system = _budget_system(system, context_messages)
        run_context.budget_system = budget_system

        def _build_current_request_view(*, drain_post_compact: bool = False):
            """Build the final request view in CC order: query context, durable
            transcript, then volatile attachments.

            Budget checks use a non-consuming peek for post-compact restore.
            The actual LLM call drains it once so compact attachments remain
            request-only and never become durable transcript messages.
            """

            post_compact_messages = (
                run_state.drain_post_compact_attachments()
                if drain_post_compact
                else run_state.peek_post_compact_attachments()
            )
            attachment_messages = (
                *request_attachment_messages(
                    tools.get_current_file_read_state(),
                    file_executor,
                ),
                *post_compact_messages,
            )
            return build_request_view(
                messages,
                query_context_messages=context_messages,
                request_attachment_messages=attachment_messages,
            )
    except BaseException:
        _close_mcp_tool_source()
        raise

    def _ret(val):
        ensure_message_uuids(run_state.messages, drop_legacy=True)
        return run_context.finish(val)

    run_state.reset_llm_idle_timer()

    run_attrs = {"task": task[:200],
                 "compact_strategy": eh.compact_strategy,
                 "compact_window": eh.compact_window or 0,
                 "compact_trigger": trigger or 0,
                 "stop_at_context": eh.stop_at_context or 0,
                 "run_metadata": run_meta,
                 "deferred_tools_enabled": deferred_policy.enabled,
                 "mcp_enabled": bool(mcp_session_lease is not None or enable_mcp or mcp_config_path is not None),
                 "mcp_server_count": len(run_context.mcp_tool_source.configs) if run_context.mcp_tool_source is not None else 0,
                 "mcp_tool_count": len(mcp_tool_definitions),
                 "project_instructions_loaded": project_profile is not None,
                 "skills_enabled": eh.skills_enabled,
                 "skill_count": len(skill_catalog.skills),
                 "tool_pool_fingerprint": tool_pool.fingerprint,
                 "tool_pool_tool_count": len(tool_pool.tools)}
    if mcp_session_lease is not None:
        run_attrs.update({
            "mcp.borrowed": mcp_session_lease.borrowed,
            "mcp.cache_hit": mcp_session_lease.cache_hit,
            "mcp.cache_key": mcp_session_lease.cache_key_summary,
            "mcp.cached": mcp_session_lease.cache_hit,
        })
    if project_profile is not None:
        run_attrs.update({
            "project_instructions_path": project_profile.relpath,
            "project_instructions_fingerprint": project_profile.fingerprint,
            "project_instructions_truncated": project_profile.truncated,
        })
    run_context.run_attrs = run_attrs

    with span("agent.run", SpanKind.AGENT, **run_attrs) as run_sp:
      prompt_hook = run_user_prompt_submit_hook(run_context, messages, _log)
      _append_hook_result_messages(messages, prompt_hook)
      if prompt_hook.blocking_error or prompt_hook.prevent_continuation:
          reason = prompt_hook.blocking_error or prompt_hook.stop_reason or "prevented by hook"
          run_sp.set(
              finished=False,
              turns=0,
              outcome="user_prompt_blocked",
              stop_reason=reason,
              hook_event="UserPromptSubmit",
          )
          return _ret(f"UserPromptSubmitBlocked: {reason}")

      # Ctrl+C 中止：run_task 阻塞在主线程，KeyboardInterrupt 天然打断（含阻塞的 LLM 调用）。
      # 捕获后不让异常逃逸：清理悬空 tool_use（保证续作 API 合法），经 return_messages 返回部分 messages。
      try:
        while True:
          while run_state.turn_no < max_turns:
            with span("agent.turn", SpanKind.AGENT) as turn_sp:
                ensure_message_uuids(messages)
                notification_count = _append_background_task_notifications(messages, run_id)
                if notification_count:
                    turn_sp.set(background_task_notifications=notification_count)

                # ── 压缩接缝（真实 loop 内，发模型前；按策略单选，便于 A/B 归因）──
                request_view = _build_current_request_view()
                ctx_for_gate = request_view.estimate_tokens(system)
                if on and ctx_for_gate > trigger:
                    before_compaction = _compact_boundary_identities(messages)
                    messages = _apply_compaction(messages, eh, cfg, trigger, budget_system,
                                                 session_memory, time.time() - run_state.last_llm_ts,
                                                 skill_agent_id=run_id,
                                                 deferred_tool_state=deferred_state,
                                                 executor=file_executor,
                                                 post_compact_sink=run_state,
                                                 read_state=tools.get_current_file_read_state())
                    run_state.replace_messages(messages)
                    run_state.increment_compactions()
                    if _has_new_compact_boundary(before_compaction, messages):
                        invalidate_memory_index_context(active_auto_memory)
                    request_view = _build_current_request_view()
                    ctx_for_gate = request_view.estimate_tokens(system)

                # 历史生成模式（eval 用）：上下文越过快照点就停，交给上层做多臂分叉。
                #   放在压缩接缝之后、发模型之前——此刻 messages 以 user/tool_result 收尾，是合法续作点。
                if eh.stop_at_context is not None and ctx_for_gate > eh.stop_at_context:
                    snap_ctx = ctx_for_gate
                    run_sp.set(n_compactions=run_state.n_compactions, turns=run_state.turn_no, finished=False,
                               outcome="snapshot_cut", peak_context_tokens=max(run_state.peak_context, snap_ctx))
                    return _ret(None)

                run_state.next_turn()

                maybe_inject_auto_memory_recall(
                    messages,
                    auto_memory=selector_auto_memory,
                    task=task,
                    run_state=run_state,
                    logger=_log,
                )

                # Recall mutates the durable request tail. Enforce budget again here
                # and report the final pre-LLM request size, including request-only context.
                request_view = _build_current_request_view()
                ctx_now = request_view.estimate_tokens(system)
                if on and ctx_now > trigger:
                    before_compaction = _compact_boundary_identities(messages)
                    messages = _apply_compaction(messages, eh, cfg, trigger, budget_system,
                                                 session_memory, time.time() - run_state.last_llm_ts,
                                                 skill_agent_id=run_id,
                                                 deferred_tool_state=deferred_state,
                                                 executor=file_executor,
                                                 post_compact_sink=run_state,
                                                 read_state=tools.get_current_file_read_state())
                    run_state.replace_messages(messages)
                    run_state.increment_compactions()
                    if _has_new_compact_boundary(before_compaction, messages):
                        invalidate_memory_index_context(active_auto_memory)
                    request_view = _build_current_request_view()
                    ctx_now = request_view.estimate_tokens(system)

                if eh.stop_at_context is not None and ctx_now > eh.stop_at_context:
                    run_sp.set(n_compactions=run_state.n_compactions, turns=run_state.turn_no, finished=False,
                               outcome="snapshot_cut", peak_context_tokens=max(run_state.peak_context, ctx_now))
                    turn_sp.set(turn_index=run_state.turn_no, context_tokens=ctx_now,
                                compacted_this_turn=(run_state.n_compactions if on else 0),
                                outcome="snapshot_cut")
                    return _ret(None)

                run_state.record_context_size(ctx_now)
                # Turn Timeline：每轮记 index + 发模型前的上下文大小（看"哪轮上下文炸/在哪轮跑偏"）
                turn_sp.set(turn_index=run_state.turn_no, context_tokens=ctx_now,
                            compacted_this_turn=(run_state.n_compactions if on else 0))

                tool_request_view = build_tool_request_view(
                    tool_pool,
                    policy=deferred_policy,
                    state=deferred_state,
                    messages=messages,
                )
                request_view = _build_current_request_view(drain_post_compact=True)
                ctx_now = request_view.estimate_tokens(system)
                run_state.record_context_size(ctx_now)
                turn_sp.set(
                    context_tokens=ctx_now,
                    durable_messages=request_view.durable_count,
                    query_context_messages=request_view.context_count,
                    request_attachments=request_view.attachment_count,
                )
                request_messages = request_view.as_messages()
                try:
                    resp = llm.chat(request_messages, system=system,
                                    tools=tool_request_view.schemas, max_tokens=4096,
                                    temperature=eh.agent_temperature)
                    if resp.stop_reason == "max_tokens":
                        # 输出被截断 ≠ 任务完成：丢弃半截回复、提高预算重试本轮，
                        # 避免一次过长输出（如 dump 大文件）就报废整个 run。
                        resp = llm.chat(request_messages, system=system,
                                        tools=tool_request_view.schemas, max_tokens=8192,
                                        temperature=eh.agent_temperature)
                except _anthropic_api_error_types() as e:
                    # eval 工程兜底：链够长时 full 臂（不压）会撞窗返 400。
                    # 把它变成可判数据点 outcome=context_overflow 后优雅返回，别崩 CLI / 污染整批跑。
                    # 与紧邻的 max_tokens 处理同款思路：把模型层信号变成可判结果。
                    # 非超窗的 400（请求构造类）不吞、re-raise，让真 bug 浮出来。
                    if not _is_context_overflow(e):
                        stop_failure_hook = run_stop_failure_hook(run_context, messages, e, _log)
                        _append_hook_result_messages(messages, stop_failure_hook)
                        raise
                    run_sp.set(n_compactions=run_state.n_compactions, turns=run_state.turn_no, finished=False,
                               stop_reason="context_overflow", outcome="context_overflow",
                               peak_context_tokens=run_state.peak_context,
                               compaction_triggered=(run_state.n_compactions > 0),
                               n_tool_errors=tools.tool_error_count())
                    return _ret("[上下文超窗 context_overflow]")
                run_state.mark_llm_completed()
                messages.append(new_assistant_message(resp.content))
                turn_sp.set(stop_reason=resp.stop_reason or "")

                if resp.stop_reason != "tool_use":
                    # 自然收尾（end_turn）：记结果摘要供 Phoenix 直接筛 trace 成败/收敛
                    final = _final_text(resp)
                    stop_hook = run_stop_hook(
                        run_context,
                        messages,
                        "finished",
                        final,
                        resp.stop_reason or "",
                        logger=_log,
                    )
                    _append_hook_result_messages(messages, stop_hook)
                    if _should_continue_after_stop(stop_hook) and run_state.allow_stop_continuation():
                        max_turns += 1
                        messages.append(_stop_feedback_message(stop_hook))
                        turn_sp.set(tools_used=[], outcome="stop_hook_continue")
                        continue
                    turn_sp.set(tools_used=[], outcome="finished")
                    run_sp.set(n_compactions=run_state.n_compactions, turns=run_state.turn_no, finished=True,
                               stop_reason=resp.stop_reason or "",
                               outcome="finished", output_value=final[:1000],
                               peak_context_tokens=run_state.peak_context,
                               compaction_triggered=(run_state.n_compactions > 0),
                               n_tool_errors=tools.tool_error_count())
                    # ── Auto Memory 接缝（opt-in）：query loop 终止（无-tool 终态）后写一次跨会话记忆。
                    #   对齐 CC「runs once at the end of each complete query loop」（`extractMemories.ts:5-6`）。
                    #   ⚠ 偏离：CC fire-and-forget（不阻塞主线）；我们**同步**调用（eval 场景够用）。
                    #   TODO：真接交互式 loop 时改异步（在 run_sp 落盘前 fire、不 await）。
                    #   try/except 包住：记忆写入失败绝不影响主任务（fire-and-forget best-effort 语义）。
                    write_auto_memory(active_auto_memory, messages, system, _log)
                    return _ret(final)

                results, tools_used = tool_runtime.execute_tool_uses(resp.content)
                # 本轮用了哪些工具 —— Turn Timeline 一眼看出"20 轮都在 bash 没 edit"这类病
                turn_sp.set(tools_used=tools_used, n_tool_calls=len(tools_used))
                result_blocks, durable_tool_messages = _split_tool_runtime_messages(results)
                messages.append(new_user_message(result_blocks))
                ensure_message_uuids(durable_tool_messages)
                messages.extend(durable_tool_messages)
                # ── Session Memory 接缝（opt-in，step1 loop 接入）：周期性 forked-LLM 写会话笔记。
                #   should_extract 三阈值控频率（不会每轮跑）；fork 工具指标已隔离（不污染主 run）。
                #   ⚠ 偏离：CC 在 postSamplingHook 每次采样后 check（含无-tool 的自然断点）；我们只在
                #   tool 处理后接入 → should_extract 的 no_tools_last 分支在此不触发，只走 enough_tools。
                extract_session_memory_after_tools(session_memory, messages, system)

          # 打满轮次未收尾：记 outcome 供 Phoenix 一眼区分"自然收尾 vs 耗尽轮次"
          stop_hook = run_stop_hook(
              run_context,
              messages,
              "max_turns_reached",
              "",
              "max_turns",
              logger=_log,
          )
          _append_hook_result_messages(messages, stop_hook)
          if _should_continue_after_stop(stop_hook) and run_state.allow_stop_continuation():
              max_turns += 1
              messages.append(_stop_feedback_message(stop_hook))
              continue
          run_sp.set(n_compactions=run_state.n_compactions, turns=run_state.turn_no, finished=False,
                     stop_reason="max_turns", outcome="max_turns_reached",
                     peak_context_tokens=run_state.peak_context,
                     compaction_triggered=(run_state.n_compactions > 0),
                     n_tool_errors=tools.tool_error_count())
          return _ret("(达到最大轮次，未收尾)")
      except KeyboardInterrupt:
        # 用户 Ctrl+C 中止。清理悬空 tool_use（保证续作 API 合法）→ 落盘的部分对话可续。
        cleaned = close_dangling_tool_uses(messages)
        stop_hook = run_stop_hook(
            run_context,
            messages,
            "interrupted",
            "",
            "interrupted",
            logger=_log,
        )
        _append_hook_result_messages(messages, stop_hook)
        run_sp.set(n_compactions=run_state.n_compactions, turns=run_state.turn_no, finished=False,
                   stop_reason="interrupted", outcome="interrupted",
                   peak_context_tokens=run_state.peak_context,
                   interrupt_cleaned_dangling=cleaned,
                   compaction_triggered=(run_state.n_compactions > 0),
                   n_tool_errors=tools.tool_error_count())
        return _ret("[已中止]")
      except BaseException:
        _close_mcp_tool_source()
        raise
