"""forked-agent 基础设施 —— 对齐 CC `runForkedAgent` 的语义。

记忆系统的写/检索/整理层（session memory / auto memory / dream）都是"分叉一个子 agent"：
读主对话历史 + 一条指令，在**锁定的工具权限**下干活（如只能 edit 记忆目录），不碰主 loop 状态。
这是 CC 记忆「LLM 管判断、代码管保证」哲学的载体——判断交子 agent，但触发/权限/落盘由确定性代码兜底。

对齐 CC（`utils/forkedAgent.ts` + `extractMemories.ts` createAutoMemCanUseTool）：
  - 独立子 loop，读主对话历史；非空能力集保留稳定的本地 tool schema 前缀，执行时再用
    role allowlist/tool_filter + executor path capability 两层拦截。CC 本身复用 parent
    cache-safe params；ACE 尚未复用 parent MCP/deferred/REPL pool，这是明确的剩余偏离。
  - 收集子 agent 写过的文件路径（对齐 `extractWrittenPaths`）——记忆模块要知道 fork 落了哪些盘。

说明 / 偏离 CC（诚实标注）：
  1. **成本归因（缓存）**：fork 复制主对话作前缀 → deepseek 的自动前缀硬盘缓存大概率命中该前缀
     （best-effort，不保证 100%），故 fork 多数输入按缓存命中计费、非全量；新增指令 + 输出按未命中
     计费。成本归 memory.fork span（usage 的 prompt_cache_hit/miss 区分）。与 CC 共享主对话 cache 殊途同归。
  2. **同步跑**：CC 后台不阻塞主线（交互式 UI 需要）；我们 eval 场景**同步**跑（调用方等它完成），
     简化、够用。真接交互式 loop 时再改异步。
"""

import copy
from dataclasses import dataclass, field

from obs.trace import SpanKind, span

from .. import llm
from ..context.request_view import build_request_view
from ..tools.file_state import FileReadState
from ..tools.pool import ToolPoolContext, assemble_tool_pool
from ..tools.runtime import ToolExecutionRuntime

# 写类工具：执行后记录 written_paths（对齐 CC extractWrittenPaths 只认 Edit/Write）
_WRITE_TOOLS = {"write_file", "edit_file"}


@dataclass
class ForkResult:
    final_text: str = ""                                   # 子 agent 自然收尾时的文本
    written_paths: list = field(default_factory=list)      # edit/write 碰过的文件
    turns: int = 0
    input_tokens: int = 0                                  # cumulative input tokens (cost attribution); fork 的主对话前缀多由 deepseek 自动缓存命中（usage 里 prompt_cache_hit 反映）
    output_tokens: int = 0
    stopped: str = ""                                      # finished / max_turns


def _usage(resp):
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0
    return (getattr(u, "input_tokens", 0) or 0,
            getattr(u, "output_tokens", 0) or 0)


def run_forked_agent(prompt, context_messages, *, system="", allowed_tools,
                     tool_filter=None, max_turns=5, max_tokens=4096, label="fork",
                     executor=None):
    """跑一个受限子 agent：读 context_messages + prompt，只用 allowed_tools 干活，返回 ForkResult。

    allowed_tools: 执行期 role allowlist。非空时不裁剪本地 tool schema，避免角色差异破坏
      fork 前缀缓存；空集仅为待 P1-A 迁移的 legacy SessionMemory 保留无工具契约。
    tool_filter(name, input) -> (ok: bool, deny_msg: str)：执行期角色/路径限制；最终文件边界
      仍由 context-bound executor fail closed。None 表示 allowlist 内一律放行。
    max_tokens: 每次 llm.chat 的输出上限（默认 4096；会话记忆传 ≥12K 以容纳整份笔记）。
    label: span/成本归因标签（session_memory / auto_memory / dream / ...）。
    """
    # CC keeps the fork's tool schema prefix stable and applies role-specific
    # restrictions in canUseTool.  ACE cannot yet reuse the exact parent pool,
    # but retaining the complete local pool avoids changing the cache key for
    # every memory role merely because its execution capability is narrower.
    # The legacy SessionMemory path still asks for an explicitly empty set and
    # performs a harness write after text generation.  Preserve that temporary
    # no-tool contract until P1-A migrates it to CC's exact-file Edit flow.
    pool_context = (
        ToolPoolContext()
        if allowed_tools
        else ToolPoolContext(include_tool_names=frozenset())
    )
    tool_pool = assemble_tool_pool(pool_context)
    tool_runtime = ToolExecutionRuntime.from_tool_pool(
        tool_pool,
        agent_type=f"memory_{label}",
        is_subagent=True,
        # Forked agents read files for memory maintenance, not for the parent
        # turn. Keep their read snapshots out of parent compact restore and
        # parent write/edit guards.
        file_state=FileReadState(),
        executor=executor,
    )
    fork_tools = tool_pool.model_schemas_for_api()
    messages = copy.deepcopy(context_messages) + [{"role": "user", "content": prompt}]
    res = ForkResult()

    with span("memory.fork", SpanKind.AGENT, **{"fork.label": label,
                                                "fork.allowed_tools": sorted(allowed_tools),
                                                "fork.max_turns": max_turns}) as sp:
        for _ in range(max_turns):
            request_messages = build_request_view(messages).as_messages()
            resp = llm.chat(request_messages, system=system, tools=fork_tools,
                            max_tokens=max_tokens, purpose=f"memory_{label}")
            ti, to = _usage(resp)
            res.input_tokens += ti
            res.output_tokens += to
            res.turns += 1
            messages.append({"role": "assistant", "content": resp.content})

            if getattr(resp, "stop_reason", None) != "tool_use":
                res.final_text = "".join(getattr(b, "text", "") for b in resp.content
                                         if getattr(b, "type", None) == "text")
                res.stopped = "finished"
                break

            results = []
            for b in resp.content:
                if getattr(b, "type", None) != "tool_use":
                    continue
                name, inp = b.name, b.input
                # 两层角色限制：① allowlist ② tool_filter；文件系统还有 executor 最终边界。
                ok, deny = True, ""
                if name not in allowed_tools:
                    ok, deny = False, f"工具 {name} 在此上下文被禁用（仅允许 {sorted(allowed_tools)}）"
                elif tool_filter is not None:
                    ok, deny = tool_filter(name, inp)
                if ok:
                    tool_messages, _ = tool_runtime.execute_tool_uses([b])
                    execution = (
                        tool_runtime.last_results[-1]
                        if tool_runtime.last_results
                        else None
                    )
                    if (
                        name in _WRITE_TOOLS
                        and isinstance(inp, dict)
                        and inp.get("path")
                        and _write_execution_succeeded(execution)
                    ):
                        res.written_paths.append(str(inp["path"]))
                    results.extend(tool_messages)
                else:
                    output = f"[拒绝] {deny}"
                if not ok:
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": output})
            messages.append({"role": "user", "content": results})
        else:
            res.stopped = "max_turns"
            # max_turns exhausted: still capture the last assistant text (needed when auto-memory/dream reuse this infrastructure)
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    res.final_text = "".join(getattr(b, "text", "") for b in m.get("content", [])
                                             if getattr(b, "type", None) == "text")
                    break

        sp.set(**{"fork.turns": res.turns, "fork.written_paths": res.written_paths,
                  "fork.input_tokens": res.input_tokens, "fork.output_tokens": res.output_tokens,
                  "fork.stopped": res.stopped})
    return res


def _write_execution_succeeded(result) -> bool:
    if result is None or getattr(result, "is_error", False):
        return False
    for message in getattr(result, "messages", ()):
        if message.get("is_error"):
            return False
        if str(message.get("content", "")).lstrip().startswith("Error:"):
            return False
    return True
