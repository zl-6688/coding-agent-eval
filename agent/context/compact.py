"""上下文压缩 —— 部分机制参考 Claude Code 公开技术资料，并结合本项目评估迭代。

真实 CC = 三套相关但职责分开的机制:
  1. **microcompact / context editing** —— 清理**旧 tool_result**(保留最近 N 个,换占位符),
     **从不碰 user 消息**。对应官方 `clear_tool_uses_20250919`(beta context-management-2025-06-27,
     参数 keep / trigger / clear_at_least / clear_tool_inputs)。
  2. **full_compact / compaction** —— 把完整 durable messages 后追加 compact summary request,
     交给 LLM 做 **9 段摘要**(<analysis>/<summary>,剥离 analysis)+ post-compact 文件/skill 恢复；
     CC full 本身不返回 messagesToKeep,不保留近端原文。
     对应官方 `compact_20260112`(beta compact-2026-01-12,默认 150K 触发)。
  3. **session-memory compact** —— 用预写 session memory 当零新 LLM 摘要,并预算感知保留近端
     (min 10K token / 5 文本消息,上限 40K,不拆 tool 配对)。本项目已有
     `session_memory_compact()`,并在 `compact_pipeline()` 里先于 full fallback 尝试。

窗口感知预算:effective = window − output_reserve(20K);auto-compact 阈值 = effective − buffer(13K)。
当前模型窗口多为 **1M**(Opus/Sonnet/Fable),Haiku 200K。

⚠ s09(教学版)的 **L1 snip**(裁中间、含 user 消息)是 s09 私货,真实 CC **没有** —— 已移除。

────────────────────────────────────────────────────────────────────
适配本系统的取舍(不是照搬 CC 生产实现,而是在"对度量重要"处有深度):
  - 模型是 **deepseek-v4(非 Claude)**,窗口实测 ≥84K、确值未知。窗口感知阈值
    (1M/20K/13K)是 **Claude 派生常数,仅未来接真实 loop 时用**;**eval 走 `target_tokens`
    显式预算驱动**(`compact_pipeline(..., target_tokens=B)`)。
  - microcompact 用 **token 预算驱动**(清到预算下、保最近 keep 个)= 官方 clear_at_least 语义。
  - 默认数字(micro keep、10–40K、min 5 条)镜像 CC 比例。
  - 摘要 prompt 是中文适配版(非 CC 原版),需在 deepseek 上验证它真能逐条保住散落事实。
  - 本项目 `full_compact()` 已回归 CC full=0：摘要输入是完整 durable messages + 追加请求,
    输出不再拼近端原文；近端保留只归 `session_memory_compact()`。
  - 摘要请求自身 prompt-too-long 时，仅在 full_compact() 的 LLM 调用内部做 PTL retry；
    这是异常逃生，不是常规 head/tail 压缩策略。

刻意**不实现**(列出以示取舍而非疏漏 —— 都与"度量"无关或属 Claude-API/生产基建专属):
  - 官方 `cache_edits` / server-side `context_management` API 层清理(保 prompt cache)——
    Claude-API 专属;我们走 OpenAI 兼容 proxy,且 eval 把上下文 flatten 成文本,缓存不在场。
  - FileStateCache、postCompactCleanup、4 级告警(Warning/Error/Blocking)、冷/热双路
    microcompact —— 生产基建,与度量无关。
"""

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from obs.trace import SpanKind, span

from .. import config as _cfg
from .. import llm
from ..runtime.messages import (
    LEGACY_MESSAGE_ID_KEY,
    ensure_message_uuids,
    message_matches_identity,
    message_uuid,
    new_message_uuid,
    new_user_message,
)
from .attachments import post_compact_file_attachment_text
from ..skills.state import invoked_skill_context_message
from ..tools.deferred import selected_deferred_tools_marker_message
from ..tools.messages import to_api_message


# ──────────────────────────────────────────────
# 配置(窗口感知;realistic 默认 1M 基准 + eval 可缩放)
# ──────────────────────────────────────────────

# Aligns CC full compact output budget; tests can still lower it per cfg.
DEFAULT_SUMMARY_MAX_TOKENS = 20_000
CONTEXT_OVERFLOW_SAFETY_BUFFER_TOKENS = 1_000
MIN_COMPACT_OUTPUT_TOKENS = 3_000


def _positive_int(value) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _configured_model_max_output_tokens() -> int:
    return _positive_int(getattr(_cfg, "MODEL_MAX_OUTPUT_TOKENS", None)) or DEFAULT_SUMMARY_MAX_TOKENS


def _effective_summary_max_tokens(requested: int | None) -> int:
    requested_value = _positive_int(requested) or DEFAULT_SUMMARY_MAX_TOKENS
    return min(requested_value, _configured_model_max_output_tokens())


@dataclass
class CompactConfig:
    # 窗口预算 —— 仅"接真实 loop"时用;eval 走显式 target_tokens(见 compact_pipeline)
    # 200K = 多数提供商默认上限(deepseek-v4 / Claude 经典 / GPT-4 等)。
    context_window: int = 200_000
    # ★ output_reserve = 给模型回复预留的 token。CC: reserve = min(maxOutputTokens, 20_000)
    #   (autoCompact.ts:34)。maxOutputTokens 是否被 cap 到 8K 取决于 feature flag tengu_otk_slot_v1:
    #     - 三方默认(claude.ts:3395 "3P default: false")→ flag 关 → 原生 32K/64K → reserve=min(.,20K)=20K → auto=167K
    #     - Anthropic 内部(flag 开)→ cap 到 8K → reserve=8K → auto=179K
    #   我们是三方场景 → 用 20K → auto=200K−20K−13K=**167K**。
    #   (订正:之前误用内部值 179K,核到 flag 才发现三方默认是 167K。见 swe-bench-journey §7.6。)
    output_reserve: int = 20_000
    compact_buffer: int = 13_000           # AUTOCOMPACT_BUFFER_TOKENS
    # microcompact:始终保留最近 keep 个 tool_result(CC keep),更旧的按 token 预算精确清
    # ★ keep=6 时一次清 158 个 tool_result（167K→71K），agent 失忆导致回归（2026-06-26 诊断）。
    #   调到 15 = 约 7 个工具轮次，覆盖编码任务近端工作窗口。
    #   ⚠ 15 是基于单次诊断的估计，非 CC 官方来源；需 eval 验证方向后再细调。
    microcompact_keep: int = 15
    # CC clear_at_least:触发时至少清这么多 token(清一大块到精确预算、避免每轮抖动)。
    # ★ 30K→10K（2026-06-26 诊断）。旧值 30K 在 micro_thr=150K 时把目标压到 140K，每次清 ~27K；
    #   原 30K 在 100K 时目标=70K，一次清 ~97K（过度）。10K 给"防抖动"缓冲已够，不再大砍。
    #   ⚠ 基于单次诊断的估计；待 eval 验证。
    microcompact_clear_at_least: int = 10_000
    # ★ microcompact **独立**触发线。CC 设计精髓:micro **早于、独立于** full_compact —— 它便宜
    #   (只删旧 tool_result、不调 LLM),所以可早触发。但触发线该多高,有两点要诚实说明:
    #   ① **150K 这个具体数字无官方来源** —— 公开技术资料未给出 micro 的确切 trigger;
    #      e1 诊断前取 100K（"早于 full"的拍脑袋近似），e1 结果显示 100K 过早、一次清 97K
    #      造成失忆性回归；调到 150K（离 full=167K 只差 17K），减少过早清除；
    #      真实合理点仍需 eval 迭代（现在的 150K 是方向性调参，非拍准值）。
    #   ② **CC 真实 micro 主要是缓存驱动(time-based 冷/热),非 token 阈值** —— token 触发只是
    #      API context_management(clear_tool_uses)那条;我们走代理无 cache_edits,只能用 token 近似。
    #   旧实现让 micro 与 full 共用 167K → micro 永不独立触发(实现缺陷,2026-06-22 修正)。
    microcompact_trigger: int = 150_000
    # ★ microcompact 缓存冷门控(2026-06-27;实测校验了参考设计的 time-based 路径)：
    #   CC 真实 micro 主路径 = 纯时间门控、无 token 阈值:gap=(now-lastAssistant.ts)/60_000,
    #   gap ≥ gapThresholdMinutes(默认 **60**,timeBasedMCConfig.ts:32) 才清旧 tool_result。
    #   WHY 60min(CC 注释原话):"60 is the safe choice: the server's 1h cache TTL is guaranteed expired,
    #   so we never force a miss that wouldn't have happened"——缓存已冷、prefix 反正要重写,此时清才「免费」。
    #   ⚠ 偏离标注:CC 缓存**温热**时不跳过,而走 Path B「cached MC」用 cache-editing API(clear_tool_uses)
    #   外科式删 tool_result 且不破热缓存;我们走 deepseek 代理**无 cache_edits**、复刻不了 Path B → 只能
    #   缓存**冷**时清(Path A)、温热时**整段跳过**(否则 content-clear 破热缓存,实测计费 input 3.5×
    #   暴涨,见 finding-microcompact-breaks-cache)。idle_seconds=None(未接 loop)视为冷→保旧行为(兼容)。
    cache_cold_seconds: int = 3600        # = CC gapThresholdMinutes 60min;连续任务轮间空闲≈秒→micro 基本不触发→full 接管
    # session-memory compact 的近端保留预算；full_compact 已回归 CC full=0,不再使用它。
    keep_min_tokens: int = 10_000
    keep_min_msgs: int = 5
    keep_max_tokens: int = 40_000
    summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS
    auto_max_failures: int = 3
    persist_dir: Path = None               # 兼容字段(当前不落盘)
    # post-compact 文件恢复（对齐 CC：POST_COMPACT_MAX_FILES_TO_RESTORE=5 / 5K/文件 / 50K 总）
    post_compact_max_files: int = 5
    post_compact_max_tokens_per_file: int = 5_000
    post_compact_token_budget: int = 50_000
    post_compact_max_tokens_per_skill: int = 5_000
    post_compact_skills_token_budget: int = 25_000


DEFAULT = CompactConfig()


# ⚠ 以下窗口感知触发线仅供"未来接真实 loop"用;eval 不走这里(走 target_tokens 显式预算)
def effective_window(cfg: "CompactConfig" = DEFAULT) -> int:
    return cfg.context_window - cfg.output_reserve


def auto_threshold(cfg: "CompactConfig" = DEFAULT) -> int:
    return effective_window(cfg) - cfg.compact_buffer


def micro_threshold(cfg: "CompactConfig" = DEFAULT, auto_thr: int = None) -> int:
    """microcompact 的独立触发线 = min(cfg.microcompact_trigger, auto_thr)。
    CC 设计:micro 早于 full;但永远不晚于 full(否则没意义)。eval 缩放时也跟着缩。"""
    auto_thr = auto_thr if auto_thr is not None else auto_threshold(cfg)
    return min(cfg.microcompact_trigger, auto_thr)


_recent_files: dict = {}     # 工具层 track_file 填充，full_compact 后回贴
_post_compact_excluded_paths: set[str] = set()
# Legacy/test-only fallback. The run_task main path passes RunState as the
# post_compact_sink so request-only post-compact context stays per-run.
_pending_post_compact_attachments: list[dict] = []
_circuit_breaker = 0
MAX_PTL_RETRIES = 3
PTL_RETRY_MARKER = "[earlier conversation truncated for compaction retry]"
COMPACT_BOUNDARY_TYPE = "system"
COMPACT_BOUNDARY_SUBTYPE = "compact_boundary"
COMPACT_BOUNDARY_CONTENT = "Conversation compacted"
RUNTIME_MESSAGE_ID_KEY = LEGACY_MESSAGE_ID_KEY


def create_compact_boundary_message(
    *,
    trigger: str,
    pre_tokens: int,
    user_context: str = "",
    messages_summarized: int = 0,
    logical_parent_uuid: str | None = None,
) -> dict:
    compact_metadata = {
        "trigger": trigger,
        "preTokens": pre_tokens,
        "userContext": user_context,
        "messagesSummarized": messages_summarized,
    }
    message = {
        "type": COMPACT_BOUNDARY_TYPE,
        "subtype": COMPACT_BOUNDARY_SUBTYPE,
        "content": COMPACT_BOUNDARY_CONTENT,
        "uuid": new_message_uuid(),
        "compactMetadata": compact_metadata,
    }
    if logical_parent_uuid:
        message["logicalParentUuid"] = logical_parent_uuid
    return message


def is_compact_boundary_message(message) -> bool:
    return (
        isinstance(message, dict)
        and message.get("type") == COMPACT_BOUNDARY_TYPE
        and message.get("subtype") == COMPACT_BOUNDARY_SUBTYPE
    )


def find_last_compact_boundary_index(messages) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if is_compact_boundary_message(messages[idx]):
            return idx
    return -1


def messages_after_compact_boundary(messages) -> list:
    idx = find_last_compact_boundary_index(messages)
    tail = messages[idx + 1:] if idx >= 0 else messages
    return [message for message in tail if not is_compact_boundary_message(message)]


def message_identity(message) -> str | None:
    """Return the stable durable-message UUID, with legacy fallback."""

    return message_uuid(message)


def ensure_runtime_message_ids(messages) -> None:
    """Compatibility wrapper for callers predating durable UUIDs."""

    ensure_message_uuids(messages)


def create_compact_summary_message(content: str, *, source: str) -> dict:
    metadata = {
        "compactSummarySource": source,
    }
    return new_user_message(
        content,
        isCompactSummary=True,
        isVisibleInTranscriptOnly=True,
        metadata=metadata,
    )


def is_compact_summary_message(message) -> bool:
    if not isinstance(message, dict):
        return False
    if message.get("isCompactSummary") is True:
        return True
    if message.get("is_compact_summary") is True:
        return True
    metadata = message.get("metadata")
    return (
        isinstance(metadata, dict)
        and (
            metadata.get("isCompactSummary") is True
            or metadata.get("is_compact_summary") is True
        )
    )


def _compact_api_messages(messages) -> list[dict]:
    api_messages = []
    for message in messages:
        if is_compact_boundary_message(message):
            continue
        rendered_memories = _render_relevant_memories(message)
        if rendered_memories is not None:
            api_messages.extend(rendered_memories)
            continue
        if "role" in message and "content" in message:
            api_messages.append(to_api_message(message))
        else:
            api_messages.append(copy.deepcopy(message))
    return api_messages


def _render_relevant_memories(message) -> list[dict] | None:
    # Keep the typed-memory renderer lazy: memory.__init__ currently imports
    # forked_agent, which imports request_view and would cycle at module load.
    from ..memory.relevant import render_relevant_memories_message

    return render_relevant_memories_message(message)


def reset_state():
    global _circuit_breaker
    _recent_files.clear()
    _post_compact_excluded_paths.clear()
    _pending_post_compact_attachments.clear()
    _circuit_breaker = 0


def drain_post_compact_attachments() -> tuple[dict, ...]:
    """Return and clear legacy/test-only post-compact attachments."""

    pending = tuple(copy.deepcopy(_pending_post_compact_attachments))
    _pending_post_compact_attachments.clear()
    return pending


def peek_post_compact_attachments() -> tuple[dict, ...]:
    """Return legacy/test-only post-compact attachments without consuming them.

    The loop main path should use RunState instead.  This compatibility lane
    exists for direct compact.py tests and isolated helper calls.
    """

    return tuple(copy.deepcopy(_pending_post_compact_attachments))


def exclude_post_compact_file(path: str | Path) -> None:
    """Register a file path that must not be restored after compact.

    Project instructions such as AGENTS.md are already injected as query-scoped
    user context. If a read_file call also records them as recent files,
    restoring them again after compact duplicates the same context.
    """
    _post_compact_excluded_paths.add(str(path))
    try:
        _post_compact_excluded_paths.add(str(Path(path).resolve()))
    except OSError:
        pass


def track_file(path: str, content: str):
    """Record a file for direct compact.py tests that do not have run state.

    The real run_task path now passes FileReadState into compaction.  Keeping
    this narrow fallback lets older direct tests exercise compact.py without
    reintroducing a process-global dependency into the main loop.
    """

    # dict 保序：重复读同一文件会移到末尾（最近），保留最新内容
    _recent_files.pop(str(path), None)
    _recent_files[str(path)] = content


def _recent_files_for_post_compact(read_state=None):
    """Snapshot and clear read state only after compact has succeeded.

    Claude Code consumes readFileState after compact so the next request can
    restore recent file context, then clears it to avoid stale pre-compact read
    records leaking into later write guards or restore attachments.
    """

    if read_state is None:
        return _recent_files
    recent_files = read_state.recent_file_items()
    read_state.reset()
    return recent_files


def _post_compact_file_attachment(
    cfg: "CompactConfig" = DEFAULT,
    exclude_paths: set = None,
    executor=None,
    recent_files=None,
) -> str:
    """compact 后把最近读过的文件重新注入（对齐 CC createPostCompactFileAttachments）。

    取最近 N 个（CC=5），每个截到 max_tokens_per_file（CC=5K），总预算 token_budget（CC=50K）。
    重读磁盘拿**最新**内容（CC 用 FileReadTool re-read，不是缓存旧值）；读不到就退回缓存内容。
    返回拼好的 attachment 文本；无文件则空串。

    对齐 CC `collectReadToolFilePaths` + `:1429 !preservedReadPaths.has(...)`：
      exclude_paths 通常来自 session-memory compact 保留近端里的完整 read_file 路径，以及全局
      治理文件排除；这些文件跳过、不再重注入，避免重复占用上下文。
      full_compact 已回归 full=0,不再传 kept-tail exclude。
      先滤后取，被排除的文件不占 N 个名额。
    """
    # ★ 观测(补盲点 2026-06-27)：后注入原本静默在 full_compact 内执行、无独立 span → 之前只能
    #   docker exec 看容器 messages 才确认它跑没跑。这里落 compact.post_attach span，让 trace 直接可见
    #   n_attached/attach_tokens/files。包整个函数体(含早退路径也落 span，n_attached=0=有近期文件但没注)。
    with span("compact.post_attach", SpanKind.INTERNAL) as sp:
        exclude = set(exclude_paths or set()) | _post_compact_excluded_paths
        recent = _recent_files if recent_files is None else recent_files
        if executor is None:
            result = post_compact_file_attachment_text(recent, cfg, exclude)
        else:
            result = post_compact_file_attachment_text(
                recent,
                cfg,
                exclude,
                executor=executor,
            )
        files = re.findall(r"^--- (.+?) ---$", result, flags=re.MULTILINE)
        sp.set(n_recent=len(recent), n_excluded=len(exclude),
               n_attached=len(files), attach_tokens=len(result) // 4,
               files=files)
        return result


def _queue_post_compact_file_attachment(
    cfg: "CompactConfig" = DEFAULT,
    exclude_paths: set = None,
    executor=None,
    post_compact_sink=None,
) -> None:
    attach = _post_compact_file_attachment(cfg, exclude_paths, executor=executor)
    _queue_post_compact_attachments(
        {"role": "user", "content": attach} if attach else None,
        post_compact_sink=post_compact_sink,
    )


def _queue_post_compact_attachments(
    *attachments: dict | None,
    post_compact_sink=None,
) -> None:
    """Replace the post-compact lane with restore context for the latest compact.

    Only the next model request needs these messages.  Keeping them here instead
    of in the compact result prevents large restore bodies from becoming durable
    transcript content that later compactions would summarize again.
    """

    _pending_post_compact_attachments.clear()
    if post_compact_sink is not None:
        post_compact_sink.queue_post_compact_attachments(*attachments)
        return

    for attachment in attachments:
        if attachment is not None:
            _pending_post_compact_attachments.append(copy.deepcopy(attachment))


def _post_compact_file_attachment_message(
    cfg: "CompactConfig" = DEFAULT,
    exclude_paths: set = None,
    executor=None,
    recent_files=None,
) -> dict | None:
    """Return the file restore as a request-only message for post-compact send.

    Compaction can erase file bodies from durable history, but the restored file
    text is only useful as immediate working context and should not persist.
    """

    attach = _post_compact_file_attachment(
        cfg,
        exclude_paths,
        executor=executor,
        recent_files=recent_files,
    )
    return {"role": "user", "content": attach} if attach else None


def _post_compact_skill_attachment(
    skill_agent_id: str | None,
    cfg: "CompactConfig" = DEFAULT,
) -> dict | None:
    """Restore invoked skill bodies without storing them in compact history.

    Skill instructions are run-local working context.  Re-sending them once
    after compact matches Claude Code attachments while avoiding durable prompt
    growth from full skill bodies.
    """

    return invoked_skill_context_message(
        skill_agent_id,
        max_tokens_per_skill=cfg.post_compact_max_tokens_per_skill,
        token_budget=cfg.post_compact_skills_token_budget,
    )


# ──────────────────────────────────────────────
# helper
# ──────────────────────────────────────────────

def _post_compact_deferred_tools_attachment(deferred_tool_state=None) -> dict | None:
    """Re-announce selected deferred tools only as next-request context.

    The run-local DeferredToolState remains the source that unlocks schemas.
    This marker is just visible continuity for the model after compact.
    """

    if deferred_tool_state is None:
        return None
    selected_names = getattr(deferred_tool_state, "selected_names", ())
    return selected_deferred_tools_marker_message(selected_names, durable=False)


def run_post_compact_cleanup(
    result: list,
    *,
    cfg: "CompactConfig" = DEFAULT,
    exclude_paths: set = None,
    skill_agent_id: str | None = None,
    deferred_tool_state=None,
    executor=None,
    post_compact_sink=None,
    read_state=None,
    session_memory=None,
) -> None:
    """Run the success-only cleanup contract shared by full and SM compact.

    The compact result is already durable before this helper runs. Everything
    queued here is request-only context for the next model call; the run-local
    read state is consumed only after success, and SM anchors advance only when
    the SM result actually replaces the transcript.
    """

    recent_files = _recent_files_for_post_compact(read_state)
    _queue_post_compact_attachments(
        _post_compact_file_attachment_message(
            cfg,
            exclude_paths,
            executor=executor,
            recent_files=recent_files,
        ),
        _post_compact_skill_attachment(skill_agent_id, cfg),
        _post_compact_deferred_tools_attachment(deferred_tool_state),
        post_compact_sink=post_compact_sink,
    )
    if session_memory is not None:
        session_memory.on_compacted(result)


def _blocks(m):
    c = m.get("content")
    return c if isinstance(c, list) else []


def _block_type(b):
    return b.get("type") if isinstance(b, dict) else getattr(b, "type", None)


def _block_attr(b, key, default=None):
    return b.get(key, default) if isinstance(b, dict) else getattr(b, key, default)


def _content_str(m) -> str:
    c = m.get("content", "")
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False, default=str)


MEDIA_BLOCK_TOKEN_ESTIMATE = 2_000


def _rough_text_tokens(value) -> int:
    return len(str(value or "")) // 4


def _rough_json_tokens(value) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str)) // 4


def _tool_result_content_tokens(content) -> int:
    if isinstance(content, str):
        return _rough_text_tokens(content)
    if isinstance(content, list):
        return sum(_content_block_tokens(block) for block in content)
    return _rough_json_tokens(content)


def _content_block_tokens(block) -> int:
    """Estimate blocks by shape so media blobs do not dominate thresholds.

    This is still a rough estimator, not CC's canonical API-usage backed
    tokenCountWithEstimation.  The point is to preserve cheap local threshold
    checks while avoiding the worst failures of dumping every block as JSON.
    """

    block_type = _block_type(block)
    if block_type == "text":
        return _rough_text_tokens(_block_attr(block, "text", ""))
    if block_type == "tool_use":
        return (
            _rough_text_tokens(_block_attr(block, "name", ""))
            + _rough_json_tokens(_block_attr(block, "input", {}))
        )
    if block_type == "tool_result":
        return _tool_result_content_tokens(_block_attr(block, "content", ""))
    if block_type in {"image", "document"}:
        return MEDIA_BLOCK_TOKEN_ESTIMATE
    if block_type == "thinking":
        return _rough_text_tokens(
            _block_attr(block, "thinking", _block_attr(block, "text", ""))
        )
    if block_type == "redacted_thinking":
        return _rough_text_tokens(
            _block_attr(block, "data", _block_attr(block, "text", ""))
        )
    return _rough_json_tokens(block)


def _message_estimate_parts(message) -> tuple[int, int]:
    rendered_memories = _render_relevant_memories(message)
    if rendered_memories is not None:
        chars = sum(
            len(str(rendered.get("content", "")))
            for rendered in rendered_memories
        )
        return chars, 0
    content = message.get("content", "")
    if isinstance(content, str):
        return len(content), 0
    if isinstance(content, list):
        return 0, sum(_content_block_tokens(block) for block in content)
    return len(json.dumps(content, ensure_ascii=False, default=str)), 0


def _estimate_message_tokens(message) -> int:
    chars, block_tokens = _message_estimate_parts(message)
    return chars // 4 + block_tokens


def estimate(messages, system: str = "") -> int:
    char_total = len(system or "")
    block_tokens = 0
    for m in messages:
        if is_compact_boundary_message(m):
            continue
        chars, tokens = _message_estimate_parts(m)
        char_total += chars
        block_tokens += tokens
    return char_total // 4 + block_tokens


def _safe_sink_attr(post_compact_sink, name: str, default=None):
    """Read optional sink attributes without letting telemetry break compact.

    Some tests and future sinks may expose duck-typed members as properties.
    Telemetry should degrade to default attrs if those properties fail instead
    of turning a successful compact into an error path.
    """

    try:
        return getattr(post_compact_sink, name, default)
    except Exception:
        return default


def _queued_post_compact_attachments(post_compact_sink=None) -> tuple[dict, ...]:
    if post_compact_sink is None:
        return peek_post_compact_attachments()
    peek = _safe_sink_attr(post_compact_sink, "peek_post_compact_attachments")
    if callable(peek):
        try:
            return tuple(peek())
        except Exception:
            return ()
    return ()


def _record_compaction_event_attrs(
    post_compact_sink,
    *,
    auto_compact_threshold: int,
    true_post_compact_tokens: int,
    will_retrigger_next_turn: bool,
) -> dict:
    compact_turn_no = _safe_sink_attr(post_compact_sink, "turn_no")
    attrs = {
        "is_recompaction_in_chain": False,
        "turns_since_previous_compact": None,
        "previous_compact_turn_no": None,
        "compact_turn_no": compact_turn_no,
    }
    record = _safe_sink_attr(post_compact_sink, "record_compaction_event")
    if not callable(record):
        return attrs
    try:
        recorded = record(
            compact_turn_no=compact_turn_no,
            auto_compact_threshold=auto_compact_threshold,
            true_post_compact_tokens=true_post_compact_tokens,
            will_retrigger_next_turn=will_retrigger_next_turn,
        )
    except Exception:
        return attrs
    if isinstance(recorded, dict):
        attrs.update({key: recorded.get(key, attrs[key]) for key in attrs})
    return attrs


def _post_compact_success_attrs(
    result: list,
    *,
    system: str,
    cfg: "CompactConfig",
    auto_thr: int | None,
    post_compact_sink=None,
) -> dict:
    threshold = auto_thr if auto_thr is not None else auto_threshold(cfg)
    durable_tokens = estimate(result)
    payload_messages = [*result, *_queued_post_compact_attachments(post_compact_sink)]
    true_tokens = estimate(payload_messages, system)
    will_retrigger = true_tokens >= threshold
    attrs = {
        "true_post_compact_tokens": true_tokens,
        "post_compact_durable_tokens": durable_tokens,
        "auto_compact_threshold": threshold,
        "will_retrigger_next_turn": will_retrigger,
    }
    attrs.update(
        _record_compaction_event_attrs(
            post_compact_sink,
            auto_compact_threshold=threshold,
            true_post_compact_tokens=true_tokens,
            will_retrigger_next_turn=will_retrigger,
        )
    )
    return attrs


def _has_tool_use(m) -> bool:
    return m.get("role") == "assistant" and any(_block_type(b) == "tool_use" for b in _blocks(m))


def _is_tool_result(m) -> bool:
    return m.get("role") == "user" and any(_block_type(b) == "tool_result" for b in _blocks(m))


def _tool_use_ids(message) -> set[str]:
    """Collect tool_use ids from one assistant message."""

    ids = set()
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return ids
    for block in _blocks(message):
        if _block_type(block) == "tool_use":
            value = _block_attr(block, "id")
            if value:
                ids.add(str(value))
    return ids


def _tool_result_ids(message) -> set[str]:
    """Collect tool_result ids from one user tool-result message."""

    ids = set()
    if not isinstance(message, dict) or message.get("role") != "user":
        return ids
    for block in _blocks(message):
        if _block_type(block) == "tool_result":
            value = _block_attr(block, "tool_use_id")
            if value:
                ids.add(str(value))
    return ids


def _has_text_content(message) -> bool:
    """Approximate CC's minTextBlockMessages check for this dict schema."""

    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    for block in content:
        block_type = _block_type(block)
        if block_type == "text" and str(_block_attr(block, "text", "")).strip():
            return True
        if block_type not in {"tool_use", "tool_result"}:
            text = _block_attr(block, "text", None)
            body = _block_attr(block, "content", None)
            if str(text or body or "").strip():
                return True
    return False


def _exception_text(exc: Exception) -> str:
    parts = [str(getattr(exc, "message", "") or ""), str(exc)]
    for attr in ("body", "error", "error_details", "errorDetails"):
        val = getattr(exc, attr, None)
        if val:
            parts.append(str(val))
    response = getattr(exc, "response", None)
    if response is not None:
        text = getattr(response, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(dict.fromkeys(p for p in parts if p))


def _is_prompt_too_long_error(exc: Exception) -> bool:
    text = _exception_text(exc).lower()
    if not text:
        return False
    direct = (
        "prompt is too long",
        "prompt too long",
        "context_length_exceeded",
        "maximum context length",
        "context window exceeded",
        "context overflow",
    )
    if any(s in text for s in direct):
        return True
    if "context" in text and any(s in text for s in ("too long", "too large", "exceed", "maximum", "length")):
        return True
    return "tokens" in text and ">" in text and "maximum" in text


def _prompt_too_long_token_gap(exc: Exception) -> int | None:
    text = _exception_text(exc)
    match = re.search(r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)", text, re.I)
    if match is None:
        match = re.search(r"(\d+)\s*tokens?\s*>\s*(\d+)", text, re.I)
    if match is None:
        return None
    actual = int(match.group(1))
    limit = int(match.group(2))
    gap = actual - limit
    return gap if gap > 0 else None


def _max_tokens_cap_from_error(exc: Exception) -> int | None:
    text = _exception_text(exc)
    lower = text.lower()
    if "max_tokens" not in lower and "max output" not in lower:
        return None
    if "input length" in lower and "context limit" in lower:
        return None
    patterns = (
        r"max_tokens[^0-9]*(?:less than or equal to|at most|no more than|<=|cannot exceed|must be)[^0-9]*(\d+)",
        r"(?:maximum|max)[^0-9]{0,40}max_tokens[^0-9]*(\d+)",
        r"max_tokens[^0-9]*(\d+)[^0-9]*(?:maximum|max|limit)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match is not None:
            return _positive_int(match.group(1))
    return None


def _context_overflow_max_tokens_from_error(exc: Exception) -> int | None:
    text = _exception_text(exc)
    match = re.search(
        r"input length and `max_tokens` exceed context limit:\s*(\d+)\s*\+\s*(\d+)\s*>\s*(\d+)",
        text,
        re.I,
    )
    if match is None:
        return None
    input_tokens = int(match.group(1))
    context_limit = int(match.group(3))
    available = context_limit - input_tokens - CONTEXT_OVERFLOW_SAFETY_BUFFER_TOKENS
    if available < MIN_COMPACT_OUTPUT_TOKENS:
        return None
    return available


def _strip_ptl_retry_marker(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    first = messages[0]
    if first.get("role") == "user" and first.get("content") == PTL_RETRY_MARKER:
        return messages[1:]
    return messages


def _group_messages_for_ptl_retry(messages: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    saw_assistant_boundary = False
    for msg in messages:
        if msg.get("role") == "assistant" and current:
            groups.append(current)
            current = [msg]
            saw_assistant_boundary = True
        else:
            current.append(msg)
    if current:
        groups.append(current)
    if saw_assistant_boundary and len(groups) >= 2:
        return groups

    groups = []
    current = []
    for msg in messages:
        if msg.get("role") == "user" and not _is_tool_result(msg) and current:
            groups.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        groups.append(current)
    return groups


def _truncate_head_for_ptl_retry(messages: list[dict], exc: Exception) -> list[dict] | None:
    input_messages = _strip_ptl_retry_marker(messages)
    groups = _group_messages_for_ptl_retry(input_messages)
    if len(groups) < 2:
        return None

    token_gap = _prompt_too_long_token_gap(exc)
    if token_gap is None:
        drop_count = max(1, len(groups) // 5)
    else:
        acc = 0
        drop_count = 0
        for group in groups:
            acc += estimate(group)
            drop_count += 1
            if acc >= token_gap:
                break
    drop_count = min(drop_count, len(groups) - 1)
    if drop_count < 1:
        return None

    sliced = [msg for group in groups[drop_count:] for msg in group]
    while sliced and _is_tool_result(sliced[0]):
        sliced = sliced[1:]
    if not sliced:
        return None
    if sliced[0].get("role") != "user":
        sliced = [{"role": "user", "content": PTL_RETRY_MARKER}, *sliced]
    return sliced


# microcompact 只清「可恢复」工具的 tool_result。
# 入选标准 = 输出可从外部真相源再取回（磁盘重读 / 命令重跑），即 CC「上下文=磁盘 cache」哲学的代码化。
# ★ update_todos 故意排除：它是 agent 的计划状态、唯一副本、磁盘上没有 → 清了不可恢复 → 续作崩。
#   (CC 的 TodoWrite 同样不在 COMPACTABLE_TOOLS 里。)
COMPACTABLE_TOOLS = {"bash", "powershell", "read_file", "write_file", "edit_file", "glob", "grep"}


def _tool_id_to_name(messages) -> dict:
    """扫 assistant 的 tool_use block，建 {tool_use_id → tool_name} 映射（对齐 CC collectCompactableToolIds）。
    tool_result 只带 tool_use_id、不带工具名 → 必须靠这个映射才能按工具类型决定清不清。"""
    m2n = {}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for b in _blocks(m):
            if _block_type(b) == "tool_use":
                tid = _block_attr(b, "id")
                if tid:
                    m2n[tid] = _block_attr(b, "name")
    return m2n


def _kept_intact_read_paths(kept) -> set:
    """收集**保留近端**(kept)里仍完好存在的 read_file 文件路径 —— 对齐 CC collectReadToolFilePaths。
    post-compact 重注入拿这个集合去重：路径已在近端原文里 → 不再重注入。
    只认 read_file：attachment 重注入的就是文件全文，write/edit 不产出全文、无重复之虞。
    ★ 我方适配（CC 无）：micro 会就地把旧 tool_result 清成占位符 → 必须排除**已被清**的，
      因为那时文件内容其实已不在场，仍需重注入，否则会把它彻底丢掉。"""
    id2path = {}
    for m in kept:
        if m.get("role") != "assistant":
            continue
        for b in _blocks(m):
            if _block_type(b) == "tool_use" and _block_attr(b, "name") == "read_file":
                inp = _block_attr(b, "input", {})
                p = inp.get("path") if isinstance(inp, dict) else None
                tid = _block_attr(b, "id")
                if p and tid:
                    id2path[tid] = str(p)
    present = set()
    for m in kept:
        if not _is_tool_result(m):
            continue
        for b in _blocks(m):
            if _block_type(b) == "tool_result" and isinstance(b, dict):
                tid = b.get("tool_use_id")
                if tid in id2path and str(b.get("content", "")) != _MC_CLEARED:
                    present.add(id2path[tid])
    return present


# ──────────────────────────────────────────────
# microcompact —— 清旧 tool_result(= clear_tool_uses;不碰 user 文本)
# ──────────────────────────────────────────────

_MC_CLEARED = "[Old tool result content cleared]"   # 稳定占位符，便于恢复逻辑识别已清理结果


def microcompact(messages, cfg: CompactConfig = DEFAULT, target_tokens: int = None):
    """清旧 tool_result —— 只清**可恢复工具**(COMPACTABLE_TOOLS)的结果,不碰 user/assistant 文本。

    按 tool_use→tool_result 配对查出工具名，**只清白名单内工具**
    (read/bash/grep/glob/edit/write),永不碰 update_todos 这种唯一副本(否则不可恢复→续作崩);
    ② 始终保留最近 `keep` 个(下限钳 1,防清空留模型零上下文,CC `Math.max(1, keepRecent)`);
    ③ 给了 target_tokens 就清到总量 ≤ target 即停,否则清完所有可清的。
    **不再按长度判清不清**(CC 无长度地板;短结果常是关键唯一值如 exit code/端口,按工具类型才安全)。"""
    with span("compact.microcompact", SpanKind.INTERNAL) as sp:
        before = estimate(messages)
        id2name = _tool_id_to_name(messages)
        blocks = []                          # 可清(白名单内)的 tool_result block,按出现顺序(旧→新)
        for m in messages:
            if not _is_tool_result(m):
                continue
            for b in _blocks(m):
                if _block_type(b) != "tool_result" or not isinstance(b, dict):
                    continue
                name = id2name.get(b.get("tool_use_id"))
                if name in COMPACTABLE_TOOLS:        # 只清可恢复工具（输出可从磁盘/命令重取回）
                    blocks.append(b)
        keep = max(1, cfg.microcompact_keep)         # 下限钳 1（CC Math.max(1, keepRecent)：永远保留最近 1 个）
        eligible = blocks[:-keep]                    # 排除最近 keep 个
        cur, cleared = before, 0
        for b in eligible:                   # 从最旧开始清
            if target_tokens is not None and cur <= target_tokens:
                break
            if b.get("content") == _MC_CLEARED:      # 已清过,跳过(防重复计数)
                continue
            content = str(b.get("content", ""))
            cur -= (len(content) - len(_MC_CLEARED)) // 4
            b["content"] = _MC_CLEARED
            cleared += 1
        after = estimate(messages)
        sp.set(**{"layer": "microcompact", "tokens_before": before, "tokens_after": after,
                  "cleared": cleared, "target": target_tokens})
        return messages


# ──────────────────────────────────────────────
# full_compact —— 完整 durable messages + 追加摘要请求 + 文件/skill/deferred 恢复
# ──────────────────────────────────────────────

_COMPACT_PROMPT = """CRITICAL: 只输出文本,不要调用任何工具(工具调用会被拒绝并浪费这次机会)。

请总结上方完整对话。

先在 <analysis> 里分析:主要任务?已完成什么?读/写/改了哪些文件?关键决策与原因?
遇到的错误与修复?还剩什么待办?

然后在 <summary> 里写简洁摘要,覆盖 9 块,**逐字保留所有事实、人名、数字、ID、端口、版本、
路径、决策、用户偏好与指令**:
1. 主要请求与意图  2. 关键技术概念与决策  3. 涉及的文件与代码片段  4. 错误与修复
5. 问题解决过程  6. 所有用户消息(非工具)← 必须逐条保留  7. 待办  8. 当前工作状态  9. 下一步
"""

# 对齐 CC prompt.ts:269-272 NO_TOOLS_TRAILER：在 summary request 尾部重申一遍禁工具。
# CC 用前后双重约束——模型读完一长段对话后,开头那条"别调工具"的指令已被冲淡,尾部重申显著降低
# 误调工具率（CC 实测 4.6 上缺尾部约 2.79% 的 turn 被工具调用浪费）。顺带把输出结构再钉一次。
# ⚠ 9 段保真在 deepseek 上仍**待 eval 验证**（中文适配版,非 CC 英文原模板,无 <example> 骨架）。
_COMPACT_TRAILER = """

REMINDER:现在开始输出摘要。**只输出文本,绝对不要调用任何工具**——这次调工具会被拒绝、白白浪费这次压缩机会。
严格按「先 <analysis>…</analysis> 后 <summary>…</summary>」结构输出;<summary> 内逐条覆盖上述 9 块,
逐字保留所有事实、人名、数字、ID、端口、版本、路径、决策、用户偏好与指令。"""

def _response_text(resp) -> str:
    return "".join(getattr(b, "text", "") for b in getattr(resp, "content", [])
                   if getattr(b, "type", None) == "text").strip()


def _response_block_types(resp) -> list[str]:
    return [
        str(getattr(b, "type", "unknown") or "unknown")
        for b in getattr(resp, "content", [])
    ]


def _format_summary(raw: str) -> str:
    body = _summary_body(raw)
    return f"[Compacted]\n{body}" if body else "[Compacted]\n(空摘要)"


def _summary_body(raw: str) -> str:
    cleaned = re.sub(r"<analysis>[\s\S]*?</analysis>", "", raw)
    mt = re.search(r"<summary>([\s\S]*?)</summary>", cleaned)
    return mt.group(1).strip() if mt else cleaned.strip()


def _keep_tail_budget(messages, cfg: CompactConfig) -> list:
    """从尾部保留近端:至少 keep_min_msgs 条且累计达 keep_min_tokens,上限 keep_max_tokens;去开头孤儿。"""
    kept = []
    toks = 0
    for m in reversed(messages):
        kept.insert(0, m)
        toks += _estimate_message_tokens(m)
        enough = len(kept) >= cfg.keep_min_msgs and toks >= cfg.keep_min_tokens
        if enough or toks >= cfg.keep_max_tokens:
            break
    while kept and _is_tool_result(kept[0]):   # 防孤儿:开头 tool_result 的 tool_use 已被摘要吞掉
        kept = kept[1:]
    return kept


def _adjust_keep_start_for_tool_pairs(messages, start_index: int, floor: int) -> int:
    """Move start_index backward when kept tool_results need their tool_use.

    Claude Code adjusts the preserved range to avoid breaking API invariants.
    This local version honors the compact-boundary floor, so it will not walk
    into a previous compact segment.
    """

    if start_index <= floor or start_index >= len(messages):
        return start_index

    adjusted = start_index
    needed = set()
    tool_uses_in_kept = set()
    for message in messages[adjusted:]:
        needed.update(_tool_result_ids(message))
        tool_uses_in_kept.update(_tool_use_ids(message))
    needed -= tool_uses_in_kept

    for idx in range(adjusted - 1, floor - 1, -1):
        found = _tool_use_ids(messages[idx]) & needed
        if not found:
            continue
        adjusted = idx
        needed -= found
        if not needed:
            break
    return adjusted


def _drop_orphan_tool_results(messages: list) -> list:
    """Drop tool_result messages whose matching tool_use is outside the kept range."""

    available_tool_uses = set()
    kept = []
    for message in messages:
        result_ids = _tool_result_ids(message)
        if result_ids and not result_ids <= available_tool_uses:
            continue
        kept.append(message)
        available_tool_uses.update(_tool_use_ids(message))
    return kept


def _session_memory_keep_start_index(messages, last_summarized_index: int, cfg: CompactConfig) -> int:
    """Calculate the CC-style messagesToKeep start index for SM compact.

    Start immediately after the summarized message.  If the boundary is unknown
    in a resumed session, start with no old messages and only expand backward to
    satisfy minimums.  Expansion never crosses the last compact boundary.
    """

    if not messages:
        return 0

    boundary_index = find_last_compact_boundary_index(messages)
    floor = boundary_index + 1 if boundary_index >= 0 else 0
    start_index = last_summarized_index + 1 if last_summarized_index >= 0 else len(messages)
    start_index = max(floor, min(start_index, len(messages)))

    total_tokens = estimate(messages[start_index:])
    text_message_count = sum(1 for message in messages[start_index:] if _has_text_content(message))
    meets_minimums = (
        total_tokens >= cfg.keep_min_tokens
        and text_message_count >= cfg.keep_min_msgs
    )
    if total_tokens >= cfg.keep_max_tokens or meets_minimums:
        return _adjust_keep_start_for_tool_pairs(messages, start_index, floor)

    for idx in range(start_index - 1, floor - 1, -1):
        message = messages[idx]
        total_tokens += estimate([message])
        if _has_text_content(message):
            text_message_count += 1
        start_index = idx
        if total_tokens >= cfg.keep_max_tokens:
            break
        if total_tokens >= cfg.keep_min_tokens and text_message_count >= cfg.keep_min_msgs:
            break

    return _adjust_keep_start_for_tool_pairs(messages, start_index, floor)


def _session_memory_messages_to_keep(messages, last_summarized_message_id, cfg: CompactConfig) -> list | None:
    """Return SM compact messagesToKeep, or None when the known anchor is stale.

    A present-but-missing last_summarized_message_id means we cannot separate
    summarized history from fresh history; matching Claude Code, that falls back
    to full compact.  With no anchor, a non-empty SM file is treated as resumed
    or unknown-boundary state: keep nothing first, then expand by minimums.
    """

    ensure_runtime_message_ids(messages)
    if last_summarized_message_id:
        summarized_index = next(
            (
                idx
                for idx, message in enumerate(messages)
                if message_matches_identity(message, last_summarized_message_id)
            ),
            -1,
        )
        if summarized_index < 0:
            return None
    else:
        summarized_index = len(messages) - 1

    start_index = _session_memory_keep_start_index(messages, summarized_index, cfg)
    kept = [
        message
        for message in messages[start_index:]
        if not is_compact_boundary_message(message)
    ]
    return _drop_orphan_tool_results(kept)


def full_compact(messages, system: str = "", cfg: CompactConfig = DEFAULT,
                 skill_agent_id: str | None = None, deferred_tool_state=None,
                 executor=None, post_compact_sink=None, read_state=None,
                 auto_thr: int | None = None):
    global _circuit_breaker
    with span("compact.full_compact", SpanKind.INTERNAL) as sp:
        ensure_message_uuids(messages)
        messages_to_summarize = copy.deepcopy(messages_after_compact_boundary(messages))
        before = estimate(messages_to_summarize)
        messages_summarized = len(messages_to_summarize)
        if _circuit_breaker >= cfg.auto_max_failures:
            sp.set(**{"layer": "full_compact", "status": "circuit_broken",
                      "tokens_before": before, "tokens_after": before,
                      "compact_llm_calls": 0, "ptl_retry_attempts": 0,
                      "compact_stop_reason": "", "compact_output_truncated": False,
                      "compact_response_block_types": ""})
            return messages
        summary_request = {"role": "user", "content": _COMPACT_PROMPT + _COMPACT_TRAILER}
        compact_llm_calls = 0
        ptl_retry_attempts = 0
        compact_stop_reason = ""
        compact_response_block_types = ""
        compact_usage = None
        requested_summary_max_tokens = _positive_int(cfg.summary_max_tokens) or DEFAULT_SUMMARY_MAX_TOKENS
        effective_summary_max_tokens = _effective_summary_max_tokens(requested_summary_max_tokens)
        max_tokens_cap_retry_attempts = 0
        context_overflow_retry_attempts = 0
        try:
            while True:
                compact_messages = _compact_api_messages(messages_to_summarize)
                compact_messages.append(copy.deepcopy(summary_request))
                compact_llm_calls += 1
                try:
                    resp = llm.chat(compact_messages,
                                    system="你是对话压缩器,只输出文本不调工具。",
                                    tools=[],
                                    max_tokens=effective_summary_max_tokens, purpose="compaction")
                except Exception as e:
                    provider_cap = _max_tokens_cap_from_error(e)
                    if provider_cap is not None and provider_cap < effective_summary_max_tokens:
                        effective_summary_max_tokens = max(1, provider_cap)
                        max_tokens_cap_retry_attempts += 1
                        continue
                    overflow_cap = _context_overflow_max_tokens_from_error(e)
                    if overflow_cap is not None and overflow_cap < effective_summary_max_tokens:
                        effective_summary_max_tokens = max(1, overflow_cap)
                        context_overflow_retry_attempts += 1
                        continue
                    if not _is_prompt_too_long_error(e):
                        raise
                    if ptl_retry_attempts >= MAX_PTL_RETRIES:
                        raise
                    truncated = _truncate_head_for_ptl_retry(messages_to_summarize, e)
                    if truncated is None:
                        raise
                    ptl_retry_attempts += 1
                    messages_to_summarize = truncated
                    continue
                compact_stop_reason = str(getattr(resp, "stop_reason", "") or "")
                compact_response_block_types = ",".join(_response_block_types(resp))
                compact_usage = getattr(resp, "usage", None)
                if compact_stop_reason == "max_tokens":
                    raise ValueError("truncated summary (stop_reason=max_tokens)")
                raw = _response_text(resp)
                if raw:
                    break
                block_types = compact_response_block_types or "none"
                raise ValueError(f"empty summary ({block_types})")
            summary_body = _summary_body(raw)
            if not summary_body:
                raise ValueError("empty summary body")
            _circuit_breaker = 0
            # 维度4 成本归因：把这次摘要调用的 token 记到 compact span，标"full_compact 的代价"
            u = compact_usage
            if u is not None:
                ci = getattr(u, "input_tokens", 0) or 0
                co = getattr(u, "output_tokens", 0) or 0
                sp.set(**{"compact_cost_input_tokens": ci, "compact_cost_output_tokens": co,
                          "compact_cost_input": ci, "compact_cost_output": co,
                          "compact_api_usage_tokens": ci + co,
                          "compact_llm_calls": compact_llm_calls,
                          "ptl_retry_attempts": ptl_retry_attempts,
                          "max_tokens_cap_retry_attempts": max_tokens_cap_retry_attempts,
                          "context_overflow_retry_attempts": context_overflow_retry_attempts,
                          "compact_requested_max_tokens": requested_summary_max_tokens,
                          "compact_effective_max_tokens": effective_summary_max_tokens,
                          "compact_max_tokens_clamped": effective_summary_max_tokens != requested_summary_max_tokens})
        except Exception as e:
            _circuit_breaker += 1
            err_attrs = {"layer": "full_compact", "status": "error", "detail": str(e)[:80],
                         "tokens_before": before, "tokens_after": before,
                         "compact_llm_calls": compact_llm_calls,
                         "ptl_retry_attempts": ptl_retry_attempts,
                         "max_tokens_cap_retry_attempts": max_tokens_cap_retry_attempts,
                         "context_overflow_retry_attempts": context_overflow_retry_attempts,
                         "compact_requested_max_tokens": requested_summary_max_tokens,
                         "compact_effective_max_tokens": effective_summary_max_tokens,
                         "compact_max_tokens_clamped": effective_summary_max_tokens != requested_summary_max_tokens,
                         "compact_stop_reason": compact_stop_reason,
                         "compact_output_truncated": compact_stop_reason == "max_tokens",
                         "compact_response_block_types": compact_response_block_types}
            if compact_usage is not None:
                ci = getattr(compact_usage, "input_tokens", 0) or 0
                co = getattr(compact_usage, "output_tokens", 0) or 0
                err_attrs.update({
                    "compact_cost_input_tokens": ci,
                    "compact_cost_output_tokens": co,
                    "compact_cost_input": ci,
                    "compact_cost_output": co,
                    "compact_api_usage_tokens": ci + co,
                })
            sp.set(**err_attrs)
            return messages

        result = [
            create_compact_boundary_message(
                trigger="auto",
                pre_tokens=before,
                user_context=system or "",
                messages_summarized=messages_summarized,
                logical_parent_uuid=(
                    message_identity(messages_to_summarize[-1])
                    if messages_to_summarize
                    else None
                ),
            ),
            create_compact_summary_message(
                f"[Compacted]\n{summary_body}",
                source="full_compact",
            ),
        ]
        run_post_compact_cleanup(
            result,
            cfg=cfg,
            skill_agent_id=skill_agent_id,
            deferred_tool_state=deferred_tool_state,
            executor=executor,
            post_compact_sink=post_compact_sink,
            read_state=read_state,
        )
        after = estimate(result)
        success_attrs = _post_compact_success_attrs(
            result,
            system=system,
            cfg=cfg,
            auto_thr=auto_thr,
            post_compact_sink=post_compact_sink,
        )
        sp.set(**{"layer": "full_compact", "status": "ok", "tokens_before": before,
                  "tokens_after": after, "ratio": round(before / max(1, after), 2),
                  "compact_llm_calls": compact_llm_calls,
                  "ptl_retry_attempts": ptl_retry_attempts,
                  "max_tokens_cap_retry_attempts": max_tokens_cap_retry_attempts,
                  "context_overflow_retry_attempts": context_overflow_retry_attempts,
                  "compact_requested_max_tokens": requested_summary_max_tokens,
                  "compact_effective_max_tokens": effective_summary_max_tokens,
                  "compact_max_tokens_clamped": effective_summary_max_tokens != requested_summary_max_tokens,
                  "compact_stop_reason": compact_stop_reason,
                  "compact_output_truncated": False,
                  "compact_response_block_types": compact_response_block_types,
                  **success_attrs})
        return result


# ──────────────────────────────────────────────
# session_memory_compact —— 免 LLM 中间层（对齐 CC trySessionMemoryCompaction）
# ──────────────────────────────────────────────

def truncate_sm_for_compact(content: str, max_tokens: int = 12_000) -> str:
    """防超长 SM 文件吃满 post-compact 预算（对齐 CC truncateSessionMemoryForCompact，P2-5）。
    ⚠ 偏离：CC 按段（# 标题）逐段截到 MAX_SECTION（2K）；我们先**整体**截到 max_total（SM 文件预算 12K），
    按行边界切——简化、够防"一个超长 SM 吃满预算"。backlog：精化为按段截（每段保头部）。"""
    max_chars = max_tokens * 4
    if len(content) <= max_chars:
        return content
    cut = content[:max_chars]
    nl = cut.rfind("\n")
    cut = cut[:nl] if nl > max_chars // 2 else cut
    return cut + "\n\n[... session memory 截断以适配压缩预算 ...]"


def session_memory_compact(messages, sm, system: str, cfg: CompactConfig, auto_thr: int,
                           skill_agent_id: str | None = None, deferred_tool_state=None,
                           executor=None, post_compact_sink=None, read_state=None):
    """用**已写好的 SM 文件**当摘要替代 LLM 摘要（零新 LLM）。对齐 CC trySessionMemoryCompaction。

    sm 是 SessionMemory 实例（duck typing：is_empty / path / wait_for_extraction / on_compacted），
    compact.py 不 import memory（避免循环）——sm 由调用方传入。
    返回新 messages，或 **None（回退 full_compact）**：无 SM 文件 / 仍空模板 / 压完仍超阈值
    （对齐 CC `:519-613` 的回退条件）。近端保留由 SessionMemory anchor 计算 messagesToKeep：
    从 last_summarized_message_id 后开始，不足最小预算才向前扩展，且不跨最后 compact boundary。
    ACE 保持线性 transcript，但每条 durable message 都有稳定 UUID；compact boundary
    通过 logicalParentUuid 指向压缩前最后一条消息。"""
    with span("compact.session_memory_compact", SpanKind.INTERNAL) as sp:
        ensure_runtime_message_ids(messages)
        source_messages = messages_after_compact_boundary(messages)
        before = estimate(source_messages, system)
        sm.wait_for_extraction()                       # P1-3：防读到写一半的 SM 文件
        if sm.is_empty():                              # 无实质内容 → 回退 full_compact（对齐 CC isSessionMemoryEmpty）
            sp.set(**{"layer": "sm_compact", "status": "fallback_empty",
                      "tokens_before": before, "tokens_after": before})
            return None
        try:
            sm_content = sm.path.read_text(encoding="utf-8")
        except Exception:
            sp.set(**{"layer": "sm_compact", "status": "fallback_no_file"})
            return None
        sm_content = truncate_sm_for_compact(sm_content)
        last_summarized_message_id = getattr(sm, "last_summarized_message_id", None)
        kept = _session_memory_messages_to_keep(messages, last_summarized_message_id, cfg)
        if kept is None:
            sp.set(**{"layer": "sm_compact", "status": "fallback_missing_summary_anchor",
                      "tokens_before": before, "tokens_after": before})
            return None
        result = [
            create_compact_boundary_message(
                trigger="auto",
                pre_tokens=before,
                user_context=system or "",
                messages_summarized=len(source_messages),
                logical_parent_uuid=(
                    message_identity(source_messages[-1])
                    if source_messages
                    else None
                ),
            ),
            create_compact_summary_message(
                "[Compacted from session memory]\n" + sm_content,
                source="session_memory_compact",
            ),
        ]
        result += kept
        after = estimate(result, system)
        if after > auto_thr:                           # 压完仍超 → 回退 full（对齐 CC postCompact 仍超检查）
            sp.set(**{"layer": "sm_compact", "status": "fallback_still_over",
                      "tokens_before": before, "tokens_after": after})
            return None
        run_post_compact_cleanup(
            result,
            cfg=cfg,
            exclude_paths=_kept_intact_read_paths(kept),
            skill_agent_id=skill_agent_id,
            deferred_tool_state=deferred_tool_state,
            executor=executor,
            post_compact_sink=post_compact_sink,
            read_state=read_state,
            session_memory=sm,
        )
        success_attrs = _post_compact_success_attrs(
            result,
            system=system,
            cfg=cfg,
            auto_thr=auto_thr,
            post_compact_sink=post_compact_sink,
        )
        sp.set(**success_attrs)
        sp.set(**{"layer": "sm_compact", "status": "ok", "tokens_before": before, "tokens_after": after,
                  "ratio": round(before / max(1, after), 2), "compact_llm_calls": 0})   # ★ 零 LLM
        return result


# ──────────────────────────────────────────────
# 管线(真实 CC 流程)+ 地板基线
# ──────────────────────────────────────────────

def _durable_target_tokens(request_target_tokens: int, system: str = "",
                           clear_at_least: int = 0) -> int:
    """把总请求预算转换为 durable messages 预算。

    `system` 可能包含固定 system prompt 和 query context 前缀的预算文本。
    microcompact 只能清理 durable messages，所以目标值必须先扣掉这些静态 token。
    """
    static_tokens = estimate([], system)
    return max(0, request_target_tokens - static_tokens - clear_at_least)


def compact_pipeline(messages, system: str = "", cfg: CompactConfig = DEFAULT, target_tokens: int = None,
                     session_memory=None, idle_seconds: float = None,
                     skill_agent_id: str | None = None, deferred_tool_state=None,
                     executor=None, post_compact_sink=None, read_state=None):
    """当前流程：**两段独立触发**（2026-06-22 根据参考设计修正）。
      - auto_thr(=167K @200K):full_compact 触发线
      - micro_thr(=min(100K, auto_thr)):microcompact **更早、独立**触发线
    梯度:< micro_thr 不压；[micro_thr, auto_thr) 只 microcompact(免费清旧 tool_result)；
          ≥ auto_thr microcompact 清完仍超 → full_compact 摘要。
    target_tokens 显式给定时作 full 触发线(eval 用),micro 线据它按比例缩。"""
    with span("compact.pipeline", SpanKind.INTERNAL) as sp:
        before = estimate(messages, system)
        auto_thr = target_tokens if target_tokens is not None else auto_threshold(cfg)
        # micro 触发线:显式 target 时按 cfg 比例缩(micro_trigger/真实auto），否则取 min(100K, auto)
        if target_tokens is not None:
            ratio = cfg.microcompact_trigger / max(1, auto_threshold(cfg))
            micro_thr = min(int(target_tokens * ratio), auto_thr)
        else:
            micro_thr = micro_threshold(cfg, auto_thr)
        msgs = copy.deepcopy(messages)
        did_micro = did_full = False
        # ① microcompact 独立触发：到 micro_thr 就先清旧 tool_result（不调 LLM、免费）
        #   清到 micro_thr 以下一截（clear_at_least=清一大块、避免每轮抖动），不是清到 auto_thr
        #   ★ 缓存冷门控：缓存温热(idle<cache_cold_seconds)时**跳过**——
        #     我们无 cache_edits,温热 content-clear 会破 prompt 缓存→计费暴涨;连续任务 idle≈秒→恒跳过→full 接管。
        #     idle=None(未接 loop)视为冷→保旧行为(兼容现有单测)。
        cache_cold = idle_seconds is None or idle_seconds >= cfg.cache_cold_seconds
        if cache_cold and estimate(msgs, system) > micro_thr:
            micro_target = _durable_target_tokens(
                micro_thr,
                system,
                clear_at_least=cfg.microcompact_clear_at_least,
            )
            msgs = microcompact(msgs, cfg, target_tokens=micro_target)
            did_micro = True
        # ★ ①.5 session_memory_compact（step1b，闭 D13）：micro 清完仍超 → 先试 SM 文件当摘要（零 LLM）。
        #   成功则跳过 full（省一次摘要 LLM 调用）；失败（无 SM/空模板/压不够）回退 None → 走 full。
        did_sm = False
        if estimate(msgs, system) > auto_thr and session_memory is not None:
            sm_kwargs = {"skill_agent_id": skill_agent_id}
            if deferred_tool_state is not None:
                sm_kwargs["deferred_tool_state"] = deferred_tool_state
            if executor is not None:
                sm_kwargs["executor"] = executor
            if post_compact_sink is not None:
                sm_kwargs["post_compact_sink"] = post_compact_sink
            if read_state is not None:
                sm_kwargs["read_state"] = read_state
            sm_res = session_memory_compact(
                msgs,
                session_memory,
                system,
                cfg,
                auto_thr,
                **sm_kwargs,
            )
            if sm_res is not None:
                msgs = sm_res
                did_sm = True
        # ② full_compact 触发：SM-compact 没接管或没压够 → LLM 摘要
        if estimate(msgs, system) > auto_thr:
            full_kwargs = {"skill_agent_id": skill_agent_id}
            if deferred_tool_state is not None:
                full_kwargs["deferred_tool_state"] = deferred_tool_state
            if executor is not None:
                full_kwargs["executor"] = executor
            if post_compact_sink is not None:
                full_kwargs["post_compact_sink"] = post_compact_sink
            if read_state is not None:
                full_kwargs["read_state"] = read_state
            full_kwargs["auto_thr"] = auto_thr
            msgs = full_compact(msgs, system, cfg, **full_kwargs)
            did_full = True
        after = estimate(msgs, system)
        sp.set(**{"strategy": "cc_faithful", "tokens_before": before, "tokens_after": after,
                  "ratio": round(before / max(1, after), 2),
                  "micro_thr": micro_thr, "auto_thr": auto_thr,
                  "did_micro": did_micro, "did_sm": did_sm, "did_full": did_full,
                  "cache_cold": cache_cold, "idle_seconds": round(idle_seconds or 0, 1)})
        return msgs


def compact_naive(messages, target_tokens: int, system: str = "",
                  keep_head: int = 2, keep_tail: int = 8):
    """地板对照「保两端」截断:保留前 keep_head 条 + 后 keep_tail 条、丢中段,再截到 target_tokens。

    WHY 从 drop-oldest 升级为保两端（对照组合理性命根）：纯 drop-oldest 必丢最早的
    承重事实（任务头建立的接口签名/契约值）= 稻草人，pipeline「赢」它毫无意义。保头（任务+早期承重）
    + 保尾（最近态）才是「合理截断」真对手——和 pipeline 一样把上下文恒定、保住任务头与最近态，差别
    只在「中段承重怎么处理」（truncate 硬丢 vs pipeline 摘要）。pipeline 赢得了它，护城河才成立。
    ⚠ 偏离前版(就地标注):旧实现仅保 head[0](≈drop-oldest)、按预算无界保尾;现 keep_head/keep_tail
      双端可配。其它复用它当 floor 的 eval(compact_eval/run·resume·overflow·answer_quality、
      swebench/session_run)随之拿到「保两端」floor——正是 F7 想要的更强对照,是设计决策不是回归。

    不变量(对齐 test_validity 三条硬不变量):保两端的拼接处会产生两类违规,就地修复——
      ① head 末尾的悬空 tool_use:它的 tool_result 落在被丢中段 → 从 head 尾部剔除;
      ② tail 开头的孤儿 tool_result:它的 tool_use 落在被丢中段 → 从 tail 头部剔除。
    head[0] 恒为 user(任务头,不含 tool_use)→ 不会被①剔除 → 「以 user 开头」自然成立。
    """
    with span("compact.naive", SpanKind.INTERNAL) as sp:
        before = estimate(messages, system)
        n = len(messages)
        if n <= keep_head + keep_tail:
            # head 与 tail 已覆盖全部消息,无中段可丢;可能仍超预算,但保两端优先(floor 语义)
            head, tail = list(messages), []
        else:
            head = list(messages[:keep_head])
            tail = list(messages[n - keep_tail:])
            # 仍超预算 → 从 tail 最旧端(紧邻被丢中段)继续丢,优先保 head(承重)+ 最近态
            while estimate(head + tail, system) > target_tokens and len(tail) > 1:
                tail.pop(0)
        # 不变量修复:先各清各端再拼接(避免拼接后下标漂移)
        while head and _has_tool_use(head[-1]):     # ① 悬空 tool_use(result 已落被丢中段)
            head.pop()
        while tail and _is_tool_result(tail[0]):     # ② 孤儿 tool_result(tool_use 已落被丢中段)
            tail.pop(0)
        kept = head + tail
        after = estimate(kept, system)
        sp.set(**{"strategy": "naive", "tokens_before": before, "tokens_after": after,
                  "ratio": round(before / max(1, after), 2),
                  "keep_head": keep_head, "keep_tail": keep_tail,
                  "dropped_middle": max(0, n - len(kept))})
        return kept
