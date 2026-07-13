"""Session Memory —— 会话级活文档（10 段 md），对齐 CC `services/SessionMemory`。

参考 Claude Code 公开技术资料的设计要点：
  - 周期性 **forked-LLM** 维护一个 10 段 Markdown 笔记文件（`sessionMemory.ts:318`）。
  - 触发三阈值（`sessionMemoryUtils.ts:32-36`）：init 10K context / 自上次增长 5K / tool 调用 3 次。
    token 阈值**始终必需**（CC 注释强调），再叠加（tool 够 OR 最后一轮无 tool＝自然断点）。
  - 文件预算（`prompts.ts:8-9`）：总 12K token、每段 2K。
  - 它是**记忆↔压缩的接缝**：写好的文件给 SM 压缩桥（step 1b）当免-LLM 摘要。

⚠ 偏离诚实标注：
  - 段描述**中文适配**：段标题保留英文（稳定锚点 + 可溯源 CC），italic 描述中文化（适配中文 agent）。
  - 触发用 **tool_use id 集合 + token 自愈**抗压缩（不依赖 message-uuid）：tool 计数数"新出现的
    tool_use id"，token 基准在 messages 缩水时自愈重置 + on_compacted 主动钩子——对齐 CC「找不到锚点
    就数全量、永不 permanently disable」。
  - **压缩边界锚点**（CC lastSummarizedMessageId）step1b 引入，届时移植 orphan 保护 + adjust。
  - 写入是真实 fork：fork 复制主对话作前缀，deepseek 自动前缀缓存大概率命中该前缀（best-effort），故非每次全量；成本归因到 memory.fork span（usage 的 prompt_cache_hit/miss 区分）。
  - **fork 生成笔记文本、harness 直接落盘**（非 CC 的 fork 用 Edit 工具写）：① CC 的工具写 ~/.claude/
    独立位置、不受 workspace 沙箱限制；我们的 agent 工具受 LocalExecutor 的 WORKDIR 沙箱限制，SM 文件
    在 workspace 外 → 用工具写会"路径越权"（真实端到端实测）。② 故让 fork **纯文本生成完整笔记**、extract
    直接 Path.write_text 落盘——绕过沙箱，对齐 CC「SM 写独立位置、不受 workspace 限制」的语义。

未实现（backlog）：manuallyExtractSessionMemory、段预算动态回灌 + truncateForCompact。
"""

import re
from dataclasses import dataclass
from pathlib import Path

from ..context import compact
from .forked_agent import run_forked_agent

# 10 段模板 —— 对齐 CC DEFAULT_SESSION_MEMORY_TEMPLATE（prompts.ts:11-41）。
# 段标题英文（锚点/溯源），italic 描述中文（适配）。fork 填描述**下方**内容、不动标题与描述行。
SESSION_MEMORY_TEMPLATE = """# Session Title
_一个简短独特的 5-10 词会话标题，信息密度高、无废话_

# Current State
_现在正在做什么？尚未完成的待办、紧接着的下一步_

# Task specification
_用户要求构建什么？任何设计决策或解释性背景_

# Files and Functions
_重要文件有哪些？简述各自包含什么、为何相关_

# Workflow
_通常按什么顺序跑哪些 bash 命令？输出如何解读（若不显然）_

# Errors & Corrections
_遇到的错误及修复方式。用户纠正了什么？哪些方法失败了、不应再试_

# Codebase and System Documentation
_重要的系统组件有哪些？它们如何工作/协作_

# Learnings
_什么有效、什么无效、要避免什么？不要与其他段重复_

# Key results
_若用户要了具体产出（答案、表格、文档），在此原样保留确切结果_

# Worklog
_逐步记录尝试/完成了什么，每步极简_
"""

_TITLE_ANCHOR = "# Session Title"


@dataclass
class SessionMemoryConfig:
    min_tokens_to_init: int = 10_000          # CC minimumMessageTokensToInit
    min_tokens_between_update: int = 5_000    # CC minimumTokensBetweenUpdate
    tool_calls_between_updates: int = 3       # CC toolCallsBetweenUpdates
    max_section_tokens: int = 2_000           # CC MAX_SECTION_LENGTH
    max_total_tokens: int = 12_000            # CC MAX_TOTAL_SESSION_MEMORY_TOKENS
    fork_max_tokens: int = 16_384             # fork 输出上限：须 ≥ max_total_tokens 否则整份笔记被截（旧值 4096 太小）
    max_turns: int = 2                        # fork 纯文本生成笔记，一轮即可（留 2 容错）


DEFAULT_SM_CONFIG = SessionMemoryConfig()


def build_update_prompt(current_notes: str, notes_path: str, cfg: SessionMemoryConfig) -> str:
    """对齐 CC getDefaultUpdatePrompt（prompts.ts:43-81）的**语义**（更新会话笔记），中文适配。
    ⚠ 偏离 CC 实现：让 fork **纯文本输出完整笔记**（不调工具）、由 harness 落盘——见模块 docstring。"""
    return f"""重要：本条消息及其指令**不属于**真实用户对话。不要在笔记里提及"记笔记/会话笔记提取/这些更新指令"。

基于上面的用户对话（**排除**本提取指令消息、system prompt、CLAUDE.md、过往会话摘要），更新会话笔记。

当前笔记内容（在此基础上更新，保持结构）：
<current_notes>
{current_notes}
</current_notes>

你的唯一任务：**直接输出**更新后的完整笔记内容，不要调用任何工具、不要加任何额外解释、不要用代码块包裹。

写出规则：
- 输出**完整笔记**：全部 10 个段标题（# 开头）+ 各段 italic _描述_ 行 + 你填的内容。
- 段标题与 italic _描述_ 行**原样保留**（描述行是模板指令，不要改/删）。
- 只在每段 italic 描述行**下方**填实际内容；某段无实质新信息就留空（仅保留标题+描述行）。
- 写**详细、信息密集**的内容：文件路径、函数名、错误信息、确切命令、技术细节。
- **务必更新 "Current State"** 反映最新工作（压缩后连续性靠它）。
- 每段控制在 ~{cfg.max_section_tokens} token 内，总计 ~{cfg.max_total_tokens // 1000}K token 内。

现在直接输出完整笔记（以 "{_TITLE_ANCHOR}" 开头）："""


def _extract_notes_body(text: str) -> str:
    """从 fork 的文本回复里取笔记主体：去 markdown 代码块包裹、从首个段标题起截（防 fork 加前言）。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    idx = t.find(_TITLE_ANCHOR)
    if idx > 0:
        t = t[idx:]
    return t.strip()


def _all_tool_use_ids(messages: list) -> set:
    """当前所有 assistant tool_use 的 id 集合。tool 计数靠它做差（数"新 id"）——
    tool_use 自带 id、压缩不会让旧 id 复活，故天然抗压缩（替代会被压缩错位的 index 锚点）。"""
    ids = set()
    for m in messages:
        if m.get("role") == "assistant":
            for b in compact._blocks(m):
                if compact._block_type(b) == "tool_use":
                    tid = compact._block_attr(b, "id")
                    if tid:
                        ids.add(tid)
    return ids


def _last_assistant_has_tools(messages: list) -> bool:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return any(compact._block_type(b) == "tool_use" for b in compact._blocks(m))
    return False


class SessionMemory:
    """会话级活文档：周期性 forked-LLM 生成 10 段笔记、harness 落盘。调用方持有实例、每轮问 should_extract。"""

    def __init__(self, path, cfg: SessionMemoryConfig = DEFAULT_SM_CONFIG):
        self.path = Path(path)
        self.cfg = cfg
        # 触发状态（对齐 CC sessionMemoryUtils 的模块状态，这里实例化）。两者都抗压缩（压缩后锦点不漂移）：
        self._initialized = False
        self._tokens_at_last = 0          # 上次提取时的 context token（算"增长"；压缩后自愈/钩子重置）
        self._seen_tool_ids = set()       # 上次提取时已见的 tool_use id（数"新 id"＝抗压缩 tool 计数）
        self._extracting = False          # in-progress flag: lets wait_for_extraction block step1b from reading a half-written notes file
        self._last_summarized_message_id = None

    @property
    def last_summarized_message_id(self):
        """Return the compact anchor produced by the latest safe extraction."""

        return self.get_last_summarized_message_id()

    def get_last_summarized_message_id(self):
        """Return the message id covered by the latest successful SM extraction."""

        return self._last_summarized_message_id

    def set_last_summarized_message_id(self, message_id) -> None:
        """Set the SM compact anchor to a known durable-message id."""

        self._last_summarized_message_id = str(message_id) if message_id else None

    def clear_last_summarized_message_id(self) -> None:
        """Clear the SM compact anchor after compaction rewrites history."""

        self._last_summarized_message_id = None

    def _record_last_summarized_message_id_if_safe(self, messages: list) -> None:
        """Advance the SM compact anchor only at a safe conversation boundary.

        Claude Code avoids setting the anchor when the latest assistant turn has
        tool calls, because a future tool_result could otherwise be separated
        from its tool_use.  P0-A gives every ACE durable message a stable UUID;
        legacy ids are accepted only while old transcripts are migrated.
        """

        if not messages or _last_assistant_has_tools(messages):
            return
        compact.ensure_runtime_message_ids(messages)
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if compact.is_compact_boundary_message(message):
                continue
            message_id = compact.message_identity(message)
            if message_id:
                self.set_last_summarized_message_id(message_id)
            return

    def _ensure_file(self):
        # 独占创建（对齐 CC setupSessionMemoryFile 的 wx＝O_CREAT|O_EXCL，sessionMemory.ts:194-200）：
        # 只有"确实新建"才写模板、已存在则跳过——避免竞态下两次 extract 都写模板把已有笔记冲掉。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, "x", encoding="utf-8") as f:
                f.write(SESSION_MEMORY_TEMPLATE)
        except FileExistsError:
            pass

    def is_empty(self) -> bool:
        """文件仍是空模板（无实质内容）→ SM 压缩桥据此回退 legacy（对齐 CC isSessionMemoryEmpty）。"""
        if not self.path.exists():
            return True
        return self.path.read_text(encoding="utf-8").strip() == SESSION_MEMORY_TEMPLATE.strip()

    def should_extract(self, messages: list) -> bool:
        """对齐 CC shouldExtractMemory（sessionMemory.ts:134-181）。锚点抗压缩（压缩不会让旧 id 复活）。

        ⚠ 与 CC 的细微差异：CC 在返 true 时**顺手推进触发锚点**；我们把锚点推进放在
        extract() 末尾 → **本方法无副作用，调用方一旦 should_extract 返 true 就必须紧跟 extract()**，
        否则触发锚点不推进、下次重复计数。loop 接入时保证这个调用契约。
        """
        cur = compact.estimate(messages)
        if not self._initialized:
            if cur < self.cfg.min_tokens_to_init:
                return False
            self._initialized = True
        # 自愈：messages 缩水（compaction 发生过）→ 绝对 token 基准失真 → 重置到当前，
        #   否则 cur-_tokens_at_last 转负、恒 < 阈值 → SM 第一次压缩后永久静默停火。on_compacted 是主动版。
        if cur < self._tokens_at_last:
            self._tokens_at_last = cur
        if (cur - self._tokens_at_last) < self.cfg.min_tokens_between_update:
            return False
        new_tools = len(_all_tool_use_ids(messages) - self._seen_tool_ids)   # 数"新 id"，抗压缩
        enough_tools = new_tools >= self.cfg.tool_calls_between_updates
        no_tools_last = not _last_assistant_has_tools(messages)              # 自然断点
        return enough_tools or no_tools_last

    def extract(self, messages: list, system: str = ""):
        """跑 forked-LLM **生成**完整会话笔记文本，由 harness 直接落盘（绕过受 WORKDIR 限制的工具沙箱，
        因为 SM 文件是 harness 记忆、在 workspace 外）。返回 ForkResult。_extracting 置位期间 wait_for_extraction 会等待。"""
        self._extracting = True
        try:
            self._ensure_file()
            current = self.path.read_text(encoding="utf-8")
            prompt = build_update_prompt(current, str(self.path), self.cfg)
            # fork 不给工具：纯文本生成完整笔记（对齐 CC「SM 写独立位置、不受 workspace 限制」的语义）
            res = run_forked_agent(prompt, messages, system=system,
                                   allowed_tools=set(), max_turns=self.cfg.max_turns,
                                   max_tokens=self.cfg.fork_max_tokens, label="session_memory")
            notes = _extract_notes_body(res.final_text)
            if notes and _TITLE_ANCHOR in notes:        # 有有效笔记才落盘（防 fork 空/跑偏覆盖已有）
                self.path.write_text(notes, encoding="utf-8")
                self._record_last_summarized_message_id_if_safe(messages)
            # 提取后推进触发锚点（对齐 CC recordExtractionTokenCount）。两个锚点都抗压缩。
            self._tokens_at_last = compact.estimate(messages)
            self._seen_tool_ids = _all_tool_use_ids(messages)
            return res
        finally:
            self._extracting = False

    def wait_for_extraction(self) -> bool:
        """对齐 CC waitForSessionMemoryExtraction（sessionMemoryUtils.ts:89-105）的接口。
        ⚠ 偏离：CC 15s 轮询等**异步**提取；我们 fork **同步**、extract 返回即写完 → 退化 no-op。
        留接口给 step1b 压缩桥（压缩前调它防读半写文件）。返回是否已就绪（同步下恒 True）。"""
        return not self._extracting

    def on_compacted(self, messages: list):
        """压缩管线压缩后调用：把 token 基准重置到压缩后基线（弥补 should_extract 内自感的被动性）。
        tool 计数用 id 集合天然抗压缩（压缩留存的 tool 仍在 _seen_tool_ids、只有新 id 算新），无需在此处理。
        should_extract 内还有 `cur < _tokens_at_last` 的自愈兜底——即使调用方忘了调本钩子也不会永久停火。"""
        self._tokens_at_last = compact.estimate(messages)
        self.clear_last_summarized_message_id()
