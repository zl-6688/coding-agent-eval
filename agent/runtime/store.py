"""SessionStore —— transcript 整文件原子覆盖落盘 + 按 id resume。

落盘模型（原子覆盖写，原因见下）：
  **整文件原子覆盖写**，不是 append。
    save(id, messages) = 写 <id>.jsonl.tmp 逐行 JSON → os.replace 到 <id>.jsonl。

WHY 覆盖而非 append（诚实偏离登记）：
  - run_task 返回的是**全量** messages（loop.py:81,84）。交给 append 增量会自相矛盾：
    无"本轮新增"游标 → N² 重复落盘 / resume 灌出重复历史。
  - loop 内会**原地压缩** messages（loop.py:99-106，list 变短、历史被重写）。append-only
    与"压缩后历史变了"冲突（盘上旧历史 vs 内存压缩后不一致）；覆盖写直接落压缩后 list、天然兼容。
  - 原子 replace 给崩溃韧性：写 tmp + os.replace，半写只污染 tmp、原文件完好。
  ⚠ 偏离 [[runtime-gap-audit]] §6.2 的 append-only 建议。代价：每轮写 O(全历史) 而非 O(新增)。
    我们规模（eval + 交互，非百万轮）完全够。TODO：规模逼近百万轮再上 append + 增量游标。

落盘**线性 messages 数组**（一行一消息），不建 CC 的 parentUuid 消息树（反膨胀，
[[runtime-gap-audit]] §5）：续作是线性单链，文件顺序即时序。
"""

import json
import os
import uuid
from pathlib import Path

from .messages import ensure_message_uuids
from .project import Project


def _first_text_block(content) -> str:
    """从 list 形式的 content 取首个 text 块文本（用于会话标题）。非 list/无 text 返回 ""。"""
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                return b["text"]
            if isinstance(b, str):
                return b
    return ""


def _to_jsonable(o):
    """忠实序列化 Anthropic SDK 块对象（真实 e2e 发现 messages 非纯 dict）。

    真实 run_task 的 messages 里 assistant content 是 SDK 块对象（ThinkingBlock/TextBlock/
    ToolUseBlock…，deepseek-v4 有 thinking 块），不是纯 dict。json.dumps 直接抛 TypeError。
    用 model_dump(mode="json") 转成 wire-format dict（{"type":"thinking",...}/{"type":"tool_use",
    ...}），resume 读回就是这些 dict，喂回 run_task(initial_messages=) 时 SDK 接受 dict 形式
    content 块 → 往返保真。

    ⚠ 否决 default=str：str 把块变成垃圾字符串、resume 读回非
    结构化 → 坏续作。这里走 model_dump 忠实结构化；真未知类型仍抛错（别有损吞掉、早暴露）。
    """
    md = getattr(o, "model_dump", None)      # pydantic v2 / Anthropic SDK 块
    if callable(md):
        return md(mode="json")               # 忠实 wire-format dict、JSON 安全
    d = getattr(o, "dict", None)             # 兜底老式 pydantic v1
    if callable(d):
        return d()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


def _is_nonpersistent_relevant_memory(message) -> bool:
    """Mirror external CC: memory attachments stay in RAM but not transcript JSONL."""

    if not isinstance(message, dict) or message.get("type") != "attachment":
        return False
    attachment = message.get("attachment")
    return (
        isinstance(attachment, dict)
        and attachment.get("type") == "relevant_memories"
    )


class SessionStore:
    """按 project 归档 transcript：new_session_id / save（覆盖写）/ resume / list_ids。"""

    def __init__(self, project: Project):
        self.project = project

    def _path(self, session_id: str) -> Path:
        return self.project.sessions_dir / f"{session_id}.jsonl"

    def new_session_id(self) -> str:
        """uuid4().hex（对齐 CC randomUUID: bootstrap/state.ts:447）。
        ⚠ 偏离：CC 带连字符 36 字符，我们 32 hex 无连字符（文件名安全唯一即可，
        功能无碍；将来若要和 CC transcript 互读工具会差一层，标注备查）。"""
        return uuid.uuid4().hex

    def save(self, session_id: str, messages: list) -> None:
        """整文件原子覆盖写：tmp 逐行 JSON → os.replace（崩溃韧性，见模块 docstring WHY）。

        原子性：os.replace 在同一文件系统上是原子的（Windows/POSIX 皆然）。进程在写 tmp
        途中被杀 → 原 <id>.jsonl 完好、只剩个残 tmp（下次 save 覆盖之）。
        """
        ensure_message_uuids(
            messages,
            migration_namespace=session_id,
            drop_legacy=True,
        )
        self.project.sessions_dir.mkdir(parents=True, exist_ok=True)
        final = self._path(session_id)
        tmp = final.with_suffix(final.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for m in messages:
                if _is_nonpersistent_relevant_memory(m):
                    continue
                # ensure_ascii=False 留中文可读。default=_to_jsonable 忠实序列化 SDK 块对象
                # （真实 e2e 发现 assistant content 是 ThinkingBlock/ToolUseBlock 等、非纯 dict）：
                # model_dump(mode="json") 转 wire-format dict、往返保真。default=str 有损已否决（见 _to_jsonable）。
                f.write(json.dumps(m, ensure_ascii=False, default=_to_jsonable) + "\n")
            f.flush()
            os.fsync(f.fileno())   # 落盘后再 replace，进一步收紧崩溃窗口。
        os.replace(tmp, final)     # 原子覆盖：要么旧文件、要么新文件，无半写中间态。

    def resume(self, session_id: str) -> list:
        """逐行读回成线性 messages 数组（出现顺序即时序，无需 parentUuid 链重建）。
        文件不存在 → 返回空 list（让调用方当新会话起；不抛，便于 eval 枚举容错）。"""
        path = self._path(session_id)
        if not path.exists():
            return []
        messages = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    messages.append(json.loads(line))
        ensure_message_uuids(
            messages,
            migration_namespace=session_id,
            drop_legacy=True,
        )
        return messages

    def _transcript_paths(self):
        """本 project 的 transcript 文件（<id>.jsonl），排除 <id>.trace.jsonl（trace 旁路）。
        notes 是 .notes.md、glob *.jsonl 本就不命中。"""
        d = self.project.sessions_dir
        if not d.exists():
            return []
        return [p for p in d.glob("*.jsonl") if not p.name.endswith(".trace.jsonl")]

    def list_ids(self) -> list[str]:
        """枚举本 project 已落盘的 session id（供 eval 脚本，非给人选菜单）。
        排除 <id>.trace.jsonl（否则 .stem='<id>.trace' 是伪 id）。"""
        return sorted(p.stem for p in self._transcript_paths())

    def list_sessions(self) -> list[dict]:
        """枚举会话给人选（6e /resume 选择器 + ace -r）：每个 {id, title, mtime}，按 mtime 倒序。

        title = 读 transcript 第一行（首条 user 消息）截断（免 LLM）。
        ⚠ 偏离 CC：CC 用 LLM 生成会话摘要标题；我们用**首条 user 消息截断**（零成本、无 LLM）。
        TODO：要更好的标题可升级为 LLM 摘要（与记忆 fork 同款），但 demo 选择器够用。
        """
        out = []
        for p in self._transcript_paths():
            out.append({"id": p.stem, "title": self._first_user_title(p), "mtime": p.stat().st_mtime})
        out.sort(key=lambda d: d["mtime"], reverse=True)   # 最近在前
        return out

    @staticmethod
    def _first_user_title(path, limit: int = 60) -> str:
        """取 transcript 首条 user 消息的文本作标题（截断）。读不出给占位。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = json.loads(line)
                    if m.get("role") == "user":
                        c = m.get("content")
                        text = c if isinstance(c, str) else _first_text_block(c)
                        text = " ".join((text or "").split())   # 压平换行/多空格
                        if text:
                            return text[:limit] + ("…" if len(text) > limit else "")
                    break   # 首条非空消息若不是 user（罕见），不再往下找，给占位
        except (OSError, ValueError):
            pass
        return "(无标题)"
