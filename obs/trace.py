"""obs/trace.py — 轻量 tracing 核心（对齐 OpenTelemetry GenAI 语义约定）。

设计推理见 learn-claude-docs/docs/architecture/13-observability.md：
  - 一条 append-only 事件流（Span 即一条事件），per-turn / span-tree 都是它的"投影"
  - 用 contextvars 持有"当前 span"，实现零侵入的父子因果传播
  - 线程安全 sink → 支持 subagent / 后台任务的并发写入
  - try/finally 保证 span 一定 close（防 span 泄漏）
  - 树重建容错：孤儿 span（找不到 parent）挂到 trace 根（防崩）

这是"极薄的对齐层"：字段名 / 属性键对齐 OTel，Day 2 可平滑导出到真 OTel / Langfuse。
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from . import otel as _otel


# ──────────────────────────────────────────────
# OTel 对齐的常量
# ──────────────────────────────────────────────

class SpanKind:
    INTERNAL = "INTERNAL"
    CLIENT = "CLIENT"     # LLM 调用按 OTel 惯例用 CLIENT
    TOOL = "TOOL"
    AGENT = "AGENT"


class SpanStatus:
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"


def _gen_id(n: int = 16) -> str:
    return uuid.uuid4().hex[:n]


def _public_attributes(attrs: dict) -> dict:
    return {
        key: value
        for key, value in dict(attrs or {}).items()
        if not str(key).startswith("tool.display.")
    }


# ──────────────────────────────────────────────
# Span：一个 span 即一条事件
# ──────────────────────────────────────────────

@dataclass
class Span:
    """字段对齐 OpenTelemetry。attributes 放 gen_ai.* 等键。"""
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    kind: str = SpanKind.INTERNAL
    start_ns: int = field(default_factory=time.time_ns)
    end_ns: Optional[int] = None
    status: str = SpanStatus.UNSET
    status_message: str = ""
    attributes: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_ns is None:
            return 0.0
        return round((self.end_ns - self.start_ns) / 1e6, 2)

    def set(self, **attrs) -> "Span":
        """链式设置属性：span.set(**{'gen_ai.usage.input_tokens': 100})"""
        self.attributes.update(attrs)
        return self

    def error(self, message: str = "") -> "Span":
        """显式标记本 span 失败（工具 rc≠0、is_error 等"非异常的失败"）。
        → Phoenix 瀑布里该 span 变红，成败一眼可见，无需点开。"""
        self.status = SpanStatus.ERROR
        if message:
            self.status_message = message[:200]
        return self

    def to_event(self) -> dict:
        d = asdict(self)
        d["attributes"] = _public_attributes(d.get("attributes", {}))
        d["duration_ms"] = self.duration_ms
        return d


# ──────────────────────────────────────────────
# Ambient context：零侵入父子传播的关键
# ──────────────────────────────────────────────

_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "current_span", default=None)
_current_trace: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_trace", default=None)


# ──────────────────────────────────────────────
# Sink：线程安全的 append-only 事件流
# ──────────────────────────────────────────────

class JsonlSink:
    """subagent / 后台线程都写它；用锁保证线程安全。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._mem: list[dict] = []
        self.dropped = 0   # count of spans swallowed by emit errors — observable after the run, never raises to caller

    def emit(self, span: Span) -> None:
        # Observability is a side-channel -- it must never crash the main run.
        # Root cause: surrogate chars from Windows stdin mis-encoding caused
        # json.dumps/write to raise UnicodeEncodeError and kill the whole run.
        # Fix: open with errors=replace (bad bytes become ?); wrap entire emit
        # in try/except, swallow errors and increment dropped -- losing one span
        # beats crashing the run.
        try:
            ev = span.to_event()
            with self._lock:
                self._mem.append(ev)
                with open(self.path, "a", encoding="utf-8", errors="replace") as f:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except Exception:
            self.dropped += 1   # 静默吞掉（不抛）；dropped 供事后排查"丢了几条 span"

    def events(self) -> list[dict]:
        with self._lock:
            return list(self._mem)


class TeeSink:
    """同一份 span 流，一头喂 JSONL（给 Phoenix 度量）、一头渲染到 stdout（给 CLI 演示）。

    WHY：pico 那种"看着 agent 干活"的
    体感纯靠 obs 层就能给、loop.py 一行不改——壳和 eval 共用同一套 span，只是渲染目标不同
    （eval→Phoenix，CLI→stdout）。"MEASURE 与演示同源"在此坐实，且不丢 trace。

    Sink 所有权：CLI/Session 启动时 set_sink(TeeSink(...)) 一次，整个
    会话生命周期归调用方掌管；Session.run 一律 trace=False、run_task 退化为"环境 sink 纯消费
    者"，不会自建 JsonlSink 把 TeeSink 冲掉。

    render_fn(span) -> str | None：返回 None＝该 span 不渲染（如 llm.call/compact.*）。注入式
    （不在 obs 里 import agent/cli），保持 obs 不依赖 CLI 层、分层干净。
    write_fn：默认 print（实际打到 stdout）；REPL 用 patch_stdout 包住 Session.run 协调与
    prompt_toolkit 的终端争用，write_fn 仍是普通 print。
    """

    def __init__(
        self,
        jsonl_path,
        render_fn,
        write_fn=None,
        *,
        render_start_fn=None,
        start_write_fn=None,
    ):
        self.jsonl = JsonlSink(jsonl_path)            # 复用现成 JSONL 落盘（含锁/内存镜像）
        self._render = render_fn
        self._write = write_fn or (lambda s: print(s, flush=True))
        self._render_start = render_start_fn
        self._write_start = start_write_fn or self._write

    def start(self, span: Span) -> None:
        if self._render_start is None:
            return
        try:
            line = self._render_start(span)
        except Exception:
            line = None
        if not line:
            return
        try:
            self._write_start(line)
        except Exception:
            pass

    def emit(self, span: Span) -> None:
        # Write to JSONL first — metric data must not be lost even if rendering raises.
        self.jsonl.emit(span)
        try:
            line = self._render(span)
        except Exception:
            line = None   # 渲染是演示糖，绝不能因它崩掉整个 run（trace 已落盘）。
        if line:
            self._write(line)

    def events(self) -> list[dict]:
        # 转发给内层 JsonlSink，保持 sink 接口一致（probe/测试可读回事件）。
        return self.jsonl.events()


_SINK: Optional[JsonlSink] = None


def set_sink(sink: JsonlSink) -> None:
    global _SINK
    _SINK = sink


def get_sink() -> JsonlSink:
    global _SINK
    if _SINK is None:
        _SINK = JsonlSink(Path(".traces") / "events.jsonl")
    return _SINK


# ──────────────────────────────────────────────
# span()：开/关一个 span，自动挂父子
# ──────────────────────────────────────────────

@contextlib.contextmanager
def span(name: str, kind: str = SpanKind.INTERNAL, **attributes):
    """开一个 span。自动挂到当前 span 之下；退出时 close 并 emit。

    try/finally 保证异常时也 close（status=ERROR），防 span 泄漏。
    """
    parent = _current_span.get()
    trace_id = _current_trace.get() or _gen_id(32)
    sp = Span(
        name=name,
        trace_id=trace_id,
        span_id=_gen_id(16),
        parent_span_id=parent.span_id if parent else None,
        kind=kind,
        attributes=dict(attributes),
    )
    t_token = _current_trace.set(trace_id)
    s_token = _current_span.set(sp)
    with _otel.otel_span(name, kind) as _otsp:
        try:
            start = getattr(get_sink(), "start", None)
            if callable(start):
                start(sp)
        except Exception:
            pass
        try:
            yield sp
            if sp.status == SpanStatus.UNSET:
                sp.status = SpanStatus.OK
                _otel.mark_ok(_otsp)
            elif sp.status == SpanStatus.ERROR:
                # 块内显式标了失败（如工具 rc≠0）→ 同步给 OTel，Phoenix 里该 span 变红
                _otel.mark_error(_otsp, sp.status_message)
        except Exception as e:
            sp.status = SpanStatus.ERROR
            sp.status_message = f"{type(e).__name__}: {e}"[:200]
            _otel.mark_error(_otsp, sp.status_message)
            raise
        finally:
            sp.end_ns = time.time_ns()
            get_sink().emit(sp)
            _otel.apply_attributes(_otsp, _public_attributes(sp.attributes))
            _current_span.reset(s_token)
            _current_trace.reset(t_token)


def current_trace_id() -> Optional[str]:
    return _current_trace.get()


def annotate(**attrs) -> None:
    """给**当前** span 追加属性。

    工具实现可在 dispatch 打开的 tool.* span 内补充结构化字段
    （如 bash 的 exit_code / stderr_head / command_kind），无需把 span 传来传去。
    无当前 span 时静默忽略（防呆，便于工具单测）。
    """
    sp = _current_span.get()
    if sp is not None:
        sp.set(**attrs)


def mark_current_error(message: str = "") -> None:
    """把**当前** span 标为失败（工具内部检测到 rc≠0 等"非异常失败"时调用）。
    无当前 span 时静默忽略。配合 span() 退出时同步给 OTel → Phoenix 里变红。"""
    sp = _current_span.get()
    if sp is not None:
        sp.error(message)


def capture_context():
    """抓当前 contextvars 上下文，给后台线程用，让其 span 落在同一 trace：

        ctx = capture_context()
        threading.Thread(target=lambda: ctx.run(work)).start()
    """
    return contextvars.copy_context()


# ──────────────────────────────────────────────
# 投影：从事件流重建 span 树（容错）
# ──────────────────────────────────────────────

def reconstruct_tree(events: list[dict]) -> list[dict]:
    """扁平事件流 → span 树。孤儿（找不到 parent）挂到根，绝不崩。"""
    by_id = {e["span_id"]: {**e, "children": []} for e in events}
    roots: list[dict] = []
    for e in events:
        node = by_id[e["span_id"]]
        pid = e.get("parent_span_id")
        if pid and pid in by_id:
            by_id[pid]["children"].append(node)
        else:
            roots.append(node)  # 根 span 或孤儿

    def _sort(nodes: list[dict]) -> None:
        nodes.sort(key=lambda n: n.get("start_ns", 0))
        for n in nodes:
            _sort(n["children"])

    _sort(roots)
    return roots


def render_tree(events: list[dict]) -> str:
    """把 span 树渲染成文本（CLI/调试用；Day 2 换成 HTML 时间线）。"""
    roots = reconstruct_tree(events)
    lines: list[str] = []

    def walk(node: dict, depth: int) -> None:
        a = node.get("attributes", {})
        extra = []
        if "gen_ai.usage.input_tokens" in a or "gen_ai.usage.output_tokens" in a:
            extra.append(f"tok={a.get('gen_ai.usage.input_tokens', 0)}"
                         f"+{a.get('gen_ai.usage.output_tokens', 0)}")
        if "cost_usd" in a:
            extra.append(f"${a['cost_usd']}")
        if "tool.name" in a:
            extra.append(a["tool.name"])
        mark = "x" if node.get("status") == SpanStatus.ERROR else " "
        lines.append(
            f"{'  ' * depth}[{mark}] {node['name']} "
            f"<{node.get('kind', '')}> {node.get('duration_ms', 0)}ms "
            f"{' '.join(extra)}".rstrip()
        )
        for c in node["children"]:
            walk(c, depth + 1)

    for r in roots:
        walk(r, 0)
    return "\n".join(lines)
