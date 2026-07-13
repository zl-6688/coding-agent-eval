"""test_trace.py — characterization tests for obs/trace.py.

Locks:
  - reconstruct_tree orphan handling (documented resilience feature)
  - span emission via context manager (agent.run / agent.turn shapes)
  - JsonlSink in-memory events() correctness
"""
import time
import uuid
import pytest

from obs.trace import (
    SpanKind, SpanStatus, Span,
    JsonlSink, TeeSink, reconstruct_tree, render_tree,
    span, set_sink, get_sink,
)
from conftest import CaptureSink


# ── reconstruct_tree ───────────────────────────────────────────────────────

def _make_event(name: str, span_id: str, parent_id=None, start_ns: int = 0) -> dict:
    """Build a minimal span event dict for reconstruct_tree input."""
    return {
        "name": name,
        "span_id": span_id,
        "parent_span_id": parent_id,
        "trace_id": "trace1",
        "start_ns": start_ns,
        "end_ns": start_ns + 1_000_000,
        "status": "OK",
        "kind": SpanKind.INTERNAL,
        "attributes": {},
    }


def test_reconstruct_tree_normal_structure():
    """Parent-child events reconstruct into the correct tree."""
    events = [
        _make_event("root", "aaa", parent_id=None, start_ns=0),
        _make_event("child1", "bbb", parent_id="aaa", start_ns=1),
        _make_event("child2", "ccc", parent_id="aaa", start_ns=2),
        _make_event("grandchild", "ddd", parent_id="bbb", start_ns=3),
    ]
    roots = reconstruct_tree(events)

    assert len(roots) == 1, "exactly one root expected"
    assert roots[0]["name"] == "root"
    assert len(roots[0]["children"]) == 2

    child1 = next(c for c in roots[0]["children"] if c["name"] == "child1")
    assert len(child1["children"]) == 1
    assert child1["children"][0]["name"] == "grandchild"


def test_reconstruct_tree_orphan_attached_to_root():
    """Orphan span (unknown parent_id) is attached to root instead of crashing.

    This is a documented resilience feature per obs/trace.py module docstring.
    The tree must reconstruct without exception even with missing parents.
    """
    events = [
        _make_event("real_root", "aaa", parent_id=None),
        _make_event("real_child", "bbb", parent_id="aaa"),
        # orphan: parent_id points to a non-existent span
        _make_event("orphan", "ccc", parent_id="NONEXISTENT"),
    ]
    # Must not raise
    roots = reconstruct_tree(events)

    all_names = {node["name"] for node in roots}
    # Orphan appears as a root-level node (alongside real_root)
    assert "orphan" in all_names, f"orphan should be in roots; got {all_names}"
    assert "real_root" in all_names


def test_reconstruct_tree_multiple_orphans():
    """Multiple orphans all attach to root without crashing."""
    events = [
        _make_event("orphan_a", "x1", parent_id="missing1"),
        _make_event("orphan_b", "x2", parent_id="missing2"),
    ]
    roots = reconstruct_tree(events)
    assert len(roots) == 2
    names = {r["name"] for r in roots}
    assert names == {"orphan_a", "orphan_b"}


def test_reconstruct_tree_empty():
    """Empty events list returns empty list (no crash)."""
    assert reconstruct_tree([]) == []


def test_reconstruct_tree_children_sorted_by_start_ns():
    """Children within each node are sorted by start_ns (chronological order)."""
    events = [
        _make_event("root", "r", parent_id=None, start_ns=0),
        _make_event("late_child", "c2", parent_id="r", start_ns=200),
        _make_event("early_child", "c1", parent_id="r", start_ns=100),
    ]
    roots = reconstruct_tree(events)
    children = roots[0]["children"]
    assert children[0]["name"] == "early_child"
    assert children[1]["name"] == "late_child"


# ── span context manager ───────────────────────────────────────────────────

def test_span_emits_to_sink(capture_sink):
    """span() context manager emits a completed Span to the active sink."""
    with span("test.unit", SpanKind.INTERNAL, foo="bar"):
        pass

    events = capture_sink.events()
    assert len(events) == 1
    ev = events[0]
    assert ev["name"] == "test.unit"
    assert ev["attributes"]["foo"] == "bar"
    assert ev["status"] == SpanStatus.OK


def test_span_parent_child_propagation(capture_sink):
    """Nested span() calls produce correct parent_span_id linkage."""
    with span("parent", SpanKind.AGENT) as parent_sp:
        with span("child", SpanKind.INTERNAL) as child_sp:
            pass

    events = capture_sink.events()
    ev_parent = next(e for e in events if e["name"] == "parent")
    ev_child = next(e for e in events if e["name"] == "child")

    assert ev_child["parent_span_id"] == ev_parent["span_id"]
    assert ev_parent["parent_span_id"] is None


def test_span_error_on_exception(capture_sink):
    """An exception inside span() marks the span as ERROR and re-raises."""
    with pytest.raises(ValueError):
        with span("errored", SpanKind.INTERNAL):
            raise ValueError("boom")

    events = capture_sink.events()
    assert events[-1]["status"] == SpanStatus.ERROR
    assert "ValueError" in events[-1]["status_message"]


def test_span_set_attributes(capture_sink):
    """span.set() updates the span's attributes at any point before close."""
    with span("annotated", SpanKind.INTERNAL) as sp:
        sp.set(key1="v1", key2=42)

    events = capture_sink.events()
    assert events[-1]["attributes"]["key1"] == "v1"
    assert events[-1]["attributes"]["key2"] == 42


def test_span_display_attributes_do_not_persist_to_event():
    sp = Span(
        name="tool.bash",
        trace_id="trace",
        span_id="span",
        parent_span_id=None,
        kind=SpanKind.TOOL,
        attributes={
            "tool.name": "bash",
            "tool.display.command": "echo SECRET",
            "tool.command_summary": "str chars=11",
        },
    )

    event = sp.to_event()

    assert event["attributes"] == {
        "tool.name": "bash",
        "tool.command_summary": "str chars=11",
    }


def test_span_calls_optional_sink_start_before_emit(tmp_path):
    order = []

    class _StartSink(JsonlSink):
        def start(self, sp):
            order.append(("start", sp.name, sp.end_ns))

        def emit(self, sp):
            order.append(("emit", sp.name, sp.end_ns))
            super().emit(sp)

    sink = _StartSink(tmp_path / "start.jsonl")
    set_sink(sink)

    with span("tool.bash", SpanKind.TOOL, **{"tool.name": "bash"}):
        pass

    assert order[0] == ("start", "tool.bash", None)
    assert order[1][0:2] == ("emit", "tool.bash")
    assert order[1][2] is not None
    assert len(sink.events()) == 1


def test_tee_sink_start_renders_without_writing_jsonl(tmp_path):
    started = []
    finished = []
    sink = TeeSink(
        tmp_path / "tee.jsonl",
        lambda _span: "finish",
        write_fn=finished.append,
        render_start_fn=lambda _span: "start",
        start_write_fn=started.append,
    )
    sp = Span(
        name="tool.bash",
        trace_id="trace",
        span_id="span",
        parent_span_id=None,
        kind=SpanKind.TOOL,
        attributes={"tool.name": "bash"},
    )

    sink.start(sp)

    assert started == ["start"]
    assert finished == []
    assert sink.events() == []

    sink.emit(sp)

    assert finished == ["finish"]
    assert len(sink.events()) == 1


# ── JsonlSink ─────────────────────────────────────────────────────────────

def test_jsonl_sink_events_returns_all(tmp_path):
    """JsonlSink.events() returns every emitted span."""
    sink = JsonlSink(tmp_path / "test.jsonl")
    set_sink(sink)

    with span("s1", SpanKind.INTERNAL):
        pass
    with span("s2", SpanKind.INTERNAL):
        pass

    events = sink.events()
    names = [e["name"] for e in events]
    assert "s1" in names
    assert "s2" in names


def test_jsonl_sink_dropped_on_bad_emit(tmp_path):
    """JsonlSink silently counts dropped spans instead of crashing (B4 fix).

    Quirk: if emit fails (e.g., unicode surrogate), dropped counter increments
    and no exception escapes.  This protects the agent loop from observability bugs.
    """
    sink = JsonlSink(tmp_path / "test2.jsonl")
    # force emit to fail by monkey-patching to_event on a span
    sp = Span(name="bad", trace_id="t", span_id="s", parent_span_id=None)

    original_to_event = sp.to_event
    sp.to_event = lambda: (_ for _ in ()).throw(RuntimeError("forced emit failure"))

    before = sink.dropped
    sink.emit(sp)
    assert sink.dropped == before + 1, "failed emit must increment dropped, not raise"


# ── render_tree (smoke) ────────────────────────────────────────────────────

def test_render_tree_smoke():
    """render_tree produces non-empty string for a simple event set."""
    events = [
        _make_event("root", "r", parent_id=None, start_ns=0),
        _make_event("child", "c", parent_id="r", start_ns=1),
    ]
    output = render_tree(events)
    assert "root" in output
    assert "child" in output
