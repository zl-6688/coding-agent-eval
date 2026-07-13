"""test_compact_pipeline.py — compact_pipeline cache-cold gate + full_compact trigger.

Locks observable behavior of the idle_seconds branch (the e2r finding) and the
pipeline strategy selection BEFORE the refactor.
"""
import pytest
from agent.context.compact import CompactConfig, compact_pipeline, estimate, microcompact, _MC_CLEARED


# ── tiny config: thresholds measured in hundreds of tokens, not 150K ────────

def _tiny_cfg() -> CompactConfig:
    """Scaled-down config so tests need only ~KB of message content, not MB."""
    return CompactConfig(
        context_window=4_000,   # effective = 4000-200-300 = 3500
        output_reserve=200,
        compact_buffer=300,
        microcompact_keep=1,
        microcompact_trigger=2_000,  # micro fires at 2K tokens; auto at 3.5K
        microcompact_clear_at_least=200,
        cache_cold_seconds=3600,
        keep_min_tokens=200,
        keep_min_msgs=2,
        keep_max_tokens=800,
        summary_max_tokens=512,
    )


def _make_messages(n_tool_rounds: int, chars_per_result: int = 2000) -> list:
    """Build a sequence of tool-round message pairs with large bash tool_results.

    Each round: assistant(tool_use bash) + user(tool_result bash).
    Large content ensures context estimate exceeds thresholds when needed.
    """
    msgs = [{"role": "user", "content": "task"}]
    for i in range(n_tool_rounds):
        tid = f"tid{i}"
        msgs.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": tid, "name": "bash",
             "input": {"command": f"echo step{i}"}}
        ]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid,
             "content": f"result-{i}: " + "X" * chars_per_result}
        ]})
    return msgs


# ── cache-cold gate tests ──────────────────────────────────────────────────

def test_cache_cold_gate_warm_skips_micro(capture_sink):
    """idle_seconds < cache_cold_seconds → microcompact SKIPPED even if context > micro_thr.

    This locks the e2r finding: running micro when cache is warm breaks prompt cache,
    causing 3.5× billing inflation.  The gate prevents it.
    """
    cfg = _tiny_cfg()
    # build large enough messages to exceed micro_thr (2000 tokens = ~8000 chars)
    msgs = _make_messages(n_tool_rounds=6, chars_per_result=1500)
    before = estimate(msgs)
    assert before > 2000, f"test setup: need > 2K tokens, got {before}"

    # idle=10s < 3600s → cache is warm → micro must be skipped
    result = compact_pipeline(msgs, cfg=cfg, target_tokens=3500, idle_seconds=10.0)

    pipeline_spans = [e for e in capture_sink.events() if e["name"] == "compact.pipeline"]
    assert len(pipeline_spans) == 1
    attrs = pipeline_spans[0]["attributes"]
    assert attrs["cache_cold"] is False
    assert attrs["did_micro"] is False, "warm cache must not trigger microcompact"


def test_cache_cold_gate_cold_triggers_micro(capture_sink):
    """idle_seconds >= cache_cold_seconds → microcompact fires when context > micro_thr."""
    cfg = _tiny_cfg()
    msgs = _make_messages(n_tool_rounds=6, chars_per_result=1500)
    before = estimate(msgs)
    assert before > 2000

    # idle=7200s >= 3600s → cache is cold → micro should fire
    # Use target_tokens large enough that full_compact is NOT triggered after micro
    result = compact_pipeline(msgs, cfg=cfg, target_tokens=3500, idle_seconds=7200.0)

    pipeline_spans = [e for e in capture_sink.events() if e["name"] == "compact.pipeline"]
    attrs = pipeline_spans[0]["attributes"]
    assert attrs["cache_cold"] is True
    assert attrs["did_micro"] is True, "cold cache should trigger microcompact"

    # Old tool_results should be cleared in the returned messages
    cleared = sum(
        1 for m in result
        for b in (m.get("content") if isinstance(m.get("content"), list) else [])
        if isinstance(b, dict) and b.get("content") == _MC_CLEARED
    )
    assert cleared >= 1, "cold-gate micro should have cleared at least one old tool_result"


def test_cache_cold_gate_none_treated_as_cold(capture_sink):
    """idle_seconds=None (loop not connected) is treated as cache-cold (backward compat)."""
    cfg = _tiny_cfg()
    msgs = _make_messages(n_tool_rounds=6, chars_per_result=1500)

    result = compact_pipeline(msgs, cfg=cfg, target_tokens=3500, idle_seconds=None)

    pipeline_spans = [e for e in capture_sink.events() if e["name"] == "compact.pipeline"]
    attrs = pipeline_spans[0]["attributes"]
    # idle=None → cache_cold=True per code comment "idle=None(未接 loop)视为冷→保旧行为"
    assert attrs["cache_cold"] is True


def test_full_compact_triggers_above_auto_thr(monkeypatch, capture_sink):
    """Context > auto_thr (after micro) → full_compact is called.

    full_compact is mocked to avoid a real LLM call — the test only asserts the
    pipeline's strategy selection (did_full=True), not the compaction output.
    """
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    cfg = _tiny_cfg()
    # build messages well above auto_thr (3500 tokens = ~14000 chars)
    msgs = _make_messages(n_tool_rounds=10, chars_per_result=1800)
    before = estimate(msgs)
    assert before > 3500, f"test setup: need > 3.5K tokens, got {before}"

    # Mock full_compact to return a tiny replacement (no LLM call)
    called = []

    def mock_full_compact(
        messages,
        system="",
        cfg=None,
        skill_agent_id=None,
        post_compact_sink=None,
        auto_thr=None,
    ):
        called.append(post_compact_sink)
        return [{"role": "user", "content": "[Compacted] mock summary"}]

    monkeypatch.setattr(_compact, "full_compact", mock_full_compact)

    # Cold cache so micro runs first; target too small → micro won't bring it below auto_thr
    sink = object()
    compact_pipeline(
        msgs,
        cfg=cfg,
        target_tokens=100,
        idle_seconds=7200.0,
        post_compact_sink=sink,
    )

    pipeline_spans = [e for e in capture_sink.events() if e["name"] == "compact.pipeline"]
    attrs = pipeline_spans[0]["attributes"]
    assert attrs["did_full"] is True, "context above auto_thr should trigger full_compact"
    assert called == [sink]


def test_compact_pipeline_passes_read_state_to_full_compact(monkeypatch):
    from agent.context import compact as _compact
    from agent.tools.file_state import FileReadState

    cfg = _tiny_cfg()
    msgs = _make_messages(n_tool_rounds=10, chars_per_result=1800)
    read_state = FileReadState()
    captured = {}

    def mock_full_compact(
        messages,
        system="",
        cfg=None,
        skill_agent_id=None,
        post_compact_sink=None,
        read_state=None,
        auto_thr=None,
    ):
        captured["read_state"] = read_state
        captured["auto_thr"] = auto_thr
        return [{"role": "user", "content": "[Compacted] mock summary"}]

    monkeypatch.setattr(_compact, "full_compact", mock_full_compact)

    compact_pipeline(
        msgs,
        cfg=cfg,
        target_tokens=100,
        idle_seconds=7200.0,
        read_state=read_state,
    )

    assert captured["read_state"] is read_state
    assert captured["auto_thr"] == 100


def test_pipeline_no_compression_when_below_micro_thr(capture_sink):
    """Context below micro_thr → neither micro nor full fires (pipeline is a no-op)."""
    cfg = _tiny_cfg()
    # very small messages → well below micro_thr (2000 tokens)
    msgs = [
        {"role": "user", "content": "small task"},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]
    before = estimate(msgs)
    assert before < 2000

    compact_pipeline(msgs, cfg=cfg, target_tokens=3500, idle_seconds=7200.0)

    pipeline_spans = [e for e in capture_sink.events() if e["name"] == "compact.pipeline"]
    attrs = pipeline_spans[0]["attributes"]
    assert attrs["did_micro"] is False
    assert attrs["did_full"] is False
