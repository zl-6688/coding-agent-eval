"""test_compact_core.py — characterization tests for compact.py core functions.

Locks observable behavior of:
  - estimate() monotonicity
  - microcompact() whitelist enforcement + keep-most-recent
  - full_compact() summary-replaces-history (with mocked LLM)
"""
import copy
import json

import pytest
from agent.context.compact import (
    CompactConfig, DEFAULT,
    DEFAULT_SUMMARY_MAX_TOKENS,
    estimate, microcompact, full_compact,
    _MC_CLEARED, COMPACTABLE_TOOLS,
    exclude_post_compact_file, track_file,
    create_compact_boundary_message,
    create_compact_summary_message,
    is_compact_boundary_message,
    is_compact_summary_message,
    messages_after_compact_boundary,
)


# ── estimate() ────────────────────────────────────────────────────────────

def test_estimate_monotonicity():
    """Adding messages increases the estimate (monotonic growth)."""
    msgs = [{"role": "user", "content": "hi"}]
    est1 = estimate(msgs)
    msgs.append({"role": "assistant", "content": "hello there"})
    est2 = estimate(msgs)
    msgs.append({"role": "user", "content": "ok " * 100})
    est3 = estimate(msgs)

    assert est1 < est2 < est3, f"estimate not monotonic: {est1} {est2} {est3}"


def test_estimate_includes_system():
    """System prompt characters count toward the estimate."""
    msgs = [{"role": "user", "content": "hi"}]
    without_system = estimate(msgs, system="")
    with_system = estimate(msgs, system="big system prompt " * 50)
    assert with_system > without_system


def test_estimate_formula():
    """String content still uses the historical total char count // 4 formula."""
    system = "S" * 8
    msgs = [{"role": "user", "content": "X" * 12}]
    # total chars = 8 + 12 = 20; tokens = 20 // 4 = 5
    assert estimate(msgs, system=system) == 5


def test_estimate_handles_content_blocks_without_stringifying_media_blobs():
    unknown = {"type": "unknown_block", "payload": {"value": "U" * 16}}

    assert estimate([{"role": "user", "content": [{"type": "text", "text": "A" * 8}]}]) == 2
    assert estimate([{"role": "assistant", "content": [{"type": "thinking", "thinking": "B" * 12}]}]) == 3
    assert estimate([{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "C" * 16}]}]) == 4
    assert estimate([{"role": "assistant", "content": [{"type": "image", "source": {"data": "I" * 20_000}}]}]) == 2_000
    assert estimate([{"role": "user", "content": [{"type": "document", "source": {"data": "D" * 20_000}}]}]) == 2_000
    assert estimate([{"role": "assistant", "content": [
        {"type": "tool_use", "name": "bash", "input": {"command": "echo hi"}},
    ]}]) == (len("bash") // 4) + (len(json.dumps({"command": "echo hi"}, ensure_ascii=False)) // 4)
    assert estimate([{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "Y" * 20},
    ]}]) == 5
    assert estimate([{"role": "user", "content": [unknown]}]) == (
        len(json.dumps(unknown, ensure_ascii=False, default=str)) // 4
    )


def test_estimate_empty():
    """Empty inputs return 0."""
    assert estimate([], system="") == 0


def test_compact_boundary_helpers_and_estimate_filter():
    boundary = create_compact_boundary_message(
        trigger="auto",
        pre_tokens=123,
        user_context="sys",
        messages_summarized=2,
        logical_parent_uuid="parent-1",
    )
    messages = [
        {"role": "user", "content": "before"},
        boundary,
        {"role": "user", "content": "after"},
    ]

    assert is_compact_boundary_message(boundary) is True
    assert messages_after_compact_boundary(messages) == [messages[-1]]
    assert estimate([boundary]) == 0
    assert boundary["compactMetadata"] == {
        "trigger": "auto",
        "preTokens": 123,
        "userContext": "sys",
        "messagesSummarized": 2,
    }
    assert boundary["logicalParentUuid"] == "parent-1"
    assert "compact_metadata" not in boundary
    assert "logical_parent_uuid" not in boundary
    assert "preserved_segment" not in boundary
    assert "preservedSegment" not in boundary["compactMetadata"]


# ── microcompact() ─────────────────────────────────────────────────────────

def _build_tool_messages(tool_name: str, n: int, chars: int = 100) -> list:
    """n tool-round pairs: assistant(tool_use) + user(tool_result) for one tool."""
    msgs = [{"role": "user", "content": "task"}]
    for i in range(n):
        tid = f"id_{tool_name}_{i}"
        msgs.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": tid, "name": tool_name,
             "input": {"command": f"step{i}"}}
        ]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid,
             "content": f"output-{i}: " + "Y" * chars}
        ]})
    return msgs


def test_microcompact_clears_compactable_tools():
    """bash (in COMPACTABLE_TOOLS) tool_results are cleared by microcompact."""
    cfg = CompactConfig(microcompact_keep=1)
    msgs = _build_tool_messages("bash", n=3, chars=50)

    result = microcompact(msgs, cfg)

    # With keep=1, all but the most recent bash result should be cleared
    cleared_count = sum(
        1 for m in result
        for b in (m.get("content") if isinstance(m.get("content"), list) else [])
        if isinstance(b, dict) and b.get("content") == _MC_CLEARED
    )
    assert cleared_count >= 2, f"expected ≥2 cleared bash results, got {cleared_count}"


def test_microcompact_keeps_most_recent_k():
    """The most recent `keep` tool_results are preserved untouched."""
    keep = 2
    cfg = CompactConfig(microcompact_keep=keep)
    msgs = _build_tool_messages("bash", n=5, chars=50)

    result = microcompact(msgs, cfg)

    # Collect all bash tool_result contents in order
    bash_results = []
    for m in result:
        for b in (m.get("content") if isinstance(m.get("content"), list) else []):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                bash_results.append(b.get("content", ""))

    # Last `keep` results should NOT be the cleared placeholder
    assert len(bash_results) >= keep
    for content in bash_results[-keep:]:
        assert content != _MC_CLEARED, "most-recent tool_results must be preserved"


def test_microcompact_never_clears_update_todos():
    """update_todos is NOT in COMPACTABLE_TOOLS — its results must never be cleared.

    WHY: update_todos is the agent's plan state, exists only in context (no disk copy),
    clearing it would lose the plan permanently (unlike bash output that can be re-run).
    """
    cfg = CompactConfig(microcompact_keep=0)  # keep=0 → clamped to 1 by max(1, keep)
    msgs = _build_tool_messages("update_todos", n=4, chars=50)

    result = microcompact(msgs, cfg)

    # No update_todos tool_result should be cleared
    for m in result:
        for b in (m.get("content") if isinstance(m.get("content"), list) else []):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                assert b.get("content") != _MC_CLEARED, \
                    "update_todos results must never be cleared by microcompact"


def test_microcompact_target_tokens_stops_early():
    """With target_tokens, clearing stops once estimate drops below the target."""
    cfg = CompactConfig(microcompact_keep=1)
    msgs = _build_tool_messages("bash", n=10, chars=200)
    before = estimate(msgs)

    # Set target just slightly below before to force only partial clearing
    target = before - 50  # clear just enough to cross target
    result = microcompact(msgs, cfg, target_tokens=target)

    after = estimate(result)
    assert after <= before, "microcompact should not increase context size"


def test_microcompact_keep_floor_at_one():
    """keep is clamped to min 1 — at least one result is always preserved.

    Quirk: even keep=0 in config is floored to 1 per 'max(1, cfg.microcompact_keep)'.
    This prevents clearing the agent's entire context.
    """
    cfg = CompactConfig(microcompact_keep=0)  # will be clamped to 1
    msgs = _build_tool_messages("bash", n=5, chars=50)

    result = microcompact(msgs, cfg)

    bash_results = [
        b.get("content", "")
        for m in result
        for b in (m.get("content") if isinstance(m.get("content"), list) else [])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    not_cleared = [c for c in bash_results if c != _MC_CLEARED]
    assert len(not_cleared) >= 1, "at least 1 result must be preserved (keep floor=1)"


def test_microcompact_returns_messages_in_place():
    """microcompact modifies messages in place AND returns the same list."""
    cfg = CompactConfig(microcompact_keep=1)
    msgs = _build_tool_messages("bash", n=3, chars=50)
    original_id = id(msgs)

    result = microcompact(msgs, cfg)

    assert id(result) == original_id, "microcompact should return the same list object"


# ── full_compact() ─────────────────────────────────────────────────────────

def test_full_compact_summary_replaces_history(monkeypatch):
    """full_compact calls LLM and returns a new message list starting with the summary.

    The history is compressed: original messages do NOT appear verbatim in the output.
    """
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _SummaryResp:
        content = [MockBlock("text", text="<summary>compacted history</summary>")]
        stop_reason = "end_turn"
        usage = MockUsage()

    calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        calls.append({
            "messages": messages,
            "system": system,
            "tools": tools,
            "max_tokens": max_tokens,
            "kwargs": kwargs,
        })
        return _SummaryResp()

    monkeypatch.setattr(_compact.llm, "chat", fake_chat)

    cfg = CompactConfig(keep_min_tokens=10, keep_min_msgs=1, keep_max_tokens=200)
    msgs = [
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": [{"type": "text", "text": "thinking..."}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                      "content": "old result"}]},
        {"role": "user", "content": "tail sentinel"},
    ]

    result = full_compact(msgs, system="sys", cfg=cfg)

    assert calls, "full_compact should call the LLM"
    compact_messages = calls[0]["messages"]
    assert compact_messages[:-1] == [
        {"role": message["role"], "content": message["content"]}
        for message in msgs
    ]
    assert compact_messages is not msgs
    assert compact_messages[-1]["role"] == "user"
    assert "总结上方完整对话" in compact_messages[-1]["content"]
    assert "对话片段:" not in compact_messages[-1]["content"]
    assert calls[0]["tools"] == []
    assert calls[0]["max_tokens"] == DEFAULT_SUMMARY_MAX_TOKENS
    assert calls[0]["kwargs"]["purpose"] == "compaction"

    assert is_compact_boundary_message(result[0]) is True
    assert result[0]["compactMetadata"]["trigger"] == "auto"
    assert result[0]["compactMetadata"]["messagesSummarized"] == len(msgs)
    assert is_compact_summary_message(result[1]) is True
    assert result[1]["role"] == "user"
    assert result[1]["isVisibleInTranscriptOnly"] is True
    assert "[Compacted]" in result[1]["content"], \
        f"expected [Compacted] prefix, got: {result[1]['content'][:100]}"
    combined = "\n".join(str(m.get("content", "")) for m in result)
    assert "original task" not in combined
    assert "old result" not in combined
    assert "tail sentinel" not in combined


def test_full_compact_allows_scaled_summary_max_tokens(monkeypatch):
    """Tests can still shrink the compaction output budget explicitly."""
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _SummaryResp:
        content = [MockBlock("text", text="<summary>small budget summary</summary>")]
        usage = MockUsage()

    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(kwargs)
        return _SummaryResp()

    monkeypatch.setattr(_compact.llm, "chat", fake_chat)

    full_compact(
        [{"role": "user", "content": "original task"}],
        cfg=CompactConfig(summary_max_tokens=512),
    )

    assert calls[0]["max_tokens"] == 512


def test_full_compact_clamps_summary_max_tokens_to_model_cap(monkeypatch, capture_sink):
    """The compaction output budget must not exceed the model/provider cap."""
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _SummaryResp:
        content = [MockBlock("text", text="<summary>clamped budget summary</summary>")]
        usage = MockUsage()

    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(kwargs)
        return _SummaryResp()

    monkeypatch.setattr(_compact.llm, "chat", fake_chat)
    monkeypatch.setattr(_compact._cfg, "MODEL_MAX_OUTPUT_TOKENS", 8192, raising=False)

    full_compact(
        [{"role": "user", "content": "original task"}],
        cfg=CompactConfig(summary_max_tokens=20_000),
    )

    assert calls[0]["max_tokens"] == 8192
    full_span = [e for e in capture_sink.events() if e["name"] == "compact.full_compact"][-1]
    attrs = full_span["attributes"]
    assert attrs["compact_requested_max_tokens"] == 20_000
    assert attrs["compact_effective_max_tokens"] == 8192
    assert attrs["compact_max_tokens_clamped"] is True


def test_full_compact_retries_when_provider_reports_max_tokens_cap(monkeypatch, capture_sink):
    """If a provider rejects 20K as too high, retry once with the reported cap."""
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _SummaryResp:
        content = [MockBlock("text", text="<summary>provider cap retry summary</summary>")]
        usage = MockUsage()

    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("max_tokens must be less than or equal to 8192")
        return _SummaryResp()

    monkeypatch.setattr(_compact.llm, "chat", fake_chat)
    monkeypatch.setattr(_compact._cfg, "MODEL_MAX_OUTPUT_TOKENS", None, raising=False)

    result = full_compact(
        [{"role": "user", "content": "original task"}],
        cfg=CompactConfig(summary_max_tokens=20_000),
    )

    assert calls[0]["max_tokens"] == 20_000
    assert calls[1]["max_tokens"] == 8192
    assert result[1]["content"] == "[Compacted]\nprovider cap retry summary"
    full_span = [e for e in capture_sink.events() if e["name"] == "compact.full_compact"][-1]
    attrs = full_span["attributes"]
    assert attrs["compact_llm_calls"] == 2
    assert attrs["max_tokens_cap_retry_attempts"] == 1
    assert attrs["compact_effective_max_tokens"] == 8192


def test_full_compact_retries_context_overflow_by_lowering_max_tokens(monkeypatch, capture_sink):
    """Mirror CC's input+max_tokens overflow retry before falling back to PTL truncation."""
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _SummaryResp:
        content = [MockBlock("text", text="<summary>context overflow retry summary</summary>")]
        usage = MockUsage()

    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "input length and `max_tokens` exceed context limit: 188059 + 20000 > 200000"
            )
        return _SummaryResp()

    monkeypatch.setattr(_compact.llm, "chat", fake_chat)
    monkeypatch.setattr(_compact._cfg, "MODEL_MAX_OUTPUT_TOKENS", None, raising=False)

    result = full_compact(
        [{"role": "user", "content": "original task"}],
        cfg=CompactConfig(summary_max_tokens=20_000),
    )

    assert calls[0]["max_tokens"] == 20_000
    assert calls[1]["max_tokens"] == 10_941
    assert result[1]["content"] == "[Compacted]\ncontext overflow retry summary"
    full_span = [e for e in capture_sink.events() if e["name"] == "compact.full_compact"][-1]
    attrs = full_span["attributes"]
    assert attrs["compact_llm_calls"] == 2
    assert attrs["context_overflow_retry_attempts"] == 1
    assert attrs["compact_effective_max_tokens"] == 10_941


def test_full_compact_second_pass_starts_after_last_boundary(monkeypatch):
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _SummaryResp:
        content = [MockBlock("text", text="<summary>second summary</summary>")]
        usage = MockUsage()

    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(copy.deepcopy(messages))
        return _SummaryResp()

    monkeypatch.setattr(_compact.llm, "chat", fake_chat)
    boundary = create_compact_boundary_message(
        trigger="auto",
        pre_tokens=100,
        user_context="old sys",
        messages_summarized=3,
    )
    prior_summary = create_compact_summary_message(
        "[Compacted]\nprior summary",
        source="full_compact",
    )
    messages = [
        {"role": "user", "content": "pre-boundary history should stay out"},
        boundary,
        prior_summary,
        {"role": "user", "content": "new tail"},
    ]

    result = full_compact(messages, system="sys", cfg=CompactConfig())

    sent = calls[0]
    sent_text = str(sent)
    assert "pre-boundary history should stay out" not in sent_text
    assert "compact_boundary" not in sent_text
    assert sent[0] == {"role": "user", "content": "[Compacted]\nprior summary"}
    assert "metadata" not in sent[0]
    assert sent[1] == {"role": "user", "content": "new tail"}
    assert result[0]["compactMetadata"]["messagesSummarized"] == 2
    assert "compact_metadata" not in result[0]
    assert "preservedSegment" not in result[0]["compactMetadata"]


def test_full_compact_ptl_retry_truncates_head_then_succeeds(monkeypatch, capture_sink):
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _Resp:
        content = [MockBlock("text", text="<summary>retry summary</summary>")]
        usage = MockUsage()

    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(copy.deepcopy(messages))
        if len(calls) == 1:
            raise RuntimeError("Prompt is too long")
        return _Resp()

    monkeypatch.setattr(_compact.llm, "chat", fake_chat)

    msgs = [
        {"role": "user", "content": "oldest content must be dropped"},
        {"role": "assistant", "content": [{"type": "text", "text": "assistant round one"}]},
        {"role": "user", "content": "middle user content"},
        {"role": "assistant", "content": [{"type": "text", "text": "assistant round two"}]},
        {"role": "user", "content": "tail user content"},
    ]

    result = full_compact(msgs, system="sys", cfg=CompactConfig())

    assert len(calls) == 2
    assert calls[0][:-1] == [
        {"role": message["role"], "content": message["content"]}
        for message in msgs
    ]
    second_messages = calls[1]
    assert "oldest content must be dropped" not in str(second_messages)
    assert second_messages[0] == {"role": "user", "content": _compact.PTL_RETRY_MARKER}
    assert second_messages[-1]["role"] == "user"
    assert "总结上方完整对话" in second_messages[-1]["content"]
    assert is_compact_boundary_message(result[0]) is True
    assert result[0]["compactMetadata"]["trigger"] == "auto"
    assert result[1]["content"] == "[Compacted]\nretry summary"
    assert is_compact_summary_message(result[1]) is True
    assert _compact._circuit_breaker == 0

    full_span = [e for e in capture_sink.events() if e["name"] == "compact.full_compact"][-1]
    attrs = full_span["attributes"]
    assert attrs["compact_llm_calls"] == 2
    assert attrs["ptl_retry_attempts"] == 1


def test_full_compact_rejects_max_tokens_truncated_summary(monkeypatch, capture_sink):
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _TruncatedResp:
        content = [MockBlock("text", text="<summary>partial compact summary")]
        stop_reason = "max_tokens"
        usage = MockUsage(input_tokens=100, output_tokens=512)

    calls = []

    def fake_chat(messages, **kwargs):
        calls.append({"messages": copy.deepcopy(messages), "kwargs": kwargs})
        return _TruncatedResp()

    monkeypatch.setattr(_compact.llm, "chat", fake_chat)

    msgs = [{"role": "user", "content": "important conversation"}]
    result = full_compact(msgs, cfg=CompactConfig(summary_max_tokens=512))

    assert len(calls) == 1
    assert result == msgs
    assert _compact._circuit_breaker == 1

    full_span = [e for e in capture_sink.events() if e["name"] == "compact.full_compact"][-1]
    attrs = full_span["attributes"]
    assert attrs["status"] == "error"
    assert "truncated summary" in attrs["detail"]
    assert attrs["compact_stop_reason"] == "max_tokens"
    assert attrs["compact_output_truncated"] is True
    assert attrs["compact_llm_calls"] == 1
    assert attrs["ptl_retry_attempts"] == 0


def test_full_compact_rejects_non_text_summary_response(monkeypatch, capture_sink):
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _ToolOnlyResp:
        content = [MockBlock("tool_use", name="", input={}, id="bad-compact")]
        stop_reason = "tool_use"
        usage = MockUsage(input_tokens=100, output_tokens=10)

    monkeypatch.setattr(_compact.llm, "chat", lambda *a, **kw: _ToolOnlyResp())

    msgs = [{"role": "user", "content": "important conversation"}]
    result = full_compact(msgs, cfg=CompactConfig(summary_max_tokens=512))

    assert result == msgs
    assert _compact._circuit_breaker == 1

    full_span = [e for e in capture_sink.events() if e["name"] == "compact.full_compact"][-1]
    attrs = full_span["attributes"]
    assert attrs["status"] == "error"
    assert attrs["detail"] == "empty summary (tool_use)"
    assert attrs["compact_stop_reason"] == "tool_use"
    assert attrs["compact_response_block_types"] == "tool_use"


def test_full_compact_rejects_analysis_only_empty_summary(monkeypatch, capture_sink):
    """A compaction response with no summary body must not be accepted."""

    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _AnalysisOnlyResp:
        content = [MockBlock("text", text="<analysis>only reasoning</analysis>")]
        stop_reason = "end_turn"
        usage = MockUsage(input_tokens=100, output_tokens=32)

    monkeypatch.setattr(_compact.llm, "chat", lambda *a, **kw: _AnalysisOnlyResp())

    msgs = [{"role": "user", "content": "important conversation"}]
    result = full_compact(msgs, cfg=CompactConfig())

    assert result == msgs
    full_span = [e for e in capture_sink.events() if e["name"] == "compact.full_compact"][-1]
    attrs = full_span["attributes"]
    assert attrs["status"] == "error"
    assert "empty summary body" in attrs["detail"]
    assert attrs["compact_stop_reason"] == "end_turn"
    assert attrs["compact_output_truncated"] is False


def test_full_compact_non_ptl_error_does_not_retry(monkeypatch):
    from agent.context import compact as _compact

    calls = []

    def fail(messages, **kwargs):
        calls.append(copy.deepcopy(messages))
        raise RuntimeError("LLM down")

    monkeypatch.setattr(_compact.llm, "chat", fail)

    msgs = [{"role": "user", "content": "task"}]
    result = full_compact(msgs, cfg=CompactConfig())

    assert len(calls) == 1
    assert result == msgs
    assert _compact._circuit_breaker == 1


def test_full_compact_ptl_retry_exhaustion_counts_one_breaker_failure(
    monkeypatch,
    capture_sink,
):
    from agent.context import compact as _compact

    calls = []

    def fail(messages, **kwargs):
        calls.append(copy.deepcopy(messages))
        raise RuntimeError("Prompt is too long")

    monkeypatch.setattr(_compact.llm, "chat", fail)

    msgs = [
        {"role": "user", "content": "oldest"},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": [{"type": "text", "text": "a3"}]},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": [{"type": "text", "text": "a4"}]},
        {"role": "user", "content": "u4"},
    ]

    result = full_compact(msgs, cfg=CompactConfig(auto_max_failures=2))

    assert len(calls) == 4
    assert result == msgs
    assert _compact._circuit_breaker == 1

    full_span = [e for e in capture_sink.events() if e["name"] == "compact.full_compact"][-1]
    attrs = full_span["attributes"]
    assert attrs["compact_llm_calls"] == 4
    assert attrs["ptl_retry_attempts"] == 3


def test_full_compact_circuit_breaker(monkeypatch):
    """Circuit breaker: after auto_max_failures consecutive LLM failures, full_compact
    returns messages unchanged (does not raise).

    Quirk: the _circuit_breaker is a module global reset by reset_state() between tests.
    """
    from agent.context import compact as _compact

    def _fail(*a, **kw):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(_compact.llm, "chat", _fail)

    cfg = CompactConfig(auto_max_failures=2)
    msgs = [{"role": "user", "content": "task"}]

    # First two calls trip the breaker
    full_compact(msgs, cfg=cfg)   # failure 1
    full_compact(msgs, cfg=cfg)   # failure 2 → breaker trips

    # Third call is blocked by the breaker → returns msgs unchanged, no LLM call
    result = full_compact(msgs, cfg=cfg)
    assert result == msgs, "circuit breaker should return msgs unchanged"


def test_full_compact_emits_span(monkeypatch, capture_sink):
    """full_compact emits a compact.full_compact span (observable via sink)."""
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _Resp:
        content = [MockBlock("text", text="<summary>s</summary>")]
        stop_reason = "end_turn"
        usage = MockUsage()

    monkeypatch.setattr(_compact.llm, "chat", lambda *a, **kw: _Resp())

    cfg = CompactConfig(keep_min_tokens=10, keep_min_msgs=1, keep_max_tokens=100)
    msgs = [{"role": "user", "content": "task"}]
    full_compact(msgs, cfg=cfg)

    span_names = [e["name"] for e in capture_sink.events()]
    assert "compact.full_compact" in span_names


def test_full_compact_span_records_post_compact_payload_and_retrigger(
    monkeypatch,
    capture_sink,
):
    from agent.context import compact as _compact
    from agent.runtime.run_context import RunState
    from agent.skills import record_invoked_skill, reset_invoked_skills
    from conftest import MockBlock, MockUsage

    class _Resp:
        content = [MockBlock("text", text="<summary>" + ("s" * 80) + "</summary>")]
        usage = MockUsage(input_tokens=31, output_tokens=7)

    monkeypatch.setattr(_compact.llm, "chat", lambda *a, **kw: _Resp())

    run_state = RunState(messages=[])
    reset_invoked_skills()
    try:
        record_invoked_skill(
            "demo",
            "skills/demo/SKILL.md",
            "ATTACHMENT BODY COUNTS TOWARD TRUE PAYLOAD",
            agent_id="run-a",
        )
        result = full_compact(
            [{"role": "user", "content": "task"}],
            system="system prompt",
            cfg=CompactConfig(),
            skill_agent_id="run-a",
            post_compact_sink=run_state,
            auto_thr=5,
        )
    finally:
        reset_invoked_skills()

    full_span = [e for e in capture_sink.events() if e["name"] == "compact.full_compact"][-1]
    attrs = full_span["attributes"]
    true_payload = [*result, *run_state.peek_post_compact_attachments()]
    assert attrs["post_compact_durable_tokens"] == estimate(result)
    assert attrs["true_post_compact_tokens"] == estimate(
        true_payload,
        system="system prompt",
    )
    assert attrs["true_post_compact_tokens"] > attrs["post_compact_durable_tokens"]
    assert attrs["auto_compact_threshold"] == 5
    assert attrs["will_retrigger_next_turn"] is True
    assert attrs["is_recompaction_in_chain"] is False
    assert attrs["previous_compact_turn_no"] is None
    assert attrs["compact_turn_no"] == 0
    assert attrs["compact_cost_input"] == 31
    assert attrs["compact_cost_output"] == 7
    assert attrs["compact_api_usage_tokens"] == 38


def test_full_compact_telemetry_tolerates_broken_sink_duck_attrs(
    monkeypatch,
    capture_sink,
):
    """Optional telemetry sink attrs can fail without failing compact success."""
    from agent.context import compact as _compact
    from conftest import MockBlock

    class _Resp:
        content = [MockBlock("text", text="<summary>summary</summary>")]

    class _BrokenTelemetrySink:
        def queue_post_compact_attachments(self, *attachments):
            self.attachments = attachments

        @property
        def peek_post_compact_attachments(self):
            raise RuntimeError("peek getter failed")

        @property
        def turn_no(self):
            raise RuntimeError("turn getter failed")

        @property
        def record_compaction_event(self):
            raise RuntimeError("record getter failed")

    monkeypatch.setattr(_compact.llm, "chat", lambda *a, **kw: _Resp())

    result = full_compact(
        [{"role": "user", "content": "task"}],
        cfg=CompactConfig(),
        post_compact_sink=_BrokenTelemetrySink(),
    )

    assert result is not None
    full_span = [e for e in capture_sink.events() if e["name"] == "compact.full_compact"][-1]
    attrs = full_span["attributes"]
    assert attrs["is_recompaction_in_chain"] is False
    assert attrs["turns_since_previous_compact"] is None
    assert attrs["previous_compact_turn_no"] is None
    assert attrs["compact_turn_no"] is None


def test_full_compact_marks_retrigger_at_threshold_boundary(
    monkeypatch,
    capture_sink,
):
    """A post-compact payload exactly at threshold still retriggers compact."""
    from agent.context import compact as _compact
    from conftest import MockBlock

    class _Resp:
        content = [MockBlock("text", text="<summary>tiny</summary>")]

    monkeypatch.setattr(_compact.llm, "chat", lambda *a, **kw: _Resp())

    result = full_compact(
        [{"role": "user", "content": "task"}],
        cfg=CompactConfig(),
    )
    exact_threshold = estimate(result)
    event_count = len(capture_sink.events())

    full_compact(
        [{"role": "user", "content": "task"}],
        cfg=CompactConfig(),
        auto_thr=exact_threshold,
    )

    new_events = capture_sink.events()[event_count:]
    full_span = [e for e in new_events if e["name"] == "compact.full_compact"][-1]
    assert full_span["attributes"]["true_post_compact_tokens"] == exact_threshold
    assert full_span["attributes"]["will_retrigger_next_turn"] is True


def test_full_compact_restore_uses_global_exclude_not_kept_tail(monkeypatch, tmp_path):
    """full=0 后 full path 不再用 kept-tail exclude，但仍尊重全局 exclude。"""
    from agent.context import compact as _compact
    from agent import config
    from conftest import MockBlock

    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("fresh file content should be restored", encoding="utf-8")
    agent_md = tmp_path / "AGENTS.md"
    agent_md.write_text("project profile should not be restored", encoding="utf-8")
    excluded_profile = tmp_path / "project-profile.txt"
    excluded_profile.write_text("global exclude should not be restored", encoding="utf-8")
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    track_file("src/app.py", source.read_text(encoding="utf-8"))
    track_file("AGENTS.md", agent_md.read_text(encoding="utf-8"))
    track_file("project-profile.txt", excluded_profile.read_text(encoding="utf-8"))
    exclude_post_compact_file(excluded_profile)

    class _Resp:
        content = [MockBlock("text", text="<summary>summary</summary>")]
        usage = None

    monkeypatch.setattr(_compact.llm, "chat", lambda *a, **kw: _Resp())

    msgs = [
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "read-1", "name": "read_file", "input": {"path": "src/app.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "read-1", "content": "old read result kept tail"},
        ]},
        {"role": "assistant", "content": [MockBlock("text", text="tail sentinel")]},
    ]
    cfg = CompactConfig(
        keep_min_tokens=10,
        keep_min_msgs=4,
        keep_max_tokens=200,
        post_compact_max_files=5,
    )
    result = full_compact(msgs, system="sys", cfg=cfg)
    combined = "\n".join(str(m.get("content", "")) for m in result)
    pending = _compact.drain_post_compact_attachments()
    pending_text = "\n".join(str(m.get("content", "")) for m in pending)

    assert "old read result kept tail" not in combined
    assert "tail sentinel" not in combined
    assert "fresh file content should be restored" in pending_text
    assert "--- src/app.py ---" in pending_text
    assert "project profile should not be restored" not in combined
    assert "project profile should not be restored" not in pending_text
    assert "--- AGENTS.md ---" not in pending_text
    assert "global exclude should not be restored" not in pending_text
    assert "--- project-profile.txt ---" not in pending_text


def test_full_compact_uses_read_state_snapshot_and_resets_on_success(monkeypatch):
    from agent.context import compact as _compact
    from agent.tools.file_state import FileReadState
    from conftest import MockBlock

    class _Exec:
        def file_snapshot(self, path):
            return {"path": path, "exists": True}

    class _Resp:
        content = [MockBlock("text", text="<summary>summary</summary>")]
        usage = None

    read_state = FileReadState()
    read_state.record_read(
        "tracked.txt",
        "content restored from read state",
        complete=True,
        executor=_Exec(),
    )
    monkeypatch.setattr(_compact.llm, "chat", lambda *a, **kw: _Resp())

    result = full_compact(
        [{"role": "user", "content": "old context"}],
        system="sys",
        cfg=CompactConfig(keep_min_tokens=10, keep_min_msgs=1, keep_max_tokens=20),
        read_state=read_state,
    )

    pending = _compact.drain_post_compact_attachments()
    pending_text = str(pending)
    assert is_compact_boundary_message(result[0]) is True
    assert result[1]["content"].startswith("[Compacted]")
    assert "content restored from read state" in pending_text
    assert read_state.records == {}


def test_full_compact_run_state_sink_clears_fallback_stale_post_compact_lane(
    monkeypatch,
):
    from agent.context import compact as _compact
    from agent.runtime.run_context import RunState
    from agent.tools.file_state import FileReadState
    from conftest import MockBlock

    class _Exec:
        def file_snapshot(self, path):
            return {"path": path, "exists": True}

    class _Resp:
        content = [MockBlock("text", text="<summary>summary</summary>")]
        usage = None

    _compact._queue_post_compact_attachments(
        {"role": "user", "content": "stale fallback restore should be cleared"}
    )
    assert _compact.peek_post_compact_attachments()

    read_state = FileReadState()
    read_state.record_read(
        "tracked.txt",
        "fresh restore via run state",
        complete=True,
        executor=_Exec(),
    )
    run_state = RunState(messages=[])
    monkeypatch.setattr(_compact.llm, "chat", lambda *a, **kw: _Resp())

    full_compact(
        [{"role": "user", "content": "old context"}],
        system="sys",
        cfg=CompactConfig(keep_min_tokens=10, keep_min_msgs=1, keep_max_tokens=20),
        post_compact_sink=run_state,
        read_state=read_state,
    )

    pending = run_state.drain_post_compact_attachments()
    assert "fresh restore via run state" in str(pending)
    assert "stale fallback restore should be cleared" not in str(pending)
    assert _compact.drain_post_compact_attachments() == ()


def test_full_compact_failure_does_not_reset_read_state(monkeypatch):
    from agent.context import compact as _compact
    from agent.tools.file_state import FileReadState

    class _Exec:
        def file_snapshot(self, path):
            return {"path": path, "exists": True}

    read_state = FileReadState()
    read_state.record_read(
        "tracked.txt",
        "still needed after failed compact",
        complete=True,
        executor=_Exec(),
    )
    monkeypatch.setattr(
        _compact.llm,
        "chat",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    messages = [{"role": "user", "content": "old context"}]

    result = full_compact(
        messages,
        system="sys",
        cfg=CompactConfig(keep_min_tokens=10, keep_min_msgs=1, keep_max_tokens=20),
        read_state=read_state,
    )

    assert result == messages
    assert read_state.recent_file_items() == (
        ("tracked.txt", "still needed after failed compact"),
    )
    assert _compact.drain_post_compact_attachments() == ()


def test_full_compact_queues_file_skill_and_deferred_post_compact_attachments(
    monkeypatch,
    tmp_path,
):
    from agent import config
    from agent.context import compact as _compact
    from agent.skills import record_invoked_skill, reset_invoked_skills
    from agent.tools.deferred import DeferredToolState
    from conftest import MockBlock

    source = tmp_path / "tracked.txt"
    source.write_text("fresh file content", encoding="utf-8")
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    track_file("tracked.txt", "cached old")
    reset_invoked_skills()
    record_invoked_skill(
        "demo",
        "skills/demo/SKILL.md",
        "COMPACT RESTORED SKILL BODY",
        agent_id="run-a",
    )

    class _Resp:
        content = [MockBlock("text", text="<summary>summary</summary>")]
        usage = None

    monkeypatch.setattr(_compact.llm, "chat", lambda *a, **kw: _Resp())

    cfg = CompactConfig(keep_min_tokens=10, keep_min_msgs=1, keep_max_tokens=20)
    try:
        result = full_compact(
            [{"role": "user", "content": "old context " * 100}],
            system="sys",
            cfg=cfg,
            skill_agent_id="run-a",
            deferred_tool_state=DeferredToolState(["mcp__fs__read"]),
        )
        result_text = str(result)
        pending = _compact.drain_post_compact_attachments()
        pending_text = str(pending)
    finally:
        reset_invoked_skills()

    assert len(pending) == 3
    assert "fresh file content" not in result_text
    assert "COMPACT RESTORED SKILL BODY" not in result_text
    assert "selected-deferred-tools" not in result_text
    assert "fresh file content" in pending_text
    assert "cached old" not in pending_text
    assert "COMPACT RESTORED SKILL BODY" in pending_text
    assert "selected-deferred-tools" in pending_text


def test_session_memory_compact_uses_read_state_snapshot_and_resets_on_success(
    tmp_path,
    capture_sink,
):
    from agent.context import compact as _compact
    from agent.skills import record_invoked_skill, reset_invoked_skills
    from agent.tools.deferred import DeferredToolState
    from agent.tools.file_state import FileReadState

    class _Exec:
        def file_snapshot(self, path):
            return {"path": path, "exists": True}

    class _SessionMemory:
        def __init__(self, path):
            self.path = path
            self.compacted = False

        def wait_for_extraction(self):
            return True

        def is_empty(self):
            return False

        def on_compacted(self, messages):
            self.compacted = True

    sm_path = tmp_path / "session-memory.md"
    sm_path.write_text("session memory body", encoding="utf-8")
    sm = _SessionMemory(sm_path)
    read_state = FileReadState()
    read_state.record_read(
        "tracked.txt",
        "session compact restored file",
        complete=True,
        executor=_Exec(),
    )
    reset_invoked_skills()
    record_invoked_skill(
        "demo",
        "skills/demo/SKILL.md",
        "SM COMPACT RESTORED SKILL BODY",
        agent_id="run-sm",
    )

    try:
        result = _compact.session_memory_compact(
            [{"role": "user", "content": "old context"}],
            sm,
            "",
            CompactConfig(keep_min_tokens=10, keep_min_msgs=1, keep_max_tokens=200),
            auto_thr=10000,
            skill_agent_id="run-sm",
            deferred_tool_state=DeferredToolState(["mcp__fs__read"]),
            read_state=read_state,
        )
        pending = _compact.drain_post_compact_attachments()
    finally:
        reset_invoked_skills()

    pending_text = str(pending)
    assert result is not None
    assert is_compact_boundary_message(result[0]) is True
    assert result[0]["compactMetadata"]["trigger"] == "auto"
    assert result[0]["compactMetadata"]["messagesSummarized"] == 1
    assert "compact_metadata" not in result[0]
    assert "preserved_segment" not in result[0]
    assert "preservedSegment" not in result[0].get("compactMetadata", {})
    assert is_compact_summary_message(result[1]) is True
    assert result[1]["content"].startswith("[Compacted from session memory]")
    assert sm.compacted is True
    assert len(pending) == 3
    assert "session compact restored file" in pending_text
    assert "SM COMPACT RESTORED SKILL BODY" in pending_text
    assert "selected-deferred-tools" in pending_text
    assert read_state.records == {}
    sm_span = [e for e in capture_sink.events() if e["name"] == "compact.session_memory_compact"][-1]
    attrs = sm_span["attributes"]
    assert attrs["status"] == "ok"
    assert attrs["compact_llm_calls"] == 0
    assert attrs["auto_compact_threshold"] == 10000
    assert attrs["post_compact_durable_tokens"] == estimate(result)
    assert attrs["true_post_compact_tokens"] >= attrs["post_compact_durable_tokens"]
    assert attrs["will_retrigger_next_turn"] is False
    assert attrs["is_recompaction_in_chain"] is False
    assert attrs["previous_compact_turn_no"] is None


def test_session_memory_compact_returns_boundary_summary_and_filters_old_boundary(
    tmp_path,
):
    from agent.context import compact as _compact

    class _SessionMemory:
        def __init__(self, path):
            self.path = path
            self.compacted_messages = None

        def wait_for_extraction(self):
            return True

        def is_empty(self):
            return False

        def on_compacted(self, messages):
            self.compacted_messages = messages

    sm_path = tmp_path / "session-memory.md"
    sm_path.write_text("session memory body", encoding="utf-8")
    sm = _SessionMemory(sm_path)
    old_boundary = create_compact_boundary_message(
        trigger="auto",
        pre_tokens=100,
        user_context="old",
        messages_summarized=3,
    )
    messages = [
        {"role": "user", "content": "pre-boundary history"},
        old_boundary,
        create_compact_summary_message("[Compacted]\nold summary", source="full_compact"),
        {"role": "user", "content": "tail sentinel"},
    ]

    result = _compact.session_memory_compact(
        messages,
        sm,
        "",
        CompactConfig(keep_min_tokens=1, keep_min_msgs=1, keep_max_tokens=200),
        auto_thr=10000,
    )

    assert result is not None
    assert is_compact_boundary_message(result[0]) is True
    assert result[0]["compactMetadata"]["trigger"] == "auto"
    assert result[0]["compactMetadata"]["messagesSummarized"] == len(
        messages_after_compact_boundary(messages)
    )
    assert "compact_metadata" not in result[0]
    assert "preserved_segment" not in result[0]
    assert "preservedSegment" not in result[0].get("compactMetadata", {})
    assert is_compact_summary_message(result[1]) is True
    assert all(not is_compact_boundary_message(message) for message in result[2:])
    assert "tail sentinel" in str(result[2:])
    assert "pre-boundary history" not in str(result)
    assert sm.compacted_messages is result


def test_session_memory_compact_fallback_does_not_queue_post_compact_attachments(
    monkeypatch,
    tmp_path,
):
    from agent import config
    from agent.context import compact as _compact
    from agent.skills import record_invoked_skill, reset_invoked_skills
    from agent.tools.deferred import DeferredToolState
    from agent.tools.file_state import FileReadState

    class _Exec:
        def file_snapshot(self, path):
            return {"path": path, "exists": True}

    class _SessionMemory:
        def __init__(self, path):
            self.path = path
            self.compacted = False

        def wait_for_extraction(self):
            return True

        def is_empty(self):
            return False

        def on_compacted(self, messages):
            self.compacted = True

    source = tmp_path / "tracked.txt"
    source.write_text("fresh file content", encoding="utf-8")
    sm_path = tmp_path / "session-memory.md"
    sm_path.write_text("session memory body", encoding="utf-8")
    monkeypatch.setattr(config, "WORKDIR", tmp_path)
    track_file("tracked.txt", "cached old")
    reset_invoked_skills()
    record_invoked_skill(
        "demo",
        "skills/demo/SKILL.md",
        "COMPACT RESTORED SKILL BODY",
        agent_id="run-a",
    )
    sm = _SessionMemory(sm_path)
    read_state = FileReadState()
    read_state.record_read(
        "tracked.txt",
        "still needed after SM fallback",
        complete=True,
        executor=_Exec(),
    )
    cfg = CompactConfig(keep_min_tokens=10, keep_min_msgs=1, keep_max_tokens=200)
    messages = [{"role": "user", "content": "old context " * 100}]

    try:
        result = _compact.session_memory_compact(
            messages,
            sm,
            "",
            cfg,
            auto_thr=1,
            skill_agent_id="run-a",
            deferred_tool_state=DeferredToolState(["mcp__fs__read"]),
            read_state=read_state,
        )
        pending = _compact.drain_post_compact_attachments()
    finally:
        reset_invoked_skills()

    assert result is None
    assert sm.compacted is False
    assert pending == ()
    assert read_state.recent_file_items() == (
        ("tracked.txt", "still needed after SM fallback"),
    )


def test_session_memory_messages_to_keep_starts_after_known_anchor():
    from agent.context import compact as _compact

    messages = [
        {"role": "user", "content": "summarized user", "id": "m1"},
        {"role": "assistant", "content": "summarized assistant", "id": "m2"},
        {"role": "user", "content": "fresh tail", "id": "m3"},
    ]

    kept = _compact._session_memory_messages_to_keep(
        messages,
        "m2",
        CompactConfig(keep_min_tokens=1, keep_min_msgs=1, keep_max_tokens=200),
    )

    assert kept == [messages[2]]


def test_session_memory_messages_to_keep_returns_none_when_anchor_missing():
    from agent.context import compact as _compact

    messages = [
        {"role": "user", "content": "old", "id": "m1"},
        {"role": "assistant", "content": "new", "id": "m2"},
    ]

    kept = _compact._session_memory_messages_to_keep(
        messages,
        "missing-anchor",
        CompactConfig(keep_min_tokens=1, keep_min_msgs=1, keep_max_tokens=200),
    )

    assert kept is None


def test_session_memory_messages_to_keep_resumed_case_stops_at_old_boundary():
    from agent.context import compact as _compact

    old_boundary = create_compact_boundary_message(
        trigger="auto",
        pre_tokens=100,
        user_context="old sys",
        messages_summarized=2,
    )
    messages = [
        {"role": "user", "content": "pre-boundary should stay summarized", "id": "m1"},
        old_boundary,
        create_compact_summary_message("[Compacted]\nold summary", source="full_compact"),
        {"role": "user", "content": "after boundary one", "id": "m2"},
        {"role": "assistant", "content": "after boundary two", "id": "m3"},
    ]

    kept = _compact._session_memory_messages_to_keep(
        messages,
        None,
        CompactConfig(keep_min_tokens=10_000, keep_min_msgs=10, keep_max_tokens=20_000),
    )

    assert old_boundary not in kept
    assert messages[0] not in kept
    assert kept == messages[2:]


def test_session_memory_anchor_records_safe_runtime_id_and_clears_on_compact(tmp_path):
    from agent.memory.session_memory import SessionMemory

    sm = SessionMemory(tmp_path / "session-memory.md")
    messages = [{"role": "assistant", "content": "safe assistant turn"}]

    sm._record_last_summarized_message_id_if_safe(messages)

    assert messages[0]["uuid"]
    assert "__ace_message_id" not in messages[0]
    assert sm.get_last_summarized_message_id() == messages[0]["uuid"]

    sm.on_compacted([{"role": "user", "content": "after compact"}])

    assert sm.last_summarized_message_id is None


def test_session_memory_anchor_does_not_record_pending_tool_turn(tmp_path):
    from agent.memory.session_memory import SessionMemory

    sm = SessionMemory(tmp_path / "session-memory.md")
    sm.set_last_summarized_message_id("previous")
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tool-1", "name": "bash", "input": {}},
            ],
        },
    ]

    sm._record_last_summarized_message_id_if_safe(messages)

    assert sm.last_summarized_message_id == "previous"
    assert "__ace_message_id" not in messages[0]


def _stub_session_memory_fork(monkeypatch, final_text):
    from agent.memory.forked_agent import ForkResult
    import agent.memory.session_memory as smmod

    calls = []

    def fake_fork(
        prompt,
        context_messages,
        *,
        system="",
        allowed_tools,
        max_turns,
        max_tokens,
        label,
        tool_filter=None,
    ):
        calls.append(
            {
                "prompt": prompt,
                "context_messages": context_messages,
                "system": system,
                "allowed_tools": allowed_tools,
                "max_turns": max_turns,
                "max_tokens": max_tokens,
                "label": label,
                "tool_filter": tool_filter,
            }
        )
        return ForkResult(final_text=final_text, stopped="finished")

    monkeypatch.setattr(smmod, "run_forked_agent", fake_fork)
    return calls


def test_session_memory_extract_advances_anchor_after_valid_notes(monkeypatch, tmp_path):
    from agent.memory.session_memory import SESSION_MEMORY_TEMPLATE, SessionMemory

    notes = SESSION_MEMORY_TEMPLATE + "\n\nimplementation detail captured"
    calls = _stub_session_memory_fork(monkeypatch, notes)
    sm = SessionMemory(tmp_path / "session-memory.md")
    messages = [
        {"role": "user", "content": "task", "id": "user-1"},
        {"role": "assistant", "content": "done", "id": "assistant-1"},
    ]

    sm.extract(messages, system="SYS")

    assert calls[0]["allowed_tools"] == set()
    assert calls[0]["label"] == "session_memory"
    assert calls[0]["system"] == "SYS"
    assert "implementation detail captured" in sm.path.read_text(encoding="utf-8")
    assert sm.last_summarized_message_id == messages[-1]["uuid"]


@pytest.mark.parametrize("final_text", ["", "notes without required title anchor"])
def test_session_memory_extract_invalid_notes_do_not_advance_anchor(
    monkeypatch,
    tmp_path,
    final_text,
):
    from agent.memory.session_memory import SESSION_MEMORY_TEMPLATE, SessionMemory

    existing_notes = SESSION_MEMORY_TEMPLATE + "\n\nexisting note"
    _stub_session_memory_fork(monkeypatch, final_text)
    sm = SessionMemory(tmp_path / "session-memory.md")
    sm.path.parent.mkdir(parents=True, exist_ok=True)
    sm.path.write_text(existing_notes, encoding="utf-8")
    sm.set_last_summarized_message_id("previous")
    messages = [{"role": "assistant", "content": "safe", "id": "assistant-1"}]

    sm.extract(messages)

    assert sm.path.read_text(encoding="utf-8") == existing_notes
    assert sm.last_summarized_message_id == "previous"


def test_session_memory_extract_valid_notes_do_not_advance_pending_tool_turn(
    monkeypatch,
    tmp_path,
):
    from agent.memory.session_memory import SESSION_MEMORY_TEMPLATE, SessionMemory

    notes = SESSION_MEMORY_TEMPLATE + "\n\nvalid update"
    _stub_session_memory_fork(monkeypatch, notes)
    sm = SessionMemory(tmp_path / "session-memory.md")
    sm.set_last_summarized_message_id("previous")
    messages = [
        {"role": "user", "content": "task", "id": "user-1"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tool-1", "name": "bash", "input": {}},
            ],
            "id": "assistant-tool-1",
        },
    ]

    sm.extract(messages)

    assert "valid update" in sm.path.read_text(encoding="utf-8")
    assert sm.last_summarized_message_id == "previous"
    assert "__ace_message_id" not in messages[-1]


def test_session_memory_messages_to_keep_drops_orphan_tool_result():
    from agent.context import compact as _compact

    boundary = create_compact_boundary_message(
        trigger="auto",
        pre_tokens=100,
        user_context="old sys",
        messages_summarized=1,
    )
    messages = [
        boundary,
        {"role": "user", "content": "safe head", "id": "head"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool-1", "content": "orphan"},
            ],
            "id": "tool-result",
        },
        {"role": "user", "content": "safe tail", "id": "tail"},
    ]

    kept = _compact._session_memory_messages_to_keep(
        messages,
        None,
        CompactConfig(keep_min_tokens=10_000, keep_min_msgs=10, keep_max_tokens=20_000),
    )

    assert kept == [messages[1], messages[3]]


def test_session_memory_messages_to_keep_expands_to_matching_tool_use():
    from agent.context import compact as _compact

    messages = [
        {"role": "user", "content": "summarized", "id": "m1"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tool-1", "name": "read_file", "input": {"path": "a.py"}},
            ],
            "id": "tool-use-message",
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool-1", "content": "file body"},
            ],
            "id": "tool-result-message",
        },
    ]

    kept = _compact._session_memory_messages_to_keep(
        messages,
        "tool-use-message",
        CompactConfig(keep_min_tokens=1, keep_min_msgs=0, keep_max_tokens=200),
    )

    assert kept == [messages[1], messages[2]]
