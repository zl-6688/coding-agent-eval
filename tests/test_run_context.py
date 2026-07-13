from agent.runtime.run_context import RunContext, RunState


class FakeMcpSource:
    """Small closeable stand-in so RunContext cleanup can be tested offline."""

    def __init__(self):
        """Start open because finish() must be the operation that closes it."""
        self.closed = False

    def close(self):
        """Record closure without hiding repeated-close bugs behind a mock."""
        self.closed = True


def test_run_state_tracks_counters_peak_and_stop_continuation():
    """RunState owns only per-run mutable counters and one-shot Stop continuation."""
    state = RunState(messages=[{"role": "user", "content": "q"}])

    assert state.next_turn() == 1
    assert state.next_turn() == 2
    assert state.increment_compactions() == 1
    assert state.record_context_size(10) == 10
    assert state.record_context_size(7) == 10
    assert state.peak_context == 10

    assert state.allow_stop_continuation() is True
    assert state.allow_stop_continuation() is False
    assert state.stop_continuation_used is True


def test_run_state_replaces_messages_after_compaction_rebind():
    """Compaction may return a new list, so finish() must see the rebound transcript."""
    original = [{"role": "user", "content": "old"}]
    compacted = [{"role": "user", "content": "new"}]
    state = RunState(messages=original)

    assert state.replace_messages(compacted) is compacted
    assert state.messages is compacted


def test_run_state_records_recompaction_chain_events():
    """Recompaction chain state is run-local and based on loop turn numbers."""
    state = RunState(messages=[])

    state.next_turn()
    first = state.record_compaction_event(
        auto_compact_threshold=100,
        true_post_compact_tokens=80,
        will_retrigger_next_turn=False,
    )
    state.next_turn()
    second = state.record_compaction_event(
        auto_compact_threshold=100,
        true_post_compact_tokens=120,
        will_retrigger_next_turn=True,
    )

    assert first == {
        "is_recompaction_in_chain": False,
        "turns_since_previous_compact": None,
        "previous_compact_turn_no": None,
        "compact_turn_no": 1,
    }
    assert second == {
        "is_recompaction_in_chain": True,
        "turns_since_previous_compact": 1,
        "previous_compact_turn_no": 1,
        "compact_turn_no": 2,
    }
    assert state.last_compact_turn_no == 2
    assert state.last_compact_will_retrigger is True


def test_run_state_post_compact_attachments_are_one_shot_and_isolated():
    """Post-compact request-only context is per-run, one-shot, and copy-isolated."""
    state = RunState(messages=[])
    source = {"role": "user", "content": {"nested": ["original"]}}

    state.queue_post_compact_attachments(source, None)
    source["content"]["nested"].append("mutated-after-queue")

    first_peek = state.peek_post_compact_attachments()
    assert first_peek == ({"role": "user", "content": {"nested": ["original"]}},)

    first_peek[0]["content"]["nested"].append("mutated-after-peek")
    second_peek = state.peek_post_compact_attachments()
    assert second_peek == ({"role": "user", "content": {"nested": ["original"]}},)

    drained = state.drain_post_compact_attachments()
    assert drained == second_peek
    assert state.peek_post_compact_attachments() == ()
    assert state.drain_post_compact_attachments() == ()


def test_run_state_resets_llm_idle_timer_at_explicit_boundary():
    """The loop can preserve the old idle timer boundary after early state setup."""
    state = RunState(messages=[])

    assert state.last_llm_ts == 0.0
    assert state.reset_llm_idle_timer(started_at=123.0) == 123.0
    assert state.mark_llm_completed(completed_at=456.0) == 456.0


def test_run_context_finish_closes_mcp_and_returns_plain_value():
    """return_messages=False preserves the old plain-value return shape after cleanup."""
    source = FakeMcpSource()
    state = RunState(messages=[{"role": "user", "content": "q"}])
    run_context = RunContext(
        task="q",
        run_id="run-1",
        run_meta={"run_id": "run-1"},
        return_messages=False,
        state=state,
        mcp_tool_source=source,
    )

    assert run_context.finish("done") == "done"
    assert source.closed is True
    assert run_context.mcp_tool_source is None


def test_run_context_finish_closes_mcp_and_returns_messages_tuple():
    """return_messages=True preserves the old tuple shape with durable messages."""
    source = FakeMcpSource()
    messages = [{"role": "user", "content": "q"}]
    state = RunState(messages=messages)
    run_context = RunContext(
        task="q",
        run_id="run-1",
        run_meta={"run_id": "run-1"},
        return_messages=True,
        state=state,
        mcp_tool_source=source,
    )

    assert run_context.finish("done") == ("done", messages)
    assert source.closed is True
    assert run_context.mcp_tool_source is None
