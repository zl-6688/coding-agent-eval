import copy
import logging

from agent.context.compact import (
    create_compact_boundary_message,
    create_compact_summary_message,
)
from agent.context.request_view import build_request_view
from agent.runtime.hooks import HookBus, HookResult
from agent.runtime.memory_integration import maybe_inject_auto_memory_recall
from agent.runtime.request_context import (
    budget_system,
    compose_llm_messages,
    request_context_messages,
)
from agent.runtime.run_context import RunContext, RunState
from agent.runtime.run_hooks import (
    append_hook_result_messages,
    run_stop_failure_hook,
    run_stop_hook,
    run_user_prompt_submit_hook,
    should_continue_after_stop,
    stop_feedback_message,
)
from agent.runtime.tool_messages import close_dangling_tool_uses, split_tool_runtime_messages
from agent.tools.deferred import selected_deferred_tools_marker_message
from agent.tools.messages import mark_durable_request_message


LOGGER = logging.getLogger(__name__)


def test_request_context_is_request_only_prefix_context_and_budgeted():
    context = {"role": "user", "content": "project profile sentinel"}
    durable_marker = mark_durable_request_message(
        {"role": "user", "content": [{"type": "text", "text": "trusted marker"}]},
        source="test",
    )
    durable_messages = [
        {"role": "user", "content": "task"},
        durable_marker,
    ]
    before = copy.deepcopy(durable_messages)

    context_messages = request_context_messages(None, context)
    request_view = compose_llm_messages(durable_messages, context_messages)

    assert durable_messages == before
    assert context_messages == (context,)
    assert request_view[0] == context
    assert request_view[1] == {"role": "user", "content": "task"}
    assert request_view[2] == {
        "role": "user",
        "content": [{"type": "text", "text": "trusted marker"}],
    }
    assert "metadata" not in request_view[2]
    assert context not in durable_messages

    estimated_system = budget_system("SYSTEM", context_messages)
    assert estimated_system.startswith("SYSTEM")
    assert "project profile sentinel" in estimated_system


def test_request_views_filter_compact_boundary_and_strip_summary_metadata():
    boundary = create_compact_boundary_message(
        trigger="auto",
        pre_tokens=500,
        user_context="sys",
        messages_summarized=3,
    )
    summary = create_compact_summary_message(
        "[Compacted]\nsummary body",
        source="full_compact",
    )
    durable_messages = [
        boundary,
        summary,
        {"role": "user", "content": "tail", "__ace_message_id": "ace-message-1"},
    ]

    legacy_context = ({"role": "user", "content": "ctx", "__ace_message_id": "ace-message-context"},)
    legacy_view = compose_llm_messages(durable_messages, legacy_context)
    request_view = build_request_view(durable_messages).as_messages()

    assert legacy_view == [
        {"role": "user", "content": "ctx"},
        {"role": "user", "content": "[Compacted]\nsummary body"},
        {"role": "user", "content": "tail"},
    ]
    assert request_view == legacy_view[1:]
    assert all(message.get("subtype") != "compact_boundary" for message in request_view)
    assert all("metadata" not in message for message in request_view)
    assert summary["metadata"] == {"compactSummarySource": "full_compact"}
    assert summary["isCompactSummary"] is True
    assert summary["isVisibleInTranscriptOnly"] is True
    assert all("isCompactSummary" not in message for message in request_view)
    assert all("isVisibleInTranscriptOnly" not in message for message in request_view)
    assert all("is_compact_summary" not in message for message in request_view)
    assert all("is_visible_in_transcript_only" not in message for message in request_view)
    assert all("__ace_message_id" not in message for message in request_view)
    assert durable_messages[2]["__ace_message_id"] == "ace-message-1"


def test_run_hooks_payloads_durable_append_and_stop_continuation_helpers(tmp_path):
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "draft"},
    ]
    state = RunState(messages=messages, turn_no=3, n_compactions=2)
    bus = HookBus()
    seen_prompt_inputs = []
    seen_stop_inputs = []
    hook_message = {"role": "user", "content": "hook durable user"}

    def prompt_hook(hook_input):
        seen_prompt_inputs.append(hook_input)
        return HookResult(
            messages=(hook_message,),
            additional_contexts=(
                "hook text context",
                {"type": "text", "text": "hook block context"},
            ),
        )

    def stop_hook(hook_input):
        seen_stop_inputs.append(hook_input)
        return HookResult(blocking_error="continue once")

    bus.register("UserPromptSubmit", prompt_hook)
    bus.register("Stop", stop_hook)
    run_context = RunContext(
        task="q",
        run_id="run-helpers",
        run_meta={"run_id": "run-helpers"},
        return_messages=True,
        state=state,
        workdir=tmp_path,
        hook_bus=bus,
    )

    prompt_result = run_user_prompt_submit_hook(run_context, messages, LOGGER)

    assert seen_prompt_inputs[0].payload == {"message_count": 2}
    assert seen_prompt_inputs[0].prompt == "q"
    append_hook_result_messages(messages, prompt_result)
    assert messages[2] == {"role": "user", "content": "hook durable user"}
    assert messages[3] == {"role": "user", "content": "hook text context"}
    assert messages[4] == {
        "role": "user",
        "content": [{"type": "text", "text": "hook block context"}],
    }
    hook_message["content"] = "mutated after append"
    assert messages[2] == {"role": "user", "content": "hook durable user"}

    stop_result = run_stop_hook(
        run_context,
        messages,
        outcome="finished",
        final_text="draft",
        stop_reason="end_turn",
        logger=LOGGER,
    )

    assert seen_stop_inputs[0].payload == {
        "outcome": "finished",
        "turns": 3,
        "n_compactions": 2,
        "final_text": "draft",
        "stop_reason": "end_turn",
    }
    assert seen_stop_inputs[0].last_assistant_message == "draft"
    assert should_continue_after_stop(stop_result) is True
    assert stop_feedback_message(stop_result) == {
        "role": "user",
        "content": "StopHookBlocked: continue once",
    }
    assert run_context.state.allow_stop_continuation() is True
    assert run_context.state.allow_stop_continuation() is False
    assert should_continue_after_stop(
        HookResult(blocking_error="blocked", prevent_continuation=True)
    ) is False


def test_run_stop_failure_hook_payload_error_and_last_assistant(tmp_path):
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "text", "text": "latest"}]},
        {"role": "user", "content": "after assistant"},
    ]
    state = RunState(messages=messages, turn_no=4)
    bus = HookBus()
    seen_inputs = []

    def failure_hook(hook_input):
        seen_inputs.append(hook_input)
        return HookResult(additional_contexts=("failure context",))

    bus.register("StopFailure", failure_hook)
    run_context = RunContext(
        task="q",
        run_id="run-stop-failure",
        run_meta={"run_id": "run-stop-failure"},
        return_messages=True,
        state=state,
        workdir=tmp_path,
        hook_bus=bus,
    )
    exc = RuntimeError("bad request")

    result = run_stop_failure_hook(run_context, messages, exc, LOGGER)

    assert seen_inputs[0].payload == {
        "turns": 4,
        "error_details": "bad request",
    }
    assert seen_inputs[0].prompt == "q"
    assert seen_inputs[0].error is exc
    assert seen_inputs[0].last_assistant_message == [
        {"type": "text", "text": "latest"}
    ]
    append_hook_result_messages(messages, result)
    assert messages[-1] == {"role": "user", "content": "failure context"}


def test_auto_memory_recall_cadence_dedup_and_typed_attachment():
    from agent.memory.relevant import collect_surfaced_memories

    class FakeAutoMemory:
        def __init__(self):
            self.calls = []

        def recall(self, query, already_surfaced):
            self.calls.append((query, set(already_surfaced)))
            if len(self.calls) == 1:
                return [{"path": "a.md", "content": "alpha"}]
            return [{"path": "b.md", "content": "beta"}]

    messages = [{"role": "user", "content": "task"}]
    run_state = RunState(messages=messages, turn_no=1)
    auto_memory = FakeAutoMemory()

    assert maybe_inject_auto_memory_recall(
        messages,
        auto_memory=auto_memory,
        task="q",
        run_state=run_state,
        logger=LOGGER,
    ) is True
    assert auto_memory.calls == [("q", set())]
    assert collect_surfaced_memories(messages).paths == frozenset({"a.md"})
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "task"
    assert messages[1]["type"] == "attachment"
    assert messages[1]["attachment"]["type"] == "relevant_memories"
    assert messages[1]["attachment"]["memories"][0]["path"] == "a.md"

    run_state.turn_no = 2
    assert maybe_inject_auto_memory_recall(
        messages,
        auto_memory=auto_memory,
        task="q",
        run_state=run_state,
        logger=LOGGER,
    ) is False
    assert len(auto_memory.calls) == 1

    run_state.turn_no = 6
    assert maybe_inject_auto_memory_recall(
        messages,
        auto_memory=auto_memory,
        task="q",
        run_state=run_state,
        logger=LOGGER,
    ) is True
    assert auto_memory.calls[1] == ("q", {"a.md"})
    assert collect_surfaced_memories(messages).paths == frozenset({"a.md", "b.md"})
    assert messages[-1]["attachment"]["memories"][0]["path"] == "b.md"


def test_tool_messages_close_dangling_tool_uses():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "running tools"},
                {"type": "tool_use", "id": "tool-1", "name": "bash", "input": {}},
                {"type": "tool_use", "id": "tool-2", "name": "read_file", "input": {}},
            ],
        }
    ]

    assert close_dangling_tool_uses(messages, note="[interrupted]") is True
    assert messages[-1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tool-1",
                "content": "[interrupted]",
            },
            {
                "type": "tool_result",
                "tool_use_id": "tool-2",
                "content": "[interrupted]",
            },
        ],
    }
    assert close_dangling_tool_uses(messages) is False
    assert close_dangling_tool_uses([]) is False
    assert close_dangling_tool_uses([
        {"role": "assistant", "content": [{"type": "text", "text": "no tools"}]}
    ]) is False


def test_tool_messages_split_durable_request_markers_from_tool_results():
    selected_marker = selected_deferred_tools_marker_message(
        ["mcp__fs__read"],
        durable=True,
    )
    assert selected_marker is not None
    sideband_marker = mark_durable_request_message(
        {"role": "user", "content": "sideband marker"},
        source="test",
    )
    tool_result = {"type": "tool_result", "tool_use_id": "tool-1", "content": "ok"}
    hook_block = {"type": "text", "text": "hook context"}

    result_blocks, durable_messages = split_tool_runtime_messages([
        tool_result,
        selected_marker,
        hook_block,
        sideband_marker,
    ])

    assert result_blocks == [tool_result, hook_block]
    assert durable_messages == [selected_marker, sideband_marker]
    assert all("metadata" in message for message in durable_messages)

    selected_marker["content"][0]["text"] = "mutated outside split"
    sideband_marker["content"] = "mutated outside split"
    assert "mutated" not in durable_messages[0]["content"][0]["text"]
    assert durable_messages[1]["content"] == "sideband marker"
