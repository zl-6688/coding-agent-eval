from __future__ import annotations

import copy

import pytest

from eval.context_eval.budget import (
    OMITTED_TOOL_OUTPUT,
    ContextBudgetConfig,
    apply_context_budget,
    estimate_request_tokens,
)


def _tool_exchange(tool_id: str, name: str, output: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": {"path": f"{tool_id}.txt"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": output,
                }
            ],
        },
    ]


def test_budget_prunes_oldest_recoverable_result_without_mutating_history() -> None:
    messages = [
        {"role": "user", "content": "keep the user request verbatim"},
        *_tool_exchange("old-read", "read_file", "old:" + "a" * 4_000),
        {"role": "assistant", "content": "keep the analysis text verbatim"},
        *_tool_exchange("new-read", "read_file", "new:" + "b" * 4_000),
    ]
    original = copy.deepcopy(messages)
    expected = copy.deepcopy(messages)
    expected[2]["content"][0]["content"] = OMITTED_TOOL_OUTPUT
    target = estimate_request_tokens(expected)

    decision = apply_context_budget(
        messages,
        ContextBudgetConfig(
            max_request_tokens=target,
            target_tokens=target,
            keep_recent_tool_results=1,
            recoverable_tools=frozenset({"read_file"}),
        ),
    )

    assert decision.outcome == "pruned"
    assert decision.pruned_results == 1
    assert decision.messages == expected
    assert decision.after_tokens <= target < decision.before_tokens
    assert decision.messages is not messages
    assert messages == original


def test_budget_pairs_results_before_deciding_recoverability() -> None:
    messages = [
        *_tool_exchange("write-1", "write_file", "cannot rerun safely:" + "x" * 3_000),
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "missing-call",
                    "content": "unpaired:" + "y" * 3_000,
                }
            ],
        },
    ]
    before = estimate_request_tokens(messages)

    decision = apply_context_budget(
        messages,
        ContextBudgetConfig(
            max_request_tokens=max(1, before // 2),
            keep_recent_tool_results=0,
            recoverable_tools=frozenset({"read_file"}),
        ),
    )

    assert decision.outcome == "exceeded"
    assert decision.pruned_results == 0
    assert decision.messages == messages


def test_budget_reports_exceeded_when_recent_result_must_be_retained() -> None:
    messages = _tool_exchange("only-read", "read_file", "z" * 4_000)
    before = estimate_request_tokens(messages)

    decision = apply_context_budget(
        messages,
        ContextBudgetConfig(
            max_request_tokens=max(1, before - 1),
            keep_recent_tool_results=1,
            recoverable_tools=frozenset({"read_file"}),
        ),
    )

    assert decision.outcome == "exceeded"
    assert decision.pruned_results == 0
    assert decision.after_tokens == before


def test_budget_uses_system_and_tool_schema_in_request_estimate() -> None:
    messages = [{"role": "user", "content": "hello"}]
    plain = estimate_request_tokens(messages)
    full = estimate_request_tokens(
        messages,
        system_prompt="system guidance",
        tool_schemas=[{"name": "read_file", "input_schema": {"type": "object"}}],
    )

    assert full > plain > 0


def test_budget_emits_a_decision_span(capture_sink) -> None:
    messages = _tool_exchange("read-1", "read_file", "q" * 2_000)
    before = estimate_request_tokens(messages)

    decision = apply_context_budget(
        messages,
        ContextBudgetConfig(
            max_request_tokens=max(1, before - 1),
            keep_recent_tool_results=0,
            recoverable_tools=frozenset({"read_file"}),
        ),
    )

    event = next(event for event in capture_sink.events() if event["name"] == "context.budget")
    attrs = event["attributes"]
    assert attrs["context.before_tokens"] == decision.before_tokens
    assert attrs["context.after_tokens"] == decision.after_tokens
    assert attrs["context.pruned_results"] == decision.pruned_results
    assert attrs["context.outcome"] == decision.outcome


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_request_tokens": 0},
        {"max_request_tokens": 10, "target_tokens": 11},
        {"max_request_tokens": 10, "target_tokens": 0},
        {"keep_recent_tool_results": -1},
    ],
)
def test_budget_rejects_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ContextBudgetConfig(**kwargs)
