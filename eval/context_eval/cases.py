"""Required deterministic cases for the context-budget offline gate."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import obs.trace as trace_module
from eval.context_eval.budget import (
    OMITTED_TOOL_OUTPUT,
    ContextBudgetConfig,
    apply_context_budget,
    estimate_request_tokens,
)


PASS = "PASS"
FAIL = "FAIL"
INVALID = "INVALID"
INCONCLUSIVE = "INCONCLUSIVE"
ERROR = "ERROR"
STATUS_VOCABULARY = (PASS, FAIL, INVALID, INCONCLUSIVE, ERROR)

SCHEMA_VERSION = "context-budget-evidence-v1"
PROTOCOL_VERSION = "context-budget-offline-v1"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    description: str
    criteria: tuple[str, ...]
    what_this_does_not_prove: str
    required: bool = True


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    description: str
    status: str
    required: bool
    evidence: Mapping[str, Any]
    what_this_does_not_prove: str
    message: str = ""


CASE_SPECS = (
    CaseSpec(
        case_id="context_unchanged_under_limit",
        description="A request below the configured limit is copied without pruning.",
        criteria=(
            "outcome is unchanged",
            "before and after estimates match",
            "content is preserved and no result is pruned",
        ),
        what_this_does_not_prove=(
            "This does not prove provider tokenizer accuracy or downstream task quality."
        ),
    ),
    CaseSpec(
        case_id="context_prunes_oldest_recoverable_and_retains_recent",
        description=(
            "The oldest recoverable result is omitted while the configured recent "
            "result remains available."
        ),
        criteria=(
            "exactly one result is pruned",
            "the oldest read result is omitted",
            "the recent read result and durable input are retained",
        ),
        what_this_does_not_prove=(
            "This does not prove that rerunning a tool is cheap, safe, or useful for a model."
        ),
    ),
    CaseSpec(
        case_id="context_nonrecoverable_and_unpaired_exceeds",
        description=(
            "Nonrecoverable and unpaired results are retained even when the request "
            "remains over budget."
        ),
        criteria=(
            "outcome is exceeded",
            "no result is pruned",
            "nonrecoverable and unpaired content remain unchanged",
        ),
        what_this_does_not_prove=(
            "This does not prove that the configured limit matches any provider limit."
        ),
    ),
    CaseSpec(
        case_id="context_pairs_results_and_preserves_input",
        description=(
            "Only a result paired to a recoverable tool call is omitted in the "
            "request copy."
        ),
        criteria=(
            "the paired recoverable result is omitted",
            "the unpaired result is retained",
            "the durable input stays unchanged and the request is a deep copy",
        ),
        what_this_does_not_prove=(
            "This does not prove compatibility with every provider message extension."
        ),
    ),
    CaseSpec(
        case_id="context_accounts_system_schema_and_emits_trace",
        description=(
            "System text and tool schemas contribute to the estimate, and the "
            "decision span exposes the required fields."
        ),
        criteria=(
            "the full estimate exceeds the message-only estimate",
            "trace values match the returned decision",
            "the required decision attributes are present",
        ),
        what_this_does_not_prove=(
            "This does not prove provider tokenizer accuracy, billing usage, or trace export."
        ),
    ),
)

REQUIRED_CASE_IDS = tuple(spec.case_id for spec in CASE_SPECS if spec.required)
_SPECS_BY_ID = {spec.case_id: spec for spec in CASE_SPECS}


class _CaptureSink:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def emit(self, span) -> None:
        self._events.append(span.to_event())

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)


def _tool_exchange(tool_id: str, name: str, output: str) -> list[dict[str, Any]]:
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


def _apply_without_ambient_trace(*args, **kwargs):
    previous = trace_module._SINK
    trace_module._SINK = _CaptureSink()
    try:
        return apply_context_budget(*args, **kwargs)
    finally:
        trace_module._SINK = previous


def _checked_result(
    spec: CaseSpec,
    *,
    checks: Mapping[str, bool],
    evidence: Mapping[str, Any],
) -> CaseResult:
    failed = [name for name, passed in checks.items() if not passed]
    payload = {
        **dict(evidence),
        "checks": dict(checks),
        "failed_checks": failed,
    }
    return CaseResult(
        case_id=spec.case_id,
        description=spec.description,
        status=PASS if not failed else FAIL,
        required=spec.required,
        evidence=payload,
        what_this_does_not_prove=spec.what_this_does_not_prove,
        message="" if not failed else "one or more deterministic checks failed",
    )


def _case_unchanged() -> CaseResult:
    spec = _SPECS_BY_ID["context_unchanged_under_limit"]
    messages = [{"role": "user", "content": "small request"}]
    original = copy.deepcopy(messages)
    before = estimate_request_tokens(messages)
    decision = _apply_without_ambient_trace(
        messages,
        ContextBudgetConfig(max_request_tokens=before + 10),
    )
    content_preserved = decision.messages == messages == original
    return _checked_result(
        spec,
        checks={
            "outcome_unchanged": decision.outcome == "unchanged",
            "estimate_unchanged": decision.before_tokens == decision.after_tokens,
            "no_pruning": decision.pruned_results == 0,
            "content_preserved": content_preserved,
        },
        evidence={
            "outcome": decision.outcome,
            "before_tokens": decision.before_tokens,
            "after_tokens": decision.after_tokens,
            "pruned_results": decision.pruned_results,
            "content_preserved": content_preserved,
        },
    )


def _case_oldest_pruned_recent_retained() -> CaseResult:
    spec = _SPECS_BY_ID[
        "context_prunes_oldest_recoverable_and_retains_recent"
    ]
    messages = [
        {"role": "user", "content": "retain this request"},
        *_tool_exchange("old-read", "read_file", "old:" + "a" * 4_000),
        {"role": "assistant", "content": "retain this assistant text"},
        *_tool_exchange("new-read", "read_file", "new:" + "b" * 4_000),
    ]
    original = copy.deepcopy(messages)
    expected = copy.deepcopy(messages)
    expected[2]["content"][0]["content"] = OMITTED_TOOL_OUTPUT
    target = estimate_request_tokens(expected)
    decision = _apply_without_ambient_trace(
        messages,
        ContextBudgetConfig(
            max_request_tokens=target,
            target_tokens=target,
            keep_recent_tool_results=1,
            recoverable_tools=frozenset({"read_file"}),
        ),
    )
    oldest_omitted = (
        decision.messages[2]["content"][0]["content"] == OMITTED_TOOL_OUTPUT
    )
    recent_retained = decision.messages[5] == messages[5]
    input_unmodified = messages == original
    return _checked_result(
        spec,
        checks={
            "outcome_pruned": decision.outcome == "pruned",
            "one_result_pruned": decision.pruned_results == 1,
            "oldest_omitted": oldest_omitted,
            "recent_retained": recent_retained,
            "input_unmodified": input_unmodified,
            "target_met": decision.after_tokens <= target < decision.before_tokens,
        },
        evidence={
            "outcome": decision.outcome,
            "before_tokens": decision.before_tokens,
            "after_tokens": decision.after_tokens,
            "target_tokens": target,
            "pruned_results": decision.pruned_results,
            "oldest_omitted": oldest_omitted,
            "recent_retained": recent_retained,
            "input_unmodified": input_unmodified,
        },
    )


def _case_nonrecoverable_unpaired_exceeded() -> CaseResult:
    spec = _SPECS_BY_ID["context_nonrecoverable_and_unpaired_exceeds"]
    messages = [
        *_tool_exchange(
            "write-1",
            "write_file",
            "nonrecoverable:" + "x" * 3_000,
        ),
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
    original = copy.deepcopy(messages)
    before = estimate_request_tokens(messages)
    decision = _apply_without_ambient_trace(
        messages,
        ContextBudgetConfig(
            max_request_tokens=max(1, before // 2),
            keep_recent_tool_results=0,
            recoverable_tools=frozenset({"read_file"}),
        ),
    )
    nonrecoverable_retained = decision.messages[1] == messages[1]
    unpaired_retained = decision.messages[2] == messages[2]
    return _checked_result(
        spec,
        checks={
            "outcome_exceeded": decision.outcome == "exceeded",
            "no_pruning": decision.pruned_results == 0,
            "nonrecoverable_retained": nonrecoverable_retained,
            "unpaired_retained": unpaired_retained,
            "input_unmodified": messages == original,
        },
        evidence={
            "outcome": decision.outcome,
            "before_tokens": decision.before_tokens,
            "after_tokens": decision.after_tokens,
            "limit_tokens": max(1, before // 2),
            "pruned_results": decision.pruned_results,
            "nonrecoverable_retained": nonrecoverable_retained,
            "unpaired_retained": unpaired_retained,
            "input_unmodified": messages == original,
        },
    )


def _case_pairing_nonmutation() -> CaseResult:
    spec = _SPECS_BY_ID["context_pairs_results_and_preserves_input"]
    messages = [
        *_tool_exchange("paired-read", "read_file", "paired:" + "p" * 3_000),
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "unknown-call",
                    "content": "unpaired:" + "u" * 2_000,
                }
            ],
        },
    ]
    original = copy.deepcopy(messages)
    expected = copy.deepcopy(messages)
    expected[1]["content"][0]["content"] = OMITTED_TOOL_OUTPUT
    target = estimate_request_tokens(expected)
    decision = _apply_without_ambient_trace(
        messages,
        ContextBudgetConfig(
            max_request_tokens=target,
            target_tokens=target,
            keep_recent_tool_results=0,
            recoverable_tools=frozenset({"read_file"}),
        ),
    )
    paired_omitted = (
        decision.messages[1]["content"][0]["content"] == OMITTED_TOOL_OUTPUT
    )
    unpaired_retained = decision.messages[2] == messages[2]
    input_unmodified = messages == original
    request_is_deep_copy = (
        decision.messages is not messages
        and decision.messages[0] is not messages[0]
        and decision.messages[0]["content"] is not messages[0]["content"]
    )
    return _checked_result(
        spec,
        checks={
            "outcome_pruned": decision.outcome == "pruned",
            "paired_result_omitted": paired_omitted,
            "unpaired_result_retained": unpaired_retained,
            "input_unmodified": input_unmodified,
            "request_is_deep_copy": request_is_deep_copy,
        },
        evidence={
            "outcome": decision.outcome,
            "before_tokens": decision.before_tokens,
            "after_tokens": decision.after_tokens,
            "target_tokens": target,
            "pruned_results": decision.pruned_results,
            "paired_result_omitted": paired_omitted,
            "unpaired_result_retained": unpaired_retained,
            "input_unmodified": input_unmodified,
            "request_is_deep_copy": request_is_deep_copy,
        },
    )


def _case_accounting_trace() -> CaseResult:
    spec = _SPECS_BY_ID[
        "context_accounts_system_schema_and_emits_trace"
    ]
    messages = [{"role": "user", "content": "hello"}]
    system_prompt = "system guidance"
    tool_schemas = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {"type": "object"},
        }
    ]
    message_only = estimate_request_tokens(messages)
    full = estimate_request_tokens(
        messages,
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
    )
    limit = full + 10
    sink = _CaptureSink()
    previous = trace_module._SINK
    trace_module.set_sink(sink)
    try:
        decision = apply_context_budget(
            messages,
            ContextBudgetConfig(max_request_tokens=limit),
            system_prompt=system_prompt,
            tool_schemas=tool_schemas,
        )
    finally:
        trace_module._SINK = previous
    event = next(
        (item for item in sink.events() if item.get("name") == "context.budget"),
        None,
    )
    attrs = dict((event or {}).get("attributes") or {})
    required_keys = (
        "context.after_tokens",
        "context.before_tokens",
        "context.outcome",
        "context.pruned_results",
        "context.target_tokens",
    )
    present_keys = sorted(key for key in required_keys if key in attrs)
    trace_matches = bool(event) and all(
        (
            attrs.get("context.before_tokens") == decision.before_tokens,
            attrs.get("context.after_tokens") == decision.after_tokens,
            attrs.get("context.pruned_results") == decision.pruned_results,
            attrs.get("context.outcome") == decision.outcome,
            attrs.get("context.target_tokens") == limit,
        )
    )
    return _checked_result(
        spec,
        checks={
            "full_estimate_exceeds_message_only": full > message_only,
            "decision_uses_full_estimate": decision.before_tokens == full,
            "trace_matches_decision": trace_matches,
            "required_trace_attributes_present": present_keys == list(required_keys),
        },
        evidence={
            "outcome": decision.outcome,
            "message_only_estimate": message_only,
            "full_estimate": full,
            "full_estimate_exceeds_message_only": full > message_only,
            "trace_matches_decision": trace_matches,
            "trace_attribute_keys": present_keys,
        },
    )


_RUNNERS: dict[str, Callable[[], CaseResult]] = {
    "context_unchanged_under_limit": _case_unchanged,
    "context_prunes_oldest_recoverable_and_retains_recent": (
        _case_oldest_pruned_recent_retained
    ),
    "context_nonrecoverable_and_unpaired_exceeds": (
        _case_nonrecoverable_unpaired_exceeded
    ),
    "context_pairs_results_and_preserves_input": _case_pairing_nonmutation,
    "context_accounts_system_schema_and_emits_trace": _case_accounting_trace,
}


def protocol_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "gate_rule": (
            "full required selection and exact result coverage with every case PASS"
        ),
        "aggregate_precedence": [ERROR, INVALID, INCONCLUSIVE, FAIL, PASS],
        "status_vocabulary": list(STATUS_VOCABULARY),
        "valid_for_rate": [PASS, FAIL],
        "excluded_from_rate": [INVALID, INCONCLUSIVE, ERROR],
        "required_case_ids": list(REQUIRED_CASE_IDS),
        "cases": [
            {
                "case_id": spec.case_id,
                "required": spec.required,
                "description": spec.description,
                "criteria": list(spec.criteria),
                "what_this_does_not_prove": spec.what_this_does_not_prove,
            }
            for spec in CASE_SPECS
        ],
    }


def protocol_fingerprint() -> str:
    encoded = json.dumps(
        protocol_manifest(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_case(case_id: str) -> CaseResult:
    spec = _SPECS_BY_ID.get(case_id)
    runner = _RUNNERS.get(case_id)
    if spec is None or runner is None:
        raise KeyError(f"unknown context-budget case: {case_id}")
    try:
        return runner()
    except Exception as exc:
        return CaseResult(
            case_id=spec.case_id,
            description=spec.description,
            status=ERROR,
            required=spec.required,
            evidence={"error_type": type(exc).__name__},
            what_this_does_not_prove=spec.what_this_does_not_prove,
            message="case execution error",
        )


def run_cases(case_ids: tuple[str, ...] | None = None) -> list[CaseResult]:
    selected = REQUIRED_CASE_IDS if case_ids is None else case_ids
    unknown = [case_id for case_id in selected if case_id not in _RUNNERS]
    if unknown:
        raise KeyError(f"unknown context-budget cases: {', '.join(unknown)}")
    return [run_case(case_id) for case_id in selected]


__all__ = [
    "ERROR",
    "FAIL",
    "INCONCLUSIVE",
    "INVALID",
    "PASS",
    "PROTOCOL_VERSION",
    "REQUIRED_CASE_IDS",
    "SCHEMA_VERSION",
    "STATUS_VOCABULARY",
    "CaseResult",
    "protocol_fingerprint",
    "protocol_manifest",
    "run_case",
    "run_cases",
]
