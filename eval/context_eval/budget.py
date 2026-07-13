"""Deterministic, eval-only request-budget adapter.

The token estimate is deliberately approximate.  It is a stable project-owned
proxy for mechanism evaluation, not the runtime compaction pipeline, a provider
tokenizer, or a billing counter.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from agent.runtime.observability import runtime_span, safe_set_current_span

OMITTED_TOOL_OUTPUT = "[tool output omitted; rerun the tool if needed]"
DEFAULT_RECOVERABLE_TOOLS = frozenset(
    {
        "bash",
        "shell",
        "powershell",
        "read",
        "read_file",
        "grep",
        "glob",
        "symbol_search",
    }
)


@dataclass(frozen=True)
class ContextBudgetConfig:
    max_request_tokens: int | None = None
    target_tokens: int | None = None
    keep_recent_tool_results: int = 2
    recoverable_tools: frozenset[str] = field(
        default_factory=lambda: DEFAULT_RECOVERABLE_TOOLS
    )

    def __post_init__(self) -> None:
        if self.max_request_tokens is not None and self.max_request_tokens <= 0:
            raise ValueError("max_request_tokens must be positive or None")
        if self.target_tokens is not None:
            if self.max_request_tokens is None:
                raise ValueError("target_tokens requires max_request_tokens")
            if self.target_tokens <= 0:
                raise ValueError("target_tokens must be positive")
            if self.target_tokens > self.max_request_tokens:
                raise ValueError("target_tokens cannot exceed max_request_tokens")
        if self.keep_recent_tool_results < 0:
            raise ValueError("keep_recent_tool_results cannot be negative")
        normalized = frozenset(
            str(name).strip().casefold() for name in self.recoverable_tools if str(name).strip()
        )
        object.__setattr__(self, "recoverable_tools", normalized)


@dataclass(frozen=True)
class ContextBudgetDecision:
    messages: list[dict[str, Any]]
    before_tokens: int
    after_tokens: int
    pruned_results: int
    outcome: Literal["unchanged", "pruned", "exceeded"]


@dataclass(frozen=True)
class _EligibleResult:
    message_index: int
    block_index: int


def estimate_request_tokens(
    messages: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str = "",
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> int:
    """Estimate request size as one token per four UTF-8 bytes, rounded up."""

    request: dict[str, Any] = {"messages": list(messages)}
    if system_prompt:
        request["system"] = system_prompt
    if tool_schemas:
        request["tools"] = list(tool_schemas)
    encoded = json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_fallback,
    ).encode("utf-8")
    return max(1, math.ceil(len(encoded) / 4))


def apply_context_budget(
    messages: Sequence[Mapping[str, Any]],
    config: ContextBudgetConfig,
    *,
    system_prompt: str = "",
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> ContextBudgetDecision:
    """Return a pruned request view while leaving the durable input untouched."""

    request_messages = copy.deepcopy([dict(message) for message in messages])
    before = estimate_request_tokens(
        request_messages,
        system_prompt=system_prompt,
        tool_schemas=tool_schemas,
    )
    target = config.target_tokens or config.max_request_tokens

    with runtime_span(
        "context.budget",
        **{
            "context.before_tokens": before,
            "context.max_request_tokens": config.max_request_tokens,
            "context.target_tokens": target,
        },
    ):
        if config.max_request_tokens is None or before <= config.max_request_tokens:
            decision = ContextBudgetDecision(
                messages=request_messages,
                before_tokens=before,
                after_tokens=before,
                pruned_results=0,
                outcome="unchanged",
            )
            _record_decision(decision, target)
            return decision

        eligible = _eligible_tool_results(request_messages, config.recoverable_tools)
        retain = min(config.keep_recent_tool_results, len(eligible))
        candidates = eligible[:-retain] if retain else eligible
        after = before
        pruned = 0

        for candidate in candidates:
            block = request_messages[candidate.message_index]["content"][candidate.block_index]
            block["content"] = OMITTED_TOOL_OUTPUT
            pruned += 1
            after = estimate_request_tokens(
                request_messages,
                system_prompt=system_prompt,
                tool_schemas=tool_schemas,
            )
            if target is not None and after <= target:
                break

        outcome: Literal["pruned", "exceeded"]
        if after <= config.max_request_tokens:
            outcome = "pruned"
        else:
            outcome = "exceeded"
        decision = ContextBudgetDecision(
            messages=request_messages,
            before_tokens=before,
            after_tokens=after,
            pruned_results=pruned,
            outcome=outcome,
        )
        _record_decision(decision, target)
        return decision


def _eligible_tool_results(
    messages: list[dict[str, Any]],
    recoverable_tools: frozenset[str],
) -> list[_EligibleResult]:
    calls_seen: dict[str, str] = {}
    eligible: list[_EligibleResult] = []

    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        role = message.get("role")
        for block_index, block in enumerate(content):
            if not isinstance(block, Mapping):
                continue
            block_type = block.get("type")
            if role == "assistant" and block_type == "tool_use":
                tool_id = str(block.get("id") or "")
                tool_name = str(block.get("name") or "").strip().casefold()
                if tool_id and tool_name:
                    calls_seen[tool_id] = tool_name
                continue
            if role != "user" or block_type != "tool_result":
                continue
            tool_id = str(block.get("tool_use_id") or "")
            tool_name = calls_seen.get(tool_id)
            if (
                tool_name in recoverable_tools
                and "content" in block
                and block.get("content") != OMITTED_TOOL_OUTPUT
            ):
                eligible.append(_EligibleResult(message_index, block_index))
    return eligible


def _record_decision(decision: ContextBudgetDecision, target: int | None) -> None:
    safe_set_current_span(
        **{
            "context.before_tokens": decision.before_tokens,
            "context.after_tokens": decision.after_tokens,
            "context.pruned_results": decision.pruned_results,
            "context.target_tokens": target,
            "context.outcome": decision.outcome,
        }
    )


def _json_fallback(value: Any) -> str:
    return f"<{type(value).__name__}>"


__all__ = [
    "DEFAULT_RECOVERABLE_TOOLS",
    "OMITTED_TOOL_OUTPUT",
    "ContextBudgetConfig",
    "ContextBudgetDecision",
    "apply_context_budget",
    "estimate_request_tokens",
]
