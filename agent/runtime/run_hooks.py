"""Non-tool hook helpers for the agent loop.

Tool hooks stay inside ToolExecutionRuntime.  This module handles the run-level
hooks whose outputs mutate the durable transcript or decide whether Stop may
continue for one more model turn.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from .hooks import HookInput, HookResult, report_hook_errors
from .run_context import RunContext


def append_hook_result_messages(messages: list, result: HookResult) -> None:
    """Append hook-produced context at the old durable transcript boundary.

    UserPromptSubmit, Stop, and StopFailure may add messages that later turns
    must see.  Centralizing the conversion keeps hook payload handling separate
    from request-only context such as AGENTS.md.
    """
    for message in result.messages:
        messages.append(hook_message_to_durable(message))
    for context in result.additional_contexts:
        messages.append(hook_context_to_durable(context))


def hook_message_to_durable(value: Any) -> dict:
    """Convert a hook message while preserving already-durable messages.

    Hooks may return either a complete message or a shorthand context block.
    Complete messages are deep-copied so hook authors cannot mutate transcript
    state after the hook returns.
    """
    if isinstance(value, dict) and "role" in value and "content" in value:
        return copy.deepcopy(value)
    return hook_context_to_durable(value)


def hook_context_to_durable(value: Any) -> dict:
    """Wrap hook shorthand context as a user message for later turns.

    The loop historically treated hook-added context as durable user content,
    not as a transient request prefix.  This helper keeps block, list, and text
    forms on the same lifecycle boundary.
    """
    if isinstance(value, dict) and "type" in value:
        return {"role": "user", "content": [copy.deepcopy(value)]}
    if isinstance(value, list):
        return {"role": "user", "content": copy.deepcopy(value)}
    return {"role": "user", "content": str(value)}


def last_assistant_message(messages: list) -> Any:
    """Return the assistant content used as hook context.

    Stop and StopFailure hooks receive the latest assistant message even after
    durable hook/tool messages have been appended.  Walking backward preserves
    that lifecycle contract without coupling hook code to turn layout.
    """
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content")
    return None


def stop_feedback_message(result: HookResult) -> dict:
    """Build the durable feedback message that triggers one more Stop turn.

    The exact text is part of the loop's observable continuation protocol, so
    the helper only centralizes construction and does not reinterpret hook
    fields.
    """
    reason = result.blocking_error or result.stop_reason or "prevented by hook"
    return {"role": "user", "content": f"StopHookBlocked: {reason}"}


def should_continue_after_stop(result: HookResult) -> bool:
    """Decide whether a Stop hook result requests the one-shot continuation.

    The RunState guard still enforces that continuation happens at most once;
    this predicate only preserves the old HookResult field interpretation.
    """
    return bool(result.blocking_error and not result.prevent_continuation)


def run_user_prompt_submit_hook(
    run_context: RunContext,
    messages: list,
    logger: logging.Logger,
) -> HookResult:
    """Run UserPromptSubmit with the exact pre-turn payload shape.

    This hook fires after setup but before any model turn, so its message_count
    must reflect the durable transcript before hook-added contexts are appended.
    """
    hook_input = HookInput(
        event="UserPromptSubmit",
        run_id=run_context.run_id,
        cwd=str(run_context.workdir),
        payload={"message_count": len(messages)},
        prompt=run_context.task,
    )
    hook_result = run_context.hook_bus.run(hook_input)
    report_hook_errors(hook_result, hook_input, logger)
    return hook_result


def run_stop_hook(
    run_context: RunContext,
    messages: list,
    outcome: str,
    final_text: str = "",
    stop_reason: str = "",
    logger: logging.Logger | None = None,
) -> HookResult:
    """Run Stop with the same lifecycle payload the loop used inline.

    Stop is a run-level hook, not a tool hook.  It sees current turn counters,
    compaction count, final text, and the latest assistant message while
    RunState separately enforces the one-shot continuation rule.
    """
    hook_input = HookInput(
        event="Stop",
        run_id=run_context.run_id,
        cwd=str(run_context.workdir),
        payload={
            "outcome": outcome,
            "turns": run_context.state.turn_no,
            "n_compactions": run_context.state.n_compactions,
            "final_text": final_text,
            "stop_reason": stop_reason,
        },
        prompt=run_context.task,
        last_assistant_message=last_assistant_message(messages),
    )
    hook_result = run_context.hook_bus.run(hook_input)
    if logger is not None:
        report_hook_errors(hook_result, hook_input, logger)
    return hook_result


def run_stop_failure_hook(
    run_context: RunContext,
    messages: list,
    exc: Exception,
    logger: logging.Logger,
) -> HookResult:
    """Run StopFailure only on the same non-overflow error path as before.

    The helper does not decide when StopFailure should fire; callers keep that
    control flow so context-overflow handling remains excluded from hook
    execution.
    """
    hook_input = HookInput(
        event="StopFailure",
        run_id=run_context.run_id,
        cwd=str(run_context.workdir),
        payload={
            "turns": run_context.state.turn_no,
            "error_details": str(exc),
        },
        prompt=run_context.task,
        error=exc,
        last_assistant_message=last_assistant_message(messages),
    )
    hook_result = run_context.hook_bus.run(hook_input)
    report_hook_errors(hook_result, hook_input, logger)
    return hook_result
