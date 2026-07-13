"""Synchronous in-process hook bus for Phase 1 lifecycle hooks."""

from __future__ import annotations

import fnmatch
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal

from .observability import runtime_span, safe_set_current_span

HookEvent = Literal[
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "Stop",
    "StopFailure",
    "SubagentStop",
]

PermissionBehavior = Literal["allow", "deny", "ask", "passthrough"]
HookMatcher = str | Callable[["HookInput"], bool] | None

HOOK_EVENTS: tuple[HookEvent, ...] = (
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "Stop",
    "StopFailure",
    "SubagentStop",
)
_HOOK_EVENT_SET = set(HOOK_EVENTS)
_PERMISSION_PRIORITY = {
    "passthrough": 0,
    "allow": 1,
    "ask": 2,
    "deny": 3,
}


@dataclass(frozen=True)
class HookInput:
    event: HookEvent | str
    run_id: str = ""
    cwd: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    tool_name: str | None = None
    tool_use_id: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any = None
    error: Any = None
    prompt: str | None = None
    last_assistant_message: Any = None


@dataclass(frozen=True)
class HookResult:
    messages: tuple[dict[str, Any], ...] = ()
    additional_contexts: tuple[Any, ...] = ()
    updated_input: dict[str, Any] | None = None
    permission_behavior: PermissionBehavior = "passthrough"
    blocking_error: str | None = None
    prevent_continuation: bool = False
    stop_reason: str | None = None
    errors: tuple[str, ...] = ()


HookHandler = Callable[[HookInput], HookResult]


@dataclass(frozen=True)
class HookRegistration:
    event: HookEvent | str
    handler: HookHandler
    matcher: HookMatcher = None


class HookBus:
    """Register, match, run, and aggregate in-process hook handlers."""

    def __init__(self, registrations: Iterable[HookRegistration] | None = None) -> None:
        self._registrations: list[HookRegistration] = []
        for registration in registrations or ():
            self.register(registration.event, registration.handler, matcher=registration.matcher)

    def register(
        self,
        event: HookEvent | str,
        handler: HookHandler,
        matcher: HookMatcher = None,
    ) -> None:
        _validate_event(event)
        if not callable(handler):
            raise TypeError("hook handler must be callable")
        self._registrations.append(HookRegistration(event=event, handler=handler, matcher=matcher))

    def has_handlers(self, event: HookEvent | str) -> bool:
        _validate_event(event)
        return any(registration.event == event for registration in self._registrations)

    def run(self, hook_input: HookInput) -> HookResult:
        _validate_event(hook_input.event)

        messages: list[dict[str, Any]] = []
        additional_contexts: list[Any] = []
        updated_input: dict[str, Any] | None = None
        permission_behavior: PermissionBehavior = "passthrough"
        blocking_error: str | None = None
        prevent_continuation = False
        stop_reason: str | None = None
        errors: list[str] = []
        handler_count = sum(
            1 for registration in self._registrations if registration.event == hook_input.event
        )
        matched_count = 0

        with runtime_span(
            "hook.run",
            **{
                "hook.event": hook_input.event,
                "hook.run_id_present": bool(hook_input.run_id),
                "hook.tool_name_present": bool(hook_input.tool_name),
                "hook.handler_count": handler_count,
            },
        ):
            for registration in self._registrations:
                if registration.event != hook_input.event:
                    continue
                try:
                    if not _matches(registration.matcher, hook_input):
                        continue
                    matched_count += 1
                    result = registration.handler(hook_input)
                    # WHY Phase 1 intentionally avoids async hooks until a task graph exists.
                    if inspect.isawaitable(result):
                        raise TypeError("async hook handlers are not supported in HookBus Phase 1")
                    if not isinstance(result, HookResult):
                        raise TypeError("hook handler must return HookResult")

                    candidate_updated_input = (
                        dict(result.updated_input) if result.updated_input is not None else None
                    )
                    candidate_permission_behavior = _merge_permission(
                        permission_behavior,
                        result.permission_behavior,
                    )

                    messages.extend(result.messages)
                    additional_contexts.extend(result.additional_contexts)
                    if candidate_updated_input is not None:
                        updated_input = candidate_updated_input
                    permission_behavior = candidate_permission_behavior
                    if blocking_error is None and result.blocking_error:
                        blocking_error = result.blocking_error
                    prevent_continuation = prevent_continuation or result.prevent_continuation
                    if stop_reason is None and result.stop_reason:
                        stop_reason = result.stop_reason
                    errors.extend(result.errors)
                except Exception as exc:  # non-blocking: a bad hook must not derail the run
                    errors.append(_error_text(exc))
                    continue
            safe_set_current_span(
                **{
                    "hook.matched_count": matched_count,
                    "hook.error_count": len(errors),
                    "hook.blocking": bool(blocking_error or prevent_continuation),
                    "hook.permission_behavior": permission_behavior,
                    "hook.messages_count": len(messages),
                    "hook.contexts_count": len(additional_contexts),
                    "hook.updated_input_present": updated_input is not None,
                }
            )

        return HookResult(
            messages=tuple(messages),
            additional_contexts=tuple(additional_contexts),
            updated_input=updated_input,
            permission_behavior=permission_behavior,
            blocking_error=blocking_error,
            prevent_continuation=prevent_continuation,
            stop_reason=stop_reason,
            errors=tuple(errors),
        )


class NoOpHookBus:
    """Default hook bus object used by callers that do not install hooks."""

    def register(
        self,
        event: HookEvent | str,
        handler: HookHandler,
        matcher: HookMatcher = None,
    ) -> None:
        _validate_event(event)
        raise RuntimeError("NoOpHookBus does not accept registrations")

    def has_handlers(self, event: HookEvent | str) -> bool:
        _validate_event(event)
        return False

    def run(self, hook_input: HookInput) -> HookResult:
        _validate_event(hook_input.event)
        return HookResult()



def report_hook_errors(
    result: HookResult,
    hook_input: HookInput,
    logger: logging.Logger | None = None,
) -> None:
    if not result.errors:
        return
    attrs = {
        "hook.event": hook_input.event,
        "hook.run_id_present": bool(hook_input.run_id),
        "hook.tool_name_present": bool(hook_input.tool_name),
        "hook.error_count": len(result.errors),
    }
    if logger is not None:
        logger.warning(
            "hook handler errors event=%s run_id=%s tool_name=%s errors=%s",
            hook_input.event,
            hook_input.run_id,
            hook_input.tool_name or "",
            list(result.errors),
        )
    safe_set_current_span(**attrs)

def _validate_event(event: HookEvent | str) -> None:
    if event not in _HOOK_EVENT_SET:
        raise ValueError(f"unknown hook event: {event}")


def _matches(matcher: HookMatcher, hook_input: HookInput) -> bool:
    if matcher is None:
        return True
    if callable(matcher):
        return bool(matcher(hook_input))
    if isinstance(matcher, str):
        value = _matcher_value(hook_input)
        return fnmatch.fnmatchcase(value, matcher)
    raise TypeError("hook matcher must be None, str, or callable")


def _matcher_value(hook_input: HookInput) -> str:
    if hook_input.tool_name:
        return hook_input.tool_name
    for key in ("matcher_value", "source"):
        value = hook_input.payload.get(key)
        if value is not None:
            return str(value)
    if hook_input.prompt:
        return hook_input.prompt
    return str(hook_input.event)


def _merge_permission(
    current: PermissionBehavior,
    candidate: PermissionBehavior,
) -> PermissionBehavior:
    if candidate not in _PERMISSION_PRIORITY:
        raise TypeError(f"unsupported permission behavior: {candidate}")
    if _PERMISSION_PRIORITY[candidate] > _PERMISSION_PRIORITY[current]:
        return candidate
    return current


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
