"""Tool execution lifecycle for Tool/ToolPool objects."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from obs.trace import SpanKind

from . import result_store
from .contracts import Tool, ToolContext, ToolResult
from .messages import is_durable_request_message
from .pool import ToolPool
from ..runtime.hooks import HookBus, HookInput, HookResult, NoOpHookBus, report_hook_errors
from ..runtime.observability import (
    content_preview_attrs,
    content_summary_attrs,
    record_permission_decision,
    runtime_span,
    safe_set_current_span,
    safe_text_length,
)
from ..runtime.permissions import PermissionDecision, PermissionEngine

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolUseRequest:
    id: str
    name: str
    input: Any

    @classmethod
    def from_block(cls, block: Any) -> "ToolUseRequest":
        return cls(
            id=str(_block_value(block, "id", "")),
            name=str(_block_value(block, "name", "")),
            input=_block_value(block, "input", None),
        )


@dataclass(frozen=True)
class HookDecision:
    updated_input: dict[str, Any] | None = None
    stop_reason: str | None = None
    additional_messages: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ToolExecutionResult:
    messages: tuple[dict[str, Any], ...]
    tool_name: str
    is_error: bool
    permission: PermissionDecision
    duration_ms: int
    additional_messages: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    counts_as_tool_error: bool = False
    persisted: bool = False
    raw_chars: int = 0
    persist_path: str = ""


class ToolHookAdapter:
    def pre_tool_use(self, tool: Tool, request: ToolUseRequest) -> HookDecision:
        return HookDecision()

    def post_tool_use(
        self,
        tool: Tool,
        request: ToolUseRequest,
        result: ToolExecutionResult,
    ) -> HookDecision:
        return HookDecision()


class ToolExecutionRuntime:
    def __init__(
        self,
        tools: ToolPool | Iterable[Tool],
        *,
        permission_engine: PermissionEngine | None = None,
        hook_adapter: ToolHookAdapter | None = None,
        hook_bus: HookBus | NoOpHookBus | None = None,
        run_id: str = "",
        cwd: str | None = None,
        project_context_message: dict[str, Any] | None = None,
        agent_id: str = "",
        agent_type: str = "main",
        is_subagent: bool = False,
        file_state: Any = None,
        executor: Any = None,
        tool_result_callback: Callable[[ToolUseRequest, ToolExecutionResult], None] | None = None,
    ) -> None:
        tool_tuple = tools.tools if isinstance(tools, ToolPool) else tuple(tools)
        self.tools = {tool.name: tool for tool in tool_tuple}
        if len(self.tools) != len(tool_tuple):
            duplicates = _duplicate_tool_names(tool_tuple)
            raise ValueError(f"duplicate tool name: {duplicates[0]}")
        self.permission_engine = permission_engine or PermissionEngine()
        self.hook_adapter = hook_adapter or ToolHookAdapter()
        self.hook_bus = hook_bus or NoOpHookBus()
        self.run_id = run_id
        self.cwd = cwd
        self.project_context_message = project_context_message
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.is_subagent = bool(is_subagent)
        self.file_state = file_state
        self.executor = executor
        self.tool_result_callback = tool_result_callback
        self.last_results: tuple[ToolExecutionResult, ...] = ()
        self._call_index = 0

    @classmethod
    def from_tool_pool(
        cls,
        pool: ToolPool,
        **kwargs: Any,
    ) -> "ToolExecutionRuntime":
        return cls(pool, **kwargs)

    def execute_tool_uses(self, blocks: Iterable[Any]) -> tuple[list[dict[str, Any]], list[str]]:
        messages: list[dict[str, Any]] = []
        tools_used: list[str] = []
        results: list[ToolExecutionResult] = []

        for block in blocks:
            if _block_value(block, "type") != "tool_use":
                continue
            request = ToolUseRequest.from_block(block)
            result = self._execute_one(request)
            self._notify_tool_result(request, result)
            results.append(result)
            tools_used.append(result.tool_name)
            messages.extend(result.messages)

        self.last_results = tuple(results)
        return messages, tools_used

    def _notify_tool_result(self, request: ToolUseRequest, result: ToolExecutionResult) -> None:
        if self.tool_result_callback is None:
            return
        try:
            self.tool_result_callback(request, result)
        except Exception:
            _log.debug("tool_result_callback failed", exc_info=True)

    def _execute_one(self, request: ToolUseRequest) -> ToolExecutionResult:
        started = time.monotonic()
        tool = self.tools.get(request.name)
        if tool is None:
            return self._error_result(
                request,
                f"UnknownToolError: unknown tool: {request.name}",
                started,
            )

        deferred_error: Exception | None = None
        result: ToolExecutionResult | None = None
        with runtime_span(
            f"tool.{tool.name}",
            SpanKind.TOOL,
            **{
                "tool.name": tool.name,
                "tool.subagent": self.is_subagent,
                "tool.fork": self.is_subagent,
                **_tool_input_attrs(request.input),
                **_tool_metadata_attrs(tool),
            },
        ) as sp:
            try:
                result = self._execute_known_tool(tool, request, started, active_span=sp)
            except Exception as exc:
                deferred_error = exc
                safe_set_current_span(
                    **{
                        "tool.is_error": True,
                        "tool.error_type": type(exc).__name__,
                    }
                )
                sp.error(f"tool_exception:{type(exc).__name__}")
            else:
                message = result.messages[0] if result.messages else {}
                output = str(message.get("content", ""))
                safe_set_current_span(
                    **{
                        **content_summary_attrs("tool.output", output),
                        **content_preview_attrs("tool.output", output),
                        "tool.output_chars": len(output),
                        "tool.output_stored": result.persisted,
                        "tool.is_error": result.is_error,
                        "tool.permission": result.permission.behavior,
                        "tool.persisted": result.persisted,
                        "tool.raw_chars": result.raw_chars,
                        "tool.persist_path": result.persist_path,
                    }
                )
                record_permission_decision(result.permission)
                if result.is_error or result.counts_as_tool_error:
                    sp.error(_tool_error_summary(result, getattr(sp, "attributes", {})))
                if result.counts_as_tool_error:
                    self._increment_main_error_count()
        if deferred_error is not None:
            raise deferred_error
        assert result is not None
        return result

    def _execute_known_tool(
        self,
        tool: Tool,
        request: ToolUseRequest,
        started: float,
        *,
        active_span: Any,
    ) -> ToolExecutionResult:
        validation_error = self._validate_tool_input(tool, request.input)
        if validation_error is not None:
            return self._error_result(request, validation_error, started)

        tool_input = dict(request.input)
        pre_decision = self.hook_adapter.pre_tool_use(tool, request)
        safe_set_current_span(
            **{
                "pre_hook.adapter_updated_input_present": pre_decision.updated_input is not None,
                "pre_hook.adapter_messages_count": len(pre_decision.additional_messages),
                "pre_hook.adapter_stop_present": bool(pre_decision.stop_reason),
            }
        )
        if pre_decision.updated_input is not None:
            validation_error = self._validate_tool_input(
                tool,
                pre_decision.updated_input,
                phase="PreToolUse updated_input",
            )
            if validation_error is not None:
                return self._error_result(
                    request,
                    validation_error,
                    started,
                    additional_messages=pre_decision.additional_messages,
                )
            tool_input = dict(pre_decision.updated_input)
        if pre_decision.stop_reason:
            return self._error_result(
                request,
                f"PreToolUseStop: {pre_decision.stop_reason}",
                started,
                additional_messages=pre_decision.additional_messages,
            )

        effective_request = ToolUseRequest(request.id, request.name, tool_input)
        pre_hook = self._run_tool_hook("PreToolUse", tool, effective_request, tool_input=tool_input)
        _record_hook_attrs("pre_hook", pre_hook)
        pre_hook_messages = _hook_result_blocks(pre_hook)
        if pre_hook.updated_input is not None:
            validation_error = self._validate_tool_input(
                tool,
                pre_hook.updated_input,
                phase="HookBus updated_input",
            )
            if validation_error is not None:
                return self._error_result(
                    request,
                    validation_error,
                    started,
                    additional_messages=pre_decision.additional_messages,
                    hook_messages=pre_hook_messages,
                )
            tool_input = dict(pre_hook.updated_input)
            effective_request = ToolUseRequest(request.id, request.name, tool_input)
        if pre_hook.blocking_error or pre_hook.prevent_continuation:
            return self._error_result(
                request,
                _pre_hook_block_message(pre_hook),
                started,
                additional_messages=pre_decision.additional_messages,
                hook_messages=pre_hook_messages,
            )
        if pre_hook.permission_behavior == "deny":
            permission = PermissionDecision(
                "deny",
                message=_hook_permission_message(pre_hook),
                source="hook_bus",
            )
            return self._error_result(
                request,
                _permission_message("HookPermissionDenied", permission),
                started,
                permission=permission,
                additional_messages=pre_decision.additional_messages,
                hook_messages=pre_hook_messages,
            )

        permission = self.permission_engine.decide(tool, tool_input)
        if permission.updated_input is not None:
            validation_error = self._validate_tool_input(
                tool,
                permission.updated_input,
                phase="Permission updated_input",
            )
            if validation_error is not None:
                return self._error_result(
                    request,
                    validation_error,
                    started,
                    permission=permission,
                    additional_messages=pre_decision.additional_messages,
                    hook_messages=pre_hook_messages,
                )
            tool_input = dict(permission.updated_input)
            effective_request = ToolUseRequest(request.id, request.name, tool_input)

        if permission.behavior == "deny":
            return self._error_result(
                request,
                _permission_message("PermissionDenied", permission),
                started,
                permission=permission,
                additional_messages=pre_decision.additional_messages,
                hook_messages=pre_hook_messages,
            )
        if permission.behavior == "ask":
            return self._error_result(
                request,
                _permission_message(
                    "PermissionAskUnsupported: permission ask is unsupported",
                    permission,
                ),
                started,
                permission=permission,
                additional_messages=pre_decision.additional_messages,
                hook_messages=pre_hook_messages,
            )
        if permission.behavior not in {"allow", "passthrough"}:
            return self._error_result(
                request,
                f"PermissionError: unsupported permission behavior: {permission.behavior}",
                started,
                permission=permission,
                additional_messages=pre_decision.additional_messages,
                hook_messages=pre_hook_messages,
            )

        approval_denial = self._approval_denial(tool, tool_input)
        if approval_denial is not None:
            return self._error_result(
                request,
                _permission_message("ApprovalDenied", approval_denial),
                started,
                permission=approval_denial,
                additional_messages=pre_decision.additional_messages,
                hook_messages=pre_hook_messages,
            )

        span_status_before_call, span_message_before_call = _span_error_snapshot(active_span)
        try:
            raw_result = tool.call(dict(tool_input), self._tool_context())
            (
                output,
                result_is_error,
                result_messages,
                persisted,
                raw_chars,
                persist_path,
            ) = self._normalize_tool_result(tool, raw_result)
            tool_marked_error = _span_error_changed(
                active_span,
                span_status_before_call,
                span_message_before_call,
            )
            if (
                tool_marked_error
                and not result_is_error
                and not getattr(active_span, "attributes", {}).get("tool.error_kind")
            ):
                safe_set_current_span(**{"tool.error_kind": "marked_error"})
        except Exception as exc:
            failure_request = ToolUseRequest(request.id, request.name, tool_input)
            failure_hook = self._run_tool_hook(
                "PostToolUseFailure",
                tool,
                failure_request,
                tool_input=tool_input,
                error=exc,
            )
            _record_hook_attrs("failure_hook", failure_hook)
            return self._error_result(
                request,
                _runtime_exception_message(exc),
                started,
                permission=permission,
                additional_messages=pre_decision.additional_messages,
                hook_messages=_hook_result_blocks(failure_hook),
                counts_as_tool_error=True,
            )

        message = {
            "type": "tool_result",
            "tool_use_id": request.id,
            "content": output,
        }
        if result_is_error:
            message["is_error"] = True
        result = ToolExecutionResult(
            messages=(message,),
            tool_name=request.name,
            is_error=result_is_error,
            permission=permission,
            duration_ms=_elapsed_ms(started),
            additional_messages=(*pre_decision.additional_messages, *result_messages),
            counts_as_tool_error=result_is_error or tool_marked_error,
            persisted=persisted,
            raw_chars=raw_chars,
            persist_path=persist_path or "",
        )
        post_decision = self.hook_adapter.post_tool_use(tool, effective_request, result)
        safe_set_current_span(
            **{
                "post_hook.adapter_messages_count": len(post_decision.additional_messages),
                "post_hook.adapter_stop_present": bool(post_decision.stop_reason),
                "post_hook.adapter_updated_input_present": post_decision.updated_input is not None,
            }
        )
        lifecycle_additional_messages = (
            *pre_decision.additional_messages,
            *result_messages,
            *post_decision.additional_messages,
        )
        post_hook = self._run_tool_hook(
            "PostToolUse",
            tool,
            effective_request,
            tool_input=tool_input,
            tool_output=output,
            result=result,
        )
        _record_hook_attrs("post_hook", post_hook)
        hook_messages = (*pre_hook_messages, *_hook_result_blocks(post_hook))
        request_view_messages = tuple(
            message for message in result_messages if is_durable_request_message(message)
        )
        return ToolExecutionResult(
            messages=(message, *hook_messages, *request_view_messages),
            tool_name=request.name,
            is_error=result_is_error,
            permission=permission,
            duration_ms=result.duration_ms,
            additional_messages=lifecycle_additional_messages,
            counts_as_tool_error=result.counts_as_tool_error,
            persisted=result.persisted,
            raw_chars=result.raw_chars,
            persist_path=result.persist_path,
        )

    def _validate_tool_input(
        self,
        tool: Tool,
        tool_input: Any,
        *,
        phase: str = "initial input",
    ) -> str | None:
        shallow_error = _validate_input(dict(tool.input_schema), tool_input)
        if shallow_error is not None:
            return shallow_error
        validator = tool.validate_input
        if validator is None:
            return None
        try:
            message = validator(dict(tool_input), self._tool_context())
        except Exception as exc:
            return f"InputValidationError: {type(exc).__name__}: {exc}"
        if not message:
            return None
        prefix = "InputValidationError"
        if str(message).startswith(prefix):
            return str(message)
        return f"{prefix}: {message}"

    def _normalize_tool_result(
        self,
        tool: Tool,
        result: ToolResult | str,
    ) -> tuple[str, bool, tuple[dict[str, Any], ...], bool, int, str | None]:
        if tool.map_result is not None:
            mapped = tool.map_result(result)
            output = str(mapped)
        elif isinstance(result, ToolResult):
            output = result.content
        else:
            output = str(result)

        if not output or not output.strip():
            output = f"({tool.name} 执行完成，无输出)"

        is_error = _result_is_error(result) or output.startswith("Error")
        raw_chars = len(output)
        self._call_index += 1
        output, persisted, persist_path = result_store.maybe_persist(
            tool.name,
            f"{tool.name}_{self._call_index}",
            output,
        )
        if persisted:
            _log.debug(
                "persisted oversized tool output",
                extra={
                    "tool_name": tool.name,
                    "raw_chars": raw_chars,
                    "persist_path": persist_path,
                },
            )
        return output, is_error, _result_messages(result), persisted, raw_chars, persist_path

    def _tool_context(self) -> ToolContext:
        from .executors import get_executor
        from .file_state import get_current_file_read_state

        return ToolContext(
            run_id=self.run_id,
            cwd=self._hook_cwd(),
            file_state=self.file_state or get_current_file_read_state(),
            executor=self.executor or get_executor(),
            hook_bus=self.hook_bus,
            permission_engine=self.permission_engine,
            project_context_message=self.project_context_message,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            is_subagent=self.is_subagent,
        )

    def _run_tool_hook(
        self,
        event: str,
        tool: Tool,
        request: ToolUseRequest,
        *,
        tool_input: dict[str, Any],
        tool_output: Any = None,
        error: Any = None,
        result: ToolExecutionResult | None = None,
    ) -> HookResult:
        hook_input = HookInput(
            event=event,
            run_id=self.run_id,
            cwd=self._hook_cwd(),
            payload={
                "tool_name": tool.name,
                "tool_use_id": request.id,
                "is_read_only": _flag_value(tool.is_read_only, tool_input),
                "is_destructive": _flag_value(tool.is_destructive, tool_input),
                "is_concurrency_safe": _flag_value(tool.is_concurrency_safe, tool_input),
                "tool_metadata": _copy_hook_metadata(tool.metadata),
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "is_subagent": self.is_subagent,
                "result": result,
            },
            tool_name=tool.name,
            tool_use_id=request.id,
            tool_input=dict(tool_input),
            tool_output=tool_output,
            error=error,
        )
        hook_result = self.hook_bus.run(hook_input)
        report_hook_errors(hook_result, hook_input, _log)
        return hook_result

    def _hook_cwd(self) -> str:
        if self.cwd is not None:
            return self.cwd
        try:
            from .executors import get_executor

            return str(get_executor().cwd)
        except Exception:
            return ""

    def _increment_main_error_count(self) -> None:
        if self.is_subagent:
            return
        try:
            from .executors import increment_tool_error_count

            increment_tool_error_count()
        except Exception:
            return

    def _approval_denial(
        self,
        tool: Tool,
        tool_input: dict[str, Any],
    ) -> PermissionDecision | None:
        if self.is_subagent:
            return None
        try:
            from .executors import get_approve_cb

            approve_cb = get_approve_cb()
        except Exception:
            return None
        if approve_cb is None:
            return None
        if approve_cb(tool.name, dict(tool_input)):
            return None
        return PermissionDecision(
            "deny",
            message=f"用户拒绝执行 {tool.name}；请换一种方式或询问用户。",
            source="approval_callback",
        )

    def _error_result(
        self,
        request: ToolUseRequest,
        content: str,
        started: float,
        *,
        permission: PermissionDecision | None = None,
        additional_messages: tuple[dict[str, Any], ...] = (),
        hook_messages: tuple[dict[str, Any], ...] = (),
        counts_as_tool_error: bool = False,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            messages=(
                {
                    "type": "tool_result",
                    "tool_use_id": request.id,
                    "content": content,
                    "is_error": True,
                },
                *hook_messages,
            ),
            tool_name=request.name,
            is_error=True,
            permission=permission or PermissionDecision("passthrough", source="not_evaluated"),
            duration_ms=_elapsed_ms(started),
            additional_messages=additional_messages,
            counts_as_tool_error=counts_as_tool_error,
        )


def _duplicate_tool_names(tools: tuple[Tool, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for tool in tools:
        if tool.name in seen:
            duplicates.append(tool.name)
        seen.add(tool.name)
    return duplicates


def _result_is_error(result: ToolResult | str) -> bool:
    return isinstance(result, ToolResult) and result.is_error


def _result_messages(result: ToolResult | str) -> tuple[dict[str, Any], ...]:
    if isinstance(result, ToolResult):
        return result.additional_messages
    return ()


def _span_error_snapshot(active_span: Any) -> tuple[str, str]:
    return (
        str(getattr(active_span, "status", "")),
        str(getattr(active_span, "status_message", "")),
    )


def _span_error_changed(
    active_span: Any,
    prior_status: str,
    prior_message: str,
) -> bool:
    return _span_error_snapshot(active_span) != (prior_status, prior_message)


def _flag_value(flag, tool_input: dict[str, Any]) -> bool:
    if callable(flag):
        try:
            return bool(flag(dict(tool_input)))
        except Exception:
            return False
    return bool(flag)


def _copy_hook_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_hook_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy_hook_metadata(item) for item in value]
    if isinstance(value, list):
        return [_copy_hook_metadata(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_copy_hook_metadata(item) for item in value]
    return value


def _tool_input_attrs(tool_input: Any) -> dict[str, Any]:
    summary_attrs = {
        **content_summary_attrs("tool.input", tool_input),
        **content_preview_attrs("tool.input", tool_input),
    }
    if isinstance(tool_input, Mapping):
        command = tool_input.get("command")
        if command:
            summary_attrs["tool.display.command"] = str(command)
            summary_attrs.update(content_summary_attrs("tool.command", command))
            summary_attrs.update(content_preview_attrs("tool.command", command))
            summary_attrs["tool.command_chars"] = safe_text_length(command)
        path = tool_input.get("path")
        if path:
            summary_attrs["tool.display.path"] = str(path)
        pattern = tool_input.get("pattern")
        if pattern:
            summary_attrs["tool.display.pattern"] = str(pattern)
        fields = tuple(sorted(str(key) for key in tool_input.keys()))
        return {
            **summary_attrs,
            "tool.input_type": "object",
            "tool.input_chars": safe_text_length(tool_input),
            "tool.input_field_count": len(fields),
            "tool.input_fields": fields,
        }
    return {
        **summary_attrs,
        "tool.input_type": type(tool_input).__name__,
        "tool.input_chars": safe_text_length(tool_input),
        "tool.input_field_count": 0,
        "tool.input_fields": (),
    }


def _tool_metadata_attrs(tool: Tool) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "tool.source": str(getattr(tool, "source", "") or ""),
        "tool.is_mcp": False,
    }
    metadata = getattr(tool, "metadata", None)
    if not isinstance(metadata, Mapping) or not bool(metadata.get("is_mcp")):
        return attrs

    attrs["tool.is_mcp"] = True
    mcp_metadata = metadata.get("mcp")
    if not isinstance(mcp_metadata, Mapping):
        return attrs

    for key in ("server_name", "tool_name", "permission_name"):
        value = mcp_metadata.get(key)
        if value:
            attrs[f"mcp.{key}"] = str(value)
    attrs["mcp.always_load"] = bool(mcp_metadata.get("always_load", False))

    annotations = mcp_metadata.get("annotations")
    if isinstance(annotations, Mapping):
        for key in ("read_only", "destructive", "open_world", "concurrency_safe"):
            if key in annotations:
                attrs[f"mcp.annotation.{key}"] = bool(annotations[key])
    return attrs


def _record_hook_attrs(prefix: str, result: HookResult) -> None:
    safe_set_current_span(
        **{
            f"{prefix}.messages_count": len(result.messages),
            f"{prefix}.contexts_count": len(result.additional_contexts),
            f"{prefix}.updated_input_present": result.updated_input is not None,
            f"{prefix}.permission_behavior": result.permission_behavior,
            f"{prefix}.blocking": bool(result.blocking_error or result.prevent_continuation),
            f"{prefix}.error_count": len(result.errors),
        }
    )


def _tool_error_summary(result: ToolExecutionResult, attrs: Mapping[str, Any]) -> str:
    permission_behavior = str(getattr(result.permission, "behavior", "") or "")
    if permission_behavior in {"deny", "ask"}:
        return f"tool_error:permission={permission_behavior}"
    error_kind = str(attrs.get("tool.error_kind") or "")
    exit_code = attrs.get("tool.exit_code")
    if error_kind:
        if exit_code is not None:
            return f"tool_error:{error_kind}:rc={exit_code}"
        return f"tool_error:{error_kind}"
    return f"tool_error:{result.tool_name}"


def _block_value(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _permission_message(prefix: str, decision: PermissionDecision) -> str:
    if decision.message:
        return f"{prefix}: {decision.message}"
    return f"{prefix}: source={decision.source}"


def _hook_permission_message(result: HookResult) -> str:
    return result.blocking_error or result.stop_reason or "denied by hook"


def _pre_hook_block_message(result: HookResult) -> str:
    if result.blocking_error:
        return f"PreToolUseBlocked: {result.blocking_error}"
    if result.stop_reason:
        return f"PreToolUseStop: {result.stop_reason}"
    return "PreToolUseStop: prevent_continuation"


def _runtime_exception_message(exc: Exception) -> str:
    return f"ToolExecutionError: {type(exc).__name__}: {exc}"


def _hook_result_blocks(result: HookResult) -> tuple[dict[str, Any], ...]:
    blocks = [_content_block(message) for message in result.messages]
    blocks.extend(_content_block(context) for context in result.additional_contexts)
    return tuple(blocks)


def _content_block(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {"type": "text", "text": str(value)}


def _validate_input(schema: dict[str, Any], tool_input: Any) -> str | None:
    if not isinstance(tool_input, dict):
        return "InputValidationError: tool input must be an object"

    required = schema.get("required") or []
    for field_name in required:
        if field_name not in tool_input:
            return f"InputValidationError: missing required field: {field_name}"

    properties = schema.get("properties") or {}
    for field_name, value in tool_input.items():
        prop_schema = properties.get(field_name) or {}
        expected = prop_schema.get("type")
        if expected is None:
            continue
        if not _matches_json_type(value, expected):
            return (
                "InputValidationError: field "
                f"{field_name} expected {expected}, got {_json_type_name(value)}"
            )
    return None


def _matches_json_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_matches_json_type(value, item) for item in expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def _json_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


__all__ = [
    "HookDecision",
    "PermissionDecision",
    "PermissionEngine",
    "ToolExecutionResult",
    "ToolExecutionRuntime",
    "ToolHookAdapter",
    "ToolUseRequest",
]
