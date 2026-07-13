"""One-shot Agent tool implementation.

The first implementation intentionally supports only synchronous fresh
subagents. It consumes the parent runtime's already-loaded request context and
reuses the parent permission and hook objects for child tool execution.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from .. import llm
from ..context.request_view import build_request_view
from ..context.system_prompt import SystemState, build_system
from ..runtime.hooks import HookInput, report_hook_errors
from ..runtime.observability import (
    content_preview_attrs,
    content_summary_attrs,
    safe_set_current_span,
)
from ..tools.file_state import FileReadState
from ..tools.pool import ToolPoolContext, assemble_tool_pool
from ..tools.runtime import ToolExecutionRuntime
from ..tools.contracts import ToolContext, ToolResult
from obs.trace import SpanKind, span


AGENT_TOOL_NAME = "Agent"
GENERAL_PURPOSE_AGENT_TYPE = "general-purpose"
DEFAULT_MAX_TURNS = 6
MAX_MAX_TURNS = 20
SUBAGENT_SYSTEM_IDENTITY = """You are a focused coding subagent.

You receive one delegated task and work independently from the parent
conversation. Use the available tools when needed, then return a concise final
answer for the parent agent. Do not attempt to spawn another Agent; that tool is
not available in this child tool pool."""


@dataclass(frozen=True)
class SubagentDefinition:
    agent_type: str
    description: str


@dataclass(frozen=True)
class SubagentConfig:
    prompt: str
    description: str
    description_provided: bool
    max_turns: int
    agent_id: str
    definition: SubagentDefinition
    cwd: str
    run_id: str
    project_context_message: Mapping[str, Any] | None
    permission_engine: Any
    hook_bus: Any


@dataclass(frozen=True)
class SubagentResult:
    agent_id: str
    agent_type: str
    description: str
    status: str
    final_text: str
    turns: int
    tool_use_count: int
    duration_ms: int
    is_error: bool = False
    messages: tuple[dict[str, Any], ...] = field(default_factory=tuple)


_GENERAL_PURPOSE = SubagentDefinition(
    agent_type=GENERAL_PURPOSE_AGENT_TYPE,
    description="General-purpose subagent for delegated coding tasks.",
)


def call_agent_tool(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    """Core tool entrypoint used by ToolExecutionRuntime."""

    agent_type = str(tool_input.get("subagent_type") or GENERAL_PURPOSE_AGENT_TYPE)
    if agent_type != GENERAL_PURPOSE_AGENT_TYPE:
        return ToolResult(
            content=(
                "UnsupportedSubagentType: "
                f"{agent_type!r}. Supported subagent types: {GENERAL_PURPOSE_AGENT_TYPE}"
            ),
            is_error=True,
            metadata={"status": "error", "agent_type": agent_type},
        )

    prompt = str(tool_input["prompt"])
    max_turns = int(tool_input.get("max_turns", DEFAULT_MAX_TURNS))
    raw_description = tool_input.get("description")
    description_provided = bool(str(raw_description or "").strip())
    description = str(raw_description) if description_provided else _default_description(prompt)
    agent_id = f"agent_{uuid.uuid4().hex[:12]}"
    config = SubagentConfig(
        prompt=prompt,
        description=description,
        description_provided=description_provided,
        max_turns=max_turns,
        agent_id=agent_id,
        definition=_GENERAL_PURPOSE,
        cwd=context.cwd,
        run_id=context.run_id,
        project_context_message=context.project_context_message,
        permission_engine=context.permission_engine,
        hook_bus=context.hook_bus,
    )
    result = SubagentRunner(config).run()
    metadata = {
        "agent_id": result.agent_id,
        "agent_type": result.agent_type,
        "description": result.description,
        "status": result.status,
        "turns": result.turns,
        "tool_use_count": result.tool_use_count,
        "duration_ms": result.duration_ms,
    }
    return ToolResult(
        content=_format_result(result),
        is_error=result.is_error,
        metadata=metadata,
        additional_messages=({"type": "metadata", "metadata": metadata},),
    )


class SubagentRunner:
    def __init__(self, config: SubagentConfig) -> None:
        self.config = config
        self._messages: list[dict[str, Any]] = [
            {"role": "user", "content": config.prompt}
        ]
        self._turns = 0
        self._tool_use_count = 0
        self._last_text = ""
        self._started = time.monotonic()

    def run(self) -> SubagentResult:
        span_attrs = {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.definition.agent_type,
            "is_subagent": True,
            **content_summary_attrs("subagent.prompt", self.config.prompt),
            **content_preview_attrs("subagent.prompt", self.config.prompt),
            **content_summary_attrs("subagent.description", self.config.description),
            **content_preview_attrs("subagent.description", self.config.description),
            "description_provided": self.config.description_provided,
            "description_chars": len(self.config.description)
            if self.config.description_provided
            else 0,
            "max_turns": self.config.max_turns,
            "parent_run_id_present": bool(self.config.run_id),
            "mcp_inherited": False,
            "tool_pool_size": 0,
            "excluded_tools_count": 0,
            "outcome": "running",
        }
        with span(
            "agent.subagent",
            SpanKind.AGENT,
            **span_attrs,
        ) as sp:
            try:
                result = self._run_loop()
            except Exception as exc:
                result = self._result(
                    status="error",
                    final_text=f"SubagentError: {type(exc).__name__}: {exc}",
                    is_error=True,
                )
            safe_set_current_span(
                **{
                    "status": result.status,
                    "outcome": result.status,
                    "turns": result.turns,
                    "tool_use_count": result.tool_use_count,
                    "duration_ms": result.duration_ms,
                }
            )
            if result.is_error:
                sp.error(f"subagent_error:{result.status}")
            self._run_stop_hook(result)
            return result

    def _run_loop(self) -> SubagentResult:
        excluded_tool_names = frozenset({AGENT_TOOL_NAME, "TaskOutput", "TaskStop"})
        child_pool = assemble_tool_pool(
            ToolPoolContext(
                workdir=self.config.cwd,
                enable_skills=False,
                exclude_tool_names=excluded_tool_names,
            )
        )
        safe_set_current_span(
            tool_pool_size=len(child_pool.tools),
            excluded_tools_count=len(excluded_tool_names),
            mcp_inherited=False,
        )
        child_runtime = ToolExecutionRuntime.from_tool_pool(
            child_pool,
            permission_engine=self.config.permission_engine,
            hook_bus=self.config.hook_bus,
            run_id=self.config.run_id,
            cwd=self.config.cwd,
            project_context_message=dict(self.config.project_context_message)
            if self.config.project_context_message is not None
            else None,
            agent_id=self.config.agent_id,
            agent_type=self.config.definition.agent_type,
            is_subagent=True,
            file_state=FileReadState(),
        )
        system = build_system(
            SystemState(
                tools=child_pool.prompt_tools_for_system(),
                workdir=self.config.cwd,
                memory_dir=None,
            ),
            identity=SUBAGENT_SYSTEM_IDENTITY,
        )

        while self._turns < self.config.max_turns:
            self._turns += 1
            request_messages = _compose_request_messages(
                self._messages,
                self.config.project_context_message,
            )
            resp = llm.chat(
                request_messages,
                system=system,
                tools=child_pool.model_schemas_for_api(),
                max_tokens=4096,
                purpose="subagent",
            )
            assistant_message = {"role": "assistant", "content": resp.content}
            self._messages.append(assistant_message)
            text = _final_text(resp.content)
            if text:
                self._last_text = text

            if getattr(resp, "stop_reason", None) != "tool_use":
                return self._result(
                    status="completed",
                    final_text=text or "(Subagent completed but returned no output.)",
                )

            tool_blocks = [
                block for block in resp.content if _block_value(block, "type") == "tool_use"
            ]
            result_messages, tools_used = child_runtime.execute_tool_uses(tool_blocks)
            self._tool_use_count += len(tools_used)
            self._messages.append({"role": "user", "content": result_messages})

        return self._result(
            status="max_turns",
            final_text=self._last_text
            or "(Subagent reached max_turns before final output.)",
        )

    def _result(self, *, status: str, final_text: str, is_error: bool = False) -> SubagentResult:
        return SubagentResult(
            agent_id=self.config.agent_id,
            agent_type=self.config.definition.agent_type,
            description=self.config.description,
            status=status,
            final_text=final_text,
            turns=self._turns,
            tool_use_count=self._tool_use_count,
            duration_ms=int((time.monotonic() - self._started) * 1000),
            is_error=is_error,
            messages=tuple(dict(message) for message in self._messages),
        )

    def _run_stop_hook(self, result: SubagentResult) -> None:
        hook_bus = self.config.hook_bus
        if hook_bus is None:
            return
        hook_input = HookInput(
            event="SubagentStop",
            run_id=self.config.run_id,
            cwd=self.config.cwd,
            payload={
                "agent_id": result.agent_id,
                "agent_type": result.agent_type,
                "is_subagent": True,
                "status": result.status,
                "turns": result.turns,
                "tool_use_count": result.tool_use_count,
                "final_text": result.final_text,
            },
            prompt=self.config.prompt,
            last_assistant_message=_last_assistant_message(self._messages),
        )
        hook_result = hook_bus.run(hook_input)
        report_hook_errors(hook_result, hook_input)


def _compose_request_messages(
    messages: list[dict[str, Any]],
    project_context_message: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build a subagent request without exposing parent transcript messages."""

    return build_request_view(
        messages,
        query_context_messages=(project_context_message,)
        if project_context_message is not None
        else (),
    ).as_messages()


def _format_result(result: SubagentResult) -> str:
    return "\n".join(
        [
            f"agentId: {result.agent_id}",
            f"agentType: {result.agent_type}",
            f"description: {_single_line(result.description)}",
            f"status: {result.status}",
            f"turns: {result.turns}",
            f"tool_uses: {result.tool_use_count}",
            f"duration_ms: {result.duration_ms}",
            "",
            result.final_text,
        ]
    )


def _default_description(prompt: str) -> str:
    compact = " ".join(prompt.strip().split())
    if len(compact) <= 80:
        return compact
    return compact[:77] + "..."


def _single_line(value: str) -> str:
    return " ".join(str(value).split())


def _final_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    text_parts: list[str] = []
    for block in content or ():
        if _block_value(block, "type") == "text":
            text_parts.append(str(_block_value(block, "text", "")))
    return "".join(text_parts)


def _last_assistant_message(messages: list[dict[str, Any]]) -> Any:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content")
    return None


def _block_value(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


__all__ = [
    "AGENT_TOOL_NAME",
    "DEFAULT_MAX_TURNS",
    "GENERAL_PURPOSE_AGENT_TYPE",
    "MAX_MAX_TURNS",
    "SubagentDefinition",
    "SubagentResult",
    "call_agent_tool",
]
