"""Per-request tool schema views.

ToolPool remains the complete runtime set. This builder decides which schemas
are sent to the LLM for one request and which reduced tool list is safe for the
system prompt tools section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import Tool
from .deferred import (
    DeferredToolPolicy,
    DeferredToolState,
    build_deferred_index_context_message,
    deferred_candidates,
)
from .pool import ToolPool


@dataclass(frozen=True)
class ToolRequestView:
    schemas: list[dict[str, Any]]
    prompt_tools: list[dict[str, str]]
    deferred_index_context_message: dict[str, Any] | None
    visible_names: frozenset[str]
    deferred_names: frozenset[str]


def build_tool_request_view(
    tool_pool: ToolPool,
    *,
    policy: DeferredToolPolicy | None = None,
    state: DeferredToolState | None = None,
    messages: Iterable[Mapping[str, Any]] | None = None,
) -> ToolRequestView:
    policy = policy or DeferredToolPolicy(enabled=False)
    if not policy.enabled:
        tools = list(tool_pool.tools)
        return ToolRequestView(
            schemas=[tool.to_model_schema() for tool in tools],
            prompt_tools=[tool.to_prompt_tool() for tool in tools],
            deferred_index_context_message=None,
            visible_names=frozenset(tool.name for tool in tools),
            deferred_names=frozenset(),
        )

    state = state or DeferredToolState()
    if messages is not None:
        state.restore_from_messages(messages)

    candidates = deferred_candidates(tool_pool.tools, policy)
    deferred_names = frozenset(tool.name for tool in candidates)
    selected_names = state.selected_names & deferred_names
    schema_tools = [
        tool for tool in tool_pool.tools if tool.name not in deferred_names or tool.name in selected_names
    ]
    prompt_tools = [tool for tool in tool_pool.tools if tool.name not in deferred_names]

    return ToolRequestView(
        schemas=[tool.to_model_schema() for tool in schema_tools],
        prompt_tools=[tool.to_prompt_tool() for tool in prompt_tools],
        deferred_index_context_message=build_deferred_index_context_message(candidates),
        visible_names=frozenset(tool.name for tool in schema_tools),
        deferred_names=deferred_names,
    )


__all__ = [
    "ToolRequestView",
    "build_tool_request_view",
]
