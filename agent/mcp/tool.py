"""Convert MCP server tools into the project's regular Tool contract."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

from agent.tools.contracts import Tool, ToolContext, ToolResult

from .names import build_mcp_tool_name
from .types import McpToolDefinition, McpToolResult


def create_mcp_tool(definition: McpToolDefinition) -> Tool:
    name = build_mcp_tool_name(definition.server_name, definition.tool_name)
    permission_name = name

    def _call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        result = definition.call(dict(tool_input), context)
        return _normalize_mcp_result(result)

    return Tool(
        name=name,
        description=definition.description,
        input_schema=definition.input_schema,
        call=_call,
        source="mcp",
        is_read_only=definition.annotations.read_only,
        is_destructive=definition.annotations.destructive,
        is_concurrency_safe=definition.annotations.concurrency_safe,
        metadata={
            "is_mcp": True,
            "mcp": {
                "server_name": definition.server_name,
                "tool_name": definition.tool_name,
                "permission_name": permission_name,
                "search_hint": definition.search_hint,
                "always_load": definition.always_load,
                "annotations": definition.annotations.to_metadata(),
            },
            "input_schema_source": "mcp",
        },
    )


def create_mcp_tools(definitions: Iterable[McpToolDefinition]) -> tuple[Tool, ...]:
    return tuple(create_mcp_tool(definition) for definition in definitions)


def _normalize_mcp_result(result: ToolResult | McpToolResult | str) -> ToolResult:
    if isinstance(result, ToolResult):
        return result
    if isinstance(result, McpToolResult):
        return ToolResult(
            content=result.content,
            is_error=result.is_error,
            metadata={"mcp": _copy_value(result.metadata)},
        )
    return ToolResult(content=str(result))


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy_value(item) for item in value]
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, frozenset):
        return {_copy_value(item) for item in value}
    if isinstance(value, set):
        return {_copy_value(item) for item in value}
    return copy.deepcopy(value)


__all__ = [
    "create_mcp_tool",
    "create_mcp_tools",
]
