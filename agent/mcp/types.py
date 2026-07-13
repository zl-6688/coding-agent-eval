"""Typed MCP tool definitions before they become regular Tool objects."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from agent.tools.contracts import ToolContext, ToolResult

McpToolHandler = Callable[
    [dict[str, Any], ToolContext],
    "ToolResult | McpToolResult | str",
]
McpToolCall = McpToolHandler


@dataclass(frozen=True)
class McpToolAnnotations:
    read_only: bool = False
    destructive: bool = True
    open_world: bool = True
    concurrency_safe: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "McpToolAnnotations":
        data = dict(value or {})
        read_only = _bool_hint(data, "read_only", "readOnlyHint", default=False)
        destructive = _bool_hint(
            data,
            "destructive",
            "destructiveHint",
            default=not read_only,
        )
        return cls(
            read_only=read_only,
            destructive=destructive,
            open_world=_bool_hint(data, "open_world", "openWorldHint", default=True),
            concurrency_safe=_bool_hint(
                data,
                "concurrency_safe",
                "concurrencySafeHint",
                "idempotentHint",
                default=False,
            ),
        )

    def to_metadata(self) -> dict[str, bool]:
        return {
            "read_only": self.read_only,
            "destructive": self.destructive,
            "open_world": self.open_world,
            "concurrency_safe": self.concurrency_safe,
        }


def _bool_hint(
    data: Mapping[str, Any],
    *keys: str,
    default: bool,
) -> bool:
    for key in keys:
        if key in data and data[key] is not None:
            return bool(data[key])
    return bool(default)


@dataclass(frozen=True)
class McpToolResult:
    content: str
    is_error: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", str(self.content))
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))


@dataclass(frozen=True)
class McpToolDefinition:
    server_name: str
    tool_name: str
    description: str
    input_schema: Mapping[str, Any]
    call: McpToolHandler
    annotations: McpToolAnnotations = field(default_factory=McpToolAnnotations)
    search_hint: str = ""
    always_load: bool = False

    def __post_init__(self) -> None:
        if not str(self.server_name).strip():
            raise ValueError("McpToolDefinition.server_name must be non-empty")
        if not str(self.tool_name).strip():
            raise ValueError("McpToolDefinition.tool_name must be non-empty")
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("McpToolDefinition.input_schema must be a mapping")
        if not callable(self.call):
            raise TypeError("McpToolDefinition.call must be callable")
        annotations = self.annotations
        if isinstance(annotations, Mapping):
            annotations = McpToolAnnotations.from_mapping(annotations)
        if not isinstance(annotations, McpToolAnnotations):
            raise TypeError("McpToolDefinition.annotations must be McpToolAnnotations")
        object.__setattr__(self, "server_name", str(self.server_name))
        object.__setattr__(self, "tool_name", str(self.tool_name))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "input_schema", _freeze_value(self.input_schema))
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(self, "search_hint", str(self.search_hint or ""))
        object.__setattr__(self, "always_load", bool(self.always_load))


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {copy.deepcopy(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return copy.deepcopy(value)


__all__ = [
    "McpToolAnnotations",
    "McpToolCall",
    "McpToolDefinition",
    "McpToolHandler",
    "McpToolResult",
]
