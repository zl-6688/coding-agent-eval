"""Tool contracts used to project model, prompt, and runtime views."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping


ToolFlag = bool | Callable[[dict[str, Any]], bool]
Validator = Callable[[dict[str, Any], "ToolContext"], str | None]
ToolCall = Callable[[dict[str, Any], "ToolContext"], "ToolResult | str"]
ResultMapper = Callable[["ToolResult | str"], str]


@dataclass(frozen=True)
class ToolContext:
    run_id: str = ""
    cwd: str = ""
    file_state: Any = None
    executor: Any = None
    hook_bus: Any = None
    permission_engine: Any = None
    project_context_message: Mapping[str, Any] | None = None
    agent_id: str = ""
    agent_type: str = "main"
    is_subagent: bool = False


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    additional_messages: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", str(self.content))
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))
        object.__setattr__(
            self,
            "additional_messages",
            tuple(copy.deepcopy(message) for message in self.additional_messages),
        )


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    call: ToolCall
    source: str = "core_builtin"
    is_read_only: ToolFlag = False
    is_destructive: ToolFlag = False
    is_concurrency_safe: ToolFlag = False
    validate_input: Validator | None = None
    map_result: ResultMapper | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool.name must be non-empty")
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("Tool.input_schema must be a mapping")
        if not callable(self.call):
            raise TypeError("Tool.call must be callable")
        object.__setattr__(self, "input_schema", _freeze_value(self.input_schema))
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))

    def to_model_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": _copy_mapping(self.input_schema),
        }

    def to_prompt_tool(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _copy_value(item) for key, item in value.items()}


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
    "ResultMapper",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolFlag",
    "ToolResult",
    "Validator",
]
