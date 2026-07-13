"""Tool pool assembly for per-run model, prompt, and runtime views."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from agent.mcp import (
    McpToolDefinition,
    build_mcp_tool_name,
    create_mcp_tool,
)
from agent.skills import SkillCatalog, create_skill_tool, discover_skill_catalog

from .builtin_tools import get_core_tools
from .contracts import Tool
from .deferred import (
    TOOL_SEARCH_NAME,
    DeferredToolPolicy,
    create_tool_search_tool,
    deferred_candidates,
)


@dataclass(frozen=True)
class ToolPoolContext:
    workdir: str | None = None
    include_tool_names: frozenset[str] | None = None
    exclude_tool_names: frozenset[str] = field(default_factory=frozenset)
    enable_skills: bool = True
    skill_user_home: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    mcp_tool_definitions: tuple[McpToolDefinition, ...] = ()
    permission_engine: Any | None = None
    enable_deferred_tools: bool = False

    def __post_init__(self) -> None:
        if self.include_tool_names is not None:
            object.__setattr__(
                self,
                "include_tool_names",
                frozenset(str(name) for name in self.include_tool_names),
            )
        object.__setattr__(
            self,
            "exclude_tool_names",
            frozenset(str(name) for name in (self.exclude_tool_names or ())),
        )
        object.__setattr__(self, "enable_skills", bool(self.enable_skills))
        if self.skill_user_home is not None:
            object.__setattr__(self, "skill_user_home", str(self.skill_user_home))
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))
        object.__setattr__(
            self,
            "mcp_tool_definitions",
            tuple(self.mcp_tool_definitions or ()),
        )
        object.__setattr__(self, "enable_deferred_tools", bool(self.enable_deferred_tools))


@dataclass(frozen=True)
class ToolPool:
    tools: tuple[Tool, ...]
    model_schemas: tuple[Mapping[str, Any], ...] = field(init=False)
    prompt_tools: tuple[Mapping[str, str], ...] = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        tools = tuple(self.tools)
        seen: set[str] = set()
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError("ToolPool.tools must contain Tool objects")
            if tool.name in seen:
                raise ValueError(f"duplicate tool name: {tool.name}")
            seen.add(tool.name)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(
            self,
            "model_schemas",
            tuple(_freeze_value(tool.to_model_schema()) for tool in tools),
        )
        object.__setattr__(
            self,
            "prompt_tools",
            tuple(_freeze_value(tool.to_prompt_tool()) for tool in tools),
        )
        object.__setattr__(self, "fingerprint", _fingerprint(tools))

    def model_schemas_for_api(self) -> list[dict[str, Any]]:
        return [_copy_mapping(schema) for schema in self.model_schemas]

    def prompt_tools_for_system(self) -> list[dict[str, str]]:
        return [_copy_mapping(tool) for tool in self.prompt_tools]

    def find_tool(self, name: str) -> Tool | None:
        return find_tool_by_name(self, name)

    def filtered(
        self,
        *,
        include_tool_names: Iterable[str] | None = None,
        exclude_tool_names: Iterable[str] | None = None,
    ) -> "ToolPool":
        include = (
            None
            if include_tool_names is None
            else frozenset(str(name) for name in include_tool_names)
        )
        exclude = frozenset(str(name) for name in (exclude_tool_names or ()))
        tools = self.tools
        if include is not None:
            tools = tuple(tool for tool in tools if tool.name in include)
        if exclude:
            tools = tuple(tool for tool in tools if tool.name not in exclude)
        return ToolPool(tools)


def get_all_base_tools(source: Iterable[Tool] | None = None) -> tuple[Tool, ...]:
    return ToolPool(_normalize_tools(source)).tools


def get_tools(
    context: ToolPoolContext | None = None,
    source: Iterable[Tool] | None = None,
) -> tuple[Tool, ...]:
    return assemble_tool_pool(context, source=source).tools


def assemble_tool_pool(
    context: ToolPoolContext | None = None,
    source: Iterable[Tool] | None = None,
) -> ToolPool:
    context = context or ToolPoolContext()
    pool = ToolPool(_tools_for_context(context, source))
    return pool.filtered(
        include_tool_names=context.include_tool_names,
        exclude_tool_names=context.exclude_tool_names,
    )


def find_tool_by_name(tools_or_pool: Any, name: str) -> Tool | None:
    candidates = tools_or_pool.tools if isinstance(tools_or_pool, ToolPool) else tools_or_pool
    for candidate in candidates:
        if isinstance(candidate, Tool) and candidate.name == name:
            return candidate
        if isinstance(candidate, Mapping) and candidate.get("name") == name:
            raise TypeError("mapping tool schemas are not executable Tool objects")
    return None


def _normalize_tools(source: Iterable[Tool] | None) -> tuple[Tool, ...]:
    return tuple(get_core_tools() if source is None else source)


def _tools_for_context(
    context: ToolPoolContext,
    source: Iterable[Tool] | None,
) -> tuple[Tool, ...]:
    tools = list(_normalize_tools(source))
    reserved_names = {tool.name for tool in tools}

    if source is None and context.enable_skills:
        catalog = _skill_catalog_for_context(context)
        if catalog.invocable_skills():
            skill_tool = create_skill_tool(catalog)
            tools.append(skill_tool)
            reserved_names.add(skill_tool.name)

    tools = _filter_exposure_denied(tools, context.permission_engine)

    mcp_tools: list[Tool] = []
    for definition in sorted(
        context.mcp_tool_definitions,
        key=lambda item: build_mcp_tool_name(item.server_name, item.tool_name),
    ):
        tool = create_mcp_tool(definition)
        if tool.name in reserved_names:
            continue
        reserved_names.add(tool.name)
        if _is_exposure_denied(context.permission_engine, tool):
            continue
        mcp_tools.append(tool)

    combined = [*tools, *mcp_tools]
    if context.enable_deferred_tools:
        policy = DeferredToolPolicy(enabled=True)
        candidates = deferred_candidates(combined, policy)
        names = {tool.name for tool in combined}
        if candidates and TOOL_SEARCH_NAME not in names:
            combined.append(create_tool_search_tool(candidates, policy=policy))

    return tuple(combined)


def _skill_catalog_for_context(context: ToolPoolContext) -> SkillCatalog:
    supplied = context.metadata.get("skill_catalog")
    if isinstance(supplied, SkillCatalog):
        return supplied
    user_home = (
        context.skill_user_home
        or context.metadata.get("user_home")
        or context.metadata.get("home")
    )
    return discover_skill_catalog(context.workdir, user_home=user_home)


def _filter_exposure_denied(tools: list[Tool], permission_engine: Any | None) -> list[Tool]:
    if permission_engine is None:
        return tools
    return [tool for tool in tools if not _is_exposure_denied(permission_engine, tool)]


def _is_exposure_denied(permission_engine: Any | None, tool: Tool) -> bool:
    if permission_engine is None:
        return False
    checker = getattr(permission_engine, "is_exposure_denied", None)
    if not callable(checker):
        return False
    return bool(checker(tool))


def _fingerprint(tools: tuple[Tool, ...]) -> str:
    payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": _copy_mapping(tool.input_schema),
            "source": tool.source,
            "is_read_only": _fingerprint_flag(tool.is_read_only),
            "is_destructive": _fingerprint_flag(tool.is_destructive),
            "is_concurrency_safe": _fingerprint_flag(tool.is_concurrency_safe),
            "metadata": _copy_mapping(tool.metadata),
        }
        for tool in tools
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _fingerprint_flag(value: Any) -> Any:
    if callable(value):
        return f"callable:{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
    return bool(value)


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
    "ToolPool",
    "ToolPoolContext",
    "assemble_tool_pool",
    "find_tool_by_name",
    "get_all_base_tools",
    "get_tools",
]
