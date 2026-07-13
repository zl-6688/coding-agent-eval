"""Local deferred tool schema selection helpers.

This module intentionally implements a local fallback, not Anthropic
``tool_reference``. ToolSearch records selected tool names in run-local state
and emits a durable marker so later request views can send the selected schemas.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import Tool, ToolContext, ToolResult
from .messages import (
    ADDITIONAL_MESSAGE_SOURCE_KEY,
    ADDITIONAL_MESSAGE_VIEW_KEY,
    DURABLE_REQUEST_VIEW,
    mark_durable_request_message,
)


TOOL_SEARCH_NAME = "ToolSearch"
DEFERRED_ADDITIONAL_MESSAGE_SOURCE = "deferred_tools"
SELECTED_DEFERRED_TOOLS_START = "<selected-deferred-tools>"
SELECTED_DEFERRED_TOOLS_END = "</selected-deferred-tools>"
_MARKER_RE = re.compile(
    re.escape(SELECTED_DEFERRED_TOOLS_START)
    + r"(?P<payload>.*?)"
    + re.escape(SELECTED_DEFERRED_TOOLS_END),
    re.DOTALL,
)
_STATE_BY_AGENT_ID: dict[str, "DeferredToolState"] = {}


@dataclass(frozen=True)
class DeferredToolPolicy:
    enabled: bool = False
    use_tool_reference: bool = False
    max_results: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        if self.use_tool_reference:
            raise ValueError("Anthropic tool_reference mode is not supported by local deferred tools")
        object.__setattr__(self, "use_tool_reference", False)
        object.__setattr__(self, "max_results", max(1, int(self.max_results)))


class DeferredToolState:
    """Run-local selected deferred tool names."""

    def __init__(self, selected_names: Iterable[str] | None = None) -> None:
        self._selected_names: set[str] = set()
        self.record_selected(selected_names or ())

    @classmethod
    def for_agent(cls, agent_id: str | None) -> "DeferredToolState":
        key = str(agent_id or "main")
        state = _STATE_BY_AGENT_ID.get(key)
        if state is None:
            state = cls()
            _STATE_BY_AGENT_ID[key] = state
        return state

    @property
    def selected_names(self) -> frozenset[str]:
        return frozenset(self._selected_names)

    def record_selected(self, names: Iterable[str]) -> tuple[str, ...]:
        added: list[str] = []
        for name in names:
            normalized = str(name).strip()
            if not normalized or normalized in self._selected_names:
                continue
            self._selected_names.add(normalized)
            added.append(normalized)
        return tuple(added)

    def restore_from_messages(self, messages: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
        return self.record_selected(extract_selected_deferred_tool_names(messages))

    def marker_message(self, *, durable: bool = True) -> dict[str, Any] | None:
        return selected_deferred_tools_marker_message(self.selected_names, durable=durable)


def reset_deferred_tool_states() -> None:
    _STATE_BY_AGENT_ID.clear()


def is_tool_search_tool(tool: Tool) -> bool:
    return tool.name == TOOL_SEARCH_NAME


def is_deferred_tool(tool: Tool, policy: DeferredToolPolicy | None = None) -> bool:
    policy = policy or DeferredToolPolicy(enabled=False)
    if not policy.enabled or is_tool_search_tool(tool):
        return False
    mcp = _mcp_metadata(tool)
    if tool.source == "mcp" and not bool(mcp.get("always_load", False)):
        return True
    return False


def deferred_candidates(
    tools: Iterable[Tool],
    policy: DeferredToolPolicy | None = None,
) -> tuple[Tool, ...]:
    policy = policy or DeferredToolPolicy(enabled=False)
    return tuple(tool for tool in tools if is_deferred_tool(tool, policy))


def create_tool_search_tool(
    candidates: Iterable[Tool],
    *,
    policy: DeferredToolPolicy | None = None,
    state: DeferredToolState | None = None,
) -> Tool:
    policy = policy or DeferredToolPolicy(enabled=True)
    candidate_tuple = tuple(candidates)
    by_name = {tool.name: tool for tool in candidate_tuple}

    def _call(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        active_state = state or DeferredToolState.for_agent(context.agent_id)
        query = _query_from_input(tool_input)
        matches = _match_candidates(query, candidate_tuple, policy.max_results)
        if not matches:
            return ToolResult(
                "No matching deferred tools. Use ToolSearch with "
                "`select:<tool_name>` for an exact local schema selection."
            )
        selected_names = tuple(tool.name for tool in matches if tool.name in by_name)
        active_state.record_selected(selected_names)
        marker = selected_deferred_tools_marker_text(selected_names)
        additional = selected_deferred_tools_marker_message(selected_names, durable=True)
        lines = [
            "Selected deferred tools for the next request:",
            *[f"- {name}" for name in selected_names],
            marker,
            "This is local schema selection; no Anthropic tool_reference blocks were emitted.",
        ]
        return ToolResult(
            "\n".join(lines),
            additional_messages=(additional,) if additional is not None else (),
        )

    return Tool(
        name=TOOL_SEARCH_NAME,
        description=(
            "Search or select locally deferred tool schemas. Use "
            "`select:<tool_name>` to make a deferred schema available on the next turn. "
            "This fallback does not support Anthropic tool_reference."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text, or select:<tool_name> for exact selection.",
                }
            },
            "required": ["query"],
        },
        call=_call,
        source="local_deferred",
        is_read_only=True,
        is_concurrency_safe=False,
        metadata={"deferred_tool_search": True},
    )


def build_deferred_index_context_message(candidates: Iterable[Tool]) -> dict[str, Any] | None:
    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        return None
    lines = [
        "<system-reminder>",
        "Deferred tool schemas are available through local ToolSearch.",
        "This fallback does not use or emit Anthropic tool_reference blocks.",
        "Use ToolSearch with `select:<tool_name>` to load a schema for the next turn.",
        "",
        "<available-deferred-tools>",
    ]
    for tool in candidate_tuple:
        hint = _search_hint(tool)
        suffix = f" | hint: {_compact_text(hint, 160)}" if hint else ""
        lines.append(f"- {tool.name}{suffix}")
    lines.extend(["</available-deferred-tools>", "</system-reminder>"])
    return {"role": "user", "content": "\n".join(lines)}


def selected_deferred_tools_marker_text(names: Iterable[str]) -> str:
    selected = sorted({str(name).strip() for name in names if str(name).strip()})
    payload = json.dumps({"selected": selected}, sort_keys=True, separators=(",", ":"))
    return f"{SELECTED_DEFERRED_TOOLS_START}{payload}{SELECTED_DEFERRED_TOOLS_END}"


def selected_deferred_tools_marker_message(
    names: Iterable[str],
    *,
    durable: bool = False,
) -> dict[str, Any] | None:
    selected = sorted({str(name).strip() for name in names if str(name).strip()})
    if not selected:
        return None
    message = {
        "role": "user",
        "content": [{"type": "text", "text": selected_deferred_tools_marker_text(selected)}],
    }
    if durable:
        return mark_durable_request_message(message, source=DEFERRED_ADDITIONAL_MESSAGE_SOURCE)
    return message


def extract_selected_deferred_tool_names(
    messages: Iterable[Mapping[str, Any]],
    *,
    trusted_only: bool = True,
) -> tuple[str, ...]:
    found: set[str] = set()
    for text in _message_texts(messages, trusted_only=trusted_only):
        for match in _MARKER_RE.finditer(text):
            payload = match.group("payload").strip()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            names = data.get("selected") if isinstance(data, Mapping) else None
            if not isinstance(names, list):
                continue
            for name in names:
                normalized = str(name).strip()
                if normalized:
                    found.add(normalized)
    return tuple(sorted(found))


def _query_from_input(tool_input: Mapping[str, Any]) -> str:
    for key in ("query", "tool", "tool_name", "name"):
        value = tool_input.get(key)
        if value is not None:
            return str(value)
    return ""


def _match_candidates(query: str, candidates: tuple[Tool, ...], max_results: int) -> tuple[Tool, ...]:
    query = str(query or "").strip()
    if not query:
        return ()
    selected = _selected_names_from_query(query)
    if selected:
        by_name = {tool.name: tool for tool in candidates}
        return tuple(by_name[name] for name in selected if name in by_name)[:max_results]
    tokens = _tokens(query)
    if not tokens:
        return ()
    scored: list[tuple[int, str, Tool]] = []
    for tool in candidates:
        score = _score_tool(tool, tokens)
        if score > 0:
            scored.append((score, tool.name, tool))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(item[2] for item in scored[:max_results])


def _selected_names_from_query(query: str) -> tuple[str, ...]:
    stripped = query.strip()
    lower = stripped.lower()
    prefix = ""
    if lower.startswith("select:"):
        prefix = stripped[len("select:") :]
    elif lower.startswith("select "):
        prefix = stripped[len("select ") :]
    if not prefix:
        return ()
    return tuple(part for part in re.split(r"[\s,]+", prefix.strip()) if part)


def _score_tool(tool: Tool, tokens: tuple[str, ...]) -> int:
    name = tool.name.lower()
    hint = _search_hint(tool).lower()
    description = str(tool.description or "").lower()
    score = 0
    for token in tokens:
        if token == name:
            score += 20
        elif token in name:
            score += 8
        if hint and token in hint:
            score += 5
        if description and token in description:
            score += 1
    return score


def _tokens(query: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in re.findall(r"[A-Za-z0-9_]+", query))


def _mcp_metadata(tool: Tool) -> Mapping[str, Any]:
    metadata = tool.metadata
    if isinstance(metadata, Mapping):
        mcp = metadata.get("mcp")
        if isinstance(mcp, Mapping):
            return mcp
    return {}


def _search_hint(tool: Tool) -> str:
    return str(_mcp_metadata(tool).get("search_hint") or "")


def _compact_text(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _message_texts(
    messages: Iterable[Mapping[str, Any]],
    *,
    trusted_only: bool,
) -> tuple[str, ...]:
    texts: list[str] = []
    for message in messages:
        if trusted_only and not _is_trusted_marker_message(message):
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, Mapping):
                    if block.get("type") == "text":
                        texts.append(str(block.get("text", "")))
                    elif "content" in block:
                        texts.append(str(block.get("content", "")))
                else:
                    text = getattr(block, "text", None)
                    if text is not None:
                        texts.append(str(text))
    return tuple(texts)


def _is_trusted_marker_message(message: Mapping[str, Any]) -> bool:
    metadata = message.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    return (
        metadata.get(ADDITIONAL_MESSAGE_VIEW_KEY) == DURABLE_REQUEST_VIEW
        and metadata.get(ADDITIONAL_MESSAGE_SOURCE_KEY) == DEFERRED_ADDITIONAL_MESSAGE_SOURCE
    )


__all__ = [
    "DEFERRED_ADDITIONAL_MESSAGE_SOURCE",
    "DeferredToolPolicy",
    "DeferredToolState",
    "TOOL_SEARCH_NAME",
    "build_deferred_index_context_message",
    "create_tool_search_tool",
    "deferred_candidates",
    "extract_selected_deferred_tool_names",
    "is_deferred_tool",
    "is_tool_search_tool",
    "reset_deferred_tool_states",
    "selected_deferred_tools_marker_message",
    "selected_deferred_tools_marker_text",
]
