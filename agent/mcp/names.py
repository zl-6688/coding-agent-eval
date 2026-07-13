"""Stable names for generated MCP tools and permission rules."""

from __future__ import annotations

import re

MCP_TOOL_PREFIX = "mcp"
MCP_SEPARATOR = "__"
_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9_]+")
_UNDERSCORE_RE = re.compile(r"_+")


def normalize_mcp_name(value: str) -> str:
    """Return a model-tool-safe MCP name segment."""

    text = str(value).strip()
    if not text:
        raise ValueError("MCP name segment must be non-empty")
    normalized = _SAFE_PART_RE.sub("_", text)
    normalized = _UNDERSCORE_RE.sub("_", normalized).strip("_")
    if not normalized:
        raise ValueError("MCP name segment must contain a safe character")
    return normalized


def build_mcp_server_name(server_name: str) -> str:
    return f"{MCP_TOOL_PREFIX}{MCP_SEPARATOR}{normalize_mcp_name(server_name)}"


def build_mcp_wildcard_name(server_name: str) -> str:
    return f"{build_mcp_server_name(server_name)}{MCP_SEPARATOR}*"


def build_mcp_tool_name(server_name: str, tool_name: str) -> str:
    return (
        f"{build_mcp_server_name(server_name)}"
        f"{MCP_SEPARATOR}{normalize_mcp_name(tool_name)}"
    )


__all__ = [
    "MCP_SEPARATOR",
    "MCP_TOOL_PREFIX",
    "build_mcp_server_name",
    "build_mcp_tool_name",
    "build_mcp_wildcard_name",
    "normalize_mcp_name",
]
