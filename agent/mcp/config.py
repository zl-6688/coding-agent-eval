"""MCP server config loading for project-level stdio servers."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    config: Mapping[str, Any]
    source: str = "inline"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("McpServerConfig.name must be non-empty")
        if not isinstance(self.config, Mapping):
            raise TypeError("McpServerConfig.config must be a mapping")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "source", str(self.source or "inline"))
        object.__setattr__(self, "config", _freeze_value(self.config))


def parse_mcp_server_configs(
    data: Mapping[str, Any],
    *,
    source: str = "inline",
) -> tuple[McpServerConfig, ...]:
    """Parse either .mcp.json shape or a direct server map."""

    if not isinstance(data, Mapping):
        raise TypeError("MCP config data must be a mapping")
    server_map = data.get("mcpServers") if "mcpServers" in data else data
    if not isinstance(server_map, Mapping):
        raise TypeError("MCP config mcpServers must be a mapping")
    configs: list[McpServerConfig] = []
    for name in sorted(server_map):
        value = server_map[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"MCP server config for {name!r} must be a mapping")
        configs.append(McpServerConfig(str(name), value, source=source))
    return tuple(configs)


def load_mcp_config_file(path: str | Path) -> tuple[McpServerConfig, ...]:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return parse_mcp_server_configs(data, source=str(config_path))


def resolve_mcp_config_path(
    *,
    workdir: str | Path | None,
    config_path: str | Path | None = None,
) -> Path | None:
    if config_path is not None:
        return Path(config_path)
    if workdir is None:
        return None
    candidate = Path(workdir) / ".mcp.json"
    return candidate if candidate.is_file() else None


def load_mcp_config_path(
    *,
    workdir: str | Path | None,
    config_path: str | Path | None = None,
) -> tuple[McpServerConfig, ...]:
    resolved = resolve_mcp_config_path(workdir=workdir, config_path=config_path)
    if resolved is None:
        return ()
    return load_mcp_config_file(resolved)


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
    "McpServerConfig",
    "load_mcp_config_file",
    "load_mcp_config_path",
    "parse_mcp_server_configs",
    "resolve_mcp_config_path",
]
