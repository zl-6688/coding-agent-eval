"""Reusable MCP source cache owned by a REPL connection manager."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .config import resolve_mcp_config_path
from .source import load_stdio_mcp_tool_source

if TYPE_CHECKING:
    from .types import McpToolDefinition


class McpToolSourceHandle(Protocol):
    """Minimal source ownership contract carried by an MCP lease."""

    @property
    def configs(self) -> tuple[object, ...]: ...

    @property
    def is_closed(self) -> bool: ...

    def close(self) -> None: ...


def build_mcp_cache_key(
    *,
    workdir: str | Path,
    mcp_config_path: str | Path | None,
) -> str:
    """Stable cache key: normalized workdir + resolved config path + file digest."""
    workdir_str = str(Path(workdir).resolve())
    resolved = resolve_mcp_config_path(workdir=workdir, config_path=mcp_config_path)
    if resolved is None:
        path_str = ""
        digest = ""
    else:
        path_str = str(resolved.resolve())
        digest = _config_digest(resolved)
    return f"{workdir_str}|{path_str}|{digest}"


def _config_digest(config_path: Path) -> str:
    if not config_path.is_file():
        return ""
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _summarize_cache_key(cache_key: str, *, max_len: int = 64) -> str:
    if len(cache_key) <= max_len:
        return cache_key
    return cache_key[:max_len] + "…"


def _has_failed_servers(source: StdioMcpToolSource) -> bool:
    return bool(source.metadata.get("failed_servers"))


@dataclass(frozen=True)
class McpSessionLease:
    source: McpToolSourceHandle
    definitions: tuple[McpToolDefinition, ...]
    cache_key: str
    borrowed: bool = True
    cache_hit: bool = False

    @property
    def cache_key_summary(self) -> str:
        return _summarize_cache_key(self.cache_key)


@dataclass
class McpSessionCache:
    """Own one source across manager leases when list fully succeeds."""

    _cache_key: str | None = None
    _source: StdioMcpToolSource | None = None
    _definitions: tuple[McpToolDefinition, ...] = ()

    def acquire(
        self,
        *,
        workdir: str | Path,
        enable_mcp: bool,
        mcp_config_path: str | Path | None,
    ) -> McpSessionLease | None:
        if not enable_mcp and mcp_config_path is None:
            return None

        cache_key = build_mcp_cache_key(workdir=workdir, mcp_config_path=mcp_config_path)
        if (
            self._cache_key == cache_key
            and self._source is not None
            and not self._source.is_closed
        ):
            return McpSessionLease(
                source=self._source,
                definitions=self._definitions,
                cache_key=cache_key,
                borrowed=True,
                cache_hit=True,
            )

        self.invalidate()

        source = load_stdio_mcp_tool_source(
            workdir=str(workdir),
            config_path=mcp_config_path,
        )
        if source is None:
            return None

        try:
            definitions = source.list_tool_definitions()
        except BaseException:
            if not source.is_closed:
                source.close()
            raise
        if _has_failed_servers(source):
            return McpSessionLease(
                source=source,
                definitions=definitions,
                cache_key=cache_key,
                borrowed=False,
                cache_hit=False,
            )

        self._cache_key = cache_key
        self._source = source
        self._definitions = definitions
        return McpSessionLease(
            source=source,
            definitions=definitions,
            cache_key=cache_key,
            borrowed=True,
            cache_hit=False,
        )

    def invalidate(self) -> None:
        if self._source is not None and not self._source.is_closed:
            self._source.close()
        self._cache_key = None
        self._definitions = ()
        self._source = None

    def close(self) -> None:
        self.invalidate()


__all__ = [
    "McpSessionCache",
    "McpSessionLease",
    "build_mcp_cache_key",
]
