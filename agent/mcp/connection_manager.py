"""REPL-scoped owner for independent per-server MCP stdio sources."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .config import McpServerConfig, load_mcp_config_path
from .session_cache import McpSessionLease, build_mcp_cache_key
from .source import create_stdio_mcp_tool_source
from .types import McpToolDefinition


@dataclass(frozen=True)
class McpServerState:
    """Immutable public state for one manager-owned MCP server."""

    server_name: str
    status: str
    phase: str
    tool_count: int
    failure_count: int
    next_retry_at: float | None
    error_type: str = ""
    error: str = ""


@dataclass(frozen=True)
class McpConnectionSnapshot:
    """Point-in-time aggregate definitions and server lifecycle state."""

    generation: int
    definitions: tuple[McpToolDefinition, ...]
    server_status: Mapping[str, McpServerState]


@dataclass
class _ManagedServer:
    config: McpServerConfig
    digest: str
    source: Any
    definitions: tuple[McpToolDefinition, ...] = ()
    status: str = "pending"
    phase: str = ""
    failure_count: int = 0
    next_retry_at: float | None = None
    error_type: str = ""
    error: str = ""
    observed_revision: int | None = None
    observed_failure_marker: tuple[str, str, str] | None = None

    def public_state(self) -> McpServerState:
        return McpServerState(
            server_name=self.config.name,
            status=self.status,
            phase=self.phase,
            tool_count=len(self.definitions),
            failure_count=self.failure_count,
            next_retry_at=self.next_retry_at,
            error_type=self.error_type,
            error=self.error,
        )


SourceFactory = Callable[[McpServerConfig], Any]


def _default_source_factory(config: McpServerConfig) -> Any:
    return create_stdio_mcp_tool_source((config,))


@dataclass
class McpConnectionManager:
    """Own independent server sources across sequential REPL Session runs.

    Manager mutations are serialized, but the ownership contract remains one
    non-concurrent REPL: callers must not reconcile configuration while an old
    lease is executing tools on another thread.
    """

    source_factory: SourceFactory = _default_source_factory
    clock: Callable[[], float] = time.monotonic
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 30.0
    _entries: dict[str, _ManagedServer] = field(default_factory=dict, init=False)
    _order: tuple[str, ...] = field(default_factory=tuple, init=False)
    _generation: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.retry_base_seconds = max(0.0, float(self.retry_base_seconds))
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            float(self.retry_max_seconds),
        )

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def configs(self) -> tuple[McpServerConfig, ...]:
        with self._lock:
            return tuple(self._entries[name].config for name in self._order)

    @property
    def server_status(self) -> Mapping[str, Mapping[str, Any]]:
        with self._lock:
            return MappingProxyType(
                {
                    name: MappingProxyType(
                        {
                            "server_name": state.server_name,
                            "status": state.status,
                            "phase": state.phase,
                            "tool_count": state.tool_count,
                            "failure_count": state.failure_count,
                            "next_retry_at": state.next_retry_at,
                            "error_type": state.error_type,
                            "error": state.error,
                        }
                    )
                    for name, state in self.snapshot.server_status.items()
                }
            )

    @property
    def metadata(self) -> Mapping[str, Any]:
        statuses = self.server_status
        return MappingProxyType(
            {
                "server_status": statuses,
                "failed_servers": tuple(
                    name
                    for name, status in statuses.items()
                    if status.get("status") == "failed"
                ),
                "tool_count": sum(
                    int(status.get("tool_count") or 0) for status in statuses.values()
                ),
                "generation": self._generation,
            }
        )

    @property
    def snapshot(self) -> McpConnectionSnapshot:
        with self._lock:
            definitions = tuple(
                definition
                for name in self._order
                for definition in self._entries[name].definitions
            )
            statuses = MappingProxyType(
                {
                    name: self._entries[name].public_state()
                    for name in self._order
                }
            )
            return McpConnectionSnapshot(
                generation=self._generation,
                definitions=definitions,
                server_status=statuses,
            )

    def acquire(
        self,
        *,
        workdir: str | Path,
        enable_mcp: bool,
        mcp_config_path: str | Path | None,
    ) -> McpSessionLease | None:
        with self._lock:
            if self._closed:
                raise RuntimeError("McpConnectionManager is closed")
            if not enable_mcp and mcp_config_path is None:
                return None

            loaded = load_mcp_config_path(
                workdir=workdir,
                config_path=mcp_config_path,
            )
            # Disabled servers are absent from the active topology, matching
            # StdioMcpToolSource's filtering contract without creating a zero-config source.
            configs = tuple(
                config
                for config in loaded
                if not bool(config.config.get("disabled", False))
            )
            topology_changed = self._reconcile(configs)
            if not configs:
                return None

            now = float(self.clock())
            state_changed = False
            for name in self._order:
                state_changed = self._sync_source_state(
                    self._entries[name],
                    now=now,
                ) or state_changed

            attempted = False
            for name in self._order:
                entry = self._entries[name]
                due = (
                    entry.status == "failed"
                    and entry.next_retry_at is not None
                    and now >= entry.next_retry_at
                )
                if entry.status == "pending" or due:
                    attempted = True
                    state_changed = self._refresh_entry(entry, now=now) or state_changed

            if topology_changed or state_changed:
                self._generation += 1
            snapshot = self.snapshot
            return McpSessionLease(
                source=self,
                definitions=snapshot.definitions,
                cache_key=build_mcp_cache_key(
                    workdir=workdir,
                    mcp_config_path=mcp_config_path,
                ),
                borrowed=True,
                cache_hit=not topology_changed and not attempted,
            )

    def invalidate(self) -> None:
        with self._lock:
            changed = bool(self._entries)
            self._close_entries(tuple(self._entries.values()))
            self._entries.clear()
            self._order = ()
            if changed:
                self._generation += 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self.invalidate()
            finally:
                self._closed = True

    def _reconcile(self, configs: tuple[McpServerConfig, ...]) -> bool:
        desired = {config.name: (config, _server_digest(config)) for config in configs}
        changed = False

        remove_names: list[str] = []
        for name, current in self._entries.items():
            desired_item = desired.get(name)
            if desired_item is not None and desired_item[1] == current.digest:
                continue
            remove_names.append(name)

        if remove_names:
            close_error: BaseException | None = None
            try:
                # One close failure must not prevent cleanup of the remaining
                # removed/changed servers.
                self._close_entries(
                    tuple(self._entries[name] for name in remove_names)
                )
            except BaseException as exc:
                close_error = exc
            finally:
                for name in remove_names:
                    self._entries.pop(name, None)
                self._order = tuple(
                    name for name in self._order if name in self._entries
                )
                changed = True
            if close_error is not None:
                self._generation += 1
                raise close_error

        for config in configs:
            if config.name in self._entries:
                continue
            self._entries[config.name] = _ManagedServer(
                config=config,
                digest=desired[config.name][1],
                source=self.source_factory(config),
            )
            changed = True

        new_order = tuple(config.name for config in configs)
        if new_order != self._order:
            self._order = new_order
            changed = True
        return changed

    def _refresh_entry(self, entry: _ManagedServer, *, now: float) -> bool:
        before = entry.public_state()
        try:
            definitions = tuple(entry.source.list_tool_definitions())
        except Exception as exc:
            self._record_failure(
                entry,
                phase="list_tools",
                error_type=type(exc).__name__,
                error=str(exc),
                now=now,
            )
            return entry.public_state() != before

        source_state = _source_state(entry)
        if source_state.get("status") == "failed":
            self._consume_failure_state(
                entry,
                source_state,
                now=now,
                force=True,
            )
        else:
            entry.definitions = definitions
            self._record_ready(entry, source_state)
        return entry.public_state() != before

    def _sync_source_state(self, entry: _ManagedServer, *, now: float) -> bool:
        before = entry.public_state()
        source_state = _source_state(entry)
        status = str(source_state.get("status") or "")
        if status == "failed":
            self._consume_failure_state(entry, source_state, now=now, force=False)
        elif status == "ready" and entry.status == "failed":
            # A user-initiated tool call may reconnect before the manager's relist
            # deadline. Backoff governs proactive relist only, not explicit calls.
            self._record_ready(entry, source_state)
        return entry.public_state() != before

    def _consume_failure_state(
        self,
        entry: _ManagedServer,
        source_state: Mapping[str, Any],
        *,
        now: float,
        force: bool,
    ) -> None:
        marker = (
            str(source_state.get("phase") or ""),
            str(source_state.get("error_type") or ""),
            str(source_state.get("error") or ""),
        )
        raw_revision = source_state.get("revision")
        revision = int(raw_revision) if isinstance(raw_revision, int) else None
        is_new = force or entry.status != "failed"
        if revision is not None:
            is_new = is_new or revision != entry.observed_revision
        else:
            is_new = is_new or marker != entry.observed_failure_marker
        if not is_new:
            return
        entry.observed_revision = revision
        entry.observed_failure_marker = marker
        self._record_failure(
            entry,
            phase=marker[0],
            error_type=marker[1],
            error=marker[2],
            now=now,
        )

    def _record_failure(
        self,
        entry: _ManagedServer,
        *,
        phase: str,
        error_type: str,
        error: str,
        now: float,
    ) -> None:
        entry.failure_count += 1
        exponent = max(0, entry.failure_count - 1)
        delay = min(
            self.retry_max_seconds,
            self.retry_base_seconds * (2**exponent),
        )
        entry.status = "failed"
        entry.phase = phase
        entry.next_retry_at = now + delay
        entry.error_type = error_type
        entry.error = error

    @staticmethod
    def _record_ready(entry: _ManagedServer, source_state: Mapping[str, Any]) -> None:
        entry.status = "ready"
        entry.phase = str(source_state.get("phase") or "list_tools")
        entry.failure_count = 0
        entry.next_retry_at = None
        entry.error_type = ""
        entry.error = ""
        raw_revision = source_state.get("revision")
        entry.observed_revision = int(raw_revision) if isinstance(raw_revision, int) else None
        entry.observed_failure_marker = None

    @staticmethod
    def _close_entries(entries: tuple[_ManagedServer, ...]) -> None:
        first_error: BaseException | None = None
        for entry in reversed(entries):
            if bool(getattr(entry.source, "is_closed", False)):
                continue
            try:
                entry.source.close()
            except BaseException as exc:  # close every server before surfacing one failure
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def _source_state(entry: _ManagedServer) -> Mapping[str, Any]:
    statuses = getattr(entry.source, "server_status", {})
    if isinstance(statuses, Mapping):
        state = statuses.get(entry.config.name)
        if isinstance(state, Mapping):
            return state
    return {}


def _server_digest(config: McpServerConfig) -> str:
    payload = {
        "name": config.name,
        "source": config.source,
        "config": _thaw(config.config),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_thaw(item) for item in value), key=repr)
    return value


__all__ = [
    "McpConnectionManager",
    "McpConnectionSnapshot",
    "McpServerState",
]
