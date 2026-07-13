"""Per-server MCP lifecycle, isolation, and recovery contracts."""

from __future__ import annotations

import json
from collections import deque

import pytest

from agent.mcp.config import McpServerConfig
from agent.mcp.types import McpToolDefinition


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ScriptedSource:
    def __init__(self, config: McpServerConfig, outcomes) -> None:
        self.configs = (config,)
        self._outcomes = deque(outcomes)
        self._closed = False
        self.list_calls = 0
        self.close_calls = 0
        self._status = {
            config.name: {
                "server_name": config.name,
                "status": "pending",
                "phase": "",
                "tool_count": 0,
            }
        }

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def server_status(self):
        return {name: dict(value) for name, value in self._status.items()}

    def list_tool_definitions(self):
        self.list_calls += 1
        outcome = self._outcomes.popleft() if self._outcomes else ("ready",)
        if outcome[0] == "failed":
            self._status[self.configs[0].name] = {
                "server_name": self.configs[0].name,
                "status": "failed",
                "phase": outcome[1],
                "tool_count": 0,
                "error_type": "RuntimeError",
                "error": outcome[2],
            }
            return ()
        definition = _definition(self.configs[0].name)
        self._status[self.configs[0].name] = {
            "server_name": self.configs[0].name,
            "status": "ready",
            "phase": "list_tools",
            "tool_count": 1,
        }
        return (definition,)

    def mark_call_failed(self, message: str = "broken pipe") -> None:
        name = self.configs[0].name
        self._status[name] = {
            "server_name": name,
            "status": "failed",
            "phase": "call_tool",
            "tool_count": 1,
            "error_type": "RuntimeError",
            "error": message,
        }

    def close(self) -> None:
        self.close_calls += 1
        self._closed = True


def _definition(server_name: str) -> McpToolDefinition:
    return McpToolDefinition(
        server_name=server_name,
        tool_name=f"{server_name}_tool",
        description="test",
        input_schema={"type": "object", "properties": {}},
        call=lambda *_args, **_kwargs: None,
    )


def _write_config(path, servers) -> None:
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def test_partial_failure_keeps_healthy_source_and_recovers_only_failed_server(tmp_path):
    from agent.mcp.connection_manager import McpConnectionManager

    clock = _Clock()
    sources = {}

    def factory(config):
        outcomes = [("ready",)] if config.name == "healthy" else [
            ("failed", "initialize", "offline"),
            ("ready",),
        ]
        source = _ScriptedSource(config, outcomes)
        sources[config.name] = source
        return source

    _write_config(
        tmp_path / ".mcp.json",
        {"healthy": {"command": "ok"}, "flaky": {"command": "sometimes"}},
    )
    manager = McpConnectionManager(source_factory=factory, clock=clock, retry_base_seconds=2)

    first = manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    immediate = manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)

    assert first is not None and immediate is not None
    assert [item.server_name for item in first.definitions] == ["healthy"]
    assert immediate.cache_hit is True
    assert sources["healthy"].list_calls == 1
    assert sources["flaky"].list_calls == 1
    assert manager.snapshot.server_status["flaky"].failure_count == 1
    assert manager.snapshot.server_status["flaky"].next_retry_at == 2.0

    clock.advance(2)
    recovered = manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)

    assert recovered is not None and recovered.cache_hit is False
    assert [item.server_name for item in recovered.definitions] == ["flaky", "healthy"]
    assert sources["healthy"].list_calls == 1
    assert sources["healthy"].close_calls == 0
    assert sources["flaky"].list_calls == 2
    assert manager.snapshot.server_status["flaky"].status == "ready"
    assert manager.snapshot.server_status["flaky"].failure_count == 0


def test_config_change_replaces_only_changed_server(tmp_path):
    from agent.mcp.connection_manager import McpConnectionManager

    created = []

    def factory(config):
        source = _ScriptedSource(config, [("ready",)])
        created.append(source)
        return source

    config_path = tmp_path / ".mcp.json"
    _write_config(config_path, {"a": {"command": "a1"}, "b": {"command": "b1"}})
    manager = McpConnectionManager(source_factory=factory)
    manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    old_a, old_b = created

    _write_config(config_path, {"a": {"command": "a1"}, "b": {"command": "b2"}})
    lease = manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)

    assert lease is not None
    assert len(created) == 3
    new_b = created[2]
    assert old_a.list_calls == 1 and old_a.close_calls == 0
    assert old_b.list_calls == 1 and old_b.close_calls == 1
    assert new_b.configs[0].name == "b"
    assert new_b.list_calls == 1


def test_call_failure_retains_definitions_until_due_recovery(tmp_path):
    from agent.mcp.connection_manager import McpConnectionManager

    clock = _Clock()
    holder = {}

    def factory(config):
        source = _ScriptedSource(config, [("ready",), ("ready",)])
        holder[config.name] = source
        return source

    _write_config(tmp_path / ".mcp.json", {"api": {"command": "api"}})
    manager = McpConnectionManager(source_factory=factory, clock=clock, retry_base_seconds=3)
    first = manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    holder["api"].mark_call_failed()

    failed = manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)

    assert first is not None and failed is not None
    assert len(failed.definitions) == 1
    assert holder["api"].list_calls == 1
    state = manager.snapshot.server_status["api"]
    assert state.status == "failed" and state.phase == "call_tool"
    assert state.failure_count == 1 and state.next_retry_at == 3.0

    clock.advance(3)
    recovered = manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    assert recovered is not None and len(recovered.definitions) == 1
    assert holder["api"].list_calls == 2
    assert manager.snapshot.server_status["api"].status == "ready"
    assert manager.snapshot.server_status["api"].failure_count == 0


def test_retry_backoff_is_exponential_and_capped(tmp_path):
    from agent.mcp.connection_manager import McpConnectionManager

    clock = _Clock()
    holder = {}

    def factory(config):
        source = _ScriptedSource(
            config,
            [
                ("failed", "initialize", "one"),
                ("failed", "initialize", "two"),
                ("failed", "initialize", "three"),
                ("failed", "initialize", "four"),
            ],
        )
        holder[config.name] = source
        return source

    _write_config(tmp_path / ".mcp.json", {"bad": {"command": "bad"}})
    manager = McpConnectionManager(
        source_factory=factory,
        clock=clock,
        retry_base_seconds=2,
        retry_max_seconds=5,
    )

    manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    assert manager.snapshot.server_status["bad"].next_retry_at == 2.0
    clock.advance(2)
    manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    assert manager.snapshot.server_status["bad"].next_retry_at == 6.0
    clock.advance(4)
    manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    assert manager.snapshot.server_status["bad"].next_retry_at == 11.0
    clock.advance(5)
    manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    assert manager.snapshot.server_status["bad"].next_retry_at == 16.0
    assert holder["bad"].list_calls == 4


def test_close_is_terminal_but_invalidate_is_reusable(tmp_path):
    from agent.mcp.connection_manager import McpConnectionManager

    created = []

    def factory(config):
        source = _ScriptedSource(config, [("ready",)])
        created.append(source)
        return source

    _write_config(tmp_path / ".mcp.json", {"a": {"command": "a"}})
    manager = McpConnectionManager(source_factory=factory)
    manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    manager.invalidate()
    manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    manager.close()

    assert [source.close_calls for source in created] == [1, 1]
    with pytest.raises(RuntimeError, match="closed"):
        manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)


def test_disabled_server_is_treated_as_absent_and_closes_existing_entry(tmp_path):
    from agent.mcp.connection_manager import McpConnectionManager

    created = []

    def factory(config):
        source = _ScriptedSource(config, [("ready",)])
        created.append(source)
        return source

    config_path = tmp_path / ".mcp.json"
    _write_config(config_path, {"a": {"command": "a"}})
    manager = McpConnectionManager(source_factory=factory)
    manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)

    _write_config(config_path, {"a": {"command": "a", "disabled": True}})
    lease = manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)

    assert lease is None
    assert len(created) == 1
    assert created[0].close_calls == 1
    assert manager.configs == ()


def test_reconcile_attempts_to_close_all_removed_servers_when_one_close_fails(tmp_path):
    from agent.mcp.connection_manager import McpConnectionManager

    created = {}

    def factory(config):
        source = _ScriptedSource(config, [("ready",)])
        created[config.name] = source
        return source

    config_path = tmp_path / ".mcp.json"
    _write_config(config_path, {"a": {"command": "a"}, "b": {"command": "b"}})
    manager = McpConnectionManager(source_factory=factory)
    manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)

    original_close = created["a"].close

    def failing_close():
        original_close()
        raise RuntimeError("close a failed")

    created["a"].close = failing_close
    config_path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="close a failed"):
        manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)

    assert created["a"].close_calls == 1
    assert created["b"].close_calls == 1
