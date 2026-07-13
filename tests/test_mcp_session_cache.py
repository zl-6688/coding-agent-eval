"""Tests for MCP source cache leases and REPL-scoped ownership (R-1～R-5)."""

from __future__ import annotations

import json

import pytest

from agent.mcp.config import McpServerConfig
from agent.mcp.session_cache import McpSessionCache, build_mcp_cache_key
from agent.mcp.types import McpToolDefinition
from tests.conftest import end_turn_resp


class _FakeSource:
    configs = ()

    def __init__(self, *, definitions=(), failed_servers=(), closed=False):
        self._closed = closed
        self._definitions = tuple(definitions)
        self._failed_servers = tuple(failed_servers)
        self.list_calls = 0
        self.close_calls = 0

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def metadata(self):
        return {"failed_servers": self._failed_servers}

    def list_tool_definitions(self):
        self.list_calls += 1
        return self._definitions

    def close(self):
        self.close_calls += 1
        self._closed = True


def _definition(name: str = "echo") -> McpToolDefinition:
    return McpToolDefinition(
        server_name="fake",
        tool_name=name,
        description="test",
        input_schema={"type": "object", "properties": {}},
        call=lambda *_a, **_k: None,
    )


def _write_mcp_config(path, *, servers=None):
    payload = {"mcpServers": servers or {"fake": {"command": "noop"}}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_mcp_cache_key_includes_digest(tmp_path):
    config_path = tmp_path / ".mcp.json"
    _write_mcp_config(config_path)
    key1 = build_mcp_cache_key(workdir=tmp_path, mcp_config_path=None)
    config_path.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")
    key2 = build_mcp_cache_key(workdir=tmp_path, mcp_config_path=None)
    assert key1 != key2


def test_acquire_cache_hit_lists_once(tmp_path, monkeypatch):
    source = _FakeSource(definitions=(_definition(),))
    load_calls = []

    def fake_load(*, workdir, config_path=None, **kwargs):
        load_calls.append((workdir, config_path))
        return source

    monkeypatch.setattr("agent.mcp.session_cache.load_stdio_mcp_tool_source", fake_load)
    _write_mcp_config(tmp_path / ".mcp.json")

    cache = McpSessionCache()
    lease1 = cache.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    lease2 = cache.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)

    assert lease1 is not None and lease2 is not None
    assert lease1.cache_hit is False
    assert lease2.cache_hit is True
    assert lease1.borrowed is True and lease2.borrowed is True
    assert source.list_calls == 1
    assert len(load_calls) == 1


def test_acquire_failed_server_not_cached(tmp_path, monkeypatch):
    source = _FakeSource(definitions=(_definition(),), failed_servers=("bad",))
    monkeypatch.setattr(
        "agent.mcp.session_cache.load_stdio_mcp_tool_source",
        lambda **kwargs: source,
    )
    _write_mcp_config(tmp_path / ".mcp.json")

    cache = McpSessionCache()
    lease1 = cache.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    lease2 = cache.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)

    assert lease1 is not None
    assert lease1.borrowed is False
    assert lease2 is not None
    assert source.list_calls == 2


def test_acquire_closed_source_rebuilds(tmp_path, monkeypatch):
    sources = [
        _FakeSource(definitions=(_definition(),)),
        _FakeSource(definitions=(_definition(),)),
    ]
    load_index = {"value": 0}

    def fake_load(**kwargs):
        source = sources[load_index["value"]]
        load_index["value"] += 1
        return source

    monkeypatch.setattr("agent.mcp.session_cache.load_stdio_mcp_tool_source", fake_load)
    _write_mcp_config(tmp_path / ".mcp.json")

    cache = McpSessionCache()
    cache.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    sources[0]._closed = True
    cache.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    assert sources[0].list_calls == 1
    assert sources[1].list_calls == 1


def test_run_task_borrowed_does_not_close_source(tmp_path, monkeypatch):
    from agent import loop
    from agent.mcp.session_cache import McpSessionLease

    source = _FakeSource(definitions=(_definition(),))
    lease = McpSessionLease(
        source=source,
        definitions=source._definitions,
        cache_key="k",
        borrowed=True,
        cache_hit=True,
    )

    def fail_load(**kwargs):
        raise AssertionError("load_stdio_mcp_tool_source should not run for borrowed lease")

    monkeypatch.setattr(loop, "load_stdio_mcp_tool_source", fail_load)
    monkeypatch.setattr(loop.llm, "chat", lambda *a, **k: end_turn_resp("ok"))

    class _Exec:
        kind = "local"
        cwd = str(tmp_path)

    monkeypatch.setattr(loop.tools, "get_executor", lambda: _Exec())

    text = loop.run_task("q", max_turns=1, trace=False, mcp_session=lease, enable_mcp=True)
    assert text == "ok"
    assert source.close_calls == 0


def test_run_task_owned_lease_closes_source(tmp_path, monkeypatch):
    from agent import loop
    from agent.mcp.session_cache import McpSessionLease

    source = _FakeSource(definitions=(_definition(),))
    lease = McpSessionLease(
        source=source,
        definitions=source._definitions,
        cache_key="k",
        borrowed=False,
        cache_hit=False,
    )

    monkeypatch.setattr(loop.llm, "chat", lambda *a, **k: end_turn_resp("ok"))

    class _Exec:
        kind = "local"
        cwd = str(tmp_path)

    monkeypatch.setattr(loop.tools, "get_executor", lambda: _Exec())

    loop.run_task("q", max_turns=1, trace=False, mcp_session=lease, enable_mcp=True)
    assert source.close_calls == 1


def test_session_run_passes_mcp_session(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    monkeypatch.setenv("ACE_MCP_CONFIG", str(tmp_path / ".mcp.json"))
    from agent.runtime import session as sess_mod
    from agent.mcp.session_cache import McpSessionLease

    captured = {}
    fake_source = _FakeSource(definitions=(_definition(),))
    fake_lease = McpSessionLease(
        source=fake_source,
        definitions=fake_source._definitions,
        cache_key="k",
        borrowed=True,
        cache_hit=False,
    )

    class _FakeManager:
        def acquire(self, **kwargs):
            captured["acquire_kwargs"] = kwargs
            return fake_lease

        def invalidate(self):
            captured["manager_invalidated"] = True

    def fake_run_task(task, **kw):
        captured["mcp_session"] = kw.get("mcp_session")
        return ("done", kw.get("initial_messages", []))

    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)
    _write_mcp_config(tmp_path / ".mcp.json")

    s = sess_mod.Session.create(tmp_path)
    s.run("hello", mcp_connection_manager=_FakeManager())

    assert captured["mcp_session"] is fake_lease
    assert captured["acquire_kwargs"]["enable_mcp"] is True


def test_connection_manager_close_releases_owned_source(tmp_path, monkeypatch):
    from agent.mcp.connection_manager import McpConnectionManager

    source = _FakeSource(definitions=(_definition(),))
    _write_mcp_config(tmp_path / ".mcp.json")

    manager = McpConnectionManager(source_factory=lambda _config: source)
    manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    manager.close()
    assert source.close_calls == 1


def test_connection_manager_digest_change_closes_old_source_and_relists(tmp_path, monkeypatch):
    from agent.mcp.connection_manager import McpConnectionManager

    sources = []

    def fake_load(**kwargs):
        source = _FakeSource(definitions=(_definition(),))
        sources.append(source)
        return source

    config_path = tmp_path / ".mcp.json"
    _write_mcp_config(config_path)
    manager = McpConnectionManager(source_factory=lambda config: fake_load())

    first = manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    config_path.write_text(
        json.dumps({"mcpServers": {"changed": {"command": "noop"}}}),
        encoding="utf-8",
    )
    second = manager.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)

    assert first is not None and second is not None
    assert first.cache_hit is False and second.cache_hit is False
    assert len(sources) == 2
    assert sources[0].list_calls == 1 and sources[0].close_calls == 1
    assert sources[1].list_calls == 1 and sources[1].close_calls == 0
    manager.close()
    assert sources[1].close_calls == 1


def test_two_sessions_borrow_one_repl_manager_and_list_once(tmp_path, monkeypatch):
    """Transcript Session replacement must not replace the REPL MCP owner."""
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    from agent.mcp.connection_manager import McpConnectionManager
    from agent.runtime import session as sess_mod

    source = _FakeSource(definitions=(_definition(),))
    _write_mcp_config(tmp_path / ".mcp.json")
    config_path = str(tmp_path / ".mcp.json")
    leases: list[tuple[bool, bool]] = []

    def fake_run_task(task, **kw):
        lease = kw.get("mcp_session")
        if lease is not None:
            leases.append((lease.cache_hit, lease.borrowed))
        messages = kw.get("initial_messages", [])
        return ("done", messages)

    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)

    manager = McpConnectionManager(source_factory=lambda _config: source)
    first = sess_mod.Session.create(tmp_path)
    second = sess_mod.Session.create(tmp_path)
    first.run(
        "hello",
        enable_mcp=True,
        mcp_config_path=config_path,
        mcp_connection_manager=manager,
    )
    second.run(
        "again",
        enable_mcp=True,
        mcp_config_path=config_path,
        mcp_connection_manager=manager,
    )

    assert source.list_calls == 1
    assert leases == [(False, True), (True, True)]
    manager.close()


def test_direct_session_run_without_manager_keeps_per_run_ownership(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    from agent.runtime import session as sess_mod

    captured = {}

    def fake_run_task(task, **kw):
        captured.update(kw)
        return ("done", kw.get("initial_messages", []))

    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)
    _write_mcp_config(tmp_path / ".mcp.json")

    session = sess_mod.Session.create(tmp_path)
    session.run("hello", enable_mcp=True)

    assert captured["mcp_session"] is None
    assert not hasattr(session, "_mcp_cache")


def test_acquire_list_exception_closes_source(tmp_path, monkeypatch):
    source = _FakeSource(definitions=(_definition(),))

    def boom():
        source.list_calls += 1
        raise RuntimeError("list failed")

    source.list_tool_definitions = boom
    monkeypatch.setattr(
        "agent.mcp.session_cache.load_stdio_mcp_tool_source",
        lambda **kwargs: source,
    )
    _write_mcp_config(tmp_path / ".mcp.json")

    cache = McpSessionCache()
    with pytest.raises(RuntimeError, match="list failed"):
        cache.acquire(workdir=tmp_path, enable_mcp=True, mcp_config_path=None)
    assert source.close_calls == 1


def test_session_run_without_mcp_invalidates_repl_manager(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    from agent.runtime import session as sess_mod

    calls = []

    class _FakeManager:
        def acquire(self, **kwargs):
            raise AssertionError("disabled MCP must not acquire")

        def invalidate(self):
            calls.append("invalidate")

    def fake_run_task(task, **kw):
        return ("done", kw.get("initial_messages", []))

    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)

    s = sess_mod.Session.create(tmp_path)
    s.run("hello", enable_mcp=False, mcp_connection_manager=_FakeManager())
    assert calls == ["invalidate"]


@pytest.mark.parametrize("switch_kind", ["clear", "resume"])
def test_repl_session_switch_keeps_source_until_repl_exit(
    switch_kind,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    monkeypatch.delenv("ACE_ENABLE_MCP", raising=False)
    monkeypatch.delenv("ACE_MCP_CONFIG", raising=False)
    from agent.cli.repl import run_repl
    from agent.runtime import session as sess_mod

    _write_mcp_config(tmp_path / ".mcp.json")
    sources = []

    def fake_load(**kwargs):
        source = _FakeSource(definitions=(_definition(),))
        sources.append(source)
        return source

    def fake_run_task(task, **kw):
        return ("done", kw.get("initial_messages", []))

    monkeypatch.setattr(
        "agent.mcp.connection_manager.create_stdio_mcp_tool_source",
        lambda configs: fake_load(),
    )
    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)

    if switch_kind == "resume":
        saved = sess_mod.Session.create(tmp_path, with_memory=False)
        saved.messages = [{"role": "user", "content": "saved"}]
        saved.store.save(saved.id, saved.messages)
        switch_command = f"/resume {saved.id}"
    else:
        switch_command = "/clear"

    script = iter(["first", switch_command, "second", "/exit"])
    run_repl(
        tmp_path,
        read_input=lambda: next(script, None),
        out=lambda _line: None,
        register_sink=False,
    )

    assert len(sources) == 1
    assert sources[0].list_calls == 1
    assert sources[0].close_calls == 1


def test_stdio_source_is_closed_property():
    from agent.mcp.source import StdioMcpToolSource

    source = StdioMcpToolSource(())
    assert source.is_closed is False
    source.close()
    assert source.is_closed is True
