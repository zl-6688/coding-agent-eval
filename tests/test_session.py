"""test_session.py — runtime Session ↔ 记忆接线（含 3b-2 Auto Memory 接线）。

ACE_HOME 指向 tmp_path，不污染真实 ~/.ace。
"""
from agent.memory.auto_memory import AutoMemory
from agent.memory.session_memory import SessionMemory


def test_session_create_wires_auto_memory(tmp_path, monkeypatch):
    """with_memory=True（默认）→ session.auto_memory 是挂 project.memory_dir 的 AutoMemory。"""
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    from agent.runtime.session import Session

    s = Session.create(tmp_path)
    assert isinstance(s.auto_memory, AutoMemory)
    # 跨会话记忆挂 **project 级** memory_dir（同 project 不同 session 共享 → 跨会话锚点）
    assert s.auto_memory.memory_dir == s.project.memory_dir
    # 与 per-session SessionMemory 不同目录、零碰撞
    assert isinstance(s.memory, SessionMemory)
    assert s.project.memory_dir != s.project.sessions_dir


def test_session_create_no_memory_gives_none(tmp_path, monkeypatch):
    """with_memory=False → auto_memory 与 memory 都为 None（不触发任何记忆 fork/召回）。"""
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    from agent.runtime.session import Session

    s = Session.create(tmp_path, with_memory=False)
    assert s.auto_memory is None
    assert s.memory is None


def test_session_run_passes_auto_memory_to_run_task(tmp_path, monkeypatch):
    """Session.run 把 self.auto_memory 透传给 run_task（opt-in 接缝真接上）。"""
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    from agent.runtime import session as sess_mod

    captured = {}

    def fake_run_task(task, **kw):
        captured.update(kw)
        # run_task(return_messages=True) 返 (text, messages)
        return ("done", kw.get("initial_messages", []))

    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)

    s = sess_mod.Session.create(tmp_path)
    s.run("hello")

    assert captured.get("auto_memory") is s.auto_memory
    assert captured.get("auto_memory") is not None
    # session_memory 同样透传（回归保护：别因加 auto_memory 漏了 SM）
    assert captured.get("session_memory") is s.memory


def test_session_run_no_memory_passes_none(tmp_path, monkeypatch):
    """with_memory=False 的会话 run → run_task 收到 auto_memory=None。"""
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    from agent.runtime import session as sess_mod

    captured = {}

    def fake_run_task(task, **kw):
        captured.update(kw)
        return ("done", kw.get("initial_messages", []))

    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)

    s = sess_mod.Session.create(tmp_path, with_memory=False)
    s.run("hello")

    assert captured.get("auto_memory") is None


def test_session_run_applies_mcp_env_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    monkeypatch.setenv("ACE_MCP_CONFIG", str(tmp_path / ".mcp.json"))
    from agent.runtime import session as sess_mod

    captured = {}

    def fake_run_task(task, **kw):
        captured.update(kw)
        return ("done", kw.get("initial_messages", []))

    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)

    s = sess_mod.Session.create(tmp_path)
    s.run("hello")

    assert captured["enable_mcp"] is True
    assert captured["mcp_config_path"] == str(tmp_path / ".mcp.json")
    assert captured["enable_deferred_tools"] is True


def test_session_run_auto_enables_workdir_mcp_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    monkeypatch.delenv("ACE_ENABLE_MCP", raising=False)
    monkeypatch.delenv("ACE_MCP_CONFIG", raising=False)
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
    from agent.runtime import session as sess_mod

    captured = {}

    def fake_run_task(task, **kw):
        captured.update(kw)
        return ("done", kw.get("initial_messages", []))

    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)

    session = sess_mod.Session.create(tmp_path)
    session.run("hello")

    assert captured["enable_mcp"] is True
    assert captured["mcp_config_path"] is None
    assert captured["enable_deferred_tools"] is True


def test_session_run_disable_mcp_suppresses_workdir_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
    from agent.runtime import session as sess_mod

    captured = {}

    def fake_run_task(task, **kw):
        captured.update(kw)
        return ("done", kw.get("initial_messages", []))

    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)

    session = sess_mod.Session.create(tmp_path)
    session.run("hello", disable_mcp=True)

    assert captured["enable_mcp"] is False
    assert captured["mcp_config_path"] is None
    assert captured["enable_deferred_tools"] is False


def test_session_run_explicit_no_deferred_overrides_mcp_default(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    monkeypatch.setenv("ACE_MCP_CONFIG", str(tmp_path / ".mcp.json"))
    from agent.runtime import session as sess_mod

    captured = {}

    def fake_run_task(task, **kw):
        captured.update(kw)
        return ("done", kw.get("initial_messages", []))

    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)

    s = sess_mod.Session.create(tmp_path)
    s.run("hello", enable_deferred_tools=False)

    assert captured["enable_mcp"] is True
    assert captured["enable_deferred_tools"] is False


def test_session_run_without_mcp_keeps_deferred_off(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    from agent.runtime import session as sess_mod

    captured = {}

    def fake_run_task(task, **kw):
        captured.update(kw)
        return ("done", kw.get("initial_messages", []))

    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)

    s = sess_mod.Session.create(tmp_path)
    s.run("hello")

    assert captured.get("enable_mcp") is False
    assert captured.get("enable_deferred_tools") is False


def test_session_run_explicit_mcp_args_override_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ACE_HOME", str(tmp_path / ".ace"))
    monkeypatch.setenv("ACE_MCP_CONFIG", str(tmp_path / "env.mcp.json"))
    from agent.runtime import session as sess_mod

    captured = {}

    def fake_run_task(task, **kw):
        captured.update(kw)
        return ("done", kw.get("initial_messages", []))

    monkeypatch.setattr(sess_mod, "run_task", fake_run_task)

    s = sess_mod.Session.create(tmp_path)
    s.run("hello", enable_mcp=False)

    assert captured["enable_mcp"] is False
    assert captured["mcp_config_path"] is None
