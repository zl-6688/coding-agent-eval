import importlib.util
import io
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def _noop_workdir(_path):
    yield


def _run_cli(monkeypatch, tmp_path, *, argv, stdin, env=None, fake_run_task=None):
    import agent.evoclaw_cli as cli

    captured = {}

    def default_run_task(_task, **kwargs):
        captured.update(kwargs)
        sm = kwargs.get("session_memory")
        if sm is not None:
            sm._initialized = True
            sm._tokens_at_last = 12345
            sm._seen_tool_ids = {"tool-1", "tool-2"}
            sm.set_last_summarized_message_id("assistant-1")
        return "done", [{"role": "assistant", "content": "ok"}]

    monkeypatch.setattr(cli, "SESS", tmp_path / ".myagent")
    monkeypatch.setattr(cli.config, "using_workdir", _noop_workdir)
    monkeypatch.setattr(cli.tools, "set_executor", lambda _executor: None)
    monkeypatch.setattr(cli, "resolve_run_task_runtime_kwargs", lambda: {})
    monkeypatch.setattr(cli.loop, "run_task", fake_run_task or default_run_task)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    monkeypatch.setenv("AGENT_WORKDIR", str(tmp_path))
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    return cli, captured


def test_evoclaw_cli_passes_session_memory_and_persists_state(monkeypatch, tmp_path):
    cli, captured = _run_cli(
        monkeypatch,
        tmp_path,
        argv=["myagent", "run", "--session-id", "sid-1"],
        stdin="task",
        env={
            "COMPACT_STRATEGY": "pipeline",
            "MYAGENT_SESSION_MEMORY": "1",
            "MYAGENT_ARM_LABEL": "pipeline_sm",
        },
    )

    sm = captured["session_memory"]
    assert sm is not None
    assert sm.path == tmp_path / ".myagent" / "session_memory" / "sid-1.notes.md"
    assert captured["eval_hooks"].compact_strategy == "pipeline"
    assert captured["meta"]["arm"] == "pipeline_sm"
    assert captured["meta"]["compact_strategy"] == "pipeline"
    assert captured["meta"]["session_memory_enabled"] is True

    state = json.loads((cli._sm_state_path("sid-1")).read_text(encoding="utf-8"))
    assert state["initialized"] is True
    assert state["tokens_at_last"] == 12345
    assert state["seen_tool_ids"] == ["tool-1", "tool-2"]
    assert state["last_summarized_message_id"] == "assistant-1"


def test_evoclaw_cli_passes_stop_at_context(monkeypatch, tmp_path):
    _, captured = _run_cli(
        monkeypatch,
        tmp_path,
        argv=["myagent", "run", "--session-id", "sid-stop"],
        stdin="task",
        env={
            "COMPACT_STRATEGY": "none",
            "MYAGENT_STOP_AT_CONTEXT": "167000",
        },
    )

    assert captured["eval_hooks"].compact_strategy == "none"
    assert captured["eval_hooks"].stop_at_context == 167000


def test_evoclaw_cli_restores_session_memory_state_on_resume(monkeypatch, tmp_path):
    import agent.evoclaw_cli as cli

    monkeypatch.setattr(cli, "SESS", tmp_path / ".myagent")
    cli._save("sid-2", [{"role": "user", "content": "old"}])
    cli._sm_state_path("sid-2").parent.mkdir(parents=True, exist_ok=True)
    cli._sm_state_path("sid-2").write_text(
        json.dumps(
            {
                "initialized": True,
                "tokens_at_last": 6789,
                "seen_tool_ids": ["tool-a"],
                "last_summarized_message_id": "assistant-prev",
            }
        ),
        encoding="utf-8",
    )

    def fake_run_task(_task, **kwargs):
        sm = kwargs["session_memory"]
        assert sm.last_summarized_message_id == "assistant-prev"
        assert sm._tokens_at_last == 6789
        assert sm._seen_tool_ids == {"tool-a"}
        assert kwargs["initial_messages"][-1] == {"role": "user", "content": "new"}
        return "done", kwargs["initial_messages"]

    _run_cli(
        monkeypatch,
        tmp_path,
        argv=["myagent", "resume", "--session-id", "sid-2"],
        stdin="new",
        env={"COMPACT_STRATEGY": "pipeline", "MYAGENT_SESSION_MEMORY": "true"},
        fake_run_task=fake_run_task,
    )


def test_evoclaw_cli_leaves_session_memory_off_by_default(monkeypatch, tmp_path):
    _, captured = _run_cli(
        monkeypatch,
        tmp_path,
        argv=["myagent", "run", "--session-id", "sid-3"],
        stdin="task",
        env={"COMPACT_STRATEGY": "pipeline"},
    )

    assert captured["session_memory"] is None
    assert captured["meta"]["arm"] == "pipeline"
    assert captured["meta"]["session_memory_enabled"] is False


def _load_myagent_module(monkeypatch, tmp_path):
    harness = types.ModuleType("harness")
    e2e = types.ModuleType("harness.e2e")
    agents = types.ModuleType("harness.e2e.agents")
    base = types.ModuleType("harness.e2e.agents.base")

    class AgentFramework:
        def __init__(self, **_kwargs):
            pass

        def get_quarantine_env_vars(self):
            return []

    def register_framework(_name):
        return lambda cls: cls

    base.AgentFramework = AgentFramework
    base.register_framework = register_framework
    monkeypatch.setitem(sys.modules, "harness", harness)
    monkeypatch.setitem(sys.modules, "harness.e2e", e2e)
    monkeypatch.setitem(sys.modules, "harness.e2e.agents", agents)
    monkeypatch.setitem(sys.modules, "harness.e2e.agents.base", base)

    src = Path("eval/evoclaw/myagent.py").resolve()
    spec = importlib.util.spec_from_file_location(f"myagent_under_test_{tmp_path.name}", src)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evoclaw_adapter_passes_session_memory_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MYAGENT_REPO", str(Path.cwd()))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("COMPACT_STRATEGY", "pipeline")
    monkeypatch.setenv("MYAGENT_SESSION_MEMORY", "1")
    monkeypatch.setenv("MYAGENT_ARM_LABEL", "pipeline_sm")

    module = _load_myagent_module(monkeypatch, tmp_path)
    env = module.MyAgentFramework().get_container_env_vars()

    assert "-e" in env
    assert "COMPACT_STRATEGY=pipeline" in env
    assert "MYAGENT_SESSION_MEMORY=1" in env
    assert "MYAGENT_ARM_LABEL=pipeline_sm" in env


def test_evoclaw_adapter_passes_stop_at_context_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MYAGENT_REPO", str(Path.cwd()))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("MYAGENT_STOP_AT_CONTEXT", "167000")

    module = _load_myagent_module(monkeypatch, tmp_path)
    env = module.MyAgentFramework().get_container_env_vars()

    assert "MYAGENT_STOP_AT_CONTEXT=167000" in env


def test_evoclaw_adapter_passes_empty_stop_at_context_to_override_image_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MYAGENT_REPO", str(Path.cwd()))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("MYAGENT_STOP_AT_CONTEXT", "")

    module = _load_myagent_module(monkeypatch, tmp_path)
    env = module.MyAgentFramework().get_container_env_vars()

    assert "MYAGENT_STOP_AT_CONTEXT=" in env
