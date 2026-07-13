import json
from types import SimpleNamespace

import pytest

from agent import config
from agent.llm import model_for_purpose
from agent.loop import run_task
from agent.runtime.llm_runtime import (
    display_model_from_settings,
    model_for_purpose as runtime_model_for_purpose,
    resolve_llm_runtime_config,
    using_repl_settings,
)
from agent.runtime.permissions import PermissionEngine, PermissionRule
from agent.runtime.settings import (
    SettingsUpdateError,
    approval_mode_from_settings,
    available_models_from_settings,
    build_permission_engine,
    build_permission_rules,
    load_merged_settings,
    merge_settings,
    parse_permission_rules,
    update_user_settings,
)
from agent.cli.commands import handle as handle_slash
from agent.cli.repl import ModelState


def _bash_spec():
    return SimpleNamespace(name="bash")


def test_load_missing_settings_returns_empty_dict(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    ace_home.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))

    merged = load_merged_settings(tmp_path / "proj")

    assert merged == {}


def test_merge_settings_concatenates_permission_lists():
    merged = merge_settings(
        {"permissions": {"allow": ["Bash(pytest *)"], "defaultMode": "default"}},
        {"permissions": {"allow": ["Read"], "deny": ["Bash(git push*)"]}},
        {"permissions": {"defaultMode": "bypassPermissions"}},
    )

    assert merged["permissions"]["allow"] == ["Bash(pytest *)", "Read"]
    assert merged["permissions"]["deny"] == ["Bash(git push*)"]
    assert merged["permissions"]["defaultMode"] == "bypassPermissions"


def test_approval_mode_from_settings():
    assert approval_mode_from_settings({}) == "ask"
    assert approval_mode_from_settings({"permissions": {"defaultMode": "default"}}) == "ask"
    assert approval_mode_from_settings({"permissions": {"defaultMode": "bypassPermissions"}}) == "auto"


def test_build_permission_rules_allow_and_deny(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    ace_home.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))
    (ace_home / "settings.json").write_text(
        json.dumps({"permissions": {"allow": ["Bash(pytest *)"]}}),
        encoding="utf-8",
    )
    project = tmp_path / "proj"
    project.mkdir()
    ace_dir = project / ".ace"
    ace_dir.mkdir()
    (ace_dir / "settings.json").write_text(
        json.dumps({"permissions": {"deny": ["Bash(git push*)"]}}),
        encoding="utf-8",
    )

    engine = build_permission_engine(project)

    allowed = engine.decide(_bash_spec(), {"command": "pytest tests/test_foo.py"})
    denied = engine.decide(_bash_spec(), {"command": "git push origin main"})
    passthrough = engine.decide(_bash_spec(), {"command": "pip install foo"})

    assert allowed.behavior == "allow"
    assert denied.behavior == "deny"
    assert passthrough.behavior == "passthrough"


def test_project_deny_overrides_user_allow(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    ace_home.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))
    (ace_home / "settings.json").write_text(
        json.dumps({"permissions": {"allow": ["Bash(git push*)"]}}),
        encoding="utf-8",
    )
    project = tmp_path / "proj"
    project.mkdir()
    ace_dir = project / ".ace"
    ace_dir.mkdir()
    (ace_dir / "settings.json").write_text(
        json.dumps({"permissions": {"deny": ["Bash(git push*)"]}}),
        encoding="utf-8",
    )

    engine = build_permission_engine(project)
    decision = engine.decide(_bash_spec(), {"command": "git push origin main"})

    assert decision.behavior == "deny"


def test_parse_permission_rules_skips_ask_entries():
    rules = parse_permission_rules(
        {"allow": ["Read"], "deny": [], "ask": ["Bash(pip install *)"]},
        layer="user",
    )

    assert len(rules) == 1
    assert rules[0].tool_name == "read_file"
    assert rules[0].behavior == "allow"


def test_run_task_without_permission_engine_stays_passthrough():
    engine = PermissionEngine()
    decision = engine.decide(_bash_spec(), {"command": "git push"})

    assert decision.behavior == "passthrough"


def test_run_task_accepts_injected_permission_engine():
    engine = PermissionEngine(
        [PermissionRule("bash", "deny", message="blocked", matcher=lambda _: True)]
    )

    assert engine.decide(_bash_spec(), {"command": "echo hi"}).behavior == "deny"

    # Seam exists on run_task signature; callers (REPL) pass permission_engine through Session.run.
    import inspect

    assert "permission_engine" in inspect.signature(run_task).parameters


def test_resolve_llm_runtime_config_from_env_and_model():
    settings = {
        "model": "deepseek-v4-pro",
        "env": {
            "ANTHROPIC_API_KEY": "sk-test",
            "ANTHROPIC_BASE_URL": "https://example.test",
            "PATH": "/should-be-ignored",
        },
        "models": {
            "memory": "deepseek-v4-flash",
            "recall": "deepseek-v4-flash",
            "compaction": "deepseek-v4-pro",
        },
    }
    runtime = resolve_llm_runtime_config(settings)

    assert runtime is not None
    assert runtime.api_key == "sk-test"
    assert runtime.base_url == "https://example.test"
    assert runtime.model_id == "deepseek-v4-pro"
    assert runtime.memory_model_id == "deepseek-v4-flash"
    assert runtime.recall_model_id == "deepseek-v4-flash"
    assert runtime.compaction_model_id == "deepseek-v4-pro"


def test_repl_llm_runtime_does_not_leak_outside_context(monkeypatch):
    monkeypatch.setattr(config, "MODEL_ID", "config-default", raising=False)

    settings = {"model": "settings-model"}
    assert model_for_purpose("agent") == "config-default"

    with using_repl_settings(settings):
        assert model_for_purpose("agent") == "settings-model"
        assert runtime_model_for_purpose("memory_recall") == "deepseek-v4-flash"
        assert runtime_model_for_purpose("memory_session_memory") == "deepseek-v4-flash"

    assert model_for_purpose("agent") == "config-default"


def test_display_model_from_settings():
    assert display_model_from_settings({"model": "deepseek-v4-flash"}) == "deepseek-v4-flash"
    assert display_model_from_settings({}) == config.MODEL_ID


def test_session_model_overrides_settings(monkeypatch):
    monkeypatch.setattr(config, "MODEL_ID", "config-default", raising=False)
    settings = {"model": "settings-model"}

    assert display_model_from_settings(settings, session_model="session-model") == "session-model"

    with using_repl_settings(settings, session_model="session-model"):
        assert model_for_purpose("agent") == "session-model"

    assert model_for_purpose("agent") == "config-default"


def test_update_user_settings_persists_model(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    ace_home.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))

    update_user_settings({"model": "deepseek-v4-flash"})
    merged = load_merged_settings(tmp_path / "proj")

    assert merged["model"] == "deepseek-v4-flash"

    update_user_settings({"model": None})
    merged = load_merged_settings(tmp_path / "proj")

    assert "model" not in merged


def test_update_user_settings_preserves_other_keys(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    ace_home.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))
    original = {
        "env": {
            "ANTHROPIC_API_KEY": "sk-test",
            "ANTHROPIC_BASE_URL": "https://example.test",
        },
        "permissions": {"allow": ["Bash(pytest *)"], "defaultMode": "default"},
        "models": {"memory": "deepseek-v4-flash"},
        "model": "deepseek-v4-pro",
        "availableModels": ["deepseek-v4-pro", "deepseek-v4-flash"],
    }
    (ace_home / "settings.json").write_text(
        json.dumps(original, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    update_user_settings({"model": "deepseek-v4-flash"})
    saved = json.loads((ace_home / "settings.json").read_text(encoding="utf-8"))

    assert saved["model"] == "deepseek-v4-flash"
    assert saved["env"] == original["env"]
    assert saved["permissions"] == original["permissions"]
    assert saved["models"] == original["models"]
    assert saved["availableModels"] == original["availableModels"]


def test_update_user_settings_refuses_corrupt_file(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    ace_home.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))
    path = ace_home / "settings.json"
    corrupt = '{\n  "env": {"ANTHROPIC_API_KEY": "sk-test"},\n  // comment breaks json\n}\n'
    path.write_text(corrupt, encoding="utf-8")

    with pytest.raises(SettingsUpdateError):
        update_user_settings({"model": "deepseek-v4-flash"})

    assert path.read_text(encoding="utf-8") == corrupt


def test_update_user_settings_patch_permissions_preserves_model(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    ace_home.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))
    original = {
        "model": "deepseek-v4-pro",
        "env": {"ANTHROPIC_API_KEY": "sk-test"},
        "permissions": {
            "allow": ["Bash(pytest *)"],
            "deny": ["Bash(git push*)"],
            "defaultMode": "default",
        },
    }
    (ace_home / "settings.json").write_text(json.dumps(original), encoding="utf-8")

    update_user_settings({"permissions": {"defaultMode": "bypassPermissions"}})
    saved = json.loads((ace_home / "settings.json").read_text(encoding="utf-8"))

    assert saved["model"] == "deepseek-v4-pro"
    assert saved["env"] == original["env"]
    assert saved["permissions"]["defaultMode"] == "bypassPermissions"
    assert saved["permissions"]["allow"] == original["permissions"]["allow"]
    assert saved["permissions"]["deny"] == original["permissions"]["deny"]


def test_update_user_settings_patch_env_preserves_model_and_permissions(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    ace_home.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))
    original = {
        "model": "deepseek-v4-pro",
        "env": {
            "ANTHROPIC_API_KEY": "sk-old",
            "ANTHROPIC_BASE_URL": "https://example.test",
        },
        "permissions": {"allow": ["Read"]},
    }
    (ace_home / "settings.json").write_text(json.dumps(original), encoding="utf-8")

    update_user_settings({"env": {"ANTHROPIC_API_KEY": "sk-new"}})
    saved = json.loads((ace_home / "settings.json").read_text(encoding="utf-8"))

    assert saved["model"] == "deepseek-v4-pro"
    assert saved["permissions"] == original["permissions"]
    assert saved["env"]["ANTHROPIC_API_KEY"] == "sk-new"
    assert saved["env"]["ANTHROPIC_BASE_URL"] == "https://example.test"


def test_available_models_from_settings(monkeypatch):
    monkeypatch.setattr(config, "MODEL_ID", "config-default", raising=False)

    models = available_models_from_settings({
        "availableModels": ["deepseek-v4-pro", "deepseek-v4-flash"],
        "model": "deepseek-v4-pro",
    })

    assert models == ["deepseek-v4-pro", "deepseek-v4-flash", "config-default"]


def test_slash_model_direct_set_is_session_only(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    ace_home.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))
    project = tmp_path / "proj"
    project.mkdir()

    from agent.runtime import Session

    session = Session.create(project)
    model_state = ModelState("old-model")
    lines: list[str] = []

    handle_slash(
        "/model deepseek-v4-flash",
        session,
        project,
        lines.append,
        model_state=model_state,
    )

    assert model_state.display == "deepseek-v4-flash"
    assert model_state.session_model == "deepseek-v4-flash"
    assert any("session only" in line for line in lines)
    assert load_merged_settings(project) == {}


def test_slash_model_default_clears_session_override_only(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    ace_home.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))
    (ace_home / "settings.json").write_text(
        json.dumps({"model": "deepseek-v4-flash"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "MODEL_ID", "config-default", raising=False)
    project = tmp_path / "proj"
    project.mkdir()

    from agent.runtime import Session

    session = Session.create(project)
    model_state = ModelState("deepseek-v4-flash")
    model_state.set("qwen3.7-plus")
    lines: list[str] = []

    handle_slash(
        "/model default",
        session,
        project,
        lines.append,
        model_state=model_state,
    )

    assert model_state.session_model is None
    assert model_state.display == "deepseek-v4-flash"
    assert load_merged_settings(project)["model"] == "deepseek-v4-flash"
    assert any("Using default model" in line for line in lines)


def test_user_settings_parse_error_reports_invalid_json(tmp_path, monkeypatch):
    ace_home = tmp_path / "ace"
    ace_home.mkdir()
    monkeypatch.setenv("ACE_HOME", str(ace_home))
    (ace_home / "settings.json").write_text(
        '{"availableModels": ["a" "b"]}',
        encoding="utf-8",
    )

    from agent.runtime.settings import user_settings_parse_error

    err = user_settings_parse_error()

    assert err is not None
    assert "无法解析" in err
