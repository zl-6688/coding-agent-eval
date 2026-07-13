from pathlib import Path

from agent.mcp.runtime_config import (
    UNSET,
    McpRuntimeConfig,
    mcp_runtime_config_from_env,
    parse_enable_mcp,
    resolve_deferred_runtime_kwargs,
    resolve_mcp_runtime_kwargs,
    resolve_run_task_runtime_kwargs,
)


def test_parse_enable_mcp_bool_values():
    for value in ("1", "true", "TRUE", "yes", "on", "y"):
        assert parse_enable_mcp(value) is True

    for value in ("0", "false", "FALSE", "no", "off", "n", "", None, "maybe"):
        assert parse_enable_mcp(value) is False


def test_env_config_path_enables_mcp():
    cfg = mcp_runtime_config_from_env({
        "ACE_MCP_CONFIG": "C:/tmp/project.mcp.json",
    })

    assert cfg == McpRuntimeConfig(
        enable_mcp=True,
        mcp_config_path="C:/tmp/project.mcp.json",
    )
    assert cfg.as_run_task_kwargs() == {
        "enable_mcp": True,
        "mcp_config_path": "C:/tmp/project.mcp.json",
    }


def test_explicit_runtime_args_override_env():
    env = {
        "ACE_ENABLE_MCP": "1",
        "ACE_MCP_CONFIG": "env.mcp.json",
    }

    assert resolve_mcp_runtime_kwargs(
        enable_mcp=False,
        mcp_config_path=UNSET,
        env=env,
    ) == {
        "enable_mcp": False,
        "mcp_config_path": None,
    }

    assert resolve_mcp_runtime_kwargs(
        enable_mcp=UNSET,
        mcp_config_path=None,
        env=env,
    ) == {
        "enable_mcp": True,
        "mcp_config_path": None,
    }

    explicit_path = Path("explicit.mcp.json")
    assert resolve_mcp_runtime_kwargs(
        enable_mcp=False,
        mcp_config_path=explicit_path,
        env=env,
    ) == {
        "enable_mcp": True,
        "mcp_config_path": "explicit.mcp.json",
    }


def test_deferred_defaults_off_without_mcp():
    assert resolve_deferred_runtime_kwargs(
        enable_mcp=False,
        mcp_config_path=None,
    ) == {"enable_deferred_tools": False}


def test_deferred_defaults_on_when_mcp_enabled():
    assert resolve_deferred_runtime_kwargs(
        enable_mcp=True,
        mcp_config_path=None,
    ) == {"enable_deferred_tools": True}
    assert resolve_deferred_runtime_kwargs(
        enable_mcp=False,
        mcp_config_path="project.mcp.json",
    ) == {"enable_deferred_tools": True}


def test_deferred_explicit_false_overrides_mcp_default():
    assert resolve_deferred_runtime_kwargs(
        enable_mcp=True,
        mcp_config_path="project.mcp.json",
        enable_deferred_tools=False,
    ) == {"enable_deferred_tools": False}


def test_run_task_runtime_kwargs_merge_mcp_and_deferred():
    env = {"ACE_MCP_CONFIG": "env.mcp.json"}
    assert resolve_run_task_runtime_kwargs(env=env) == {
        "enable_mcp": True,
        "mcp_config_path": "env.mcp.json",
        "enable_deferred_tools": True,
    }
    assert resolve_run_task_runtime_kwargs(
        enable_mcp=False,
        mcp_config_path=UNSET,
        enable_deferred_tools=False,
        env=env,
    ) == {
        "enable_mcp": False,
        "mcp_config_path": None,
        "enable_deferred_tools": False,
    }


def test_workdir_mcp_json_auto_enables_mcp_and_deferred(tmp_path):
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")

    assert resolve_run_task_runtime_kwargs(workdir=tmp_path, env={}) == {
        "enable_mcp": True,
        "mcp_config_path": None,
        "enable_deferred_tools": True,
    }


def test_explicit_false_suppresses_workdir_auto_discovery(tmp_path):
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")

    assert resolve_run_task_runtime_kwargs(
        enable_mcp=False,
        workdir=tmp_path,
        env={},
    ) == {
        "enable_mcp": False,
        "mcp_config_path": None,
        "enable_deferred_tools": False,
    }


def test_disable_mcp_suppresses_explicit_env_and_workdir_config(tmp_path):
    (tmp_path / ".mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")

    assert resolve_run_task_runtime_kwargs(
        enable_mcp=True,
        mcp_config_path="explicit.mcp.json",
        disable_mcp=True,
        workdir=tmp_path,
        env={"ACE_ENABLE_MCP": "1", "ACE_MCP_CONFIG": "env.mcp.json"},
    ) == {
        "enable_mcp": False,
        "mcp_config_path": None,
        "enable_deferred_tools": False,
    }
