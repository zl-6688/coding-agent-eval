import json
import subprocess
import sys


def test_core_tool_modules_import_without_cycles():
    code = """
import importlib
import json
import agent.tools
importlib.import_module("agent.tools.contracts")
importlib.import_module("agent.tools.builtin_tools")
importlib.import_module("agent.tools.file_state")
importlib.import_module("agent.tools.executors")
importlib.import_module("agent.tools.symbol_search")
importlib.import_module("agent.tools.handlers")
import agent.tools.pool
import agent.tools.runtime
from agent.tools.pool import assemble_tool_pool
print(json.dumps({
    "pool": [tool.name for tool in assemble_tool_pool().tools],
    "schemas": [tool["name"] for tool in assemble_tool_pool().model_schemas_for_api()],
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["pool"] == payload["schemas"]
    assert "grep" in payload["pool"]
    assert "symbol_search" in payload["pool"]
    assert "powershell" in payload["pool"]


def test_builtin_tools_does_not_eagerly_import_subagent_runner():
    code = """
import importlib
import json
import sys

module = importlib.import_module("agent.tools.builtin_tools")
module.get_core_tools()

print(json.dumps({
    "subagents_package_loaded": "agent.subagents" in sys.modules,
    "subagent_runner_loaded": "agent.subagents.agent_tool" in sys.modules,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload == {
        "subagents_package_loaded": False,
        "subagent_runner_loaded": False,
    }


def test_subagent_agent_tool_imports_without_cycles_and_old_root_module_is_removed():
    code = """
import importlib
import json

import agent.loop
import agent.tools.builtin_tools
subagent_module = importlib.import_module("agent.subagents.agent_tool")

try:
    importlib.import_module("agent.agent_tool")
except ModuleNotFoundError as exc:
    old_root = str(exc)
else:
    old_root = "present"

print(json.dumps({
    "module": subagent_module.__name__,
    "callable": callable(subagent_module.call_agent_tool),
    "old_root": old_root,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["module"] == "agent.subagents.agent_tool"
    assert payload["callable"] is True
    assert "agent.agent_tool" in payload["old_root"]


def test_agent_tools_package_exports_new_api_and_handlers():
    import agent.tools as core_tools
    import agent.tools.file_state as file_read_state
    import agent.tools as tools
    from agent.tools import builtin_tools, contracts, file_state

    assert core_tools.Tool is contracts.Tool
    assert core_tools.ToolResult is contracts.ToolResult
    assert core_tools.ToolFlag is contracts.ToolFlag
    assert core_tools.Validator is contracts.Validator
    assert core_tools.ToolCall is contracts.ToolCall
    assert core_tools.ResultMapper is contracts.ResultMapper
    assert core_tools.get_core_tools is builtin_tools.get_core_tools
    assert file_read_state.FileReadState is file_state.FileReadState
    assert file_read_state.FileReadSnapshot is file_state.FileReadSnapshot

    for name in (
        "Tool",
        "ToolContext",
        "ToolResult",
        "set_executor",
        "reset_executor",
        "get_executor",
        "set_approve_cb",
        "reset_approve_cb",
        "run_bash",
        "run_powershell",
        "run_read",
        "run_write",
        "run_edit",
        "run_glob",
        "run_grep",
        "run_symbol_search",
        "run_update_todos",
        "reset_file_read_state",
        "get_file_read_state",
    ):
        assert hasattr(tools, name)

    for removed in (
        "TOOLS",
        "TOOL_HANDLERS",
        "dispatch",
        "LegacyToolAdapter",
        "CoreToolContext",
        "CoreToolResult",
        "CoreToolSpec",
    ):
        assert not hasattr(tools, removed)
        assert not hasattr(core_tools, removed)
        assert not hasattr(contracts, removed)


def test_dispatch_submodule_is_removed():
    code = """
import json
import importlib

try:
    importlib.import_module("agent.tools.dispatch")
except ModuleNotFoundError as exc:
    message = str(exc)
else:
    message = "no error"

print(json.dumps({"message": message}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    payload = json.loads(result.stdout)

    assert "agent.tools.dispatch" in payload["message"]


def test_runtime_governance_modules_import_without_cycles_and_root_shims():
    code = """
import importlib
import json

import agent.loop
import agent.runtime.hooks
import agent.tools.runtime
from agent.runtime import Session

results = {
    "lazy_session": Session.__name__,
    "new_hooks": importlib.import_module("agent.runtime.hooks").__name__,
    "new_permissions": importlib.import_module("agent.runtime.permissions").__name__,
}

for name in ("agent.hook_bus", "agent.permission_manager"):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        results[name] = "missing"
    else:
        results[name] = "present"

print(json.dumps(results))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload == {
        "lazy_session": "Session",
        "new_hooks": "agent.runtime.hooks",
        "new_permissions": "agent.runtime.permissions",
        "agent.hook_bus": "missing",
        "agent.permission_manager": "missing",
    }


def test_root_context_modules_are_removed():
    code = """
import importlib
import json
import agent

results = {}
for name in ("compact", "project_instructions", "system_prompt"):
    try:
        importlib.import_module(f"agent.{name}")
    except ModuleNotFoundError:
        module_import = "missing"
    else:
        module_import = "present"
    results[name] = {
        "module_import": module_import,
        "package_attr": hasattr(agent, name),
    }

importlib.import_module("agent.context.compact")
importlib.import_module("agent.context.project_instructions")
importlib.import_module("agent.context.system_prompt")
print(json.dumps(results))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload == {
        "compact": {"module_import": "missing", "package_attr": False},
        "project_instructions": {"module_import": "missing", "package_attr": False},
        "system_prompt": {"module_import": "missing", "package_attr": False},
    }
