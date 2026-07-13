"""Public API for the tools package."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def reset_file_read_state() -> None:
    from .file_state import reset_current_file_read_state

    reset_current_file_read_state()


def get_file_read_state():
    from .file_state import get_current_file_read_state

    return get_current_file_read_state()


_EXPORTS = {
    "DockerExecutor": ("agent.tools.executors", "DockerExecutor"),
    "DeferredToolPolicy": ("agent.tools.deferred", "DeferredToolPolicy"),
    "DeferredToolState": ("agent.tools.deferred", "DeferredToolState"),
    "FileReadRecord": ("agent.tools.file_state", "FileReadRecord"),
    "FileReadSnapshot": ("agent.tools.file_state", "FileReadSnapshot"),
    "FileReadState": ("agent.tools.file_state", "FileReadState"),
    "FileReadStateError": ("agent.tools.file_state", "FileReadStateError"),
    "LocalExecutor": ("agent.tools.executors", "LocalExecutor"),
    "ResultMapper": ("agent.tools.contracts", "ResultMapper"),
    "Tool": ("agent.tools.contracts", "Tool"),
    "ToolCall": ("agent.tools.contracts", "ToolCall"),
    "ToolContext": ("agent.tools.contracts", "ToolContext"),
    "ToolExecutionResult": ("agent.tools.runtime", "ToolExecutionResult"),
    "ToolExecutionRuntime": ("agent.tools.runtime", "ToolExecutionRuntime"),
    "ToolPool": ("agent.tools.pool", "ToolPool"),
    "ToolPoolContext": ("agent.tools.pool", "ToolPoolContext"),
    "ToolFlag": ("agent.tools.contracts", "ToolFlag"),
    "HookDecision": ("agent.tools.runtime", "HookDecision"),
    "ToolHookAdapter": ("agent.tools.runtime", "ToolHookAdapter"),
    "ToolResult": ("agent.tools.contracts", "ToolResult"),
    "ToolRequestView": ("agent.tools.request", "ToolRequestView"),
    "ToolUseRequest": ("agent.tools.runtime", "ToolUseRequest"),
    "Validator": ("agent.tools.contracts", "Validator"),
    "assemble_tool_pool": ("agent.tools.pool", "assemble_tool_pool"),
    "bind_file_access": ("agent.tools.executors", "bind_file_access"),
    "bind_memory_file_access": ("agent.tools.executors", "bind_memory_file_access"),
    "build_tool_request_view": ("agent.tools.request", "build_tool_request_view"),
    "create_tool_search_tool": ("agent.tools.deferred", "create_tool_search_tool"),
    "find_tool_by_name": ("agent.tools.pool", "find_tool_by_name"),
    "get_all_base_tools": ("agent.tools.pool", "get_all_base_tools"),
    "get_core_tools": ("agent.tools.builtin_tools", "get_core_tools"),
    "get_executor": ("agent.tools.executors", "get_executor"),
    "get_tools": ("agent.tools.pool", "get_tools"),
    "get_current_file_read_state": ("agent.tools.file_state", "get_current_file_read_state"),
    "is_persisted_path": ("agent.tools.result_store", "is_persisted_path"),
    "maybe_persist": ("agent.tools.result_store", "maybe_persist"),
    "reset_approve_cb": ("agent.tools.executors", "reset_approve_cb"),
    "reset_bash_history": ("agent.tools.executors", "reset_bash_history"),
    "reset_current_file_read_state": ("agent.tools.file_state", "reset_current_file_read_state"),
    "reset_deferred_tool_states": ("agent.tools.deferred", "reset_deferred_tool_states"),
    "reset_executor": ("agent.tools.executors", "reset_executor"),
    "reset_todos": ("agent.tools.handlers", "reset_todos"),
    "run_bash": ("agent.tools.handlers", "run_bash"),
    "run_edit": ("agent.tools.handlers", "run_edit"),
    "run_glob": ("agent.tools.handlers", "run_glob"),
    "run_grep": ("agent.tools.handlers", "run_grep"),
    "run_powershell": ("agent.tools.handlers", "run_powershell"),
    "run_read": ("agent.tools.handlers", "run_read"),
    "run_symbol_search": ("agent.tools.handlers", "run_symbol_search"),
    "run_task_create": ("agent.tools.handlers", "run_task_create"),
    "run_task_get": ("agent.tools.handlers", "run_task_get"),
    "run_task_list": ("agent.tools.handlers", "run_task_list"),
    "run_task_output": ("agent.tools.handlers", "run_task_output"),
    "run_task_stop": ("agent.tools.handlers", "run_task_stop"),
    "run_task_update": ("agent.tools.handlers", "run_task_update"),
    "run_update_todos": ("agent.tools.handlers", "run_update_todos"),
    "run_write": ("agent.tools.handlers", "run_write"),
    "safe_path": ("agent.tools.executors", "safe_path"),
    "set_approve_cb": ("agent.tools.executors", "set_approve_cb"),
    "set_executor": ("agent.tools.executors", "set_executor"),
    "tool_error_count": ("agent.tools.executors", "tool_error_count"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "DockerExecutor",
    "DeferredToolPolicy",
    "DeferredToolState",
    "FileReadRecord",
    "FileReadSnapshot",
    "FileReadState",
    "FileReadStateError",
    "LocalExecutor",
    "ResultMapper",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolExecutionResult",
    "ToolExecutionRuntime",
    "ToolPool",
    "ToolPoolContext",
    "ToolFlag",
    "HookDecision",
    "ToolHookAdapter",
    "ToolResult",
    "ToolRequestView",
    "ToolUseRequest",
    "Validator",
    "assemble_tool_pool",
    "bind_file_access",
    "bind_memory_file_access",
    "build_tool_request_view",
    "create_tool_search_tool",
    "find_tool_by_name",
    "get_all_base_tools",
    "get_core_tools",
    "get_current_file_read_state",
    "get_executor",
    "get_file_read_state",
    "get_tools",
    "is_persisted_path",
    "maybe_persist",
    "reset_approve_cb",
    "reset_bash_history",
    "reset_current_file_read_state",
    "reset_deferred_tool_states",
    "reset_executor",
    "reset_file_read_state",
    "reset_todos",
    "run_bash",
    "run_edit",
    "run_glob",
    "run_grep",
    "run_powershell",
    "run_read",
    "run_symbol_search",
    "run_task_create",
    "run_task_get",
    "run_task_list",
    "run_task_output",
    "run_task_stop",
    "run_task_update",
    "run_update_todos",
    "run_write",
    "safe_path",
    "set_approve_cb",
    "set_executor",
    "tool_error_count",
]
