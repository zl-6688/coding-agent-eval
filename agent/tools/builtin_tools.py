"""Ordered built-in core tool definitions and validators."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import Tool, ToolContext, ToolResult, ToolFlag, Validator
from .handlers import (
    run_bash,
    run_edit,
    run_glob,
    run_grep,
    run_powershell,
    run_read,
    run_symbol_search,
    run_task_create,
    run_task_get,
    run_task_list,
    run_task_output,
    run_task_stop,
    run_task_update,
    run_update_todos,
    run_write,
)


def get_core_tools() -> tuple[Tool, ...]:
    """Return the ordered built-in core tools."""

    return (
        _handler_tool(
            "bash",
            "Run a shell command in the workspace.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "run_in_background": {"type": "boolean"},
                },
                "required": ["command"],
            },
            is_read_only=_shell_is_read_only,
            is_destructive=_shell_is_destructive,
            is_concurrency_safe=False,
            validate_input=_validate_command,
            call=lambda tool_input, context: run_bash(
                tool_input["command"],
                tool_input.get("run_in_background", False),
                context,
            ),
        ),
        _handler_tool(
            "powershell",
            "Run a PowerShell command in the workspace.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "run_in_background": {"type": "boolean"},
                },
                "required": ["command"],
            },
            is_read_only=_shell_is_read_only,
            is_destructive=_shell_is_destructive,
            is_concurrency_safe=False,
            validate_input=_validate_command,
            call=lambda tool_input, context: run_powershell(
                tool_input["command"],
                tool_input.get("run_in_background", False),
                context,
            ),
        ),
        _handler_tool(
            "TaskCreate",
            "Create a persistent graph task with subject, description, active form, metadata, cwd, and worktree.",
            {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "active_form": {"type": ["string", "null"]},
                    "metadata": {"type": ["object", "null"]},
                    "cwd": {"type": ["string", "null"]},
                    "worktree": {"type": ["string", "null"]},
                },
                "required": ["subject", "description"],
            },
            is_read_only=False,
            is_destructive=False,
            is_concurrency_safe=True,
            validate_input=_validate_task_create,
            call=lambda tool_input, context: run_task_create(tool_input, context),
        ),
        _handler_tool(
            "TaskList",
            "List persistent graph tasks. Completed blockers are omitted from each task's blocked_by view.",
            {
                "type": "object",
                "properties": {},
            },
            is_read_only=True,
            is_destructive=False,
            is_concurrency_safe=True,
            call=lambda tool_input, context: run_task_list(tool_input),
        ),
        _handler_tool(
            "TaskGet",
            "Get a persistent graph task by task_id. Missing tasks return task: null.",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                },
                "required": ["task_id"],
            },
            is_read_only=True,
            is_destructive=False,
            is_concurrency_safe=True,
            validate_input=_validate_task_get,
            call=lambda tool_input, context: run_task_get(tool_input["task_id"]),
        ),
        _handler_tool(
            "TaskUpdate",
            (
                "Update a persistent graph task. Use claim_owner to claim, "
                "complete_evidence to complete with evidence, or status=deleted to delete."
            ),
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "deleted"],
                    },
                    "owner": {"type": ["string", "null"]},
                    "claim_owner": {"type": "string"},
                    "cwd": {"type": ["string", "null"]},
                    "worktree": {"type": ["string", "null"]},
                    "blocks": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "blocked_by": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "add_blocks": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "add_blocked_by": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "evidence": {
                        "type": ["object", "array", "string", "null"],
                    },
                    "complete_evidence": {
                        "type": ["object", "array", "string", "null"],
                    },
                    "metadata": {"type": ["object", "null"]},
                    "active_form": {"type": ["string", "null"]},
                },
                "required": ["task_id"],
            },
            is_read_only=False,
            is_destructive=_task_update_is_destructive,
            is_concurrency_safe=True,
            validate_input=_validate_task_update,
            call=lambda tool_input, context: run_task_update(tool_input, context),
        ),
        _handler_tool(
            "TaskOutput",
            "Read output from a runtime background task. Defaults to waiting up to 30000 ms.",
            {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "block": {"type": "boolean", "default": True},
                    "timeout": {
                        "type": "integer",
                        "default": 30000,
                        "description": "Maximum blocking wait in milliseconds. Defaults to 30000.",
                    },
                },
                "required": ["task_id"],
            },
            is_read_only=True,
            is_destructive=False,
            is_concurrency_safe=True,
            validate_input=_validate_task_output,
            call=lambda tool_input, context: run_task_output(
                tool_input["task_id"],
                tool_input.get("block", True),
                tool_input.get("timeout", 30000),
            ),
        ),
        _handler_tool(
            "TaskStop",
            "Stop a running runtime background task.",
            {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
            is_read_only=False,
            is_destructive=True,
            is_concurrency_safe=False,
            validate_input=_validate_task_id,
            call=lambda tool_input, context: run_task_stop(tool_input["task_id"]),
        ),
        _handler_tool(
            "read_file",
            "Read a text file with line numbers; offset/limit can read a line range.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "0-based start line"},
                    "limit": {"type": "integer", "description": "maximum lines to read"},
                },
                "required": ["path"],
            },
            is_read_only=True,
            is_destructive=False,
            is_concurrency_safe=True,
            validate_input=_validate_read_file,
            call=lambda tool_input, context: run_read(
                tool_input["path"],
                tool_input.get("offset", 0),
                tool_input.get("limit"),
                context,
            ),
        ),
        _handler_tool(
            "write_file",
            "Write a file. Existing files require a fresh complete read_file first.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            is_read_only=False,
            is_destructive=True,
            is_concurrency_safe=False,
            validate_input=_validate_path,
            call=lambda tool_input, context: run_write(
                tool_input["path"],
                tool_input["content"],
                context,
            ),
        ),
        _handler_tool(
            "edit_file",
            "Replace exact text in a file; replace_all must be true for multiple matches.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_text", "new_text"],
            },
            is_read_only=False,
            is_destructive=True,
            is_concurrency_safe=False,
            validate_input=_validate_edit_file,
            call=lambda tool_input, context: run_edit(
                tool_input["path"],
                tool_input["old_text"],
                tool_input["new_text"],
                tool_input.get("replace_all", False),
                context,
            ),
        ),
        _handler_tool(
            "glob",
            "Find files by glob pattern.",
            {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
            is_read_only=True,
            is_destructive=False,
            is_concurrency_safe=True,
            validate_input=_validate_pattern,
            call=lambda tool_input, context: run_glob(tool_input["pattern"], context),
        ),
        _handler_tool(
            "grep",
            "Search file contents without using the shell tool.",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                    "case_insensitive": {"type": "boolean"},
                    "line_numbers": {"type": "boolean"},
                    "head_limit": {"type": "integer"},
                    "offset": {"type": "integer"},
                },
                "required": ["pattern"],
            },
            is_read_only=True,
            is_destructive=False,
            is_concurrency_safe=True,
            validate_input=_validate_grep,
            call=lambda tool_input, context: run_grep(
                tool_input["pattern"],
                tool_input.get("path", "."),
                tool_input.get("glob"),
                tool_input.get("case_insensitive", False),
                tool_input.get("line_numbers", True),
                tool_input.get("head_limit", 100),
                tool_input.get("offset", 0),
                context,
            ),
        ),
        _handler_tool(
            "symbol_search",
            "Lightweight Python AST symbol lookup with explicit text-search fallback; not a real LSP client.",
            {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["document_symbols", "definition", "references"],
                    },
                    "file_path": {"type": "string"},
                    "symbol": {"type": "string"},
                    "line": {
                        "type": "integer",
                        "description": "1-based line used to infer a symbol when symbol is omitted.",
                    },
                    "character": {
                        "type": "integer",
                        "description": "1-based character used with line to infer a symbol.",
                    },
                    "include_fallback": {"type": "boolean", "default": True},
                },
                "required": ["operation", "file_path"],
            },
            is_read_only=True,
            is_destructive=False,
            is_concurrency_safe=True,
            validate_input=_validate_symbol_search,
            call=lambda tool_input, context: run_symbol_search(
                tool_input["operation"],
                tool_input["file_path"],
                symbol=tool_input.get("symbol"),
                line=tool_input.get("line"),
                character=tool_input.get("character"),
                include_fallback=tool_input.get("include_fallback", True),
            ),
        ),
        _handler_tool(
            "update_todos",
            "Maintain the current task plan and progress list.",
            {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["content", "status"],
                        },
                    }
                },
                "required": ["todos"],
            },
            is_read_only=False,
            is_destructive=False,
            is_concurrency_safe=False,
            call=lambda tool_input, context: run_update_todos(tool_input["todos"]),
        ),
        Tool(
            name="Agent",
            description=(
                "Run a synchronous one-shot fresh subagent for an isolated task. "
                "Only the general-purpose subagent is supported in this version."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Complete task instructions for the subagent.",
                    },
                    "subagent_type": {
                        "type": "string",
                        "description": "Optional subagent type. Currently only general-purpose.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short label for tracing or progress display.",
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "Maximum child LLM turns, from 1 to 20. Defaults to 6.",
                    },
                },
                "required": ["prompt"],
            },
            is_read_only=True,
            is_destructive=False,
            is_concurrency_safe=False,
            validate_input=_validate_agent_tool,
            call=_call_agent_tool,
            metadata={"subagent_types": ["general-purpose"]},
        ),
    )


def _handler_tool(
    name: str,
    description: str,
    input_schema: Mapping[str, Any],
    *,
    is_read_only: ToolFlag,
    is_destructive: ToolFlag,
    is_concurrency_safe: ToolFlag,
    validate_input: Validator | None = None,
    call,
) -> Tool:
    return Tool(
        name=name,
        description=description,
        input_schema=input_schema,
        is_read_only=is_read_only,
        is_destructive=is_destructive,
        is_concurrency_safe=is_concurrency_safe,
        validate_input=validate_input,
        call=call,
    )


def _validate_command(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    command = tool_input.get("command", "")
    if not str(command).strip():
        return "command must be non-empty"
    return None


def _validate_path(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    path = tool_input.get("path", "")
    if not str(path).strip():
        return "path must be non-empty"
    return None


def _validate_read_file(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    path_error = _validate_path(tool_input, context)
    if path_error is not None:
        return path_error
    for field in ("offset", "limit"):
        if field in tool_input and tool_input[field] is not None and tool_input[field] < 0:
            return f"{field} must be non-negative"
    return None


def _validate_edit_file(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    path_error = _validate_path(tool_input, context)
    if path_error is not None:
        return path_error
    if tool_input.get("old_text", "") == "":
        return "old_text must be non-empty"
    return None


def _validate_pattern(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    pattern = tool_input.get("pattern", "")
    if not str(pattern).strip():
        return "pattern must be non-empty"
    return None


def _validate_grep(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    pattern_error = _validate_pattern(tool_input, context)
    if pattern_error is not None:
        return pattern_error
    for field in ("head_limit", "offset"):
        if field in tool_input and tool_input[field] is not None and tool_input[field] < 0:
            return f"{field} must be non-negative"
    return None


def _validate_symbol_search(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    field_error = _validate_known_fields(tool_input, _SYMBOL_SEARCH_FIELDS)
    if field_error is not None:
        return field_error

    operation = str(tool_input.get("operation") or "").strip()
    if operation not in _SYMBOL_SEARCH_OPERATIONS:
        return "operation must be document_symbols, definition, or references"

    file_path = str(tool_input.get("file_path") or "").strip()
    if not file_path:
        return "file_path must be non-empty"

    if "symbol" in tool_input and tool_input["symbol"] is not None:
        if not isinstance(tool_input["symbol"], str):
            return "symbol must be a string"
        if not tool_input["symbol"].strip():
            return "symbol must be non-empty when provided"

    for field in ("line", "character"):
        if field in tool_input and tool_input[field] is not None:
            value = tool_input[field]
            if isinstance(value, bool) or not isinstance(value, int):
                return f"{field} must be an integer"
            if value < 1:
                return f"{field} must be positive"

    if "character" in tool_input and tool_input.get("character") is not None:
        if tool_input.get("line") is None:
            return "line must be provided when character is provided"

    if "include_fallback" in tool_input and tool_input["include_fallback"] is not None:
        if not isinstance(tool_input["include_fallback"], bool):
            return "include_fallback must be a boolean"

    if operation in {"definition", "references"}:
        has_symbol = bool(str(tool_input.get("symbol") or "").strip())
        has_line = tool_input.get("line") is not None
        if not has_symbol and not has_line:
            return "symbol or line must be provided for definition and references"
    return None


def _validate_task_create(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    field_error = _validate_known_fields(tool_input, _TASK_CREATE_FIELDS)
    if field_error is not None:
        return field_error
    subject = tool_input.get("subject", "")
    if not str(subject).strip():
        return "subject must be non-empty"
    if "description" in tool_input and tool_input["description"] is None:
        return "description must be a string"
    return None


def _validate_task_get(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    field_error = _validate_known_fields(tool_input, _TASK_GET_FIELDS)
    if field_error is not None:
        return field_error
    return _validate_graph_task_id(tool_input, context)


def _validate_graph_task_id(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    task_id = _graph_task_id(tool_input)
    if not task_id.strip():
        return "task_id must be non-empty"
    return None


def _validate_task_update(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    field_error = _validate_known_fields(tool_input, _TASK_UPDATE_FIELDS)
    if field_error is not None:
        return field_error
    task_id_error = _validate_graph_task_id(tool_input, context)
    if task_id_error is not None:
        return task_id_error
    status = tool_input.get("status")
    if status is not None and status not in {
        "pending",
        "in_progress",
        "completed",
        "deleted",
    }:
        return "status must be pending, in_progress, completed, or deleted"
    if "claim_owner" in tool_input:
        owner = tool_input.get("claim_owner") or context.agent_id
        if not str(owner).strip():
            return "claim_owner must be non-empty"
    subject = tool_input.get("subject")
    if subject is not None and not str(subject).strip():
        return "subject must be non-empty"
    return _validate_graph_edges(tool_input)


def _validate_graph_edges(tool_input: dict[str, Any]) -> str | None:
    for field in (
        "blocks",
        "blocked_by",
        "add_blocks",
        "add_blocked_by",
    ):
        if field in tool_input and tool_input[field] is not None:
            if not isinstance(tool_input[field], list):
                return f"{field} must be an array"
            if any(not str(item).strip() for item in tool_input[field]):
                return f"{field} entries must be non-empty"
    return None


def _validate_task_id(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    task_id = tool_input.get("task_id", "")
    if not str(task_id).strip():
        return "task_id must be non-empty"
    return None


def _validate_task_output(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    task_id_error = _validate_task_id(tool_input, context)
    if task_id_error is not None:
        return task_id_error
    if "timeout" in tool_input and tool_input["timeout"] is not None:
        if tool_input["timeout"] < 0 or tool_input["timeout"] > 600000:
            return "timeout must be between 0 and 600000"
    return None


def _validate_agent_tool(tool_input: dict[str, Any], context: ToolContext) -> str | None:
    prompt = tool_input.get("prompt", "")
    if not str(prompt).strip():
        return "prompt must be non-empty"
    if "max_turns" in tool_input and tool_input["max_turns"] is not None:
        max_turns = tool_input["max_turns"]
        if max_turns < 1 or max_turns > 20:
            return "max_turns must be between 1 and 20"
    return None


def _call_agent_tool(tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
    from agent.subagents.agent_tool import call_agent_tool

    return call_agent_tool(tool_input, context)


def _graph_task_id(tool_input: dict[str, Any]) -> str:
    return str(tool_input.get("task_id") or "")


def _validate_known_fields(tool_input: dict[str, Any], allowed: frozenset[str]) -> str | None:
    extra = sorted(set(tool_input) - allowed)
    if extra:
        return f"unsupported field: {extra[0]}"
    return None


_TASK_CREATE_FIELDS = frozenset(
    {"subject", "description", "active_form", "metadata", "cwd", "worktree"}
)
_TASK_GET_FIELDS = frozenset({"task_id"})
_TASK_UPDATE_FIELDS = frozenset(
    {
        "task_id",
        "subject",
        "description",
        "status",
        "owner",
        "claim_owner",
        "cwd",
        "worktree",
        "blocks",
        "blocked_by",
        "add_blocks",
        "add_blocked_by",
        "evidence",
        "complete_evidence",
        "metadata",
        "active_form",
    }
)
_SYMBOL_SEARCH_OPERATIONS = frozenset({"document_symbols", "definition", "references"})
_SYMBOL_SEARCH_FIELDS = frozenset(
    {"operation", "file_path", "symbol", "line", "character", "include_fallback"}
)


def _task_update_is_destructive(tool_input: dict[str, Any]) -> bool:
    return tool_input.get("status") == "deleted"


_READ_ONLY_SHELL_COMMANDS = frozenset(
    {
        "cat",
        "cd",
        "dir",
        "echo",
        "find",
        "findstr",
        "git",
        "grep",
        "get-childitem",
        "get-content",
        "get-location",
        "head",
        "ls",
        "pwd",
        "rg",
        "select-string",
        "stat",
        "tail",
        "tree",
        "type",
        "wc",
        "where",
        "which",
    }
)
_DESTRUCTIVE_MARKERS = (
    " rm ",
    " rm-",
    " del ",
    " erase ",
    " remove-item ",
    " rmdir ",
    " shutdown",
    " reboot",
    " mkfs",
    "sudo ",
    "> /dev/",
)


def _shell_is_read_only(tool_input: dict[str, Any]) -> bool:
    command = str(tool_input.get("command") or "").strip().lower()
    if not command or _shell_is_destructive(tool_input):
        return False
    first = command.replace(";", " ").replace("&", " ").replace("|", " ").split()[0]
    return first in _READ_ONLY_SHELL_COMMANDS


def _shell_is_destructive(tool_input: dict[str, Any]) -> bool:
    padded = f" {str(tool_input.get('command') or '').lower()} "
    return any(marker in padded for marker in _DESTRUCTIVE_MARKERS)


__all__ = ["get_core_tools"]
