"""Concrete core tool handlers."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from obs.trace import mark_current_error

from agent.runtime.observability import (
    content_preview_attrs,
    content_summary_attrs,
    safe_set_current_span,
    safe_text_length,
)
from agent.tasks import (
    format_task_output,
    format_task_start,
    format_task_stop,
    read_task_output,
    start_background_shell_task,
    stop_task,
)
from agent.tasks.graph import TaskGraphError, TaskGraphStore

from .contracts import ToolContext, ToolResult
from .executors import (
    _DANGEROUS,
    command_kind,
    get_executor,
    seen_before,
)
from .file_state import get_current_file_read_state


_READ_OUTPUT_LIMIT = 30000
_TODOS: list[Any] = []


def run_bash(
    command: str,
    run_in_background: bool = False,
    context: ToolContext | None = None,
) -> str | ToolResult:
    ex = get_executor()
    timeout = getattr(ex, "default_timeout", 120)
    meta = _shell_meta(command, repeated_key=command)
    if any(d in command for d in _DANGEROUS):
        safe_set_current_span(
            **{
                **meta,
                **_shell_output_attrs("Error: dangerous command blocked"),
                "tool.exit_code": -1,
                "tool.error_kind": "blocked",
                "tool.stdout_chars": 0,
                "tool.stderr_chars": 0,
                "tool.output_chars": 0,
            }
        )
        return "Error: \u5371\u9669\u547d\u4ee4\u5df2\u62e6\u622a"
    if run_in_background:
        return _start_background_command("local_bash", command, context, meta)
    try:
        stdout, stderr, rc = ex.exec_shell(command, timeout=timeout)
        out = (stdout + stderr).strip()
        safe_set_current_span(
            **{
                **meta,
                **_shell_output_attrs(out),
                "tool.exit_code": rc,
                "tool.error_kind": "ok" if rc == 0 else "nonzero_exit",
                "tool.stdout_chars": safe_text_length(stdout),
                "tool.stderr_chars": safe_text_length(stderr),
                "tool.output_chars": safe_text_length(out),
            }
        )
        if rc != 0:
            mark_current_error(f"tool_error:nonzero_exit:rc={rc}")
        return out[:30000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        safe_set_current_span(
            **{
                **meta,
                **_shell_output_attrs(f"Error: timeout ({timeout}s)"),
                "tool.exit_code": -1,
                "tool.error_kind": "timeout",
                "tool.stdout_chars": 0,
                "tool.stderr_chars": 0,
                "tool.output_chars": 0,
            }
        )
        return f"Error: \u8d85\u65f6 ({timeout}s)"
    except Exception as e:
        safe_set_current_span(
            **{
                **meta,
                **_shell_output_attrs(f"Error: {type(e).__name__}: {e}"),
                "tool.exit_code": -1,
                "tool.error_kind": "exception",
                "tool.error_type": type(e).__name__,
                "tool.stderr_chars": safe_text_length(e),
                "tool.output_chars": 0,
            }
        )
        return f"Error: {type(e).__name__}: {e}"


def run_powershell(
    command: str,
    run_in_background: bool = False,
    context: ToolContext | None = None,
) -> str | ToolResult:
    ex = get_executor()
    timeout = getattr(ex, "default_timeout", 120)
    meta = _shell_meta(command, repeated_key=f"powershell:{command}")
    if any(d in command for d in _DANGEROUS):
        safe_set_current_span(
            **{
                **meta,
                **_shell_output_attrs("Error: dangerous command blocked"),
                "tool.exit_code": -1,
                "tool.error_kind": "blocked",
                "tool.stdout_chars": 0,
                "tool.stderr_chars": 0,
                "tool.output_chars": 0,
            }
        )
        return "Error: dangerous command blocked"
    if run_in_background:
        return _start_background_command("local_powershell", command, context, meta)
    try:
        stdout, stderr, rc = ex.exec_powershell(command, timeout=timeout)
        out = (stdout + stderr).strip()
        safe_set_current_span(
            **{
                **meta,
                **_shell_output_attrs(out),
                "tool.exit_code": rc,
                "tool.error_kind": "ok" if rc == 0 else "nonzero_exit",
                "tool.stdout_chars": safe_text_length(stdout),
                "tool.stderr_chars": safe_text_length(stderr),
                "tool.output_chars": safe_text_length(out),
            }
        )
        if rc != 0:
            mark_current_error(f"tool_error:nonzero_exit:rc={rc}")
        return out[:30000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        safe_set_current_span(
            **{
                **meta,
                **_shell_output_attrs(f"Error: timeout ({timeout}s)"),
                "tool.exit_code": -1,
                "tool.error_kind": "timeout",
                "tool.stdout_chars": 0,
                "tool.stderr_chars": 0,
                "tool.output_chars": 0,
            }
        )
        return f"Error: timeout ({timeout}s)"
    except Exception as e:
        safe_set_current_span(
            **{
                **meta,
                **_shell_output_attrs(f"Error: {type(e).__name__}: {e}"),
                "tool.exit_code": -1,
                "tool.error_kind": "exception",
                "tool.error_type": type(e).__name__,
                "tool.stderr_chars": safe_text_length(e),
                "tool.output_chars": 0,
            }
        )
        return f"Error: {type(e).__name__}: {e}"


def _shell_meta(command: str, *, repeated_key: str) -> dict[str, Any]:
    return {
        **content_summary_attrs("tool.command", command),
        **content_preview_attrs("tool.command", command),
        "tool.command_chars": safe_text_length(command),
        "tool.command_kind": command_kind(command),
        "tool.repeated_command": seen_before(repeated_key),
    }


def _shell_output_attrs(output: str) -> dict[str, Any]:
    return {
        **content_summary_attrs("tool.output", output),
        **content_preview_attrs("tool.output", output),
    }


def run_task_output(task_id: str, block: bool = False, timeout: int = 0) -> ToolResult:
    result = read_task_output(
        str(task_id),
        block=bool(block),
        timeout_ms=max(0, int(timeout or 0)),
        mark_notified=True,
    )
    if result is None:
        return ToolResult(content=f"Error: task not found: {task_id}", is_error=True)
    return ToolResult(content=format_task_output(result), metadata=result)


def run_task_stop(task_id: str) -> ToolResult:
    result = stop_task(str(task_id))
    if result is None:
        return ToolResult(content=f"Error: task not found: {task_id}", is_error=True)
    if result["status"] != "killed":
        return ToolResult(
            content=f"Error: task is not running: {task_id} (status={result['status']})",
            is_error=True,
            metadata=result,
        )
    return ToolResult(content=format_task_stop(result), metadata=result)


def run_task_create(tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
    store = TaskGraphStore()
    try:
        task = store.create_task(
            subject=tool_input.get("subject", ""),
            description=tool_input.get("description", ""),
            cwd=tool_input.get("cwd") or _context_cwd(context),
            worktree=tool_input.get("worktree"),
            metadata=tool_input.get("metadata"),
            active_form=tool_input.get("active_form"),
        )
    except TaskGraphError as exc:
        return _task_graph_error(exc)
    return _json_tool_result({"task": task})


def run_task_list(tool_input: dict[str, Any] | None = None) -> ToolResult:
    store = TaskGraphStore()
    try:
        tasks = [
            task
            for task in store.list_tasks()
            if not task.get("metadata", {}).get("_internal")
        ]
    except TaskGraphError as exc:
        return _task_graph_error(exc)
    by_id = {task["id"]: task for task in tasks}
    views = [_task_list_view(task, by_id) for task in tasks]
    return _json_tool_result({"tasks": views})


def run_task_get(task_id: str) -> ToolResult:
    store = TaskGraphStore()
    try:
        task = store.get_task(str(task_id))
    except TaskGraphError as exc:
        return _task_graph_error(exc)
    return _json_tool_result({"task": task})


def run_task_update(
    tool_input: dict[str, Any],
    context: ToolContext | None = None,
) -> ToolResult:
    store = TaskGraphStore()
    task_id = _task_id(tool_input)
    try:
        if "claim_owner" in tool_input:
            owner = tool_input.get("claim_owner")
            owner = owner or getattr(context, "agent_id", "")
            result = store.claim_task(task_id, str(owner))
            return _json_tool_result(result)

        if "complete_evidence" in tool_input:
            evidence = tool_input.get("complete_evidence")
            task = store.complete_task(
                task_id,
                evidence=evidence,
                owner=tool_input.get("owner"),
            )
            return _json_tool_result({"task": task})

        task = store.update_task(task_id, **_task_update_payload(tool_input))
    except TaskGraphError as exc:
        return _task_graph_error(exc)
    return _json_tool_result({"task": task})


def _start_background_command(
    task_type: str,
    command: str,
    context: ToolContext | None,
    meta: dict[str, Any],
) -> ToolResult:
    try:
        cwd = _background_cwd(context)
        task = start_background_shell_task(
            command,
            task_type=task_type,  # type: ignore[arg-type]
            run_id=getattr(context, "run_id", "") if context is not None else "",
            cwd=cwd,
        )
        snapshot = task.snapshot()
        safe_set_current_span(
            **{
                **meta,
                "tool.background": True,
                "tool.task_id": task.id,
                "tool.output_file": task.output_file,
                "tool.error_kind": "background_started",
            }
        )
        return ToolResult(content=format_task_start(task), metadata=snapshot)
    except Exception as e:
        safe_set_current_span(
            **{
                **meta,
                "tool.background": True,
                "tool.exit_code": -1,
                "tool.error_kind": "exception",
                "tool.error_type": type(e).__name__,
                "tool.stderr_chars": safe_text_length(e),
                "tool.output_chars": 0,
            }
        )
        mark_current_error(f"tool_error:exception:{type(e).__name__}")
        return ToolResult(content=f"Error: {type(e).__name__}: {e}", is_error=True)


def _background_cwd(context: ToolContext | None) -> str:
    executor = getattr(context, "executor", None) if context is not None else None
    if executor is None:
        executor = get_executor()
    if getattr(executor, "kind", "local") != "local":
        raise RuntimeError("background shell tasks are only supported for local executors")
    host_cwd = getattr(executor, "host_cwd", None)
    if host_cwd:
        return str(host_cwd)
    return str(getattr(executor, "cwd", ""))


def _context_cwd(context: ToolContext | None) -> str | None:
    if context is not None and context.cwd:
        return str(context.cwd)
    try:
        return str(get_executor().cwd)
    except Exception:
        return None


def _json_tool_result(payload: dict[str, Any]) -> ToolResult:
    return ToolResult(
        content=json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
        metadata=payload,
    )


def _task_graph_error(exc: Exception) -> ToolResult:
    return ToolResult(content=f"Error: {type(exc).__name__}: {exc}", is_error=True)


def _task_id(tool_input: dict[str, Any]) -> str:
    return str(tool_input.get("task_id") or "")


def _task_update_payload(tool_input: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    aliases = {
        "subject": ("subject",),
        "description": ("description",),
        "status": ("status",),
        "owner": ("owner",),
        "cwd": ("cwd",),
        "worktree": ("worktree",),
        "active_form": ("active_form",),
        "metadata": ("metadata",),
        "evidence": ("evidence",),
        "blocks": ("blocks",),
        "blocked_by": ("blocked_by",),
        "add_blocks": ("add_blocks",),
        "add_blocked_by": ("add_blocked_by",),
    }
    for field, names in aliases.items():
        name = names[0]
        if name in tool_input:
            updates[field] = tool_input[name]
    return updates


def _task_list_view(task: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    blockers = []
    for blocker in _unique_strings([*task.get("blocked_by", ())]):
        blocker_task = by_id.get(blocker)
        if blocker_task is None or blocker_task.get("status") != "completed":
            blockers.append(blocker)
    return {
        "id": task["id"],
        "subject": task.get("subject", ""),
        "status": task.get("status", "pending"),
        "owner": task.get("owner"),
        "blocked_by": blockers,
    }


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def run_grep(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    case_insensitive: bool = False,
    line_numbers: bool = True,
    head_limit: int = 100,
    offset: int = 0,
    context: ToolContext | None = None,
) -> str:
    ex = _context_executor(context)
    try:
        timeout = getattr(ex, "default_timeout", 120)
        stdout, stderr, rc = ex.grep_files(
            pattern,
            path=path or ".",
            glob_pattern=glob,
            case_insensitive=case_insensitive,
            line_numbers=line_numbers,
            timeout=timeout,
        )
        if rc == 1:
            return "(no matches)"
        if rc != 0:
            return f"Error: {(stderr or stdout).strip() or f'rg exited with {rc}'}"
        lines = (stdout or "").splitlines()
        offset = max(0, int(offset or 0))
        head_limit = max(0, int(head_limit or 100))
        selected = lines[offset : offset + head_limit if head_limit else None]
        out = "\n".join(selected)
        if offset + len(selected) < len(lines):
            out += f"\n... {len(lines) - offset - len(selected)} more matches"
        return out[:30000] if out else "(no matches)"
    except FileNotFoundError as e:
        return f"Error: {e}"
    except subprocess.TimeoutExpired:
        return f"Error: timeout ({getattr(ex, 'default_timeout', 120)}s)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


def run_symbol_search(
    operation: str,
    file_path: str,
    symbol: str | None = None,
    line: int | None = None,
    character: int | None = None,
    include_fallback: bool = True,
) -> ToolResult:
    """Run the lightweight symbol helper through the handlers API.

    Keeping this as a thin lazy wrapper avoids import cycles while preserving the
    existing ``run_*`` surface used by tests and callers.
    """

    from .symbol_search import run_symbol_search as _run_symbol_search

    return _run_symbol_search(
        operation,
        file_path,
        symbol=symbol,
        line=line,
        character=character,
        include_fallback=include_fallback,
    )


def run_read(
    path: str,
    offset: int = 0,
    limit: int | None = None,
    context: ToolContext | None = None,
) -> str:
    ex = _context_executor(context)
    read_state = _context_file_state(context)
    try:
        raw = ex.read_file_raw(path)
        lines = raw.splitlines()
        raw_lines = raw.splitlines(keepends=True)
        total = len(lines)
        offset = max(0, offset)
        end = total if not limit else min(total, offset + limit)
        numbered = [f"{offset + i + 1:6d}\t{ln}" for i, ln in enumerate(lines[offset:end])]
        head = f"# {path}  (\u884c {offset + 1}-{end} / \u5171 {total})\n"
        body = "\n".join(numbered)
        if end < total:
            body += f"\n... \u8fd8\u6709 {total - end} \u884c\uff0c\u7528 offset={end} \u7ee7\u7eed"
        rendered = head + body
        # A full line range can still be partial if the display budget truncates
        # the tool result; stale guard must only unlock content the model saw.
        complete = offset == 0 and end == total and len(rendered) <= _READ_OUTPUT_LIMIT
        visible_content = _fully_visible_read_content(
            raw_lines[offset:end],
            numbered,
            head=head,
        )
        read_state.record_read(
            path,
            raw,
            complete=complete,
            visible_content=visible_content,
            executor=ex,
        )
        return rendered[:_READ_OUTPUT_LIMIT]
    except Exception as e:
        return f"Error: {e}"


def run_write(
    path: str,
    content: str,
    context: ToolContext | None = None,
) -> str:
    ex = _context_executor(context)
    read_state = _context_file_state(context)
    try:
        read_state.assert_can_write(path, executor=ex)
        n = ex.write_file_raw(path, content)
        read_state.record_write(path, content, executor=ex)
        return f"\u5df2\u5199\u5165 {n} \u5b57\u8282 \u2192 {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
    context: ToolContext | None = None,
) -> str:
    ex = _context_executor(context)
    read_state = _context_file_state(context)
    try:
        authorization = read_state.assert_can_edit(
            path,
            old_text=old_text,
            replace_all=replace_all,
            executor=ex,
        )
        text = authorization.content
        updated = (
            text.replace(old_text, new_text)
            if replace_all
            else text.replace(old_text, new_text, 1)
        )
        ex.write_file_raw(path, updated)
        read_state.record_edit(
            path,
            updated,
            record_path=authorization.record_path,
            old_text=old_text,
            new_text=new_text,
            executor=ex,
        )
        suffix = " (replace_all)" if replace_all else ""
        return f"\u5df2\u7f16\u8f91 {path}{suffix}"
    except Exception as e:
        return f"Error: {e}"


def _context_executor(context: ToolContext | None):
    """Return the executor bound to a tool call, falling back to global state."""

    return context.executor if context is not None and context.executor is not None else get_executor()


def _context_file_state(context: ToolContext | None):
    """Return the file-read state scoped to this tool call."""

    if context is not None and context.file_state is not None:
        return context.file_state
    return get_current_file_read_state()


def _fully_visible_read_content(
    source_lines: list[str],
    numbered_lines: list[str],
    *,
    head: str,
) -> str | None:
    """Return source lines whose numbered rendering fits wholly in the budget."""

    position = len(head)
    visible: list[str] = []
    for index, (source_line, numbered_line) in enumerate(
        zip(source_lines, numbered_lines, strict=True)
    ):
        if index:
            position += 1
        position += len(numbered_line)
        if position > _READ_OUTPUT_LIMIT:
            break
        visible.append(source_line)
    return "".join(visible) if visible else None


def run_glob(pattern: str, context: ToolContext | None = None) -> str:
    try:
        matches = _context_executor(context).glob_files(pattern)
        return "\n".join(matches) if matches else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


def reset_todos():
    _TODOS.clear()


def run_update_todos(todos) -> str:
    _TODOS[:] = todos if isinstance(todos, list) else []
    if not _TODOS:
        return "(\u0074odo \u5217\u8868\u4e3a\u7a7a)"
    mark = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}
    out = []
    for t in _TODOS:
        s = t.get("status", "pending") if isinstance(t, dict) else "pending"
        c = t.get("content", str(t)) if isinstance(t, dict) else str(t)
        out.append(f"  {mark.get(s, '[ ]')} {c}")
    return "\u5f53\u524d\u8ba1\u5212:\n" + "\n".join(out)


__all__ = [
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
]
