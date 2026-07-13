"""Runtime background task registry for local shell tasks."""

from __future__ import annotations

import re
import secrets
import os
import signal
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agent import config
from agent.runtime.observability import (
    content_preview_attrs,
    content_summary_attrs,
    runtime_span,
    safe_set_current_span,
    safe_text_length,
)

TaskStatus = Literal["pending", "running", "completed", "failed", "killed"]
TaskType = Literal["local_bash", "local_powershell"]

TERMINAL_STATUSES = frozenset({"completed", "failed", "killed"})
TASK_OUTPUT_MAX_CHARS = 30_000
NOTIFICATION_TAIL_CHARS = 2_000

_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_GLOBAL_RUN_ID = "global"
_TASKS: dict[str, "BackgroundTask"] = {}
_REGISTRY_LOCK = threading.RLock()


@dataclass
class BackgroundTask:
    id: str
    type: TaskType
    command: str
    cwd: str
    run_id: str
    status: TaskStatus
    start: float
    end: float | None
    exit_code: int | None
    output_file: str
    notified: bool = False
    _process: Any = field(default=None, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "task_id": self.id,
                "type": self.type,
                "status": self.status,
                "command": self.command,
                "cwd": self.cwd,
                "run_id": self.run_id,
                "start": self.start,
                "end": self.end,
                "exit_code": self.exit_code,
                "output_file": self.output_file,
                "notified": self.notified,
            }


def start_background_shell_task(
    command: str,
    *,
    task_type: TaskType,
    run_id: str | None = None,
    cwd: str | Path | None = None,
) -> BackgroundTask:
    """Start a local shell command and register its runtime task state."""

    command = str(command)
    task_run_id = _normalize_run_id(run_id)
    task_id = _new_task_id()
    output_file = _task_output_path(task_run_id, task_id)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.touch(exist_ok=False)

    cwd_path = Path(cwd or config.WORKDIR).resolve()
    task = BackgroundTask(
        id=task_id,
        type=task_type,
        command=command,
        cwd=str(cwd_path),
        run_id=task_run_id,
        status="pending",
        start=time.time(),
        end=None,
        exit_code=None,
        output_file=str(output_file),
    )

    deferred_error: Exception | None = None
    with runtime_span(
        "background_task.start",
        **{
            "background_task.task_id": task_id,
            "background_task.run_id": task_run_id,
            "background_task.run_id_present": bool(task_run_id),
            "background_task.task_type": task_type,
            "background_task.status": "pending",
            **content_summary_attrs("background_task.command", command),
            **content_preview_attrs("background_task.command", command),
            "background_task.command_chars": safe_text_length(command),
        },
    ) as active_span:
        try:
            process = _spawn_process(
                command,
                task_type=task_type,
                cwd=cwd_path,
                output_file=output_file,
            )
            with task._lock:
                task._process = process
                task.status = "running"
            with _REGISTRY_LOCK:
                _TASKS[task_id] = task

            thread = threading.Thread(
                target=_wait_for_task_exit,
                args=(task_id,),
                name=f"background-task-{task_id}",
                daemon=True,
            )
            thread.start()
            safe_set_current_span(**{"background_task.status": "running"})
        except Exception as exc:
            deferred_error = exc
            safe_set_current_span(
                **{
                    "background_task.status": "failed",
                    "background_task.error_type": type(exc).__name__,
                }
            )
            active_span.error(f"background_task_start_error:{type(exc).__name__}")
    if deferred_error is not None:
        raise deferred_error
    return task


def get_task(task_id: str) -> BackgroundTask | None:
    task = _lookup_task(task_id)
    if task is not None:
        _refresh_task(task)
    return task


def read_task_output(
    task_id: str,
    *,
    block: bool = False,
    timeout_ms: int = 0,
    max_chars: int = TASK_OUTPUT_MAX_CHARS,
    mark_notified: bool = False,
) -> dict[str, Any] | None:
    task = get_task(task_id)
    if task is None:
        return None
    if block and timeout_ms > 0:
        task._done.wait(timeout=max(0, timeout_ms) / 1000)
        _refresh_task(task)
    elif block:
        _refresh_task(task)

    snapshot = task.snapshot()
    snapshot["output"] = _read_tail(Path(task.output_file), max_chars=max_chars)
    snapshot["output_truncated"] = len(snapshot["output"]) >= max_chars
    if mark_notified and snapshot["status"] in TERMINAL_STATUSES:
        with task._lock:
            task.notified = True
            snapshot["notified"] = True
        _evict_task(task.id)
    return snapshot


def stop_task(task_id: str) -> dict[str, Any] | None:
    task = get_task(task_id)
    if task is None:
        return None

    with task._lock:
        if task.status not in {"pending", "running"}:
            return task.snapshot()
        process = task._process
        task.status = "killed"
        task.end = time.time()

    _append_output(task.output_file, "\n[task killed]\n")
    if process is not None:
        _terminate_process(process)

    rc = getattr(process, "returncode", None) if process is not None else None
    _finish_task(task, rc, force_status="killed")
    return task.snapshot()


def drain_task_notifications(run_id: str | None = None) -> list[str]:
    """Return terminal task notifications and mark them as notified."""

    selected_run_id = _normalize_run_id(run_id) if run_id is not None else None
    notifications: list[str] = []
    with _REGISTRY_LOCK:
        tasks = list(_TASKS.values())

    for task in tasks:
        if selected_run_id is not None and task.run_id not in {selected_run_id, _GLOBAL_RUN_ID}:
            continue
        _refresh_task(task)
        with task._lock:
            if task.status not in TERMINAL_STATUSES or task.notified:
                continue
            task.notified = True
            snapshot = task.snapshot()
        notification = _format_notification(snapshot)
        with runtime_span(
            "background_task.notification",
            **{
                "background_task.task_id": snapshot["task_id"],
                "background_task.run_id": snapshot["run_id"],
                "background_task.task_type": snapshot["type"],
                "background_task.status": snapshot["status"],
                "background_task.exit_code": snapshot.get("exit_code"),
                **content_summary_attrs("background_task.notification", notification),
                **content_preview_attrs("background_task.notification", notification),
                "background_task.count": 1,
                "background_task.notified": True,
            },
        ):
            notifications.append(notification)
        _evict_task(task.id)
    return notifications


def reset_task_registry(*, kill_running: bool = True) -> None:
    """Clear in-memory tasks, optionally terminating running processes first."""

    with _REGISTRY_LOCK:
        tasks = list(_TASKS.values())
        _TASKS.clear()
    if not kill_running:
        return
    for task in tasks:
        with task._lock:
            process = task._process
            running = task.status in {"pending", "running"}
        if running and process is not None:
            _terminate_process(process)
            _finish_task(task, getattr(process, "returncode", None), force_status="killed")


def format_task_start(task: BackgroundTask) -> str:
    snapshot = task.snapshot()
    return (
        "Started background task\n"
        f"task_id: {snapshot['task_id']}\n"
        f"status: {snapshot['status']}\n"
        f"output_file: {snapshot['output_file']}"
    )


def format_task_output(result: dict[str, Any]) -> str:
    output = result.get("output") or ""
    if not output:
        output = "(no output yet)"
    return (
        f"task_id: {result['task_id']}\n"
        f"status: {result['status']}\n"
        f"exit_code: {_display_exit_code(result.get('exit_code'))}\n"
        f"output_file: {result['output_file']}\n"
        "--- output ---\n"
        f"{output}"
    )


def format_task_stop(result: dict[str, Any]) -> str:
    return (
        f"task_id: {result['task_id']}\n"
        f"status: {result['status']}\n"
        f"exit_code: {_display_exit_code(result.get('exit_code'))}\n"
        f"output_file: {result['output_file']}"
    )


def _spawn_process(
    command: str,
    *,
    task_type: TaskType,
    cwd: Path,
    output_file: Path,
) -> subprocess.Popen:
    stdout = output_file.open("ab", buffering=0)
    try:
        if task_type == "local_bash":
            return subprocess.Popen(
                command,
                shell=True,
                cwd=str(cwd),
                stdout=stdout,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=(os.name != "nt"),
            )

        exe = shutil.which("pwsh") or shutil.which("powershell")
        if not exe:
            raise FileNotFoundError("PowerShell executable not found")
        return subprocess.Popen(
            [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
            cwd=str(cwd),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=(os.name != "nt"),
        )
    finally:
        stdout.close()


def _wait_for_task_exit(task_id: str) -> None:
    task = _lookup_task(task_id)
    if task is None:
        return
    process = task._process
    if process is None:
        return
    try:
        rc = process.wait()
    except Exception:
        rc = getattr(process, "returncode", None)
    _finish_task(task, rc)


def _refresh_task(task: BackgroundTask) -> None:
    process = task._process
    if process is None:
        return
    with task._lock:
        if task.status in TERMINAL_STATUSES:
            return
    rc = process.poll()
    if rc is not None:
        _finish_task(task, rc)


def _finish_task(
    task: BackgroundTask,
    exit_code: int | None,
    *,
    force_status: TaskStatus | None = None,
) -> None:
    with task._lock:
        if task.status in TERMINAL_STATUSES and force_status is None:
            if task.exit_code is None:
                task.exit_code = exit_code
            task._done.set()
            return
        task.exit_code = exit_code
        task.end = task.end or time.time()
        if force_status is not None:
            task.status = force_status
        elif exit_code == 0:
            task.status = "completed"
        else:
            task.status = "failed"
        attrs = _background_task_attrs(task)
    with runtime_span("background_task.finish", **attrs):
        pass
    with task._lock:
        task._done.set()


def _background_task_attrs(task: BackgroundTask) -> dict[str, Any]:
    duration_ms = 0
    if task.end is not None:
        duration_ms = int(max(0.0, task.end - task.start) * 1000)
    return {
        "background_task.task_id": task.id,
        "background_task.run_id": task.run_id,
        "background_task.task_type": task.type,
        "background_task.status": task.status,
        "background_task.exit_code": task.exit_code,
        "background_task.duration_ms": duration_ms,
        "background_task.notified": task.notified,
    }


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
            process.wait(timeout=1)
            return
        except Exception:
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=1)
            return
        except ProcessLookupError:
            return
        except Exception:
            pass
    try:
        process.terminate()
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            return
    except Exception:
        try:
            process.kill()
        except Exception:
            return


def _lookup_task(task_id: str) -> BackgroundTask | None:
    with _REGISTRY_LOCK:
        return _TASKS.get(str(task_id))


def _evict_task(task_id: str) -> None:
    with _REGISTRY_LOCK:
        _TASKS.pop(str(task_id), None)


def _new_task_id() -> str:
    with _REGISTRY_LOCK:
        while True:
            candidate = "b" + _random_base36(8)
            if candidate not in _TASKS:
                return candidate


def _random_base36(length: int) -> str:
    value = secrets.randbelow(36**length)
    chars: list[str] = []
    for _ in range(length):
        value, remainder = divmod(value, 36)
        chars.append(_BASE36_ALPHABET[remainder])
    return "".join(reversed(chars))


def _task_output_path(run_id: str, task_id: str) -> Path:
    return config.TRACES_DIR / ".tool_results" / "tasks" / run_id / f"{task_id}.txt"


def _normalize_run_id(run_id: str | None) -> str:
    text = str(run_id or "").strip() or _GLOBAL_RUN_ID
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return sanitized or _GLOBAL_RUN_ID


def _read_tail(path: Path, *, max_chars: int) -> str:
    max_chars = max(0, int(max_chars))
    if max_chars == 0 or not path.exists():
        return ""
    max_bytes = max_chars * 4
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        data = handle.read()
    text = data.decode("utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _append_output(output_file: str, text: str) -> None:
    try:
        with Path(output_file).open("ab") as handle:
            handle.write(text.encode("utf-8", errors="replace"))
    except Exception:
        return


def _format_notification(snapshot: dict[str, Any]) -> str:
    tail = _read_tail(Path(snapshot["output_file"]), max_chars=NOTIFICATION_TAIL_CHARS)
    if not tail:
        tail = "(no output)"
    return (
        "<background-task-notification>\n"
        f"task_id: {snapshot['task_id']}\n"
        f"status: {snapshot['status']}\n"
        f"exit_code: {_display_exit_code(snapshot.get('exit_code'))}\n"
        f"output_file: {snapshot['output_file']}\n"
        f"summary: {snapshot['type']} task {snapshot['status']}\n"
        "tail:\n"
        f"{tail}\n"
        "</background-task-notification>"
    )


def _display_exit_code(exit_code: Any) -> str:
    return "none" if exit_code is None else str(exit_code)
