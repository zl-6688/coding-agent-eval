"""File-backed persistent task graph store."""

from __future__ import annotations

import copy
import json
import os
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

from agent import config

TaskGraphStatus = Literal["pending", "in_progress", "completed"]

TASK_GRAPH_VERSION = 1
TASK_GRAPH_STATUSES: frozenset[str] = frozenset(
    {"pending", "in_progress", "completed"}
)
TASK_GRAPH_DELETED_STATUS = "deleted"

_BASE36_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
_STORE_LOCK = threading.RLock()


class TaskGraphError(Exception):
    """Base error for persistent task graph operations."""


class TaskGraphNotFoundError(TaskGraphError):
    """Raised when a task graph operation targets a missing task."""


class TaskGraphValidationError(TaskGraphError):
    """Raised when a task graph payload is invalid."""


def default_task_graph_path() -> Path:
    return config.TRACES_DIR / ".tool_results" / "tasks_graph" / "task_graph.json"


class TaskGraphStore:
    """Small JSON-backed task graph.

    The first implementation intentionally keeps graph tasks separate from
    runtime background tasks. This store tracks planning ownership and blockers;
    it does not know about running processes or task output files.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_task_graph_path()

    def create_task(
        self,
        *,
        subject: str,
        description: str,
        owner: str | None = None,
        cwd: str | None = None,
        worktree: str | None = None,
        dependencies: list[Any] | tuple[Any, ...] | None = None,
        blocks: list[Any] | tuple[Any, ...] | None = None,
        blocked_by: list[Any] | tuple[Any, ...] | None = None,
        evidence: Any = None,
        metadata: Mapping[str, Any] | None = None,
        active_form: str | None = None,
    ) -> dict[str, Any]:
        subject = str(subject).strip()
        if not subject:
            raise TaskGraphValidationError("subject must be non-empty")
        now = _now()

        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            tasks = _normalized_tasks(state.get("tasks"))
            task_ids = {task["id"] for task in tasks}
            task = {
                "id": _new_task_id(task_ids),
                "subject": subject,
                "description": str(description),
                "status": "pending",
                "owner": _optional_string(owner),
                "cwd": _optional_string(cwd),
                "worktree": _optional_string(worktree),
                "dependencies": [],
                "blocks": _string_list(blocks),
                "blocked_by": _blocked_by_alias(blocked_by, dependencies),
                "evidence": _evidence_list(evidence),
                "metadata": _metadata_dict(metadata),
                "created_at": now,
                "updated_at": now,
            }
            if active_form is not None:
                task["active_form"] = str(active_form)
            tasks.append(task)
            _validate_references(tasks)
            _normalize_edges(tasks)
            state["version"] = TASK_GRAPH_VERSION
            state["tasks"] = tasks
            return copy.deepcopy(task)

        return self._mutate(mutate)

    def list_tasks(self) -> list[dict[str, Any]]:
        with _STORE_LOCK:
            state = self._read_state()
            tasks = _normalized_tasks(state.get("tasks"))
            _normalize_edges(tasks)
            return copy.deepcopy(tasks)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        task_id = str(task_id)
        with _STORE_LOCK:
            state = self._read_state()
            tasks = _normalized_tasks(state.get("tasks"))
            _normalize_edges(tasks)
            for task in tasks:
                if task["id"] == task_id:
                    return copy.deepcopy(task)
        return None

    def update_task(self, task_id: str, **updates: Any) -> dict[str, Any]:
        task_id = str(task_id)

        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            tasks = _normalized_tasks(state.get("tasks"))
            task = _require_task(tasks, task_id)
            status = updates.get("status")
            if status == TASK_GRAPH_DELETED_STATUS:
                _delete_task(tasks, task_id)
                _normalize_edges(tasks)
                state["version"] = TASK_GRAPH_VERSION
                state["tasks"] = tasks
                return {"task_id": task_id, "deleted": True}

            changed = False
            for field in (
                "subject",
                "description",
                "owner",
                "cwd",
                "worktree",
                "active_form",
            ):
                if field not in updates:
                    continue
                value = updates[field]
                if field == "subject":
                    value = str(value).strip()
                    if not value:
                        raise TaskGraphValidationError("subject must be non-empty")
                elif value is not None:
                    value = str(value)
                task[field] = value
                changed = True

            if status is not None:
                if status not in TASK_GRAPH_STATUSES:
                    raise TaskGraphValidationError(f"invalid status: {status}")
                task["status"] = str(status)
                changed = True

            if "blocks" in updates:
                _replace_blocks(tasks, task_id, updates["blocks"])
                changed = True
            if "blocked_by" in updates:
                _replace_blocked_by(tasks, task_id, updates["blocked_by"])
                changed = True
            if "dependencies" in updates:
                _replace_blocked_by(tasks, task_id, updates["dependencies"])
                changed = True

            if "add_blocks" in updates:
                _add_blocks(tasks, task_id, updates["add_blocks"])
                changed = True
            if "add_blocked_by" in updates:
                _add_blocked_by(tasks, task_id, updates["add_blocked_by"])
                changed = True
            if "add_dependencies" in updates:
                _add_blocked_by(tasks, task_id, updates["add_dependencies"])
                changed = True

            if "metadata" in updates:
                task["metadata"] = _merge_metadata(task.get("metadata"), updates["metadata"])
                changed = True

            if "evidence" in updates:
                task["evidence"] = [*task.get("evidence", ()), *_evidence_list(updates["evidence"])]
                changed = True

            if changed:
                task["updated_at"] = _now()

            _validate_references(tasks)
            _normalize_edges(tasks)
            state["version"] = TASK_GRAPH_VERSION
            state["tasks"] = tasks
            return copy.deepcopy(task)

        return self._mutate(mutate)

    def claim_task(self, task_id: str, owner: str) -> dict[str, Any]:
        task_id = str(task_id)
        owner = str(owner).strip()
        if not owner:
            raise TaskGraphValidationError("owner must be non-empty")

        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            tasks = _normalized_tasks(state.get("tasks"))
            task = _find_task(tasks, task_id)
            if task is None:
                return {"ok": False, "reason": "task_not_found", "task_id": task_id}
            if task["status"] == "completed":
                return {
                    "ok": False,
                    "reason": "already_resolved",
                    "task_id": task_id,
                    "task": copy.deepcopy(task),
                }
            current_owner = str(task.get("owner") or "").strip()
            if current_owner and current_owner != owner:
                return {
                    "ok": False,
                    "reason": "already_claimed",
                    "task_id": task_id,
                    "owner": current_owner,
                    "task": copy.deepcopy(task),
                }

            _normalize_edges(tasks)
            blockers = _unresolved_blockers(task, tasks)
            if blockers:
                return {
                    "ok": False,
                    "reason": "blocked",
                    "task_id": task_id,
                    "unresolved_blockers": blockers,
                    "task": copy.deepcopy(task),
                }

            task["owner"] = owner
            task["updated_at"] = _now()
            state["version"] = TASK_GRAPH_VERSION
            state["tasks"] = tasks
            return {"ok": True, "task": copy.deepcopy(task)}

        return self._mutate(mutate)

    def complete_task(
        self,
        task_id: str,
        *,
        evidence: Any = None,
        owner: str | None = None,
    ) -> dict[str, Any]:
        task_id = str(task_id)

        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            tasks = _normalized_tasks(state.get("tasks"))
            task = _require_task(tasks, task_id)
            if owner is not None:
                task["owner"] = str(owner)
            task["status"] = "completed"
            task["evidence"] = [*task.get("evidence", ()), *_evidence_list(evidence)]
            task["updated_at"] = _now()
            _normalize_edges(tasks)
            state["version"] = TASK_GRAPH_VERSION
            state["tasks"] = tasks
            return copy.deepcopy(task)

        return self._mutate(mutate)

    def _mutate(self, fn):
        with _STORE_LOCK:
            state = self._read_state()
            result = fn(state)
            self._write_state(state)
            return result

    def _read_state(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": TASK_GRAPH_VERSION, "tasks": []}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TaskGraphValidationError(f"invalid task graph JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise TaskGraphValidationError("task graph root must be an object")
        tasks = raw.get("tasks", [])
        if not isinstance(tasks, list):
            raise TaskGraphValidationError("task graph tasks must be a list")
        return {"version": raw.get("version", TASK_GRAPH_VERSION), "tasks": tasks}

    def _write_state(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": TASK_GRAPH_VERSION,
            "tasks": _normalized_tasks(state.get("tasks")),
        }
        _normalize_edges(payload["tasks"])
        data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        tmp = self.path.with_name(f".{self.path.name}.{secrets.token_hex(8)}.tmp")
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, self.path)


def create_task(**kwargs: Any) -> dict[str, Any]:
    return TaskGraphStore().create_task(**kwargs)


def list_tasks() -> list[dict[str, Any]]:
    return TaskGraphStore().list_tasks()


def get_task(task_id: str) -> dict[str, Any] | None:
    return TaskGraphStore().get_task(task_id)


def update_task(task_id: str, **updates: Any) -> dict[str, Any]:
    return TaskGraphStore().update_task(task_id, **updates)


def claim_task(task_id: str, owner: str) -> dict[str, Any]:
    return TaskGraphStore().claim_task(task_id, owner)


def complete_task(task_id: str, *, evidence: Any = None, owner: str | None = None) -> dict[str, Any]:
    return TaskGraphStore().complete_task(task_id, evidence=evidence, owner=owner)


def _normalized_tasks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    tasks = [_normalize_task(item) for item in value if isinstance(item, Mapping)]
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for task in tasks:
        if task["id"] in seen:
            continue
        seen.add(task["id"])
        unique.append(task)
    return unique


def _normalize_task(value: Mapping[str, Any]) -> dict[str, Any]:
    now = _now()
    task_id = str(value.get("id") or "").strip()
    if not task_id:
        task_id = _new_task_id(set())
    status = str(value.get("status") or "pending")
    if status not in TASK_GRAPH_STATUSES:
        status = "pending"
    task = {
        "id": task_id,
        "subject": str(value.get("subject") or ""),
        "description": str(value.get("description") or ""),
        "status": status,
        "owner": _optional_string(value.get("owner")),
        "cwd": _optional_string(value.get("cwd")),
        "worktree": _optional_string(value.get("worktree")),
        "dependencies": [],
        "blocks": _string_list(value.get("blocks")),
        "blocked_by": _blocked_by_alias(
            value.get("blocked_by") if "blocked_by" in value else None,
            value.get("dependencies"),
        ),
        "evidence": _evidence_list(value.get("evidence")),
        "metadata": _metadata_dict(value.get("metadata")),
        "created_at": str(value.get("created_at") or now),
        "updated_at": str(value.get("updated_at") or value.get("created_at") or now),
    }
    if "active_form" in value:
        task["active_form"] = _optional_string(value.get("active_form"))
    return task


def _validate_references(tasks: list[dict[str, Any]]) -> None:
    ids = {task["id"] for task in tasks}
    for task in tasks:
        task_id = task["id"]
        for field in ("blocks", "blocked_by"):
            for ref in task.get(field, ()):
                if ref == task_id:
                    raise TaskGraphValidationError(f"{field} cannot reference self: {task_id}")
                if ref not in ids:
                    raise TaskGraphValidationError(f"{field} references missing task: {ref}")


def _normalize_edges(tasks: list[dict[str, Any]]) -> None:
    ids = {task["id"] for task in tasks}
    relations: dict[str, set[str]] = {task_id: set() for task_id in ids}

    for task in tasks:
        task_id = task["id"]
        task["blocked_by"] = [
            ref for ref in _string_list(task.get("blocked_by")) if ref in ids and ref != task_id
        ]
        task["blocks"] = [
            ref for ref in _string_list(task.get("blocks")) if ref in ids and ref != task_id
        ]
        for blocker in task["blocked_by"]:
            if blocker in ids and blocker != task_id:
                relations[blocker].add(task_id)
        for target in task["blocks"]:
            if target in ids and target != task_id:
                relations[task_id].add(target)

    for task in tasks:
        task_id = task["id"]
        incoming = sorted(
            blocker for blocker, blocked in relations.items() if task_id in blocked
        )
        task["blocked_by"] = incoming
        task["blocks"] = sorted(relations[task_id])
        task["dependencies"] = list(incoming)


def _replace_blocks(tasks: list[dict[str, Any]], task_id: str, refs: Any) -> None:
    task = _require_task(tasks, task_id)
    next_refs = _string_list(refs)
    for candidate in tasks:
        if candidate["id"] not in next_refs:
            candidate["blocked_by"] = _without_ref(candidate.get("blocked_by"), task_id)
    task["blocks"] = next_refs
    for ref in next_refs:
        target = _find_task(tasks, ref)
        if target is not None:
            target["blocked_by"] = _unique_strings(
                [*target.get("blocked_by", ()), task_id]
            )


def _replace_blocked_by(tasks: list[dict[str, Any]], task_id: str, refs: Any) -> None:
    task = _require_task(tasks, task_id)
    next_refs = _string_list(refs)
    for candidate in tasks:
        if candidate["id"] not in next_refs:
            candidate["blocks"] = _without_ref(candidate.get("blocks"), task_id)
    task["blocked_by"] = next_refs
    for ref in next_refs:
        blocker = _find_task(tasks, ref)
        if blocker is not None:
            blocker["blocks"] = _unique_strings([*blocker.get("blocks", ()), task_id])


def _add_blocks(tasks: list[dict[str, Any]], task_id: str, refs: Any) -> None:
    task = _require_task(tasks, task_id)
    next_refs = _unique_strings([*task.get("blocks", ()), *_string_list(refs)])
    _replace_blocks(tasks, task_id, next_refs)


def _add_blocked_by(tasks: list[dict[str, Any]], task_id: str, refs: Any) -> None:
    task = _require_task(tasks, task_id)
    next_refs = _unique_strings([*task.get("blocked_by", ()), *_string_list(refs)])
    _replace_blocked_by(tasks, task_id, next_refs)


def _delete_task(tasks: list[dict[str, Any]], task_id: str) -> None:
    tasks[:] = [task for task in tasks if task["id"] != task_id]
    for task in tasks:
        for field in ("dependencies", "blocks", "blocked_by"):
            task[field] = [ref for ref in _string_list(task.get(field)) if ref != task_id]


def _find_task(tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any] | None:
    for task in tasks:
        if task["id"] == task_id:
            return task
    return None


def _require_task(tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
    task = _find_task(tasks, task_id)
    if task is None:
        raise TaskGraphNotFoundError(f"task not found: {task_id}")
    return task


def _unresolved_blockers(task: dict[str, Any], tasks: list[dict[str, Any]]) -> list[str]:
    by_id = {candidate["id"]: candidate for candidate in tasks}
    blockers = _string_list(task.get("blocked_by"))
    unresolved: list[str] = []
    for blocker in blockers:
        blocker_task = by_id.get(blocker)
        if blocker_task is None or blocker_task.get("status") != "completed":
            unresolved.append(blocker)
    return unresolved


def _new_task_id(existing: set[str]) -> str:
    while True:
        candidate = "t" + _random_base36(8)
        if candidate not in existing:
            return candidate


def _random_base36(length: int) -> str:
    value = secrets.randbelow(36**length)
    chars: list[str] = []
    for _ in range(length):
        value, remainder = divmod(value, 36)
        chars.append(_BASE36_ALPHABET[remainder])
    return "".join(reversed(chars))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _blocked_by_alias(blocked_by: Any, dependencies: Any) -> list[str]:
    if blocked_by is not None:
        return _string_list(blocked_by)
    return _string_list(dependencies)


def _without_ref(value: Any, ref: str) -> list[str]:
    return [item for item in _string_list(value) if item != ref]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return _unique_strings([value])
    if not isinstance(value, (list, tuple, set, frozenset)):
        return _unique_strings([value])
    return _unique_strings(value)


def _unique_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _metadata_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TaskGraphValidationError("metadata must be an object")
    return copy.deepcopy(dict(value))


def _merge_metadata(current: Any, patch: Any) -> dict[str, Any]:
    if patch is None:
        return {}
    merged = _metadata_dict(current)
    if not isinstance(patch, Mapping):
        raise TaskGraphValidationError("metadata must be an object")
    for key, value in patch.items():
        name = str(key)
        if value is None:
            merged.pop(name, None)
        else:
            merged[name] = copy.deepcopy(value)
    return merged


def _evidence_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return copy.deepcopy(value)
    if isinstance(value, tuple):
        return [copy.deepcopy(item) for item in value]
    return [copy.deepcopy(value)]


__all__ = [
    "TASK_GRAPH_DELETED_STATUS",
    "TASK_GRAPH_STATUSES",
    "TaskGraphError",
    "TaskGraphNotFoundError",
    "TaskGraphStatus",
    "TaskGraphStore",
    "TaskGraphValidationError",
    "claim_task",
    "complete_task",
    "create_task",
    "default_task_graph_path",
    "get_task",
    "list_tasks",
    "update_task",
]
