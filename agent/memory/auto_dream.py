"""Minimal AutoDream daemon for semantic AutoMemory maintenance.

AutoDream intentionally keeps semantic decisions inside a restricted forked agent.
The deterministic code here only schedules runs, owns lock/state safety, and
repairs mechanical MEMORY.md damage after the fork has had a chance to prune.
"""

from __future__ import annotations

import copy
import errno
import json
import logging
import os
import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from ..tools.executors import bind_memory_file_access, get_executor
from .forked_agent import run_forked_agent
from .governance import MemoryPrunePlan, repair_memory_index

try:  # pragma: no cover - tests exercise behavior, not tracing availability.
    from obs.trace import SpanKind, span
except Exception:  # pragma: no cover
    SpanKind = None

    class span:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "span":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def set(self, **kwargs: Any) -> None:
            return None


STATE_VERSION = 1
_ALLOWED_TOOLS = frozenset({"read_file", "glob", "grep", "write_file", "edit_file"})
_WRITE_TOOLS = frozenset({"write_file", "edit_file"})


@dataclass(frozen=True)
class AutoDreamConfig:
    """Configuration for the first AutoDream slice.

    The defaults are deliberately inert: callers must opt in with enabled=True
    and provide a memory directory so background maintenance never surprises the
    main loop during tests or unrelated tasks.
    """

    enabled: bool = False
    memory_dir: Path | None = None
    interval_minutes: int = 24 * 60
    min_runs_since_last: int = 0
    scan_throttle_minutes: int = 10
    stale_lock_minutes: int = 60
    failure_backoff_minutes: int = 60
    max_turns: int = 5
    max_tokens: int = 4096
    state_file_name: str = ".auto-dream.json"
    lock_file_name: str = ".auto-dream.lock"
    daemon_poll_seconds: float = 600.0

    def memory_root(self, context: "AutoDreamRunContext | None" = None) -> Path | None:
        """Resolve the memory root from config first so scheduler state is stable."""
        raw = self.memory_dir if self.memory_dir is not None else getattr(context, "memory_dir", None)
        return Path(raw).resolve() if raw is not None else None

    def state_path(self, memory_root: Path) -> Path:
        """Return the sidecar state path; it is not a memory topic."""
        return memory_root / self.state_file_name

    def lock_path(self, memory_root: Path) -> Path:
        """Return the mutual-exclusion lock path for this memory root."""
        return memory_root / self.lock_file_name


@dataclass
class AutoDreamState:
    """Persistent scheduler state kept outside memory topics for observability."""

    version: int = STATE_VERSION
    last_started_at: str | None = None
    last_finished_at: str | None = None
    last_success_at: str | None = None
    last_status: str = "never"
    skip_reason: str | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    next_eligible_at: str | None = None
    runs_since_last: int = 0
    last_scan_at: str | None = None
    last_summary: str | None = None
    written_paths: list[str] = field(default_factory=list)
    repair_actions: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "AutoDreamState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls(last_status="failed", last_error="state file could not be read")
        known = set(cls.__dataclass_fields__)
        clean = {key: value for key, value in data.items() if key in known}
        return cls(**clean)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)


@dataclass(frozen=True)
class AutoDreamRunContext:
    """Immutable snapshot consumed by the daemon instead of live loop state."""

    memory_dir: Path
    messages_snapshot: list[dict[str, Any]] = field(default_factory=list)
    system: str = ""
    run_id: str = ""
    logger: logging.Logger | None = None


@dataclass(frozen=True)
class AutoDreamAcquireResult:
    acquired: bool
    owner_token: str | None = None
    reason: str = "locked"
    reclaimed: bool = False


class AutoDreamLock:
    """Owner-token file lock so a failed run cannot delete someone else's lock."""

    def __init__(self, path: str | Path, *, stale_lock_minutes: int = 60) -> None:
        self.path = Path(path)
        self.stale_after = timedelta(minutes=stale_lock_minutes)

    def acquire(self, *, run_id: str) -> AutoDreamAcquireResult:
        """Create the lock atomically and return the owner token needed to release it."""
        owner_token = secrets.token_hex(16)
        payload = {
            "pid": os.getpid(),
            "run_id": run_id,
            "started_at": _iso_now(),
            "owner_token": owner_token,
        }
        result = self._try_create(payload, owner_token, reclaimed=False)
        if result.acquired:
            return result
        if not self._can_reclaim_stale_lock():
            return result
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            return AutoDreamAcquireResult(False, reason="locked")
        return self._try_create(payload, owner_token, reclaimed=True)

    def release(self, owner_token: str) -> None:
        """Release only when the stored owner token still belongs to this run."""
        if not owner_token:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        if data.get("owner_token") != owner_token:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            return

    def _try_create(
        self,
        payload: dict[str, Any],
        owner_token: str,
        *,
        reclaimed: bool,
    ) -> AutoDreamAcquireResult:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(str(self.path), flags)
        except FileExistsError:
            return AutoDreamAcquireResult(False, reason="locked")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True)
        return AutoDreamAcquireResult(
            True,
            owner_token=owner_token,
            reason="stale_reclaimed" if reclaimed else "acquired",
            reclaimed=reclaimed,
        )

    def _can_reclaim_stale_lock(self) -> bool:
        """Reclaim only when a stale lock's PID is clearly no longer alive."""
        if not self._is_stale_by_mtime():
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        pid = data.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        return _pid_is_running(pid) is False

    def _is_stale_by_mtime(self) -> bool:
        try:
            mtime = datetime.fromtimestamp(self.path.stat().st_mtime, UTC)
        except OSError:
            return False
        return _utcnow() - mtime >= self.stale_after


@dataclass(frozen=True)
class _AutoDreamRunResult:
    summary: str
    written_paths: list[str]
    repair_actions: list[str]
    input_tokens: int = 0
    output_tokens: int = 0


class _AutoDreamRunError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        written_paths: list[str] | None = None,
        repair_actions: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.written_paths = written_paths or []
        self.repair_actions = repair_actions or []


class AutoDreamRunner:
    """Run one semantic memory consolidation under memory-only tool permissions."""

    def __init__(
        self,
        config: AutoDreamConfig,
        *,
        fork_runner: Callable[..., Any] | None = None,
        repair_func: Callable[..., MemoryPrunePlan] | None = None,
    ) -> None:
        self.config = config
        self.fork_runner = fork_runner or run_forked_agent
        self.repair_func = repair_func or repair_memory_index

    def run_once(self, context: AutoDreamRunContext) -> _AutoDreamRunResult:
        """Run the semantic fork and always attempt structural repair afterwards."""
        prompt = build_auto_dream_prompt(context.memory_dir)
        tool_filter = self.tool_filter_for(context.memory_dir)
        fork_result: Any | None = None
        fork_error: BaseException | None = None
        repair_actions: list[str] = []
        repair_error: BaseException | None = None

        try:
            memory_executor = bind_memory_file_access(
                get_executor(), context.memory_dir
            )
            fork_result = self.fork_runner(
                prompt,
                copy.deepcopy(context.messages_snapshot),
                system=context.system,
                allowed_tools=_ALLOWED_TOOLS,
                tool_filter=tool_filter,
                max_turns=self.config.max_turns,
                max_tokens=self.config.max_tokens,
                label="auto_dream",
                executor=memory_executor,
            )
        except Exception as exc:  # Fork may have already written memory files.
            fork_error = exc
        finally:
            try:
                plan = self.repair_func(context.memory_dir, dry_run=False, add_orphans=False)
                repair_actions = list(getattr(plan, "actions", []))
            except Exception as exc:
                repair_error = exc

        written_paths = [str(path) for path in getattr(fork_result, "written_paths", [])]
        if fork_error is not None or repair_error is not None:
            parts = []
            if fork_error is not None:
                parts.append(f"fork failed: {fork_error}")
            if repair_error is not None:
                parts.append(f"repair failed: {repair_error}")
            raise _AutoDreamRunError(
                "; ".join(parts),
                written_paths=written_paths,
                repair_actions=repair_actions,
            )

        return _AutoDreamRunResult(
            summary=str(getattr(fork_result, "final_text", "")),
            written_paths=written_paths,
            repair_actions=repair_actions,
            input_tokens=int(getattr(fork_result, "input_tokens", 0) or 0),
            output_tokens=int(getattr(fork_result, "output_tokens", 0) or 0),
        )

    @staticmethod
    def tool_filter_for(memory_dir: str | Path) -> Callable[[str, dict[str, Any]], tuple[bool, str]]:
        memory_root = Path(memory_dir).resolve()

        def tool_filter(name: str, tool_input: dict[str, Any]) -> tuple[bool, str]:
            if name not in _ALLOWED_TOOLS:
                return False, f"tool {name} is not allowed for AutoDream"
            if name in {"read_file", "write_file", "edit_file"}:
                path_value = tool_input.get("path")
                if not path_value:
                    return False, f"tool {name} requires a path inside memory_dir"
                return _allow_path(Path(str(path_value)), memory_root, write=name in _WRITE_TOOLS)
            if name == "grep":
                path_value = tool_input.get("path")
                if not path_value:
                    return False, "AutoDream grep must be scoped to a top-level memory Markdown file"
                ok, deny = _allow_top_level_markdown_path(
                    Path(str(path_value)), memory_root, allow_index=True
                )
                if not ok:
                    return ok, deny
                glob_value = tool_input.get("glob")
                if glob_value:
                    return _allow_grep_glob(str(glob_value))
                return True, ""
            if name == "glob":
                pattern = str(tool_input.get("pattern", ""))
                return _allow_top_level_markdown_pattern(pattern, memory_root)
            return False, f"tool {name} is not allowed for AutoDream"

        return tool_filter


class AutoDreamDaemon:
    """Small in-process scheduler; it never reaches into mutable loop objects."""

    def __init__(
        self,
        config: AutoDreamConfig,
        *,
        context_provider: Callable[[], AutoDreamRunContext | None] | None = None,
        runner: AutoDreamRunner | None = None,
    ) -> None:
        self.config = config
        self.context_provider = context_provider
        self.runner = runner or AutoDreamRunner(config)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_context: AutoDreamRunContext | None = None

    def start(self) -> None:
        """Start the optional polling thread only when a context provider exists."""
        if self._thread is not None or self.context_provider is None:
            return
        self._thread = threading.Thread(target=self._loop, name="auto-dream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to stop without touching lock or state files."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def record_query_completed(self, context: AutoDreamRunContext) -> None:
        """Record one safe-point completion so run-count gates can become eligible."""
        self._last_context = context
        root = self.config.memory_root(context)
        if root is None or not root.exists():
            return
        state_path = self.config.state_path(root)
        state = AutoDreamState.load(state_path)
        state.runs_since_last += 1
        state.save(state_path)

    def tick_once(self, context: AutoDreamRunContext | None = None, *, force: bool = False) -> AutoDreamState:
        """Evaluate gates, acquire the lock, run once, and persist a terminal state."""
        with span("memory.auto_dream", SpanKind.AGENT if SpanKind is not None else None) as sp:
            context = context or (self.context_provider() if self.context_provider else self._last_context)
            root = self.config.memory_root(context)

            if not self.config.enabled and not force:
                return self._skip(root, self._load_state(root), "disabled", sp)
            if context is None:
                return self._skip(root, self._load_state(root), "context_missing", sp)
            if root is None:
                state = AutoDreamState(last_status="skipped", skip_reason="context_missing")
                sp.set(**{"auto_dream.status": "skipped", "auto_dream.skip_reason": "context_missing"})
                return state
            if Path(context.memory_dir).resolve() != root:
                return self._skip(root, self._load_state(root), "context_memory_dir_mismatch", sp)
            if not root.exists():
                state = AutoDreamState(last_status="skipped", skip_reason="memory_dir_missing")
                sp.set(**{"auto_dream.status": "skipped", "auto_dream.skip_reason": "memory_dir_missing"})
                return state

            state_path = self.config.state_path(root)
            state = AutoDreamState.load(state_path)

            if not force:
                skip_reason = self._schedule_skip_reason(state)
                if skip_reason:
                    if skip_reason == "runs_not_due":
                        state.last_scan_at = _iso_now()
                    return self._skip(root, state, skip_reason, sp)

            lock = AutoDreamLock(self.config.lock_path(root), stale_lock_minutes=self.config.stale_lock_minutes)
            acquired = lock.acquire(run_id=context.run_id or secrets.token_hex(8))
            sp.set(**{"auto_dream.lock_result": acquired.reason, "auto_dream.lock_reclaimed": acquired.reclaimed})
            if not acquired.acquired or acquired.owner_token is None:
                return self._skip(root, state, "lock_held", sp)

            state.last_started_at = _iso_now()
            state.last_status = "running"
            state.skip_reason = None
            state.last_error = None
            state.save(state_path)

            try:
                result = self.runner.run_once(context)
            except _AutoDreamRunError as exc:
                state = self._failure(state, str(exc), exc.written_paths, exc.repair_actions)
                sp.set(**{"auto_dream.status": "failed", "auto_dream.error": str(exc)})
            except Exception as exc:
                state = self._failure(state, str(exc), [], [])
                sp.set(**{"auto_dream.status": "failed", "auto_dream.error": str(exc)})
            else:
                state.last_finished_at = _iso_now()
                state.last_success_at = state.last_finished_at
                state.last_status = "completed"
                state.skip_reason = None
                state.last_error = None
                state.consecutive_failures = 0
                state.next_eligible_at = _iso(_utcnow() + timedelta(minutes=self.config.interval_minutes))
                state.runs_since_last = 0
                state.last_summary = result.summary
                state.written_paths = result.written_paths
                state.repair_actions = result.repair_actions
                sp.set(
                    **{
                        "auto_dream.status": "completed",
                        "auto_dream.files_touched_count": len(result.written_paths),
                        "auto_dream.input_tokens": result.input_tokens,
                        "auto_dream.output_tokens": result.output_tokens,
                    }
                )
            finally:
                lock.release(acquired.owner_token)
                state.save(state_path)
            return state

    def _load_state(self, root: Path | None) -> AutoDreamState:
        if root is None or not root.exists():
            return AutoDreamState()
        return AutoDreamState.load(self.config.state_path(root))

    def _loop(self) -> None:
        while not self._stop_event.wait(self.config.daemon_poll_seconds):
            try:
                self.tick_once()
            except Exception:
                logging.getLogger(__name__).exception("AutoDream daemon tick failed")

    def _schedule_skip_reason(self, state: AutoDreamState) -> str | None:
        now = _utcnow()
        next_eligible = _parse_time(state.next_eligible_at)
        if state.consecutive_failures > 0 and next_eligible and next_eligible > now:
            return "failure_backoff"
        last_success = _parse_time(state.last_success_at)
        if last_success and last_success + timedelta(minutes=self.config.interval_minutes) > now:
            return "interval_not_due"
        if state.runs_since_last < self.config.min_runs_since_last:
            last_scan = _parse_time(state.last_scan_at)
            if last_scan and last_scan + timedelta(minutes=self.config.scan_throttle_minutes) > now:
                return "scan_throttled"
            return "runs_not_due"
        return None

    def _skip(
        self,
        root: Path | None,
        state: AutoDreamState,
        reason: str,
        sp: Any | None = None,
    ) -> AutoDreamState:
        state.last_finished_at = _iso_now()
        state.last_status = "skipped"
        state.skip_reason = reason
        if sp is not None:
            sp.set(**{"auto_dream.status": "skipped", "auto_dream.skip_reason": reason})
        if root is not None and root.exists():
            state.save(self.config.state_path(root))
        return state

    def _failure(
        self,
        state: AutoDreamState,
        error: str,
        written_paths: list[str],
        repair_actions: list[str],
    ) -> AutoDreamState:
        state.last_finished_at = _iso_now()
        state.last_status = "failed"
        state.skip_reason = None
        state.last_error = error
        state.consecutive_failures += 1
        state.next_eligible_at = _iso(_utcnow() + timedelta(minutes=self.config.failure_backoff_minutes))
        state.written_paths = written_paths
        state.repair_actions = repair_actions
        return state


def run_auto_dream_once(
    context: AutoDreamRunContext,
    config: AutoDreamConfig,
    *,
    force: bool = False,
) -> AutoDreamState:
    """Convenience entry for tests and one-shot maintenance commands."""
    daemon = AutoDreamDaemon(config)
    return daemon.tick_once(context, force=force)


def maybe_start_auto_dream_daemon(
    config: AutoDreamConfig,
    *,
    context_provider: Callable[[], AutoDreamRunContext | None] | None = None,
) -> AutoDreamDaemon | None:
    """Create an AutoDream daemon only when the feature is explicitly enabled."""
    if not config.enabled:
        return None
    daemon = AutoDreamDaemon(config, context_provider=context_provider)
    daemon.start()
    return daemon


def build_auto_dream_prompt(memory_dir: str | Path) -> str:
    """Build the narrow four-phase memory prompt aligned to Claude Code autoDream."""
    root = Path(memory_dir).resolve()
    return (
        "You are AutoDream, a restricted memory-maintenance forked agent.\n"
        f"Memory root: {root}\n\n"
        "Phase 1 - Orient\n"
        "Read MEMORY.md first if it exists, then inspect only top-level memory topic files under the memory root. "
        "Do not read repository code, docs, traces, workspaces, or a full transcript.\n\n"
        "Phase 2 - Gather recent signal\n"
        "Use only the current memory index and topic files as recent signal in this first version. "
        "Use narrow grep/glob inside the memory root only when a topic points to a specific term.\n\n"
        "Phase 3 - Consolidate\n"
        "Merge duplicate or obviously same-subject topic files when doing so preserves the useful source details. "
        "Prefer updating existing topics over creating near-duplicates.\n\n"
        "Phase 4 - Prune and index\n"
        "You, not deterministic code, decide whether stale, wrong, superseded, or contradictory memory pointers should be removed. "
        "Keep MEMORY.md short, remove links that should no longer guide retrieval, and ensure remaining links point to useful topics. "
        "After you finish, deterministic repair may remove missing/duplicate/unsafe links but will not re-add orphan topics.\n"
    )


def _allow_path(path: Path, memory_root: Path, *, write: bool) -> tuple[bool, str]:
    ok, deny = _allow_top_level_markdown_path(path, memory_root, allow_index=True)
    if not ok:
        return ok, deny
    resolved = path.resolve()
    if write and resolved.name in {".auto-dream.json", ".auto-dream.lock"}:
        return False, "AutoDream fork may not edit daemon state or lock files"
    return True, ""


def _allow_top_level_markdown_path(
    path: Path,
    memory_root: Path,
    *,
    allow_index: bool,
) -> tuple[bool, str]:
    resolved = path.resolve()
    if not _path_stays_within(resolved, memory_root):
        return False, f"AutoDream may only access files inside memory_dir: {memory_root}"
    try:
        rel = resolved.relative_to(memory_root.resolve())
    except ValueError:
        return False, f"AutoDream may only access files inside memory_dir: {memory_root}"
    if len(rel.parts) != 1:
        return False, "AutoDream may only access top-level memory Markdown files"
    if resolved.name == "MEMORY.md" and allow_index:
        return True, ""
    if resolved.name.startswith(".") or resolved.suffix.casefold() != ".md":
        return False, "AutoDream may only access top-level memory Markdown files"
    return True, ""


def _allow_grep_glob(pattern: str) -> tuple[bool, str]:
    normalized = pattern.replace("\\", "/")
    if "/" in normalized or "**" in normalized:
        return False, "AutoDream grep glob must not include directories or recursion"
    if normalized in {"*.md", "MEMORY.md"}:
        return True, ""
    return False, "AutoDream grep glob must target MEMORY.md or top-level *.md"


def _allow_top_level_markdown_pattern(pattern: str, memory_root: Path) -> tuple[bool, str]:
    if not pattern:
        return False, "AutoDream glob must target MEMORY.md or top-level *.md"
    normalized = pattern.replace("\\", "/")
    if "**" in normalized:
        return False, "AutoDream does not allow recursive memory globs"
    root_text = memory_root.resolve().as_posix().rstrip("/")
    if normalized == f"{root_text}/*.md" or normalized == f"{root_text}/MEMORY.md":
        return True, ""
    if "/" in normalized:
        return False, "AutoDream only allows memory root top-level Markdown globs"
    return False, "AutoDream glob must include the memory root path"


def _path_stays_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _pid_is_running(pid: int) -> bool | None:
    """Best-effort local PID liveness check used only for stale lock reclaim."""
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetLastError(0)
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            error = kernel32.GetLastError()
            if error == 87:  # ERROR_INVALID_PARAMETER means the PID is not valid.
                return False
            return None
        except Exception:
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        return None
    return True


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso_now() -> str:
    return _iso(_utcnow())


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = [
    "AutoDreamConfig",
    "AutoDreamState",
    "AutoDreamRunContext",
    "AutoDreamAcquireResult",
    "AutoDreamLock",
    "AutoDreamRunner",
    "AutoDreamDaemon",
    "build_auto_dream_prompt",
    "run_auto_dream_once",
    "maybe_start_auto_dream_daemon",
]
