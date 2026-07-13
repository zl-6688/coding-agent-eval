from __future__ import annotations

import json
import os

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import agent.memory.auto_dream as auto_dream_mod
from agent.memory.auto_dream import (
    AutoDreamConfig,
    AutoDreamDaemon,
    AutoDreamLock,
    AutoDreamRunContext,
    AutoDreamRunner,
    AutoDreamState,
)
from agent.memory.governance import MemoryPrunePlan


def _context(memory_dir: Path) -> AutoDreamRunContext:
    return AutoDreamRunContext(
        memory_dir=memory_dir,
        messages_snapshot=[{"role": "user", "content": "remember the project facts"}],
        system="system",
        run_id="test-run",
    )


def _config(memory_dir: Path, **kwargs) -> AutoDreamConfig:
    return AutoDreamConfig(enabled=True, memory_dir=memory_dir, **kwargs)


def _state_path(memory_dir: Path) -> Path:
    return memory_dir / ".auto-dream.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _future(minutes: int = 60) -> str:
    return (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat()


def _fake_fork(**overrides):
    values = {
        "final_text": "dream done",
        "written_paths": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _fake_repair(actions=None):
    def repair(memory_dir, *, dry_run=True, add_orphans=False):
        assert dry_run is False
        assert add_orphans is False
        return MemoryPrunePlan(issues=[], actions=list(actions or []), dry_run=False, applied=False)

    return repair


def _write_topic(memory_dir: Path, file_name: str, desc: str = "desc") -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / file_name).write_text(
        f"---\nname: {Path(file_name).stem}\ndescription: {desc}\ntype: user\n---\n\nbody\n",
        encoding="utf-8",
    )


def _write_index(memory_dir: Path, *lines: str) -> None:
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "MEMORY.md").write_text("# Memory Index\n" + "".join(lines), encoding="utf-8")


def test_disabled_gate_skips_without_runner(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    called = False

    def fork(*args, **kwargs):
        nonlocal called
        called = True
        return _fake_fork()

    config = AutoDreamConfig(enabled=False, memory_dir=memory_dir)
    runner = AutoDreamRunner(config, fork_runner=fork, repair_func=_fake_repair())
    state = AutoDreamDaemon(config, runner=runner).tick_once(_context(memory_dir))

    assert state.last_status == "skipped"
    assert state.skip_reason == "disabled"
    assert called is False
    assert AutoDreamState.load(_state_path(memory_dir)).skip_reason == "disabled"


def test_context_missing_gate_skips(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    config = _config(memory_dir)

    state = AutoDreamDaemon(config).tick_once(None)

    assert state.last_status == "skipped"
    assert state.skip_reason == "context_missing"


def test_early_gate_trace_records_skip_reason(tmp_path, monkeypatch):
    memory_dir = tmp_path / "missing-memory"
    records = []

    class FakeSpan:
        def __init__(self, *args, **kwargs):
            records.append(("open", args, kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def set(self, **attrs):
            records.append(("set", attrs))

    monkeypatch.setattr(auto_dream_mod, "span", FakeSpan)

    state = AutoDreamDaemon(_config(memory_dir)).tick_once(_context(memory_dir))

    assert state.last_status == "skipped"
    assert state.skip_reason == "memory_dir_missing"
    assert ("set", {"auto_dream.status": "skipped", "auto_dream.skip_reason": "memory_dir_missing"}) in records

def test_memory_dir_missing_gate_skips_without_creating_it(tmp_path):
    memory_dir = tmp_path / "missing-memory"
    config = _config(memory_dir)

    state = AutoDreamDaemon(config).tick_once(_context(memory_dir))

    assert state.last_status == "skipped"
    assert state.skip_reason == "memory_dir_missing"
    assert not memory_dir.exists()


def test_interval_not_due_gate_skips(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    AutoDreamState(last_success_at=_now(), runs_since_last=10).save(_state_path(memory_dir))
    config = _config(memory_dir, interval_minutes=60)

    state = AutoDreamDaemon(config).tick_once(_context(memory_dir))

    assert state.last_status == "skipped"
    assert state.skip_reason == "interval_not_due"


def test_scan_throttle_gate_skips_when_runs_gate_was_recently_checked(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    AutoDreamState(runs_since_last=0, last_scan_at=_now()).save(_state_path(memory_dir))
    config = _config(memory_dir, min_runs_since_last=2, scan_throttle_minutes=10)

    state = AutoDreamDaemon(config).tick_once(_context(memory_dir))

    assert state.last_status == "skipped"
    assert state.skip_reason == "scan_throttled"


def test_failure_backoff_gate_skips(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    AutoDreamState(consecutive_failures=1, next_eligible_at=_future()).save(_state_path(memory_dir))
    config = _config(memory_dir)

    state = AutoDreamDaemon(config).tick_once(_context(memory_dir))

    assert state.last_status == "skipped"
    assert state.skip_reason == "failure_backoff"


def test_locked_skip_does_not_release_other_owner(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    config = _config(memory_dir)
    held = AutoDreamLock(config.lock_path(memory_dir)).acquire(run_id="other")

    state = AutoDreamDaemon(config).tick_once(_context(memory_dir))

    assert state.last_status == "skipped"
    assert state.skip_reason == "lock_held"
    assert config.lock_path(memory_dir).exists()
    AutoDreamLock(config.lock_path(memory_dir)).release(held.owner_token or "")


def test_stale_lock_with_live_current_pid_is_not_reclaimed(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    lock_path = memory_dir / ".auto-dream.lock"
    lock_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "run_id": "still-running",
                "started_at": _now(),
                "owner_token": "live-owner",
            }
        ),
        encoding="utf-8",
    )
    old = (datetime.now(UTC) - timedelta(minutes=30)).timestamp()
    os.utime(lock_path, (old, old))

    result = AutoDreamLock(lock_path, stale_lock_minutes=1).acquire(run_id="new-run")

    assert result.acquired is False
    assert result.reason == "locked"
    assert json.loads(lock_path.read_text(encoding="utf-8"))["owner_token"] == "live-owner"


def test_stale_lock_with_damaged_json_is_not_reclaimed(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    lock_path = memory_dir / ".auto-dream.lock"
    lock_path.write_text("{not-json", encoding="utf-8")
    old = (datetime.now(UTC) - timedelta(minutes=30)).timestamp()
    os.utime(lock_path, (old, old))

    result = AutoDreamLock(lock_path, stale_lock_minutes=1).acquire(run_id="new-run")

    assert result.acquired is False
    assert result.reason == "locked"
    assert lock_path.read_text(encoding="utf-8") == "{not-json"


def test_stale_lock_with_missing_or_invalid_pid_is_not_reclaimed(tmp_path):
    for payload in ({"owner_token": "missing"}, {"pid": "123"}, {"pid": 0}, {"pid": -1}):
        memory_dir = tmp_path / f"memory-{len(str(payload))}"
        memory_dir.mkdir()
        lock_path = memory_dir / ".auto-dream.lock"
        lock_path.write_text(json.dumps(payload), encoding="utf-8")
        old = (datetime.now(UTC) - timedelta(minutes=30)).timestamp()
        os.utime(lock_path, (old, old))

        result = AutoDreamLock(lock_path, stale_lock_minutes=1).acquire(run_id="new-run")

        assert result.acquired is False
        assert result.reason == "locked"
        assert json.loads(lock_path.read_text(encoding="utf-8")) == payload


def test_stale_lock_with_unknown_pid_liveness_is_not_reclaimed(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    lock_path = memory_dir / ".auto-dream.lock"
    lock_path.write_text(json.dumps({"pid": 999999, "owner_token": "unknown"}), encoding="utf-8")
    old = (datetime.now(UTC) - timedelta(minutes=30)).timestamp()
    os.utime(lock_path, (old, old))
    monkeypatch.setattr(auto_dream_mod, "_pid_is_running", lambda pid: None)

    result = AutoDreamLock(lock_path, stale_lock_minutes=1).acquire(run_id="new-run")

    assert result.acquired is False
    assert result.reason == "locked"
    assert json.loads(lock_path.read_text(encoding="utf-8"))["owner_token"] == "unknown"

def test_tool_filter_denies_paths_outside_memory_dir(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    outside = tmp_path / "outside.md"
    inside = memory_dir / "alpha.md"
    nested = memory_dir / "nested" / "beta.md"
    state_file = memory_dir / ".auto-dream.json"
    tool_filter = AutoDreamRunner.tool_filter_for(memory_dir)

    assert tool_filter("write_file", {"path": str(inside)}) == (True, "")
    assert tool_filter("edit_file", {"path": str(outside)})[0] is False
    assert tool_filter("write_file", {"path": str(nested)})[0] is False
    assert tool_filter("read_file", {"path": str(state_file)})[0] is False
    assert tool_filter("read_file", {"path": str(outside)})[0] is False
    assert tool_filter("bash", {"command": "echo nope"})[0] is False


def test_grep_filter_allows_only_top_level_markdown_file_and_safe_glob(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    tool_filter = AutoDreamRunner.tool_filter_for(memory_dir)

    assert tool_filter("grep", {"pattern": "x", "path": str(memory_dir / "alpha.md")}) == (True, "")
    assert tool_filter("grep", {"pattern": "x", "path": str(memory_dir / "MEMORY.md")}) == (True, "")
    assert tool_filter("grep", {"pattern": "x"})[0] is False
    assert tool_filter("grep", {"pattern": "x", "path": str(memory_dir)})[0] is False
    assert tool_filter("grep", {"pattern": "x", "path": str(memory_dir / "nested")})[0] is False
    assert tool_filter("grep", {"pattern": "x", "path": str(memory_dir / "nested" / "beta.md")})[0] is False
    assert tool_filter("grep", {"pattern": "x", "path": str(memory_dir / "alpha.md"), "glob": "*.md"}) == (True, "")
    for bad_glob in ("**/*.md", "sub/*.md", "*/x.md", str(memory_dir / "*.md"), str(memory_dir / "*" / "*.md")):
        assert tool_filter("grep", {"pattern": "x", "path": str(memory_dir / "alpha.md"), "glob": bad_glob})[0] is False


def test_glob_filter_allows_only_memory_root_markdown_shapes(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    tool_filter = AutoDreamRunner.tool_filter_for(memory_dir)

    assert tool_filter("glob", {"pattern": str(memory_dir / "*.md")}) == (True, "")
    assert tool_filter("glob", {"pattern": str(memory_dir / "MEMORY.md")}) == (True, "")
    for bad_pattern in (
        "*.md",
        "MEMORY.md",
        "**/*.md",
        "sub/*.md",
        "*/x.md",
        str(memory_dir / "*" / "*.md"),
        str(memory_dir / "*" / "*" / "*.md"),
    ):
        assert tool_filter("glob", {"pattern": bad_pattern})[0] is False


def test_fork_error_after_write_triggers_repair_and_releases_lock(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    repair_calls = []

    def fork(*args, **kwargs):
        _write_topic(memory_dir, "partial.md")
        raise RuntimeError("fork exploded")

    def repair(memory_dir_arg, *, dry_run=True, add_orphans=False):
        repair_calls.append((Path(memory_dir_arg), dry_run, add_orphans))
        return MemoryPrunePlan(issues=[], actions=["repair ran"], dry_run=False, applied=False)

    config = _config(memory_dir)
    runner = AutoDreamRunner(config, fork_runner=fork, repair_func=repair)
    state = AutoDreamDaemon(config, runner=runner).tick_once(_context(memory_dir))

    assert state.last_status == "failed"
    assert "fork exploded" in (state.last_error or "")
    assert repair_calls == [(memory_dir.resolve(), False, False)]
    assert state.repair_actions == ["repair ran"]
    assert not config.lock_path(memory_dir).exists()


def test_repair_error_releases_lock_and_marks_failure(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    def repair(*args, **kwargs):
        raise RuntimeError("repair exploded")

    config = _config(memory_dir)
    runner = AutoDreamRunner(config, fork_runner=lambda *a, **kw: _fake_fork(), repair_func=repair)
    state = AutoDreamDaemon(config, runner=runner).tick_once(_context(memory_dir))

    assert state.last_status == "failed"
    assert "repair exploded" in (state.last_error or "")
    assert not config.lock_path(memory_dir).exists()


def test_auto_dream_repair_does_not_readd_orphan_topic(tmp_path):
    memory_dir = tmp_path / "memory"
    _write_topic(memory_dir, "alpha.md", "alpha canonical")
    _write_topic(memory_dir, "beta.md", "beta should stay orphan")
    _write_index(memory_dir, "- [alpha](alpha.md) — stale desc\n")

    config = _config(memory_dir)
    runner = AutoDreamRunner(config, fork_runner=lambda *a, **kw: _fake_fork())
    state = AutoDreamDaemon(config, runner=runner).tick_once(_context(memory_dir), force=True)

    index = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert state.last_status == "completed"
    assert "- [alpha](alpha.md) — alpha canonical\n" in index
    assert "beta.md" not in index
    assert not any(action.startswith("add orphan topic index") for action in state.repair_actions)


def test_prompt_uses_four_claude_code_phases_and_memory_only_signal(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    captured = {}

    def fork(prompt, context_messages, **kwargs):
        captured["prompt"] = prompt
        captured["allowed_tools"] = kwargs["allowed_tools"]
        return _fake_fork(final_text="prompt ok")

    config = _config(memory_dir)
    runner = AutoDreamRunner(config, fork_runner=fork, repair_func=_fake_repair())
    AutoDreamDaemon(config, runner=runner).tick_once(_context(memory_dir), force=True)

    prompt = captured["prompt"]
    assert "Phase 1 - Orient" in prompt
    assert "Phase 2 - Gather recent signal" in prompt
    assert "Phase 3 - Consolidate" in prompt
    assert "Phase 4 - Prune and index" in prompt
    assert "stale, wrong, superseded" in prompt
    assert "Do not read repository code, docs, traces, workspaces, or a full transcript" in prompt
    assert "bash" not in captured["allowed_tools"]
