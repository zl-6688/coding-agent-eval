"""MCP behavior eval cases — live loop grading with optional fake mode."""

from __future__ import annotations

import importlib.util
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent.mcp.runtime_config import resolve_run_task_runtime_kwargs
from agent.runtime.permissions import PermissionEngine, PermissionRule

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mcp" / ".mcp.json"
DEFERRED_CONFIG = REPO_ROOT / "examples" / "mcp" / ".mcp.deferred.json"

PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"
ERROR = "ERROR"
INCONCLUSIVE = "INCONCLUSIVE"

BEHAVIOR_CASE_IDS = (
    "mcp_behavior_01_echo_via_deferred",
    "mcp_behavior_02_permission_deny",
    "mcp_behavior_03_session_deferred_reuse",
)


@dataclass(frozen=True)
class BehaviorRunArtifacts:
    final_text: str
    messages: list[dict[str, Any]]
    events: list[dict[str, Any]]
    error: str = ""


@dataclass(frozen=True)
class BehaviorCaseResult:
    case_id: str
    status: str
    duration_ms: int
    mode: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    message: str = ""
    trial: int | None = None
    repeat: int | None = None

    def to_record(self, *, commit: str, timestamp: str) -> dict[str, Any]:
        record = {
            "case_id": self.case_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "mode": self.mode,
            "evidence": dict(self.evidence),
            "commit": commit,
            "timestamp": timestamp,
        }
        if self.trial is not None:
            record["trial"] = self.trial
        if self.repeat is not None:
            record["repeat"] = self.repeat
        if self.message:
            record["message"] = self.message
        return record


class SkipBehaviorCase(Exception):
    pass


def _mcp_installed() -> bool:
    return importlib.util.find_spec("mcp") is not None


def _api_key_available() -> bool:
    from agent import config

    return bool(config.API_KEY)


class _CaptureSink:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def emit(self, span) -> None:
        self._events.append(span.to_event())

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)


def tool_names_from_events(events: Iterable[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for event in events:
        if not str(event.get("name", "")).startswith("tool."):
            continue
        attrs = event.get("attributes") or {}
        tool_name = attrs.get("tool.name")
        if tool_name:
            names.append(str(tool_name))
    return names


def tool_names_from_messages(messages: Iterable[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                name = block.get("name")
                if name:
                    names.append(str(name))
    return names


def _first_index(names: Sequence[str], target: str) -> int | None:
    try:
        return names.index(target)
    except ValueError:
        return None


def _text_blob(messages: Iterable[Mapping[str, Any]], final_text: str) -> str:
    parts = [final_text or ""]
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, Mapping) and block.get("type") == "tool_result":
                result_content = block.get("content")
                if isinstance(result_content, str):
                    parts.append(result_content)
                elif isinstance(result_content, list):
                    for item in result_content:
                        if isinstance(item, Mapping) and item.get("type") == "text":
                            parts.append(str(item.get("text") or ""))
    return "\n".join(parts)


def grade_echo_via_deferred(artifacts: BehaviorRunArtifacts) -> tuple[str, Mapping[str, Any], str]:
    event_tools = tool_names_from_events(artifacts.events)
    message_tools = tool_names_from_messages(artifacts.messages)
    tool_sequence = event_tools or message_tools
    evidence = {
        "event_tool_names": event_tools,
        "message_tool_names": message_tools,
        "final_text_preview": (artifacts.final_text or "")[:300],
    }
    if artifacts.error:
        return ERROR, evidence, artifacts.error

    search_idx = _first_index(tool_sequence, "ToolSearch")
    echo_idx = _first_index(tool_sequence, "mcp__echo__echo")
    blob = _text_blob(artifacts.messages, artifacts.final_text)
    evidence["tool_sequence"] = tool_sequence
    evidence["contains_echo_hello"] = "echo:hello" in blob

    if echo_idx is None:
        return FAIL, evidence, "mcp__echo__echo was never called"
    if search_idx is None:
        return INCONCLUSIVE, evidence, "ToolSearch not observed; deferred path unclear"
    if search_idx > echo_idx:
        return INCONCLUSIVE, evidence, "ToolSearch appeared after MCP echo call"
    if "echo:hello" not in blob:
        return FAIL, evidence, "expected echo:hello in outputs"
    return PASS, evidence, ""


def grade_permission_deny(artifacts: BehaviorRunArtifacts) -> tuple[str, Mapping[str, Any], str]:
    event_tools = tool_names_from_events(artifacts.events)
    message_tools = tool_names_from_messages(artifacts.messages)
    evidence = {
        "event_tool_names": event_tools,
        "message_tool_names": message_tools,
        "final_text_preview": (artifacts.final_text or "")[:300],
    }
    if artifacts.error:
        return ERROR, evidence, artifacts.error

    if "mcp__echo__echo" in event_tools or "mcp__echo__echo" in message_tools:
        return FAIL, evidence, "denied MCP tool still executed"
    return PASS, evidence, ""


def _run_timed(
    case_id: str,
    mode: str,
    fn: Callable[[], BehaviorCaseResult],
) -> BehaviorCaseResult:
    started = time.perf_counter()
    try:
        result = fn()
        duration_ms = int((time.perf_counter() - started) * 1000)
        if result.duration_ms == 0:
            return BehaviorCaseResult(
                case_id=result.case_id,
                status=result.status,
                duration_ms=duration_ms,
                mode=result.mode,
                evidence=result.evidence,
                message=result.message,
            )
        return result
    except SkipBehaviorCase as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return BehaviorCaseResult(
            case_id=case_id,
            status=SKIPPED,
            duration_ms=duration_ms,
            mode=mode,
            message=str(exc),
        )
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return BehaviorCaseResult(
            case_id=case_id,
            status=ERROR,
            duration_ms=duration_ms,
            mode=mode,
            message=f"{type(exc).__name__}: {exc}",
            evidence={"traceback": traceback.format_exc()},
        )


def run_live_task(
    *,
    task: str,
    workdir: Path,
    mcp_config_path: Path,
    permission_engine: PermissionEngine | None = None,
    max_turns: int = 10,
) -> BehaviorRunArtifacts:
    from agent import config, loop
    from obs.trace import set_sink

    sink = _CaptureSink()
    set_sink(sink)
    runtime_kwargs = resolve_run_task_runtime_kwargs(
        enable_mcp=True,
        mcp_config_path=str(mcp_config_path),
    )
    try:
        with config.using_workdir(workdir):
            final_text, messages = loop.run_task(
                task,
                max_turns=max_turns,
                trace=False,
                return_messages=True,
                permission_engine=permission_engine,
                eval_hooks=loop.EvalHooks(compact_strategy="none"),
                **runtime_kwargs,
            )
    except Exception as exc:
        return BehaviorRunArtifacts(
            final_text="",
            messages=[],
            events=sink.events(),
            error=f"{type(exc).__name__}: {exc}",
        )
    return BehaviorRunArtifacts(
        final_text=str(final_text or ""),
        messages=list(messages or []),
        events=sink.events(),
    )


def _agent_run_attrs(events: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    attrs_list: list[Mapping[str, Any]] = []
    for event in events:
        if event.get("name") != "agent.run":
            continue
        attrs = event.get("attributes") or {}
        if isinstance(attrs, Mapping):
            attrs_list.append(attrs)
    return attrs_list


def run_session_live_tasks(
    *,
    tasks: Sequence[str],
    workdir: Path,
    mcp_config_path: Path,
    permission_engine: PermissionEngine | None = None,
    max_turns: int = 10,
) -> tuple[BehaviorRunArtifacts, Mapping[str, Any]]:
    """Run consecutive REPL-style Session turns through one MCP manager."""
    from agent import config, loop
    from agent.mcp.connection_manager import McpConnectionManager
    from agent.runtime.session import Session
    from obs.trace import set_sink

    sink = _CaptureSink()
    set_sink(sink)
    session = Session.create(workdir, with_memory=False)
    manager = McpConnectionManager()
    final_text = ""
    try:
        with config.using_workdir(workdir):
            for task in tasks:
                final_text = session.run(
                    task,
                    mcp_connection_manager=manager,
                    enable_mcp=True,
                    mcp_config_path=str(mcp_config_path),
                    permission_engine=permission_engine,
                    max_turns=max_turns,
                    eval_hooks=loop.EvalHooks(compact_strategy="none"),
                )
    except Exception as exc:
        return (
            BehaviorRunArtifacts(
                final_text="",
                messages=list(session.messages or []),
                events=sink.events(),
                error=f"{type(exc).__name__}: {exc}",
            ),
            {},
        )
    finally:
        manager.close()
    events = sink.events()
    run_attrs = _agent_run_attrs(events)
    cache_evidence = {
        "agent_run_count": len(run_attrs),
        "cache_hits": [bool(attrs.get("mcp.cache_hit")) for attrs in run_attrs],
        "borrowed": [bool(attrs.get("mcp.borrowed")) for attrs in run_attrs],
        "second_run_cache_hit": bool(run_attrs[1].get("mcp.cache_hit")) if len(run_attrs) >= 2 else False,
    }
    return (
        BehaviorRunArtifacts(
            final_text=str(final_text or ""),
            messages=list(session.messages or []),
            events=events,
        ),
        cache_evidence,
    )


def grade_session_deferred_reuse(
    artifacts: BehaviorRunArtifacts,
    cache_evidence: Mapping[str, Any],
) -> tuple[str, dict[str, Any], str]:
    status, evidence, message = grade_echo_via_deferred(artifacts)
    evidence = dict(evidence)
    evidence["session_cache"] = dict(cache_evidence)
    if status != PASS:
        return status, evidence, message
    if not cache_evidence.get("second_run_cache_hit"):
        return (
            FAIL,
            evidence,
            "Session second run did not report mcp.cache_hit=true",
        )
    return PASS, evidence, ""


def _behavior_01_task() -> str:
    return (
        "请使用 MCP echo 工具把文本 hello 原样回显。"
        "如果 echo 工具的 schema 还没加载，先用 ToolSearch 执行 "
        "`select:mcp__echo__echo` 选中它，再调用 MCP echo。"
        "最终回复里必须包含 `echo:hello`。"
    )


def run_behavior_01_live(workdir: Path) -> BehaviorCaseResult:
    if not _mcp_installed():
        raise SkipBehaviorCase("mcp package not installed")
    if not _api_key_available():
        raise SkipBehaviorCase("ANTHROPIC_API_KEY not configured")
    if not DEFERRED_CONFIG.is_file():
        raise SkipBehaviorCase(f"missing config: {DEFERRED_CONFIG}")

    artifacts = run_live_task(
        task=_behavior_01_task(),
        workdir=workdir,
        mcp_config_path=DEFERRED_CONFIG,
    )
    status, evidence, message = grade_echo_via_deferred(artifacts)
    return BehaviorCaseResult(
        case_id="mcp_behavior_01_echo_via_deferred",
        status=status,
        duration_ms=0,
        mode="live",
        evidence=evidence,
        message=message,
    )


def run_behavior_02_live(workdir: Path) -> BehaviorCaseResult:
    if not _mcp_installed():
        raise SkipBehaviorCase("mcp package not installed")
    if not _api_key_available():
        raise SkipBehaviorCase("ANTHROPIC_API_KEY not configured")
    if not EXAMPLE_CONFIG.is_file():
        raise SkipBehaviorCase(f"missing config: {EXAMPLE_CONFIG}")

    engine = PermissionEngine([PermissionRule("mcp__echo__echo", "deny")])
    artifacts = run_live_task(
        task=_behavior_01_task(),
        workdir=workdir,
        mcp_config_path=EXAMPLE_CONFIG,
        permission_engine=engine,
    )
    status, evidence, message = grade_permission_deny(artifacts)
    return BehaviorCaseResult(
        case_id="mcp_behavior_02_permission_deny",
        status=status,
        duration_ms=0,
        mode="live",
        evidence=evidence,
        message=message,
    )


def run_behavior_01_fake() -> BehaviorCaseResult:
    artifacts = BehaviorRunArtifacts(
        final_text="done echo:hello",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "ToolSearch", "input": {"query": "select:mcp__echo__echo"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "Selected deferred tools for the next request:"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "mcp__echo__echo", "input": {"text": "hello"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": [{"type": "text", "text": "echo:hello"}]},
                ],
            },
        ],
        events=[
            {"name": "tool.ToolSearch", "attributes": {"tool.name": "ToolSearch"}},
            {"name": "tool.mcp__echo__echo", "attributes": {"tool.name": "mcp__echo__echo"}},
        ],
    )
    status, evidence, message = grade_echo_via_deferred(artifacts)
    return BehaviorCaseResult(
        case_id="mcp_behavior_01_echo_via_deferred",
        status=status,
        duration_ms=0,
        mode="fake",
        evidence=evidence,
        message=message,
    )


def run_behavior_02_fake() -> BehaviorCaseResult:
    artifacts = BehaviorRunArtifacts(
        final_text="I could not call the denied MCP tool.",
        messages=[
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "MCP echo is unavailable."}],
            }
        ],
        events=[{"name": "tool.read_file", "attributes": {"tool.name": "read_file"}}],
    )
    status, evidence, message = grade_permission_deny(artifacts)
    return BehaviorCaseResult(
        case_id="mcp_behavior_02_permission_deny",
        status=status,
        duration_ms=0,
        mode="fake",
        evidence=evidence,
        message=message,
    )


def run_behavior_03_live(workdir: Path) -> BehaviorCaseResult:
    if not _mcp_installed():
        raise SkipBehaviorCase("mcp package not installed")
    if not _api_key_available():
        raise SkipBehaviorCase("ANTHROPIC_API_KEY not configured")
    if not DEFERRED_CONFIG.is_file():
        raise SkipBehaviorCase(f"missing config: {DEFERRED_CONFIG}")

    import os

    ace_home = workdir / ".ace"
    ace_home.mkdir(parents=True, exist_ok=True)
    previous_ace = os.environ.get("ACE_HOME")
    os.environ["ACE_HOME"] = str(ace_home)
    try:
        artifacts, cache_evidence = run_session_live_tasks(
            tasks=(_behavior_01_task(), _behavior_01_task()),
            workdir=workdir,
            mcp_config_path=DEFERRED_CONFIG,
        )
    finally:
        if previous_ace is None:
            os.environ.pop("ACE_HOME", None)
        else:
            os.environ["ACE_HOME"] = previous_ace
    status, evidence, message = grade_session_deferred_reuse(artifacts, cache_evidence)
    return BehaviorCaseResult(
        case_id="mcp_behavior_03_session_deferred_reuse",
        status=status,
        duration_ms=0,
        mode="live",
        evidence=evidence,
        message=message,
    )


def run_behavior_03_fake() -> BehaviorCaseResult:
    artifacts = BehaviorRunArtifacts(
        final_text="echo:hello",
        messages=[],
        events=[
            {"name": "agent.run", "attributes": {"mcp.cache_hit": False, "mcp.borrowed": True}},
            {"name": "tool.ToolSearch", "attributes": {"tool.name": "ToolSearch"}},
            {"name": "tool.mcp__echo__echo", "attributes": {"tool.name": "mcp__echo__echo"}},
            {"name": "agent.run", "attributes": {"mcp.cache_hit": True, "mcp.borrowed": True}},
            {"name": "tool.mcp__echo__echo", "attributes": {"tool.name": "mcp__echo__echo"}},
        ],
    )
    cache_evidence = {
        "agent_run_count": 2,
        "cache_hits": [False, True],
        "borrowed": [True, True],
        "second_run_cache_hit": True,
    }
    status, evidence, message = grade_session_deferred_reuse(artifacts, cache_evidence)
    return BehaviorCaseResult(
        case_id="mcp_behavior_03_session_deferred_reuse",
        status=status,
        duration_ms=0,
        mode="fake",
        evidence=evidence,
        message=message,
    )


LIVE_RUNNERS: dict[str, Callable[[Path], BehaviorCaseResult]] = {
    "mcp_behavior_01_echo_via_deferred": run_behavior_01_live,
    "mcp_behavior_02_permission_deny": run_behavior_02_live,
    "mcp_behavior_03_session_deferred_reuse": run_behavior_03_live,
}

FAKE_RUNNERS: dict[str, Callable[[], BehaviorCaseResult]] = {
    "mcp_behavior_01_echo_via_deferred": run_behavior_01_fake,
    "mcp_behavior_02_permission_deny": run_behavior_02_fake,
    "mcp_behavior_03_session_deferred_reuse": run_behavior_03_fake,
}


def run_behavior_case(case_id: str, *, mode: str, workdir: Path) -> BehaviorCaseResult:
    if mode == "fake":
        runner = FAKE_RUNNERS.get(case_id)
        if runner is None:
            return BehaviorCaseResult(
                case_id=case_id,
                status=ERROR,
                duration_ms=0,
                mode=mode,
                message=f"unknown fake case: {case_id}",
            )
        return _run_timed(case_id, mode, runner)
    if mode == "live":
        runner = LIVE_RUNNERS.get(case_id)
        if runner is None:
            return BehaviorCaseResult(
                case_id=case_id,
                status=ERROR,
                duration_ms=0,
                mode=mode,
                message=f"unknown live case: {case_id}",
            )
        return _run_timed(case_id, mode, lambda: runner(workdir))
    return BehaviorCaseResult(
        case_id=case_id,
        status=ERROR,
        duration_ms=0,
        mode=mode,
        message=f"unknown mode: {mode}",
    )


def run_behavior_cases(
    case_ids: Sequence[str] | None = None,
    *,
    mode: str,
    workdir: Path,
    repeat: int = 1,
) -> list[BehaviorCaseResult]:
    selected = tuple(case_ids or BEHAVIOR_CASE_IDS)
    trials = max(1, int(repeat))
    results: list[BehaviorCaseResult] = []
    for case_id in selected:
        for trial in range(trials):
            result = run_behavior_case(case_id, mode=mode, workdir=workdir)
            if trial > 0 or trials > 1:
                result = BehaviorCaseResult(
                    case_id=result.case_id,
                    status=result.status,
                    duration_ms=result.duration_ms,
                    mode=result.mode,
                    evidence=result.evidence,
                    message=result.message,
                    trial=trial,
                    repeat=trials,
                )
            results.append(result)
    return results


def summarize_behavior_trials(results: list[BehaviorCaseResult]) -> dict[str, dict[str, Any]]:
    """Per-case pass counts across repeat trials."""
    grouped: dict[str, list[str]] = {}
    for result in results:
        grouped.setdefault(result.case_id, []).append(result.status)
    summary: dict[str, dict[str, Any]] = {}
    for case_id, statuses in grouped.items():
        pass_count = statuses.count(PASS)
        summary[case_id] = {
            "trials": len(statuses),
            "pass_count": pass_count,
            "fail_count": statuses.count(FAIL),
            "inconclusive_count": statuses.count(INCONCLUSIVE),
            "error_count": statuses.count(ERROR),
            "skipped_count": statuses.count(SKIPPED),
            "all_pass": pass_count == len(statuses) and len(statuses) > 0,
            "statuses": statuses,
        }
    return summary


def summarize_behavior_results(results: list[BehaviorCaseResult]) -> dict[str, Any]:
    counts = {PASS: 0, FAIL: 0, SKIPPED: 0, ERROR: 0, INCONCLUSIVE: 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    blocking = [result.case_id for result in results if result.status in {FAIL, ERROR}]
    inconclusive = [result.case_id for result in results if result.status == INCONCLUSIVE]
    gate_pass = not blocking
    return {
        "counts": counts,
        "gate_pass": gate_pass,
        "blocking": blocking,
        "inconclusive": inconclusive,
        "per_case": summarize_behavior_trials(results),
    }
