"""Deterministic MCP per-server lifecycle reliability cases (no LLM)."""

from __future__ import annotations

import json
import tempfile
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from agent.mcp.config import McpServerConfig
from agent.mcp.connection_manager import McpConnectionManager
from agent.mcp.source import StdioMcpToolSource
from agent.mcp.types import McpToolDefinition

PASS = "PASS"
FAIL = "FAIL"
ERROR = "ERROR"

RELIABILITY_CASE_IDS = (
    "mcp_reliability_01_partial_recovery",
    "mcp_reliability_02_config_isolation",
    "mcp_reliability_03_call_failure_recovery",
    "mcp_reliability_04_backoff_cap",
)


@dataclass(frozen=True)
class ReliabilityCaseResult:
    case_id: str
    status: str
    duration_ms: int
    evidence: Mapping[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_record(self, *, commit: str, timestamp: str) -> dict[str, Any]:
        record = {
            "case_id": self.case_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "evidence": dict(self.evidence),
            "commit": commit,
            "timestamp": timestamp,
        }
        if self.message:
            record["message"] = self.message
        return record


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _ScriptedSource:
    def __init__(self, config: McpServerConfig, outcomes) -> None:
        self.configs = (config,)
        self._outcomes = deque(outcomes)
        self._closed = False
        self._revision = 0
        self.list_calls = 0
        self.close_calls = 0
        self._status = {
            config.name: {
                "server_name": config.name,
                "status": "pending",
                "phase": "",
                "tool_count": 0,
                "revision": 0,
            }
        }

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def server_status(self):
        return {name: dict(value) for name, value in self._status.items()}

    def list_tool_definitions(self):
        self.list_calls += 1
        self._revision += 1
        outcome = self._outcomes.popleft() if self._outcomes else ("ready",)
        name = self.configs[0].name
        if outcome[0] == "failed":
            self._status[name] = {
                "server_name": name,
                "status": "failed",
                "phase": outcome[1],
                "tool_count": 0,
                "error_type": "RuntimeError",
                "error": outcome[2],
                "revision": self._revision,
            }
            return ()
        self._status[name] = {
            "server_name": name,
            "status": "ready",
            "phase": "list_tools",
            "tool_count": 1,
            "revision": self._revision,
        }
        return (_definition(name),)

    def close(self) -> None:
        self.close_calls += 1
        self._closed = True


def _definition(server_name: str) -> McpToolDefinition:
    return McpToolDefinition(
        server_name=server_name,
        tool_name=f"{server_name}_tool",
        description="reliability fixture",
        input_schema={"type": "object", "properties": {}},
        call=lambda *_args, **_kwargs: None,
    )


def _write_config(path: Path, servers: Mapping[str, Any]) -> None:
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def _state(manager: McpConnectionManager, name: str) -> dict[str, Any]:
    return asdict(manager.snapshot.server_status[name])


def _verdict(
    case_id: str,
    assertions: Mapping[str, bool],
    *,
    evidence: Mapping[str, Any] | None = None,
) -> ReliabilityCaseResult:
    failed = [name for name, passed in assertions.items() if not passed]
    merged = {"assertions": dict(assertions), **dict(evidence or {})}
    return ReliabilityCaseResult(
        case_id=case_id,
        status=PASS if not failed else FAIL,
        duration_ms=0,
        evidence=merged,
        message="" if not failed else f"failed assertions: {', '.join(failed)}",
    )


def _case_partial_recovery() -> ReliabilityCaseResult:
    case_id = RELIABILITY_CASE_IDS[0]
    clock = _Clock()
    sources: dict[str, _ScriptedSource] = {}

    def factory(config):
        outcomes = [("ready",)] if config.name == "healthy" else [
            ("failed", "initialize", "offline"),
            ("ready",),
        ]
        source = _ScriptedSource(config, outcomes)
        sources[config.name] = source
        return source

    with tempfile.TemporaryDirectory(prefix="mcp_rel_partial_") as raw:
        workdir = Path(raw)
        _write_config(
            workdir / ".mcp.json",
            {"healthy": {"command": "ok"}, "flaky": {"command": "sometimes"}},
        )
        manager = McpConnectionManager(
            source_factory=factory,
            clock=clock,
            retry_base_seconds=2,
        )
        first = manager.acquire(workdir=workdir, enable_mcp=True, mcp_config_path=None)
        immediate = manager.acquire(workdir=workdir, enable_mcp=True, mcp_config_path=None)
        before = _state(manager, "flaky")
        immediate_flaky_calls = sources["flaky"].list_calls
        clock.advance(2)
        recovered = manager.acquire(workdir=workdir, enable_mcp=True, mcp_config_path=None)
        after = _state(manager, "flaky")
        assertions = {
            "first_exposes_only_healthy": [d.server_name for d in first.definitions] == ["healthy"],
            "immediate_is_cache_hit": immediate.cache_hit is True,
            "healthy_listed_once": sources["healthy"].list_calls == 1,
            "healthy_not_closed": sources["healthy"].close_calls == 0,
            "failed_not_retried_early": before["failure_count"] == 1 and immediate_flaky_calls == 1,
            "failed_retried_once_when_due": sources["flaky"].list_calls == 2,
            "recovery_exposes_both": [d.server_name for d in recovered.definitions] == ["flaky", "healthy"],
            "recovery_resets_failure": after["status"] == "ready" and after["failure_count"] == 0,
        }
        evidence = {"before": before, "after": after}
        result = _verdict(case_id, assertions, evidence=evidence)
        manager.close()
        return result


def _case_config_isolation() -> ReliabilityCaseResult:
    case_id = RELIABILITY_CASE_IDS[1]
    created: list[_ScriptedSource] = []

    def factory(config):
        source = _ScriptedSource(config, [("ready",)])
        created.append(source)
        return source

    with tempfile.TemporaryDirectory(prefix="mcp_rel_config_") as raw:
        workdir = Path(raw)
        path = workdir / ".mcp.json"
        _write_config(path, {"a": {"command": "a1"}, "b": {"command": "b1"}})
        manager = McpConnectionManager(source_factory=factory)
        manager.acquire(workdir=workdir, enable_mcp=True, mcp_config_path=None)
        old_a, old_b = created
        _write_config(path, {"a": {"command": "a1"}, "b": {"command": "b2"}})
        lease = manager.acquire(workdir=workdir, enable_mcp=True, mcp_config_path=None)
        new_b = created[2]
        assertions = {
            "only_one_source_recreated": len(created) == 3,
            "a_reused": old_a.list_calls == 1 and old_a.close_calls == 0,
            "old_b_closed": old_b.close_calls == 1,
            "new_b_listed": new_b.configs[0].name == "b" and new_b.list_calls == 1,
            "both_definitions_exposed": len(lease.definitions) == 2,
        }
        result = _verdict(case_id, assertions)
        manager.close()
        return result


class _AsyncContext:
    def __init__(self, value, exits) -> None:
        self.value = value
        self.exits = exits

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        self.exits.append(self.value)
        return False


class _SdkSession:
    def __init__(self, *, fail_call: bool) -> None:
        self.fail_call = fail_call

    async def initialize(self):
        return None

    async def list_tools(self, cursor=None):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="issue",
                    description="fixture",
                    inputSchema={"type": "object", "properties": {}},
                    annotations={"readOnlyHint": True},
                )
            ],
            nextCursor=None,
        )

    async def call_tool(self, name, arguments=None):
        if self.fail_call:
            raise LookupError("transport dropped")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="recovered")],
            structuredContent=None,
            isError=False,
        )


class _MutableSdkFactory:
    def __init__(self, session: _SdkSession) -> None:
        self.session = session
        self.exits: list[Any] = []

    def server_parameters(self, config, *, default_cwd=None):
        return {"server_name": config.name}

    def stdio_client(self, parameters):
        return _AsyncContext(("read", "write"), self.exits)

    def client_session(self, read_stream, write_stream, **kwargs):
        return _AsyncContext(self.session, self.exits)


def _case_call_failure_recovery() -> ReliabilityCaseResult:
    case_id = RELIABILITY_CASE_IDS[2]
    clock = _Clock()
    sdk_factory = _MutableSdkFactory(_SdkSession(fail_call=True))
    source_holder: dict[str, StdioMcpToolSource] = {}

    def source_factory(config):
        source = StdioMcpToolSource((config,), client_factory=sdk_factory)
        source_holder[config.name] = source
        return source

    with tempfile.TemporaryDirectory(prefix="mcp_rel_call_") as raw:
        workdir = Path(raw)
        _write_config(workdir / ".mcp.json", {"api": {"command": "fake"}})
        manager = McpConnectionManager(
            source_factory=source_factory,
            clock=clock,
            retry_base_seconds=3,
        )
        first = manager.acquire(workdir=workdir, enable_mcp=True, mcp_config_path=None)
        source = source_holder["api"]
        ready_revision = source.server_status["api"]["revision"]
        failed_result = source.call_tool("api", "issue", {})
        failed_source_state = dict(source.server_status["api"])
        failed_lease = manager.acquire(workdir=workdir, enable_mcp=True, mcp_config_path=None)
        failed_manager_state = _state(manager, "api")

        sdk_factory.session = _SdkSession(fail_call=False)
        recovered_result = source.call_tool("api", "issue", {})
        manager.acquire(workdir=workdir, enable_mcp=True, mcp_config_path=None)
        recovered_state = _state(manager, "api")
        assertions = {
            "initial_definition_listed": len(first.definitions) == 1,
            "transport_error_result": failed_result.is_error is True,
            "source_records_failed_revision": (
                failed_source_state["status"] == "failed"
                and failed_source_state["phase"] == "call_tool"
                and failed_source_state["revision"] > ready_revision
            ),
            "failed_connection_discarded": sdk_factory.exits.count(("read", "write")) >= 1,
            "manager_retains_definition": len(failed_lease.definitions) == 1,
            "manager_schedules_relist": failed_manager_state["next_retry_at"] == 3.0,
            "explicit_call_can_reconnect": recovered_result.is_error is False,
            "successful_call_resets_manager": (
                recovered_state["status"] == "ready"
                and recovered_state["phase"] == "call_tool"
                and recovered_state["failure_count"] == 0
            ),
        }
        evidence = {
            "failed_source_state": failed_source_state,
            "failed_manager_state": failed_manager_state,
            "recovered_manager_state": recovered_state,
        }
        result = _verdict(case_id, assertions, evidence=evidence)
        manager.close()
        return result


def _case_backoff_cap() -> ReliabilityCaseResult:
    case_id = RELIABILITY_CASE_IDS[3]
    clock = _Clock()
    holder: dict[str, _ScriptedSource] = {}

    def factory(config):
        source = _ScriptedSource(
            config,
            [("failed", "initialize", str(i)) for i in range(4)],
        )
        holder[config.name] = source
        return source

    with tempfile.TemporaryDirectory(prefix="mcp_rel_backoff_") as raw:
        workdir = Path(raw)
        _write_config(workdir / ".mcp.json", {"bad": {"command": "bad"}})
        manager = McpConnectionManager(
            source_factory=factory,
            clock=clock,
            retry_base_seconds=2,
            retry_max_seconds=5,
        )
        deadlines = []
        for advance in (0, 2, 4, 5):
            clock.advance(advance)
            manager.acquire(workdir=workdir, enable_mcp=True, mcp_config_path=None)
            deadlines.append(_state(manager, "bad")["next_retry_at"])
        assertions = {
            "deadlines_double_then_cap": deadlines == [2.0, 6.0, 11.0, 16.0],
            "one_attempt_per_deadline": holder["bad"].list_calls == 4,
        }
        result = _verdict(case_id, assertions, evidence={"deadlines": deadlines})
        manager.close()
        return result


_CASE_RUNNERS: Mapping[str, Callable[[], ReliabilityCaseResult]] = {
    RELIABILITY_CASE_IDS[0]: _case_partial_recovery,
    RELIABILITY_CASE_IDS[1]: _case_config_isolation,
    RELIABILITY_CASE_IDS[2]: _case_call_failure_recovery,
    RELIABILITY_CASE_IDS[3]: _case_backoff_cap,
}


def run_reliability_case(case_id: str) -> ReliabilityCaseResult:
    started = time.perf_counter()
    try:
        result = _CASE_RUNNERS[case_id]()
        return ReliabilityCaseResult(
            case_id=result.case_id,
            status=result.status,
            duration_ms=int((time.perf_counter() - started) * 1000),
            evidence=result.evidence,
            message=result.message,
        )
    except Exception as exc:
        return ReliabilityCaseResult(
            case_id=case_id,
            status=ERROR,
            duration_ms=int((time.perf_counter() - started) * 1000),
            evidence={"assertions": {}, "traceback": traceback.format_exc()},
            message=f"{type(exc).__name__}: {exc}",
        )


def run_reliability_cases(case_ids=None) -> list[ReliabilityCaseResult]:
    selected = tuple(case_ids or RELIABILITY_CASE_IDS)
    return [run_reliability_case(case_id) for case_id in selected]


def summarize_reliability_results(results) -> dict[str, Any]:
    counts = {status: 0 for status in (PASS, FAIL, ERROR)}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return {
        "counts": counts,
        "gate_pass": counts.get(FAIL, 0) == 0 and counts.get(ERROR, 0) == 0,
    }


__all__ = [
    "ERROR",
    "FAIL",
    "PASS",
    "RELIABILITY_CASE_IDS",
    "ReliabilityCaseResult",
    "run_reliability_case",
    "run_reliability_cases",
    "summarize_reliability_results",
]
