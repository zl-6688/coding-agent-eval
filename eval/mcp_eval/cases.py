"""MCP Phase 1 smoke cases — rule graders only, no LLM."""

from __future__ import annotations

import importlib.util
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from agent.mcp import (
    McpServerConfig,
    McpToolDefinition,
    StdioMcpToolSource,
    create_mcp_tool,
    create_stdio_mcp_tool_source,
    load_mcp_config_file,
)
from agent.mcp.runtime_config import UNSET, resolve_run_task_runtime_kwargs
from agent.runtime.permissions import PermissionEngine, PermissionRule
from agent.tools.contracts import ToolContext
from agent.tools.pool import ToolPoolContext, assemble_tool_pool

from ._fakes import FakeHealthyMcpSession, FakeInitializeFailureSession, PerServerMcpFactory

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "examples" / "mcp" / ".mcp.json"


def load_example_configs() -> tuple[McpServerConfig, ...]:
    """示例配置的 command 是可移植的 "python"；这里钉到当前解释器,保证子进程带 mcp 包。"""
    import sys

    return tuple(
        McpServerConfig(
            config.name,
            {**dict(config.config), "command": sys.executable},
            source=config.source,
        )
        for config in load_mcp_config_file(EXAMPLE_CONFIG)
    )

PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"
ERROR = "ERROR"

REQUIRED_CASE_IDS = (
    "mcp_smoke_01_list_call",
    "mcp_smoke_02_permission_deny",
    "mcp_smoke_03_server_isolation",
    "mcp_smoke_04_deferred_default",
    "mcp_smoke_05_no_deferred_override",
)

BACKLOG_CASE_IDS = (
    "mcp_smoke_06_repl_status_fields",
)


@dataclass(frozen=True)
class CaseResult:
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


def _mcp_installed() -> bool:
    return importlib.util.find_spec("mcp") is not None


def _run_timed(case_id: str, fn: Callable[[], Mapping[str, Any]]) -> CaseResult:
    started = time.perf_counter()
    try:
        evidence = fn()
        duration_ms = int((time.perf_counter() - started) * 1000)
        return CaseResult(case_id=case_id, status=PASS, duration_ms=duration_ms, evidence=evidence)
    except SkipCase as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return CaseResult(
            case_id=case_id,
            status=SKIPPED,
            duration_ms=duration_ms,
            message=str(exc),
        )
    except AssertionError as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return CaseResult(
            case_id=case_id,
            status=FAIL,
            duration_ms=duration_ms,
            message=str(exc) or "assertion failed",
        )
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        return CaseResult(
            case_id=case_id,
            status=ERROR,
            duration_ms=duration_ms,
            message=f"{type(exc).__name__}: {exc}",
            evidence={"traceback": traceback.format_exc()},
        )


class SkipCase(Exception):
    """Case cannot run in this environment (e.g. missing mcp package)."""


def _mcp_tool_definition(server: str, tool: str) -> McpToolDefinition:
    return McpToolDefinition(
        server_name=server,
        tool_name=tool,
        description=f"{server} {tool}",
        input_schema={"type": "object", "properties": {}},
        call=lambda tool_input, context: f"mcp:{server}:{tool}",
    )


def run_mcp_smoke_01_list_call() -> CaseResult:
    def _body() -> Mapping[str, Any]:
        if not _mcp_installed():
            raise SkipCase("mcp package not installed")
        if not EXAMPLE_CONFIG.is_file():
            raise SkipCase(f"example config missing: {EXAMPLE_CONFIG}")

        configs = load_example_configs()
        source = create_stdio_mcp_tool_source(
            configs,
            read_timeout_seconds=10.0,
            operation_timeout_seconds=20.0,
        )
        try:
            definitions = source.list_tool_definitions()
            by_name = {definition.tool_name: definition for definition in definitions}
            if "echo" not in by_name:
                raise AssertionError(f"expected echo tool, got {sorted(by_name)}")
            tool = create_mcp_tool(by_name["echo"])
            result = tool.call({"text": "hello"}, ToolContext(agent_id="mcp_eval"))
        finally:
            source.close()

        if tool.name != "mcp__echo__echo":
            raise AssertionError(f"unexpected tool name: {tool.name}")
        if result.content != "echo:hello":
            raise AssertionError(f"unexpected call result: {result.content!r}")
        return {
            "config_path": str(EXAMPLE_CONFIG),
            "tool_name": tool.name,
            "content": result.content,
            "server_count": len(configs),
        }

    return _run_timed("mcp_smoke_01_list_call", _body)


def run_mcp_smoke_02_permission_deny() -> CaseResult:
    def _body() -> Mapping[str, Any]:
        definitions = (
            _mcp_tool_definition("echo", "echo"),
            _mcp_tool_definition("git", "status"),
        )
        engine = PermissionEngine([PermissionRule("mcp__echo__echo", "deny")])
        pool = assemble_tool_pool(
            ToolPoolContext(
                mcp_tool_definitions=definitions,
                permission_engine=engine,
            )
        )
        tool_names = [tool.name for tool in pool.tools]
        schema_names = [schema["name"] for schema in pool.model_schemas_for_api()]

        if "mcp__echo__echo" in tool_names:
            raise AssertionError("denied MCP tool still present in ToolPool")
        if "mcp__echo__echo" in schema_names:
            raise AssertionError("denied MCP tool still exposed in model schemas")
        if "mcp__git__status" not in tool_names:
            raise AssertionError("non-denied MCP tool missing from ToolPool")
        return {
            "denied_tool": "mcp__echo__echo",
            "remaining_mcp_tools": [name for name in tool_names if name.startswith("mcp__")],
            "schema_count": len(schema_names),
        }

    return _run_timed("mcp_smoke_02_permission_deny", _body)


def run_mcp_smoke_03_server_isolation() -> CaseResult:
    def _body() -> Mapping[str, Any]:
        source = StdioMcpToolSource(
            (
                McpServerConfig("bad", {"command": "fake-server"}),
                McpServerConfig("good", {"command": "fake-server"}),
            ),
            client_factory=PerServerMcpFactory(
                {
                    "bad": FakeInitializeFailureSession(),
                    "good": FakeHealthyMcpSession(),
                }
            ),
        )
        try:
            definitions = source.list_tool_definitions()
            status = dict(source.server_status)
            metadata = dict(source.metadata)
        finally:
            source.close()

        if not definitions:
            raise AssertionError("expected healthy server definitions")
        if definitions[0].server_name != "good":
            raise AssertionError(f"unexpected healthy server: {definitions[0].server_name}")
        if status.get("good", {}).get("status") != "ready":
            raise AssertionError(f"good server status: {status.get('good')}")
        if status.get("bad", {}).get("status") != "failed":
            raise AssertionError(f"bad server status: {status.get('bad')}")
        if "bad" not in set(metadata.get("failed_servers") or ()):
            raise AssertionError(f"failed_servers missing bad: {metadata.get('failed_servers')}")
        return {
            "healthy_servers": [definition.server_name for definition in definitions],
            "server_status": status,
            "failed_servers": list(metadata.get("failed_servers") or ()),
            "tool_count": metadata.get("tool_count"),
        }

    return _run_timed("mcp_smoke_03_server_isolation", _body)


def run_mcp_smoke_04_deferred_default() -> CaseResult:
    def _body() -> Mapping[str, Any]:
        kwargs = resolve_run_task_runtime_kwargs(enable_mcp=True)
        if kwargs.get("enable_deferred_tools") is not True:
            raise AssertionError(f"expected deferred default on, got {kwargs}")
        return dict(kwargs)

    return _run_timed("mcp_smoke_04_deferred_default", _body)


def run_mcp_smoke_05_no_deferred_override() -> CaseResult:
    def _body() -> Mapping[str, Any]:
        kwargs = resolve_run_task_runtime_kwargs(
            enable_mcp=True,
            enable_deferred_tools=False,
        )
        if kwargs.get("enable_deferred_tools") is not False:
            raise AssertionError(f"expected explicit deferred off, got {kwargs}")
        return dict(kwargs)

    return _run_timed("mcp_smoke_05_no_deferred_override", _body)


def run_backlog_case(case_id: str) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        status=SKIPPED,
        duration_ms=0,
        message="backlog: not part of Phase 1 MCP core gate",
    )


CASE_RUNNERS: dict[str, Callable[[], CaseResult]] = {
    "mcp_smoke_01_list_call": run_mcp_smoke_01_list_call,
    "mcp_smoke_02_permission_deny": run_mcp_smoke_02_permission_deny,
    "mcp_smoke_03_server_isolation": run_mcp_smoke_03_server_isolation,
    "mcp_smoke_04_deferred_default": run_mcp_smoke_04_deferred_default,
    "mcp_smoke_05_no_deferred_override": run_mcp_smoke_05_no_deferred_override,
    "mcp_smoke_06_repl_status_fields": lambda: run_backlog_case("mcp_smoke_06_repl_status_fields"),
}

DEFAULT_CASE_IDS = REQUIRED_CASE_IDS


def available_case_ids() -> tuple[str, ...]:
    return tuple(CASE_RUNNERS)


def run_case(case_id: str) -> CaseResult:
    runner = CASE_RUNNERS.get(case_id)
    if runner is None:
        return CaseResult(
            case_id=case_id,
            status=ERROR,
            duration_ms=0,
            message=f"unknown case_id: {case_id}",
        )
    return runner()


def run_cases(case_ids: tuple[str, ...] | None = None) -> list[CaseResult]:
    selected = case_ids or DEFAULT_CASE_IDS
    return [run_case(case_id) for case_id in selected]


def summarize_results(results: list[CaseResult]) -> dict[str, Any]:
    counts = {PASS: 0, FAIL: 0, SKIPPED: 0, ERROR: 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    required = [result for result in results if result.case_id in REQUIRED_CASE_IDS]
    required_fail = [result.case_id for result in required if result.status == FAIL]
    required_error = [result.case_id for result in required if result.status == ERROR]
    gate_pass = not required_fail and not required_error
    return {
        "counts": counts,
        "gate_pass": gate_pass,
        "required_fail": required_fail,
        "required_error": required_error,
    }
