"""Paired MCP coding-benefit probe with a shared deterministic grader."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.mcp.runtime_config import resolve_run_task_runtime_kwargs
from agent.runtime.permissions import PermissionEngine
from eval.mcp_eval.behavior_cases import BehaviorRunArtifacts

REPO_ROOT = Path(__file__).resolve().parents[2]
ISSUE_SERVER = REPO_ROOT / "examples" / "mcp" / "issue_server.py"

PASS = "PASS"
FAIL = "FAIL"
INVALID = "INVALID"
SKIPPED = "SKIPPED"
ERROR = "ERROR"
OK = "OK"

CONTROL = "MCP unavailable"
TREATMENT = "MCP issue context available"
CASE_ID = "mcp_benefit_01_issue_context_patch"

_FIXTURE_SOURCE = '''"""Shipping label helpers."""


def build_shipping_label(order_id: int, region: str) -> str:
    """Build the legacy downstream label."""
    return f"{region}:{order_id}"
'''


@dataclass(frozen=True)
class BenefitGraderResult:
    passed: bool
    checks: Mapping[str, bool]
    error: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "error": self.error,
            "grader": "python:hidden_build_shipping_label_grader",
        }


@dataclass(frozen=True)
class BenefitConditionResult:
    condition: str
    status: str
    duration_ms: int
    hidden_tests_passed: bool = False
    issue_tool_called: bool = False
    issue_call_before_target_write: bool = False
    causal_evidence_source: str = ""
    tool_sequence: tuple[str, ...] = ()
    artifact: Mapping[str, Any] = field(default_factory=dict)
    grader: Mapping[str, Any] = field(default_factory=dict)
    final_text_preview: str = ""
    error: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "hidden_tests_passed": self.hidden_tests_passed,
            "issue_tool_called": self.issue_tool_called,
            "issue_call_before_target_write": self.issue_call_before_target_write,
            "causal_evidence_source": self.causal_evidence_source,
            "tool_sequence": list(self.tool_sequence),
            "artifact": dict(self.artifact),
            "grader": dict(self.grader),
            "final_text_preview": self.final_text_preview,
            "error": self.error,
        }


@dataclass(frozen=True)
class BenefitPairResult:
    case_id: str
    pair_index: int
    nonce: str
    mode: str
    model_id: str
    order: tuple[str, str]
    status: str
    control: BenefitConditionResult
    treatment: BenefitConditionResult
    message: str = ""

    def to_record(self, *, commit: str, timestamp: str) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "pair_index": self.pair_index,
            "nonce": self.nonce,
            "mode": self.mode,
            "model_id": self.model_id,
            "order": list(self.order),
            "status": self.status,
            "message": self.message,
            "control": self.control.to_mapping(),
            "treatment": self.treatment.to_mapping(),
            "commit": commit,
            "timestamp": timestamp,
        }


def create_benefit_fixture(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "orders.py").write_text(_FIXTURE_SOURCE, encoding="utf-8")
    (workspace / "README.md").write_text(
        "# Shipping labels\n\nResolve issue ACE-MCP-001 in `orders.py`.\n",
        encoding="utf-8",
    )


def apply_known_good_patch(workspace: Path, nonce: str) -> None:
    (workspace / "orders.py").write_text(
        '''"""Shipping label helpers."""


def build_shipping_label(order_id: int, region: str) -> str:
    """Build the ACE-MCP-001 downstream label."""
    normalized_region = region.strip().upper()
    if not normalized_region:
        raise ValueError("region must be non-empty")
    return f"ACE-''' + nonce + ''':{normalized_region}:{int(order_id):06d}"
''',
        encoding="utf-8",
    )


def grade_benefit_workspace(workspace: Path, nonce: str) -> BenefitGraderResult:
    target = workspace / "orders.py"
    checks = {
        "target_exists": target.is_file(),
        "primary_format": False,
        "secondary_format": False,
        "empty_region_rejected": False,
    }
    if not target.is_file():
        return BenefitGraderResult(False, checks, "orders.py missing")
    try:
        module_name = f"mcp_benefit_orders_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, target)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load orders.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = module.build_shipping_label
        checks["primary_format"] = fn(42, " eu ") == f"ACE-{nonce}:EU:000042"
        checks["secondary_format"] = fn(7, "us") == f"ACE-{nonce}:US:000007"
        try:
            fn(1, "   ")
        except ValueError:
            checks["empty_region_rejected"] = True
    except Exception as exc:
        return BenefitGraderResult(False, checks, f"{type(exc).__name__}: {exc}")
    return BenefitGraderResult(all(checks.values()), checks)


def grade_benefit_pair(
    control: BenefitConditionResult,
    treatment: BenefitConditionResult,
) -> tuple[str, str]:
    # Infrastructure and declared dependency outcomes take precedence over
    # causal/behavior grading.
    if ERROR in (control.status, treatment.status):
        return ERROR, "condition infrastructure error"
    if SKIPPED in (control.status, treatment.status):
        return SKIPPED, "optional live dependency unavailable"
    if control.hidden_tests_passed:
        return INVALID, "control passed; task was not discriminating"
    if treatment.hidden_tests_passed and not treatment.issue_tool_called:
        return INVALID, "treatment passed without observed MCP issue call"
    if treatment.hidden_tests_passed and not treatment.issue_call_before_target_write:
        return INVALID, "MCP issue call was not observed before target-file write"
    if treatment.hidden_tests_passed:
        return PASS, "observation consistent with MCP benefit"
    return FAIL, "treatment failed hidden acceptance tests"


class _CaptureSink:
    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def emit(self, span) -> None:
        self._events.append(span.to_event())

    def events(self) -> list[dict[str, Any]]:
        return list(self._events)


def _task_prompt() -> str:
    return (
        "Resolve issue ACE-MCP-001 in this repository. Retrieve the issue contract "
        "from the issue tracker tool if that tool is available, inspect orders.py, "
        "implement the contract, and run a focused check. Do not invent an issue "
        "contract when the tracker is unavailable."
    )


def _tool_sequence(messages: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                names.append(str(block.get("name") or ""))
    return tuple(name for name in names if name)


def _causal_order(
    messages: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]] = (),
) -> tuple[bool, bool, str]:
    issue_index: int | None = None
    target_write_index: int | None = None
    index = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                continue
            name = str(block.get("name") or "")
            payload = block.get("input") or {}
            if name == "mcp__issue__get_issue" and issue_index is None:
                issue_index = index
            if name.lower() in {"bash", "powershell", "background_bash", "shell"}:
                if target_write_index is None:
                    # Shells can mutate orders.py without a file-tool event.
                    # Treat any pre-MCP shell as a conservative mutation boundary.
                    target_write_index = index
            if name in {"write_file", "edit_file"} and isinstance(payload, Mapping):
                path = str(payload.get("path") or payload.get("file_path") or "")
                if Path(path).name == "orders.py" and target_write_index is None:
                    target_write_index = index
            index += 1
    if issue_index is not None:
        before = target_write_index is None or issue_index < target_write_index
        return True, before, "messages"

    issue_index = None
    target_write_index = None
    for index, event in enumerate(events):
        if not str(event.get("name") or "").startswith("tool."):
            continue
        attrs = event.get("attributes") or {}
        if not isinstance(attrs, Mapping):
            continue
        name = str(attrs.get("tool.name") or "")
        if name == "mcp__issue__get_issue" and issue_index is None:
            issue_index = index
        if name.lower() in {"bash", "powershell", "background_bash", "shell"}:
            if target_write_index is None:
                target_write_index = index
        if name in {"write_file", "edit_file"} and target_write_index is None:
            # Tools execute sequentially in this runtime. When path display data
            # is unavailable, the fixture's only requested mutation is orders.py,
            # so any first file write is the conservative boundary.
            path = str(attrs.get("tool.display.path") or "")
            if not path or Path(path).name == "orders.py":
                target_write_index = index
    called = issue_index is not None
    before = called and (target_write_index is None or issue_index < target_write_index)
    return called, before, "events" if called else "none"


def _event_tool_sequence(events: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: list[str] = []
    for event in events:
        attrs = event.get("attributes") or {}
        if str(event.get("name") or "").startswith("tool.") and isinstance(attrs, Mapping):
            name = str(attrs.get("tool.name") or "")
            if name:
                names.append(name)
    return tuple(names)


def _artifact(workspace: Path) -> dict[str, Any]:
    target = workspace / "orders.py"
    if not target.is_file():
        return {"exists": False, "sha256": "", "preview": ""}
    raw = target.read_bytes()
    return {
        "exists": True,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "preview": raw.decode("utf-8", errors="replace")[:500],
    }


def _issue_config(path: Path, nonce: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mcpServers": {
            "issue": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(ISSUE_SERVER)],
                "env": {"ACE_MCP_BENEFIT_NONCE": nonce},
                "tools": {
                    "get_issue": {
                        "always_load": True,
                        "search_hint": "retrieve issue acceptance contracts",
                    }
                },
            }
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _run_live_condition(
    *,
    condition: str,
    workspace: Path,
    nonce: str,
    config_path: Path,
    max_turns: int,
) -> BenefitConditionResult:
    from agent import config, loop
    from obs.trace import get_sink, set_sink

    started = time.perf_counter()
    sink = _CaptureSink()
    previous_sink = get_sink()
    set_sink(sink)
    treatment = condition == TREATMENT
    runtime_kwargs = resolve_run_task_runtime_kwargs(
        enable_mcp=True if treatment else False,
        mcp_config_path=str(config_path) if treatment else None,
        disable_mcp=not treatment,
        enable_deferred_tools=True,
        workdir=workspace,
        env={},
    )
    artifacts: BehaviorRunArtifacts
    try:
        try:
            with config.using_workdir(workspace):
                final_text, messages = loop.run_task(
                    _task_prompt(),
                    max_turns=max_turns,
                    trace=False,
                    return_messages=True,
                    permission_engine=PermissionEngine(),
                    eval_hooks=loop.EvalHooks(compact_strategy="none"),
                    **runtime_kwargs,
                )
            artifacts = BehaviorRunArtifacts(
                final_text=str(final_text or ""),
                messages=list(messages or []),
                events=sink.events(),
            )
        except Exception as exc:
            artifacts = BehaviorRunArtifacts(
                final_text="",
                messages=[],
                events=sink.events(),
                error=f"{type(exc).__name__}: {exc}",
            )
    finally:
        set_sink(previous_sink)
    grader = grade_benefit_workspace(workspace, nonce)
    called, before, evidence_source = _causal_order(artifacts.messages, artifacts.events)
    sequence = _tool_sequence(artifacts.messages) or _event_tool_sequence(artifacts.events)
    return BenefitConditionResult(
        condition=condition,
        status=ERROR if artifacts.error else OK,
        duration_ms=int((time.perf_counter() - started) * 1000),
        hidden_tests_passed=grader.passed,
        issue_tool_called=called,
        issue_call_before_target_write=before,
        causal_evidence_source=evidence_source,
        tool_sequence=sequence,
        artifact=_artifact(workspace),
        grader=grader.to_mapping(),
        final_text_preview=artifacts.final_text[:500],
        error=artifacts.error,
    )


def _fake_condition(
    condition: str,
    workspace: Path,
    nonce: str,
) -> BenefitConditionResult:
    started = time.perf_counter()
    create_benefit_fixture(workspace)
    if condition == TREATMENT:
        apply_known_good_patch(workspace, nonce)
        sequence = ("mcp__issue__get_issue", "read_file", "edit_file")
        called = True
        before = True
    else:
        sequence = ("read_file",)
        called = False
        before = False
    grader = grade_benefit_workspace(workspace, nonce)
    return BenefitConditionResult(
        condition=condition,
        status=OK,
        duration_ms=int((time.perf_counter() - started) * 1000),
        hidden_tests_passed=grader.passed,
        issue_tool_called=called,
        issue_call_before_target_write=before,
        causal_evidence_source="synthetic_events",
        tool_sequence=sequence,
        artifact=_artifact(workspace),
        grader=grader.to_mapping(),
        final_text_preview="offline fixture/grader self-test",
    )


def _live_dependencies() -> str:
    if importlib.util.find_spec("mcp") is None:
        return "mcp package not installed"
    from agent import config

    if not config.API_KEY:
        return "ANTHROPIC_API_KEY not configured"
    if not ISSUE_SERVER.is_file():
        return f"missing issue server: {ISSUE_SERVER}"
    return ""


def _skipped_condition(condition: str, message: str) -> BenefitConditionResult:
    return BenefitConditionResult(
        condition=condition,
        status=SKIPPED,
        duration_ms=0,
        error=message,
    )


def run_benefit_pairs(
    *,
    mode: str,
    root: Path | None = None,
    repeat: int = 1,
    max_turns: int = 10,
) -> list[BenefitPairResult]:
    if mode not in {"fake", "live"}:
        raise ValueError(f"unsupported mode: {mode}")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
    root = root or Path(tempfile.mkdtemp(prefix="mcp_benefit_"))
    root.mkdir(parents=True, exist_ok=True)
    dependency_error = _live_dependencies() if mode == "live" else ""
    if mode == "live":
        from agent import config

        model_id = config.MODEL_ID
    else:
        model_id = "none"
    results: list[BenefitPairResult] = []

    for index in range(1, repeat + 1):
        nonce = f"fake{index:04d}" if mode == "fake" else uuid.uuid4().hex[:10]
        pair_root = root / f"pair_{index:03d}"
        control_workspace = pair_root / "control" / "workspace"
        treatment_workspace = pair_root / "treatment" / "workspace"
        config_path = pair_root / "harness" / "issue.mcp.json"
        order = (CONTROL, TREATMENT) if index % 2 else (TREATMENT, CONTROL)

        if dependency_error:
            control = _skipped_condition(CONTROL, dependency_error)
            treatment = _skipped_condition(TREATMENT, dependency_error)
        elif mode == "fake":
            by_condition = {
                CONTROL: lambda: _fake_condition(CONTROL, control_workspace, nonce),
                TREATMENT: lambda: _fake_condition(TREATMENT, treatment_workspace, nonce),
            }
            executed = {condition: by_condition[condition]() for condition in order}
            control = executed[CONTROL]
            treatment = executed[TREATMENT]
        else:
            # Without process-level filesystem isolation, a counterbalanced
            # treatment-first run could leak its nonce/patch into the control.
            # Finish control before creating any current treatment or config.
            order = (CONTROL, TREATMENT)
            create_benefit_fixture(control_workspace)
            control = _run_live_condition(
                condition=CONTROL,
                workspace=control_workspace,
                nonce=nonce,
                config_path=config_path,
                max_turns=max_turns,
            )
            create_benefit_fixture(treatment_workspace)
            _issue_config(config_path, nonce)
            treatment = _run_live_condition(
                condition=TREATMENT,
                workspace=treatment_workspace,
                nonce=nonce,
                config_path=config_path,
                max_turns=max_turns,
            )

        status, message = grade_benefit_pair(control, treatment)
        if mode == "fake" and status == PASS:
            message = "harness fixture/grader self-test passed"
        results.append(
            BenefitPairResult(
                case_id=CASE_ID,
                pair_index=index,
                nonce=nonce,
                mode=mode,
                model_id=model_id,
                order=order,
                status=status,
                control=control,
                treatment=treatment,
                message=message,
            )
        )
    return results


def summarize_benefit_pairs(results: Sequence[BenefitPairResult]) -> dict[str, Any]:
    counts = {status: 0 for status in (PASS, FAIL, INVALID, SKIPPED, ERROR)}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    judged = [result for result in results if result.status != SKIPPED]
    gate_pass = (
        counts[ERROR] == 0
        and counts[FAIL] == 0
        and counts[INVALID] == 0
        and (not judged or all(result.status == PASS for result in judged))
    )
    all_skipped = bool(results) and counts[SKIPPED] == len(results)
    gate_status = SKIPPED if all_skipped else (PASS if gate_pass else FAIL)
    all_judged_pass = bool(judged) and all(result.status == PASS for result in judged)
    if all_judged_pass and all(result.mode == "live" for result in judged):
        claim = "observations_consistent_with_mcp_benefit"
    elif all_judged_pass and all(result.mode == "fake" for result in judged):
        claim = "harness_self_test_only"
    else:
        claim = "no_positive_benefit_claim"
    return {
        "counts": counts,
        "gate_pass": gate_pass,
        "gate_status": gate_status,
        "claim": claim,
    }


__all__ = [
    "CASE_ID",
    "CONTROL",
    "ERROR",
    "FAIL",
    "INVALID",
    "PASS",
    "SKIPPED",
    "TREATMENT",
    "BenefitConditionResult",
    "BenefitGraderResult",
    "BenefitPairResult",
    "apply_known_good_patch",
    "create_benefit_fixture",
    "grade_benefit_pair",
    "grade_benefit_workspace",
    "run_benefit_pairs",
    "summarize_benefit_pairs",
]
