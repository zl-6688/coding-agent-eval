#!/usr/bin/env python3
"""Aggregate offline checks into one provenance-rich release-gate report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Sequence

try:
    from scripts.regression_gate import (
        EXIT_INFRA,
        EXIT_OK,
        EXIT_REGRESSION,
        GateResult,
        InstanceResult,
        SweSummary,
        aggregate_swe_results,
        run_mcp_reliability,
        run_mcp_smoke,
        run_offline,
        run_swe_check,
        suite_ids_sha256,
        write_swe_baseline,
    )
except ModuleNotFoundError:  # Direct ``python scripts/release_gate.py`` execution.
    from regression_gate import (  # type: ignore[no-redef]
        EXIT_INFRA,
        EXIT_OK,
        EXIT_REGRESSION,
        GateResult,
        InstanceResult,
        SweSummary,
        aggregate_swe_results,
        run_mcp_reliability,
        run_mcp_smoke,
        run_offline,
        run_swe_check,
        suite_ids_sha256,
        write_swe_baseline,
    )


SCHEMA_VERSION = "ace.offline-gate.v1"

PASS = "PASS"
FAIL = "FAIL"
INVALID = "INVALID"
INCONCLUSIVE = "INCONCLUSIVE"
ERROR = "ERROR"
JUDGE_ERROR = "JUDGE_ERROR"
GRADER_ERROR = "GRADER_ERROR"
SKIPPED = "SKIPPED"

VALID_STATUSES = frozenset(
    {
        PASS,
        FAIL,
        INVALID,
        INCONCLUSIVE,
        ERROR,
        JUDGE_ERROR,
        GRADER_ERROR,
        SKIPPED,
    }
)

REQUIRED_CHECK_IDS = (
    "unit_tests",
    "context_budget",
    "mcp_smoke",
    "mcp_reliability",
    "mcp_benefit_synthetic",
    "swe_checker_selftest",
)

CLAIM_SCOPES = {
    "unit_tests": "deterministic_runtime_regression",
    "context_budget": "deterministic_context_budget_invariants",
    "mcp_smoke": "mcp_mechanism_smoke",
    "mcp_reliability": "mcp_lifecycle_reliability",
    "mcp_benefit_synthetic": "synthetic_harness_self_test",
    "swe_checker_selftest": "checker_contract_regression",
}

WHAT_THIS_DOES_NOT_PROVE = (
    "It does not measure end-to-end quality with a live language model.",
    "It does not report an official SWE-bench resolved rate.",
    "Synthetic MCP and checker self-tests do not prove product benefit.",
    "It does not replace production monitoring or cross-platform CI.",
)

MINIMUM_UNIT_TESTS = 300
REQUIRED_UNIT_SENTINELS = (
    (
        "tests.test_loop",
        "test_run_task_single_tool_then_end",
    ),
    (
        "tests.test_public_release_policy",
        "test_text_sources_do_not_contain_machine_specific_release_paths",
    ),
    (
        "tests.test_context_eval",
        "test_required_matrix_and_protocol_fingerprint_are_complete_and_stable",
    ),
    (
        "tests.test_mcp_stdio_smoke",
        "test_example_stdio_mcp_server_lists_and_calls_tool",
    ),
    (
        "tests.test_swebench_checker",
        "test_fixture_selftest_passes_without_becoming_gate_evidence_or_agent_score",
    ),
)

AGGREGATE_PROTOCOL_DESCRIPTOR: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "aggregate_claim": "offline_mechanism_regression_gate",
    "required_checks": [
        {"check_id": check_id, "claim_scope": CLAIM_SCOPES[check_id]}
        for check_id in REQUIRED_CHECK_IDS
    ],
    "command_plan": {
        "unit_tests": [
            "python",
            "-m",
            "pytest",
            "-q",
            "--junitxml",
            "<RUN>/unit-tests.xml",
        ],
        "context_budget": ["python", "-m", "eval.context_eval.run"],
        "mcp_smoke": ["python", "-m", "eval.mcp_eval.smoke"],
        "mcp_reliability": ["python", "-m", "eval.mcp_eval.reliability"],
        "mcp_benefit_synthetic": [
            "python",
            "-m",
            "eval.mcp_eval.benefit",
            "--mode",
            "fake",
            "--repeat",
            "2",
        ],
        "swe_checker_selftest": [
            "python",
            "-m",
            "eval.swebench.checker",
            "--allow-fixture-baseline",
        ],
    },
    "evidence_inspection": {
        "all_tracks": [
            "evidence code_version must equal the requested gate version",
            "required ids and actual records must have exact unique coverage",
            "status, counts, metrics, and gate flags are recomputed",
            "current protocol fingerprints and implementation bytes are verified",
            "the final report hash must refer to the inspected bytes",
        ],
        "unit_tests": {
            "junit_rule": (
                "counts and testcase records must agree; testcase IDs are non-empty and "
                "unique; skips remain explicit"
            ),
            "minimum_collected_tests": MINIMUM_UNIT_TESTS,
            "required_executed_sentinels": [list(item) for item in REQUIRED_UNIT_SENTINELS],
            "normalization": (
                "machine-local repository, home, temp, hostname, pytest user, and pytest "
                "run metadata plus synthetic credential-shaped values are redacted before "
                "inspection and hashing"
            ),
        },
        "context_budget": "current deterministic context protocol and case records",
        "mcp": "current descriptor/source hashes plus identical run metadata per record",
        "swe_checker_selftest": "current synthetic fixture hashes and recomputed outcomes",
    },
    "code_version_provenance": {
        "auto": "clean Git HEAD becomes its unique 12-character revision; otherwise WORKTREE",
        "explicit_worktree": "WORKTREE is permitted as a provisional label",
        "explicit_revision": "any other label must resolve to the current clean Git HEAD",
    },
    "aggregation": {
        "status_precedence": [ERROR, JUDGE_ERROR, GRADER_ERROR, FAIL, INCONCLUSIVE, PASS],
        "pass_rule": "every fixed required check is present, fully covered, and PASS",
        "partial_rule": "missing, skipped, invalid, inconclusive, or partial evidence cannot PASS",
        "unknown_required_rule": "unknown required checks are rejected",
        "process_exit_mapping": {"0": "inspect evidence", "1": "non-pass", "other": ERROR},
    },
    "synthetic_checker_semantics": {
        "required_as": "checker_contract_regression",
        "benchmark_evidence_eligible": False,
        "agent_score_contribution": False,
    },
    "status_vocabulary": sorted(VALID_STATUSES),
    "what_this_does_not_prove": list(WHAT_THIS_DOES_NOT_PROVE),
}


@dataclass(frozen=True)
class GateCheck:
    """One required or optional check consumed by the aggregate gate."""

    check_id: str
    status: str
    full_coverage: bool
    claim_scope: str
    command: tuple[str, ...] = ()
    artifact_path: Path | None = None
    inspected_sha256: str | None = None
    message: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        if not self.check_id.strip():
            raise ValueError("check_id must be non-empty")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"unsupported gate status: {self.status!r}")
        if not self.claim_scope.strip():
            raise ValueError("claim_scope must be non-empty")
        object.__setattr__(self, "command", tuple(str(part) for part in self.command))
        if self.artifact_path is not None:
            object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        if self.inspected_sha256 is not None:
            _validated_sha256(self.inspected_sha256, "inspected_sha256")


@dataclass(frozen=True)
class EvidenceVerdict:
    """Strict inspection result for one track-owned evidence artifact."""

    status: str
    full_coverage: bool
    message: str
    artifact_sha256: str = ""
    artifact_bytes: int = 0


@dataclass(frozen=True)
class OfflineCommandSpec:
    """One deterministic command in the offline release-gate plan."""

    check_id: str
    command: tuple[str, ...]
    artifact_path: Path | None = None

    def __post_init__(self) -> None:
        if self.check_id not in REQUIRED_CHECK_IDS:
            raise ValueError(f"unsupported offline check: {self.check_id}")
        if not self.command:
            raise ValueError("offline command must not be empty")
        object.__setattr__(self, "command", tuple(str(part) for part in self.command))
        if self.artifact_path is not None:
            object.__setattr__(self, "artifact_path", Path(self.artifact_path))


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess result used by the orchestrator and its tests."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[[OfflineCommandSpec, Path], CommandResult]


def inspect_track_evidence(
    check_id: str,
    artifact_path: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str] | None = None,
    expected_code_version: str | None = None,
) -> EvidenceVerdict:
    """Validate a track artifact instead of trusting its filename or exit code."""

    path = Path(artifact_path)
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    try:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        if check_id == "context_budget":
            verdict = _inspect_context_evidence(text, expected_code_version)
        elif check_id in {
            "mcp_smoke",
            "mcp_reliability",
            "mcp_benefit_synthetic",
        }:
            verdict = _inspect_mcp_evidence(
                check_id,
                text,
                repo_root=root,
                expected_code_version=expected_code_version,
            )
        elif check_id == "swe_checker_selftest":
            verdict = _inspect_swe_checker_evidence(
                text,
                repo_root=root,
                expected_code_version=expected_code_version,
            )
        else:
            verdict = _evidence_error(f"unsupported evidence track: {check_id}")
        return EvidenceVerdict(
            status=verdict.status,
            full_coverage=verdict.full_coverage,
            message=verdict.message,
            artifact_sha256=digest,
            artifact_bytes=len(raw),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return _evidence_error(f"malformed {check_id} evidence: {exc}")


def _inspect_pytest_junit(path: Path) -> EvidenceVerdict:
    try:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        root = ET.fromstring(raw)
        suites = list(root.iter("testsuite"))
        if not suites:
            raise ValueError("JUnit contains no testsuite")
        totals = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
        testcase_count = 0
        testcase_keys: list[tuple[str, str]] = []
        executed_keys: set[tuple[str, str]] = set()
        child_counts = {name: 0 for name in ("failure", "error", "skipped")}
        for suite in suites:
            for name in totals:
                value = int(suite.attrib.get(name, "0"))
                if value < 0:
                    raise ValueError(f"negative JUnit {name}")
                totals[name] += value
            cases = list(suite.findall("testcase"))
            testcase_count += len(cases)
            for case in cases:
                key = (case.attrib.get("classname", ""), case.attrib.get("name", ""))
                testcase_keys.append(key)
                if case.find("skipped") is None:
                    executed_keys.add(key)
                for name in child_counts:
                    child_counts[name] += len(case.findall(name))
        if totals["tests"] <= 0 or totals["tests"] != testcase_count:
            raise ValueError("JUnit test count does not match testcase records")
        if totals["failures"] != child_counts["failure"]:
            raise ValueError("JUnit failure count does not match testcase records")
        if totals["errors"] != child_counts["error"]:
            raise ValueError("JUnit error count does not match testcase records")
        if totals["skipped"] != child_counts["skipped"]:
            raise ValueError("JUnit skipped count does not match testcase records")
        executed = totals["tests"] - totals["skipped"]
        missing_sentinels = sorted(set(REQUIRED_UNIT_SENTINELS) - executed_keys)
        sufficient_collection = totals["tests"] >= MINIMUM_UNIT_TESTS
        duplicate_testcase_ids = len(testcase_keys) - len(set(testcase_keys))
        valid_testcase_ids = all(classname and name for classname, name in testcase_keys)
        full_coverage = (
            executed > 0
            and sufficient_collection
            and not missing_sentinels
            and duplicate_testcase_ids == 0
            and valid_testcase_ids
        )
        if totals["errors"]:
            status = ERROR
        elif totals["failures"]:
            status = FAIL
        elif not full_coverage:
            status = INCONCLUSIVE
        else:
            status = PASS
        message = (
            f"tests={totals['tests']} failures={totals['failures']} "
            f"errors={totals['errors']} skipped={totals['skipped']} executed={executed} "
            f"missing_sentinels={len(missing_sentinels)} "
            f"duplicate_testcase_ids={duplicate_testcase_ids}"
        )
        return EvidenceVerdict(
            status=status,
            full_coverage=full_coverage,
            message=message,
            artifact_sha256=digest,
            artifact_bytes=len(raw),
        )
    except (OSError, ET.ParseError, TypeError, ValueError) as exc:
        return _evidence_error(f"malformed pytest JUnit evidence: {exc}")


def offline_command_specs(
    *,
    repo_root: str | os.PathLike[str],
    evidence_dir: str | os.PathLike[str],
    code_version: str,
) -> tuple[OfflineCommandSpec, ...]:
    """Return the fixed, complete command plan for the offline gate."""

    root = Path(repo_root).expanduser().resolve()
    output = Path(evidence_dir).expanduser().resolve()
    version = _validated_code_version(code_version)
    python = sys.executable
    fixture = root / "eval" / "swebench" / "checker_fixture"
    return (
        OfflineCommandSpec(
            check_id="unit_tests",
            command=(
                python,
                "-m",
                "pytest",
                "-q",
                "--junitxml",
                str(output / "unit-tests.xml"),
            ),
            artifact_path=output / "unit-tests.xml",
        ),
        OfflineCommandSpec(
            check_id="context_budget",
            command=(
                python,
                "-m",
                "eval.context_eval.run",
                "--output",
                str(output / "context-budget.jsonl"),
                "--code-version",
                version,
            ),
            artifact_path=output / "context-budget.jsonl",
        ),
        OfflineCommandSpec(
            check_id="mcp_smoke",
            command=(
                python,
                "-m",
                "eval.mcp_eval.smoke",
                "--output",
                str(output / "mcp-smoke.jsonl"),
                "--code-version",
                version,
            ),
            artifact_path=output / "mcp-smoke.jsonl",
        ),
        OfflineCommandSpec(
            check_id="mcp_reliability",
            command=(
                python,
                "-m",
                "eval.mcp_eval.reliability",
                "--output",
                str(output / "mcp-reliability.jsonl"),
                "--code-version",
                version,
            ),
            artifact_path=output / "mcp-reliability.jsonl",
        ),
        OfflineCommandSpec(
            check_id="mcp_benefit_synthetic",
            command=(
                python,
                "-m",
                "eval.mcp_eval.benefit",
                "--mode",
                "fake",
                "--repeat",
                "2",
                "--output",
                str(output / "mcp-benefit-synthetic.jsonl"),
                "--code-version",
                version,
            ),
            artifact_path=output / "mcp-benefit-synthetic.jsonl",
        ),
        OfflineCommandSpec(
            check_id="swe_checker_selftest",
            command=(
                python,
                "-m",
                "eval.swebench.checker",
                "--suite",
                str(fixture / "suite.json"),
                "--results",
                str(fixture / "results.jsonl"),
                "--baseline",
                str(fixture / "baseline.fixture.json"),
                "--output",
                str(output / "swe-checker-selftest.json"),
                "--code-version",
                version,
                "--allow-fixture-baseline",
            ),
            artifact_path=output / "swe-checker-selftest.json",
        ),
    )


def run_offline_gate(
    *,
    repo_root: str | os.PathLike[str],
    evidence_dir: str | os.PathLike[str],
    code_version: str,
    runner: CommandRunner | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Run every required offline check, inspect evidence, and write one report."""

    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = Path(evidence_dir).expanduser().resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError("evidence_dir must be inside repo_root") from exc

    specs = offline_command_specs(
        repo_root=root,
        evidence_dir=output,
        code_version=code_version,
    )
    report_path = output / "offline-gate.json"
    if output.exists():
        names = ", ".join(sorted(path.name for path in output.iterdir())) or "empty directory"
        raise FileExistsError(
            f"refusing to reuse a prior round directory; choose a new evidence directory ({names})"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)

    execute = runner or _subprocess_runner
    checks: list[GateCheck] = []
    for spec in specs:
        try:
            result = execute(spec, root)
        except Exception as exc:  # noqa: BLE001 - a failed runner is gate evidence
            checks.append(
                GateCheck(
                    check_id=spec.check_id,
                    status=ERROR,
                    full_coverage=False,
                    claim_scope=CLAIM_SCOPES[spec.check_id],
                    command=spec.command,
                    artifact_path=(
                        spec.artifact_path
                        if spec.artifact_path is not None and spec.artifact_path.is_file()
                        else None
                    ),
                    message=f"runner error: {type(exc).__name__}: {exc}",
                )
            )
            continue

        if spec.check_id == "unit_tests":
            artifact = spec.artifact_path
            if artifact is None or not artifact.is_file():
                checks.append(
                    GateCheck(
                        check_id=spec.check_id,
                        status=ERROR,
                        full_coverage=False,
                        claim_scope=CLAIM_SCOPES[spec.check_id],
                        command=spec.command,
                        message=f"pytest exit code {result.returncode}; JUnit artifact missing",
                    )
                )
                continue
            try:
                _sanitize_junit_artifact(artifact, root)
            except (OSError, ET.ParseError, ValueError) as exc:
                checks.append(
                    GateCheck(
                        check_id=spec.check_id,
                        status=ERROR,
                        full_coverage=False,
                        claim_scope=CLAIM_SCOPES[spec.check_id],
                        command=spec.command,
                        artifact_path=artifact,
                        message=f"JUnit sanitization error: {type(exc).__name__}: {exc}",
                    )
                )
                continue
            verdict = _inspect_pytest_junit(artifact)
            if result.returncode == 0 and verdict.status == PASS:
                status = PASS
                full_coverage = verdict.full_coverage
            elif result.returncode == 0 and verdict.status == INCONCLUSIVE:
                status = INCONCLUSIVE
                full_coverage = False
            elif result.returncode == 1 and verdict.status == FAIL:
                status = FAIL
                full_coverage = verdict.full_coverage
            else:
                status = ERROR
                full_coverage = False
            checks.append(
                GateCheck(
                    check_id=spec.check_id,
                    status=status,
                    full_coverage=full_coverage,
                    claim_scope=CLAIM_SCOPES[spec.check_id],
                    command=spec.command,
                    artifact_path=artifact,
                    inspected_sha256=verdict.artifact_sha256 or None,
                    message=f"pytest exit code {result.returncode}; {verdict.message}",
                )
            )
            continue

        artifact = spec.artifact_path
        if artifact is None or not artifact.is_file():
            checks.append(
                GateCheck(
                    check_id=spec.check_id,
                    status=ERROR,
                    full_coverage=False,
                    claim_scope=CLAIM_SCOPES[spec.check_id],
                    command=spec.command,
                    message=f"process exit {result.returncode}; evidence artifact missing",
                )
            )
            continue

        verdict = inspect_track_evidence(
            spec.check_id,
            artifact,
            repo_root=root,
            expected_code_version=code_version,
        )
        status = verdict.status
        full_coverage = verdict.full_coverage
        message = f"process exit {result.returncode}; {verdict.message}"
        if result.returncode not in {0, 1}:
            status = ERROR
            full_coverage = False
            message = f"process exit {result.returncode}; command did not complete normally"
        elif result.returncode == 1 and verdict.status == PASS:
            status = ERROR
            full_coverage = False
            message = "process exit 1 contradicts PASS evidence"
        checks.append(
            GateCheck(
                check_id=spec.check_id,
                status=status,
                full_coverage=full_coverage,
                claim_scope=CLAIM_SCOPES[spec.check_id],
                command=spec.command,
                artifact_path=artifact,
                inspected_sha256=verdict.artifact_sha256 or None,
                message=message,
            )
        )

    finalized: list[GateCheck] = []
    for check in checks:
        if check.artifact_path is None or check.inspected_sha256 is None:
            finalized.append(check)
            continue
        if check.check_id == "unit_tests":
            verdict = _inspect_pytest_junit(check.artifact_path)
        else:
            verdict = inspect_track_evidence(
                check.check_id,
                check.artifact_path,
                repo_root=root,
                expected_code_version=code_version,
            )
        if (
            verdict.artifact_sha256 != check.inspected_sha256
            or verdict.status != check.status
            or verdict.full_coverage != check.full_coverage
        ):
            finalized.append(
                GateCheck(
                    check_id=check.check_id,
                    status=ERROR,
                    full_coverage=False,
                    claim_scope=check.claim_scope,
                    command=check.command,
                    artifact_path=check.artifact_path,
                    inspected_sha256=verdict.artifact_sha256 or None,
                    message=(
                        "artifact changed after inspection or no longer matches its "
                        f"verdict: {verdict.message}"
                    ),
                    required=check.required,
                )
            )
        else:
            finalized.append(check)
    checks = finalized

    report = build_gate_report(
        checks,
        repo_root=root,
        code_version=code_version,
        generated_at=generated_at,
    )
    write_report(report_path, report)
    return report


def build_gate_report(
    checks: Sequence[GateCheck],
    *,
    repo_root: str | os.PathLike[str],
    code_version: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a portable report without trusting filenames as evidence."""

    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    by_id: dict[str, GateCheck] = {}
    for check in checks:
        if check.check_id in by_id:
            raise ValueError(f"duplicate check_id: {check.check_id}")
        by_id[check.check_id] = check

    unexpected_required = sorted(
        check.check_id
        for check in checks
        if check.required and check.check_id not in REQUIRED_CHECK_IDS
    )
    if unexpected_required:
        raise ValueError(
            f"unexpected required checks: {', '.join(unexpected_required)}"
        )
    for check_id in REQUIRED_CHECK_IDS:
        check = by_id.get(check_id)
        if check is not None and (
            not check.required or check.claim_scope != CLAIM_SCOPES[check_id]
        ):
            raise ValueError(f"invalid required check contract: {check_id}")

    missing = [check_id for check_id in REQUIRED_CHECK_IDS if check_id not in by_id]
    required_checks = [by_id[check_id] for check_id in REQUIRED_CHECK_IDS if check_id in by_id]
    full_coverage = not missing and all(check.full_coverage for check in required_checks)
    status = _aggregate_status(required_checks, missing=missing, full_coverage=full_coverage)
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")

    serialized = [_serialize_check(check, root) for check in checks]
    counts = Counter(check.status for check in checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_fingerprint": protocol_fingerprint(),
        "generated_at": timestamp.astimezone(timezone.utc).isoformat(),
        "code_version": str(code_version),
        "environment": {
            "python": platform.python_version(),
            "system": platform.system() or "unknown",
            "machine": platform.machine() or "unknown",
        },
        "claim": "offline_mechanism_regression_gate",
        "status": status,
        "gate_pass": status == PASS and full_coverage,
        "full_gate_coverage": full_coverage,
        "required_check_ids": list(REQUIRED_CHECK_IDS),
        "missing_required_checks": missing,
        "counts": {name: counts.get(name, 0) for name in sorted(VALID_STATUSES)},
        "checks": serialized,
        "what_this_does_not_prove": list(WHAT_THIS_DOES_NOT_PROVE),
    }


def protocol_fingerprint() -> str:
    """Hash the stable aggregate protocol, independent of run results."""

    return _fingerprint_protocol_descriptor(AGGREGATE_PROTOCOL_DESCRIPTOR)


def _fingerprint_protocol_descriptor(descriptor: dict[str, Any]) -> str:
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_CONTEXT_SUMMARY_FIELDS = {
    "code_version",
    "counts",
    "environment",
    "excluded_case_count",
    "excluded_statuses",
    "full_gate_coverage",
    "gate_pass",
    "gate_status",
    "protocol_sha256",
    "protocol_version",
    "record_type",
    "required_case_ids",
    "result_case_ids",
    "schema_version",
    "selected_case_ids",
    "selected_result_coverage",
    "timestamp_utc",
    "valid_case_count",
    "valid_statuses",
    "what_this_does_not_prove",
}
_CONTEXT_CASE_FIELDS = {
    "case_id",
    "code_version",
    "description",
    "evidence",
    "message",
    "protocol_sha256",
    "protocol_version",
    "record_type",
    "required",
    "schema_version",
    "status",
    "timestamp_utc",
    "what_this_does_not_prove",
}


def _inspect_context_evidence(
    text: str,
    expected_code_version: str | None,
) -> EvidenceVerdict:
    from eval.context_eval.cases import (
        PROTOCOL_VERSION as CURRENT_PROTOCOL_VERSION,
        REQUIRED_CASE_IDS as CURRENT_REQUIRED_CASE_IDS,
        STATUS_VOCABULARY as CONTEXT_STATUSES,
        protocol_fingerprint as current_protocol_fingerprint,
    )

    records = _read_jsonl_text(text)
    summary = records[0]
    if set(summary) != _CONTEXT_SUMMARY_FIELDS:
        return _evidence_error("context summary fields do not match the evidence schema")
    if summary.get("record_type") != "run_summary":
        return _evidence_error("context evidence must begin with run_summary")
    if summary.get("schema_version") != "context-budget-evidence-v1":
        return _evidence_error("unexpected context evidence schema")
    if summary.get("protocol_version") != CURRENT_PROTOCOL_VERSION:
        return _evidence_error("context protocol version does not match current code")
    timestamp = _validated_utc_timestamp(summary.get("timestamp_utc"), "context timestamp")
    _validated_environment(summary.get("environment"), "context environment")
    if summary.get("valid_statuses") != [PASS, FAIL]:
        return _evidence_error("context valid_statuses declaration is inconsistent")
    if summary.get("excluded_statuses") != [INVALID, INCONCLUSIVE, ERROR]:
        return _evidence_error("context excluded_statuses declaration is inconsistent")

    required = _unique_string_list(summary.get("required_case_ids"), "required_case_ids")
    if required != list(CURRENT_REQUIRED_CASE_IDS):
        return _evidence_error("context required ids do not match the current protocol")
    selected = _unique_string_list(summary.get("selected_case_ids"), "selected_case_ids")
    declared_results = _unique_string_list(
        summary.get("result_case_ids"), "result_case_ids"
    )
    case_records = records[1:]
    if any(record.get("record_type") != "case_result" for record in case_records):
        return _evidence_error("context evidence contains a non-case result record")
    actual_results = _unique_string_list(
        [record.get("case_id") for record in case_records], "case result ids"
    )
    exact_coverage = (
        bool(required)
        and selected == required
        and declared_results == required
        and actual_results == required
    )
    declared_coverage = (
        summary.get("full_gate_coverage") is True
        and summary.get("selected_result_coverage") is True
    )
    if not exact_coverage or not declared_coverage:
        return _evidence_error(
            "inconsistent context coverage: summary and case records must exactly cover required ids"
        )

    status = _validated_status(summary.get("gate_status"), "context gate_status")
    gate_pass = summary.get("gate_pass")
    if not isinstance(gate_pass, bool) or gate_pass != (status == PASS):
        return _evidence_error("inconsistent context gate status and gate_pass")
    protocol_sha = _validated_sha256(
        summary.get("protocol_sha256"), "context protocol_sha256"
    )
    if protocol_sha != current_protocol_fingerprint():
        return _evidence_error("context protocol fingerprint does not match current code")
    code_version = _validated_code_version(summary.get("code_version"))
    if expected_code_version is not None and code_version != _validated_code_version(
        expected_code_version
    ):
        return _evidence_error("context code_version does not match requested gate version")
    _validated_boundary(summary.get("what_this_does_not_prove"))

    case_statuses: list[str] = []
    for record in case_records:
        if set(record) != _CONTEXT_CASE_FIELDS:
            return _evidence_error("context case fields do not match the evidence schema")
        if record.get("schema_version") != summary["schema_version"]:
            return _evidence_error("inconsistent context record schema")
        if record.get("protocol_version") != CURRENT_PROTOCOL_VERSION:
            return _evidence_error("inconsistent context protocol version")
        if record.get("protocol_sha256") != protocol_sha:
            return _evidence_error("inconsistent context protocol fingerprint")
        if record.get("code_version") != code_version:
            return _evidence_error("inconsistent context code_version")
        if record.get("timestamp_utc") != timestamp:
            return _evidence_error("inconsistent context timestamp")
        if record.get("required") is not True:
            return _evidence_error("required context cases must declare required=true")
        _validated_boundary(record.get("what_this_does_not_prove"))
        if not isinstance(record.get("description"), str) or not isinstance(
            record.get("message"), str
        ) or not isinstance(record.get("evidence"), dict):
            return _evidence_error("invalid context case payload types")
        case_statuses.append(_validated_status(record.get("status"), "context case status"))

    actual_counts = Counter(case_statuses)
    expected_counts = {name: actual_counts[name] for name in CONTEXT_STATUSES}
    if summary.get("counts") != expected_counts:
        return _evidence_error("inconsistent context status counts")
    valid_count = actual_counts[PASS] + actual_counts[FAIL]
    excluded_count = sum(
        actual_counts[name] for name in (INVALID, INCONCLUSIVE, ERROR)
    )
    if summary.get("valid_case_count") != valid_count:
        return _evidence_error("inconsistent context valid_case_count")
    if summary.get("excluded_case_count") != excluded_count:
        return _evidence_error("inconsistent context excluded_case_count")
    if actual_counts[ERROR]:
        expected_status = ERROR
    elif actual_counts[INVALID] or actual_counts[INCONCLUSIVE]:
        expected_status = INCONCLUSIVE
    elif actual_counts[FAIL]:
        expected_status = FAIL
    elif case_statuses and all(case_status == PASS for case_status in case_statuses):
        expected_status = PASS
    else:
        expected_status = INCONCLUSIVE
    if status != expected_status:
        return _evidence_error("inconsistent context gate status and case records")
    return EvidenceVerdict(status=status, full_coverage=True, message="context evidence verified")


_MCP_EXPECTED_CLAIMS = {
    "mcp_smoke": "offline_mcp_mechanism_gate",
    "mcp_reliability": "offline_mcp_lifecycle_reliability_gate",
    "mcp_benefit_synthetic": "synthetic_harness_self_test",
}


_MCP_PROTOCOL_SPECS: dict[str, dict[str, Any]] = {
    "mcp_smoke": {
        "protocol_id": "mcp-smoke",
        "required_ids": [
            "mcp_smoke_01_list_call",
            "mcp_smoke_02_permission_deny",
            "mcp_smoke_03_server_isolation",
            "mcp_smoke_04_deferred_default",
            "mcp_smoke_05_no_deferred_override",
        ],
        "descriptor": {
            "track": "mechanism_smoke",
            "required_case_ids": [
                "mcp_smoke_01_list_call",
                "mcp_smoke_02_permission_deny",
                "mcp_smoke_03_server_isolation",
                "mcp_smoke_04_deferred_default",
                "mcp_smoke_05_no_deferred_override",
            ],
            "grader": "deterministic_rule_based",
            "gate_rule": "all required cases appear exactly once and have status PASS",
            "pass_rate_denominator": ["PASS", "FAIL"],
        },
        "source_paths": [
            "eval/mcp_eval/evidence.py",
            "eval/mcp_eval/cases.py",
            "eval/mcp_eval/smoke.py",
        ],
    },
    "mcp_reliability": {
        "protocol_id": "mcp-reliability",
        "required_ids": [
            "mcp_reliability_01_partial_recovery",
            "mcp_reliability_02_config_isolation",
            "mcp_reliability_03_call_failure_recovery",
            "mcp_reliability_04_backoff_cap",
        ],
        "descriptor": {
            "track": "lifecycle_reliability",
            "required_case_ids": [
                "mcp_reliability_01_partial_recovery",
                "mcp_reliability_02_config_isolation",
                "mcp_reliability_03_call_failure_recovery",
                "mcp_reliability_04_backoff_cap",
            ],
            "grader": "deterministic_assertions",
            "gate_rule": "all required cases appear exactly once and have status PASS",
            "pass_rate_denominator": ["PASS", "FAIL"],
        },
        "source_paths": [
            "eval/mcp_eval/evidence.py",
            "eval/mcp_eval/reliability_cases.py",
            "eval/mcp_eval/reliability.py",
        ],
    },
    "mcp_benefit_synthetic": {
        "protocol_id": "mcp-benefit-synthetic",
        "required_ids": ["pair_001", "pair_002"],
        "descriptor": {
            "track": "benefit_synthetic",
            "case_id": "mcp_benefit_01_issue_context_patch",
            "mode": "fake",
            "expected_pair_count": 2,
            "conditions": ["MCP unavailable", "MCP issue context available"],
            "grader": "isolated_hidden_python_fixture_grader",
            "gate_rule": "all expected pairs complete and have status PASS",
            "pass_rate_denominator": ["PASS", "FAIL"],
        },
        "source_paths": [
            "eval/mcp_eval/evidence.py",
            "eval/mcp_eval/benefit_cases.py",
            "eval/mcp_eval/benefit.py",
        ],
    },
}

_MCP_ENVELOPE_FIELDS = {
    "interpretation",
    "payload",
    "protocol",
    "record_type",
    "run",
    "schema_version",
}
_MCP_RUN_FIELDS = {
    "api_calls",
    "code_version",
    "environment",
    "execution_mode",
    "llm_calls",
    "run_id",
    "timestamp_utc",
}


def _inspect_mcp_evidence(
    check_id: str,
    text: str,
    *,
    repo_root: Path,
    expected_code_version: str | None,
) -> EvidenceVerdict:
    from eval.mcp_eval.evidence import build_protocol, status_metrics

    records = _read_jsonl_text(text)
    summary = records[0]
    if any(set(record) != _MCP_ENVELOPE_FIELDS for record in records):
        return _evidence_error("MCP envelope fields do not match the evidence schema")
    if _contains_forbidden_score_field(records):
        return _evidence_error("synthetic or mechanism evidence must not contain score fields")
    if summary.get("record_type") != "run_summary":
        return _evidence_error("MCP evidence must begin with run_summary")
    if summary.get("schema_version") != "ace.mcp_evidence.v1":
        return _evidence_error("unexpected MCP evidence schema")

    interpretation = _mapping(summary.get("interpretation"), "interpretation")
    if set(interpretation) != {"claim", "what_this_does_not_prove"}:
        return _evidence_error("MCP interpretation fields are invalid")
    expected_claim = _MCP_EXPECTED_CLAIMS[check_id]
    if interpretation.get("claim") != expected_claim:
        return _evidence_error(f"unexpected MCP claim for {check_id}")
    _validated_boundary(interpretation.get("what_this_does_not_prove"))

    payload = _mapping(summary.get("payload"), "payload")
    coverage = _mapping(payload.get("coverage"), "coverage")
    required = _unique_string_list(coverage.get("required_ids"), "required_ids")
    protocol_spec = _MCP_PROTOCOL_SPECS[check_id]
    if required != protocol_spec["required_ids"]:
        return _evidence_error("MCP required ids do not match the current protocol")
    observed = _unique_string_list(coverage.get("observed_ids"), "observed_ids")
    if not required:
        return _evidence_error("MCP required_ids must not be empty")
    if coverage.get("full_coverage") is not True or observed != required:
        return _evidence_error("inconsistent MCP declared coverage")
    for key in ("duplicate_ids", "missing_ids", "unexpected_ids"):
        if coverage.get(key) != []:
            return _evidence_error(f"inconsistent MCP coverage field: {key}")

    case_records = records[1:]
    if any(record.get("record_type") != "case_result" for record in case_records):
        return _evidence_error("MCP evidence contains a non-case result record")
    actual_ids: list[str] = []
    case_statuses: list[str] = []
    for record in case_records:
        case_payload = _mapping(record.get("payload"), "case payload")
        if check_id == "mcp_benefit_synthetic":
            pair_index = case_payload.get("pair_index")
            if not isinstance(pair_index, int) or isinstance(pair_index, bool) or pair_index < 1:
                return _evidence_error("invalid MCP synthetic pair_index")
            actual_ids.append(f"pair_{pair_index:03d}")
        else:
            case_id = case_payload.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                return _evidence_error("invalid MCP case_id")
            actual_ids.append(case_id)
        case_statuses.append(_validated_status(case_payload.get("status"), "MCP case status"))

    try:
        actual = _unique_string_list(actual_ids, "MCP case record ids")
    except ValueError as exc:
        return _evidence_error(f"MCP record coverage invalid: {exc}")
    if actual != required:
        return _evidence_error("MCP record coverage does not exactly match required ids")

    status = _validated_status(payload.get("gate_status"), "MCP gate_status")
    if payload.get("status") != status:
        return _evidence_error("inconsistent MCP status fields")
    gate_pass = payload.get("gate_pass")
    if not isinstance(gate_pass, bool) or gate_pass != (status == PASS):
        return _evidence_error("inconsistent MCP gate status and gate_pass")
    if status == PASS and any(case_status != PASS for case_status in case_statuses):
        return _evidence_error("inconsistent MCP PASS with non-PASS case record")

    protocol = _mapping(summary.get("protocol"), "protocol")
    protocol_sha = _validated_sha256(protocol.get("sha256"), "MCP protocol sha256")
    expected_protocol = build_protocol(
        protocol_id=protocol_spec["protocol_id"],
        protocol_version="1.0.0",
        descriptor=protocol_spec["descriptor"],
        repo_root=repo_root,
        source_paths=protocol_spec["source_paths"],
    )
    if protocol != expected_protocol or protocol_sha != expected_protocol["sha256"]:
        return _evidence_error("MCP protocol fingerprint does not match current code")
    run = _mapping(summary.get("run"), "run")
    if set(run) != _MCP_RUN_FIELDS:
        return _evidence_error("MCP run fields do not match the evidence schema")
    code_version = _validated_code_version(run.get("code_version"))
    if expected_code_version is not None and code_version != _validated_code_version(
        expected_code_version
    ):
        return _evidence_error("MCP code_version does not match requested gate version")
    expected_mode = (
        "offline_synthetic" if check_id == "mcp_benefit_synthetic" else "offline_deterministic"
    )
    if run.get("execution_mode") != expected_mode:
        return _evidence_error("unexpected MCP execution mode")
    if run.get("api_calls") != 0 or run.get("llm_calls") != 0:
        return _evidence_error("offline MCP evidence reports external model or API calls")
    timestamp = _validated_utc_timestamp(run.get("timestamp_utc"), "MCP timestamp")
    _validated_mcp_environment(run.get("environment"))
    run_id = run.get("run_id")
    expected_run_id = hashlib.sha256(
        json.dumps(
            {
                "protocol_sha256": protocol_sha,
                "timestamp_utc": timestamp,
                "code_version": code_version,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    if run_id != expected_run_id:
        return _evidence_error("MCP run_id does not match protocol, time, and code version")

    expected_metrics = status_metrics(case_statuses)
    if payload.get("metrics") != expected_metrics:
        return _evidence_error("inconsistent MCP status metrics")
    declared_counts = _mapping(payload.get("status_counts"), "status_counts")
    actual_counts = Counter(case_statuses)
    if any(
        key not in VALID_STATUSES
        or not isinstance(value, int)
        or isinstance(value, bool)
        or value != actual_counts[key]
        for key, value in declared_counts.items()
    ) or any(actual_counts[key] and key not in declared_counts for key in actual_counts):
        return _evidence_error("inconsistent MCP status counts")
    if actual_counts[ERROR]:
        expected_status = ERROR
    elif any(actual_counts[name] for name in (SKIPPED, INVALID, INCONCLUSIVE)):
        expected_status = INCONCLUSIVE
    elif actual_counts[FAIL]:
        expected_status = FAIL
    elif case_statuses and all(case_status == PASS for case_status in case_statuses):
        expected_status = PASS
    else:
        expected_status = INCONCLUSIVE
    if status != expected_status:
        return _evidence_error("inconsistent MCP gate status and case records")

    for record in case_records:
        if record.get("schema_version") != summary["schema_version"]:
            return _evidence_error("inconsistent MCP record schema")
        if record.get("interpretation") != interpretation:
            return _evidence_error("inconsistent MCP record interpretation")
        if record.get("protocol") != protocol:
            return _evidence_error("inconsistent MCP protocol fingerprint")
        if record.get("run") != run:
            return _evidence_error("inconsistent MCP run metadata")
    return EvidenceVerdict(status=status, full_coverage=True, message="MCP evidence verified")


_SWE_FIXTURE_PATHS = {
    "suite": "eval/swebench/checker_fixture/suite.json",
    "results": "eval/swebench/checker_fixture/results.jsonl",
    "baseline": "eval/swebench/checker_fixture/baseline.fixture.json",
}
_SWE_EVIDENCE_FIELDS = {
    "artifact_hashes",
    "baseline_id",
    "claim",
    "code_version",
    "environment",
    "gate_eligible",
    "instance_count",
    "instance_outcomes",
    "protocol_fingerprint",
    "protocol_version",
    "run_id",
    "schema_version",
    "status",
    "suite_id",
    "timestamp_utc",
    "what_this_does_not_prove",
}


def _inspect_swe_checker_evidence(
    text: str,
    *,
    repo_root: Path,
    expected_code_version: str | None,
) -> EvidenceVerdict:
    from eval.swebench.checker import evaluate_files

    evidence = json.loads(text)
    if not isinstance(evidence, dict):
        return _evidence_error("SWE checker evidence must be a JSON object")
    if set(evidence) != _SWE_EVIDENCE_FIELDS:
        return _evidence_error("SWE checker fields do not match the exact output schema")
    if _contains_forbidden_score_field(evidence):
        return _evidence_error("SWE checker self-test must not contain score fields")
    if evidence.get("schema_version") != "coding-agent-eval.swe-checker.output.v1":
        return _evidence_error("unexpected SWE checker evidence schema")
    if evidence.get("claim") != "synthetic_checker_self_test":
        return _evidence_error("unexpected SWE checker claim")
    if evidence.get("gate_eligible") is not False:
        return _evidence_error("synthetic SWE checker evidence must keep gate_eligible=false")
    _validated_utc_timestamp(evidence.get("timestamp_utc"), "SWE checker timestamp")
    _validated_environment(evidence.get("environment"), "SWE checker environment")

    status = _validated_status(evidence.get("status"), "SWE checker status")
    code_version = _validated_code_version(evidence.get("code_version"))
    if expected_code_version is not None and code_version != _validated_code_version(
        expected_code_version
    ):
        return _evidence_error("SWE checker code_version does not match requested gate version")
    _validated_sha256(
        evidence.get("protocol_fingerprint"), "SWE checker protocol_fingerprint"
    )
    boundary = evidence.get("what_this_does_not_prove")
    if not isinstance(boundary, list) or not boundary or not all(
        isinstance(item, str) and item.strip() for item in boundary
    ):
        return _evidence_error("SWE checker boundary statement is missing")

    outcomes = evidence.get("instance_outcomes")
    count = evidence.get("instance_count")
    if (
        not isinstance(outcomes, list)
        or not outcomes
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count != len(outcomes)
    ):
        return _evidence_error("inconsistent SWE checker instance coverage")
    ids = _unique_string_list(
        [item.get("instance_id") if isinstance(item, dict) else None for item in outcomes],
        "SWE checker instance ids",
    )
    if len(ids) != count:
        return _evidence_error("inconsistent SWE checker instance coverage")
    outcome_statuses = [
        _validated_status(item.get("status"), "SWE checker instance status")
        for item in outcomes
    ]
    if status == PASS and any(item_status != PASS for item_status in outcome_statuses):
        return _evidence_error("inconsistent SWE checker PASS with non-PASS outcome")

    artifact_hashes = _mapping(evidence.get("artifact_hashes"), "artifact_hashes")
    if set(artifact_hashes) != set(_SWE_FIXTURE_PATHS):
        return _evidence_error("unexpected SWE checker fixture artifacts")
    for name, relative in _SWE_FIXTURE_PATHS.items():
        record = _mapping(artifact_hashes.get(name), f"artifact_hashes.{name}")
        if record.get("path") != relative:
            return _evidence_error("unexpected SWE checker fixture path")
        declared_sha = _validated_sha256(
            record.get("sha256"), f"artifact_hashes.{name}.sha256"
        )
        actual_sha = hashlib.sha256((repo_root / relative).read_bytes()).hexdigest()
        if declared_sha != actual_sha:
            return _evidence_error("SWE checker fixture hash does not match current bytes")

    expected = evaluate_files(
        repo_root / _SWE_FIXTURE_PATHS["suite"],
        repo_root / _SWE_FIXTURE_PATHS["results"],
        repo_root / _SWE_FIXTURE_PATHS["baseline"],
        code_version=code_version,
        allow_fixture_baseline=True,
    )
    stable_keys = {
        "baseline_id",
        "claim",
        "code_version",
        "gate_eligible",
        "instance_count",
        "instance_outcomes",
        "protocol_fingerprint",
        "protocol_version",
        "run_id",
        "schema_version",
        "status",
        "suite_id",
        "what_this_does_not_prove",
    }
    if any(evidence.get(key) != expected.get(key) for key in stable_keys):
        return _evidence_error(
            "SWE checker evidence does not match the current fixture and protocol"
        )
    return EvidenceVerdict(status=status, full_coverage=True, message="SWE checker evidence verified")


def _read_jsonl_text(text: str) -> list[dict[str, Any]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("evidence file is empty")
    records = [json.loads(line) for line in lines]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("every JSONL record must be an object")
    return records


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _unique_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} contains duplicate ids")
    return value


def _validated_status(value: Any, name: str) -> str:
    if not isinstance(value, str) or value not in VALID_STATUSES:
        raise ValueError(f"{name} is not in the status vocabulary")
    return value


def _validated_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validated_code_version(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", value) is None:
        raise ValueError("code_version must be a path-free revision label")
    return value


def _validated_boundary(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("what_this_does_not_prove must be non-empty")
    return value


def _validated_utc_timestamp(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be an RFC 3339 UTC timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must use UTC")
    return value


def _validated_environment(value: Any, name: str) -> dict[str, Any]:
    environment = _mapping(value, name)
    expected = {"machine", "platform", "python_implementation", "python_version"}
    if set(environment) != expected:
        raise ValueError(f"{name} fields do not match the environment schema")
    if not all(
        isinstance(item, str)
        and item.strip()
        and not _looks_machine_path(item)
        for item in environment.values()
    ):
        raise ValueError(f"{name} contains an invalid or path-like value")
    return environment


def _validated_mcp_environment(value: Any) -> dict[str, Any]:
    environment = _mapping(value, "MCP environment")
    expected = {
        "architecture",
        "os",
        "os_release",
        "packages",
        "python_implementation",
        "python_version",
    }
    if set(environment) != expected:
        raise ValueError("MCP environment fields do not match the evidence schema")
    packages = _mapping(environment.get("packages"), "MCP environment packages")
    if set(packages) != {"mcp"} or not (
        packages["mcp"] is None
        or isinstance(packages["mcp"], str) and bool(packages["mcp"].strip())
    ):
        raise ValueError("MCP package metadata is invalid")
    scalar_values = [environment[key] for key in expected if key != "packages"]
    if not all(
        isinstance(item, str)
        and item.strip()
        and not _looks_machine_path(item)
        for item in scalar_values
    ):
        raise ValueError("MCP environment contains an invalid or path-like value")
    return environment


def _looks_machine_path(value: str) -> bool:
    stripped = value.strip()
    if stripped.lower().startswith("file:"):
        return True
    return (
        PurePosixPath(stripped).is_absolute()
        or PureWindowsPath(stripped).is_absolute()
        or stripped.startswith(("\\", "/", "~/", "~\\"))
    )


def _contains_forbidden_score_field(value: Any) -> bool:
    forbidden = {"agent_score", "benchmark_score", "resolved_rate"}
    if isinstance(value, dict):
        return any(
            str(key).lower() in forbidden or _contains_forbidden_score_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_score_field(item) for item in value)
    return False


def _evidence_error(message: str) -> EvidenceVerdict:
    return EvidenceVerdict(status=ERROR, full_coverage=False, message=message)


def _subprocess_runner(spec: OfflineCommandSpec, cwd: Path) -> CommandResult:
    completed = subprocess.run(
        spec.command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.stdout:
        _safe_console_write(_sanitize_console_paths(completed.stdout, cwd), sys.stdout)
    if completed.stderr:
        _safe_console_write(_sanitize_console_paths(completed.stderr, cwd), sys.stderr)
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _safe_console_write(value: str, stream: Any) -> None:
    """Preserve diagnostics without letting terminal encoding fail the gate."""

    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        safe = value.encode(encoding, errors="backslashreplace").decode(encoding)
    except LookupError:
        safe = value.encode("utf-8", errors="backslashreplace").decode("utf-8")
    stream.write(safe)
    stream.flush()


def _sanitize_console_paths(value: str, repo_root: Path) -> str:
    return _sanitize_path_text(value, repo_root)


def _sanitize_path_text(value: str, repo_root: Path) -> str:
    candidates: list[tuple[str, str]] = []
    for path, marker in (
        (repo_root.resolve(), "<REPO_ROOT>"),
        (Path(tempfile.gettempdir()).resolve(), "<TEMP>"),
        (Path.home().resolve(), "<HOME>"),
    ):
        native = str(path)
        forms = {
            native,
            native.replace("\\", "/"),
            native.replace("/", "\\"),
            native.replace("\\", "\\\\"),
        }
        candidates.extend((form, marker) for form in forms if form)
    sanitized = value
    for form, marker in sorted(candidates, key=lambda item: len(item[0]), reverse=True):
        sanitized = sanitized.replace(form, marker)
    sanitized = re.sub(r"pytest-of-[A-Za-z0-9_.-]+", "pytest-of-<USER>", sanitized)
    sanitized = re.sub(r"pytest-\d+", "pytest-<RUN>", sanitized)
    return sanitized


_JUNIT_CREDENTIAL_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])" + "sk-" + "ant-" + r"[A-Za-z0-9_-]{20,}"),
    re.compile("gh" + r"[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])" + "sk-" + r"(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile("AK" + r"IA[0-9A-Z]{16}"),
    re.compile("AI" + r"za[0-9A-Za-z_-]{35}"),
    re.compile("-----BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)


def _sanitize_junit_text(value: str, repo_root: Path) -> str:
    sanitized = _sanitize_path_text(value, repo_root)
    for pattern in _JUNIT_CREDENTIAL_PATTERNS:
        sanitized = pattern.sub("<CREDENTIAL>", sanitized)
    return sanitized


def _sanitize_junit_artifact(path: Path, repo_root: Path) -> None:
    """Atomically redact machine paths and credential-shaped test fixtures."""

    tree = ET.parse(path)
    for element in tree.getroot().iter():
        if element.tag == "testsuite":
            element.attrib.pop("hostname", None)
        if element.text:
            element.text = _sanitize_junit_text(element.text, repo_root)
        if element.tail:
            element.tail = _sanitize_junit_text(element.tail, repo_root)
        for key, value in tuple(element.attrib.items()):
            element.set(key, _sanitize_junit_text(value, repo_root))
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _detect_code_version(repo_root: Path) -> str:
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if status.returncode != 0 or status.stdout.strip():
        return "WORKTREE"
    revision = subprocess.run(
        ("git", "rev-parse", "--short=12", "HEAD"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    candidate = revision.stdout.strip()
    if revision.returncode == 0:
        try:
            return _validated_code_version(candidate)
        except ValueError:
            pass
    return "WORKTREE"


def _resolve_offline_code_version(repo_root: Path, requested: str | None) -> str:
    """Bind explicit commit labels to the clean repository being evaluated.

    ``WORKTREE`` remains an intentional provisional label. Any other explicit
    label must resolve to the current clean HEAD; accepting an arbitrary string
    would let a dirty or unrelated tree masquerade as commit-pinned evidence.
    """

    if requested is None:
        return _detect_code_version(repo_root)
    label = _validated_code_version(requested)
    if label == "WORKTREE":
        return label

    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if status.returncode != 0:
        raise ValueError("explicit code_version requires a Git worktree")
    if status.stdout.strip():
        raise ValueError("explicit code_version requires a clean Git worktree")

    head = subprocess.run(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    requested_revision = subprocess.run(
        ("git", "rev-parse", "--verify", f"{label}^{{commit}}"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if (
        head.returncode != 0
        or requested_revision.returncode != 0
        or head.stdout.strip() != requested_revision.stdout.strip()
    ):
        raise ValueError("explicit code_version does not resolve to clean HEAD")
    return label


def _swe_checker_main(argv: Sequence[str]) -> int:
    _ensure_repo_import_path()
    from eval.swebench.checker import main as checker_main

    return checker_main(argv)


def _ensure_repo_import_path() -> None:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    offline = subparsers.add_parser(
        "offline",
        help="run every deterministic offline release check",
    )
    offline.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    offline.add_argument(
        "--evidence-dir",
        type=Path,
        help="new output directory; defaults to an ignored timestamped run directory",
    )
    offline.add_argument("--code-version")

    swe = subparsers.add_parser(
        "swe-checker-selftest",
        help="self-test normalized SWE-style artifacts with the synthetic checker",
    )
    swe.add_argument("--suite", type=Path, required=True)
    swe.add_argument("--results", type=Path, required=True)
    swe.add_argument("--baseline", type=Path, required=True)
    swe.add_argument("--output", type=Path)
    swe.add_argument("--code-version", required=True)
    swe.add_argument("--allow-fixture-baseline", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_repo_import_path()
    args = build_parser().parse_args(argv)
    if args.command == "swe-checker-selftest":
        checker_argv = [
            "--suite",
            str(args.suite),
            "--results",
            str(args.results),
            "--baseline",
            str(args.baseline),
            "--code-version",
            args.code_version,
        ]
        if args.output is not None:
            checker_argv.extend(("--output", str(args.output)))
        if args.allow_fixture_baseline:
            checker_argv.append("--allow-fixture-baseline")
        return _swe_checker_main(checker_argv)

    root = args.repo_root.expanduser().resolve()
    try:
        code_version = _resolve_offline_code_version(root, args.code_version)
    except ValueError as exc:
        print(f"Offline gate error: ValueError: {exc}", file=sys.stderr)
        return 2
    if args.evidence_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        evidence_dir = root / "eval" / "reports" / "offline" / stamp
    else:
        evidence_dir = args.evidence_dir.expanduser()
        if not evidence_dir.is_absolute():
            evidence_dir = root / evidence_dir
    try:
        report = run_offline_gate(
            repo_root=root,
            evidence_dir=evidence_dir,
            code_version=code_version,
        )
    except (OSError, ValueError) as exc:
        print(f"Offline gate error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Offline gate: {report['status']}")
    report_path = (evidence_dir / "offline-gate.json").resolve()
    try:
        display_path = report_path.relative_to(root).as_posix()
    except ValueError:
        display_path = report_path.name
    print(f"Evidence: {display_path}")
    return 0 if report["gate_pass"] else 1


def write_report(path: str | os.PathLike[str], report: dict[str, Any]) -> None:
    """Atomically write one UTF-8 JSON report."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _aggregate_status(
    checks: Sequence[GateCheck],
    *,
    missing: Sequence[str],
    full_coverage: bool,
) -> str:
    statuses = {check.status for check in checks}
    if statuses & {ERROR, JUDGE_ERROR, GRADER_ERROR}:
        return ERROR
    if FAIL in statuses:
        return FAIL
    if missing or not full_coverage:
        return INCONCLUSIVE
    if statuses & {INVALID, INCONCLUSIVE, SKIPPED}:
        return INCONCLUSIVE
    if len(checks) == len(REQUIRED_CHECK_IDS) and statuses == {PASS}:
        return PASS
    return INCONCLUSIVE


def _serialize_check(check: GateCheck, root: Path) -> dict[str, Any]:
    item: dict[str, Any] = {
        "check_id": check.check_id,
        "required": check.required,
        "status": check.status,
        "full_coverage": check.full_coverage,
        "claim_scope": check.claim_scope,
        "command": _portable_command(check.command, root),
        "message": check.message,
    }
    if check.artifact_path is not None:
        item["artifact"] = _artifact_record(
            check.artifact_path,
            root,
            expected_sha256=check.inspected_sha256,
        )
    return item


def _artifact_record(
    path: Path,
    root: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact is outside repository: {resolved.name}") from exc
    content = resolved.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"artifact changed after inspection: {relative.as_posix()}")
    return {
        "path": relative.as_posix(),
        "bytes": len(content),
        "sha256": digest,
    }


def _portable_command(command: Sequence[str], root: Path) -> list[str]:
    portable: list[str] = []
    executable = Path(sys.executable).resolve()
    for index, raw in enumerate(command):
        value = str(raw)
        candidate = Path(value).expanduser()
        if index == 0:
            try:
                if candidate.resolve() == executable:
                    portable.append("python")
                    continue
            except OSError:
                pass
        if candidate.is_absolute():
            resolved = candidate.resolve()
            try:
                value = resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(
                    f"command contains path outside repository: {resolved.name}"
                ) from exc
        portable.append(value.replace("\\", "/"))
    return portable


__all__ = [
    "AGGREGATE_PROTOCOL_DESCRIPTOR",
    "CLAIM_SCOPES",
    "CommandResult",
    "ERROR",
    "EvidenceVerdict",
    "FAIL",
    "GateCheck",
    "INCONCLUSIVE",
    "OfflineCommandSpec",
    "PASS",
    "REQUIRED_CHECK_IDS",
    "SKIPPED",
    "build_gate_report",
    "build_parser",
    "inspect_track_evidence",
    "main",
    "offline_command_specs",
    "protocol_fingerprint",
    "run_offline_gate",
    "write_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
