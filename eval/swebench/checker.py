"""Validate normalized SWE-style artifacts and recompute instance resolution.

The bundled artifacts are synthetic checker fixtures. They are deliberately
ineligible for benchmark or release gates unless a caller explicitly requests
the self-test mode, which remains ineligible even when it passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform as runtime_platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SUITE_SCHEMA_VERSION = "coding-agent-eval.swe-checker.suite.v1"
RESULT_SCHEMA_VERSION = "coding-agent-eval.swe-checker.result.v1"
BASELINE_SCHEMA_VERSION = "coding-agent-eval.swe-checker.baseline.v1"
OUTPUT_SCHEMA_VERSION = "coding-agent-eval.swe-checker.output.v1"
PROTOCOL_VERSION = "swe-checker-protocol-v1"

FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"
REQUIRED_TEST_KINDS = (FAIL_TO_PASS, PASS_TO_PASS)
TEST_STATUSES = ("PASS", "FAIL", "INVALID")
OUTPUT_STATUSES = ("PASS", "FAIL", "INVALID", "ERROR")

REPO_ROOT = Path(__file__).resolve().parents[2]
_REVISION_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]{0,127}$")

_PROTOCOL_DESCRIPTOR = {
    "protocol_version": PROTOCOL_VERSION,
    "schemas": {
        "suite": SUITE_SCHEMA_VERSION,
        "result": RESULT_SCHEMA_VERSION,
        "baseline": BASELINE_SCHEMA_VERSION,
        "output": OUTPUT_SCHEMA_VERSION,
    },
    "required_test_kinds": REQUIRED_TEST_KINDS,
    "test_statuses": TEST_STATUSES,
    "output_statuses": OUTPUT_STATUSES,
    "coverage_rule": (
        "suite, results, and baseline instance IDs must match exactly; each result "
        "must contain every suite-declared test ID exactly once with the declared kind"
    ),
    "resolution_rule": (
        "computed_resolved is true exactly when every required FAIL_TO_PASS "
        "and PASS_TO_PASS test is PASS; any INVALID status makes resolution undefined"
    ),
    "comparison_rule": "computed_resolved must equal baseline expected_resolved",
    "fixture_rule": (
        "a not_for_gate baseline requires --allow-fixture-baseline and always emits "
        "gate_eligible=false with claim=synthetic_checker_self_test"
    ),
    "exit_codes": {"PASS": 0, "FAIL_OR_INVALID": 1, "ERROR": 2},
    "evidence_metadata": {
        "timestamp": "timestamp_utc in UTC Z form",
        "environment": (
            "python_version, python_implementation, platform, and machine only"
        ),
        "sample_count": "instance_count",
    },
    "code_version_rule": (
        "1-128 characters; starts alphanumeric; remaining characters are "
        "alphanumeric, dot, underscore, plus, or hyphen"
    ),
    "artifact_hash_rule": (
        "SHA-256 is computed from the same single-read bytes used for parsing"
    ),
}

_FIXTURE_LIMITS = [
    "This synthetic self-test does not measure a coding agent or model.",
    "It does not establish a SWE-bench resolved rate or benchmark performance.",
    (
        "It does not validate repository checkout, patch application, container "
        "execution, or the official SWE-bench harness."
    ),
]

_NORMAL_LIMITS = [
    "This consistency check does not establish official SWE-bench artifact provenance.",
    "It does not measure behavior outside the suite's declared required tests.",
    (
        "It does not validate repository checkout, patch application, container "
        "execution, or the official SWE-bench harness."
    ),
]


class CheckerInputError(ValueError):
    """Raised when an input artifact violates the normalized protocol."""


class FixtureBaselineRejected(CheckerInputError):
    """Raised when a fixture baseline is used without explicit self-test intent."""


@dataclass(frozen=True)
class SuiteTest:
    test_id: str
    kind: str


@dataclass(frozen=True)
class SuiteInstance:
    instance_id: str
    tests: tuple[SuiteTest, ...]


@dataclass(frozen=True)
class NormalizedSuite:
    suite_id: str
    not_for_gate: bool
    instances: Mapping[str, SuiteInstance]


@dataclass(frozen=True)
class ResultTest:
    test_id: str
    kind: str
    status: str


@dataclass(frozen=True)
class ResultRecord:
    instance_id: str
    tests: tuple[ResultTest, ...]


@dataclass(frozen=True)
class NormalizedResults:
    suite_id: str
    run_id: str
    not_for_gate: bool
    records: Mapping[str, ResultRecord]


@dataclass(frozen=True)
class NormalizedBaseline:
    baseline_id: str
    suite_id: str
    not_for_gate: bool
    expected_resolutions: Mapping[str, bool]


def protocol_fingerprint() -> str:
    """Return a stable SHA-256 fingerprint of the checker protocol contract."""

    encoded = json.dumps(
        _PROTOCOL_DESCRIPTOR,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_files(
    suite_path: str | Path,
    results_path: str | Path,
    baseline_path: str | Path,
    *,
    code_version: str,
    allow_fixture_baseline: bool = False,
) -> dict[str, Any]:
    """Validate three artifacts and return deterministic structured evidence."""

    version = _revision_label(code_version, "code_version")
    suite_file = Path(suite_path)
    results_file = Path(results_path)
    baseline_file = Path(baseline_path)

    raw_suite, suite_bytes = _load_json(suite_file, "suite")
    raw_results, results_bytes = _load_jsonl(results_file)
    raw_baseline, baseline_bytes = _load_json(baseline_file, "baseline")
    suite = _normalize_suite(raw_suite)
    results = _normalize_results(raw_results, suite)
    baseline = _normalize_baseline(raw_baseline, suite)
    _validate_fixture_consistency(suite, results, baseline)

    if baseline.not_for_gate and not allow_fixture_baseline:
        raise FixtureBaselineRejected(
            "fixture baseline rejected; rerun only as a checker self-test with "
            "--allow-fixture-baseline"
        )

    fixture_mode = baseline.not_for_gate
    outcomes = _compute_outcomes(suite, results, baseline)
    if any(outcome["status"] == "INVALID" for outcome in outcomes):
        status = "INVALID"
    elif any(outcome["status"] == "FAIL" for outcome in outcomes):
        status = "FAIL"
    else:
        status = "PASS"

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_fingerprint(),
        "code_version": version,
        "timestamp_utc": _timestamp_utc(),
        "environment": _environment(),
        "instance_count": len(outcomes),
        "status": status,
        "claim": (
            "synthetic_checker_self_test"
            if fixture_mode
            else "swe_resolution_consistency_check"
        ),
        "gate_eligible": bool(
            not fixture_mode and status in {"PASS", "FAIL"}
        ),
        "suite_id": suite.suite_id,
        "run_id": results.run_id,
        "baseline_id": baseline.baseline_id,
        "artifact_hashes": _artifact_hashes(
            suite=(suite_file, suite_bytes),
            results=(results_file, results_bytes),
            baseline=(baseline_file, baseline_bytes),
        ),
        "instance_outcomes": outcomes,
        "what_this_does_not_prove": (
            list(_FIXTURE_LIMITS) if fixture_mode else list(_NORMAL_LIMITS)
        ),
    }


def _read_utf8(path: Path, label: str) -> tuple[str, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CheckerInputError(f"unable to read {label} artifact: {type(exc).__name__}") from exc
    try:
        return raw.decode("utf-8"), raw
    except UnicodeDecodeError as exc:
        raise CheckerInputError(f"{label} artifact is not valid UTF-8") from exc


def _load_json(path: Path, label: str) -> tuple[Any, bytes]:
    text, raw = _read_utf8(path, label)
    try:
        return json.loads(text), raw
    except json.JSONDecodeError as exc:
        raise CheckerInputError(
            f"{label} is not valid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc


def _load_jsonl(path: Path) -> tuple[list[Any], bytes]:
    text, raw = _read_utf8(path, "results")
    lines = text.splitlines()
    if not lines:
        raise CheckerInputError("results are empty")

    records: list[Any] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise CheckerInputError(f"results line {line_number} is empty")
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise CheckerInputError(
                f"results line {line_number} is not valid JSON at column {exc.colno}"
            ) from exc
    return records, raw


def _normalize_suite(raw: Any) -> NormalizedSuite:
    data = _object(raw, "suite")
    _exact_keys(
        data,
        {"schema_version", "suite_id", "not_for_gate", "instances"},
        "suite",
    )
    _schema(data["schema_version"], SUITE_SCHEMA_VERSION, "suite")
    suite_id = _nonempty_string(data["suite_id"], "suite.suite_id")
    not_for_gate = _boolean(data["not_for_gate"], "suite.not_for_gate")
    raw_instances = _list(data["instances"], "suite.instances")
    if not raw_instances:
        raise CheckerInputError("suite.instances is empty")

    instances: dict[str, SuiteInstance] = {}
    for index, raw_instance in enumerate(raw_instances):
        label = f"suite.instances[{index}]"
        instance = _object(raw_instance, label)
        _exact_keys(instance, {"instance_id", "tests"}, label)
        instance_id = _nonempty_string(instance["instance_id"], f"{label}.instance_id")
        if instance_id in instances:
            raise CheckerInputError(f"suite has duplicate instance ID: {instance_id}")
        tests = _normalize_suite_tests(instance["tests"], instance_id)
        instances[instance_id] = SuiteInstance(instance_id=instance_id, tests=tests)

    return NormalizedSuite(
        suite_id=suite_id,
        not_for_gate=not_for_gate,
        instances=instances,
    )


def _normalize_suite_tests(raw: Any, instance_id: str) -> tuple[SuiteTest, ...]:
    raw_tests = _list(raw, f"suite instance {instance_id} tests")
    if not raw_tests:
        raise CheckerInputError(f"suite instance {instance_id} tests are empty")
    tests: list[SuiteTest] = []
    seen: set[str] = set()
    kinds: set[str] = set()
    for index, raw_test in enumerate(raw_tests):
        label = f"suite instance {instance_id} test[{index}]"
        test = _object(raw_test, label)
        _exact_keys(test, {"test_id", "kind"}, label)
        test_id = _nonempty_string(test["test_id"], f"{label}.test_id")
        if test_id in seen:
            raise CheckerInputError(
                f"suite instance {instance_id} has duplicate test ID: {test_id}"
            )
        kind = _choice(test["kind"], REQUIRED_TEST_KINDS, f"{label}.kind")
        seen.add(test_id)
        kinds.add(kind)
        tests.append(SuiteTest(test_id=test_id, kind=kind))
    for required_kind in REQUIRED_TEST_KINDS:
        if required_kind not in kinds:
            raise CheckerInputError(
                f"suite instance {instance_id} requires at least one {required_kind} test"
            )
    return tuple(tests)


def _normalize_results(raw_records: list[Any], suite: NormalizedSuite) -> NormalizedResults:
    if not raw_records:
        raise CheckerInputError("results are empty")
    records: dict[str, ResultRecord] = {}
    run_id: str | None = None
    results_not_for_gate: bool | None = None

    for index, raw_record in enumerate(raw_records):
        label = f"results record[{index}]"
        record = _object(raw_record, label)
        _exact_keys(
            record,
            {
                "schema_version",
                "suite_id",
                "run_id",
                "not_for_gate",
                "instance_id",
                "tests",
            },
            label,
        )
        _schema(record["schema_version"], RESULT_SCHEMA_VERSION, label)
        record_suite_id = _nonempty_string(record["suite_id"], f"{label}.suite_id")
        if record_suite_id != suite.suite_id:
            raise CheckerInputError(
                f"{label}.suite_id does not match suite.suite_id"
            )
        current_run_id = _nonempty_string(record["run_id"], f"{label}.run_id")
        if run_id is None:
            run_id = current_run_id
        elif current_run_id != run_id:
            raise CheckerInputError("results contain inconsistent run_id values")
        current_not_for_gate = _boolean(
            record["not_for_gate"], f"{label}.not_for_gate"
        )
        if results_not_for_gate is None:
            results_not_for_gate = current_not_for_gate
        elif current_not_for_gate != results_not_for_gate:
            raise CheckerInputError("results contain inconsistent not_for_gate values")

        instance_id = _nonempty_string(
            record["instance_id"], f"{label}.instance_id"
        )
        if instance_id in records:
            raise CheckerInputError(f"results have duplicate instance ID: {instance_id}")
        tests = _normalize_result_tests(record["tests"], instance_id)
        records[instance_id] = ResultRecord(instance_id=instance_id, tests=tests)

    _exact_id_coverage(
        expected=set(suite.instances),
        actual=set(records),
        label="results instance",
    )
    for instance_id, result in records.items():
        _validate_test_coverage(suite.instances[instance_id], result)

    assert run_id is not None
    assert results_not_for_gate is not None
    return NormalizedResults(
        suite_id=suite.suite_id,
        run_id=run_id,
        not_for_gate=results_not_for_gate,
        records=records,
    )


def _normalize_result_tests(raw: Any, instance_id: str) -> tuple[ResultTest, ...]:
    raw_tests = _list(raw, f"results instance {instance_id} tests")
    if not raw_tests:
        raise CheckerInputError(f"results instance {instance_id} tests are empty")
    tests: list[ResultTest] = []
    seen: set[str] = set()
    for index, raw_test in enumerate(raw_tests):
        label = f"results instance {instance_id} test[{index}]"
        test = _object(raw_test, label)
        _exact_keys(test, {"test_id", "kind", "status"}, label)
        test_id = _nonempty_string(test["test_id"], f"{label}.test_id")
        if test_id in seen:
            raise CheckerInputError(
                f"results instance {instance_id} has duplicate test ID: {test_id}"
            )
        kind = _choice(test["kind"], REQUIRED_TEST_KINDS, f"{label}.kind")
        status = _choice(test["status"], TEST_STATUSES, f"{label}.status")
        seen.add(test_id)
        tests.append(ResultTest(test_id=test_id, kind=kind, status=status))
    return tuple(tests)


def _validate_test_coverage(suite: SuiteInstance, result: ResultRecord) -> None:
    expected = {test.test_id: test.kind for test in suite.tests}
    actual = {test.test_id: test.kind for test in result.tests}
    _exact_id_coverage(
        expected=set(expected),
        actual=set(actual),
        label=f"results test for {suite.instance_id}",
    )
    for test_id in sorted(expected):
        if actual[test_id] != expected[test_id]:
            raise CheckerInputError(
                f"results test {suite.instance_id}/{test_id} kind mismatch: "
                f"expected {expected[test_id]}, got {actual[test_id]}"
            )


def _normalize_baseline(raw: Any, suite: NormalizedSuite) -> NormalizedBaseline:
    data = _object(raw, "baseline")
    _exact_keys(
        data,
        {
            "schema_version",
            "baseline_id",
            "suite_id",
            "not_for_gate",
            "expected_resolutions",
        },
        "baseline",
    )
    _schema(data["schema_version"], BASELINE_SCHEMA_VERSION, "baseline")
    baseline_id = _nonempty_string(data["baseline_id"], "baseline.baseline_id")
    suite_id = _nonempty_string(data["suite_id"], "baseline.suite_id")
    if suite_id != suite.suite_id:
        raise CheckerInputError("baseline.suite_id does not match suite.suite_id")
    not_for_gate = _boolean(data["not_for_gate"], "baseline.not_for_gate")
    raw_expected = _list(
        data["expected_resolutions"], "baseline.expected_resolutions"
    )
    if not raw_expected:
        raise CheckerInputError("baseline.expected_resolutions is empty")

    expected_resolutions: dict[str, bool] = {}
    for index, raw_item in enumerate(raw_expected):
        label = f"baseline.expected_resolutions[{index}]"
        item = _object(raw_item, label)
        _exact_keys(item, {"instance_id", "resolved"}, label)
        instance_id = _nonempty_string(item["instance_id"], f"{label}.instance_id")
        if instance_id in expected_resolutions:
            raise CheckerInputError(f"baseline has duplicate instance ID: {instance_id}")
        expected_resolutions[instance_id] = _boolean(
            item["resolved"], f"{label}.resolved"
        )

    _exact_id_coverage(
        expected=set(suite.instances),
        actual=set(expected_resolutions),
        label="baseline instance",
    )
    return NormalizedBaseline(
        baseline_id=baseline_id,
        suite_id=suite_id,
        not_for_gate=not_for_gate,
        expected_resolutions=expected_resolutions,
    )


def _validate_fixture_consistency(
    suite: NormalizedSuite,
    results: NormalizedResults,
    baseline: NormalizedBaseline,
) -> None:
    markers = {
        "suite": suite.not_for_gate,
        "results": results.not_for_gate,
        "baseline": baseline.not_for_gate,
    }
    if len(set(markers.values())) != 1:
        values = ", ".join(f"{name}={value}" for name, value in markers.items())
        raise CheckerInputError(f"not_for_gate markers must agree: {values}")


def _compute_outcomes(
    suite: NormalizedSuite,
    results: NormalizedResults,
    baseline: NormalizedBaseline,
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for instance_id in sorted(suite.instances):
        record = results.records[instance_id]
        ordered_tests = sorted(record.tests, key=lambda test: (test.kind, test.test_id))
        test_statuses = {
            kind: [
                {"test_id": test.test_id, "status": test.status}
                for test in ordered_tests
                if test.kind == kind
            ]
            for kind in REQUIRED_TEST_KINDS
        }
        if any(test.status == "INVALID" for test in ordered_tests):
            computed_resolved: bool | None = None
            outcome_status = "INVALID"
        else:
            computed_resolved = all(test.status == "PASS" for test in ordered_tests)
            outcome_status = (
                "PASS"
                if computed_resolved == baseline.expected_resolutions[instance_id]
                else "FAIL"
            )
        outcomes.append(
            {
                "instance_id": instance_id,
                "status": outcome_status,
                "computed_resolved": computed_resolved,
                "expected_resolved": baseline.expected_resolutions[instance_id],
                "test_statuses": test_statuses,
            }
        )
    return outcomes


def _exact_id_coverage(*, expected: set[str], actual: set[str], label: str) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if not missing and not extra:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing IDs: {', '.join(missing)}")
    if extra:
        details.append(f"extra IDs: {', '.join(extra)}")
    raise CheckerInputError(f"{label} coverage mismatch; {'; '.join(details)}")


def _artifact_hashes(
    **artifacts: tuple[Path, bytes],
) -> dict[str, dict[str, str]]:
    return {
        label: {
            "path": _display_path(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for label, (path, raw) in artifacts.items()
    }


def _best_effort_artifact_hashes(**paths: Path) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        item: dict[str, Any] = {"path": _display_path(path)}
        try:
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            item["sha256"] = None
        artifacts[label] = item
    return artifacts


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    for base in (REPO_ROOT, Path.cwd().resolve()):
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.name or "artifact"


def _error_evidence(
    exc: BaseException,
    *,
    suite_path: Path,
    results_path: Path,
    baseline_path: Path,
    code_version: str,
) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_fingerprint(),
        "code_version": _safe_revision_label(code_version),
        "timestamp_utc": _timestamp_utc(),
        "environment": _environment(),
        "instance_count": 0,
        "status": "ERROR",
        "claim": "checker_input_validation",
        "gate_eligible": False,
        "artifact_hashes": _best_effort_artifact_hashes(
            suite=suite_path,
            results=results_path,
            baseline=baseline_path,
        ),
        "error": {"type": type(exc).__name__, "message": str(exc)},
        "what_this_does_not_prove": list(_NORMAL_LIMITS),
    }


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _environment() -> dict[str, str]:
    """Return reproducibility metadata without hostnames, users, or paths."""

    return {
        "python_version": runtime_platform.python_version() or "unknown",
        "python_implementation": (
            runtime_platform.python_implementation() or "unknown"
        ),
        "platform": runtime_platform.system() or sys.platform or "unknown",
        "machine": runtime_platform.machine() or "unknown",
    }


def _revision_label(value: Any, label: str) -> str:
    text = _nonempty_string(value, label)
    if _REVISION_LABEL.fullmatch(text) is None:
        raise CheckerInputError(
            f"{label} must be a path-free revision label: start with a letter or "
            "digit and use only letters, digits, '.', '_', '+', or '-'"
        )
    return text


def _safe_revision_label(value: Any) -> str:
    try:
        return _revision_label(value, "code_version")
    except CheckerInputError:
        return "INVALID"


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CheckerInputError(f"{label} must be a JSON object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CheckerInputError(f"{label} must be a JSON array")
    return value


def _exact_keys(data: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(data)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise CheckerInputError(f"{label} has missing fields: {', '.join(missing)}")
    if extra:
        raise CheckerInputError(f"{label} has extra fields: {', '.join(extra)}")


def _schema(value: Any, expected: str, label: str) -> None:
    if value != expected:
        raise CheckerInputError(
            f"{label}.schema_version must be {expected!r}, got {value!r}"
        )


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CheckerInputError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise CheckerInputError(f"{label} must not contain leading or trailing whitespace")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise CheckerInputError(f"{label} must be a boolean")
    return value


def _choice(value: Any, choices: Sequence[str], label: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise CheckerInputError(
            f"{label} must be one of {', '.join(choices)}, got {value!r}"
        )
    return value


def _write_evidence(path: Path, evidence: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--code-version", required=True)
    parser.add_argument(
        "--allow-fixture-baseline",
        action="store_true",
        help="run an explicit synthetic checker self-test; never gate eligible",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evidence = evaluate_files(
            args.suite,
            args.results,
            args.baseline,
            code_version=args.code_version,
            allow_fixture_baseline=args.allow_fixture_baseline,
        )
    except CheckerInputError as exc:
        evidence = _error_evidence(
            exc,
            suite_path=args.suite,
            results_path=args.results,
            baseline_path=args.baseline,
            code_version=args.code_version,
        )
        if args.output is not None:
            _write_evidence(args.output, evidence)
        else:
            print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
        print(f"Checker error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.output is not None:
        _write_evidence(args.output, evidence)
        print(f"Evidence: {_display_path(args.output)}")
    else:
        print(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
    print(f"Checker status: {evidence['status']}")
    return 0 if evidence["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
