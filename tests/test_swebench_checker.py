from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from eval.swebench import checker


def _payloads() -> tuple[dict, list[dict], dict]:
    suite = {
        "schema_version": checker.SUITE_SCHEMA_VERSION,
        "suite_id": "synthetic-checker-selftest-v1",
        "not_for_gate": True,
        "instances": [
            {
                "instance_id": "synthetic-resolved",
                "tests": [
                    {"test_id": "fix-regression", "kind": "FAIL_TO_PASS"},
                    {"test_id": "keep-behavior", "kind": "PASS_TO_PASS"},
                ],
            },
            {
                "instance_id": "synthetic-unresolved",
                "tests": [
                    {"test_id": "fix-still-fails", "kind": "FAIL_TO_PASS"},
                    {"test_id": "keep-still-passes", "kind": "PASS_TO_PASS"},
                ],
            },
        ],
    }
    results = [
        {
            "schema_version": checker.RESULT_SCHEMA_VERSION,
            "suite_id": suite["suite_id"],
            "run_id": "synthetic-checker-run-v1",
            "not_for_gate": True,
            "instance_id": "synthetic-resolved",
            "tests": [
                {
                    "test_id": "fix-regression",
                    "kind": "FAIL_TO_PASS",
                    "status": "PASS",
                },
                {
                    "test_id": "keep-behavior",
                    "kind": "PASS_TO_PASS",
                    "status": "PASS",
                },
            ],
        },
        {
            "schema_version": checker.RESULT_SCHEMA_VERSION,
            "suite_id": suite["suite_id"],
            "run_id": "synthetic-checker-run-v1",
            "not_for_gate": True,
            "instance_id": "synthetic-unresolved",
            "tests": [
                {
                    "test_id": "fix-still-fails",
                    "kind": "FAIL_TO_PASS",
                    "status": "FAIL",
                },
                {
                    "test_id": "keep-still-passes",
                    "kind": "PASS_TO_PASS",
                    "status": "PASS",
                },
            ],
        },
    ]
    baseline = {
        "schema_version": checker.BASELINE_SCHEMA_VERSION,
        "baseline_id": "synthetic-checker-oracle-v1",
        "suite_id": suite["suite_id"],
        "not_for_gate": True,
        "expected_resolutions": [
            {"instance_id": "synthetic-resolved", "resolved": True},
            {"instance_id": "synthetic-unresolved", "resolved": False},
        ],
    }
    return suite, results, baseline


def _write_payloads(
    tmp_path: Path,
    suite: dict,
    results: list[dict],
    baseline: dict,
) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    suite_path = tmp_path / "suite.json"
    results_path = tmp_path / "results.jsonl"
    baseline_path = tmp_path / "baseline.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in results),
        encoding="utf-8",
    )
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    return suite_path, results_path, baseline_path


def _evaluate(
    tmp_path: Path,
    *,
    suite: dict | None = None,
    results: list[dict] | None = None,
    baseline: dict | None = None,
    allow_fixture_baseline: bool = True,
) -> dict:
    default_suite, default_results, default_baseline = _payloads()
    paths = _write_payloads(
        tmp_path,
        suite if suite is not None else default_suite,
        results if results is not None else default_results,
        baseline if baseline is not None else default_baseline,
    )
    return checker.evaluate_files(
        *paths,
        code_version="WORKTREE",
        allow_fixture_baseline=allow_fixture_baseline,
    )


def _all_keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            *(str(key) for key in value),
            *(nested for item in value.values() for nested in _all_keys(item)),
        ]
    if isinstance(value, list):
        return [nested for item in value for nested in _all_keys(item)]
    return []


def test_fixture_selftest_passes_without_becoming_gate_evidence_or_agent_score(
    tmp_path: Path,
):
    suite, results, baseline = _payloads()
    paths = _write_payloads(tmp_path, suite, results, baseline)

    evidence = checker.evaluate_files(
        *paths,
        code_version="WORKTREE",
        allow_fixture_baseline=True,
    )

    assert evidence["status"] == "PASS"
    assert evidence["claim"] == "synthetic_checker_self_test"
    assert evidence["gate_eligible"] is False
    assert evidence["code_version"] == "WORKTREE"
    assert evidence["protocol_fingerprint"] == checker.protocol_fingerprint()
    assert evidence["instance_count"] == 2
    assert evidence["timestamp_utc"].endswith("Z")
    datetime.fromisoformat(evidence["timestamp_utc"].removesuffix("Z") + "+00:00")
    assert set(evidence["environment"]) == {
        "python_version",
        "python_implementation",
        "platform",
        "machine",
    }
    assert all(evidence["environment"].values())
    assert [item["instance_id"] for item in evidence["instance_outcomes"]] == [
        "synthetic-resolved",
        "synthetic-unresolved",
    ]
    assert [item["computed_resolved"] for item in evidence["instance_outcomes"]] == [
        True,
        False,
    ]
    assert all("score" not in key.casefold() for key in _all_keys(evidence))
    assert evidence["what_this_does_not_prove"]

    for label, path in zip(("suite", "results", "baseline"), paths, strict=True):
        artifact = evidence["artifact_hashes"][label]
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert not Path(artifact["path"]).is_absolute()


def test_resolution_output_is_deterministic_across_input_order(tmp_path: Path):
    suite, results, baseline = _payloads()
    first = _evaluate(
        tmp_path / "first",
        suite=copy.deepcopy(suite),
        results=copy.deepcopy(results),
        baseline=copy.deepcopy(baseline),
    )
    suite["instances"].reverse()
    for instance in suite["instances"]:
        instance["tests"].reverse()
    results.reverse()
    for record in results:
        record["tests"].reverse()
    baseline["expected_resolutions"].reverse()
    second = _evaluate(
        tmp_path / "second",
        suite=suite,
        results=results,
        baseline=baseline,
    )

    assert second["protocol_fingerprint"] == first["protocol_fingerprint"]
    assert second["instance_outcomes"] == first["instance_outcomes"]


def test_artifact_hashes_bind_the_same_bytes_that_were_parsed(
    tmp_path: Path,
    monkeypatch,
):
    suite, results, baseline = _payloads()
    suite_path, results_path, baseline_path = _write_payloads(
        tmp_path,
        suite,
        results,
        baseline,
    )
    original = suite_path.read_bytes()
    calls = 0
    real_read_text = Path.read_text
    real_read_bytes = Path.read_bytes

    def changing_bytes(path: Path) -> bytes:
        nonlocal calls
        if path.resolve() != suite_path.resolve():
            return real_read_bytes(path)
        calls += 1
        return original if calls == 1 else b'{"changed_after_parse":true}'

    def changing_text(path: Path, *args, **kwargs) -> str:
        if path.resolve() != suite_path.resolve():
            return real_read_text(path, *args, **kwargs)
        return changing_bytes(path).decode(kwargs.get("encoding") or "utf-8")

    monkeypatch.setattr(Path, "read_bytes", changing_bytes)
    monkeypatch.setattr(Path, "read_text", changing_text)

    evidence = checker.evaluate_files(
        suite_path,
        results_path,
        baseline_path,
        code_version="WORKTREE",
        allow_fixture_baseline=True,
    )

    assert calls == 1
    assert evidence["artifact_hashes"]["suite"]["sha256"] == hashlib.sha256(
        original
    ).hexdigest()


def test_fixture_baseline_is_rejected_without_explicit_flag(tmp_path: Path):
    with pytest.raises(checker.FixtureBaselineRejected, match="allow-fixture-baseline"):
        _evaluate(tmp_path, allow_fixture_baseline=False)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_result_instance_ids_require_exact_unique_suite_coverage(
    tmp_path: Path,
    mutation: str,
):
    suite, results, baseline = _payloads()
    if mutation == "missing":
        results.pop()
    elif mutation == "extra":
        extra = copy.deepcopy(results[0])
        extra["instance_id"] = "synthetic-extra"
        results.append(extra)
    else:
        results.append(copy.deepcopy(results[0]))

    with pytest.raises(checker.CheckerInputError, match=mutation):
        _evaluate(tmp_path, suite=suite, results=results, baseline=baseline)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_result_test_ids_require_exact_unique_instance_coverage(
    tmp_path: Path,
    mutation: str,
):
    suite, results, baseline = _payloads()
    tests = results[0]["tests"]
    if mutation == "missing":
        tests.pop()
    elif mutation == "extra":
        tests.append(
            {"test_id": "not-required", "kind": "PASS_TO_PASS", "status": "PASS"}
        )
    else:
        tests.append(copy.deepcopy(tests[0]))

    with pytest.raises(checker.CheckerInputError, match=mutation):
        _evaluate(tmp_path, suite=suite, results=results, baseline=baseline)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_baseline_instance_ids_require_exact_unique_suite_coverage(
    tmp_path: Path,
    mutation: str,
):
    suite, results, baseline = _payloads()
    expected = baseline["expected_resolutions"]
    if mutation == "missing":
        expected.pop()
    elif mutation == "extra":
        expected.append({"instance_id": "synthetic-extra", "resolved": False})
    else:
        expected.append(copy.deepcopy(expected[0]))

    with pytest.raises(checker.CheckerInputError, match=mutation):
        _evaluate(tmp_path, suite=suite, results=results, baseline=baseline)


def test_suite_rejects_duplicate_instance_and_test_ids(tmp_path: Path):
    suite, results, baseline = _payloads()
    duplicate_instance = copy.deepcopy(suite)
    duplicate_instance["instances"].append(copy.deepcopy(suite["instances"][0]))
    with pytest.raises(checker.CheckerInputError, match="duplicate"):
        _evaluate(
            tmp_path / "instance",
            suite=duplicate_instance,
            results=results,
            baseline=baseline,
        )

    duplicate_test = copy.deepcopy(suite)
    duplicate_test["instances"][0]["tests"].append(
        copy.deepcopy(duplicate_test["instances"][0]["tests"][0])
    )
    with pytest.raises(checker.CheckerInputError, match="duplicate"):
        _evaluate(
            tmp_path / "test",
            suite=duplicate_test,
            results=results,
            baseline=baseline,
        )


def test_suite_requires_fail_to_pass_and_pass_to_pass_tests(tmp_path: Path):
    suite, results, baseline = _payloads()
    suite["instances"][0]["tests"] = [suite["instances"][0]["tests"][0]]

    with pytest.raises(checker.CheckerInputError, match="PASS_TO_PASS"):
        _evaluate(tmp_path, suite=suite, results=results, baseline=baseline)


def test_result_rejects_kind_mismatch_and_malformed_status(tmp_path: Path):
    suite, results, baseline = _payloads()
    kind_mismatch = copy.deepcopy(results)
    kind_mismatch[0]["tests"][0]["kind"] = "PASS_TO_PASS"
    with pytest.raises(checker.CheckerInputError, match="kind"):
        _evaluate(
            tmp_path / "kind",
            suite=suite,
            results=kind_mismatch,
            baseline=baseline,
        )

    malformed = copy.deepcopy(results)
    malformed[0]["tests"][0]["status"] = "success"
    with pytest.raises(checker.CheckerInputError, match="status"):
        _evaluate(
            tmp_path / "status",
            suite=suite,
            results=malformed,
            baseline=baseline,
        )


def test_mismatched_expected_resolution_is_fail_not_checker_pass(tmp_path: Path):
    suite, results, baseline = _payloads()
    baseline["expected_resolutions"][0]["resolved"] = False

    evidence = _evaluate(tmp_path, suite=suite, results=results, baseline=baseline)

    assert evidence["status"] == "FAIL"
    outcome = next(
        item
        for item in evidence["instance_outcomes"]
        if item["instance_id"] == "synthetic-resolved"
    )
    assert outcome["status"] == "FAIL"
    assert outcome["computed_resolved"] is True
    assert outcome["expected_resolved"] is False


def test_empty_suite_and_empty_results_are_rejected(tmp_path: Path):
    suite, results, baseline = _payloads()
    empty_suite = copy.deepcopy(suite)
    empty_suite["instances"] = []
    with pytest.raises(checker.CheckerInputError, match="empty"):
        _evaluate(
            tmp_path / "suite",
            suite=empty_suite,
            results=results,
            baseline=baseline,
        )

    with pytest.raises(checker.CheckerInputError, match="empty"):
        _evaluate(
            tmp_path / "results",
            suite=suite,
            results=[],
            baseline=baseline,
        )


def test_all_invalid_results_are_invalid_and_cannot_pass(tmp_path: Path):
    suite, results, baseline = _payloads()
    for record in results:
        for test in record["tests"]:
            test["status"] = "INVALID"

    evidence = _evaluate(tmp_path, suite=suite, results=results, baseline=baseline)

    assert evidence["status"] == "INVALID"
    assert all(item["status"] == "INVALID" for item in evidence["instance_outcomes"])
    assert all(item["computed_resolved"] is None for item in evidence["instance_outcomes"])


def test_schema_versions_and_fixture_markers_are_strict(tmp_path: Path):
    suite, results, baseline = _payloads()
    wrong_schema = copy.deepcopy(suite)
    wrong_schema["schema_version"] = "swe-checker-suite-v0"
    with pytest.raises(checker.CheckerInputError, match="schema_version"):
        _evaluate(
            tmp_path / "schema",
            suite=wrong_schema,
            results=results,
            baseline=baseline,
        )

    mixed_fixture = copy.deepcopy(results)
    mixed_fixture[0]["not_for_gate"] = False
    with pytest.raises(checker.CheckerInputError, match="not_for_gate"):
        _evaluate(
            tmp_path / "fixture",
            suite=suite,
            results=mixed_fixture,
            baseline=baseline,
        )


@pytest.mark.parametrize(
    "invalid_version",
    ["/tmp/rev", r"C:\repo\rev", "feature/rev"],
)
def test_code_version_rejects_path_like_values_without_echoing_them_to_evidence(
    tmp_path: Path,
    invalid_version: str,
    capsys,
):
    suite, results, baseline = _payloads()
    suite_path, results_path, baseline_path = _write_payloads(
        tmp_path,
        suite,
        results,
        baseline,
    )
    output_path = tmp_path / "error-evidence.json"

    rc = checker.main(
        [
            "--suite",
            str(suite_path),
            "--results",
            str(results_path),
            "--baseline",
            str(baseline_path),
            "--output",
            str(output_path),
            "--code-version",
            invalid_version,
            "--allow-fixture-baseline",
        ]
    )

    assert rc == 2
    serialized = output_path.read_text(encoding="utf-8")
    evidence = json.loads(serialized)
    assert evidence["status"] == "ERROR"
    assert evidence["code_version"] == "INVALID"
    assert invalid_version not in serialized
    assert invalid_version not in capsys.readouterr().err


def test_cli_exit_codes_and_structured_evidence_never_turn_bad_input_into_pass(
    tmp_path: Path,
):
    suite, results, baseline = _payloads()
    suite_path, results_path, baseline_path = _write_payloads(
        tmp_path,
        suite,
        results,
        baseline,
    )
    output_path = tmp_path / "evidence.json"
    common = [
        "--suite",
        str(suite_path),
        "--results",
        str(results_path),
        "--baseline",
        str(baseline_path),
        "--output",
        str(output_path),
        "--code-version",
        "WORKTREE",
    ]

    assert checker.main(common) == 2
    rejected = json.loads(output_path.read_text(encoding="utf-8"))
    assert rejected["status"] == "ERROR"
    assert rejected["gate_eligible"] is False
    assert rejected["instance_count"] == 0
    assert rejected["timestamp_utc"].endswith("Z")
    assert set(rejected["environment"]) == {
        "python_version",
        "python_implementation",
        "platform",
        "machine",
    }

    assert checker.main([*common, "--allow-fixture-baseline"]) == 0
    passed = json.loads(output_path.read_text(encoding="utf-8"))
    assert passed["status"] == "PASS"
    assert passed["claim"] == "synthetic_checker_self_test"

    baseline["expected_resolutions"][0]["resolved"] = False
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    assert checker.main([*common, "--allow-fixture-baseline"]) == 1
    failed = json.loads(output_path.read_text(encoding="utf-8"))
    assert failed["status"] == "FAIL"
