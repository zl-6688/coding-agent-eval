from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import release_gate as regression_gate


ROOT = Path(__file__).resolve().parents[1]


def _passing_checks(tmp_path):
    checks = []
    for index, check_id in enumerate(regression_gate.REQUIRED_CHECK_IDS):
        artifact = tmp_path / "evidence" / f"{check_id}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps({"check_id": check_id, "status": "PASS"}),
            encoding="utf-8",
        )
        checks.append(
            regression_gate.GateCheck(
                check_id=check_id,
                status="PASS",
                full_coverage=True,
                claim_scope=regression_gate.CLAIM_SCOPES[check_id],
                command=("python", "-m", check_id),
                artifact_path=artifact,
                message=f"check {index} passed",
            )
        )
    return checks


def test_complete_offline_gate_records_relative_hashed_evidence(tmp_path):
    checks = _passing_checks(tmp_path)

    report = regression_gate.build_gate_report(
        checks,
        repo_root=tmp_path,
        code_version="WORKTREE",
        generated_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
    )

    assert report["schema_version"] == "ace.offline-gate.v1"
    assert report["status"] == "PASS"
    assert report["gate_pass"] is True
    assert report["full_gate_coverage"] is True
    assert report["missing_required_checks"] == []
    assert len(report["protocol_fingerprint"]) == 64
    assert report["code_version"] == "WORKTREE"
    assert report["generated_at"] == "2026-07-12T00:00:00+00:00"
    assert report["what_this_does_not_prove"]

    by_id = {item["check_id"]: item for item in report["checks"]}
    for source in checks:
        item = by_id[source.check_id]
        expected_bytes = source.artifact_path.read_bytes()
        assert item["artifact"]["path"] == (
            f"evidence/{source.check_id}.json"
        )
        assert item["artifact"]["bytes"] == len(expected_bytes)
        assert item["artifact"]["sha256"] == hashlib.sha256(expected_bytes).hexdigest()
        assert not item["artifact"]["path"].startswith(("/", "\\"))


def test_missing_required_check_is_inconclusive_not_a_partial_pass(tmp_path):
    checks = _passing_checks(tmp_path)[:-1]

    report = regression_gate.build_gate_report(
        checks,
        repo_root=tmp_path,
        code_version="WORKTREE",
    )

    assert report["status"] == "INCONCLUSIVE"
    assert report["gate_pass"] is False
    assert report["full_gate_coverage"] is False
    assert report["missing_required_checks"] == [
        regression_gate.REQUIRED_CHECK_IDS[-1]
    ]


@pytest.mark.parametrize(
    ("status", "full_coverage", "expected_gate_status"),
    [
        ("FAIL", True, "FAIL"),
        ("ERROR", True, "ERROR"),
        ("INCONCLUSIVE", True, "INCONCLUSIVE"),
        ("SKIPPED", True, "INCONCLUSIVE"),
        ("PASS", False, "INCONCLUSIVE"),
    ],
)
def test_nonpassing_or_partial_required_check_cannot_pass_gate(
    tmp_path,
    status,
    full_coverage,
    expected_gate_status,
):
    checks = _passing_checks(tmp_path)
    original = checks[0]
    checks[0] = regression_gate.GateCheck(
        check_id=original.check_id,
        status=status,
        full_coverage=full_coverage,
        claim_scope=original.claim_scope,
        command=original.command,
        artifact_path=original.artifact_path,
        message="counterexample",
    )

    report = regression_gate.build_gate_report(
        checks,
        repo_root=tmp_path,
        code_version="WORKTREE",
    )

    assert report["status"] == expected_gate_status
    assert report["gate_pass"] is False


def test_duplicate_check_ids_are_rejected(tmp_path):
    checks = _passing_checks(tmp_path)

    with pytest.raises(ValueError, match="duplicate check_id"):
        regression_gate.build_gate_report(
            [*checks, checks[0]],
            repo_root=tmp_path,
            code_version="WORKTREE",
        )


def test_artifact_outside_repository_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    checks = _passing_checks(tmp_path)

    with pytest.raises(ValueError, match="outside repository"):
        regression_gate.build_gate_report(
            checks,
            repo_root=repo,
            code_version="WORKTREE",
        )


def test_write_report_is_utf8_json_and_creates_parent(tmp_path):
    report = regression_gate.build_gate_report(
        _passing_checks(tmp_path),
        repo_root=tmp_path,
        code_version="WORKTREE",
    )
    output = tmp_path / "nested" / "offline-gate.json"

    regression_gate.write_report(output, report)

    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize(
    ("check_id", "relative_path"),
    [
        ("context_budget", "docs/evals/evidence/context-budget.jsonl"),
        ("mcp_smoke", "docs/evals/evidence/mcp-smoke.jsonl"),
        ("mcp_reliability", "docs/evals/evidence/mcp-reliability.jsonl"),
        (
            "mcp_benefit_synthetic",
            "docs/evals/evidence/mcp-benefit-synthetic.jsonl",
        ),
        (
            "swe_checker_selftest",
            "docs/evals/evidence/swe-checker-selftest.json",
        ),
    ],
)
def test_current_track_evidence_passes_strict_inspection(check_id, relative_path):
    verdict = regression_gate.inspect_track_evidence(
        check_id,
        ROOT / relative_path,
    )

    assert verdict.status == "PASS", verdict.message
    assert verdict.full_coverage is True


def test_context_summary_cannot_claim_pass_without_full_coverage(tmp_path):
    source = ROOT / "docs/evals/evidence/context-budget.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    records[0]["full_gate_coverage"] = False
    target = tmp_path / "context.jsonl"
    target.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    verdict = regression_gate.inspect_track_evidence("context_budget", target)

    assert verdict.status == "ERROR"
    assert verdict.full_coverage is False
    assert "inconsistent" in verdict.message


def test_context_counts_and_current_protocol_are_recomputed(tmp_path):
    source = ROOT / "docs/evals/evidence/context-budget.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    records[0]["counts"]["PASS"] = 0
    records[0]["counts"]["FAIL"] = 999
    records[0]["protocol_sha256"] = "0" * 64
    for record in records[1:]:
        record["protocol_sha256"] = "0" * 64
    target = tmp_path / "context.jsonl"
    target.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    verdict = regression_gate.inspect_track_evidence("context_budget", target)

    assert verdict.status == "ERROR"
    assert verdict.full_coverage is False


def test_context_metadata_and_requiredness_are_strict(tmp_path):
    source = ROOT / "docs/evals/evidence/context-budget.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    records[0].pop("environment")
    records[0]["valid_statuses"] = ["FAIL", "PASS"]
    records[0]["excluded_statuses"] = ["ERROR"]
    for index, record in enumerate(records[1:]):
        record["required"] = False
        record["protocol_version"] = "wrong"
        record["timestamp_utc"] = f"2026-07-12T00:00:0{index}Z"
        record.pop("what_this_does_not_prove")
    target = tmp_path / "context.jsonl"
    target.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    verdict = regression_gate.inspect_track_evidence("context_budget", target)

    assert verdict.status == "ERROR"
    assert verdict.full_coverage is False


def test_mcp_summary_cannot_hide_a_missing_case_record(tmp_path):
    source = ROOT / "docs/evals/evidence/mcp-smoke.jsonl"
    lines = source.read_text(encoding="utf-8").splitlines()
    target = tmp_path / "mcp.jsonl"
    target.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    verdict = regression_gate.inspect_track_evidence("mcp_smoke", target)

    assert verdict.status == "ERROR"
    assert verdict.full_coverage is False
    assert "record coverage" in verdict.message


def test_mcp_case_run_metadata_must_match_summary(tmp_path):
    source = ROOT / "docs/evals/evidence/mcp-smoke.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    records[1]["run"]["api_calls"] = 99
    records[1]["run"]["llm_calls"] = 7
    target = tmp_path / "mcp.jsonl"
    target.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    verdict = regression_gate.inspect_track_evidence(
        "mcp_smoke",
        target,
        repo_root=ROOT,
    )

    assert verdict.status == "ERROR"
    assert verdict.full_coverage is False


def test_mcp_run_identity_and_environment_schema_are_strict(tmp_path):
    source = ROOT / "docs/evals/evidence/mcp-smoke.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    forged_run = {
        **records[0]["run"],
        "timestamp_utc": "not-a-time",
        "run_id": "forged",
        "environment": {"path": "C:/sensitive/example"},
    }
    for record in records:
        record["run"] = forged_run
    target = tmp_path / "mcp.jsonl"
    target.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    verdict = regression_gate.inspect_track_evidence(
        "mcp_smoke",
        target,
        repo_root=ROOT,
    )

    assert verdict.status == "ERROR"
    assert verdict.full_coverage is False


def test_synthetic_checker_must_remain_gate_ineligible(tmp_path):
    source = ROOT / "docs/evals/evidence/swe-checker-selftest.json"
    evidence = json.loads(source.read_text(encoding="utf-8"))
    evidence["gate_eligible"] = True
    target = tmp_path / "swe.json"
    target.write_text(json.dumps(evidence), encoding="utf-8")

    verdict = regression_gate.inspect_track_evidence(
        "swe_checker_selftest",
        target,
    )

    assert verdict.status == "ERROR"
    assert verdict.full_coverage is False
    assert "gate_eligible" in verdict.message


def test_swe_checker_recomputes_protocol_fixture_and_resolution(tmp_path):
    source = ROOT / "docs/evals/evidence/swe-checker-selftest.json"
    evidence = json.loads(source.read_text(encoding="utf-8"))
    evidence["protocol_fingerprint"] = "0" * 64
    evidence["instance_outcomes"][0]["computed_resolved"] = False
    evidence["instance_outcomes"][0]["test_statuses"] = {}
    target = tmp_path / "swe.json"
    target.write_text(json.dumps(evidence), encoding="utf-8")

    verdict = regression_gate.inspect_track_evidence(
        "swe_checker_selftest",
        target,
        repo_root=ROOT,
    )

    assert verdict.status == "ERROR"
    assert verdict.full_coverage is False


def test_swe_checker_rejects_missing_metadata_and_score_fields(tmp_path):
    source = ROOT / "docs/evals/evidence/swe-checker-selftest.json"
    evidence = json.loads(source.read_text(encoding="utf-8"))
    evidence.pop("timestamp_utc")
    evidence.pop("environment")
    evidence["agent_score"] = 1.0
    evidence["resolved_rate"] = 1.0
    target = tmp_path / "swe.json"
    target.write_text(json.dumps(evidence), encoding="utf-8")

    verdict = regression_gate.inspect_track_evidence(
        "swe_checker_selftest",
        target,
        repo_root=ROOT,
    )

    assert verdict.status == "ERROR"
    assert verdict.full_coverage is False


def test_evidence_code_version_must_match_requested_gate_version():
    verdict = regression_gate.inspect_track_evidence(
        "context_budget",
        ROOT / "docs/evals/evidence/context-budget.jsonl",
        repo_root=ROOT,
        expected_code_version="DIFFERENT123",
    )

    assert verdict.status == "ERROR"
    assert verdict.full_coverage is False
    assert "code_version" in verdict.message


def test_explicit_commit_code_version_must_match_clean_head(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Evidence Test")
    git("config", "user.email", "evidence@example.invalid")
    tracked = repo / "tracked.txt"
    tracked.write_text("v1\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "--quiet", "-m", "fixture")
    full_revision = git("rev-parse", "HEAD")
    short_revision = full_revision[:12]

    assert (
        regression_gate._resolve_offline_code_version(repo, short_revision)
        == short_revision
    )
    assert (
        regression_gate._resolve_offline_code_version(repo, full_revision)
        == full_revision
    )
    assert regression_gate._resolve_offline_code_version(repo, "WORKTREE") == "WORKTREE"

    with pytest.raises(ValueError, match="does not resolve to clean HEAD"):
        regression_gate._resolve_offline_code_version(repo, "deadbeef")

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires a clean Git worktree"):
        regression_gate._resolve_offline_code_version(repo, short_revision)


_TRACK_EVIDENCE = {
    "context_budget": "context-budget.jsonl",
    "mcp_smoke": "mcp-smoke.jsonl",
    "mcp_reliability": "mcp-reliability.jsonl",
    "mcp_benefit_synthetic": "mcp-benefit-synthetic.jsonl",
    "swe_checker_selftest": "swe-checker-selftest.json",
}


def test_offline_plan_contains_every_required_check_once(tmp_path):
    specs = regression_gate.offline_command_specs(
        repo_root=ROOT,
        evidence_dir=tmp_path,
        code_version="WORKTREE",
    )

    assert [spec.check_id for spec in specs] == list(
        regression_gate.REQUIRED_CHECK_IDS
    )
    assert specs[0].artifact_path == tmp_path / "unit-tests.xml"
    assert all(spec.command[0] == regression_gate.sys.executable for spec in specs)
    assert all(str(ROOT) not in " ".join(spec.command) for spec in specs[:1])
    assert {
        spec.artifact_path.name
        for spec in specs
        if spec.artifact_path is not None
    } == {"unit-tests.xml", *_TRACK_EVIDENCE.values()}


def _copying_runner(*, failing_check_id=None, omit_artifact_for=None):
    calls = []

    def run(spec, cwd):
        calls.append((spec.check_id, tuple(spec.command), cwd))
        _prepare_protocol_repo(cwd)
        if spec.artifact_path is not None and spec.check_id != omit_artifact_for:
            spec.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            if spec.check_id == "unit_tests":
                failures = 1 if spec.check_id == failing_check_id else 0
                cases = [
                    '<testcase classname="test_sample" name="test_skip_1"><skipped /></testcase>',
                    '<testcase classname="test_sample" name="test_skip_2"><skipped /></testcase>',
                ]
                cases.extend(
                    f'<testcase classname="{classname}" name="{name}" />'
                    for classname, name in regression_gate.REQUIRED_UNIT_SENTINELS
                )
                cases.extend(
                    f'<testcase classname="test_sample" name="test_ok_{index}" />'
                    for index in range(
                        regression_gate.MINIMUM_UNIT_TESTS - len(cases) - 1
                    )
                )
                cases.append(
                    '<testcase classname="test_sample" name="test_last"><failure /></testcase>'
                    if failures
                    else '<testcase classname="test_sample" name="test_last" />'
                )
                spec.artifact_path.write_text(
                    (
                        '<?xml version="1.0" encoding="utf-8"?>'
                        '<testsuites name="pytest tests">'
                        f'<testsuite name="pytest" errors="0" failures="{failures}" '
                        f'skipped="2" tests="{regression_gate.MINIMUM_UNIT_TESTS}" '
                        'time="0.1" timestamp="2026-07-12T00:00:00Z">'
                        + "".join(cases)
                        +
                        '</testsuite></testsuites>'
                    ),
                    encoding="utf-8",
                )
            else:
                source = ROOT / "docs" / "evals" / "evidence" / _TRACK_EVIDENCE[spec.check_id]
                shutil.copy2(source, spec.artifact_path)
        return regression_gate.CommandResult(
            returncode=1 if spec.check_id == failing_check_id else 0,
            stdout="test output",
            stderr="test error" if spec.check_id == failing_check_id else "",
        )

    return run, calls


def _prepare_protocol_repo(repo_root):
    if repo_root.resolve() == ROOT:
        return
    relative_paths = {
        "eval/mcp_eval/evidence.py",
        "eval/mcp_eval/cases.py",
        "eval/mcp_eval/smoke.py",
        "eval/mcp_eval/reliability_cases.py",
        "eval/mcp_eval/reliability.py",
        "eval/mcp_eval/benefit_cases.py",
        "eval/mcp_eval/benefit.py",
        "eval/swebench/checker_fixture/suite.json",
        "eval/swebench/checker_fixture/results.jsonl",
        "eval/swebench/checker_fixture/baseline.fixture.json",
    }
    for relative in relative_paths:
        source = ROOT / relative
        target = repo_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)


def test_offline_gate_runs_all_checks_inspects_artifacts_and_writes_report(tmp_path):
    runner, calls = _copying_runner()
    evidence_dir = tmp_path / "evidence"

    report = regression_gate.run_offline_gate(
        repo_root=tmp_path,
        evidence_dir=evidence_dir,
        code_version="WORKTREE",
        runner=runner,
        generated_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
    )

    assert report["status"] == "PASS"
    assert report["gate_pass"] is True
    assert [call[0] for call in calls] == list(regression_gate.REQUIRED_CHECK_IDS)
    assert all(call[2] == tmp_path for call in calls)
    assert json.loads((evidence_dir / "offline-gate.json").read_text(encoding="utf-8")) == report
    by_id = {item["check_id"]: item for item in report["checks"]}
    assert by_id["unit_tests"]["artifact"]["path"] == "evidence/unit-tests.xml"
    assert f"tests={regression_gate.MINIMUM_UNIT_TESTS}" in by_id["unit_tests"]["message"]
    assert "skipped=2" in by_id["unit_tests"]["message"]
    assert all(
        by_id[check_id]["artifact"]["sha256"]
        for check_id in ("unit_tests", *_TRACK_EVIDENCE)
    )


def test_missing_track_artifact_is_error_and_cannot_partially_pass(tmp_path):
    runner, _ = _copying_runner(omit_artifact_for="mcp_smoke")

    report = regression_gate.run_offline_gate(
        repo_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        code_version="WORKTREE",
        runner=runner,
    )

    by_id = {item["check_id"]: item for item in report["checks"]}
    assert report["status"] == "ERROR"
    assert report["gate_pass"] is False
    assert by_id["mcp_smoke"]["status"] == "ERROR"
    assert by_id["mcp_smoke"]["full_coverage"] is False


def test_gate_version_cannot_be_relabelled_over_old_track_evidence(tmp_path):
    runner, _ = _copying_runner()

    report = regression_gate.run_offline_gate(
        repo_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        code_version="DIFFERENT123",
        runner=runner,
    )

    assert report["code_version"] == "DIFFERENT123"
    assert report["status"] == "ERROR"
    assert report["gate_pass"] is False
    assert all(
        item["status"] == "ERROR"
        for item in report["checks"]
        if item["check_id"] != "unit_tests"
    )


def test_zero_exit_with_invalid_evidence_is_error(tmp_path):
    evidence_dir = tmp_path / "evidence"

    def runner(spec, cwd):
        del cwd
        if spec.artifact_path is not None:
            spec.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            spec.artifact_path.write_text("{}\n", encoding="utf-8")
        return regression_gate.CommandResult(returncode=0)

    report = regression_gate.run_offline_gate(
        repo_root=tmp_path,
        evidence_dir=evidence_dir,
        code_version="WORKTREE",
        runner=runner,
    )

    assert report["status"] == "ERROR"
    assert report["gate_pass"] is False
    assert all(
        item["status"] == "ERROR"
        for item in report["checks"]
    )


def test_evidence_replaced_after_inspection_cannot_pass(tmp_path):
    runner, _ = _copying_runner()

    def tampering_runner(spec, cwd):
        result = runner(spec, cwd)
        if spec.check_id == "swe_checker_selftest":
            context_path = spec.artifact_path.parent / "context-budget.jsonl"
            context_path.write_text('{"tampered_after_inspection":true}\n', encoding="utf-8")
        return result

    report = regression_gate.run_offline_gate(
        repo_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        code_version="WORKTREE",
        runner=tampering_runner,
    )

    by_id = {item["check_id"]: item for item in report["checks"]}
    assert report["status"] == "ERROR"
    assert report["gate_pass"] is False
    assert by_id["context_budget"]["status"] == "ERROR"
    assert "changed after inspection" in by_id["context_budget"]["message"]


@pytest.mark.parametrize(
    ("returncode", "expected_status"),
    [(1, "FAIL"), (2, "ERROR"), (5, "ERROR")],
)
def test_pytest_exit_code_is_classified_without_skipping_other_tracks(
    tmp_path,
    returncode,
    expected_status,
):
    runner, calls = _copying_runner(failing_check_id="unit_tests")

    def classified_runner(spec, cwd):
        result = runner(spec, cwd)
        if spec.check_id == "unit_tests":
            return regression_gate.CommandResult(returncode=returncode)
        return result

    report = regression_gate.run_offline_gate(
        repo_root=tmp_path,
        evidence_dir=tmp_path / "evidence",
        code_version="WORKTREE",
        runner=classified_runner,
    )

    by_id = {item["check_id"]: item for item in report["checks"]}
    assert by_id["unit_tests"]["status"] == expected_status
    assert len(calls) == len(regression_gate.REQUIRED_CHECK_IDS)
    assert report["gate_pass"] is False


def test_nonempty_output_directory_is_rejected_to_preserve_prior_round(tmp_path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "unrelated-note.txt").write_text("old round", encoding="utf-8")
    runner, calls = _copying_runner()

    with pytest.raises(FileExistsError, match="prior round"):
        regression_gate.run_offline_gate(
            repo_root=tmp_path,
            evidence_dir=evidence_dir,
            code_version="WORKTREE",
            runner=runner,
        )

    assert calls == []


def test_unknown_required_check_is_rejected(tmp_path):
    checks = _passing_checks(tmp_path)
    checks.append(
        regression_gate.GateCheck(
            check_id="extra_required",
            status="FAIL",
            full_coverage=True,
            claim_scope="extra",
            required=True,
        )
    )

    with pytest.raises(ValueError, match="unexpected required"):
        regression_gate.build_gate_report(
            checks,
            repo_root=tmp_path,
            code_version="WORKTREE",
        )


@pytest.mark.parametrize(
    ("required", "claim_scope"),
    [(False, regression_gate.CLAIM_SCOPES["unit_tests"]), (True, "wrong-scope")],
)
def test_fixed_required_check_contract_cannot_be_weakened(
    tmp_path,
    required,
    claim_scope,
):
    checks = _passing_checks(tmp_path)
    original = checks[0]
    checks[0] = regression_gate.GateCheck(
        check_id=original.check_id,
        status=original.status,
        full_coverage=original.full_coverage,
        claim_scope=claim_scope,
        artifact_path=original.artifact_path,
        required=required,
    )

    with pytest.raises(ValueError, match="invalid required check contract"):
        regression_gate.build_gate_report(
            checks,
            repo_root=tmp_path,
            code_version="WORKTREE",
        )


def test_aggregate_protocol_fingerprint_covers_material_rules():
    descriptor = regression_gate.AGGREGATE_PROTOCOL_DESCRIPTOR

    assert {
        "aggregate_claim",
        "aggregation",
        "code_version_provenance",
        "command_plan",
        "evidence_inspection",
        "required_checks",
        "synthetic_checker_semantics",
    } <= set(descriptor)
    changed = json.loads(json.dumps(descriptor))
    changed["aggregation"]["status_precedence"] = ["PASS"]

    assert regression_gate._fingerprint_protocol_descriptor(changed) != (
        regression_gate.protocol_fingerprint()
    )


def test_swe_checker_selftest_subcommand_delegates_without_enabling_fixture(
    monkeypatch,
    tmp_path,
):
    captured = []

    def fake_checker(argv):
        captured.extend(argv)
        return 7

    monkeypatch.setattr(regression_gate, "_swe_checker_main", fake_checker)
    paths = [tmp_path / name for name in ("suite.json", "results.jsonl", "baseline.json")]

    returncode = regression_gate.main(
        [
            "swe-checker-selftest",
            "--suite",
            str(paths[0]),
            "--results",
            str(paths[1]),
            "--baseline",
            str(paths[2]),
            "--code-version",
            "abc123",
        ]
    )

    assert returncode == 7
    assert "--allow-fixture-baseline" not in captured


def test_swe_checker_selftest_script_entrypoint_can_import_checker_and_rejects_fixture(
    tmp_path,
):
    fixture = ROOT / "eval" / "swebench" / "checker_fixture"
    completed = subprocess.run(
        [
            regression_gate.sys.executable,
            "scripts/release_gate.py",
            "swe-checker-selftest",
            "--suite",
            str(fixture / "suite.json"),
            "--results",
            str(fixture / "results.jsonl"),
            "--baseline",
            str(fixture / "baseline.fixture.json"),
            "--output",
            str(tmp_path / "evidence.json"),
            "--code-version",
            "WORKTREE",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 2
    assert "ModuleNotFoundError" not in completed.stderr
    assert "FixtureBaselineRejected" in completed.stderr


def test_script_entrypoint_initializes_repository_import_path(monkeypatch):
    root_text = str(ROOT)
    monkeypatch.setattr(
        regression_gate.sys,
        "path",
        [entry for entry in regression_gate.sys.path if entry != root_text],
    )

    regression_gate._ensure_repo_import_path()

    assert regression_gate.sys.path[0] == root_text


def test_subprocess_runner_does_not_reemit_untrusted_unicode(monkeypatch, tmp_path):
    class AsciiOnlyStream:
        encoding = "ascii"

        def write(self, value):
            value.encode("ascii")

        def flush(self):
            return None

    monkeypatch.setattr(
        regression_gate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="collected 1 item \ufeff\n",
            stderr="warning \u2014 diagnostic\n",
        ),
    )
    monkeypatch.setattr(regression_gate.sys, "stdout", AsciiOnlyStream())
    monkeypatch.setattr(regression_gate.sys, "stderr", AsciiOnlyStream())
    spec = regression_gate.OfflineCommandSpec(
        check_id="unit_tests",
        command=(regression_gate.sys.executable, "-m", "pytest"),
    )

    result = regression_gate._subprocess_runner(spec, tmp_path)

    assert result.returncode == 0
    assert "\ufeff" in result.stdout
    assert "\u2014" in result.stderr


def test_console_path_sanitizer_handles_repr_escaped_windows_paths():
    raw = repr(str(ROOT / "eval" / "reports" / "round.json"))

    sanitized = regression_gate._sanitize_console_paths(raw, ROOT)

    assert "D:" not in sanitized
    assert str(ROOT).replace("\\", "\\\\") not in sanitized
    assert ROOT.as_posix() not in sanitized
    assert "<REPO_ROOT>" in sanitized


def test_junit_artifact_paths_are_structurally_redacted(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    junit = repo / "unit-tests.xml"
    source_path = repo / "tests" / "test_sample.py"
    temp_path = tmp_path / "outside.txt"
    junit.write_text(
        (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<testsuites><testsuite errors="0" failures="0" skipped="1" tests="1" '
            'hostname="PRIVATE-HOST">'
            '<testcase classname="test_sample" name="test_skip">'
            f'<skipped>{source_path}: skipped via {temp_path}</skipped>'
            '</testcase></testsuite></testsuites>'
        ),
        encoding="utf-8",
    )

    regression_gate._sanitize_junit_artifact(junit, repo)
    text = junit.read_text(encoding="utf-8")
    verdict = regression_gate._inspect_pytest_junit(junit)

    assert str(repo) not in text
    assert str(tmp_path) not in text
    assert "&lt;REPO_ROOT&gt;" in text
    assert "PRIVATE-HOST" not in text
    assert "hostname=" not in text
    assert re.search(r"pytest-of-[A-Za-z0-9_.-]+", text) is None
    assert re.search(r"pytest-\d+", text) is None
    assert verdict.status == "INCONCLUSIVE"
    assert "skipped=1" in verdict.message


def test_junit_artifact_redacts_synthetic_credential_values(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    junit = repo / "unit-tests.xml"
    anthropic = "sk-" + "ant-api03-" + "A1b2_C3d4-E5f6_G7h8-I9j0_K1l2"
    aws = "AK" + "IA" + "A1B2C3D4E5F6G7H8"
    private_key = "-----BEGIN " + "PRIVATE KEY-----"
    junit.write_text(
        (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<testsuites><testsuite errors="0" failures="0" skipped="0" tests="1">'
            f'<testcase classname="tests.test_secret_scan" name="case[{anthropic}]">'
            f'<system-out>{aws} {private_key}</system-out>'
            '</testcase></testsuite></testsuites>'
        ),
        encoding="utf-8",
    )

    regression_gate._sanitize_junit_artifact(junit, repo)
    text = junit.read_text(encoding="utf-8")

    assert anthropic not in text
    assert aws not in text
    assert private_key not in text
    assert text.count("&lt;CREDENTIAL&gt;") >= 3


def test_all_skipped_or_collapsed_junit_cannot_pass(tmp_path):
    junit = tmp_path / "unit-tests.xml"
    junit.write_text(
        (
            '<testsuites><testsuite errors="0" failures="0" skipped="2" tests="2">'
            '<testcase classname="sample" name="a"><skipped /></testcase>'
            '<testcase classname="sample" name="b"><skipped /></testcase>'
            '</testsuite></testsuites>'
        ),
        encoding="utf-8",
    )

    verdict = regression_gate._inspect_pytest_junit(junit)

    assert verdict.status == "INCONCLUSIVE"
    assert verdict.full_coverage is False
    assert "executed=0" in verdict.message


def test_duplicate_junit_testcase_ids_cannot_satisfy_collection_floor(tmp_path):
    junit = tmp_path / "unit-tests.xml"
    cases = [
        f'<testcase classname="{classname}" name="{name}" />'
        for classname, name in regression_gate.REQUIRED_UNIT_SENTINELS
    ]
    duplicate = '<testcase classname="duplicate" name="same" />'
    cases.extend(
        duplicate
        for _ in range(regression_gate.MINIMUM_UNIT_TESTS - len(cases))
    )
    junit.write_text(
        (
            '<testsuites><testsuite errors="0" failures="0" skipped="0" '
            f'tests="{regression_gate.MINIMUM_UNIT_TESTS}">'
            + "".join(cases)
            + '</testsuite></testsuites>'
        ),
        encoding="utf-8",
    )

    verdict = regression_gate._inspect_pytest_junit(junit)

    assert verdict.status == "INCONCLUSIVE"
    assert verdict.full_coverage is False
    assert "duplicate_testcase_ids" in verdict.message


@pytest.mark.parametrize(
    "path_value",
    ["/opt/private/tool", r"\\server\share\secret", "file:///tmp/secret"],
)
def test_swe_environment_rejects_posix_unc_and_file_uri_paths(tmp_path, path_value):
    source = ROOT / "docs/evals/evidence/swe-checker-selftest.json"
    evidence = json.loads(source.read_text(encoding="utf-8"))
    evidence["environment"]["machine"] = path_value
    target = tmp_path / "swe.json"
    target.write_text(json.dumps(evidence), encoding="utf-8")

    verdict = regression_gate.inspect_track_evidence(
        "swe_checker_selftest",
        target,
        repo_root=ROOT,
    )

    assert verdict.status == "ERROR"
    assert verdict.full_coverage is False
