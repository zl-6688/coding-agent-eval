import json
from pathlib import Path

import pytest


def _row(instance_id: str, rep: int, resolved: bool, status: str = "scored") -> dict:
    return {
        "instance_id": instance_id,
        "rep": rep,
        "resolved": resolved,
        "score_status": status,
    }


def test_cli_help_exposes_canonical_regression_commands(capsys):
    from scripts import regression_gate

    with pytest.raises(SystemExit) as exc_info:
        regression_gate.main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "offline" in help_text
    assert "swe-check" in help_text
    assert "swe-baseline" in help_text
    assert "swe-checker-selftest" not in help_text


def test_swe_aggregate_uses_fixed_denominator_and_majority_vote():
    from scripts.regression_gate import aggregate_swe_results

    suite_ids = ["case-a", "case-b", "case-c"]
    rows = [
        _row("case-a", 0, True),
        _row("case-a", 1, True),
        _row("case-a", 2, False),
        _row("case-b", 0, True),
        _row("case-b", 1, False),
        _row("case-b", 2, False),
        _row("case-c", 0, True),
        _row("case-c", 1, True),
        _row("case-c", 2, True),
    ]

    summary = aggregate_swe_results(rows, suite_ids, repeat=3)

    assert summary.denominator == 3
    assert summary.resolved_count == 2
    assert summary.flaky_count == 2
    assert summary.instance_results["case-a"].resolved is True
    assert summary.instance_results["case-b"].resolved is False
    assert summary.instance_results["case-a"].flaky is True
    assert summary.instance_results["case-c"].flaky is False


def test_swe_aggregate_treats_missing_repeats_as_infra_error():
    from scripts.regression_gate import aggregate_swe_results

    summary = aggregate_swe_results(
        [_row("case-a", 0, True), _row("case-b", 0, False)],
        ["case-a", "case-b"],
        repeat=3,
    )

    assert summary.infra_error_count == 2
    assert any("missing reps" in error for error in summary.infra_errors)


def test_swe_aggregate_treats_error_status_as_infra_not_failure():
    from scripts.regression_gate import aggregate_swe_results

    summary = aggregate_swe_results(
        [{"instance_id": "case-a", "rep": 0, "resolved": None, "score_status": "ERROR"}],
        ["case-a"],
        repeat=1,
    )

    assert summary.resolved_count == 0
    assert summary.infra_error_count == 2
    assert any("infra status=ERROR" in error for error in summary.infra_errors)


def test_swe_gate_uses_regression_and_infra_exit_codes(tmp_path):
    from scripts.regression_gate import EXIT_INFRA, EXIT_OK, EXIT_REGRESSION, run_swe_check

    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps({"suite": "tiny", "instances": ["case-a", "case-b"]}),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "suite": "tiny",
                "instances_file": str(suite),
                "repeat": 3,
                "denominator": 2,
                "resolved_count": 2,
                "flaky_count": 0,
                "thresholds": {
                    "max_resolved_drop_abs": 0,
                    "max_flaky_increase_abs": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    passing = tmp_path / "passing.jsonl"
    passing.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _row("case-a", 0, True),
                _row("case-a", 1, True),
                _row("case-a", 2, True),
                _row("case-b", 0, True),
                _row("case-b", 1, True),
                _row("case-b", 2, False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert run_swe_check(suite, baseline, passing).exit_code == EXIT_OK

    regressed = tmp_path / "regressed.jsonl"
    regressed.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _row("case-a", 0, False),
                _row("case-a", 1, False),
                _row("case-a", 2, False),
                _row("case-b", 0, True),
                _row("case-b", 1, True),
                _row("case-b", 2, True),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert run_swe_check(suite, baseline, regressed).exit_code == EXIT_REGRESSION

    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text(json.dumps(_row("case-a", 0, True)) + "\n", encoding="utf-8")
    assert run_swe_check(suite, baseline, incomplete).exit_code == EXIT_INFRA


def test_fixture_baseline_requires_explicit_opt_in(tmp_path):
    from scripts.regression_gate import EXIT_INFRA, EXIT_OK, run_swe_check

    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps({"suite": "tiny", "instances": ["case-a"]}),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "suite": "tiny",
                "not_for_gate": True,
                "repeat": 1,
                "denominator": 1,
                "resolved_count": 1,
                "flaky_count": 0,
                "thresholds": {"max_resolved_drop_abs": 0, "max_flaky_increase_abs": 0},
            }
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(_row("case-a", 0, True)) + "\n", encoding="utf-8")

    assert run_swe_check(suite, baseline, results).exit_code == EXIT_INFRA
    assert run_swe_check(suite, baseline, results, allow_fixture_baseline=True).exit_code == EXIT_OK


def test_swe_baseline_records_protocol_metadata(tmp_path):
    from scripts.regression_gate import EXIT_OK, write_swe_baseline

    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "suite": "tiny",
                "selection_rule": "fixed tiny suite for unit test",
                "instances": ["case-a"],
            }
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n".join(json.dumps(_row("case-a", rep, True)) for rep in range(3)) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "baseline.json"

    gate = write_swe_baseline(
        suite,
        results,
        out,
        model_id="deepseek-test",
        repeat=3,
        temperature="0.2",
        seed="123",
    )

    baseline = json.loads(out.read_text(encoding="utf-8"))
    assert gate.exit_code == EXIT_OK
    assert baseline["model_id"] == "deepseek-test"
    assert baseline["temperature"] == "0.2"
    assert baseline["seed"] == "123"
    assert baseline["repeat"] == 3
    assert baseline["resolved_count"] == 1
    assert baseline["denominator"] == 1
    assert baseline["instances_sha256"]
    assert baseline["generated_at"]


def _runs_module(cmd: list[str], module: str) -> bool:
    return any(
        cmd[index : index + 2] == ["-m", module]
        for index in range(max(0, len(cmd) - 1))
    )


def _assert_ignored_mcp_output(cmd: list[str], filename: str) -> None:
    output_index = cmd.index("--output")
    output = Path(cmd[output_index + 1])
    assert output.as_posix().endswith(f"eval/reports/regression-gate/{filename}")


def test_run_mcp_smoke_propagates_failure(monkeypatch):
    from scripts.regression_gate import run_mcp_smoke

    def fake_run(cmd, cwd=None):
        assert _runs_module(cmd, "eval.mcp_eval.smoke")
        _assert_ignored_mcp_output(cmd, "mcp-smoke.jsonl")

        class _Proc:
            returncode = 1

        return _Proc()

    monkeypatch.setattr("scripts.regression_gate.subprocess.run", fake_run)
    assert run_mcp_smoke() == 1


def test_run_mcp_reliability_propagates_failure(monkeypatch):
    from scripts.regression_gate import run_mcp_reliability

    def fake_run(cmd, cwd=None):
        assert _runs_module(cmd, "eval.mcp_eval.reliability")
        _assert_ignored_mcp_output(cmd, "mcp-reliability.jsonl")
        return type("P", (), {"returncode": 1})()

    monkeypatch.setattr("scripts.regression_gate.subprocess.run", fake_run)
    assert run_mcp_reliability() == 1


def test_run_offline_runs_mcp_smoke_after_pytest(monkeypatch):
    from scripts.regression_gate import run_offline

    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):
        calls.append(list(cmd))
        if "pytest" in cmd:
            return type("P", (), {"returncode": 0})()
        if _runs_module(cmd, "eval.mcp_eval.smoke"):
            return type("P", (), {"returncode": 0})()
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr("scripts.regression_gate.subprocess.run", fake_run)
    assert run_offline(skip_validate_tasks=True) == 0
    assert any(_runs_module(cmd, "eval.mcp_eval.smoke") for cmd in calls)
    assert any(_runs_module(cmd, "eval.mcp_eval.reliability") for cmd in calls)


def test_run_offline_skip_mcp_smoke(monkeypatch):
    from scripts.regression_gate import run_offline

    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):
        calls.append(list(cmd))
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr("scripts.regression_gate.subprocess.run", fake_run)
    assert run_offline(skip_validate_tasks=True, skip_mcp_smoke=True) == 0
    assert not any(_runs_module(cmd, "eval.mcp_eval.smoke") for cmd in calls)
    assert any(_runs_module(cmd, "eval.mcp_eval.reliability") for cmd in calls)


def test_run_offline_skip_mcp_reliability(monkeypatch):
    from scripts.regression_gate import run_offline

    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):
        calls.append(list(cmd))
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr("scripts.regression_gate.subprocess.run", fake_run)
    assert run_offline(skip_validate_tasks=True, skip_mcp_reliability=True) == 0
    assert any(_runs_module(cmd, "eval.mcp_eval.smoke") for cmd in calls)
    assert not any(_runs_module(cmd, "eval.mcp_eval.reliability") for cmd in calls)
