"""Offline tests for the MCP benefit pair and its real fixture grader."""

from __future__ import annotations

import json
from dataclasses import replace


def test_hidden_grader_distinguishes_fixture_and_nonce_patch(tmp_path):
    from eval.mcp_eval.benefit_cases import (
        apply_known_good_patch,
        create_benefit_fixture,
        grade_benefit_workspace,
    )

    workspace = tmp_path / "workspace"
    create_benefit_fixture(workspace)
    assert grade_benefit_workspace(workspace, "n0nce").passed is False
    apply_known_good_patch(workspace, "n0nce")
    assert grade_benefit_workspace(workspace, "n0nce").passed is True
    assert grade_benefit_workspace(workspace, "wrong").passed is False


def test_pair_verdict_precedence_and_causal_requirements():
    from eval.mcp_eval.benefit_cases import (
        ERROR,
        FAIL,
        INVALID,
        PASS,
        SKIPPED,
        BenefitConditionResult,
        grade_benefit_pair,
    )

    def condition(name, *, passed=False, called=False, before=False, status="OK"):
        return BenefitConditionResult(
            condition=name,
            status=status,
            duration_ms=1,
            hidden_tests_passed=passed,
            issue_tool_called=called,
            issue_call_before_target_write=before,
        )

    control_fail = condition("MCP unavailable")
    treatment_pass = condition(
        "MCP issue context available", passed=True, called=True, before=True
    )
    assert grade_benefit_pair(control_fail, treatment_pass)[0] == PASS
    assert grade_benefit_pair(condition("control", passed=True), treatment_pass)[0] == INVALID
    assert grade_benefit_pair(control_fail, condition("treatment", passed=True))[0] == INVALID
    assert grade_benefit_pair(control_fail, condition("treatment"))[0] == FAIL
    assert grade_benefit_pair(condition("control", status=SKIPPED), treatment_pass)[0] == SKIPPED
    assert grade_benefit_pair(condition("control", status=ERROR), treatment_pass)[0] == ERROR


def test_causal_order_falls_back_to_trace_events():
    from eval.mcp_eval.benefit_cases import _causal_order

    events = [
        {
            "name": "tool.mcp__issue__get_issue",
            "attributes": {"tool.name": "mcp__issue__get_issue"},
        },
        {
            "name": "tool.edit_file",
            "attributes": {"tool.name": "edit_file", "tool.display.path": "orders.py"},
        },
    ]
    assert _causal_order([], events) == (True, True, "events")


def test_shell_before_mcp_is_a_conservative_mutation_boundary():
    from eval.mcp_eval.benefit_cases import _causal_order

    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": "bash",
                    "input": {"command": "Set-Content orders.py hacked"},
                },
                {
                    "type": "tool_use",
                    "name": "mcp__issue__get_issue",
                    "input": {"issue_id": "ACE-MCP-001"},
                },
            ],
        }
    ]
    assert _causal_order(messages) == (True, False, "messages")


def test_benefit_fake_mode_runs_real_fixture_grader(tmp_path):
    from eval.mcp_eval.benefit_cases import PASS, run_benefit_pairs, summarize_benefit_pairs

    results = run_benefit_pairs(mode="fake", root=tmp_path, repeat=2)
    assert len(results) == 2
    assert all(result.status == PASS for result in results)
    assert all(result.control.grader["passed"] is False for result in results)
    assert all(result.treatment.grader["passed"] is True for result in results)
    assert all(result.treatment.issue_call_before_target_write for result in results)
    summary = summarize_benefit_pairs(results)
    assert summary["gate_pass"] is True
    assert summary["gate_status"] == "PASS"
    assert summary["claim"] == "harness_self_test_only"

    live_results = [replace(result, mode="live") for result in results]
    assert summarize_benefit_pairs(live_results)["claim"] == (
        "observations_consistent_with_mcp_benefit"
    )


def test_all_skipped_is_not_presented_as_gate_pass():
    from eval.mcp_eval.benefit_cases import (
        CASE_ID,
        CONTROL,
        SKIPPED,
        TREATMENT,
        BenefitConditionResult,
        BenefitPairResult,
        summarize_benefit_pairs,
    )

    control = BenefitConditionResult(CONTROL, SKIPPED, 0)
    treatment = BenefitConditionResult(TREATMENT, SKIPPED, 0)
    pair = BenefitPairResult(
        case_id=CASE_ID,
        pair_index=1,
        nonce="n",
        mode="live",
        model_id="m",
        order=(CONTROL, TREATMENT),
        status=SKIPPED,
        control=control,
        treatment=treatment,
    )
    summary = summarize_benefit_pairs([pair])
    assert summary["gate_pass"] is True
    assert summary["gate_status"] == SKIPPED
    assert summary["claim"] == "no_positive_benefit_claim"


def test_live_condition_restores_previous_trace_sink(tmp_path, monkeypatch):
    from agent import loop
    from eval.mcp_eval.benefit_cases import CONTROL, _run_live_condition, create_benefit_fixture
    from obs.trace import get_sink, set_sink

    class _Sink:
        def emit(self, span):
            return None

    previous = _Sink()
    set_sink(previous)
    workspace = tmp_path / "workspace"
    create_benefit_fixture(workspace)
    monkeypatch.setattr(
        loop,
        "run_task",
        lambda *args, **kwargs: ("done", [{"role": "assistant", "content": []}]),
    )
    _run_live_condition(
        condition=CONTROL,
        workspace=workspace,
        nonce="nonce",
        config_path=tmp_path / "unused.json",
        max_turns=1,
    )
    assert get_sink() is previous


def test_live_control_finishes_before_treatment_or_nonce_config_exists(tmp_path, monkeypatch):
    import eval.mcp_eval.benefit_cases as cases

    observed_orders = []

    def fake_run(*, condition, workspace, nonce, config_path, max_turns):
        if condition == cases.CONTROL:
            pair_root = config_path.parent.parent
            assert not config_path.exists()
            assert not (pair_root / "treatment" / "workspace").exists()
            return cases.BenefitConditionResult(
                condition=condition,
                status=cases.OK,
                duration_ms=1,
                hidden_tests_passed=False,
            )
        return cases.BenefitConditionResult(
            condition=condition,
            status=cases.OK,
            duration_ms=1,
            hidden_tests_passed=True,
            issue_tool_called=True,
            issue_call_before_target_write=True,
        )

    monkeypatch.setattr(cases, "_live_dependencies", lambda: "")
    monkeypatch.setattr(cases, "_run_live_condition", fake_run)
    results = cases.run_benefit_pairs(mode="live", root=tmp_path, repeat=2)
    observed_orders.extend(result.order for result in results)
    assert observed_orders == [
        (cases.CONTROL, cases.TREATMENT),
        (cases.CONTROL, cases.TREATMENT),
    ]
    assert all(result.status == cases.PASS for result in results)


def test_benefit_cli_fake_writes_pair_jsonl(tmp_path):
    from eval.mcp_eval.benefit import main

    output = tmp_path / "benefit.jsonl"
    assert main(["--mode", "fake", "--output", str(output)]) == 0
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "PASS"
    assert records[0]["control"]["condition"] == "MCP unavailable"
    assert records[0]["treatment"]["condition"] == "MCP issue context available"
