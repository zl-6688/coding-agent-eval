"""Tests for memory-eval outcome classification."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MEMORY_EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MEMORY_EVAL))

from outcome import classify_outcome


def test_track1_incremental_b_fail_is_expected_control_behavior():
    outcome = classify_outcome(
        case_id="H_usr",
        arm="B",
        verdict="FAIL",
        sample_status="VALID",
    )

    assert outcome["expected_verdict"] == "FAIL"
    assert outcome["expectation_met"] is True
    assert outcome["expectation_score"] == 1.0
    assert outcome["outcome_class"] == "expected"


def test_track1_incremental_b_pass_is_unexpected_control_leak_or_weak_case():
    outcome = classify_outcome(
        case_id="H_ref",
        arm="B",
        verdict="PASS",
        sample_status="VALID",
    )

    assert outcome["expected_verdict"] == "FAIL"
    assert outcome["expectation_met"] is False
    assert outcome["expectation_score"] == 0.0
    assert outcome["outcome_class"] == "unexpected"


def test_track1_incremental_a_fail_is_unexpected_memory_failure():
    outcome = classify_outcome(
        case_id="H_usr",
        arm="A",
        verdict="FAIL",
        sample_status="VALID",
    )

    assert outcome["expected_verdict"] == "PASS"
    assert outcome["expectation_met"] is False
    assert outcome["outcome_class"] == "unexpected"


def test_precision_case_expects_both_arms_to_pass():
    for arm in ("A", "B"):
        outcome = classify_outcome(
            case_id="H_prec",
            arm=arm,
            verdict="PASS",
            sample_status="VALID",
        )

        assert outcome["expected_verdict"] == "PASS"
        assert outcome["expectation_met"] is True


def test_runtime_error_is_not_scored_as_eval_expectation():
    outcome = classify_outcome(
        case_id="H_usr",
        arm="A",
        verdict="ERROR",
        sample_status="ERROR",
    )

    assert outcome["expected_verdict"] == "PASS"
    assert outcome["expectation_met"] is None
    assert outcome["expectation_score"] is None
    assert outcome["outcome_class"] == "runtime_error"


def test_invalid_write_gate_sample_is_excluded_not_failed():
    outcome = classify_outcome(
        case_id="H_usr",
        arm="A",
        verdict="WRITE_FAIL",
        sample_status="INVALID",
    )

    assert outcome["expected_verdict"] == "PASS"
    assert outcome["expectation_met"] is None
    assert outcome["outcome_class"] == "excluded"


def test_structurally_excluded_h_fb1_is_not_scored():
    outcome = classify_outcome(
        case_id="H_fb1",
        arm="B",
        verdict="PASS",
        sample_status="VALID",
    )

    assert outcome["expected_verdict"] is None
    assert outcome["expectation_met"] is None
    assert outcome["outcome_class"] == "excluded"


def test_record_adds_outcome_fields_for_expected_control_fail():
    from graders import GradeResult
    from harness import ArmResult
    from run import _record

    result = ArmResult(
        case_id="H_usr",
        arm="B",
        grade=GradeResult("FAIL", "missing Go analogy", ""),
        sample_status="VALID",
    )

    rec = _record(result, run_idx=0)

    assert rec["expected_verdict"] == "FAIL"
    assert rec["expectation_met"] is True
    assert rec["expectation_score"] == 1.0
    assert rec["outcome_class"] == "expected"


def test_root_span_runtime_error_gate_ignores_eval_fail_and_s1_incomplete():
    from graders import GradeResult
    from harness import ArmResult
    from run import _span_should_mark_runtime_error

    eval_fail = ArmResult(
        case_id="H_usr",
        arm="A",
        grade=GradeResult("FAIL", "did not use Go analogy", ""),
        sample_status="VALID",
    )
    s1_incomplete = ArmResult(
        case_id="H_prec",
        arm="A",
        grade=GradeResult("S1_INCOMPLETE", "max turns", ""),
        sample_status="ERROR",
    )
    grader_error = ArmResult(
        case_id="H_usr",
        arm="A",
        grade=GradeResult("SKIP", "grader error", ""),
        sample_status="ERROR",
    )
    runtime_error = ArmResult(
        case_id="H_usr",
        arm="A",
        error="APIConnectionError",
        sample_status="ERROR",
    )

    assert _span_should_mark_runtime_error(eval_fail, "FAIL") is False
    assert _span_should_mark_runtime_error(s1_incomplete, "S1_INCOMPLETE") is False
    assert _span_should_mark_runtime_error(grader_error, "SKIP") is True
    assert _span_should_mark_runtime_error(runtime_error, "ERROR") is True


def test_phoenix_packet_includes_experiment_expectation():
    from to_phoenix import _build_judgment_packet

    packet = _build_judgment_packet(
        {
            "case_id": "H_usr",
            "arm": "B",
            "run_idx": 0,
            "sample_status": "VALID",
            "verdict": "FAIL",
            "reason": "missing Go analogy",
            "transcript": "React state explanation without Go.",
        },
        {"run_id": "unit"},
    )

    assert "expected_verdict=`FAIL`" in packet
    assert "expectation_met=`True`" in packet
    assert "outcome_class=`expected`" in packet
