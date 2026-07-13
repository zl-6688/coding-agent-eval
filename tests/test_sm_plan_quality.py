from eval.compression_eval.sm_plan_quality import (
    CORRECTION_TEXT,
    WRONG_PLAN_TEXT,
    render_plan_quality_report,
    run_plan_quality_smoke,
)


def test_plan_quality_smoke_detects_stale_plan_conflict(tmp_path):
    result = run_plan_quality_smoke(tmp_path)

    assert result.status == "PASS"
    assert result.risk_status == "RISK_DETECTED"
    assert result.capture_gate is True
    assert result.takeover_gate is True
    assert result.same_state_gate is True
    assert result.correction_tail_gate is True
    assert result.no_old_message_leak_gate is True
    assert result.sm_wrong_plan_survival is True
    assert result.sm_actionable_wrong_plan_survival is True
    assert result.sm_correction_survival is True
    assert result.sm_conflict_survival is True
    assert result.sm_actionable_conflict_survival is True
    assert result.full_wrong_plan_survivals == [False, False, False]
    assert result.full_actionable_wrong_plan_survivals == [False, False, False]
    assert result.full_correction_survivals == [True, True, True]
    assert result.full_conflict_survivals == [False, False, False]
    assert result.full_actionable_conflict_survivals == [False, False, False]


def test_plan_quality_report_is_self_describing(tmp_path):
    result = run_plan_quality_smoke(tmp_path)
    report = render_plan_quality_report(result)

    assert "SessionMemory Plan-Quality Probe" in report
    assert "risk status" in report
    assert "correction-tail gate" in report
    assert "no-old-message-leak gate" in report
    assert WRONG_PLAN_TEXT in report
    assert CORRECTION_TEXT in report
