from eval.compression_eval.sm_fidelity import (
    SM_FIDELITY_OLD_FACT,
    SM_FIDELITY_TAIL_FACT,
    render_fidelity_report,
    run_fidelity_smoke,
)


def test_fidelity_smoke_runs_paired_ab_gates(tmp_path):
    result = run_fidelity_smoke(tmp_path)

    assert result.status == "PASS"
    assert result.capture_gate is True
    assert result.takeover_gate is True
    assert result.same_state_gate is True
    assert result.no_kept_tail_gate is True
    assert result.sm_summary_survival is True
    assert result.tail_survival is True
    assert result.full_summary_survivals == [False, False, False]
    assert result.full_summary_survival_rate == 0.0
    assert result.summary_delta == 1.0


def test_fidelity_report_is_self_describing(tmp_path):
    result = run_fidelity_smoke(tmp_path)
    report = render_fidelity_report(result)

    assert "SessionMemory Fidelity Smoke" in report
    assert "capture gate" in report
    assert "takeover gate" in report
    assert "same-state gate" in report
    assert "no-kept-tail gate" in report
    assert SM_FIDELITY_OLD_FACT in report
    assert SM_FIDELITY_TAIL_FACT in report
