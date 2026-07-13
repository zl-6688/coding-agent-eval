from eval.compression_eval.sm_capture import SM_CAPTURE_FACTS
from eval.compression_eval.sm_fidelity_live import (
    render_live_fidelity_report,
    run_live_fidelity_probe,
)


def test_live_fidelity_probe_fake_mode_records_all_gates(tmp_path):
    result = run_live_fidelity_probe(tmp_path, live=False, full_repeat_count=3)

    assert result.status == "PASS"
    assert result.mode == "fake"
    assert result.capture_gate is True
    assert result.takeover_gate is True
    assert result.same_state_gate is True
    assert result.no_kept_tail_gate is True
    assert result.tail_survival is True
    assert result.sm_summary_survival_rate == 1.0
    assert result.full_summary_survival_rate == 0.0
    assert result.summary_delta == 1.0
    assert result.full_repeat_count == 3
    assert len(result.facts) == len(SM_CAPTURE_FACTS)
    assert all(fact.sm_summary_survival for fact in result.facts)
    assert all(not any(fact.full_summary_survivals) for fact in result.facts)


def test_live_fidelity_report_is_self_describing(tmp_path):
    result = run_live_fidelity_probe(tmp_path, live=False, full_repeat_count=2)
    report = render_live_fidelity_report(result)

    assert "SessionMemory Live Fidelity Probe" in report
    assert "capture gate" in report
    assert "full summary survival rate" in report
    assert "fake" in report
    for fact in SM_CAPTURE_FACTS:
        assert fact.fact_id in report
