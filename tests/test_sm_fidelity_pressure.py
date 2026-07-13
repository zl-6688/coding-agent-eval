from agent.context import compact
from eval.compression_eval.sm_fidelity_pressure import (
    PRESSURE_TARGET_FACTS,
    render_pressure_report,
    run_pressure_fidelity_probe,
)


def test_pressure_fidelity_fake_mode_records_pressure_gates(tmp_path):
    result = run_pressure_fidelity_probe(tmp_path, live=False, full_repeat_count=3)

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
    assert len(result.facts) == len(PRESSURE_TARGET_FACTS)
    assert result.summary_max_tokens == compact.DEFAULT_SUMMARY_MAX_TOKENS


def test_pressure_fidelity_report_is_self_describing(tmp_path):
    result = run_pressure_fidelity_probe(tmp_path, live=False, full_repeat_count=2)
    report = render_pressure_report(result)

    assert "SessionMemory Pressure Fidelity Probe" in report
    assert "pressure" in report.lower()
    assert "summary delta" in report
    for fact in PRESSURE_TARGET_FACTS:
        assert fact.fact_id in report
