from agent.context import compact
from eval.compression_eval.sm_long_task_fidelity import (
    LONG_TASK_FACTS,
    render_long_task_report,
    run_long_task_fidelity_probe,
)


def test_long_task_fidelity_fake_mode_records_overwrite_and_stale_gates(tmp_path):
    result = run_long_task_fidelity_probe(
        tmp_path,
        live=False,
        full_repeat_count=2,
        extract_count=3,
        distractor_rounds=4,
    )

    assert result.status == "PASS"
    assert result.mode == "fake"
    assert result.capture_gate is True
    assert result.takeover_gate is True
    assert result.same_state_gate is True
    assert result.no_kept_tail_gate is True
    assert result.tail_survival is True
    assert result.overwrite_gate is True
    assert result.stale_correction_gate is True
    assert result.sm_summary_survival_rate == 1.0
    assert result.full_summary_survival_rate == 0.0
    assert result.summary_delta == 1.0
    assert result.extract_count == 3
    assert len(result.extract_output_tokens) == 3
    assert len(result.facts) == len(LONG_TASK_FACTS)
    assert result.summary_max_tokens == compact.DEFAULT_SUMMARY_MAX_TOKENS
    assert all(fact.sm_summary_survival for fact in result.facts)
    assert all(not any(fact.full_summary_survivals) for fact in result.facts)


def test_long_task_fidelity_report_is_self_describing(tmp_path):
    result = run_long_task_fidelity_probe(
        tmp_path,
        live=False,
        full_repeat_count=2,
        extract_count=3,
        distractor_rounds=2,
    )
    report = render_long_task_report(result)

    assert "SessionMemory Long Task Fidelity Probe" in report
    assert "overwrite gate" in report
    assert "stale correction gate" in report
    assert "summary delta" in report
    for fact in LONG_TASK_FACTS:
        assert fact.fact_id in report
