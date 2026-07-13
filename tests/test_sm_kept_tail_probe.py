from eval.compression_eval.sm_kept_tail_probe import (
    KEPT_TAIL_OLD_FACT,
    KEPT_TAIL_RECENT_FACT,
    KEPT_TAIL_SUMMARY_FACT,
    render_probe_report,
    run_kept_tail_probe,
)


def test_kept_tail_probe_isolates_recent_fact_survival(tmp_path):
    result = run_kept_tail_probe(tmp_path)

    assert result.status == "PASS"
    assert result.compact_status == "ok"
    assert result.recent_fact_in_kept_tail is True
    assert result.recent_fact_in_summary is False
    assert result.recent_fact_survives_without_tail is False
    assert result.old_fact_leaked is False
    assert result.summary_fact_in_summary is True


def test_kept_tail_probe_report_names_the_mechanism(tmp_path):
    result = run_kept_tail_probe(tmp_path)
    report = render_probe_report(result)

    assert "SessionMemory Kept-Tail Probe" in report
    assert "kept tail" in report
    assert KEPT_TAIL_RECENT_FACT in report
    assert KEPT_TAIL_OLD_FACT in report
    assert KEPT_TAIL_SUMMARY_FACT in report
