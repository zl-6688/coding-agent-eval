from eval.compression_eval.sm_capture import (
    SM_CAPTURE_FACTS,
    render_capture_report,
    run_capture_smoke,
)


def test_capture_smoke_fake_mode_records_all_target_facts(tmp_path):
    result = run_capture_smoke(tmp_path, live=False)

    assert result.status == "PASS"
    assert result.mode == "fake"
    assert result.extract_stopped == "finished"
    assert result.capture_rate == 1.0
    assert result.last_summarized_message_id
    assert {fact.fact_id for fact in SM_CAPTURE_FACTS} == {item.fact_id for item in result.facts}
    assert all(item.captured for item in result.facts)


def test_capture_report_is_self_describing(tmp_path):
    result = run_capture_smoke(tmp_path, live=False)
    report = render_capture_report(result)

    assert "SessionMemory Capture Smoke" in report
    assert "capture rate" in report
    assert "fake" in report
    for fact in SM_CAPTURE_FACTS:
        assert fact.fact_id in report
