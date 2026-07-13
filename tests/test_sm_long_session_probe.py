from eval.compression_eval.sm_long_session_probe import (
    render_probe_report,
    run_controlled_long_session_probe,
)


def test_controlled_long_session_probe_writes_sm_then_takes_over(tmp_path):
    result = run_controlled_long_session_probe(tmp_path)
    summary = result.takeover_summary

    assert result.final_text == "controlled long-session probe complete"
    assert result.sm_written is True
    assert result.capture_gate is True
    assert result.tool_turns == 23
    assert result.main_llm_calls == 24
    assert result.fork_llm_calls == 11
    assert result.memory_fork_spans == 11
    assert result.full_stub_spans == 0

    assert summary["files_scanned"] == 1
    assert summary["malformed_lines"] == 0
    assert summary["sm_attempts"] == 1
    assert summary["sm_ok"] == 1
    assert summary["takeover_rate"] == 1.0
    assert summary["pipeline_did_sm_true"] == 1
    assert summary["pipeline_did_full_true"] == 0
    assert summary["saved_full_compact_calls_estimate"] == 1
    assert summary["sm_status_counts"] == {"ok": 1}


def test_controlled_long_session_probe_report_names_gates(tmp_path):
    result = run_controlled_long_session_probe(tmp_path)

    report = render_probe_report(result)

    assert "| SM written | True |" in report
    assert "| Capture gate | True |" in report
    assert "| SM takeover rate | 100.00% |" in report
    assert "| Avoided sync full_compact calls estimate | 1 |" in report
