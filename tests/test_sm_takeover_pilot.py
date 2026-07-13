from eval.compression_eval.sm_takeover import analyze_paths, stats_to_dict
from eval.compression_eval.sm_takeover_pilot import (
    DEFAULT_CASES,
    render_case_table,
    run_pilot_cases,
)


def test_sm_takeover_pilot_generates_expected_pipeline_statuses(tmp_path):
    results = run_pilot_cases(tmp_path)

    assert {item.name for item in results} == {case.name for case in DEFAULT_CASES}
    assert all(item.trace_path.exists() for item in results)

    by_name = {item.name: item for item in results}
    assert by_name["seeded_ok"].capture_gate is True
    assert by_name["seeded_ok"].sm_status == "ok"
    assert by_name["seeded_ok"].pipeline_did_sm is True
    assert by_name["seeded_ok"].pipeline_did_full is False

    assert by_name["empty_note"].capture_gate is False
    assert by_name["empty_note"].sm_status == "fallback_empty"
    assert by_name["empty_note"].pipeline_did_sm is False
    assert by_name["empty_note"].pipeline_did_full is True

    assert by_name["missing_anchor"].capture_gate is True
    assert by_name["missing_anchor"].sm_status == "fallback_missing_summary_anchor"
    assert by_name["missing_anchor"].pipeline_did_sm is False
    assert by_name["missing_anchor"].pipeline_did_full is True

    assert by_name["still_over"].capture_gate is True
    assert by_name["still_over"].sm_status == "fallback_still_over"
    assert by_name["still_over"].pipeline_did_sm is False
    assert by_name["still_over"].pipeline_did_full is True


def test_sm_takeover_pilot_traces_are_compatible_with_takeover_analyzer(tmp_path):
    run_pilot_cases(tmp_path)

    stats = analyze_paths([tmp_path], require_pipeline_parent=True)
    data = stats_to_dict(stats)

    assert data["files_scanned"] == len(DEFAULT_CASES)
    assert data["malformed_lines"] == 0
    assert data["sm_attempts"] == 4
    assert data["sm_ok"] == 1
    assert data["takeover_rate"] == 0.25
    assert data["sm_direct_or_unlinked"] == 0
    assert data["pipeline_spans"] == 4
    assert data["pipeline_with_sm_attempt"] == 4
    assert data["pipeline_did_sm_true"] == 1
    assert data["pipeline_did_full_true"] == 3
    assert data["pipeline_did_full_true_after_sm_attempt"] == 3
    assert data["saved_full_compact_calls_estimate"] == 1
    assert data["sm_status_counts"] == {
        "fallback_empty": 1,
        "fallback_missing_summary_anchor": 1,
        "fallback_still_over": 1,
        "ok": 1,
    }


def test_render_case_table_includes_case_level_gate_and_status(tmp_path):
    results = run_pilot_cases(tmp_path)

    report = render_case_table(results)

    assert "| `seeded_ok` | True | `ok` | True | False |" in report
    assert "| `empty_note` | False | `fallback_empty` | False | True |" in report
