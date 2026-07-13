"""Deterministic evaluation contracts for context-budget handling."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from eval.context_eval.cases import (
    ERROR,
    FAIL,
    INCONCLUSIVE,
    INVALID,
    PASS,
    REQUIRED_CASE_IDS,
    protocol_fingerprint,
    protocol_manifest,
    run_case,
    run_cases,
)
from eval.context_eval.run import build_run_records, main, summarize_results


ROOT = Path(__file__).resolve().parents[1]


def _record_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _record_strings(key)
            yield from _record_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _record_strings(item)


def _looks_absolute_path(value: str) -> bool:
    return value.startswith(("/", "\\\\")) or bool(
        re.match(r"^[A-Za-z]:[\\/]", value)
    )


def test_unchanged_case_preserves_content_below_the_limit():
    result = run_case("context_unchanged_under_limit")

    assert result.status == PASS
    assert result.evidence["outcome"] == "unchanged"
    assert result.evidence["pruned_results"] == 0
    assert result.evidence["before_tokens"] == result.evidence["after_tokens"]
    assert result.evidence["content_preserved"] is True


def test_pruning_case_removes_oldest_recoverable_and_retains_recent():
    result = run_case("context_prunes_oldest_recoverable_and_retains_recent")

    assert result.status == PASS
    assert result.evidence["outcome"] == "pruned"
    assert result.evidence["pruned_results"] == 1
    assert result.evidence["oldest_omitted"] is True
    assert result.evidence["recent_retained"] is True
    assert result.evidence["input_unmodified"] is True


def test_nonrecoverable_and_unpaired_case_reports_exceeded_without_pruning():
    result = run_case("context_nonrecoverable_and_unpaired_exceeds")

    assert result.status == PASS
    assert result.evidence["outcome"] == "exceeded"
    assert result.evidence["pruned_results"] == 0
    assert result.evidence["nonrecoverable_retained"] is True
    assert result.evidence["unpaired_retained"] is True


def test_pairing_case_changes_only_the_paired_recoverable_copy():
    result = run_case("context_pairs_results_and_preserves_input")

    assert result.status == PASS
    assert result.evidence["paired_result_omitted"] is True
    assert result.evidence["unpaired_result_retained"] is True
    assert result.evidence["input_unmodified"] is True
    assert result.evidence["request_is_deep_copy"] is True


def test_accounting_case_includes_system_schema_and_trace_decision_fields():
    result = run_case("context_accounts_system_schema_and_emits_trace")

    assert result.status == PASS
    assert result.evidence["full_estimate_exceeds_message_only"] is True
    assert result.evidence["trace_matches_decision"] is True
    assert result.evidence["trace_attribute_keys"] == [
        "context.after_tokens",
        "context.before_tokens",
        "context.outcome",
        "context.pruned_results",
        "context.target_tokens",
    ]


def test_required_matrix_and_protocol_fingerprint_are_complete_and_stable():
    results = run_cases()

    assert tuple(result.case_id for result in results) == REQUIRED_CASE_IDS
    assert all(result.required for result in results)
    assert all(result.status == PASS for result in results)
    assert re.fullmatch(r"[0-9a-f]{64}", protocol_fingerprint())
    assert protocol_fingerprint() == protocol_fingerprint()
    assert protocol_manifest()["gate_rule"] == (
        "full required selection and exact result coverage with every case PASS"
    )


def test_case_matrix_does_not_write_to_the_ambient_trace_sink(tmp_path, monkeypatch):
    import obs.trace as trace_module

    previous = trace_module._SINK
    trace_module._SINK = None
    monkeypatch.chdir(tmp_path)
    try:
        assert all(result.status == PASS for result in run_cases())
    finally:
        trace_module._SINK = previous

    assert not (tmp_path / ".traces").exists()


def test_summary_excludes_invalid_inconclusive_and_error_from_valid_count():
    results = run_cases()
    statuses = (PASS, FAIL, INVALID, INCONCLUSIVE, ERROR)
    mixed = [
        replace(result, status=status)
        for result, status in zip(results, statuses, strict=True)
    ]

    summary = summarize_results(mixed, selected_case_ids=REQUIRED_CASE_IDS)

    assert summary["counts"] == {
        PASS: 1,
        FAIL: 1,
        INVALID: 1,
        INCONCLUSIVE: 1,
        ERROR: 1,
    }
    assert summary["valid_case_count"] == 2
    assert summary["excluded_case_count"] == 3
    assert summary["gate_status"] == ERROR
    assert summary["gate_pass"] is False


def test_full_runner_emits_auditable_path_free_jsonl(tmp_path):
    output = tmp_path / "context-budget.jsonl"

    assert main(["--output", str(output), "--code-version", "WORKTREE"]) == 0
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(records) == len(REQUIRED_CASE_IDS) + 1
    summary = records[0]
    assert summary["record_type"] == "run_summary"
    assert summary["schema_version"] == "context-budget-evidence-v1"
    assert summary["protocol_version"] == "context-budget-offline-v1"
    assert summary["protocol_sha256"] == protocol_fingerprint()
    assert summary["code_version"] == "WORKTREE"
    assert summary["timestamp_utc"].endswith("Z")
    assert set(summary["environment"]) == {
        "machine",
        "platform",
        "python_implementation",
        "python_version",
    }
    assert summary["full_gate_coverage"] is True
    assert summary["gate_status"] == PASS
    assert summary["gate_pass"] is True
    assert summary["valid_case_count"] == len(REQUIRED_CASE_IDS)
    assert summary["excluded_case_count"] == 0
    assert all(record["what_this_does_not_prove"] for record in records[1:])
    assert not any(
        _looks_absolute_path(value)
        for record in records
        for value in _record_strings(record)
    )


def test_selected_subset_is_explicitly_non_gate(tmp_path):
    output = tmp_path / "subset.jsonl"
    case_id = REQUIRED_CASE_IDS[0]

    assert main(["--output", str(output), "--cases", case_id]) == 1
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert records[0]["selected_case_ids"] == [case_id]
    assert records[0]["full_gate_coverage"] is False
    assert records[0]["gate_status"] == INCONCLUSIVE
    assert records[0]["gate_pass"] is False


def test_missing_required_result_cannot_open_the_full_gate():
    incomplete = run_cases()[:-1]

    summary = summarize_results(
        incomplete,
        selected_case_ids=REQUIRED_CASE_IDS,
    )

    assert summary["full_gate_coverage"] is False
    assert summary["gate_status"] == INCONCLUSIVE
    assert summary["gate_pass"] is False


def test_empty_case_selection_is_rejected_without_evidence(tmp_path):
    output = tmp_path / "empty.jsonl"

    assert main(["--output", str(output), "--cases", ","]) == 2
    assert not output.exists()


def test_record_builder_rejects_path_like_code_versions():
    results = run_cases()

    for invalid in ("/tmp/revision", r"C:\repo\revision", "feature/revision"):
        try:
            build_run_records(
                results,
                selected_case_ids=REQUIRED_CASE_IDS,
                code_version=invalid,
            )
        except ValueError as exc:
            assert "code_version" in str(exc)
        else:
            raise AssertionError(f"path-like code version accepted: {invalid}")


def test_design_report_and_provisional_evidence_are_separate_and_traceable():
    design = ROOT / "docs" / "evals" / "context-budget-design.md"
    report = ROOT / "docs" / "evals" / "context-budget-report.md"
    evidence = ROOT / "docs" / "evals" / "evidence" / "context-budget.jsonl"

    assert design.is_file() and report.is_file() and evidence.is_file()
    design_text = design.read_text(encoding="utf-8")
    report_text = report.read_text(encoding="utf-8")
    for text in (design_text, report_text):
        assert text.startswith("---\n")
        assert "Glossary" in text
        assert "TL;DR" in text
        assert "What this does not prove" in text
    assert "context-budget.jsonl" in report_text
    assert "provider tokenizer accuracy" in report_text
    assert "task quality" in report_text

    records = [json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines()]
    assert records[0]["code_version"] == "WORKTREE"
    assert records[0]["protocol_sha256"] == protocol_fingerprint()
    assert records[0]["gate_status"] == PASS
    assert records[0]["full_gate_coverage"] is True
