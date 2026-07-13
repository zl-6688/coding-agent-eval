"""Offline reliability gate contracts."""

from __future__ import annotations

import json


def test_all_reliability_cases_pass():
    from eval.mcp_eval.reliability_cases import (
        RELIABILITY_CASE_IDS,
        run_reliability_cases,
        summarize_reliability_results,
    )

    results = run_reliability_cases()
    assert [result.case_id for result in results] == list(RELIABILITY_CASE_IDS)
    assert all(result.status == "PASS" for result in results), [
        (result.case_id, result.status, result.message) for result in results
    ]
    assert summarize_reliability_results(results)["gate_pass"] is True


def test_reliability_cli_writes_auditable_jsonl(tmp_path):
    from eval.mcp_eval.reliability import main

    output = tmp_path / "reliability.jsonl"
    assert main(["--output", str(output)]) == 0
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records
    assert all(record["status"] == "PASS" for record in records)
    assert all("assertions" in record["evidence"] for record in records)
