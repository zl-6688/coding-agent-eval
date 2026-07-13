"""Pytest wrapper for eval/mcp_eval Phase 1 smoke gate."""

from __future__ import annotations

import pytest

from eval.mcp_eval.cases import (
    ERROR,
    FAIL,
    PASS,
    REQUIRED_CASE_IDS,
    SKIPPED,
    run_case,
    summarize_results,
)


@pytest.mark.parametrize("case_id", REQUIRED_CASE_IDS)
def test_mcp_smoke_required_case(case_id: str):
    result = run_case(case_id)
    assert result.status in {PASS, SKIPPED}, (
        f"{case_id}: expected PASS or SKIPPED, got {result.status}: {result.message}"
    )


def test_mcp_smoke_gate_summary():
    results = [run_case(case_id) for case_id in REQUIRED_CASE_IDS]
    summary = summarize_results(results)
    assert summary["gate_pass"] is True, (
        f"gate failed: fail={summary['required_fail']} error={summary['required_error']}"
    )


def test_mcp_smoke_02_always_runs_without_mcp_package():
    result = run_case("mcp_smoke_02_permission_deny")
    assert result.status == PASS


def test_mcp_smoke_03_always_runs_without_mcp_package():
    result = run_case("mcp_smoke_03_server_isolation")
    assert result.status == PASS
