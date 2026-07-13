"""Live MCP behavior eval — requires API key, mcp package, and proxy."""

from __future__ import annotations

import pytest

from eval.mcp_eval.behavior_cases import BEHAVIOR_CASE_IDS, INCONCLUSIVE, PASS, SKIPPED, run_behavior_case


@pytest.mark.live
@pytest.mark.parametrize("case_id", BEHAVIOR_CASE_IDS)
def test_mcp_behavior_live_case(case_id: str, tmp_path):
    result = run_behavior_case(case_id, mode="live", workdir=tmp_path)
    assert result.status in {PASS, SKIPPED, INCONCLUSIVE}, (
        f"{case_id}: unexpected {result.status}: {result.message}"
    )
