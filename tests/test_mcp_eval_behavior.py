"""Offline tests for MCP behavior eval graders (fake mode)."""

from __future__ import annotations

import pytest

from eval.mcp_eval.behavior_cases import (
    BEHAVIOR_CASE_IDS,
    FAIL,
    INCONCLUSIVE,
    PASS,
    BehaviorRunArtifacts,
    grade_echo_via_deferred,
    grade_permission_deny,
    run_behavior_case,
    summarize_behavior_results,
)


def test_grade_echo_via_deferred_passes_on_toolsearch_then_mcp():
    artifacts = BehaviorRunArtifacts(
        final_text="echo:hello",
        messages=[],
        events=[
            {"name": "tool.ToolSearch", "attributes": {"tool.name": "ToolSearch"}},
            {"name": "tool.mcp__echo__echo", "attributes": {"tool.name": "mcp__echo__echo"}},
        ],
    )
    status, _, message = grade_echo_via_deferred(artifacts)
    assert status == PASS, message


def test_grade_echo_via_deferred_fails_without_mcp_call():
    artifacts = BehaviorRunArtifacts(
        final_text="no tool",
        messages=[],
        events=[{"name": "tool.ToolSearch", "attributes": {"tool.name": "ToolSearch"}}],
    )
    status, _, message = grade_echo_via_deferred(artifacts)
    assert status == FAIL
    assert "never called" in message


def test_grade_echo_via_deferred_inconclusive_without_toolsearch():
    artifacts = BehaviorRunArtifacts(
        final_text="echo:hello",
        messages=[],
        events=[{"name": "tool.mcp__echo__echo", "attributes": {"tool.name": "mcp__echo__echo"}}],
    )
    status, _, message = grade_echo_via_deferred(artifacts)
    assert status == INCONCLUSIVE
    assert "ToolSearch" in message


def test_grade_permission_deny_passes_when_mcp_not_called():
    artifacts = BehaviorRunArtifacts(
        final_text="cannot use mcp",
        messages=[],
        events=[{"name": "tool.read_file", "attributes": {"tool.name": "read_file"}}],
    )
    status, _, message = grade_permission_deny(artifacts)
    assert status == PASS, message


def test_grade_permission_deny_fails_when_mcp_called():
    artifacts = BehaviorRunArtifacts(
        final_text="oops",
        messages=[],
        events=[{"name": "tool.mcp__echo__echo", "attributes": {"tool.name": "mcp__echo__echo"}}],
    )
    status, _, _ = grade_permission_deny(artifacts)
    assert status == FAIL


@pytest.mark.parametrize("case_id", BEHAVIOR_CASE_IDS)
def test_behavior_fake_cases(case_id: str, tmp_path):
    result = run_behavior_case(case_id, mode="fake", workdir=tmp_path)
    assert result.status == PASS, result.message


def test_behavior_repeat_fake_runs_each_case_n_times(tmp_path):
    from eval.mcp_eval.behavior_cases import PASS, run_behavior_cases, summarize_behavior_results

    results = run_behavior_cases(mode="fake", workdir=tmp_path, repeat=3)
    assert len(results) == len(BEHAVIOR_CASE_IDS) * 3
    summary = summarize_behavior_results(results)
    assert summary["gate_pass"] is True
    assert summary["per_case"]["mcp_behavior_01_echo_via_deferred"]["pass_count"] == 3
    assert summary["per_case"]["mcp_behavior_03_session_deferred_reuse"]["pass_count"] == 3
    assert all(r.trial is not None for r in results)


def test_grade_session_deferred_reuse_requires_cache_hit():
    from eval.mcp_eval.behavior_cases import grade_session_deferred_reuse

    artifacts = BehaviorRunArtifacts(
        final_text="echo:hello",
        messages=[],
        events=[
            {"name": "tool.ToolSearch", "attributes": {"tool.name": "ToolSearch"}},
            {"name": "tool.mcp__echo__echo", "attributes": {"tool.name": "mcp__echo__echo"}},
        ],
    )
    status, _, message = grade_session_deferred_reuse(
        artifacts,
        {"second_run_cache_hit": False, "agent_run_count": 2},
    )
    assert status == FAIL
    assert "cache_hit" in message


def test_behavior_fake_gate_summary(tmp_path):
    results = [run_behavior_case(case_id, mode="fake", workdir=tmp_path) for case_id in BEHAVIOR_CASE_IDS]
    summary = summarize_behavior_results(results)
    assert summary["gate_pass"] is True
