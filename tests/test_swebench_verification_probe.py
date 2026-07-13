import json
from pathlib import Path

import pytest

from eval.swebench.verification_probe import (
    CC_ALIGNED_VERIFIER_SYSTEM_IDENTITY,
    DEFAULT_SUITE,
    PROMPT_MODE_CC_ALIGNED,
    PROMPT_MODE_CURRENT,
    VERIFIER_REPORTER_IDENTITY,
    VERIFIER_SYSTEM_IDENTITY,
    build_verifier_prompt,
    load_model_patch,
    load_suite,
    parse_verdict,
    snapshot_delta,
    system_identity_for_mode,
)


def test_probe_suite_keeps_two_bad_patches_and_two_resolved_controls():
    suite = load_suite(DEFAULT_SUITE)

    assert len(suite) == 4
    assert len({case["instance_id"] for case in suite}) == 4
    assert [case["bucket"] for case in suite].count("historical_false_green") == 2
    assert [case["bucket"] for case in suite].count("officially_resolved_control") == 2
    assert all(str(case.get("approach_taken") or "").strip() for case in suite)


def test_cc_aligned_system_keeps_source_backed_verification_gates():
    system = system_identity_for_mode(PROMPT_MODE_CC_ALIGNED)

    assert system == CC_ALIGNED_VERIFIER_SYSTEM_IDENTITY
    assert "verification avoidance" in system
    assert "first 80%" in system
    assert "A check without a command" in system
    assert "at least one adversarial probe" in system
    assert "BEFORE ISSUING FAIL" in system
    assert "SWE-bench" not in system
    assert "runtests.py" not in system


def test_current_system_mode_preserves_previous_probe_baseline():
    assert system_identity_for_mode(PROMPT_MODE_CURRENT) == VERIFIER_SYSTEM_IDENTITY


def test_cc_style_handoff_adds_neutral_approach_without_oracle_fields():
    prompt = build_verifier_prompt(
        problem_statement="Original issue text.",
        changed_files=["src/example.py"],
        approach_taken="Changed Example.lookup() to use the configured source value.",
        handoff_mode="cc_style",
        max_turns=20,
    )

    assert "## Approach taken" in prompt
    assert "Changed Example.lookup()" in prompt
    assert "Original issue text." in prompt
    assert "official_resolved" not in prompt
    assert "FAIL_TO_PASS" not in prompt
    assert "gold patch" not in prompt.lower()


def test_minimal_handoff_does_not_expose_approach_text():
    prompt = build_verifier_prompt(
        problem_statement="Original issue text.",
        changed_files=["src/example.py"],
        approach_taken="SECRET APPROACH SUMMARY",
        handoff_mode="minimal",
        max_turns=20,
    )

    assert "SECRET APPROACH SUMMARY" not in prompt
    assert "## Approach taken" not in prompt


def test_snapshot_delta_preserves_read_only_violation_evidence():
    delta = snapshot_delta(" M a.py\n---DIFF---\nold\n", " M a.py\n?? test.db\n---DIFF---\nold\n")

    assert "+?? test.db" in delta
    assert "--- before-verifier" in delta
    assert "+++ after-verifier" in delta


def test_verifier_prompt_is_independent_and_does_not_embed_oracle_data():
    prompt = build_verifier_prompt(
        problem_statement="Public behavior should preserve inherited marks.",
        changed_files=["src/example.py"],
        max_turns=30,
    )

    assert "Public behavior should preserve inherited marks." in prompt
    assert "src/example.py" in prompt
    assert "independently" in prompt.lower()
    assert "official_resolved" not in prompt
    assert "FAIL_TO_PASS" not in prompt
    assert "gold patch" not in prompt.lower()
    assert "graded test" not in prompt.lower()
    assert "30 total turns" in prompt
    assert "Inspect the current diff" in VERIFIER_SYSTEM_IDENTITY
    assert "SWE-bench" not in VERIFIER_SYSTEM_IDENTITY
    assert "runtests.py" not in VERIFIER_SYSTEM_IDENTITY
    assert "must not invent" in VERIFIER_REPORTER_IDENTITY
    assert "SWE-bench" not in VERIFIER_REPORTER_IDENTITY


@pytest.mark.parametrize("verdict", ["PASS", "FAIL", "PARTIAL"])
def test_parse_verdict_accepts_one_exact_final_line(verdict):
    assert parse_verdict(f"Evidence:\n- checked behavior\nVERDICT: {verdict}") == verdict


@pytest.mark.parametrize(
    "text",
    [
        "No verdict",
        "VERDICT: PASS\nextra text",
        "VERDICT: PASS\nVERDICT: FAIL",
        "verdict: PASS",
        "VERDICT: UNKNOWN",
    ],
)
def test_parse_verdict_rejects_ambiguous_or_nonfinal_output(text):
    with pytest.raises(ValueError):
        parse_verdict(text)


def test_load_model_patch_reads_prediction_artifact(tmp_path: Path):
    prediction = tmp_path / "prediction.jsonl"
    prediction.write_text(
        json.dumps(
            {
                "instance_id": "owner__repo-1",
                "model_patch": "diff --git a/a.py b/a.py\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_model_patch(prediction, "owner__repo-1") == "diff --git a/a.py b/a.py\n"


def test_load_model_patch_rejects_wrong_instance(tmp_path: Path):
    prediction = tmp_path / "prediction.jsonl"
    prediction.write_text(
        json.dumps({"instance_id": "other__repo-2", "model_patch": "diff"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="instance_id"):
        load_model_patch(prediction, "owner__repo-1")
