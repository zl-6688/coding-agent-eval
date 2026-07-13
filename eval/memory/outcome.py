"""Outcome classification for memory-eval samples.

This module separates three concepts that Phoenix otherwise tends to blur:

* verdict: what the grader said about the task output.
* sample_status: whether this sample is valid for scoring.
* expectation: whether the verdict matches the experimental condition.
"""

from __future__ import annotations

from typing import Any


TRACK1_INCREMENTAL_IDS = frozenset({"H_ref", "H_proj", "H_fb2", "H_usr"})
TRACK1_PRECISION_IDS = frozenset({"H_prec"})
TRACK2_IDS = frozenset({"H_drift", "H_ignore", "H_neg", "H_neg_clean"})
EXCLUDED_CASE_IDS = frozenset({"H_fb1"})


def expected_verdict(case_id: str, arm: str) -> str | None:
    """Return the verdict expected by the experiment design, if scorable."""
    if case_id in EXCLUDED_CASE_IDS:
        return None
    if case_id in TRACK1_INCREMENTAL_IDS:
        if arm == "A":
            return "PASS"
        if arm == "B":
            return "FAIL"
    if case_id in TRACK1_PRECISION_IDS:
        if arm in {"A", "B"}:
            return "PASS"
    if case_id in TRACK2_IDS:
        return "PASS"
    return None


def classify_outcome(
    *,
    case_id: str,
    arm: str,
    verdict: str,
    sample_status: str | None,
) -> dict[str, Any]:
    """Classify one sample without conflating eval failure and runtime failure."""
    status = (sample_status or "").upper()
    expected = expected_verdict(case_id, arm)

    if case_id in EXCLUDED_CASE_IDS:
        outcome_class = "excluded"
        expectation_met = None
    elif status == "ERROR":
        outcome_class = "runtime_error"
        expectation_met = None
    elif status in {"INVALID", "INCONCLUSIVE"}:
        outcome_class = "excluded"
        expectation_met = None
    elif expected is None:
        outcome_class = "unknown"
        expectation_met = None
    else:
        expectation_met = verdict == expected
        outcome_class = "expected" if expectation_met else "unexpected"

    return {
        "expected_verdict": expected,
        "expectation_met": expectation_met,
        "expectation_score": (
            1.0 if expectation_met is True else
            0.0 if expectation_met is False else
            None
        ),
        "outcome_class": outcome_class,
    }


def classify_record(record: dict[str, Any]) -> dict[str, Any]:
    """Classify a JSONL-style memory-eval record."""
    return classify_outcome(
        case_id=str(record.get("case_id") or ""),
        arm=str(record.get("arm") or ""),
        verdict=str(record.get("verdict") or ""),
        sample_status=record.get("sample_status"),
    )
