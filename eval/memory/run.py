"""Memory eval orchestrator — A/B × k runs, metrics, report.

Usage:
    python eval/memory/run.py                         # all cases, k=5
    python eval/memory/run.py --cases H_ref H_fb1     # subset
    python eval/memory/run.py --k 1 --cases H_ref     # quick smoke test
    python eval/memory/run.py --k 1 --smoke           # H_ref only (live smoke)

Metrics (track-1):
  Incremental cases (5): P(write), P(use|write), end-to-end, Δ per case + total
  H_prec (precision):    P(A pass), P(B pass), Δ≈0 check — REPORTED SEPARATELY
Track-2: pass^k rate per case (no Δ).

Outputs:
  eval/memory/results/<timestamp>.jsonl   raw per-run records
  eval/memory/results/<timestamp>.md      human-readable report

Design reference: 00-design.md § 2.6 metrics + § 2.7 model/run.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add repo root so `agent` and `eval.memory` imports work
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))  # for sibling imports

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from cases import (
    ALL_CASES,
    CASES_BY_ID,
    TRACK1_INCREMENTAL,
    TRACK1_PRECISION,
    TRACK2_ALL,
    Case,
)
from graders import RUBRIC_VERSION
from harness import ArmResult, _MAX_TURNS, run_arm
from outcome import classify_outcome, classify_record

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PHOENIX = "http://localhost:6006"
PHOENIX_PROJECT = "memory-eval"


def _get_git_hash() -> str:
    """Return current git commit hash + dirty flag (best-effort)."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"],
                               cwd=REPO, capture_output=True, check=False).returncode != 0
        return f"{h}{'*' if dirty else ''}" if h else "unknown"
    except Exception:
        return "unknown"


# ── Per-run record ────────────────────────────────────────────────────────────


def _agent_changes(art: dict) -> str:
    """Real on-disk diff (ground truth) of the agent's file changes vs BASE.

    WHY: `transcript` is the agent's *self-narration* — it may claim edits it never
    made, or say it wrote helpers.py while actually writing elsewhere.  The git diff
    collected by the harness is the ground truth; human calibration must judge against
    it, not the model's own words.
    """
    if not isinstance(art, dict):
        return ""
    if "agent_changes" in art:
        return art.get("agent_changes") or ""
    # H_ignore: artifacts hold foil/ignore sub-arms, each with its own agent_changes
    parts = []
    for sub in ("foil", "ignore"):
        if isinstance(art.get(sub), dict):
            parts.append(f"### [{sub} 臂]\n{art[sub].get('agent_changes', '') or '（无改动）'}")
    return "\n\n".join(parts)


def _record(result: ArmResult, run_idx: int) -> dict:
    grade = result.grade
    rec = {
        # ── existing fields ───────────────────────────────────────────────────
        "case_id":        result.case_id,
        "arm":            result.arm,
        "run_idx":        run_idx,
        "s1_complete":    result.s1_complete,   # P1-3: None=B-arm/track-2; True=natural; False=max_turns
        "write_pass":     result.write_pass,
        "write_evidence": result.write_evidence,
        "verdict":        grade.verdict if grade else "ERROR",
        "reason":         grade.reason  if grade else result.error,
        "evidence":       grade.evidence if grade else "",
        # WHY: human calibration in Phoenix needs the full text to show per-span
        # evidence; previously only length was stored, making transcript invisible in UI.
        "transcript":     result.transcript,
        "transcript_len": len(result.transcript),
        # WHY: ground-truth diff — transcript is self-narration and can lie; the human
        # must judge code cases against what actually landed on disk (see _agent_changes).
        "agent_changes":  _agent_changes(result.artifacts),
        "error":          result.error,
        # ── Phase-2 per-sample fields (spec §2 Group A) ───────────────────────
        "s1_transcript":             result.s1_transcript,
        "write_fork_decision":       result.write_fork_decision,       # fork JSON or None if not captured
        "recall_tier1_lines":        result.recall_tier1_lines,        # MEMORY.md content before S2
        "recall_tier2_files":        result.recall_tier2_files,        # sideQuery-selected filenames
        "judge_raw_full":            result.judge_raw_full,            # None for code graders
        "judge_input_truncated_at":  result.judge_input_truncated_at,  # None = no truncation
        "sample_status":             result.sample_status,             # VALID|INVALID|ERROR|INCONCLUSIVE
        "error_detail":              result.error_detail,
        "token_usage":               result.token_usage,               # None = not yet captured (known gap)
        "started_at":                result.started_at,
        "ended_at":                  result.ended_at,
        "latency_ms":                result.latency_ms,
        "fixture_hash":              result.fixture_hash,
        "probe_hash":                result.probe_hash,
        "is_b1_sanity":              result.is_b1_sanity,
    }
    rec.update(classify_record(rec))
    return rec


def _span_should_mark_runtime_error(result: ArmResult, verdict: str) -> bool:
    """Return True only for infrastructure/runtime failures, not eval FAILs."""
    if result.error:
        return True
    return result.sample_status == "ERROR" and verdict == "SKIP"


# ── Metrics ───────────────────────────────────────────────────────────────────


def _metrics_track1_incremental(records: list[dict]) -> dict:
    """Aggregate track-1 incremental metrics over k runs.

    Returns per-case dicts and total Δ (only 5 incremental cases).
    """
    from collections import defaultdict
    by_case: dict[str, dict] = defaultdict(lambda: {"A": [], "B": []})
    for r in records:
        arm = r["arm"]
        if arm not in ("A", "B"):
            continue
        by_case[r["case_id"]][arm].append(r)

    per_case = {}
    total_delta_sum = 0.0
    total_cases = 0

    for case_id, arms in sorted(by_case.items()):
        a_runs = arms["A"]
        b_runs = arms["B"]

        # P1-3: S1_INCOMPLETE runs have write_pass=None — exclude from write-side
        # denominator.  Counting them deflates P(write) (false attribution to memory
        # when the real cause is S1 exhausting max_turns before auto_memory fires).
        a_eligible = [r for r in a_runs if r.get("write_pass") is not None]
        n_s1_incomplete = len(a_runs) - len(a_eligible)

        # P(write) = fraction of eligible A runs where write_pass=True
        a_write = [r for r in a_eligible if r.get("write_pass") is True]
        p_write = len(a_write) / max(1, len(a_eligible))

        # P(use|write) = among write-pass A runs, fraction with verdict=PASS
        a_use = [r for r in a_write if r.get("verdict") == "PASS"]
        p_use_given_write = len(a_use) / max(1, len(a_write))

        # End-to-end = P(A overall PASS) among eligible runs only
        a_pass = [r for r in a_eligible if r.get("verdict") == "PASS"]
        p_a_e2e = len(a_pass) / max(1, len(a_eligible))

        # P(B pass)
        b_pass = [r for r in b_runs if r.get("verdict") == "PASS"]
        p_b = len(b_pass) / max(1, len(b_runs))

        # Δ at "use|write" level (isolates write-side noise)
        delta = p_use_given_write - p_b
        total_delta_sum += delta
        total_cases += 1

        per_case[case_id] = {
            "n_a": len(a_runs),
            "n_a_eligible": len(a_eligible),
            "n_s1_incomplete": n_s1_incomplete,
            "n_b": len(b_runs),
            "p_write": round(p_write, 3),
            "p_use_given_write": round(p_use_given_write, 3),
            "p_a_e2e": round(p_a_e2e, 3),
            "p_b": round(p_b, 3),
            "delta": round(delta, 3),
        }

    total_delta = round(total_delta_sum / max(1, total_cases), 3)
    return {"per_case": per_case, "total_delta": total_delta}


def _metrics_h_prec(records: list[dict]) -> dict:
    """H_prec precision axis — want Δ≈0 (A≈B). SEPARATE from total Δ.

    P1-5: a valid A sample is one where the decoy WAS stored (write_pass is True)
    — only then can we test whether the scoped memory is mis-applied.  Exclude both
    write_pass=False (WRITE_FAIL: decoy not stored) AND write_pass=None
    (S1_INCOMPLETE: S1 ran out of max_turns, memory never written, S2 skipped, diff
    empty).  Counting these invalid samples as FAIL deflates p_a into a false
    "RED: memory misleads" — the agent never produced an answer to misjudge.
    Fixed: previously used `is not False`, which let S1_INCOMPLETE (None) through.
    """
    a_runs = [r for r in records if r["arm"] == "A"]
    b_runs = [r for r in records if r["arm"] == "B"]
    # valid = decoy actually stored (write_pass is True); excludes WRITE_FAIL and S1_INCOMPLETE
    a_valid = [r for r in a_runs if r.get("write_pass") is True]
    a_vacuous = len(a_runs) - len(a_valid)
    p_a = sum(1 for r in a_valid if r["verdict"] == "PASS") / max(1, len(a_valid))
    p_b = sum(1 for r in b_runs if r["verdict"] == "PASS") / max(1, len(b_runs))
    delta = round(p_a - p_b, 3)
    # WHY: when n_a_valid == 0 (decoy never stored → all A runs are vacuous),
    # p_a collapses to 0 and delta = -p_b → looks like "RED: A<<B" which is a
    # false alarm.  0 valid samples means we have no evidence either way; mark
    # INCONCLUSIVE rather than RED so readers don't misread it as a real failure.
    if len(a_valid) == 0:
        interp = "INCONCLUSIVE (no valid A samples — decoy not stored)"
    elif abs(delta) < 0.2:
        interp = "OK (Δ≈0)"
    elif delta < -0.2:
        interp = "RED: A<<B (memory misleads)"
    else:
        interp = "WARN: A>>B (unexpected)"
    return {
        "n_a_total": len(a_runs),
        "n_a_valid": len(a_valid),
        "n_a_vacuous": a_vacuous,
        "p_a": round(p_a, 3),
        "p_b": round(p_b, 3),
        "delta": delta,
        "interpretation": interp,
    }


def _metrics_track2(records: list[dict]) -> dict:
    """Track-2: pass^k rates per case.

    WHY exclude ERROR/SKIP from denominator (spec §4 + Iron Law 5): ERROR samples
    are system failures, not logical failures — counting them deflates pass_rate
    into a false signal.  Same fix as H_prec / track1 INVALID exclusion.
    """
    from collections import defaultdict
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_case[r["case_id"]].append(r)
    result = {}
    for case_id, runs in sorted(by_case.items()):
        # Only count VALID samples in pass_rate denominator
        valid_runs = [r for r in runs if r.get("sample_status") == "VALID"]
        n_valid = len(valid_runs)
        n_pass = sum(1 for r in valid_runs if r["verdict"] == "PASS")
        result[case_id] = {
            "n": len(runs),
            "n_valid": n_valid,
            "n_error": sum(1 for r in runs if r.get("sample_status") == "ERROR"),
            "n_invalid": sum(1 for r in runs if r.get("sample_status") == "INVALID"),
            "pass_rate": round(n_pass / max(1, n_valid), 3),
            "verdicts": [r["verdict"] for r in runs],
        }
    return result


# ── Report rendering ──────────────────────────────────────────────────────────


def _render_report(
    ts: str,
    track1_inc_metrics: dict,
    prec_metrics: dict,
    track2_metrics: dict,
    all_records: list[dict],
) -> str:
    lines = [
        f"# Memory Eval Report — {ts}",
        "",
        "## Track-1 Incremental Metrics (5 cases, want Δ≫0)",
        "",
        "| case | n_A | n_A_elig | n_S1inc | n_B | P(write) | P(use\\|write) | e2e A | P(B) | Δ |",
        "|------|-----|---------|---------|-----|----------|---------------|-------|------|---|",
    ]
    for cid, m in track1_inc_metrics["per_case"].items():
        lines.append(
            f"| {cid} | {m['n_a']} | {m.get('n_a_eligible', m['n_a'])} | "
            f"{m.get('n_s1_incomplete', 0)} | {m['n_b']} | {m['p_write']:.2f} | "
            f"{m['p_use_given_write']:.2f} | {m['p_a_e2e']:.2f} | "
            f"{m['p_b']:.2f} | **{m['delta']:+.2f}** |"
        )
    lines += [
        "",
        f"**Total Δ (5 incremental cases):** {track1_inc_metrics['total_delta']:+.3f}",
        "",
        "## H_prec Precision Axis (want Δ≈0 — SEPARATE)",
        "",
        f"| n_A(total) | n_A(valid) | n_A(vacuous) | P(A) | P(B) | Δ | Interpretation |",
        f"|-----------|-----------|-------------|------|------|---|----------------|",
        f"| {prec_metrics.get('n_a_total', '?')} | {prec_metrics.get('n_a_valid', '?')} | "
        f"{prec_metrics.get('n_a_vacuous', '?')} | "
        f"{prec_metrics['p_a']:.2f} | {prec_metrics['p_b']:.2f} | "
        f"{prec_metrics['delta']:+.2f} | {prec_metrics['interpretation']} |",
        "",
        "## Track-2 CC Behaviour Replication (no Δ)",
        "",
        "| case | n | pass_rate | verdicts |",
        "|------|---|-----------|----------|",
    ]
    for cid, m in track2_metrics.items():
        lines.append(
            f"| {cid} | {m['n']} | {m['pass_rate']:.2f} | {m['verdicts']} |"
        )
    lines += [
        "",
        "## Raw Records (truncated)",
        "",
        "```",
    ]
    for r in all_records[:40]:
        lines.append(json.dumps(r, ensure_ascii=False))
    if len(all_records) > 40:
        lines.append(f"... ({len(all_records) - 40} more, see .jsonl file)")
    lines.append("```")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory eval runner")
    p.add_argument("--k", type=int, default=5, help="runs per arm (default 5)")
    p.add_argument(
        "--cases", nargs="+", default=None,
        help="case IDs to run (default: all).  Example: H_ref H_fb1",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="shorthand for --k 1 --cases H_ref (live smoke test)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke:
        args.k = 1
        args.cases = ["H_ref"]

    # Select cases
    if args.cases:
        cases: list[Case] = []
        for cid in args.cases:
            if cid not in CASES_BY_ID:
                print(f"Unknown case: {cid}.  Available: {list(CASES_BY_ID.keys())}")
                sys.exit(1)
            cases.append(CASES_BY_ID[cid])
    else:
        cases = list(ALL_CASES)

    k = args.k
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    jsonl_path = RESULTS_DIR / f"{ts}.jsonl"
    md_path    = RESULTS_DIR / f"{ts}.md"

    # ── Phase-2: init native OTel tracing to Phoenix ─────────────────────────
    # WHY init here not in harness: run.py owns the batch lifecycle; each arm run
    # will wrap run_arm() in a root span so agent sub-spans nest automatically via
    # contextvars (obs/trace.py span() → obs/otel.py otel_span() uses current span).
    from obs.otel import init_otel
    from obs.trace import SpanKind
    from obs.trace import span as tr_span
    from agent import config as _config

    otel_ok = init_otel(project_name=PHOENIX_PROJECT, endpoint=f"{PHOENIX}/v1/traces")
    if not otel_ok:
        log.warning("OTel init failed — Phoenix trace will be skipped (local JSONL still runs)")

    # ── Phase-2: write run_meta as JSONL first line (batch constants) ─────────
    run_meta = {
        "type":                 "run_meta",   # sentinel so readers can skip this line
        "run_id":               ts,
        "model_id":             _config.MODEL_ID,
        "base_url":             _config.BASE_URL or "",
        "judge_model_id":       _config.JUDGE_MODEL_ID,
        "judge_rubric_version": RUBRIC_VERSION,
        "max_turns":            _MAX_TURNS,
        "compact_strategy":     "none",
        "temperature":          0.0,
        "dataset_version":      "v1",
        "cli_command":          " ".join(sys.argv),
        "harness_commit":       _get_git_hash(),
        "k":                    k,
        "cases":                [c.id for c in cases],
        "phoenix_project":      PHOENIX_PROJECT,
    }
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(run_meta, ensure_ascii=False) + "\n")

    all_records: list[dict] = []
    track1_inc_records: list[dict] = []
    prec_records: list[dict] = []
    track2_records: list[dict] = []

    for case in cases:
        if case.track == 1:
            arms = ["A", "B"]
        else:
            arms = ["single"]

        for run_idx in range(k):
            for arm in arms:
                print(f"\n{'='*60}")
                print(f"  {case.id}  arm={arm}  run={run_idx+1}/{k}")
                print(f"{'='*60}")

                # ── Phase-2: root span wraps the entire arm run ───────────────
                # WHY root span here: all agent sub-spans (agent.run/llm.call/tool/memory.fork)
                # are created via obs.trace.span() → obs.otel.otel_span(), which uses
                # OTel's contextvars to parent under the currently-active span.
                # Wrapping run_arm() here makes memory_eval.* the parent of every
                # S1+S2 sub-span, producing the waterfall Phoenix needs.
                span_name = f"memory_eval.{case.id}.{arm}.run{run_idx}"
                started_at = datetime.now(timezone.utc).isoformat()
                t0 = time.time()

                # Phase-2: reset per-run token accumulator before each arm run
                from agent.llm import get_token_usage, reset_token_counter
                reset_token_counter()

                try:
                    with tr_span(span_name, SpanKind.AGENT) as root_sp:
                        try:
                            result = run_arm(case, arm)
                        except Exception as exc:
                            log.exception("run_arm raised: %s", exc)
                            result = ArmResult(
                                case_id=case.id, arm=arm, write_pass=None,
                                error=str(exc), error_detail=f"{type(exc).__name__}: {exc}",
                                sample_status="ERROR",
                            )
                        # Inject timing + token_usage (run.py owns batch lifecycle)
                        result.started_at = started_at
                        result.ended_at = datetime.now(timezone.utc).isoformat()
                        result.latency_ms = round((time.time() - t0) * 1000, 1)
                        result.token_usage = get_token_usage(_config.MODEL_ID)
                        # is_b1_sanity: flag first B-arm run as prior sanity check
                        if arm == "B" and run_idx == 0:
                            result.is_b1_sanity = True
                        # Propagate eval semantics to the root span without conflating
                        # expected control FAILs with runtime span errors.
                        verdict_str = result.grade.verdict if result.grade else "ERROR"
                        outcome = classify_outcome(
                            case_id=case.id,
                            arm=arm,
                            verdict=verdict_str,
                            sample_status=result.sample_status,
                        )
                        root_sp.set(
                            case_id=case.id, arm=arm, run_idx=run_idx,
                            verdict=verdict_str,
                            sample_status=result.sample_status,
                            expected_verdict=outcome["expected_verdict"] or "",
                            expectation_met=outcome["expectation_met"],
                            expectation_score=outcome["expectation_score"],
                            outcome_class=outcome["outcome_class"],
                            **{
                                "eval.verdict": verdict_str,
                                "eval.sample_status": result.sample_status,
                                "eval.expected_verdict": outcome["expected_verdict"] or "",
                                "eval.expectation_met": outcome["expectation_met"],
                                "eval.expectation_score": outcome["expectation_score"],
                                "eval.outcome_class": outcome["outcome_class"],
                            },
                            task=f"{case.id} arm={arm} run{run_idx}",
                            output_value=(
                                f"{outcome['outcome_class']} / {verdict_str} — "
                                f"{(result.grade.reason if result.grade else result.error)[:120]}"
                            ),
                        )
                        if _span_should_mark_runtime_error(result, verdict_str):
                            root_sp.error(result.grade.reason if result.grade else result.error)
                except Exception as exc:
                    # Span itself errored — should not happen; degrade gracefully
                    log.exception("root span raised: %s", exc)
                    result = ArmResult(
                        case_id=case.id, arm=arm, write_pass=None,
                        error=str(exc), error_detail=f"{type(exc).__name__}: {exc}",
                        sample_status="ERROR",
                        started_at=started_at,
                        ended_at=datetime.now(timezone.utc).isoformat(),
                        latency_ms=round((time.time() - t0) * 1000, 1),
                    )

                rec = _record(result, run_idx)
                all_records.append(rec)

                verdict = rec["verdict"]
                print(f"  verdict={verdict}  sample_status={rec['sample_status']}  reason={rec['reason'][:80]}")
                wfd = rec.get("write_fork_decision")
                print(f"  write_fork_decision={'<captured>' if wfd else 'None'}")

                # Categorise for metrics
                if case.id in {c.id for c in TRACK1_INCREMENTAL}:
                    track1_inc_records.append(rec)
                elif case.id in {c.id for c in TRACK1_PRECISION}:
                    prec_records.append(rec)
                else:
                    track2_records.append(rec)

                # Flush JSONL after each run (crash-safe; first line is run_meta)
                with jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Compute metrics
    track1_inc_metrics = (
        _metrics_track1_incremental(track1_inc_records)
        if track1_inc_records
        else {"per_case": {}, "total_delta": 0.0}
    )
    prec_metrics = (
        _metrics_h_prec(prec_records)
        if prec_records
        else {"p_a": 0, "p_b": 0, "delta": 0, "interpretation": "n/a"}
    )
    track2_met = _metrics_track2(track2_records) if track2_records else {}

    # Render and save report
    report = _render_report(ts, track1_inc_metrics, prec_metrics, track2_met, all_records)
    md_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(report)
    print(f"\nRaw records: {jsonl_path}")
    print(f"Report:      {md_path}")


if __name__ == "__main__":
    main()
