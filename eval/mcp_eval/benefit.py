"""Paired MCP coding-benefit probe entrypoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.mcp_eval.benefit_cases import (  # noqa: E402
    ERROR,
    FAIL,
    INVALID,
    PASS,
    SKIPPED,
    run_benefit_pairs,
    summarize_benefit_pairs,
)
from eval.mcp_eval.evidence import (  # noqa: E402
    build_protocol,
    required_coverage,
    resolve_code_version,
    status_metrics,
    write_evidence_jsonl,
)

INCONCLUSIVE = "INCONCLUSIVE"

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _default_output_path(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"benefit_{mode}_{stamp}.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fake", "live"), default="fake")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--code-version",
        help="portable code version recorded in evidence (default: git or WORKTREE)",
    )
    args = parser.parse_args(argv)
    root = args.work_root or Path(tempfile.mkdtemp(prefix="mcp_benefit_"))
    results = run_benefit_pairs(
        mode=args.mode,
        root=root,
        repeat=args.repeat,
        max_turns=args.max_turns,
    )
    summary = summarize_benefit_pairs(results)
    public_claim = (
        "synthetic_harness_self_test"
        if args.mode == "fake" and summary["claim"] == "harness_self_test_only"
        else summary["claim"]
    )
    output = args.output or _default_output_path(args.mode)
    if args.code_version is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        commit = _git_commit()
        timestamp = datetime.now(timezone.utc).isoformat()
        with output.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(
                    json.dumps(
                        result.to_record(commit=commit, timestamp=timestamp),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"=== MCP benefit pair ({args.mode}) ===")
        for result in results:
            print(
                f"[{result.status}] pair={result.pair_index} order={list(result.order)} "
                f"- {result.message}"
            )
        counts = summary["counts"]
        print(
            f"Summary: PASS={counts.get(PASS, 0)} FAIL={counts.get(FAIL, 0)} "
            f"INVALID={counts.get(INVALID, 0)} SKIPPED={counts.get(SKIPPED, 0)} "
            f"ERROR={counts.get(ERROR, 0)}"
        )
        print(f"Claim: {summary['claim']}")
        print(f"Gate: {summary['gate_status']}")
        print(f"Results: {output}")
        return 0 if summary["gate_pass"] else 1
    expected_pair_ids = [f"pair_{index:03d}" for index in range(1, args.repeat + 1)]
    observed_pair_ids = [f"pair_{result.pair_index:03d}" for result in results]
    coverage = required_coverage(expected_pair_ids, observed_pair_ids)
    gate_status = (
        summary["gate_status"]
        if coverage["full_coverage"]
        else INCONCLUSIVE
    )
    gate_pass = summary["gate_pass"] and coverage["full_coverage"]
    protocol_id = (
        "mcp-benefit-synthetic" if args.mode == "fake" else "mcp-benefit-live"
    )
    protocol = build_protocol(
        protocol_id=protocol_id,
        protocol_version="1.0.0",
        descriptor={
            "track": "benefit_synthetic" if args.mode == "fake" else "benefit_live",
            "case_id": results[0].case_id if results else "mcp_benefit_01_issue_context_patch",
            "mode": args.mode,
            "expected_pair_count": args.repeat,
            "conditions": ["MCP unavailable", "MCP issue context available"],
            "grader": "isolated_hidden_python_fixture_grader",
            "gate_rule": "all expected pairs complete and have status PASS",
            "pass_rate_denominator": ["PASS", "FAIL"],
        },
        repo_root=REPO_ROOT,
        source_paths=(
            "eval/mcp_eval/evidence.py",
            "eval/mcp_eval/benefit_cases.py",
            "eval/mcp_eval/benefit.py",
        ),
    )
    case_payloads = []
    for result in results:
        payload = result.to_record(commit="", timestamp="")
        payload.pop("commit", None)
        payload.pop("timestamp", None)
        payload["eligible_for_pass_rate"] = result.status in {PASS, FAIL}
        case_payloads.append(payload)
    what_this_does_not_prove = (
        "real model or product benefit"
        if args.mode == "fake"
        else "population-level or production product benefit"
    )
    write_evidence_jsonl(
        output=output,
        protocol=protocol,
        repo_root=REPO_ROOT,
        code_version=resolve_code_version(REPO_ROOT, args.code_version),
        interpretation={
            "claim": public_claim,
            "what_this_does_not_prove": what_this_does_not_prove,
        },
        summary_payload={
            "track": "benefit_synthetic" if args.mode == "fake" else "benefit_live",
            "status": gate_status,
            "gate_status": gate_status,
            "gate_pass": gate_pass,
            "coverage": coverage,
            "status_counts": dict(summary["counts"]),
            "metrics": status_metrics([result.status for result in results]),
            "claim": public_claim,
            "what_this_does_not_prove": what_this_does_not_prove,
        },
        case_payloads=case_payloads,
        execution_mode="offline_synthetic" if args.mode == "fake" else "live_model",
        api_calls=0 if args.mode == "fake" else None,
        llm_calls=0 if args.mode == "fake" else None,
    )

    print(f"=== MCP benefit pair ({args.mode}) ===")
    for result in results:
        print(
            f"[{result.status}] pair={result.pair_index} order={list(result.order)} "
            f"- {result.message}"
        )
    counts = summary["counts"]
    print(
        f"Summary: PASS={counts.get(PASS, 0)} FAIL={counts.get(FAIL, 0)} "
        f"INVALID={counts.get(INVALID, 0)} SKIPPED={counts.get(SKIPPED, 0)} "
        f"ERROR={counts.get(ERROR, 0)}"
    )
    print(f"Claim: {public_claim}")
    print(f"Gate: {gate_status}")
    print(f"Results: {output}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
