"""MCP per-server lifecycle reliability gate entrypoint (no LLM)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.mcp_eval.reliability_cases import (  # noqa: E402
    ERROR,
    FAIL,
    PASS,
    RELIABILITY_CASE_IDS,
    run_reliability_cases,
    summarize_reliability_results,
)
from eval.mcp_eval.evidence import (  # noqa: E402
    build_protocol,
    required_coverage,
    required_gate_status,
    resolve_code_version,
    status_metrics,
    write_evidence_jsonl,
)

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


def _write_runtime_jsonl(path: Path, results) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    commit = _git_commit()
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    result.to_record(commit=commit, timestamp=timestamp),
                    ensure_ascii=False,
                )
                + "\n"
            )


def _default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"reliability_{stamp}.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", help="Comma-separated reliability case ids")
    parser.add_argument("--output", type=Path, help="JSONL output path")
    parser.add_argument(
        "--code-version",
        help="portable code version recorded in evidence (default: git or WORKTREE)",
    )
    args = parser.parse_args(argv)
    case_ids = (
        tuple(part.strip() for part in args.cases.split(",") if part.strip())
        if args.cases
        else RELIABILITY_CASE_IDS
    )
    unknown = [case_id for case_id in case_ids if case_id not in RELIABILITY_CASE_IDS]
    if unknown:
        print(f"Unknown case ids: {', '.join(unknown)}", file=sys.stderr)
        return 2

    results = run_reliability_cases(case_ids)
    summary = summarize_reliability_results(results)
    output = args.output or _default_output_path()
    coverage = required_coverage(
        RELIABILITY_CASE_IDS,
        [result.case_id for result in results],
    )
    statuses = [result.status for result in results]
    summary["full_gate_coverage"] = coverage["full_coverage"]
    summary["selected_pass"] = bool(results) and all(
        result.status == PASS for result in results
    )
    if args.code_version is None:
        _write_runtime_jsonl(output, results)
        print("=== MCP reliability gate ===")
        for result in results:
            suffix = f" - {result.message}" if result.message else ""
            print(f"[{result.status}] {result.case_id} ({result.duration_ms}ms){suffix}")
        counts = summary["counts"]
        print(
            f"Summary: PASS={counts.get(PASS, 0)} FAIL={counts.get(FAIL, 0)} "
            f"ERROR={counts.get(ERROR, 0)}"
        )
        print(f"Gate: {'PASS' if summary['gate_pass'] else 'FAIL'}")
        print(f"Results: {output}")
        return 0 if summary["gate_pass"] else 1
    gate_status = required_gate_status(
        full_coverage=coverage["full_coverage"],
        statuses=statuses,
        gate_pass=summary["gate_pass"],
    )
    protocol = build_protocol(
        protocol_id="mcp-reliability",
        protocol_version="1.0.0",
        descriptor={
            "track": "lifecycle_reliability",
            "required_case_ids": list(RELIABILITY_CASE_IDS),
            "grader": "deterministic_assertions",
            "gate_rule": "all required cases appear exactly once and have status PASS",
            "pass_rate_denominator": ["PASS", "FAIL"],
        },
        repo_root=REPO_ROOT,
        source_paths=(
            "eval/mcp_eval/evidence.py",
            "eval/mcp_eval/reliability_cases.py",
            "eval/mcp_eval/reliability.py",
        ),
    )
    write_evidence_jsonl(
        output=output,
        protocol=protocol,
        repo_root=REPO_ROOT,
        code_version=resolve_code_version(REPO_ROOT, args.code_version),
        interpretation={
            "claim": "offline_mcp_lifecycle_reliability_gate",
            "what_this_does_not_prove": (
                "production uptime, long-duration stability, or real-server diversity"
            ),
        },
        summary_payload={
            "track": "lifecycle_reliability",
            "status": gate_status,
            "gate_status": gate_status,
            "gate_pass": summary["gate_pass"],
            "coverage": coverage,
            "status_counts": dict(summary["counts"]),
            "metrics": status_metrics(statuses),
        },
        case_payloads=[
            {
                "case_id": result.case_id,
                "status": result.status,
                "eligible_for_pass_rate": result.status in {PASS, FAIL},
                "duration_ms": result.duration_ms,
                "evidence": dict(result.evidence),
                "message": result.message,
            }
            for result in results
        ],
        execution_mode="offline_deterministic",
        api_calls=0,
        llm_calls=0,
    )

    print("=== MCP reliability gate ===")
    for result in results:
        suffix = f" - {result.message}" if result.message else ""
        print(f"[{result.status}] {result.case_id} ({result.duration_ms}ms){suffix}")
    counts = summary["counts"]
    print(
        f"Summary: PASS={counts.get(PASS, 0)} FAIL={counts.get(FAIL, 0)} "
        f"ERROR={counts.get(ERROR, 0)}"
    )
    if summary["full_gate_coverage"]:
        print(f"Gate (all reliability cases): {'PASS' if summary['gate_pass'] else 'FAIL'}")
    else:
        print(
            "Selected checks: "
            f"{'PASS' if summary['selected_pass'] else 'FAIL'} "
            "(full reliability gate not run)"
        )
    print(f"Results: {output}")
    passed = (
        summary["gate_pass"]
        if summary["full_gate_coverage"]
        else summary["selected_pass"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
