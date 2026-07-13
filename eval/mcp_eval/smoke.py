"""MCP mechanism smoke evaluation entry point.

    python -m eval.mcp_eval.smoke
    python -m eval.mcp_eval.smoke --cases mcp_smoke_01_list_call,mcp_smoke_02_permission_deny
"""

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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from eval.mcp_eval.cases import (  # noqa: E402
    DEFAULT_CASE_IDS,
    ERROR,
    FAIL,
    PASS,
    REQUIRED_CASE_IDS,
    SKIPPED,
    available_case_ids,
    run_cases,
    summarize_results,
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


def _parse_case_list(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"smoke_{stamp}.jsonl"


def _print_summary(results, summary, output_path: Path) -> None:
    print("=== MCP smoke eval ===")
    for result in results:
        line = f"[{result.status}] {result.case_id} ({result.duration_ms}ms)"
        if result.message:
            line += f" — {result.message}"
        print(line)
    counts = summary["counts"]
    print(
        f"\nSummary: PASS={counts.get(PASS, 0)} "
        f"FAIL={counts.get(FAIL, 0)} "
        f"SKIPPED={counts.get(SKIPPED, 0)} "
        f"ERROR={counts.get(ERROR, 0)}"
    )
    if summary["full_gate_coverage"]:
        print(f"Gate (all required cases): {'PASS' if summary['gate_pass'] else 'FAIL'}")
    else:
        print(
            "Selected checks: "
            f"{'PASS' if summary['selected_pass'] else 'FAIL'} "
            "(full required gate not run)"
        )
    print(f"Results: {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCP mechanism smoke eval (no LLM)")
    parser.add_argument(
        "--cases",
        help="Comma-separated case ids (default: 01-05 full required gate)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="jsonl output path (default: eval/mcp_eval/results/smoke_<utc>.jsonl)",
    )
    parser.add_argument(
        "--code-version",
        help="portable code version recorded in evidence (default: git or WORKTREE)",
    )
    args = parser.parse_args(argv)

    case_ids = _parse_case_list(args.cases)
    if case_ids:
        unknown = [case_id for case_id in case_ids if case_id not in available_case_ids()]
        if unknown:
            print(f"Unknown case ids: {', '.join(unknown)}", file=sys.stderr)
            print(f"Available: {', '.join(available_case_ids())}", file=sys.stderr)
            return 2

    results = run_cases(case_ids)
    summary = summarize_results(results)
    output_path = args.output or _default_output_path()
    coverage = required_coverage(
        REQUIRED_CASE_IDS,
        [result.case_id for result in results],
    )
    required_results = [
        result for result in results if result.case_id in REQUIRED_CASE_IDS
    ]
    required_statuses = [result.status for result in required_results]
    # The runtime summary intentionally stays small; the public evidence layer
    # derives selection/coverage fields without changing the case runners.
    summary["full_gate_coverage"] = coverage["full_coverage"]
    summary["selected_pass"] = bool(results) and all(
        result.status == PASS for result in results
    )
    summary["required_non_pass"] = [
        result.case_id for result in required_results if result.status != PASS
    ]
    if args.code_version is None:
        _write_runtime_jsonl(output_path, results)
        _print_summary(results, summary, output_path)
        passed = (
            summary["gate_pass"]
            if summary["full_gate_coverage"]
            else summary["selected_pass"]
        )
        return 0 if passed else 1
    gate_status = required_gate_status(
        full_coverage=coverage["full_coverage"],
        statuses=required_statuses,
        gate_pass=summary["gate_pass"],
    )
    protocol = build_protocol(
        protocol_id="mcp-smoke",
        protocol_version="1.0.0",
        descriptor={
            "track": "mechanism_smoke",
            "required_case_ids": list(REQUIRED_CASE_IDS),
            "grader": "deterministic_rule_based",
            "gate_rule": "all required cases appear exactly once and have status PASS",
            "pass_rate_denominator": ["PASS", "FAIL"],
        },
        repo_root=REPO_ROOT,
        source_paths=(
            "eval/mcp_eval/evidence.py",
            "eval/mcp_eval/cases.py",
            "eval/mcp_eval/smoke.py",
        ),
    )
    case_payloads = [
        {
            "case_id": result.case_id,
            "status": result.status,
            "eligible_for_pass_rate": result.status in {PASS, FAIL},
            "duration_ms": result.duration_ms,
            "evidence": dict(result.evidence),
            "message": result.message,
        }
        for result in results
    ]
    write_evidence_jsonl(
        output=output_path,
        protocol=protocol,
        repo_root=REPO_ROOT,
        code_version=resolve_code_version(REPO_ROOT, args.code_version),
        interpretation={
            "claim": "offline_mcp_mechanism_gate",
            "what_this_does_not_prove": (
                "real-model task success, production-scale reliability, or product benefit"
            ),
        },
        summary_payload={
            "track": "mechanism_smoke",
            "status": gate_status,
            "gate_status": gate_status,
            "gate_pass": summary["gate_pass"],
            "coverage": coverage,
            "status_counts": dict(summary["counts"]),
            "metrics": status_metrics(required_statuses),
            "required_non_pass": list(summary["required_non_pass"]),
        },
        case_payloads=case_payloads,
        execution_mode="offline_deterministic",
        api_calls=0,
        llm_calls=0,
    )
    _print_summary(results, summary, output_path)
    passed = (
        summary["gate_pass"]
        if summary["full_gate_coverage"]
        else summary["selected_pass"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
