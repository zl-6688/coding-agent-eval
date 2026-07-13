"""MCP behavior eval entrypoint (live loop + fake grader self-test).

    python -m eval.mcp_eval.behavior --mode fake
    python -m eval.mcp_eval.behavior --mode live
"""

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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from eval.mcp_eval.behavior_cases import (  # noqa: E402
    BEHAVIOR_CASE_IDS,
    ERROR,
    FAIL,
    INCONCLUSIVE,
    PASS,
    SKIPPED,
    run_behavior_cases,
    summarize_behavior_results,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _parse_case_list(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _default_output_path(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"behavior_{mode}_{stamp}.jsonl"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_summary(results, summary, output_path: Path, mode: str) -> None:
    print(f"=== MCP behavior eval ({mode}) ===")
    for result in results:
        line = f"[{result.status}] {result.case_id}"
        if result.trial is not None:
            line += f" trial={result.trial}"
        line += f" ({result.duration_ms}ms)"
        if result.message:
            line += f" — {result.message}"
        print(line)
    counts = summary["counts"]
    print(
        f"\nSummary: PASS={counts.get(PASS, 0)} "
        f"FAIL={counts.get(FAIL, 0)} "
        f"INCONCLUSIVE={counts.get(INCONCLUSIVE, 0)} "
        f"SKIPPED={counts.get(SKIPPED, 0)} "
        f"ERROR={counts.get(ERROR, 0)}"
    )
    if summary["inconclusive"]:
        print(f"Inconclusive: {', '.join(summary['inconclusive'])}")
    per_case = summary.get("per_case") or {}
    if per_case:
        print("Per-case trials:")
        for case_id, stats in per_case.items():
            print(
                f"  {case_id}: PASS {stats['pass_count']}/{stats['trials']} "
                f"(statuses={stats['statuses']})"
            )
    print(f"Gate: {'PASS' if summary['gate_pass'] else 'FAIL'}")
    print(f"Results: {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCP behavior eval (fake graders or live loop)")
    parser.add_argument("--mode", choices=("fake", "live"), default="fake")
    parser.add_argument("--repeat", type=int, default=1, help="Run each case N times (live stability)")
    parser.add_argument("--cases", help="Comma-separated behavior case ids")
    parser.add_argument("--workdir", type=Path, help="Workspace for live runs (default: temp dir)")
    parser.add_argument("--output", type=Path, help="jsonl output path")
    args = parser.parse_args(argv)

    case_ids = _parse_case_list(args.cases)
    if case_ids:
        unknown = [case_id for case_id in case_ids if case_id not in BEHAVIOR_CASE_IDS]
        if unknown:
            print(f"Unknown case ids: {', '.join(unknown)}", file=sys.stderr)
            print(f"Available: {', '.join(BEHAVIOR_CASE_IDS)}", file=sys.stderr)
            return 2

    workdir = args.workdir
    if args.mode == "live":
        workdir = workdir or Path(tempfile.mkdtemp(prefix="mcp_behavior_"))
    else:
        workdir = workdir or REPO_ROOT

    results = run_behavior_cases(case_ids, mode=args.mode, workdir=workdir, repeat=args.repeat)
    summary = summarize_behavior_results(results)
    commit = _git_commit()
    timestamp = datetime.now(timezone.utc).isoformat()
    records = [result.to_record(commit=commit, timestamp=timestamp) for result in results]
    output_path = args.output or _default_output_path(args.mode)
    _write_jsonl(output_path, records)
    _print_summary(results, summary, output_path, args.mode)
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
