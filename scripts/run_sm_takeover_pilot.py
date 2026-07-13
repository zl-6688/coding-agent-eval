"""Run the deterministic SessionMemory compact takeover pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.compression_eval.sm_takeover import analyze_paths, render_markdown, stats_to_dict
from eval.compression_eval.sm_takeover_pilot import render_case_table, run_pilot_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run seeded SessionMemory compact takeover cases and summarize the generated traces."
        ),
    )
    parser.add_argument(
        "--out",
        default=".traces/sm_takeover_pilot",
        help="Output directory for pilot trace JSONL files.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    results = run_pilot_cases(out_dir)
    stats = analyze_paths([out_dir], require_pipeline_parent=True)

    if args.format == "json":
        payload = {
            "cases": [
                {
                    "name": item.name,
                    "trace_path": str(item.trace_path),
                    "capture_gate": item.capture_gate,
                    "sm_status": item.sm_status,
                    "pipeline_did_sm": item.pipeline_did_sm,
                    "pipeline_did_full": item.pipeline_did_full,
                    "output_tokens": item.output_tokens,
                }
                for item in results
            ],
            "summary": stats_to_dict(stats),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("# SessionMemory Takeover Pilot")
        print()
        print(render_case_table(results), end="")
        print()
        print(render_markdown(stats), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
