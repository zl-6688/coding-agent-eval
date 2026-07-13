"""Run the controlled SessionMemory long-session write-to-takeover probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.compression_eval.sm_long_session_probe import (  # noqa: E402
    render_probe_report,
    run_controlled_long_session_probe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a no-API long-session SessionMemory write-to-takeover probe.",
    )
    parser.add_argument(
        "--out",
        default=".traces/sm_long_session_probe",
        help="Output directory for trace and SessionMemory artifacts.",
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
    result = run_controlled_long_session_probe(args.out)
    if args.format == "json":
        print(
            json.dumps(
                {
                    "trace_path": str(result.trace_path),
                    "workspace": str(result.workspace),
                    "sm_path": str(result.sm_path),
                    "final_text": result.final_text,
                    "sm_written": result.sm_written,
                    "capture_gate": result.capture_gate,
                    "initial_context_tokens": result.initial_context_tokens,
                    "one_tool_result_tokens": result.one_tool_result_tokens,
                    "compact_threshold": result.compact_threshold,
                    "main_llm_calls": result.main_llm_calls,
                    "fork_llm_calls": result.fork_llm_calls,
                    "memory_fork_spans": result.memory_fork_spans,
                    "full_stub_spans": result.full_stub_spans,
                    "takeover_summary": result.takeover_summary,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_probe_report(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
