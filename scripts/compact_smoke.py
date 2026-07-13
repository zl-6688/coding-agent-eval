#!/usr/bin/env python3
"""Compatibility entry point for the current offline context-budget gate.

The historical layered NIAH script is retained under ``eval/_archive`` for
design archaeology, but it targets APIs that no longer exist. This command
therefore runs the maintained deterministic context-budget protocol instead;
it does not claim to reproduce the archived experiment.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSONL evidence here; omit for a temporary smoke artifact",
    )
    parser.add_argument("--cases", help="comma-separated context gate case IDs")
    parser.add_argument("--code-version", default="WORKTREE")
    return parser


def _run(output: Path, *, cases: str | None, code_version: str) -> int:
    from eval.context_eval.run import main as context_gate_main

    argv = ["--output", str(output), "--code-version", code_version]
    if cases:
        argv.extend(["--cases", cases])
    print(
        "[compat] Running the maintained context-budget offline gate. "
        "The archived layered NIAH protocol is not being reproduced."
    )
    return context_gate_main(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output is not None:
        return _run(args.output, cases=args.cases, code_version=args.code_version)
    with tempfile.TemporaryDirectory(prefix="ace-context-smoke-") as directory:
        output = Path(directory) / "context-budget.jsonl"
        status = _run(output, cases=args.cases, code_version=args.code_version)
        print("[compat] Temporary evidence was validated and removed.")
        return status


if __name__ == "__main__":
    raise SystemExit(main())
