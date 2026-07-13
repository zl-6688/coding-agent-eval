"""Render a local JSONL trace as an offline HTML report.

Examples:
    python scripts/view_trace.py
    python scripts/view_trace.py path/to/run.jsonl
    python scripts/view_trace.py path/to/run.jsonl --output report.html
    python scripts/view_trace.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent import config
from obs.viewer import to_html


def _load(path: Path) -> list[dict]:
    events: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        events.append(event)
    return events


def _trace_files() -> list[Path]:
    return sorted(
        (path for path in config.TRACES_DIR.glob("*.jsonl") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
    )


def _run_measurements(events: list[dict]) -> tuple[str, str]:
    run = next((event for event in events if event.get("name") == "agent.run"), None)
    attrs = run.get("attributes", {}) if isinstance(run, dict) else {}
    if not isinstance(attrs, dict):
        attrs = {}
    outcome_value = attrs.get("outcome") if "outcome" in attrs else attrs.get("run.outcome")
    outcome = str(outcome_value or "not recorded")
    raw_turns = attrs.get("turns") if "turns" in attrs else attrs.get("run.turns")
    turns = str(raw_turns) if isinstance(raw_turns, int) and not isinstance(raw_turns, bool) else "?"
    return outcome, turns


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path, help="JSONL trace to render")
    parser.add_argument("--list", action="store_true", help="list recent trace files")
    parser.add_argument("--output", type=Path, help="HTML output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    traces = _trace_files()

    if args.list:
        if not traces:
            print(f"No trace files found in {config.TRACES_DIR}.", file=sys.stderr)
            return 1
        for path in traces[-30:]:
            try:
                events = _load(path)
                outcome, turns = _run_measurements(events)
                print(
                    f"{path.name}  {len(events)} spans  "
                    f"outcome={outcome}  turns={turns}"
                )
            except (OSError, ValueError) as exc:
                print(f"{path.name}  unreadable ({exc})")
        return 0

    if args.trace is not None:
        trace_path = args.trace
    elif traces:
        trace_path = traces[-1]
    else:
        print(f"No trace files found in {config.TRACES_DIR}.", file=sys.stderr)
        return 1

    output_path = args.output or trace_path.with_suffix(".html")
    try:
        events = _load(trace_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            to_html(events, f"{trace_path.stem} trace"),
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"Unable to render {trace_path}: {exc}", file=sys.stderr)
        return 2

    print(f"Trace: {trace_path} ({len(events)} spans)")
    print(f"HTML:  {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
