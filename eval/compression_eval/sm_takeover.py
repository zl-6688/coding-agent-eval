"""Analyze SessionMemory compact takeover from trace JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SM_SPAN = "compact.session_memory_compact"
PIPELINE_SPAN = "compact.pipeline"


@dataclass
class SmTakeoverStats:
    files_scanned: int = 0
    events_scanned: int = 0
    malformed_lines: int = 0
    sm_attempts: int = 0
    sm_status_counts: Counter[str] = field(default_factory=Counter)
    sm_direct_or_unlinked: int = 0
    pipeline_spans: int = 0
    pipeline_with_sm_attempt: int = 0
    pipeline_did_sm_true: int = 0
    pipeline_did_full_true: int = 0
    pipeline_did_full_true_after_sm_attempt: int = 0
    saved_full_compact_calls_estimate: int = 0

    @property
    def sm_ok(self) -> int:
        return self.sm_status_counts.get("ok", 0)

    @property
    def takeover_rate(self) -> float:
        return self.sm_ok / self.sm_attempts if self.sm_attempts else 0.0


def discover_jsonl(paths: Iterable[str | Path]) -> list[Path]:
    """Return JSONL files under files/directories, sorted for stable reports."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(p for p in path.rglob("*.jsonl") if p.is_file())
        elif path.is_file() and path.suffix.lower() == ".jsonl":
            found.append(path)
    return sorted(set(found))


def parse_since(value: str | None) -> int | None:
    """Parse YYYY-MM-DD, ISO datetime, or integer nanoseconds into epoch ns."""
    if not value:
        return None
    text = value.strip()
    if text.isdigit():
        return int(text)
    if len(text) == 10:
        dt = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _load_events(files: Iterable[Path], *, since_ns: int | None = None) -> tuple[list[dict], int, int]:
    events: list[dict] = []
    malformed = 0
    total = 0
    for path in files:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                total += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                start_ns = event.get("start_ns")
                if since_ns is not None and isinstance(start_ns, int) and start_ns < since_ns:
                    continue
                events.append(event)
    return events, total, malformed


def analyze_paths(
    paths: Iterable[str | Path],
    *,
    since: str | None = None,
    require_pipeline_parent: bool = False,
) -> SmTakeoverStats:
    """Analyze SM compact spans and related pipeline spans from trace files."""
    files = discover_jsonl(paths)
    since_ns = parse_since(since)
    events, total, malformed = _load_events(files, since_ns=since_ns)

    stats = SmTakeoverStats(
        files_scanned=len(files),
        events_scanned=total,
        malformed_lines=malformed,
    )

    pipelines = [
        event for event in events
        if event.get("name") == PIPELINE_SPAN
    ]
    pipeline_by_span = {
        (event.get("trace_id"), event.get("span_id")): event
        for event in pipelines
    }
    sm_children_by_pipeline: dict[tuple[str | None, str | None], list[dict]] = {}

    sm_events = [event for event in events if event.get("name") == SM_SPAN]
    selected_sm: list[dict] = []
    for event in sm_events:
        parent_key = (event.get("trace_id"), event.get("parent_span_id"))
        linked = parent_key in pipeline_by_span
        if linked:
            sm_children_by_pipeline.setdefault(parent_key, []).append(event)
        if require_pipeline_parent and not linked:
            continue
        selected_sm.append(event)
        if not linked:
            stats.sm_direct_or_unlinked += 1

    stats.sm_attempts = len(selected_sm)
    for event in selected_sm:
        attrs = event.get("attributes") or {}
        stats.sm_status_counts[str(attrs.get("status", "unknown"))] += 1

    stats.pipeline_spans = len(pipelines)
    for pipeline in pipelines:
        attrs = pipeline.get("attributes") or {}
        key = (pipeline.get("trace_id"), pipeline.get("span_id"))
        has_selected_sm = bool(sm_children_by_pipeline.get(key))
        if has_selected_sm:
            stats.pipeline_with_sm_attempt += 1
        did_sm = bool(attrs.get("did_sm"))
        did_full = bool(attrs.get("did_full"))
        if did_sm:
            stats.pipeline_did_sm_true += 1
        if did_full:
            stats.pipeline_did_full_true += 1
        if has_selected_sm and did_full:
            stats.pipeline_did_full_true_after_sm_attempt += 1
        if did_sm and not did_full:
            stats.saved_full_compact_calls_estimate += 1

    return stats


def stats_to_dict(stats: SmTakeoverStats) -> dict:
    return {
        "files_scanned": stats.files_scanned,
        "events_scanned": stats.events_scanned,
        "malformed_lines": stats.malformed_lines,
        "sm_attempts": stats.sm_attempts,
        "sm_ok": stats.sm_ok,
        "takeover_rate": round(stats.takeover_rate, 4),
        "sm_status_counts": dict(sorted(stats.sm_status_counts.items())),
        "sm_direct_or_unlinked": stats.sm_direct_or_unlinked,
        "pipeline_spans": stats.pipeline_spans,
        "pipeline_with_sm_attempt": stats.pipeline_with_sm_attempt,
        "pipeline_did_sm_true": stats.pipeline_did_sm_true,
        "pipeline_did_full_true": stats.pipeline_did_full_true,
        "pipeline_did_full_true_after_sm_attempt": stats.pipeline_did_full_true_after_sm_attempt,
        "saved_full_compact_calls_estimate": stats.saved_full_compact_calls_estimate,
    }


def render_markdown(stats: SmTakeoverStats) -> str:
    data = stats_to_dict(stats)
    lines = [
        "# SessionMemory Compact Takeover",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Files scanned | {data['files_scanned']} |",
        f"| Events scanned | {data['events_scanned']} |",
        f"| Malformed lines | {data['malformed_lines']} |",
        f"| SM attempts | {data['sm_attempts']} |",
        f"| SM ok | {data['sm_ok']} |",
        f"| SM takeover rate | {data['takeover_rate']:.2%} |",
        f"| Direct/unlinked SM attempts | {data['sm_direct_or_unlinked']} |",
        f"| Pipeline spans | {data['pipeline_spans']} |",
        f"| Pipelines with SM attempt | {data['pipeline_with_sm_attempt']} |",
        f"| Pipelines did_sm=true | {data['pipeline_did_sm_true']} |",
        f"| Pipelines did_full=true | {data['pipeline_did_full_true']} |",
        f"| Pipelines did_full=true after SM attempt | {data['pipeline_did_full_true_after_sm_attempt']} |",
        f"| Avoided sync full_compact calls estimate | {data['saved_full_compact_calls_estimate']} |",
        "",
        "## SM Status Counts",
        "",
        "| status | count |",
        "|---|---:|",
    ]
    if stats.sm_status_counts:
        for status, count in sorted(stats.sm_status_counts.items()):
            lines.append(f"| `{status}` | {count} |")
    else:
        lines.append("| _none_ | 0 |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze compact.session_memory_compact takeover and fallback distribution from trace JSONL files.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[".traces"],
        help="Trace JSONL file(s) or directory/directories to scan. Defaults to .traces.",
    )
    parser.add_argument(
        "--since",
        help="Only include events at or after YYYY-MM-DD, ISO datetime, or epoch nanoseconds.",
    )
    parser.add_argument(
        "--require-pipeline-parent",
        action="store_true",
        help="Count only SM spans whose parent is a compact.pipeline span.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stats = analyze_paths(
        args.paths,
        since=args.since,
        require_pipeline_parent=args.require_pipeline_parent,
    )
    if args.format == "json":
        print(json.dumps(stats_to_dict(stats), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_markdown(stats), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
