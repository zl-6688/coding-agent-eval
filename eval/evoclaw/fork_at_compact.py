#!/usr/bin/env python3
"""Validate EvoClaw seed traces for a fork-at-compact experiment.

This script intentionally does not clone Docker state. It only answers the
first gate: did the seed arm stop cleanly before compaction, with enough trace
evidence to justify freezing that state for later A/B forks?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


COMPACT_SPAN_PREFIX = "compact."
RUN_SPAN = "agent.run"
TURN_SPAN = "agent.turn"


@dataclass
class SeedValidation:
    status: str
    trace_files: int = 0
    run_spans: int = 0
    snapshot_cut_spans: int = 0
    compact_spans: int = 0
    compact_strategies: list[str] = field(default_factory=list)
    stop_at_context: int = 0
    peak_context_tokens: int = 0
    max_turn_context_tokens: int = 0
    arms: list[str] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict:
        return asdict(self)


def _iter_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.jsonl") if p.is_file())


def _load_events(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if isinstance(event, dict):
            yield event


def _int_attr(attrs: dict, key: str) -> int:
    try:
        return int(attrs.get(key) or 0)
    except Exception:
        return 0


def validate_seed_traces(path: str | Path) -> SeedValidation:
    """Validate that seed traces represent a clean pre-compact snapshot cut."""

    trace_path = Path(path)
    trace_files = _iter_trace_files(trace_path)
    result = SeedValidation(status="FAIL", trace_files=len(trace_files))

    compact_strategies: set[str] = set()
    arms: set[str] = set()
    session_ids: set[str] = set()
    stop_thresholds: set[int] = set()

    for file_path in trace_files:
        for event in _load_events(file_path):
            name = event.get("name")
            attrs = event.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}

            if isinstance(name, str) and name.startswith(COMPACT_SPAN_PREFIX):
                result.compact_spans += 1
                continue

            if name == TURN_SPAN:
                result.max_turn_context_tokens = max(
                    result.max_turn_context_tokens,
                    _int_attr(attrs, "context_tokens"),
                )
                continue

            if name != RUN_SPAN:
                continue

            result.run_spans += 1
            strategy = str(attrs.get("compact_strategy") or "")
            if strategy:
                compact_strategies.add(strategy)
            threshold = _int_attr(attrs, "stop_at_context")
            if threshold > 0:
                stop_thresholds.add(threshold)
                result.stop_at_context = max(result.stop_at_context, threshold)
            result.peak_context_tokens = max(
                result.peak_context_tokens,
                _int_attr(attrs, "peak_context_tokens"),
            )
            meta = attrs.get("run_metadata") or {}
            if isinstance(meta, dict):
                if meta.get("arm"):
                    arms.add(str(meta["arm"]))
                if meta.get("session_id"):
                    session_ids.add(str(meta["session_id"]))
            if attrs.get("outcome") == "snapshot_cut":
                result.snapshot_cut_spans += 1

    result.compact_strategies = sorted(compact_strategies)
    result.arms = sorted(arms)
    result.session_ids = sorted(session_ids)

    if not trace_files:
        result.issues.append("no_trace_files")
    if result.run_spans == 0:
        result.issues.append("no_agent_run_span")
    if result.snapshot_cut_spans == 0:
        result.issues.append("no_snapshot_cut")
    if result.compact_spans:
        result.issues.append("compact_span_present")
    if compact_strategies and compact_strategies != {"none"}:
        result.issues.append("seed_compact_strategy_not_none")
    if not stop_thresholds:
        result.issues.append("missing_stop_at_context")
    elif len(stop_thresholds) > 1:
        result.issues.append("inconsistent_stop_at_context")
    if result.stop_at_context and result.peak_context_tokens < result.stop_at_context:
        result.issues.append("peak_context_below_stop_threshold")

    if result.snapshot_cut_spans > 1:
        result.warnings.append("multiple_snapshot_cuts_possible_recovery_pollution")
    if len(session_ids) > 1:
        result.warnings.append("multiple_session_ids")
    if len(arms) > 1:
        result.warnings.append("multiple_arms")

    result.status = "PASS" if not result.issues else "FAIL"
    return result


def _format_human(result: SeedValidation) -> str:
    lines = [
        f"status: {result.status}",
        f"trace_files: {result.trace_files}",
        f"run_spans: {result.run_spans}",
        f"snapshot_cut_spans: {result.snapshot_cut_spans}",
        f"compact_spans: {result.compact_spans}",
        f"compact_strategies: {', '.join(result.compact_strategies) or '-'}",
        f"stop_at_context: {result.stop_at_context}",
        f"peak_context_tokens: {result.peak_context_tokens}",
        f"max_turn_context_tokens: {result.max_turn_context_tokens}",
        f"arms: {', '.join(result.arms) or '-'}",
        f"session_ids: {', '.join(result.session_ids) or '-'}",
    ]
    if result.issues:
        lines.append("issues: " + ", ".join(result.issues))
    if result.warnings:
        lines.append("warnings: " + ", ".join(result.warnings))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, help="Trace JSONL file or directory")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--out", help="Optional JSON output path")
    args = parser.parse_args(argv)

    result = validate_seed_traces(args.traces)
    payload = result.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_format_human(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
