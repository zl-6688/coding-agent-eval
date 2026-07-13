#!/usr/bin/env python3
"""Build a compact trace-to-patch evidence table for EvoClaw paired arms.

This helper is deliberately diagnostic. It joins paired evaluation outcomes,
compact spans, and keyword evidence from trace JSONL files, but it does not
claim compression causality by itself.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass
class KeywordHit:
    file: str
    span_name: str
    purpose: str
    start_utc: str | None
    field: str
    keyword: str
    snippet: str


@dataclass
class CompactEvent:
    arm: str
    file: str
    name: str
    start_utc: str | None
    tokens_before: int | None
    tokens_after: int | None
    compact_turn_no: int | None
    compact_llm_calls: int | None


@dataclass
class DifferentialMilestone:
    milestone_id: str
    direction: str
    full_resolved: bool
    sm_resolved: bool
    full_fail_to_pass_failures: list[str] = field(default_factory=list)
    sm_fail_to_pass_failures: list[str] = field(default_factory=list)
    full_pass_to_pass_failures: list[str] = field(default_factory=list)
    sm_pass_to_pass_failures: list[str] = field(default_factory=list)
    root_cause_candidate: str = ""
    evidence_keywords: list[str] = field(default_factory=list)
    full_keyword_hits: list[KeywordHit] = field(default_factory=list)
    sm_keyword_hits: list[KeywordHit] = field(default_factory=list)


@dataclass
class TraceToPatchEvidence:
    schema_version: int
    run_id: str
    status: str
    compact_events: dict[str, list[CompactEvent]]
    differential_milestones: list[DifferentialMilestone]
    limits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iter_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.jsonl") if p.is_file())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if isinstance(event, dict):
            yield event


def _iso_from_ns(value: Any) -> str | None:
    try:
        ns = int(value)
    except Exception:
        return None
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _snippet(text: str, keyword: str, *, radius: int = 180) -> str:
    lowered = text.lower()
    idx = lowered.find(keyword.lower())
    if idx < 0:
        return text[: radius * 2]
    start = max(0, idx - radius)
    end = min(len(text), idx + len(keyword) + radius)
    return text[start:end].replace("\n", "\\n")


def _event_text_fields(event: dict[str, Any]) -> Iterable[tuple[str, str]]:
    attrs = event.get("attributes") or {}
    if not isinstance(attrs, dict):
        return
    for key in ("llm.input", "llm.output", "status_message"):
        value = attrs.get(key)
        if isinstance(value, str):
            yield key, value


def extract_compact_events(path: Path, *, arm: str) -> list[CompactEvent]:
    events: list[CompactEvent] = []
    for file_path in _iter_trace_files(path):
        for event in _load_jsonl(file_path):
            name = str(event.get("name") or "")
            if name not in {"compact.full_compact", "compact.session_memory_compact"}:
                continue
            attrs = event.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            events.append(
                CompactEvent(
                    arm=arm,
                    file=file_path.name,
                    name=name,
                    start_utc=_iso_from_ns(event.get("start_ns")),
                    tokens_before=_int_or_none(attrs.get("tokens_before")),
                    tokens_after=_int_or_none(attrs.get("tokens_after")),
                    compact_turn_no=_int_or_none(attrs.get("compact_turn_no")),
                    compact_llm_calls=_int_or_none(attrs.get("compact_llm_calls")),
                )
            )
    return events


def find_keyword_hits(path: Path, *, keywords: Iterable[str], max_hits: int = 8) -> list[KeywordHit]:
    keyword_list = sorted(
        [keyword for keyword in keywords if keyword],
        key=lambda item: (-len(item), item.lower()),
    )
    hits: list[KeywordHit] = []
    trace_files = _iter_trace_files(path)
    for keyword in keyword_list:
        for file_path in trace_files:
            for event in _load_jsonl(file_path):
                attrs = event.get("attributes") or {}
                if not isinstance(attrs, dict):
                    attrs = {}
                for field_name, text in _event_text_fields(event):
                    lower = text.lower()
                    if keyword.lower() not in lower:
                        continue
                    hits.append(
                        KeywordHit(
                            file=file_path.name,
                            span_name=str(event.get("name") or ""),
                            purpose=str(attrs.get("llm.purpose") or ""),
                            start_utc=_iso_from_ns(event.get("start_ns")),
                            field=field_name,
                            keyword=keyword,
                            snippet=_snippet(text, keyword),
                        )
                    )
                    if len(hits) >= max_hits:
                        return hits
    return hits


def _failures(side: dict[str, Any], bucket: str) -> list[str]:
    raw = side.get(f"{bucket}_failures") or {}
    if isinstance(raw, dict):
        value = raw.get("fail_to_pass" if bucket == "full" else "fail_to_pass")
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _pass_to_pass(side: dict[str, Any], bucket: str) -> list[str]:
    raw = side.get(f"{bucket}_failures") or {}
    if isinstance(raw, dict):
        value = raw.get("pass_to_pass")
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _candidate_for(row: dict[str, Any]) -> tuple[str, list[str], str]:
    milestone = str(row.get("milestone_id") or "")
    full_failures = _failures(row, "full")
    sm_failures = _failures(row, "sm")
    sm_p2p = _pass_to_pass(row, "sm")
    joined = " ".join([milestone, *full_failures, *sm_failures, *sm_p2p])
    if "json::replacement" in joined:
        return (
            "json_replacement_schema",
            ["json::replacement", "Data::from_bytes", "serialize_field", "replacement"],
            "SM-only required-test failure likely needs JSON replacement patch diff inspection.",
        )
    if "context_" in joined or "absolute_byte_offset" in joined:
        return (
            "searcher_context_byte_count",
            ["context_code", "context_sherlock", "absolute_byte_offset", "core.pos"],
            "SM pass-to-pass regressions likely need Searcher::finish byte-count diff inspection.",
        )
    if row.get("direction") != "same":
        return (
            "paired_outcome_difference",
            [milestone, *sm_failures, *full_failures],
            "Arm outcome differs; inspect patch diff and trace around the milestone tag.",
        )
    if row.get("sm_pass_to_pass_failed", 0) != row.get("full_pass_to_pass_failed", 0):
        return (
            "pass_to_pass_delta",
            [milestone, *sm_p2p],
            "Pass-to-pass failure count differs; inspect regression tests and patch diff.",
        )
    return "", [], ""


def _differential_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = summary.get("paired_results", {}).get("raw_paired", [])
    if not isinstance(rows, list):
        return []
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("direction") != "same":
            selected.append(row)
            continue
        if row.get("sm_pass_to_pass_failed", 0) != row.get("full_pass_to_pass_failed", 0):
            selected.append(row)
            continue
        if row.get("sm_fail_to_pass") != row.get("full_fail_to_pass"):
            selected.append(row)
    return selected


def build_evidence(
    summary_path: str | Path,
    *,
    full_traces: str | Path,
    sm_traces: str | Path,
    max_hits: int = 8,
) -> TraceToPatchEvidence:
    summary = _load_json(Path(summary_path))
    full_path = Path(full_traces)
    sm_path = Path(sm_traces)

    milestones: list[DifferentialMilestone] = []
    for row in _differential_rows(summary):
        candidate, keywords, description = _candidate_for(row)
        full_failures = row.get("full_failures") or {}
        sm_failures = row.get("sm_failures") or {}
        milestones.append(
            DifferentialMilestone(
                milestone_id=str(row.get("milestone_id") or ""),
                direction=str(row.get("direction") or ""),
                full_resolved=bool(row.get("full_resolved")),
                sm_resolved=bool(row.get("sm_resolved")),
                full_fail_to_pass_failures=[str(item) for item in full_failures.get("fail_to_pass", [])],
                sm_fail_to_pass_failures=[str(item) for item in sm_failures.get("fail_to_pass", [])],
                full_pass_to_pass_failures=[str(item) for item in full_failures.get("pass_to_pass", [])],
                sm_pass_to_pass_failures=[str(item) for item in sm_failures.get("pass_to_pass", [])],
                root_cause_candidate=description or candidate,
                evidence_keywords=keywords,
                full_keyword_hits=find_keyword_hits(full_path, keywords=keywords, max_hits=max_hits),
                sm_keyword_hits=find_keyword_hits(sm_path, keywords=keywords, max_hits=max_hits),
            )
        )

    return TraceToPatchEvidence(
        schema_version=1,
        run_id=str(summary.get("run_id") or ""),
        status=str(summary.get("status") or ""),
        compact_events={
            "full": extract_compact_events(full_path, arm="full"),
            "sm": extract_compact_events(sm_path, arm="sm"),
        },
        differential_milestones=milestones,
        limits=[
            "keyword hits are trace evidence, not proof of compression causality",
            "patch diffs and official evaluation logs remain the source of truth for root cause",
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="fork-at-compact summary JSON")
    parser.add_argument("--full-traces", required=True, help="full arm trace JSONL file or directory")
    parser.add_argument("--sm-traces", required=True, help="SM arm trace JSONL file or directory")
    parser.add_argument("--max-hits", type=int, default=8)
    parser.add_argument("--out", help="Optional JSON output path")
    args = parser.parse_args(argv)

    result = build_evidence(
        args.summary,
        full_traces=args.full_traces,
        sm_traces=args.sm_traces,
        max_hits=args.max_hits,
    )
    payload = result.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
