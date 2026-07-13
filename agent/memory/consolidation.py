"""Minimal deterministic consolidation for AutoMemory topic files.

This module intentionally stays smaller than Claude Code autoDream. It only
merges exact duplicate topics by normalized name and type, then delegates index
repair to governance so consolidation remains predictable and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import secret_scan
from .governance import (
    MEMORY_TYPES,
    MemoryPrunePlan,
    MemoryRecord,
    delete_memory,
    list_memories,
    prune_memories,
    update_memory,
)


@dataclass(frozen=True)
class MemoryMergeGroup:
    """One exact duplicate set because file-level auditability is required."""

    normalized_name: str
    memory_type: str
    canonical_file: str
    duplicate_files: list[str]


@dataclass(frozen=True)
class MemoryConsolidationSkip:
    """Skipped work is structured because secret and safety decisions must be visible."""

    reason: str
    files: list[str]
    secret_hits: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MemoryConsolidationPlan:
    """Consolidation output keeps merge and prune evidence together for review."""

    dry_run: bool
    applied: bool
    actions: list[str]
    prune_plan: MemoryPrunePlan
    merged_groups: list[MemoryMergeGroup]
    skipped: list[MemoryConsolidationSkip]


def consolidate_memories(
    memory_dir: str | Path,
    *,
    dry_run: bool = True,
) -> MemoryConsolidationPlan:
    """Merge only mechanically identical topic memories and prune the index.

    The function is synchronous and deterministic because this project needs a
    safe maintenance primitive before any LLM-based or background autoDream
    workflow. A topic is mergeable only when its normalized frontmatter name and
    exact type match another topic; no semantic similarity judgment is made.
    """

    root = Path(memory_dir)
    records = list_memories(root)
    groups = _duplicate_groups(records)
    actions: list[str] = []
    merged_groups: list[MemoryMergeGroup] = []
    skipped: list[MemoryConsolidationSkip] = []
    wrote_topics = False

    for records_in_group in groups:
        canonical, duplicates = _choose_canonical(records_in_group)
        merge_group = MemoryMergeGroup(
            normalized_name=_normalize_topic_name(canonical.name),
            memory_type=canonical.type,
            canonical_file=canonical.file_name,
            duplicate_files=[record.file_name for record in duplicates],
        )
        merged_body = _merged_body(canonical, duplicates)
        scan_payload = "\n".join(
            [canonical.name, canonical.description, canonical.type, merged_body]
        )
        secret_hits = secret_scan.scan(scan_payload)
        if secret_hits:
            skipped.append(
                MemoryConsolidationSkip(
                    reason="secret_scan",
                    files=[canonical.file_name, *merge_group.duplicate_files],
                    secret_hits=secret_hits,
                )
            )
            actions.append(
                "skip merge "
                f"{canonical.file_name} <- {', '.join(merge_group.duplicate_files)} "
                f"because secret scan matched {', '.join(secret_hits)}"
            )
            continue

        merged_groups.append(merge_group)
        if dry_run:
            actions.append(
                "would merge "
                f"{canonical.file_name} <- {', '.join(merge_group.duplicate_files)}"
            )
            continue

        update_memory(root, canonical.file_name, body=merged_body)
        for duplicate in duplicates:
            delete_memory(root, duplicate.file_name)
        wrote_topics = True
        actions.append(
            f"merge {canonical.file_name} <- {', '.join(merge_group.duplicate_files)}"
        )

    prune_plan = prune_memories(root, dry_run=dry_run)
    applied = False if dry_run else wrote_topics or prune_plan.applied
    return MemoryConsolidationPlan(
        dry_run=dry_run,
        applied=applied,
        actions=actions,
        prune_plan=prune_plan,
        merged_groups=merged_groups,
        skipped=skipped,
    )


def _duplicate_groups(records: list[MemoryRecord]) -> list[list[MemoryRecord]]:
    """Group only exact topic duplicates so consolidation cannot infer semantics."""
    by_key: dict[tuple[str, str], list[MemoryRecord]] = {}
    for record in records:
        normalized_name = _normalize_topic_name(record.name)
        if not normalized_name or record.type not in MEMORY_TYPES:
            continue
        by_key.setdefault((normalized_name, record.type), []).append(record)
    return [
        sorted(group, key=lambda record: record.file_name.casefold())
        for group in by_key.values()
        if len(group) >= 2
    ]


def _choose_canonical(records: list[MemoryRecord]) -> tuple[MemoryRecord, list[MemoryRecord]]:
    """Pick a stable canonical file so dry-run and apply plans are reproducible."""
    sorted_records = sorted(records, key=lambda record: record.file_name.casefold())
    return sorted_records[0], sorted_records[1:]


def _normalize_topic_name(name: str) -> str:
    """Normalize only whitespace and case to avoid surprising topic equivalence."""
    return " ".join(str(name).split()).casefold()


def _merged_body(canonical: MemoryRecord, duplicates: list[MemoryRecord]) -> str:
    """Append duplicate bodies under source sections to keep the audit chain intact."""
    parts = [canonical.body] if canonical.body else []
    for duplicate in duplicates:
        parts.append(_source_section(duplicate))
    return "\n\n".join(parts)


def _source_section(record: MemoryRecord) -> str:
    """Render duplicate source metadata next to its original body for later review."""
    body = record.body
    metadata = (
        f"Source name: {record.name}\n"
        f"Source type: {record.type}\n"
        f"Source description: {record.description}"
    )
    if body:
        return f"## Consolidated From {record.file_name}\n\n{metadata}\n\n{body}"
    return f"## Consolidated From {record.file_name}\n\n{metadata}"


__all__ = [
    "MemoryMergeGroup",
    "MemoryConsolidationSkip",
    "MemoryConsolidationPlan",
    "consolidate_memories",
]
