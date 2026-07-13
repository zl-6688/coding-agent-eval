"""Deterministic file governance for AutoMemory topic files.

The functions in this module deliberately work on the existing flat Markdown
format only. They do not infer semantic relevance, create new scopes, or add a
storage backend; they make the on-disk memory directory mechanically inspectable
and repairable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import secret_scan


INDEX_FILE_NAME = "MEMORY.md"
INDEX_HEADER = "# Memory Index\n"
INDEX_SEPARATOR = " — "
INDEX_LINE_MAX_CHARS = 240
MEMORY_TYPES = frozenset({"user", "feedback", "project", "reference"})

_INDEX_LINK_RE = re.compile(r"^\s*-\s+\[([^\]]+)\]\(([^)]+)\)(?P<tail>.*)$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class MemoryRecord:
    """Parsed memory topic plus index membership used by governance callers."""

    name: str
    file_name: str
    path: Path
    type: str
    description: str
    body: str
    mtime: float
    indexed: bool


@dataclass(frozen=True)
class MemoryHealthIssue:
    """One deterministic issue found in a memory directory."""

    kind: str
    message: str
    file_name: str | None = None
    path: Path | None = None
    line_number: int | None = None


@dataclass(frozen=True)
class MemoryPrunePlan:
    """Dry-run or applied repair plan returned by prune_memories."""

    issues: list[MemoryHealthIssue]
    actions: list[str] = field(default_factory=list)
    dry_run: bool = True
    applied: bool = False


@dataclass(frozen=True)
class _TopicParts:
    frontmatter: dict[str, str]
    body: str
    valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _IndexEntry:
    line_number: int
    raw: str
    label: str
    target: str
    description: str


def list_memories(memory_dir: str | Path) -> list[MemoryRecord]:
    """Return topic files with parsed frontmatter and index membership."""
    root = _memory_root(memory_dir)
    if not root.exists():
        return []

    indexed_names = {entry.target for entry in _read_index_entries(root) if _is_safe_topic_name(entry.target)}
    records: list[MemoryRecord] = []
    for path in _topic_files(root):
        parts = _read_topic_parts(path)
        stat = path.stat()
        records.append(
            MemoryRecord(
                name=parts.frontmatter.get("name") or path.stem,
                file_name=path.name,
                path=path,
                type=parts.frontmatter.get("type", ""),
                description=parts.frontmatter.get("description", ""),
                body=parts.body,
                mtime=stat.st_mtime,
                indexed=path.name in indexed_names,
            )
        )
    return sorted(records, key=lambda record: record.file_name.casefold())


def read_memory(memory_dir: str | Path, file_name: str) -> MemoryRecord:
    """Read one topic file, rejecting path traversal and MEMORY.md."""
    root = _memory_root(memory_dir)
    path = _resolve_topic_path(root, file_name)
    if not path.exists():
        raise FileNotFoundError(path)
    indexed_names = {entry.target for entry in _read_index_entries(root) if _is_safe_topic_name(entry.target)}
    parts = _read_topic_parts(path)
    stat = path.stat()
    return MemoryRecord(
        name=parts.frontmatter.get("name") or path.stem,
        file_name=path.name,
        path=path,
        type=parts.frontmatter.get("type", ""),
        description=parts.frontmatter.get("description", ""),
        body=parts.body,
        mtime=stat.st_mtime,
        indexed=path.name in indexed_names,
    )


def update_memory(
    memory_dir: str | Path,
    file_name: str,
    *,
    name: str | None = None,
    description: str | None = None,
    memory_type: str | None = None,
    body: str | None = None,
) -> MemoryRecord:
    """Update frontmatter/body, block secrets, and synchronize MEMORY.md."""
    root = _memory_root(memory_dir)
    path = _resolve_topic_path(root, file_name)
    if not path.exists():
        raise FileNotFoundError(path)

    current = read_memory(root, file_name)
    new_name = _clean_frontmatter_value(current.name if name is None else name)
    new_description = _clean_frontmatter_value(
        current.description if description is None else description
    )
    requested_type = _clean_frontmatter_value(
        current.type if memory_type is None else memory_type
    )
    new_body = current.body if body is None else str(body)

    hits = secret_scan.scan(f"{new_name}\n{new_description}\n{requested_type}\n{new_body}")
    if hits:
        raise ValueError(f"memory update blocked by secret scan: {', '.join(hits)}")

    new_type = requested_type if requested_type in MEMORY_TYPES else "reference"
    _write_topic(path, new_name, new_description, new_type, new_body)
    _upsert_index_line(root, path.name, new_description)
    return read_memory(root, path.name)


def delete_memory(memory_dir: str | Path, file_name: str) -> None:
    """Delete one topic file and remove every matching MEMORY.md index line."""
    root = _memory_root(memory_dir)
    path = _resolve_topic_path(root, file_name)
    if not path.exists():
        raise FileNotFoundError(path)
    path.unlink()
    _remove_index_lines(root, path.name)


def inspect_memory_health(memory_dir: str | Path) -> list[MemoryHealthIssue]:
    """Return read-only health issues for the memory directory."""
    root = _memory_root(memory_dir)
    if not root.exists():
        return []

    issues: list[MemoryHealthIssue] = []
    entries = _read_index_entries(root)
    seen_targets: set[str] = set()
    linked_existing_targets: set[str] = set()

    for entry in entries:
        if entry.target in seen_targets:
            issues.append(
                MemoryHealthIssue(
                    kind="duplicate_index",
                    message=f"duplicate MEMORY.md link for {entry.target}",
                    file_name=entry.target,
                    line_number=entry.line_number,
                )
            )
            continue
        seen_targets.add(entry.target)

        if len(entry.raw.rstrip("\n")) > INDEX_LINE_MAX_CHARS:
            issues.append(
                MemoryHealthIssue(
                    kind="oversized_index_line",
                    message=f"index line exceeds {INDEX_LINE_MAX_CHARS} characters",
                    file_name=entry.target,
                    line_number=entry.line_number,
                )
            )

        if not _is_safe_topic_name(entry.target):
            issues.append(
                MemoryHealthIssue(
                    kind="missing_target",
                    message=f"index target is not a top-level topic file: {entry.target}",
                    file_name=entry.target,
                    line_number=entry.line_number,
                )
            )
            continue

        path = _resolve_topic_path(root, entry.target)
        if not path.exists():
            issues.append(
                MemoryHealthIssue(
                    kind="missing_target",
                    message=f"indexed topic file is missing: {entry.target}",
                    file_name=entry.target,
                    path=path,
                    line_number=entry.line_number,
                )
            )
            continue
        linked_existing_targets.add(entry.target)

    for path in _topic_files(root):
        parts = _read_topic_parts(path)
        if path.name not in linked_existing_targets:
            issues.append(
                MemoryHealthIssue(
                    kind="orphan_topic",
                    message=f"topic file is not linked from {INDEX_FILE_NAME}",
                    file_name=path.name,
                    path=path,
                )
            )
        if not parts.valid:
            issues.append(
                MemoryHealthIssue(
                    kind="bad_frontmatter",
                    message="; ".join(parts.errors),
                    file_name=path.name,
                    path=path,
                )
            )

    return issues


def repair_memory_index(
    memory_dir: str | Path,
    *,
    dry_run: bool = True,
    add_orphans: bool = False,
) -> MemoryPrunePlan:
    """Repair structural MEMORY.md issues without overriding semantic pruning.

    AutoDream lets a forked agent decide whether an unindexed topic is stale,
    wrong, or superseded. This narrower repair path therefore defaults to not
    re-adding orphan topics, while prune_memories keeps its historical behavior.
    """
    return _repair_memory_index(memory_dir, dry_run=dry_run, add_orphans=add_orphans)


def prune_memories(memory_dir: str | Path, *, dry_run: bool = True) -> MemoryPrunePlan:
    """Plan or apply deterministic repairs without semantic memory deletion."""
    return _repair_memory_index(memory_dir, dry_run=dry_run, add_orphans=True)


def _repair_memory_index(
    memory_dir: str | Path,
    *,
    dry_run: bool,
    add_orphans: bool,
) -> MemoryPrunePlan:
    root = _memory_root(memory_dir)
    issues = inspect_memory_health(root)
    lines = _read_index_lines(root)
    entries_by_line = {entry.line_number: entry for entry in _read_index_entries(root)}
    seen_targets: set[str] = set()
    topic_parts = {path.name: _read_topic_parts(path) for path in _topic_files(root)}
    existing_topics = set(topic_parts)
    actions: list[str] = []
    new_lines: list[str] = []
    write_needed = False

    if lines:
        source_lines = lines
    else:
        source_lines = [INDEX_HEADER]
        if any(parts.valid for parts in topic_parts.values()):
            actions.append(f"create {INDEX_FILE_NAME}")
            write_needed = True

    for line_number, line in enumerate(source_lines, start=1):
        entry = entries_by_line.get(line_number)
        if entry is None:
            new_lines.append(line)
            continue

        if not _is_safe_topic_name(entry.target) or entry.target not in existing_topics:
            actions.append(f"remove missing index link {entry.target}")
            write_needed = True
            continue

        if entry.target in seen_targets:
            actions.append(f"remove duplicate index link {entry.target}")
            write_needed = True
            continue
        seen_targets.add(entry.target)

        parts = topic_parts[entry.target]
        if not parts.valid:
            new_lines.append(line)
            continue

        canonical = _make_index_line(
            entry.target,
            parts.frontmatter.get("description") or entry.description,
        )
        if secret_scan.scan(canonical):
            actions.append(f"skip secret index {entry.target}")
            new_lines.append(line)
            continue
        if canonical != line:
            actions.append(f"normalize index link {entry.target}")
            write_needed = True
        new_lines.append(canonical)

    if add_orphans:
        for topic_name in sorted(existing_topics - seen_targets, key=str.casefold):
            parts = topic_parts[topic_name]
            if not parts.valid:
                actions.append(f"skip invalid topic index {topic_name}")
                continue
            new_line = _make_index_line(topic_name, parts.frontmatter.get("description", ""))
            if secret_scan.scan(new_line):
                actions.append(f"skip secret index {topic_name}")
                continue
            _ensure_final_newline(new_lines)
            new_lines.append(new_line)
            actions.append(f"add orphan topic index {topic_name}")
            write_needed = True

    if dry_run:
        return MemoryPrunePlan(issues=issues, actions=actions, dry_run=True, applied=False)

    if write_needed:
        output = "".join(new_lines)
        hits = secret_scan.scan(output)
        if hits:
            raise ValueError(f"memory prune blocked by secret scan: {', '.join(hits)}")
        root.mkdir(parents=True, exist_ok=True)
        _index_path(root).write_text(output, encoding="utf-8")
    return MemoryPrunePlan(issues=issues, actions=actions, dry_run=False, applied=write_needed)


def _memory_root(memory_dir: str | Path) -> Path:
    return Path(memory_dir).resolve()


def _index_path(root: Path) -> Path:
    return root / INDEX_FILE_NAME


def _topic_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (
            path
            for path in root.glob("*.md")
            if path.is_file()
            and path.name.casefold() != INDEX_FILE_NAME.casefold()
            and _path_stays_within(path, root)
        ),
        key=lambda path: path.name.casefold(),
    )


def _resolve_topic_path(root: Path, file_name: str) -> Path:
    if not _is_safe_topic_name(file_name):
        raise ValueError(f"memory file must be a top-level .md topic file: {file_name!r}")
    path = (root / file_name).resolve()
    if not _path_stays_within(path, root):
        raise ValueError(f"memory file escapes memory_dir: {file_name!r}")
    return path


def _path_stays_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _is_safe_topic_name(file_name: str) -> bool:
    if not isinstance(file_name, str) or not file_name:
        return False
    if "/" in file_name or "\\" in file_name:
        return False
    candidate = Path(file_name)
    return (
        not candidate.is_absolute()
        and candidate.name == file_name
        and file_name.casefold() != INDEX_FILE_NAME.casefold()
        and file_name.endswith(".md")
    )


def _read_topic_parts(path: Path) -> _TopicParts:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return _TopicParts({}, "", False, (f"cannot read topic file: {exc}",))
    return _parse_topic_text(text)


def _parse_topic_text(text: str) -> _TopicParts:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return _TopicParts({}, text, False, ("missing opening frontmatter delimiter",))

    end_index: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_index = idx
            break

    if end_index is None:
        return _TopicParts({}, "", False, ("missing closing frontmatter delimiter",))

    frontmatter: dict[str, str] = {}
    errors: list[str] = []
    for line in lines[1:end_index]:
        stripped = line.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            errors.append(f"invalid frontmatter line: {stripped}")
            continue
        key, _, value = stripped.partition(":")
        frontmatter[key.strip()] = value.strip()

    for required in ("name", "description", "type"):
        if not frontmatter.get(required):
            errors.append(f"missing frontmatter field: {required}")
    if frontmatter.get("type") and frontmatter["type"] not in MEMORY_TYPES:
        errors.append(f"invalid frontmatter field: type={frontmatter['type']}")

    body = "".join(lines[end_index + 1 :])
    if body.startswith("\r\n"):
        body = body[2:]
    elif body.startswith("\n"):
        body = body[1:]
    if body.endswith("\r\n"):
        body = body[:-2]
    elif body.endswith("\n"):
        body = body[:-1]
    return _TopicParts(frontmatter, body, not errors, tuple(errors))


def _write_topic(path: Path, name: str, description: str, memory_type: str, body: str) -> None:
    content = (
        f"---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"type: {memory_type}\n"
        f"---\n\n"
        f"{body}\n"
    )
    path.write_text(content, encoding="utf-8")


def _read_index_lines(root: Path) -> list[str]:
    path = _index_path(root)
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def _read_index_entries(root: Path) -> list[_IndexEntry]:
    entries: list[_IndexEntry] = []
    for line_number, line in enumerate(_read_index_lines(root), start=1):
        match = _INDEX_LINK_RE.match(line.rstrip("\n"))
        if not match:
            continue
        tail = match.group("tail").strip()
        if tail[:1] in {"-", "—", "–"}:
            tail = tail[1:].strip()
        entries.append(
            _IndexEntry(
                line_number=line_number,
                raw=line,
                label=match.group(1).strip(),
                target=match.group(2).strip(),
                description=tail,
            )
        )
    return entries


def _upsert_index_line(root: Path, file_name: str, description: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    lines = _read_index_lines(root) or [INDEX_HEADER]
    new_line = _make_index_line(file_name, description)
    entries_by_line = {entry.line_number: entry for entry in _read_index_entries(root)}
    found = False
    output: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        entry = entries_by_line.get(line_number)
        if entry and entry.target == file_name:
            if not found:
                output.append(new_line)
                found = True
            continue
        output.append(line)
    if not found:
        _ensure_final_newline(output)
        output.append(new_line)
    _index_path(root).write_text("".join(output), encoding="utf-8")


def _remove_index_lines(root: Path, file_name: str) -> None:
    path = _index_path(root)
    if not path.exists():
        return
    entries_by_line = {entry.line_number: entry for entry in _read_index_entries(root)}
    output: list[str] = []
    for line_number, line in enumerate(_read_index_lines(root), start=1):
        entry = entries_by_line.get(line_number)
        if entry and entry.target == file_name:
            continue
        output.append(line)
    path.write_text("".join(output), encoding="utf-8")


def _make_index_line(file_name: str, description: str) -> str:
    stem = Path(file_name).stem
    prefix = f"- [{stem}]({file_name}){INDEX_SEPARATOR}"
    desc = _clean_index_description(description)
    return f"{prefix}{desc}\n"


def _clean_frontmatter_value(value: object) -> str:
    return _CONTROL_RE.sub(" ", str(value)).strip()


def _clean_index_description(value: object) -> str:
    return " ".join(_clean_frontmatter_value(value).split())


def _ensure_final_newline(lines: list[str]) -> None:
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"


__all__ = [
    "MemoryRecord",
    "MemoryHealthIssue",
    "MemoryPrunePlan",
    "list_memories",
    "read_memory",
    "update_memory",
    "delete_memory",
    "inspect_memory_health",
    "repair_memory_index",
    "prune_memories",
]
