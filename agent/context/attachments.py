"""Request-view attachments for dynamic agent context.

Attachments are user-role meta messages that exist only for the current model
request.  Keeping them out of the durable transcript and the system prompt lets
the loop surface volatile context, such as files changed after a read, without
changing prompt-cache ownership or compaction history.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping

from .. import config
from ..tools.file_state import (
    FileReadSnapshot,
    FileReadState,
    content_matches_for_attachment,
    snapshot_for_path,
)


_POST_COMPACT_GOVERNANCE_NAMES = {
    ".agent-memory",
    ".codex-memory",
    "AGENT.md",
    "AGENTS.md",
    "MEMORY.md",
}


@dataclass(frozen=True)
class Attachment:
    """A small unit of transient context rendered as a user meta message.

    ``metadata`` is deliberately sideband data for tracing or tests.  It is not
    sent to the model because the request API should only see the rendered text.
    """

    kind: str
    title: str
    body: str
    metadata: Mapping[str, Any] | None = None


def render_attachment(attachment: Attachment) -> str:
    """Render an attachment with a system-reminder wrapper but no system role.

    Claude Code uses ``<system-reminder>`` as a textual convention inside user
    messages.  Mirroring that shape keeps the authority boundary clear: the
    content is visible context, not a system prompt section.
    """

    title = attachment.title.strip()
    body = attachment.body.strip("\n")
    header = f"# {title}\n\n" if title else ""
    return f"<system-reminder>\n{header}{body}\n</system-reminder>\n"


def attachment_message(attachment: Attachment) -> dict[str, str]:
    """Return the API-facing user message for one transient attachment."""

    return {"role": "user", "content": render_attachment(attachment)}


def changed_files_attachment(
    read_state: FileReadState,
    executor: Any,
) -> Attachment | None:
    """Warn when a fully read file no longer matches its read snapshot.

    This producer is intentionally based on ``FileReadState`` instead of git
    state.  The model only needs a warning for files it has already seen; repo
    changes outside that read set are normal discovery work, not stale context.
    Partial reads are skipped because the model never saw the complete file.
    """

    changed: list[tuple[str, str]] = []
    for record in read_state.records.values():
        if not record.complete:
            continue
        try:
            current = snapshot_for_path(executor, record.snapshot.path)
        except Exception:
            continue
        if content_matches_for_attachment(record.snapshot, current):
            continue
        changed.append((record.snapshot.path, _snapshot_change_reason(current)))

    if not changed:
        return None

    lines = [
        "The following files were read earlier, but their current disk snapshot no longer matches what the model saw.",
        "Run read_file again before relying on or editing these paths.",
        "",
    ]
    lines.extend(f"- {path}: {reason}" for path, reason in changed)
    return Attachment(
        kind="changed_files",
        title="Files changed after read",
        body="\n".join(lines),
        metadata={"paths": [path for path, _reason in changed]},
    )


def changed_files_message(read_state: FileReadState, executor: Any) -> dict[str, str] | None:
    """Return a request-only changed-files message, or ``None`` when clean."""

    attachment = changed_files_attachment(read_state, executor)
    return attachment_message(attachment) if attachment is not None else None


def request_attachment_messages(
    read_state: FileReadState,
    executor: Any,
) -> tuple[dict[str, str], ...]:
    """Collect dynamic request-only attachments for one LLM call.

    The loop calls this immediately before sampling so volatile context is fresh
    without becoming durable transcript state.  Today this only includes changed
    files; the single entry point leaves room for IDE/MCP/memory producers later.
    """

    messages = [changed_files_message(read_state, executor)]
    return tuple(message for message in messages if message is not None)


def post_compact_file_attachment(
    recent_files: Mapping[str, str] | Iterable[tuple[str, str]],
    cfg: Any,
    exclude_paths: Iterable[str | Path] | None = None,
    executor: Any | None = None,
) -> Attachment | None:
    """Build the post-compact file restore attachment from recent read files.

    Compaction can summarize away file contents that the model was using.  This
    helper restores a small, bounded set of recently read files through the same
    renderer as other attachments, while skipping files already preserved in the
    kept tail and project-profile files that are injected separately.
    """

    excluded = _expanded_exclude_paths(exclude_paths or ())
    selected: list[tuple[str, str]] = []
    for path, cached in _recent_file_items(recent_files):
        if _is_post_compact_excluded(path, excluded):
            continue
        selected.append((path, cached))

    selected = selected[-int(cfg.post_compact_max_files) :][::-1]
    parts: list[str] = []
    used_chars = 0
    per_file_chars = int(cfg.post_compact_max_tokens_per_file) * 4
    budget_chars = int(cfg.post_compact_token_budget) * 4
    attached_paths: list[str] = []
    for path, cached in selected:
        content = _read_latest_text(path, cached, executor=executor)[:per_file_chars]
        if used_chars + len(content) > budget_chars:
            break
        used_chars += len(content)
        attached_paths.append(path)
        parts.append(f"--- {path} ---\n{content}")

    if not parts:
        return None

    body = (
        "[压缩后文件恢复 — 重读了最近访问的文件，避免再次 read_file]\n\n"
        "Post-compact file restore: recently read files are restored here so the "
        "model can continue without an immediate read_file.\n\n"
        + "\n\n".join(parts)
    )
    return Attachment(
        kind="post_compact_files",
        title="Post-compact file restore",
        body=body,
        metadata={
            "paths": attached_paths,
            "attached_count": len(attached_paths),
            "estimated_tokens": used_chars // 4,
        },
    )


def post_compact_file_attachment_text(
    recent_files: Mapping[str, str] | Iterable[tuple[str, str]],
    cfg: Any,
    exclude_paths: Iterable[str | Path] | None = None,
    executor: Any | None = None,
) -> str:
    """Render post-compact file restore text for compact.py compatibility."""

    attachment = post_compact_file_attachment(
        recent_files,
        cfg,
        exclude_paths,
        executor=executor,
    )
    return render_attachment(attachment) if attachment is not None else ""


def _snapshot_change_reason(current: FileReadSnapshot) -> str:
    if not current.exists:
        return "deleted after it was read"
    return "changed after it was read"


def _recent_file_items(
    recent_files: Mapping[str, str] | Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    if isinstance(recent_files, Mapping):
        return [(str(path), str(content)) for path, content in recent_files.items()]
    return [(str(path), str(content)) for path, content in recent_files]


def _expanded_exclude_paths(paths: Iterable[str | Path]) -> set[str]:
    expanded: set[str] = set()
    for path in paths:
        expanded.update(_path_variants(str(path)))
    return expanded


def _is_post_compact_excluded(path: str, excluded: set[str]) -> bool:
    if _path_parts(path) & _POST_COMPACT_GOVERNANCE_NAMES:
        return True
    return bool(_path_variants(path) & excluded)


def _path_name(path: str) -> str:
    name = Path(path).name
    win_name = PureWindowsPath(path).name
    if "\\" in name and win_name:
        return win_name
    return name or win_name


def _path_parts(path: str) -> set[str]:
    parts = set(Path(path).parts)
    parts.update(PureWindowsPath(path).parts)
    name = _path_name(path)
    if name:
        parts.add(name)
    return {part.strip("\\/") for part in parts if part.strip("\\/")}


def _path_variants(path: str) -> set[str]:
    variants = {path}
    try:
        variants.add(str(Path(path).resolve()))
    except OSError:
        pass
    try:
        variants.add(str((config.WORKDIR / path).resolve()))
    except Exception:
        pass
    return variants


def _read_latest_text(path: str, cached: str, *, executor: Any | None = None) -> str:
    if executor is not None:
        try:
            return executor.read_file_raw(path)
        except Exception:
            return cached

    try:
        fp = Path(path)
        if not fp.is_absolute():
            fp = config.WORKDIR / path
        return fp.read_text(encoding="utf-8", errors="replace") if fp.exists() else cached
    except Exception:
        return cached


__all__ = [
    "Attachment",
    "attachment_message",
    "changed_files_attachment",
    "changed_files_message",
    "post_compact_file_attachment",
    "post_compact_file_attachment_text",
    "render_attachment",
    "request_attachment_messages",
]
