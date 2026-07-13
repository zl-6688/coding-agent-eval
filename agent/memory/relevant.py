"""Typed relevant-memory attachments and their API-facing renderer.

Claude Code keeps recalled memories as durable ``relevant_memories`` attachment
messages and renders them only while building the provider request.  ACE keeps
the same semantic boundary without importing CC's full TypeScript message union.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..runtime.messages import new_message_uuid


ATTACHMENT_MESSAGE_TYPE = "attachment"
RELEVANT_MEMORIES_TYPE = "relevant_memories"
# Mirrors CC RELEVANT_MEMORIES_CONFIG.MAX_SESSION_BYTES.  Despite the upstream
# name, attachments.ts accumulates JavaScript String.length (UTF-16 code units).
MAX_SESSION_SURFACED_UNITS = 60 * 1024


@dataclass(frozen=True)
class SurfacedMemories:
    """Trusted surfaced state reconstructed from the current transcript."""

    paths: frozenset[str]
    total_bytes: int


def memory_header(path: str, mtime: float) -> str:
    """Render and freeze the freshness header used for one recalled memory."""

    age_days = max(0, int((time.time() - mtime) / 86400))
    if age_days > 1:
        staleness = (
            f"此记忆已 {age_days} 天。"
            "记忆是时间点快照，而非实时状态——"
            "关于代码行为的断言或 file:line 引用可能已过时。断言前请核对当前代码。"
        )
        return f"{staleness}\n\n记忆: {path}:"
    if age_days == 1:
        return f"记忆（保存于昨天）: {path}:"
    return f"记忆（保存于今天）: {path}:"


def create_relevant_memories_message(
    memories: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create one durable typed attachment from selector results.

    ``header`` is computed once instead of during every request render.  This
    mirrors CC's prompt-cache safeguard: a relative-age label must not silently
    change the bytes of an otherwise unchanged transcript on a later turn.
    """

    stored_memories = [_normalize_memory(memory) for memory in memories]
    if not stored_memories:
        raise ValueError("relevant memory attachment requires at least one memory")
    return {
        "type": ATTACHMENT_MESSAGE_TYPE,
        "uuid": new_message_uuid(),
        "attachment": {
            "type": RELEVANT_MEMORIES_TYPE,
            "memories": stored_memories,
        },
    }


def is_relevant_memories_message(message: Any) -> bool:
    if not isinstance(message, Mapping):
        return False
    attachment = message.get("attachment")
    return (
        message.get("type") == ATTACHMENT_MESSAGE_TYPE
        and isinstance(attachment, Mapping)
        and attachment.get("type") == RELEVANT_MEMORIES_TYPE
        and isinstance(attachment.get("memories"), list)
    )


def collect_surfaced_memories(messages: Iterable[Any]) -> SurfacedMemories:
    """Rebuild selector de-dup and session budget from typed attachments only.

    This intentionally does not parse rendered reminder text.  Full compaction
    therefore resets surfaced state by removing old attachments, while an SM
    kept-tail or resumed transcript preserves it without a separate run cache.
    ``total_bytes`` retains the CC-facing name but mirrors upstream
    ``content.length`` using UTF-16 code units rather than UTF-8 bytes.
    """

    paths: set[str] = set()
    total_units = 0
    for message in messages:
        if not is_relevant_memories_message(message):
            continue
        for memory in message["attachment"]["memories"]:
            if not isinstance(memory, Mapping):
                continue
            path = memory.get("path")
            content = memory.get("content")
            if not isinstance(path, str) or not isinstance(content, str):
                continue
            paths.add(path)
            total_units += _javascript_string_length(content)
    return SurfacedMemories(frozenset(paths), total_units)


def render_relevant_memories_message(
    message: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    """Expand a typed attachment into provider-safe user reminder messages.

    ``None`` means the input is not this attachment type.  Returning one user
    message per memory follows CC's normalization path and ensures no durable
    sideband field (UUID, path metadata, attachment discriminator) reaches the
    provider API.
    """

    if not is_relevant_memories_message(message):
        return None
    rendered = []
    for memory in message["attachment"]["memories"]:
        if not isinstance(memory, Mapping):
            continue
        path = str(memory.get("path", ""))
        content = str(memory.get("content", ""))
        header = memory.get("header")
        if not header:
            mtime_ms = _number(memory.get("mtime_ms"), default=0.0)
            header = (
                memory_header(path, mtime_ms / 1000.0)
                if mtime_ms > 0
                else f"记忆: {path}:"
            )
        rendered.append({
            "role": "user",
            "content": f"<system-reminder>\n{header}\n\n{content}\n</system-reminder>",
        })
    return rendered


def _normalize_memory(memory: Mapping[str, Any]) -> dict[str, Any]:
    path = str(memory["path"])
    content = str(memory["content"])
    if memory.get("mtime_ms") is not None:
        mtime_ms = int(_number(memory.get("mtime_ms"), default=0.0))
        mtime_seconds = mtime_ms / 1000.0
    else:
        mtime_seconds = _number(memory.get("mtime"), default=0.0)
        mtime_ms = int(mtime_seconds * 1000)
    header = memory.get("header")
    if not header:
        header = (
            memory_header(path, mtime_seconds)
            if mtime_seconds > 0
            else f"记忆: {path}:"
        )
    return {
        "path": path,
        "content": content,
        "mtime_ms": mtime_ms,
        "header": str(header),
        "limit": memory.get("limit"),
    }


def _number(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _javascript_string_length(value: str) -> int:
    """Return JavaScript String.length for exact CC budget parity."""

    return len(value.encode("utf-16-le")) // 2


__all__ = [
    "ATTACHMENT_MESSAGE_TYPE",
    "MAX_SESSION_SURFACED_UNITS",
    "RELEVANT_MEMORIES_TYPE",
    "SurfacedMemories",
    "collect_surfaced_memories",
    "create_relevant_memories_message",
    "is_relevant_memories_message",
    "memory_header",
    "render_relevant_memories_message",
]
