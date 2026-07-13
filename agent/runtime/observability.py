"""Thin runtime observability helpers.

The runtime should emit useful spans without making tracing a second control
plane.  These helpers keep annotation best-effort and centralize the small set
of attribute shapes that are safe to record.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections.abc import Mapping
from typing import Any

from obs.trace import SpanKind, annotate, span

_CONTENT_MODES = {"safe", "redacted", "raw"}
_DEFAULT_PREVIEW_CHARS = 500
_MAX_PREVIEW_CHARS = 50_000
_SECRET_PATTERNS = (
    (
        re.compile(
            r"(?i)(\bauthorization\b[\"']?\s*[:=]\s*[\"']?\s*bearer\s+)([^\"'\s,;&}]+)"
        ),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r"(?i)(\b(?:x[-_]?api[-_]?key|api[_-]?key|token|password|secret)\b[\"']?\s*[:=]\s*)([\"']?)([^\"'\s,;&}]+)([\"']?)"
        ),
        r"\1\2[REDACTED]\4",
    ),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[REDACTED]"),
)


@contextlib.contextmanager
def runtime_span(name: str, kind: str = SpanKind.INTERNAL, **attrs: Any):
    """Open a runtime span while keeping business exceptions visible to callers."""

    with span(name, kind, **safe_attrs(attrs)) as active_span:
        yield active_span


def safe_set_current_span(**attrs: Any) -> None:
    """Annotate the current span best-effort so observability cannot break runtime work."""

    try:
        annotate(**safe_attrs(attrs))
    except Exception:
        return


def record_permission_decision(decision: Any, prefix: str = "permission") -> None:
    """Record only permission decision metadata because raw inputs may contain secrets."""

    if decision is None:
        return
    safe_set_current_span(
        **{
            f"{prefix}.behavior": str(getattr(decision, "behavior", "") or ""),
            f"{prefix}.source": str(getattr(decision, "source", "") or ""),
            f"{prefix}.reason": str(getattr(decision, "reason", "") or ""),
            f"{prefix}.updated_input_present": bool(
                getattr(decision, "updated_input", None) is not None
            ),
        }
    )


def safe_text_length(value: Any) -> int:
    """Return only text length so callers can measure payload size without storing it."""

    if value is None:
        return 0
    if isinstance(value, bytes):
        return len(value.decode("utf-8", errors="replace"))
    try:
        return len(str(value))
    except Exception:
        return 0


def content_summary_attrs(prefix: str, value: Any) -> dict[str, Any]:
    """Record shape-only content summaries so safe traces stay useful without raw text."""

    summary = _content_summary(value)
    return {f"{prefix}_summary": summary}


def content_preview_attrs(prefix: str, value: Any) -> dict[str, Any]:
    """Return explicit raw/redacted previews only when developers opt into content tracing.

    Default traces must remain safe for shared Phoenix projects.  The env gate keeps raw
    content out of span attributes unless a developer deliberately asks for a bounded
    debug preview.
    """

    mode = _content_preview_mode()
    if mode == "safe":
        return {}
    text = _content_to_preview_text(value)
    if mode == "redacted":
        text = _redact_preview_text(text)
    limit = _content_preview_chars()
    truncated = len(text) > limit
    preview = text[:limit]
    return {
        f"{prefix}.preview": preview,
        f"{prefix}.preview_mode": mode,
        f"{prefix}.preview_truncated": truncated,
        f"{prefix}.preview_chars": len(text),
    }


def safe_attrs(attrs: Mapping[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    """Normalize span attributes to JSON-ish primitives without expanding objects."""

    merged: dict[str, Any] = {}
    if attrs:
        merged.update(dict(attrs))
    merged.update(extra)
    return {str(key): _safe_attr_value(value) for key, value in merged.items()}


def _safe_attr_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "chars": safe_text_length(value)}
    if isinstance(value, Mapping):
        return {
            "type": type(value).__name__,
            "field_count": len(value),
            "fields": sorted(str(key) for key in value.keys()),
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if all(item is None or isinstance(item, (bool, int, float, str)) for item in items):
            return items
        return {"type": type(value).__name__, "count": len(items)}
    return {"type": type(value).__name__}


def _content_preview_mode() -> str:
    mode = str(os.environ.get("ACE_TRACE_CONTENT") or "safe").strip().lower()
    return mode if mode in _CONTENT_MODES else "safe"


def _content_preview_chars() -> int:
    raw = str(os.environ.get("ACE_TRACE_PREVIEW_CHARS") or "").strip()
    try:
        value = int(raw) if raw else _DEFAULT_PREVIEW_CHARS
    except ValueError:
        value = _DEFAULT_PREVIEW_CHARS
    return max(0, min(value, _MAX_PREVIEW_CHARS))


def _content_to_preview_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _content_summary(value: Any) -> str:
    chars = safe_text_length(value)
    if value is None:
        return "none chars=0"
    if isinstance(value, Mapping):
        fields = ",".join(sorted(str(key) for key in value.keys())[:8])
        suffix = ",..." if len(value) > 8 else ""
        return f"object fields={fields}{suffix} chars={chars}"
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"{type(value).__name__} count={len(value)} chars={chars}"
    return f"{type(value).__name__} chars={chars}"


def _redact_preview_text(text: str) -> str:
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


__all__ = [
    "content_preview_attrs",
    "content_summary_attrs",
    "record_permission_decision",
    "runtime_span",
    "safe_attrs",
    "safe_set_current_span",
    "safe_text_length",
]
