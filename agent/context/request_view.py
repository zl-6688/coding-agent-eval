"""Build the final per-request message view without mutating transcript state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from .compact import is_compact_boundary_message, is_compact_summary_message
from ..runtime.request_context import context_message_list
from ..tools.messages import is_durable_request_message, to_api_message


@dataclass(frozen=True)
class RequestView:
    """API-facing messages plus counts for the three request-only layers.

    The order aligns with Claude Code ``prependUserContext()``: stable query-scoped
    context first, durable transcript second, volatile attachments last.
    Attachments stay at the tail so changed-files and post-compact restore do
    not shift the durable prefix away from provider prompt caches.
    """

    messages: tuple[dict[str, Any], ...]
    durable_count: int
    context_count: int
    attachment_count: int

    def as_messages(self) -> list[dict[str, Any]]:
        """Return a fresh list suitable for an LLM client call."""

        return list(self.messages)

    def estimate_tokens(self, system: str = "") -> int:
        """Estimate the full request size, including tail attachments."""

        total = len(system or "")
        for message in self.messages:
            total += len(_content_as_text(message))
        return total // 4


def build_request_view(
    durable_messages: Iterable[dict[str, Any]],
    query_context_messages: Any = None,
    request_attachment_messages: Any = None,
) -> RequestView:
    """Create the final LLM request view without changing durable messages.

    Claude Code prepends user context (``claudeMd``) before the durable transcript
    and keeps volatile attachments at the request tail.  This project mirrors
    that split: query-scoped context, then durable transcript, then attachments.
    """

    durable_view = tuple(_api_messages(durable_messages))
    context_view = tuple(_api_messages(context_message_list(query_context_messages)))
    attachment_view = tuple(_api_messages(context_message_list(request_attachment_messages)))
    return RequestView(
        messages=(*context_view, *durable_view, *attachment_view),
        durable_count=len(durable_view),
        context_count=len(context_view),
        attachment_count=len(attachment_view),
    )


def _api_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    # Local import avoids memory.__init__ -> forked_agent -> request_view cycles
    # while the existing memory package still exposes eager compatibility APIs.
    from ..memory.relevant import render_relevant_memories_message

    visible = []
    for message in messages:
        rendered_memories = render_relevant_memories_message(message)
        if rendered_memories is not None:
            visible.extend(rendered_memories)
            continue
        api_message = _api_message(message)
        if api_message is not None:
            visible.append(api_message)
    return visible


def _api_message(message: dict[str, Any]) -> dict[str, Any] | None:
    if is_compact_boundary_message(message):
        return None
    if (
        is_durable_request_message(message)
        or is_compact_summary_message(message)
        or ("role" in message and "content" in message)
    ):
        return to_api_message(message)
    return message


def _content_as_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


__all__ = ["RequestView", "build_request_view"]
