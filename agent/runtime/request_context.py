"""Request-only context helpers for the agent loop.

The loop keeps durable transcript messages separate from transient request
context such as AGENTS.md, skill listings, and deferred-tool indexes.  These
helpers make that boundary explicit without changing transcript state.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..context.compact import (
    is_compact_boundary_message,
    is_compact_summary_message,
)
from ..tools.messages import is_durable_request_message, to_api_message


_MEMORY_INDEX_SNAPSHOT_ATTR = "_ace_memory_index_context_snapshot"


def context_message_list(context_messages: Any) -> list[dict]:
    """Normalize request-only context without creating durable transcript state.

    Setup code may have no context, one context message, or a tuple of independent
    context messages.  The model request needs a flat list, while compaction and
    return_messages must continue to see only the durable transcript.
    """
    if context_messages is None:
        return []
    if isinstance(context_messages, dict):
        return [context_messages]
    return [message for message in context_messages if message is not None]


def request_context_messages(*context_messages: dict | None) -> tuple[dict, ...]:
    """Freeze the per-run request-only context at setup time.

    The tuple filters optional messages while documenting that AGENTS.md, skill
    listing, and deferred index messages are request-only context rather than
    conversation history.  The final request builder decides where this context
    is placed relative to durable messages and volatile attachments.
    """
    return tuple(message for message in context_messages if message is not None)


def memory_index_context_message(
    auto_memory: Any,
    *,
    enabled: bool,
    recall_mode: str,
) -> dict[str, str] | None:
    """Return the session-cached Auto Memory index in index mode.

    CC memoizes both ``getUserContext`` and ``getMemoryFiles`` for the
    conversation.  The same snapshot is therefore reused across model turns
    and user queries until a compact/clear boundary invalidates it.
    """

    if not enabled or recall_mode != "index" or auto_memory is None:
        return None
    if hasattr(auto_memory, _MEMORY_INDEX_SNAPSHOT_ATTR):
        return copy.deepcopy(getattr(auto_memory, _MEMORY_INDEX_SNAPSHOT_ATTR))
    memory_dir = getattr(auto_memory, "memory_dir", None)
    if memory_dir is None:
        return None
    path = Path(memory_dir) / "MEMORY.md"
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        result = None
        setattr(auto_memory, _MEMORY_INDEX_SNAPSHOT_ATTR, result)
        return result
    if not content.strip():
        result = None
        setattr(auto_memory, _MEMORY_INDEX_SNAPSHOT_ATTR, result)
        return result
    truncated = auto_memory.truncate_index_for_injection(content)
    result = {
        "role": "user",
        "content": (
            "<system-reminder>\n"
            "The following is the current Auto Memory index. It is dynamic user context, not system policy.\n\n"
            f"Contents of {path} (user's auto-memory, persists across conversations):\n\n"
            f"{truncated.strip()}\n"
            "</system-reminder>"
        ),
    }
    setattr(auto_memory, _MEMORY_INDEX_SNAPSHOT_ATTR, result)
    return copy.deepcopy(result)


def invalidate_memory_index_context(auto_memory: Any) -> None:
    """Invalidate the cached index for the query after a compact/clear boundary."""

    if auto_memory is None:
        return
    try:
        delattr(auto_memory, _MEMORY_INDEX_SNAPSHOT_ATTR)
    except AttributeError:
        return


def compose_llm_messages(messages: list, context_messages: Any) -> list:
    """Return the per-request message view without mutating durable messages.

    Durable internal metadata is converted only for the API view.  Request-only
    context is prepended before the durable transcript, matching Claude Code
    ``prependUserContext()``; volatile attachments are handled by
    ``build_request_view()`` at the request tail.
    """
    durable_view = []
    for message in messages:
        if is_compact_boundary_message(message):
            continue
        if (
            is_durable_request_message(message)
            or is_compact_summary_message(message)
            or ("role" in message and "content" in message)
        ):
            durable_view.append(to_api_message(message))
        else:
            durable_view.append(message)
    context_view = []
    for message in context_message_list(context_messages):
        if is_compact_boundary_message(message):
            continue
        if (
            is_durable_request_message(message)
            or is_compact_summary_message(message)
            or ("role" in message and "content" in message)
        ):
            context_view.append(to_api_message(message))
        else:
            context_view.append(message)
    if not context_view:
        return durable_view
    return [*context_view, *durable_view]


def budget_system(system: str, context_messages: Any) -> str:
    """Fold static request context into the estimate-only budget string.

    ``compact.estimate`` accepts a system string and durable messages, not a
    separate request context.  Rendering context here preserves budget pressure
    from AGENTS.md and deferred-tool indexes without sending them as system
    prompt content.
    """
    context = context_message_list(context_messages)
    if not context:
        return system
    rendered = "\n\n".join(str(message.get("content", "")) for message in context)
    return f"{system}\n\n{rendered}"
