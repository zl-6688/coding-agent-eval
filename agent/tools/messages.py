"""Tool runtime message view markers."""

from __future__ import annotations

import copy
from typing import Any, Mapping


ADDITIONAL_MESSAGE_VIEW_KEY = "coding_agent_view"
DURABLE_REQUEST_VIEW = "durable_request"
ADDITIONAL_MESSAGE_SOURCE_KEY = "source"


def mark_durable_request_message(message: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    marked = copy.deepcopy(dict(message))
    metadata = dict(marked.get("metadata") or {})
    metadata[ADDITIONAL_MESSAGE_VIEW_KEY] = DURABLE_REQUEST_VIEW
    metadata[ADDITIONAL_MESSAGE_SOURCE_KEY] = source
    marked["metadata"] = metadata
    return marked


def is_durable_request_message(message: Any) -> bool:
    if not isinstance(message, Mapping):
        return False
    if message.get("role") != "user" or "content" not in message:
        return False
    metadata = message.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    return metadata.get(ADDITIONAL_MESSAGE_VIEW_KEY) == DURABLE_REQUEST_VIEW


def to_api_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Strip sideband marker metadata before the message reaches the LLM API."""

    return {
        "role": str(message["role"]),
        "content": copy.deepcopy(message["content"]),
    }


def durable_request_messages(messages: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    return tuple(to_api_message(message) for message in messages if is_durable_request_message(message))


__all__ = [
    "ADDITIONAL_MESSAGE_SOURCE_KEY",
    "ADDITIONAL_MESSAGE_VIEW_KEY",
    "DURABLE_REQUEST_VIEW",
    "durable_request_messages",
    "is_durable_request_message",
    "mark_durable_request_message",
    "to_api_message",
]
