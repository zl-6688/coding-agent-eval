"""Durable transcript message identity and constructors.

Claude Code assigns a UUID when every typed message is created.  ACE keeps its
lighter wire-shaped dictionaries, but gives each durable top-level message the
same stable identity property.  API request builders strip the sideband UUID;
SessionStore persists it across save/resume and deterministically migrates old
transcripts that predate this module.
"""

from __future__ import annotations

import uuid as _uuid
from collections.abc import Iterable, MutableMapping
from typing import Any


MESSAGE_UUID_KEY = "uuid"
LEGACY_MESSAGE_ID_KEY = "__ace_message_id"


def new_message_uuid() -> str:
    """Return an RFC 4122 UUID string for a newly created durable message."""

    return str(_uuid.uuid4())


def new_durable_message(role: str, content: Any, **fields: Any) -> dict[str, Any]:
    """Create one wire-shaped durable message with a stable UUID.

    ACE intentionally keeps the existing ``role``/``content`` representation
    instead of copying Claude Code's full TypeScript message union.  The UUID
    is transcript metadata and is removed by the API-facing request view.
    """

    message = {"role": role, "content": content, **fields}
    message[MESSAGE_UUID_KEY] = new_message_uuid()
    return message


def new_user_message(content: Any, **fields: Any) -> dict[str, Any]:
    return new_durable_message("user", content, **fields)


def new_assistant_message(content: Any, **fields: Any) -> dict[str, Any]:
    return new_durable_message("assistant", content, **fields)


def message_uuid(message: Any) -> str | None:
    """Return the stable message UUID, accepting the legacy key during migration."""

    if not isinstance(message, MutableMapping):
        return None
    value = message.get(MESSAGE_UUID_KEY)
    if value:
        return str(value)
    legacy = message.get(LEGACY_MESSAGE_ID_KEY)
    return str(legacy) if legacy else None


def message_matches_identity(message: Any, identity: Any) -> bool:
    """Match a new UUID or a pre-P0-A anchor during transcript migration."""

    if not isinstance(message, MutableMapping) or identity is None:
        return False
    target = str(identity)
    return any(
        str(value) == target
        for value in (
            message.get(MESSAGE_UUID_KEY),
            message.get("id"),
            message.get("messageId"),
            message.get(LEGACY_MESSAGE_ID_KEY),
        )
        if value
    )


def ensure_message_uuids(
    messages: Iterable[Any],
    *,
    migration_namespace: str | None = None,
    drop_legacy: bool = False,
) -> None:
    """Add UUIDs in place without rewriting identities that already exist.

    Old transcript files need repeatable IDs even if they are resumed multiple
    times before the next save.  A SessionStore namespace plus line position
    provides a deterministic UUID5 migration.  Live messages use UUID4.
    ``__ace_message_id`` is removed at the SessionStore/return migration
    boundary; compact can keep it briefly to match a pre-P0-A anchor.
    """

    for index, message in enumerate(messages):
        if not isinstance(message, MutableMapping):
            continue
        existing = message.get(MESSAGE_UUID_KEY)
        if existing:
            if drop_legacy:
                message.pop(LEGACY_MESSAGE_ID_KEY, None)
            continue

        legacy = (
            message.get(LEGACY_MESSAGE_ID_KEY)
            or message.get("id")
            or message.get("messageId")
        )
        if migration_namespace is not None:
            migration_key = legacy or f"position:{index}"
            value = _uuid.uuid5(
                _uuid.NAMESPACE_URL,
                f"ace-session:{migration_namespace}:{migration_key}",
            )
            message[MESSAGE_UUID_KEY] = str(value)
        elif legacy:
            message[MESSAGE_UUID_KEY] = str(
                _uuid.uuid5(_uuid.NAMESPACE_URL, f"ace-legacy:{legacy}")
            )
        else:
            message[MESSAGE_UUID_KEY] = new_message_uuid()
        if drop_legacy:
            message.pop(LEGACY_MESSAGE_ID_KEY, None)


__all__ = [
    "LEGACY_MESSAGE_ID_KEY",
    "MESSAGE_UUID_KEY",
    "ensure_message_uuids",
    "message_matches_identity",
    "message_uuid",
    "new_assistant_message",
    "new_durable_message",
    "new_message_uuid",
    "new_user_message",
]
