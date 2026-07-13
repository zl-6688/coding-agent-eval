"""Skill state and compatibility exports for request messages."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

from ..tools.messages import (
    ADDITIONAL_MESSAGE_SOURCE_KEY,
    ADDITIONAL_MESSAGE_VIEW_KEY,
    DURABLE_REQUEST_VIEW,
    durable_request_messages,
    is_durable_request_message,
    mark_durable_request_message,
    to_api_message,
)


SKILL_ADDITIONAL_MESSAGE_SOURCE = "skill"
POST_COMPACT_MAX_TOKENS_PER_SKILL = 5_000
POST_COMPACT_SKILLS_TOKEN_BUDGET = 25_000


@dataclass(frozen=True)
class InvokedSkill:
    name: str
    path: str
    content: str
    invoked_at: float
    agent_id: str = ""


_invoked_skills: dict[tuple[str, str], InvokedSkill] = {}


def _scope(agent_id: str | None) -> str:
    return str(agent_id or "")


def _key(agent_id: str | None, name: str) -> tuple[str, str]:
    return (_scope(agent_id), str(name or "").strip())


def _estimate_chars(tokens: int) -> int:
    return max(0, int(tokens)) * 4


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    max_chars = _estimate_chars(max_tokens)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    newline = cut.rfind("\n")
    if newline > max_chars // 2:
        cut = cut[:newline]
    return cut.rstrip() + "\n\n[... skill content truncated after compact budget ...]"


def record_invoked_skill(
    name: str,
    path: Any,
    content: str,
    *,
    agent_id: str | None = "",
    invoked_at: float | None = None,
) -> InvokedSkill:
    """Record the rendered body that actually reached the model request."""

    skill = InvokedSkill(
        name=str(name or "").strip(),
        path=str(path or ""),
        content=str(content or ""),
        invoked_at=time.time() if invoked_at is None else float(invoked_at),
        agent_id=_scope(agent_id),
    )
    if skill.name:
        _invoked_skills[_key(agent_id, skill.name)] = skill
    return skill


def get_invoked_skills(agent_id: str | None = "") -> tuple[InvokedSkill, ...]:
    scope = _scope(agent_id)
    skills = [skill for skill in _invoked_skills.values() if skill.agent_id == scope]
    return tuple(sorted(skills, key=lambda skill: skill.invoked_at, reverse=True))


def reset_invoked_skills(agent_id: str | None = None) -> None:
    if agent_id is None:
        _invoked_skills.clear()
        return
    scope = _scope(agent_id)
    for key in [key for key, skill in _invoked_skills.items() if skill.agent_id == scope]:
        _invoked_skills.pop(key, None)


def invoked_skill_context_message(
    agent_id: str | None = "",
    *,
    max_tokens_per_skill: int = POST_COMPACT_MAX_TOKENS_PER_SKILL,
    token_budget: int = POST_COMPACT_SKILLS_TOKEN_BUDGET,
) -> dict[str, str] | None:
    skills = get_invoked_skills(agent_id)
    if not skills:
        return None

    parts: list[str] = []
    used = 0
    budget_chars = _estimate_chars(token_budget)
    for skill in skills:
        body = _truncate_to_token_budget(skill.content, max_tokens_per_skill)
        path = skill.path or "<unknown>"
        part = f"### Skill: {skill.name}\nPath: {path}\n\n{body.rstrip()}"
        if used + len(part) > budget_chars:
            break
        used += len(part)
        parts.append(part)

    if not parts:
        return None
    content = (
        "<system-reminder>\n"
        "The following skills were invoked earlier in this run. Their complete "
        "instructions are restored after compaction so you can continue following them.\n\n"
        + "\n\n".join(parts)
        + "\n</system-reminder>\n"
    )
    return {"role": "user", "content": content}


def restore_invoked_skills_from_messages(
    messages: Iterable[dict[str, Any]],
    *,
    agent_id: str | None = "",
) -> tuple[InvokedSkill, ...]:
    restored: list[InvokedSkill] = []
    restored_at = time.time()
    for message in messages:
        text = _message_text(message)
        records = _extract_invoked_skill_records(text)
        if _has_compact_restore_records(text):
            invoked_times = [
                restored_at + len(records) - index for index in range(len(records))
            ]
        else:
            invoked_times = [
                restored_at + index + 1 for index in range(len(records))
            ]
        restored_at += len(records)
        for (name, path, body), invoked_at in zip(records, invoked_times):
            restored.append(
                record_invoked_skill(
                    name, path, body, agent_id=agent_id, invoked_at=invoked_at
                )
            )
    return tuple(restored)


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block.get("content"), str):
                parts.append(str(block.get("content") or ""))
        else:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(str(text))
    return "\n".join(part for part in parts if part)


def _extract_invoked_skill_records(text: str) -> tuple[tuple[str, str, str], ...]:
    records: list[tuple[str, str, str]] = []
    records.extend(_extract_skill_body_messages(text))
    records.extend(_extract_compact_restore_messages(text))
    return tuple(records)


def _has_compact_restore_records(text: str) -> bool:
    return bool(_extract_compact_restore_messages(text))


_SKILL_BODY_RE = re.compile(
    r"<system-reminder>\s*# skill:\s*(?P<name>[^\n]+)\n\n(?P<body>.*?)\n?</system-reminder>",
    re.DOTALL,
)
_COMPACT_SKILL_RE = re.compile(
    r"### Skill:\s*(?P<name>[^\n]+)\nPath:\s*(?P<path>[^\n]*)\n\n(?P<body>.*?)(?=\n### Skill:|\n</system-reminder>|$)",
    re.DOTALL,
)


def _extract_skill_body_messages(text: str) -> tuple[tuple[str, str, str], ...]:
    records: list[tuple[str, str, str]] = []
    for match in _SKILL_BODY_RE.finditer(text or ""):
        name = match.group("name").strip()
        body = match.group("body").strip()
        if name and body:
            records.append((name, "", body))
    return tuple(records)


def _extract_compact_restore_messages(text: str) -> tuple[tuple[str, str, str], ...]:
    if "The following skills were invoked" not in (text or ""):
        return ()
    records: list[tuple[str, str, str]] = []
    for match in _COMPACT_SKILL_RE.finditer(text):
        name = match.group("name").strip()
        path = match.group("path").strip()
        body = match.group("body").strip()
        if name and body:
            records.append((name, "" if path == "<unknown>" else path, body))
    return tuple(records)


__all__ = [
    "ADDITIONAL_MESSAGE_SOURCE_KEY",
    "ADDITIONAL_MESSAGE_VIEW_KEY",
    "DURABLE_REQUEST_VIEW",
    "InvokedSkill",
    "POST_COMPACT_MAX_TOKENS_PER_SKILL",
    "POST_COMPACT_SKILLS_TOKEN_BUDGET",
    "SKILL_ADDITIONAL_MESSAGE_SOURCE",
    "durable_request_messages",
    "get_invoked_skills",
    "invoked_skill_context_message",
    "is_durable_request_message",
    "mark_durable_request_message",
    "record_invoked_skill",
    "reset_invoked_skills",
    "restore_invoked_skills_from_messages",
    "to_api_message",
]
