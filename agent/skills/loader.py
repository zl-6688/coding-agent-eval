"""Call-time skill body rendering."""

from __future__ import annotations

from typing import Any

from .catalog import SkillDefinition
from .state import SKILL_ADDITIONAL_MESSAGE_SOURCE, mark_durable_request_message


def render_skill_body(
    skill: SkillDefinition,
    *,
    args: str = "",
    run_id: str = "",
) -> str:
    body = skill.body.rstrip("\n")
    base_dir = str(skill.base_dir)
    if base_dir:
        body = f"Base directory for this skill: {base_dir}\n\n{body}"
    args_text = str(args or "")
    body = body.replace("$ARGUMENTS", args_text)
    skill_dir = base_dir.replace("\\", "/")
    body = body.replace("${CLAUDE_SKILL_DIR}", skill_dir)
    body = body.replace("${CLAUDE_SESSION_ID}", str(run_id or ""))
    return body


def skill_body_context_message(
    skill: SkillDefinition,
    *,
    args: str = "",
    run_id: str = "",
    rendered_body: str | None = None,
) -> dict[str, Any]:
    body = (
        rendered_body
        if rendered_body is not None
        else render_skill_body(skill, args=args, run_id=run_id)
    )
    content = (
        "<system-reminder>\n"
        f"# skill: {skill.name}\n\n"
        f"{body}\n"
        "</system-reminder>\n"
    )
    return mark_durable_request_message(
        {"role": "user", "content": content},
        source=SKILL_ADDITIONAL_MESSAGE_SOURCE,
    )


__all__ = ["render_skill_body", "skill_body_context_message"]
