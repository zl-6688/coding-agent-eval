"""Skill tool definition."""

from __future__ import annotations

from typing import Any

from ..runtime.observability import (
    content_preview_attrs,
    content_summary_attrs,
    runtime_span,
    safe_set_current_span,
    safe_text_length,
)
from ..tools.contracts import Tool, ToolContext, ToolResult
from .catalog import SkillCatalog, normalize_skill_name
from .loader import render_skill_body, skill_body_context_message
from .state import record_invoked_skill


SKILL_TOOL_NAME = "Skill"


def create_skill_tool(catalog: SkillCatalog) -> Tool:
    return Tool(
        name=SKILL_TOOL_NAME,
        description=(
            "Load an available skill's complete instructions into the next model request. "
            "Use this after selecting a relevant skill from the skill listing."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "Skill name from the available skill listing.",
                },
                "args": {
                    "type": "string",
                    "description": "Optional argument string passed to the skill body.",
                },
            },
            "required": ["skill"],
        },
        source="skill_catalog",
        is_read_only=True,
        is_destructive=False,
        is_concurrency_safe=True,
        validate_input=lambda tool_input, context: validate_skill_input(tool_input, context, catalog),
        call=lambda tool_input, context: call_skill_tool(tool_input, context, catalog),
        metadata={
            "skill_count": len(catalog.skills),
            "skill_names": tuple(skill.name for skill in catalog.skills),
        },
    )


def validate_skill_input(
    tool_input: dict[str, Any],
    context: ToolContext,
    catalog: SkillCatalog,
) -> str | None:
    skill_name = normalize_skill_name(str(tool_input.get("skill") or ""))
    if not skill_name:
        return "skill must be non-empty"
    skill = catalog.find(skill_name)
    if skill is None:
        return f"unknown skill: {skill_name}"
    if skill.disable_model_invocation:
        return f"skill is disabled for model invocation: {skill.name}"
    return None


def call_skill_tool(
    tool_input: dict[str, Any],
    context: ToolContext,
    catalog: SkillCatalog,
) -> ToolResult:
    skill_name = normalize_skill_name(str(tool_input.get("skill") or ""))
    args = str(tool_input.get("args") or "")
    with runtime_span(
        "skill.invoke",
        **{
            "skill.name": skill_name,
            "skill.args_present": bool(args),
            **content_summary_attrs("skill.args", args),
            **content_preview_attrs("skill.args", args),
            "skill.run_id_present": bool(context.run_id),
            "skill.agent_id_present": bool(context.agent_id),
        },
    ):
        skill = catalog.find(skill_name)
        if skill is None:
            safe_set_current_span(
                **{
                    "skill.status": "unknown",
                    "skill.source": "",
                    "skill.body_chars": 0,
                }
            )
            return ToolResult(
                content=f"UnknownSkillError: unknown skill: {skill_name}",
                is_error=True,
            )
        safe_set_current_span(
            **{
                "skill.name": skill.name,
                "skill.source": skill.source,
                "skill.body_chars": safe_text_length(skill.body),
                "skill.status": "disabled" if skill.disable_model_invocation else "ok",
            }
        )
        if skill.disable_model_invocation:
            return ToolResult(
                content=f"SkillDisabledError: skill is disabled for model invocation: {skill.name}",
                is_error=True,
            )

        rendered_body = render_skill_body(skill, args=args, run_id=context.run_id)
        safe_set_current_span(
            **{
                **content_summary_attrs("skill.body", rendered_body),
                **content_preview_attrs("skill.body", rendered_body),
                "skill.body_chars": safe_text_length(rendered_body),
            }
        )
        record_invoked_skill(
            skill.name,
            skill.path,
            rendered_body,
            agent_id=context.agent_id,
        )
        return ToolResult(
            content=f"Launching skill: {skill.name}",
            metadata={"skill": skill.name, "source": skill.source},
            additional_messages=(
                skill_body_context_message(
                    skill,
                    args=args,
                    run_id=context.run_id,
                    rendered_body=rendered_body,
                ),
            ),
        )


__all__ = [
    "SKILL_TOOL_NAME",
    "call_skill_tool",
    "create_skill_tool",
    "validate_skill_input",
]
