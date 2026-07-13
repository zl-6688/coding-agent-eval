"""Skill catalog discovery and summary rendering."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


_log = logging.getLogger(__name__)

PROJECT_SKILLS_RELATIVE_DIR = Path(".claude") / "skills"
ACE_SKILLS_SUBDIR = "skills"
# CC 用户级 skill 落在 config home；与 ~/.claude/skills 对齐，作兼容扫描。
CC_USER_SKILLS_RELATIVE_DIR = Path(".claude") / "skills"
SKILL_FILE_NAME = "SKILL.md"


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    display_name: str = ""
    when_to_use: str = ""
    allowed_tools: tuple[str, ...] = ()
    disable_model_invocation: bool = False
    user_invocable: bool | None = None
    argument_hint: str = ""
    arguments: tuple[str, ...] = ()
    body: str = ""
    raw_content: str = ""
    path: Path = Path()
    base_dir: Path = Path()
    source: str = ""

    @property
    def model_invocable(self) -> bool:
        return not self.disable_model_invocation


@dataclass(frozen=True)
class SkillCatalog:
    skills: tuple[SkillDefinition, ...] = ()
    _by_name: Mapping[str, SkillDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_name", {skill.name: skill for skill in self.skills})

    def find(self, name: str) -> SkillDefinition | None:
        return self._by_name.get(normalize_skill_name(name))

    def invocable_skills(self) -> tuple[SkillDefinition, ...]:
        return tuple(skill for skill in self.skills if skill.model_invocable)

    def is_empty(self) -> bool:
        return not self.skills


def resolve_skill_workdir(workdir: str | Path | None) -> Path | None:
    """Resolve project skill discovery root; prefer git root when available."""
    if workdir is None:
        return None
    from ..context.project_instructions import ProjectInstructionsLoader

    return ProjectInstructionsLoader()._project_root(workdir)


def discover_skill_catalog(
    workdir: str | Path | None,
    *,
    user_home: str | Path | None = None,
) -> SkillCatalog:
    """Discover project and user skills for one run boundary.

    Project skills are loaded before user skills so project-local definitions
    win name conflicts without mutating either source.
    """

    if workdir is None:
        return SkillCatalog()

    project_dir = Path(workdir).expanduser() / PROJECT_SKILLS_RELATIVE_DIR
    from ..runtime.project import ace_home

    home = Path(user_home).expanduser() if user_home is not None else Path.home()
    user_dirs = (
        ("user", ace_home() / ACE_SKILLS_SUBDIR),
        ("user", home / CC_USER_SKILLS_RELATIVE_DIR),
    )

    seen: set[str] = set()
    skills: list[SkillDefinition] = []
    for source, skills_dir in (("project", project_dir), *user_dirs):
        for skill in _load_skills_from_dir(skills_dir, source=source):
            if skill.name in seen:
                continue
            seen.add(skill.name)
            skills.append(skill)
    return SkillCatalog(tuple(skills))


def skill_summary_text(skill: SkillDefinition) -> str:
    """Best-effort one-paragraph summary for CLI detail view."""
    desc = _scalar(skill.description).strip()
    if desc and desc not in {"-", "—", "–", "|", "|-"}:
        return desc
    when = _scalar(skill.when_to_use).strip()
    if when:
        return when
    display = _scalar(skill.display_name).strip()
    if display and display != skill.name:
        return display
    for line in skill.body.splitlines():
        cleaned = line.strip()
        if cleaned and not cleaned.startswith("#") and cleaned != "---":
            return cleaned.lstrip("#").strip()
    return ""


def format_cli_skill_listing(catalog: SkillCatalog) -> str:
    """Compact plain-text skill list for non-TTY slash commands."""
    if catalog.is_empty():
        from ..runtime.project import ace_home

        ace_skills = ace_home() / ACE_SKILLS_SUBDIR
        return (
            "未发现 skill。\n"
            f"  项目级：{PROJECT_SKILLS_RELATIVE_DIR}/<skill名>/{SKILL_FILE_NAME}\n"
            f"  用户级：{ace_skills}/<skill名>/{SKILL_FILE_NAME}\n"
            f"         或 ~/{CC_USER_SKILLS_RELATIVE_DIR}/<skill名>/{SKILL_FILE_NAME}（Claude Code 兼容）"
        )

    lines = ["Skills", f"{len(catalog.skills)} skills", ""]
    for skill in catalog.skills:
        status = "on" if skill.model_invocable else "off"
        mark = "✓" if skill.model_invocable else "○"
        tok = format_skill_token_label(estimate_skill_tokens(skill))
        lines.append(f"  {mark} {status} · {skill.name} · {skill.source} · {tok}")
    return "\n".join(lines)


def estimate_skill_tokens(skill: SkillDefinition) -> int:
    """Rough token estimate for full SKILL.md (body loaded on invoke)."""
    text = skill.raw_content or skill.body
    return max(1, len(text) // 4)


def format_skill_token_label(tokens: int) -> str:
    """Human label aligned with Claude Code skill picker (~N tok / < 20 tok)."""
    if tokens < 20:
        return "< 20 tok"
    return f"~{tokens} tok"


def render_skill_listing(catalog: SkillCatalog) -> str:
    """Render model-visible skill summaries, excluding full SKILL.md bodies."""

    visible = catalog.invocable_skills()
    if not visible:
        return ""

    lines = [
        "# skill_listing",
        "Available skills are listed below. These are summaries only; call the Skill tool with a skill name to load the complete instructions when relevant.",
    ]
    for skill in visible:
        lines.append("")
        lines.append(f"- {skill.name}: {skill.description}")
        if skill.display_name and skill.display_name != skill.name:
            lines.append(f"  display_name: {skill.display_name}")
        if skill.when_to_use:
            lines.append(f"  when_to_use: {skill.when_to_use}")
        if skill.argument_hint:
            lines.append(f"  argument_hint: {skill.argument_hint}")
        lines.append(f"  source: {skill.source}")
    return "\n".join(lines).rstrip()


def skill_listing_context_message(catalog: SkillCatalog) -> dict[str, str] | None:
    listing = render_skill_listing(catalog)
    if not listing:
        return None
    return {
        "role": "user",
        "content": (
            "<system-reminder>\n"
            f"{listing}\n"
            "IMPORTANT: Do not assume a skill's full procedure from this summary. Use the Skill tool first when the skill is relevant.\n"
            "</system-reminder>\n"
        ),
    }


def normalize_skill_name(name: str) -> str:
    return str(name or "").strip().lstrip("/")


def _load_skills_from_dir(skills_dir: Path, *, source: str) -> tuple[SkillDefinition, ...]:
    try:
        children = sorted(skills_dir.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return ()

    skills: list[SkillDefinition] = []
    for child in children:
        if not child.is_dir():
            continue
        skill_path = child / SKILL_FILE_NAME
        if not skill_path.is_file():
            continue
        skill = _load_skill_file(skill_path, source=source)
        if skill is not None:
            skills.append(skill)
    return tuple(skills)


def _load_skill_file(path: Path, *, source: str) -> SkillDefinition | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        _log.warning("skipping unreadable skill file %s: %s", path, exc)
        return None

    frontmatter, body = _split_frontmatter(raw)
    name = normalize_skill_name(path.parent.name)
    description = _scalar(frontmatter.get("description")) or _derive_description(body)
    return SkillDefinition(
        name=name,
        description=description,
        display_name=_scalar(frontmatter.get("name")),
        when_to_use=_scalar(frontmatter.get("when_to_use")),
        allowed_tools=_list_field(frontmatter.get("allowed-tools")),
        disable_model_invocation=_bool_field(frontmatter.get("disable-model-invocation")),
        user_invocable=_optional_bool(frontmatter.get("user-invocable")),
        argument_hint=_scalar(frontmatter.get("argument-hint")),
        arguments=_list_field(frontmatter.get("arguments")),
        body=body,
        raw_content=raw,
        path=path,
        base_dir=path.parent,
        source=source,
    )


def _split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, raw

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            frontmatter_text = "".join(lines[1:index])
            body = "".join(lines[index + 1 :])
            return _parse_frontmatter(frontmatter_text), body
    return {}, raw


def _parse_frontmatter(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    result: dict[str, Any] = {}
    current_list_key: str | None = None
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            i += 1
            continue

        if raw_line[:1].isspace() and current_list_key:
            item = raw_line.strip()
            if item.startswith("- "):
                current = result.setdefault(current_list_key, [])
                if isinstance(current, list):
                    current.append(_parse_scalar(item[2:].strip()))
            i += 1
            continue

        key, sep, value = raw_line.partition(":")
        if not sep:
            current_list_key = None
            i += 1
            continue
        key = key.strip()
        value = value.strip()
        if not key:
            current_list_key = None
            i += 1
            continue

        if value in {"|", "|-", ">", ">-"}:
            block, i = _read_block_scalar(lines, i + 1)
            result[key] = "\n".join(block).strip()
            current_list_key = None
            continue

        if value == "":
            if i + 1 < len(lines) and lines[i + 1][:1] in (" ", "\t"):
                next_line = lines[i + 1].strip()
                if next_line.startswith("- "):
                    result[key] = []
                    current_list_key = key
                    i += 1
                    continue
            current_list_key = None
            i += 1
            continue

        result[key] = _parse_scalar(value)
        current_list_key = None
        i += 1
    return result


def _read_block_scalar(lines: list[str], start: int) -> tuple[list[str], int]:
    collected: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            if collected:
                collected.append("")
            i += 1
            continue
        if line[:1] not in (" ", "\t"):
            break
        collected.append(line.strip())
        i += 1
    return collected, i


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    return value


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(_scalar(item) for item in value if _scalar(item))
    return str(value).strip()


def _list_field(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(item for item in (_scalar(part) for part in value) if item)
    text = _scalar(value)
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _bool_field(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _scalar(value).lower() in {"true", "yes", "on", "1"}


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return _bool_field(value)


def _derive_description(body: str) -> str:
    for line in body.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned.lstrip("#").strip()
    return ""


__all__ = [
    "ACE_SKILLS_SUBDIR",
    "CC_USER_SKILLS_RELATIVE_DIR",
    "PROJECT_SKILLS_RELATIVE_DIR",
    "SKILL_FILE_NAME",
    "SkillCatalog",
    "SkillDefinition",
    "discover_skill_catalog",
    "estimate_skill_tokens",
    "format_cli_skill_listing",
    "format_skill_token_label",
    "normalize_skill_name",
    "render_skill_listing",
    "resolve_skill_workdir",
    "skill_listing_context_message",
    "skill_summary_text",
]
