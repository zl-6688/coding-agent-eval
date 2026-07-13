"""REPL skill browser — compact CC-style rows + colored TTY picker."""

from __future__ import annotations

import asyncio

from prompt_toolkit.styles import Style

from ..skills.catalog import (
    SkillCatalog,
    SkillDefinition,
    estimate_skill_tokens,
    format_cli_skill_listing,
    format_skill_token_label,
    skill_summary_text,
)

SKILL_PICKER_STYLE = Style.from_dict({
    "header": "#af87ff bold",
    "dim": "#585858",
    "skill-on": "#5fff87",
    "skill-off": "#8a8a8a",
    "skill-name": "#ffffff",
    "skill-name-active": "#00d7ff bold",
    "skill-meta": "#8a8a8a",
    "reverse": "reverse",
})


def show_skill_listing(catalog: SkillCatalog, out, *, is_tty: bool = False) -> None:
    """Print skill catalog; TTY uses colored inline picker."""
    if catalog.is_empty():
        out(format_cli_skill_listing(catalog))
        return

    skills = list(catalog.skills)
    if is_tty:
        _show_skill_picker(skills, out, is_tty=is_tty)
        return

    from . import style as style_mod

    style_mod.emit_segments(out, _listing_header_segments(len(skills)), is_tty=False)
    for skill in skills:
        style_mod.emit_segments(out, _skill_row_segments(skill), is_tty=False)


def _show_skill_picker(skills: list[SkillDefinition], out, *, is_tty: bool) -> None:
    from . import style as style_mod

    header = (
        f"Skills — {len(skills)} skills · "
        "↑↓ 选择 · Enter 查看 · Esc 关闭"
    )
    chosen = _select_skill_inline(header, skills)
    if chosen is None:
        style_mod.emit_block(out, "dim", "(已取消)", is_tty)
        return

    out(_skill_detail_text(chosen))


def _select_skill_inline(header: str, skills: list[SkillDefinition]):
    """Colored inline picker; returns chosen skill or None."""
    if not skills:
        return None

    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    from .repl import _SelectState

    st = _SelectState(len(skills))
    visible = 10

    def _text():
        start = max(0, min(st.index - visible + 1, len(skills) - visible)) if len(skills) > visible else 0
        start = max(0, start)
        lines: list[tuple[str, str]] = [("class:header", header + "\n")]
        for i in range(start, min(start + visible, len(skills))):
            selected = i == st.index
            prefix = "> " if selected else "  "
            row_style = "class:reverse" if selected else ""
            row_body = _skill_row_fragments(skills[i], selected=selected)
            lines.append((row_style, prefix))
            for frag_style, frag_text in row_body:
                lines.append((frag_style, frag_text))
            lines.append(("", "\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    def _(event):
        st.move(-1)

    @kb.add("down")
    def _(event):
        st.move(1)

    @kb.add("enter")
    def _(event):
        event.app.exit(result=skills[st.index])

    @kb.add("escape", eager=True)
    def _(event):
        event.app.exit(result=None)

    @kb.add("c-c")
    def _(event):
        event.app.exit(result=None)

    app = Application(
        layout=Layout(Window(FormattedTextControl(_text), always_hide_cursor=True)),
        key_bindings=kb,
        full_screen=False,
        mouse_support=False,
        erase_when_done=True,
        style=SKILL_PICKER_STYLE,
    )
    try:
        asyncio.get_running_loop()
        in_thread = True
    except RuntimeError:
        in_thread = False
    return app.run(in_thread=in_thread)


def _skill_row_fragments(skill: SkillDefinition, *, selected: bool) -> list[tuple[str, str]]:
    status = "on" if skill.model_invocable else "off"
    mark = "✓ " if skill.model_invocable else "○ "
    tok = format_skill_token_label(estimate_skill_tokens(skill))
    name_class = "class:skill-name-active" if selected else "class:skill-name"
    status_class = "class:skill-on" if skill.model_invocable else "class:skill-off"
    return [
        (status_class, mark + status),
        ("class:dim", " · "),
        (name_class, skill.name),
        ("class:skill-meta", f" · {skill.source} · {tok}"),
    ]


def _listing_header_segments(count: int) -> list[tuple[str, str]]:
    return [
        ("skill-header", "Skills\n"),
        ("dim", f"{count} skills\n\n"),
    ]


def _skill_row_segments(skill: SkillDefinition, *, highlight: bool = False) -> list[tuple[str, str]]:
    status = "on" if skill.model_invocable else "off"
    mark = "✓ " if skill.model_invocable else "○ "
    tok = format_skill_token_label(estimate_skill_tokens(skill))
    name_role = "skill-name" if highlight else "answer"
    return [
        ("dim", "  "),
        ("skill-on", mark),
        ("skill-on" if skill.model_invocable else "dim", status),
        ("dim", " · "),
        (name_role, skill.name),
        ("dim", f" · {skill.source} · {tok}\n"),
    ]


def _skill_detail_text(skill: SkillDefinition) -> str:
    summary = skill_summary_text(skill)
    preview = _body_preview(skill.body)
    parts: list[str] = []

    if summary:
        parts.append(summary)
    if preview and preview != summary and not preview.startswith(summary + "\n"):
        parts.append(preview)
    return "\n".join(parts) or "（无内容）"


def _body_preview(body: str, *, max_lines: int = 6, max_chars: int = 480) -> str:
    lines: list[str] = []
    total = 0
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line == "---":
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        if not line:
            continue
        if total + len(line) > max_chars or len(lines) >= max_lines:
            lines.append("…")
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)
