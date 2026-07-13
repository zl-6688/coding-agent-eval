"""REPL model picker — CC-shaped /model TTY selector."""

from __future__ import annotations

from typing import Callable


def pick_model(
    candidates: list[str],
    *,
    current: str | None = None,
    pick_fn: Callable[[str, list[str], Callable], str | None] | None = None,
) -> str | None:
    """TTY inline picker; returns chosen model id or None (cancel)."""
    if not candidates:
        return None

    def render_row(model_id: str, selected: bool) -> str:
        marker = " (当前)" if current and model_id == current else ""
        return f"{model_id}{marker}"

    header = "选择模型（↑↓ 选 · Enter 确认 · Esc 取消）"
    if pick_fn is not None:
        return pick_fn(header, candidates, render_row)

    from .repl import select_inline

    return select_inline(header, candidates, render_row)


def make_model_pick_cb(workpath, model_state, out):
    """Build TTY picker callback for /model (no arg)."""
    from ..runtime.settings import (
        available_models_from_settings,
        load_merged_settings,
        user_settings_parse_error,
    )

    warned = False

    def _pick() -> str | None:
        nonlocal warned
        parse_err = user_settings_parse_error()
        if parse_err and not warned:
            out(f"警告：{parse_err}")
            warned = True
        merged = load_merged_settings(workpath)
        candidates = available_models_from_settings(
            merged,
            current_model=model_state.display,
        )
        if not candidates:
            out("（没有可选模型；在 ~/.ace/settings.json 配置 availableModels）")
            return None
        chosen = pick_model(candidates, current=model_state.display)
        return chosen

    return _pick
