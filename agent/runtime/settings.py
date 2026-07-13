"""ACE settings loader — CC-shaped permissions JSON from ~/.ace and project .ace/.

Cold start: files are optional (ENOENT → empty dict). REPL builds a PermissionEngine
from merged rules; eval/direct run_task callers omit injection and keep empty engine.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .permissions import PermissionEngine, PermissionRule
from .project import ace_home

_ACE_DIR = ".ace"
_USER_SETTINGS = "settings.json"
_PROJECT_SETTINGS = "settings.json"
_LOCAL_SETTINGS = "settings.local.json"


@dataclass(frozen=True)
class MemoryRuntimeSettings:
    """Trusted runtime interpretation of the user/project memory settings."""

    enabled: bool = True
    recall_mode: str = "index"


def memory_runtime_settings_from_settings(
    settings: dict[str, Any] | None,
) -> MemoryRuntimeSettings:
    """Normalize the small P0-D memory config surface.

    Unknown modes fall back to index, matching CC's false feature-gate fallback. This
    function never accepts a memory path override; the trusted Project object
    remains the only owner of that path.
    """

    raw = (settings or {}).get("memory")
    memory = raw if isinstance(raw, dict) else {}
    enabled_value = memory.get("enabled", True)
    enabled = enabled_value if isinstance(enabled_value, bool) else True
    mode_value = str(memory.get("recall_mode", "index")).strip().lower()
    recall_mode = mode_value if mode_value in {"selector", "index"} else "index"
    return MemoryRuntimeSettings(enabled=enabled, recall_mode=recall_mode)


class SettingsUpdateError(RuntimeError):
    """Refuse to overwrite settings when existing file cannot be parsed safely."""


# CC legacy PascalCase tool names → ACE snake_case builtins.
_LEGACY_TOOL_ALIASES: dict[str, str] = {
    "Bash": "bash",
    "PowerShell": "powershell",
    "Read": "read_file",
    "Write": "write_file",
    "Edit": "edit_file",
    "Glob": "glob",
    "Grep": "grep",
}

_SHELL_TOOLS = frozenset({"bash", "powershell"})
_PATH_TOOLS = frozenset({"read_file", "write_file", "edit_file"})
_PATTERN_TOOLS = frozenset({"grep", "glob"})

_SOURCE_BY_LAYER = {
    "user": "user",
    "project": "project",
    "local": "session",
}


def settings_paths(workpath: str | Path) -> dict[str, Path]:
    """Resolved settings file paths (files may be missing)."""
    root = Path(workpath).resolve()
    return {
        "user": ace_home() / _USER_SETTINGS,
        "project": root / _ACE_DIR / _PROJECT_SETTINGS,
        "local": root / _ACE_DIR / _LOCAL_SETTINGS,
    }


def load_settings_file(path: Path) -> dict[str, Any]:
    """Parse one settings JSON file; missing/empty → {}."""
    try:
        # utf-8-sig: Windows 记事本 / 部分编辑器会写 BOM，裸 utf-8 解析会失败。
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    text = text.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _load_user_settings_for_update(path: Path) -> dict[str, Any]:
    """Load user settings for merge-write; fail closed if file exists but is invalid."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise SettingsUpdateError(f"无法读取 {path}：{exc}") from exc
    text = text.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SettingsUpdateError(
            f"无法解析 {path}（含注释/尾逗号会导致此错误），已拒绝写入以免覆盖原有配置：{exc}"
        ) from exc
    if not isinstance(data, dict):
        raise SettingsUpdateError(f"{path} 根节点必须是 JSON 对象，已拒绝写入。")
    return data


def merge_settings(
    user: dict[str, Any] | None = None,
    project: dict[str, Any] | None = None,
    local: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shallow merge user → project → local (later keys override earlier)."""
    merged: dict[str, Any] = {}
    for layer in (user or {}, project or {}, local or {}):
        for key, value in layer.items():
            if key == "permissions" and isinstance(value, dict):
                merged_permissions = dict(merged.get("permissions") or {})
                for perm_key, perm_value in value.items():
                    if perm_key in {"allow", "deny", "ask"} and isinstance(perm_value, list):
                        existing = list(merged_permissions.get(perm_key) or [])
                        merged_permissions[perm_key] = existing + list(perm_value)
                    else:
                        merged_permissions[perm_key] = perm_value
                merged["permissions"] = merged_permissions
            else:
                merged[key] = value
    return merged


def load_merged_settings(workpath: str | Path) -> dict[str, Any]:
    """Load and merge ACE settings for a workspace."""
    paths = settings_paths(workpath)
    return merge_settings(
        load_settings_file(paths["user"]),
        load_settings_file(paths["project"]),
        load_settings_file(paths["local"]),
    )


def approval_mode_from_settings(settings: dict[str, Any] | None) -> str:
    """Map permissions.defaultMode to REPL ApprovalState mode (ask/auto)."""
    permissions = (settings or {}).get("permissions")
    if not isinstance(permissions, dict):
        return "ask"
    default_mode = permissions.get("defaultMode")
    if default_mode == "bypassPermissions":
        return "auto"
    return "ask"


def _normalize_tool_name(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return raw
    return _LEGACY_TOOL_ALIASES.get(raw, raw)


def _parse_rule_string(rule_string: str) -> tuple[str, str | None]:
    """Parse CC rule text: ToolName or ToolName(content)."""
    text = str(rule_string or "").strip()
    if not text:
        raise ValueError("empty permission rule")
    open_index = _first_unescaped_char(text, "(")
    if open_index == -1:
        return _normalize_tool_name(text), None
    if not text.endswith(")"):
        raise ValueError(f"unbalanced parentheses in permission rule: {text}")
    tool_name = _normalize_tool_name(text[:open_index])
    rule_content = _unescape_rule_content(text[open_index + 1 : -1])
    return tool_name, rule_content


def _first_unescaped_char(text: str, char: str) -> int:
    index = 0
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == char:
            return index
        index += 1
    return -1


def _unescape_rule_content(content: str) -> str:
    return (
        content.replace("\\(", "(")
        .replace("\\)", ")")
        .replace("\\\\", "\\")
    )


def _match_rule_content(tool_name: str, tool_input: dict[str, Any], rule_content: str) -> bool:
    if tool_name in _SHELL_TOOLS:
        command = str(tool_input.get("command") or "")
        return _shell_rule_matches(command, rule_content)
    if tool_name in _PATH_TOOLS:
        path = str(tool_input.get("path") or "")
        return fnmatch.fnmatchcase(path, rule_content)
    if tool_name in _PATTERN_TOOLS:
        pattern = str(tool_input.get("pattern") or "")
        return fnmatch.fnmatchcase(pattern, rule_content)
    return False


def _shell_rule_matches(command: str, rule_content: str) -> bool:
    """Prefix/wildcard match for shell commands (v1 subset of CC shellRuleMatching)."""
    command = str(command or "").strip()
    rule = str(rule_content or "").strip()
    if not rule:
        return False
    if "*" in rule or "?" in rule:
        return fnmatch.fnmatchcase(command, rule)
    return command.startswith(rule)


def _make_matcher(tool_name: str, rule_content: str | None):
    if rule_content is None:
        return None

    def _matcher(tool_input: dict[str, Any]) -> bool:
        return _match_rule_content(tool_name, tool_input, rule_content)

    return _matcher


def parse_permission_rules(
    permissions: dict[str, Any] | None,
    *,
    layer: str,
) -> list[PermissionRule]:
    """Convert merged permissions.allow/deny into PermissionRule objects.

    ask rules are skipped in v1 — they would need PermissionEngine ask → REPL bridge.
    """
    if not isinstance(permissions, dict):
        return []
    source = _SOURCE_BY_LAYER.get(layer, "user")
    rules: list[PermissionRule] = []
    for behavior in ("deny", "allow"):
        entries = permissions.get(behavior)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, str):
                continue
            try:
                tool_name, rule_content = _parse_rule_string(entry)
            except ValueError:
                continue
            if not tool_name:
                continue
            rules.append(
                PermissionRule(
                    tool_name,
                    behavior,  # type: ignore[arg-type]
                    source=source,  # type: ignore[arg-type]
                    matcher=_make_matcher(tool_name, rule_content),
                    message=f"{layer} {behavior}: {entry}",
                )
            )
    return rules


def build_permission_rules(workpath: str | Path) -> list[PermissionRule]:
    """Load layered settings and flatten into PermissionRule list."""
    paths = settings_paths(workpath)
    layers = (
        ("user", load_settings_file(paths["user"])),
        ("project", load_settings_file(paths["project"])),
        ("local", load_settings_file(paths["local"])),
    )
    rules: list[PermissionRule] = []
    for layer_name, settings in layers:
        permissions = settings.get("permissions")
        if not isinstance(permissions, dict):
            continue
        rules.extend(parse_permission_rules(permissions, layer=layer_name))
    return rules


def build_permission_engine(workpath: str | Path) -> PermissionEngine:
    """Build PermissionEngine for interactive REPL (reads ACE settings only)."""
    return PermissionEngine(build_permission_rules(workpath))


def user_settings_parse_error() -> str | None:
    """Return parse error for ~/.ace/settings.json, or None if missing/valid."""
    path = ace_home() / _USER_SETTINGS
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"无法读取 {path}：{exc}"
    text = text.strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return (
            f"无法解析 {path}：{exc}。"
            "请手动修复 JSON（注释/缺逗号会导致 availableModels 等配置不生效）。"
        )
    if not isinstance(data, dict):
        return f"{path} 根节点必须是 JSON 对象。"
    return None


def _apply_settings_patch(base: dict[str, Any], partial: dict[str, Any]) -> dict[str, Any]:
    """Apply a partial update without clobbering unrelated keys.

    Top-level: only keys present in ``partial`` are touched (``None`` deletes that key).
    Nested ``env`` / ``permissions`` / ``models``: deep-merge sub-keys so e.g. patching
    ``permissions.defaultMode`` does not wipe ``allow``/``deny`` or top-level ``model``.
    """
    result = dict(base)
    nested_keys = frozenset({"env", "permissions", "models"})

    for key, value in partial.items():
        if value is None:
            result.pop(key, None)
            continue
        if key in nested_keys and isinstance(value, dict):
            existing = result.get(key)
            nested = dict(existing) if isinstance(existing, dict) else {}
            for sub_key, sub_value in value.items():
                if sub_value is None:
                    nested.pop(sub_key, None)
                else:
                    nested[sub_key] = sub_value
            result[key] = nested
        else:
            result[key] = value
    return result


def update_user_settings(partial: dict[str, Any]) -> dict[str, Any]:
    """Patch ~/.ace/settings.json — single write seam for REPL-driven settings changes.

    Callers pass only the keys they intend to change (e.g. ``{"model": "..."}`` or
    ``{"permissions": {"defaultMode": "bypassPermissions"}}``). Unmentioned keys
    are preserved. Parse failures refuse the write (fail closed).
    """
    path = ace_home() / _USER_SETTINGS
    current = _load_user_settings_for_update(path)
    merged = _apply_settings_patch(current, partial)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return merged


def available_models_from_settings(
    settings: dict[str, Any] | None,
    *,
    current_model: str | None = None,
) -> list[str]:
    """Candidate list for /model picker: availableModels → current → config default."""
    from .. import config
    from .llm_runtime import display_model_from_settings

    seen: set[str] = set()
    result: list[str] = []

    def _add(model: str | None) -> None:
        text = str(model or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)

    raw = (settings or {}).get("availableModels")
    if isinstance(raw, list):
        for entry in raw:
            _add(str(entry).strip() if entry is not None else None)

    _add(current_model or display_model_from_settings(settings))
    _add(config.MODEL_ID)
    return result
