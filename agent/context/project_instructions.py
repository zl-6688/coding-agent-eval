"""AGENTS.md project profile loader and initializer.

AGENTS.md is a harness-owned project profile for the target repository. Legacy
AGENT.md is read only as a fallback when AGENTS.md is absent. CLAUDE.md and other
external-agent files may be useful human context, but this agent runtime should
not silently merge them into its own project contract.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_INSTRUCTIONS_FILENAME = "AGENTS.md"
LEGACY_PROJECT_INSTRUCTIONS_FILENAME = "AGENT.md"
AGENT_FILENAME = PROJECT_INSTRUCTIONS_FILENAME
PROJECT_INSTRUCTIONS_CONTEXT_KEY = "project_instructions"
MAX_INSTRUCTION_LINES = 200
MAX_INSTRUCTION_BYTES = 25_000

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".traces",
    ".venv",
    "__pycache__",
    "node_modules",
    "workspace",
}

_KNOWN_DIR_HINTS = {
    "agent": "agent runtime, loop, tools, memory, CLI, and project/session state",
    "docs": "design notes, decisions, module docs, eval reports, and reference analysis",
    "eval": "evaluation harnesses and benchmark-specific runners",
    "obs": "observability, trace schema, sinks, and viewers",
    "scripts": "developer and demo scripts",
    "tests": "offline and live regression tests",
}

_KNOWN_FILE_HINTS = {
    "README.md": "human entrypoint and current project overview",
    "pyproject.toml": "Python project and pytest configuration",
    "requirements.txt": "Python dependency list",
    "AGENTS.md": "agent-owned project profile",
    "AGENT.md": "legacy project profile fallback; read-only when AGENTS.md is absent",
    "CLAUDE.md": "external collaboration guide; not loaded by this harness by default",
}


@dataclass(frozen=True)
class AgentProjectProfile:
    """Loaded project instructions content plus a stable cache key."""

    path: Path
    root: Path
    content: str
    fingerprint: str
    truncated: bool = False

    @property
    def relpath(self) -> str:
        try:
            return self.path.relative_to(self.root).as_posix()
        except ValueError:
            return self.path.as_posix()

    def render_user_context(self) -> str:
        """Render project instructions as query-scoped user context."""
        lines = [
            f"# {PROJECT_INSTRUCTIONS_CONTEXT_KEY}",
            "Project instructions from the agent harness profile. AGENTS.md is preferred; legacy AGENT.md is used only when AGENTS.md is absent.",
            f"\n### {self.relpath}",
            self.content.rstrip("\n"),
        ]
        if self.truncated:
            lines.append("[TRUNCATED: project instructions were shortened for prompt injection]")
        return "\n".join(lines).rstrip("\n")

    def render(self) -> str:
        """Backward-compatible alias for query-scoped rendering."""
        return self.render_user_context()

    def to_context_message(self) -> dict:
        """Return the transient meta user message attached to each LLM request.

        This message is part of the request view, not the durable transcript.
        It mirrors Claude Code's user-context shape as a meta user message
        wrapped in <system-reminder>, while the request builder decides the
        cache-friendly position relative to durable transcript messages.
        """
        return {
            "role": "user",
            "content": (
                "<system-reminder>\n"
                "回答用户问题时，可以使用以下项目上下文：\n"
                f"{self.render_user_context()}\n\n"
                "IMPORTANT: 这些上下文可能与当前任务相关，也可能无关。"
                "只有在与当前任务高度相关时才使用它，不要为了复述上下文而回应。\n"
                "</system-reminder>\n"
            ),
        }


class ProjectInstructionsLoader:
    """Load or create the harness-owned AGENTS.md for a project."""

    def __init__(
        self,
        *,
        root: Path | str | None = None,
        filename: str = PROJECT_INSTRUCTIONS_FILENAME,
        max_lines: int = MAX_INSTRUCTION_LINES,
        max_bytes: int = MAX_INSTRUCTION_BYTES,
    ) -> None:
        self.root = Path(root).resolve() if root is not None else None
        self.filename = filename
        self.max_lines = max_lines
        self.max_bytes = max_bytes

    def load(self, workdir: Path | str) -> AgentProjectProfile | None:
        """Return root project instructions, preferring AGENTS.md over legacy AGENT.md."""
        root = self._project_root(workdir)
        path = next((candidate for candidate in self._candidate_paths(root) if candidate.is_file()), None)
        if path is None:
            return None

        try:
            raw = path.read_text(encoding="utf-8")
            stat = path.stat()
        except OSError:
            return None

        content, truncated = _truncate_for_injection(
            raw,
            max_lines=self.max_lines,
            max_bytes=self.max_bytes,
        )
        fingerprint = hashlib.md5(
            f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
        ).hexdigest()[:16]
        return AgentProjectProfile(
            path=path,
            root=root,
            content=content,
            fingerprint=fingerprint,
            truncated=truncated,
        )

    def ensure(
        self,
        workdir: Path | str,
        *,
        overwrite: bool = False,
        project_name: str | None = None,
    ) -> AgentProjectProfile:
        """Create AGENTS.md when missing, then load it.

        WHY creation lives here: AGENTS.md is part of the harness' project bootstrap,
        like a compact project map for future sessions. The default template is
        deterministic and reviewable; richer LLM-maintained updates can come later.
        """
        root = self._project_root(workdir)
        path = root / self.filename
        if overwrite or not path.exists():
            content = render_agent_md_template(root, project_name=project_name)
            path.write_text(content, encoding="utf-8")

        profile = self.load(workdir)
        if profile is None:
            raise FileNotFoundError(f"failed to create or load {path}")
        return profile

    def _project_root(self, workdir: Path | str) -> Path:
        cwd = Path(workdir).resolve()
        return (self.root or _find_git_root(cwd) or cwd).resolve()

    def _candidate_paths(self, root: Path) -> list[Path]:
        paths = [root / self.filename]
        if self.filename == PROJECT_INSTRUCTIONS_FILENAME:
            paths.append(root / LEGACY_PROJECT_INSTRUCTIONS_FILENAME)
        return paths


def ensure_agent_md(workdir: Path | str, *, overwrite: bool = False) -> AgentProjectProfile:
    """Convenience wrapper for project bootstrap code. Creates AGENTS.md."""
    return ProjectInstructionsLoader().ensure(workdir, overwrite=overwrite)


def render_agent_md_template(root: Path | str, *, project_name: str | None = None) -> str:
    """Render a deterministic starter AGENTS.md from the current project layout."""
    root = Path(root).resolve()
    name = project_name or root.name
    inventory = _project_inventory(root)
    return "\n".join(
        [
            f"# AGENTS.md - {name}",
            "",
            "> Agent-owned project profile. Keep this file concise; it is injected as query-scoped user context.",
            "",
            "## Project Purpose",
            "",
            "- TODO: summarize what this project builds and how success is measured.",
            "",
            "## Project Map",
            "",
            *inventory,
            "",
            "## Commands",
            "",
            "- TODO: add the smallest reliable test command.",
            "- TODO: add any required environment variables or setup steps.",
            "",
            "## Working Rules",
            "",
            "- Prefer current code over stale notes.",
            "- Keep changes scoped and reviewable.",
            "- Do not commit secrets, generated traces, or local tool output.",
            "- Update this file when project structure or reliable commands change.",
            "",
            "## External Notes",
            "",
            "- CLAUDE.md may exist for other tools, but this harness does not load it by default.",
            "- Legacy AGENT.md is read only as a fallback when AGENTS.md is absent; bootstrap creates AGENTS.md.",
        ]
    ).rstrip("\n") + "\n"


def _find_git_root(workdir: Path) -> Path | None:
    """Best-effort git root detection; callers have a deterministic cwd fallback."""
    try:
        out = subprocess.run(
            ["git", "-C", str(workdir), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    top = out.stdout.strip()
    return Path(top) if top else None


def _truncate_for_injection(text: str, *, max_lines: int, max_bytes: int) -> tuple[str, bool]:
    """Return a prompt-sized copy of text without mutating the file on disk."""
    truncated = False

    lines = text.splitlines(keepends=True)
    if max_lines >= 0 and len(lines) > max_lines:
        text = "".join(lines[:max_lines])
        truncated = True

    data = text.encode("utf-8")
    if max_bytes >= 0 and len(data) > max_bytes:
        text = data[:max_bytes].decode("utf-8", errors="ignore")
        truncated = True

    return text.rstrip("\n"), truncated


def _project_inventory(root: Path, *, max_entries: int = 24) -> list[str]:
    entries: list[str] = []
    try:
        children = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError:
        return ["- TODO: inspect project structure."]

    for child in children:
        if len(entries) >= max_entries:
            break
        if child.name in _SKIP_DIRS or child.name.startswith(".") and child.name not in {".env.example"}:
            continue
        if child.is_dir():
            hint = _KNOWN_DIR_HINTS.get(child.name, "TODO: describe this directory")
            entries.append(f"- `{child.name}/` - {hint}")
        elif child.is_file() and child.name in _KNOWN_FILE_HINTS:
            entries.append(f"- `{child.name}` - {_KNOWN_FILE_HINTS[child.name]}")

    return entries or ["- TODO: inspect project structure."]
