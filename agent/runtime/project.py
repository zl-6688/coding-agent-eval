"""Stable project identity and project-scoped state paths.

The working directory is preserved for task execution. Sessions and memory use
the resolved Git root when one is available, so every working directory inside
the same repository sees the same existing project state. Outside Git, the
resolved working directory is the identity boundary.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path


_SANITIZE = re.compile(r"[^A-Za-z0-9]")
_KEY_MAX = 200


def ace_home() -> Path:
    """Return the application-state root for the current process."""

    return _ace_root()


def _ace_root() -> Path:
    return Path(os.environ.get("ACE_HOME") or (Path.home() / ".ace"))


def _normalize(path: str | os.PathLike[str]) -> Path:
    return Path(path).resolve()


def _git_root(workpath: Path) -> Path | None:
    """Return the repository root, or ``None`` when Git cannot identify one."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(workpath), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return Path(value) if value else None


def _key_for(workpath: Path) -> str:
    """Return the state-directory key used by existing ACE installations."""

    identity_root = _normalize(_git_root(workpath) or workpath)
    canonical = os.path.normcase(str(identity_root))
    key = _SANITIZE.sub("-", canonical)
    if len(key) > _KEY_MAX:
        suffix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
        key = key[:_KEY_MAX] + "-" + suffix
    return key


class Project:
    """One working directory plus its stable project-scoped state layout."""

    def __init__(self, workpath: Path, key: str, root: Path) -> None:
        self.workpath = workpath
        self.key = key
        self.root = root

    @classmethod
    def from_cwd(cls, workpath: str | os.PathLike[str]) -> "Project":
        normalized_workpath = _normalize(workpath)
        return cls(
            workpath=normalized_workpath,
            key=_key_for(normalized_workpath),
            root=_ace_root(),
        )

    @property
    def _project_dir(self) -> Path:
        return self.root / "projects" / self.key

    @property
    def sessions_dir(self) -> Path:
        return self._project_dir / "sessions"

    @property
    def memory_dir(self) -> Path:
        return self._project_dir / "memory"

    def __repr__(self) -> str:
        return f"Project(key={self.key!r}, workpath={self.workpath})"


__all__ = ["Project", "ace_home"]
