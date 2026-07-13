"""Immutable, per-runtime file-root capabilities.

The ordinary workspace remains the default read/write root.  Trusted harness
code may add narrower roots (for example Auto Memory) without mutating global
executor state or accepting paths from repository settings.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


AccessKind = Literal["read", "write", "metadata"]
_GLOB_MAGIC = re.compile(r"[*?[]")


@dataclass(frozen=True)
class PathGrant:
    """One lexical root pinned to its resolved filesystem root."""

    lexical: Path
    resolved: Path
    secret_scan: bool = False
    reject_hardlinked_reads: bool = False
    reject_hardlinked_writes: bool = False


@dataclass(frozen=True)
class AuthorizedPath:
    lexical: Path
    resolved: Path
    grant: PathGrant

    @property
    def requires_secret_scan(self) -> bool:
        return self.grant.secret_scan

    @property
    def rejects_hardlinked_writes(self) -> bool:
        return self.grant.reject_hardlinked_writes

    @property
    def rejects_hardlinked_reads(self) -> bool:
        return self.grant.reject_hardlinked_reads


@dataclass(frozen=True)
class FileAccessPolicy:
    """Operation-specific roots with same-root symlink/junction containment."""

    workdir: PathGrant
    read_grants: tuple[PathGrant, ...]
    write_grants: tuple[PathGrant, ...]

    @classmethod
    def create(
        cls,
        workdir: str | Path,
        *,
        read_roots: Iterable[str | Path] = (),
        write_roots: Iterable[str | Path] = (),
        secret_scan_roots: Iterable[str | Path] = (),
    ) -> "FileAccessPolicy":
        workspace = _make_grant(workdir, trusted_workspace=True)
        additional_reads = tuple(_make_grant(root) for root in read_roots)
        reads = _dedupe_grants([workspace, *additional_reads])
        additional_writes = tuple(_make_grant(root) for root in write_roots)
        writes = _dedupe_grants([workspace, *additional_writes])
        secret_grants = _dedupe_grants(
            [_make_grant(root, secret_scan=True) for root in secret_scan_roots]
        )

        write_keys = {_grant_key(grant) for grant in writes}
        for grant in secret_grants:
            if _grant_key(grant) not in write_keys:
                raise ValueError(
                    f"secret-scan root is not an authorized write root: {grant.lexical}"
                )
        secret_keys = {_grant_key(grant) for grant in secret_grants}
        protected_read_keys = {
            _grant_key(grant) for grant in additional_reads
        }
        protected_write_keys = {
            _grant_key(grant) for grant in additional_writes
        }
        reads = tuple(
            PathGrant(
                lexical=grant.lexical,
                resolved=grant.resolved,
                secret_scan=grant.secret_scan,
                reject_hardlinked_reads=(
                    _grant_key(grant) in protected_read_keys
                ),
                reject_hardlinked_writes=grant.reject_hardlinked_writes,
            )
            for grant in reads
        )
        writes = tuple(
            PathGrant(
                lexical=grant.lexical,
                resolved=grant.resolved,
                secret_scan=_grant_key(grant) in secret_keys,
                reject_hardlinked_reads=(
                    _grant_key(grant) in protected_write_keys
                ),
                reject_hardlinked_writes=(
                    _grant_key(grant) in protected_write_keys
                ),
            )
            for grant in writes
        )
        return cls(workdir=workspace, read_grants=reads, write_grants=writes)

    def authorize(self, path: str | Path, access: AccessKind) -> AuthorizedPath:
        text = os.fspath(path)
        _reject_unsafe_path_text(text)
        raw = Path(text)
        lexical = _lexical_absolute(
            raw if raw.is_absolute() else self.workdir.lexical / raw
        )
        grants = self._grants_for(access)

        # Pick the most-specific lexical grant first.  Once selected, the
        # resolved target must remain under that same grant; it cannot fall
        # through to a second allowed root after crossing a symlink/junction.
        matching = [
            grant for grant in grants if _is_within(lexical, grant.lexical)
        ]
        if not matching:
            raise ValueError(f"path is outside allowed {access} roots: {path}")
        grant = max(matching, key=lambda item: len(os.fspath(item.lexical)))
        resolved = lexical.resolve(strict=False)
        _reject_unsafe_path_text(os.fspath(resolved))
        if not _is_within(resolved, grant.resolved):
            raise ValueError(
                "path escapes its granted root through a symlink or junction: "
                f"{path}"
            )
        # Root selection is most-specific, but protection is monotonic: adding
        # a narrower overlapping grant or a second lexical alias to the same
        # canonical tree must never turn off a parent's secret/hardlink guard.
        # Physical matches inherit restrictions only; they cannot authorize a
        # path that failed the selected grant's containment check above.
        protection_matches = [
            item for item in grants if _is_within(resolved, item.resolved)
        ]
        effective_grant = PathGrant(
            lexical=grant.lexical,
            resolved=grant.resolved,
            secret_scan=any(item.secret_scan for item in protection_matches),
            reject_hardlinked_reads=any(
                item.reject_hardlinked_reads for item in protection_matches
            ),
            reject_hardlinked_writes=any(
                item.reject_hardlinked_writes for item in protection_matches
            ),
        )
        return AuthorizedPath(
            lexical=lexical,
            resolved=resolved,
            grant=effective_grant,
        )

    def protected_grants(self) -> tuple[PathGrant, ...]:
        """Return roots whose inode aliases must not bypass added-root guards."""

        return _dedupe_grants(
            grant
            for grant in (*self.read_grants, *self.write_grants)
            if (
                grant.secret_scan
                or grant.reject_hardlinked_reads
                or grant.reject_hardlinked_writes
            )
        )

    def glob_search_root(self, pattern: str) -> AuthorizedPath:
        """Authorize the non-magic prefix before glob enumerates the filesystem."""

        _reject_unsafe_path_text(pattern)
        path = Path(pattern)
        prefix: list[str] = []
        for part in path.parts:
            if _GLOB_MAGIC.search(part):
                break
            prefix.append(part)
        if prefix:
            base = Path(*prefix)
        else:
            base = Path(".")
        authorized = self.authorize(base, "read")
        if _grant_key(authorized.grant) != _grant_key(self.workdir):
            # Python glob can follow a symlink matched by a wildcard directory
            # before result filtering. Extra roots therefore require explicit
            # directory components and permit magic only in the final segment.
            if any(part == ".." for part in path.parts):
                raise ValueError(
                    "glob traversal segments are not allowed in additional roots"
                )
            if any(_GLOB_MAGIC.search(part) for part in path.parts[:-1]):
                raise ValueError(
                    "wildcard directory traversal is not allowed in additional roots"
                )
        return authorized

    def _grants_for(self, access: AccessKind) -> tuple[PathGrant, ...]:
        if access == "read":
            return self.read_grants
        if access == "write":
            return self.write_grants
        return _dedupe_grants([*self.read_grants, *self.write_grants])


def _make_grant(
    root: str | Path,
    *,
    secret_scan: bool = False,
    trusted_workspace: bool = False,
) -> PathGrant:
    text = os.fspath(root)
    _reject_unsafe_path_text(text)
    raw = Path(text)
    if not trusted_workspace and not raw.is_absolute():
        raise ValueError(f"additional file root must be absolute: {root}")
    lexical = _lexical_absolute(raw)
    resolved = lexical.resolve(strict=False)
    _reject_unsafe_path_text(os.fspath(resolved))
    if not trusted_workspace and (_is_filesystem_root(lexical) or _is_filesystem_root(resolved)):
        raise ValueError(f"filesystem root cannot be granted as an additional root: {root}")
    # Match CC's near-root rejection for custom memory paths (`/a`), while
    # retaining ordinary Windows paths such as C:\x.
    if not trusted_workspace and len(os.fspath(lexical).rstrip("/\\")) < 3:
        raise ValueError(f"additional file root is too broad: {root}")
    return PathGrant(lexical=lexical, resolved=resolved, secret_scan=secret_scan)


def _lexical_absolute(path: Path) -> Path:
    # abspath/normpath collapses `..` without resolving symlinks.  Keeping that
    # distinction is what lets authorize() pin a lexical grant to one resolved
    # root instead of accepting a target merely because another root allows it.
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _is_filesystem_root(path: Path) -> bool:
    anchor = path.anchor
    return bool(anchor) and os.path.normcase(os.fspath(path)) == os.path.normcase(anchor)


def _is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            [os.path.normcase(os.fspath(path)), os.path.normcase(os.fspath(root))]
        ) == os.path.normcase(os.fspath(root))
    except ValueError:
        return False


def _reject_unsafe_path_text(path: str) -> None:
    if "\0" in path:
        raise ValueError("null bytes are not allowed in file paths")
    normalized = path.replace("\\", "/")
    if normalized.startswith("//"):
        raise ValueError("UNC and device paths are not allowed")
    if os.name == "nt":
        raw = Path(path)
        if raw.anchor and not raw.is_absolute():
            raise ValueError(
                "Windows anchored-relative file paths are not allowed"
            )
        _drive, tail = os.path.splitdrive(path)
        if ":" in tail:
            raise ValueError("NTFS alternate data stream paths are not allowed")


def _grant_key(grant: PathGrant) -> tuple[str, str]:
    return (
        os.path.normcase(os.fspath(grant.lexical)),
        os.path.normcase(os.fspath(grant.resolved)),
    )


def _dedupe_grants(grants: Iterable[PathGrant]) -> tuple[PathGrant, ...]:
    result: list[PathGrant] = []
    positions: dict[tuple[str, str], int] = {}
    for grant in grants:
        key = _grant_key(grant)
        existing_position = positions.get(key)
        if existing_position is not None:
            existing = result[existing_position]
            result[existing_position] = PathGrant(
                lexical=existing.lexical,
                resolved=existing.resolved,
                secret_scan=existing.secret_scan or grant.secret_scan,
                reject_hardlinked_reads=(
                    existing.reject_hardlinked_reads
                    or grant.reject_hardlinked_reads
                ),
                reject_hardlinked_writes=(
                    existing.reject_hardlinked_writes
                    or grant.reject_hardlinked_writes
                ),
            )
            continue
        positions[key] = len(result)
        result.append(grant)
    return tuple(result)


__all__ = ["AuthorizedPath", "FileAccessPolicy", "PathGrant"]
