"""Shared, versioned evidence envelopes for the public MCP evaluations.

The writer keeps run metadata separate from case payloads, fingerprints both
the declared protocol and its implementation sources, and removes local paths
or credential-shaped text before canonical JSONL is persisted.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "ace.mcp_evidence.v1"
PROTOCOL_FINGERPRINT_ALGORITHM = "sha256"

PASS = "PASS"
FAIL = "FAIL"
INVALID = "INVALID"
INCONCLUSIVE = "INCONCLUSIVE"
ERROR = "ERROR"
SKIPPED = "SKIPPED"

ALLOWED_STATUSES = (PASS, FAIL, INVALID, INCONCLUSIVE, ERROR, SKIPPED)
EXCLUDED_STATUSES = (INVALID, INCONCLUSIVE, ERROR, SKIPPED)

_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_TOKEN = re.compile(
    r"(?i)\b(?:sk-ant-|sk-(?:proj-)?|gh[pousr]_)[A-Za-z0-9_-]{20,}"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:anthropic_api_key|openai_api_key|api[_-]?key|token|password|secret)"
    r"\b\s*[:=]\s*)([^\s,;]+)"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9_<>])(?:[A-Za-z]:[\\/])[^\r\n\t\"']+"
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_source_bytes(path: Path) -> bytes:
    """Hash text sources independently of Git checkout line-ending policy."""

    raw = path.read_bytes()
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def build_protocol(
    *,
    protocol_id: str,
    protocol_version: str,
    descriptor: Mapping[str, Any],
    repo_root: Path,
    source_paths: Sequence[str],
) -> dict[str, Any]:
    """Build a self-describing protocol fingerprint from safe relative inputs."""

    if not _VERSION.fullmatch(protocol_id) or not _VERSION.fullmatch(protocol_version):
        raise ValueError("protocol id/version must be portable identifiers")
    root = Path(repo_root).resolve()
    source_sha256: dict[str, str] = {}
    for raw_path in source_paths:
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"protocol source must be repository-relative: {raw_path}")
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise FileNotFoundError(f"protocol source not found: {raw_path}")
        normalized = path.as_posix()
        source_sha256[normalized] = _sha256_bytes(_canonical_source_bytes(candidate))

    safe_descriptor = sanitize_for_evidence(dict(descriptor), repo_root=root)
    descriptor_sha256 = _sha256_bytes(_canonical_json(safe_descriptor))
    fingerprint_material = {
        "id": protocol_id,
        "version": protocol_version,
        "descriptor": safe_descriptor,
        "source_sha256": source_sha256,
    }
    return {
        **fingerprint_material,
        "algorithm": PROTOCOL_FINGERPRINT_ALGORITHM,
        "descriptor_sha256": descriptor_sha256,
        "sha256": _sha256_bytes(_canonical_json(fingerprint_material)),
    }


def resolve_code_version(repo_root: Path, override: str | None = None) -> str:
    """Return an explicit portable version, using WORKTREE for dirty/unborn trees."""

    if override is not None:
        value = override.strip()
        if not _VERSION.fullmatch(value):
            raise ValueError("code version must be a portable non-empty identifier")
        return value

    root = Path(repo_root).resolve()
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        if status.returncode != 0 or status.stdout.strip():
            return "WORKTREE"
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        value = commit.stdout.strip()
        return value if commit.returncode == 0 and _VERSION.fullmatch(value) else "WORKTREE"
    except (OSError, subprocess.SubprocessError):
        return "WORKTREE"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def environment_metadata() -> dict[str, Any]:
    """Describe the runtime without hostnames, executable paths, or environment values."""

    try:
        mcp_version: str | None = importlib.metadata.version("mcp")
    except importlib.metadata.PackageNotFoundError:
        mcp_version = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.system() or "unknown",
        "os_release": platform.release() or "unknown",
        "architecture": platform.machine() or "unknown",
        "packages": {"mcp": mcp_version},
    }


def required_coverage(
    required_ids: Sequence[str],
    observed_ids: Sequence[str],
) -> dict[str, Any]:
    counts = Counter(observed_ids)
    missing = [case_id for case_id in required_ids if counts[case_id] == 0]
    duplicates = [case_id for case_id in required_ids if counts[case_id] > 1]
    unexpected = [case_id for case_id in counts if case_id not in set(required_ids)]
    return {
        "required_ids": list(required_ids),
        "observed_ids": list(observed_ids),
        "missing_ids": missing,
        "duplicate_ids": duplicates,
        "unexpected_ids": unexpected,
        "full_coverage": not missing and not duplicates,
    }


def status_metrics(statuses: Iterable[str]) -> dict[str, Any]:
    counts = Counter(statuses)
    unknown = sorted(set(counts) - set(ALLOWED_STATUSES))
    if unknown:
        raise ValueError(f"unknown evidence statuses: {unknown}")
    pass_count = counts[PASS]
    fail_count = counts[FAIL]
    eligible_count = pass_count + fail_count
    return {
        "pass_count": pass_count,
        "fail_count": fail_count,
        "eligible_count": eligible_count,
        "pass_rate": (pass_count / eligible_count) if eligible_count else None,
        "excluded_counts": {
            status: counts[status]
            for status in EXCLUDED_STATUSES
        },
    }


def required_gate_status(
    *,
    full_coverage: bool,
    statuses: Sequence[str],
    gate_pass: bool,
) -> str:
    """Map a required mechanism gate to one uniform report status."""

    counts = Counter(statuses)
    if counts[ERROR]:
        return ERROR
    if not full_coverage or counts[SKIPPED] or counts[INVALID] or counts[INCONCLUSIVE]:
        return INCONCLUSIVE
    if counts[FAIL]:
        return FAIL
    return PASS if gate_pass and bool(statuses) else INCONCLUSIVE


def _replace_root(text: str, root: Path, marker: str) -> str:
    candidates = {
        str(root.resolve()),
        str(root.resolve()).replace("\\", "/"),
        str(root.resolve()).replace("/", "\\"),
    }
    for candidate in sorted(candidates, key=len, reverse=True):
        text = text.replace(candidate, marker)
    return text


def sanitize_text(text: str, *, repo_root: Path | None = None) -> str:
    sanitized = str(text)
    roots: list[tuple[Path, str]] = []
    if repo_root is not None:
        roots.append((Path(repo_root), "<REPO_ROOT>"))
    roots.extend(
        [
            (Path(tempfile.gettempdir()), "<TEMP>"),
            (Path.home(), "<HOME>"),
        ]
    )
    for root, marker in roots:
        sanitized = _replace_root(sanitized, root, marker)
    sanitized = _TOKEN.sub("<REDACTED_TOKEN>", sanitized)
    sanitized = _SECRET_ASSIGNMENT.sub(r"\1<REDACTED>", sanitized)
    sanitized = _WINDOWS_ABSOLUTE_PATH.sub("<ABSOLUTE_PATH>", sanitized)
    return sanitized


def sanitize_for_evidence(value: Any, *, repo_root: Path | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_for_evidence(item, repo_root=repo_root)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize_for_evidence(item, repo_root=repo_root) for item in value]
    if isinstance(value, Path):
        return sanitize_text(str(value), repo_root=repo_root)
    if isinstance(value, str):
        return sanitize_text(value, repo_root=repo_root)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_text(str(value), repo_root=repo_root)


def write_evidence_jsonl(
    *,
    output: Path,
    protocol: Mapping[str, Any],
    repo_root: Path,
    code_version: str,
    interpretation: Mapping[str, str],
    summary_payload: Mapping[str, Any],
    case_payloads: Sequence[Mapping[str, Any]],
    execution_mode: str,
    api_calls: int | None,
    llm_calls: int | None,
    timestamp_utc: str | None = None,
    environment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Atomically write one summary followed by case records using one envelope."""

    timestamp = timestamp_utc or utc_timestamp()
    safe_code_version = resolve_code_version(repo_root, code_version)
    run_id = _sha256_bytes(
        _canonical_json(
            {
                "protocol_sha256": protocol["sha256"],
                "timestamp_utc": timestamp,
                "code_version": safe_code_version,
            }
        )
    )[:24]
    run = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "code_version": safe_code_version,
        "execution_mode": execution_mode,
        "api_calls": api_calls,
        "llm_calls": llm_calls,
        "environment": dict(environment or environment_metadata()),
    }
    shared = {
        "schema_version": SCHEMA_VERSION,
        "protocol": dict(protocol),
        "run": run,
        "interpretation": dict(interpretation),
    }
    records = [
        {
            **shared,
            "record_type": "run_summary",
            "payload": dict(summary_payload),
        }
    ]
    records.extend(
        {
            **shared,
            "record_type": "case_result",
            "payload": dict(payload),
        }
        for payload in case_payloads
    )
    safe_records = [
        sanitize_for_evidence(record, repo_root=Path(repo_root).resolve())
        for record in records
    ]

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for record in safe_records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return safe_records


__all__ = [
    "ALLOWED_STATUSES",
    "ERROR",
    "FAIL",
    "INCONCLUSIVE",
    "INVALID",
    "PASS",
    "SCHEMA_VERSION",
    "SKIPPED",
    "build_protocol",
    "environment_metadata",
    "required_coverage",
    "required_gate_status",
    "resolve_code_version",
    "sanitize_for_evidence",
    "sanitize_text",
    "status_metrics",
    "utc_timestamp",
    "write_evidence_jsonl",
]
