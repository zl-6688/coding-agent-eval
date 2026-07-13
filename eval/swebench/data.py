"""Load ID-only SWE-bench suites and hydrate them from external datasets.

The repository owns suite selection metadata, not the benchmark instances.
Callers must supply a complete dataset JSON explicitly; this module validates
that boundary before any Docker, model, or scoring work starts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUITE_SCHEMA = "ace.swebench-suite.v1"
REQUIRED_INSTANCE_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "patch",
    "test_patch",
)
_MANIFEST_FIELDS = frozenset(
    {"schema", "name", "source_dataset", "external_dataset_required", "instance_ids"}
)


class SuiteManifestError(ValueError):
    """The project-owned ID-only suite manifest is invalid."""


class DatasetError(ValueError):
    """The caller-supplied benchmark dataset is missing or invalid."""


@dataclass(frozen=True)
class SuiteManifest:
    name: str
    source: str
    instance_ids: tuple[str, ...]
    path: Path


def _read_json(path: Path, *, label: str, error_type: type[ValueError]) -> Any:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise error_type(
            f"{label} not found: {path}. Supply an existing JSON path explicitly."
        ) from exc
    except OSError as exc:
        raise error_type(f"Could not read {label} {path}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise error_type(
            f"Invalid JSON in {label} {path}: line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicate: list[str] = []
    for value in values:
        if value in seen and value not in duplicate:
            duplicate.append(value)
        seen.add(value)
    return duplicate


def load_suite_manifest(path: str | Path) -> SuiteManifest:
    """Validate and load an ``ace.swebench-suite.v1`` ID-only manifest."""

    resolved = Path(path).expanduser().resolve()
    payload = _read_json(
        resolved,
        label="SWE-bench suite manifest",
        error_type=SuiteManifestError,
    )
    if not isinstance(payload, Mapping):
        raise SuiteManifestError(
            f"Suite manifest {resolved} must be a JSON object using schema "
            f"{SUITE_SCHEMA!r}; full benchmark rows belong in --instances, not the repo."
        )
    unknown = sorted(set(payload) - _MANIFEST_FIELDS)
    if unknown:
        raise SuiteManifestError(
            f"Suite manifest {resolved} contains unsupported fields: {', '.join(unknown)}. "
            "The manifest may contain IDs only; supply problem statements and patches "
            "through the external --instances dataset."
        )
    if payload.get("schema") != SUITE_SCHEMA:
        raise SuiteManifestError(
            f"Suite manifest {resolved} must declare schema={SUITE_SCHEMA!r}; "
            f"got {payload.get('schema')!r}."
        )
    suite = str(payload.get("name") or "").strip()
    if not suite:
        raise SuiteManifestError(f"Suite manifest {resolved} requires a non-empty 'name'.")
    source = str(payload.get("source_dataset") or "").strip()
    if not source:
        raise SuiteManifestError(
            f"Suite manifest {resolved} requires a non-empty 'source_dataset'."
        )
    if payload.get("external_dataset_required") is not True:
        raise SuiteManifestError(
            f"Suite manifest {resolved} must set external_dataset_required=true."
        )
    values = payload.get("instance_ids")
    if not isinstance(values, list) or not values:
        raise SuiteManifestError(
            f"Suite manifest {resolved} requires a non-empty 'instance_ids' JSON list."
        )
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise SuiteManifestError(
            f"Suite manifest {resolved} instance_ids must contain non-empty strings only; "
            "do not vendor instance objects."
        )
    instance_ids = [value.strip() for value in values]
    duplicate = _duplicates(instance_ids)
    if duplicate:
        raise SuiteManifestError(
            f"Suite manifest {resolved} has duplicate instance_ids: {', '.join(duplicate)}."
        )
    return SuiteManifest(
        name=suite,
        source=source,
        instance_ids=tuple(instance_ids),
        path=resolved,
    )


def load_instance_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Validate a caller-supplied JSON list of complete SWE-bench instances."""

    resolved = Path(path).expanduser().resolve()
    payload = _read_json(
        resolved,
        label="external SWE-bench dataset",
        error_type=DatasetError,
    )
    if not isinstance(payload, list):
        raise DatasetError(
            f"External SWE-bench dataset {resolved} must be a JSON list of instance objects."
        )
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    for index, value in enumerate(payload):
        if not isinstance(value, Mapping):
            raise DatasetError(
                f"External SWE-bench dataset {resolved} row {index} must be a JSON object."
            )
        instance_id = str(value.get("instance_id") or "").strip()
        label = instance_id or f"row {index}"
        missing = [
            field
            for field in REQUIRED_INSTANCE_FIELDS
            if field not in value
            or not isinstance(value[field], str)
            or not value[field].strip()
        ]
        if missing:
            raise DatasetError(
                f"SWE-bench instance {label!r} in {resolved} is missing required fields "
                f"or has empty/non-string values: {', '.join(missing)}. Export complete "
                "benchmark rows before running the agent."
            )
        row = dict(value)
        row["instance_id"] = instance_id
        rows.append(row)
        ids.append(instance_id)
    duplicate = _duplicates(ids)
    if duplicate:
        raise DatasetError(
            f"External SWE-bench dataset {resolved} has duplicate instance_id rows: "
            f"{', '.join(duplicate)}."
        )
    return rows


def hydrate_suite(
    manifest_path: str | Path,
    dataset_path: str | Path,
) -> list[dict[str, Any]]:
    """Join an ID-only suite to complete external rows in suite order."""

    manifest = load_suite_manifest(manifest_path)
    dataset = load_instance_dataset(dataset_path)
    by_id = {row["instance_id"]: row for row in dataset}
    missing = [instance_id for instance_id in manifest.instance_ids if instance_id not in by_id]
    if missing:
        raise DatasetError(
            f"Suite {manifest.name!r} references instance IDs {', '.join(missing)} that "
            "were not found in the external dataset. Use a matching benchmark "
            "split/version."
        )
    return [dict(by_id[instance_id]) for instance_id in manifest.instance_ids]


def load_instances_for_run(
    dataset_path: str | Path,
    suite_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load all external rows or hydrate the optional ID-only suite."""

    if suite_path is not None:
        return hydrate_suite(suite_path, dataset_path)
    return load_instance_dataset(dataset_path)


__all__ = [
    "DatasetError",
    "REQUIRED_INSTANCE_FIELDS",
    "SUITE_SCHEMA",
    "SuiteManifest",
    "SuiteManifestError",
    "hydrate_suite",
    "load_instance_dataset",
    "load_instances_for_run",
    "load_suite_manifest",
]
