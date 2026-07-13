"""CLI runner and gate aggregation for the context-budget offline evaluation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .cases import (
    ERROR,
    FAIL,
    INCONCLUSIVE,
    INVALID,
    PASS,
    PROTOCOL_VERSION,
    REQUIRED_CASE_IDS,
    SCHEMA_VERSION,
    STATUS_VOCABULARY,
    CaseResult,
    protocol_fingerprint,
    run_cases,
)


_CODE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


def _parse_cases(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return REQUIRED_CASE_IDS
    selected: list[str] = []
    for part in raw.split(","):
        case_id = part.strip()
        if case_id and case_id not in selected:
            selected.append(case_id)
    return tuple(selected)


def summarize_results(
    results: Sequence[CaseResult],
    *,
    selected_case_ids: Sequence[str],
) -> dict[str, Any]:
    counts = {status: 0 for status in STATUS_VOCABULARY}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    selected = tuple(selected_case_ids)
    actual = tuple(result.case_id for result in results)
    selected_is_full = (
        len(selected) == len(REQUIRED_CASE_IDS)
        and set(selected) == set(REQUIRED_CASE_IDS)
    )
    actual_is_full = (
        len(actual) == len(REQUIRED_CASE_IDS)
        and set(actual) == set(REQUIRED_CASE_IDS)
    )
    selected_result_coverage = (
        len(selected) == len(actual) and set(selected) == set(actual)
    )
    full_coverage = selected_is_full and actual_is_full and selected_result_coverage

    statuses = tuple(result.status for result in results)
    if not full_coverage:
        gate_status = INCONCLUSIVE
    elif ERROR in statuses:
        gate_status = ERROR
    elif INVALID in statuses:
        gate_status = INVALID
    elif INCONCLUSIVE in statuses:
        gate_status = INCONCLUSIVE
    elif FAIL in statuses:
        gate_status = FAIL
    elif statuses and all(status == PASS for status in statuses):
        gate_status = PASS
    else:
        gate_status = ERROR

    gate_pass = full_coverage and gate_status == PASS
    return {
        "counts": counts,
        "valid_case_count": counts.get(PASS, 0) + counts.get(FAIL, 0),
        "excluded_case_count": (
            counts.get(INVALID, 0)
            + counts.get(INCONCLUSIVE, 0)
            + counts.get(ERROR, 0)
        ),
        "result_case_ids": list(actual),
        "selected_result_coverage": selected_result_coverage,
        "full_gate_coverage": full_coverage,
        "gate_status": gate_status,
        "gate_pass": gate_pass,
    }


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _environment() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.system() or "unknown",
        "machine": platform.machine() or "unknown",
    }


def _validate_code_version(code_version: str) -> str:
    value = str(code_version).strip()
    if not _CODE_VERSION.fullmatch(value):
        raise ValueError(
            "code_version must be a path-free revision label using letters, "
            "digits, '.', '_', '+', or '-'"
        )
    return value


def build_run_records(
    results: Sequence[CaseResult],
    *,
    selected_case_ids: Sequence[str],
    code_version: str,
    timestamp_utc: str | None = None,
) -> list[dict[str, Any]]:
    revision = _validate_code_version(code_version)
    timestamp = timestamp_utc or _timestamp_utc()
    selected = tuple(selected_case_ids)
    summary = summarize_results(results, selected_case_ids=selected)
    fingerprint = protocol_fingerprint()
    environment = _environment()

    records: list[dict[str, Any]] = [
        {
            "record_type": "run_summary",
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": fingerprint,
            "code_version": revision,
            "timestamp_utc": timestamp,
            "environment": environment,
            "required_case_ids": list(REQUIRED_CASE_IDS),
            "selected_case_ids": list(selected),
            **summary,
            "valid_statuses": [PASS, FAIL],
            "excluded_statuses": [INVALID, INCONCLUSIVE, ERROR],
            "what_this_does_not_prove": (
                "This deterministic mechanism gate does not prove provider "
                "tokenizer accuracy, model behavior, or task quality."
            ),
        }
    ]
    for result in results:
        records.append(
            {
                "record_type": "case_result",
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "protocol_sha256": fingerprint,
                "code_version": revision,
                "timestamp_utc": timestamp,
                "case_id": result.case_id,
                "description": result.description,
                "required": result.required,
                "status": result.status,
                "evidence": dict(result.evidence),
                "message": result.message,
                "what_this_does_not_prove": result.what_this_does_not_prove,
            }
        )
    return records


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}-",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic context-budget offline gate."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", help="comma-separated case IDs")
    parser.add_argument("--code-version", default="WORKTREE")
    args = parser.parse_args(argv)

    selected = _parse_cases(args.cases)
    if not selected:
        print("No case IDs selected.", file=sys.stderr)
        return 2
    unknown = [case_id for case_id in selected if case_id not in REQUIRED_CASE_IDS]
    if unknown:
        print(f"Unknown case IDs: {', '.join(unknown)}", file=sys.stderr)
        return 2
    try:
        results = run_cases(selected)
        records = build_run_records(
            results,
            selected_case_ids=selected,
            code_version=args.code_version,
        )
        _write_jsonl(args.output, records)
    except (OSError, ValueError) as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    summary = records[0]
    counts = summary["counts"]
    print(
        "Cases: "
        + " ".join(
            f"{status}={counts.get(status, 0)}" for status in STATUS_VOCABULARY
        )
    )
    print(f"Full gate coverage: {summary['full_gate_coverage']}")
    print(f"Gate: {summary['gate_status']}")
    print("Results written.")
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_run_records", "main", "summarize_results"]
