#!/usr/bin/env python3
"""Canonical local regression gate entry points.

The script deliberately separates experiment runners from checkers. For
SWE-bench, it checks already-produced JSONL rows against a fixed suite and a
same-protocol baseline; it does not run Docker or the official harness itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_INFRA = 2

REPO = Path(__file__).resolve().parents[1]
SCORED_STATUSES = {"scored", "scored_retry", "empty_patch_unresolved", None, ""}
INFRA_STATUSES = {"ERROR", "runner_error", "score_error"}


@dataclass
class InstanceResult:
    instance_id: str
    pass_count: int
    repeat: int
    resolved: bool
    flaky: bool
    statuses: list[str] = field(default_factory=list)


@dataclass
class SweSummary:
    suite: str
    repeat: int
    denominator: int
    resolved_count: int
    flaky_count: int
    instance_results: dict[str, InstanceResult]
    infra_errors: list[str] = field(default_factory=list)
    unexpected_instance_ids: list[str] = field(default_factory=list)

    @property
    def infra_error_count(self) -> int:
        return len(self.infra_errors)


@dataclass
class GateResult:
    exit_code: int
    messages: list[str]
    summary: SweSummary | None = None


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO / path


def _read_json(path: str | Path) -> Any:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in _resolve(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_suite(path: str | Path) -> tuple[str, list[str], dict[str, Any]]:
    data = _read_json(path)
    meta: dict[str, Any] = {}
    if isinstance(data, dict):
        meta = data
        suite = str(data.get("name") or data.get("suite") or _resolve(path).stem)
        instances = data.get("instance_ids") if data.get("schema") == "ace.swebench-suite.v1" else data.get("instances")
        if instances is None and data.get("instances_file"):
            _, instances, nested = load_suite(data["instances_file"])
            meta = {**nested, **data}
        if instances is None:
            raise ValueError(f"suite file has no instances: {path}")
    elif isinstance(data, list):
        suite = _resolve(path).stem
        instances = data
    else:
        raise ValueError(f"unsupported suite format: {path}")

    ids: list[str] = []
    for item in instances:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict) and item.get("instance_id"):
            ids.append(str(item["instance_id"]))
        else:
            raise ValueError(f"unsupported suite instance entry: {item!r}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"suite has duplicate instance ids: {path}")
    return suite, ids, meta


def suite_ids_sha256(instance_ids: list[str]) -> str:
    payload = json.dumps(instance_ids, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_rep(row: dict[str, Any]) -> int:
    rep = row.get("rep", row.get("run_idx", row.get("repeat_index", 0)))
    try:
        return int(rep)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid rep value for {row.get('instance_id')}: {rep!r}") from exc


def _is_scored_row(row: dict[str, Any]) -> bool:
    return isinstance(row.get("resolved"), bool) and row.get("score_status") in SCORED_STATUSES


def aggregate_swe_results(rows: list[dict[str, Any]], suite_ids: list[str], repeat: int, suite: str = "") -> SweSummary:
    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    suite_set = set(suite_ids)
    by_run: dict[tuple[str, int], dict[str, Any]] = {}
    infra_errors: list[str] = []
    unexpected: set[str] = set()

    for row in rows:
        instance_id = str(row.get("instance_id") or "")
        if not instance_id:
            infra_errors.append("row without instance_id")
            continue
        if instance_id not in suite_set:
            unexpected.add(instance_id)
            continue

        try:
            rep = _row_rep(row)
        except ValueError as exc:
            infra_errors.append(str(exc))
            continue
        if rep < 0 or rep >= repeat:
            infra_errors.append(f"{instance_id}: rep {rep} outside expected repeat=0..{repeat - 1}")
            continue

        status = row.get("score_status")
        if status in INFRA_STATUSES or row.get("resolved") is None:
            infra_errors.append(f"{instance_id} rep {rep}: infra status={status}")
            continue
        if not _is_scored_row(row):
            infra_errors.append(f"{instance_id} rep {rep}: unscored row status={status!r}")
            continue
        by_run[(instance_id, rep)] = row

    instance_results: dict[str, InstanceResult] = {}
    majority = repeat // 2 + 1
    for instance_id in suite_ids:
        missing = [rep for rep in range(repeat) if (instance_id, rep) not in by_run]
        if missing:
            infra_errors.append(f"{instance_id}: missing reps {missing}")
            continue
        runs = [by_run[(instance_id, rep)] for rep in range(repeat)]
        pass_count = sum(1 for row in runs if row.get("resolved") is True)
        instance_results[instance_id] = InstanceResult(
            instance_id=instance_id,
            pass_count=pass_count,
            repeat=repeat,
            resolved=pass_count >= majority,
            flaky=0 < pass_count < repeat,
            statuses=[str(row.get("score_status") or "scored") for row in runs],
        )

    resolved_count = sum(1 for result in instance_results.values() if result.resolved)
    flaky_count = sum(1 for result in instance_results.values() if result.flaky)
    return SweSummary(
        suite=suite,
        repeat=repeat,
        denominator=len(suite_ids),
        resolved_count=resolved_count,
        flaky_count=flaky_count,
        instance_results=instance_results,
        infra_errors=infra_errors,
        unexpected_instance_ids=sorted(unexpected),
    )


def _thresholds(baseline: dict[str, Any]) -> tuple[int, int]:
    thresholds = baseline.get("thresholds") or {}
    max_drop = int(thresholds.get("max_resolved_drop_abs", 0))
    max_flaky_inc = int(thresholds.get("max_flaky_increase_abs", 0))
    return max_drop, max_flaky_inc


def run_swe_check(
    suite_path: str | Path,
    baseline_path: str | Path,
    results_path: str | Path,
    *,
    allow_fixture_baseline: bool = False,
) -> GateResult:
    suite, suite_ids, suite_meta = load_suite(suite_path)
    baseline = _read_json(baseline_path)
    messages: list[str] = []

    if baseline.get("not_for_gate") and not allow_fixture_baseline:
        return GateResult(EXIT_INFRA, ["baseline is marked not_for_gate; pass --allow-fixture-baseline for checker smoke only"])

    repeat = int(baseline.get("repeat") or 0)
    if repeat < 1:
        return GateResult(EXIT_INFRA, ["baseline.repeat must be >= 1"])

    expected_hash = baseline.get("instances_sha256")
    current_hash = suite_ids_sha256(suite_ids)
    if expected_hash and expected_hash != current_hash:
        return GateResult(EXIT_INFRA, [f"suite hash mismatch: expected {expected_hash}, got {current_hash}"])

    if baseline.get("suite") and baseline["suite"] != suite:
        messages.append(f"warning: baseline suite={baseline['suite']} current suite={suite}")
    if baseline.get("denominator") is not None and int(baseline["denominator"]) != len(suite_ids):
        return GateResult(EXIT_INFRA, [f"denominator mismatch: baseline={baseline['denominator']} suite={len(suite_ids)}"])

    rows = _read_jsonl(results_path)
    summary = aggregate_swe_results(rows, suite_ids, repeat=repeat, suite=suite)

    if summary.unexpected_instance_ids:
        messages.append(f"unexpected instances ignored: {len(summary.unexpected_instance_ids)}")
    if summary.infra_errors:
        messages.extend(summary.infra_errors)
        return GateResult(EXIT_INFRA, messages, summary)

    baseline_resolved = int(baseline.get("resolved_count", 0))
    baseline_flaky = int(baseline.get("flaky_count", 0))
    max_drop, max_flaky_inc = _thresholds(baseline)
    resolved_drop = baseline_resolved - summary.resolved_count
    flaky_increase = summary.flaky_count - baseline_flaky

    messages.append(
        "swe-check "
        f"suite={suite} repeat={repeat} resolved={summary.resolved_count}/{summary.denominator} "
        f"baseline={baseline_resolved}/{summary.denominator} drop={resolved_drop} "
        f"flaky={summary.flaky_count} baseline_flaky={baseline_flaky} flaky_delta={flaky_increase}"
    )

    if resolved_drop > max_drop:
        messages.append(f"REGRESSION: resolved drop {resolved_drop} > allowed {max_drop}")
        return GateResult(EXIT_REGRESSION, messages, summary)
    if flaky_increase > max_flaky_inc:
        messages.append(f"REGRESSION: flaky increase {flaky_increase} > allowed {max_flaky_inc}")
        return GateResult(EXIT_REGRESSION, messages, summary)
    messages.append("PASS")
    return GateResult(EXIT_OK, messages, summary)


def write_swe_baseline(
    suite_path: str | Path,
    results_path: str | Path,
    out_path: str | Path,
    *,
    model_id: str,
    repeat: int,
    temperature: str = "",
    seed: str = "",
    max_resolved_drop_abs: int = 1,
    max_flaky_increase_abs: int = 2,
    allow_infra: bool = False,
) -> GateResult:
    suite, suite_ids, suite_meta = load_suite(suite_path)
    summary = aggregate_swe_results(_read_jsonl(results_path), suite_ids, repeat=repeat, suite=suite)
    messages: list[str] = []
    if summary.infra_errors and not allow_infra:
        return GateResult(EXIT_INFRA, summary.infra_errors, summary)

    baseline = {
        "suite": suite,
        "status": "active",
        "instances_file": str(Path(suite_path).as_posix()),
        "instances_sha256": suite_ids_sha256(suite_ids),
        "source_selection_rule": suite_meta.get("selection_rule", ""),
        "model_id": model_id,
        "temperature": temperature,
        "seed": seed,
        "repeat": repeat,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "denominator": len(suite_ids),
        "resolved_count": summary.resolved_count,
        "flaky_count": summary.flaky_count,
        "thresholds": {
            "max_resolved_drop_abs": max_resolved_drop_abs,
            "max_flaky_increase_abs": max_flaky_increase_abs,
        },
    }
    out = _resolve(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    messages.append(f"wrote baseline: {out}")
    return GateResult(EXIT_OK, messages, summary)


def run_mcp_smoke() -> int:
    """Run MCP mechanism smoke (no LLM). SKIPPED when mcp package missing (exit 0)."""
    output = REPO / "eval" / "reports" / "regression-gate" / "mcp-smoke.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.mcp_eval.smoke",
            "--output",
            str(output),
        ],
        cwd=REPO,
    )
    return proc.returncode


def run_mcp_reliability() -> int:
    """Run deterministic MCP per-server lifecycle reliability cases."""
    output = (
        REPO
        / "eval"
        / "reports"
        / "regression-gate"
        / "mcp-reliability.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "eval.mcp_eval.reliability",
            "--output",
            str(output),
        ],
        cwd=REPO,
    )
    return proc.returncode


def run_offline(
    *,
    collect_only: bool = False,
    skip_validate_tasks: bool = False,
    skip_mcp_smoke: bool = False,
    skip_mcp_reliability: bool = False,
) -> int:
    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "eval/memory/tests",
        "eval/runtime_eval",
        "eval/memory_eval",
        "-m",
        "not live",
        "-q",
    ]
    if collect_only:
        pytest_cmd.insert(3, "--collect-only")
    proc = subprocess.run(pytest_cmd, cwd=REPO)
    if proc.returncode != 0:
        return proc.returncode
    if not skip_mcp_smoke and not collect_only:
        smoke_code = run_mcp_smoke()
        if smoke_code != 0:
            return smoke_code
    if not skip_mcp_reliability and not collect_only:
        reliability_code = run_mcp_reliability()
        if reliability_code != 0:
            return reliability_code
    if skip_validate_tasks or collect_only:
        return EXIT_OK
    validate = subprocess.run([sys.executable, "scripts/validate_tasks.py"], cwd=REPO)
    return validate.returncode


def _print_result(result: GateResult) -> None:
    for message in result.messages:
        print(message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    offline = sub.add_parser("offline", help="run offline regression tests")
    offline.add_argument("--collect-only", action="store_true")
    offline.add_argument("--skip-validate-tasks", action="store_true")
    offline.add_argument("--skip-mcp-smoke", action="store_true", help="skip eval.mcp_eval.smoke after pytest")
    offline.add_argument("--skip-mcp-reliability", action="store_true", help="skip eval.mcp_eval.reliability after smoke")

    check = sub.add_parser("swe-check", help="check SWE-bench JSONL against a fixed baseline")
    check.add_argument("--suite", required=True)
    check.add_argument("--baseline", required=True)
    check.add_argument("--results", required=True)
    check.add_argument("--allow-fixture-baseline", action="store_true")

    freeze = sub.add_parser("swe-baseline", help="freeze a same-protocol SWE-bench baseline")
    freeze.add_argument("--suite", required=True)
    freeze.add_argument("--results", required=True)
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--model-id", required=True)
    freeze.add_argument("--repeat", type=int, required=True)
    freeze.add_argument("--temperature", default="")
    freeze.add_argument("--seed", default="")
    freeze.add_argument("--max-resolved-drop-abs", type=int, default=1)
    freeze.add_argument("--max-flaky-increase-abs", type=int, default=2)
    freeze.add_argument("--allow-infra", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "offline":
        return run_offline(
            collect_only=args.collect_only,
            skip_validate_tasks=args.skip_validate_tasks,
            skip_mcp_smoke=args.skip_mcp_smoke,
            skip_mcp_reliability=args.skip_mcp_reliability,
        )
    if args.command == "swe-check":
        result = run_swe_check(
            args.suite,
            args.baseline,
            args.results,
            allow_fixture_baseline=args.allow_fixture_baseline,
        )
        _print_result(result)
        return result.exit_code
    if args.command == "swe-baseline":
        result = write_swe_baseline(
            args.suite,
            args.results,
            args.out,
            model_id=args.model_id,
            repeat=args.repeat,
            temperature=args.temperature,
            seed=args.seed,
            max_resolved_drop_abs=args.max_resolved_drop_abs,
            max_flaky_increase_abs=args.max_flaky_increase_abs,
            allow_infra=args.allow_infra,
        )
        _print_result(result)
        return result.exit_code
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
