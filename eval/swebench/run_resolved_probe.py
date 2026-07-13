"""Run a controlled SWE-bench resolved-rate probe.

This runner exists for model/prompt A/Bs where we want official harness
outcomes, not localization proxies. It intentionally reuses the current
in-Docker prompt from ``run_swe.run_one_indocker`` and only changes the model
by mutating ``agent.config.MODEL_ID`` inside this process after ``.env`` has
been loaded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SCORED_STATUSES = {"scored", "scored_retry", "empty_patch_unresolved", None, ""}
INFRA_FAILURE_REASONS = {"docker_error", "llm_api_error", "runner_error"}
DEFAULT_CONDITION = "strong_model_current_prompt"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def safe_run_id(tag: str, index: int, instance_id: str, rep: int = 0) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_id)
    suffix = f"_r{rep}" if rep else ""
    return f"{tag}_{index:02d}_{clean}{suffix}"


def build_condition_label(
    *,
    explicit: str,
    test_entry_hint_mode: str,
    verification_prompt_mode: str,
    identity_prompt_mode: str = "current",
    skills_mode: str = "on",
) -> str:
    explicit = (explicit or "").strip()
    if explicit:
        return explicit
    return (
        f"identity_{identity_prompt_mode}_hint_{test_entry_hint_mode}_"
        f"verify_{verification_prompt_mode}_skills_{skills_mode}"
    )


def _row_complete(row: dict[str, Any]) -> bool:
    return _is_scored_row(row)


def _is_scored_row(row: dict[str, Any]) -> bool:
    return isinstance(row.get("resolved"), bool) and row.get("score_status") in SCORED_STATUSES


def scored_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _is_scored_row(row)]


def result_is_infra_error(result: dict[str, Any]) -> bool:
    if result.get("run_status") == "error":
        return True
    return bool(result.get("error")) and result.get("failure_reason") in INFRA_FAILURE_REASONS


def error_row_from_result(
    *,
    inst: dict[str, Any],
    run_id: str,
    tag: str,
    model_id: str,
    rep: int,
    repeat: int,
    result: dict[str, Any],
    elapsed_sec: float,
    condition: str = DEFAULT_CONDITION,
    test_entry_hint_mode: str | None = None,
    verification_prompt_mode: str | None = None,
    identity_prompt_mode: str | None = None,
    skills_enabled: bool | None = None,
) -> dict[str, Any]:
    return {
        "instance_id": inst["instance_id"],
        "repo": inst.get("repo"),
        "run_id": run_id,
        "tag": tag,
        "condition": condition,
        "model_id": model_id,
        "rep": rep,
        "repeat": repeat,
        "test_entry_hint_mode": test_entry_hint_mode,
        "verification_prompt_mode": verification_prompt_mode,
        "identity_prompt_mode": identity_prompt_mode,
        "skills_enabled": skills_enabled,
        "resolved": None,
        "score_status": "ERROR",
        "run_status": "error",
        "has_patch": False,
        "patch_files": [],
        "patch_file_count": 0,
        "patch_chars": 0,
        "failure_reason": result.get("failure_reason") or "runner_error",
        "error_kind": result.get("error_kind") or "RunnerError",
        "error": result.get("error", ""),
        "modified": result.get("modified", []),
        "untracked": result.get("untracked", []),
        "turns": result.get("turns"),
        "max_turns_reached": result.get("max_turns_reached"),
        "stop_reason": result.get("stop_reason"),
        "trace_path": result.get("trace_path"),
        "evidence_dir": result.get("evidence_dir"),
        "predictions_path": "",
        "score_artifacts_dir": "",
        "elapsed_sec": round(elapsed_sec, 2),
    }


def completed_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for row in rows:
        instance_id = row.get("instance_id")
        if not instance_id or not _row_complete(row):
            continue
        try:
            rep = int(row.get("rep", 0))
        except (TypeError, ValueError):
            continue
        keys.add((str(instance_id), rep))
    return keys


def _patch_files(patch: str) -> list[str]:
    files: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[len("+++ b/") :])
    return files


def _write_prediction(path: Path, instance_id: str, run_id: str, patch: str) -> None:
    row = {
        "instance_id": instance_id,
        "model_name_or_path": run_id,
        "model_patch": patch,
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def _score_with_retry(score_harness, pred: Path, run_id: str, instance_id: str) -> tuple[bool | None, str]:
    resolved = score_harness(pred, run_id, instance_id)
    if resolved is not None:
        return resolved, "scored"

    retry_run_id = f"{run_id}_retry"
    retry_pred = pred.with_name(f"{pred.stem}_retry{pred.suffix}")
    patch = json.loads(pred.read_text(encoding="utf-8").splitlines()[0])["model_patch"]
    _write_prediction(retry_pred, instance_id, retry_run_id, patch)
    resolved = score_harness(retry_pred, retry_run_id, instance_id)
    if resolved is not None:
        return resolved, "scored_retry"
    return None, "score_error"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        required=True,
        help="Path to a complete external SWE-bench dataset JSON list.",
    )
    parser.add_argument(
        "--suite",
        default="",
        help="Optional ace.swebench-suite.v1 ID-only manifest to hydrate.",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--test-entry-hint", choices=["auto", "off"], default="auto")
    parser.add_argument("--verification-prompt", choices=["default", "strong", "coverage"], default="default")
    parser.add_argument(
        "--identity-prompt",
        choices=["current", "legacy", "cc-core-cn"],
        default="current",
    )
    parser.add_argument("--skills", choices=["on", "off"], default="on")
    parser.add_argument("--condition", default="")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    condition = build_condition_label(
        explicit=args.condition,
        test_entry_hint_mode=args.test_entry_hint,
        verification_prompt_mode=args.verification_prompt,
        identity_prompt_mode=args.identity_prompt,
        skills_mode=args.skills,
    )

    from agent import config  # noqa: WPS433

    config.MODEL_ID = args.model_id

    from eval.swebench.run_swe import run_one_indocker  # noqa: WPS433
    from eval.swebench.variance_probe import score_artifact_dir, score_harness  # noqa: WPS433

    from eval.swebench.data import (  # noqa: WPS433
        DatasetError,
        SuiteManifestError,
        load_instances_for_run,
    )

    instances_path = Path(args.instances).expanduser()
    if not instances_path.is_absolute():
        instances_path = REPO / instances_path
    suite_path = Path(args.suite).expanduser() if args.suite else None
    if suite_path is not None and not suite_path.is_absolute():
        suite_path = REPO / suite_path
    out = Path(args.out) if args.out else REPO / "eval" / "swebench" / f"resolved_{args.tag}.jsonl"
    out = out.resolve()
    prediction_dir = REPO / "eval" / "swebench"

    try:
        insts = load_instances_for_run(instances_path, suite_path)
    except (DatasetError, SuiteManifestError) as exc:
        parser.error(str(exc))
    if args.limit:
        insts = insts[: args.limit]

    completed = completed_keys(_read_jsonl(out))
    total_runs = len(insts) * args.repeat

    print(
        f"START tag={args.tag} model={config.MODEL_ID} instances={len(insts)} repeat={args.repeat} "
        f"condition={condition} test_entry_hint={args.test_entry_hint} "
        f"verification_prompt={args.verification_prompt} identity_prompt={args.identity_prompt} "
        f"skills={args.skills} "
        f"already_done={len(completed)} out={out}",
        flush=True,
    )

    started = time.time()
    for index, inst in enumerate(insts):
        instance_id = inst["instance_id"]
        for rep in range(args.repeat):
            ordinal = index * args.repeat + rep + 1
            if (instance_id, rep) in completed:
                print(f"[{ordinal}/{total_runs}] skip done {instance_id} rep={rep}", flush=True)
                continue

            run_id = safe_run_id(args.tag, index, instance_id, rep=rep)
            meta = {
                "benchmark": "swebench_verified",
                "condition": condition,
                "model_id": args.model_id,
                "tag": args.tag,
                "run_id": run_id,
                "instance_id": instance_id,
                "rep": rep,
                "repeat": args.repeat,
                "test_entry_hint_mode": args.test_entry_hint,
                "verification_prompt_mode": args.verification_prompt,
                "identity_prompt_mode": args.identity_prompt,
                "skills_enabled": args.skills == "on",
            }
            print(f"\n[{ordinal}/{total_runs}] run {instance_id} rep={rep} run_id={run_id}", flush=True)
            t0 = time.time()
            try:
                result = run_one_indocker(
                    inst,
                    max_turns=args.max_turns,
                    meta=meta,
                    test_entry_hint_mode=args.test_entry_hint,
                    verification_prompt_mode=args.verification_prompt,
                    identity_prompt_mode=args.identity_prompt,
                    skills_enabled=args.skills == "on",
                )
                if result_is_infra_error(result):
                    row = error_row_from_result(
                        inst=inst,
                        run_id=run_id,
                        tag=args.tag,
                        model_id=args.model_id,
                        rep=rep,
                        repeat=args.repeat,
                        result=result,
                        elapsed_sec=time.time() - t0,
                        condition=condition,
                        test_entry_hint_mode=args.test_entry_hint,
                        verification_prompt_mode=args.verification_prompt,
                        identity_prompt_mode=args.identity_prompt,
                        skills_enabled=args.skills == "on",
                    )
                    _jsonl_append(out, row)
                    print(
                        f"[{ordinal}/{total_runs}] done {instance_id} rep={rep} "
                        f"resolved={row.get('resolved')} status={row.get('score_status')} "
                        f"error_kind={row.get('error_kind')} elapsed={row.get('elapsed_sec')}s",
                        flush=True,
                    )
                    continue

                patch = result.get("model_patch", "") or ""
                patch_files = _patch_files(patch)
                pred = prediction_dir / f"predictions_{run_id}.jsonl"

                if patch.strip():
                    _write_prediction(pred, instance_id, run_id, patch)
                    resolved, score_status = _score_with_retry(score_harness, pred, run_id, instance_id)
                else:
                    resolved, score_status = False, "empty_patch_unresolved"

                row = {
                    "instance_id": instance_id,
                    "repo": inst.get("repo"),
                    "run_id": run_id,
                    "tag": args.tag,
                    "condition": condition,
                    "model_id": args.model_id,
                    "rep": rep,
                    "repeat": args.repeat,
                    "test_entry_hint_mode": result.get("test_entry_hint_mode") or args.test_entry_hint,
                    "verification_prompt_mode": result.get("verification_prompt_mode")
                    or args.verification_prompt,
                    "identity_prompt_mode": result.get("identity_prompt_mode") or args.identity_prompt,
                    "skills_enabled": result.get("skills_enabled", args.skills == "on"),
                    "resolved": resolved,
                    "score_status": score_status,
                    "has_patch": bool(patch.strip()),
                    "patch_files": patch_files,
                    "patch_file_count": len(patch_files),
                    "patch_chars": len(patch),
                    "failure_reason": result.get("failure_reason"),
                    "modified": result.get("modified", []),
                    "untracked": result.get("untracked", []),
                    "turns": result.get("turns"),
                    "max_turns_reached": result.get("max_turns_reached"),
                    "stop_reason": result.get("stop_reason"),
                    "trace_path": result.get("trace_path"),
                    "evidence_dir": result.get("evidence_dir"),
                    "predictions_path": str(pred) if patch.strip() else "",
                    "score_artifacts_dir": str(score_artifact_dir(run_id, instance_id)) if patch.strip() else "",
                    "elapsed_sec": round(time.time() - t0, 2),
                }
            except Exception as exc:
                row = error_row_from_result(
                    inst=inst,
                    run_id=run_id,
                    tag=args.tag,
                    model_id=args.model_id,
                    rep=rep,
                    repeat=args.repeat,
                    result={
                        "failure_reason": "runner_error",
                        "error_kind": type(exc).__name__,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    elapsed_sec=time.time() - t0,
                    condition=condition,
                    test_entry_hint_mode=args.test_entry_hint,
                    verification_prompt_mode=args.verification_prompt,
                    identity_prompt_mode=args.identity_prompt,
                    skills_enabled=args.skills == "on",
                )

            _jsonl_append(out, row)
            print(
                f"[{ordinal}/{total_runs}] done {instance_id} rep={rep} "
                f"resolved={row.get('resolved')} status={row.get('score_status')} "
                f"elapsed={row.get('elapsed_sec')}s",
                flush=True,
            )
            if _row_complete(row):
                completed.add((instance_id, rep))

    rows = _read_jsonl(out)
    scored = scored_rows(rows)
    errors = [row for row in rows if not _is_scored_row(row)]
    resolved_n = sum(1 for r in scored if r.get("resolved") is True)
    print(
        f"\nDONE tag={args.tag} rows={len(rows)} scored={len(scored)} errors={len(errors)} "
        f"resolved={resolved_n}/{len(scored) if scored else 0} "
        f"rate={(resolved_n / len(scored)) if scored else 0:.3f} "
        f"elapsed_total={time.time() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
