"""方差探针：筛稳定 target T —— 把「陷阱题」（自验证假阳性 → resolved 掷硬币）淘汰。

为什么（DECISIONS/会话设计 §8）：多 issue 会话实验要测「长上下文掉幅」，但若 target T 本身
resolved 方差大（如 sympy-13031：obvious 文件已修、真 bug 在副本，agent 用 dense repro 自验全绿
= 假阳性），solo 就忽上忽下，根本分不清 full 的变化是 damage 还是噪声。
→ 先对候选实例跑 solo×N，看每个的 resolved 方差，**只留稳定解出的（4/4 或 3/4）当 target T**。

流程：每个实例 in-docker 跑 N 次（compact=none，单发）→ 产 N 个补丁 → 官方 harness 评 resolved →
统计每个实例的 resolved 命中率/方差。harness 评分**脚本内调 WSL**（不再手动）。

用法（项目根，带代理）：
  python -m eval.swebench.variance_probe --instances <dataset.json>
  python -m eval.swebench.variance_probe --instances <dataset.json> django__django-11066 2
"""

import argparse
import glob
import json
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import config                                           # noqa: E402
from eval.swebench.run_swe import run_one_indocker                 # noqa: E402

HERE = Path(__file__).resolve().parent
# 候选：cmp_indocker 里 resolved 过的（已知至少解得出一次），筛哪些是【稳定】解出的
CANDIDATES = ["django__django-11066", "sphinx-doc__sphinx-10449",
              "pydata__xarray-3993", "astropy__astropy-12907"]


def load_inst(iid: str, instances: list[dict]) -> dict:
    for i in instances:
        if i["instance_id"] == iid:
            return i
    raise KeyError(iid)


def _win_to_wsl_path(path: Path) -> str:
    win = str(path.resolve())
    if len(win) >= 3 and win[1] == ":":
        return "/mnt/" + win[0].lower() + win[2:].replace("\\", "/")
    return win.replace("\\", "/")


def _prediction_model_name(predictions_path: Path) -> str | None:
    try:
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                model = json.loads(line).get("model_name_or_path")
                return str(model) if model else None
    except Exception:
        return None
    return None


def _load_summary_from_wsl_unc(model: str, run_id: str) -> dict | None:
    names = list(dict.fromkeys([f"{model}.{run_id}.json", f"{run_id}.{run_id}.json"]))
    patterns = []
    for root in (r"\\wsl$\Ubuntu\home", r"\\wsl.localhost\Ubuntu\home"):
        patterns.extend(fr"{root}\*\swe-runs\{name}" for name in names)
        patterns.append(fr"{root}\*\swe-runs\*.{run_id}.json")
    for pattern in patterns:
        for match in sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True):
            try:
                return json.loads(Path(match).read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in value)[:180] or "unknown"


def score_artifact_dir(run_id: str, instance_id: str) -> Path:
    return REPO / ".traces" / "swebench-score" / _safe_path_part(run_id) / _safe_path_part(instance_id)


def _unique_existing(paths: list[str]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        try:
            p = Path(path)
            key = str(p)
            if key in seen or not p.exists():
                continue
            seen.add(key)
            out.append(p)
        except Exception:
            continue
    return out


def _wsl_log_artifact_candidates(model: str, run_id: str, instance_id: str) -> dict[str, list[Path]]:
    names = list(dict.fromkeys([model, run_id]))
    report_patterns: list[str] = []
    test_output_patterns: list[str] = []
    for root in (r"\\wsl$\Ubuntu\home", r"\\wsl.localhost\Ubuntu\home"):
        for name in names:
            base_patterns = [
                fr"{root}\*\swe-runs\logs\run_evaluation\{name}.{run_id}\*\{instance_id}",
                fr"{root}\*\swe-runs\logs\run_evaluation\{name}.{run_id}\{name}\{instance_id}",
                fr"{root}\*\swe-runs\logs\run_evaluation\*{run_id}*\*\{instance_id}",
                fr"{root}\*\swe-runs\logs\run_evaluation\*{run_id}*\*{run_id}*\{instance_id}",
            ]
            report_patterns.extend(fr"{pattern}\report.json" for pattern in base_patterns)
            test_output_patterns.extend(fr"{pattern}\test_output.txt" for pattern in base_patterns)
    return {
        "report": _unique_existing([match for pattern in report_patterns for match in glob.glob(pattern)]),
        "test_output": _unique_existing(
            [match for pattern in test_output_patterns for match in glob.glob(pattern)]
        ),
    }


def _copy_first(candidates: list[Path], dest: Path) -> str:
    for src in candidates:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
            return str(dest)
        except Exception:
            continue
    return ""


def materialize_score_artifacts(
    predictions_path: Path,
    run_id: str,
    instance_id: str,
    summary: dict | None = None,
) -> dict[str, str]:
    """Copy official SWE-bench scorer evidence to a stable local directory."""

    model = _prediction_model_name(predictions_path) or run_id
    out_dir = score_artifact_dir(run_id, instance_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"artifact_dir": str(out_dir)}

    if summary is not None:
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["summary"] = str(summary_path)

    candidates = _wsl_log_artifact_candidates(model, run_id, instance_id)
    report_path = _copy_first(candidates.get("report", []), out_dir / "report.json")
    test_output_path = _copy_first(candidates.get("test_output", []), out_dir / "test_output.txt")
    if report_path:
        paths["report"] = report_path
    if test_output_path:
        paths["test_output"] = test_output_path

    (out_dir / "manifest.json").write_text(
        json.dumps(paths, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def score_harness(predictions_path: Path, run_id: str, instance_id: str) -> bool | None:
    """Run the official WSL SWE-bench harness and read its schema-v2 summary."""
    wsl_path = _win_to_wsl_path(predictions_path)
    cmd = (
        "source ~/swe/bin/activate; cd ~/swe-runs; "
        "python -m swebench.harness.run_evaluation "
        "--dataset_name princeton-nlp/SWE-bench_Verified "
        f"--predictions_path {wsl_path} --max_workers 1 --run_id {run_id} "
        f"--instance_ids {instance_id} --cache_level instance"
    )
    subprocess.run(
        ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=1800,
    )

    dump = REPO / ".traces" / f"vp_report_{run_id}.json"
    dump.parent.mkdir(parents=True, exist_ok=True)
    model = _prediction_model_name(predictions_path) or run_id
    rep = _load_summary_from_wsl_unc(model, run_id)
    if rep:
        dump.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        materialize_score_artifacts(predictions_path, run_id, instance_id, rep)
        return instance_id in rep.get("resolved_ids", [])

    models = " ".join(shlex.quote(m) for m in dict.fromkeys([model, run_id]))
    rcmd = (
        f"dest={shlex.quote(_win_to_wsl_path(dump))}; "
        f"run_id={shlex.quote(run_id)}; "
        f"for model in {models}; do "
        'f="$HOME/swe-runs/${model}.${run_id}.json"; '
        'if [ -f "$f" ]; then cp "$f" "$dest"; exit 0; fi; '
        "done; "
        'f=$(ls -t "$HOME"/swe-runs/*.${run_id}.json 2>/dev/null | head -n1); '
        'if [ -n "$f" ]; then cp "$f" "$dest"; else echo "{}" > "$dest"; fi'
    )
    subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", rcmd], capture_output=True, timeout=60)
    try:
        rep = json.loads(dump.read_text(encoding="utf-8"))
        if not rep:
            materialize_score_artifacts(predictions_path, run_id, instance_id, None)
            return None
        materialize_score_artifacts(predictions_path, run_id, instance_id, rep)
        return instance_id in rep.get("resolved_ids", [])
    except Exception:
        materialize_score_artifacts(predictions_path, run_id, instance_id, None)
        return None


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        required=True,
        type=Path,
        help="Path to a complete external SWE-bench dataset JSON list.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        help="Optional ace.swebench-suite.v1 ID-only manifest to hydrate.",
    )
    parser.add_argument("instance_id", nargs="?")
    parser.add_argument("reps", nargs="?", type=int, default=4)
    args = parser.parse_args(argv)
    if args.reps < 1:
        parser.error("reps must be >= 1")

    from eval.swebench.data import (  # noqa: WPS433
        DatasetError,
        SuiteManifestError,
        load_instances_for_run,
    )

    try:
        instances = load_instances_for_run(args.instances, args.suite)
    except (DatasetError, SuiteManifestError) as exc:
        parser.error(str(exc))
    ids = [args.instance_id] if args.instance_id else CANDIDATES
    reps = args.reps
    by_id = {row["instance_id"]: row for row in instances}
    missing = [instance_id for instance_id in ids if instance_id not in by_id]
    if missing:
        parser.error(f"instance IDs are not present in the selected data: {', '.join(missing)}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"方差探针：{len(ids)} 候选 × solo reps={reps}（筛稳定 target T）", flush=True)

    results = defaultdict(list)   # iid -> [resolved bool/None per rep]
    for iid in ids:
        inst = by_id[iid]
        for rep in range(reps):
            tag = f"vp_{iid.split('__')[-1]}_r{rep}"
            print(f"\n[{iid} rep{rep}] in-docker 跑 agent …", flush=True)
            r = run_one_indocker(inst, max_turns=120,
                                 meta={"probe": "variance", "instance_id": iid, "rep": rep})
            patch = r.get("model_patch", "")
            if not patch:
                print(f"   空补丁（{r.get('failure_reason')}）→ 记 unresolved", flush=True)
                results[iid].append(False)
                continue
            pred = HERE / f"predictions_{tag}.jsonl"
            pred.write_text(json.dumps({"instance_id": iid, "model_name_or_path": tag,
                                        "model_patch": patch}, ensure_ascii=False) + "\n",
                            encoding="utf-8")
            print(f"   补丁文件数={patch.count('+++ b/')} → harness 评分 …", flush=True)
            rv = score_harness(pred, tag, iid)
            results[iid].append(rv)
            print(f"   resolved={rv}", flush=True)

    # 汇总：每实例命中率 + 稳定判定
    print(f"\n=== 方差探针汇总（{stamp}）===")
    print(f"{'instance':<34}{'resolved':>16}{'命中率':>8}{'判定':>10}")
    report = {}
    for iid in ids:
        vals = results[iid]
        hits = sum(1 for v in vals if v is True)
        n = len(vals)
        rate = hits / max(1, n)
        verdict = ("稳定✓" if rate >= 0.75 else ("陷阱✗" if 0 < rate < 0.75 else "全失败"))
        print(f"{iid:<34}{str(vals):>16}{f'{hits}/{n}':>8}{verdict:>10}")
        report[iid] = {"resolved": vals, "rate": rate, "verdict": verdict}
    stable = [iid for iid in ids if report[iid]["rate"] >= 0.75]
    print(f"\n稳定 target T 候选（命中率≥75%）：{stable or '（无——需放宽或换候选）'}")
    out = REPO / ".traces" / f"variance_probe_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告 → {out}")


if __name__ == "__main__":
    main()
