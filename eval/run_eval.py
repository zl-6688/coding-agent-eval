#!/usr/bin/env python3
"""eval/run_eval.py — 评估 harness（客观校验 + LLM-judge + 回归门禁 + repeat-N 稳定性）。

对每个任务：隔离 workspace → 跑 agent（产 trace）→ 拷隐藏测试 → 跑测试（exit 0 = PASS）
→ 从 trace 抽 turns/tokens/延迟 →（可选）LLM-judge 给代码质量打分。
repeat-N：每任务跑 N 次，算 pass_fraction + 标记 flaky。
回归门禁：与上一次 baseline 比，pass_rate 掉超过阈值则报 REGRESSION（exit 1）。

用法:
    python eval/run_eval.py                      # 全部任务，单次
    python eval/run_eval.py T01 H04              # 只跑指定（名字前缀匹配）
    python eval/run_eval.py --repeat 3           # 每任务跑 3 次测稳定性
    python eval/run_eval.py --no-judge           # 跳过 LLM-judge
"""

import argparse
import json
import shutil
from collections import Counter
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import config
from agent.loop import run_task
from agent.mcp.runtime_config import UNSET, resolve_run_task_runtime_kwargs
from eval import judge
from obs.trace import get_sink

TASKS_DIR = Path(__file__).resolve().parent / "tasks"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REGRESSION_THRESHOLD = 5.0  # pass_rate 掉超过这么多个百分点判回归


# ──────────────────────────────────────────────
# 加载 / 工具
# ──────────────────────────────────────────────

def load_tasks(filters=None):
    tasks = []
    for d in sorted(TASKS_DIR.iterdir()):
        if not d.is_dir() or not (d / "task.json").exists():
            continue
        if filters and not any(f in d.name for f in filters):
            continue
        meta = json.loads((d / "task.json").read_text(encoding="utf-8"))
        meta.setdefault("category", "bugfix" if "fix" in meta["id"].lower() else "codegen")
        meta["_dir"] = d
        tasks.append(meta)
    return tasks


def _copy_into(src_dir: Path, dst: Path):
    if not src_dir.exists():
        return
    for f in src_dir.rglob("*"):
        if f.is_file():
            target = dst / f.relative_to(src_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)


def _snapshot_code(ws: Path, limit: int = 4000) -> str:
    """快照 workspace 里的 .py（agent 产出 + 被改的 setup），给 judge 看。"""
    parts = []
    for f in sorted(ws.rglob("*.py")):
        try:
            parts.append(f"# === {f.relative_to(ws)} ===\n"
                         + f.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    return "\n\n".join(parts)[:limit]


def _trace_metrics(events):
    turns = sum(1 for e in events if e["name"] == "agent.turn")
    llm = [e for e in events if e["name"] == "llm.call"]
    intok = sum(e.get("attributes", {}).get("gen_ai.usage.input_tokens", 0) for e in llm)
    outtok = sum(e.get("attributes", {}).get("gen_ai.usage.output_tokens", 0) for e in llm)
    run = next((e for e in events if e["name"] == "agent.run"), None)
    latency = round(run.get("duration_ms", 0) / 1000, 2) if run else 0
    peak_ctx = max((e.get("attributes", {}).get("context.tokens_sent", 0) for e in llm), default=0)
    return turns, intok, outtok, latency, peak_ctx


def classify_failure(r: dict, max_turns: int, files_changed=None):
    """把一次失败归类（failure taxonomy）。pass 返回 None。

    类别：超时 / 工具错误 / 没定位 / 上下文爆炸 / 没收敛 / 改错了。
    回答"它卡在哪"，而不只是"它几分"。
    """
    if r.get("passed"):
        return None
    err = (r.get("error") or "").lower()
    if "timeout" in err:
        return "超时"
    if err:
        return "工具错误"
    if files_changed is not None and len(files_changed) == 0:
        return "没定位(未改文件)"
    if (r.get("peak_context", 0) or 0) > 30000:
        return "上下文爆炸"
    if r.get("turns", 0) >= max_turns:
        return "没收敛(顶满轮次)"
    return "改错了(未过测试)"


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0


def _std(xs):
    if not xs:
        return 0.0
    m = _mean(xs)
    return round((sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5, 1)


# ──────────────────────────────────────────────
# 单次运行 + repeat-N 聚合
# ──────────────────────────────────────────────

def run_one(task: dict, capture_code: bool = False,
            mcp_run_task_kwargs: dict | None = None) -> dict:
    """跑一个任务一次并客观判分。capture_code=True 时快照代码（给 judge）。"""
    ws = Path(tempfile.mkdtemp(prefix="evalws_"))
    r = {"id": task["id"], "passed": False, "error": None, "verify_rc": None,
         "turns": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0,
         "peak_context": 0, "failure_reason": None, "_code": None}
    try:
        _copy_into(task["_dir"] / "setup", ws)
        mcp_run_task_kwargs = mcp_run_task_kwargs or resolve_run_task_runtime_kwargs()
        with config.using_workdir(ws):
            try:
                run_task(
                    task["prompt"],
                    max_turns=task.get("max_turns", 15),
                    **mcp_run_task_kwargs,
                )
            except Exception as e:
                r["error"] = f"agent: {type(e).__name__}: {e}"[:200]
            r["turns"], r["input_tokens"], r["output_tokens"], r["latency_s"], r["peak_context"] = \
                _trace_metrics(get_sink().events())
            if capture_code:
                r["_code"] = _snapshot_code(ws)

        _copy_into(task["_dir"] / "verify", ws)
        parts = task["verify_cmd"].split()
        if parts and parts[0] == "python":
            parts[0] = sys.executable
        try:
            proc = subprocess.run(parts, cwd=ws, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=60)
            r["verify_rc"] = proc.returncode
            r["passed"] = (proc.returncode == 0)
            if proc.returncode != 0:
                r["verify_out"] = (proc.stdout + proc.stderr)[-300:]
        except subprocess.TimeoutExpired:
            r["error"] = (r["error"] or "") + " | verify timeout"
    finally:
        shutil.rmtree(ws, ignore_errors=True)
    r["failure_reason"] = classify_failure(r, task.get("max_turns", 15))
    return r


def run_repeated(task: dict, n: int, do_judge: bool,
                 mcp_run_task_kwargs: dict | None = None) -> dict:
    """跑 n 次，聚合：pass_fraction / flaky / 稳定性 / judge（仅首次）。"""
    runs = [
        run_one(
            task,
            capture_code=(do_judge and i == 0),
            mcp_run_task_kwargs=mcp_run_task_kwargs,
        )
        for i in range(n)
    ]
    passes = sum(1 for r in runs if r["passed"])
    turns = [r["turns"] for r in runs]
    agg = {
        "id": task["id"],
        "runs": n,
        "passes": passes,
        "pass_fraction": round(passes / n, 2),
        "flaky": 0 < passes < n,
        "avg_turns": round(_mean(turns), 1),
        "std_turns": _std(turns),
        "avg_input_tokens": round(_mean([r["input_tokens"] for r in runs])),
        "avg_output_tokens": round(_mean([r["output_tokens"] for r in runs])),
        "avg_latency_s": round(_mean([r["latency_s"] for r in runs]), 2),
        "errors": [r["error"] for r in runs if r["error"]],
    }
    agg["category"] = task.get("category", "?")
    fr = [r["failure_reason"] for r in runs if not r["passed"] and r.get("failure_reason")]
    agg["failure_reason"] = Counter(fr).most_common(1)[0][0] if fr else None
    if do_judge:
        agg["judge"] = judge.judge_code(task["prompt"], runs[0].get("_code"))
    return agg


# ──────────────────────────────────────────────
# 回归门禁
# ──────────────────────────────────────────────

def regression_check(cur_pr: float, prev_pr: float, threshold: float = REGRESSION_THRESHOLD):
    """返回 (delta, is_regression)。pass_rate 掉超过 threshold 个百分点判回归。"""
    delta = round(cur_pr - prev_pr, 1)
    return delta, (delta < -threshold)


def _load_latest():
    latest = REPORTS_DIR / "LATEST"
    if not latest.exists():
        return None
    p = Path(latest.read_text(encoding="utf-8").strip())
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="评估 harness")
    parser.add_argument("filters", nargs="*", help="只跑名字含这些前缀的任务")
    parser.add_argument("--repeat", type=int, default=1, help="每任务跑几次（测稳定性）")
    parser.add_argument("--judge", action="store_true", help="强制开启 LLM-judge")
    parser.add_argument("--no-judge", action="store_true", help="强制关闭 LLM-judge")
    parser.add_argument("--enable-mcp", action="store_true", help="启用 MCP stdio 工具加载")
    parser.add_argument("--mcp-config", help="MCP 配置文件路径；提供路径即启用 MCP")
    args = parser.parse_args()

    if args.judge and args.no_judge:
        print("--judge 和 --no-judge 不能同时用")
        return
    do_judge = judge.judge_available() if not (args.judge or args.no_judge) else args.judge
    if args.judge and not judge.judge_available():
        print("[judge] 警告：JUDGE_MODEL_ID 未配置或与被测模型相同 → 跳过 judge")
        do_judge = False
    mcp_run_task_kwargs = resolve_run_task_runtime_kwargs(
        enable_mcp=True if args.enable_mcp else UNSET,
        mcp_config_path=args.mcp_config if args.mcp_config is not None else UNSET,
    )

    tasks = load_tasks(args.filters or None)
    if not tasks:
        print("没有匹配的任务。")
        return

    judge_info = f"on (judge={config.JUDGE_MODEL_ID})" if do_judge else "off"
    print(f"运行 {len(tasks)} 个任务 · model={config.MODEL_ID} · repeat={args.repeat} · judge={judge_info}\n")

    results = []
    for i, t in enumerate(tasks):
        print(f"[{i + 1}/{len(tasks)}] {t['id']} ...", end=" ", flush=True)
        agg = run_repeated(t, args.repeat, do_judge, mcp_run_task_kwargs=mcp_run_task_kwargs)
        results.append(agg)
        flag = " FLAKY" if agg["flaky"] else ""
        jd = ""
        if agg.get("judge") and agg["judge"].get("avg"):
            jd = f" judge={agg['judge']['avg']}"
        fr = f" [{agg['failure_reason']}]" if agg.get("failure_reason") else ""
        print(f"{agg['passes']}/{agg['runs']} pass{flag}  "
              f"({agg['avg_turns']}±{agg['std_turns']}t {agg['avg_input_tokens']}+{agg['avg_output_tokens']}tok){jd}{fr}")

    # 汇总
    total_runs = sum(r["runs"] for r in results)
    total_passes = sum(r["passes"] for r in results)
    pass_rate = round(total_passes / total_runs * 100, 1) if total_runs else 0
    flaky = [r["id"] for r in results if r["flaky"]]
    judged = [r["judge"]["avg"] for r in results if r.get("judge") and r["judge"].get("avg")]
    avg_judge = round(_mean(judged), 2) if judged else None

    # 回归门禁（与上一次比，再保存）
    prev = _load_latest()
    delta = is_reg = None
    if prev and "summary" in prev and "pass_rate" in prev["summary"]:
        delta, is_reg = regression_check(pass_rate, prev["summary"]["pass_rate"])

    summary = {
        "total_tasks": len(results),
        "total_runs": total_runs,
        "pass_rate": pass_rate,
        "flaky_count": len(flaky),
        "flaky_tasks": flaky,
        "avg_judge": avg_judge,
        "avg_turns": round(_mean([r["avg_turns"] for r in results]), 1),
        "avg_latency_s": round(_mean([r["avg_latency_s"] for r in results]), 2),
        "regression_delta": delta,
        "regression": bool(is_reg),
        "failure_breakdown": dict(Counter(r["failure_reason"] for r in results if r.get("failure_reason"))),
    }
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": config.MODEL_ID,
        "judge_model": config.JUDGE_MODEL_ID if do_judge else None,
        "repeat": args.repeat,
        "summary": summary,
        "results": results,
    }

    print("\n" + "=" * 56)
    print(f"  PASS RATE: {total_passes}/{total_runs} = {pass_rate}%")
    if avg_judge is not None:
        print(f"  AVG JUDGE: {avg_judge}/5.0")
    print(f"  FLAKY: {len(flaky)} {flaky if flaky else ''}")
    if delta is not None:
        arrow = "▲" if delta >= 0 else "▼"
        print(f"  vs 上次 baseline: {arrow} {delta:+}pp" + ("  ⚠️ REGRESSION" if is_reg else ""))
    print("=" * 56)
    fr_counts = Counter(r["failure_reason"] for r in results if r.get("failure_reason"))
    if fr_counts:
        print("  失败原因: " + " · ".join(f"{k}×{v}" for k, v in fr_counts.most_common()))
    cat = {}
    for r in results:
        d = cat.setdefault(r.get("category", "?"), [0, 0])
        d[1] += 1
        if r["passes"] == r["runs"] and r["runs"]:
            d[0] += 1
    print("  分类通过率: " + " · ".join(f"{c} {p}/{t}" for c, (p, t) in cat.items()))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / (report["ts"][:19].replace(":", "-") + ".json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / "LATEST").write_text(str(path), encoding="utf-8")
    print(f"\n报告: {path}")

    sys.exit(1 if is_reg else 0)


if __name__ == "__main__":
    main()
