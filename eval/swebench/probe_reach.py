"""可达性探针：难实例在 no-compression / 高 turn 上限下，上下文能自然涨到多高？

回答 resume 评测的前置问题（DECISIONS §3 / journey §7.7 的"自然长轨迹"方案）：
  正式实验要在"agent 真实峰值（目标 80-100K）"处三臂分叉。但 §7.6 实测 max_turns=30 时
  单 issue 峰值才 ~50K。本探针把压缩关掉、max_turns 拉到 150，量两件事：
    ① peak_context —— 自然峰值到底够不够阈值
    ② A1 落盘触发次数 —— 峰值若低，是不是大结果落盘（tool_store）已经替我们管住了上下文
  （若 A1 是主因 → "压缩对单 issue 非因素"再添一证；要逼高需另加杠杆，如关 A1。）

用法（项目根，带代理）：
  python -m eval.swebench.probe_reach --instances <dataset.json> matplotlib__matplotlib-14623 [max_turns]
  python -m eval.swebench.probe_reach --instances <dataset.json> --all
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import config, tools                          # noqa: E402
from agent.loop import run_task                          # noqa: E402
from obs.trace import JsonlSink, set_sink                 # noqa: E402
from eval.swebench.run_swe import docker_instance, build_task_indocker  # noqa: E402

# 本地大 repo 候选（探索量大，最可能涨高）；按 repo 体量/难度挑
DEFAULT_CANDIDATES = ["matplotlib__matplotlib-14623", "pydata__xarray-3993",
                      "scikit-learn__scikit-learn-12585"]


def _load(instance_id: str, instances: list[dict]) -> dict:
    for i in instances:
        if i["instance_id"] == instance_id:
            return i
    raise KeyError(instance_id)


def probe_one(instance_id: str, max_turns: int = 150, *, instance: dict | None = None) -> dict:
    if instance is None:
        raise ValueError("probe_one requires a hydrated external SWE-bench instance row")
    inst = instance
    stamp = time.strftime("%H%M%S")
    trace = config.TRACES_DIR / f"probe_reach_{instance_id}_{stamp}.jsonl"
    set_sink(JsonlSink(trace))
    print(f"\n=== 探针 {instance_id}  max_turns={max_turns}  compact=none ===", flush=True)
    t0 = time.time()
    try:
        with docker_instance(instance_id) as container:
            tools.set_executor(tools.DockerExecutor(container))
            try:
                task = build_task_indocker(inst, container)
                # trace=False：用上面 set_sink 的 probe trace（run_task 默认 trace=True 会另设 sink 覆盖它）
                run_task(task, max_turns=max_turns,
                         trace=False, meta={"probe": "reach"})
            finally:
                tools.reset_executor()
    except Exception as e:
        return {"instance_id": instance_id, "error": f"{type(e).__name__}: {str(e)[:200]}"}

    # 从 trace 聚合：峰值上下文（loop 记在 agent.run）+ A1 落盘次数 + 轮次/收尾
    events = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines() if l.strip()]
    run = next((e for e in events if e["name"] == "agent.run"), {})
    a = run.get("attributes", {})
    persisted = sum(1 for e in events
                    if e.get("attributes", {}).get("tool.persisted"))
    # 每轮发模型前的 context（agent.turn 的 context_tokens）→ 看增长曲线峰值
    turn_ctx = [e["attributes"].get("context_tokens", 0) for e in events if e["name"] == "agent.turn"]
    return {"instance_id": instance_id,
            "peak_context": a.get("peak_context_tokens", max(turn_ctx, default=0)),
            "turns": a.get("turns", len(turn_ctx)),
            "outcome": a.get("outcome", "?"),
            "a1_persisted": persisted,           # A1 落盘触发次数（压着上下文的嫌疑）
            "n_tool_errors": a.get("n_tool_errors", 0),
            "secs": round(time.time() - t0), "trace": str(trace)}


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
    parser.add_argument("--all", action="store_true")
    parser.add_argument("instance_id", nargs="?")
    parser.add_argument("max_turns", nargs="?", type=int, default=150)
    args = parser.parse_args(argv)
    if args.max_turns < 1:
        parser.error("max_turns must be >= 1")

    from eval.swebench.data import (  # noqa: WPS433
        DatasetError,
        SuiteManifestError,
        load_instances_for_run,
    )

    try:
        instances = load_instances_for_run(args.instances, args.suite)
    except (DatasetError, SuiteManifestError) as exc:
        parser.error(str(exc))
    ids = DEFAULT_CANDIDATES if args.all else ([args.instance_id] if args.instance_id else DEFAULT_CANDIDATES[:1])
    by_id = {row["instance_id"]: row for row in instances}
    missing = [instance_id for instance_id in ids if instance_id not in by_id]
    if missing:
        parser.error(f"instance IDs are not present in the selected data: {', '.join(missing)}")
    mt = args.max_turns
    out = []
    for iid in ids:
        r = probe_one(iid, mt, instance=by_id[iid])
        out.append(r)
        print(f"  → {r}", flush=True)
    print("\n=== 汇总（峰值 vs 80-100K 阈值；A1 落盘多=它在压着）===")
    for r in out:
        if "error" in r:
            print(f"  {r['instance_id']}: ERROR {r['error']}")
        else:
            print(f"  {r['instance_id']}: peak={r['peak_context']} tok  turns={r['turns']}  "
                  f"A1落盘={r['a1_persisted']}  {r['outcome']}  ({r['secs']}s)")
    rep = config.TRACES_DIR / f"probe_reach_{time.strftime('%Y%m%d_%H%M%S')}.json"
    rep.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告 → {rep}")


if __name__ == "__main__":
    main()
