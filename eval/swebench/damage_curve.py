"""长上下文 damage 曲线：在压缩区（~179K）测「不压 vs 我们的压缩 vs naive 截断」的 resolved + turns。

设计见 multi-issue-session-design.md §2.7-2.9 / D9-D10。核心：
  1. **前缀链跑一次、存快照**（D9）：5 前缀 → snapshots[N] = (实际token, messages)；reps 只变 target、
     历史冻结 → 干净测 target 方差。3/4/5 档复用同一链，不重跑前缀。
  2. **拐点档(N=5, ~195K 超 179K)三臂 × reps**：
       none       = 不压（>179K 可能超模型窗 → 报错本身=「必须压」的数据）
       compressed = 我们的 CC 压缩（compact_pipeline 压回 179K 内）
       truncated  = naive 截断到同样大小（地板对照）
  3. **damage 双维度**（D10）：resolved 掉幅(硬) + 每 issue turns(软)；target max_turns=200 排除截断污染。

用法（项目根，带代理）：
  python eval/swebench/damage_curve.py            # sphinx-10449, 5前缀, 拐点档三臂×reps=3
  python eval/swebench/damage_curve.py --reps=2   # 快验
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import config, loop                                     # noqa: E402
from agent.context import compact                                  # noqa: E402
from eval.swebench.session_run import SessionRunner, load_inst      # noqa: E402
from eval.swebench.variance_probe import score_harness             # noqa: E402

HERE = Path(__file__).resolve().parent
SNAP_DIR = config.TRACES_DIR / "snapshots"   # 前缀快照稳定存放（跨 run 复用/续跑，不随时间戳变）
SNAP_DIR.mkdir(parents=True, exist_ok=True)
T_ID = "sphinx-doc__sphinx-10449"       # 已验证稳定 4/4
# 10 个同 repo 不同子模块前缀（前 5 已有快照 175K；后 5 续堆 → 让 thinking+摘要本身也超 179K，
# 此时 micro 清完 tool_result 仍超阈值 → full 必触发，micro 的"推迟 full、降本"价值才可测）。
PREFIXES = ["sphinx-doc__sphinx-11510", "sphinx-doc__sphinx-10466", "sphinx-doc__sphinx-10323",
            "sphinx-doc__sphinx-10614", "sphinx-doc__sphinx-8638",
            "sphinx-doc__sphinx-7454", "sphinx-doc__sphinx-8621", "sphinx-doc__sphinx-9591",
            "sphinx-doc__sphinx-8551", "sphinx-doc__sphinx-8120"]
TREATMENTS = ("none", "compressed", "truncated")


def _jsonable_msgs(msgs: list) -> list:
    """assistant 历史里的 content 是 anthropic SDK 对象（非 dict）→ json.dumps 会崩。
    转成 dict（API 和我们的 compact 函数都接受 dict content block，round-trip 无损）。"""
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            nc = []
            for b in c:
                if isinstance(b, dict):
                    nc.append(b)
                elif hasattr(b, "model_dump"):
                    nc.append(b.model_dump())
                else:
                    nc.append({"type": "text", "text": str(getattr(b, "text", b))})
            out.append({**m, "content": nc})
        else:
            out.append(m)
    return out


def build_chain(T, root, instances: list[dict]) -> list:
    """跑前缀链，**每个前缀跑完就增量落盘**（网络抖动重跑可从已存前缀续，不白跑整条链）。
    返回 snapshots（messages 已转 dict）。snap_file 固定在 SNAP_DIR，跨 run 复用。"""
    snap_file = SNAP_DIR / f"prefix_snapshots_{T_ID.split('__')[-1]}.json"
    snaps = []
    if snap_file.exists():
        snaps = json.loads(snap_file.read_text(encoding="utf-8"))
        print(f"已有 {len(snaps)}/{len(PREFIXES)} 个前缀快照（{snap_file.name}），续跑剩余…", flush=True)
    if len(snaps) >= len(PREFIXES):
        print(f"前缀链已完整（{len(snaps)} 个，峰值 {snaps[-1]['tokens']} tok）→ 直接复用", flush=True)
        return snaps
    with SessionRunner(T, root / "chain") as sess:
        # 把已存快照的历史灌回 sess，从断点续（messages 用最后一个快照的）
        if snaps:
            sess.messages = list(snaps[-1]["messages"])
        for k in range(len(snaps), len(PREFIXES)):
            inst = load_inst(PREFIXES[k], instances)
            sc, trace = sess._run_issue(inst, f"prefix{k}_{PREFIXES[k]}", max_turns=60)
            tok = compact.estimate(sess.messages, loop.SYSTEM)
            t = sess._issue_turns(trace)
            snaps.append({"n": k + 1, "tokens": tok, "turns": t["turns"], "prefix_id": PREFIXES[k],
                          "messages": _jsonable_msgs(sess.messages)})
            snap_file.write_text(json.dumps(snaps, ensure_ascii=False), encoding="utf-8")   # 增量落盘
            print(f"   prefix{k+1} {PREFIXES[k]}: →{tok} tok ({tok/179000:.0%}/179K) turns={t['turns']}"
                  f"  [已存 {len(snaps)}/{len(PREFIXES)}]", flush=True)
    print(f"前缀链完整，快照 → {snap_file}", flush=True)
    return snaps


def run_target_arm(T, root, prefix_msgs, treatment, rep) -> dict:
    """新起容器跑一档一臂一次 target（隔离）。捕获超窗错误（= 必须压的数据）。"""
    base = root / f"{treatment}_r{rep}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        with SessionRunner(T, base) as sess:
            r = sess.run_target_from(copy.deepcopy(prefix_msgs), treatment=treatment)
    except Exception as e:
        return {"treatment": treatment, "rep": rep, "resolved": None, "patch_files": 0,
                "target_turns": None, "max_turns_reached": False,
                "error": f"{type(e).__name__}: {str(e)[:120]}", "score_note": "run_error"}
    r["rep"] = rep
    tag = f"dmg_{treatment}_r{rep}"
    pred = HERE / f"predictions_{tag}.jsonl"
    pred.write_text(json.dumps({"instance_id": T_ID, "model_name_or_path": tag,
                                "model_patch": r["patch"]}, ensure_ascii=False) + "\n", encoding="utf-8")
    if not r["patch"]:
        r["resolved"] = None
        r["score_note"] = "inconclusive_truncated" if r["max_turns_reached"] else "empty_patch"
    else:
        print(f"     {treatment} r{rep}: 补丁{r['patch_files']}文件 → 评分…", flush=True)
        r["resolved"] = score_harness(pred, tag, T_ID)
        r["score_note"] = "scored"
    return {k: v for k, v in r.items() if k != "patch"}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--chain-only", action="store_true")
    args = parser.parse_args(argv)
    if args.reps < 1:
        parser.error("--reps must be >= 1")
    from eval.swebench.data import DatasetError, SuiteManifestError, load_instances_for_run
    try:
        instances = load_instances_for_run(args.instances, args.suite)
        T = load_inst(T_ID, instances)
        for prefix_id in PREFIXES:
            load_inst(prefix_id, instances)
    except (DatasetError, SuiteManifestError, KeyError) as exc:
        parser.error(f"selected data cannot build the damage curve: {exc}")
    reps = args.reps
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = config.TRACES_DIR / f"damage_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    print(f"damage 曲线：T={T_ID}  拐点档 N={len(PREFIXES)}前缀  三臂×reps={reps}\n", flush=True)

    snaps = build_chain(T, root, instances)
    peak = snaps[-1]
    print(f"\n前缀链：N=1..{len(snaps)}，峰值 {peak['tokens']} tok ({peak['tokens']/179000:.0%}/179K)", flush=True)
    print(f"前缀 token 曲线: {[(s['n'], s['tokens']) for s in snaps]}", flush=True)
    print(f"前缀 turns: {[s['turns'] for s in snaps]}", flush=True)
    if args.chain_only:
        print("\n--chain-only：只建前缀链快照（供后续实验复用），不跑 target。", flush=True)
        return
    prefix_msgs = peak["messages"]      # 拐点档 = 最长那个快照

    results = []
    for treatment in TREATMENTS:
        for rep in range(reps):
            print(f"\n[{treatment} rep{rep}] 从 {peak['tokens']}tok 前缀起跑 target …", flush=True)
            r = run_target_arm(T, root, prefix_msgs, treatment, rep)
            results.append(r)
            print(f"   resolved={r['resolved']}  turns={r.get('target_turns')}"
                  f"  start_ctx={r.get('start_ctx')}  {r.get('error','')}", flush=True)

    # ── 汇总：三臂 resolved + turns（硬+软 damage）
    print(f"\n=== damage 拐点档汇总（N={len(PREFIXES)}前缀≈{peak['tokens']}tok, reps={reps}）===")
    print(f"{'treatment':12s}{'resolved':>16}{'命中':>7}{'turns':>16}{'start_ctx均':>12}")
    summ = {}
    for tr in TREATMENTS:
        rs = [r for r in results if r["treatment"] == tr]
        rv = [r["resolved"] for r in rs]
        hits = sum(1 for v in rv if v is True)
        scored = sum(1 for v in rv if v is not None)
        turns = [r["target_turns"] for r in rs if r["target_turns"] is not None]
        sctx = [r["start_ctx"] for r in rs if r.get("start_ctx")]
        print(f"{tr:12s}{str(rv):>16}{f'{hits}/{scored}':>7}{str(turns):>16}"
              f"{(sum(sctx)//len(sctx) if sctx else 0):>12}")
        summ[tr] = {"resolved": rv, "hits": hits, "scored": scored, "turns": turns}
    # 对照解读
    print(f"\n── 解读（拐点 ~{peak['tokens']}tok）──")
    n, c, t = summ["none"], summ["compressed"], summ["truncated"]
    print(f"  none(不压) {n['hits']}/{n['scored']}  | compressed(我们) {c['hits']}/{c['scored']}"
          f"  | truncated(截断) {t['hits']}/{t['scored']}")
    errs = [r for r in results if r.get("error")]
    if errs:
        print(f"  [!] none 臂报错 {len(errs)} 次（可能超窗 = 必须压的硬证据）: {errs[0].get('error')}")
    print(f"  → compressed−truncated = 智能压缩相对截断的增量；compressed−none = 压缩的代价/收益")

    out = root / "damage_results.json"
    out.write_text(json.dumps({"T": T_ID, "n_prefix": len(PREFIXES), "peak_tokens": peak["tokens"],
                               "prefix_turns": [s["turns"] for s in snaps], "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告 → {out}\ntrace → {root}")


if __name__ == "__main__":
    main()
