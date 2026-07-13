"""多 issue 长会话 v1 薄切片：SessionRunner（一容器全程）+ solo vs full-N gate。

公开运行约束见 eval/swebench/README.md。核心修正：**一个容器 / 一个 base_commit
（目标 T 的），全程不换** —— 否则前缀历史里记的文件原文在新容器找不到 = 上下文对 agent 撒谎，
掉幅归因到「环境割裂」而非「上下文变长」。本设计让环境与上下文全程一致：
  - 前缀 issue 在【同一】容器（T 的 base_commit）里跑 → 撑上下文，【不评分】
    （前缀不需要自己的镜像，只借 problem_statement 给 agent 派活）
  - 跑目标 T 前 `git checkout -- .` 复位 → 只延续 agent 的「记忆」，不让前缀的代码改动污染 T 补丁
  - 只评 T，在 T 自己的 base_commit 上，干净
gate：solo（无前缀）vs full-N（带前缀历史）的 resolved 掉幅 = 长上下文的代价（baseline damage）。

用法（项目根，带代理）：
  python -m eval.swebench.session_run --instances <dataset.json> [target_id] [prefix_id,prefix_id,...] [reps]
  # 默认 T=sympy-13031(已知 solo 可解)，prefix=sympy-13480,sympy-20428，reps=2
  python -m eval.swebench.session_run --instances <dataset.json> sympy__sympy-13031 sympy__sympy-13480 1
"""

import argparse
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

from agent import config, loop, tools                               # noqa: E402
from agent.context import compact                                   # noqa: E402
from agent.loop import run_task                                     # noqa: E402
from obs.trace import JsonlSink, set_sink                            # noqa: E402
from eval.swebench.run_swe import (                                 # noqa: E402
    docker_instance, build_task_indocker, container_diff, _dexec)
from eval.swebench.variance_probe import score_harness             # noqa: E402  (复用已验证的评分)

HERE = Path(__file__).resolve().parent
ARMS = ("solo", "full")


def load_inst(iid: str, instances: list[dict]) -> dict:
    for i in instances:
        if i["instance_id"] == iid:
            return i
    raise KeyError(iid)


# 新 issue 的醒目边界：真实多任务会话里，每个新任务都会明确说"现在做这一步"。
# 不加这个，agent 带着前缀（同模块）的历史会误判"这个修复我已经做过了"→ 不动手（实测 bug）。
_NEW_ISSUE_BANNER = (
    "\n\n━━━━━━━━━━ 新任务（与上面的工作无关）━━━━━━━━━━\n"
    "下面是一个**全新、独立**的 GitHub issue。**忽略你之前对其他 issue 的任何改动**——"
    "仓库已复位到干净状态，你之前的编辑都不在了。请**从头在当前代码上重新定位并修复**，"
    "不要假设任何修复已经存在。\n\n")


def _append_new_issue(messages: list, task: str) -> list:
    """把新 issue 接到历史末尾，**始终作为干净的新 user turn**。

    关键修复：前缀若打满轮次，末条是 user(tool_result)——直接 append user 会两条连续 user → API 400；
    旧实现合并进 tool_result → 新任务描述被埋进工具结果里，agent 看不清"现在要做新的"。
    解法：末条是 user 时，先插一条 assistant 占位（承接上文），再起干净的新 user turn。
    第一个 issue（无前序历史）不加 banner（上面没东西）。"""
    if not messages:
        return [{"role": "user", "content": task}]
    new_user = {"role": "user", "content": _NEW_ISSUE_BANNER + task}
    if messages[-1].get("role") == "user":
        bridge = {"role": "assistant", "content": "（上一个任务到此为止。）"}
        return messages + [bridge, new_user]
    return messages + [new_user]


class SessionRunner:
    """一个容器（T 的 base_commit）全程；messages 跨任务持久。"""

    def __init__(self, target_inst: dict, base_dir: Path):
        self.T = target_inst
        self.base = base_dir
        self.container = None
        self.messages = []

    def __enter__(self):
        self._cm = docker_instance(self.T["instance_id"])     # 用 T 的镜像/base_commit
        self.container = self._cm.__enter__()
        tools.set_executor(tools.DockerExecutor(self.container))
        return self

    def __exit__(self, *a):
        tools.reset_executor()
        self._cm.__exit__(*a)

    def _run_issue(self, inst: dict, label: str, max_turns: int):
        # 方案①：每个 issue 前都复位 repo 到干净 base_commit → repo 对所有 issue 恒定，
        #   唯一变量 = 累积的上下文（最干净的隔离）。配合 banner 诚实告知 agent「repo 干净、从头定位」，
        #   测的是「任务边界清晰后，~75K 上下文本身还伤不伤」。首个 issue 时是 no-op（容器本就干净）。
        _dexec(self.container, "cd /testbed && git checkout -- . 2>&1")
        task = build_task_indocker(inst, self.container)       # 前缀也用 in-docker 任务模板
        msgs = _append_new_issue(self.messages, task)
        start_ctx = compact.estimate(msgs, loop.SYSTEM)        # T 开始前的上下文 = 关键自变量
        trace = self.base / f"{label}.jsonl"
        set_sink(JsonlSink(trace))
        _, self.messages = run_task("(session)", max_turns=max_turns, trace=False,
                                    initial_messages=msgs,
                                    return_messages=True, meta={"session_label": label})
        return start_ctx, trace

    def _issue_turns(self, trace: Path) -> dict:
        ev = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines() if l.strip()]
        ra = next((e for e in ev if e["name"] == "agent.run"), {}).get("attributes", {})
        return {"turns": ra.get("turns"), "outcome": ra.get("outcome"),
                "max_turns_reached": ra.get("outcome") == "max_turns_reached"}

    def run_prefix(self, prefix_insts: list):
        for k, inst in enumerate(prefix_insts):
            sc, _ = self._run_issue(inst, f"prefix{k}_{inst['instance_id']}", max_turns=60)
            print(f"   prefix{k} {inst['instance_id']}: 起始{sc}→结束"
                  f"{compact.estimate(self.messages, loop.SYSTEM)} tok", flush=True)

    def run_prefix_chain(self, prefix_insts: list) -> list:
        """跑前缀链，每跑完一个存快照 → 一条链产 N=1..K 所有中间点（D9：确定性可复用）。
        返回 [{n, tokens, turns, messages(深拷贝)}]，供各档 target 复用、不重跑前缀。"""
        import copy as _copy
        snaps = []
        for k, inst in enumerate(prefix_insts):
            sc, trace = self._run_issue(inst, f"prefix{k}_{inst['instance_id']}", max_turns=60)
            tok = compact.estimate(self.messages, loop.SYSTEM)
            t = self._issue_turns(trace)
            snaps.append({"n": k + 1, "tokens": tok, "turns": t["turns"],
                          "prefix_id": inst["instance_id"], "messages": _copy.deepcopy(self.messages)})
            print(f"   prefix{k+1} {inst['instance_id']}: →{tok} tok ({tok/179000:.0%}/179K) "
                  f"turns={t['turns']}", flush=True)
        return snaps

    def run_target_from(self, prefix_messages: list, treatment: str = "none") -> dict:
        """从给定前缀快照（messages）起跑 target。treatment 决定 target 前怎么处理历史：
          none=不压（>179K 可能超窗）/ compressed=我们的 CC 压缩 / truncated=naive 截断。
        ★ max_turns=200（D10）：195K 上给足轮次排除截断污染；turns 仍记录当 damage 指标。"""
        self.messages = list(prefix_messages)      # 用快照历史
        pre_tok = compact.estimate(self.messages, loop.SYSTEM)
        if treatment == "compressed":
            self.messages = compact.compact_pipeline(self.messages, system=loop.SYSTEM,
                                                     target_tokens=compact.auto_threshold(compact.DEFAULT))
        elif treatment == "truncated":
            self.messages = compact.compact_naive(self.messages,
                                                  target_tokens=compact.auto_threshold(compact.DEFAULT),
                                                  system=loop.SYSTEM)
        start_ctx, trace = self._run_issue(self.T, f"target_{treatment}_{self.T['instance_id']}", max_turns=200)
        patch = container_diff(self.container)
        t = self._issue_turns(trace)
        return {"treatment": treatment, "prefix_tokens": pre_tok, "start_ctx": start_ctx,
                "patch": patch, "patch_files": patch.count("+++ b/"),
                "target_turns": t["turns"], "max_turns_reached": t["max_turns_reached"]}

    def run_target(self) -> dict:
        """旧接口（solo/full gate 用）：不压、max_turns=200。"""
        return run_target_compat(self)


def run_target_compat(sess) -> dict:
    start_ctx, trace = sess._run_issue(sess.T, f"target_{sess.T['instance_id']}", max_turns=200)
    patch = container_diff(sess.container)
    t = sess._issue_turns(trace)
    return {"start_ctx": start_ctx, "end_ctx": compact.estimate(sess.messages, loop.SYSTEM),
            "patch": patch, "patch_files": patch.count("+++ b/"),
            "target_turns": t["turns"], "target_outcome": t["outcome"],
            "max_turns_reached": t["max_turns_reached"]}


# 各 T 的前缀（同 repo、刻意不同子模块 → 避开「前缀直接关于 T 代码」的串味；测纯上下文长度效应）
PREFIX_MAP = {
    "sphinx-doc__sphinx-10449": ["sphinx-doc__sphinx-10435", "sphinx-doc__sphinx-10466"],
    "django__django-11066": ["django__django-10880", "django__django-10914"],
    "astropy__astropy-12907": ["astropy__astropy-13236", "astropy__astropy-13453"],
}


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
    parser.add_argument("target_id", nargs="?", default="sphinx-doc__sphinx-10449")
    parser.add_argument("prefix_ids", nargs="?", default="")
    parser.add_argument("reps", nargs="?", type=int, default=3)
    args = parser.parse_args(argv)

    from eval.swebench.data import (  # noqa: WPS433
        DatasetError,
        SuiteManifestError,
        load_instances_for_run,
    )

    try:
        instances = load_instances_for_run(args.instances, args.suite)
    except (DatasetError, SuiteManifestError) as exc:
        parser.error(str(exc))
    target_id = args.target_id
    prefix_ids = args.prefix_ids.split(",") if args.prefix_ids else PREFIX_MAP.get(target_id)
    reps = args.reps
    if reps < 1:
        parser.error("reps must be >= 1")
    if not prefix_ids:
        parser.error(f"no prefix IDs configured for {target_id!r}; pass a comma-separated list")
    try:
        T = load_inst(target_id, instances)
        prefix = [load_inst(i, instances) for i in prefix_ids]
    except KeyError as exc:
        parser.error(f"instance_id {exc.args[0]!r} is not present in the selected data")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = config.TRACES_DIR / f"session_{stamp}"
    short = target_id.split("__")[-1]
    print(f"目标 T={target_id}  前缀={prefix_ids}（同repo不同子模块）  reps={reps}", flush=True)

    results = []
    for rep in range(reps):
        for arm in ARMS:
            base = root / f"{arm}_r{rep}"
            base.mkdir(parents=True, exist_ok=True)
            print(f"\n[{arm} rep{rep}] 起容器(T 的 base_commit) …", flush=True)
            with SessionRunner(T, base) as sess:
                if arm == "full":
                    sess.run_prefix(prefix)
                r = sess.run_target()
            r.update(arm=arm, rep=rep, instance_id=target_id)
            # 评分：每 (arm,rep) 单独 predictions + harness（防归因污染：打满轮次的空补丁标 inconclusive）
            tag = f"sess_{short}_{arm}_r{rep}"
            pred = HERE / f"predictions_{tag}.jsonl"
            pred.write_text(json.dumps({"instance_id": target_id, "model_name_or_path": tag,
                                        "model_patch": r["patch"]}, ensure_ascii=False) + "\n",
                            encoding="utf-8")
            if not r["patch"]:
                r["resolved"] = None
                r["score_note"] = "inconclusive_truncated" if r["max_turns_reached"] else "empty_patch"
            else:
                print(f"   补丁文件数={r['patch_files']} → harness 评分 …", flush=True)
                r["resolved"] = score_harness(pred, tag, target_id)
                r["score_note"] = "scored"
            results.append(r)
            print(f"[{arm} rep{rep}] T起始ctx={r['start_ctx']}  补丁={r['patch_files']}文件"
                  f"  轮次={r.get('target_turns')}{'(★截断)' if r['max_turns_reached'] else ''}"
                  f"  resolved={r['resolved']}", flush=True)

    # ── gate 汇总：配对 solo vs full 的 resolved 率（同 T，掉幅=长上下文代价）
    def rate(arm):
        vals = [r["resolved"] for r in results if r["arm"] == arm and r["resolved"] is not None]
        return sum(1 for v in vals if v), len(vals)
    sh, sn = rate("solo")
    fh, fn = rate("full")
    sctx = [r["start_ctx"] for r in results if r["arm"] == "solo"]
    fctx = [r["start_ctx"] for r in results if r["arm"] == "full"]
    print(f"\n=== solo vs full GATE（T={target_id}, reps={reps}）===")
    print(f"{'arm':6s}{'resolved':>16}{'命中':>7}{'T起始ctx均':>12}{'截断/空':>9}")
    for arm in ARMS:
        rs = [r for r in results if r["arm"] == arm]
        vals = [r["resolved"] for r in rs]
        bad = sum(1 for r in rs if r["resolved"] is None)
        ctx = sum(r["start_ctx"] for r in rs) // max(1, len(rs))
        h, n = rate(arm)
        print(f"{arm:6s}{str(vals):>16}{f'{h}/{n}':>7}{ctx:>12}{bad:>9}")
    print(f"\nsolo {sh}/{sn}  full {fh}/{fn}   上下文 solo≈{sum(sctx)//max(1,len(sctx))} → full≈{sum(fctx)//max(1,len(fctx))} tok")
    if sn and fn:
        drop = sh / sn - fh / fn
        if drop > 0:
            print(f"  → full 掉了 {drop:.0%}（baseline damage 信号；样本小，看原始值+下一步扩 reps/T）")
        elif drop < 0:
            print(f"  → full 反而高 {-drop:.0%}（localization 噪声主导？n 太小）")
        else:
            print(f"  → 两臂持平（此 T 此链长无可见掉幅；可能 ~Xk 不施压 → 上长链）")
    rep_json = root / "session_results.json"
    rep_json.write_text(json.dumps([{k: v for k, v in r.items() if k != 'patch'}
                                     for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 → {rep_json}\ntrace → {root}")


if __name__ == "__main__":
    main()
