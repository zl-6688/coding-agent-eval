"""会话上下文可达性探针：N 个前缀能把上下文堆到多少 token？

为什么：纵向拉长链找 damage 拐点前，先确认前缀真能堆进压缩区（→179K）。
2 前缀实测才 ~50K，8 前缀理论 ~170K——但 agent 解题深浅不一，必须实测，
否则铺全曲线跑几小时才发现到不了压缩区 = 白跑。

只跑前缀、记每加一个前缀后的累积 token，不评分、不跑 target。快。

用法（项目根，带代理）：
  python eval/swebench/session_reach_probe.py            # sphinx-10449 容器里堆 8 个前缀
"""

import argparse
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

T_ID = "sphinx-doc__sphinx-10449"
# 8 个同 repo 不同子模块前缀（ps 长的排前面，撑得快）
PREFIXES = ["sphinx-doc__sphinx-11510", "sphinx-doc__sphinx-10466", "sphinx-doc__sphinx-10323",
            "sphinx-doc__sphinx-10614", "sphinx-doc__sphinx-8638", "sphinx-doc__sphinx-7454",
            "sphinx-doc__sphinx-10435", "sphinx-doc__sphinx-8621"]


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--suite", type=Path)
    args = parser.parse_args(argv)
    from eval.swebench.data import DatasetError, SuiteManifestError, load_instances_for_run
    try:
        instances = load_instances_for_run(args.instances, args.suite)
        T = load_inst(T_ID, instances)
        prefixes = [load_inst(prefix_id, instances) for prefix_id in PREFIXES]
    except (DatasetError, SuiteManifestError, KeyError) as exc:
        parser.error(f"selected data cannot build the reach probe: {exc}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = config.TRACES_DIR / f"reach_{stamp}"
    base.mkdir(parents=True, exist_ok=True)
    print(f"会话可达性探针：T={T_ID} 容器里堆 {len(PREFIXES)} 前缀，看累积 token", flush=True)
    print(f"压缩触发线 ≈179K；想找的是哪个 N 把上下文推进压缩区\n", flush=True)

    curve = []
    with SessionRunner(T, base) as sess:
        for k, (pid, prefix) in enumerate(zip(PREFIXES, prefixes)):
            sess.run_prefix([prefix])      # 一次加一个，复用现有逐前缀打印
            ctx = compact.estimate(sess.messages, loop.SYSTEM)
            curve.append((k + 1, pid, ctx))
            print(f"  → 累积 {k+1} 前缀: {ctx} tok  ({ctx/179000:.0%} of 179K 压缩线)", flush=True)

    print(f"\n=== 上下文增长曲线 ===")
    print(f"{'N前缀':>5}{'累积tok':>10}{'/179K':>8}")
    for n, _, ctx in curve:
        print(f"{n:>5}{ctx:>10}{ctx/179000:>7.0%}")
    peak = curve[-1][2] if curve else 0
    print(f"\n峰值 {peak} tok。", end="")
    if peak >= 179000:
        print(" ✓ 能进压缩区 → 可铺全曲线（含压缩臂）。")
    elif peak >= 120000:
        print(" 接近压缩区 → 加 1-2 前缀或挑更长 ps 的可到 179K。")
    else:
        print(f" 远未到压缩区（{peak/179000:.0%}）→ 单 issue 撑得太慢，找 damage 拐点需更多前缀或换策略。")


if __name__ == "__main__":
    main()
