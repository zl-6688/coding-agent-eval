"""overflow regime 硬实验 —— 压缩策略的真战场。

动机(reviewer Round 2 + 用户):舒适区(上下文装得下)里 none=1.0、naive==4layer 都是早产结论。
压缩的真价值在**触顶**:上下文 >> 预算时 none 装不下(enablement),且被迫激进压缩(20-30x)时
"怎么丢"才决定性分高下。本实验进入那个 regime:

- 构造**巨大**上下文(~150K+ token,远超预算),15 个硬 needle 按位置/类型散布(user 消息 + 大 tool 输出)。
- **固定预算 B(冻结在代码里,不准看结果再调)**,三条件:none(装不下→退场/失败)/ naive→B / 4layer→B。
- 客观判分(每 needle 一道 QA),repeat 抗抖动,真实 API。

读法:naive vs 4layer 在 B 下若**分裂** → 四层的复杂度终于挣到了钱(L3 落盘/L4 摘要保住了 naive 丢的);
若仍**逐 needle 相等** → 印证 reviewer M2:needle 摆放让两策略同构,是 case 设计问题不是 regime 问题。

用法:
    python eval/_archive/compact_eval/overflow_eval.py --repeat 1 --n-turns 150
    python eval/_archive/compact_eval/overflow_eval.py --repeat 3 --n-turns 400
"""

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import llm
from agent.context import compact
from eval._archive.compact_eval.answer_quality import _flatten, answer, score_answer

REPORTS = Path(__file__).resolve().parent / "reports"

# ── 冻结的硬 needle 集(15 个,位置/类型/难度散布;不准看结果再调)──
NEEDLES = [
    {"key": "XK-9920", "pos": 0.03, "where": "user", "text": "重要:部署用的 API key 是 XK-9920。",
     "q": "部署用的 API key 是什么?", "expect": ["XK-9920"]},
    {"key": "Alice", "pos": 0.05, "where": "user", "text": "项目负责人是 Alice。",
     "q": "项目负责人是谁?", "expect": ["Alice"]},
    {"key": "tabs", "pos": 0.07, "where": "user", "text": "我的偏好:代码缩进用 tabs,不要 spaces。",
     "q": "我对代码缩进的偏好是什么?", "expect": ["tab"]},
    {"key": "8443", "pos": 0.12, "where": "user", "text": "把服务监听端口从 80 改成 8443。",
     "q": "服务最终监听哪个端口?", "expect": ["8443"]},
    {"key": "v2.3.1", "pos": 0.18, "where": "user", "text": "目标发布版本是 v2.3.1。",
     "q": "目标发布版本号是多少?", "expect": ["v2.3.1"]},
    {"key": "Redis", "pos": 0.24, "where": "user", "text": "缓存层用 Redis。",
     "q": "缓存层用什么技术?", "expect": ["Redis"]},
    {"key": "Frankfurt", "pos": 0.31, "where": "user", "text": "数据中心选在 Frankfurt。",
     "q": "数据中心选在哪个城市?", "expect": ["Frankfurt"]},
    {"key": "42", "pos": 0.36, "where": "tool", "text": "关键:测试结果是 42 failed, 308 passed。",
     "q": "上一次测试有多少个用例失败?", "expect": ["42"]},
    {"key": "PostgreSQL", "pos": 0.43, "where": "user", "text": "更正:数据库不用 MySQL 了,改用 PostgreSQL。",
     "q": "最终决定用哪个数据库?", "expect": ["PostgreSQL"], "forbid": ["MySQL"]},
    {"key": "30", "pos": 0.50, "where": "user", "text": "请求超时统一设成 30 秒。",
     "q": "请求超时设成多少秒?", "expect": ["30"]},
    {"key": "DARK_MODE", "pos": 0.58, "where": "tool", "text": "新功能开关名是 DARK_MODE,默认关闭。",
     "q": "新功能的开关(feature flag)叫什么名字?", "expect": ["DARK_MODE"]},
    {"key": "bob@acme", "pos": 0.66, "where": "user", "text": "运维联系人邮箱 bob@acme.io。",
     "q": "运维联系人的邮箱是什么?", "expect": ["bob@acme"]},
    {"key": "eu-west-1", "pos": 0.74, "where": "tool", "text": "部署区域确定为 eu-west-1。",
     "q": "部署在哪个 region?", "expect": ["eu-west-1"]},
    {"key": "2026-09-01", "pos": 0.85, "where": "user", "text": "上线截止日期是 2026-09-01。",
     "q": "上线截止日期是哪天?", "expect": ["2026-09-01"]},
    {"key": "5000", "pos": 0.95, "where": "user", "text": "本月云成本预算上限 5000 美元。",
     "q": "本月云成本预算上限是多少美元?", "expect": ["5000"]},
]

FILLER = "普通代码行,与关键信息无关,仅用于占据上下文字节,模型无需记住这一行。\n"


def build_big(n_turns: int) -> list:
    """构造巨大 agent 对话:n_turns 个工具轮(填充)+ 15 个 needle 按位置散布。"""
    msgs = [{"role": "user", "content": "项目启动,我会陆续给你关键信息,请全部记住,后面要提问。"}]
    tools = ["read_file", "bash", "grep", "glob"]
    for i in range(n_turns):
        tid = f"t{i}"
        out = f"# src/mod_{i}.py\n" + FILLER * 12
        msgs.append({"role": "assistant",
                     "content": [{"type": "tool_use", "id": tid, "name": tools[i % 4], "input": {"path": f"src/mod_{i}.py"}}]})
        msgs.append({"role": "user",
                     "content": [{"type": "tool_result", "tool_use_id": tid, "content": out}]})

    for nd in NEEDLES:
        idx = max(1, int(nd["pos"] * len(msgs)))
        if nd.get("where") == "tool":
            tid = f"big_{nd['key'][:6]}"
            big = FILLER * 1200                      # >30K 字符 → L3 budget 会落盘;needle 埋在深处
            content = big + nd["text"] + "\n" + FILLER * 150
            msgs.insert(idx, {"role": "assistant",
                              "content": [{"type": "tool_use", "id": tid, "name": "read_file", "input": {"path": "big.log"}}]})
            msgs.insert(idx + 1, {"role": "user",
                                  "content": [{"type": "tool_result", "tool_use_id": tid, "content": content}]})
        else:
            msgs.insert(idx, {"role": "user", "content": nd["text"]})
    msgs.append({"role": "user", "content": "以上是全部背景,接下来我会基于它逐条提问。"})
    return msgs


def compress_to(cond: str, conv: list, B: int, cfg) -> list:
    """压到预算 B。none=原样(会装不下)/ naive=纯截断到 B / 4layer=四层管线 target=B。"""
    m = copy.deepcopy(conv)
    if cond == "none":
        return m
    if cond == "naive":
        return compact.compact_naive(m, B)
    if cond == "4layer":
        compact.reset_state()
        m = compact.budget(m, cfg)
        m = compact.snip(m, cfg)
        m = compact.micro(m, cfg)
        if compact.estimate(m) > B:
            m = compact.summarize(m, "", cfg)
        return m
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=8000, help="固定预算 B(token);冻结值")
    ap.add_argument("--n-turns", type=int, default=400, help="填充轮数,决定上下文有多大")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    cfg = compact.DEFAULT                   # realistic 阈值(非缩放),进真实 overflow regime
    conv = build_big(args.n_turns)
    full_tok = compact.estimate(conv)
    print(f"overflow 实验 · budget B={args.budget} tok · n_turns={args.n_turns} · repeat={args.repeat}")
    print(f"  完整上下文 ≈ {full_tok:,} token(≈{full_tok / args.budget:.0f}x 预算)→ none 必须装不下\n")

    # none:试一次,看是否 prompt_too_long(enablement 的物理证据)
    none_status = "?"
    try:
        a = answer(conv, NEEDLES[0]["q"])
        none_status = f"竟然答了(模型窗口够大):{a[:40]!r} —— 说明该模型窗口 > {full_tok:,} tok,需再加大 n_turns"
    except Exception as e:
        none_status = f"FAIL({type(e).__name__}: {str(e)[:60]}) —— 物理装不下,这就是 enablement"
    print(f"  none(全上下文直发):{none_status}\n")

    conds = ["naive", "4layer"]
    hits = {c: [0.0] * len(NEEDLES) for c in conds}
    sizes = {c: [] for c in conds}
    for rep in range(args.repeat):
        for c in conds:
            comp = compress_to(c, conv, args.budget, cfg)
            sizes[c].append(compact.estimate(comp))
            for i, nd in enumerate(NEEDLES):
                hits[c][i] += score_answer(answer(comp, nd["q"]), nd)
        print(f"  rep {rep + 1}/{args.repeat} done")

    print(f"\n  {'needle':12} {'pos':>5} {'where':>5} " + " ".join(f"{c:>8}" for c in conds))
    for i, nd in enumerate(NEEDLES):
        cells = " ".join(f"{round(hits[c][i] / args.repeat, 2):>8}" for c in conds)
        flag = "  ←分裂" if abs(hits["naive"][i] - hits["4layer"][i]) > 1e-9 else ""
        print(f"  {nd['key']:12} {nd['pos']:>5} {nd.get('where', 'user'):>5} {cells}{flag}")

    n = len(NEEDLES)
    print(f"\n  {'平均':12} {'':>5} {'':>5} "
          + " ".join(f"{round(sum(hits[c]) / args.repeat / n, 3):>8}" for c in conds))
    print(f"  压缩后规模(token):" + " ".join(f"{c}≈{round(sum(sizes[c]) / len(sizes[c]))}" for c in conds))
    split = sum(1 for i in range(n) if abs(hits["naive"][i] - hits["4layer"][i]) > 1e-9)
    print(f"\n  逐 needle 分裂数:{split}/{n}。split>0 ⇒ 四层在 overflow 下挣到了钱;split=0 ⇒ 印证 M2(策略同构/case 设计问题)。")

    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    path = REPORTS / ("overflow_" + ts[:19].replace(":", "-") + ".json")
    path.write_text(json.dumps({"ts": ts, "budget": args.budget, "n_turns": args.n_turns,
                                "repeat": args.repeat, "full_tok": full_tok, "none": none_status,
                                "avg": {c: round(sum(hits[c]) / args.repeat / n, 3) for c in conds},
                                "per_needle": {c: [round(hits[c][i] / args.repeat, 2) for i in range(n)] for c in conds},
                                "sizes": {c: round(sum(sizes[c]) / len(sizes[c])) for c in conds}},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {path}")


if __name__ == "__main__":
    main()
