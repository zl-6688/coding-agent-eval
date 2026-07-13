"""压缩 NIAH 保留率 eval —— 四层压缩 vs naive，**逐层测 recall 定位薄弱层**。

构造 agent 风格长对话（tool_use/tool_result + needle 埋在 user 消息和大 tool 输出里）→
逐层应用 budget→snip→micro→summary，每层后测 recall → 看是哪一层、哪类 needle 丢的。
L0–L2 无 API；L4 summary 需 API（--with-summary）。

用法:
    python eval/_archive/compact_eval/run.py
    python eval/_archive/compact_eval/run.py --with-summary
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

from agent.context import compact

CASES = Path(__file__).resolve().parent / "cases.json"
REPORTS = Path(__file__).resolve().parent / "reports"
_POS = {"early": 0.1, "mid": 0.5, "late": 0.9}
_TOOLS = ["read_file", "bash", "grep", "glob"]


def _turn(i: int, tool: str, output: str) -> list:
    tid = f"t{i}"
    return [{"role": "assistant", "content": [{"type": "tool_use", "name": tool, "id": tid,
                                               "input": {"path": f"src/mod_{i}.py"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": output}]}]


def build_conversation(case: dict, n_turns: int = 28) -> list:
    msgs = [{"role": "user", "content": "开始项目，请记住后续我给的所有关键信息。"}]
    for i in range(n_turns):
        tool = _TOOLS[i % len(_TOOLS)]
        big = (i == 8)  # 一个超大输出，触发 L3 budget
        body = "普通代码行，无需特别记住的关键信息，这行用来占字节。\n" * (300 if big else 6)
        msgs += _turn(i, tool, f"# src/mod_{i}.py\n{body}")
        if i % 5 == 2:
            msgs.append({"role": "user", "content": f"（第 {i} 条说明）继续看下一个文件，这段不重要。"})

    for nd in case["needles"]:
        idx = 1 + int(_POS[nd["pos"]] * len(msgs))
        if nd.get("where") == "tool":
            tid = f"toolout_{idx}"   # 中性 id:不要把 needle 文本泄进 tool_use_id / L3 面包屑
            head = "日志填充行，用于占据字节，没有关键信息。\n" * 350   # > 6000 字符，needle 落在预览之外
            content = head + nd["text"] + "\n日志尾部填充。\n" * 40
            msgs.insert(idx, {"role": "assistant", "content": [{"type": "tool_use", "name": "read_file",
                                                                "id": tid, "input": {"path": "big.log"}}]})
            msgs.insert(idx + 1, {"role": "user", "content": [{"type": "tool_result",
                                                               "tool_use_id": tid, "content": content}]})
        else:
            msgs.insert(idx, {"role": "user", "content": nd["text"]})
    msgs.append({"role": "user", "content": "好，请基于以上所有信息继续。"})
    return msgs


def _all_text(messages) -> str:
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    parts.append(f"{b.get('content', '')} {b.get('text', '')} "
                                 f"{json.dumps(b.get('input', {}), ensure_ascii=False)}")
    return "\n".join(parts)


def _recall(messages, needles) -> float:
    """in-window recall:needle 是否还在活动上下文里。"""
    t = _all_text(messages)
    return round(sum(1 for nd in needles if nd["key"] in t) / len(needles), 2)


def _recall_retained(messages, needles) -> float:
    """retained recall:needle 在活动窗口 **或** 已被 L3 登记(未被物理删除)即算"留着了"。

    ⚠ 诚实说明(reviewer C2):这**不**度量可用性,只度量"没被物理删除"。eval 里 persist
    是内存登记、且重取链路从不触发(没有 agent loop),所以本指标偏循环(budget 存了→这里
    必查得到)。真正的"可恢复但 passive 不可用"由 answer_quality.py 的 4layer vs 4layer+read
    行为级度量(且那条也需改成按 ref 真读盘才非平凡)。
    in-window 与 retained 的差 = budget 登记的 tool 输出;retained 仍<1 = 用户口头事实(无登记)。
    """
    t = _all_text(messages) + "\n" + "\n".join(compact.persisted_texts())
    return round(sum(1 for nd in needles if nd["key"] in t) / len(needles), 2)


def run_case(case: dict, target: int, with_summary: bool) -> dict:
    compact.reset_state()                  # 清空落盘登记/熔断,避免跨 case 串扰
    cfg = compact.eval_profile()           # 缩放 profile;persist_dir=None → 仅内存登记,不写盘
    conv = build_conversation(case)
    needles = case["needles"]
    before = compact.estimate(conv)

    m = copy.deepcopy(conv)
    layers = [("L0_full", _recall(m, needles))]
    m = compact.budget(m, cfg); layers.append(("L3_budget", _recall(m, needles)))
    m = compact.snip(m, cfg);   layers.append(("L1_snip", _recall(m, needles)))
    m = compact.micro(m, cfg);  layers.append(("L2_micro", _recall(m, needles)))
    if with_summary and compact.estimate(m) > target:
        m = compact.summarize(m, "", cfg); layers.append(("L4_summary", _recall(m, needles)))

    ratio = round(before / max(1, compact.estimate(m)), 2)
    naive = compact.compact_naive(copy.deepcopy(conv), target)
    stale = any(nd.get("anti") and nd["anti"] in _all_text(m) for nd in needles)
    return {"id": case["id"], "dimension": case["dimension"], "before": before,
            "layers": dict(layers), "final_recall": layers[-1][1], "ratio": ratio,
            "retained_recall": _recall_retained(m, needles),
            "naive_recall": _recall(naive, needles), "stale_present": stale}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--with-summary", action="store_true", help="跑 L4 摘要（需 API key）")
    args = ap.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    layer_names = ["L0_full", "L3_budget", "L1_snip", "L2_micro"] + (["L4_summary"] if args.with_summary else [])
    print(f"压缩保留率 eval（逐层）· target={args.target} · L4={'on' if args.with_summary else 'off'}\n")

    rows = [run_case(c, args.target, args.with_summary) for c in cases]
    print(f"  {'case':14} {'dim':12} " + " ".join(f"{n[:9]:>9}" for n in layer_names)
          + f" {'retain':>6} {'naive':>7} {'ratio':>6}")
    for r in rows:
        cells = " ".join(f"{r['layers'].get(n, '—'):>9}" for n in layer_names)
        st = " STALE!" if r["stale_present"] else ""
        print(f"  {r['id']:14} {r['dimension']:12} {cells} {r['retained_recall']:>6} "
              f"{r['naive_recall']:>7} {r['ratio']:>5}x{st}")

    print("\n逐层平均 recall（看在哪一层掉得最多 = 薄弱层）:")
    prev = None
    for n in layer_names:
        avg = round(sum(r["layers"].get(n, 0) for r in rows) / len(rows), 3)
        drop = f"  (↓{round(prev - avg, 3)})" if prev is not None and avg < prev else ""
        print(f"  {n:12}: {avg}{drop}")
        prev = avg
    naive_avg = round(sum(r["naive_recall"] for r in rows) / len(rows), 3)
    print(f"  {'naive(地板)':12}: {naive_avg}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    path = REPORTS / (ts[:19].replace(":", "-") + ".json")
    path.write_text(json.dumps({"ts": ts, "target": args.target, "with_summary": args.with_summary,
                                "cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {path}")


if __name__ == "__main__":
    main()
