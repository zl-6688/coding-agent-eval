"""answer-quality 轴 —— 第三根轴:在不同压缩条件下,让模型回答"必须用到 needle 才能答对"
的问题,**客观判分**(答案里有没有那个事实)。这是唯一能看见"去噪/倒 U"的轴:string-match
recall 只看"事实在不在",这根轴看"模型能不能用上"。

三个条件:none(不压,全上下文)/ naive(纯截断地板)/ 4layer(我们的四层)。
- 4layer ≥ none ⇒ 压缩没伤,甚至去噪获益;4layer < none ⇒ 这个规模下压缩在丢信息。
- 加大 --n-turns(提高噪声)看 4layer 是否反超 none = sweet spot 出现 = 倒 U。

为什么客观判分而非 LLM-judge:事实型 needle 有唯一正确答案,客观 string-check 比 judge
更严谨、零成本、无自评偏差——这正是 NIAH 的标准做法。judge 留给开放式答案(本模块用不上)。

为什么把上下文 flatten 成文本再问:避免 Anthropic 对 tool_use 块要求 tools 参数的约束,
且度量的是"事实可用性"(模型最终看到的就是文本)。lost-in-the-middle 对长文本同样成立。

用法:
    python eval/_archive/compact_eval/answer_quality.py --stub
    python eval/_archive/compact_eval/answer_quality.py
    python eval/_archive/compact_eval/answer_quality.py --n-turns 80 --repeat 3
"""

import argparse
import copy
import json
import re
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
from eval._archive.compact_eval.run import build_conversation

CASES = Path(__file__).resolve().parent / "cases.json"
REPORTS = Path(__file__).resolve().parent / "reports"
CONDITIONS = ["none", "naive", "4layer", "4layer+read"]


def _flatten(messages) -> str:
    """把(可能已压缩的)消息列表转成纯文本对话记录。"""
    lines = []
    for m in messages:
        role = m.get("role", "?")
        c = m.get("content")
        if isinstance(c, str):
            lines.append(f"{role}: {c}")
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    lines.append(f"{role}: {getattr(b, 'text', str(b))}")
                    continue
                t = b.get("type")
                if t == "text":
                    lines.append(f"{role}: {b.get('text', '')}")
                elif t == "tool_use":
                    lines.append(f"{role}: [调用 {b.get('name')} {json.dumps(b.get('input', {}), ensure_ascii=False)}]")
                elif t == "tool_result":
                    lines.append(f"tool: {b.get('content', '')}")
    return "\n".join(lines)


def _four_layer(m: list, cfg, target: int) -> list:
    compact.reset_state()
    m = compact.budget(m, cfg)
    m = compact.snip(m, cfg)
    m = compact.micro(m, cfg)
    if compact.estimate(m) > target:
        m = compact.summarize(m, "", cfg)
    return m


def compress(condition: str, conv: list, cfg, target: int) -> list:
    """按条件压缩对话。
    none=原样 / naive=纯截断地板 / 4layer=完整四层(passive)/
    4layer+read=四层后**模拟 agent re-read 落盘内容**(用来实证"可恢复≠可用":
                4layer 答不出的落盘 needle,re-read 后能答出 → 差额=recoverability 的使用代价)。
    """
    m = copy.deepcopy(conv)
    if condition == "none":
        return m
    if condition == "naive":
        return compact.compact_naive(m, target)
    if condition == "4layer":
        return _four_layer(m, cfg, target)
    if condition == "4layer+read":
        m = _four_layer(m, cfg, target)
        recovered = compact.persisted_texts()   # budget 本次落盘的内容(模拟 agent 看到面包屑后重取)
        if recovered:
            m = m + [{"role": "user", "content": "[agent re-read 了落盘内容]\n" + "\n".join(recovered)}]
        return m
    raise ValueError(f"未知条件 {condition}")


def answer(messages: list, question: str) -> str:
    """让**候选模型**仅依据(压缩后的)上下文回答问题。"""
    transcript = _flatten(messages)
    prompt = (f"以下是之前的对话记录：\n\n{transcript}\n\n"
              f"请仅根据上面的记录回答问题，只给答案、不要解释；记录里没有就回答“不知道”。\n"
              f"问题：{question}")
    resp = llm.chat([{"role": "user", "content": prompt}],
                    system="你根据提供的对话记录如实回答。", max_tokens=200)
    return "".join(getattr(b, "text", "") for b in resp.content
                   if getattr(b, "type", None) == "text").strip()


def stub_extract(transcript: str, qa: dict) -> str:
    """离线 oracle:完美阅读器,返回 transcript 中确实出现的 expect 键。

    用途:不烧 API 验证整条 harness(compress→flatten→answer→score)接线正确,
    且结果应与 recall 一致(事实在上下文里 oracle 就答得出,不在就答不出)。
    """
    return " ".join(e for e in qa.get("expect", []) if e in transcript)


def _hit(answer: str, key: str) -> bool:
    """命中判定:纯 ascii 短 token 用词边界(允许复数 s),避免子串误判
    ('42'≠'6420'、'tab'≠'table');含特殊字符的 key(v2.3.1 / XK-9920)走子串。"""
    a, k = answer.lower(), key.lower()
    if re.fullmatch(r"[a-z0-9]+", k):
        return re.search(r"\b" + re.escape(k) + r"s?\b", a) is not None
    return k in a


def score_answer(ans: str, qa: dict) -> float:
    """客观判分:答案命中 expect 键的比例;命中任一 forbid 键直接判 0(如保留了过期事实)。"""
    a = ans or ""
    if any(_hit(a, f) for f in qa.get("forbid", [])):
        return 0.0
    exp = qa.get("expect", [])
    return round(sum(1 for e in exp if _hit(a, e)) / len(exp), 2) if exp else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1500)
    ap.add_argument("--n-turns", type=int, default=28, help="对话轮数,越大噪声越大")
    ap.add_argument("--repeat", type=int, default=1, help="每个(case×条件)重复次数,抗模型抖动")
    ap.add_argument("--stub", action="store_true", help="离线 oracle,验证接线(不烧 API)")
    args = ap.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    cfg = compact.eval_profile()
    mode = "STUB(离线 oracle)" if args.stub else "LIVE(候选模型)"
    print(f"answer-quality 轴 · target={args.target} · n_turns={args.n_turns} · "
          f"repeat={args.repeat} · {mode}\n")
    print(f"  {'case':14} " + " ".join(f"{c:>11}" for c in CONDITIONS))

    agg = {c: 0.0 for c in CONDITIONS}
    rows = []
    for case in cases:
        qa = case["qa"]
        cells = {}
        for cond in CONDITIONS:
            s = 0.0
            for _ in range(args.repeat):
                conv = build_conversation(case, args.n_turns)
                comp = compress(cond, conv, cfg, args.target)
                ans = stub_extract(_flatten(comp), qa) if args.stub else answer(comp, qa["q"])
                s += score_answer(ans, qa)
            cells[cond] = round(s / args.repeat, 2)
            agg[cond] += cells[cond]
        rows.append({"id": case["id"], "scores": cells})
        print(f"  {case['id']:14} " + " ".join(f"{cells[c]:>11}" for c in CONDITIONS))

    n = len(cases)
    avg = {c: round(agg[c] / n, 3) for c in CONDITIONS}
    print(f"\n  {'平均':12} " + " ".join(f"{avg[c]:>11}" for c in CONDITIONS))
    print("\n  读法:4layer ≥ none ⇒ 压缩没伤(甚至去噪获益);4layer < none ⇒ 此规模下压缩在丢信息。")
    print("       加大 --n-turns 提高噪声,看 4layer 是否反超 none = 倒 U 的 sweet spot 出现。")
    print("  注意:answer 轴比 recoverable recall 更严 —— 落盘的 tool 输出若不主动 re-read,"
          "passive 问答里仍答不出(可恢复≠免费可用,这正是'agent 得知道去重取'的代价)。")

    if not args.stub:
        REPORTS.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        path = REPORTS / ("answerq_" + ts[:19].replace(":", "-") + ".json")
        path.write_text(json.dumps({"ts": ts, "target": args.target, "n_turns": args.n_turns,
                                    "repeat": args.repeat, "avg": avg, "rows": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告: {path}")


if __name__ == "__main__":
    main()
