"""从 trace JSONL 直接算 per-session 峰值窗口 + 累计 token 成本 + 到达里程碑数。
用法: python3 peak_cost.py <traces_dir> [<traces_dir2> ...]
峰值窗口 = max(agent.run.peak_context_tokens, max agent.turn.context_tokens)；
Σsent = Σ llm.call.context.tokens_sent（总成本口径）；tags = 自标里程碑集合。"""
import json, sys, glob, os, re
from collections import defaultdict

TAG = re.compile(r"agent-impl-([A-Za-z0-9][A-Za-z0-9_.\-]*)")

def analyze(tdir):
    rows = []
    for p in sorted(glob.glob(os.path.join(tdir, "run_*.jsonl"))):
        seen_run = False; peak = 0; sent = 0; inp = 0; outcome = ""
        sid = None; arm = None; tags = set()
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            n = ev.get("name"); a = ev.get("attributes", {}) or {}
            if n == "agent.run":
                seen_run = True
                outcome = str(a.get("outcome") or "")
                meta = a.get("run_metadata") or {}
                sid = meta.get("session_id") or meta.get("run_id")
                arm = meta.get("arm") or a.get("compact_strategy")
                peak = max(peak, int(a.get("peak_context_tokens") or 0))
            elif n == "agent.turn":
                peak = max(peak, int(a.get("context_tokens") or 0))
            elif n == "llm.call":
                sent += int(a.get("context.tokens_sent") or 0)
                inp += int(a.get("gen_ai.usage.input_tokens") or 0)
            elif isinstance(n, str) and n.startswith("tool."):
                for k in ("tool.command", "tool.arg"):
                    v = a.get(k)
                    if v:
                        m = TAG.search(str(v))
                        if m:
                            tags.add(m.group(1))
        if seen_run:
            rows.append([os.path.basename(p), arm, sid, outcome, peak, sent, inp, tags])

    # per-session 聚合（同 session 多 exec：峰值取 max、token 累加、tags 并集）
    agg = defaultdict(lambda: [0, 0, 0, set(), set()])  # peak, sent, inp, tags, outcomes
    for fn, arm, sid, outcome, peak, sent, inp, tags in rows:
        g = agg[(str(arm), str(sid))]
        g[0] = max(g[0], peak); g[1] += sent; g[2] += inp
        g[3] |= tags
        if outcome:
            g[4].add(outcome)

    print(f"\n========== {tdir} ==========")
    print(f"{'file':30} {'arm':9} {'outcome':16} {'peak':>9} {'Ssent':>13} {'tags'}")
    for fn, arm, sid, outcome, peak, sent, inp, tags in rows:
        print(f"{fn[:30]:30} {str(arm)[:9]:9} {outcome[:16]:16} {peak:>9,} {sent:>13,} {len(tags)}")
    print("  ---- per session (跨 exec 累计) ----")
    for (arm, sid), g in sorted(agg.items()):
        print(f"  arm={arm:9} sid={sid[:20]:20} peak={g[0]:>9,}  Ssent={g[1]:>13,}  Sinp={g[2]:>12,}  里程碑数={len(g[3])}  outcomes={sorted(g[4])}")

for d in sys.argv[1:]:
    analyze(d)
