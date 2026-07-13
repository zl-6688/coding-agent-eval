"""结构合法性 property test —— 任意层组合后,消息列表必须仍是 API-valid。

压缩最阴险的 bug 不是"丢信息",是"产出一个 API 会 400 的消息列表"(孤儿 tool_result /
悬空 tool_use / 不以 user 开头)→ 生产里直接崩。这里用随机对话 × 随机层组合做 property
test 兜住三条硬不变量。纯离线(mock 掉 L4 的 LLM 调用),无 API。

    python eval/_archive/compact_eval/test_validity.py
"""

import copy
import itertools
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent.context import compact


# ── 把 L4 的 LLM 调用换成确定性桩(保留所有事实的理想摘要)──
class _Blk:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Resp:
    def __init__(self, text):
        self.content = [_Blk(text)]


def _fake_chat(messages, system="", max_tokens=0):
    return _Resp("<analysis>略</analysis><summary>摘要:逐条保留了所有 KEY 与用户事实。</summary>")


compact.llm.chat = _fake_chat


def _blocks(m):
    c = m.get("content")
    return c if isinstance(c, list) else []


# ──────────────────────────────────────────────
# 三条硬不变量(违反任意一条 → Anthropic API 400)
# ──────────────────────────────────────────────

def assert_valid(messages, ctx=""):
    assert isinstance(messages, list) and messages, f"{ctx}: 空消息列表"
    assert messages[0].get("role") == "user", f"{ctx}: 必须以 user 开头,实际 {messages[0].get('role')}"
    seen_tool_use = set()
    pending = set()  # 已出现但还没收到 result 的 tool_use id
    for m in messages:
        for b in _blocks(m):
            t = b.get("type") if isinstance(b, dict) else None
            if t == "tool_use":
                bid = b.get("id")
                seen_tool_use.add(bid)
                pending.add(bid)
            elif t == "tool_result":
                tuid = b.get("tool_use_id")
                assert tuid in seen_tool_use, f"{ctx}: 孤儿 tool_result(无对应 tool_use){tuid}"
                pending.discard(tuid)
    assert not pending, f"{ctx}: 悬空 tool_use(无对应 tool_result){pending}"


# ──────────────────────────────────────────────
# 随机生成 well-formed agent 对话(成对相邻、以 user 收尾)
# ──────────────────────────────────────────────

def gen_conversation(rng, n_turns):
    msgs = [{"role": "user", "content": "开始任务,请记住关键信息。"}]
    for i in range(n_turns):
        r = rng.random()
        if r < 0.6:  # 一对 tool_use / tool_result
            tid = f"t{i}_{rng.randint(0, 99999)}"
            tool = rng.choice(["bash", "read_file", "grep", "glob", "write_file", "edit_file"])
            big = rng.random() < 0.15
            out = ("行内容\n" * rng.randint(8, 30)) if not big else ("大输出填充\n" * rng.randint(400, 900))
            msgs.append({"role": "assistant",
                         "content": [{"type": "tool_use", "id": tid, "name": tool, "input": {"p": f"f{i}.py"}}]})
            msgs.append({"role": "user",
                         "content": [{"type": "tool_result", "tool_use_id": tid, "content": out}]})
        elif r < 0.8:  # assistant 文本 + user 文本
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"思考第 {i} 步"}]})
            msgs.append({"role": "user", "content": f"用户补充 {i}:记住 KEY-{i}。"})
        else:  # 独立 user 备注
            msgs.append({"role": "user", "content": f"备注 {i}:这段不太重要。"})
    msgs.append({"role": "user", "content": "基于以上所有信息继续。"})
    return msgs


# ──────────────────────────────────────────────
# 主流程:随机对话 × 随机层组合,全部断言合法
# ──────────────────────────────────────────────

LAYERS = {
    "budget": lambda m, c: compact.budget(m, c),
    "snip": lambda m, c: compact.snip(m, c),
    "micro": lambda m, c: compact.micro(m, c),
}


def main():
    rng = random.Random(20260620)
    cfg = compact.eval_profile()  # persist_dir=None → 不写盘
    trials = 0

    # 1) 随机对话 × 随机层子集/顺序(含空集、全排列)
    combos = [()]
    for r in range(1, 4):
        combos += list(itertools.permutations(["budget", "snip", "micro"], r))

    for _ in range(250):
        base = gen_conversation(rng, rng.randint(10, 45))
        for combo in combos:
            compact.reset_state()
            m = copy.deepcopy(base)   # 深拷贝,避免层间原地修改污染 base
            for name in combo:
                m = LAYERS[name](m, cfg)
            assert_valid(m, ctx=f"combo={combo or '∅'}")
            trials += 1

    # 2) 完整管线 budget→snip→micro→summary,多种 keep_tail
    for kt in (0, 1, 2, 4, 8):
        for _ in range(40):
            base = gen_conversation(rng, rng.randint(30, 60))
            c2 = compact.eval_profile()
            c2.summary_keep_tail = kt
            compact.reset_state()
            m = copy.deepcopy(base)
            m = compact.budget(m, c2)
            m = compact.snip(m, c2)
            m = compact.micro(m, c2)
            m = compact.summarize(m, "", c2)
            assert_valid(m, ctx=f"pipeline+summary keep_tail={kt}")
            trials += 1

    # 3) 定向最坏边界:tail 恰好以孤儿 tool_result 开头(keep_tail=1)
    edge = [{"role": "user", "content": "start"}]
    for i in range(45):
        tid = f"e{i}"
        edge.append({"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": "bash", "input": {}}]})
        edge.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": "x" * 80}]})
    c3 = compact.eval_profile()
    c3.summary_keep_tail = 1
    compact.reset_state()
    m = compact.summarize(copy.deepcopy(edge), "", c3)
    assert_valid(m, ctx="edge: 孤儿边界 keep_tail=1")
    trials += 1

    # 4) 定向回归:microcompact 工具白名单(D1+D2)—— update_todos 永不被清(唯一副本不可恢复)
    wl = [{"role": "user", "content": "task"}]
    specs = [("read_file", "X" * 5000, True), ("update_todos", "plan: locate/fix", False),
             ("bash", "Y" * 5000, True), ("read_file", "Z" * 5000, "keep")]
    for i, (name, body, _) in enumerate(specs):
        tid = f"w{i}"
        wl.append({"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": name, "input": {}}]})
        wl.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": body}]})
    out = compact.microcompact(copy.deepcopy(wl), compact.CompactConfig(microcompact_keep=1), target_tokens=0)
    CL = "[Old tool result content cleared]"
    res = [out[2 + 2 * i]["content"][0]["content"] for i in range(4)]   # 每个 tool_result 在 user 消息的 content[0]
    assert res[1] != CL, "update_todos 被清 = 致命 BUG(唯一副本不可恢复)"
    assert res[3] != CL, "最近的可清结果没被 keep 保留"
    assert res[0] == CL and res[2] == CL, "可恢复工具(read_file/bash)的旧结果应被清"
    trials += 1

    # 5) 定向回归:D14 post-compact 文件恢复去重 —— 已在保留近端原文里的文件不重注入
    compact.reset_state()
    # 保留近端 kept:read fileA(结果完好)、read fileB(结果已被 micro 清成占位符)
    kept = [
        {"role": "user", "content": "继续"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "rA", "name": "read_file", "input": {"path": "fileA.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "rA", "content": "AAA fileA 原文"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "rB", "name": "read_file", "input": {"path": "fileB.py"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "rB", "content": compact._MC_CLEARED}]},
    ]
    intact = compact._kept_intact_read_paths(kept)
    assert intact == {"fileA.py"}, f"D14: 应只认完好的 fileA(fileB 已被 micro 清),实际 {intact}"
    compact.track_file("fileA.py", "AAA fileA 原文")   # 在近端、完好 → 应被去重跳过
    compact.track_file("fileB.py", "BBB fileB 原文")   # 在近端但已清 → 内容不在场,仍须重注入
    compact.track_file("fileC.py", "CCC fileC 原文")   # 不在近端 → 正常重注入
    att = compact._post_compact_file_attachment(compact.CompactConfig(), exclude_paths=intact)
    assert "fileA.py" not in att, "D14: 近端已完好存在的 fileA 不该再被重注入(重复)"
    assert "fileB.py" in att and "fileC.py" in att, "D14: 已清的 fileB / 不在近端的 fileC 都应重注入"
    trials += 1

    # 5b) 先滤后取:被排除的文件不占 N 个名额(否则会挤掉一个更旧、本该保留的文件)
    compact.reset_state()
    for i in range(6):
        compact.track_file(f"g{i}.py", f"g{i} content")   # g5 最近;max_files=5
    att2 = compact._post_compact_file_attachment(compact.CompactConfig(), exclude_paths={"g5.py"})
    assert "g5.py" not in att2 and "g0.py" in att2, \
        "D14: 排除最近文件后应先滤再取5个 → 最旧的 g0 仍入选(被排除项不占名额)"
    trials += 1

    # 6) 定向回归:compact_naive 保两端(truncate 臂 = 对照组命根,红队 F7)——
    #    随机对话 × 多组 (keep_head, keep_tail, target) 下,丢中段后三条硬不变量仍成立,
    #    且早期承重(head)与最近态(tail)都被保住(护城河前提:不是 drop-oldest 稻草人)。
    for kh, kt in [(1, 1), (2, 4), (2, 8), (3, 2), (4, 10)]:
        for _ in range(60):
            base = gen_conversation(rng, rng.randint(20, 70))
            for tgt in (0, 200, 1500, 10 ** 9):   # 含极紧(逼到只剩尾1) 与 极松(只丢中段)
                m = compact.compact_naive(copy.deepcopy(base), target_tokens=tgt,
                                          system="SYS", keep_head=kh, keep_tail=kt)
                assert_valid(m, ctx=f"naive 保两端 kh={kh} kt={kt} tgt={tgt}")
                # 保头:任务头(base[0]) 必在结果第一条(承重锚点没被丢)
                assert m[0] == base[0], f"naive 保两端:任务头丢了 kh={kh} kt={kt}"
                # 保尾:最近一条(base[-1]) 必在结果末尾(最近态没被丢)
                assert m[-1] == base[-1], f"naive 保两端:最近态丢了 kh={kh} kt={kt}"
                trials += 1

    print(f"[OK] 结构合法性 property test 通过:{trials} 个 (对话 × 层组合) trial,")
    print("      三条硬不变量全部成立(以 user 开头 / 无孤儿 tool_result / 无悬空 tool_use)。")
    print("      + microcompact 工具白名单:update_todos 永不被清、可恢复工具清、keep 保最近。")
    print("      + D14 post-compact 去重:近端已完好的文件不重注入、已清的仍重注入、先滤后取不占名额。")
    print("      + compact_naive 保两端(truncate 臂):丢中段后不变量成立 + 任务头/最近态都保住。")


if __name__ == "__main__":
    main()
