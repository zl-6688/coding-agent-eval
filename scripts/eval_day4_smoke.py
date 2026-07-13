"""验证 Day 4 三件事 —— 无需 API key、不花钱：
  1) repeat-N + flaky 检测（让 agent 一次对一次错 → 应标 flaky）
  2) LLM-judge 接入，且 judge==被测模型时正确跳过（守自评偏差）
  3) 回归门禁逻辑
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("usage: eval_day4_smoke.py\n\nRun the offline repeat-N, judge, and regression smoke checks.")
    raise SystemExit(0)

from agent import config, llm
from eval import judge
from eval.run_eval import load_tasks, regression_check, run_repeated


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class _Resp:
    def __init__(self, content, stop_reason, usage):
        self.content, self.stop_reason, self.usage = content, stop_reason, usage


GOOD = ("def quicksort(lst):\n"
        "    if len(lst) <= 1:\n"
        "        return list(lst)\n"
        "    p = lst[len(lst) // 2]\n"
        "    return (quicksort([x for x in lst if x < p])\n"
        "            + [x for x in lst if x == p]\n"
        "            + quicksort([x for x in lst if x > p]))\n")
BAD = "def quicksort(lst):\n    return lst\n"


def alternating_client(impls):
    """create 奇数次 -> 写文件(取下一个 impl)，偶数次 -> 收尾。"""
    st = {"n": 0, "wi": 0}

    class M:
        def create(self, **kw):
            st["n"] += 1
            if st["n"] % 2 == 1:
                impl = impls[st["wi"] % len(impls)]
                st["wi"] += 1
                return _Resp([_Block(type="tool_use", name="write_file", id=f"t{st['n']}",
                                     input={"path": "quicksort.py", "content": impl})],
                             "tool_use", _Usage(500, 40))
            return _Resp([_Block(type="text", text="done")], "end_turn", _Usage(600, 10))

    class C:
        def __init__(self):
            self.messages = M()

    return C()


def fake_judge_client(json_text):
    class M:
        def create(self, **kw):
            return _Resp([_Block(type="text", text=json_text)], "end_turn", _Usage(100, 20))

    class C:
        def __init__(self):
            self.messages = M()

    return C()


def main():
    t01 = {t["id"]: t for t in load_tasks()}["T01-quicksort"]

    # 1) repeat-N + flaky
    llm._client = alternating_client([GOOD, BAD])      # run1 对, run2 错
    agg = run_repeated(t01, n=2, do_judge=False)
    print(f"[1] repeat-2: passes={agg['passes']}/2  pass_fraction={agg['pass_fraction']}  flaky={agg['flaky']}")
    assert agg["passes"] == 1 and agg["flaky"] is True, f"flaky 检测错: {agg}"

    # 2) judge 接入（judge≠被测）
    config.JUDGE_MODEL_ID = "fake-judge-model"
    assert judge.judge_available(), "judge 应可用"
    judge._client = fake_judge_client('{"correctness":4,"readability":3,"robustness":5}')
    llm._client = alternating_client([GOOD, GOOD])
    agg2 = run_repeated(t01, n=1, do_judge=True)
    print(f"[2] judge -> {agg2.get('judge')}")
    assert agg2["judge"]["avg"] == 4.0, f"judge 分数错: {agg2.get('judge')}"

    # 2b) judge==被测模型 -> 跳过（守自评偏差）
    config.JUDGE_MODEL_ID = config.MODEL_ID
    assert not judge.judge_available(), "judge==被测时应不可用"
    print("[2b] judge==被测模型 -> 正确跳过（守自评偏差）")

    # 3) 回归门禁
    d1, reg1 = regression_check(70.0, 83.3)
    d2, reg2 = regression_check(85.0, 83.3)
    print(f"[3] 70 vs 83.3 -> {d1:+}pp regression={reg1} | 85 vs 83.3 -> {d2:+}pp regression={reg2}")
    assert reg1 is True and reg2 is False, "回归门禁逻辑错"

    print("\n[OK] Day4 验证通过：repeat-N flaky 检测、judge 接入(judge≠被测守门)、回归门禁 都正常。")


if __name__ == "__main__":
    main()
