"""验证 eval 流水线 —— 无需 API key、不花钱。

打桩 agent，让它对 T01 产出①正确实现②错误实现，确认 run_one 分别报告 PASS / FAIL，
即证明：workspace 隔离、setup/verify 拷贝、验证子进程、trace 指标抽取全部正常。
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
    print("usage: eval_smoke.py\n\nRun the offline evaluation-pipeline smoke check.")
    raise SystemExit(0)

from agent import llm
from eval.run_eval import load_tasks, run_one


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class _Resp:
    def __init__(self, content, stop_reason, usage):
        self.content, self.stop_reason, self.usage = content, stop_reason, usage


def make_fake(impl_code):
    calls = {"n": 0}

    class M:
        def create(self, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Resp([_Block(type="tool_use", name="write_file", id="t1",
                                     input={"path": "quicksort.py", "content": impl_code})],
                             "tool_use", _Usage(500, 40))
            return _Resp([_Block(type="text", text="done")], "end_turn", _Usage(600, 10))

    class C:
        def __init__(self):
            self.messages = M()

    return C()


GOOD = ("def quicksort(lst):\n"
        "    if len(lst) <= 1:\n"
        "        return list(lst)\n"
        "    p = lst[len(lst) // 2]\n"
        "    return (quicksort([x for x in lst if x < p])\n"
        "            + [x for x in lst if x == p]\n"
        "            + quicksort([x for x in lst if x > p]))\n")
BAD = "def quicksort(lst):\n    return lst\n"


def main():
    t01 = {t["id"]: t for t in load_tasks()}["T01-quicksort"]

    llm._client = make_fake(GOOD)
    r_good = run_one(t01)
    print(f"正确实现 -> {'PASS' if r_good['passed'] else 'FAIL'}  "
          f"(turns={r_good['turns']}, tok={r_good['input_tokens']}+{r_good['output_tokens']}, "
          f"{r_good['latency_s']}s)")
    assert r_good["passed"], f"正确实现应当 PASS，但得到: {r_good}"

    llm._client = make_fake(BAD)
    r_bad = run_one(t01)
    print(f"错误实现 -> {'PASS' if r_bad['passed'] else 'FAIL'}  (verify_rc={r_bad['verify_rc']})")
    assert not r_bad["passed"], "错误实现应当 FAIL"

    print("\n[OK] eval 流水线验证通过：正确→PASS、错误→FAIL；"
          "workspace 隔离 + 隐藏测试子进程 + trace 指标抽取都正常。")


if __name__ == "__main__":
    main()
