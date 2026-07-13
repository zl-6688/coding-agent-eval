"""任务自检 —— 确保 eval 任务本身是合理的（不需要 API key）。

对每个难任务，用一份**参考解**跑它的隐藏测试，确认能 PASS（即测试不是不可能过的）；
并确认 H03 的 buggy setup 会 FAIL（即 bug 真的会被测到）。
这样 agent 跑出来的失败才反映 agent 能力，而不是我把测试写错了。
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from eval.run_eval import _copy_into, load_tasks

# ── 参考解（正确实现）──
REFS = {
    "H01-roman": ("roman.py", r'''
_VALS = [(1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),
         (50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")]
def to_roman(n):
    out = []
    for v, s in _VALS:
        while n >= v:
            out.append(s); n -= v
    return "".join(out)
def from_roman(s):
    m = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
    total = 0; prev = 0
    for ch in reversed(s):
        cur = m[ch]
        total += cur if cur >= prev else -cur
        prev = cur
    return total
'''),
    "H02-lru-cache": ("lru.py", r'''
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity; self.d = OrderedDict()
    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)
        return self.d[key]
    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
'''),
    "H03-fix-intervals": ("intervals.py", r'''
def merge(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    result = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
'''),
    "H04-calc": ("calc.py", r'''
import re
def evaluate(expr):
    tokens = re.findall(r"\d+\.\d+|\d+|[+\-*/()]", expr)
    pos = [0]
    def factor():
        t = tokens[pos[0]]
        if t == "(":
            pos[0] += 1
            v = expr_()
            pos[0] += 1
            return v
        pos[0] += 1
        return float(t) if "." in t else int(t)
    def term():
        v = factor()
        while pos[0] < len(tokens) and tokens[pos[0]] in ("*", "/"):
            op = tokens[pos[0]]; pos[0] += 1
            r = factor()
            v = v * r if op == "*" else v / r
        return v
    def expr_():
        v = term()
        while pos[0] < len(tokens) and tokens[pos[0]] in ("+", "-"):
            op = tokens[pos[0]]; pos[0] += 1
            r = term()
            v = v + r if op == "+" else v - r
        return v
    return expr_()
'''),
    "H05-flatten": ("flatten.py", r'''
def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = prefix + str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out
'''),
    "H06-topk": ("topk.py", r'''
from collections import Counter
def top_k(text, k):
    words = text.lower().split()
    if not words:
        return []
    c = Counter(words)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
'''),
}


def run_verify(task, files):
    ws = Path(tempfile.mkdtemp(prefix="valid_"))
    try:
        for fn, code in files.items():
            (ws / fn).write_text(code, encoding="utf-8")
        _copy_into(task["_dir"] / "verify", ws)
        parts = task["verify_cmd"].split()
        if parts and parts[0] == "python":
            parts[0] = sys.executable
        proc = subprocess.run(parts, cwd=ws, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=60)
        return proc.returncode, (proc.stdout + proc.stderr).strip()[-200:]
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def main():
    tasks = {t["id"]: t for t in load_tasks()}
    ok = True

    for tid, (fn, code) in REFS.items():
        rc, out = run_verify(tasks[tid], {fn: code})
        print(f"{tid:20} 参考解 -> {'PASS' if rc == 0 else 'FAIL ' + out}")
        ok = ok and rc == 0

    # H03 的 buggy setup 应当 FAIL（证明 bug 会被测到）
    t = tasks["H03-fix-intervals"]
    buggy = (t["_dir"] / "setup" / "intervals.py").read_text(encoding="utf-8")
    rc, _ = run_verify(t, {"intervals.py": buggy})
    print(f"{'H03 buggy setup':20} -> {'FAIL（符合预期，bug 被测到）' if rc != 0 else 'PASS（❌ bug 没被测到）'}")
    ok = ok and rc != 0

    print("\n" + ("[OK] 所有难任务的测试都可过，且 H03 的 bug 会被测到——任务集合理。"
                  if ok else "[!] 有任务不合理，需要修。"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
