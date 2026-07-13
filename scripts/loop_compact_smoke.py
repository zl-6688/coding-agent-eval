"""真实 loop 压缩 smoke —— 压缩接进 loop(小阈值强制触发),看:
① 任务跑通 ② 触发几次压缩 ③ compact.* span 的模块级指标(before/after/ratio)从真实运行里捞出来。
这正是"instrumented 真实 loop"主线的最小验证。需 API。

    python scripts/loop_compact_smoke.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import config, loop

TASK = ("创建 fib.py:实现 fib(n) 返回第 n 个斐波那契数(fib(0)=0,fib(1)=1)。"
        "写完后 read_file 确认一遍,再用 bash 跑 `python -c \"from fib import fib; print(fib(10))\"` 验证输出 55。")
THRESHOLD = 250   # 调到很小,强制普通任务也触发(照 CC 的测试 env-var 思路)


def _compact_spans(trace_path):
    rows = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        name = rec.get("name", "")
        if name.startswith("compact."):
            rows.append((name, rec.get("attributes", {})))
    return rows


def main():
    print(f"=== compaction strategy=pipeline · threshold={THRESHOLD} tok ===")
    res = loop.run_task(TASK, max_turns=12,
                        eval_hooks=loop.EvalHooks(compact_strategy="pipeline", compact_threshold=THRESHOLD))
    print("RESULT:", (res or "")[:200])

    traces = sorted(config.TRACES_DIR.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not traces:
        print("(无 trace)")
        return
    latest = traces[-1]
    print(f"\ntrace: {latest.name}")
    rows = _compact_spans(latest)
    pipes = [a for n, a in rows if n == "compact.pipeline"]
    for name, a in rows:
        print(f"  {name:22} before={a.get('tokens_before')} after={a.get('tokens_after')} "
              f"ratio={a.get('ratio')} cleared={a.get('cleared', '')} status={a.get('status', '')}")
    print(f"\n压缩触发(compact.pipeline)次数: {len(pipes)} · compact.* span 总数: {len(rows)}")
    if pipes:
        print("[OK] 压缩已接进真实 loop,模块级指标从真实运行的 span 里读到了。")
    else:
        print("[!] 本次没触发压缩(上下文没超阈值)——把 THRESHOLD 调更小再跑。")


if __name__ == "__main__":
    main()
