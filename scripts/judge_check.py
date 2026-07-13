"""验证 judge 修复 —— 用真实 judge 模型打 1 个分（需 key，仅 1 次调用，便宜）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("usage: judge_check.py\n\nRun one configured live judge request.")
    raise SystemExit(0)

from agent import config
from eval import judge

if not judge.judge_available():
    print(f"judge 不可用：MODEL_ID={config.MODEL_ID!r} JUDGE_MODEL_ID={config.JUDGE_MODEL_ID!r}"
          f"（需配置 JUDGE_MODEL_ID 且与被测模型不同）")
    sys.exit(1)

print(f"judge 模型 = {config.JUDGE_MODEL_ID} | 被测 = {config.MODEL_ID}\n")

code = "def quicksort(lst):\n    return sorted(lst)\n"
result = judge.judge_code("实现 quicksort(lst)：对列表升序排序并返回新列表。", code)
print("judge 返回:", result)

if result.get("avg"):
    print(f"\n[OK] judge 修复生效，拿到分数 avg={result['avg']}/5.0")
    sys.exit(0)
print(f"\n[!] 仍未拿到分数 —— 把上面的返回贴给我继续排查。")
sys.exit(1)
