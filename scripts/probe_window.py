"""探 deepseek-v4-fresh 真实上下文窗口:发递增长度 prompt,看在哪一档 400(prompt_too_long)。

决定 SWE-bench eval 用 200K 还是 128K:窗口 = 最大不 400 的那档。
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import llm
from agent.context import compact

UNIT = "The quick brown fox jumps over the lazy dog. "   # ~11 tokens / 45 chars


def main():
    print("探 deepseek-v4-fresh 上下文窗口(发递增长度,看在哪档 400)…\n")
    ceiling = None
    for ktok in [50, 80, 110, 140, 170, 200, 230, 260]:
        n = max(1, ktok * 1000 // 11)
        content = UNIT * n + "\n以上是填充。只回复 OK。"
        est = compact.estimate([{"role": "user", "content": content}])
        try:
            resp = llm.chat([{"role": "user", "content": content}], max_tokens=8)
            txt = "".join(getattr(b, "text", "") for b in resp.content
                          if getattr(b, "type", None) == "text").strip()
            print(f"  ~{ktok}K tok (est {est:,}): OK   -> {txt[:24]!r}")
            ceiling = ktok
        except Exception as e:
            print(f"  ~{ktok}K tok (est {est:,}): FAIL -> {type(e).__name__}: {str(e)[:140]}")
            break
    print(f"\n最大通过档 ≈ {ceiling}K tok（真实窗口在它和下一档之间）")
    if ceiling and ceiling >= 200:
        print("→ 支持 200K:SWE-bench eval 用 200K(忠实 CC 的设计点)。")
    elif ceiling and ceiling >= 110:
        print(f"→ 不到 200K 但够大:用 ~{ceiling}K（同样套 CC 绝对常数,仍忠实,只是跑在更小窗口模型上)。")
    else:
        print("→ 窗口偏小,需再议(可能要换个大窗口模型跑 agent)。")


if __name__ == "__main__":
    main()
