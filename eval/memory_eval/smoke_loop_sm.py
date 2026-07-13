"""小端到端真实验证（step1）—— 开 Session Memory 跑一个真任务，验 SM 真被 forked-LLM 写。

★ 真实 deepseek 调用（主 loop + SM fork 都真跑）。需带代理。
小 SM cfg 让短任务也触发 should_extract。隔离 workdir（tempfile，repo 外）。

    HTTPS_PROXY=http://<your-proxy> HTTP_PROXY=http://<your-proxy> NO_PROXY=localhost,127.0.0.1 \
      PYTHONIOENCODING=utf-8 python eval/memory_eval/smoke_loop_sm.py
"""

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import config, loop
from agent.memory.session_memory import SessionMemory, SessionMemoryConfig


def main():
    print(f"[config] MODEL_ID={config.MODEL_ID}  BASE_URL={config.BASE_URL}")
    if not config.API_KEY:
        print("!! 无 ANTHROPIC_API_KEY（.env），跳过真实调用。")
        return

    tmp = Path(tempfile.mkdtemp())
    ws = tmp / "ws"
    sm_path = tmp / "session_memory.md"
    # 小 cfg：让短任务也触发 SM extract（init 600 token / 增长 300 / tool 3）
    cfg = SessionMemoryConfig(min_tokens_to_init=200, min_tokens_between_update=100,
                              tool_calls_between_updates=2)
    sm = SessionMemory(sm_path, cfg)

    task = (
        "在当前目录创建一个 Python 文件 utils.py，写两个函数："
        "add(a, b) 返回两数之和、is_even(n) 判断 n 是否偶数。"
        "然后用 bash 运行 `python -c \"from utils import add, is_even; print(add(2,3), is_even(4))\"` 验证。"
        "再创建 README.md 简要说明这两个函数。最后用一两句话总结你做了什么。"
    )

    print(f"[run] workdir={ws}  sm={sm_path}")
    with config.using_workdir(ws):
        out = loop.run_task(task, max_turns=14, trace=True,
                            session_memory=sm)

    print("\n=== run_task 输出（截断）===")
    print((out or "")[:600])
    print("\n=== Session Memory 验证 ===")
    print("文件存在:", sm_path.exists())
    print("is_empty（仍是空模板?）:", sm.is_empty())
    if sm_path.exists():
        content = sm_path.read_text(encoding="utf-8")
        print(f"SM 文件长度: {len(content)} chars")
        print("--- SM 文件内容（截断 2500）---")
        print(content[:2500])
    # 结论
    wrote = sm_path.exists() and not sm.is_empty()
    print("\n[结论] SM 被 forked-LLM 真实写入:", "✓ 是" if wrote else "✗ 否（未触发或 fork 没写）")


if __name__ == "__main__":
    main()
