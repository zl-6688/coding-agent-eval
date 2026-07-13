"""验证 tracing 核心 —— 无需 API key。

构造一个假的 agent run（含 subagent + 后台任务 + 一个失败的工具调用），证明：
  - span 自动挂父子（subagent 内部 span 落在 subagent.run 之下）
  - 并发线程的 span 通过 copy_context 落在同一 trace
  - 扁平事件流能重建成正确的 span 树
  - 异常 span 标 ERROR 但不让程序崩
"""

import sys
import threading
import time
from pathlib import Path

# Windows 控制台默认非 UTF-8，重配 stdout 以正确显示中文
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from obs.trace import (
    JsonlSink, SpanKind, capture_context, get_sink, render_tree, set_sink, span,
)


def main() -> None:
    set_sink(JsonlSink(REPO / ".traces" / "demo.jsonl"))

    with span("agent.turn", SpanKind.AGENT, task="修复登录 bug"):
        # 1) LLM 调用：带 OTel gen_ai.* 属性
        with span("llm.call", SpanKind.CLIENT, **{
            "gen_ai.request.model": "demo-model",
            "gen_ai.usage.input_tokens": 1200,
            "gen_ai.usage.output_tokens": 300,
            "cost_usd": 0.012,
        }):
            time.sleep(0.01)

        # 2) 工具调用
        with span("tool.dispatch", SpanKind.TOOL, **{"tool.name": "bash"}):
            time.sleep(0.005)

        # 3) subagent：内部 span 自动挂到 subagent.run 之下
        with span("subagent.run", SpanKind.AGENT, **{"subagent": "test-writer"}):
            with span("llm.call", SpanKind.CLIENT,
                      **{"gen_ai.request.model": "demo-model"}):
                time.sleep(0.008)
            with span("tool.dispatch", SpanKind.TOOL, **{"tool.name": "write_file"}):
                time.sleep(0.003)

        # 4) 后台任务：另一线程，copy_context 让它落在同一 trace
        def bg() -> None:
            with span("background.task", SpanKind.INTERNAL, **{"cmd": "long build"}):
                time.sleep(0.01)

        ctx = capture_context()
        t = threading.Thread(target=lambda: ctx.run(bg))
        t.start()
        t.join()

        # 5) 故意失败的 span：应标 ERROR，但不让程序崩
        try:
            with span("tool.dispatch", SpanKind.TOOL, **{"tool.name": "edit_file"}):
                raise ValueError("文件未找到")
        except ValueError:
            pass

    events = get_sink().events()
    print(f"\n捕获 {len(events)} 个 span。重建出的 trace 树：\n")
    print(render_tree(events))
    print(f"\n事件流已写入: {get_sink().path}")


if __name__ == "__main__":
    main()
