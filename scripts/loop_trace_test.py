"""验证 Day 2 埋点 —— 无需 API key、不花钱。

打桩底层 Anthropic client（绕过真实网络），让 llm.chat / loop / tools.dispatch
的埋点逻辑全部真实运行，然后断言 trace 里有 agent.run / agent.turn / llm.call / tool.* span，
并生成一个示例 HTML 时间线。
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
    print("usage: loop_trace_test.py\n\nRun the stubbed loop-to-trace smoke check.")
    raise SystemExit(0)

from agent import config, llm, loop
from obs.trace import get_sink, render_tree
from obs.viewer import to_html


# ── 构造假的 Anthropic 响应对象 ──
class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Usage:
    def __init__(self, i, o):
        self.input_tokens, self.output_tokens = i, o


class _Resp:
    def __init__(self, content, stop_reason, usage):
        self.content, self.stop_reason, self.usage = content, stop_reason, usage


_calls = {"n": 0}


class _FakeMessages:
    def create(self, **kw):
        _calls["n"] += 1
        if _calls["n"] == 1:
            # 第一轮：让 agent 调用 write_file 工具
            return _Resp(
                content=[_Block(type="tool_use", name="write_file", id="t1",
                                input={"path": "hello.py", "content": "print('hi')"})],
                stop_reason="tool_use", usage=_Usage(1000, 50))
        # 第二轮：收尾
        return _Resp(
            content=[_Block(type="text", text="完成：写了 hello.py 并打印 hi")],
            stop_reason="end_turn", usage=_Usage(1200, 30))


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()


def main() -> None:
    # 注入假 client：绕过真实网络，但 llm.chat 的 span 逻辑照常运行
    llm._client = _FakeClient()

    result = loop.run_task("写 hello.py 并验证", max_turns=5)
    events = get_sink().events()
    names = [e["name"] for e in events]

    print(f"结果: {result}")
    print(f"捕获 spans: {names}\n")
    print(render_tree(events))

    # 断言 4 类接缝 span 都在
    assert any(n == "agent.run" for n in names), "缺 agent.run"
    assert any(n == "agent.turn" for n in names), "缺 agent.turn"
    assert any(n == "llm.call" for n in names), "缺 llm.call"
    assert any(n.startswith("tool.") for n in names), "缺 tool.*"
    # 断言 token 被记录
    llm_spans = [e for e in events if e["name"] == "llm.call"]
    assert any(e["attributes"].get("gen_ai.usage.input_tokens", 0) > 0 for e in llm_spans), \
        "llm.call 未记录 token"
    print("\n[OK] 埋点验证通过：agent.run / agent.turn / llm.call / tool.* 都在，且 token 已记录")

    html_path = config.TRACES_DIR / "sample_trace.html"
    html_path.write_text(to_html(events, "Sample (stubbed) run"), encoding="utf-8")
    print(f"[OK] 示例 HTML 时间线: {html_path}")


if __name__ == "__main__":
    main()
