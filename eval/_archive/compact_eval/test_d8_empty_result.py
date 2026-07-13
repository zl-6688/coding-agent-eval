"""D8 回归 —— 空 tool_result 注"完成、无输出"占位(对齐 CC isToolResultContentEmpty)。

纯空/纯空白的 tool_result 会让某些模型触发停止序列(CC inc-4586)。dispatch 在工具执行层
把空输出换成 `(<tool> 执行完成，无输出)` 占位:既非空、又区分于报错(空输出常是成功 no-op)。
离线:monkeypatch 一个返回空串的 handler,断言 dispatch 的行为。无 API、无真实工具。

    python eval/_archive/compact_eval/test_d8_empty_result.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import tools


def _with_handler(name, fn):
    """临时塞一个测试 handler,跑完恢复(避免污染全局 TOOL_HANDLERS)。"""
    saved = tools.TOOL_HANDLERS.get(name)
    tools.TOOL_HANDLERS[name] = fn
    try:
        return tools.dispatch(name, {})
    finally:
        if saved is None:
            tools.TOOL_HANDLERS.pop(name, None)
        else:
            tools.TOOL_HANDLERS[name] = saved


def main():
    trials = 0

    # 1) 纯空串 → 占位
    out = _with_handler("bash", lambda **kw: "")
    assert out == "(bash 执行完成，无输出)", f"空串应换占位,实际 {out!r}"
    trials += 1

    # 2) 纯空白(空格/换行/Tab)→ 占位
    out = _with_handler("bash", lambda **kw: "  \n\t \n")
    assert out == "(bash 执行完成，无输出)", f"纯空白应换占位,实际 {out!r}"
    trials += 1

    # 3) 占位符不得被误判为 error(空输出是成功 no-op,不是失败)
    #    dispatch 内 is_error = output.startswith("Error");占位不以 Error 开头 → 不计失败。
    before = tools._TOOL_ERRORS[0]
    _with_handler("glob", lambda **kw: "")
    assert tools._TOOL_ERRORS[0] == before, "空输出占位被错误计入工具失败数"
    trials += 1

    # 4) 非空输出原样透传(不受影响)
    out = _with_handler("bash", lambda **kw: "hello world")
    assert out == "hello world", f"非空输出应原样透传,实际 {out!r}"
    trials += 1

    # 5) 真正的报错仍按报错走(以 Error 开头,不被占位逻辑吞掉)
    out = _with_handler("bash", lambda **kw: "Error: boom")
    assert out == "Error: boom", f"报错应原样透传,实际 {out!r}"
    trials += 1

    print(f"[OK] D8 空结果占位回归通过:{trials} 个 case。")
    print("      空/纯空白→占位、占位非 error、非空透传、报错透传。")


if __name__ == "__main__":
    main()
