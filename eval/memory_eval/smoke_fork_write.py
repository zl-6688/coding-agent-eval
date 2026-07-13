"""隔离验证：fork 能否真把 Session Memory 文件写出来（合成对话 + 手动 extract，省主 loop 钱）。

只触发 1 次 SM fork（几次 deepseek 调用）。暴露：fork 的 edit 成没成、written_paths、为什么。
需带代理。

    HTTPS_PROXY=http://<your-proxy> HTTP_PROXY=http://<your-proxy> NO_PROXY=localhost,127.0.0.1 \
      PYTHONIOENCODING=utf-8 python eval/memory_eval/smoke_fork_write.py
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
import agent.memory.forked_agent as forked_mod
from agent.memory.session_memory import SessionMemory
from agent.tools.pool import ToolPool
from agent.tools.contracts import Tool

# 诊断：trace fork 的工具调用（看 write 的 content + ToolRuntime 返回）
_orig_assemble_tool_pool = forked_mod.assemble_tool_pool


def _wrap_tool(tool):
    def call(inp, context, _tool=tool):
        out = _tool.call(inp, context)
        if context.is_subagent:
            content = getattr(out, "content", out)
            print(f"  [fork tool] {_tool.name} path={inp.get('path')!r} → ret={str(content)[:140]!r}")
            if _tool.name == "write_file":
                print(
                    "  [fork write] "
                    f"content_len={len(inp.get('content', ''))} "
                    f"head={inp.get('content', '')[:250]!r}"
                )
        return out

    return Tool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        call=call,
        source=tool.source,
        is_read_only=tool.is_read_only,
        is_destructive=tool.is_destructive,
        is_concurrency_safe=tool.is_concurrency_safe,
        validate_input=tool.validate_input,
        map_result=tool.map_result,
        metadata=tool.metadata,
    )


def _traced_assemble_tool_pool(context=None):
    pool = _orig_assemble_tool_pool(context)
    return ToolPool(tuple(_wrap_tool(tool) for tool in pool.tools))


forked_mod.assemble_tool_pool = _traced_assemble_tool_pool


def main():
    print(f"[config] MODEL_ID={config.MODEL_ID}  BASE_URL={config.BASE_URL}")
    if not config.API_KEY:
        print("!! 无 API key，跳过。")
        return

    tmp = Path(tempfile.mkdtemp())
    ws = tmp / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    sm = SessionMemory(tmp / "sm.md")   # SM 文件在 WORKDIR(ws) 外 —— 验证 fork 能否写 WORKDIR 外的绝对路径

    # 合成一段有实质内容的对话（让 fork 有东西可记）
    messages = [
        {"role": "user", "content": "在当前目录创建 utils.py，写 add(a,b) 和 is_even(n) 两个函数"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "1", "name": "write_file",
             "input": {"path": "utils.py", "content": "def add(a,b): return a+b\ndef is_even(n): return n%2==0"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "已写入 60 字节 → utils.py"}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "2", "name": "bash",
             "input": {"command": "python -c 'from utils import add; print(add(2,3))'"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "2", "content": "5"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "创建了 utils.py，add(2,3)=5 验证通过"}]},
    ]

    print(f"[run] 手动 sm.extract（fork）... sm_path={sm.path}  (在 WORKDIR={ws} 外)")
    with config.using_workdir(ws):
        res = sm.extract(messages, loop.SYSTEM)

    print(f"\n[fork result] stopped={res.stopped}  turns={res.turns}")
    print(f"[fork result] written_paths={res.written_paths}")
    print(f"[fork result] final_text={res.final_text[:400]!r}")
    print(f"\n[SM is_empty]: {sm.is_empty()}")
    print("--- SM 文件内容（截断 2500）---")
    print(sm.path.read_text(encoding="utf-8")[:2500])


if __name__ == "__main__":
    main()
