"""Session Memory 回归 —— 触发逻辑 + 文件初始化 + extract 权限锁（离线，mock fork）。

    python eval/memory_eval/test_session_memory.py
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

from agent.context import compact
import agent.memory.session_memory as smmod
from agent.memory.session_memory import (
    SESSION_MEMORY_TEMPLATE,
    SessionMemory,
    SessionMemoryConfig,
)
from agent.memory.forked_agent import ForkResult


def conv(n_chars, n_tools):
    """造一个含 n_tools 个 tool_use 对、总量约 n_chars 的对话。"""
    msgs = [{"role": "user", "content": "start"}]
    per = max(1, n_chars // max(1, n_tools))
    for i in range(n_tools):
        msgs.append({"role": "assistant",
                     "content": [{"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {}}]})
        msgs.append({"role": "user",
                     "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * per}]})
    return msgs


def main():
    # 小 cfg 方便离线测触发逻辑（不必造真实 10K token）
    cfg = SessionMemoryConfig(min_tokens_to_init=100, min_tokens_between_update=50,
                              tool_calls_between_updates=2)
    tmp = Path(tempfile.mkdtemp()) / "sm.md"
    sm = SessionMemory(tmp, cfg)
    trials = 0

    # 1. context 不够 init → 不触发，且不标 initialized
    assert not sm.should_extract([{"role": "user", "content": "x" * 200}]), "below-init 不该触发"
    assert not sm._initialized
    trials += 1

    # 2. 够 init + 增长够 + tool 够 → 触发（并标 initialized）
    big = conv(800, 2)   # ~200 token > 100 init；2 个 tool
    assert sm.should_extract(big), "够阈值应触发"
    assert sm._initialized
    trials += 1

    # 3. 文件初始化 + is_empty 判定
    assert sm.is_empty(), "未 ensure 前视为空"
    sm._ensure_file()
    assert tmp.read_text(encoding="utf-8").strip() == SESSION_MEMORY_TEMPLATE.strip()
    assert sm.is_empty(), "刚写模板＝空"
    tmp.write_text(SESSION_MEMORY_TEMPLATE + "\n实质内容", encoding="utf-8")
    assert not sm.is_empty(), "有实质内容＝非空"
    sm._ensure_file()   # wx 独占（P1-2）：文件已存在 → 不该覆盖已有笔记
    assert "实质内容" in tmp.read_text(encoding="utf-8"), "wx 独占创建不该覆盖已有笔记"
    tmp.write_text(SESSION_MEMORY_TEMPLATE, encoding="utf-8")   # 还原
    trials += 1

    # 4. extract：mock fork（纯文本生成笔记），验 fork 不给工具 + harness 落盘 + 锚点推进
    captured = {}

    def fake_fork(prompt, ctx, *, system="", allowed_tools, max_turns, label, tool_filter=None):
        captured.update(allowed_tools=allowed_tools, label=label, max_turns=max_turns)
        return ForkResult(final_text=SESSION_MEMORY_TEMPLATE + "\n# 实质\n做了 X", stopped="finished")
    smmod.run_forked_agent = fake_fork

    sm.extract(big, system="SYS")
    assert captured["allowed_tools"] == set(), f"fork 应不给工具（纯文本生成），实际 {captured['allowed_tools']}"
    assert captured["label"] == "session_memory"
    assert "做了 X" in sm.path.read_text(encoding="utf-8"), "extract 应把 fork 生成的笔记文本落盘"
    # 锚点推进 → 紧接着同一对话不再触发（增长归零）
    assert sm._tokens_at_last > 0 and sm._seen_tool_ids == {"t0", "t1"}, \
        f"锚点应推进到 big 的 tool id，实际 {sm._seen_tool_ids}"
    assert sm.wait_for_extraction() is True, "extract 返回后 _extracting=False → wait 应就绪(P1-3)"
    assert not sm.should_extract(big), "提取后增长归零，不该再触发"
    trials += 1

    # 5. ★ P0-1 回归：压缩后短列表不该让 SM 永久停火（旧 index 锚点版会在此挂）
    #    extract(big) 后 _tokens_at_last≈estimate(big)。模拟 full_compact 产出的全新短列表：
    compacted = [{"role": "user", "content": "[Compacted] 摘要"}]   # estimate 骤降
    assert not sm.should_extract(compacted), "刚压缩这轮增长归零，本就不触发（合理）"
    # 关键：token 基准被自愈重置到压缩后基线，而非停在 big 的绝对值（否则负 delta 永久停火）
    assert sm._tokens_at_last <= compact.estimate(compacted) + 1, \
        f"压缩后 token 基准应自愈重置，实际 {sm._tokens_at_last}"
    # 压缩后上下文重新增长 + 全新 tool id → 必须能再次触发（证明没永久停火；旧代码因负 delta 在此挂）
    grown = compacted + [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "new1", "name": "bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "new1", "content": "y" * 400}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "new2", "name": "bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "new2", "content": "z" * 400}]},
    ]
    assert sm.should_extract(grown), "★ 压缩后重新增长+新 tool 必须能再触发（旧 index 锚点会因负 delta 永久停火）"
    # on_compacted 主动钩子也能重置（与自愈等价的另一路径）
    sm2 = SessionMemory(Path(tempfile.mkdtemp()) / "sm2.md", cfg)
    sm2._tokens_at_last = 999_999   # 模拟压缩前的大基准
    sm2.on_compacted(compacted)
    assert sm2._tokens_at_last <= compact.estimate(compacted) + 1, "on_compacted 应重置 token 基准"
    trials += 1

    print(f"[OK] session memory 回归通过：{trials} 组。")
    print("      触发三阈值 / 文件模板+is_empty / extract 权限锁死本文件 + 锚点推进 / ★压缩后不永久停火(P0-1)。")


if __name__ == "__main__":
    main()
