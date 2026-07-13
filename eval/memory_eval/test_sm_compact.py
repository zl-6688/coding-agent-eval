"""SM 压缩桥回归（step1b）—— 零 LLM、回退、API 合法性、pipeline 接管。

session_memory_compact 用已写好的 SM 文件当摘要、不调 LLM，故纯离线可测（不 mock llm）。
pipeline 测里 mock full_compact 成桩，防回退路径真打 LLM。

    python eval/memory_eval/test_sm_compact.py
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
from agent.memory.session_memory import SESSION_MEMORY_TEMPLATE, SessionMemory


def big_text(n=6, per=300):
    """造一段不可被 micro 清理的对话（纯 text/user，无 tool_result）→ 逼出 SM-compact/full。"""
    msgs = [{"role": "user", "content": "task start"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"思考{i} " + "x" * per}]})
        msgs.append({"role": "user", "content": f"补充{i} " + "y" * per})
    return msgs


def _assert_valid(messages):
    """API 合法性核心：以 user 开头、开头不是孤儿 tool_result（防 step1b 切断 tool 对→400）。"""
    assert messages[0]["role"] == "user", f"必须以 user 开头，实际 {messages[0]['role']}"
    c = messages[0].get("content")
    if isinstance(c, list):
        assert not any((b.get("type") if isinstance(b, dict) else None) == "tool_result" for b in c), \
            "开头不能是孤儿 tool_result"


def main():
    cfg = compact.CompactConfig(keep_min_tokens=10, keep_min_msgs=1, keep_max_tokens=50)
    tmp = Path(tempfile.mkdtemp())
    big = big_text()
    trials = 0

    # 1. SM 空模板 → 回退 None（走 full）
    sm_empty = SessionMemory(tmp / "empty.md")
    sm_empty._ensure_file()
    assert sm_empty.is_empty()
    assert compact.session_memory_compact(big, sm_empty, "", cfg, auto_thr=10_000) is None, \
        "空模板应回退 full"
    trials += 1

    # 2. SM 有内容 → result（零 LLM / 合法 / 含近端 / on_compacted 重置锚点）
    sm = SessionMemory(tmp / "sm.md")
    sm._ensure_file()
    sm.path.write_text(SESSION_MEMORY_TEMPLATE + "\n\n# 实质\n做了 A、B，改了 main.py", encoding="utf-8")
    assert not sm.is_empty()
    r = compact.session_memory_compact(big, sm, "", cfg, auto_thr=10_000)
    assert r is not None, "有内容应成功"
    assert r[0]["role"] == "user" and "session memory" in r[0]["content"], "首条应是 SM 摘要"
    _assert_valid(r)
    assert any("补充5" in str(m.get("content", "")) for m in r), "应保留近端尾部原文"
    assert sm._tokens_at_last <= compact.estimate(r) + 1, "on_compacted 应把锚点重置到压缩后基线"
    trials += 1

    # 3. 压完仍超阈值（auto_thr 极小）→ 回退 None
    assert compact.session_memory_compact(big, sm, "", cfg, auto_thr=5) is None, \
        "压完仍超阈值应回退 full"
    trials += 1

    # 4 & 5：pipeline 接管 vs 回退（mock full_compact 成桩，防回退路径真打 LLM）
    _orig_full = compact.full_compact
    compact.full_compact = lambda msgs, system="", cfg=None: [{"role": "user", "content": "[FULL-FALLBACK]"}]
    try:
        # 4. pipeline 带有内容 SM → SM-compact 接管（零 LLM），不走 full
        out = compact.compact_pipeline(big, system="", cfg=cfg, target_tokens=400, session_memory=sm)
        _assert_valid(out)
        assert "session memory" in str(out[0].get("content", "")), "SM 应接管（非 full fallback）"
        assert not any("FULL-FALLBACK" in str(m.get("content", "")) for m in out), "SM 接管时不该走 full"
        trials += 1

        # 5. pipeline 带空 SM → 回退 full（桩）
        out = compact.compact_pipeline(big, system="", cfg=cfg, target_tokens=400, session_memory=sm_empty)
        assert any("FULL-FALLBACK" in str(m.get("content", "")) for m in out), "空 SM 应回退 full"
        trials += 1

        # 6. pipeline 不传 session_memory → 老行为（直接 full），不受影响
        out = compact.compact_pipeline(big, system="", cfg=cfg, target_tokens=400)
        assert any("FULL-FALLBACK" in str(m.get("content", "")) for m in out), "无 SM 参数＝老行为走 full"
        trials += 1
    finally:
        compact.full_compact = _orig_full

    print(f"[OK] SM 压缩桥回归通过：{trials} 组。")
    print("      零LLM摘要 / 空模板回退 / 压不够回退 / API合法(无孤儿) / pipeline接管 vs 回退 / 老行为不变。")


if __name__ == "__main__":
    main()
