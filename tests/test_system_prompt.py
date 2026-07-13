"""test_system_prompt.py — agent/context/system_prompt.py 的 mock 单元测试。

验证：
  - 无 memory_dir → 3 段（identity/tools/workspace）
  - AGENTS.md 不进入 system prompt section cache
  - identity 段不含已删除的「定位并修复 bug」尾巴
  - tools 段列出 ToolPool prompt tools 的工具名
  - workspace 段含 workdir
  - memory 启用 → system 只含稳定策略，不含动态 MEMORY.md 内容
  - index mode 的 MEMORY.md 在 request context 中读取并截断
  - selector / disabled mode 不注入动态索引
  - 缓存：同 state 连调两次 → 第二次命中缓存（section 只渲染一次）
  - state 变（workdir 变 / memory 出现）→ 重算
"""

import importlib
from pathlib import Path

import pytest

from agent.tools.pool import assemble_tool_pool
from agent.context.system_prompt import (
    DEFAULT_IDENTITY,
    EXPERIMENTAL_IDENTITY,
    LEGACY_IDENTITY,
    SystemState,
    _CACHE,
    _section_identity,
    _section_tools,
    _section_workspace,
    _section_memory,
    build_system,
)


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clear_section_cache():
    """每个测试前后清空段缓存，防跨测试污染。"""
    _CACHE.clear()
    yield
    _CACHE.clear()


def _make_state(
    workdir: str = "/test/workdir",
    memory_dir: str | None = None,
    memory_enabled: bool | None = None,
    memory_recall_mode: str = "selector",
) -> SystemState:
    return SystemState(
        tools=assemble_tool_pool().prompt_tools_for_system(),
        workdir=workdir,
        memory_dir=memory_dir,
        memory_enabled=bool(memory_dir) if memory_enabled is None else memory_enabled,
        memory_recall_mode=memory_recall_mode,
    )


# ── 基本分段 ───────────────────────────────────────────────────────────────────


def test_no_memory_dir_gives_three_sections():
    """无 memory_dir → build_system 输出包含 identity/tools/workspace，不含 memory 段。

    WHY 不用 split("\n\n") 计数：identity 内部本身含 \n\n（## 工作方法 / ## 原则 之间），
    导致 split 结果 > 3。改用段特征字符串检查：三段应有，memory 段不应有。
    """
    state = _make_state()
    result = build_system(state)
    # identity 段特征
    assert "coding agent" in result
    # tools 段特征
    assert "## 可用工具" in result
    # workspace 段特征
    assert "## 工作目录" in result
    # memory 段应缺席
    assert "长期记忆" not in result


def test_identity_does_not_contain_bug_tail():
    """identity 段**不含**「定位并修复 bug」——已从第一行去掉。"""
    state = _make_state()
    result = build_system(state)
    assert "定位并修复 bug" not in result, "identity 段不应含已删除的 bug 修复尾巴"


def test_identity_first_line_correct():
    """identity 第一行精确匹配：去尾后的正确表述。"""
    assert DEFAULT_IDENTITY.splitlines()[0] == "你是一个 coding agent，在代码仓库里完成软件工程任务。"


def test_default_identity_is_legacy_baseline():
    """默认 identity 回退到已确认的 legacy 基线，实验 prompt 只能显式启用。"""
    assert DEFAULT_IDENTITY == LEGACY_IDENTITY


def test_identity_retains_work_method_section():
    """默认 identity 保留用户确认过的 legacy 工作方法结构。"""
    assert "## 工作方法（按步骤）" in DEFAULT_IDENTITY
    assert "理解任务描述的问题与期望行为" in DEFAULT_IDENTITY
    assert "定位（最关键）" in DEFAULT_IDENTITY
    assert "grep / glob" in DEFAULT_IDENTITY
    assert "read_file" in DEFAULT_IDENTITY


def test_identity_retains_principles_section():
    """identity 保留原则段。"""
    assert "## 原则" in DEFAULT_IDENTITY
    assert "该做的做完就收尾" in DEFAULT_IDENTITY


def test_experimental_identity_uses_capability_language_not_core_tool_names():
    """实验 identity 表达工程能力，不把该变量混入默认产品行为。"""
    lower_identity = EXPERIMENTAL_IDENTITY.lower()
    for tool_word in ["grep / glob", "read_file", "edit_file", "update_todos", "bash"]:
        assert tool_word not in lower_identity


def test_experimental_identity_requires_honest_post_change_verification():
    """实验 identity 保留强验证契约，用于后续受控 A/B。"""
    assert "真实行为" in EXPERIMENTAL_IDENTITY
    assert "最后一次源码改动后" in EXPERIMENTAL_IDENTITY
    assert "不能用更窄或无关的通过结果覆盖" in EXPERIMENTAL_IDENTITY
    assert "无法验证" in EXPERIMENTAL_IDENTITY


def test_tools_section_lists_tool_names():
    """tools 段列出 ToolPool prompt tools 中每个工具的名字。"""
    state = _make_state()
    result = build_system(state)
    for t in assemble_tool_pool().prompt_tools_for_system():
        assert t["name"] in result, f"工具 {t['name']} 未出现在 system prompt 中"


def test_tools_section_has_header():
    """tools 段有 ## 可用工具 标题。"""
    state = _make_state()
    result = build_system(state)
    assert "## 可用工具" in result


def test_tools_section_fingerprint_includes_description():
    """Same tool name with a changed description must render a new tools section."""
    first = SystemState(
        tools=[
            {
                "name": "demo",
                "description": "first description",
                "input_schema": {"type": "object"},
            }
        ],
        workdir="/wd",
    )
    second = SystemState(
        tools=[
            {
                "name": "demo",
                "description": "second description",
                "input_schema": {"type": "object"},
            }
        ],
        workdir="/wd",
    )

    assert "first description" in _section_tools(first)
    rendered = _section_tools(second)

    assert "second description" in rendered
    assert "first description" not in rendered


def test_workspace_section_contains_workdir():
    """workspace 段含传入的 workdir。"""
    wd = "/my/specific/workdir"
    state = _make_state(workdir=wd)
    result = build_system(state)
    assert wd in result, f"workdir {wd!r} 未出现在 system prompt 中"


def test_workspace_section_has_header():
    """workspace 段有 ## 工作目录 标题。"""
    state = _make_state()
    result = build_system(state)
    assert "## 工作目录" in result


# ── project instructions 不属于 system prompt ────────────────────────────────


def test_system_prompt_has_no_project_instructions_section():
    """AGENTS.md 不再作为 system prompt section，也不产生 project cache key。"""
    result = build_system(_make_state())

    assert "项目画像（AGENTS.md）" not in result
    assert "project_instructions" not in {k[0] for k in _CACHE}


# ── memory 段行为 ──────────────────────────────────────────────────────────────


def test_memory_section_absent_without_memory_dir():
    """无 memory_dir → memory 段完全缺席（不出现在 prompt 里）。"""
    state = _make_state(memory_dir=None)
    result = build_system(state)
    assert "# auto memory" not in result


def test_memory_policy_present_without_memory_md(tmp_path):
    """Memory policy is stable and does not depend on MEMORY.md existing."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    # 不创建 MEMORY.md
    state = _make_state(memory_dir=str(mem_dir))
    result = build_system(state)
    assert "# auto memory" in result
    assert "Memory Index" not in result


def test_memory_policy_excludes_memory_md_content(tmp_path):
    """Dynamic MEMORY.md data never becomes system prompt content."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "MEMORY.md").write_text(
        "# Memory Index\n- [my-fact](my-fact.md) — 记了一件事\n",
        encoding="utf-8",
    )
    state = _make_state(memory_dir=str(mem_dir))
    result = build_system(state)
    assert "# auto memory" in result
    assert "Memory Index" not in result
    assert "记了一件事" not in result


def test_enabled_memory_adds_policy_section(tmp_path):
    """Enabled memory adds stable policy alongside the three base sections."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "MEMORY.md").write_text(
        "# Memory Index\n- [note](note.md) — a note\n",
        encoding="utf-8",
    )
    state = _make_state(memory_dir=str(mem_dir))
    result = build_system(state)
    assert "# auto memory" in result
    assert "## 可用工具" in result
    assert "## 工作目录" in result
    assert "a note" not in result


def test_memory_index_context_truncation_long(tmp_path):
    """A long MEMORY.md is truncated in index-mode request context."""
    from agent.memory.auto_memory import AutoMemory
    from agent.runtime.request_context import memory_index_context_message

    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    lines = ["# Memory Index\n"] + [f"- [e{i}](e{i}.md) — desc {i}\n" for i in range(210)]
    (mem_dir / "MEMORY.md").write_text("".join(lines), encoding="utf-8")

    message = memory_index_context_message(
        AutoMemory(mem_dir), enabled=True, recall_mode="index"
    )
    assert message is not None
    assert "WARNING" in message["content"], "超 200 行应触发截断 WARNING"


def test_memory_index_context_truncation_large_bytes(tmp_path):
    """A >25KB MEMORY.md is truncated in index-mode request context."""
    from agent.memory.auto_memory import AutoMemory
    from agent.runtime.request_context import memory_index_context_message

    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    # 构造超 25KB 的内容（含中文使字节数快速增长）
    big_line = "- [item](item.md) — " + "描述内容" * 20 + "\n"
    content = "# Memory Index\n" + big_line * 100
    assert len(content.encode("utf-8")) > 25_000
    (mem_dir / "MEMORY.md").write_text(content, encoding="utf-8")

    message = memory_index_context_message(
        AutoMemory(mem_dir), enabled=True, recall_mode="index"
    )
    assert message is not None
    assert "WARNING" in message["content"], "超 25KB 应触发截断 WARNING"


# ── 缓存行为 ──────────────────────────────────────────────────────────────────


def test_cache_hit_on_repeated_call_same_state():
    """同 state 连调两次 → _CACHE 条目数不增（第二次全命中缓存）。"""
    state = _make_state(workdir="/same/dir")
    build_system(state)
    cache_size_after_first = len(_CACHE)
    assert cache_size_after_first > 0, "第一次调用后 _CACHE 应有条目"

    build_system(state)  # 第二次
    assert len(_CACHE) == cache_size_after_first, (
        "同 state 第二次调用不应新增 _CACHE 条目（应全部命中缓存）"
    )


def test_cache_miss_on_workdir_change():
    """workdir 变 → workspace 段 cache miss，_CACHE 增加新条目。"""
    state1 = _make_state(workdir="/dir/one")
    build_system(state1)
    size_after_first = len(_CACHE)

    state2 = _make_state(workdir="/dir/two")
    build_system(state2)
    assert len(_CACHE) > size_after_first, "workdir 变后 _CACHE 应增加 workspace 的新条目"


def test_memory_policy_cache_ignores_memory_file_changes(tmp_path):
    """Creating MEMORY.md does not invalidate the stable system policy cache."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()

    state = _make_state(workdir="/wd", memory_dir=str(mem_dir))
    first = build_system(state)
    size_before_index = len(_CACHE)

    (mem_dir / "MEMORY.md").write_text("# Memory Index\n- [x](x.md) — item\n", encoding="utf-8")

    second = build_system(state)

    assert second == first
    assert "Memory Index" not in second
    assert "item" not in second
    assert len(_CACHE) == size_before_index


def test_memory_policy_cache_varies_by_recall_mode(tmp_path):
    """Selector and index modes cache distinct stable save policies."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()

    selector = build_system(
        _make_state(memory_dir=str(mem_dir), memory_recall_mode="selector")
    )
    index = build_system(
        _make_state(memory_dir=str(mem_dir), memory_recall_mode="index")
    )

    memory_keys = [k for k in _CACHE if k[0] == "memory"]
    assert len(memory_keys) == 2
    assert "Saving a memory has two steps" not in selector
    assert "Saving a memory is a two-step process" in index


# ── 必修-1b：memory 段含防漂移三段指引 ──────────────────────────────────────────


def test_memory_section_contains_drift_guide(tmp_path):
    """Stable CC-aligned policy includes access and drift verification rules."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    state = _make_state(memory_dir=str(mem_dir))
    result = build_system(state)

    assert "become stale" in result
    assert "## When to access memories" in result
    assert "## Before recommending from memory" in result


def test_memory_policy_absent_when_memory_disabled(tmp_path):
    """Disabled memory omits all memory policy regardless of directory state."""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    state = _make_state(memory_dir=str(mem_dir), memory_enabled=False)
    result = build_system(state)
    assert "# auto memory" not in result
    assert "## When to access memories" not in result


def test_section_identity_uses_cache(monkeypatch):
    """同 identity 内容连调两次 _section_identity → 第二次命中缓存（hash 不重算）。

    用 monkeypatch 计数 _fingerprint 调用来验证。
    """
    import agent.context.system_prompt as sp_mod

    call_count = {"n": 0}
    original_fp = sp_mod._fingerprint

    def counting_fingerprint(val):
        call_count["n"] += 1
        return original_fp(val)

    monkeypatch.setattr(sp_mod, "_fingerprint", counting_fingerprint)
    _CACHE.clear()

    # 第一次
    _section_identity(DEFAULT_IDENTITY)
    count_after_first = call_count["n"]
    assert count_after_first >= 1

    # 第二次：_section_identity 内先检查 key in _CACHE，命中则不再调 _fingerprint 以外的计算
    # 但 _fingerprint 本身每次都会被调（用来生成 key）；验证的是 _CACHE[key] 直接命中
    _section_identity(DEFAULT_IDENTITY)
    # cache 大小不变（没有新 key 写入）
    assert len([k for k in _CACHE if k[0] == "identity"]) == 1, (
        "同 identity 连调不应新增 cache 条目"
    )
