"""test_auto_memory.py — mock 单元测试 agent.memory.auto_memory + loop 接入。

所有测试 mock 掉 run_forked_agent + llm.chat，不打真 LLM。
验证：文件落盘、frontmatter 格式、密钥拦截、MEMORY.md 双截、manifest 读取、loop 接入、
      recall 召回（选 ≤5 / 滤已浮现 / 字节上限 / 非法 JSON / 文件缺失）、loop tier-2 注入。
"""
import json
from pathlib import Path

import pytest

from agent.memory.auto_memory import (
    AutoMemory,
    AutoMemoryConfig,
    _parse_memories_json,
    _parse_selector_json,
    _sanitize_name,
)
from agent.memory.forked_agent import ForkResult
from agent.runtime.settings import MemoryRuntimeSettings


SELECTOR_MEMORY = MemoryRuntimeSettings(enabled=True, recall_mode="selector")


# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_fork_result(memories: list[dict]) -> ForkResult:
    """构造包含 JSON 的 ForkResult，模拟 fork 正常输出。"""
    res = ForkResult()
    res.final_text = json.dumps({"memories": memories}, ensure_ascii=False)
    res.stopped = "finished"
    return res


def _make_am(tmp_path: Path, **cfg_kwargs) -> AutoMemory:
    cfg = AutoMemoryConfig(**cfg_kwargs) if cfg_kwargs else AutoMemoryConfig()
    return AutoMemory(tmp_path / "memory", cfg)


def _patch_loop_tool_pool(monkeypatch, loop, *, output: str = "ok"):
    from agent.tools.pool import ToolPool
    from agent.tools.contracts import Tool

    fake_bash = Tool(
        name="bash",
        description="fake bash",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
        },
        call=lambda tool_input, context: output,
    )
    pool = ToolPool((fake_bash,))
    monkeypatch.setattr(loop, "assemble_tool_pool", lambda *a, **kw: pool)
    return pool


# ── 基本写入 ──────────────────────────────────────────────────────────────────


def test_write_creates_memory_files(tmp_path, monkeypatch):
    """fork 返回 2 条记忆 → 写 2 个文件，frontmatter 平铺正确，type 路由正确。"""
    from agent.memory import auto_memory as am_mod

    memories = [
        {"name": "user-pref-python", "description": "偏好 Python", "type": "user", "body": "用户喜欢用 Python。"},
        {"name": "test-no-mock-db", "description": "测试别 mock 数据库", "type": "feedback",
         "body": "不要 mock 数据库。\n\n**Why:** 真实行为需要真实库。\n\n**How to apply:** 测试时用真实库。"},
    ]
    monkeypatch.setattr(am_mod, "run_forked_agent", lambda *a, **kw: _make_fork_result(memories))

    am = _make_am(tmp_path)
    result = am.write([{"role": "user", "content": "hi"}])

    assert result["written"] == 2
    assert result["skipped_secret"] == 0
    assert result["total"] == 2

    # 文件存在
    mem_dir = tmp_path / "memory"
    assert (mem_dir / "user-pref-python.md").exists()
    assert (mem_dir / "test-no-mock-db.md").exists()


def test_write_flat_frontmatter_format(tmp_path, monkeypatch):
    """写出的文件 frontmatter 是平铺 name/description/type（非嵌套 metadata）。

    对齐参考设计的平铺字段；偏离本 harness 自身的嵌套变体（intentional，见模块 docstring）。
    """
    from agent.memory import auto_memory as am_mod

    memories = [
        {"name": "my-project", "description": "当前项目", "type": "project", "body": "在做 coding agent eval。"},
    ]
    monkeypatch.setattr(am_mod, "run_forked_agent", lambda *a, **kw: _make_fork_result(memories))

    am = _make_am(tmp_path)
    am.write([])

    content = (tmp_path / "memory" / "my-project.md").read_text(encoding="utf-8")
    # 平铺 frontmatter
    assert "---\n" in content
    assert "name: my-project\n" in content
    assert "description: 当前项目\n" in content
    assert "type: project\n" in content
    # 不是嵌套 metadata:
    assert "metadata:" not in content
    # body 在 frontmatter 之后
    assert "在做 coding agent eval。" in content


def test_write_type_routing(tmp_path, monkeypatch):
    """四类 type 路由正确（user/feedback/project/reference 各写进文件）。"""
    from agent.memory import auto_memory as am_mod

    for mem_type in ("user", "feedback", "project", "reference"):
        memories = [
            {"name": f"test-{mem_type}", "description": f"desc-{mem_type}",
             "type": mem_type, "body": f"body for {mem_type}"},
        ]
        monkeypatch.setattr(am_mod, "run_forked_agent", lambda *a, mems=memories, **kw: _make_fork_result(mems))
        am = _make_am(tmp_path / mem_type)
        result = am.write([])
        assert result["written"] == 1
        written_file = tmp_path / mem_type / "memory" / f"test-{mem_type}.md"
        assert written_file.exists()
        text = written_file.read_text(encoding="utf-8")
        assert f"type: {mem_type}\n" in text


# ── 密钥拦截 ──────────────────────────────────────────────────────────────────


def test_write_skips_secret_body(tmp_path, monkeypatch):
    """body 含密钥 → 跳过写盘、skipped_secret 计数正确、文件不存在。"""
    from agent.memory import auto_memory as am_mod

    aws_key = "AK" + "IAIOSFODNN7EXAMPLE"
    memories = [
        {"name": "with-secret", "description": "含密钥", "type": "reference",
         "body": f"服务地址 https://example.com key={aws_key}"},
        {"name": "clean-memory", "description": "干净", "type": "user", "body": "用户喜欢 Python。"},
    ]
    monkeypatch.setattr(am_mod, "run_forked_agent", lambda *a, **kw: _make_fork_result(memories))

    am = _make_am(tmp_path)
    result = am.write([])

    # 含密钥的跳过，干净的写入
    assert result["written"] == 1
    assert result["skipped_secret"] == 1
    assert result["total"] == 2
    # 含密钥的文件不存在
    assert not (tmp_path / "memory" / "with-secret.md").exists()
    # 干净的文件存在
    assert (tmp_path / "memory" / "clean-memory.md").exists()


# ── MEMORY.md 索引更新与双截断 ────────────────────────────────────────────────


def test_write_creates_memory_md_index(tmp_path, monkeypatch):
    """写入后 MEMORY.md 索引含新条目。"""
    from agent.memory import auto_memory as am_mod

    memories = [
        {"name": "ref-docs", "description": "文档链接", "type": "reference",
         "body": "https://example.com/docs"},
    ]
    monkeypatch.setattr(am_mod, "run_forked_agent", lambda *a, **kw: _make_fork_result(memories))

    am = _make_am(tmp_path)
    am.write([])

    idx = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "ref-docs.md" in idx
    assert "文档链接" in idx


def test_update_index_dedup_replaces_existing_line(tmp_path, monkeypatch):
    """同 name 的条目第二次写入 → 替换旧行，不追加重复行。"""
    from agent.memory import auto_memory as am_mod

    am = _make_am(tmp_path)
    am.memory_dir.mkdir(parents=True, exist_ok=True)

    # 先写一次
    memories = [{"name": "my-note", "description": "旧描述", "type": "user", "body": "旧内容"}]
    monkeypatch.setattr(am_mod, "run_forked_agent", lambda *a, **kw: _make_fork_result(memories))
    am.write([])

    # 再写一次（描述更新）
    memories2 = [{"name": "my-note", "description": "新描述", "type": "user", "body": "新内容"}]
    monkeypatch.setattr(am_mod, "run_forked_agent", lambda *a, **kw: _make_fork_result(memories2))
    am.write([])

    idx = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    # 只有一行 my-note.md
    assert idx.count("my-note.md") == 1, f"期望去重后只有 1 行，实际:\n{idx}"
    assert "新描述" in idx
    assert "旧描述" not in idx


def test_truncate_index_for_injection_by_lines(tmp_path):
    """P0: truncate_index_for_injection（读时注入）按行截断并追加 WARNING。
    磁盘 MEMORY.md 保持完整（_update_index 不再截断）。
    """
    cfg = AutoMemoryConfig(max_entrypoint_lines=5, max_entrypoint_bytes=100_000)
    am = AutoMemory(tmp_path / "memory", cfg)

    # 构造 6 行内容（超过 5 行上限）
    content = "# Memory Index\n" + "".join(f"- [e{i}](e{i}.md) — desc\n" for i in range(6))
    result = am.truncate_index_for_injection(content)

    assert "WARNING" in result, f"期望截断 WARNING，实际:\n{result}"
    assert "max_entrypoint_lines=5" in result


def test_truncate_index_for_injection_by_bytes(tmp_path):
    """P0: truncate_index_for_injection 按字节截断并追加 WARNING。"""
    cfg = AutoMemoryConfig(max_entrypoint_lines=10_000, max_entrypoint_bytes=200)
    am = AutoMemory(tmp_path / "memory", cfg)

    # 构造超 200 字节的内容（含中文使字节数快速增长）
    content = "# Memory Index\n" + "".join(f"- [e{i}](e{i}.md) — 长描述{'描' * 10}\n" for i in range(5))
    result = am.truncate_index_for_injection(content)

    assert "WARNING" in result
    assert len(result.encode("utf-8")) < 500  # 截断后不会太大


def test_write_does_not_truncate_on_disk(tmp_path, monkeypatch):
    """P0: _update_index 写盘时不截断 —— CC truncateEntrypointContent 是读时注入（CC memdir.ts:296）。"""
    from agent.memory import auto_memory as am_mod

    cfg = AutoMemoryConfig(max_entrypoint_lines=5, max_entrypoint_bytes=100_000)
    am = AutoMemory(tmp_path / "memory", cfg)

    # 写 6 条，超过 5 行上限
    for i in range(6):
        memories = [{"name": f"mem-{i:03d}", "description": f"desc {i}", "type": "user",
                     "body": f"body {i}"}]
        monkeypatch.setattr(am_mod, "run_forked_agent", lambda *a, mems=memories, **kw: _make_fork_result(mems))
        am.write([])

    idx = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    # 磁盘上必须有全部 6 条，不能被截断到 5 条
    for i in range(6):
        assert f"mem-{i:03d}.md" in idx, f"磁盘 MEMORY.md 丢失条目 mem-{i:03d}，P0 截断 bug 复现"


# ── manifest 只读前 30 行 ─────────────────────────────────────────────────────


def test_scan_manifest_reads_only_frontmatter_max_lines(tmp_path):
    """_scan_manifest 只读每个文件前 frontmatter_max_lines 行（30），不读完整大文件。"""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()

    # 写一个 frontmatter 在前 5 行、第 35 行才有 description 字段的文件
    lines = ["---\n", "name: tricky\n", "type: user\n", "---\n", "\n"]
    lines += [f"# 正文行 {j}\n" for j in range(30)]   # 行 5~34
    lines += ["description: 这行超过30行\n"]            # 行 35（超过 frontmatter_max_lines）
    (mem_dir / "tricky.md").write_text("".join(lines), encoding="utf-8")

    cfg = AutoMemoryConfig(frontmatter_max_lines=30)
    am = AutoMemory(mem_dir, cfg)
    manifest = am._scan_manifest()

    # description 字段在第 35 行，超出 30 行读取上限，不应出现在 manifest 里
    assert "这行超过30行" not in manifest
    # 但 name 应该被读到
    assert "tricky" in manifest


# ── 空 memories list ──────────────────────────────────────────────────────────


def test_write_empty_memories_no_files_no_error(tmp_path, monkeypatch):
    """fork 返回空 memories list → 不写文件、不报错、返回全零计数。"""
    from agent.memory import auto_memory as am_mod

    monkeypatch.setattr(am_mod, "run_forked_agent", lambda *a, **kw: _make_fork_result([]))

    am = _make_am(tmp_path)
    result = am.write([{"role": "user", "content": "nothing to remember"}])

    assert result == {"written": 0, "skipped_secret": 0, "total": 0}
    # memory_dir 可能被创建，但不应有 .md 文件（除 MEMORY.md 外）
    mem_dir = tmp_path / "memory"
    if mem_dir.exists():
        md_files = [f for f in mem_dir.glob("*.md") if f.name != "MEMORY.md"]
        assert md_files == [], f"期望无记忆文件，实际：{md_files}"


# ── _parse_memories_json 健壮性 ────────────────────────────────────────────────


def test_parse_memories_json_raw():
    raw = '{"memories": [{"name": "n", "description": "d", "type": "user", "body": "b"}]}'
    result = _parse_memories_json(raw)
    assert len(result) == 1
    assert result[0]["name"] == "n"


def test_parse_memories_json_with_code_block():
    wrapped = '```json\n{"memories": [{"name": "n2", "description": "d2", "type": "feedback", "body": "b2"}]}\n```'
    result = _parse_memories_json(wrapped)
    assert len(result) == 1
    assert result[0]["type"] == "feedback"


def test_parse_memories_json_with_preamble():
    text = '以下是提取的记忆：\n{"memories": []}\n谢谢。'
    result = _parse_memories_json(text)
    assert result == []


def test_parse_memories_json_invalid_returns_empty():
    assert _parse_memories_json("不是 JSON") == []
    assert _parse_memories_json("") == []
    assert _parse_memories_json("{bad json}") == []


def test_parse_memories_json_empty_text():
    assert _parse_memories_json("") == []


# ── _sanitize_name ────────────────────────────────────────────────────────────


def test_sanitize_name_basic():
    assert _sanitize_name("hello-world") == "hello-world"
    assert _sanitize_name("Hello World") == "hello-world"
    assert _sanitize_name("user_pref_python") == "user-pref-python"


def test_sanitize_name_strips_edge_hyphens():
    assert _sanitize_name("-leading") == "leading"
    assert _sanitize_name("trailing-") == "trailing"


def test_sanitize_name_empty_falls_back():
    # P1-3: fallback 改为 "untitled-memory"，旧的 "memory" → MEMORY.md 碰撞
    assert _sanitize_name("") == "untitled-memory"
    assert _sanitize_name("---") == "untitled-memory"


# ── loop 接入 ─────────────────────────────────────────────────────────────────


def test_loop_auto_memory_none_no_side_effects(monkeypatch):
    """auto_memory=None 时 loop 无副作用（不调 write）。"""
    from agent import loop, llm
    from conftest import end_turn_resp, CaptureSink
    from obs.trace import set_sink

    set_sink(CaptureSink())
    monkeypatch.setattr(llm, "chat", lambda *a, **kw: end_turn_resp("done"))

    # 没传 auto_memory，不应报任何 auto_memory 错误
    result = loop.run_task("test task", max_turns=2, trace=False, auto_memory=None)
    assert result == "done"


def test_loop_auto_memory_called_once_at_end_turn(tmp_path, monkeypatch):
    """auto_memory 非 None 时，end_turn 后调一次 write（不调多次）。"""
    from agent import loop, llm
    from conftest import end_turn_resp, CaptureSink
    from obs.trace import set_sink
    from agent.memory import auto_memory as am_mod

    set_sink(CaptureSink())
    monkeypatch.setattr(llm, "chat", lambda *a, **kw: end_turn_resp("done"))
    monkeypatch.setattr(am_mod, "run_forked_agent", lambda *a, **kw: _make_fork_result([]))

    call_count = {"n": 0}

    class FakeAutoMemory:
        def write(self, messages, system=""):
            call_count["n"] += 1
            return {"written": 0, "skipped_secret": 0, "total": 0}

    am = FakeAutoMemory()
    result = loop.run_task("test task", max_turns=2, trace=False, auto_memory=am)

    assert result == "done"
    assert call_count["n"] == 1, f"期望 write 被调一次，实际 {call_count['n']} 次"


def test_loop_auto_memory_error_does_not_crash_main_task(monkeypatch):
    """auto_memory.write 抛异常 → 主任务仍正常返回，不崩溃。"""
    from agent import loop, llm
    from conftest import end_turn_resp, CaptureSink
    from obs.trace import set_sink

    set_sink(CaptureSink())
    monkeypatch.setattr(llm, "chat", lambda *a, **kw: end_turn_resp("done"))

    class BrokenAutoMemory:
        def write(self, messages, system=""):
            raise RuntimeError("模拟记忆写入失败")

    result = loop.run_task("test task", max_turns=2, trace=False, auto_memory=BrokenAutoMemory())
    assert result == "done", "记忆写入失败不应影响主任务返回值"


def test_loop_auto_memory_not_called_on_max_turns(monkeypatch):
    """达到 max_turns 但未 end_turn → auto_memory.write 不被调。"""
    from agent import loop, llm
    from conftest import tool_use_resp, CaptureSink
    from obs.trace import set_sink

    set_sink(CaptureSink())
    monkeypatch.setattr(llm, "chat", lambda *a, **kw: tool_use_resp())
    _patch_loop_tool_pool(monkeypatch, loop, output="output")

    call_count = {"n": 0}

    class TrackingAutoMemory:
        def write(self, messages, system=""):
            call_count["n"] += 1
            return {"written": 0, "skipped_secret": 0, "total": 0}

    result = loop.run_task("infinite task", max_turns=2, trace=False, auto_memory=TrackingAutoMemory())

    assert "达到最大轮次" in result
    assert call_count["n"] == 0, "未 end_turn 时不应触发 auto_memory.write"


# ── P1-3 _sanitize_name 强化 ──────────────────────────────────────────────────


def test_sanitize_name_path_traversal_slash():
    """路径穿越：斜杠不能存活于 safe 文件名中。"""
    result = _sanitize_name("../../etc/passwd")
    assert "/" not in result
    assert "\\" not in result
    assert ".." not in result


def test_sanitize_name_windows_memory_collision():
    """'memory' → MEMORY.md 碰撞（Windows 大小写不敏感），必须前缀 note-。"""
    result = _sanitize_name("memory")
    assert result == "note-memory"


def test_sanitize_name_windows_reserved_con():
    """Windows 保留文件名 CON 必须被阻断。"""
    assert _sanitize_name("con") == "note-con"


def test_sanitize_name_windows_reserved_nul():
    assert _sanitize_name("nul") == "note-nul"


def test_sanitize_name_length_cap():
    """长度上限 80 字符。"""
    long_name = "a" * 100
    result = _sanitize_name(long_name)
    assert len(result) <= 80


# ── P1-1 description 也被扫密钥 ──────────────────────────────────────────────


def test_write_skips_entry_with_secret_in_description(tmp_path, monkeypatch):
    """P1-1: description 含密钥 → 跳过（description 进 MEMORY.md 索引同样有泄露风险）。"""
    from agent.memory import auto_memory as am_mod

    aws_key = "AK" + "IAIOSFODNN7EXAMPLE"
    memories = [
        {"name": "bad-ref", "description": f"key is {aws_key}", "type": "reference", "body": "clean body"},
    ]
    monkeypatch.setattr(am_mod, "run_forked_agent", lambda *a, **kw: _make_fork_result(memories))

    am = _make_am(tmp_path)
    result = am.write([])

    assert result["skipped_secret"] == 1
    assert result["written"] == 0
    idx_path = tmp_path / "memory" / "MEMORY.md"
    if idx_path.exists():
        assert "bad-ref" not in idx_path.read_text(encoding="utf-8")


# ── _parse_selector_json ──────────────────────────────────────────────────────


def test_parse_selector_json_basic():
    raw = '{"selected": ["foo.md", "bar.md"]}'
    assert _parse_selector_json(raw) == ["foo.md", "bar.md"]


def test_parse_selector_json_code_block():
    wrapped = '```json\n{"selected": ["a.md"]}\n```'
    assert _parse_selector_json(wrapped) == ["a.md"]


def test_parse_selector_json_empty_selected():
    assert _parse_selector_json('{"selected": []}') == []


def test_parse_selector_json_invalid_returns_empty():
    assert _parse_selector_json("not json at all") == []
    assert _parse_selector_json("") == []
    assert _parse_selector_json('{"selected": "bad type"}') == []


# ── recall() 单元测试（mock llm.chat）────────────────────────────────────────


def _write_memory_file(mem_dir: Path, name: str, desc: str = "desc", mtype: str = "user"):
    """辅助：在 mem_dir 下写一条 frontmatter 记忆文件。"""
    mem_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {desc}\ntype: {mtype}\n---\n\nbody of {name}\n"
    path = mem_dir / f"{name}.md"
    path.write_text(content, encoding="utf-8")
    return path


def _fake_llm_chat_selecting(filenames: list[str]):
    """构造 mock llm.chat，返回选中指定 filenames 的选择器 JSON。"""
    class _MockResp:
        class _Block:
            type = "text"
            text = json.dumps({"selected": filenames})
        content = [_Block()]
    return lambda *a, **kw: _MockResp()


def test_recall_empty_dir_returns_empty(tmp_path):
    """memory_dir 不存在 → recall() 返 []。"""
    am = AutoMemory(tmp_path / "memory")
    result = am.recall(query="anything", already_surfaced=set())
    assert result == []


def test_recall_no_files_returns_empty(tmp_path):
    """memory_dir 存在但无 .md 文件（只有 MEMORY.md）→ []。"""
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "MEMORY.md").write_text("# Memory Index\n", encoding="utf-8")
    am = AutoMemory(mem_dir)
    result = am.recall(query="anything", already_surfaced=set())
    assert result == []


def test_recall_rejects_out_of_bounds_selector_names(tmp_path, monkeypatch):
    """P1-2 安全回归：选择器（LLM，输出不可信）返回越界文件名（路径穿越/绝对路径/不存在）
    → recall() 只读 manifest 白名单内的真实文件，越界名全丢，绝不读 memory_dir 外。"""
    import agent.memory.auto_memory as am_mod

    mem_dir = tmp_path / "memory"
    _write_memory_file(mem_dir, "legit", desc="real memory")
    # 目录外预埋"机密"文件，确认绝不会被读出
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")

    # 选择器幻觉出各种越界名 + 一个合法名
    monkeypatch.setattr(am_mod, "_llm_mod", type("_M", (), {
        "chat": staticmethod(_fake_llm_chat_selecting([
            "../secret.txt", "../../etc/passwd", "/etc/passwd",
            str(secret), "secret.txt", "nonexistent.md", "legit.md",
        ]))
    })())

    am = AutoMemory(mem_dir)
    result = am.recall(query="test", already_surfaced=set())

    paths = [r["path"] for r in result]
    contents = "".join(r["content"] for r in result)
    assert len(result) == 1 and all("legit.md" in p for p in paths), \
        f"应只返回白名单内的 legit.md，实际 {paths}"
    assert "TOP SECRET" not in contents, "绝不能读出 memory_dir 外的文件"


def test_recall_selects_up_to_five(tmp_path, monkeypatch):
    """6 个文件，选择器返回 6 个文件名 → 只取前 5（≤5 上限）。"""
    import agent.memory.auto_memory as am_mod

    mem_dir = tmp_path / "memory"
    for i in range(6):
        _write_memory_file(mem_dir, f"file-{i:02d}", desc=f"desc {i}")

    # 选择器返回全部 6 个
    six_names = [f"file-{i:02d}.md" for i in range(6)]
    monkeypatch.setattr(am_mod, "_llm_mod", type("_M", (), {
        "chat": staticmethod(_fake_llm_chat_selecting(six_names))
    })())

    am = AutoMemory(mem_dir)
    result = am.recall(query="test", already_surfaced=set())
    assert len(result) <= 5, f"recall 结果应 ≤5，实际 {len(result)}"


def test_recall_filters_already_surfaced(tmp_path, monkeypatch):
    """already_surfaced 里的路径在选择器调用前被过滤掉，不进 manifest。"""
    import agent.memory.auto_memory as am_mod

    mem_dir = tmp_path / "memory"
    p1 = _write_memory_file(mem_dir, "mem-a", desc="memory a")
    _write_memory_file(mem_dir, "mem-b", desc="memory b")

    # 仅选 mem-b（mem-a 已浮现）
    monkeypatch.setattr(am_mod, "_llm_mod", type("_M", (), {
        "chat": staticmethod(_fake_llm_chat_selecting(["mem-b.md"]))
    })())

    am = AutoMemory(mem_dir)
    result = am.recall(query="test", already_surfaced={str(p1)})

    paths = [r["path"] for r in result]
    assert not any("mem-a" in p for p in paths), "mem-a 已浮现，不应被召回"
    assert any("mem-b" in p for p in paths), "mem-b 应被召回"


def test_recall_byte_limit_per_file(tmp_path, monkeypatch):
    """文件超 4096 字节 → 内容被截断 + 截断标注。"""
    import agent.memory.auto_memory as am_mod

    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    big_content = "---\nname: big\ndescription: big file\ntype: user\n---\n\n" + "x" * 8000
    (mem_dir / "big.md").write_text(big_content, encoding="utf-8")

    monkeypatch.setattr(am_mod, "_llm_mod", type("_M", (), {
        "chat": staticmethod(_fake_llm_chat_selecting(["big.md"]))
    })())

    am = AutoMemory(mem_dir)
    result = am.recall(query="test", already_surfaced=set())

    assert len(result) == 1
    content = result[0]["content"]
    assert "已截断" in content, "超 4096 字节应有截断标注"
    assert len(content.encode("utf-8")) < 6000, "截断后内容不应远超 4096 字节"


def test_recall_invalid_json_returns_empty(tmp_path, monkeypatch):
    """选择器返非法 JSON → recall 返 []，不崩。"""
    import agent.memory.auto_memory as am_mod

    mem_dir = tmp_path / "memory"
    _write_memory_file(mem_dir, "note", desc="some note")

    class _BadResp:
        class _Block:
            type = "text"
            text = "这不是 JSON"
        content = [_Block()]

    monkeypatch.setattr(am_mod, "_llm_mod", type("_M", (), {
        "chat": staticmethod(lambda *a, **kw: _BadResp())
    })())

    am = AutoMemory(mem_dir)
    result = am.recall(query="test", already_surfaced=set())
    assert result == []


def test_recall_missing_file_skipped(tmp_path, monkeypatch):
    """文件在扫描后被删除 → 跳过不崩，其他文件仍返回。"""
    import agent.memory.auto_memory as am_mod

    mem_dir = tmp_path / "memory"
    p_exists = _write_memory_file(mem_dir, "exists", desc="exists")
    p_gone = _write_memory_file(mem_dir, "gone", desc="gone")

    # 选择器选两个（一个即将消失）
    monkeypatch.setattr(am_mod, "_llm_mod", type("_M", (), {
        "chat": staticmethod(_fake_llm_chat_selecting(["gone.md", "exists.md"]))
    })())

    # 删除 gone.md（模拟文件消失）
    p_gone.unlink()

    am = AutoMemory(mem_dir)
    result = am.recall(query="test", already_surfaced=set())

    # 不应崩溃；exists 应返回
    paths = [r["path"] for r in result]
    assert any("exists" in p for p in paths), "exists.md 应被正常召回"
    assert not any("gone" in p for p in paths), "gone.md 已删除，不应出现"


def test_recall_selector_error_returns_empty(tmp_path, monkeypatch):
    """选择器 LLM 调用抛异常 → recall 返 []，不崩。"""
    import agent.memory.auto_memory as am_mod

    mem_dir = tmp_path / "memory"
    _write_memory_file(mem_dir, "note", desc="some note")

    def _raise(*a, **kw):
        raise RuntimeError("模拟 LLM 故障")

    monkeypatch.setattr(am_mod, "_llm_mod", type("_M", (), {
        "chat": staticmethod(_raise)
    })())

    am = AutoMemory(mem_dir)
    result = am.recall(query="test", already_surfaced=set())
    assert result == []


# ── loop Tier-2 注入测试 ──────────────────────────────────────────────────────


def _make_recall_mock(mems: list[dict]):
    """构造返回固定 mems 的 recall mock。"""
    class _FakeAM:
        memory_dir = Path("/fake/memory")

        def write(self, messages, system=""):
            return {"written": 0, "skipped_secret": 0, "total": 0}

        def recall(self, query, already_surfaced, recent_tools=None):
            # 过滤已浮现
            return [m for m in mems if m["path"] not in already_surfaced]
    return _FakeAM()


def test_loop_tier2_recall_on_turn_1(monkeypatch, tmp_path):
    """auto_memory 非 None → 第 1 轮 recall 被调用，注入内容出现在 messages 尾部 user 消息里。"""
    from agent import loop, llm
    from conftest import end_turn_resp, CaptureSink
    from obs.trace import set_sink

    set_sink(CaptureSink())

    captured_messages = {}

    def fake_chat(messages, system="", tools=None, max_tokens=4096, model=None, purpose="agent", **kwargs):
        captured_messages["last"] = [m for m in messages]
        captured_messages["system"] = system
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(loop, "build_system", lambda state: "SYSTEM")

    mems = [{"path": "/fake/memory/note.md", "content": "重要记忆内容"}]
    am = _make_recall_mock(mems)

    result = loop.run_task("my task", max_turns=2, trace=False, auto_memory=am,
                           memory_settings=SELECTOR_MEMORY)
    assert result == "done"

    # 注入内容应出现在 messages 尾部 user 消息里（非 system）
    msgs = captured_messages["last"]
    system_str = captured_messages["system"]
    assert "重要记忆内容" not in system_str, "召回内容不应进 system prompt（缓存铁律）"
    # 找 user 消息中包含召回的那条
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    has_recall = any(
        "重要记忆内容" in str(m.get("content", ""))
        for m in user_msgs
    )
    assert has_recall, "召回内容应注入到 messages 尾部 user 消息里"


def test_loop_tier2_throttle(monkeypatch, tmp_path):
    """turn 1 + turn 6 注入，turn 2-5 不重注（TURNS_BETWEEN_ATTACHMENTS=5 节流）。"""
    from agent import loop, llm
    from conftest import tool_use_resp, end_turn_resp, CaptureSink
    from obs.trace import set_sink

    set_sink(CaptureSink())

    # 5 轮 tool_use，第 6 轮 end_turn
    call_n = {"n": 0}
    def fake_chat(*a, **kw):
        call_n["n"] += 1
        if call_n["n"] <= 5:
            return tool_use_resp()
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    _patch_loop_tool_pool(monkeypatch, loop, output="ok")
    monkeypatch.setattr(loop, "build_system", lambda state: "SYSTEM")

    recall_turns = []

    class _ThrottleAM:
        memory_dir = Path("/fake/memory")

        def write(self, messages, system=""):
            return {"written": 0, "skipped_secret": 0, "total": 0}

        def recall(self, query, already_surfaced, recent_tools=None):
            recall_turns.append(call_n["n"])  # 记录哪轮被调
            return []  # 不注入内容

    result = loop.run_task("task", max_turns=10, trace=False, auto_memory=_ThrottleAM(),
                           memory_settings=SELECTOR_MEMORY)
    assert result == "done"
    # recall 在 turn 1 和 turn 6 各调一次（call_n["n"] 对应 llm.chat 调用计数）
    # turn_no=1 → call_n["n"]=0（recall 在 chat 前）；故 recall_turns 有 0 和 5 两条
    # 实际：recall 在 chat 前调，此时 call_n["n"] 还未加（加在 fake_chat 里）
    # → recall_turns 共 2 次
    assert len(recall_turns) == 2, (
        f"期望 recall 被调 2 次（turn 1 + turn 6），实际：{len(recall_turns)} 次"
    )


def test_loop_tier2_no_duplicate_surfaced(monkeypatch):
    """同一条记忆 recall 第一轮返回后，第二次召回（turn 6）时该路径被过滤掉。"""
    from agent import loop, llm
    from conftest import tool_use_resp, end_turn_resp, CaptureSink
    from obs.trace import set_sink

    set_sink(CaptureSink())

    call_n = {"n": 0}
    def fake_chat(*a, **kw):
        call_n["n"] += 1
        if call_n["n"] <= 5:
            return tool_use_resp()
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    _patch_loop_tool_pool(monkeypatch, loop, output="ok")
    monkeypatch.setattr(loop, "build_system", lambda state: "SYSTEM")

    surfaced_on_second_call = []

    class _DedupeAM:
        memory_dir = Path("/fake/memory")
        _call_count = 0

        def write(self, messages, system=""):
            return {"written": 0, "skipped_secret": 0, "total": 0}

        def recall(self, query, already_surfaced, recent_tools=None):
            self.__class__._call_count += 1
            if self.__class__._call_count == 2:
                # 第二次调用时记录 already_surfaced（应含第一次返回的路径）
                surfaced_on_second_call.extend(already_surfaced)
            return [{"path": "/fake/memory/note.md", "content": "note"}]

    result = loop.run_task("task", max_turns=10, trace=False, auto_memory=_DedupeAM(),
                           memory_settings=SELECTOR_MEMORY)
    assert result == "done"
    # 第二次 recall 调用时，第一次返回的 /fake/memory/note.md 应已在 already_surfaced 里
    assert "/fake/memory/note.md" in surfaced_on_second_call, (
        "第一次浮现的路径应在第二次 recall 的 already_surfaced 里"
    )


def test_loop_tier2_injection_not_in_system(monkeypatch):
    """recall 注入内容不进 system prompt（缓存铁律）。"""
    from agent import loop, llm
    from conftest import end_turn_resp, CaptureSink
    from obs.trace import set_sink

    set_sink(CaptureSink())

    captured = {}

    def fake_chat(messages, system="", tools=None, max_tokens=4096, model=None, purpose="agent", **kwargs):
        captured["system"] = system
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(loop, "build_system", lambda state: "THE_SYSTEM_PROMPT")

    mems = [{"path": "/fake/note.md", "content": "SECRET_RECALL_CONTENT"}]
    am = _make_recall_mock(mems)

    loop.run_task("task", max_turns=2, trace=False, auto_memory=am,
                  memory_settings=SELECTOR_MEMORY)

    assert "SECRET_RECALL_CONTENT" not in captured.get("system", ""), (
        "召回内容不应出现在 system prompt（缓存铁律）"
    )


def test_loop_tier2_messages_tool_pair_valid(monkeypatch):
    """recall 注入后 messages 仍合法：无悬空 tool_use（tool_use/tool_result 配对完整）。"""
    from agent import loop, llm
    from conftest import tool_use_resp, end_turn_resp, CaptureSink
    from obs.trace import set_sink

    set_sink(CaptureSink())

    call_n = {"n": 0}
    def fake_chat(*a, **kw):
        call_n["n"] += 1
        if call_n["n"] == 1:
            return tool_use_resp("bash", {}, "id1")
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    _patch_loop_tool_pool(monkeypatch, loop, output="ok")
    monkeypatch.setattr(loop, "build_system", lambda state: "SYSTEM")

    mems = [{"path": "/fake/note.md", "content": "recall content"}]
    am = _make_recall_mock(mems)

    _, msgs = loop.run_task("task", max_turns=5, trace=False, auto_memory=am,
                            memory_settings=SELECTOR_MEMORY, return_messages=True)

    # 验证：每个 tool_use 都有配对的 tool_result
    tool_use_ids = set()
    tool_result_ids = set()
    for m in msgs:
        content = m.get("content", [])
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict):
                    if b.get("type") == "tool_use":
                        tool_use_ids.add(b["id"])
                    elif b.get("type") == "tool_result":
                        tool_result_ids.add(b["tool_use_id"])
                else:
                    if getattr(b, "type", None) == "tool_use":
                        tool_use_ids.add(getattr(b, "id", None))
    # 所有 tool_use_id 都应有对应的 tool_result
    assert tool_use_ids <= tool_result_ids, (
        f"tool_use 配对不完整：tool_use_ids={tool_use_ids}, tool_result_ids={tool_result_ids}"
    )


def test_loop_memory_policy_in_system_without_dynamic_index(monkeypatch, tmp_path):
    """The loop keeps stable policy in system and excludes MEMORY.md data."""
    from agent import loop, llm
    from conftest import end_turn_resp, CaptureSink
    from obs.trace import set_sink
    from agent.memory.auto_memory import AutoMemory
    from agent.memory import auto_memory as am_mod

    set_sink(CaptureSink())

    # 建真实 MEMORY.md
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "MEMORY.md").write_text(
        "# Memory Index\n- [my-fact](my-fact.md) — 重要事实\n", encoding="utf-8"
    )

    captured = {}

    def fake_chat(messages, system="", tools=None, max_tokens=4096, model=None, purpose="agent", **kwargs):
        captured["system"] = system
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    # selector recall is irrelevant here; only the system/data boundary matters.
    monkeypatch.setattr(am_mod, "_llm_mod", type("_M", (), {
        "chat": staticmethod(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("不应被调")))
    })())

    am = AutoMemory(mem_dir)
    loop.run_task("task", max_turns=2, trace=False, auto_memory=am,
                  memory_settings=SELECTOR_MEMORY)

    system_str = captured.get("system", "")
    assert "# auto memory" in system_str
    assert "Memory Index" not in system_str
    assert "重要事实" not in system_str


# ── 必修-1a：recall 返回 mtime + loop 新鲜度头 ────────────────────────────────


def test_recall_returns_mtime(tmp_path, monkeypatch):
    """必修-1a：recall() 返回的每条 dict 含 mtime 字段（float seconds since epoch）。"""
    import agent.memory.auto_memory as am_mod

    mem_dir = tmp_path / "memory"
    _write_memory_file(mem_dir, "note", desc="a note")

    monkeypatch.setattr(am_mod, "_llm_mod", type("_M", (), {
        "chat": staticmethod(_fake_llm_chat_selecting(["note.md"]))
    })())

    am = AutoMemory(mem_dir)
    result = am.recall(query="test", already_surfaced=set())

    assert len(result) == 1
    assert "mtime" in result[0], "recall 结果应含 mtime 字段（必修-1a）"
    assert isinstance(result[0]["mtime"], float), "mtime 应为 float（秒级时间戳）"
    assert result[0]["mtime"] > 0, "mtime 应为正值"


def test_loop_tier2_freshness_header_today(monkeypatch, tmp_path):
    """必修-1a：recall mtime=今天 → 注入文本含「今天」+ path（对齐 CC attachments.ts:2327-2332）。"""
    import time
    from pathlib import Path
    from agent import loop, llm
    from conftest import end_turn_resp, CaptureSink
    from obs.trace import set_sink

    set_sink(CaptureSink())
    captured = {}

    def fake_chat(messages, **kw):
        captured["last"] = messages
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(loop, "build_system", lambda state: "SYSTEM")

    now = time.time()
    mems = [{"path": "/fake/memory/note.md", "content": "内容", "mtime": now}]

    class _FakeAM:
        memory_dir = Path("/fake/memory")
        def write(self, messages, system=""): return {}
        def recall(self, query, already_surfaced, recent_tools=None): return mems

    loop.run_task("task", max_turns=2, trace=False, auto_memory=_FakeAM(),
                  memory_settings=SELECTOR_MEMORY)

    user_text = str(captured.get("last", []))
    assert "今天" in user_text, "mtime=今天 → 注入含「今天」"
    assert "note.md" in user_text, "注入含 path"


def test_loop_tier2_freshness_header_stale(monkeypatch, tmp_path):
    """必修-1a：recall mtime > 1天前 → 注入含 staleness 提示（对齐 CC memoryAge.ts:33-41）。"""
    import time
    from pathlib import Path
    from agent import loop, llm
    from conftest import end_turn_resp, CaptureSink
    from obs.trace import set_sink

    set_sink(CaptureSink())
    captured = {}

    def fake_chat(messages, **kw):
        captured["last"] = messages
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(loop, "build_system", lambda state: "SYSTEM")

    stale_mtime = time.time() - 5 * 86400  # 5 天前
    mems = [{"path": "/fake/memory/note.md", "content": "内容", "mtime": stale_mtime}]

    class _FakeAM:
        memory_dir = Path("/fake/memory")
        def write(self, messages, system=""): return {}
        def recall(self, query, already_surfaced, recent_tools=None): return mems

    loop.run_task("task", max_turns=2, trace=False, auto_memory=_FakeAM(),
                  memory_settings=SELECTOR_MEMORY)

    user_text = str(captured.get("last", []))
    # staleness 词（对齐 CC memoryFreshnessText：points-in-time, outdated）
    assert "过时" in user_text or "快照" in user_text, "stale memory → 注入含 staleness 词（过时/快照）"


# ── 必修-2：manifest 行以 filename 领头 ───────────────────────────────────────


def test_do_recall_manifest_filename_leads(tmp_path, monkeypatch):
    """必修-2：_do_recall 拼的 manifest 行以 filename 领头（对齐 CC memoryScan.ts:84-94）。"""
    import re
    import agent.memory.auto_memory as am_mod

    mem_dir = tmp_path / "memory"
    _write_memory_file(mem_dir, "my-note", desc="test description", mtype="user")

    captured_manifest = {}

    def recording_chat(messages, system="", tools=None, max_tokens=256, model=None,
                       purpose="memory_recall", **kwargs):
        # messages[-1]["content"] 含 manifest 文本
        content = messages[-1].get("content", "")
        captured_manifest["text"] = content

        class _Resp:
            content = [type("B", (), {"type": "text",
                                      "text": '{"selected": ["my-note.md"]}'})()]
        return _Resp()

    monkeypatch.setattr(am_mod, "_llm_mod", type("_M", (), {
        "chat": staticmethod(recording_chat)
    })())

    am = AutoMemory(mem_dir)
    result = am.recall(query="test", already_surfaced=set())

    manifest_text = captured_manifest.get("text", "")
    assert "my-note.md" in manifest_text, "manifest 应含文件名"
    # 文件名领头：行格式 `- {filename} [{type}] (ISO): description`
    lines = manifest_text.splitlines()
    manifest_lines = [l for l in lines if "my-note.md" in l]
    assert manifest_lines, "manifest 中应有含 my-note.md 的行"
    assert any(re.match(r"^- my-note\.md ", l) for l in manifest_lines), (
        f"manifest 行应以 filename 领头（`- my-note.md ...`），实际: {manifest_lines}"
    )
    # recall 仍能命中（非回归）
    assert len(result) == 1 and "my-note" in result[0]["path"], "manifest 主键对齐后 recall 仍正常"


def test_loop_no_auto_memory_has_no_memory_policy(monkeypatch):
    """auto_memory=None omits the Auto Memory system policy."""
    from agent import loop, llm
    from conftest import end_turn_resp, CaptureSink
    from obs.trace import set_sink

    set_sink(CaptureSink())

    captured = {}

    def fake_chat(messages, system="", tools=None, max_tokens=4096, model=None, purpose="agent", **kwargs):
        captured["system"] = system
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    loop.run_task("task", max_turns=2, trace=False, auto_memory=None)

    assert "# auto memory" not in captured.get("system", "")
