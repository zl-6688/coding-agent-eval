"""会话运行时（G1–G4）回归 —— 纯单测、不调 LLM（离线）。

    python eval/runtime_eval/test_runtime.py

覆盖评审钉的点（docs/runtime/03-design-review.md）：
  ① 覆盖写原子性（写一半模拟中断、原文件完好）          —— P0-2 落盘模型
  ② 压缩轮落盘（messages 变短后 save→resume 一致、无残留旧行） —— P0-2 压缩×落盘
  ③ keying 稳定（同 workpath 两次→同 key；ACE_HOME override 生效，全程写 tmp 不污染真实 ~/.ace）—— P1/P1-2
  ④ resume 往返线性数组一致 + 多轮 messages 不重复       —— P0-2 重复落盘

⚠ 全程 ACE_HOME 指向 tmp、绝不写真实 home（评审 P1 / 测试纪律硬约束）。
Session.run 的 LLM 路径不在此测（留给后续 e2e）；这里只测 Project keying / SessionStore
save-resume 往返 / 多轮 messages 不重复（不实例化 Session.run）。
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── 测试纪律：导入 runtime 前就把 ACE_HOME 钉到 tmp，任何 Project 构造都落 tmp、绝不碰真实 home。
_ACE_TMP = Path(tempfile.mkdtemp(prefix="ace_test_"))
os.environ["ACE_HOME"] = str(_ACE_TMP)

from agent.runtime import Project, Session, SessionStore   # noqa: E402
from agent.runtime import project as project_mod  # noqa: E402


# ──────────────────────────────────────────────────────────────────
# ③ keying 稳定 + ACE_HOME override
# ──────────────────────────────────────────────────────────────────
def test_keying_stable_and_ace_home():
    wp = REPO   # 用本仓库做 workpath（它是 git repo，会走 git-root 分支）
    p1 = Project.from_cwd(wp)
    p2 = Project.from_cwd(wp)
    # 同 workpath 两次构造 → 同 key（稳定性是评审硬要求）。
    assert p1.key == p2.key, f"同 workpath 两次 key 应相同，得 {p1.key!r} vs {p2.key!r}"

    # 大小写不同的同一路径（Windows 盘符 / 大小写）应归一到同 key（normcase 抹平）。
    # 在 POSIX normcase 是 no-op，此断言只在 Windows 真正发力；用 resolve 后路径构造对照。
    alt = str(wp).swapcase() if os.name == "nt" else str(wp)
    p3 = Project.from_cwd(alt)
    if os.name == "nt":
        assert p1.key == p3.key, f"大小写差异应归一到同 key：{p1.key!r} vs {p3.key!r}"

    # ACE_HOME override 生效：root 在 tmp 下，绝不是真实 ~/.ace。
    assert str(p1.root) == str(_ACE_TMP), f"root 应为 ACE_HOME={_ACE_TMP}，得 {p1.root}"
    real_ace = Path.home() / ".ace"
    assert real_ace not in p1.root.parents and p1.root != real_ace, "绝不能落真实 ~/.ace"

    # 目录布局形状（对齐 CC ~/.claude/projects/<key>/{sessions,memory}）。
    assert p1.sessions_dir == _ACE_TMP / "projects" / p1.key / "sessions"
    assert p1.memory_dir == _ACE_TMP / "projects" / p1.key / "memory"

    # key 只含字母数字和 '-'（sanitize 正确）。
    assert all(c.isalnum() or c == "-" for c in p1.key), f"key 含非法字符：{p1.key!r}"
    return 1


def test_keying_git_fallback_deterministic():
    """git 探测失败必须确定性退 workpath（评审 P1-2：别让 git 偶发失败导致 key 抖动）。"""
    # monkeypatch _git_root 恒返回 None（模拟无 git/非 git），key 应稳定退到 workpath。
    orig = project_mod._git_root
    try:
        project_mod._git_root = lambda wp: None
        tmpdir = Path(tempfile.mkdtemp())
        k1 = Project.from_cwd(tmpdir).key
        k2 = Project.from_cwd(tmpdir).key
        assert k1 == k2, f"无 git 时同 workpath 两次 key 应相同：{k1!r} vs {k2!r}"
        # 退到 workpath：key 应反映 tmpdir 路径（sanitize 后含其末段名；末段名也要同样 sanitize 比对，
        # 因为 mktemp 名可能含下划线等会被转成 '-' 的字符）。
        sanitized_name = project_mod._SANITIZE.sub("-", os.path.normcase(tmpdir.name))
        assert sanitized_name in k1, \
            f"无 git 应退 workpath，key 应含 sanitize 后目录名 {sanitized_name!r}：{k1!r}"
    finally:
        project_mod._git_root = orig
    return 1


# ──────────────────────────────────────────────────────────────────
# P0-A 防回归：SessionMemory 笔记按会话挂 sessions_dir/<id>.notes.md，
#             绝不等于 memory_dir/MEMORY.md（后者留给将来 Auto Memory）
# ──────────────────────────────────────────────────────────────────
def test_session_memory_path_per_session():
    """Session.create/resume 的 memory.path 必须落在 sessions_dir/<id>.notes.md、
    不等于 memory_dir/MEMORY.md（评审 P0-A）。with_memory=True 不调 LLM 即可验 path。"""
    s = Session.create(REPO, with_memory=True)
    expected = s.project.sessions_dir / f"{s.id}.notes.md"
    assert s.memory.path == expected, f"SessionMemory 应按会话挂 {expected}，得 {s.memory.path}"

    auto_mem_index = s.project.memory_dir / "MEMORY.md"
    assert s.memory.path != auto_mem_index, \
        f"SessionMemory 绝不该撞 Auto Memory 索引 {auto_mem_index}（评审 P0-A）"
    # 与 transcript 同目录平铺（<id>.notes.md 与 <id>.jsonl 同在 sessions_dir）。
    assert s.memory.path.parent == s.project.sessions_dir, "SessionMemory 应与 transcript 同目录"

    # 同 project 不同 session → 不同笔记文件（不再互相覆盖，"会话内"语义恢复）。
    s2 = Session.create(REPO, with_memory=True)
    assert s.id != s2.id and s.memory.path != s2.memory.path, \
        "同 project 不同 session 应有各自独立的笔记文件"

    # resume 同 id → 落同一笔记文件（会话内笔记跨进程接得住）。
    sr = Session.resume(s.id, REPO, with_memory=True)
    assert sr.memory.path == s.memory.path, "resume 同 id 应挂回同一会话笔记文件"

    # with_memory=False → 无 memory（不实例化、不碰盘）。
    s3 = Session.create(REPO, with_memory=False)
    assert s3.memory is None, "with_memory=False 应无 SessionMemory"
    return 1


def _no_uuid(message):
    """比较时剥掉 uuid 元数据（store 落盘会给每条消息注入 uuid，语义比较不看它）。"""
    return {k: v for k, v in message.items() if k != "uuid"}


def test_run_appends_task_b1():
    """B1 防回归：Session.run 必须把新 task 追加进历史再传 initial_messages，否则追问没接上。

    loop.py:81 给了 initial_messages 就忽略 task → 必须靠 Session.run 把 task 注入历史。
    stub run_task 记录入参，验 fresh + resumed 两路传入的 initial_messages 末条都是新 task。
    """
    import agent.runtime.session as session_mod

    captured = {}

    def stub_run_task(task, *, initial_messages=None, return_messages=False, **kw):
        captured["task"] = task
        captured["initial_messages"] = list(initial_messages) if initial_messages else None
        # 模拟 run_task：返回 (text, messages)，messages = 传入的历史 + 一条 assistant 回复。
        msgs = list(initial_messages) + [{"role": "assistant", "content": f"答:{task}"}]
        return (f"答:{task}", msgs)

    orig = session_mod.run_task
    try:
        session_mod.run_task = stub_run_task

        # 路 1：fresh 会话（空历史）。initial_messages 末条应是新 task（等价原 [{user:task}]）。
        s = Session.create(REPO, with_memory=False)
        s.run("第一个问题")
        im = captured["initial_messages"]
        assert im and _no_uuid(im[-1]) == {"role": "user", "content": "第一个问题"}, \
            f"fresh：initial_messages 末条应是新 task，得 {im}"
        assert len(im) == 1, f"fresh 会话应只有 1 条（新 task），得 {len(im)}"

        # 路 2：同会话追问。新 task 必须接在已有历史后（不是覆盖、不是丢弃）。
        s.run("追问")
        im = captured["initial_messages"]
        assert _no_uuid(im[-1]) == {"role": "user", "content": "追问"}, \
            f"追问：initial_messages 末条应是追问 task，得末条 {im[-1]}"
        assert any(m.get("content") == "第一个问题" for m in im), "追问时旧历史应仍在场（上下文接力）"

        # 路 3：resumed 会话（盘上有历史）。run(新task) 后 initial_messages 末条仍是新 task。
        sid = s.id
        sr = Session.resume(sid, REPO, with_memory=False)
        n_before = len(sr.messages)
        assert n_before > 0, "resumed 会话应有历史"
        sr.run("resume 后的新问题")
        im = captured["initial_messages"]
        assert _no_uuid(im[-1]) == {"role": "user", "content": "resume 后的新问题"}, \
            f"resumed：initial_messages 末条应是新 task，得末条 {im[-1]}"
        assert len(im) == n_before + 1, \
            f"resumed：应在 {n_before} 条历史后追加 1 条新 task，得 {len(im)}"
    finally:
        session_mod.run_task = orig
    return 1


def test_close_dangling_tool_uses_step7():
    """step7：中止后清理悬空 tool_use → 每个 tool_use 都配上 tool_result、续作 API 合法。"""
    from agent.loop import close_dangling_tool_uses
    from agent.runtime.store import _to_jsonable
    import json

    # 末尾 assistant 含 2 个悬空 tool_use（中止在 dispatch 前/中，tool_result 还没 append）。
    messages = [
        {"role": "user", "content": "改两个文件"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "好的，开始"},
            {"type": "tool_use", "id": "tu1", "name": "edit_file", "input": {"path": "a.py"}},
            {"type": "tool_use", "id": "tu2", "name": "bash", "input": {"command": "ls"}},
        ]},
    ]
    cleaned = close_dangling_tool_uses(messages)
    assert cleaned is True, "有悬空 tool_use 应清理"
    # 追加了一条 user 消息，含 2 个 tool_result，id 与悬空 tool_use 一一对应。
    last = messages[-1]
    assert last["role"] == "user", "清理应追加 user 消息"
    results = {b["tool_use_id"]: b for b in last["content"]}
    assert set(results) == {"tu1", "tu2"}, f"每个 tool_use 都应有 tool_result，得 {set(results)}"
    assert all(b["type"] == "tool_result" for b in last["content"])
    assert all("Interrupted" in b["content"] for b in last["content"]), "占位内容应标中止"
    # 清理后整段可序列化（喂回 API 不崩）——用已验的 _to_jsonable。
    json.dumps(messages, ensure_ascii=False, default=_to_jsonable)

    # 边界 1：末尾已是 user(tool_result，本轮工具已配对) → 不触发。
    paired = messages[:]   # 上面已 append 了 tool_result，末尾是 user
    assert close_dangling_tool_uses(paired) is False, "已配对完不该再清理"

    # 边界 2：中止在 LLM 调用中（末尾是干净上一轮 user，无悬空 tool_use）→ 不触发。
    clean_prev = [{"role": "user", "content": "问题"}]
    assert close_dangling_tool_uses(clean_prev) is False, "无悬空不触发"

    # 边界 3：末尾 assistant 只有 text、无 tool_use（自然收尾被中止）→ 不触发。
    text_only = [{"role": "assistant", "content": [{"type": "text", "text": "答完了"}]}]
    assert close_dangling_tool_uses(text_only) is False, "无 tool_use 不触发"
    return 1


# ──────────────────────────────────────────────────────────────────
# ④ resume 往返线性数组一致 + 多轮不重复
# ──────────────────────────────────────────────────────────────────
def test_save_resume_roundtrip():
    store = SessionStore(Project.from_cwd(REPO))
    sid = store.new_session_id()
    assert len(sid) == 32 and all(c in "0123456789abcdef" for c in sid), f"uuid4().hex 形状：{sid!r}"

    msgs = [
        {"role": "user", "content": "第一个任务"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "bash", "input": {"command": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "file.py"}]},
        {"role": "assistant", "content": "做完了"},
    ]
    store.save(sid, msgs)
    got = store.resume(sid)
    assert got == msgs, f"resume 往返应字节级一致\n期望 {msgs}\n得到 {got}"

    # 文件确实逐行 JSON（行数 == 消息数，线性数组无 parentUuid 树）。
    path = store.project.sessions_dir / f"{sid}.jsonl"
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == len(msgs), f"应一行一消息，得 {len(lines)} 行 / {len(msgs)} 消息"
    return 1


def test_save_sdk_block_faithful():
    """真实 e2e 暴露的 P0：assistant content 含 SDK 块对象（ThinkingBlock 等、非纯 dict）。
    save→resume 后块应变成**等价 wire-format dict**（忠实、非 str），且可再 json.dumps 不崩。"""
    import json

    # 优先用真 ThinkingBlock（真实 e2e 的崩溃类型）；不可 import 则退一个带 model_dump 的小对象。
    try:
        from anthropic.types import ThinkingBlock
        block = ThinkingBlock(type="thinking", thinking="让我先定位", signature="sig123")
        expected = block.model_dump(mode="json")   # 忠实 wire-format dict
    except Exception:
        class _FakeBlock:
            def model_dump(self, mode="python"):
                return {"type": "thinking", "thinking": "让我先定位", "signature": "sig123"}
        block = _FakeBlock()
        expected = block.model_dump(mode="json")

    store = SessionStore(Project.from_cwd(REPO))
    sid = store.new_session_id()
    # 模拟真实 run_task：assistant content 是块对象列表（含 thinking 块）。
    msgs = [
        {"role": "user", "content": "任务"},
        {"role": "assistant", "content": [block, {"type": "text", "text": "好的"}]},
    ]
    store.save(sid, msgs)   # 旧实现在此抛 TypeError: ThinkingBlock not JSON serializable
    got = store.resume(sid)

    # 块被忠实序列化成结构化 dict（不是 "ThinkingBlock(...)" 这种 str）。
    restored_block = got[1]["content"][0]
    assert isinstance(restored_block, dict), f"块应回成 dict，得 {type(restored_block).__name__}"
    assert restored_block == expected, f"应忠实 wire-format dict，期望 {expected}，得 {restored_block}"
    assert restored_block.get("type") == "thinking", "type 字段应保留（喂回 SDK 用）"
    # 纯 dict 部分原样保真。
    assert got[1]["content"][1] == {"type": "text", "text": "好的"}
    # resume 回来的全是 JSON-native dict → 可再次 json.dumps 不崩（往返闭环）。
    json.dumps(got, ensure_ascii=False)
    return 1


def test_multi_round_no_duplicate():
    """多轮 run 模拟：每轮 run_task 返**全量** messages，save 覆盖写 → resume 不重复（评审 P0-2）。

    不调 LLM——直接用"全量 messages 每轮变长"模拟 run_task 的返回，验 save 覆盖语义。
    """
    store = SessionStore(Project.from_cwd(REPO))
    sid = store.new_session_id()

    round1 = [{"role": "user", "content": "task1"}, {"role": "assistant", "content": "r1"}]
    store.save(sid, round1)
    assert store.resume(sid) == round1

    # 第 2 轮：run_task 返回的是**累积全量**（含第 1 轮）。覆盖写后不该出现重复。
    round2 = round1 + [{"role": "user", "content": "task2"}, {"role": "assistant", "content": "r2"}]
    store.save(sid, round2)
    got = store.resume(sid)
    assert got == round2, f"第 2 轮覆盖写应得全量、无重复，得 {got}"
    assert len(got) == 4, f"4 条消息（非 2+4=6 的重复），得 {len(got)}"

    # resume 回灌喂给下一轮 initial_messages 的应是干净线性历史，无 N² 重复。
    stripped = [_no_uuid(m) for m in got]
    assert stripped.count({"role": "user", "content": "task1"}) == 1, "task1 不该重复出现"
    return 1


# ──────────────────────────────────────────────────────────────────
# ① 覆盖写原子性（写一半模拟中断、原文件完好）
# ──────────────────────────────────────────────────────────────────
def test_atomic_overwrite_on_crash():
    """模拟 save 写 tmp 途中进程死：原 <id>.jsonl 必须完好（os.replace 原子性）。"""
    store = SessionStore(Project.from_cwd(REPO))
    sid = store.new_session_id()
    good = [{"role": "user", "content": "已成功落盘的旧内容"}, {"role": "assistant", "content": "old"}]
    store.save(sid, good)
    path = store.project.sessions_dir / f"{sid}.jsonl"
    before = path.read_text(encoding="utf-8")

    # 模拟"写 tmp 写一半崩溃"：手动写一个半截 tmp 文件（json 行不完整），但**不** os.replace。
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text('{"role": "user", "content": "半截写到一', encoding="utf-8")  # 故意截断
    # 崩溃 ≈ 进程在 os.replace 之前死。此时再读原文件：必须还是旧的好内容。
    assert path.read_text(encoding="utf-8") == before, "崩溃在 replace 前，原文件必须完好"
    assert store.resume(sid) == good, "原文件可正常 resume（半截 tmp 不污染）"

    # 清掉残 tmp（下次正常 save 也会覆盖它），再正常 save 一次验证恢复正常。
    new = good + [{"role": "user", "content": "崩溃后新一轮"}]
    store.save(sid, new)   # 正常 save 覆盖残 tmp
    assert store.resume(sid) == new, "崩溃后下一次正常 save 应成功覆盖"
    assert not tmp.exists(), "save 成功后 tmp 应已被 os.replace 消费（不残留）"
    return 1


# ──────────────────────────────────────────────────────────────────
# ② 压缩轮落盘（messages 变短后 save→resume 一致、无残留旧行）
# ──────────────────────────────────────────────────────────────────
def test_compaction_round_shrink():
    """loop 内会原地压缩 messages（list 变短）。覆盖写必须落"压缩后短 list"、无残留旧长历史。

    这是 append-only 的死穴（盘上旧历史 vs 内存压缩后不一致），覆盖写天然兼容（评审 P0-2）。
    """
    store = SessionStore(Project.from_cwd(REPO))
    sid = store.new_session_id()

    # 第 1 轮：长历史（8 条）。
    long_hist = []
    for i in range(4):
        long_hist.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {}}]})
        long_hist.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * 100}]})
    store.save(sid, long_hist)
    assert len(store.resume(sid)) == 8

    # 第 2 轮：压缩发生，messages 被重绑成更短的压缩后 list（2 条）。
    compacted = [
        {"role": "user", "content": "[Compacted] 前 8 条已摘要"},
        {"role": "assistant", "content": "继续"},
    ]
    store.save(sid, compacted)
    got = store.resume(sid)
    # 关键：盘上必须只有压缩后 2 条，**无残留旧 8 行**（append-only 会留旧行→不一致）。
    assert got == compacted, f"压缩轮应落压缩后短 list，得 {got}"
    assert len(got) == 2, f"无残留旧长历史，应 2 条，得 {len(got)}"
    # 文件物理行数也必须是 2（确认是覆盖写、不是 append 追加在旧 8 行后）。
    path = store.project.sessions_dir / f"{sid}.jsonl"
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2, f"物理行数应 2（覆盖写），得 {len(lines)}（append 会是 10）"
    return 1


def test_list_ids():
    store = SessionStore(Project.from_cwd(Path(tempfile.mkdtemp())))   # 干净 project，无旧 session
    assert store.list_ids() == [], "空 project 应无 session"
    a = store.new_session_id(); store.save(a, [{"role": "user", "content": "a"}])
    b = store.new_session_id(); store.save(b, [{"role": "user", "content": "b"}])
    assert sorted(store.list_ids()) == sorted([a, b]), f"应列出 2 个 session：{store.list_ids()}"
    # resume 不存在的 id → 空 list（容错，不抛）。
    assert store.resume("does-not-exist") == [], "不存在的 id resume 应返回空 list"
    return 1


def test_list_sessions_6e():
    """6e：list_sessions 返回 {id,title,mtime}、按 mtime 倒序、排除 trace 文件。"""
    import time
    store = SessionStore(Project.from_cwd(Path(tempfile.mkdtemp())))
    assert store.list_sessions() == [], "空 project 应无会话"

    a = store.new_session_id()
    store.save(a, [{"role": "user", "content": "修复登录 bug"},
                   {"role": "assistant", "content": "好的"}])
    time.sleep(0.02)   # 拉开 mtime
    b = store.new_session_id()
    store.save(b, [{"role": "user", "content": [{"type": "text", "text": "重构缓存层"}]}])

    # 造一个 trace 文件，验它被排除（不当成会话）。
    (store.project.sessions_dir / f"{a}.trace.jsonl").write_text('{"x":1}\n', encoding="utf-8")

    sessions = store.list_sessions()
    assert len(sessions) == 2, f"应 2 个会话（trace 文件排除），得 {len(sessions)}：{[s['id'] for s in sessions]}"
    # 按 mtime 倒序：b（后写）在前。
    assert sessions[0]["id"] == b and sessions[1]["id"] == a, "应按 mtime 倒序（最近在前）"
    # title = 首条 user 消息截断（str 与 list-block 两种 content 都取得到）。
    assert sessions[0]["title"] == "重构缓存层", f"list-block content 应取 text，得 {sessions[0]['title']!r}"
    assert sessions[1]["title"] == "修复登录 bug", f"str content 应取整段，得 {sessions[1]['title']!r}"
    assert all(isinstance(s["mtime"], float) for s in sessions), "mtime 应是 float 时间戳"
    # list_ids 同样排除 trace（防 <id>.trace 伪 id）。
    assert sorted(store.list_ids()) == sorted([a, b]), f"list_ids 应排除 trace，得 {store.list_ids()}"
    return 1


def main():
    tests = [
        test_keying_stable_and_ace_home,
        test_keying_git_fallback_deterministic,
        test_session_memory_path_per_session,
        test_run_appends_task_b1,
        test_close_dangling_tool_uses_step7,
        test_save_resume_roundtrip,
        test_save_sdk_block_faithful,
        test_multi_round_no_duplicate,
        test_atomic_overwrite_on_crash,
        test_compaction_round_shrink,
        test_list_ids,
        test_list_sessions_6e,
    ]
    total = 0
    for t in tests:
        n = t()
        total += n
        print(f"  [OK] {t.__name__}")
    print(f"\n[OK] 运行时 G1–G4 回归通过：{len(tests)} 个测试 / {total} 组断言。")
    print("      ① 覆盖写原子性 ② 压缩轮落盘 ③ keying 稳定+ACE_HOME override ④ resume 往返+多轮不重复。")
    print(f"      （全程 ACE_HOME={_ACE_TMP}，未污染真实 ~/.ace）")


if __name__ == "__main__":
    main()
