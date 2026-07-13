"""EvoClaw 容器内入口（BYO-agent adapter 的 agent 侧）：run / resume 两子命令。

设计权威：docs/compression-eval/01-design-framework.md §2.4 + docs/evoclaw-validation.md §7b.4。
本文件是 B1 架构（agent-in-container + LocalExecutor，cwd=/testbed）里"被 EvoClaw 驱动的 agent"
的薄包装：把宿主侧的 agent.loop.run_task 包成一个 docker exec 可调的 CLI。

⚠ 偏离标注（命名）：设计/任务书写的是新增 `agent/cli.py`，但本仓库 `agent/cli/` 目录已被
   **交互式 REPL 壳** 占用（agent/cli/__init__.py 的 "交互式 CLI 壳"）。`agent/cli.py` 与
   `agent/cli/` 包同名会冲突 → 改名 `agent/evoclaw_cli.py`。功能与骨架一致，仅模块名不同；
   wrapper（/usr/local/bin/myagent）相应 `from agent.evoclaw_cli import main`。TODO：若将来把
   交互壳并入子模块腾出 cli.py，可回归原名。

数据流（WHY 要持久化 messages）：EvoClaw 两次 `docker exec` 是**独立进程**，session 必须跨进程
持久。run 跑完把全量 messages 落 `~/.myagent/<sid>.json`，resume 读回当 `initial_messages`
接续 → 这正是上下文累积点：messages 越长，full 臂上下文越大（context rot）、pipeline 臂越要压。
"""

import argparse
import json
import os
import pathlib
import sys

from . import config, loop, tools
from .memory.session_memory import SessionMemory
from .mcp.runtime_config import resolve_run_task_runtime_kwargs
# WHY 复用 runtime.store._to_jsonable：真实 run_task 的 messages 里 assistant content 是
# Anthropic SDK 块对象（ThinkingBlock/TextBlock/ToolUseBlock，deepseek-v4 带 thinking 块），
# 非纯 dict。骨架里的 default=str 会把块变成垃圾字符串、resume 读回坏续作（store.py P1-A 已否决）。
# model_dump(mode="json") 转 wire-format dict，往返保真。这里直接复用那份被实测验证过的序列化器，
# 不重复造轮子、不重蹈 default=str 的坑。
from .runtime.store import _to_jsonable

# 容器内 session 持久目录（HOME=/home/fakeroot → ~/.myagent）。
SESS = pathlib.Path.home() / ".myagent"

# 运行环境编排说明：把 EvoClaw 面向 claude-code 的 v1 prompt 桥接到我们的工具集。
# ★ 三臂同构：本 preamble 三臂完全相同，只有 COMPACT_STRATEGY 不同 → 不破坏"只差一个压缩策略"。
# WHY 必要：① v1 prompt 让 agent "spawn Sub-Agent"——我们没有 sub-agent 工具，需说明"你自己顺序做"；
#   ② 队列/SRS 在 /e2e_workspace（/testbed 之外），read_file 工具被 safe_path 限制在 WORKDIR=/testbed
#   内、读它会"路径越权" → 必须改用 bash `cat`（exec_shell 不受 WORKDIR 限制）；③ 重申完成信号=git tag。
_ORCHESTRATION_PREAMBLE = """\
[运行环境说明 —— EvoClaw 连续里程碑]
- 你就是实现者：直接读 SRS、改代码、提交、打 tag。本环境没有独立 sub-agent 工具，下面提到的
  "spawn Sub-Agent" 按"你自己一个一个里程碑顺序做"理解即可。
- 任务队列与 SRS 在 /e2e_workspace 下（在 /testbed 之外）：用 **bash 工具** 读它们，例如
  `cat /e2e_workspace/TASK_QUEUE.md`、`cat /e2e_workspace/srs/<milestone_id>_SRS.md`。
  read_file 工具被限制在 /testbed 内、读 /e2e_workspace 会报"路径越权"，所以读队列/SRS 一律用 bash cat。
- 代码改动只在源码目录内（见下方 Source Code）。改完在 /testbed 里依次：
  `git add <src> && git commit -m "Implement <milestone_id>" && git tag agent-impl-<milestone_id>`。
  **打 tag 是里程碑完成的唯一信号**，打了就触发外部评分、不可回退。
- 做完一个里程碑立刻重新 `cat /e2e_workspace/TASK_QUEUE.md` 看有没有新解锁的任务；直到队列显示
  "(No tasks currently available)" 再收尾。

----- 以下是任务说明 -----
"""


def _load(sid: str):
    """resume：读回上次落盘的 messages 当续作种子。不存在返回 None（当新会话起）。"""
    f = SESS / f"{sid}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def _save(sid: str, messages: list) -> None:
    """整文件覆盖落盘全量 messages（供下次 resume 读回）。default=_to_jsonable 忠实序列化 SDK 块。"""
    SESS.mkdir(parents=True, exist_ok=True)
    f = SESS / f"{sid}.json"
    f.write_text(json.dumps(messages, ensure_ascii=False, default=_to_jsonable), encoding="utf-8")


def _truthy_env(name: str) -> bool:
    v = os.environ.get(name)
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _sm_dir() -> pathlib.Path:
    return SESS / "session_memory"


def _sm_notes_path(sid: str) -> pathlib.Path:
    return _sm_dir() / f"{sid}.notes.md"


def _sm_state_path(sid: str) -> pathlib.Path:
    return _sm_dir() / f"{sid}.state.json"


def _load_session_memory(sid: str) -> SessionMemory:
    """Create EvoClaw per-session SM and restore cross-exec runtime anchors."""

    sm = SessionMemory(_sm_notes_path(sid))
    state_path = _sm_state_path(sid)
    if not state_path.exists():
        return sm
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return sm
    sm._initialized = bool(data.get("initialized", False))
    sm._tokens_at_last = int(data.get("tokens_at_last", 0) or 0)
    sm._seen_tool_ids = set(data.get("seen_tool_ids") or [])
    if data.get("last_summarized_message_id"):
        sm.set_last_summarized_message_id(data["last_summarized_message_id"])
    return sm


def _save_session_memory_state(sid: str, sm: SessionMemory | None) -> None:
    if sm is None:
        return
    _sm_dir().mkdir(parents=True, exist_ok=True)
    data = {
        "initialized": bool(getattr(sm, "_initialized", False)),
        "tokens_at_last": int(getattr(sm, "_tokens_at_last", 0) or 0),
        "seen_tool_ids": sorted(str(x) for x in getattr(sm, "_seen_tool_ids", set())),
        "last_summarized_message_id": sm.get_last_summarized_message_id(),
        "notes_path": str(sm.path),
    }
    _sm_state_path(sid).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _build_meta(sid: str, strat: str, *, session_memory_enabled: bool = False) -> dict:
    """★ 共享 trace meta 契约（CoderA 的曲线脚本按 arm/milestone/session_id/instance_id 聚合）。

    字段名严格钉死，别自造别名。
      - arm        ：当前压缩臂（= COMPACT_STRATEGY），三臂分叉的归因键。
      - milestone  ：当前里程碑 id。⚠ v1 prompt 让 agent **自驱整条队列**（一次 exec 连做多个
                     里程碑，agent_runner.py:1099 只替换 {src_dirs}、{milestone_id} 留作字面量）→
                     单个 run_task 跨多里程碑，run 级 milestone 本质是粗粒度的。这里取
                     MYAGENT_MILESTONE（若上层按里程碑驱动则注入），否则填 "self-driven" 兜底。
                     **per-里程碑 resolve×长度曲线须由 EvoClaw 的 git-tag 时刻与本 trace 的 turn span
                     按时间 join**，不靠这个粗 milestone 字段。（已在返回里向 lead 标注此交界面口径。）
      - session_id ：EvoClaw 传入的 UUID（跨 exec 持久键）。
      - instance_id：链/repo id（如 ripgrep）。取 MYAGENT_INSTANCE_ID，上层 launch 脚本注入。
    """
    arm = os.environ.get("MYAGENT_ARM_LABEL")
    if not arm:
        arm = "pipeline_sm" if session_memory_enabled else strat
    return {
        "arm": arm,
        "compact_strategy": strat,
        "session_memory_enabled": session_memory_enabled,
        "milestone": os.environ.get("MYAGENT_MILESTONE", "self-driven"),
        "session_id": sid,
        "instance_id": os.environ.get("MYAGENT_INSTANCE_ID", ""),
    }


def _int_env(name: str, default):
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(prog="myagent", description="EvoClaw 容器内 agent 入口")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "resume"):
        p = sub.add_parser(c)
        p.add_argument("--session-id", required=True)
    a = ap.parse_args()

    # ★ EvoClaw 把 prompt(run)/message(resume) 经 stdin 喂进来（base.py 契约：命令以 `< path` 收尾）。
    stdin_text = sys.stdin.read()

    # 三臂开关：COMPACT_STRATEGY ∈ {none, pipeline, truncate}（truncate 由 CoderA 在 loop 实现，
    #   值是字符串、本 CLI 不用改即可透传）。adapter 的 get_container_env_vars 注入。
    strat = os.environ.get("COMPACT_STRATEGY", "none")
    sid = a.session_id

    # 容器内 = 本地执行，cwd=/testbed（LocalExecutor 默认就是当前 _EX；显式设一遍更清楚）。
    tools.set_executor(tools.LocalExecutor())
    # WORKDIR=/testbed：LocalExecutor 的 bash cwd / 文件操作根 = 此处。AGENT_WORKDIR 可覆盖
    #   （宿主测试时指 tmp，别污染真实环境）。/e2e_workspace 在 WORKDIR 外、经 bash cat 读（见 preamble）。
    workdir = os.environ.get("AGENT_WORKDIR", "/testbed")
    max_turns = _int_env("MYAGENT_MAX_TURNS", 200)   # 连续多里程碑 → 放大轮数（设计 §2.4）
    cwin = _int_env("MYAGENT_COMPACT_WINDOW", None)      # 可选：覆盖压缩窗口（默认走 loop DEFAULT）
    cthr = _int_env("MYAGENT_COMPACT_THRESHOLD", None)   # 可选：覆盖压缩触发阈值
    stop_at_context = _int_env("MYAGENT_STOP_AT_CONTEXT", None)
    session_memory_enabled = _truthy_env("MYAGENT_SESSION_MEMORY")
    session_memory = _load_session_memory(sid) if session_memory_enabled else None

    if a.cmd == "resume":
        init = _load(sid) or []
        # resume 的 stdin 是"新一条 user 消息"（claude-code 语义：--resume <sid> < message）。
        # run_task 给了 initial_messages 时会忽略 task、不把它加进 messages → 必须自己把新消息
        # append 成一条 user 轮，否则 resume 的指令丢失。落盘 messages 总是 run_task 干净返回的可续态
        #   （finished=assistant 收尾 / max_turns=user(tool_result) 收尾 / interrupted 已清悬空），
        #   追加一条 user 合法。
        init.append({"role": "user", "content": stdin_text})
        task = stdin_text
    else:  # run：首跑，stdin 是渲染后的 v1 prompt；前置环境编排说明。
        init = None
        task = _ORCHESTRATION_PREAMBLE + stdin_text

    meta = _build_meta(sid, strat, session_memory_enabled=session_memory_enabled)

    with config.using_workdir(workdir):
        mcp_run_task_kwargs = resolve_run_task_runtime_kwargs()
        _final, messages = loop.run_task(
            task,
            eval_hooks=loop.EvalHooks(
                compact_strategy=strat,
                compact_window=cwin,
                compact_threshold=cthr,
                stop_at_context=stop_at_context,
            ),
            initial_messages=init,
            session_memory=session_memory,
            return_messages=True,   # 复用续作接缝，拿回全量 messages 落盘
            meta=meta,              # 带全 4 字段 trace meta 契约
            max_turns=max_turns,
            **mcp_run_task_kwargs,
        )

    _save(sid, messages)   # ★ 持久化，供下次 resume 累积上下文
    _save_session_memory_state(sid, session_memory)
    # 退出码语义弱：完成判定靠 git tag（base.py 契约 #5），正常返回一律 exit 0。
    sys.exit(0)


if __name__ == "__main__":
    main()
