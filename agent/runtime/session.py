"""Session —— 会话句柄：薄包 run_task 把多任务串成一个会话 + 落盘 + 记忆挂载。

它怎么薄叠（对 loop/tools/compact 零侵入，[[runtime-gap-audit]] §6.4 已核）：
  Session.run = using_workdir(已有, config.py:32)
              + run_task(initial_messages/return_messages/session_memory，已有接缝 loop.py:44-45,162)
              + store.save（唯一真新增）。

两个评审硬约束，就地标注：
  - **trace=False 必带**（sink 所有权：sink 生命周期归调用方，不让 run_task 自建 sink 冲掉）：run_task(trace=True，默认)
    每轮自建 JsonlSink 并 set_sink（loop.py:67-68），会**冲掉** CLI/调用方设的 TeeSink →
    实时输出失效。决策：sink 生命周期归 CLI/Session 掌管，run_task 一律 trace=False、退化为
    "环境 sink 纯消费者"。照搬 SessionRunner 现成做法（eval/swebench/session_run.py:102-104）。
  - **记忆挂 project.memory_dir**（替掉调用方任意传 path）：同一 project 不同 session →
    同 memory_dir → 跨会话记忆有锚点（[[runtime-gap-audit]] §3.5 的机制根）。

落盘语义见 store.py：每轮 run 后整文件覆盖写全量 self.messages（免增量游标、
压缩轮天然兼容）。
"""

from .. import config
from ..loop import run_task
from ..memory.session_memory import SessionMemory
from ..memory.auto_memory import AutoMemory
from ..mcp.connection_manager import McpConnectionManager
from ..mcp.runtime_config import UNSET, resolve_run_task_runtime_kwargs
from .messages import new_user_message
from .project import Project
from .store import SessionStore


class Session:
    """一个落盘的会话句柄。持 id/project/store/memory/messages，run() 把任务串起来。"""

    def __init__(self, session_id: str, project: Project, store: SessionStore,
                 memory: SessionMemory | None, messages: list,
                 auto_memory: "AutoMemory | None" = None):
        self.id = session_id
        self.project = project
        self.store = store
        self.memory = memory          # 会话内 SessionMemory（per-session 笔记）
        self.auto_memory = auto_memory  # 跨会话 Auto Memory（per-project 记忆 + 索引）
        self.messages = messages

    @classmethod
    def create(cls, workpath, *, with_memory: bool = True) -> "Session":
        """新会话：new id + 空 messages。SessionMemory 笔记按**会话**挂（各会话独立、不互相覆盖）。

        with_memory 默认 True；纯运行时单测/不需要记忆的场景可关，避免无谓 fork。
        """
        project = Project.from_cwd(workpath)
        store = SessionStore(project)
        session_id = store.new_session_id()
        memory = cls._make_memory(project, session_id) if with_memory else None
        auto_memory = cls._make_auto_memory(project) if with_memory else None
        return cls(session_id, project, store, memory, [], auto_memory)

    @classmethod
    def resume(cls, session_id: str, workpath, *, with_memory: bool = True) -> "Session":
        """续会话：store.resume() 回灌 messages（跨进程接得住）。SessionMemory 仍挂同会话的笔记文件。"""
        project = Project.from_cwd(workpath)
        store = SessionStore(project)
        messages = store.resume(session_id)
        memory = cls._make_memory(project, session_id) if with_memory else None
        auto_memory = cls._make_auto_memory(project) if with_memory else None
        return cls(session_id, project, store, memory, messages, auto_memory)

    @staticmethod
    def _make_memory(project: Project, session_id: str) -> SessionMemory:
        """SessionMemory 接线：会话内活文档，按会话挂 sessions_dir/<id>.notes.md。

        确认两类记忆**不同名不同目录、零碰撞**：
          - 会话内 SessionMemory 笔记 = per-session（CC: projects/<cwd>/<id>/session-memory/
            summary.md，filesystem.ts:270）。每会话独立、不互相覆盖。
          - 跨会话 Auto Memory 索引 = per-project 的 MEMORY.md（CC: memdir/paths.ts:93）。
        旧实现误接 memory_dir/MEMORY.md（项目级、且撞将来 Auto Memory 的索引文件）：① 同
        project 不同 session 共享一文件互相覆盖、丢"会话内"语义；② 将来建 Auto Memory 撞同名。

        ⚠ 诚实偏离 CC：CC 是嵌套 <id>/session-memory/summary.md；我们平铺 <id>.notes.md
        （与 transcript <id>.jsonl 同目录）。作用域一致（按会话），仅目录层级更平。
        project.memory_dir/MEMORY.md 留给将来 Auto Memory，本步不占用（见 project.py 占位注释）。
        """
        return SessionMemory(project.sessions_dir / f"{session_id}.notes.md")

    @staticmethod
    def _make_auto_memory(project: Project) -> AutoMemory:
        """Auto Memory 接线：跨会话经验记忆，挂 **project 级** memory_dir（project.py:109 占位已就位）。
        同 project 不同 session 共享 → 跨会话记忆有锚点；与 per-session SessionMemory 不同目录、零碰撞。
        opt-in 同 with_memory：关掉则 run_task 收 auto_memory=None，写入/召回/索引常驻全不触发。
        """
        return AutoMemory(project.memory_dir)

    def run(
        self,
        task: str,
        *,
        permission_engine=None,
        mcp_connection_manager: McpConnectionManager | None = None,
        **run_task_kwargs,
    ) -> str:
        """跑一个任务、接力上下文、落盘。返回最终文本。

        trace=False 强制（sink 所有权）：调用方若误传 trace 会被这里覆盖，保证 sink 不被 run_task 抢。

        permission_engine：仅 REPL 等交互入口注入；默认不传 → run_task 用空 PermissionEngine。

        ── Fix: resume was missing the new task (found in real end-to-end testing) ──
        loop.py:81 `messages = deepcopy(initial_messages) if initial_messages else [{user:task}]`
        —— 给了 initial_messages 就**只用它、忽略 task**。旧实现在非空会话上传 initial_messages=
        旧历史 → 新 task 从未进消息 → agent 重跑旧对话。故这里**先把新 task 作为 user 消息追加
        进 self.messages、再当 initial_messages 传**：对空会话等价于原 [{user:task}]，对 resumed
        会话正确接上新问题。仍零侵入 loop（只改 Session.run 的入参组织）。
        """
        run_task_kwargs.pop("trace", None)   # 防调用方误传冲掉 sink 所有权契约。
        enable_mcp_arg = run_task_kwargs.pop("enable_mcp", UNSET)
        mcp_config_path_arg = run_task_kwargs.pop("mcp_config_path", UNSET)
        enable_deferred_arg = run_task_kwargs.pop("enable_deferred_tools", UNSET)
        disable_mcp = bool(run_task_kwargs.pop("disable_mcp", False))
        runtime_kwargs = resolve_run_task_runtime_kwargs(
            enable_mcp=enable_mcp_arg,
            mcp_config_path=mcp_config_path_arg,
            enable_deferred_tools=enable_deferred_arg,
            disable_mcp=disable_mcp,
            workdir=self.project.workpath,
        )
        if permission_engine is not None:
            run_task_kwargs["permission_engine"] = permission_engine
        mcp_lease = None
        if mcp_connection_manager is not None and (
            runtime_kwargs.get("enable_mcp") or runtime_kwargs.get("mcp_config_path")
        ):
            mcp_lease = mcp_connection_manager.acquire(
                workdir=self.project.workpath,
                enable_mcp=bool(runtime_kwargs.get("enable_mcp")),
                mcp_config_path=runtime_kwargs.get("mcp_config_path"),
            )
        elif mcp_connection_manager is not None:
            mcp_connection_manager.invalidate()
        self.messages = (self.messages or []) + [new_user_message(task)]
        with config.using_workdir(self.project.workpath):
            text, self.messages = run_task(
                task,
                initial_messages=self.messages,   # 含刚追加的新 task；loop 直接续接这条历史
                return_messages=True,
                session_memory=self.memory,
                auto_memory=self.auto_memory,      # 跨会话记忆（写入/召回/索引常驻）；None 时全不触发
                trace=False,                       # ← sink 所有权：caller holds the sink
                mcp_session=mcp_lease,
                **runtime_kwargs,
                **run_task_kwargs,
            )
        self.store.save(self.id, self.messages)   # ← 整文件覆盖落盘
        return text
