"""slash 命令路由（步 4）：/help /exit /clear /resume /skills /model。非 slash 当普通任务（在 repl 里）。

反膨胀（cli-shell-plan §6）：只留 REPL 必需命令。CC 全套几十个 slash（/config /cost /doctor …）
是产品运维命令，demo 不需要全抄；/skills 对齐 CC 的 skill 浏览；/model 对齐 CC 主模型切换。

返回约定（给 repl 主循环判断）：
  - EXIT 哨兵      → 退出 REPL。
  - 一个 Session   → 换会话（/clear 新建 / /resume 回灌），repl 重设 sink。
  - None           → 已处理（如 /help），继续用原 session。
"""

from ..runtime import Session
from .skills_ui import show_skill_listing
from ..skills import discover_skill_catalog, resolve_skill_workdir
from ..runtime.settings import load_merged_settings

# 退出哨兵（用唯一对象身份判等，避免和正常返回值混淆）。
EXIT = object()

_HELP = """可用命令：
  /help            显示本帮助
  /exit            退出并显示恢复命令（同 Ctrl-D）
  /clear           开新会话（旧会话已落盘，可用 /resume <旧id> 找回）
  /resume <id>     回灌某会话历史，接着问
  /skills          浏览可用 skill（↑↓ 选择 · Enter 查看）
  /model           切换主模型（TTY 弹出选择器，仅本次 REPL 会话）
  /model <id>      直接切换（仅本次 REPL 会话）
  /model default   清除会话内覆盖，回到 settings.json 的 model
永久修改模型请手动编辑 ~/.ace/settings.json。
其余输入都当作交给 agent 的任务。"""


def _handle_model(
    arg: str,
    *,
    model_state,
    workpath,
    out,
    model_pick_cb=None,
) -> None:
    """/model: session-only override; permanent changes = edit settings.json."""
    if arg.lower() == "default":
        model_state.clear()
        model_state.refresh_from_settings(load_merged_settings(workpath))
        out(f"Using default model {model_state.display}")
        return

    model_id = arg.strip()
    if not model_id:
        if model_pick_cb is not None:
            model_id = model_pick_cb() or ""
            if not model_id:
                return
        else:
            out("用法：/model <model_id> 或 /model default")
            return

    previous = getattr(model_state, "display", "")
    model_state.set(model_id)
    if model_id == previous:
        out(f"Kept model as {model_id}")
    else:
        out(f"Set model to {model_id} (session only)")


def handle(
    line: str,
    session: Session,
    workpath,
    out,
    pick_cb=None,
    *,
    is_tty: bool = False,
    model_state=None,
    model_pick_cb=None,
):
    """处理一条 slash。out 是打印函数。返回 EXIT / 新 Session / None。

    pick_cb（6e，可选）：() -> session_id | None 的交互式选择器（TTY-only）。/resume 无参时
    若提供就弹选择器；非 TTY/未提供则退回"用法"提示。
    model_state / model_pick_cb：/model 专用（仅 REPL 注入，eval/run_task 不传）。
    """
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        return EXIT

    if cmd == "/help":
        out(_HELP)
        return None

    if cmd in ("/skill", "/skills"):
        skill_root = resolve_skill_workdir(workpath)
        catalog = discover_skill_catalog(skill_root)
        show_skill_listing(catalog, out, is_tty=is_tty)
        return None

    if cmd == "/model":
        if model_state is None:
            out("（/model 仅 REPL 可用）")
            return None
        _handle_model(
            arg,
            model_state=model_state,
            workpath=workpath,
            out=out,
            model_pick_cb=model_pick_cb,
        )
        return None

    if cmd == "/clear":
        new = Session.create(workpath)
        # 打印旧 id，否则用户不知道旧 id 找不回。
        out(f"已开新会话 {new.id[:12]}；旧会话 {session.id[:12]} 已落盘，可 /resume {session.id} 找回。")
        return new

    if cmd == "/resume":
        if not arg:
            # 6e：无参 → 弹交互式选择器（TTY）。非 TTY/无 pick_cb → 退回用法提示。
            if pick_cb is not None:
                arg = pick_cb()
                if not arg:        # 用户 Esc 取消 / 空列表 → 不换 session
                    return None
            else:
                out("用法：/resume <session_id>")
                return None
        resumed = Session.resume(arg, workpath)
        if not resumed.messages:
            # 空历史：id 不存在或会话从未落盘。诚实告知，不静默假装成功。
            out(f"会话 {arg} 无历史（id 不存在或未落盘）。仍可在此 id 上继续。")
        else:
            out(f"已回灌会话 {arg[:12]}（{len(resumed.messages)} 条消息），接着问。")
        return resumed

    out(f"未知命令 {cmd}；/help 看可用命令。")
    return None
