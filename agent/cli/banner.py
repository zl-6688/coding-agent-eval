"""启动 ASCII banner + 状态框（步 3）。

状态框字段：WORKSPACE / MODEL / APPROVAL / BRANCH / SESSION。
  - WORKSPACE = project.workpath（真实工作目录）
  - MODEL     = config.MODEL_ID
  - APPROVAL  = 先恒显 'ask'（逻辑步 5 接 approve_cb；此处只显示，不实现权限）
  - BRANCH    = git 当前分支（读不到显 '-'）
  - SESSION   = session.id（短显前 12 位，够区分、不刷屏）
"""

import subprocess

from .. import config

_BANNER = r"""
   __ _  ___ ___
  / _` |/ __/ _ \   ace —— 本地 coding agent
 | (_| | (_|  __/   loop 范式 · 可度量 · 可续作
  \__,_|\___\___|
"""


def _git_branch(workpath) -> str:
    """读 workpath 的 git 当前分支；任何失败返回 '-'（确定性，不抖）。"""
    try:
        out = subprocess.run(
            ["git", "-C", str(workpath), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            br = out.stdout.strip()
            if br:
                return br
    except (OSError, subprocess.SubprocessError):
        pass
    return "-"


def render_status(session, approval=None, *, model_id: str | None = None) -> str:
    """渲染状态框（字段真实，从 config/Project/git 读）。

    ⚠ step6a：**删掉 APPROVAL 行**（用户明确要：框里那行是启动时快照、不随 shift+tab 更新 →
    误导）。approval 现由 PromptSession 底部常驻状态栏（bottom_toolbar）实时显示。approval 参数
    保留只为向后兼容旧调用方，**不再渲染**。
    """
    rows = [
        ("WORKSPACE", str(session.project.workpath)),
        ("MODEL", model_id or config.MODEL_ID),
        ("BRANCH", _git_branch(session.project.workpath)),
        ("SESSION", session.id[:12]),
    ]
    width = max(len(f"{k}  {v}") for k, v in rows) + 2
    top = "┌" + "─" * width + "┐"
    bot = "└" + "─" * width + "┘"
    lines = [top]
    for k, v in rows:
        content = f"{k:<9} {v}"
        lines.append("│ " + content.ljust(width - 1) + "│")
    lines.append(bot)
    return "\n".join(lines)


def render_banner(session, approval=None, *, model_id: str | None = None) -> str:
    """完整启动输出：banner + 状态框 + 一行提示。approval 参数保留向后兼容、不渲染（见 render_status）。"""
    return (_BANNER + "\n" + render_status(session, model_id=model_id)
            + "\n输入任务开始；/help 看命令，/exit 退出。\n")
