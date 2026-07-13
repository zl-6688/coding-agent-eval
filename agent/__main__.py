"""`python -m agent` 入口：读 cwd → Project → Session → 进 REPL。

薄到只剩一行转发：所有逻辑在 agent/cli/repl.py。读当前目录 = 把 Project.workpath 设为
os.getcwd()（cli-shell-plan §2：CLI 的"读当前目录"经 using_workdir 注入，在 Session.run 内）。
"""

from .cli.repl import main

if __name__ == "__main__":
    main()
