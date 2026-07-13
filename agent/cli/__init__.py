"""交互式 CLI 壳（步 1–4）：坐在 runtime 层（G1–G4）之上，给它 stdin/stdout 入口。

  - render.py   ：span → 人读单行（"› grep …"/"✎ edit foo.py"/"· turn"），供 TeeSink 调。
  - banner.py   ：启动 ASCII banner + 状态框。
  - commands.py ：slash 路由（/help /exit /clear /resume /skill）。
  - repl.py     ：REPL 主循环（prompt_toolkit + patch_stdout + Session.run）。

设计权威：docs/runtime/00-design.md §3 文件布局 / §5 sink 所有权 / cli-shell-plan.md §3。
对 loop/tools/compact/记忆零改逻辑——壳只是 Session 的另一个调用方 + obs 的第二个 sink。
"""
