"""Memory eval case definitions.

Probe text and PASS predicates are verbatim from 00-design.md § Appendix A —
do NOT paraphrase; discriminability lives in the exact wording.

Each Case carries:
  id                  — short identifier matching design doc (H_fb1 … H_neg)
  track               — 1 (A/B-Δ) or 2 (single-condition CC behaviour)
  setup_task          — S1 instruction ("teach" turn for track-1; single-arm setup for track-2)
  probe_task          — S2 instruction (track-1) or probe for track-2 single-condition
  grader_type         — "code" | "judge" | "code+judge"
  write_gate_tokens   — list of required tokens in memory body after S1
                        each element is str (exact match) or list[str] (any-of/OR)
  write_gate_reverse  — True for H_prec: gate checks that decoy *was* stored (reversed polarity)
  track2_sub          — track-2 cases may have sub-condition keys (e.g. H_ignore)
  notes               — inline rationale excerpt from design
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    id: str
    track: int
    setup_task: str
    probe_task: str
    grader_type: str                        # "code" | "judge" | "code+judge"
    write_gate_tokens: tuple = ()           # required tokens; list[str|list[str]]
    write_gate_reverse: bool = False        # True → gate checks decoy IS stored (H_prec)
    notes: str = ""


# ── Track 1 · Incremental cases (want Δ≫0) ───────────────────────────────────

H_fb1 = Case(
    id="H_fb1",
    track=1,
    # setup_task: verbatim from Appendix A
    setup_task=(
        "提醒你：给这个项目写测试**别 mock 数据库**——上次 mock 掉 DB 漏了个迁移 bug。"
        "以后一律连真实测试库/测试容器。"
    ),
    # probe_task: verbatim from Appendix A (neutral — no "integration test" hint)
    probe_task="给 `OrderRepo` 写个测试。",
    grader_type="code",
    # write gate: memory body must contain all of: mock, (数据库 or db), (Why: or How to apply:)
    # Gate: mock AND (数据库/db/DB) — why/how headers excluded because
    # auto_memory writes Chinese and may not use English header literals.
    write_gate_tokens=(
        "mock",
        ["数据库", "db", "DB"],
    ),
    notes=(
        "B FAIL reason: without memory, 'write a test' defaults to mock-DB unit test → "
        "hits `mock` token → FAIL.  B-1 sanity: first run must verify B really does mock."
    ),
)

H_ref = Case(
    id="H_ref",
    track=1,
    # verbatim from Appendix A
    setup_task="记一下：ingest pipeline 的 bug 上下文都在我们 Linear 的 **INGEST** project 里。",
    probe_task="这个 ingest 报错我想查历史上下文，去哪找？",
    grader_type="code",
    # write gate: memory body must contain INGEST AND Linear
    write_gate_tokens=("INGEST", "Linear"),
    notes=(
        "B FAIL: without memory, agent guesses logs/git/generic → no 'Linear+INGEST' → FAIL.  "
        "Fixture must contain ZERO occurrences of 'Linear' or 'INGEST' to block grep false-PASS."
    ),
)

H_proj = Case(
    id="H_proj",
    track=1,
    # verbatim from Appendix A
    setup_task=(
        "项目约束：**2026-07-09（含）之后冻结所有非关键 merge**，只放紧急修复。"
    ),
    # probe uses absolute date already inside freeze window
    probe_task=(
        "我想在 **2026-07-15** 合一个非关键的重构 PR，帮我看下计划。"
    ),
    grader_type="judge",
    write_gate_tokens=("2026-07-09", "冻结", "非关键"),
    notes=(
        "B FAIL: without memory, agent doesn't know about freeze → directly helps schedule "
        "merge without warning → FAIL.  Absolute date removes 'date arithmetic' as a "
        "memory-unrelated failure source."
    ),
)

H_fb2 = Case(
    id="H_fb2",
    track=1,
    # verbatim from Appendix A
    setup_task=(
        "我的习惯：重构**打成一个 bundled PR**，别拆成一堆碎 commit/PR——一次 review 看完。"
    ),
    # probe deliberately presents three natural split points to tempt splitting
    probe_task=(
        "重构 auth 中间件（涉及拆出校验逻辑、重写 handler、更新测试三块），给我提交计划。"
    ),
    grader_type="judge",
    # Gate: (bundled/一个PR/单个PR/不拆/打包/一次review) — broaden for Chinese variants
    write_gate_tokens=(["bundled", "一个 PR", "单个 PR", "不拆", "打包", "一次 review"],),
    notes=(
        "B FAIL: three natural split points + 'commit plan' prompt → agent defaults to "
        "staged multi-commit/multi-PR → FAIL."
    ),
)

H_usr = Case(
    id="H_usr",
    track=1,
    # verbatim from Appendix A
    setup_task=(
        "背景：我写了十年 Go，**第一次碰 React 前端**，这仓库前端对我是新的。"
    ),
    # probe points at the fixture React component
    probe_task=(
        "解释下这个 React 组件的 state 流。"
        "（组件在 components/UserCard.jsx）"
    ),
    grader_type="judge",
    write_gate_tokens=("Go", ["React", "前端"], ["新手", "第一次"]),
    notes=(
        "B FAIL: without memory, agent explains React normally, no Go bridge → no Go anchor → FAIL.  "
        "PASS rubric only counts explicit Go analogies; 'beginner-friendly' is default behaviour."
    ),
)

# ── Track 1 · Precision case (want Δ≈0) ──────────────────────────────────────

H_prec = Case(
    id="H_prec",
    track=1,
    # WHY: keep this fact-shaped. Action-shaped wording ("must/forbid/violates")
    # makes the coding agent inspect or change the repo during S1 and can exhaust
    # max_turns before auto_memory.write() fires.
    setup_task=(
        "记一下一条长期测试约定，先记住就行、这轮不用改代码也不用翻仓库："
        "我们项目里 `PaymentService` 的测试要用 stub 把 `PaymentGateway` 替身掉"
        "（mock 支付网关），不连真实网关。"
    ),
    # probe asks about a different service with NO gateway dependency
    probe_task="给 `OrderRepo` 写个测试。",
    grader_type="code",   # graders.py _grade_h_prec: pure code (regex pattern scan); no _judge_call
    # REVERSED gate: verify the PaymentService decoy WAS stored (otherwise test is vacuous)
    write_gate_tokens=(["PaymentService", "支付网关", "gateway", "stub"],),
    write_gate_reverse=True,
    notes=(
        "Polarity REVERSED: want Δ≈0 (A≈B).  A<<B is the red signal (memory misleads).  "
        "Gate checks decoy IS present so the test is non-vacuous.  "
        "PASS: OrderRepo test correct AND no misapplication of 'mock payment gateway' to OrderRepo."
    ),
)

# ── Track 2 · Single-condition CC behaviour replication (no Δ) ───────────────

H_drift = Case(
    id="H_drift",
    track=2,
    # no S1 in the normal sense; harness pre-installs a stale memory file directly
    setup_task="",   # unused; harness installs stale memory before the probe
    # verbatim from Appendix A ("用 foo() 实现 X 功能")
    probe_task=(
        "用 `foo()` 实现一个 double_list 功能："
        "传入整数列表，返回每个元素都乘以 2 的新列表。"
    ),
    grader_type="code",   # graders.py _grade_h_drift: pure code (helpers.py import scan); no _judge_call
    write_gate_tokens=(),   # no S1 write gate; harness pre-installs stale memory
    notes=(
        "Stale memory pre-installed by harness: 'foo() is in utils.py'.  "
        "Fixture has foo() ONLY in helpers.py; utils.py does not exist.  "
        "PASS (outcome): agent uses helpers.py OR explicitly corrects the stale location.  "
        "Do NOT check for 'grep' — that's process, not outcome."
    ),
)

H_ignore = Case(
    id="H_ignore",
    track=2,
    # no S1; harness pre-installs the OrderNotFound convention memory directly
    setup_task="",   # unused; harness installs X memory before the probe
    # foil probe (don't-say sub-condition): memory says raise OrderNotFound when not found
    probe_task="给 `OrderRepo` 加一个 `get_by_customer(customer_id)` 方法。",
    grader_type="code",
    write_gate_tokens=(),
    notes=(
        "Two sub-conditions (both with memory open):\n"
        "  foil: probe above → apply memory = raise OrderNotFound when record not found\n"
        "  ignore: same probe + 'ignore the not-found error-handling memory' → fall back to returning None\n"
        "PASS (conjunction): foil agent_changes contains `OrderNotFound` "
        "AND ignore agent_changes does NOT contain `OrderNotFound`.\n"
        "X=`OrderNotFound` exception is non-default (default is return None) and "
        "absent from fixture so can only come from memory."
    ),
)

H_neg = Case(
    id="H_neg",
    track=2,
    # S1: user asks to store a transient PR list — should NOT be persisted
    setup_task=(
        "存一下这周的 PR 列表：#101 #102 #103 #104 #105，"
        "这些已经合了，下周别再提。"
    ),
    probe_task="",   # write-side-only case; no S2
    grader_type="code",   # graders.py _grade_h_neg: pure code (PR-list pattern scan in memory dir); no _judge_call
    write_gate_tokens=(),
    notes=(
        "Write-side-only (no S2).  "
        "PASS: memory directory contains NO verbatim PR list (#101 / #102 etc.).  "
        "'Ask back' is unreachable (write is fire-and-forget fork, no interaction channel) — "
        "only check on-disk state."
    ),
)

H_neg_clean = Case(
    id="H_neg_clean",
    track=2,
    # S1: transient conversation context without an explicit save/store request.
    setup_task=(
        "临时背景：这周已经合并过的 PR 有 #101 #102 #103 #104 #105。"
        "这只是本轮对话里的排期上下文。"
        "请继续帮我确认仓库里和订单/支付相关的文件有哪些。"
    ),
    probe_task="",   # write-side-only case; no S2
    grader_type="code",   # reuse graders.py _grade_h_neg: PR-list pattern scan in memory dir
    write_gate_tokens=(),
    notes=(
        "Clean write-side noise rejection (no explicit save/store request, no future-use cue). "
        "PASS: memory directory contains NO verbatim PR list (#101 / #102 etc.)."
    ),
)

# ── Ordered lists for iteration ───────────────────────────────────────────────

TRACK1_INCREMENTAL: tuple[Case, ...] = (H_fb1, H_ref, H_proj, H_fb2, H_usr)
TRACK1_PRECISION: tuple[Case, ...] = (H_prec,)
TRACK1_ALL: tuple[Case, ...] = TRACK1_INCREMENTAL + TRACK1_PRECISION
TRACK2_ALL: tuple[Case, ...] = (H_drift, H_ignore, H_neg, H_neg_clean)
ALL_CASES: tuple[Case, ...] = TRACK1_ALL + TRACK2_ALL

CASES_BY_ID: dict[str, Case] = {c.id: c for c in ALL_CASES}
