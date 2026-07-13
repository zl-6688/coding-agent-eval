"""resume-from-compressed-context 评测（薄切片）。

回答的问题（对齐 DECISIONS.md §3 / §3.1）：
  对 coding agent，压缩的成败不是「压缩后还答得出原文事实吗」（那是通用 QA），
  而是「**压缩后 agent 还能继续把代码任务做完吗**」。本脚本做最小可信的度量：

  1. 生成一段真实长历史：让我们的 agent 通读 3 个真实源文件，上下文一过切点 C 就快照停。
  2. 同一快照分三臂（同一 repo 拷贝、同一续作指令，只有「历史怎么处理」不同）：
        full       —— 不压（拿完整 C tokens 续作）          = 上界对照
        compressed —— compact_pipeline（micro→full→A2 重注文件）= 我们的 CC 式智能压缩
        truncated  —— compact_naive（drop-oldest 硬截断）      = 地板对照
     **compressed − truncated 才是「智能压缩」的真实增量**；full 标出「不压能到哪」。
  3. 续作指令故意要求「**基于你刚读过的三个文件**写 notes.md」→ 谁丢了文件原文谁就得重读。
     头指标 = **重读率**（续作里 read_file 命中历史已读过的路径）：full≤compressed≤truncated
     就是压缩损伤的可视证据，且 compressed 靠 A2（post-compact 文件重注）把重读压下来 = 验证 A2。

薄切片刻意的简化（先证「测量机器 + trace 能讲损伤故事」，再扩 N）：
  - 切点缩放到 8K token（非生产 128K/179K）、本地源码当 repo、不接 Docker、不评官方 resolved。
  - 单任务、单次历史。多轮压缩漂移 / 配对 McNemar / 约束保留探针留作下一步（见 TODO.md）。

跑法（项目根，需带代理 + .env）：
  HTTPS_PROXY=http://<your-proxy> HTTP_PROXY=http://<your-proxy> python eval/_archive/compact_eval/resume_eval.py
"""

import copy
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print(
        "usage: resume_eval.py [cut_tokens] [target_tokens] [repetitions]\n\n"
        "Run the archived live resume-from-compressed-context experiment."
    )
    raise SystemExit(0)

from agent import config, loop          # noqa: E402
from agent.context import compact       # noqa: E402
from eval import judge                            # noqa: E402
from obs.trace import JsonlSink, set_sink         # noqa: E402

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
ARMS = ("full", "compressed", "truncated")
# CONT_TASK 问到的具体函数 → judge 只拿这几段真实源码当基准（比塞整文件小得多、更准、judge 更稳）
REFERENCE_FUNCS = {"compact.py": ["microcompact"], "tools.py": ["dispatch"],
                   "loop.py": ["run_task"], "llm.py": ["chat", "_create_with_retry"]}

# 让 agent 通读的真实源文件（多读几个、强制一轮一个 → 历史变长变碎，压缩/截断才有"渐进退化"可言；
# 否则一轮读完全部 → 历史只有 3 条消息，截断变成全有/全无的极端、压缩也退化，数字不可信）。
SEED_FILES = ["compact.py", "tools.py", "loop.py", "llm.py", "config.py", "tool_store.py"]
GEN_TASK = (
    "仓库根目录下有 6 个 Python 文件：compact.py、tools.py、loop.py、llm.py、config.py、tool_store.py。"
    "请**一轮只 read_file 一个文件**（绝对不要在同一轮里读多个），逐个完整通读："
    "每读完一个，用一句话说它干嘛，然后再读下一个，直到 6 个都读完才收尾。"
)
CONT_TASK = (
    "现在**基于你刚才逐个读过的那些源文件**，在 notes.md 里分别写出这几个函数各自做什么："
    "compact.py 的 microcompact、tools.py 的 dispatch、loop.py 的 run_task、llm.py 里实际发请求的函数"
    "（每个两三句，依据你读到的源码内容写，别重新猜）。写完用 write_file 存成 notes.md 就收尾。"
)


def _scaled_cfg(target: int) -> "compact.CompactConfig":
    """把 keep / A2 重注预算缩到 target 量级。

    CC 的 A2 常量（5 文件 / 5K 每文件 / 50K 总）是 128K 窗口尺度的；直接用在缩放 regime 里，
    重注的文件体积会**压过整段历史**（compressed 反而比 full 还大）。按 target 比例缩，让
    compressed 真的 < full，对比才公平。keep 同理（keep_min_msgs=5 在短历史里会留全部 → 不缩）。"""
    return compact.CompactConfig(
        keep_min_tokens=max(500, target // 4), keep_min_msgs=2, keep_max_tokens=max(1000, target // 2),
        post_compact_max_files=3, post_compact_max_tokens_per_file=max(500, target // 6),
        post_compact_token_budget=max(1000, target // 2),
    )


def _seed_workspace(dst: Path):
    """把真实源码铺进一个隔离 workspace，给 agent 当「仓库」探索。"""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in SEED_FILES:
        shutil.copy(REPO / "agent" / name, dst / name)


def _extract_func(src: str, fname: str) -> str:
    """抽一个顶层 `def fname(...)` 的函数体（到下一个顶层 def/class 或 EOF 为止）。
    judge 只需要被问到的函数原文，不必看整文件——prompt 更小、判得更准、更不易失败。"""
    lines = src.splitlines()
    out, capturing = [], False
    for ln in lines:
        if not capturing and ln.startswith(f"def {fname}("):
            capturing = True
        elif capturing and (ln.startswith("def ") or ln.startswith("class ")):
            break          # 下一个顶层定义 → 本函数结束
        if capturing:
            out.append(ln)
    return "\n".join(out)


def _reference_source() -> str:
    """拼 CONT_TASK 问到的那几个函数的真实源码当 judge 基准。"""
    parts = []
    for fname, funcs in REFERENCE_FUNCS.items():
        src = (REPO / "agent" / fname).read_text(encoding="utf-8")
        for fn in funcs:
            body = _extract_func(src, fn)
            if body:
                parts.append(f"# ===== {fname} :: {fn} =====\n{body}")
    return "\n\n".join(parts)


def _append_instruction(messages: list, text: str):
    """把续作指令接到历史末尾——末条是 user(tool_result) 就并进它，避免出现两条连续 user。"""
    last = messages[-1]
    if last.get("role") == "user":
        c = last.get("content")
        if isinstance(c, list):
            c.append({"type": "text", "text": text})
        else:
            last["content"] = f"{c}\n\n{text}"
    else:
        messages.append({"role": "user", "content": text})


def _transform(arm: str, messages: list, target: int) -> list:
    """三臂各自的历史处理；compressed 用 A2 需要 _recent_files 仍在场（调用方保证）。"""
    msgs = copy.deepcopy(messages)
    if arm == "full":
        return msgs
    if arm == "compressed":
        return compact.compact_pipeline(msgs, system=loop.SYSTEM, cfg=_scaled_cfg(target), target_tokens=target)
    if arm == "truncated":
        return compact.compact_naive(msgs, target_tokens=target, system=loop.SYSTEM)
    raise ValueError(arm)


def _read_paths(trace_path: Path) -> list:
    """从一条 trace 里抽所有 read_file 读过的路径（按出现顺序）。"""
    paths = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        a = ev.get("attributes", {})
        if a.get("tool.name") == "read_file":
            paths.append(str(a.get("tool.arg", "")))
    return paths


def _metrics(trace_path: Path, hist_reads: set) -> dict:
    """从续作 trace 抽过程指标（差异化护城河——别人只报成败，我们能在 trace 上指出损伤在哪）。"""
    run = {}
    rereads = repeated_bash = nav = writes_notes = 0
    cont_reads = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        a = ev.get("attributes", {})
        if ev.get("name") == "agent.run":
            run = a
        tn = a.get("tool.name")
        if tn == "read_file":
            p = str(a.get("tool.arg", ""))
            cont_reads.append(p)
            if p in hist_reads:        # 续作里又读了历史已读过的文件 = 压缩把它弄丢了，被迫重读
                rereads += 1
        elif tn == "bash":
            if a.get("tool.repeated_command"):
                repeated_bash += 1
            if a.get("tool.command_kind") == "nav":
                nav += 1
        elif tn == "write_file" and "notes.md" in str(a.get("tool.arg", "")):
            writes_notes += 1
    return {
        "finished": bool(run.get("finished")),
        "outcome": run.get("outcome", "?"),
        "turns": run.get("turns", 0),
        "rereads": rereads,                       # ← 头指标：压缩损伤的可视证据
        "cont_reads": len(cont_reads),
        "repeated_bash": repeated_bash,
        "nav_calls": nav,
        "tool_errors": run.get("n_tool_errors", 0),
        "wrote_notes": writes_notes > 0,          # 任务是否真完成（产出 notes.md）
    }


def main():
    cut_c = int(sys.argv[1]) if len(sys.argv) > 1 else 14_000  # 快照切点（< 读完6文件的~16K → 探索中途快照）
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 6_000  # compressed 的压缩目标（truncated 会对齐其实际大小）
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3        # repeat-N：单跑是噪声，看 fabrications 分布
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = config.TRACES_DIR / f"resume_{stamp}"
    base.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # ── 1. 生成长历史（压缩=none，过切点即快照停）。在隔离 workspace 里跑，A2 后面要重读这些真文件。
    hist_ws = base / "ws_hist"
    _seed_workspace(hist_ws)
    hist_trace = base / "hist.jsonl"
    set_sink(JsonlSink(hist_trace))
    print(f"[1/3] 生成历史：通读 {SEED_FILES}，切点 {cut_c} tok …")
    with config.using_workdir(hist_ws):
        _, history = loop.run_task(GEN_TASK, max_turns=16, trace=False,
                                   compact_strategy="none", stop_at_context=cut_c,
                                   return_messages=True)
    hist_tokens = compact.estimate(history, loop.SYSTEM)
    hist_reads = set(_read_paths(hist_trace))
    print(f"      历史 = {hist_tokens} tok，{len(history)} 条消息，历史读过文件 {sorted(hist_reads)}")
    if hist_tokens < cut_c * 0.6:
        print("      ⚠ 历史没到切点（agent 提前收尾？）——结果仅供管线自检。")

    # ── 2. 三臂变换（必须在 _recent_files 仍在场时做完 compressed 的 A2 重注；故先全变换，再各自续作）。
    #    变换在 hist_ws 下做，A2 能重读到真实文件原文（CC 用 FileReadTool re-read 最新内容）。
    #    **同预算公平对比**：full_compact 不把 target 当下限、会压到最狠 → 先算 compressed 的实际大小 S，
    #    再把 truncated 截到同样的 S。否则两臂大小不同，测的是"压得狠 vs 松"而非"智能 vs 截断"。
    transformed = {}
    with config.using_workdir(hist_ws):
        transformed["full"] = _transform("full", history, target)
        transformed["compressed"] = _transform("compressed", history, target)
        s_compressed = compact.estimate(transformed["compressed"], loop.SYSTEM)
        transformed["truncated"] = _transform("truncated", history, s_compressed)
    print(f"      同预算对齐：compressed≈truncated≈{s_compressed} tok（full={hist_tokens}）")
    for arm in ARMS:
        _append_instruction(transformed[arm], CONT_TASK)

    # ── 3. 各臂续作 × repeat-N：同一历史、同一压缩后上下文，每臂续作跑 reps 次。
    #    为什么 repeat-N：agent「重读恢复 vs 硬写幻觉」的决策是随机的、且主导结果 → N=1 是噪声，
    #    单跑能让智能压缩看起来赢也能看起来输。固定上下文、只重复续作，隔离这层随机性看分布。
    reference = _reference_source()       # judge 的事实基准（真实源码），只建一次
    if not judge.judge_available():
        print("      ⚠ judge 不可用（JUDGE_MODEL_ID 未配或＝被测模型）→ 只出过程指标，无正确性轴。")
    trials = {arm: [] for arm in ARMS}    # arm -> 每次 rep 的指标 dict
    start_tok = {arm: compact.estimate(transformed[arm], loop.SYSTEM) for arm in ARMS}
    for rep in range(reps):
        for arm in ARMS:
            arm_ws = base / f"ws_{arm}_r{rep}"
            shutil.copytree(hist_ws, arm_ws)          # 每次 rep 一份干净 repo 快照
            arm_trace = base / f"{arm}_r{rep}.jsonl"
            set_sink(JsonlSink(arm_trace))
            print(f"[2/3] rep{rep} 续作 {arm:10s}：起始 {start_tok[arm]} tok（full≈{hist_tokens}）…")
            with config.using_workdir(arm_ws):
                loop.run_task(f"[resume:{arm}:r{rep}]", max_turns=8, trace=False,
                              compact_strategy="none", initial_messages=transformed[arm])
            m = _metrics(arm_trace, hist_reads)
            notes_fp = arm_ws / "notes.md"
            notes = notes_fp.read_text(encoding="utf-8") if notes_fp.exists() else ""
            jr = judge.judge_notes_accuracy(notes, reference)
            m["accuracy"] = jr.get("accuracy")
            m["fabrications"] = jr.get("fabrications")
            trials[arm].append(m)

    # ── 聚合 + 报告（均值是头条，原始列表暴露方差——单跑会骗人）
    def _vals(arm, k):
        return [t[k] for t in trials[arm] if t.get(k) is not None]

    def _mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    print(f"\n[3/3] 结果（切点 {cut_c} / 压缩目标 {target} / reps={reps}；历史 {hist_tokens} tok）")
    print(f"\n{'arm':12s}{'start_tok':>11s}{'acc均值':>11s}{'编造均值':>11s}{'重读均值':>11s}{'finished':>11s}")
    for arm in ARMS:
        fin = sum(1 for t in trials[arm] if t.get("finished"))
        print(f"{arm:12s}{start_tok[arm]:>11d}{str(_mean(_vals(arm,'accuracy'))):>11s}"
              f"{str(_mean(_vals(arm,'fabrications'))):>11s}{str(_mean(_vals(arm,'rereads'))):>11s}"
              f"{f'{fin}/{reps}':>11s}")
    print("\n── 各 rep 原始值（看方差——这正是 repeat-N 要暴露的）──")
    for arm in ARMS:
        print(f"  {arm:11s} 编造={_vals(arm,'fabrications')}  重读={_vals(arm,'rereads')}  "
              f"准确={_vals(arm,'accuracy')}")

    mf_f = _mean(_vals("full", "fabrications"))
    mf_c = _mean(_vals("compressed", "fabrications"))
    mf_t = _mean(_vals("truncated", "fabrications"))
    print(f"\n编造均值 full={mf_f}  compressed={mf_c}  truncated={mf_t}")
    if None not in (mf_c, mf_t):
        if mf_c < mf_t:
            print("  → 均值上 compressed 编造 < truncated：同预算下智能压缩更「可用」（不只是可恢复）。")
        elif mf_c > mf_t:
            print("  → 均值上 compressed 编造 > truncated：智能压缩此 regime 反而更差；首要嫌疑="
                  "A2 把每文件截到 ~target//6 太狠、给出「残缺却伪装完整」的文件 → 该修。")
        else:
            print("  → 两臂编造均值持平：差异在方差里，需加大 reps 或加难度。")

    report = {"cut_c": cut_c, "target": target, "reps": reps, "hist_tokens": hist_tokens,
              "hist_reads": sorted(hist_reads), "start_tokens": start_tok,
              "trials": trials, "trace_dir": str(base)}
    out = REPORTS / f"resume_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告 → {out}\ntrace → {base}")


if __name__ == "__main__":
    main()
