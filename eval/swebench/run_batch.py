"""批量在 SWE-bench Lite 全部实例上跑 agent —— 断点续跑 + 结果落盘 + 聚合 localization%。

设计要点(为什么这么写):
  - **落盘 batch_results.jsonl**:300 实例跑 20-30h,不能靠内存/context 攒结果;每个实例跑完
    立刻 append 一行,进程崩了/被停了也不丢,天然支持断点续跑(已在 results 里的跳过)。
  - **顺序执行**:agent 用全局 config.WORKDIR(using_workdir 上下文),线程并发会互踩;要提速
    用多进程分片(各跑各的实例子集),但 append 同一文件在 Windows 上可能交错,故默认单进程。
  - **proxy 评分**:与官方 Docker harness 一致的口径是"改对文件没有"(gold patch 文件重合)。

用法:
    python -m eval.swebench.run_batch --instances <dataset.json>
    python -m eval.swebench.run_batch --instances <dataset.json> 20
    python -m eval.swebench.run_batch --instances <dataset.json> --summary
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agent import config
from agent.loop import EvalHooks, run_task
from agent.mcp.runtime_config import resolve_run_task_runtime_kwargs
from obs.trace import get_sink
from eval.swebench.run_swe import (agent_changed_files, build_task, clone,
                                   git_diff, gold_files, maybe_init_otel,
                                   run_one_indocker)

INSTANCES: Path | None = None
RESULTS = Path(__file__).resolve().parent / "batch_results.jsonl"
IN_DOCKER = False   # --in-docker 时 True：agent 在实例容器内跑（能跑真测试）
VERSION_TAG = ""    # --tag 值：作为 trace metadata.version，供 Phoenix 按版本对比


def _meta(inst: dict) -> dict:
    """身份 metadata（盖到 agent.run span，供 Phoenix 筛选/对比 + 区分每次执行）。"""
    return {"instance_id": inst["instance_id"], "repo": inst.get("repo", ""),
            "version": VERSION_TAG or "default",
            "mode": "in-docker" if IN_DOCKER else "blind"}


def done_ids() -> set:
    """已"成功跑过"的实例 = 有非 error 结果。error 行(如 clone 失败)不算 done，下次自动重试。"""
    if not RESULTS.exists():
        return set()
    out = set()
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                if not r.get("error"):
                    out.add(r["instance_id"])
            except Exception:
                pass
    return out


def run_one(inst: dict) -> dict:
    """跑一个实例，从 trace 聚合**诊断字段**：不止 hit/miss，还有失败原因、读/改了哪些文件、
    bash 报错数、轮次、stop_reason、trace 路径——让 miss 可复盘、瓶颈可定位。
    """
    maybe_init_otel()   # 每个子进程独立初始化（OTEL_EXPORT=1 时导 Phoenix；幂等）
    gf = gold_files(inst["patch"])
    ws = Path(tempfile.mkdtemp(prefix="swe_"))
    try:
        clone(inst["repo"], inst["base_commit"], ws)
        task = build_task(inst, ws)
        mcp_run_task_kwargs = resolve_run_task_runtime_kwargs()
        with config.using_workdir(ws):
            final_text = run_task(task, max_turns=50,
                                  eval_hooks=EvalHooks(compact_strategy="pipeline", compact_window=200_000),
                                  meta=_meta(inst),
                                  **mcp_run_task_kwargs)
            sink = get_sink()
            events = sink.events()
            trace_path = str(getattr(sink, "path", ""))

        llm_calls = [e for e in events if e["name"] == "llm.call"]
        peak = max((e["attributes"].get("context.tokens_sent", 0) for e in llm_calls), default=0)
        stop_reason = (llm_calls[-1]["attributes"].get("gen_ai.response.stop_reason", "")
                       if llm_calls else "")
        turns = sum(1 for e in events if e["name"] == "agent.turn")
        max_turns_reached = bool(final_text and final_text.startswith("(达到最大轮次"))

        tool_counts, read_files, edited_files, bash_errors = {}, [], [], 0
        bash_kinds, repeated_bash, nonzero_exit = {}, 0, 0
        for e in events:
            if not e["name"].startswith("tool."):
                continue
            a = e["attributes"]
            tn = a.get("tool.name", "?")
            tool_counts[tn] = tool_counts.get(tn, 0) + 1
            arg = a.get("tool.arg", "")
            if tn == "read_file":
                read_files.append(arg)
            elif tn in ("edit_file", "write_file"):
                edited_files.append(arg)
            if tn == "bash":
                k = a.get("tool.command_kind", "unknown")
                bash_kinds[k] = bash_kinds.get(k, 0) + 1
                if a.get("tool.repeated_command"):
                    repeated_bash += 1
                ec = a.get("tool.exit_code")
                if isinstance(ec, int) and ec != 0:
                    nonzero_exit += 1
                if a.get("tool.is_error"):
                    bash_errors += 1

        cf = agent_changed_files(ws)
        patch = git_diff(ws)   # 在 rmtree 前抓补丁，供官方 harness 评 resolved
        overlap = sorted(set(cf["all"]) & set(gf))
        hit = bool(overlap)
        if hit:
            reason = ""
        elif not cf["all"]:
            reason = "no_edit_max_turns" if max_turns_reached else "no_edit"
        else:
            reason = "wrong_file"
        return {"instance_id": inst["instance_id"], "repo": inst["repo"], "gold": gf,
                "localization_hit": hit, "failure_reason": reason, "overlap": overlap,
                "modified": cf["modified"], "untracked": cf["untracked"],
                "peak_context_estimated": peak, "turns": turns,
                "max_turns_reached": max_turns_reached, "stop_reason": stop_reason,
                "n_llm_calls": len(llm_calls),
                "n_compact": sum(1 for e in events if e["name"] == "compact.pipeline"),
                "tool_counts": tool_counts, "bash_error_count": bash_errors,
                "bash_kinds": bash_kinds, "repeated_bash": repeated_bash,
                "bash_nonzero_exit": nonzero_exit,
                "read_files": sorted(set(filter(None, read_files))),
                "edited_files": sorted(set(filter(None, edited_files))),
                "final_text": (final_text or "")[:500], "trace_path": trace_path,
                "model_patch": patch}
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        return {"instance_id": inst["instance_id"], "repo": inst["repo"],
                "localization_hit": False,
                "failure_reason": "clone_error" if "clone" in msg.lower() else "exception",
                "error": msg}
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def summarize():
    if not RESULTS.exists():
        print("(还没有结果)")
        return
    raw = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    dedup = {}
    for r in raw:
        dedup[r.get("instance_id")] = r   # 同 id 后写覆盖前：重试成功覆盖旧 error 行
    rows = list(dedup.values())
    n = len(rows)
    hits = sum(1 for r in rows if r.get("localization_hit"))
    errs = sum(1 for r in rows if r.get("error"))
    misses = [r for r in rows if not r.get("localization_hit") and not r.get("error")]
    fr = {}
    for r in misses:
        k = r.get("failure_reason", "?")
        fr[k] = fr.get(k, 0) + 1
    ctxs = sorted(r.get("peak_context_estimated", 0) for r in rows if not r.get("error"))
    compacted = sum(1 for r in rows if r.get("n_compact", 0) > 0)
    by_repo = {}
    for r in rows:
        rp = r.get("repo", "?")
        by_repo.setdefault(rp, [0, 0])
        by_repo[rp][1] += 1
        if r.get("localization_hit"):
            by_repo[rp][0] += 1
    print(f"\n=== 聚合 (n={n}) ===")
    print("注意: localization_hit = agent 改到了 gold 文件，**≠ SWE resolved**(未跑官方测试)；"
          "上下文为**估算**(字符//4)。")
    print(f"localization_hit: {hits}/{n} = {hits / max(1, n):.1%}   (运行错误 {errs})")
    if fr:
        print("miss 分类: " + ", ".join(f"{k}={v}" for k, v in sorted(fr.items(), key=lambda x: -x[1])))
    if ctxs:
        print(f"峰值上下文(估算): 中位 {ctxs[len(ctxs) // 2]:,} / 最大 {ctxs[-1]:,} tok   "
              f"触发压缩(>167K): {compacted}/{n}")
    for rp, (h, t) in sorted(by_repo.items()):
        print(f"  {rp}: {h}/{t} = {h / max(1, t):.0%}")


def export_predictions(tag):
    """把 batch_results 转成 SWE-bench predictions JSONL（交 sb-cli / 官方 harness 评分）。

    每行 {instance_id, model_name_or_path, model_patch}；model_name_or_path 编入版本号，
    便于按 agent 版本对比 resolved 率。无补丁的样本也写空 patch（计入 attempted 分母）。
    """
    if INSTANCES is None:
        raise RuntimeError("run_batch requires --instances before exporting predictions")
    if not RESULTS.exists():
        print("(还没有结果，先跑 batch)")
        return
    raw = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    dedup = {}
    for r in raw:
        dedup[r.get("instance_id")] = r   # 同 id 后写覆盖前
    rows = [r for r in dedup.values() if not r.get("error")]
    model = f"coding-agent-{tag or 'default'}"
    outp = INSTANCES.parent / f"predictions_{tag or 'default'}.jsonl"
    with_patch = 0
    with outp.open("w", encoding="utf-8") as f:
        for r in rows:
            mp = r.get("model_patch", "") or ""
            with_patch += bool(mp.strip())
            f.write(json.dumps({"instance_id": r["instance_id"],
                                "model_name_or_path": model,
                                "model_patch": mp}, ensure_ascii=False) + "\n")
    print(f"predictions → {outp}")
    print(f"  {len(rows)} 实例（有补丁 {with_patch} / 空补丁 {len(rows) - with_patch}），"
          f"model_name_or_path={model}")
    print(f"  提交: sb-cli submit swe-bench_lite test --predictions_path {outp.name} --run_id {tag or 'run'}")


def _write(r):
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _progress(k, total, r, secs):
    tag = "✓HIT" if r.get("localization_hit") else ("ERR " if r.get("error") else "✗" + (r.get("failure_reason") or "miss"))
    extra = r.get("error") or f"ctx={r.get('peak_context_estimated', '-')} turns={r.get('turns', '-')}"
    t = f" ({secs:.0f}s)" if secs else ""
    print(f"[{k}/{total}] {tag} {r['instance_id']:32s} {extra}{t}", flush=True)


def run_sequential(todo):
    consec = 0
    for k, inst in enumerate(todo, 1):
        t0 = time.time()
        r = run_one_indocker(inst, meta=_meta(inst)) if IN_DOCKER else run_one(inst)
        _write(r)
        _progress(k, len(todo), r, time.time() - t0)
        if r.get("failure_reason") == "clone_error":
            consec += 1
            if consec >= 5:
                print(f"\n⚠️ 连续 {consec} 个 clone 失败 —— 网络/代理疑似中断，提前中止。"
                      f"恢复后重跑本命令即可自动续。", flush=True)
                break
        else:
            consec = 0


def run_parallel(todo, workers):
    """多进程并行：每进程独立 WORKDIR/sink 全局，互不串；主进程单写结果，避免 append 竞争。
    deepseek-v4-flash 支持高并发，瓶颈在 git clone + 30 轮串行 LLM，故按"多实例并行"提速。"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    clone_err = 0
    done_n = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, inst): inst for inst in todo}
        for fut in as_completed(futs):
            r = fut.result()
            _write(r)
            done_n += 1
            _progress(done_n, len(todo), r, 0)
            if r.get("failure_reason") == "clone_error":
                clone_err += 1
                if clone_err >= 10:
                    print(f"\n⚠️ 已 {clone_err} 个 clone 失败 —— 网络/代理疑似中断，取消剩余任务。"
                          f"恢复后重跑自动续(error 行不算 done)。", flush=True)
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
            else:
                clone_err = max(0, clone_err - 1)   # 偶发不累积，只在持续失败时触发


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        required=True,
        type=Path,
        help="Path to a complete external SWE-bench dataset JSON list.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        help="Optional ace.swebench-suite.v1 ID-only manifest to hydrate.",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--in-docker", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--predictions", action="store_true")
    parser.add_argument("limit", nargs="?", type=int)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    tag = args.tag
    workers = args.workers
    global RESULTS, INSTANCES, IN_DOCKER, VERSION_TAG
    IN_DOCKER = args.in_docker
    VERSION_TAG = tag
    INSTANCES = args.instances.expanduser().resolve()
    if IN_DOCKER:
        workers = 1   # 容器并发吃 RAM，强制串行
    RESULTS = INSTANCES.parent / (f"batch_results_{tag}.jsonl" if tag else "batch_results.jsonl")

    if args.summary:
        summarize()
        return

    if args.predictions:
        export_predictions(tag)
        return

    from eval.swebench.data import (  # noqa: WPS433
        DatasetError,
        SuiteManifestError,
        load_instances_for_run,
    )

    try:
        insts = load_instances_for_run(INSTANCES, args.suite)
    except (DatasetError, SuiteManifestError) as exc:
        parser.error(str(exc))
    done = done_ids()
    todo = [i for i in insts if i["instance_id"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"总 {len(insts)} 实例，已完成 {len(done)}，本次跑 {len(todo)}"
          f"（tag={tag or 'default'}, workers={workers}）", flush=True)

    if workers > 1:
        run_parallel(todo, workers)
    else:
        run_sequential(todo)

    summarize()


if __name__ == "__main__":
    main()
