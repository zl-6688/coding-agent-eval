"""EvoClaw 逐里程碑隔离评分 → extract_curves `--verdicts` 的 JSONL 桥。

extract_curves 的 `--verdicts` 契约（`eval/compression_eval/extract_curves.py::load_verdicts`）：
  每行 `{"session_id":..., "milestone":..., "resolved":0/1}`，**join 键 = (session_id, milestone)**。
  其中 `milestone` 必须 == agent 实际打的 `git tag agent-impl-<mid>` 里的 `<mid>`
  （extract_curves 从 trace 的 tool.bash span 用 `agent-impl-([..])` 正则抽 mid）；
  `session_id` 必须 == trace `meta.session_id`（= harness 传给 `--session-id` 的 UUID）。
  → 本桥**只负责把 EvoClaw 的真分搬成这个格式**，不做 normalize；命名对不对得上由真跑核对（P1-1）。

★★ infra 失败 ≠ 真失败（必修，防 sb-cli 0/30 血泪重演）：
  EvoClaw 隔离评分容器可能因 **环境/infra 原因**起不来（docker `--cpus` 超宿主核数 exit125、
  超时、OOM）→ 里程碑落 `milestone_status.error`、**没产 evaluation_result.json**。
  **这种 infra 失败绝不能当 `resolved=0`**：那会把"评分管线坏了"误读成"agent/压缩能力差"，污染退化
  曲线的 resolve 高度（红队反复咬过的 confound）。三态区分：
    - **scored**（测试真跑了，产了 evaluation_result.json，或 summary 记 passed/failed）→ resolved=0/1（真分）。
    - **infra_error**（summary 记 error 且无 evaluation_result.json）→ resolved=null + status="infra_error"。
    - **not_evaluated**（available/blocked/submitted/skipped，没轮到评）→ resolved=null + status="not_evaluated"。
  本桥输出**每行带 `status`**；infra_error/not_evaluated 行的 `resolved=null`，**不当 0**。

  ⚠ 对 extract_curves 的要求（已报 lead 转 A）：当前 `load_verdicts` 做 `1 if d.get("resolved") else 0`
  会把 `null` 强制成 0（= 又把 infra 当真失败）。**A 需改 load_verdicts：status=="infra_error"/"not_evaluated"
  或 resolved is None 的行 → 整点逐出（既不当真 0、也不当 proxy 1），不混进退化曲线。** 在 A 改之前，
  喂当前 extract_curves 请加 `--scored-only`（只出 scored 行；本次 E0 单里程碑全 scored、无差别）。

EvoClaw 评分产物（`<trial_root>/evaluation/`，源码：orchestrator.py / evaluator.py）：
  - 每里程碑 `<milestone_id>/evaluation_result(_filtered).json`，含 `milestone_id` + `resolved`(bool)。
  - `summary.json`：`milestone_status.{passed,failed,error,available,blocked,submitted,skipped}` = milestone_id 列表。
  session_id：trial 下 `session_id.txt`（agent_runner 落盘）。

用法（WSL，纯 stdlib，无依赖）：
    python3 eval/evoclaw/verdicts_bridge.py --trial <trial_root> [--session-id <sid>] [--out verdicts.jsonl] [--scored-only]
    # --out 省略或 '-' → stdout。诊断（scored/infra 分类 + 三方真实值）打 stderr。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# summary.milestone_status 各桶 → (status, resolved)。error=infra（不是真 0！）。
_BUCKET = {
    "passed": ("scored", 1),
    "failed": ("scored", 0),          # 真跑了测试没过 = 真失败
    "error": ("infra_error", None),   # ★ 评分 infra 坏了，不是 agent 没过 → null，绝不当 0
    "available": ("not_evaluated", None),
    "blocked": ("not_evaluated", None),
    "submitted": ("not_evaluated", None),   # 提交了但还没评完
    "skipped": ("not_evaluated", None),
    "early_unlocked": ("not_evaluated", None),  # 仅 DAG 解锁标记，非评分态
}


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def find_session_id(trial: Path, override: str | None) -> str:
    if override:
        return override
    for sf in sorted(trial.rglob("session_id.txt")):
        txt = sf.read_text(encoding="utf-8", errors="replace").strip()
        if txt:
            return txt
    for sj in sorted(trial.rglob("summary.json")):
        d = _load_json(sj) or {}
        for k in ("session_id", "agent_session_id"):
            if d.get(k):
                return str(d[k])
    return ""


def scored_from_eval_results(trial: Path) -> dict[str, int]:
    """每里程碑 evaluation_result(_filtered).json → {mid: resolved01}。**有这文件 = 测试真跑了 = scored**。
    优先 _filtered（按阈值过滤后的最终判定）。"""
    out: dict[str, int] = {}
    seen_filtered: set[str] = set()
    for fname, is_filtered in (("evaluation_result_filtered.json", True),
                               ("evaluation_result.json", False)):
        for p in sorted(trial.rglob(fname)):
            d = _load_json(p)
            if not d or not d.get("milestone_id"):
                continue
            mid = str(d["milestone_id"])
            if is_filtered:
                out[mid] = 1 if d.get("resolved") else 0
                seen_filtered.add(mid)
            elif mid not in seen_filtered:
                out[mid] = 1 if d.get("resolved") else 0
    return out


def buckets_from_summary(trial: Path) -> dict[str, str]:
    """summary.json milestone_status → {mid: bucket}。同 mid 多桶取"最确定"的（scored > infra > not_eval）。"""
    prio = {"passed": 3, "failed": 3, "error": 2, "submitted": 1,
            "available": 0, "blocked": 0, "skipped": 0, "early_unlocked": 0}
    best: dict[str, str] = {}
    for sj in sorted(trial.rglob("summary.json")):
        d = _load_json(sj) or {}
        ms = d.get("milestone_status") or {}
        for bucket, mids in ms.items():
            for mid in (mids or []):
                mid = str(mid)
                if mid not in best or prio.get(bucket, 0) > prio.get(best[mid], 0):
                    best[mid] = bucket
    return best


def build_rows(trial: Path, session_id: str) -> list[dict]:
    """三态分类每里程碑 → 行 {session_id, milestone, resolved(0/1/None), status}。"""
    scored = scored_from_eval_results(trial)    # 权威真分（测试真跑过）
    buckets = buckets_from_summary(trial)       # 分类来源（区分 scored/infra/not-eval）
    rows = []
    for mid in sorted(set(scored) | set(buckets)):
        if mid in scored:
            status, resolved = "scored", scored[mid]
        else:
            status, resolved = _BUCKET.get(buckets.get(mid, ""), ("not_evaluated", None))
        rows.append({"session_id": session_id, "milestone": mid,
                     "resolved": resolved, "status": status})
    # 诊断打 stderr（不污染 stdout JSONL），供 P1-1 三方核对 + headroom 体检。
    n_scored = sum(1 for r in rows if r["status"] == "scored")
    n_pass = sum(1 for r in rows if r["resolved"] == 1)
    n_fail = sum(1 for r in rows if r["status"] == "scored" and r["resolved"] == 0)
    n_infra = sum(1 for r in rows if r["status"] == "infra_error")
    n_ne = sum(1 for r in rows if r["status"] == "not_evaluated")
    print(f"[verdicts_bridge] session_id={session_id!r}  里程碑={len(rows)}  "
          f"scored={n_scored}(pass={n_pass}/fail={n_fail})  infra_error={n_infra}  not_evaluated={n_ne}",
          file=sys.stderr)
    for r in rows:
        print(f"  milestone={r['milestone']!r}  resolved={r['resolved']}  status={r['status']}", file=sys.stderr)
    if n_infra:
        print("  ⚠ infra_error 行 resolved=null（评分 infra 坏了，非真失败）——extract_curves 应逐出这些点，"
              "别当真 0（见模块 docstring 对 A 的要求）。", file=sys.stderr)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EvoClaw 评分 → extract_curves --verdicts JSONL 桥（infra≠真失败）")
    ap.add_argument("--trial", required=True, type=Path, help="EvoClaw trial 根目录（含 evaluation/ 或 summary.json）")
    ap.add_argument("--session-id", default=None, help="覆盖 session_id（默认从 trial 下 session_id.txt 读）")
    ap.add_argument("--out", default="-", help="输出 JSONL 路径（默认 '-' = stdout）")
    ap.add_argument("--scored-only", action="store_true",
                    help="只出 scored 行（resolved 0/1），逐出 infra_error/not_evaluated。"
                         "喂当前未改的 extract_curves 时用（避免 null→0 误当真失败）。")
    a = ap.parse_args(argv)

    if not a.trial.exists():
        print(f"[ERR] trial 目录不存在: {a.trial}", file=sys.stderr)
        return 2
    sid = find_session_id(a.trial, a.session_id)
    if not sid:
        print("[WARN] 没找到 session_id（session_id.txt 缺失且未 --session-id）→ join 会失败，"
              "请显式传 --session-id（= harness UUID = trace meta.session_id）", file=sys.stderr)
    rows = build_rows(a.trial, sid)
    if a.scored_only:
        rows = [{"session_id": r["session_id"], "milestone": r["milestone"], "resolved": r["resolved"]}
                for r in rows if r["status"] == "scored"]
        print(f"[verdicts_bridge] --scored-only：保留 {len(rows)} 个 scored 行", file=sys.stderr)
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")
    if a.out == "-":
        sys.stdout.write(text)
    else:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"[verdicts_bridge] 写出 {len(rows)} 行 → {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
