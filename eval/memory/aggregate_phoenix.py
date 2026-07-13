"""eval/memory/aggregate_phoenix.py — Rollup aggregate table + Phoenix Experiment.

从 JSONL 计算两张聚合表（复用 run.py 口径），推成 Phoenix Experiment。

Usage:
    NO_PROXY=localhost,127.0.0.1 python eval/memory/aggregate_phoenix.py \
        --jsonl eval/memory/results/2026-07-06T06-51-31.jsonl

WHY: Phoenix 原生 Experiment 提供 per-example 下钻 + 总通过率；
     rollup 表放进 description 让 Experiments 页一眼看到聚合全貌。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from outcome import classify_record

DEFAULT_JSONL_PATH = REPO / "eval/memory/results/2026-06-30T09-24-44.jsonl"
DEFAULT_PHOENIX = "http://localhost:6006"


@dataclass(frozen=True)
class AggregateConfig:
    jsonl_path: Path
    batch: str
    experiment_name: str
    phoenix_url: str
    run_meta: dict[str, Any]
    reference_check: bool = False

# Track 分类（与 run.py 的 TRACK1_INCREMENTAL / TRACK1_PRECISION / TRACK2_ALL 对应）
TRACK1_INC_IDS  = {"H_ref", "H_proj", "H_fb2", "H_usr", "H_fb1"}
TRACK1_PREC_IDS = {"H_prec"}
TRACK2_IDS      = {"H_drift", "H_ignore", "H_neg", "H_neg_clean"}

# H_fb1 是结构性剔除例，不计入增量统计（沿用 02-results 处理方式）
EXCLUDED_INC_IDS = {"H_fb1"}


# ── 复用 run.py 指标口径（逐字复制，不改算法） ─────────────────────────────────

def _metrics_track1_incremental(records: list[dict]) -> dict:
    """Aggregate track-1 incremental metrics over k runs.
    Verbatim copy from run.py _metrics_track1_incremental.
    """
    by_case: dict[str, dict] = defaultdict(lambda: {"A": [], "B": []})
    for r in records:
        arm = r["arm"]
        if arm not in ("A", "B"):
            continue
        by_case[r["case_id"]][arm].append(r)

    per_case = {}
    total_delta_sum = 0.0
    total_cases = 0

    for case_id, arms in sorted(by_case.items()):
        a_runs = arms["A"]
        b_runs = arms["B"]

        a_eligible = [r for r in a_runs if r.get("write_pass") is not None]
        n_s1_incomplete = len(a_runs) - len(a_eligible)

        a_write = [r for r in a_eligible if r.get("write_pass") is True]
        p_write = len(a_write) / max(1, len(a_eligible))

        a_use = [r for r in a_write if r.get("verdict") == "PASS"]
        p_use_given_write = len(a_use) / max(1, len(a_write))

        a_pass = [r for r in a_eligible if r.get("verdict") == "PASS"]
        p_a_e2e = len(a_pass) / max(1, len(a_eligible))

        b_pass = [r for r in b_runs if r.get("verdict") == "PASS"]
        p_b = len(b_pass) / max(1, len(b_runs))

        delta = p_use_given_write - p_b
        total_delta_sum += delta
        total_cases += 1

        per_case[case_id] = {
            "n_a":             len(a_runs),
            "n_a_eligible":    len(a_eligible),
            "n_s1_incomplete": n_s1_incomplete,
            "n_b":             len(b_runs),
            "p_write":         round(p_write, 3),
            "p_use_given_write": round(p_use_given_write, 3),
            "p_a_e2e":         round(p_a_e2e, 3),
            "p_b":             round(p_b, 3),
            "delta":           round(delta, 3),
            # raw counts for display
            "_a_write_n":      len(a_write),
            "_a_use_n":        len(a_use),
            "_a_pass_n":       len(a_pass),
            "_b_pass_n":       len(b_pass),
        }

    total_delta = round(total_delta_sum / max(1, total_cases), 3)
    return {"per_case": per_case, "total_delta": total_delta}


def _metrics_h_prec(records: list[dict]) -> dict:
    """H_prec precision axis. Synced from run.py _metrics_h_prec (P1-5 fix).

    WHY use `is True` not `is not False`: write_pass="" (S1_INCOMPLETE) is not False
    but is also not True.  Counting those runs as valid deflates p_a into a false RED.
    Only runs where the decoy was actually stored (write_pass is True) are valid.
    """
    a_runs  = [r for r in records if r["arm"] == "A"]
    b_runs  = [r for r in records if r["arm"] == "B"]
    # valid = decoy actually stored; excludes WRITE_FAIL (False) and S1_INCOMPLETE ("")
    a_valid = [r for r in a_runs if r.get("write_pass") is True]
    a_vacuous = len(a_runs) - len(a_valid)
    p_a   = sum(1 for r in a_valid if r["verdict"] == "PASS") / max(1, len(a_valid))
    p_b   = sum(1 for r in b_runs  if r["verdict"] == "PASS") / max(1, len(b_runs))
    delta = round(p_a - p_b, 3)
    if len(a_valid) == 0:
        interp = "INCONCLUSIVE (no valid A samples — decoy not stored)"
    elif abs(delta) < 0.2:
        interp = "OK (Δ≈0)"
    elif delta < -0.2:
        interp = "RED: A<<B (memory misleads)"
    else:
        interp = "WARN: A>>B (unexpected)"
    return {
        "n_a_total":    len(a_runs),
        "n_a_valid":    len(a_valid),
        "n_a_vacuous":  a_vacuous,
        "p_a":          round(p_a, 3),
        "p_b":          round(p_b, 3),
        "delta":        delta,
        "interpretation": interp,
    }


def _metrics_track2(records: list[dict]) -> dict:
    """Track-2 pass^k rates. Synced from run.py _metrics_track2 (VALID-only denominator).

    WHY filter by sample_status==VALID: ERROR samples (system failures, not logic failures)
    should not count in the denominator — they deflate pass_rate with false negatives.
    """
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_case[r["case_id"]].append(r)
    result = {}
    for case_id, runs in sorted(by_case.items()):
        # Only count VALID samples in denominator (mirrors run.py Iron Law 5)
        valid_runs = [r for r in runs if r.get("sample_status") == "VALID"]
        n_valid = len(valid_runs)
        n_pass = sum(1 for r in valid_runs if r["verdict"] == "PASS")
        result[case_id] = {
            "n":         len(runs),
            "n_valid":   n_valid,
            "pass_rate": round(n_pass / max(1, n_valid), 3),
            "n_pass":    n_pass,
            "verdicts":  [r["verdict"] for r in runs],
        }
    return result


# ── 读取 JSONL 并分桶 ────────────────────────────────────────────────────────────


def load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """读 JSONL，返回 (run_meta, records)。

    run.py 从 Phase-2 起把批次常量写到首行 run_meta；旧批次没有 run_meta 时
    保持兼容，返回空 dict 并继续读取样本记录。
    """
    run_meta: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "run_meta":
                run_meta = obj
                continue   # skip batch-level metadata row
            all_records.append(obj)

    return run_meta, all_records


def load_and_split(path: Path) -> tuple[list, list, list, list]:
    """读 JSONL，按 Track 分桶返回 (all, inc, prec, t2)。

    首行 run_meta（type=="run_meta"）自动跳过，不混入 all_records。
    """
    _, all_records = load_run(path)
    inc_records  = [r for r in all_records if r["case_id"] in TRACK1_INC_IDS]
    prec_records = [r for r in all_records if r["case_id"] in TRACK1_PREC_IDS]
    t2_records   = [r for r in all_records if r["case_id"] in TRACK2_IDS]
    return all_records, inc_records, prec_records, t2_records


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO / candidate


def _case_slug(cases: list[str]) -> str:
    if not cases:
        return "all"
    if len(cases) <= 3:
        return "-".join(cases)
    return f"{len(cases)}cases"


def _default_experiment_name(batch: str, run_meta: dict[str, Any], records: list[dict[str, Any]]) -> str:
    k = run_meta.get("k")
    cases = run_meta.get("cases")
    if not isinstance(cases, list) or not all(isinstance(c, str) for c in cases):
        cases = sorted({str(r.get("case_id", "unknown")) for r in records})

    if k:
        return f"memory-k{k}-{_case_slug(cases)}-{batch}"
    return f"memory-eval-{batch}"


def resolve_config(
    *,
    jsonl_path: str | Path = DEFAULT_JSONL_PATH,
    experiment_name: str | None = None,
    phoenix_url: str = DEFAULT_PHOENIX,
    reference_check: bool = False,
) -> AggregateConfig:
    """从 CLI 参数和 run_meta 派生 Phoenix 聚合配置。"""
    path = _resolve_path(jsonl_path)
    run_meta, records = load_run(path)
    batch = str(run_meta.get("run_id") or path.stem)
    exp_name = experiment_name or _default_experiment_name(batch, run_meta, records)

    return AggregateConfig(
        jsonl_path=path,
        batch=batch,
        experiment_name=exp_name,
        phoenix_url=phoenix_url.rstrip("/"),
        run_meta=run_meta,
        reference_check=reference_check,
    )


# ── 渲染 Rollup Markdown 表 ────────────────────────────────────────────────────

# H_fb1 的友好名（结构性剔除，显示时附注）
_CASE_LABEL: dict[str, str] = {
    "H_ref":    "mem_ref · Linear INGEST 位置",
    "H_proj":   "mem_proj · 冻结期冲突提示",
    "H_fb2":    "mem_fb2 · bundled PR 提交计划",
    "H_usr":    "mem_usr · Go 类比讲 React",
    "H_fb1":    "mem_fb1 · 别 mock DB（结构性剔除）",
    "H_prec":   "mem_prec · 精度轴（want Δ≈0）",
    "H_drift":  "mem_drift · CC H1 漂移验证",
    "H_ignore": "mem_ignore · CC H6 忽略遵守",
    "H_neg":    "mem_neg · CC H2 噪声拒绝",
    "H_neg_clean": "mem_neg_clean · 纯临时上下文拒写",
}

_STATUS_THRESHOLDS = {
    # 增量轨：Δ>0 + 通过视为 PASS，结构性剔除单列
    "inc": lambda m: (
        "剔除（结构性）" if m["case_id"] in EXCLUDED_INC_IDS
        else "PASS" if m["delta"] > 0
        else "FAIL"
    ),
    # 行为轨：通过率 ≥ 0.8 → 稳定；0.4~0.8 → 弱（有效测量，非 INCONCLUSIVE）；< 0.4 → 不稳
    # WHY 不写 INCONCLUSIVE：状态词纪律——INCONCLUSIVE 专指「0 有效样本、测不出」，
    # 而 0.4~0.8 是全 VALID 样本下的真实弱通过率，是「测出来就是弱」，不是「测不出」。
    "t2": lambda rate: (
        "稳定" if rate >= 0.8
        else "弱（有效·复刻不稳）" if rate >= 0.4
        else "不稳（复刻差）"
    ),
}


def render_rollup(
    inc_metrics: dict,
    prec_metrics: dict,
    t2_metrics: dict,
    batch: str,
) -> str:
    """渲染两张聚合 Markdown 表（增量轨 + 行为轨）+ 顶部总览。"""
    lines: list[str] = []

    # ── 顶部总览 ─────────────────────────────────────────────────────────────
    inc_pc = inc_metrics["per_case"]
    valid_inc = {cid: m for cid, m in inc_pc.items() if cid not in EXCLUDED_INC_IDS}
    all_delta_pos = all(m["delta"] > 0 for m in valid_inc.values())
    weak_cases_t2 = [cid for cid, m in t2_metrics.items() if m["pass_rate"] < 0.6]

    lines += [
        f"## Rollup 聚合视图 · batch={batch}",
        "",
        "### 总览",
        "",
        f"- **增量轨有效例**（不含结构性剔除 H_fb1）：{len(valid_inc)} 例，"
        f"Δ 全为正：{'是' if all_delta_pos else '否'}，"
        f"平均 Δ = {inc_metrics['total_delta']:+.3f}",
        f"- **精度轴（H_prec）**：{prec_metrics['interpretation']}",
        f"- **行为复刻轨**：{len(t2_metrics)} 例，"
        f"弱 / FAIL 用例：{', '.join(weak_cases_t2) if weak_cases_t2 else '无'}",
        "",
        "> **一眼定位弱点**：" + (
            "增量轨全正" if all_delta_pos else
            "增量轨有 Δ ≤ 0 用例——" +
            ", ".join(c for c, m in valid_inc.items() if m["delta"] <= 0)
        ) + (
            "；行为轨 " + (
                ", ".join(weak_cases_t2) + " 通过率低"
                if weak_cases_t2 else "行为轨全过"
            )
        ),
        "",
    ]

    # ── 表 1：增量轨（含 H_fb1 单独行 + H_prec 精度轴） ──────────────────────
    lines += [
        "### 增量轨（Track-1，A/B 对照，want Δ≫0）",
        "",
        "| 用例 | 写入成功率 | 召回应用率 | 端到端（A） | 对照组通过率（B） | Δ | 状态 |",
        "|------|--------:|-------:|-------:|-------:|--:|------|",
    ]
    for cid in ("H_ref", "H_proj", "H_fb2", "H_usr", "H_fb1"):
        if cid not in inc_pc:
            continue
        m = inc_pc[cid]
        na_e = m["n_a_eligible"]
        nb   = m["n_b"]
        write_str = f"{m['_a_write_n']}/{na_e}" if na_e > 0 else "n/a"
        use_str   = f"{m['_a_use_n']}/{m['_a_write_n']}" if m["_a_write_n"] > 0 else "n/a"
        e2e_str   = f"{m['_a_pass_n']}/{na_e}" if na_e > 0 else "n/a"
        b_str     = f"{m['_b_pass_n']}/{nb}"
        excl      = " （剔除）" if cid in EXCLUDED_INC_IDS else ""
        status    = "剔除（结构性）" if cid in EXCLUDED_INC_IDS else (
            "PASS" if m["delta"] > 0 else "FAIL"
        )
        label = _CASE_LABEL.get(cid, cid)
        lines.append(
            f"| `{cid}` {label} | {write_str} | {use_str} | "
            f"{e2e_str} | {b_str} | **{m['delta']:+.2f}** | {status} |"
        )

    lines += [""]

    # H_prec 精度轴
    pm = prec_metrics
    lines += [
        "**H_prec 精度轴（极性相反，want Δ≈0，不计入增量 Δ 平均）：**",
        "",
        f"| n_A(total) | n_A(valid) | n_A(vacuous) | P(A) | P(B) | Δ | 解读 |",
        f"|-----------|-----------|-------------|------|------|---|------|",
        f"| {pm['n_a_total']} | {pm['n_a_valid']} | {pm['n_a_vacuous']} | "
        f"{pm['p_a']:.2f} | {pm['p_b']:.2f} | {pm['delta']:+.2f} | {pm['interpretation']} |",
        "",
        f"**增量轨（4 有效例）平均 Δ = {inc_metrics['total_delta']:+.3f}**",
        "",
    ]

    # ── 表 2：行为复刻轨 ─────────────────────────────────────────────────────
    lines += [
        "### 行为复刻轨（Track-2，单条件，不计 Δ）",
        "",
        "| 用例 | 通过率（n/5） | 各次判定 | 状态 |",
        "|------|-------:|---------|------|",
    ]
    for cid in ("H_drift", "H_ignore", "H_neg", "H_neg_clean"):
        if cid not in t2_metrics:
            continue
        m    = t2_metrics[cid]
        rate = m["pass_rate"]
        n    = m["n"]
        np_  = m["n_pass"]
        stat = _STATUS_THRESHOLDS["t2"](rate)
        label = _CASE_LABEL.get(cid, cid)
        lines.append(
            f"| `{cid}` {label} | {np_}/{n} ({rate:.2f}) | "
            f"{','.join(m['verdicts'])} | {stat} |"
        )
    lines.append("")

    return "\n".join(lines)


# ── Phoenix Experiment ────────────────────────────────────────────────────────


def push_phoenix_experiment(
    all_records: list[dict],
    rollup_md: str,
    config: AggregateConfig,
) -> str:
    """创建 Phoenix Dataset + Experiment，返回 Experiment URL。

    - Dataset：每条 run 一个 example；input=case_id/arm/run_idx/probe；metadata 完整
    - Experiment：task 回放已记录的 verdict；evaluators pass(verdict==PASS→1 else 0)
    - description：任务1的 rollup Markdown 表（Experiments 页一眼看到聚合）
    API 参考：client.datasets.create_dataset / client.experiments.run_experiment
    """
    from phoenix.client import Client

    client = Client(base_url=config.phoenix_url)
    run_meta = config.run_meta

    # 预先构建 verdict lookup by (case_id, arm, run_idx)
    verdict_lookup: dict[tuple, dict] = {}
    for r in all_records:
        key = (r["case_id"], r["arm"], r["run_idx"])
        verdict_lookup[key] = r

    # ── 构建 inputs / outputs / metadata 列表（不用 DataFrame，避免 pandas 依赖） ──
    inputs   = []
    outputs  = []
    metadata = []
    for r in all_records:
        outcome = classify_record(r)
        inputs.append({
            "case_id": r["case_id"],
            "arm":     r["arm"],
            "run_idx": r["run_idx"],
            "probe":   _CASE_LABEL.get(r["case_id"], r["case_id"]),
        })
        outputs.append({
            "verdict":       r.get("verdict", ""),
            "write_pass":    str(r.get("write_pass", "")),
            "write_evidence": (r.get("write_evidence") or "")[:120],
            "expected_verdict": outcome["expected_verdict"] or "",
            "expectation_met": outcome["expectation_met"],
            "expectation_score": outcome["expectation_score"],
            "outcome_class": outcome["outcome_class"],
        })
        metadata.append({
            "run_id":      config.batch,
            "k":           run_meta.get("k", ""),
            "cases":       ",".join(run_meta.get("cases", [])) if isinstance(run_meta.get("cases"), list) else "",
            "harness_commit": run_meta.get("harness_commit", ""),
            "track":       (
                "track1_inc"  if r["case_id"] in TRACK1_INC_IDS  else
                "track1_prec" if r["case_id"] in TRACK1_PREC_IDS else
                "track2"
            ),
            "grader_kind": (
                "LLM"  if r["case_id"] in {"H_proj", "H_fb2", "H_usr"} else
                "CODE"
            ),
            "sample_status": r.get("sample_status", ""),
            "expected_verdict": outcome["expected_verdict"] or "",
            "expectation_met": str(outcome["expectation_met"]),
            "outcome_class": outcome["outcome_class"],
        })

    # Dataset 名含 batch 保证唯一
    dataset_name = f"memory-eval-{config.batch}"
    print(f"[phoenix] creating dataset '{dataset_name}' ({len(inputs)} examples)…")

    try:
        k_text = run_meta.get("k", "?")
        cases = run_meta.get("cases", [])
        cases_text = ", ".join(cases) if isinstance(cases, list) else "unknown"
        dataset = client.datasets.create_dataset(
            name=dataset_name,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            dataset_description=(
                f"Memory eval · batch {config.batch} · k={k_text} · "
                f"cases={cases_text} · runs={len(inputs)}"
            ),
        )
        dataset_id = dataset.id   # Dataset is an object with .id attribute
        print(f"[phoenix] dataset created: id={dataset_id}")
    except Exception as e:
        print(f"[phoenix][error] create_dataset failed: {e}")
        return f"{config.phoenix_url}/projects (dataset creation failed)"

    # ── run_experiment — task 回放已记录的 verdict（不调 LLM） ─────────────────
    def replay_task(example: dict) -> dict:
        """回放已记录的 verdict；example 是 DatasetExample（dict-like）。

        Phoenix DatasetExample dict 键：input/expected_output/metadata/id/updated_at
        """
        inp = example.get("input", {}) or {}
        key = (
            inp.get("case_id", ""),
            inp.get("arm", ""),
            int(inp.get("run_idx", 0)),
        )
        rec = verdict_lookup.get(key, {})
        outcome = classify_record(rec) if rec else {
            "expected_verdict": None,
            "expectation_met": None,
            "expectation_score": None,
            "outcome_class": "missing",
        }
        return {
            "verdict":           rec.get("verdict", "UNKNOWN"),
            "reason":            (rec.get("reason") or "")[:200],
            "write_pass":        str(rec.get("write_pass", "")),
            "expected_verdict":  outcome["expected_verdict"] or "",
            "expectation_met":   outcome["expectation_met"],
            "expectation_score": outcome["expectation_score"],
            "outcome_class":     outcome["outcome_class"],
        }

    def eval_expectation_met(output: dict, expected: dict) -> float:
        """1.0 when the verdict matches the experimental expectation."""
        if not isinstance(output, dict):
            return 0.0
        score = output.get("expectation_score")
        return float(score) if score is not None else 0.0

    def eval_write_success(output: dict, expected: dict) -> float:
        """1.0 if write_pass=='True' else 0.0."""
        wp = output.get("write_pass", "") if isinstance(output, dict) else ""
        return 1.0 if wp == "True" else 0.0

    exp_desc = (
        f"Memory eval · batch {config.batch}\n\n"
        "按 case_id 列排序可看每用例通过情况；下方嵌有 rollup 聚合表。\n\n"
        + rollup_md
    )

    print(f"[phoenix] running experiment '{config.experiment_name}'…")
    try:
        ran = client.experiments.run_experiment(
            dataset=dataset,
            task=replay_task,
            evaluators=[eval_expectation_met, eval_write_success],
            experiment_name=config.experiment_name,
            experiment_description=exp_desc,
        )
        exp_id = experiment_id_from_result(ran)
        exp_url = client.experiments.get_experiment_url(
            dataset_id=dataset_id,
            experiment_id=exp_id,
        )
        print(f"[phoenix] experiment done: id={exp_id}")
        return exp_url
    except Exception as e:
        print(f"[phoenix][error] run_experiment failed: {e}")
        # fallback: create experiment skeleton only
        try:
            exp = client.experiments.create(
                dataset_id=dataset_id,
                experiment_name=config.experiment_name,
                experiment_description=exp_desc,
            )
            exp_id = exp.get("id", "") if isinstance(exp, dict) else str(exp)
            exp_url = client.experiments.get_experiment_url(
                dataset_id=dataset_id, experiment_id=exp_id)
            print(f"[phoenix][fallback] experiment created (no runs): {exp_url}")
            return exp_url
        except Exception as e2:
            print(f"[phoenix][fallback] create failed: {e2}")
            try:
                ds_url = client.experiments.get_dataset_experiments_url(dataset_id=dataset_id)
            except Exception:
                ds_url = f"{config.phoenix_url}/datasets"
            return f"{ds_url}  (experiment create failed; see dataset)"


def experiment_id_from_result(result: Any) -> str:
    """Extract Phoenix experiment id across client return shapes."""
    if isinstance(result, dict):
        for key in ("id", "experiment_id"):
            value = result.get(key)
            if value:
                return str(value)

    for key in ("id", "experiment_id"):
        value = getattr(result, key, None)
        if value:
            return str(value)

    return str(result)


# ── Main ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push memory-eval JSONL rollup into Phoenix Experiment."
    )
    parser.add_argument(
        "--jsonl",
        default=str(DEFAULT_JSONL_PATH.relative_to(REPO)),
        help="评估结果 JSONL 路径；默认保留旧 2026-06-30 定版批次。",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Phoenix Experiment 名称；默认从 run_meta 的 k/cases/run_id 派生。",
    )
    parser.add_argument(
        "--phoenix-url",
        default=DEFAULT_PHOENIX,
        help="Phoenix base URL。",
    )
    parser.add_argument(
        "--reference-check",
        action="store_true",
        help="显式对照 02-results 的旧 k=5 数字；默认关闭，避免新批次错报。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = resolve_config(
        jsonl_path=args.jsonl,
        experiment_name=args.experiment_name,
        phoenix_url=args.phoenix_url,
        reference_check=args.reference_check,
    )

    print(f"[aggregate] loading {config.jsonl_path}…")
    print(f"[aggregate] batch={config.batch}  experiment={config.experiment_name}")
    all_records, inc_records, prec_records, t2_records = load_and_split(config.jsonl_path)
    print(f"  total={len(all_records)}  inc={len(inc_records)}  prec={len(prec_records)}  t2={len(t2_records)}")

    # 计算指标（run.py 口径）
    inc_metrics  = _metrics_track1_incremental(inc_records)
    prec_metrics = _metrics_h_prec(prec_records)
    t2_metrics   = _metrics_track2(t2_records)

    # 渲染 rollup 表
    rollup_md = render_rollup(inc_metrics, prec_metrics, t2_metrics, config.batch)

    print("\n" + "=" * 70)
    print(rollup_md)
    print("=" * 70)

    if config.reference_check:
        # 与 02-results §7 核对（从报告数字硬编码期望值，人工对比）
        print("\n── 与 02-results §7 核对 ────────────────────────────────────────────")
        ref = {
            "H_proj": {"p_write": 1.0, "p_use_given_write": 1.0, "p_a_e2e": 1.0, "p_b": 0.0, "delta": +1.00},
            "H_fb2":  {"p_write": 1.0, "p_use_given_write": 0.8, "p_a_e2e": 0.8, "p_b": 0.0, "delta": +0.80},
            "H_usr":  {"p_write": 0.6, "p_use_given_write": 0.667, "p_a_e2e": 0.4, "p_b": 0.0, "delta": +0.667},
            "H_ref":  {"p_write": 0.8, "p_use_given_write": 1.0, "p_a_e2e": 0.8, "p_b": 0.4, "delta": +0.60},
        }
        ref_t2 = {
            "H_drift":  {"pass_rate": 1.0},
            "H_ignore": {"pass_rate": 0.8},
            "H_neg":    {"pass_rate": 0.4},
        }

        match_all = True
        for cid, r in ref.items():
            m = inc_metrics["per_case"].get(cid, {})
            if not m:
                print(f"  [{cid}] MISSING from results")
                match_all = False
                continue
            diffs = []
            for key, exp_val in r.items():
                got = m.get(key)
                if got is None:
                    diffs.append(f"{key}=MISSING")
                elif abs(float(got) - float(exp_val)) > 0.02:
                    diffs.append(f"{key}: expected {exp_val:.3f}, got {got:.3f}")
            if diffs:
                print(f"  [{cid}] DIFF: {'; '.join(diffs)}")
                match_all = False
            else:
                print(f"  [{cid}] 一致")
        for cid, r in ref_t2.items():
            m = t2_metrics.get(cid, {})
            if not m:
                print(f"  [{cid}] MISSING from t2 results")
                match_all = False
                continue
            got = m["pass_rate"]
            exp_val = r["pass_rate"]
            if abs(got - exp_val) > 0.02:
                print(f"  [{cid}] DIFF: pass_rate expected {exp_val:.2f}, got {got:.2f}")
                match_all = False
            else:
                print(f"  [{cid}] 一致")
        # H_prec 核对
        pd_ = prec_metrics["delta"]
        pi  = prec_metrics["interpretation"]
        if abs(pd_) < 0.21 and pi.startswith("OK"):
            print(f"  [H_prec] 一致 (Δ={pd_:+.2f}, {pi})")
        else:
            print(f"  [H_prec] DIFF: expected Δ≈0/OK, got Δ={pd_:+.2f} / {pi}")
            match_all = False

        print(f"\n核对结论: {'全部一致' if match_all else '存在差异（见上方 DIFF 行）'}")
        print("  注：这是显式 --reference-check，用来人工对照旧 02-results，不作为任意新批次默认门禁。")

    # 推 Phoenix Experiment
    print("\n── 推 Phoenix Experiment ────────────────────────────────────────────")
    exp_url = push_phoenix_experiment(all_records, rollup_md, config)

    print(f"\n[DONE] Phoenix Experiment URL: {exp_url}")
    print(f"[DONE] 在 Experiments 页面按 case_id 列排序/筛选可看 per-example 下钻。")
    print(f"[DONE] Experiment description 里嵌有 rollup 聚合表。")


if __name__ == "__main__":
    main()
