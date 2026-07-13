"""把评估结论写回 Phoenix span annotation —— 打通**观测↔评估**。

观测(trace)记录 agent 怎么跑的；评估结论(localization_hit / failure_reason / resolved)
原本只在 batch jsonl / harness 日志里，与 trace 断开。本脚本把结论作为 **annotation**
(Phoenix 的评估层，annotator_kind=CODE)写到对应 trace 的 agent.run span 上，于是在 Phoenix UI 里：
  - 可按 annotation 筛"所有 unresolved / no_edit 的 trace"，点进去看它为什么失败
  - trace 上直接显示 resolved=0 / failure_reason=...，无需回翻 jsonl
  - 平台上聚合成功率

口径(核对官方 API)：POST /v1/span_annotations，body=
  {data:[{name, annotator_kind:"CODE", span_id, result:{label, score, explanation}}]}
span_id = OTel hex；按 batch 行的 metadata.run_id 反查 Phoenix 的 agent.run span。

用法:
  python eval/swebench/annotate_phoenix.py --tag=cmp_indocker            # 用 batch 的 localization/failure
  python eval/swebench/annotate_phoenix.py --tag=cmp_indocker --resolved=<run_id>  # 再带上 harness 真 resolved
"""

import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PHOENIX = "http://localhost:6006"
PROJECT = "coding-agent-eval"
HERE = Path(__file__).resolve().parent


def _get(url):
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def _project_id():
    for p in _get(f"{PHOENIX}/v1/projects").get("data", []):
        if p.get("name") == PROJECT:
            return p["id"]
    raise RuntimeError(f"Phoenix 里没有项目 {PROJECT}")


def runid_to_spanid(project_id: str) -> dict:
    """拉所有 agent.run span，建 metadata.run_id → OTel span_id 映射（annotation 挂到根 span）。
    Phoenix spans 端点单页有上限，用 cursor 分页拉全。"""
    out = {}
    cursor = None
    for _ in range(200):   # 最多 200 页，防死循环
        url = f"{PHOENIX}/v1/projects/{project_id}/spans?limit=100"
        if cursor:
            url += f"&cursor={cursor}"
        resp = _get(url)
        for s in resp.get("data", []):
            if s.get("name") != "agent.run":
                continue
            rid = (s.get("attributes", {}) or {}).get("metadata.run_id")
            sid = s.get("context", {}).get("span_id")
            if rid and sid:
                out[rid] = sid
        cursor = resp.get("next_cursor")
        if not cursor:
            break
    return out


def post_annotations(items: list) -> int:
    """批量 POST span annotation。items=[{span_id,name,label,score,explanation}]。"""
    data = [{"name": it["name"], "annotator_kind": "CODE", "span_id": it["span_id"],
             "result": {k: it[k] for k in ("label", "score", "explanation") if it.get(k) is not None}}
            for it in items]
    req = urllib.request.Request(f"{PHOENIX}/v1/span_annotations?sync=true",
                                 data=json.dumps({"data": data}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status


def load_resolved(run_id: str) -> dict:
    """从 WSL harness 日志读官方 resolved（可选）。返回 {instance_id: bool}。
    日志路径在 WSL 内：~/swe-runs/logs/run_evaluation/<run_id>/<model>/<iid>/report.json。
    这里走 git-bash 可见的 \\wsl$ 挂载；读不到就跳过（只写 localization/failure）。"""
    import glob
    res = {}
    bases = glob.glob(fr"\\wsl$\Ubuntu\home\*\swe-runs\logs\run_evaluation\{run_id}\*")
    bases += glob.glob(fr"\\wsl.localhost\Ubuntu\home\*\swe-runs\logs\run_evaluation\{run_id}\*")
    for model_dir in bases:
        for rp in glob.glob(model_dir + r"\*\report.json"):
            try:
                d = json.loads(Path(rp).read_text(encoding="utf-8"))
                iid = list(d.keys())[0]
                res[iid] = bool(d[iid]["resolved"])
            except Exception:
                pass
    return res


def main():
    flags = {a.split("=", 1)[0]: (a.split("=", 1)[1] if "=" in a else "")
             for a in sys.argv[1:] if a.startswith("--")}
    tag = flags.get("--tag", "")
    if not tag:
        print("用法: --tag=<batch tag> [--resolved=<harness run_id>]")
        return
    batch = HERE / f"batch_results_{tag}.jsonl"
    if not batch.exists():
        print(f"找不到 {batch}")
        return
    rows = {}
    for line in batch.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["instance_id"]] = r   # 后写覆盖（重试取最新）

    resolved_map = load_resolved(flags["--resolved"]) if flags.get("--resolved") else {}

    pid = _project_id()
    r2s = runid_to_spanid(pid)
    print(f"Phoenix agent.run span: {len(r2s)} 个 | batch 行: {len(rows)} | resolved 映射: {len(resolved_map)}")

    items, miss = [], []
    for iid, r in rows.items():
        rid = (r.get("model_patch") is not None) and None  # placeholder
        # batch 行里没直接存 run_id；用 trace_path 文件名兜底匹配 run_id
        rid = None
        tp = r.get("trace_path", "")
        if tp:
            stem = Path(tp).stem  # run_<id>
            rid = stem[4:] if stem.startswith("run_") else stem
        sid = r2s.get(rid)
        if not sid:
            miss.append(iid)
            continue
        hit = r.get("localization_hit")
        rv = resolved_map.get(iid)   # None=没评分; True/False=官方真值
        # ① resolved（headline 真值，二元 score+label）
        if rv is not None:
            items.append({"span_id": sid, "name": "resolved",
                          "label": "resolved" if rv else "unresolved", "score": 1.0 if rv else 0.0,
                          "explanation": "官方 harness 跑测试通过" if rv else "补丁 apply 了但测试没过/空补丁"})
        # ② localization_hit（诊断，二元 score+label）
        items.append({"span_id": sid, "name": "localization_hit",
                      "label": "hit" if hit else "miss", "score": 1.0 if hit else 0.0,
                      "explanation": "改到了 gold 文件" if hit else "没改到 gold 文件"})
        # ③ failure_reason（分类 label，**仅当确有失败**才写，给有意义的值，不写 "ok"）
        fr = None
        if not hit:
            fr = r.get("failure_reason") or "miss"               # no_edit / wrong_file / ...
        elif rv is False:
            fr = "localized_but_unresolved"                       # 命中文件但测试没过 = close-but-wrong
        if fr:
            items.append({"span_id": sid, "name": "failure_reason", "label": fr, "score": None,
                          "explanation": "为什么没成功（localization 层 或 resolve 层）"})

    if not items:
        print("没有可写的 annotation（trace 可能没导出到 Phoenix，或 run_id 对不上）。")
        if miss:
            print("未匹配实例:", [m.split('__')[-1] for m in miss][:10])
        return
    status = post_annotations(items)
    print(f"✅ POST {len(items)} 条 annotation → HTTP {status}（{len(rows)-len(miss)} 个 trace 标注）")
    if miss:
        print(f"⚠️ {len(miss)} 个实例没匹配到 Phoenix trace（旧代码跑的没 run_id / 没开 OTEL_EXPORT）:",
              [m.split('__')[-1] for m in miss][:10])
    print(f"→ 现在去 Phoenix 项目 {PROJECT}，可按 annotation 'resolved'/'localization_hit'/'failure_reason' 筛选 trace。")


if __name__ == "__main__":
    main()
