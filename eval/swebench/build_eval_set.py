"""生成可复用评估集清单 eval_set.json（本地已有 + 计划下载的难实例）。

设计目标：多样性 × 难度 × 同 repo 可串。派生元数据只从调用方显式提供的完整
SWE-bench dataset JSON 读取；benchmark rows 不随仓库分发。

用法: python -m eval.swebench.build_eval_set --instances <dataset.json> --out eval_set.json
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from eval.swebench.run_swe import instance_image  # noqa: E402

# 用户拍板的下载清单（难度优先,排除本地已有）。键=repo 简称,值=完整 instance_id 列表。
PLAN = {
    "pylint-dev/pylint": ["pylint-dev__pylint-4551", "pylint-dev__pylint-6386",
                          "pylint-dev__pylint-8898", "pylint-dev__pylint-6528"],
    "sphinx-doc/sphinx": ["sphinx-doc__sphinx-9461", "sphinx-doc__sphinx-10673",
                          "sphinx-doc__sphinx-7590", "sphinx-doc__sphinx-8551"],
    "sympy/sympy": ["sympy__sympy-13091", "sympy__sympy-16597",
                    "sympy__sympy-20438", "sympy__sympy-14248"],
    "django/django": ["django__django-11532", "django__django-11138",
                      "django__django-15629", "django__django-13121"],
    "astropy/astropy": ["astropy__astropy-13398", "astropy__astropy-14369",
                        "astropy__astropy-8707"],
    "pydata/xarray": ["pydata__xarray-6938", "pydata__xarray-3095"],
    "scikit-learn/scikit-learn": ["scikit-learn__scikit-learn-25102",
                                  "scikit-learn__scikit-learn-12682"],
    "matplotlib/matplotlib": ["matplotlib__matplotlib-25775"],
}
PLANNED_IDS = {iid for ids in PLAN.values() for iid in ids}


def _gold_files(patch: str) -> list:
    return sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch, re.M)))


def _local_images() -> dict:
    """本地 swebench 镜像 → {instance_id: size_str}。"""
    out = subprocess.run(["docker", "images", "--format", "{{.Repository}} {{.Size}}"],
                         capture_output=True, text=True).stdout
    local = {}
    for ln in out.splitlines():
        m = re.search(r"sweb\.eval\.x86_64\.(\S+?)\s+(\S+)", ln)
        if m:
            local[m.group(1).replace("_1776_", "__")] = m.group(2)
    return local


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        required=True,
        type=Path,
        help="Path to a complete external SWE-bench dataset JSON list.",
    )
    parser.add_argument("--out", type=Path, default=Path("eval_set.json"))
    args = parser.parse_args(argv)

    from eval.swebench.data import DatasetError, load_instance_dataset

    try:
        full = {row["instance_id"]: row for row in load_instance_dataset(args.instances)}
    except DatasetError as exc:
        parser.error(str(exc))
    local = _local_images()
    ids = sorted(set(local) | PLANNED_IDS)

    entries = []
    for iid in ids:
        meta = full.get(iid)
        gf = _gold_files(meta["patch"]) if meta else []
        entries.append({
            "instance_id": iid,
            "repo": iid.split("__")[0],                 # 簇键（同 repo 可串）
            "image": instance_image(iid),
            "local": iid in local,
            "in_plan": iid in PLANNED_IDS,
            "gold_files": gf,
            "gold_count": len(gf) if meta else None,
            "ps_chars": len(meta["problem_statement"]) if meta else None,
            "in_verified": meta is not None,            # 本地少数可能来自 Lite,非 Verified
        })
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总
    by_repo = defaultdict(lambda: {"local": 0, "topull": 0})
    for e in entries:
        by_repo[e["repo"]]["local" if e["local"] else "topull"] += 1
    topull = [e for e in entries if not e["local"]]
    print(f"评估集 = {len(entries)} 实例 / {len(by_repo)} 簇(repo);本地 {sum(1 for e in entries if e['local'])} "
          f"+ 待拉 {len(topull)}")
    print(f"\n{'簇(repo)':<22}{'本地':>5}{'待拉':>5}{'可串≥3?':>9}")
    for repo in sorted(by_repo, key=lambda r: -(by_repo[r]['local'] + by_repo[r]['topull'])):
        c = by_repo[repo]
        tot = c["local"] + c["topull"]
        print(f"{repo:<22}{c['local']:>5}{c['topull']:>5}{('✓' if tot >= 3 else '—'):>9}")
    print(f"\n待拉 {len(topull)} 个 → 跑 `python eval/swebench/pull_eval_set.py` (磁盘安全/可续)。")
    print(f"清单 → {out_path}")


if __name__ == "__main__":
    main()
