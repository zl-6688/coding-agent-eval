"""拉取 SWE-bench Lite 实例（用 HF datasets-server 的 JSON API，绕开 pyarrow/Docker）。

用法: python eval/swebench/fetch.py            # 拉 100 个，存 instances.json，打印 repo 分布
"""

import json
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATASET = "princeton-nlp/SWE-bench_Lite"
OUT = Path(__file__).resolve().parent / "instances.json"

# 各 repo 的大致克隆体量（粗分，用于挑小的）
_SMALL = {"psf/requests", "pallets/flask", "marshmallow-code/marshmallow",
          "pallets/jinja", "pallets/click", "psf/requests-html"}
_MEDIUM = {"pylint-dev/pylint", "pytest-dev/pytest", "sphinx-doc/sphinx",
           "pydata/xarray", "mwaskom/seaborn", "pyvista/pyvista"}


def fetch(offset: int, length: int) -> list:
    url = ("https://datasets-server.huggingface.co/rows?"
           f"dataset={urllib.parse.quote(DATASET)}&config=default&split=test"
           f"&offset={offset}&length={length}")
    with urllib.request.urlopen(url, timeout=40) as r:
        return json.loads(r.read().decode("utf-8")).get("rows", [])


def main():
    print(f"拉取 {DATASET} ...")
    rows = []
    try:
        for off in (0, 100, 200):
            rows += fetch(off, 100)
    except Exception as e:
        print(f"datasets-server 失败：{e}\n可改用 hf-mirror 或 git clone 数据集。")
        return

    insts = [r["row"] for r in rows]
    OUT.write_text(json.dumps(insts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已存 {len(insts)} 个实例 -> {OUT}\n")

    by_repo = Counter(i["repo"] for i in insts)
    print("repo 分布（按数量）：")
    for repo, n in by_repo.most_common():
        tier = "小" if repo in _SMALL else ("中" if repo in _MEDIUM else "大")
        print(f"  [{tier}] {repo:35} {n}")

    print("\n小/中 repo 的候选实例（挑这些来跑，克隆和上下文都可控）：")
    cand = [i for i in insts if i["repo"] in (_SMALL | _MEDIUM)]
    for i in cand[:12]:
        ps = i["problem_statement"].replace("\n", " ")[:70]
        print(f"  {i['instance_id']:40} {ps}")
    print(f"\n候选 {len(cand)} 个（小/中 repo）。下一步挑 1 个 clone + 跑 agent 看上下文压力。")


if __name__ == "__main__":
    main()
