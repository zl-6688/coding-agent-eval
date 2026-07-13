"""eval/judge.py — LLM-as-judge：对 agent 产出的代码做质量打分。

关键设计：judge 模型必须 ≠ 被测模型，否则就是"自己给自己打分"（self-grading bias）——
这是评估系统最常见的陷阱，judge_available() 据此守门。
单次调用 temperature=0 降方差；多次采样取一致可进一步降方差（留作 v2）。
"""

import json
import re

from anthropic import Anthropic

from agent import config

_DIMS = ["correctness", "readability", "robustness"]
_SYSTEM = ("You are a strict senior code reviewer. "
           "Respond with ONLY a single JSON object, no other text, no explanation.")

_client = None


def _judge_client():
    global _client
    if _client is None:
        _client = Anthropic(api_key=config.API_KEY, base_url=config.BASE_URL)
    return _client


def judge_available() -> bool:
    """judge 是否可用：配了 JUDGE_MODEL_ID 且与被测模型不同。"""
    jm = config.JUDGE_MODEL_ID
    return bool(jm) and jm != config.MODEL_ID


def judge_code(task_prompt: str, code: str) -> dict:
    if not judge_available():
        return {"skipped": "JUDGE_MODEL_ID 未配置或与被测模型相同（避免自评偏差）"}
    if not code:
        return {"error": "no_code"}

    prompt = (
        f"## 编程任务\n{task_prompt}\n\n"
        f"## 候选人提交的代码\n```python\n{code[:4000]}\n```\n\n"
        f"请在 correctness（逻辑是否正确）、readability（可读性）、robustness（边界处理）"
        f"三个维度各打 1-5 分（整数），严格打分。\n"
        f'只返回一个 JSON，例如：{{"correctness": 4, "readability": 3, "robustness": 4}}'
    )
    try:
        resp = _judge_client().messages.create(
            model=config.JUDGE_MODEL_ID, system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048, temperature=0,   # 留足空间：推理模型(如 deepseek-v4-pro)的 thinking 会吃掉 token
        )
        # 聚合 text + thinking：推理模型可能把内容放 thinking 块、text 为空
        text_raw = "".join(getattr(b, "text", "") for b in resp.content
                           if getattr(b, "type", None) == "text").strip()
        think_raw = "".join(getattr(b, "thinking", "") for b in resp.content
                            if getattr(b, "type", None) == "thinking").strip()
    except Exception as e:
        return {"error": f"judge_call: {type(e).__name__}: {e}"[:120]}

    scores = _extract_json(text_raw) or _extract_json(think_raw)
    if not scores:
        return {"error": f"parse_failed: text={text_raw[:60]!r} think={think_raw[:60]!r}"}

    out = {"judge_model": config.JUDGE_MODEL_ID}
    vals = []
    for d in _DIMS:
        v = scores.get(d)
        if isinstance(v, (int, float)):
            out[d] = max(1, min(5, int(v)))
            vals.append(out[d])
        else:
            out[d] = None
    out["avg"] = round(sum(vals) / len(vals), 2) if vals else None
    return out


def judge_notes_accuracy(notes: str, reference: str) -> dict:
    """给一份"对源码的说明"按**事实准确性**打分（对照真实源码 reference）。

    用于压缩评测的**产出正确性轴**：压缩后 agent 写的说明，有多少是真的、有多少是**编造**的。
    这正是抓"硬截断丢了内容却自信胡说"的指标——`fabrications`>0 = 它在幻觉。
    judge≠被测模型（同 judge_available 守门）。返回 {accuracy:1-5, fabrications:int, judge_model}。"""
    if not judge_available():
        return {"skipped": "JUDGE_MODEL_ID 未配置或与被测模型相同（避免自评偏差）"}
    if not notes:
        return {"error": "no_notes", "accuracy": None, "fabrications": None}

    prompt = (
        "下面给你【真实源码】和一份【待评说明】（说明是别人读了源码后写的函数职责描述）。\n"
        "请**严格对照真实源码**评判这份说明：\n"
        "  - accuracy：说明整体的事实准确度，1-5 整数（5=完全符合源码，1=大量与源码不符）。\n"
        "  - fabrications：说明里**与源码矛盾或源码中根本不存在**的具体说法**条数**（编造/张冠李戴算数；"
        "措辞不同但意思对不算）。\n\n"
        f"## 真实源码\n```python\n{reference[:50000]}\n```\n\n"
        f"## 待评说明\n{notes[:6000]}\n\n"
        '只返回一个 JSON，例如：{"accuracy": 4, "fabrications": 1}'
    )
    # 重试：judge 走同一条不稳代理 + 推理模型偶发空响应/解析失败 → 瞬时失败重试 3 次，
    #   否则 None 会污染 repeat-N 的均值（实测单 None 能让结论翻盘）。
    scores = None
    last_err = ""
    for _ in range(3):
        try:
            resp = _judge_client().messages.create(
                model=config.JUDGE_MODEL_ID, system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048, temperature=0,
            )
            text_raw = "".join(getattr(b, "text", "") for b in resp.content
                               if getattr(b, "type", None) == "text").strip()
            think_raw = "".join(getattr(b, "thinking", "") for b in resp.content
                                if getattr(b, "type", None) == "thinking").strip()
        except Exception as e:
            last_err = f"judge_call: {type(e).__name__}: {e}"[:120]
            continue
        scores = _extract_json(text_raw) or _extract_json(think_raw)
        if scores:
            break
        last_err = f"parse_failed: text={text_raw[:60]!r}"
    if not scores:
        return {"error": last_err, "accuracy": None, "fabrications": None}
    acc = scores.get("accuracy")
    fab = scores.get("fabrications")
    return {"judge_model": config.JUDGE_MODEL_ID,
            "accuracy": max(1, min(5, int(acc))) if isinstance(acc, (int, float)) else None,
            "fabrications": int(fab) if isinstance(fab, (int, float)) else None}


def _extract_json(raw: str):
    """多策略 JSON 抽取：代码块 -> 结尾的花括号（推理模型常把答案放最后）-> 整体。"""
    if not raw:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        obj = _try_json(m.group(1))
        if obj is not None:
            return obj
    start, end = raw.rfind("{"), raw.rfind("}")
    if 0 <= start < end:
        obj = _try_json(raw[start:end + 1])
        if obj is not None:
            return obj
    return _try_json(raw)


def _try_json(s):
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        return None
