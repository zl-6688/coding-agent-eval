"""LLM 客户端封装 —— LLM 接缝（seam）。

在 chat() 里开一个 OTel 风格的 CLIENT span，记录 model / token / 延迟，
而 loop.py 完全无需感知埋点的存在。
"""

import json
import time
from typing import Any

import anthropic
from anthropic import Anthropic

from obs.trace import SpanKind, annotate, span

from . import config
from .runtime.llm_runtime import model_for_purpose

_client: Anthropic | None = None
_client_key: tuple[Any, ...] | None = None

# ── Per-run token accumulator (Phase-2 cost capture) ─────────────────────────
# WHY global + reset: eval runs serially (one arm at a time); a global counter
# with an explicit reset before each arm is simpler than threading.local and
# avoids the span roll-up complexity (reading attrs back out of closed spans).
# reset_token_counter() is called by run.py before each arm run; get_token_usage()
# is called after to populate ArmResult.token_usage.
_tok: dict = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}


def reset_token_counter() -> None:
    """Reset per-run accumulator. Call before each arm run."""
    _tok["input"] = _tok["output"] = _tok["cache_read"] = _tok["cache_creation"] = 0


def get_token_usage(model_id: str | None = None) -> dict:
    """Return accumulated token counts + rough cost estimate.

    Deepseek v4 flash pricing (2026-06) — rough estimate only, not for billing:
      input/cache_creation: ~$0.27/1M tokens
      output:               ~$1.10/1M tokens
      cache_read:           ~$0.07/1M tokens
    ⚠ Deviation: actual pricing may differ; use for order-of-magnitude only.
    """
    inp  = _tok["input"]
    out  = _tok["output"]
    cr   = _tok["cache_read"]
    cc   = _tok["cache_creation"]
    cost = (inp + cc) * 0.27e-6 + out * 1.10e-6 + cr * 0.07e-6
    return {
        "input":           inp,
        "output":          out,
        "cache_read":      cr,
        "cache_creation":  cc,
        "cost_est_usd":    round(cost, 6),
        "model_id":        model_id or "",
    }

# ── 错误恢复 L1：瞬时错误指数退避重试 ──
# 网络抖动/限流/服务过载是真实风险（实测代理不稳，连测 3 个实例都因瞬时错全废）。
# 一次瞬时失败不该报废整个 run；退避重试把"抖动"吸收掉。
_RETRYABLE = (
    anthropic.APIConnectionError,   # 连接错（网络/代理抖动）
    anthropic.APITimeoutError,      # 请求超时
    anthropic.RateLimitError,       # 429 限流
    anthropic.InternalServerError,  # 5xx（含 OverloadedError 529）
)
_MAX_RETRIES = 5
_BACKOFF_BASE = 2.0   # 退避：2,4,8,16,32s（上限 32）


def _estimate_context_tokens(messages: list, system: str) -> int:
    """粗估我们实际发出去的上下文大小（~4 字符/token）。

    不依赖 provider 的计费字段，能稳定反映"上下文随轮次增长"，
    不受缓存 / proxy 差异影响。
    """
    total = len(system or "")
    for m in messages:
        total += len(str(m.get("content", "")))
    return total // 4


def reset_client() -> None:
    """Drop cached Anthropic client (REPL settings/env switch)."""
    global _client, _client_key
    _client = None
    _client_key = None


def client() -> Anthropic:
    global _client, _client_key
    from .runtime.llm_runtime import effective_api_key, effective_base_url

    api_key = effective_api_key()
    base_url = effective_base_url()
    cache_key = (api_key, base_url)
    if _client is None or _client_key != cache_key:
        # WHY timeout: 实测 deepseek 代理偶发连接挂死（连接建立、响应永不返回）。
        # 无 timeout → SDK 默认很长 → 整个 run 干等卡死（定版两次栽在 H_prec/H_ignore
        # 这类跑满 max_turns 的硬用例上）。设 300s 硬超时 → 挂死调用抛 APITimeoutError →
        # _create_with_retry 退避重试接住 → 瞬时挂死能自愈，持久故障也只让该 run 变 ERROR
        # 而非掀翻整场。300s 对 flash 的 4096-token 正常响应留足余量，不会误触发。
        _client = Anthropic(api_key=api_key, base_url=base_url, timeout=300.0)
        _client_key = cache_key
    return _client


def _create_with_retry(sp, **kwargs):
    """对瞬时错误（网络/超时/限流/过载）指数退避重试；其余错误立即抛。

    重试信息记到当前 llm.call span（n_retries / retried_on）→ recovery 行为可观测：
    Phoenix 里能看出"哪些调用靠重试才成功、网络多抖"。
    """
    last = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = client().messages.create(**kwargs)
            if attempt > 0:
                annotate(**{"llm.n_retries": attempt,
                            "llm.retried_on": type(last).__name__})
            return resp
        except _RETRYABLE as e:
            last = e
            if attempt >= _MAX_RETRIES:
                annotate(**{"llm.n_retries": attempt, "llm.retry_exhausted": True,
                            "llm.retried_on": type(e).__name__})
                raise
            delay = min(_BACKOFF_BASE * (2 ** attempt), 32.0)
            time.sleep(delay)


def chat(messages: list, system: str, tools: list | None = None,
         max_tokens: int = 4096, model: str | None = None, purpose: str = "agent",
         temperature: float | None = None):
    """一次 LLM 调用，返回原始 response。

    purpose: 这次调用的用途 → llm.purpose span 属性，让成本可**按用途归因**。
      'agent'（默认，主循环推理）/ 'compaction'（full_compact 的摘要调用 = 压缩成本）/
      'judge' 等。维度4：能从 trace 抽出"压缩本身花了多少 token/$"，与主循环成本分开。
    """
    mdl = model_for_purpose(purpose, model)
    # 最后一条 user 消息作为本次调用的"输入"摘要（Phoenix input 列），截断防爆
    last_user = next((str(m.get("content", "")) for m in reversed(messages)
                      if m.get("role") == "user"), "")
    with span("llm.call", SpanKind.CLIENT, **{
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": mdl,
        "context.tokens_sent": _estimate_context_tokens(messages, system),
        "llm.input": last_user[:4000],
        "llm.n_messages": len(messages),
        "llm.purpose": purpose,
    }) as sp:
        kwargs = dict(model=mdl, system=system, messages=messages,
                      tools=tools or [], max_tokens=max_tokens)
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = _create_with_retry(sp, **kwargs)
        # 回复文本（text 块）作为"输出"（Phoenix output 列）
        out_text = "".join(getattr(b, "text", "") for b in resp.content
                           if getattr(b, "type", None) == "text")
        tool_calls = [getattr(b, "name", "") for b in resp.content
                      if getattr(b, "type", None) == "tool_use"]
        block_types = [
            str(getattr(b, "type", "unknown") or "unknown")
            for b in resp.content
        ]
        if out_text:
            rendered_output = out_text
        elif tool_calls:
            rendered_output = "[tool_use] " + ", ".join(tool_calls)
        else:
            rendered_output = "[non_text] " + ", ".join(block_types)
        sp.set(**{
            "llm.output": rendered_output[:4000],
            "llm.output_block_types": ",".join(block_types),
        })
        usage = getattr(resp, "usage", None)
        if usage is not None:
            try:
                u = usage.model_dump()           # pydantic：拿全部字段（含 proxy 自定义的）
            except Exception:
                u = {k: getattr(usage, k, 0) for k in
                     ("input_tokens", "output_tokens",
                      "cache_read_input_tokens", "cache_creation_input_tokens")
                     if hasattr(usage, k)}
            _ti  = u.get("input_tokens", 0) or 0
            _to  = u.get("output_tokens", 0) or 0
            _tcr = u.get("cache_read_input_tokens", 0) or 0
            _tcc = u.get("cache_creation_input_tokens", 0) or 0
            sp.set(**{
                "gen_ai.usage.input_tokens": _ti,
                "gen_ai.usage.output_tokens": _to,
                # 缓存字段：上下文真正的大小藏在 cache_read 里（input_tokens 只是未命中的新增量）
                "gen_ai.usage.cache_read_input_tokens": _tcr,
                "gen_ai.usage.cache_creation_input_tokens": _tcc,
                # 原始 usage 留一份，方便看 proxy 到底回传了哪些字段
                "gen_ai.usage.raw": json.dumps(u, ensure_ascii=False, default=str),
            })
            # Phase-2: accumulate into per-run counter so harness can read token_usage
            _tok["input"]         += _ti
            _tok["output"]        += _to
            _tok["cache_read"]    += _tcr
            _tok["cache_creation"] += _tcc
        sp.set(**{"gen_ai.response.stop_reason": getattr(resp, "stop_reason", "")})
        return resp
