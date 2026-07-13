"""obs/otel.py — 可选的 OpenTelemetry 桥接层。

我们自己的 span() 在每个接缝产出一个轻量 Span（写 JSONL + 本地 HTML）。
启用本桥接后，同一个 span() 还会产出一个**标准 OTel span**，经 OTLP 导出到
Arize Phoenix（或任何 OTLP 后端）。

设计要点：
- opentelemetry 惰性导入：不装也能跑（只是没有 OTel 导出），项目不会因观测依赖挂掉。
- OTel 自身用 contextvars 管父子，与我们的 contextvars 并行；
  capture_context() 会一并捕获两者，所以后台线程 / subagent 的 OTel span 也能正确嵌套。
"""

from __future__ import annotations

import contextlib
import json as _json

_enabled = False
_tracer = None

# 把我们的 SpanKind 映射到 OpenInference 的 span kind，让 Phoenix UI 正确分类
_OPENINFERENCE_KIND = {
    "AGENT": "AGENT",
    "CLIENT": "LLM",
    "TOOL": "TOOL",
    "INTERNAL": "CHAIN",
}


def init_otel(project_name: str = "coding-agent-eval",
              endpoint: str | None = None, console: bool = False) -> bool:
    """初始化 OTel 导出。console=True 时打印到控制台（用于验证）。

    返回是否成功启用（未安装 opentelemetry 时返回 False，优雅降级）。
    """
    global _enabled, _tracer
    if _enabled:
        return True   # 幂等：每个子进程首调初始化，重复调用直接复用（多进程批跑安全）
    try:
        from opentelemetry import trace as ot_trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter, SimpleSpanProcessor,
        )
    except ImportError:
        print("[otel] 未安装 opentelemetry，跳过（本地 JSONL + HTML 仍可用）")
        return False

    # openinference.project.name：Phoenix 17.x 按此 resource 属性归项目（否则全进 default；
    # 旧的 x-phoenix-project-name header 在 OTLP/HTTP 路径上不被 17.9 认）。
    provider = TracerProvider(resource=Resource.create({
        "service.name": project_name,
        "openinference.project.name": project_name,
    }))
    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        target = "console"
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        target = endpoint or "http://localhost:6006/v1/traces"
        # 开发期用 Simple（即时导出，避免脚本退出前 batch 没 flush）；
        # 生产环境改用 BatchSpanProcessor。
        # x-phoenix-project-name：让 span 落在 Phoenix 的命名项目里（否则进 "default"）。
        # 对非 Phoenix 的 OTLP 后端，这个 header 会被忽略，无害。
        provider.add_span_processor(SimpleSpanProcessor(
            OTLPSpanExporter(endpoint=target, headers={"x-phoenix-project-name": project_name})))

    ot_trace.set_tracer_provider(provider)
    _tracer = ot_trace.get_tracer("coding-agent-eval")
    _enabled = True
    print(f"[otel] 已启用 → {target}")
    return True


@contextlib.contextmanager
def otel_span(name: str, kind: str):
    """产出一个 OTel span（若已启用）；否则 yield None（零开销）。"""
    if not _enabled or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as otsp:
        otsp.set_attribute("openinference.span.kind", _OPENINFERENCE_KIND.get(kind, "CHAIN"))
        yield otsp


# 内部属性键 → OpenInference 语义键（让 Phoenix 的 input/output/token/model 列认得）。
# 我们的 gen_ai.*（OTel GenAI 语义）原样保留——batch 分析依赖它；这里只是**额外补**一份镜像。
_OI_MIRROR = {
    "llm.input": "input.value",
    "llm.output": "output.value",
    "gen_ai.request.model": "llm.model_name",
    "gen_ai.usage.input_tokens": "llm.token_count.prompt",
    "gen_ai.usage.output_tokens": "llm.token_count.completion",
    "task": "input.value",                       # agent.run 的 input 列显示任务
    "output_value": "output.value",              # agent.run 的 output 列显示最终总结
}

_INPUT_PREVIEW_KEYS = (
    "tool.command.preview",
    "tool.input.preview",
    "mcp.input.preview",
    "skill.args.preview",
    "skill.body.preview",
    "background_task.command.preview",
    "subagent.prompt.preview",
    "subagent.description.preview",
)
_INPUT_SUMMARY_KEYS = (
    "tool.command_summary",
    "tool.input_summary",
    "mcp.input_summary",
    "skill.args_summary",
    "skill.body_summary",
    "background_task.command_summary",
    "subagent.prompt_summary",
    "subagent.description_summary",
)
_INPUT_LEGACY_KEYS = ("tool.arg",)
_OUTPUT_PREVIEW_KEYS = (
    "tool.output.preview",
    "mcp.output.preview",
    "background_task.notification.preview",
)
_OUTPUT_SUMMARY_KEYS = (
    "tool.output_summary",
    "mcp.output_summary",
    "background_task.notification_summary",
)
_OUTPUT_LEGACY_KEYS = ("tool.output_tail",)


def apply_attributes(otsp, attrs: dict) -> None:
    """把我们 Span 的属性复制到 OTel span（在 finally 里调用，此时 OTel span 仍打开）。

    额外补一份 OpenInference 镜像键，让 Phoenix UI 的 input/output/token/model 列填上。
    """
    if otsp is None:
        return
    for k, v in attrs.items():
        if v is None:
            continue
        # run_metadata（身份 dict）→ 只发 OpenInference 标准 metadata 键（Phoenix 自动展开成
        # metadata.version / metadata.instance_id … 可筛）；不发原 run_metadata.* 避免重复。
        if k == "run_metadata":
            otsp.set_attribute("metadata", _safe_json(v))
            continue
        # OTel 只接受标量/同型数组；dict/复杂值序列化成 JSON 字符串
        val = v if isinstance(v, (str, bool, int, float)) else _safe_json(v)
        try:
            otsp.set_attribute(k, val)
        except Exception:
            otsp.set_attribute(k, str(v))
    for oi, val in _openinference_attrs(attrs).items():
        try:
            otsp.set_attribute(oi, val)
        except Exception:
            otsp.set_attribute(oi, str(val))


def _openinference_attrs(attrs: dict) -> dict:
    mirrors = {}
    for k, oi in _OI_MIRROR.items():
        if oi in attrs or oi in mirrors or k not in attrs:
            continue
        mirrors[oi] = _otel_value(attrs[k])
    if "input.value" not in attrs:
        value = _first_attr(attrs, _INPUT_PREVIEW_KEYS)
        if value is None:
            value = _first_attr(attrs, _INPUT_SUMMARY_KEYS)
        if value is None:
            value = _first_attr(attrs, _INPUT_LEGACY_KEYS)
        if value is not None:
            mirrors["input.value"] = _otel_value(value)
    if "output.value" not in attrs:
        value = _first_attr(attrs, _OUTPUT_PREVIEW_KEYS)
        if value is None:
            value = _first_attr(attrs, _OUTPUT_SUMMARY_KEYS)
        if value is None:
            value = _first_attr(attrs, _OUTPUT_LEGACY_KEYS)
        if value is not None:
            mirrors["output.value"] = _otel_value(value)
    return mirrors


def _first_attr(attrs: dict, keys: tuple[str, ...]):
    for key in keys:
        value = attrs.get(key)
        if value not in (None, ""):
            return value
    return None


def _otel_value(v):
    return v if isinstance(v, (str, bool, int, float)) else _safe_json(v)


def _safe_json(v) -> str:
    try:
        return _json.dumps(v, ensure_ascii=False, default=str)
    except Exception:
        return str(v)


def mark_error(otsp, message: str) -> None:
    if otsp is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode
        otsp.set_status(Status(StatusCode.ERROR, message))
    except Exception:
        pass


def mark_ok(otsp) -> None:
    """显式标 OK（否则 OTel 默认 UNSET；让 Phoenix 状态列干净区分 OK/ERROR）。"""
    if otsp is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode
        otsp.set_status(Status(StatusCode.OK))
    except Exception:
        pass
