"""验证 OTel 桥接 —— 用 console exporter 打印 OTel span。

需先: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("usage: otel_check.py\n\nEmit synthetic OpenTelemetry spans to the console.")
    raise SystemExit(0)

from obs.otel import init_otel
from obs.trace import SpanKind, span

if not init_otel(console=True):
    print("opentelemetry 未安装：pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http")
    sys.exit(1)

with span("agent.run", SpanKind.AGENT, **{"task": "otel 验证"}):
    with span("llm.call", SpanKind.CLIENT, **{
        "gen_ai.request.model": "demo",
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 5,
    }):
        pass
    with span("tool.bash", SpanKind.TOOL, **{"tool.name": "bash"}):
        pass

print("\n[OK] 若上方打印了 OTel span（含 openinference.span.kind / gen_ai.*），桥接就通了。")
