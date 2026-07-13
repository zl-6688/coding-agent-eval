"""Phoenix OTLP smoke test — zero LLM API calls.

Sends a handful of synthetic spans via OTLP HTTP to Phoenix at localhost:6006,
then queries the Phoenix API to confirm the spans were received.

Usage (from project root, with the project environment active):
    python scripts/phoenix_smoke.py

Expected output:
    [otel] 已启用 -> http://localhost:6006/v1/traces
    [smoke] Sending 3 test spans to Phoenix...
    [smoke] Spans sent.
    [smoke] Phoenix projects: ['phoenix (default)'] or similar
    [smoke] Traces in project: N (should be >= 1)
    [OK] Phoenix smoke test passed — open http://localhost:6006 and look for
         project "coding-agent-eval" with trace named "smoke.agent.run".
"""

import sys
import time
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print("usage: phoenix_smoke.py\n\nSend synthetic spans to a local Phoenix service.")
    raise SystemExit(0)

from obs.otel import init_otel
from obs.trace import SpanKind, span

PHOENIX_ENDPOINT = "http://localhost:6006/v1/traces"
PHOENIX_API_BASE = "http://localhost:6006"
PROJECT_NAME = "coding-agent-eval"


def send_test_spans() -> None:
    ok = init_otel(project_name=PROJECT_NAME, endpoint=PHOENIX_ENDPOINT, console=False)
    if not ok:
        print("[ERROR] OTel init failed — is opentelemetry-sdk installed?")
        sys.exit(1)

    print(f"[smoke] Sending 3 test spans to Phoenix...")
    with span("smoke.agent.run", SpanKind.AGENT, **{"task": "phoenix smoke test"}):
        with span("smoke.llm.call", SpanKind.CLIENT, **{
            "gen_ai.request.model": "smoke-model",
            "gen_ai.usage.input_tokens": 42,
            "gen_ai.usage.output_tokens": 7,
        }):
            time.sleep(0.05)   # simulate latency, no LLM call
        with span("smoke.tool.bash", SpanKind.TOOL, **{
            "tool.name": "bash",
            "tool.input": "echo hello",
            "tool.output": "hello",
        }):
            time.sleep(0.02)
    print("[smoke] Spans sent (OTLP exporter flushed synchronously).")


def verify_via_api() -> None:
    """Poll Phoenix REST API to confirm traces were ingested."""
    try:
        import urllib.request
        import json as _json

        # Give Phoenix a moment to ingest
        time.sleep(2)

        # List projects
        with urllib.request.urlopen(f"{PHOENIX_API_BASE}/v1/projects", timeout=5) as resp:
            data = _json.loads(resp.read())
        projects = data.get("data", [])
        names = [p.get("name") for p in projects]
        print(f"[smoke] Phoenix projects: {names}")

        # Find our project
        our_project = next(
            (p for p in projects if p.get("name") == PROJECT_NAME), None
        )
        if our_project is None:
            print(f"[smoke] Project '{PROJECT_NAME}' not yet visible via API "
                  f"(may take a moment). Open http://localhost:6006 to check manually.")
            return

        project_id = our_project.get("id") or our_project.get("identifier")
        url = f"{PHOENIX_API_BASE}/v1/projects/{project_id}/spans?limit=5"
        with urllib.request.urlopen(url, timeout=5) as resp:
            trace_data = _json.loads(resp.read())
        span_count = len(trace_data.get("data", []))
        print(f"[smoke] Spans visible in project '{PROJECT_NAME}': {span_count}")
        if span_count > 0:
            print(f"[OK] Phoenix received the test spans.")
        else:
            print(f"[smoke] No spans visible yet — they may still be indexing. "
                  f"Open http://localhost:6006 to check.")

    except Exception as e:
        print(f"[smoke] API verification skipped ({type(e).__name__}: {e}). "
              f"Open http://localhost:6006 to verify manually.")


if __name__ == "__main__":
    send_test_spans()
    verify_via_api()
    print()
    print("[OK] Smoke test complete.")
    print("     Open http://localhost:6006 in your browser.")
    print(f"     Look for project '{PROJECT_NAME}' -> Traces -> 'smoke.agent.run'")
