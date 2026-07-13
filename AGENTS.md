# Contributor Guide for Coding Agents

Treat current code and executable contracts as the source of truth. Documentation should describe the runtime state explicitly: default, opt-in, tested library, evaluation-only, or backlog.

## Project map

- `agent/loop.py` — shared `run_task()` orchestration seam.
- `agent/cli/` — interactive REPL, rendering, model/skill selectors, and resume flow.
- `agent/runtime/` — sessions, projects, settings, hooks, permissions, request state, and durable storage.
- `agent/context/` — project instructions, request views, attachments, and compaction.
- `agent/tools/` — tool contracts, pool/exposure/runtime layers, built-ins, deferred discovery, and result storage.
- `agent/memory/` — SessionMemory, AutoMemory, typed recall, governance, consolidation, and AutoDream.
- `agent/skills/` — discovery, progressive loading, invocation state, and compact/resume restoration.
- `agent/subagents/` — synchronous one-shot `Agent` tool.
- `agent/tasks/` — background process registry and persistent dependency graph.
- `agent/mcp/` — optional stdio configuration, connections, lifecycle management, and tool adaptation.
- `agent/plugins/` — MCP configuration compatibility imports only; not a general plugin platform.
- `obs/` — structured local traces, viewer, and optional OpenTelemetry export.
- `eval/` — runtime, context/compression, memory, MCP, EvoClaw, and SWE-bench evaluation code.
- `tests/` — root regression and public-release policy tests.

## Working rules

1. Prefer current code over stale prose and verify call paths before changing claims.
2. Keep each change coherent and preserve unrelated work.
3. Add a failing regression test before fixing a bug.
4. Explain why a boundary exists; do not preserve private-source identifiers or line mappings in comments.
5. Never commit credentials, raw model transcripts, private traces, `.tool_results`, local environment files, generated benchmark outputs, or machine-specific paths.
6. Keep evaluation claims within the artifact's declared `claim` and `what_this_does_not_prove` boundary.
7. Treat synthetic MCP and checker fixtures as contract self-tests, not model-quality or product-benefit evidence.
8. Do not turn library-only features such as AutoDream into default-runtime claims without wiring and tests.

## Verification

Use Python 3.12 or newer:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pytest eval/memory/tests eval/runtime_eval eval/memory_eval -m "not live"
.\.venv\Scripts\python.exe scripts/regression_gate.py offline
```

On POSIX, use the equivalent `.venv/bin/python` path. The offline gate must create a new evidence directory; never overwrite a preserved round. Packaging changes also require a wheel install/import/CLI check in a fresh environment.
