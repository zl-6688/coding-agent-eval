"""Source-aligned tests for the offline trace viewer and its public CLI."""

from __future__ import annotations

import json

from obs.viewer import to_html
from scripts import view_trace


def _event(
    name: str,
    span_id: str,
    *,
    parent: str | None = None,
    start: int = 0,
    end: int = 10_000_000,
    kind: str = "INTERNAL",
    status: str = "OK",
    attributes: dict | None = None,
    trace_id: str = "trace-1",
) -> dict:
    return {
        "name": name,
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent,
        "kind": kind,
        "start_ns": start,
        "end_ns": end,
        "duration_ms": round((end - start) / 1e6, 2),
        "status": status,
        "status_message": "",
        "attributes": attributes or {},
    }


def _runtime_events(*, trace_id: str = "trace-1") -> list[dict]:
    return [
        _event(
            "agent.run",
            "run",
            end=100_000_000,
            kind="AGENT",
            attributes={
                "task": "repair the failing test",
                "outcome": "finished",
                "turns": 3,
                "peak_context_tokens": 54_321,
                "compaction_triggered": True,
                "n_tool_errors": 1,
            },
            trace_id=trace_id,
        ),
        _event(
            "agent.turn",
            "turn-3",
            parent="run",
            start=5_000_000,
            end=95_000_000,
            kind="AGENT",
            attributes={
                "turn_index": 3,
                "context_tokens": 12_345,
                "stop_reason": "tool_use",
            },
            trace_id=trace_id,
        ),
        _event(
            "llm.call",
            "llm-1",
            parent="turn-3",
            start=10_000_000,
            end=30_000_000,
            kind="CLIENT",
            attributes={
                "context.tokens_sent": 12_345,
                "gen_ai.usage.input_tokens": 345,
                "gen_ai.usage.output_tokens": 67,
                "gen_ai.request.model": "runtime-model",
            },
            trace_id=trace_id,
        ),
        _event(
            "tool.bash",
            "tool-1",
            parent="turn-3",
            start=31_000_000,
            end=40_000_000,
            kind="TOOL",
            status="ERROR",
            attributes={
                "tool.name": "bash",
                "tool.command.preview": "pytest -q",
                "tool.output.preview": "FAILED tests/test_real.py::test_bug",
            },
            trace_id=trace_id,
        ),
    ]


def test_viewer_projects_current_runtime_measurements() -> None:
    rendered = to_html(_runtime_events(), "Runtime trace")

    for expected in (
        "finished",
        "runtime-model",
        "pytest -q",
        "FAILED tests/test_real.py::test_bug",
        "12345",
    ):
        assert expected in rendered


def test_viewer_escapes_dynamic_title_name_and_trace_id() -> None:
    events = _runtime_events(trace_id="trace-<public>")
    events.append(
        _event(
            "<img src=x onerror=alert(1)>",
            "custom",
            parent="run",
            start=96_000_000,
            end=97_000_000,
            trace_id="trace-<public>",
        )
    )

    rendered = to_html(events, "<b>unsafe title</b>")

    assert "<b>unsafe title</b>" not in rendered
    assert "&lt;b&gt;unsafe title&lt;/b&gt;" in rendered
    assert "trace-&lt;public&gt;" in rendered
    assert "<img src=x onerror=alert(1)>" not in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered


def test_viewer_empty_trace_is_still_a_complete_document() -> None:
    rendered = to_html([])

    assert rendered.startswith("<!DOCTYPE html>")
    assert "(no spans)" in rendered
    assert rendered.endswith("</html>")


def test_cli_lists_and_renders_to_explicit_output(tmp_path, monkeypatch, capsys) -> None:
    trace_path = tmp_path / "events.jsonl"
    trace_path.write_text(
        "\n".join(json.dumps(event) for event in _runtime_events()) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(view_trace.config, "TRACES_DIR", tmp_path)

    assert view_trace.main(["--list"]) == 0
    assert "events.jsonl" in capsys.readouterr().out

    output_path = tmp_path / "report.html"
    assert view_trace.main([str(trace_path), "--output", str(output_path)]) == 0
    assert output_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_cli_reports_an_empty_trace_directory(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(view_trace.config, "TRACES_DIR", tmp_path)

    assert view_trace.main([]) == 1
    assert "No trace files found" in capsys.readouterr().err
