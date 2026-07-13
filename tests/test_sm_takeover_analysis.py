import json

from eval.compression_eval.sm_takeover import analyze_paths, render_markdown, stats_to_dict


def _span(name, trace_id, span_id, *, parent=None, attrs=None, start_ns=100):
    return {
        "name": name,
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent,
        "kind": "INTERNAL",
        "start_ns": start_ns,
        "end_ns": start_ns + 1,
        "status": "OK",
        "status_message": "",
        "attributes": attrs or {},
        "duration_ms": 0.01,
    }


def _write_jsonl(path, events):
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            if event == "bad-json":
                fh.write("{bad-json\n")
            else:
                fh.write(json.dumps(event) + "\n")


def test_analyze_sm_takeover_counts_statuses_and_pipeline_links(tmp_path):
    trace = tmp_path / "events.jsonl"
    events = [
        _span(
            "compact.pipeline",
            "t1",
            "p1",
            attrs={"did_sm": True, "did_full": False},
        ),
        _span(
            "compact.session_memory_compact",
            "t1",
            "s1",
            parent="p1",
            attrs={"status": "ok", "compact_llm_calls": 0},
        ),
        _span(
            "compact.pipeline",
            "t2",
            "p2",
            attrs={"did_sm": False, "did_full": True},
        ),
        _span(
            "compact.session_memory_compact",
            "t2",
            "s2",
            parent="p2",
            attrs={"status": "fallback_still_over"},
        ),
        _span(
            "compact.pipeline",
            "t3",
            "p3",
            attrs={"did_sm": False, "did_full": True},
        ),
        _span(
            "compact.session_memory_compact",
            "direct",
            "s3",
            attrs={"status": "fallback_empty"},
        ),
        "bad-json",
    ]
    _write_jsonl(trace, events)

    stats = analyze_paths([tmp_path])
    data = stats_to_dict(stats)

    assert data["files_scanned"] == 1
    assert data["malformed_lines"] == 1
    assert data["sm_attempts"] == 3
    assert data["sm_ok"] == 1
    assert data["sm_status_counts"] == {
        "fallback_empty": 1,
        "fallback_still_over": 1,
        "ok": 1,
    }
    assert data["sm_direct_or_unlinked"] == 1
    assert data["pipeline_spans"] == 3
    assert data["pipeline_with_sm_attempt"] == 2
    assert data["pipeline_did_sm_true"] == 1
    assert data["pipeline_did_full_true"] == 2
    assert data["pipeline_did_full_true_after_sm_attempt"] == 1
    assert data["saved_full_compact_calls_estimate"] == 1


def test_require_pipeline_parent_counts_only_child_sm_spans(tmp_path):
    trace = tmp_path / "events.jsonl"
    events = [
        _span("compact.pipeline", "t1", "p1", attrs={"did_sm": True, "did_full": False}),
        _span(
            "compact.session_memory_compact",
            "t1",
            "s1",
            parent="p1",
            attrs={"status": "ok"},
        ),
        _span(
            "compact.session_memory_compact",
            "direct",
            "s2",
            attrs={"status": "fallback_empty"},
        ),
    ]
    _write_jsonl(trace, events)

    stats = analyze_paths([trace], require_pipeline_parent=True)
    data = stats_to_dict(stats)

    assert data["sm_attempts"] == 1
    assert data["sm_status_counts"] == {"ok": 1}
    assert data["sm_direct_or_unlinked"] == 0
    assert data["pipeline_with_sm_attempt"] == 1


def test_render_markdown_includes_takeover_summary(tmp_path):
    trace = tmp_path / "events.jsonl"
    _write_jsonl(
        trace,
        [
            _span("compact.pipeline", "t1", "p1", attrs={"did_sm": True, "did_full": False}),
            _span(
                "compact.session_memory_compact",
                "t1",
                "s1",
                parent="p1",
                attrs={"status": "ok"},
            ),
        ],
    )

    report = render_markdown(analyze_paths([trace]))

    assert "# SessionMemory Compact Takeover" in report
    assert "| SM attempts | 1 |" in report
    assert "| SM takeover rate | 100.00% |" in report
    assert "| Avoided sync full_compact calls estimate | 1 |" in report
    assert "| `ok` | 1 |" in report
