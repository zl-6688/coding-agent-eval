from __future__ import annotations

import logging

from agent.runtime.memory_integration import extract_session_memory_after_tools
from obs.trace import SpanStatus


_MESSAGES = [{"role": "user", "content": "continue the coding task"}]


class _SessionMemoryProbe:
    def __init__(
        self,
        *,
        should_extract: bool = True,
        should_error: Exception | None = None,
        extract_error: Exception | None = None,
    ) -> None:
        self.should_extract_result = should_extract
        self.should_error = should_error
        self.extract_error = extract_error
        self.should_calls = 0
        self.extract_calls = 0

    def should_extract(self, messages: list) -> bool:
        self.should_calls += 1
        if self.should_error is not None:
            raise self.should_error
        return self.should_extract_result

    def extract(self, messages: list, system: str) -> None:
        self.extract_calls += 1
        if self.extract_error is not None:
            raise self.extract_error


def _session_span(capture_sink) -> dict:
    matches = [
        event
        for event in capture_sink.events()
        if event["name"] == "memory.session.extract"
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_safe_failure_logs(caplog, *, phase: str, error_type: str, secret: str) -> None:
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert f"phase={phase}" in warnings[0].getMessage()
    assert f"error_type={error_type}" in warnings[0].getMessage()
    assert secret not in warnings[0].getMessage()
    assert warnings[0].exc_info is None

    debug = [record for record in caplog.records if record.levelno == logging.DEBUG]
    assert len(debug) == 1
    assert f"phase={phase}" in debug[0].getMessage()
    assert f"error_type={error_type}" in debug[0].getMessage()
    assert secret not in debug[0].getMessage()
    assert debug[0].exc_info is not None


def test_should_extract_failure_is_best_effort_and_traceable(
    capture_sink,
    caplog,
):
    secret = "credential=SHOULD-NOT-ENTER-SPAN"
    memory = _SessionMemoryProbe(should_error=LookupError(secret))
    caplog.set_level(logging.DEBUG, logger="agent.runtime.memory_integration")

    result = extract_session_memory_after_tools(memory, _MESSAGES, "system")

    assert result is None
    assert memory.should_calls == 1
    assert memory.extract_calls == 0
    _assert_safe_failure_logs(
        caplog,
        phase="should_extract",
        error_type="LookupError",
        secret=secret,
    )
    event = _session_span(capture_sink)
    assert event["status"] == SpanStatus.ERROR
    assert event["attributes"] == {
        "memory.session.phase": "should_extract",
        "memory.session.status": "error",
        "memory.session.error_type": "LookupError",
    }
    assert "should_extract" in event["status_message"]
    assert "LookupError" in event["status_message"]
    assert secret not in event["status_message"]


def test_extract_failure_is_best_effort_and_traceable(capture_sink, caplog):
    secret = "provider-response=SHOULD-NOT-ENTER-SPAN"
    memory = _SessionMemoryProbe(extract_error=RuntimeError(secret))
    caplog.set_level(logging.DEBUG, logger="agent.runtime.memory_integration")

    result = extract_session_memory_after_tools(memory, _MESSAGES, "system")

    assert result is None
    assert memory.should_calls == 1
    assert memory.extract_calls == 1
    _assert_safe_failure_logs(
        caplog,
        phase="extract",
        error_type="RuntimeError",
        secret=secret,
    )
    event = _session_span(capture_sink)
    assert event["status"] == SpanStatus.ERROR
    assert event["attributes"] == {
        "memory.session.phase": "extract",
        "memory.session.status": "error",
        "memory.session.should_extract": True,
        "memory.session.error_type": "RuntimeError",
    }
    assert "extract" in event["status_message"]
    assert "RuntimeError" in event["status_message"]
    assert secret not in event["status_message"]


def test_success_records_extraction_outcome(capture_sink, caplog):
    memory = _SessionMemoryProbe()
    caplog.set_level(logging.DEBUG, logger="agent.runtime.memory_integration")

    extract_session_memory_after_tools(memory, _MESSAGES, "system")

    assert memory.should_calls == 1
    assert memory.extract_calls == 1
    assert not caplog.records
    event = _session_span(capture_sink)
    assert event["status"] == SpanStatus.OK
    assert event["status_message"] == ""
    assert event["attributes"] == {
        "memory.session.phase": "extract",
        "memory.session.status": "success",
        "memory.session.should_extract": True,
    }


def test_noop_records_threshold_decision_without_extracting(capture_sink, caplog):
    memory = _SessionMemoryProbe(should_extract=False)
    caplog.set_level(logging.DEBUG, logger="agent.runtime.memory_integration")

    extract_session_memory_after_tools(memory, _MESSAGES, "system")

    assert memory.should_calls == 1
    assert memory.extract_calls == 0
    assert not caplog.records
    event = _session_span(capture_sink)
    assert event["status"] == SpanStatus.OK
    assert event["status_message"] == ""
    assert event["attributes"] == {
        "memory.session.phase": "should_extract",
        "memory.session.status": "noop",
        "memory.session.should_extract": False,
    }


def test_disabled_session_memory_emits_no_span(capture_sink):
    extract_session_memory_after_tools(None, _MESSAGES, "system")

    assert capture_sink.events() == []
