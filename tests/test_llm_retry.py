"""test_llm_retry.py — characterization tests for agent.llm._create_with_retry.

Locks the 5-retry exponential backoff behavior BEFORE any signature changes.
All tests are offline: the anthropic client is replaced with a mock.
"""
import pytest
import anthropic
import httpx

from conftest import MockBlock, MockUsage


# ── helpers ────────────────────────────────────────────────────────────────

def _mock_httpx_response(status_code: int = 429) -> httpx.Response:
    """Build a minimal httpx.Response (required by anthropic's APIStatusError constructors)."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(status_code, request=request)


class _MockMessages:
    """messages.create that raises the given errors in order, then returns success_resp."""

    def __init__(self, errors: list, success_resp):
        self._queue = list(errors)
        self._resp = success_resp
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        if self._queue:
            raise self._queue.pop(0)
        return self._resp


class _MockClient:
    def __init__(self, messages_mock):
        self.messages = messages_mock


class _MockResp:
    """Minimal success response accepted by llm.chat's span-filling code."""
    def __init__(self, text: str = "ok"):
        self.content = [MockBlock("text", text=text)]
        self.stop_reason = "end_turn"
        self.usage = MockUsage()


@pytest.fixture
def no_sleep(monkeypatch):
    """Suppress time.sleep so retry backoff doesn't actually wait."""
    import agent.llm as _llm
    monkeypatch.setattr(_llm.time, "sleep", lambda _: None)


# ── tests ─────────────────────────────────────────────────────────────────

def test_retry_transient_then_success(monkeypatch, no_sleep, capture_sink):
    """Two transient failures → success on the third attempt.

    Asserts: response returned + llm.call span has n_retries=2.
    """
    import agent.llm as _llm

    mock_msgs = _MockMessages(
        errors=[
            anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com")),
            anthropic.RateLimitError(message="rate limit",
                                     response=_mock_httpx_response(429), body=None),
        ],
        success_resp=_MockResp("recovered"),
    )
    # Patch client() — assigning _client alone is overwritten when cache_key mismatches.
    monkeypatch.setattr(_llm, "client", lambda: _MockClient(mock_msgs))

    resp = _llm.chat([{"role": "user", "content": "hi"}], system="s")

    assert resp.stop_reason == "end_turn"

    # llm.call span should carry retry metadata
    llm_spans = [e for e in capture_sink.events() if e["name"] == "llm.call"]
    assert len(llm_spans) == 1
    assert llm_spans[0]["attributes"].get("llm.n_retries") == 2
    assert mock_msgs.call_count == 3


def test_retry_single_failure_then_success(monkeypatch, no_sleep, capture_sink):
    """Single transient failure → success on the second attempt (n_retries=1)."""
    import agent.llm as _llm

    mock_msgs = _MockMessages(
        errors=[anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com"))],
        success_resp=_MockResp("ok"),
    )
    monkeypatch.setattr(_llm, "client", lambda: _MockClient(mock_msgs))

    resp = _llm.chat([{"role": "user", "content": "hi"}], system="s")
    assert resp.stop_reason == "end_turn"
    assert mock_msgs.call_count == 2


def test_retry_exhausted_raises(monkeypatch, no_sleep):
    """Exhausting all 5 retries should raise the last error (not silently return)."""
    import agent.llm as _llm

    n_retries = _llm._MAX_RETRIES  # 5
    mock_msgs = _MockMessages(
        # one more than max: ensures it fails on all attempts including the final one
        errors=[anthropic.InternalServerError(message="5xx",
                                              response=_mock_httpx_response(500), body=None)]
              * (n_retries + 1),
        success_resp=_MockResp("never reached"),
    )
    monkeypatch.setattr(_llm, "client", lambda: _MockClient(mock_msgs))

    with pytest.raises(anthropic.InternalServerError):
        _llm.chat([{"role": "user", "content": "hi"}], system="s")

    # All 6 attempts were made (initial + 5 retries)
    assert mock_msgs.call_count == n_retries + 1


def test_non_retryable_error_raises_immediately(monkeypatch, no_sleep):
    """A non-retryable error (BadRequestError) should raise immediately without retry."""
    import agent.llm as _llm

    mock_msgs = _MockMessages(
        errors=[anthropic.BadRequestError(message="bad request",
                                          response=_mock_httpx_response(400), body=None)],
        success_resp=_MockResp(),
    )
    monkeypatch.setattr(_llm, "client", lambda: _MockClient(mock_msgs))

    with pytest.raises(anthropic.BadRequestError):
        _llm.chat([{"role": "user", "content": "hi"}], system="s")

    assert mock_msgs.call_count == 1, "non-retryable error must not be retried"


def test_no_error_no_retry(monkeypatch, no_sleep, capture_sink):
    """Clean first-call success → n_retries attribute should NOT be set on span."""
    import agent.llm as _llm

    mock_msgs = _MockMessages(errors=[], success_resp=_MockResp("first try"))
    monkeypatch.setattr(_llm, "client", lambda: _MockClient(mock_msgs))

    _llm.chat([{"role": "user", "content": "hi"}], system="s")

    llm_spans = [e for e in capture_sink.events() if e["name"] == "llm.call"]
    assert "llm.n_retries" not in llm_spans[0]["attributes"]
    assert mock_msgs.call_count == 1


def test_chat_observability_labels_non_text_blocks_without_claiming_tool_use(
    monkeypatch,
    capture_sink,
):
    import agent.llm as _llm

    class _ThinkingResp:
        content = [MockBlock("thinking", text="internal reasoning")]
        stop_reason = "max_tokens"
        usage = MockUsage()

    mock_msgs = _MockMessages(errors=[], success_resp=_ThinkingResp())
    monkeypatch.setattr(_llm, "client", lambda: _MockClient(mock_msgs))

    _llm.chat([{"role": "user", "content": "hi"}], system="s", purpose="compaction")

    llm_span = [e for e in capture_sink.events() if e["name"] == "llm.call"][-1]
    attrs = llm_span["attributes"]
    assert attrs["llm.output"] == "[non_text] thinking"
    assert attrs["llm.output_block_types"] == "thinking"
    assert attrs["gen_ai.response.stop_reason"] == "max_tokens"
