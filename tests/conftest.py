"""conftest.py — shared fixtures for the core characterization test suite.

These tests are OFFLINE (no real LLM API calls) unless marked @pytest.mark.live.
They lock observable behavior of the surviving core before the EvalHooks refactor.
"""
import sys
from pathlib import Path

import pytest

# Add repo root so agent/* and obs/* are importable without `pip install -e .`
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class CaptureSink:
    """In-memory span sink that records every emitted Span as a plain dict."""

    def __init__(self):
        self._events: list[dict] = []

    def emit(self, span) -> None:  # noqa: ANN001
        self._events.append(span.to_event())

    def events(self) -> list[dict]:
        return list(self._events)


@pytest.fixture
def capture_sink():
    """Install a CaptureSink for the test; restore the prior sink on teardown."""
    import obs.trace as _trace
    prior = _trace._SINK
    sink = CaptureSink()
    _trace.set_sink(sink)
    yield sink
    _trace._SINK = prior


@pytest.fixture(autouse=True)
def reset_compact_state():
    """Reset compact module globals between tests to prevent cross-run bleed."""
    from agent.context import compact
    compact.reset_state()
    yield
    compact.reset_state()


@pytest.fixture(autouse=True)
def reset_tool_state():
    """Reset tool module globals (bash history + error counter) between tests."""
    from agent import tools
    from agent.tasks import reset_task_registry
    tools.reset_bash_history()
    tools.reset_file_read_state()
    reset_task_registry()
    yield
    reset_task_registry()
    tools.reset_bash_history()
    tools.reset_file_read_state()


# ── Mock helpers shared across test modules ──────────────────────────────────

class MockBlock:
    """Minimal Anthropic content block that the loop and llm modules will accept."""

    def __init__(self, blk_type: str, **attrs):
        self.type = blk_type
        for k, v in attrs.items():
            setattr(self, k, v)


class MockUsage:
    """Minimal usage object — model_dump() will fail → llm.chat falls back to getattr."""

    def __init__(self, input_tokens: int = 10, output_tokens: int = 5):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockResponse:
    """Minimal Anthropic messages.create response."""

    def __init__(self, blocks: list, stop_reason: str = "end_turn"):
        self.content = blocks
        self.stop_reason = stop_reason
        self.usage = MockUsage()


def tool_use_resp(name: str = "bash", inp: dict = None, tid: str = "tid1") -> MockResponse:
    """LLM response that requests a single tool call."""
    blk = MockBlock("tool_use", name=name, input=inp or {"command": "echo hi"}, id=tid)
    return MockResponse([blk], "tool_use")


def end_turn_resp(text: str = "done") -> MockResponse:
    """LLM response that signals task completion."""
    blk = MockBlock("text", text=text)
    return MockResponse([blk], "end_turn")


def script(*responses) -> "callable":
    """Return a fake llm.chat that pops from *responses in order, then loops end_turn."""
    seq = list(responses)

    def fake_chat(
        messages,
        system="",
        tools=None,
        max_tokens=4096,
        model=None,
        purpose="agent",
        temperature=None,
    ):
        return seq.pop(0) if seq else end_turn_resp("(script exhausted)")

    return fake_chat
