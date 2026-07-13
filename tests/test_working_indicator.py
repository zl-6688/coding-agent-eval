import io
import sys
import time

from agent.cli.working_indicator import (
    WorkingIndicator,
    active,
    format_elapsed,
    set_terminal_title,
)


def test_format_elapsed():
    assert format_elapsed(3.2) == "3.2s"
    assert format_elapsed(12.0) == "12s"
    assert format_elapsed(75) == "1m 15s"


def test_active_updates_title_only():
    buf = io.StringIO()
    old_stdout = sys.__stdout__
    sys.__stdout__ = buf
    try:
        with active(is_tty=True):
            time.sleep(0.6)
        output = buf.getvalue()
        assert "2K" not in output
        assert "\r" not in output
        assert "\033]0;" in output
    finally:
        sys.__stdout__ = old_stdout


def test_working_indicator_non_tty_is_noop():
    with active(is_tty=False) as indicator:
        time.sleep(0.02)
    assert indicator.elapsed_s >= 0


def test_set_terminal_title_writes_osc():
    buf = io.StringIO()
    old_stdout = sys.__stdout__
    sys.__stdout__ = buf
    try:
        set_terminal_title("ace · 3s")
        assert "\033]0;ace · 3s\007" in buf.getvalue()
    finally:
        sys.__stdout__ = old_stdout
