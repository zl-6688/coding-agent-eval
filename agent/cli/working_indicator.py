"""Working indicator — elapsed in terminal title only.

CC keeps spinner in a dedicated Ink layer. ACE cannot safely repaint a spinner
row on Windows stdout (ANSI leaks as ``?[2K`` and mixes with answers). Live
elapsed goes to the window/tab title; transcript = tool lines → separator → answer.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from typing import Iterator

_DEFAULT_TITLE = "ace"


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s" if seconds < 10 else f"{int(seconds)}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def set_terminal_title(title: str) -> None:
    """OSC title — visible in terminal chrome, never in conversation transcript."""
    try:
        sys.__stdout__.write(f"\033]0;{title}\007")
        sys.__stdout__.flush()
    except OSError:
        pass


class WorkingIndicator:
    """Update terminal title with elapsed time for one REPL turn."""

    def __init__(self, *, is_tty: bool = False, interval: float = 0.5):
        self._is_tty = is_tty
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self.elapsed_s = 0.0

    def _stop_timer(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        set_terminal_title(_DEFAULT_TITLE)

    def __enter__(self) -> WorkingIndicator:
        self._t0 = time.monotonic()
        if self._is_tty:
            self._thread = threading.Thread(target=self._loop, name="ace-working", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed_s = time.monotonic() - self._t0
        self._stop_timer()
        return False

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            elapsed = format_elapsed(time.monotonic() - self._t0)
            set_terminal_title(f"{_DEFAULT_TITLE} · {elapsed}")


@contextlib.contextmanager
def active(*, is_tty: bool = False) -> Iterator[WorkingIndicator]:
    with WorkingIndicator(is_tty=is_tty) as indicator:
        yield indicator
