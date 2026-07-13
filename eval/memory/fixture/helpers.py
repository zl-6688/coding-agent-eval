"""Shared helper utilities used across the project.

All general-purpose utility functions live here — NOT in utils.py (which
does not exist in this repo).
"""


def foo(x: int) -> int:
    """Return double of x.  Used throughout the codebase as a simple transform."""
    return x * 2


def bar(items: list) -> list:
    """Return a reversed copy of *items* without mutating the original."""
    return list(reversed(items))


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))
