"""Runtime package entry points.

Keep this module lightweight: ``agent.runtime.hooks`` and
``agent.runtime.permissions`` are imported by the loop/tool runtime, while
``Session`` imports the loop. Eagerly importing ``Session`` here would turn
submodule imports into an import-order cycle.
"""

from .project import Project, ace_home
from .run_context import RunContext, RunState
from .store import SessionStore

__all__ = ["Project", "RunContext", "RunState", "Session", "SessionStore", "ace_home"]


def __getattr__(name: str):
    if name == "Session":
        from .session import Session

        return Session
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
