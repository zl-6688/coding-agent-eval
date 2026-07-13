"""Runtime configuration with shell > launch-directory ``.env`` > defaults."""

import contextlib
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load the target project's .env without overriding explicit shell values.
# Using the launch directory keeps installed package directories read-only.
load_dotenv(Path.cwd() / ".env", override=False)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL") or None
MODEL_ID = os.environ.get("MODEL_ID", "claude-sonnet-4-6")


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


# Optional provider/model output cap for endpoints that do not publish a
# reliable model capability descriptor. compact.py also adapts to provider
# budget errors at runtime.
MODEL_MAX_OUTPUT_TOKENS = _positive_int_env("MODEL_MAX_OUTPUT_TOKENS")
# Keep the judge distinct from the evaluated model to avoid direct self-rating.
JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "")

# Keep mutable default runtime state out of both the installed package and the
# caller's project. Session/script entrypoints provide an explicit workdir.
_ACE_HOME = Path(os.environ.get("ACE_HOME") or (Path.home() / ".ace")).expanduser().resolve()

# Direct callers without an explicit session use an isolated application
# workspace. Session and evaluation entrypoints override it via using_workdir.
WORKDIR = Path(
    os.environ.get("AGENT_WORKDIR") or (_ACE_HOME / "workspaces" / "default")
).expanduser().resolve()

# trace 事件输出目录
TRACES_DIR = Path(os.environ.get("TRACES_DIR") or (_ACE_HOME / "traces")).expanduser().resolve()


def ensure_workdir(path=None) -> Path:
    """Create and return a runtime workspace at the point it is needed."""

    target = Path(path or WORKDIR).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


@contextlib.contextmanager
def using_workdir(path):
    """Temporarily bind the process-global workspace for a serial evaluation.

    Concurrent evaluations require an executor- or context-local workspace;
    this compatibility helper always restores the previous value on exit.
    """
    global WORKDIR
    old = WORKDIR
    WORKDIR = ensure_workdir(path)
    try:
        yield WORKDIR
    finally:
        WORKDIR = old
