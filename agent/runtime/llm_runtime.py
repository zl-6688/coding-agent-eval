"""REPL-scoped LLM runtime overrides from ACE settings (CC-shaped env + model).

Eval and direct run_task callers never enter this context — they keep using agent.config
and repo .env defaults. REPL wraps each Session.run in using_repl_llm_runtime().
"""

from __future__ import annotations

import contextlib
import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

from .. import config

# Fallback when no REPL settings override (eval path): cheap selector, not main MODEL_ID.
_DEFAULT_RECALL_MODEL = "deepseek-v4-flash"
_DEFAULT_MEMORY_MODEL = "deepseek-v4-flash"

# CC applies settings.env after trust; we whitelist provider/proxy keys only so a
# project .ace/settings.json cannot rewrite PATH/LD_PRELOAD (managedEnv.ts pattern).
_SAFE_SETTINGS_ENV_KEYS = frozenset({
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "MODEL_ID",
    "MODEL_MAX_OUTPUT_TOKENS",
    "JUDGE_MODEL_ID",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "ACE_MEMORY_MODEL",
    "ACE_COMPACTION_MODEL",
    "ACE_RECALL_MODEL",
})

_current: ContextVar["LlmRuntimeConfig | None"] = ContextVar(
    "ace_llm_runtime",
    default=None,
)


@dataclass(frozen=True)
class LlmRuntimeConfig:
    """Effective LLM credentials/models for one REPL task."""

    api_key: str | None = None
    base_url: str | None = None
    model_id: str | None = None
    memory_model_id: str | None = None
    compaction_model_id: str | None = None
    recall_model_id: str | None = None

    def is_active(self) -> bool:
        return any((
            self.api_key,
            self.base_url is not None,
            self.model_id,
            self.memory_model_id,
            self.compaction_model_id,
            self.recall_model_id,
        ))


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _filter_settings_env(env: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(env, dict):
        return {}
    filtered: dict[str, str] = {}
    for key, value in env.items():
        name = str(key).strip()
        if name not in _SAFE_SETTINGS_ENV_KEYS:
            continue
        if value is None:
            continue
        filtered[name] = str(value)
    return filtered


def display_model_from_settings(
    settings: dict[str, Any] | None,
    *,
    session_model: str | None = None,
) -> str:
    """Banner label: session override > settings.model > config.MODEL_ID."""
    if session_model:
        return session_model
    runtime = resolve_llm_runtime_config(settings)
    if runtime is not None and runtime.model_id:
        return runtime.model_id
    return config.MODEL_ID


def resolve_llm_runtime_config(
    settings: dict[str, Any] | None,
    *,
    session_model: str | None = None,
) -> LlmRuntimeConfig | None:
    """Build runtime config from merged ACE settings (CC: env + model + models)."""
    if not isinstance(settings, dict) or not settings:
        if session_model:
            return LlmRuntimeConfig(model_id=session_model)
        return None

    env = _filter_settings_env(settings.get("env"))
    models = settings.get("models") if isinstance(settings.get("models"), dict) else {}

    model_id = (
        _clean_str(session_model)
        or _clean_str(settings.get("model"))
        or _clean_str(env.get("MODEL_ID"))
    )
    memory_model = (
        _clean_str(models.get("memory"))
        or _clean_str(env.get("ACE_MEMORY_MODEL"))
    )
    compaction_model = (
        _clean_str(models.get("compaction"))
        or _clean_str(env.get("ACE_COMPACTION_MODEL"))
    )
    recall_model = (
        _clean_str(models.get("recall"))
        or _clean_str(env.get("ACE_RECALL_MODEL"))
    )

    cfg = LlmRuntimeConfig(
        api_key=_clean_str(env.get("ANTHROPIC_API_KEY")),
        base_url=_clean_str(env.get("ANTHROPIC_BASE_URL")),
        model_id=model_id,
        memory_model_id=memory_model,
        compaction_model_id=compaction_model,
        recall_model_id=recall_model,
    )
    return cfg if cfg.is_active() else None


def current_llm_runtime() -> LlmRuntimeConfig | None:
    return _current.get()


def effective_api_key() -> str | None:
    runtime = current_llm_runtime()
    if runtime is not None and runtime.api_key:
        return runtime.api_key
    return config.API_KEY


def effective_base_url() -> str | None:
    runtime = current_llm_runtime()
    if runtime is not None and runtime.base_url is not None:
        return runtime.base_url or None
    return config.BASE_URL


def model_for_purpose(purpose: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    runtime = current_llm_runtime()
    if runtime is not None:
        if purpose == "compaction":
            if runtime.compaction_model_id:
                return runtime.compaction_model_id
        elif purpose == "memory_recall":
            if runtime.recall_model_id:
                return runtime.recall_model_id
            return _DEFAULT_RECALL_MODEL
        elif purpose.startswith("memory_"):
            if runtime.memory_model_id:
                return runtime.memory_model_id
            return _DEFAULT_MEMORY_MODEL
        if runtime.model_id:
            return runtime.model_id
    if purpose == "memory_recall":
        return _DEFAULT_RECALL_MODEL
    if purpose.startswith("memory_"):
        return _DEFAULT_MEMORY_MODEL
    return config.MODEL_ID


@contextlib.contextmanager
def using_repl_llm_runtime(
    runtime: LlmRuntimeConfig | None,
    *,
    env_overlay: dict[str, str] | None = None,
) -> Iterator[None]:
    """Apply REPL-only LLM overrides for the duration of one interactive task."""
    token = _current.set(runtime)
    saved_env: dict[str, str | None] = {}
    overlay = dict(env_overlay or {})
    try:
        if overlay:
            for key, value in overlay.items():
                if key not in _SAFE_SETTINGS_ENV_KEYS:
                    continue
                saved_env[key] = os.environ.get(key)
                os.environ[key] = value
            if overlay.get("ANTHROPIC_BASE_URL"):
                os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
        from .. import llm

        llm.reset_client()
        yield
    finally:
        for key, previous in saved_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        _current.reset(token)
        from .. import llm

        llm.reset_client()


@contextlib.contextmanager
def using_repl_settings(
    settings: dict[str, Any] | None,
    *,
    session_model: str | None = None,
) -> Iterator[None]:
    """Convenience: resolve runtime config + env overlay from merged settings."""
    runtime = resolve_llm_runtime_config(settings, session_model=session_model)
    env_overlay = _filter_settings_env((settings or {}).get("env"))
    if runtime is None and not env_overlay:
        yield
        return
    with using_repl_llm_runtime(runtime, env_overlay=env_overlay):
        yield
