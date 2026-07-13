from __future__ import annotations

import platform

from eval.context_eval import run as context_run
from eval.mcp_eval import evidence as mcp_evidence


def test_release_evidence_uses_portable_fallbacks_for_empty_platform_fields(
    monkeypatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "")
    monkeypatch.setattr(platform, "release", lambda: "")
    monkeypatch.setattr(platform, "machine", lambda: "")

    context_environment = context_run._environment()
    mcp_environment = mcp_evidence.environment_metadata()

    assert context_environment["platform"] == "unknown"
    assert context_environment["machine"] == "unknown"
    assert mcp_environment["os"] == "unknown"
    assert mcp_environment["os_release"] == "unknown"
    assert mcp_environment["architecture"] == "unknown"
