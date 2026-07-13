"""Deterministic builders for schema-valid SessionMemory eval fixtures."""

from agent.memory.session_memory import ACE_CHECKPOINT_SECTIONS, SESSION_MEMORY_TEMPLATE


def checkpoint_fixture(content: str, *, section: str = "Files and evidence") -> str:
    """Insert deterministic eval content into one of the six checkpoint sections."""

    if section not in ACE_CHECKPOINT_SECTIONS:
        raise ValueError(f"Unknown ACE checkpoint section: {section}")
    body = content.strip()
    if not body:
        return SESSION_MEMORY_TEMPLATE
    heading = f"## {section}\n"
    return SESSION_MEMORY_TEMPLATE.replace(heading, f"{heading}{body}\n", 1)
