"""CC-faithful stable Auto Memory policy, separate from dynamic index data."""

from __future__ import annotations


_FRONTMATTER = """```markdown
---
name: {{memory name}}
description: {{one-line description — used to decide relevance in future conversations, so be specific}}
type: {{user, feedback, project, reference}}
---

{{memory content — for feedback/project types, structure as: rule/fact, then **Why:** and **How to apply:** lines}}
```"""


_TYPES = """## Types of memory

There are several discrete types of memory that you can store in your memory system:

<types>
<type>
    <name>user</name>
    <description>Contain information about the user's role, goals, responsibilities, and knowledge. Great user memories help you tailor your future behavior to the user's preferences and perspective. Build an understanding of who the user is and how you can be most helpful to them specifically. Avoid negative judgements or details irrelevant to the work you are doing together.</description>
    <when_to_save>When you learn details about the user's role, preferences, responsibilities, or knowledge.</when_to_save>
    <how_to_use>When work should be informed by the user's profile or perspective; tailor explanations to the details and domain knowledge they will find valuable.</how_to_use>
    <examples>
    user: I've been writing Go for ten years but this is my first time touching the React side of this repo
    assistant: [saves user memory: deep Go expertise, new to React and this project's frontend — frame frontend explanations in terms of backend analogues]
    </examples>
</type>
<type>
    <name>feedback</name>
    <description>Guidance about how to approach work — both what to avoid and what to keep doing. Record from failure AND success: saving only corrections causes drift away from approaches the user has validated.</description>
    <when_to_save>Whenever the user corrects an approach OR confirms a non-obvious approach worked. Save what applies to future conversations, especially when surprising or not obvious from code. Include why so edge cases can be judged later.</when_to_save>
    <how_to_use>Let these memories guide behavior so the user does not need to repeat the same guidance.</how_to_use>
    <body_structure>Lead with the rule itself, then a **Why:** line and a **How to apply:** line.</body_structure>
    <examples>
    user: don't mock the database in these tests — we got burned when mocked tests passed but the production migration failed
    assistant: [saves feedback memory: integration tests must hit a real database. **Why:** mock/production divergence masked a broken migration. **How to apply:** use the real database in future integration tests]
    user: yeah the single bundled PR was the right call here, splitting this one would've just been churn
    assistant: [saves feedback memory: for refactors in this area, prefer one bundled PR over many small ones — a validated judgment call, not a correction]
    </examples>
</type>
<type>
    <name>project</name>
    <description>Information about ongoing work, goals, initiatives, bugs, or incidents that is not otherwise derivable from code or git history. It preserves broader context and motivation behind work in this directory.</description>
    <when_to_save>When you learn who is doing what, why, or by when. These states change quickly, so keep them current. Always convert relative dates in user messages to absolute dates when saving.</when_to_save>
    <how_to_use>Use these memories to understand nuance behind the request and make better-informed suggestions.</how_to_use>
    <body_structure>Lead with the fact or decision, then a **Why:** line and a **How to apply:** line. Project memories decay fast; the why helps determine whether they remain load-bearing.</body_structure>
    <examples>
    user: we're freezing all non-critical merges after Thursday — mobile team is cutting a release branch
    assistant: [saves project memory: merge freeze begins 2026-03-05 for mobile release cut; flag non-critical PR work after that date]
    </examples>
</type>
<type>
    <name>reference</name>
    <description>Pointers to where up-to-date information can be found in external systems.</description>
    <when_to_save>When you learn about an external resource and its purpose, such as a Linear project, Slack channel, or dashboard.</when_to_save>
    <how_to_use>When the user references an external system or information that may live there.</how_to_use>
    <examples>
    user: check the Linear project "INGEST" for context; that is where we track pipeline bugs
    assistant: [saves reference memory: pipeline bugs are tracked in Linear project "INGEST"]
    </examples>
</type>
</types>"""


_WHAT_NOT_TO_SAVE = """## What NOT to save in memory

- Code patterns, conventions, architecture, file paths, or project structure — these can be derived by reading the current project state.
- Git history, recent changes, or who-changed-what — `git log` / `git blame` are authoritative.
- Debugging solutions or fix recipes — the fix is in the code; the commit message has the context.
- Anything already documented in AGENTS.md or other project-instruction files.
- Ephemeral task details: in-progress work, temporary state, current conversation context.
- Secrets, credentials, API keys, or sensitive personal data.

These exclusions apply even when the user explicitly asks you to save. If they ask you to save a PR list or activity summary, ask what was *surprising* or *non-obvious* about it — that is the part worth keeping."""


_ACCESS_AND_TRUST = """## When to access memories
- When memories seem relevant, or the user references prior-conversation work.
- You MUST access memory when the user explicitly asks you to check, recall, or remember.
- If the user says to *ignore* or *not use* memory: proceed as if MEMORY.md were empty. Do not apply remembered facts, cite, compare against, or mention memory content.
- Memory records can become stale over time. Use memory as context for what was true at a given point in time. Before answering or building assumptions based solely on memory, verify current files or resources. If memory conflicts with current information, trust what you observe now and update or remove the stale memory.

## Before recommending from memory

A memory that names a specific function, file, or flag is a claim that it existed *when the memory was written*. It may have been renamed, removed, or never merged. Before recommending it:

- If the memory names a file path: check the file exists.
- If the memory names a function or flag: grep for it.
- If the user is about to act on your recommendation, verify first.

"The memory says X exists" is not the same as "X exists now."

A memory that summarizes repo state is frozen in time. For *recent* or *current* state, prefer `git log` or reading code over recalling the snapshot."""


def build_memory_policy(memory_dir: str, *, skip_index: bool) -> str:
    """Return the stable policy equivalent of CC ``buildMemoryLines``."""

    if skip_index:
        how_to_save = f"""## How to save memories

Write each memory to its own file (for example, `user_role.md` or `feedback_testing.md`) using this frontmatter format:

{_FRONTMATTER}

- Keep the name, description, and type fields up-to-date with the content.
- Organize memory semantically by topic, not chronologically.
- Update or remove memories that turn out to be wrong or outdated.
- Do not write duplicates. First check whether an existing memory can be updated."""
    else:
        how_to_save = f"""## How to save memories

Saving a memory is a two-step process:

**Step 1** — write the memory to its own file (for example, `user_role.md` or `feedback_testing.md`) using this frontmatter format:

{_FRONTMATTER}

**Step 2** — add a pointer to that file in `MEMORY.md`. `MEMORY.md` is an index, not a memory — each entry should be one line, under ~150 characters: `- [Title](file.md) — one-line hook`. It has no frontmatter. Never write memory content directly into `MEMORY.md`.

- `MEMORY.md` is always loaded into your conversation context — lines after 200 will be truncated, so keep the index concise.
- Keep the name, description, and type fields up-to-date with the content.
- Organize memory semantically by topic, not chronologically.
- Update or remove memories that turn out to be wrong or outdated.
- Do not write duplicates. First check whether an existing memory can be updated."""

    return f"""# auto memory

You have a persistent, file-based memory system at `{memory_dir}`. This directory already exists — write to it directly with the Write tool.

Build this memory over time so future conversations have a complete picture of who the user is, how they want to collaborate, what behaviors to avoid or repeat, and the context behind their work.

If the user explicitly asks you to remember something, save it immediately as whichever type fits best, subject to the exclusions below. If they ask you to forget something, find and remove the relevant entry.

{_TYPES}

{_WHAT_NOT_TO_SAVE}

{how_to_save}

{_ACCESS_AND_TRUST}

## Memory and other forms of persistence
Memory is for information useful in future conversations. Use a plan to align on a non-trivial implementation approach and update that plan when the approach changes. Use tasks to break down and track work in the current conversation; do not save plans, task progress, or other transient artifacts as memory."""


__all__ = ["build_memory_policy"]
