from types import SimpleNamespace
from pathlib import Path

import pytest

from conftest import end_turn_resp, tool_use_resp

from agent.skills import (
    discover_skill_catalog,
    get_invoked_skills,
    invoked_skill_context_message,
    record_invoked_skill,
    render_skill_body,
    render_skill_listing,
    reset_invoked_skills,
    restore_invoked_skills_from_messages,
)
from agent.tools.contracts import Tool, ToolResult
from agent.tools.pool import ToolPool, ToolPoolContext, assemble_tool_pool
from agent.tools.runtime import ToolExecutionRuntime


@pytest.fixture(autouse=True)
def _isolated_ace_home(tmp_path, monkeypatch):
    ace = tmp_path / "ace"
    monkeypatch.setenv("ACE_HOME", str(ace))
    return ace


@pytest.fixture(autouse=True)
def _reset_invoked_skill_state():
    reset_invoked_skills()
    yield
    reset_invoked_skills()


def _tool_use(name="Skill", inp=None, tid="skill1"):
    return SimpleNamespace(type="tool_use", name=name, input=inp or {"skill": "demo"}, id=tid)


def _write_project_skill(root, dirname, content):
    path = root / ".claude" / "skills" / dirname / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_user_skill(ace_home, dirname, content):
    path = Path(ace_home) / "skills" / dirname / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_claude_user_skill(home, dirname, content):
    path = home / ".claude" / "skills" / dirname / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_no_skill_does_not_add_skill_tool(tmp_path):
    home = tmp_path / "home"

    pool = assemble_tool_pool(
        ToolPoolContext(workdir=str(tmp_path), metadata={"user_home": str(home)})
    )

    assert "Skill" not in [tool.name for tool in pool.tools]


def test_project_skill_discovery_and_listing_uses_summary_only(tmp_path):
    home = tmp_path / "home"
    _write_project_skill(
        tmp_path,
        "review",
        """---
name: code-review
description: Review code for defects.
when_to_use: Before accepting a code change.
allowed-tools: read_file, grep
argument-hint: changed files
---
FULL BODY SENTINEL
Check every changed file.
""",
    )

    catalog = discover_skill_catalog(tmp_path, user_home=home)
    listing = render_skill_listing(catalog)
    pool = assemble_tool_pool(
        ToolPoolContext(
            workdir=str(tmp_path),
            metadata={"skill_catalog": catalog, "user_home": str(home)},
        )
    )

    skill = catalog.find("review")
    assert skill is not None
    assert skill.display_name == "code-review"
    assert skill.source == "project"
    assert skill.allowed_tools == ("read_file", "grep")
    assert "review: Review code for defects." in listing
    assert "display_name: code-review" in listing
    assert "Before accepting a code change." in listing
    assert "FULL BODY SENTINEL" not in listing
    assert "Skill" in [tool.name for tool in pool.tools]


def test_user_skill_discovery(tmp_path, _isolated_ace_home):
    home = tmp_path / "home"
    _write_user_skill(
        _isolated_ace_home,
        "notes",
        """---
name: notes
description: Keep concise notes.
---
User skill body.
""",
    )

    catalog = discover_skill_catalog(tmp_path, user_home=home)

    skill = catalog.find("notes")
    assert skill is not None
    assert skill.source == "user"
    assert "User skill body." in skill.body


def test_claude_user_skill_discovery(tmp_path):
    home = tmp_path / "home"
    _write_claude_user_skill(
        home,
        "mentor",
        """---
description: Agent engineering mentor guidance.
---
Mentor body.
""",
    )

    catalog = discover_skill_catalog(tmp_path, user_home=home)

    skill = catalog.find("mentor")
    assert skill is not None
    assert skill.source == "user"
    assert "Mentor body." in skill.body


def test_multiline_frontmatter_description(tmp_path):
    home = tmp_path / "home"
    _write_project_skill(
        tmp_path,
        "mentor",
        """---
description: |-
  Use when discussing Agent Harness engineering.
  Focus on architecture reasoning.
---
# Body title
""",
    )
    from agent.skills import skill_summary_text

    catalog = discover_skill_catalog(tmp_path, user_home=home)
    skill = catalog.find("mentor")
    assert skill is not None
    assert "Agent Harness engineering" in skill.description
    assert "architecture reasoning" in skill.description
    assert "Agent Harness engineering" in skill_summary_text(skill)


def test_estimate_skill_tokens_and_compact_cli_listing(tmp_path):
    home = tmp_path / "home"
    _write_project_skill(
        tmp_path,
        "tiny",
        """---
description: Tiny skill.
---
Hi
""",
    )
    from agent.skills import (
        estimate_skill_tokens,
        format_cli_skill_listing,
        format_skill_token_label,
    )

    catalog = discover_skill_catalog(tmp_path, user_home=home)
    skill = catalog.find("tiny")
    assert format_skill_token_label(estimate_skill_tokens(skill)) == "< 20 tok"

    listing = format_cli_skill_listing(catalog)
    assert "Skills" in listing
    assert "✓ on · tiny · project · < 20 tok" in listing
    assert "路径" not in listing


def test_invalid_utf8_skill_is_skipped(tmp_path):
    home = tmp_path / "home"
    path = tmp_path / ".claude" / "skills" / "broken" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe\xfa")

    catalog = discover_skill_catalog(tmp_path, user_home=home)

    assert catalog.is_empty()


def test_skill_without_frontmatter_derives_description_from_body(tmp_path):
    home = tmp_path / "home"
    _write_project_skill(
        tmp_path,
        "plain",
        """# Plain Skill

Body line.
""",
    )

    catalog = discover_skill_catalog(tmp_path, user_home=home)

    skill = catalog.find("plain")
    assert skill is not None
    assert skill.description == "Plain Skill"
    assert skill.body.startswith("# Plain Skill")


def test_render_skill_body_replaces_claude_placeholders(tmp_path):
    home = tmp_path / "home"
    _write_project_skill(
        tmp_path,
        "paths",
        """---
description: Path skill.
---
dir=${CLAUDE_SKILL_DIR}
session=${CLAUDE_SESSION_ID}
args=$ARGUMENTS
""",
    )
    catalog = discover_skill_catalog(tmp_path, user_home=home)
    skill = catalog.find("paths")

    rendered = render_skill_body(skill, args="topic", run_id="run-123")

    assert rendered.startswith("Base directory for this skill:")
    assert "dir=" in rendered
    skill_dir_value = rendered.split("dir=", 1)[1].splitlines()[0]
    assert "\\" not in skill_dir_value
    assert "session=run-123" in rendered
    assert "args=topic" in rendered


def test_project_skill_wins_same_name_over_user_skill(tmp_path, _isolated_ace_home):
    home = tmp_path / "home"
    _write_user_skill(
        _isolated_ace_home,
        "same",
        """---
name: same
description: user version
---
USER BODY
""",
    )
    _write_project_skill(
        tmp_path,
        "same",
        """---
name: same
description: project version
---
PROJECT BODY
""",
    )

    catalog = discover_skill_catalog(tmp_path, user_home=home)

    skill = catalog.find("same")
    assert skill is not None
    assert skill.source == "project"
    assert skill.description == "project version"
    assert "PROJECT BODY" in skill.body
    assert "USER BODY" not in skill.body


def test_skill_call_injects_full_body_next_turn_without_system_pollution(monkeypatch, tmp_path):
    from agent import loop, llm, tools

    home = tmp_path / "home"
    _write_project_skill(
        tmp_path,
        "demo",
        """---
name: demo
description: Demo summary only.
argument-hint: topic
---
Use demo with $ARGUMENTS.
FULL BODY SENTINEL
""",
    )

    class _Exec:
        cwd = str(tmp_path)

    calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        calls.append({"messages": messages, "system": system, "tools": tools or []})
        if len(calls) == 1:
            assert "Skill" in [tool["name"] for tool in tools or []]
            return tool_use_resp("Skill", {"skill": "demo", "args": "topic-1"}, "skill-call")
        return end_turn_resp("done")

    real_discover = discover_skill_catalog
    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(tools, "get_executor", lambda: _Exec())
    monkeypatch.setattr(
        loop,
        "discover_skill_catalog",
        lambda workdir: real_discover(workdir, user_home=home),
    )

    text, durable = loop.run_task("q", max_turns=3, trace=False, return_messages=True)

    assert text == "done"
    assert len(calls) == 2
    first_request = str(calls[0]["messages"])
    second_request = str(calls[1]["messages"])
    assert "Demo summary only." in first_request
    assert "FULL BODY SENTINEL" not in first_request
    assert "FULL BODY SENTINEL" in second_request
    assert "Use demo with topic-1." in second_request
    assert "Base directory for this skill:" in second_request
    assert all("FULL BODY SENTINEL" not in call["system"] for call in calls)
    assert "FULL BODY SENTINEL" in str(durable)
    assert "Demo summary only." not in str(durable)


def test_skill_call_records_invoked_body_for_agent_scope(tmp_path):
    home = tmp_path / "home"
    _write_project_skill(
        tmp_path,
        "demo",
        """---
name: demo
description: Demo summary only.
---
Body for $ARGUMENTS.
""",
    )
    catalog = discover_skill_catalog(tmp_path, user_home=home)
    pool = assemble_tool_pool(
        ToolPoolContext(workdir=str(tmp_path), metadata={"skill_catalog": catalog})
    )
    runtime = ToolExecutionRuntime.from_tool_pool(pool, run_id="run-1", agent_id="agent-a")

    runtime.execute_tool_uses([_tool_use(inp={"skill": "demo", "args": "topic"}, tid="skill-call")])

    invoked = get_invoked_skills("agent-a")
    assert len(invoked) == 1
    assert invoked[0].name == "demo"
    assert "Body for topic." in invoked[0].content
    assert get_invoked_skills("other-agent") == ()
    reminder = invoked_skill_context_message("agent-a")
    assert reminder is not None
    assert "### Skill: demo" in reminder["content"]
    assert "Body for topic." in reminder["content"]


def test_restore_invoked_skills_from_durable_and_compact_messages():
    durable_messages = [
        {
            "role": "user",
            "content": (
                "<system-reminder>\n"
                "# skill: demo\n\n"
                "Durable skill body.\n"
                "</system-reminder>\n"
            ),
        }
    ]

    restored = restore_invoked_skills_from_messages(durable_messages, agent_id="resume-run")
    assert len(restored) == 1
    assert get_invoked_skills("resume-run")[0].content == "Durable skill body."

    compact_message = invoked_skill_context_message("resume-run")
    reset_invoked_skills()
    restored_again = restore_invoked_skills_from_messages([compact_message], agent_id="new-run")

    assert len(restored_again) == 1
    assert get_invoked_skills("new-run")[0].name == "demo"
    assert get_invoked_skills("new-run")[0].content == "Durable skill body."


def test_restore_compact_message_preserves_recent_skill_priority_under_budget():
    record_invoked_skill(
        "old",
        "skills/old/SKILL.md",
        "OLD_SKILL_BODY " * 18,
        agent_id="original-run",
        invoked_at=1.0,
    )
    record_invoked_skill(
        "new",
        "skills/new/SKILL.md",
        "NEW_SKILL_BODY " * 18,
        agent_id="original-run",
        invoked_at=2.0,
    )
    compact_message = invoked_skill_context_message(
        "original-run",
        token_budget=1_000,
        max_tokens_per_skill=1_000,
    )
    assert compact_message is not None
    assert compact_message["content"].index("### Skill: new") < compact_message[
        "content"
    ].index("### Skill: old")

    reset_invoked_skills()
    restore_invoked_skills_from_messages([compact_message], agent_id="restored-run")

    restored_names = [skill.name for skill in get_invoked_skills("restored-run")]
    assert restored_names == ["new", "old"]
    next_compact_message = invoked_skill_context_message(
        "restored-run",
        token_budget=80,
        max_tokens_per_skill=1_000,
    )

    assert next_compact_message is not None
    assert "### Skill: new" in next_compact_message["content"]
    assert "NEW_SKILL_BODY" in next_compact_message["content"]
    assert "### Skill: old" not in next_compact_message["content"]
    assert "OLD_SKILL_BODY" not in next_compact_message["content"]


def test_full_compact_queues_invoked_skill_body_as_post_compact_attachment(monkeypatch):
    from agent.context import compact as _compact
    from conftest import MockBlock, MockUsage

    class _SummaryResp:
        content = [MockBlock("text", text="<summary>summary without skill</summary>")]
        stop_reason = "end_turn"
        usage = MockUsage()

    monkeypatch.setattr(_compact.llm, "chat", lambda *a, **kw: _SummaryResp())
    from agent.skills import record_invoked_skill

    record_invoked_skill(
        "demo",
        "skills/demo/SKILL.md",
        "COMPACT RESTORED SKILL BODY",
        agent_id="run-a",
    )
    cfg = _compact.CompactConfig(keep_min_tokens=10, keep_min_msgs=1, keep_max_tokens=20)

    result = _compact.full_compact(
        [{"role": "user", "content": "old context " * 100}],
        cfg=cfg,
        skill_agent_id="run-a",
    )
    combined = "\n".join(str(m.get("content", "")) for m in result)
    pending = _compact.drain_post_compact_attachments()
    pending_text = "\n".join(str(m.get("content", "")) for m in pending)

    assert "summary without skill" in combined
    assert "COMPACT RESTORED SKILL BODY" not in combined
    assert "COMPACT RESTORED SKILL BODY" in pending_text
    assert _compact.drain_post_compact_attachments() == ()


def test_run_task_resets_invoked_skill_state_between_runs(monkeypatch):
    from agent import loop, llm
    from agent.skills import record_invoked_skill

    record_invoked_skill("demo", "", "SHOULD NOT LEAK", agent_id="old-run")

    monkeypatch.setattr(llm, "chat", lambda *a, **kw: end_turn_resp("done"))
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)
    monkeypatch.setattr(loop, "discover_skill_catalog", lambda workdir: discover_skill_catalog(None))

    text, _durable = loop.run_task("q", max_turns=1, trace=False, return_messages=True)

    assert text == "done"
    assert get_invoked_skills("old-run") == ()


def test_run_task_restores_initial_skill_body_before_compaction(monkeypatch):
    from agent import loop, llm
    from conftest import MockBlock, MockUsage

    old_skill_message = {
        "role": "user",
        "content": (
            "<system-reminder>\n"
            "# skill: demo\n\n"
            "RESUME RESTORED SKILL BODY\n"
            "</system-reminder>\n"
        ),
    }
    initial_messages = [old_skill_message] + [
        {"role": "user", "content": f"tail-{i} " + ("X" * 10_000)}
        for i in range(5)
    ]
    agent_requests = []

    class _SummaryResp:
        content = [MockBlock("text", text="<summary>summary without resume body</summary>")]
        stop_reason = "end_turn"
        usage = MockUsage()

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        if kwargs.get("purpose") == "compaction":
            return _SummaryResp()
        agent_requests.append(messages)
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(loop, "build_system", lambda state: "SYS")
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)
    monkeypatch.setattr(loop, "discover_skill_catalog", lambda workdir: discover_skill_catalog(None))

    text, _durable = loop.run_task(
        "q",
        max_turns=1,
        trace=False,
        initial_messages=initial_messages,
        return_messages=True,
        eval_hooks=loop.EvalHooks(compact_strategy="full", compact_threshold=100),
    )

    assert text == "done"
    assert agent_requests
    assert "RESUME RESTORED SKILL BODY" in str(agent_requests[0])


def test_skill_tool_unknown_and_disabled_errors(tmp_path):
    home = tmp_path / "home"
    _write_project_skill(
        tmp_path,
        "disabled",
        """---
name: disabled
description: Disabled skill.
disable-model-invocation: true
---
Disabled body.
""",
    )
    _write_project_skill(
        tmp_path,
        "enabled",
        """---
description: Enabled skill.
---
Enabled body.
""",
    )
    catalog = discover_skill_catalog(tmp_path, user_home=home)
    pool = assemble_tool_pool(
        ToolPoolContext(workdir=str(tmp_path), metadata={"skill_catalog": catalog})
    )
    runtime = ToolExecutionRuntime(pool)

    unknown_messages, _ = runtime.execute_tool_uses(
        [_tool_use(inp={"skill": "missing"}, tid="unknown")]
    )
    disabled_messages, _ = runtime.execute_tool_uses(
        [_tool_use(inp={"skill": "disabled"}, tid="disabled")]
    )

    assert unknown_messages[0]["is_error"] is True
    assert "unknown skill: missing" in unknown_messages[0]["content"]
    assert disabled_messages[0]["is_error"] is True
    assert "skill is disabled for model invocation: disabled" in disabled_messages[0]["content"]


def test_disabled_only_catalog_does_not_add_skill_tool(tmp_path):
    home = tmp_path / "home"
    _write_project_skill(
        tmp_path,
        "disabled",
        """---
description: Disabled skill.
disable-model-invocation: true
---
Disabled body.
""",
    )
    catalog = discover_skill_catalog(tmp_path, user_home=home)
    pool = assemble_tool_pool(
        ToolPoolContext(workdir=str(tmp_path), metadata={"skill_catalog": catalog})
    )

    assert "Skill" not in [tool.name for tool in pool.tools]


def test_metadata_only_additional_messages_are_not_durable(monkeypatch):
    from agent import loop, llm

    sideband_secret = "METADATA_ONLY_SENTINEL"
    tool = Tool(
        name="sideband",
        description="sideband tool",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        call=lambda tool_input, context: ToolResult(
            content="sideband output",
            additional_messages=(
                {"type": "metadata", "metadata": {"secret": sideband_secret}},
            ),
        ),
    )
    calls = []

    def fake_chat(messages, system="", tools=None, max_tokens=4096, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return tool_use_resp("sideband", {"text": "go"}, "sideband-call")
        return end_turn_resp("done")

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(loop, "assemble_tool_pool", lambda context=None: ToolPool((tool,)))
    monkeypatch.setattr(loop.ProjectInstructionsLoader, "load", lambda self, workdir: None)

    text, durable = loop.run_task("q", max_turns=3, trace=False, return_messages=True)

    assert text == "done"
    assert sideband_secret not in str(calls[1])
    assert sideband_secret not in str(durable)
