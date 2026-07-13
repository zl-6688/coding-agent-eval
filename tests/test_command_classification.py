from agent.tools.executors import command_kind


def test_command_kind_recognizes_generic_test_commands_through_shell_wrappers():
    cases = [
        "cd frontend && npm test -- --runInBand",
        "cd app && python manage.py test polls | tail -20",
        "git stash && python -m pytest tests/test_bug.py -q",
        "env RUST_BACKTRACE=1 cargo test parser::tests",
        "make test",
        "go test ./...",
        "pnpm run test:unit",
    ]

    for command in cases:
        assert command_kind(command) == "test"


def test_command_kind_keeps_project_specific_runners_out_of_generic_agent_layer():
    assert command_kind("cd /testbed && python tests/runtests.py expressions | grep OK") == "grep"
    assert command_kind("cd /testbed && python bin/test sympy/core/tests/test_basic.py") == "custom_script"


def test_command_kind_does_not_treat_grep_over_test_files_as_validation():
    assert command_kind('cd /testbed && grep -n "INSTALLED_APPS" tests/runtests.py | head -5') == "grep"
