from pathlib import Path


def test_resolve_run_paths_preserves_nested_workdir_and_invocation_config(tmp_path):
    from scripts.run_task import resolve_run_paths

    invocation = tmp_path / "launcher"
    requested = tmp_path / "repo" / "src" / "nested"
    invocation.mkdir()
    requested.mkdir(parents=True)

    workdir, config_path = resolve_run_paths(
        requested,
        Path("configs/mcp.json"),
        invocation_workdir=invocation,
    )

    assert workdir == requested.resolve()
    assert config_path == (invocation / "configs" / "mcp.json").resolve()


def test_single_task_parser_exposes_real_runtime_controls():
    from scripts.run_task import build_parser

    args = build_parser().parse_args(
        [
            "inspect the project",
            "--compact-strategy",
            "pipeline",
            "--compact-threshold",
            "12000",
            "--no-skills",
            "--no-mcp",
        ]
    )

    assert args.compact_strategy == "pipeline"
    assert args.compact_threshold == 12000
    assert args.no_skills is True
    assert args.no_mcp is True
