#!/usr/bin/env python3
"""Run one coding task through the complete runtime and save its trace."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", help="task to run")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--max-turns", type=_positive_int, default=20)
    parser.add_argument(
        "--compact-strategy",
        choices=("none", "micro", "full", "pipeline", "session_memory", "truncate"),
        default="none",
        help="programmatic context-compression strategy",
    )
    parser.add_argument("--compact-window", type=_positive_int)
    parser.add_argument("--compact-threshold", type=_positive_int)
    parser.add_argument(
        "--no-skills",
        action="store_true",
        help="disable skill discovery for this evaluation-style run",
    )
    mcp = parser.add_mutually_exclusive_group()
    mcp.add_argument(
        "--mcp-config",
        type=Path,
        help="MCP JSON config, resolved from the invocation directory",
    )
    mcp.add_argument(
        "--no-mcp",
        action="store_true",
        help="disable MCP environment settings and project auto-discovery",
    )
    parser.add_argument(
        "--html",
        type=Path,
        help="optional HTML trace path; defaults to <TRACES_DIR>/latest_run.html",
    )
    return parser


def resolve_run_paths(
    workdir: str | Path,
    mcp_config: str | Path | None,
    *,
    invocation_workdir: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Resolve explicit paths without replacing a requested nested workdir."""

    invocation = Path(invocation_workdir or Path.cwd()).expanduser().resolve()
    requested = Path(workdir).expanduser()
    if not requested.is_absolute():
        requested = invocation / requested
    resolved_workdir = requested.resolve()
    if not resolved_workdir.is_dir():
        raise NotADirectoryError(resolved_workdir)

    if mcp_config is None:
        return resolved_workdir, None
    config_path = Path(mcp_config).expanduser()
    if not config_path.is_absolute():
        config_path = invocation / config_path
    return resolved_workdir, config_path.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from agent import config
        from agent.loop import EvalHooks, run_task
        from agent.mcp.runtime_config import UNSET, resolve_run_task_runtime_kwargs
        from obs.otel import init_otel
        from obs.trace import get_sink, render_tree
        from obs.viewer import to_html

        invocation = Path.cwd().resolve()
        workdir, explicit_config = resolve_run_paths(
            args.workdir,
            args.mcp_config,
            invocation_workdir=invocation,
        )
        runtime_options = resolve_run_task_runtime_kwargs(
            mcp_config_path=(explicit_config if explicit_config is not None else UNSET),
            disable_mcp=args.no_mcp,
            workdir=workdir,
        )
        hooks = EvalHooks(
            compact_strategy=args.compact_strategy,
            compact_window=args.compact_window,
            compact_threshold=args.compact_threshold,
            skills_enabled=not args.no_skills,
        )
        if os.getenv("OTEL_EXPORT"):
            init_otel(
                endpoint=os.getenv("OTEL_ENDPOINT")
                or "http://localhost:6006/v1/traces"
            )

        with config.using_workdir(workdir):
            answer = run_task(
                args.task,
                max_turns=args.max_turns,
                trace=True,
                eval_hooks=hooks,
                **runtime_options,
            )

        sink = get_sink()
        events = sink.events()
        print(answer or "")
        print(render_tree(events), file=sys.stderr)
        html_path = (args.html or (config.TRACES_DIR / "latest_run.html")).resolve()
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(to_html(events, args.task[:80]), encoding="utf-8")
        trace_path = getattr(sink, "path", None)
        if trace_path is not None:
            print(f"Trace JSONL: {trace_path}", file=sys.stderr)
        print(f"Trace HTML: {html_path}", file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
