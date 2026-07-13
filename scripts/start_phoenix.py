#!/usr/bin/env python3
"""Start a Phoenix trace UI using an executable available on PATH."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable",
        default=os.environ.get("PHOENIX_BIN", "phoenix"),
        help="Phoenix CLI name or path (default: PHOENIX_BIN or phoenix)",
    )
    parser.add_argument("--host", help="optional host passed to `phoenix serve`")
    parser.add_argument("--port", type=int, help="optional port passed to `phoenix serve`")
    parser.add_argument("--dry-run", action="store_true", help="print the command without starting Phoenix")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    executable = shutil.which(args.executable)
    if executable is None:
        print(
            f"Phoenix CLI not found: {args.executable!r}. "
            "Install Arize Phoenix in the active environment or pass --executable."
        )
        return 2
    command = [executable, "serve"]
    if args.host:
        command.extend(["--host", args.host])
    if args.port is not None:
        command.extend(["--port", str(args.port)])
    print("Starting:", shlex.join(command))
    if args.dry_run:
        return 0
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        print("Phoenix stopped.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
