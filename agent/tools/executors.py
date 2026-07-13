"""Tool execution backends and mutable per-run executor state."""

from __future__ import annotations

import base64
import glob as _glob
import hashlib
import json as _json
import os
import posixpath
import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, Callable

from agent import config

from .command_classification import command_kind
from .file_state import FileReadSnapshot
from .path_access import AuthorizedPath, FileAccessPolicy


_DANGEROUS = ["rm -rf /", "sudo ", "shutdown", "reboot", "> /dev/", "mkfs", ":(){"]
_BASH_SEEN: set[str] = set()
_TOOL_ERRORS = [0]
_APPROVE_CB: Callable[[str, dict[str, Any]], bool] | None = None


def reset_bash_history() -> None:
    _BASH_SEEN.clear()
    _TOOL_ERRORS[0] = 0


def tool_error_count() -> int:
    return _TOOL_ERRORS[0]


def increment_tool_error_count() -> None:
    _TOOL_ERRORS[0] += 1


def seen_before(cmd: str) -> bool:
    norm = " ".join(cmd.split())
    seen = norm in _BASH_SEEN
    _BASH_SEEN.add(norm)
    return seen


def safe_path(p: str) -> Path:
    policy = FileAccessPolicy.create(config.WORKDIR)
    return policy.authorize(p, "metadata").resolved


class LocalExecutor:
    kind = "local"
    default_timeout = 120

    def __init__(
        self,
        *,
        workdir: str | Path | None = None,
        file_access: FileAccessPolicy | None = None,
    ) -> None:
        self._workdir = Path(workdir) if workdir is not None else None
        self._file_access = file_access

    def with_file_access(
        self,
        *,
        read_roots=(),
        write_roots=(),
        secret_scan_roots=(),
    ) -> "LocalExecutor":
        """Return a run-scoped executor; never mutate the process-global one."""

        policy = FileAccessPolicy.create(
            self._workdir_path,
            read_roots=read_roots,
            write_roots=write_roots,
            secret_scan_roots=secret_scan_roots,
        )
        return LocalExecutor(workdir=policy.workdir.lexical, file_access=policy)

    @property
    def _workdir_path(self) -> Path:
        selected = self._workdir if self._workdir is not None else config.WORKDIR
        return config.ensure_workdir(selected)

    @property
    def _access(self) -> FileAccessPolicy:
        return self._file_access or FileAccessPolicy.create(self._workdir_path)

    @property
    def cwd(self) -> str:
        return str(self._workdir_path)

    @property
    def host_cwd(self) -> str:
        return str(self._workdir_path)

    def exec_shell(self, cmd: str, timeout: int = 120):
        r = subprocess.run(
            cmd,
            shell=True,
            cwd=self._workdir_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return r.stdout or "", r.stderr or "", r.returncode

    def exec_powershell(self, cmd: str, timeout: int = 120):
        exe = shutil.which("pwsh") or shutil.which("powershell")
        if not exe:
            raise FileNotFoundError("PowerShell executable not found")
        r = subprocess.run(
            [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", cmd],
            cwd=self._workdir_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return r.stdout or "", r.stderr or "", r.returncode

    def file_snapshot(self, path: str) -> FileReadSnapshot:
        authorized = self._access.authorize(path, "metadata")
        fp = authorized.resolved
        self._assert_hardlink_access_safe(authorized, operation="read")
        if not fp.exists():
            return FileReadSnapshot(path=str(fp), exists=False)
        st = fp.stat()
        h = hashlib.sha256()
        with fp.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return FileReadSnapshot(
            path=str(fp),
            exists=True,
            mtime_ns=st.st_mtime_ns,
            size=st.st_size,
            content_hash=h.hexdigest(),
        )

    def grep_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        glob_pattern: str | None = None,
        case_insensitive: bool = False,
        line_numbers: bool = True,
        timeout: int = 120,
    ):
        rg = self._trusted_ripgrep_path()
        authorized = self._access.authorize(path or ".", "read")
        self._assert_additional_grep_target_safe(authorized)
        # Host ripgrep config is ambient authority. In particular `--follow`
        # would invalidate the executor's symlink/junction boundary.
        cmd = [rg, "--no-config", "--color", "never"]
        if line_numbers:
            cmd.append("--line-number")
        if case_insensitive:
            cmd.append("--ignore-case")
        if glob_pattern:
            cmd.extend(["--glob", glob_pattern])
        # Execute the authorized target, never the raw path.  A raw relative
        # path such as `link/../secret` is not equivalent to lexical normpath
        # when `link` is a symlink. Keep canonical WORKDIR targets relative so
        # rg's `file:line:text` output avoids a Windows drive-colon.
        try:
            relative_target = authorized.resolved.relative_to(
                self._access.workdir.resolved
            )
        except ValueError:
            search_path = str(authorized.resolved)
        else:
            search_path = str(relative_target) or "."
        cmd.extend(["--", pattern, search_path])
        r = subprocess.run(
            cmd,
            cwd=self._workdir_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return r.stdout or "", r.stderr or "", r.returncode

    def _trusted_ripgrep_path(self) -> str:
        discovered = shutil.which("rg")
        if not discovered:
            raise FileNotFoundError("ripgrep (rg) is not installed")
        candidate = Path(discovered)
        if not candidate.is_absolute():
            raise ValueError(
                "untrusted ripgrep executable: PATH resolved inside the "
                "current directory"
            )
        resolved = candidate.resolve(strict=False)
        if _path_is_within(resolved, self._access.workdir.resolved):
            raise ValueError(
                "untrusted ripgrep executable: repository-local binaries "
                "are not allowed"
            )
        return str(resolved)

    def read_file_raw(self, path: str) -> str:
        authorized = self._access.authorize(path, "read")
        fp = authorized.resolved
        self._assert_hardlink_access_safe(authorized, operation="read")
        return fp.read_text(encoding="utf-8", errors="replace")

    def write_file_raw(self, path: str, content: str) -> int:
        authorized = self._access.authorize(path, "write")
        if authorized.requires_secret_scan:
            from agent.memory.secret_scan import scan

            hits = scan(content)
            if hits:
                raise ValueError(
                    "memory write blocked by secret scan: " + ", ".join(hits)
                )
        fp = authorized.resolved
        self._assert_hardlink_access_safe(authorized, operation="write")
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return len(content)

    def glob_files(self, pattern: str):
        self._access.glob_search_root(pattern)
        # glob must execute the same lexical normalization that policy
        # authorization inspected. Otherwise `symlink/../*` can be authorized
        # as a WORKDIR path but enumerated outside by the OS.
        normalized_pattern = os.path.normpath(pattern)
        matches = _glob.glob(normalized_pattern, root_dir=self._workdir_path)
        return [
            match
            for match in matches
            if self._glob_match_is_authorized(match)
        ]

    def _glob_match_is_authorized(self, match: str) -> bool:
        try:
            self._access.authorize(match, "read")
        except ValueError:
            return False
        return True

    def _assert_additional_grep_target_safe(
        self, authorized: AuthorizedPath
    ) -> None:
        """Preflight an added-root tree before handing recursive reads to rg.

        Canonical containment alone cannot detect an in-root hardlink whose
        inode also has an out-of-root name.  Added roots are small durable
        stores, so fail closed before rg reads any candidate.  WORKDIR keeps
        its existing behavior, and the ordinary local-race caveat still
        applies between this preflight and rg execution.
        """

        if not authorized.rejects_hardlinked_reads:
            self._assert_unprotected_grep_target_safe(authorized)
            return
        try:
            root_stat = authorized.resolved.stat()
        except FileNotFoundError:
            return
        if stat.S_ISREG(root_stat.st_mode):
            _reject_hardlinked_regular_file(
                authorized.resolved, operation="read"
            )
            return
        if not stat.S_ISDIR(root_stat.st_mode):
            return

        pending = [authorized.lexical]
        seen_directories: set[str] = set()
        while pending:
            lexical_dir = pending.pop()
            current = self._access.authorize(lexical_dir, "read")
            directory_key = os.path.normcase(os.fspath(current.resolved))
            if directory_key in seen_directories:
                continue
            seen_directories.add(directory_key)
            with os.scandir(lexical_dir) as entries:
                for entry in entries:
                    child = self._access.authorize(entry.path, "read")
                    try:
                        child_stat = child.resolved.stat()
                    except FileNotFoundError as exc:
                        raise ValueError(
                            "grep target changed during additional-root preflight"
                        ) from exc
                    if stat.S_ISREG(child_stat.st_mode):
                        _reject_hardlinked_regular_file(
                            child.resolved, operation="read"
                        )
                    elif stat.S_ISDIR(child_stat.st_mode):
                        pending.append(Path(entry.path))

    def _assert_hardlink_access_safe(
        self, authorized: AuthorizedPath, *, operation: str
    ) -> None:
        file_stat = _hardlinked_regular_stat(authorized.resolved)
        if file_stat is None:
            return

        directly_protected = (
            authorized.rejects_hardlinked_reads
            if operation == "read"
            else (
                authorized.rejects_hardlinked_writes
                or authorized.requires_secret_scan
            )
        )
        if directly_protected:
            raise ValueError(
                f"{operation} through a hardlink is not allowed in an "
                "additional file root"
            )

        protected_grants = self._access.protected_grants()
        if not protected_grants:
            return
        if not file_stat.st_ino:
            raise ValueError(
                f"{operation} through a hardlink is not allowed while "
                "additional protected roots are active"
            )
        identity = (file_stat.st_dev, file_stat.st_ino)
        if self._protected_roots_contain_identity(protected_grants, identity):
            raise ValueError(
                f"{operation} through a hardlink alias to an additional "
                "protected root is not allowed"
            )

    def _protected_roots_contain_identity(
        self,
        grants,
        identity: tuple[int, int],
    ) -> bool:
        """Find reverse aliases without weakening ordinary WORKDIR hardlinks."""

        return identity in self._protected_hardlink_identities(grants)

    def _protected_hardlink_identities(self, grants) -> set[tuple[int, int]]:
        """Collect multi-link file identities from small protected roots."""

        identities: set[tuple[int, int]] = set()
        seen_directories: set[str] = set()
        for grant in grants:
            pending = [grant.lexical]
            while pending:
                lexical_path = pending.pop()
                current = self._access.authorize(lexical_path, "metadata")
                try:
                    current_stat = current.resolved.stat()
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(current_stat.st_mode):
                    if current_stat.st_nlink > 1:
                        identities.add((current_stat.st_dev, current_stat.st_ino))
                    continue
                if not stat.S_ISDIR(current_stat.st_mode):
                    continue

                directory_key = os.path.normcase(os.fspath(current.resolved))
                if directory_key in seen_directories:
                    continue
                seen_directories.add(directory_key)
                with os.scandir(lexical_path) as entries:
                    for entry in entries:
                        child = self._access.authorize(entry.path, "metadata")
                        try:
                            child_stat = child.resolved.stat()
                        except FileNotFoundError as exc:
                            raise ValueError(
                                "protected root changed during hardlink preflight"
                            ) from exc
                        if stat.S_ISREG(child_stat.st_mode):
                            if child_stat.st_nlink > 1:
                                identities.add(
                                    (child_stat.st_dev, child_stat.st_ino)
                                )
                        elif stat.S_ISDIR(child_stat.st_mode):
                            pending.append(Path(entry.path))
        return identities

    def _assert_unprotected_grep_target_safe(
        self, authorized: AuthorizedPath
    ) -> None:
        """Block WORKDIR Grep from exposing a protected-root inode alias."""

        protected_identities = self._protected_hardlink_identities(
            self._access.protected_grants()
        )
        if not protected_identities:
            return
        try:
            root_stat = authorized.resolved.stat()
        except FileNotFoundError:
            return
        if stat.S_ISREG(root_stat.st_mode):
            if (
                root_stat.st_nlink > 1
                and (root_stat.st_dev, root_stat.st_ino)
                in protected_identities
            ):
                raise ValueError(
                    "grep through a hardlink alias to an additional "
                    "protected root is not allowed"
                )
            return
        if not stat.S_ISDIR(root_stat.st_mode):
            return

        pending = [authorized.resolved]
        seen_directories: set[str] = set()
        while pending:
            directory = pending.pop()
            directory_key = os.path.normcase(os.fspath(directory.resolve()))
            if directory_key in seen_directories:
                continue
            seen_directories.add(directory_key)
            with os.scandir(directory) as entries:
                for entry in entries:
                    entry_path = Path(entry.path)
                    if entry.is_symlink() or _is_junction(entry_path):
                        continue
                    try:
                        # Windows DirEntry.stat(follow_symlinks=False) can
                        # report zeroed inode/link metadata. Directory aliases
                        # were excluded above, so Path.stat is safe here.
                        entry_stat = entry_path.stat()
                    except FileNotFoundError as exc:
                        raise ValueError(
                            "grep target changed during hardlink preflight"
                        ) from exc
                    if stat.S_ISREG(entry_stat.st_mode):
                        if (
                            entry_stat.st_nlink > 1
                            and (entry_stat.st_dev, entry_stat.st_ino)
                            in protected_identities
                        ):
                            raise ValueError(
                                "grep through a hardlink alias to an additional "
                                "protected root is not allowed"
                            )
                    elif stat.S_ISDIR(entry_stat.st_mode):
                        pending.append(entry_path)


def _reject_hardlinked_regular_file(path: Path, *, operation: str) -> None:
    if _hardlinked_regular_stat(path) is not None:
        raise ValueError(
            f"{operation} through a hardlink is not allowed in an "
            "additional file root"
        )


def _hardlinked_regular_stat(path: Path):
    try:
        file_stat = path.stat()
    except FileNotFoundError:
        return None
    if stat.S_ISREG(file_stat.st_mode) and file_stat.st_nlink > 1:
        return file_stat
    return None


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker()) if callable(checker) else False


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            [os.path.normcase(os.fspath(path)), os.path.normcase(os.fspath(root))]
        ) == os.path.normcase(os.fspath(root))
    except ValueError:
        return False


class DockerExecutor:
    kind = "docker"
    default_timeout = 300

    def __init__(
        self,
        container: str,
        workdir: str = "/testbed",
        conda: str = "testbed",
        host_cwd: str | Path | None = None,
    ):
        self.container = container
        self.workdir = workdir
        self._host_cwd = str(host_cwd) if host_cwd is not None else None
        self._prefix = (
            "set -o pipefail; "
            f"source /opt/miniconda3/bin/activate {shlex.quote(conda)} 2>/dev/null; "
            f"cd {shlex.quote(workdir)} && "
        )

    @property
    def cwd(self) -> str:
        return f"{self.workdir} @docker:{self.container[:12]}"

    @property
    def host_cwd(self) -> str | None:
        return self._host_cwd

    def _exec(self, bash_cmd: str, timeout: int = 120, *, stdin: bytes | None = None):
        args = ["docker", "exec"]
        if stdin is not None:
            args.append("-i")
        args.extend([self.container, "bash", "-lc", bash_cmd])
        run_kwargs: dict[str, Any] = {
            "capture_output": True,
            "timeout": timeout,
        }
        if stdin is not None:
            run_kwargs["input"] = stdin
        else:
            run_kwargs.update(
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        r = subprocess.run(args, **run_kwargs)
        stdout = r.stdout or ""
        stderr = r.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return stdout, stderr, r.returncode

    def _abs(self, path: str) -> str:
        return path if path.startswith("/") else posixpath.join(self.workdir, path)

    def exec_shell(self, cmd: str, timeout: int = 120):
        return self._exec(self._prefix + cmd, timeout)

    def exec_powershell(self, cmd: str, timeout: int = 120):
        encoded = base64.b64encode(cmd.encode("utf-16le")).decode("ascii")
        ps = "pwsh"
        _check, _err, rc = self._exec("command -v pwsh >/dev/null 2>&1")
        if rc != 0:
            _check, _err, rc = self._exec("command -v powershell >/dev/null 2>&1")
            ps = "powershell"
        if rc != 0:
            raise FileNotFoundError("PowerShell executable not found in container")
        command = (
            self._prefix
            + f"{ps} -NoProfile -NonInteractive -EncodedCommand {shlex.quote(encoded)}"
        )
        return self._exec(command, timeout)

    def file_snapshot(self, path: str) -> FileReadSnapshot:
        ap = self._abs(path)
        code = (
            "import hashlib,json,os,sys;"
            "p=sys.argv[1];"
            "ap=os.path.abspath(p);"
            "\nif not os.path.exists(ap):\n"
            " print(json.dumps({'path':ap,'exists':False})); sys.exit(0)\n"
            "st=os.stat(ap); h=hashlib.sha256();\n"
            "f=open(ap,'rb')\n"
            "try:\n"
            "  [h.update(chunk) for chunk in iter(lambda:f.read(1048576), b'')]\n"
            "finally:\n"
            "  f.close()\n"
            "print(json.dumps({'path':ap,'exists':True,'mtime_ns':getattr(st,'st_mtime_ns',int(st.st_mtime*1e9)),'size':st.st_size,'content_hash':h.hexdigest()}))"
        )
        out, err, rc = self._exec(f"python -c {shlex.quote(code)} {shlex.quote(ap)}")
        if rc != 0:
            raise OSError(err.strip() or "file snapshot failed")
        return FileReadSnapshot.from_mapping(_json.loads(out.strip()))

    def grep_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        glob_pattern: str | None = None,
        case_insensitive: bool = False,
        line_numbers: bool = True,
        timeout: int = 120,
    ):
        ap = self._abs(path)
        cmd = ["rg", "--color", "never"]
        if line_numbers:
            cmd.append("--line-number")
        if case_insensitive:
            cmd.append("--ignore-case")
        if glob_pattern:
            cmd.extend(["--glob", glob_pattern])
        cmd.extend(["--", pattern, ap])
        shell_cmd = self._prefix + " ".join(shlex.quote(part) for part in cmd)
        stdout, stderr, rc = self._exec(shell_cmd, timeout)
        missing_rg = rc == 127 and "not found" in (stdout + stderr).lower()
        if missing_rg:
            return self._grep_files_python(
                pattern,
                path=path,
                glob_pattern=glob_pattern,
                case_insensitive=case_insensitive,
                line_numbers=line_numbers,
                timeout=timeout,
            )
        return stdout, stderr, rc

    def _grep_files_python(
        self,
        pattern: str,
        *,
        path: str = ".",
        glob_pattern: str | None = None,
        case_insensitive: bool = False,
        line_numbers: bool = True,
        timeout: int = 120,
    ):
        ap = self._abs(path)
        script = r"""
import fnmatch
import os
import re
import sys

pattern, root, glob_pattern, case_insensitive, line_numbers = sys.argv[1:6]
flags = re.IGNORECASE if case_insensitive == "1" else 0
try:
    regex = re.compile(pattern, flags)
except re.error as exc:
    print(f"regex error: {exc}", file=sys.stderr)
    raise SystemExit(2)

cwd = os.getcwd()
root = os.path.abspath(root)
skip_dirs = {".git", ".hg", ".svn", "__pycache__", ".tox", ".venv", "venv", "node_modules"}

def display_path(file_path):
    try:
        return os.path.relpath(file_path, cwd).replace(os.sep, "/")
    except ValueError:
        return file_path.replace(os.sep, "/")

def glob_matches(file_path):
    if not glob_pattern:
        return True
    rel = display_path(file_path)
    return fnmatch.fnmatch(rel, glob_pattern) or fnmatch.fnmatch(os.path.basename(file_path), glob_pattern)

def iter_files(root_path):
    if os.path.isfile(root_path):
        yield root_path
        return
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [name for name in dirnames if name not in skip_dirs]
        for filename in filenames:
            yield os.path.join(dirpath, filename)

matched = False
for file_path in iter_files(root):
    if not glob_matches(file_path):
        continue
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                if not regex.search(line):
                    continue
                matched = True
                text = line.rstrip("\n")
                if line_numbers == "1":
                    print(f"{display_path(file_path)}:{lineno}:{text}")
                else:
                    print(f"{display_path(file_path)}:{text}")
    except OSError:
        continue

raise SystemExit(0 if matched else 1)
""".strip()
        args = [
            pattern,
            ap,
            glob_pattern or "",
            "1" if case_insensitive else "0",
            "1" if line_numbers else "0",
        ]
        shell_cmd = (
            self._prefix
            + f"python -c {shlex.quote(script)} "
            + " ".join(shlex.quote(arg) for arg in args)
        )
        return self._exec(shell_cmd, timeout)

    def read_file_raw(self, path: str) -> str:
        out, err, rc = self._exec(f"cat {shlex.quote(self._abs(path))}")
        if rc != 0:
            raise FileNotFoundError(err.strip() or path)
        return out

    def write_file_raw(self, path: str, content: str) -> int:
        ap = self._abs(path)
        cmd = (
            f"mkdir -p {shlex.quote(posixpath.dirname(ap) or '/')} && "
            f"cat > {shlex.quote(ap)}"
        )
        _out, err, rc = self._exec(cmd, stdin=content.encode("utf-8"))
        if rc != 0:
            raise OSError(err.strip() or "write failed")
        return len(content)

    def glob_files(self, pattern: str):
        code = f"import glob,json;print(json.dumps(glob.glob({pattern!r},recursive=True)))"
        out, _err, _rc = self.exec_shell(f"python -c {shlex.quote(code)}")
        try:
            return _json.loads(out.strip() or "[]")
        except Exception:
            return []


_EX = LocalExecutor()


def set_executor(ex) -> None:
    global _EX
    _EX = ex


def get_executor():
    return _EX


def bind_file_access(
    executor,
    *,
    read_roots=(),
    write_roots=(),
    secret_scan_roots=(),
):
    """Bind trusted roots when the backend can enforce host-path containment.

    Docker paths live in a different filesystem namespace, so a host Auto
    Memory root is deliberately not projected into that executor implicitly.
    """

    binder = getattr(executor, "with_file_access", None)
    if not callable(binder):
        raise ValueError(
            f"executor {type(executor).__name__} cannot bind trusted host file roots"
        )
    return binder(
        read_roots=read_roots,
        write_roots=write_roots,
        secret_scan_roots=secret_scan_roots,
    )


def bind_memory_file_access(executor, memory_root: str | Path):
    """Bind the shared Auto Memory capability used by main/background agents."""

    return bind_file_access(
        executor,
        read_roots=(memory_root,),
        write_roots=(memory_root,),
        secret_scan_roots=(memory_root,),
    )


def reset_executor() -> None:
    global _EX
    _EX = LocalExecutor()


def set_approve_cb(cb) -> None:
    global _APPROVE_CB
    _APPROVE_CB = cb


def get_approve_cb():
    return _APPROVE_CB


def reset_approve_cb() -> None:
    global _APPROVE_CB
    _APPROVE_CB = None


__all__ = [
    "bind_file_access",
    "bind_memory_file_access",
    "DockerExecutor",
    "LocalExecutor",
    "command_kind",
    "get_approve_cb",
    "get_executor",
    "increment_tool_error_count",
    "reset_approve_cb",
    "reset_bash_history",
    "reset_executor",
    "safe_path",
    "seen_before",
    "set_approve_cb",
    "set_executor",
    "tool_error_count",
]
