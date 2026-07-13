"""Per-run read state used to guard file overwrites and edits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping


class FileReadStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileReadSnapshot:
    path: str
    exists: bool
    mtime_ns: int | None = None
    size: int | None = None
    content_hash: str | None = None
    hash_only: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FileReadSnapshot":
        return cls(
            path=str(value.get("path") or ""),
            exists=bool(value.get("exists")),
            mtime_ns=value.get("mtime_ns"),
            size=value.get("size"),
            content_hash=value.get("content_hash"),
            hash_only=bool(value.get("hash_only", False)),
        )


@dataclass(frozen=True)
class FileReadRecord:
    snapshot: FileReadSnapshot
    complete: bool
    content: str
    visible_segments: tuple[str, ...]
    timestamp: int


@dataclass(frozen=True)
class FileEditAuthorization:
    record_path: str
    content: str


class FileReadState:
    """Tracks complete reads inside one run_task boundary."""

    def __init__(self) -> None:
        self._records: dict[str, FileReadRecord] = {}
        self._timestamp = 0

    @property
    def records(self) -> Mapping[str, FileReadRecord]:
        return dict(self._records)

    def reset(self) -> None:
        """Clear run-local read state after compact consumes its recovery data.

        The state is scoped to one active transcript.  A successful compact
        snapshots recent reads for the next request and must then forget them so
        stale pre-compact reads do not keep unlocking writes or restorations.
        """

        self._records.clear()
        self._timestamp = 0

    def recent_file_items(self) -> tuple[tuple[str, str], ...]:
        """Return recently read file bodies for post-compact restoration.

        Compaction can remove the original read_file output from durable
        history.  Returning the run-local read bodies lets compact restore that
        working context without depending on compact.py process globals.
        """

        return tuple(
            (record.snapshot.path, record.content)
            for record in sorted(
                self._records.values(),
                key=lambda record: record.timestamp,
            )
        )

    def record_read(
        self,
        path: str,
        content: str,
        *,
        complete: bool,
        visible_content: str | None = None,
        executor: Any,
    ) -> FileReadSnapshot:
        snapshot = _snapshot_for_path(executor, path, content=content)
        if not snapshot.exists:
            snapshot = FileReadSnapshot(
                path=snapshot.path or str(path),
                exists=True,
                content_hash=_hash_text(content),
                hash_only=True,
            )
        if snapshot.content_hash is None:
            snapshot = FileReadSnapshot(
                path=snapshot.path,
                exists=snapshot.exists,
                mtime_ns=snapshot.mtime_ns,
                size=snapshot.size,
                content_hash=_hash_text(content),
                hash_only=True,
            )

        previous = self._records.get(snapshot.path)
        effective_complete = bool(complete)
        visible_segments: tuple[str, ...]
        if previous is not None and previous.complete and _same_snapshot(
            previous.snapshot, snapshot
        ):
            effective_complete = True
            visible_segments = previous.visible_segments
        elif complete:
            visible_segments = (content,)
        elif previous is not None and _same_snapshot(previous.snapshot, snapshot):
            visible_segments = previous.visible_segments
            if visible_content and visible_content not in visible_segments:
                visible_segments = (*visible_segments, visible_content)
        else:
            visible_segments = (visible_content,) if visible_content else ()
        self._records.pop(snapshot.path, None)
        self._timestamp += 1
        self._records[snapshot.path] = FileReadRecord(
            snapshot=snapshot,
            complete=effective_complete,
            content=content,
            visible_segments=visible_segments,
            timestamp=self._timestamp,
        )
        return snapshot

    def record_write(self, path: str, content: str, *, executor: Any) -> FileReadSnapshot:
        return self.record_read(path, content, complete=True, executor=executor)

    def record_edit(
        self,
        path: str,
        content: str,
        *,
        record_path: str,
        old_text: str,
        new_text: str,
        executor: Any,
    ) -> FileReadSnapshot:
        snapshot = _snapshot_for_path(executor, path, content=content)
        if not snapshot.exists or snapshot.content_hash is None:
            snapshot = FileReadSnapshot(
                path=record_path,
                exists=True,
                content_hash=_hash_text(content),
                hash_only=True,
            )
        elif snapshot.path != record_path:
            snapshot = FileReadSnapshot(
                path=record_path,
                exists=snapshot.exists,
                mtime_ns=snapshot.mtime_ns,
                size=snapshot.size,
                content_hash=snapshot.content_hash,
                hash_only=snapshot.hash_only,
            )

        previous = self._records.get(record_path)
        if previous is None:
            raise FileReadStateError(
                f"File read state was lost after editing; read_file again: {path}"
            )

        if previous.complete:
            visible_segments = (content,)
        else:
            visible_segments = tuple(
                segment.replace(old_text, new_text, 1)
                if old_text in segment
                else segment
                for segment in previous.visible_segments
            )

        self._records.pop(record_path, None)
        self._timestamp += 1
        self._records[snapshot.path] = FileReadRecord(
            snapshot=snapshot,
            complete=previous.complete,
            content=content,
            visible_segments=visible_segments,
            timestamp=self._timestamp,
        )
        return snapshot

    def assert_can_write(self, path: str, *, executor: Any) -> FileReadSnapshot:
        current = _snapshot_for_path(executor, path)
        if not current.exists:
            return current

        record = self._records.get(current.path)
        if record is None:
            raise FileReadStateError(
                f"Existing file must be read with read_file before writing: {path}"
            )
        if not record.complete:
            raise FileReadStateError(
                f"File was only partially read; read the complete file before writing: {path}"
            )
        if not _same_snapshot(record.snapshot, current):
            raise FileReadStateError(
                f"File changed since last read; read_file again before writing: {path}"
            )
        return current

    def assert_can_edit(
        self,
        path: str,
        *,
        old_text: str,
        replace_all: bool,
        executor: Any,
    ) -> FileEditAuthorization:
        current = _snapshot_for_path(executor, path)
        record = self._records.get(current.path)
        if record is None:
            raise FileReadStateError(
                f"Existing file must be read with read_file before editing: {path}"
            )
        if not _same_snapshot(record.snapshot, current):
            raise FileReadStateError(
                f"File changed since last read; read_file again before editing: {path}"
            )
        if not record.complete:
            if replace_all:
                raise FileReadStateError(
                    "replace_all requires a complete read of the current file"
                )
            if not any(old_text in segment for segment in record.visible_segments):
                raise FileReadStateError(
                    f"old_text was not visible in the partial read of {path}; "
                    "read the lines containing it before editing"
                )
        text = executor.read_file_raw(path)
        count = text.count(old_text)
        if count == 0:
            raise FileReadStateError(f"old_text was not found in {path}")
        if count > 1 and not replace_all:
            raise FileReadStateError(
                "old_text appears multiple times; set replace_all=true to replace all matches"
            )
        return FileEditAuthorization(record_path=current.path, content=text)


_CURRENT = FileReadState()


def get_current_file_read_state() -> FileReadState:
    return _CURRENT


def reset_current_file_read_state() -> None:
    _CURRENT.reset()


def _snapshot_for_path(
    executor: Any,
    path: str,
    *,
    content: str | None = None,
) -> FileReadSnapshot:
    snapshot_func = getattr(executor, "file_snapshot", None)
    if callable(snapshot_func):
        try:
            return _coerce_snapshot(snapshot_func(path))
        except FileNotFoundError:
            return FileReadSnapshot(path=str(path), exists=False)
        except Exception:
            if content is not None:
                return FileReadSnapshot(
                    path=str(path),
                    exists=True,
                    content_hash=_hash_text(content),
                    hash_only=True,
                )

    try:
        raw = executor.read_file_raw(path)
    except FileNotFoundError:
        return FileReadSnapshot(path=str(path), exists=False)
    except Exception as exc:
        raise FileReadStateError(
            f"Could not verify file state for {path}: {type(exc).__name__}: {exc}"
        ) from exc
    return FileReadSnapshot(
        path=str(path),
        exists=True,
        content_hash=_hash_text(raw),
        hash_only=True,
    )


def _coerce_snapshot(value: Any) -> FileReadSnapshot:
    if isinstance(value, FileReadSnapshot):
        return value
    if isinstance(value, Mapping):
        return FileReadSnapshot.from_mapping(value)
    raise TypeError(f"unsupported file snapshot type: {type(value).__name__}")


def _same_snapshot(left: FileReadSnapshot, right: FileReadSnapshot) -> bool:
    if not left.exists or not right.exists:
        return left.exists == right.exists
    if not left.content_hash or not right.content_hash:
        return False
    if left.content_hash != right.content_hash:
        return False
    if left.hash_only or right.hash_only:
        return True
    return left.mtime_ns == right.mtime_ns and left.size == right.size


def content_matches_for_attachment(
    left: FileReadSnapshot,
    right: FileReadSnapshot,
) -> bool:
    """Return whether two snapshots have the same model-visible bytes.

    Request-only changed-file reminders should not fire when only metadata
    changed.  The stricter ``_same_snapshot`` remains the write/edit stale guard
    so mtime/size drift still forces a reread before mutation.
    """

    if not left.exists or not right.exists:
        return left.exists == right.exists
    if left.content_hash and right.content_hash:
        return left.content_hash == right.content_hash
    return _same_snapshot(left, right)


def snapshot_for_path(
    executor: Any,
    path: str,
    *,
    content: str | None = None,
) -> FileReadSnapshot:
    """Expose read-state snapshotting for request attachments.

    The stale-write guard and changed-files attachment must compare files with
    the same rules.  Keeping the implementation here prevents the attachment
    layer from inventing a second definition of "same file state".
    """

    return _snapshot_for_path(executor, path, content=content)


def snapshots_match(left: FileReadSnapshot, right: FileReadSnapshot) -> bool:
    """Return strict file-state equality used by write/edit stale guards."""

    return _same_snapshot(left, right)


def _hash_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


__all__ = [
    "FileReadRecord",
    "FileReadSnapshot",
    "FileReadState",
    "FileReadStateError",
    "content_matches_for_attachment",
    "get_current_file_read_state",
    "reset_current_file_read_state",
    "snapshot_for_path",
    "snapshots_match",
]
