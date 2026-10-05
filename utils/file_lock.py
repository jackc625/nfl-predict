"""An exclusive OS byte-range lock on a dedicated lock file (Phase 34, D-13).

WHY AN OS LOCK AND NOT A PID FILE
---------------------------------
``msvcrt.locking`` places a byte-range lock the operating system owns: "If a process terminates
with a portion of a file locked or closes a file that has outstanding locks, the locks are unlocked
by the operating system" (Win32 LockFile / LockFileEx). A writer that crashes, is killed or loses
power therefore never leaves a stale lock behind, so there is no pid file, no liveness probe and no
"delete it if nothing is running" instruction (34-RESEARCH section G).

WHY A SEPARATE LOCK FILE
------------------------
The lock is taken on its OWN file, never on the data file it guards: a locked region blocks every
other handle, including a second handle in the same process, and the guarded file must stay
replaceable with ``os.replace``.

``msvcrt.locking`` locks from the CURRENT file position, so the position is set to 0 before both the
lock and the unlock. The region may extend past the end of the (empty) file, which Windows permits.

Windows-only: this project runs on Windows (the scheduled task is a Windows S4U task). On another
platform the lock refuses with a clear error rather than silently not locking.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator
from pathlib import Path

__all__ = ["FileLockHeldError", "exclusive_file_lock"]

_LOCKED_BYTES = 1


class FileLockHeldError(Exception):
    """Another holder has the lock, and it did not come free within the allowed wait."""


@contextlib.contextmanager
def exclusive_file_lock(
    lock_path: Path | str,
    *,
    wait_seconds: float = 0.0,
    poll_seconds: float = 0.25,
) -> Iterator[Path]:
    """Hold an exclusive lock on *lock_path* for the duration of the block.

    Args:
        lock_path: The dedicated lock file. Created if absent (its parent must exist).
        wait_seconds: How long to keep retrying a held lock. ``0.0`` fails fast.
        poll_seconds: The pause between retries.

    Yields:
        The lock file's path.

    Raises:
        FileLockHeldError: the lock is held by another handle and stayed held for *wait_seconds*.
        RuntimeError: the platform has no ``msvcrt`` (not Windows).
    """
    try:
        import msvcrt
    except ImportError as error:  # pragma: no cover - this project runs on Windows only
        msg = "exclusive_file_lock needs the Windows msvcrt module; this platform has none."
        raise RuntimeError(msg) from error

    path = Path(lock_path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, _LOCKED_BYTES)
                break
            except OSError as error:
                if time.monotonic() >= deadline:
                    msg = (
                        f"the lock {path.as_posix()} is held by another writer "
                        f"(waited {wait_seconds:g}s)."
                    )
                    raise FileLockHeldError(msg) from error
                time.sleep(poll_seconds)
        try:
            yield path
        finally:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, _LOCKED_BYTES)
    finally:
        os.close(fd)
