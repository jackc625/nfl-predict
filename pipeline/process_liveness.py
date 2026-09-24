"""Is a recorded pipeline process still running? (33.2 review C1 CR-05 = B WR-05)

The execution log records the PID of the run that wrote it. A run killed mid-flight (a reboot,
sleep, a Task Scheduler stop, a ``SystemExit`` that escaped every handler) leaves that log saying
``running`` forever, and the staleness gate used to refuse every later run on the word alone.
This module answers the one question that separates a live run from an abandoned one: is that
process still there, and is it the SAME process?

PID REUSE IS CHECKED, NOT ASSUMED AWAY. Windows recycles PIDs, so "some process has this PID" is
not "the run is alive". On Windows the process's CREATION time is read too: a process created
after the logged run started is a newer process that happens to hold a recycled PID, so the
logged run is gone. Elsewhere only existence is checked (``os.kill(pid, 0)``); the gate's age
limit is the backstop there.

WHY NOT ``os.kill(pid, 0)`` ON WINDOWS: there, signal 0 is ``CTRL_C_EVENT``. Probing a PID that
way would INTERRUPT the process being asked about.

No third-party dependency: ``psutil`` is not in the project, and two Win32 calls do not justify
adding it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime, timedelta

__all__ = ["process_is_alive"]

#: Clock slack when comparing a process's creation time with the run's logged start: the log's
#: start is taken a moment AFTER the process was created, so only a creation clearly later than
#: the start marks a recycled PID.
_CREATION_SLACK = timedelta(seconds=2)

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_FILETIME_EPOCH = datetime(1601, 1, 1, tzinfo=UTC)


def process_is_alive(pid: int, *, started_at: datetime | None = None) -> bool:
    """True when process *pid* is running and (where checkable) is the run that started then.

    Args:
        pid: The process id the execution log recorded.
        started_at: When the logged run started (tz-aware). When given, a process with this PID
            created clearly AFTER it is a recycled PID, and the logged run is reported gone.

    Returns:
        ``False`` when no such process exists, when it has exited, or when the PID now belongs to
        a newer process. ``True`` otherwise -- including when the process exists but cannot be
        inspected (access denied), because treating an unreadable live run as dead would let
        two runs write at once.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _windows_process_is_alive(pid, started_at)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_process_is_alive(pid: int, started_at: datetime | None) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
        ctypes.POINTER(wintypes.FILETIME)
    ] * 4
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # No such process (ERROR_INVALID_PARAMETER), or one we may not open. Only the latter
        # proves something is running.
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        if exit_code.value != _STILL_ACTIVE:
            return False
        if started_at is None:
            return True
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return True
        ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
        creation = _FILETIME_EPOCH + timedelta(microseconds=ticks // 10)
        return creation <= started_at + _CREATION_SLACK
    finally:
        kernel32.CloseHandle(handle)
