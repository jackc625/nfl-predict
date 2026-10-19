"""The ONE publish lock: the website push and every web-cache rebuild hold it (Phase 36, D-18).

WHY ONE LOCK
------------
The push reads the cache's data format, stamps a copy of the cache, takes the file's digest and
uploads it. A rebuild replaces that same file wholesale (``api.cache.populate_cache`` builds a temp
file and renames it over the live one). If the two overlapped, the push could stamp, digest or
upload a cache that is being swapped out underneath it. So both hold this one exclusive OS lock,
and the manual push refuses at once while either a push or a rebuild is running (SPEC R11).

WHY AT THE OUTER ENTRY POINTS (research Pitfall 1)
--------------------------------------------------
``utils.file_lock.exclusive_file_lock`` blocks every other handle, INCLUDING a second handle in the
same process. The push's D-03 self-heal rebuilds the cache while it already holds the lock, so the
rebuild it calls must not take the lock again. The lock is therefore taken by the outer callers --
``pipeline.steps.step_populate_web_cache`` (both registries and the settle-day rebuild go through
it), ``scripts/populate_cache.py`` and ``pipeline.site_push.push_site`` -- and the push's self-heal
calls the unlocked ``pipeline.steps.populate_web_cache_unlocked``.

The lock is a dedicated file beside the logs, never the cache itself, so the cache stays
replaceable. :data:`CACHE_PUBLISH_LOCK` is read at call time, so the test suite can point it at a
per-test path.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

from utils.file_lock import exclusive_file_lock

__all__ = ["CACHE_PUBLISH_LOCK", "REBUILD_LOCK_WAIT_SECONDS", "cache_publish_lock"]

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The lock file. Read at call time (``tests/conftest.py`` quarantines it per test).
CACHE_PUBLISH_LOCK: Path = REPO_ROOT / "logs" / ".cache_publish.lock"

# TRADE-OFF (review 36-04/36-11 MEDIUM), accepted rather than lengthening the wait: a rebuild waits
# up to 600 s for the lock, but a push can hold it longer (a deploy bounded at 900 s, a read-back of
# about 130 s and a 45 s self-heal). So a MANUAL push overlapping the 17:00 ET rebuild can make that
# rebuild time out; the run is then degraded and its own push refuses ``rebuild_incomplete`` (one
# "FAILED: website push"). The RUNBOOK says not to push by hand between 16:30 and 18:15 ET.
REBUILD_LOCK_WAIT_SECONDS = 600.0


@contextlib.contextmanager
def cache_publish_lock(*, wait_seconds: float) -> Iterator[Path]:
    """Hold the publish lock for the duration of the block.

    Args:
        wait_seconds: How long to keep retrying a held lock; ``0.0`` fails fast.

    Yields:
        The lock file's path.

    Raises:
        utils.file_lock.FileLockHeldError: a push or a rebuild held the lock for *wait_seconds*.
    """
    lock_path = CACHE_PUBLISH_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(lock_path, wait_seconds=wait_seconds) as held:
        yield held
