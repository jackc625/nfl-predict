"""Content-based boundary guards for the production stores under ``data/`` and ``artifacts/``.

WHY THIS EXISTS
---------------
Every "the data/ hard boundary held" claim in this repository through Phase 30 and the
first ten plans of Phase 31 was made by running::

    git status --porcelain data/

That check is STRUCTURALLY INCAPABLE OF FAILING. ``.gitignore:22`` blankets ``data/``,
so git reports nothing whether the archive is intact, silently rewritten, or deleted.
An empty result was read as proof of an untouched boundary; it was proof of an ignore
rule. Plan 31-11's own plan text says as much twice (acceptance criterion "never by git
status on the data directory") and the check was still the one being run, because no
content-based instrument existed to run instead.

This module is that instrument. It hashes file CONTENT, so a write it did not expect is
a failure it can actually report, naming the file and the digest on both sides.

WHAT IT CAUGHT (had it existed four waves earlier)
--------------------------------------------------
``tests/integration/test_elo_integration.py`` called ``save_dataframe(..., layer=
"silver")`` with no sandbox, rebuilding Elo from 2018 and overwriting the production
``data/silver/elo_game_snapshots.parquet`` on EVERY full integration run. It fired at
least four times during Phase 31 -- the last at 2026-09-04 22:05:25, inside Plan 31-10's
post-merge gate -- and moved fourteen Elo columns in the 2025 slice of gold. Nothing
reported it, because the only boundary check in the phase was the vacuous git one.

``tests/integration/test_lift_validation.py`` was worse in kind: it retrained all three
targets into the PRODUCTION ``artifacts/`` directory and rewrote ``artifacts/latest.json``,
which is the deployed-model manifest. Its backup directory carries mtime 2026-09-04 22:06,
so it too ran during Phase 31; only the training subprocess failing kept the deployed
models from being swapped mid-phase.

USAGE
-----
As a pytest fixture (opt in per module -- see ``tests/conftest.py``)::

    def test_something(data_boundary_guard):     # digests data/ before and after
        ...

    def test_something_else(artifacts_boundary_guard):

As a helper inside a test that needs the two sides explicitly::

    before = digest_tree(PRODUCTION_DATA_ROOT)
    ...
    assert_tree_unchanged(before, digest_tree(PRODUCTION_DATA_ROOT), PRODUCTION_DATA_ROOT)

As a command-line instrument around a long run that pytest is not driving::

    python -m tests.data_boundary snapshot data outputs/before.json
    python -m tests.data_boundary verify   data outputs/before.json

NOT A REPLACEMENT FOR THE FINGERPRINT LADDER. A digest says a file changed; the ladder in
``scripts/fingerprint_gold.py`` says WHICH COLUMN in WHICH SEASON changed and why. This
guard is the cheap always-on tripwire that tells the ladder it has work to do.

A LOCKED PRODUCTION FILE NOW FAILS A NORMAL TEST RUN, BY NAME (D33-32)
----------------------------------------------------------------------
This is a NEW failure mode, added deliberately by Plan 33-01, and it is worth stating
plainly because it will be met before it is read about.

``digest_file`` still falls back to a ``stat-size-mtime:`` signature when a file cannot be
opened, and it still SAYS SO in the value it returns. That machinery is kept: the CLI
snapshot/verify path depends on it, and the self-declaration is what makes a mixed
comparison detectable at all.

What changed is what a VERDICT may rest on. A stat signature is an INABILITY TO PROVE
content integrity -- and it degrades exactly for the files the guard most needs to cover,
because ``data/nfl_predictions.duckdb`` is the store DuckDB holds open. So
``require_content_digest`` closes handles (``gc.collect()`` plus dropping the global
DuckDB connection), RETRIES, and if the bytes still cannot be read it RAISES
:class:`LockedStoreDigestError` naming the file, the underlying error and the likely
holder. The autouse write guard in ``tests/conftest.py`` and ``assert_tree_unchanged``
both route through it, so no comparison in this repository is ever settled by metadata.

If a run stops with that error, CLOSE THE HOLDER -- usually a DuckDB connection left open
by an earlier run, or an editor sitting in ``data/`` -- and re-run. The refusal is the
intended behaviour: an unprovable comparison is a refusal, not a pass.

AND AN UNDECIDED COMPARISON IS NEITHER (NF-02)
-----------------------------------------------
Where one side of a comparison is a content hash and the other is a stat signature, the
two values disagree as raw strings but say NOTHING about the bytes. ``diff_digests``
reports those keys as ``mixed`` rather than folding them into ``changed``, and
``format_digest_diff`` renders them as UNDECIDED. A mixed key is a HARD FAILURE: it is
never silently a data move, and never silently a pass.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import sys
from pathlib import Path

PRODUCTION_DATA_ROOT = Path("data")
PRODUCTION_ARTIFACTS_ROOT = Path("artifacts")

# Everything a production store is made of. Deliberately NOT a "parquet only" list:
# ``data/nfl_predictions.duckdb`` is the second half of every gold write, and
# ``artifacts/latest.json`` is the deployed-model manifest -- both are exactly the files
# an accidental write is most damaging to.
TRACKED_SUFFIXES = (
    ".parquet",
    ".duckdb",
    ".db",  # data/optuna/*.db -- the tuning studies
    ".json",
    ".csv",
    ".pkl",
    ".joblib",
)

_CHUNK_BYTES = 1 << 20

# Marks a value produced by the locked-file fallback in ``digest_file`` rather than by
# a content hash. Self-describing on purpose: a digest document that mixes the two must
# be readable as such months later.
_STAT_PREFIX = "stat-size-mtime:"


# How many times `require_content_digest` re-reads a file after closing handles
# before it gives up and raises. Two, because the holders worth closing -- an
# unreferenced connection object and the module-global one -- are closed by
# different mechanisms (`gc.collect()` and `close_db_connection()`), and a single
# retry cannot demonstrate that the second one helped.
DIGEST_RETRY_ATTEMPTS: int = 2

# VERBATIM from scripts/promote_models.py:558-563. Copied rather than re-phrased
# on purpose: this repository already answers "why can I not read this file on
# Windows" in one voice, and a second phrasing of the same diagnostic is a second
# answer to the same question. Copied rather than IMPORTED because that module's
# own message is about a staging path and ends with advice about --skip-train,
# which would be actively misleading here.
_LIKELY_HOLDER_CLAUSE = (
    "On Windows this is almost always a HELD HANDLE or a read-only file. The two "
    "likely holders are (1) an open DuckDB/SQLite connection left behind by an "
    "earlier run, and (2) a file browser or editor sitting in that directory."
)

# How often the locked-file path actually fired. Measured rather than assumed:
# Plan 33-01 Task 3(d) records the count over an armed integration tier, because
# "the fallback degrades the guard" and "the fallback never fires" are different
# worlds and nobody had counted.
_LOCKED_READ_OBSERVATIONS = {"count": 0}


class DataBoundaryViolation(AssertionError):
    """A production store under a guarded root was added to, removed from or rewritten."""


class LockedStoreDigestError(DataBoundaryViolation):
    """A content digest could not be taken, even after closing handles and retrying.

    A subclass of :class:`DataBoundaryViolation` so every existing caller that
    catches a boundary violation still catches this: an unprovable comparison is a
    boundary finding, not a separate kind of problem.
    """


def locked_read_observations() -> int:
    """How many times a content read has hit the locked-file path this process."""
    return _LOCKED_READ_OBSERVATIONS["count"]


def reset_locked_read_observations() -> None:
    """Zero the counter -- for a measurement that wants one run's worth."""
    _LOCKED_READ_OBSERVATIONS["count"] = 0


def _content_digest(path: Path) -> str:
    """The sha256 of *path*'s bytes. Raises if the bytes cannot be read."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close_probable_holders() -> None:
    """Drop the handles a test session is most likely to be holding.

    `gc.collect()` releases connection objects nothing references any more --
    a pattern this suite produces constantly, because a test that calls
    `load_dataframe` leaves a connection behind when its frame goes out of scope.
    `data.storage.close_db_connection` closes the MODULE-GLOBAL one, which no
    collection can reach because the module still references it. Both are needed;
    neither is sufficient.
    """
    gc.collect()
    with contextlib.suppress(ImportError, Exception):
        from data.storage import close_db_connection

        close_db_connection()


def require_content_digest(
    path: Path | str,
    *,
    attempts: int = DIGEST_RETRY_ATTEMPTS,
) -> str:
    """The sha256 of *path*'s CONTENT, or a named refusal. Never a stat signature.

    D33-32. Where `digest_file` degrades to metadata so that the CLI can keep
    reporting, this is the function a VERDICT goes through, and it does not
    degrade: it closes handles, retries, and then raises
    :class:`LockedStoreDigestError` naming the file, the underlying exception and
    the likely holder.

    That a legitimately-locked production store now fails a normal test run is the
    intended consequence, not an accident. The guard's whole claim is content-based
    evidence; an unprovable comparison is a refusal, and the named diagnostic tells
    the operator exactly which handle to close.
    """
    target = Path(path)
    try:
        return _content_digest(target)
    except (PermissionError, OSError) as first_failure:
        _LOCKED_READ_OBSERVATIONS["count"] += 1
        last_failure: OSError = first_failure

    for _ in range(max(1, attempts)):
        _close_probable_holders()
        try:
            return _content_digest(target)
        except (PermissionError, OSError) as retry_failure:
            last_failure = retry_failure

    raise LockedStoreDigestError(
        f"CANNOT PROVE THE CONTENT of {target.resolve().as_posix()} -- "
        f"{type(last_failure).__name__}: {last_failure}. "
        f"Closed handles and retried {max(1, attempts)} time(s); the bytes are "
        "still unreadable.\n\n"
        f"{_LIKELY_HOLDER_CLAUSE} Close the holder and re-run.\n\n"
        "A stat signature WOULD have been available here and is deliberately NOT "
        "accepted (D33-32): size and mtime cannot show that content is unchanged, "
        "and they degrade for exactly the locked DuckDB stores this guard most "
        "needs to cover. An unprovable comparison is a refusal, never a pass."
    ) from last_failure


def digest_file(path: Path) -> str:
    """Return the sha256 of *path*'s bytes, read in chunks so a 40 MB store is cheap.

    LOCKED FILES. On Windows, DuckDB holds an exclusive lock on an open database, and
    ``data/nfl_predictions.duckdb`` is open for the whole of any pytest session that
    touched ``load_dataframe``. Opening it for reading raises ``PermissionError``.
    Rather than crash (a guard that errors gets disabled) or skip the file silently (a
    guard that lies), fall back to a stat signature and SAY SO in the returned value:
    the digest string itself declares which instrument produced it, so a comparison can
    never quietly mix a content hash on one side with a stat signature on the other and
    call the difference a data move.

    A stat signature is weaker than a content hash -- it would miss a rewrite that
    preserved both size and mtime -- but it detects every write DuckDB actually makes,
    because a write updates mtime.
    """
    try:
        return _content_digest(path)
    except (PermissionError, OSError):
        stat = path.stat()
        return f"{_STAT_PREFIX}{stat.st_size}:{stat.st_mtime_ns}"


def is_stat_signature(value: str) -> bool:
    """True when *value* came from the locked-file fallback rather than a content hash."""
    return value.startswith(_STAT_PREFIX)


def digest_tree(
    root: Path | str,
    suffixes: tuple[str, ...] = TRACKED_SUFFIXES,
    digest=digest_file,
) -> dict[str, str]:
    """Map every tracked file under *root* to the sha256 of its contents.

    Keys are POSIX-relative to *root* so a snapshot taken on one platform reads the same
    on another. A missing *root* yields an empty mapping rather than raising: a checkout
    that never built gold has no boundary to cross, and a guard that explodes on it would
    be disabled rather than fixed.

    *digest* is the per-file instrument. It defaults to ``digest_file``, which degrades
    to a self-declaring stat signature on a locked file so the CLI can still report.
    Pass ``require_content_digest`` -- or use ``content_digest_tree`` -- where the result
    will settle a VERDICT and a signature must not be accepted (D33-32).
    """
    root_path = Path(root)
    if not root_path.exists():
        return {}

    lowered = tuple(suffix.lower() for suffix in suffixes)
    digests: dict[str, str] = {}
    for candidate in sorted(root_path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in lowered:
            continue
        digests[candidate.relative_to(root_path).as_posix()] = digest(candidate)
    return digests


def content_digest_tree(
    root: Path | str,
    suffixes: tuple[str, ...] = TRACKED_SUFFIXES,
) -> dict[str, str]:
    """``digest_tree`` that refuses to return a stat signature for any file (D33-32)."""
    return digest_tree(root, suffixes, require_content_digest)


def mixed_instrument_keys(
    before: dict[str, str],
    after: dict[str, str],
) -> list[str]:
    """Keys whose two values were produced by DIFFERENT instruments.

    NF-02. A content hash on one side and a stat signature on the other compare
    unequal as raw strings while saying nothing whatever about the bytes. The
    module docstring has forbidden that mixing since Plan 31-11 and
    ``is_stat_signature`` was written to detect it; this is the caller that was
    missing. The transition is reachable in a normal run: the write guard holds its
    baseline across a whole tier, which is exactly the window in which DuckDB opens
    ``data/nfl_predictions.duckdb`` and locks it.
    """
    return sorted(
        key
        for key in set(before) & set(after)
        if is_stat_signature(before[key]) != is_stat_signature(after[key])
    )


def diff_digests(
    before: dict[str, str],
    after: dict[str, str],
) -> dict[str, list[str]]:
    """Return the added, removed, changed and (where present) MIXED keys.

    The ``mixed`` key is OMITTED when empty, so the mapping every existing caller
    already destructures is unchanged in the overwhelmingly common case. Where it is
    present, those keys are deliberately kept OUT of ``changed``: an undecided
    comparison is not a data move, and reporting it as one is the false alarm that
    gets a guard switched off.
    """
    before_keys = set(before)
    after_keys = set(after)
    mixed = set(mixed_instrument_keys(before, after))
    diff = {
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changed": sorted(
            key
            for key in before_keys & after_keys
            if key not in mixed and before[key] != after[key]
        ),
    }
    if mixed:
        diff["mixed"] = sorted(mixed)
    return diff


def is_clean(diff: dict[str, list[str]]) -> bool:
    """True when nothing was added, removed, rewritten or left undecided.

    A MIXED key is NOT clean. "We could not tell" must never read as "nothing
    happened" -- that is the precise failure this module's own docstring warns
    about.
    """
    return not (
        diff["added"] or diff["removed"] or diff["changed"] or diff.get("mixed")
    )


def format_digest_diff(
    diff: dict[str, list[str]],
    before: dict[str, str],
    after: dict[str, str],
    root: Path | str,
) -> str:
    """Render a violation as a message that names the files and both digests.

    A boundary report that says only "data/ changed" sends the reader back to the
    filesystem to find out what. The point of a content check is that it already knows.
    """
    lines = [
        f"PRODUCTION BOUNDARY CROSSED under {Path(root).as_posix()}/ -- "
        "content digests moved.",
        "",
        "This is a CONTENT comparison, not a git one. `git status --porcelain data/` "
        "cannot report this because .gitignore blankets the directory; that check "
        "returns empty whether the store is intact or destroyed.",
    ]
    if diff["changed"]:
        lines.append("")
        lines.append("REWRITTEN:")
        for key in diff["changed"]:
            lines.append(f"  {key}")
            lines.append(f"    before sha256 {before[key]}")
            lines.append(f"    after  sha256 {after[key]}")
    if diff["added"]:
        lines.append("")
        lines.append("ADDED:")
        lines.extend(f"  {key}  sha256 {after[key]}" for key in diff["added"])
    if diff["removed"]:
        lines.append("")
        lines.append("REMOVED:")
        lines.extend(f"  {key}  sha256 {before[key]}" for key in diff["removed"])
    if diff.get("mixed"):
        lines.append("")
        lines.append("UNDECIDED (the digest instrument changed between observations):")
        for key in diff["mixed"]:
            lines.append(f"  {key}")
            lines.append(f"    before {before[key]}")
            lines.append(f"    after  {after[key]}")
        lines.append(
            "  One side is a content hash and the other is the locked-file stat "
            "signature, so these two values CANNOT be compared: the content "
            "comparison is UNDECIDED. This is reported as a hard failure rather than "
            "as a data move, and never as a pass -- see D33-32 and the module "
            "docstring. Close whatever holds the file open and re-run."
        )
    lines.append("")
    lines.append(
        "A test must not write a production store. Redirect the write to tmp_path -- "
        "for data.storage callers, monkeypatch data.storage._parquet_manager and "
        "data.storage._db_connection onto a sandbox root."
    )
    return "\n".join(lines)


def _content_resolved(after: dict[str, str], root: Path | str) -> dict[str, str]:
    """Re-read every AFTER-side stat signature as a content digest (D33-32).

    ONLY the after side. The before side is a HISTORICAL observation: re-reading
    that file now would return its CURRENT content, which would make every
    comparison trivially equal and silently void the whole guard. Where the before
    side is a signature and the after side is a content hash the comparison stays
    genuinely UNDECIDED, and ``diff_digests`` reports it as MIXED.
    """
    root_path = Path(root)
    return {
        key: (
            require_content_digest(root_path / key)
            if is_stat_signature(value)
            else value
        )
        for key, value in after.items()
    }


def assert_tree_unchanged(
    before: dict[str, str],
    after: dict[str, str],
    root: Path | str,
) -> None:
    """Raise :class:`DataBoundaryViolation` if anything under *root* moved.

    Routes the after side through ``require_content_digest`` first, so no verdict
    here can rest on a stat signature. A file that cannot be read even after
    handles are closed raises :class:`LockedStoreDigestError` -- which IS a
    :class:`DataBoundaryViolation`, so callers already catching one catch this too.
    """
    resolved = _content_resolved(after, root)
    diff = diff_digests(before, resolved)
    if is_clean(diff):
        return
    raise DataBoundaryViolation(format_digest_diff(diff, before, resolved, root))


def _main(argv: list[str]) -> int:
    """``snapshot ROOT OUT`` writes a digest document; ``verify ROOT DOC`` compares."""
    if len(argv) != 3 or argv[0] not in {"snapshot", "verify"}:
        print(
            "usage: python -m tests.data_boundary snapshot ROOT OUT.json\n"
            "       python -m tests.data_boundary verify   ROOT DOC.json",
            file=sys.stderr,
        )
        return 2

    command, root, document = argv[0], Path(argv[1]), Path(argv[2])
    if command == "snapshot":
        digests = digest_tree(root)
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text(json.dumps(digests, indent=2), encoding="utf-8")
        print(f"digested {len(digests)} file(s) under {root.as_posix()}/ -> {document}")
        return 0

    before = json.loads(document.read_text(encoding="utf-8"))
    after = digest_tree(root)
    diff = diff_digests(before, after)
    if is_clean(diff):
        print(
            f"UNCHANGED: {len(after)} file(s) under {root.as_posix()}/ match {document}"
        )
        return 0
    print(format_digest_diff(diff, before, after, root), file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover - exercised through _main in tests
    sys.exit(_main(sys.argv[1:]))
