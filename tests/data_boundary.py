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
"""

from __future__ import annotations

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


class DataBoundaryViolation(AssertionError):
    """A production store under a guarded root was added to, removed from or rewritten."""


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
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (PermissionError, OSError):
        stat = path.stat()
        return f"{_STAT_PREFIX}{stat.st_size}:{stat.st_mtime_ns}"


def is_stat_signature(value: str) -> bool:
    """True when *value* came from the locked-file fallback rather than a content hash."""
    return value.startswith(_STAT_PREFIX)


def digest_tree(
    root: Path | str,
    suffixes: tuple[str, ...] = TRACKED_SUFFIXES,
) -> dict[str, str]:
    """Map every tracked file under *root* to the sha256 of its contents.

    Keys are POSIX-relative to *root* so a snapshot taken on one platform reads the same
    on another. A missing *root* yields an empty mapping rather than raising: a checkout
    that never built gold has no boundary to cross, and a guard that explodes on it would
    be disabled rather than fixed.
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
        digests[candidate.relative_to(root_path).as_posix()] = digest_file(candidate)
    return digests


def diff_digests(
    before: dict[str, str],
    after: dict[str, str],
) -> dict[str, list[str]]:
    """Return the added, removed and changed keys between two digest mappings."""
    before_keys = set(before)
    after_keys = set(after)
    return {
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changed": sorted(
            key for key in before_keys & after_keys if before[key] != after[key]
        ),
    }


def is_clean(diff: dict[str, list[str]]) -> bool:
    """True when nothing was added, removed or rewritten."""
    return not (diff["added"] or diff["removed"] or diff["changed"])


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
    lines.append("")
    lines.append(
        "A test must not write a production store. Redirect the write to tmp_path -- "
        "for data.storage callers, monkeypatch data.storage._parquet_manager and "
        "data.storage._db_connection onto a sandbox root."
    )
    return "\n".join(lines)


def assert_tree_unchanged(
    before: dict[str, str],
    after: dict[str, str],
    root: Path | str,
) -> None:
    """Raise :class:`DataBoundaryViolation` if anything under *root* moved."""
    diff = diff_digests(before, after)
    if is_clean(diff):
        return
    raise DataBoundaryViolation(format_digest_diff(diff, before, after, root))


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
