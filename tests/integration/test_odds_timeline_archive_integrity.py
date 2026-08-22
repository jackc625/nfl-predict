"""The paid ``odds_timeline`` archive's integrity gate (SPEC R3, T-30-12, Plan 30-04).

Phase 30 drops the fifteen ``line_movement`` columns from gold. It must NOT drop the paid
archive those columns were built from: ``data/silver/odds_timeline.parquet`` is 9,957 rows
bought with real Odds API credits (Plan 29-05, re-keyed by quick task 260816-u0e), and it is
not re-derivable from anything in this repository. Re-buying it costs money.

WHY THIS FILE EXISTS RATHER THAN A GIT CHECK (N-03). The obvious guard --
``git status --porcelain data/`` -- is VACUOUS here. ``.gitignore`` line 22 is ``data/``, so
porcelain returns empty whether the archive is intact or has been deleted. A phase that
asserted the archive's safety that way would have asserted nothing at all. **No assertion in
this module uses a git-status check**, and none ever should. The substantive gate is the one
below: row count, distinct ``(game_id, snapshot_ts)`` pair count, and per-season counts, all
read THROUGH ``data.storage.load_dataframe`` rather than by reading the parquet path
directly -- because ``load_dataframe(source="auto")`` resolves DuckDB first and only falls
back to parquet, so asserting through the same seam the feature builders use is what makes
this a statement about what the pipeline sees.

The expected counts are imported from ``tests.phase30_state``, the git-TRACKED Phase-30
manifest, rather than declared as literals here. That is what makes the phase-START assertion
in Plan 30-04 and the phase-END assertion in Plan 30-07 provably the SAME assertion rather
than two hand-copied lists that agree today.

TEST CLASS (Plan 30-04's phase-wide rule): **integration / slow**. Every test carries
``@pytest.mark.integration``, reads generated ``data/`` state, and skips cleanly with a
remediation-carrying message when that state is absent -- the
``tests/integration/test_activation_parity.py`` shape. Read-only: nothing here writes under
``data/``.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from data.storage import load_dataframe
from tests.phase30_state import (
    ODDS_TIMELINE_PAIR_LIST_SHA256,
    ODDS_TIMELINE_PAIRS,
    ODDS_TIMELINE_ROWS,
    ODDS_TIMELINE_SEASON_COUNTS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_ARCHIVE_PATH = REPO_ROOT / "data" / "silver" / "odds_timeline.parquet"

_PAIR_KEYS = ["game_id", "snapshot_ts"]


def _require_archive() -> None:
    """Skip cleanly when the paid archive is absent from this checkout.

    ``data/`` is gitignored, so a fresh clone legitimately has no archive. The skip
    message names the remediation rather than leaving a bare skip, per the house shape.
    """
    if not _ARCHIVE_PATH.exists():
        pytest.skip(
            f"paid odds_timeline archive not present at {_ARCHIVE_PATH} -- it is "
            "gitignored runtime data bought with Odds API credits (Plan 29-05). "
            "Restore it from backup; do NOT re-pull it to make this test run."
        )


@pytest.fixture(scope="module")
def archive() -> pd.DataFrame:
    """The archive read through the same seam every feature builder reads it through."""
    _require_archive()
    return load_dataframe("odds_timeline", layer="silver")


def _season_counts(df: pd.DataFrame) -> dict[int, int]:
    """Per-season row counts, derived from the season prefix of ``game_id``."""
    seasons = df["game_id"].str.slice(0, 4).astype(int)
    return {int(season): int(count) for season, count in seasons.value_counts().items()}


def pair_list_sha256(df: pd.DataFrame) -> str:
    """Return the sha256 of the deterministically sorted (game_id, snapshot_ts) pair list.

    Shared with the capture that wrote ``outputs/n01/odds_timeline_baseline.json``, so the
    baseline document and this assertion are computed by the same expression rather than by
    two that happen to agree.
    """
    pairs = (
        df[_PAIR_KEYS]
        .drop_duplicates()
        .assign(snapshot_ts=lambda frame: frame["snapshot_ts"].astype(str))
        .sort_values(_PAIR_KEYS)
    )
    joined = "\n".join(
        f"{game_id}\x1f{snapshot_ts}"
        for game_id, snapshot_ts in zip(pairs["game_id"], pairs["snapshot_ts"])
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@pytest.mark.integration
class TestPaidArchiveIntegrity:
    """T-30-12: the archive is intact, asserted by content and never by git status."""

    def test_row_count_is_the_committed_baseline(self, archive: pd.DataFrame):
        assert len(archive) == ODDS_TIMELINE_ROWS, (
            f"the paid odds_timeline archive reads {len(archive)} rows against the "
            f"committed baseline of {ODDS_TIMELINE_ROWS} (tests/phase30_state.py). This "
            "archive is not re-derivable and re-buying it costs Odds API credits."
        )

    def test_distinct_pair_count_is_the_committed_baseline(self, archive: pd.DataFrame):
        pairs = len(archive[_PAIR_KEYS].drop_duplicates())
        assert pairs == ODDS_TIMELINE_PAIRS, (
            f"the archive holds {pairs} distinct (game_id, snapshot_ts) pairs against the "
            f"committed baseline of {ODDS_TIMELINE_PAIRS}. A row count that matches while "
            "the pair count moves means rows were swapped, not lost -- which a count-only "
            "check would miss (the 29-06 lesson applied to rows)."
        )

    def test_every_row_is_a_distinct_pair(self, archive: pd.DataFrame):
        assert ODDS_TIMELINE_ROWS == ODDS_TIMELINE_PAIRS
        assert len(archive) == len(archive[_PAIR_KEYS].drop_duplicates())

    def test_per_season_counts_are_unchanged(self, archive: pd.DataFrame):
        assert _season_counts(archive) == ODDS_TIMELINE_SEASON_COUNTS

    def test_the_season_counts_sum_to_the_row_count(self):
        assert sum(ODDS_TIMELINE_SEASON_COUNTS.values()) == ODDS_TIMELINE_ROWS, (
            "the committed per-season baseline does not sum to the committed row count, so "
            "at least one of the two is transcribed rather than measured"
        )

    def test_the_pair_list_digest_is_the_committed_baseline(
        self, archive: pd.DataFrame
    ):
        assert pair_list_sha256(archive) == ODDS_TIMELINE_PAIR_LIST_SHA256, (
            "the archive's (game_id, snapshot_ts) pair list no longer digests to the "
            "committed baseline. The counts can all match while the CONTENT has moved; "
            "this is the assertion that catches that."
        )

    def test_no_closing_line_leaked_into_the_archive_schema(
        self, archive: pd.DataFrame
    ):
        leaked = [column for column in archive.columns if column.startswith("closing_")]
        assert leaked == [], f"closing-line columns present in the archive: {leaked}"

    def test_this_module_shells_out_to_nothing(self):
        """The guard that makes N-03 unrepeatable: no git check may stand in for content.

        A git-status check on ``data/`` is VACUOUS -- the directory is gitignored at
        ``.gitignore`` line 22, so porcelain reports success on a destroyed archive. The
        assertion here is mechanical rather than textual: this module runs no subprocess at
        all, and without a subprocess there is no git check to reach for.
        """
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        forbidden = ({"subprocess", "os", "sh", "commands"} & imported) | (
            {"__import__", "eval", "exec"} & called
        )
        assert forbidden == set(), (
            f"this module reaches for {sorted(forbidden)}, which is how a shell-out gets "
            "written. Integrity here is asserted from CONTENT read through load_dataframe, "
            "never from a git check -- a git check on gitignored data/ reports success on "
            "a destroyed archive (N-03). The assertion is on the parsed imports rather "
            "than on the source text, so it cannot match its own wording."
        )
