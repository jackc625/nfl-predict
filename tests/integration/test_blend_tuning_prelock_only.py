"""The blend's tuning corpus is owned PRE-LOCK lines only, and a missing line is never filled.

WHAT THIS MODULE PINS (Plan 33.2-24 Task 1, D33.2-03 / D33.2-10)
-----------------------------------------------------------------
The blend used to be tuned on nflverse CLOSING lines for 2010-2017, fetched live. A closing
line did not exist at a game's lock, so a blend tuned on it learned how much to trust a
number nobody could have had. The corpus is now the owned ``odds_timeline`` (the Phase-29
purchase, 2020-2024, real capture times), and for each game the loader takes the LATEST
snapshot timed AT OR BEFORE that game's own lock (D33.2-01).

Four properties, each with its own test, because each fails in a different way:

1. AT-LOCK IS ADMISSIBLE. A snapshot stamped exactly at the lock is in; one second later is
   out -- the same ``<=`` as ``utils.game_lock.is_admissible``.
2. LATEST-AT-OR-BEFORE WINS, DETERMINISTICALLY, ONE ROW PER GAME. Several pre-lock snapshots
   resolve to the latest; an identical ``snapshot_ts`` resolves by the declared
   ``created_at`` tie-break; and a frame that still carries two rows for one game RAISES
   rather than silently doubling that game's weight in the fit. The writer's own
   ``(game_id, snapshot_ts)`` dedupe is not relied on.
3. NO FILL, EVER. A game with no pre-lock snapshot -- including one whose only line is a
   CLOSING line -- is EXCLUDED and returned in the excluded set. It is never filled from a
   closing line, a later snapshot or a league average.
4. THE LOADER DOES NOT READ THE CLOSING-LINE STORE. Measured on the PARSED tree, never on
   raw text, so the module's own explanation of what it does not do cannot trip the check.

A non-vacuity control runs the loader on the REAL corpus: an empty frame would satisfy
"every row is pre-lock" trivially, so the row count is asserted to sit in the neighbourhood
of the 1,346 games measured at plan time.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from utils.game_lock import game_lock

REPO_ROOT = Path(__file__).resolve().parents[2]
LOADER_MODULE_PATH = REPO_ROOT / "models" / "blending_data.py"
SILVER_DIR = REPO_ROOT / "data" / "silver"

#: The measured size of the owned pre-lock corpus at plan time (33.2-RESEARCH.md 7.3,
#: re-measured 2026-09-23): 1,346 of 1,408 scheduled 2020-2024 games carry a snapshot at or
#: before their lock. The band is a NEIGHBOURHOOD, not a pin: its job is to make an empty or
#: truncated frame fail, not to freeze a count a later capture could legitimately move.
MEASURED_PRELOCK_GAMES = 1346
NEIGHBOURHOOD = (1300, 1400)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _kickoff(day: int, hour_utc: int = 17) -> pd.Timestamp:
    """A tz-aware kickoff instant in September 2022."""
    return pd.Timestamp(datetime(2022, 9, day, hour_utc, 0)).tz_localize("UTC")


def _schedule(*game_ids_and_kickoffs: tuple[str, pd.Timestamp]) -> pd.DataFrame:
    """A silver-``games``-shaped schedule for the named games."""
    return pd.DataFrame(
        {
            "game_id": [gid for gid, _ in game_ids_and_kickoffs],
            "season": 2022,
            "week": 1,
            "game_type": "REG",
            "kickoff_et": [kickoff for _, kickoff in game_ids_and_kickoffs],
        }
    )


def _snapshot(
    game_id: str,
    snapshot_ts: pd.Timestamp,
    spread: float,
    total: float = 45.0,
    created_at: pd.Timestamp | None = None,
) -> dict[str, object]:
    """One ``odds_timeline`` row. ``spread`` is on the TIMELINE's own (inverted) sign."""
    return {
        "game_id": game_id,
        "snapshot_ts": snapshot_ts,
        "total": total,
        "spread": spread,
        "sportsbook": "consensus_median",
        "region": "us",
        "created_at": created_at
        if created_at is not None
        else pd.Timestamp("2026-08-16 18:00:00", tz="UTC"),
    }


def _lock_of(kickoff: pd.Timestamp) -> pd.Timestamp:
    """The game's lock, from THE rule (never re-derived here)."""
    return pd.Timestamp(game_lock(kickoff)).tz_convert("UTC")


# ---------------------------------------------------------------------------
# 1. At-lock is admissible; one second later is not
# ---------------------------------------------------------------------------


class TestTheLockBoundary:
    def test_a_snapshot_exactly_at_the_lock_is_admitted(self) -> None:
        from models.blending_data import select_prelock_lines

        kickoff = _kickoff(11)
        lock = _lock_of(kickoff)
        corpus = select_prelock_lines(
            pd.DataFrame([_snapshot("2022_W01_A@B", lock, spread=-3.0)]),
            _schedule(("2022_W01_A@B", kickoff)),
        )
        assert list(corpus.frame["game_id"]) == ["2022_W01_A@B"]
        assert corpus.frame["snapshot_ts"].iloc[0] == lock
        assert corpus.excluded.empty

    def test_a_snapshot_one_second_after_the_lock_is_not(self) -> None:
        from models.blending_data import EXCLUSION_NO_PRELOCK_LINE, select_prelock_lines

        kickoff = _kickoff(11)
        lock = _lock_of(kickoff)
        corpus = select_prelock_lines(
            pd.DataFrame(
                [_snapshot("2022_W01_A@B", lock + timedelta(seconds=1), spread=-3.0)]
            ),
            _schedule(("2022_W01_A@B", kickoff)),
        )
        assert corpus.frame.empty
        assert list(corpus.excluded["game_id"]) == ["2022_W01_A@B"]
        assert list(corpus.excluded["reason"]) == [EXCLUSION_NO_PRELOCK_LINE]


# ---------------------------------------------------------------------------
# 2. Latest-at-or-before wins, deterministically, one row per game
# ---------------------------------------------------------------------------


class TestTheLatestPreLockSnapshotWins:
    def test_of_three_snapshots_the_latest_pre_lock_one_is_used(self) -> None:
        from models.blending_data import select_prelock_lines

        kickoff = _kickoff(11)
        lock = _lock_of(kickoff)
        rows = [
            _snapshot("2022_W01_A@B", lock - timedelta(days=3), spread=-1.0),
            _snapshot("2022_W01_A@B", lock - timedelta(hours=1), spread=-2.0),
            _snapshot("2022_W01_A@B", lock + timedelta(hours=1), spread=-9.0),
        ]
        corpus = select_prelock_lines(
            pd.DataFrame(rows), _schedule(("2022_W01_A@B", kickoff))
        )
        assert len(corpus.frame) == 1
        assert corpus.frame["snapshot_ts"].iloc[0] == lock - timedelta(hours=1)
        # The timeline stores the OPPOSITE sign (D33.2-23); the frame carries the
        # home-margin scale, POSITIVE when the home team is favoured.
        assert corpus.frame["market_spread"].iloc[0] == pytest.approx(2.0)

    def test_it_is_neither_the_first_snapshot_nor_an_average(self) -> None:
        from models.blending_data import select_prelock_lines

        kickoff = _kickoff(11)
        lock = _lock_of(kickoff)
        rows = [
            _snapshot(
                "2022_W01_A@B", lock - timedelta(days=2), spread=-1.0, total=40.0
            ),
            _snapshot(
                "2022_W01_A@B", lock - timedelta(days=1), spread=-5.0, total=50.0
            ),
        ]
        corpus = select_prelock_lines(
            pd.DataFrame(rows), _schedule(("2022_W01_A@B", kickoff))
        )
        assert corpus.frame["market_total"].iloc[0] == pytest.approx(50.0)
        assert corpus.frame["market_spread"].iloc[0] == pytest.approx(5.0)

    def test_an_identical_snapshot_ts_resolves_by_the_declared_created_at_tie_break(
        self,
    ) -> None:
        """Two rows at the SAME stamp -- a hand-built frame can bypass the writer's dedupe."""
        from models.blending_data import select_prelock_lines

        kickoff = _kickoff(11)
        stamp = _lock_of(kickoff) - timedelta(hours=5)
        early = pd.Timestamp("2026-08-16 18:00:00", tz="UTC")
        late = pd.Timestamp("2026-08-16 18:05:00", tz="UTC")
        rows = [
            _snapshot("2022_W01_A@B", stamp, spread=-4.0, created_at=late),
            _snapshot("2022_W01_A@B", stamp, spread=-6.0, created_at=early),
        ]
        corpus = select_prelock_lines(
            pd.DataFrame(rows), _schedule(("2022_W01_A@B", kickoff))
        )
        assert len(corpus.frame) == 1
        assert corpus.frame["created_at"].iloc[0] == late
        assert corpus.frame["market_spread"].iloc[0] == pytest.approx(4.0)

    def test_a_frame_still_carrying_two_rows_for_one_game_raises(self) -> None:
        from models.blending_data import (
            DuplicateTuningRowError,
            assert_one_row_per_game,
        )

        frame = pd.DataFrame(
            {"game_id": ["2022_W01_A@B", "2022_W01_A@B", "2022_W01_C@D"]}
        )
        with pytest.raises(DuplicateTuningRowError, match="2022_W01_A@B"):
            assert_one_row_per_game(frame)

    def test_a_frame_with_one_row_per_game_passes(self) -> None:
        """The no-false-positive control for the assertion above."""
        from models.blending_data import assert_one_row_per_game

        assert_one_row_per_game(pd.DataFrame({"game_id": ["2022_W01_A@B", "x", "y"]}))


# ---------------------------------------------------------------------------
# 3. No fill, ever
# ---------------------------------------------------------------------------


class TestNoMissingLineIsFilled:
    def test_a_game_with_no_snapshot_is_excluded_and_returned(self) -> None:
        from models.blending_data import EXCLUSION_NO_PRELOCK_LINE, select_prelock_lines

        kick_a, kick_b = _kickoff(11), _kickoff(12)
        corpus = select_prelock_lines(
            pd.DataFrame(
                [_snapshot("2022_W01_A@B", _lock_of(kick_a) - timedelta(hours=2), -3.0)]
            ),
            _schedule(("2022_W01_A@B", kick_a), ("2022_W01_C@D", kick_b)),
        )
        assert list(corpus.frame["game_id"]) == ["2022_W01_A@B"]
        excluded = corpus.excluded.set_index("game_id")
        assert list(excluded.index) == ["2022_W01_C@D"]
        assert excluded.loc["2022_W01_C@D", "reason"] == EXCLUSION_NO_PRELOCK_LINE
        assert excluded.loc["2022_W01_C@D", "detail"] == "absent_from_timeline"

    def test_a_closing_line_only_game_is_excluded_not_filled(self) -> None:
        """THE PROHIBITION TEST (T-33.2-24-01).

        The game's only line is a CLOSING line -- a snapshot at kickoff, after its lock. The
        row must be ABSENT from the tuning frame and PRESENT in the excluded set. A filled
        row is a row the tuner treats as evidence about a line that did not exist at the lock.
        """
        from models.blending_data import EXCLUSION_NO_PRELOCK_LINE, select_prelock_lines

        kickoff = _kickoff(11)
        corpus = select_prelock_lines(
            pd.DataFrame([_snapshot("2022_W01_A@B", kickoff, spread=-7.0)]),
            _schedule(("2022_W01_A@B", kickoff)),
        )
        assert "2022_W01_A@B" not in set(corpus.frame["game_id"])
        excluded = corpus.excluded.set_index("game_id")
        assert excluded.loc["2022_W01_A@B", "reason"] == EXCLUSION_NO_PRELOCK_LINE
        assert excluded.loc["2022_W01_A@B", "detail"] == "post_lock_snapshots_only"

    def test_the_excluded_set_is_returned_with_the_frame_not_merely_logged(
        self,
    ) -> None:
        from models.blending_data import PrelockTuningCorpus, select_prelock_lines

        kickoff = _kickoff(11)
        corpus = select_prelock_lines(
            pd.DataFrame(columns=list(_snapshot("x", kickoff, 0.0))),
            _schedule(("2022_W01_A@B", kickoff)),
        )
        assert isinstance(corpus, PrelockTuningCorpus)
        assert isinstance(corpus.excluded, pd.DataFrame)
        assert {"game_id", "season", "reason", "detail"} <= set(corpus.excluded.columns)
        assert len(corpus.excluded) == 1

    def test_a_timeline_row_naming_no_scheduled_game_is_reported_not_joined(
        self,
    ) -> None:
        from models.blending_data import select_prelock_lines

        kickoff = _kickoff(11)
        corpus = select_prelock_lines(
            pd.DataFrame(
                [
                    _snapshot(
                        "2022_W01_A@B", _lock_of(kickoff) - timedelta(hours=2), -3.0
                    ),
                    _snapshot(
                        "2022_W99_X@Y", _lock_of(kickoff) - timedelta(hours=2), -3.0
                    ),
                ]
            ),
            _schedule(("2022_W01_A@B", kickoff)),
        )
        assert corpus.unjoined_timeline_ids == ("2022_W99_X@Y",)
        assert list(corpus.frame["game_id"]) == ["2022_W01_A@B"]

    def test_a_naive_snapshot_time_is_refused_not_relabelled(self) -> None:
        from models.blending_data import TuningCorpusError, select_prelock_lines

        kickoff = _kickoff(11)
        naive = (_lock_of(kickoff) - timedelta(hours=2)).tz_localize(None)
        with pytest.raises(TuningCorpusError, match="naive"):
            select_prelock_lines(
                pd.DataFrame([_snapshot("2022_W01_A@B", naive, -3.0)]),
                _schedule(("2022_W01_A@B", kickoff)),
            )


# ---------------------------------------------------------------------------
# 4. The loader does not read the closing-line store (parsed tree, planted control)
# ---------------------------------------------------------------------------


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids of the bare string-statement constants: docstrings and prose, never code."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            ids.add(id(node.value))
    return ids


def closing_store_violations(source: str) -> list[str]:
    """Every CODE reference to the closing-line store or a live schedule fetch in *source*.

    A string constant CONTAINING ``odds_snapshot`` (so a ``.../odds_snapshot.parquet`` path
    is caught, which an exact-match scan would miss), any identifier or attribute naming it,
    and any CALL to ``load_schedules``. Bare string statements -- docstrings and prose -- are
    skipped deliberately: the loader is REQUIRED to explain that it has no such fallback, and
    a scan the explanation could trip would force a choice between the two.
    """
    tree = ast.parse(source)
    prose = _docstring_nodes(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in prose
            and "odds_snapshot" in node.value
        ):
            hits.append(f"constant:{node.value}")
        elif isinstance(node, ast.Name) and "odds_snapshot" in node.id:
            hits.append(f"name:{node.id}")
        elif isinstance(node, ast.Attribute) and "odds_snapshot" in node.attr:
            hits.append(f"attribute:{node.attr}")
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if name == "load_schedules":
                hits.append("call:load_schedules")
    return hits


class TestTheLoaderReadsNoClosingLineStore:
    def test_the_loader_module_has_no_closing_store_reference_in_code(self) -> None:
        source = LOADER_MODULE_PATH.read_text(encoding="utf-8")
        assert closing_store_violations(source) == []

    def test_the_loader_module_names_the_owned_timeline(self) -> None:
        """Non-vacuity: the module under scan IS the timeline loader."""
        from models.blending_data import OWNED_TIMELINE_TABLE

        assert OWNED_TIMELINE_TABLE == "odds_timeline"
        assert "odds_timeline" in LOADER_MODULE_PATH.read_text(encoding="utf-8")

    def test_a_planted_closing_store_read_is_flagged(self) -> None:
        planted = (
            "import pandas as pd\n"
            "def fallback():\n"
            "    return pd.read_parquet('data/silver/odds_snapshot.parquet')\n"
        )
        assert closing_store_violations(planted) == [
            "constant:data/silver/odds_snapshot.parquet"
        ]

    def test_a_planted_live_schedule_fetch_is_flagged(self) -> None:
        planted = "import nflreadpy\nnflreadpy.load_schedules(seasons=[2015])\n"
        assert closing_store_violations(planted) == ["call:load_schedules"]

    def test_prose_naming_the_store_is_not_a_violation(self) -> None:
        """No-false-positive control: the required explanation cannot fail the check."""
        planted = (
            '"""There is no fallback to odds_snapshot and no load_schedules call."""\n'
            "x = 1\n"
        )
        assert closing_store_violations(planted) == []


# ---------------------------------------------------------------------------
# The REAL corpus: every row pre-lock, one row per game, non-vacuous
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_corpus():
    if (
        not (SILVER_DIR / "odds_timeline.parquet").exists()
        or not (SILVER_DIR / "games.parquet").exists()
    ):
        pytest.skip(
            "evidence-backed skip: silver odds_timeline / games are absent from this "
            "checkout, so there is no owned corpus to load"
        )
    from models.blending_data import load_tuning_period_data

    return load_tuning_period_data(silver_dir=SILVER_DIR)


@pytest.mark.integration
class TestTheRealCorpus:
    def test_the_frame_is_non_empty_and_near_the_measured_size(
        self, real_corpus
    ) -> None:
        low, high = NEIGHBOURHOOD
        assert low <= len(real_corpus.frame) <= high, (
            f"{len(real_corpus.frame)} tuning rows; {MEASURED_PRELOCK_GAMES} were measured "
            "at plan time. An empty or truncated frame would pass every pre-lock assertion "
            "below vacuously."
        )

    def test_every_tuning_row_is_at_or_before_its_own_lock(self, real_corpus) -> None:
        frame = real_corpus.frame
        assert (frame["snapshot_ts"] <= frame["lock"]).all()

    def test_every_lock_is_the_rules_lock(self, real_corpus) -> None:
        """The frame's lock is THE rule's, not a second derivation (spot-checked on 25)."""
        from utils.game_lock import lock_frame

        games = pd.read_parquet(SILVER_DIR / "games.parquet")
        locks = lock_frame(games[games["game_id"].isin(real_corpus.frame["game_id"])])
        sample = real_corpus.frame.head(25)
        for row in sample.itertuples():
            assert row.lock == pd.Timestamp(locks.loc[row.game_id])

    def test_the_frame_has_exactly_one_row_per_game(self, real_corpus) -> None:
        assert real_corpus.frame["game_id"].is_unique

    def test_every_row_carries_a_spread_and_a_total(self, real_corpus) -> None:
        assert real_corpus.frame["market_spread"].notna().all()
        assert real_corpus.frame["market_total"].notna().all()

    def test_no_game_is_both_tuned_and_excluded(self, real_corpus) -> None:
        tuned = set(real_corpus.frame["game_id"])
        excluded = set(real_corpus.excluded["game_id"])
        assert tuned.isdisjoint(excluded)

    def test_tuned_plus_excluded_is_the_whole_schedule(self, real_corpus) -> None:
        from models.market_probability import OWNED_LINE_SEASONS

        games = pd.read_parquet(SILVER_DIR / "games.parquet")
        scheduled = set(games.loc[games["season"].isin(OWNED_LINE_SEASONS), "game_id"])
        tuned = set(real_corpus.frame["game_id"])
        excluded = set(real_corpus.excluded["game_id"])
        assert tuned | excluded == scheduled

    def test_the_home_margin_sign_is_the_serving_convention(self, real_corpus) -> None:
        """POSITIVE market_spread means the home team is favoured.

        MEASURED rather than assumed: over the real corpus the home-margin-scale spread is
        POSITIVELY correlated with the realized home margin. The stored timeline sign is the
        opposite (D33.2-23); a missing or doubled flip would make this negative.
        """
        games = pd.read_parquet(SILVER_DIR / "games.parquet")
        joined = real_corpus.frame.merge(
            games[["game_id", "home_score", "away_score"]], on="game_id"
        )
        margin = joined["home_score"] - joined["away_score"]
        assert joined["market_spread"].corr(margin) > 0.3
