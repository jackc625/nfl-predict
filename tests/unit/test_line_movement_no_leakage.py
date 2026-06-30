"""Leakage proof for the line-movement signal (SIG-04 / SIG-06 / D-15).

Following the `tests/unit/test_situational_no_leakage.py` /
`tests/unit/test_elo_no_leakage.py` convention. `LineMovementBuilder` (Plan
29-04) derives the four D-09 line-movement families from the `odds_timeline`
trajectory, fenced strictly to snapshots at/before EACH game's OWN Friday-6PM-ET
freeze. This module makes the D-15 3-part standard LIVE for the line-movement
group.

Leakage contract (D-15): only snapshots strictly `<= freeze` (ET, not UTC) may
enter a feature; the true closing line is reserved for CLV grading and is NEVER
emitted. The freeze is PER-GAME (derived from each game's kickoff date), so the
proof must use >=2 games with DIFFERENT kickoffs to show the fence is per-game,
not one global Friday (review 29-04 HIGH).

Parts:
  1. TestTimeFence -- each game fences to its OWN Friday-6PM-ET freeze; an
     ET-evening (22:00 UTC = 18:00 ET) snapshot a UTC-localized cutoff would drop
     IS included (WR-02 positive control); a naive-local `datetime.now()` default
     does NOT raise TypeError (review 29-04 HIGH).
  2. TestWithholdFuture -- appending GENUINELY post-freeze (Sat/Sun/post-close)
     timeline rows + game results does NOT change the emitted features
     (byte-identical); a positive control proves that moving a game's freeze PAST
     a Saturday row DOES change them (the fence is load-bearing).
  3. TestKeywordGuard -- no emitted column carries a closing/post-freeze spelling,
     and the emitted columns do not collide with the LeakageGate keywords.
"""

from datetime import datetime

import pandas as pd

from features.line_movement import LEAGUE_AVERAGE_TOTAL, LineMovementBuilder
from features.validation import LeakageGate

# Two games on DIFFERENT kickoff dates -> DIFFERENT per-game Friday freezes
# (review 29-04 HIGH). Game A: Sun 2023-09-10 -> freeze Fri 2023-09-08 18:00 ET
# (= 22:00 UTC). Game B: Sun 2023-09-17 -> freeze Fri 2023-09-15 18:00 ET.
_GAME_A = "2023_01_DET_KC"
_GAME_B = "2023_02_LV_BUF"


def _make_games(*, move_game_a: bool = False, reveal_results: bool = False):
    """Two games with different kickoffs (hence different per-game freezes).

    Args:
        move_game_a: When True, Game A's kickoff slips one week later so its
            Friday freeze (2023-09-15) lands PAST the Saturday 2023-09-09 row --
            the withhold-future positive control.
        reveal_results: When True, attach future game RESULTS (scores) -- the
            leakage-sensitive future information the builder must ignore.
    """
    a_kickoff = (
        datetime(2023, 9, 17, 13, 0) if move_game_a else datetime(2023, 9, 10, 13, 0)
    )
    rows = [
        {
            "game_id": _GAME_A,
            "season": 2023,
            "week": 1,
            "home_team": "KC",
            "away_team": "DET",
            "kickoff_et": a_kickoff,
        },
        {
            "game_id": _GAME_B,
            "season": 2023,
            "week": 2,
            "home_team": "BUF",
            "away_team": "LV",
            "kickoff_et": datetime(2023, 9, 17, 13, 0),
        },
    ]
    if reveal_results:
        for r, (h, a) in zip(rows, [(21.0, 20.0), (24.0, 17.0)], strict=True):
            r["home_score"] = h
            r["away_score"] = a
    return pd.DataFrame(rows)


def _ts(spec: str) -> pd.Timestamp:
    """A tz-aware UTC timestamp."""
    return pd.Timestamp(spec, tz="UTC")


def _pre_freeze_timeline() -> pd.DataFrame:
    """Only snapshots strictly at/before each game's own Friday-6PM-ET freeze.

    Game A (freeze Fri 2023-09-08 22:00 UTC): open 44.0 (Tue) -> 44.5 (Wed) ->
    45.0 at the Fri 22:00 UTC (= 18:00 ET) freeze. The 45.0 is the WR-02
    positive control: a UTC-localized 18:00 cutoff would drop a 22:00 UTC row.
    Game B (freeze Fri 2023-09-15 22:00 UTC): open 48.0 (Tue) -> 47.0 at freeze.
    """
    return pd.DataFrame(
        {
            "game_id": [_GAME_A, _GAME_A, _GAME_A, _GAME_B, _GAME_B],
            "snapshot_ts": [
                _ts("2023-09-05 16:00"),
                _ts("2023-09-06 16:00"),
                _ts("2023-09-08 22:00"),  # 18:00 ET == freeze (ET-evening control)
                _ts("2023-09-12 16:00"),
                _ts("2023-09-15 22:00"),  # 18:00 ET == freeze
            ],
            "total": [44.0, 44.5, 45.0, 48.0, 47.0],
        }
    )


def _post_freeze_rows() -> pd.DataFrame:
    """Genuinely POST-freeze rows (Sat/Sun/post-close), AFTER each game's freeze.

    For Game A (freeze 2023-09-08): a Sat 2023-09-09 (46.0) and Sun 2023-09-10
    (47.5) row -- both strictly after the freeze, so the per-game fence MUST drop
    them. For Game B (freeze 2023-09-15): a Sat 2023-09-16 (50.0) row.
    """
    return pd.DataFrame(
        {
            "game_id": [_GAME_A, _GAME_A, _GAME_B],
            "snapshot_ts": [
                _ts("2023-09-09 18:00"),
                _ts("2023-09-10 17:00"),
                _ts("2023-09-16 18:00"),
            ],
            "total": [46.0, 47.5, 50.0],
        }
    )


def _full_timeline() -> pd.DataFrame:
    """Pre-freeze snapshots + genuinely post-freeze rows."""
    return pd.concat([_pre_freeze_timeline(), _post_freeze_rows()], ignore_index=True)


# A freeze cutoff after BOTH games' Friday freezes (so the as_of cap never binds;
# the PER-GAME freeze is what excludes the post-freeze rows).
_AS_OF_AFTER_BOTH = datetime(2023, 9, 20, 18, 0)


class TestTimeFence:
    """Part 1: each game fences to its OWN Friday-6PM-ET freeze (ET, not UTC)."""

    def test_per_game_freeze_excludes_post_freeze_rows(self):
        """Game A and Game B each use only snapshots <= their OWN Friday freeze.

        With the full timeline (pre- + post-freeze rows), the per-game fence keeps
        Game A's drift at +1.0 (open 44.0 -> freeze 45.0) and Game B's at -1.0
        (open 48.0 -> freeze 47.0). A single GLOBAL Friday (e.g. the later
        2023-09-15) would wrongly admit Game A's Sat/Sun rows and inflate its
        drift to +3.5 -- so drift == +1.0 proves the fence is per-game (review
        29-04 HIGH).
        """
        builder = LineMovementBuilder(timeline_df=_full_timeline())
        out = builder.build_features(_make_games(), _AS_OF_AFTER_BOTH).set_index(
            "game_id"
        )

        assert out.loc[_GAME_A, "opening_total"] == 44.0
        assert out.loc[_GAME_A, "total_drift"] == 1.0
        assert out.loc[_GAME_B, "opening_total"] == 48.0
        assert out.loc[_GAME_B, "total_drift"] == -1.0

    def test_et_evening_snapshot_is_included(self):
        """WR-02: the Fri 22:00 UTC (= 18:00 ET) freeze snapshot IS included.

        A UTC-localized Friday-18:00 cutoff would compare 22:00 UTC > 18:00 UTC
        and DROP the legitimate ET-evening snapshot, collapsing Game A's freeze
        total to 44.5 and its range to 0.5. The ET cutoff keeps the 45.0 row, so
        opening->freeze range == 1.0 proves the ET fence is load-bearing.
        """
        builder = LineMovementBuilder(timeline_df=_pre_freeze_timeline())
        out = builder.build_features(_make_games(), _AS_OF_AFTER_BOTH).set_index(
            "game_id"
        )

        # The 45.0 ET-evening row survived -> covered, full open-to-freeze span.
        assert out.loc[_GAME_A, "line_movement_coverage"] == 1.0
        assert out.loc[_GAME_A, "total_range"] == 1.0
        assert out.loc[_GAME_A, "total_drift"] == 1.0

    def test_naive_now_default_does_not_raise(self):
        """A naive-local `datetime.now()` as_of must not raise TypeError.

        The full-build default is a naive-local `datetime.now()`
        (build_features.py:111-112/:852-853); the builder canonicalizes it to
        tz-aware UTC before comparing against the tz-aware snapshot_ts (review
        29-04 HIGH). `now()` is far after the 2023 freezes, so the per-game freeze
        still binds and the drifts are unchanged.
        """
        builder = LineMovementBuilder(timeline_df=_full_timeline())
        out = builder.build_features(_make_games(), datetime.now()).set_index("game_id")

        assert out.loc[_GAME_A, "total_drift"] == 1.0
        assert out.loc[_GAME_B, "total_drift"] == -1.0


class TestWithholdFuture:
    """Part 2: revealing genuinely post-freeze rows must not move the features."""

    def test_revealing_post_freeze_rows_does_not_change_features(self):
        """Build with only pre-freeze snapshots, then again after appending
        GENUINELY post-freeze (Sat/Sun/post-close) timeline rows AND game results;
        the emitted features must be BYTE-IDENTICAL.

        The appended rows are strictly after each game's own Friday freeze (review
        29-04 HIGH), so the per-game fence drops them -- the test exercises the
        fence rather than passing vacuously.
        """
        base = LineMovementBuilder(timeline_df=_pre_freeze_timeline()).build_features(
            _make_games(), _AS_OF_AFTER_BOTH
        )
        revealed = LineMovementBuilder(timeline_df=_full_timeline()).build_features(
            _make_games(reveal_results=True), _AS_OF_AFTER_BOTH
        )

        pd.testing.assert_frame_equal(
            base.sort_values("game_id").reset_index(drop=True),
            revealed.sort_values("game_id").reset_index(drop=True),
        )

    def test_moving_freeze_past_saturday_row_changes_features(self):
        """Positive control: moving Game A's freeze PAST a Saturday row DOES move
        the features (the fence is load-bearing, mirroring the Phase-28 control).

        Baseline Game A (kickoff 2023-09-10 -> freeze 2023-09-08) has drift +1.0.
        Slipping its kickoff one week (freeze 2023-09-15) admits the Sat 46.0 and
        Sun 47.5 rows -> open 44.0, freeze 47.5, drift +3.5 != +1.0.
        """
        full = _full_timeline()
        base = LineMovementBuilder(timeline_df=full).build_features(
            _make_games(), _AS_OF_AFTER_BOTH
        )
        moved = LineMovementBuilder(timeline_df=full).build_features(
            _make_games(move_game_a=True), _AS_OF_AFTER_BOTH
        )

        base_a = base.set_index("game_id").loc[_GAME_A, "total_drift"]
        moved_a = moved.set_index("game_id").loc[_GAME_A, "total_drift"]
        assert base_a == 1.0
        assert moved_a == 3.5
        assert base_a != moved_a


class TestKeywordGuard:
    """Part 3: no emitted column carries a closing/post-freeze line spelling."""

    def _emitted_columns(self) -> list[str]:
        builder = LineMovementBuilder(timeline_df=_full_timeline())
        out = builder.build_features(_make_games(), _AS_OF_AFTER_BOTH)
        return [c for c in out.columns if c != "game_id"]

    def test_no_closing_or_post_freeze_column(self):
        """No emitted column names a closing / post-freeze / result value (D-15)."""
        banned = ("clos", "closing", "final", "result", "post_freeze", "postfreeze")
        for col in self._emitted_columns():
            lc = col.lower()
            assert not any(token in lc for token in banned), col

    def test_columns_do_not_collide_with_leakage_keywords(self):
        """The emitted columns pass the LeakageGate substring keyword guard.

        `opening_total` / `total_drift` / ... are legitimate non-leaky names; none
        contains a LeakageGate.LEAKAGE_KEYWORDS spelling (e.g. `total_score`).
        """
        for col in self._emitted_columns():
            lc = col.lower()
            for keyword in LeakageGate.LEAKAGE_KEYWORDS:
                assert keyword not in lc, (
                    f"{col} collides with leakage keyword {keyword}"
                )

    def test_closing_total_absent_uncovered_opening_not_ood(self):
        """An uncovered game emits no closing line and a non-OOD opening_total.

        A game with NO trajectory coverage (empty timeline) gets
        line_movement_coverage=0.0, drift/path=0.0, and opening_total imputed from
        the prior-only LEAGUE_AVERAGE_TOTAL constant -- never a literal 0.0
        (out-of-distribution for a ~40-50 line, review 29-04 MED).
        """
        builder = LineMovementBuilder(
            timeline_df=pd.DataFrame(columns=["game_id", "snapshot_ts", "total"])
        )
        out = builder.build_features(_make_games(), _AS_OF_AFTER_BOTH).set_index(
            "game_id"
        )

        assert out.loc[_GAME_A, "line_movement_coverage"] == 0.0
        assert out.loc[_GAME_A, "total_drift"] == 0.0
        assert out.loc[_GAME_A, "opening_total"] == LEAGUE_AVERAGE_TOTAL
        assert out.loc[_GAME_A, "opening_total"] != 0.0
