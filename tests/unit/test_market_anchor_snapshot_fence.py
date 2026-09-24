"""The market-anchor gold builder admits a line by its CAPTURE time, never its label.

TEST CLASS: plain unit tests. The odds frame is built in the test and injected by
monkeypatching ``features.market_anchors.load_dataframe``; no production store is read
or written, and the module passes on a fresh checkout with no ``data/``.

THE HISTORY. The stored ``snapshot_ts`` is a STRING column holding two spellings
(``2018-09-19T18:00:00-04:00`` and ``2025-08-29 22:00:00+00:00``), and a bare
``pd.to_datetime`` once NaT'd every row in the second spelling, so a whole season's market
anchors fell to the neutral default (Plan 31-11). Plan 33.2-14 then ruled that a line counts
only with a RECORDED capture time (``created_at``) at or before its game's lock, and the gold
builder stopped reading the label at all. What stays pinned here is that OUTCOME.

RETIRED (33.2 review batch 3): the classes that pinned ``_parse_snapshot_column`` and the
deprecated ``identify_opening_lines`` / ``select_snapshot_lines_at_lock`` path. That path
existed only to write silver ``market_anchor_features``, which no production code read, and
it was deleted with the step that wrote it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from features.market_anchors import MarketAnchorFeaturesCalculator

# The two spellings the live column holds, with the SAME instant expressed both ways so a
# comparison between them is meaningful: 18:00 Eastern on 2025-08-29 is 22:00 UTC.
LEGACY_SPELLING = "2018-09-19T18:00:00-04:00"
TZ_AWARE_SPELLING = "2025-08-29 22:00:00+00:00"


def _odds_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2018_W03_LAC@LA",
                "sportsbook": "consensus",
                "snapshot_ts": LEGACY_SPELLING,
                "spread": -3.0,
                "total": 46.5,
                "ml_home": -150,
                "ml_away": 130,
            },
            {
                "game_id": "2025_W01_DAL@PHI",
                "sportsbook": "consensus",
                "snapshot_ts": TZ_AWARE_SPELLING,
                "spread": -7.5,
                "total": 51.0,
                "ml_home": -320,
                "ml_away": 260,
            },
        ]
    )


def _captured_odds_frame() -> pd.DataFrame:
    """The same two rows, each CAPTURED at the instant its label names (``created_at``)."""
    frame = _odds_frame()
    frame["created_at"] = [
        pd.Timestamp(LEGACY_SPELLING).tz_convert("UTC"),
        pd.Timestamp(TZ_AWARE_SPELLING).tz_convert("UTC"),
    ]
    return frame


_BUILT_GAMES = pd.DataFrame(
    [
        {
            "game_id": "2018_W03_LAC@LA",
            "season": 2018,
            "week": 3,
            "kickoff_et": pd.Timestamp("2018-09-23 17:00:00", tz="UTC"),
        },
        {
            "game_id": "2025_W01_DAL@PHI",
            "season": 2025,
            "week": 1,
            "kickoff_et": pd.Timestamp("2025-09-05 00:20:00", tz="UTC"),
        },
    ]
)


class TestTheGoldBuilderReadsTheCaptureTimeNeverTheLabel:
    """End to end through ``build_features`` (retargeted by Plan 33.2-14).

    Owner ruling 2026-09-22: a line counts for a game only with a RECORDED capture time
    (``created_at``) at or before its lock, and ``snapshot_ts`` is a label, never an
    information time. So the spelling defect can no longer reach gold -- the gold
    builder does not parse the label at all -- and what stays pinned is its OUTCOME: a
    stored line that WAS known before the lock reaches the feature, whichever spelling its
    label uses, and one whose only time is the label does not.
    """

    @staticmethod
    def _build(monkeypatch: pytest.MonkeyPatch, odds: pd.DataFrame) -> pd.DataFrame:
        import features.market_anchors as module

        monkeypatch.setattr(module, "load_dataframe", lambda *a, **k: odds)
        return (
            MarketAnchorFeaturesCalculator()
            .build_features(
                _BUILT_GAMES, pd.Timestamp("2030-01-01", tz="UTC").to_pydatetime()
            )
            .set_index("game_id")
        )

    def test_a_pre_lock_capture_reaches_the_feature_whatever_its_label_spelling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built = self._build(monkeypatch, _captured_odds_frame())

        assert built.loc["2025_W01_DAL@PHI", "snapshot_spread"] == -7.5
        assert built.loc["2025_W01_DAL@PHI", "snapshot_total"] == 51.0
        assert built.loc["2018_W03_LAC@LA", "snapshot_spread"] == -3.0
        assert built.loc["2018_W03_LAC@LA", "snapshot_total"] == 46.5

    def test_a_row_whose_only_time_is_its_label_is_the_honest_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stored 2018-2024 shape: a label, and created_at NULL."""
        odds = _odds_frame()
        odds["created_at"] = pd.NaT
        built = self._build(monkeypatch, odds)

        assert built["snapshot_spread"].isna().all()
        assert built["snapshot_ml_prob_home_fair"].isna().all()

    def test_a_capture_after_the_lock_is_the_honest_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guard the guard: the fence must still FENCE, or these tests pass for free."""
        odds = _captured_odds_frame()
        odds["created_at"] = pd.Timestamp("2026-09-05 04:59:49", tz="UTC")
        built = self._build(monkeypatch, odds)

        assert built["snapshot_spread"].isna().all(), (
            "a line captured AFTER its game's lock still reached the feature, so the "
            "time fence is no longer fencing"
        )
