"""The market-anchor time fence parses BOTH stored snapshot spellings, or refuses.

TEST CLASS: plain unit tests. The odds frame is built in the test and injected by
monkeypatching ``features.market_anchors.load_dataframe``; no production store is read
or written, and the module passes on a fresh checkout with no ``data/``.

THE DEFECT THESE TESTS PIN, measured on 2026-09-05 during Plan 31-11's rung 2.
``MarketAnchorFeaturesCalculator.build_features`` fenced its odds with a bare
``pd.to_datetime(column, errors="coerce")``. The stored ``snapshot_ts`` is a STRING
column holding two known spellings:

  legacy per-season   2018-09-19T18:00:00-04:00     (ISO, 'T', Eastern offset)
  tz-aware write      2025-08-29 22:00:00+00:00     (space-separated, UTC offset)

``pd.to_datetime`` infers ONE format from the first element. Every row in the other
spelling became ``NaT``; ``NaT <= cutoff`` is False, so those rows left the fenced
population; a game with no surviving odds row falls through to
``_default_compressed_market_features``. All 285 rows of the freshly ingested 2025
season were destroyed that way, and gold's 2025 market anchors came out at the neutral
default -- 0 of 285 non-zero, WORSE than the 3 of 285 they replaced. The ingest reached
silver and never reached gold, and nothing said so.

Two things are proved here, and the second is what makes the first non-vacuous:

1. the fixed parse handles both spellings and refuses what it cannot parse;
2. the OLD parse really did destroy the mixed row -- so the fix is not guarding a
   hazard that was never there.
"""

from __future__ import annotations

import pandas as pd
import pytest

from features.market_anchors import MarketAnchorFeaturesCalculator

# The two spellings the live column holds, with the SAME instant expressed both ways so a
# comparison between them is meaningful: 18:00 Eastern on 2025-08-29 is 22:00 UTC.
LEGACY_SPELLING = "2018-09-19T18:00:00-04:00"
TZ_AWARE_SPELLING = "2025-08-29 22:00:00+00:00"
TZ_AWARE_AS_LEGACY = "2025-08-29T18:00:00-04:00"


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


class TestTheOldParseReallyDestroyedTheMixedRow:
    """Without this, the fix below is a guard against a hazard nobody has shown exists."""

    def test_a_bare_to_datetime_coerces_the_second_spelling_to_nat(self) -> None:
        column = pd.Series([LEGACY_SPELLING, TZ_AWARE_SPELLING])

        parsed = pd.to_datetime(column, errors="coerce")

        assert pd.isna(parsed.iloc[1]), (
            "pd.to_datetime no longer NaTs the space-separated spelling when it follows "
            "the ISO one. If pandas has fixed this, the fence's dedicated parse may be "
            "removable -- but check that BEFORE removing it, because the failure it "
            "prevents is silent."
        )
        assert not pd.isna(parsed.iloc[0])

    def test_a_nat_never_survives_a_less_than_or_equal_fence(self) -> None:
        """The second half of the mechanism: NaT does not merely sort oddly, it VANISHES."""
        cutoff = pd.Timestamp("2030-01-01", tz="UTC")
        parsed = pd.to_datetime(
            pd.Series([LEGACY_SPELLING, TZ_AWARE_SPELLING]), errors="coerce"
        )
        kept = parsed <= cutoff.tz_convert(parsed.dt.tz)

        assert list(kept) == [True, False], (
            "a NaT row is not dropped by the fence, so the 2025 season would not have "
            "been lost this way and this module is pinning the wrong mechanism"
        )


class TestTheFixedParseHandlesBothSpellings:
    def test_neither_spelling_becomes_nat(self) -> None:
        parsed = MarketAnchorFeaturesCalculator._parse_snapshot_column(
            pd.Series([LEGACY_SPELLING, TZ_AWARE_SPELLING])
        )

        assert parsed.isna().sum() == 0
        assert str(parsed.dt.tz) == "UTC"

    def test_the_two_spellings_of_one_instant_parse_to_one_instant(self) -> None:
        """Proves the parse is instant-preserving, not merely non-crashing."""
        parsed = MarketAnchorFeaturesCalculator._parse_snapshot_column(
            pd.Series([TZ_AWARE_SPELLING, TZ_AWARE_AS_LEGACY])
        )

        assert parsed.iloc[0] == parsed.iloc[1]

    def test_a_naive_value_is_anchored_in_EASTERN_not_utc(self) -> None:
        """The anchor choice is the point of the shared parse path, so it is asserted.

        An unqualified wall-clock time in this project's odds data is market-local.
        Reading it as UTC moves it four or five hours -- across the 6 PM freeze.
        """
        parsed = MarketAnchorFeaturesCalculator._parse_snapshot_column(
            pd.Series(["2025-08-29 18:00:00"])
        )

        assert parsed.iloc[0] == pd.Timestamp(TZ_AWARE_SPELLING)

    def test_an_unparseable_value_raises_and_names_itself(self) -> None:
        with pytest.raises(ValueError) as error:
            MarketAnchorFeaturesCalculator._parse_snapshot_column(
                pd.Series([LEGACY_SPELLING, "not-a-timestamp"])
            )

        message = str(error.value)
        assert "not-a-timestamp" in message
        assert "neutral default" in message, (
            "the refusal does not say what the silent alternative would have cost, so a "
            "future reader can talk themselves back into coercing"
        )

    def test_a_null_is_refused_rather_than_treated_as_fresh(self) -> None:
        with pytest.raises(ValueError):
            MarketAnchorFeaturesCalculator._parse_snapshot_column(
                pd.Series([LEGACY_SPELLING, None])
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
    information time. So the spelling defect above can no longer reach gold -- the gold
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


class TestTheDeprecatedPathUsesTheSameOneParse:
    """WR-01: ``identify_opening_lines`` and the snapshot selection kept the bare parse.

    The snapshot method was renamed ``select_snapshot_lines_at_lock`` by Plan 33.2-02, which
    also deleted its one-global-Friday cutoff: each game is now cut at its own day-before lock
    (D33.2-01). The parse property below is unchanged.

    ``build_features`` was fixed above, but the two DEPRECATED methods were not -- and they
    are the ones the scheduled run reaches. ``pipeline/steps.py::step_build_market_anchors``
    calls ``build_market_anchor_features``, which calls both of them. Measured on live silver
    on 2026-09-09: the bare parse NaT'd 1,855 of 2,140 rows, and a NaT fails both the
    ``>= min_opening_hours`` fence and the ``<= cutoff`` one, so 87% of games lost their odds
    and fell through to ``_default_market_features`` with no count and no warning.
    """

    _GAMES = pd.DataFrame(
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
                "kickoff_et": pd.Timestamp("2025-09-04 20:20:00", tz="UTC"),
            },
        ]
    )

    @classmethod
    def _patch(cls, monkeypatch: pytest.MonkeyPatch) -> None:
        import features.market_anchors as module

        def _dispatch(table: str, *_args: object, **_kwargs: object) -> pd.DataFrame:
            return cls._GAMES.copy() if table == "games" else _odds_frame()

        monkeypatch.setattr(module, "load_dataframe", _dispatch)

    def test_both_spellings_survive_identify_opening_lines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch)

        opening = MarketAnchorFeaturesCalculator().identify_opening_lines(_odds_frame())

        assert set(opening["game_id"]) == {"2018_W03_LAC@LA", "2025_W01_DAL@PHI"}, (
            "a game vanished from the opening-line population. Under the bare parse the "
            "space-separated spelling became NaT, NaT >= min_opening_hours is False, and "
            "the game silently fell through to the neutral market default."
        )

    def test_both_spellings_survive_select_snapshot_lines_at_lock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both games keep their pre-lock snapshot, each judged at its OWN lock.

        Stronger than the check it replaces, which asserted only the legacy-spelled game:
        under a per-game lock both rows are before their games' locks, so both must survive.
        Admission reads each row's recorded capture (Plan 33.2-14), so the rows carry one.
        """
        self._patch(monkeypatch)

        snapshots = MarketAnchorFeaturesCalculator().select_snapshot_lines_at_lock(
            _captured_odds_frame(), self._GAMES
        )

        assert set(snapshots["game_id"]) == {"2018_W03_LAC@LA", "2025_W01_DAL@PHI"}

    def test_an_unparseable_snapshot_is_refused_rather_than_silently_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deprecated path now REFUSES what it cannot read, like the fixed one."""
        import features.market_anchors as module

        broken = _odds_frame()
        broken.loc[1, "snapshot_ts"] = "not-a-timestamp"
        monkeypatch.setattr(
            module,
            "load_dataframe",
            lambda table, *a, **k: (
                self._GAMES.copy() if table == "games" else broken.copy()
            ),
        )

        with pytest.raises(ValueError, match="not-a-timestamp"):
            MarketAnchorFeaturesCalculator().identify_opening_lines(broken)
