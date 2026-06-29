"""Leakage proof for the injury signal (SIG-01 / SIG-06 / SC1).

Three-part canonical standard (D-18a), following the
`tests/unit/test_elo_no_leakage.py` + `test_snap_no_leakage.py` convention. The
InjuryBuilder (Plan 28-05) Friday-freeze fence is the load-bearing novel code:
injury rows are filtered to `date_modified <= as_of_datetime` (the Friday 6 PM ET
freeze), so revealing a Saturday/Sunday game-day report must not change any
pre-game feature value.

Parts (LIVE):
  1. Time-fence assertion -- the builder's `date_modified <= as_of_datetime`
     fence drops every post-freeze row (no fenced row is after the cutoff).
  2. Withhold-future byte-unchanged (the SC1 test) -- reveal a Saturday/Sunday
     `date_modified` row and assert qb_out_flag / backup_quality_delta /
     availability_fraction are byte-identical; a positive control proves the
     same Saturday row WOULD flip the flag if it were not fenced.
  3. Status-vocab -- the {Out, Doubtful} out-flag derivation reads
     `report_status`; {None, Questionable} do NOT trip the qb_out_flag.
"""

from datetime import datetime

import pandas as pd
import pytest

from features.injury import InjuryBuilder
from features.snaps import SnapCountBuilder

# Mahomes (the KC QB1) and a backup gsis_id used across the fixtures.
_QB1_ID = "00-0033873"
_QB2_ID = "00-0000002"

# Friday pre-freeze vs Saturday game-day report timestamps (UTC).
_FRIDAY_DM = "2023-09-08T21:00:00Z"
_SATURDAY_DM = "2023-09-09T18:00:00Z"

# A Friday 6 PM ET freeze that admits the Friday report and excludes the
# Saturday game-day update (naive -> interpreted as UTC by the builder fence).
_AS_OF_FRIDAY = datetime(2023, 9, 8, 23, 0)
# A cutoff AFTER the Saturday update -- the positive control (no fence effect).
_AS_OF_AFTER_GAMEDAY = datetime(2023, 9, 11, 0, 0)


def _injury_row(
    gsis_id: str,
    position: str,
    status: str | None,
    date_modified: str,
    team: str = "KC",
) -> dict:
    """One injury silver row (shape of `nfl.load_injuries(season).to_pandas()`)."""
    return {
        "season": 2023,
        "week": 1,
        "team": team,
        "gsis_id": gsis_id,
        "position": position,
        "full_name": "Player",
        "report_status": status,
        "date_modified": date_modified,
    }


def _injuries(rows: list[dict]) -> pd.DataFrame:
    """Assemble an injuries frame with a UTC-aware date_modified column."""
    df = pd.DataFrame(rows)
    df["date_modified"] = pd.to_datetime(df["date_modified"], utc=True)
    return df


def _mahomes_fixture(include_saturday: bool) -> pd.DataFrame:
    """KC QB1 injury fixture: a Friday `Questionable` row, optionally followed by
    a Saturday game-day `Out` downgrade for the same player.

    Args:
        include_saturday: When True, append the post-freeze Saturday `Out` row.
    """
    rows = [_injury_row(_QB1_ID, "QB", "Questionable", _FRIDAY_DM)]
    if include_saturday:
        rows.append(_injury_row(_QB1_ID, "QB", "Out", _SATURDAY_DM))
    return _injuries(rows)


def _depth_charts() -> pd.DataFrame:
    """Depth-chart fixture: Mahomes is KC QB1, a backup is KC QB2 (week 1)."""
    return pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "club_code": ["KC", "KC"],
            "position": ["QB", "QB"],
            "depth_team": ["1", "2"],
            "gsis_id": [_QB1_ID, _QB2_ID],
            "full_name": ["Patrick Mahomes", "Backup QB"],
        }
    )


def _snap_builder() -> SnapCountBuilder:
    """A SnapCountBuilder seeded with a minimal in-memory snap frame.

    Week-1 targets have no prior games, so the per-position prior shares are empty
    and availability defaults to 1.0 -- the injected frame only keeps the snap
    silver load hermetic (no parquet dependency)."""
    snaps = pd.DataFrame(
        {
            "game_id": ["2023_01_BUF_KC"],
            "pfr_player_id": ["kc_qb"],
            "player": ["kc_qb"],
            "position": ["QB"],
            "team": ["KC"],
            "opponent": ["BUF"],
            "season": [2023],
            "week": [1],
            "offense_snaps": [70.0],
            "offense_pct": [1.0],
            "defense_snaps": [0.0],
            "defense_pct": [0.0],
            "st_snaps": [0.0],
            "st_pct": [0.0],
        }
    )
    return SnapCountBuilder(snaps_df=snaps)


def _make_test_games() -> pd.DataFrame:
    """Minimal games_df for a single KC (home) vs BUF (away) week-1 game."""
    return pd.DataFrame(
        [
            {
                "game_id": "2023_01_BUF_KC",
                "season": 2023,
                "week": 1,
                "home_team": "KC",
                "away_team": "BUF",
            }
        ]
    )


def _builder(
    injuries: pd.DataFrame,
) -> InjuryBuilder:
    """An InjuryBuilder wired to the fixtures, with an empty PBP frame so the
    backup-quality path never touches nflreadpy."""
    return InjuryBuilder(
        snap_builder=_snap_builder(),
        injuries_df=injuries,
        depth_charts_df=_depth_charts(),
        pbp_df=pd.DataFrame(),
    )


class TestInjuryTimeFence:
    """LIVE: InjuryBuilder date_modified fence (Plan 28-05)."""

    def test_injury_builder_respects_date_modified_fence(self):
        """The fence drops any injury row whose date_modified is after the
        as_of_datetime freeze: no fenced row survives past the cutoff, and the
        Saturday game-day row is specifically excluded."""
        builder = _builder(_mahomes_fixture(include_saturday=True))

        fenced, dm_applied = builder.fenced_injuries(2023, 1, _AS_OF_FRIDAY)

        assert dm_applied, "the date_modified fence must be applied (column present)"
        assert len(fenced) > 0, "the Friday pre-freeze row must survive the fence"

        cutoff = pd.Timestamp(_AS_OF_FRIDAY, tz="UTC")
        dm = pd.to_datetime(fenced["date_modified"], utc=True)
        assert (dm <= cutoff).all(), (
            "time-fence violation: a fenced injury row has date_modified after "
            f"the cutoff:\n{fenced[dm > cutoff]}"
        )
        # The Saturday game-day report is specifically excluded.
        assert not (dm == pd.Timestamp(_SATURDAY_DM)).any()


class TestInjuryWithholdFuture:
    """LIVE: withhold-future byte-unchanged proof (the SC1 test)."""

    def test_revealing_gameday_report_does_not_change_features(self):
        """Building the injury features once with the Saturday game-day row
        present and once without (Friday row only) must produce byte-identical
        qb_out_flag, backup_quality_delta and availability_fraction -- revealing
        the post-freeze game-day inactive must not change a Friday-fenced
        feature."""
        feature_cols = [
            "home_qb_out_flag",
            "home_backup_quality_delta",
            "home_availability_fraction",
        ]

        withheld = _builder(_mahomes_fixture(include_saturday=False)).build_features(
            _make_test_games(), _AS_OF_FRIDAY, target_season=2023, target_week=1
        )
        revealed = _builder(_mahomes_fixture(include_saturday=True)).build_features(
            _make_test_games(), _AS_OF_FRIDAY, target_season=2023, target_week=1
        )

        # Byte-identical fenced feature values regardless of the Saturday row.
        pd.testing.assert_frame_equal(
            withheld[feature_cols], revealed[feature_cols], check_exact=True
        )
        # The fenced Friday status is `Questionable`, so the flag stays 0.0.
        assert float(revealed.iloc[0]["home_qb_out_flag"]) == 0.0

    def test_positive_control_unfenced_saturday_row_would_flip_the_flag(self):
        """Positive control: the very same Saturday `Out` row DOES set
        qb_out_flag=1 when the cutoff is moved past it -- proving the fence (not
        an inert fixture) is what protects the feature in the withhold test."""
        leaked = _builder(_mahomes_fixture(include_saturday=True)).build_features(
            _make_test_games(),
            _AS_OF_AFTER_GAMEDAY,
            target_season=2023,
            target_week=1,
        )
        assert float(leaked.iloc[0]["home_qb_out_flag"]) == 1.0


class TestInjuryStatusVocab:
    """LIVE: report_status vocabulary check (Plan 28-05)."""

    @pytest.mark.parametrize(
        ("status", "expected_flag"),
        [
            ("Out", 1.0),
            ("Doubtful", 1.0),
            ("Questionable", 0.0),
            (None, 0.0),
        ],
    )
    def test_report_status_vocabulary(self, status, expected_flag):
        """The {Out, Doubtful} out-flag derivation reads `report_status`: Out and
        Doubtful raise qb_out_flag to 1, while None and Questionable leave it 0.
        The cutoff is after the row's Friday timestamp so status (not the fence)
        is the only thing under test."""
        injuries = _injuries([_injury_row(_QB1_ID, "QB", status, _FRIDAY_DM)])
        out = _builder(injuries).build_features(
            _make_test_games(), _AS_OF_FRIDAY, target_season=2023, target_week=1
        )
        assert float(out.iloc[0]["home_qb_out_flag"]) == expected_flag
