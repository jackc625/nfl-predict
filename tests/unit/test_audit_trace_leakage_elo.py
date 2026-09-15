"""AUDIT-03/04 deep hand-trace tests -- area 1 (LeakageGate + temporal fence)
and area 4 (Elo snapshot-then-update).

Phase 20 plan 20-04 (D-08). These codify the two highest-stakes fragile-area
hand-traces as re-runnable assertions (D-12 harness): leakage silently inflates
every backtest number, and Elo feeds every target. Per D-10 these CATALOG the
as-found behavior -- they do not fix anything.

Method (D-07 depth): synthetic fixtures with known answers (mirroring the
analog tests test_leakage_gate.py / test_elo_no_leakage.py) PLUS targeted
real-row assertions against current gold / silver. External Elo references
(nfelo/538-style) are corroborating only -- we trace for order-of-magnitude and
the internal snapshot-then-update invariant, NOT bit-exact (A5).
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from data.storage import load_dataframe
from features.validation import LeakageGate, LeakageViolation
from ratings.elo import EloRatingSystem, is_divisional_game
from tests.phase33_state import ELO_SEASON_COVERAGE

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

# Leakage-keyword columns that must NOT reach a deployed model feature_list.
LEAKAGE_KEYWORDS_SAMPLE = ["result", "margin", "final"]

GOLD_MATRICES = {"wp": "features_wp", "ats": "features_ats", "ou": "features_ou"}


@pytest.fixture(scope="module")
def gate() -> LeakageGate:
    return LeakageGate()


@pytest.fixture(scope="module")
def gold_wp() -> pd.DataFrame:
    return load_dataframe("features_wp", layer="gold")


@pytest.fixture(scope="module")
def elo_snapshots() -> pd.DataFrame:
    return load_dataframe("elo_game_snapshots", layer="silver")


def _real_2024_source_frame() -> pd.DataFrame:
    """Build a small per-builder-style source frame from real 2024 wk1 games.

    check_time_fence inspects game_date / kickoff_et / snapshot_ts; gold strips
    those, so we reconstruct a builder-style frame from silver games (which DO
    carry kickoff_et) for the time-fence trace.
    """
    games = load_dataframe("games", layer="silver")
    g = games[(games["season"] == 2024) & (games["week"] == 1)].copy()
    return g[["game_id", "season", "week", "home_team", "away_team", "kickoff_et"]]


# ===========================================================================
# AREA 1 -- LeakageGate + temporal fence
# ===========================================================================


class TestArea1TemporalFence:
    """check_time_fence holds on a real 2024 row and catches injected future."""

    def test_time_fence_passes_on_real_2024_rows(self, gate):
        """A real 2024 wk1 builder frame with an as_of AFTER the games passes."""
        src = _real_2024_source_frame()
        assert len(src) > 0, "expected real 2024 wk1 games in silver"

        # as_of well after the latest 2024 wk1 kickoff -- nothing is in the future
        latest_kickoff = pd.to_datetime(src["kickoff_et"]).max()
        as_of = pd.Timestamp(latest_kickoff) + pd.Timedelta(days=1)

        # Should NOT raise
        gate.check_time_fence(src, as_of.to_pydatetime(), "team_form_real_2024")

    def test_time_fence_raises_on_injected_future_kickoff(self, gate):
        """Injecting a kickoff_et AFTER as_of makes check_time_fence RAISE.

        This is the core leakage guard: a builder must never emit a row whose
        timestamp is later than the time-fence cutoff.
        """
        src = _real_2024_source_frame().copy()
        kickoffs = pd.to_datetime(src["kickoff_et"])
        # Cutoff just before the earliest real kickoff so the real rows are fine,
        as_of = (kickoffs.min() - pd.Timedelta(hours=1)).to_pydatetime()

        # ...then inject one clearly-future row (a full week after the cutoff).
        future_row = src.iloc[0].copy()
        future_row["game_id"] = "INJECTED_FUTURE"
        future_row["kickoff_et"] = kickoffs.min() + pd.Timedelta(days=7)
        injected = pd.concat([src, future_row.to_frame().T], ignore_index=True)

        with pytest.raises(LeakageViolation) as exc_info:
            gate.check_time_fence(injected, as_of, "team_form_injected")

        details = exc_info.value.details
        assert details["violation_type"] == "time_fence"
        assert details["affected_rows"] >= 1
        assert details["column"] in {"kickoff_et", "game_date"}

    def test_synthetic_clean_vs_leaked(self, gate):
        """Known-answer synthetic: clean frame passes, future-dated frame raises."""
        as_of = datetime(2024, 10, 4, 18, 0, 0)
        clean = pd.DataFrame(
            {
                "game_id": ["G1", "G2"],
                "kickoff_et": [
                    as_of - timedelta(days=7),
                    as_of - timedelta(days=1),
                ],
            }
        )
        leaked = pd.DataFrame(
            {
                "game_id": ["G1", "G2"],
                "kickoff_et": [
                    as_of - timedelta(days=1),
                    as_of + timedelta(days=6),  # FUTURE
                ],
            }
        )
        gate.check_time_fence(clean, as_of, "synthetic_clean")  # no raise
        with pytest.raises(LeakageViolation):
            gate.check_time_fence(leaked, as_of, "synthetic_leaked")


class TestArea1LeakageKeywords:
    """Leakage-keyword columns must not reach a deployed model feature_list."""

    def test_no_leakage_keyword_reaches_deployed_model(self):
        """F-LEAK-01 trace: home_margin exists in features_ats gold as a benign
        label-sibling, but NO result/margin/final column is in any deployed
        model feature_list (so it never reaches a model)."""
        import json
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent.parent
        latest = json.loads(
            (project_root / "artifacts" / "latest.json").read_text(encoding="utf-8")
        )

        reaching = []
        for target in GOLD_MATRICES:
            artifact_dir = latest.get(target)
            if not artifact_dir:
                continue
            fl_path = project_root / "artifacts" / artifact_dir / "feature_list.json"
            if not fl_path.exists():
                continue
            raw = json.loads(fl_path.read_text(encoding="utf-8"))
            fl = raw if isinstance(raw, list) else raw.get("feature_list", [])
            for col in fl:
                if any(kw in col.lower() for kw in LEAKAGE_KEYWORDS_SAMPLE):
                    reaching.append(f"{target}:{col}")

        assert reaching == [], (
            f"Leakage-keyword columns reach a deployed model feature_list: {reaching}"
        )

    def test_home_margin_is_label_sibling_not_a_feature(self, gold_wp):
        """home_margin (the raw final margin) is NOT in features_wp/ou gold and,
        where present (features_ats), is a label-derivation sibling excluded
        from the model's feature columns. Confirms the gate flag is a benign
        label column, not a leaked feature (cataloged, not fixed -- D-10)."""
        # features_wp must carry no margin/result/final column at all
        leak_cols = [
            c
            for c in gold_wp.columns
            if any(kw in c.lower() for kw in LEAKAGE_KEYWORDS_SAMPLE)
        ]
        assert leak_cols == [], (
            f"features_wp gold unexpectedly carries leakage-keyword columns: "
            f"{leak_cols}"
        )

        # features_ats carries home_margin as a label-sibling (point margin)
        ats = load_dataframe("features_ats", layer="gold")
        assert "home_margin" in ats.columns
        # It holds real point differentials (integers-ish), i.e. a label
        assert ats["home_margin"].dropna().abs().max() > 1.5


# ===========================================================================
# AREA 4 -- Elo snapshot-then-update
# ===========================================================================


def _make_games(matchups, season=2023):
    rows = []
    for i, (home, away, hs, as_) in enumerate(matchups):
        rows.append(
            {
                "game_id": f"T_{season}_{i:02d}_{away}@{home}",
                "season": season,
                "week": (i // 2) + 1,
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": as_,
                "kickoff_et": datetime(season, 9, 7 + i, 13, 0),
            }
        )
    return pd.DataFrame(rows)


def _snapshot_then_update(games_df, season):
    """Replicate the build_elo snapshot-then-update loop: capture PRE-game
    ratings, then update with the result."""
    elo = EloRatingSystem()
    elo.apply_season_carryover(season)
    snaps = []
    for _, game in games_df.sort_values("kickoff_et").iterrows():
        home, away = game["home_team"], game["away_team"]
        home_pre = elo.get_or_create_rating(home, season).rating
        away_pre = elo.get_or_create_rating(away, season).rating
        snaps.append(
            {
                "game_id": game["game_id"],
                "home_team": home,
                "away_team": away,
                "home_elo_pre": home_pre,
                "away_elo_pre": away_pre,
            }
        )
        elo.update_ratings(
            home_team=home,
            away_team=away,
            home_score=int(game["home_score"]),
            away_score=int(game["away_score"]),
            season=season,
            game_date=game["kickoff_et"],
            game_id=game["game_id"],
            is_divisional=is_divisional_game(home, away),
        )
    return pd.DataFrame(snaps)


class TestArea4EloSnapshotThenUpdate:
    """Elo features are the PRE-game snapshot, not post-game state."""

    def test_snapshot_is_pre_game_not_post_game(self):
        """KC wins games 1 and 2, so KC's pre-game Elo for game 3 must be
        strictly higher than its pre-game Elo for game 2 -- i.e. the snapshot
        reflects state BEFORE the game's own result was applied."""
        games = _make_games(
            [
                ("KC", "BUF", 28, 21),
                ("KC", "PHI", 35, 14),
                ("KC", "SF", 24, 17),
            ]
        )
        snaps = _snapshot_then_update(games, 2023)
        g2 = snaps[snaps["game_id"].str.contains("PHI@KC")].iloc[0]
        g3 = snaps[snaps["game_id"].str.contains("SF@KC")].iloc[0]
        assert g3["home_elo_pre"] > g2["home_elo_pre"], (
            "KC won game 2; its pre-game Elo for game 3 must be higher "
            "(snapshot-then-update, no post-game leakage)"
        )

    def test_season_boundary_regresses_25pct_toward_1500(self):
        """Season carryover regresses each rating 25% toward 1500 (the
        validate_temporal_consistency invariant: new = old*0.75 + 1500*0.25)."""
        elo = EloRatingSystem()
        # Seed a team well above 1500 in 2023
        rating = elo.get_or_create_rating("KC", 2023)
        rating.rating = 1700.0
        rating.season = 2023

        elo.apply_season_carryover(2024)
        carried = elo.ratings["KC"].rating
        expected = 1700.0 * 0.75 + 1500.0 * 0.25  # = 1650.0 (25% regression)
        assert carried == pytest.approx(expected, abs=1e-6)
        # 25% of the 200-pt edge (50 pts) is shed
        assert (1700.0 - carried) == pytest.approx(50.0, abs=1e-6)

    def test_check_elo_ordering_passes_and_raises(self, gate):
        """check_elo_ordering passes on chronological data, raises out-of-order."""
        ok = pd.DataFrame(
            {
                "season": [2024, 2024, 2024],
                "team": ["BUF", "BUF", "BUF"],
                "game_date": [
                    datetime(2024, 9, 8),
                    datetime(2024, 9, 15),
                    datetime(2024, 9, 22),
                ],
            }
        )
        gate.check_elo_ordering(ok)  # no raise

        bad = pd.DataFrame(
            {
                "season": [2024, 2024, 2024],
                "team": ["BUF", "BUF", "BUF"],
                "game_date": [
                    datetime(2024, 9, 8),
                    datetime(2024, 9, 22),  # out of order
                    datetime(2024, 9, 15),
                ],
            }
        )
        with pytest.raises(LeakageViolation) as exc_info:
            gate.check_elo_ordering(bad)
        assert exc_info.value.details["violation_type"] == "elo_ordering"


class TestArea4EloRealData:
    """Real-data corroboration: pre-game snapshots + pre-2018 N/A handling."""

    def test_real_2024_snapshot_order_of_magnitude(self, elo_snapshots):
        """2024_W01_BAL@KC pre-game Elo is in the plausible NFL band (both are
        strong teams ~1650-1750) and KC (home) is favored. Order-of-magnitude
        + internal sanity, not bit-exact vs an external source (A5)."""
        row = elo_snapshots[elo_snapshots["game_id"] == "2024_W01_BAL@KC"]
        assert len(row) == 1, "expected one snapshot for 2024_W01_BAL@KC"
        r = row.iloc[0]
        # NFL Elo centers on 1500; strong teams sit ~1600-1800
        assert 1400.0 < r["home_elo_pre"] < 1900.0
        assert 1400.0 < r["away_elo_pre"] < 1900.0
        # Both KC and BAL were strong entering 2024 -> both above league mean
        assert r["home_elo_pre"] > 1550.0
        assert r["away_elo_pre"] > 1550.0
        # KC at home, narrowly favored -> elo_prob_home in a sensible band
        assert 0.5 < r["elo_prob_home"] < 0.75

    def test_every_gold_game_carries_an_elo_snapshot(self, elo_snapshots):
        """Every gold game has a pre-game snapshot, back to the 2002 burn-in floor.

        THIS TEST USED TO ASSERT THE OPPOSITE, and the inversion is the point.
        Under the name ``test_pre_burn_in_games_have_no_spurious_elo_snapshot`` it
        required ``elo_game_snapshots["season"].min() == 2018`` and required that NO
        pre-2018 gold game carry a snapshot, on the reading that pre-2018 games were
        "snapshot-less by design (N/A)". Per this module's own D-10 note, these
        area-4 tests CATALOG as-found behaviour rather than fix it -- and the
        as-found behaviour it catalogued was a defect, not a design.

        WHAT WAS ACTUALLY MEASURED (Phase 33, F-03). The 2,227-row 2018-2025
        snapshot table was written by ``tests/integration/test_elo_integration.py``
        running against production with no sandbox, at least four times during Phase
        31 -- not by a deliberate market-context floor. Because gold's Elo columns
        are a LEFT JOIN off that table, 4,288 of 6,499 gold rows carried a
        FABRICATED 0.0 Elo (all of 2002-2017 plus week 1 of 2018), and the deployed
        ATS model trained with 61.2% of its rows carrying that zero across six Elo
        features. The claim in the old docstring that pre-2018 gold games "have
        populated Elo features" was true only in the sense that 0.0 is a value.

        Plan 33-13 re-derived the canonical 2002-2025 chain from
        ``data/silver/games.parquet`` under an owner ruling, so the invariant is now
        the one stated here. ``tests/integration/test_elo_burn_in_canonical.py``
        holds the full set of canonical checks on that chain; this is the
        gold-facing corroboration of it.
        """
        burn_in_floor = ELO_SEASON_COVERAGE[0]
        assert elo_snapshots["season"].min() == burn_in_floor, (
            f"elo_game_snapshots should start at the canonical burn-in floor "
            f"{burn_in_floor}; got {elo_snapshots['season'].min()}. A floor of 2018 "
            "specifically is the pre-Phase-33 defect: it is the span the "
            "sandbox-less integration test happened to rebuild."
        )
        snap_ids = set(elo_snapshots["game_id"])
        gold = load_dataframe("features_wp", layer="gold")

        # The burn-in half: pre-2018 gold games now carry a REAL snapshot. Before
        # the re-derivation every one of them joined to nothing and took a
        # fabricated 0.0 Elo into training.
        pre_2018 = gold[gold["season"] < 2018]
        assert len(pre_2018) > 0, "expected pre-2018 gold rows"
        unsnapshotted = [gid for gid in pre_2018["game_id"] if gid not in snap_ids]
        assert unsnapshotted == [], (
            f"{len(unsnapshotted)} pre-2018 gold game(s) still have no Elo "
            f"snapshot, e.g. {unsnapshotted[:8]}. Each of those rows LEFT JOINs to "
            "nothing and carries a fabricated Elo into every model trained on it."
        )

        # And the rest of gold, so the assertion covers the whole matrix rather
        # than only the half that used to fail.
        missing = [gid for gid in gold["game_id"] if gid not in snap_ids]
        assert missing == [], (
            f"{len(missing)} gold game(s) are missing an Elo snapshot, e.g. "
            f"{missing[:8]}"
        )

    def test_real_snapshots_are_chronologically_ordered(self, gate, elo_snapshots):
        """The full real elo_game_snapshots set, reshaped per team with a
        kickoff_et game_date proxy, passes check_elo_ordering (chronological
        within season)."""
        games = load_dataframe("games", layer="silver")[
            ["game_id", "kickoff_et"]
        ].drop_duplicates("game_id")
        merged = elo_snapshots.merge(games, on="game_id", how="left")
        merged = merged.sort_values(["season", "kickoff_et", "game_id"])
        home = merged[["season", "kickoff_et", "home_team"]].rename(
            columns={"kickoff_et": "game_date", "home_team": "team"}
        )
        away = merged[["season", "kickoff_et", "away_team"]].rename(
            columns={"kickoff_et": "game_date", "away_team": "team"}
        )
        long_df = pd.concat([home, away], ignore_index=False).sort_values(
            ["season", "game_date"]
        )
        # Should NOT raise
        gate.check_elo_ordering(long_df)
