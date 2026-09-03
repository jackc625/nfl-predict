"""Wave-0 pre-ingest gates for the Phase-31 one-shot 2025 run (Plan 31-02).

This module holds the assertions that must hold BEFORE anything writes production silver or
rebuilds the 2025 gold slice. It writes nothing itself: Plan 31-11 owns the only silver write
in this phase, and every measurement here is either read-only or confined to ``tmp_path``.

What it pins:

* **The ATS residual bias does not drift.** Plan 31-02 Task 2 re-derived the bias from the
  DEPLOYED artifact and APPENDED the result to ``tests/phase31_state.py``. This module asserts
  the live re-score still equals those constants to 17 significant digits, so a later
  disagreement is a FAILURE rather than a silently different number in a later run.
* **The completeness gate (SPEC R2).** Zero 2025 rows, or fewer than 285 games carrying a
  total, is a HARD STOP before gold is touched.
* **The no-extra-rows gate (DEFECT-3).** The synthetic ``2025_W01_TEST@HOME`` fixture sitting
  in production silver passes the OUM-06 sportsbook allowlist and must be caught by a
  ``game_id`` shape-and-existence gate instead.
* **The DEFECT-2 dry run (assumption A2).** Whether an all-season re-ingest ADDS ``LA``-keyed
  Rams rows beside the stored ``LAR`` ones or REPLACES them, measured by upserting into a
  ``tmp_path`` copy -- never by writing production silver.

TEST CLASS: integration. Reads ``data/gold``, ``artifacts/`` and ``data/silver`` -- all
gitignored runtime state -- behind evidence-backed skip guards whose reasons are registered in
``tests/conftest._EVIDENCE_SKIP_MARKERS``, so a control that did not run is named in the
terminal summary rather than disappearing into a green suite (WR-10).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from tests.phase31_state import (
    ATS_RESIDUAL_BY_SEASON,
    ATS_RESIDUAL_FIELDS,
    ATS_RESIDUAL_POOLED,
    ATS_RESIDUAL_POOLED_PROVENANCE,
)

_GOLD_ATS = Path("data/gold/features_ats.parquet")
_GOLD_OU = Path("data/gold/features_ou.parquet")
_SILVER_ODDS = Path("data/silver/odds_snapshot.parquet")
_ARTIFACTS_MANIFEST = Path("artifacts/latest.json")

# The ONE explicit specifier every Phase-31 float is compared through.
_FMT = "{:.17g}"

# The synthetic fixture row DEFECT-3 names. Identified here so its removal is a NAMED
# pre-ingest step rather than an accident of downstream join semantics.
_SYNTHETIC_FIXTURE_GAME_ID = "2025_W01_TEST@HOME"


def _require_rescore_inputs() -> None:
    """Skip with a REGISTERED reason when the re-score's gitignored inputs are absent."""
    if not _GOLD_ATS.is_file():
        pytest.skip(
            "live gold absent (features_ats) -- data/ is gitignored runtime state."
        )
    if not _ARTIFACTS_MANIFEST.is_file():
        pytest.skip("production manifest not present at artifacts/latest.json")


def _require_silver_odds() -> None:
    if not _SILVER_ODDS.is_file():
        pytest.skip(
            "live silver odds not present at data/silver/odds_snapshot.parquet -- "
            "data/ is gitignored runtime state."
        )


def _require_gold_ou() -> None:
    if not _GOLD_OU.is_file():
        pytest.skip(
            "live gold absent (features_ou) -- data/ is gitignored runtime state."
        )


def _load_live_schedule(season: int) -> pd.DataFrame:
    """The live nflreadpy schedule for *season*, or a REGISTERED skip when it cannot load.

    An offline checkout is the same fact as an absent gitignored run record: the control did
    not run. Its reason is registered in ``tests/conftest._EVIDENCE_SKIP_MARKERS`` so the
    terminal summary names it, rather than letting it pass for an ordinary environment skip.
    """
    import nflreadpy as nfl

    try:
        return nfl.load_schedules([season]).to_pandas()
    except Exception as exc:
        pytest.skip(
            f"the live nflreadpy {season} schedule could not be loaded on this checkout "
            f"(offline or upstream unavailable): {exc}"
        )


@pytest.fixture(scope="module")
def ats_bias_block() -> dict:
    """The LIVE re-score of the deployed ATS artifact (read-only, no data/ writes)."""
    _require_rescore_inputs()
    from scripts.audit_odds_preingest import (
        assert_silver_unchanged,
        measure_ats_residual_bias,
        silver_parquet_digests,
    )

    before = silver_parquet_digests()
    block = measure_ats_residual_bias()
    assert_silver_unchanged(before, "measure_ats_residual_bias (test)")
    return block


class TestTheAppendedATSResidualConstantsStillHold:
    """The live re-score must equal the constants Plan 31-02 appended, to 17 significant digits.

    The pre-registration Plan 31-05 freezes binds to these numbers. If the deployed artifact,
    the gold matrix or the residual contract moves, this test is where that shows up -- as a
    failure naming both figures, not as a quietly different number in a later run.
    """

    def test_the_scored_artifact_is_the_one_the_constants_name(
        self, ats_bias_block: dict
    ) -> None:
        expected = ATS_RESIDUAL_POOLED_PROVENANCE["artifact_id"]
        assert ats_bias_block["artifact_id"] == expected, (
            f"the live re-score scored artifact {ats_bias_block['artifact_id']!r} but "
            f"tests/phase31_state.py records {expected!r}. The bias the pre-registration "
            "freezes must be attributable to a NAMED deployed model (T-31-08); a different "
            "artifact means a different number, and the constants must be re-measured by a "
            "new plan rather than edited in place."
        )

    @pytest.mark.parametrize("season", sorted(ATS_RESIDUAL_BY_SEASON))
    def test_each_per_season_figure_matches(
        self, ats_bias_block: dict, season: int
    ) -> None:
        live = ats_bias_block["by_season"][str(season)]
        for field, appended in zip(
            ATS_RESIDUAL_FIELDS, ATS_RESIDUAL_BY_SEASON[season], strict=True
        ):
            expected = str(appended) if field == "n" else _FMT.format(float(appended))
            assert str(live[field]) == expected, (
                f"season {season} field '{field}' drifted: live {live[field]} vs the "
                f"value {expected} appended by Plan 31-02 Task 2. Every per-season mean is "
                "carried into the pre-registration verbatim, including the NEGATIVE 2022 "
                "season, so a drift here changes what the pre-registration states."
            )

    def test_the_pooled_figure_matches(self, ats_bias_block: dict) -> None:
        live = ats_bias_block["pooled"]
        for field, appended in zip(
            ATS_RESIDUAL_FIELDS, ATS_RESIDUAL_POOLED, strict=True
        ):
            expected = str(appended) if field == "n" else _FMT.format(float(appended))
            assert str(live[field]) == expected, (
                f"pooled field '{field}' drifted: live {live[field]} vs the value "
                f"{expected} appended by Plan 31-02 Task 2."
            )

    def test_the_pooled_direction_is_positive_and_only_the_pooled_sign_is_gated(
        self, ats_bias_block: dict
    ) -> None:
        """The guard the pre-registration rests on, and the guard it deliberately does NOT have.

        The O/U residual contract corrects a NEGATIVE bias, so an ATS sign guard copied from it
        would assert the wrong direction. The pooled ATS mean is POSITIVE and that is asserted.
        The per-season signs are NOT asserted: 2022 is negative, and a per-season gate would
        hard-stop the phase on a fact that is simply true (REVIEW-ATS).
        """
        assert float(ats_bias_block["pooled"]["mean"]) > 0.0
        assert ats_bias_block["pooled_direction_asserted"] is True
        assert ats_bias_block["per_season_sign_asserted"] is False
        assert ats_bias_block["numeric_tolerance_used"] is None, (
            "a numeric tolerance appeared in the ATS bias measurement. Choosing a magnitude "
            "threshold after seeing the measured values is the post-hoc threshold selection "
            "this phase forbids everywhere else (T-31-08c)."
        )
        assert ats_bias_block["seasons_with_negative_mean"] == [2022], (
            "the set of negative-mean seasons moved. 2022 being negative is ON THE RECORD by "
            "design; if that changes, the pre-registration's per-season table changes with it."
        )


class TestTheCompletenessGate:
    """SPEC R2: an empty or thin 2025 population HARD STOPS before gold is touched."""

    @staticmethod
    def _frame(n_games: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_id": [
                    f"2025_W{i // 16 + 1:02d}_AAA@BBB{i}" for i in range(n_games)
                ],
                "total": [45.5] * n_games,
            }
        )

    def test_zero_rows_is_a_hard_stop(self) -> None:
        from scripts.audit_odds_preingest import assert_2025_odds_completeness

        with pytest.raises(ValueError, match="ZERO rows"):
            assert_2025_odds_completeness(pd.DataFrame({"game_id": [], "total": []}))

    def test_284_games_with_a_total_is_a_hard_stop(self) -> None:
        from scripts.audit_odds_preingest import assert_2025_odds_completeness

        with pytest.raises(ValueError, match="only 284 games with a total"):
            assert_2025_odds_completeness(self._frame(284))

    def test_285_games_with_a_total_passes(self) -> None:
        from scripts.audit_odds_preingest import assert_2025_odds_completeness

        assert assert_2025_odds_completeness(self._frame(285)) == 285

    def test_a_game_whose_total_is_null_does_not_count(self) -> None:
        """The floor is games CARRYING A TOTAL, not games merely present."""
        from scripts.audit_odds_preingest import assert_2025_odds_completeness

        frame = self._frame(285)
        frame.loc[0, "total"] = None
        with pytest.raises(ValueError, match="only 284 games with a total"):
            assert_2025_odds_completeness(frame)

    def test_the_live_2025_schedule_meets_the_floor(self) -> None:
        """The floor is not hypothetical: nflreadpy's live 2025 schedule carries 285 games."""
        from scripts.audit_odds_preingest import MIN_2025_GAMES_WITH_TOTAL

        schedule = _load_live_schedule(2025)

        assert len(schedule) == MIN_2025_GAMES_WITH_TOTAL, (
            f"the live nflreadpy 2025 schedule carries {len(schedule)} games, not "
            f"{MIN_2025_GAMES_WITH_TOTAL}. The SPEC R2 floor was set from this source; if the "
            "source moved, the floor is measuring the wrong thing."
        )
        for column in ("total_line", "spread_line", "home_moneyline", "away_moneyline"):
            n_present = int(schedule[column].notna().sum())
            assert n_present == MIN_2025_GAMES_WITH_TOTAL, (
                f"the live 2025 schedule carries {column} on {n_present} of "
                f"{len(schedule)} games. The one-shot run needs a total, a spread and both "
                "moneylines on every game; a partial column silently thins the population."
            )


class TestTheNoExtraRowsGate:
    """DEFECT-3: the synthetic fixture row the OUM-06 sportsbook allowlist admits."""

    def test_the_allowlist_provably_admits_the_synthetic_row(self) -> None:
        """The premise of the whole gate: a forged game_id under a legitimate sportsbook."""
        _require_silver_odds()
        from backtest.ou_divergence import _ALLOWED_SPORTSBOOKS

        odds = pd.read_parquet(_SILVER_ODDS)
        fixture = odds[odds["game_id"] == _SYNTHETIC_FIXTURE_GAME_ID]
        assert len(fixture) == 1, (
            f"expected exactly one {_SYNTHETIC_FIXTURE_GAME_ID} row in production silver; "
            f"found {len(fixture)}. This gate exists because that row is there."
        )
        assert fixture.iloc[0]["sportsbook"] in _ALLOWED_SPORTSBOOKS, (
            "the synthetic row's sportsbook is no longer on the OUM-06 allowlist, so the hole "
            "this gate closes may have changed shape. Re-read DEFECT-3 before relaxing "
            "anything."
        )
        assert not bool(fixture.iloc[0]["is_live"])

    def test_it_raises_on_the_current_silver_2025_slice(self) -> None:
        _require_silver_odds()
        _require_gold_ou()
        from scripts.audit_odds_preingest import assert_no_synthetic_game_ids

        odds = pd.read_parquet(_SILVER_ODDS)
        slice_2025 = odds[odds["game_id"].astype(str).str.startswith("2025_")]
        features_ou = pd.read_parquet(_GOLD_OU, columns=["game_id"])

        with pytest.raises(ValueError, match=_SYNTHETIC_FIXTURE_GAME_ID):
            assert_no_synthetic_game_ids(slice_2025, features_ou)

    def test_it_returns_once_the_fixture_row_is_excluded(self) -> None:
        """Removing the row is a NAMED pre-ingest step, not an accident of join semantics."""
        _require_silver_odds()
        _require_gold_ou()
        from scripts.audit_odds_preingest import assert_no_synthetic_game_ids

        odds = pd.read_parquet(_SILVER_ODDS)
        slice_2025 = odds[odds["game_id"].astype(str).str.startswith("2025_")]
        cleaned = slice_2025[slice_2025["game_id"] != _SYNTHETIC_FIXTURE_GAME_ID]

        assert_no_synthetic_game_ids(
            cleaned, pd.read_parquet(_GOLD_OU, columns=["game_id"])
        )

    def test_a_well_formed_but_orphan_id_is_also_caught(self) -> None:
        """Shape alone is not enough: a real-looking id naming no gold game is still forged."""
        from scripts.audit_odds_preingest import assert_no_synthetic_game_ids

        with pytest.raises(ValueError, match="ORPHAN"):
            assert_no_synthetic_game_ids(
                pd.DataFrame({"game_id": ["2025_W01_BUF@MIA"]}),
                pd.DataFrame({"game_id": ["2025_W01_KC@DEN"]}),
            )


class TestTheDefect2DryRun:
    """A2, settled by measurement inside ``tmp_path`` -- production silver is never written."""

    def test_a_rams_reingest_adds_rows_rather_than_replacing_them(
        self, tmp_path: Path
    ) -> None:
        _require_silver_odds()
        from data.storage import upsert_silver
        from scripts.audit_odds_preingest import sha256_file
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        source_digest_before = sha256_file(_SILVER_ODDS)

        sandbox_silver = tmp_path / "silver"
        sandbox_silver.mkdir(parents=True)
        sandbox_odds = sandbox_silver / "odds_snapshot.parquet"
        shutil.copy2(_SILVER_ODDS, sandbox_odds)

        transformed = transform_nfl_odds_to_standard_format(_load_live_schedule(2024))
        rams = transformed[
            transformed["game_id"].str.contains(r"_LA@|@LA$", regex=True)
        ].copy()
        assert not rams.empty, (
            "the ingest transform produced no LA-keyed Rams rows for 2024, so the dry run "
            "would measure nothing. create_standard_game_id normalizes LAR -> LA; if that "
            "changed, DEFECT-2 has changed shape."
        )

        before = pd.read_parquet(sandbox_odds)
        n_before = len(before)
        n_lar_before = int(before["game_id"].str.contains("LAR").sum())
        n_la_before = int(
            before["game_id"].str.contains(r"_LA@|@LA$", regex=True).sum()
        )

        upsert_silver(rams, "odds_snapshot", key_column="game_id", base_path=tmp_path)

        after = pd.read_parquet(sandbox_odds)
        n_after = len(after)
        n_lar_after = int(after["game_id"].str.contains("LAR").sum())
        n_la_after = int(after["game_id"].str.contains(r"_LA@|@LA$", regex=True).sum())

        measured = (
            f"MEASURED upsert behaviour (assumption A2, DEFECT-2): upserting {len(rams)} "
            f"LA-keyed Rams rows for 2024 into a copy of production silver took the table "
            f"from {n_before} to {n_after} rows, a delta of {n_after - n_before}. The "
            f"pre-existing LAR-keyed Rams rows went {n_lar_before} -> {n_lar_after} and the "
            f"LA-keyed rows went {n_la_before} -> {n_la_after}. upsert_silver keys on "
            "game_id, and 'LAR' and 'LA' are DIFFERENT KEYS, so the re-ingest ADDS "
            "duplicate-game rows and REPLACES NONE. An all-season re-ingest must therefore "
            "either replace the odds table outright or normalize the stored LAR keys first; "
            "a plain merge silently duplicates every Rams game."
        )

        assert n_after == n_before + len(rams), measured
        assert n_lar_after == n_lar_before, measured
        assert n_la_after == n_la_before + len(rams), measured

        assert sha256_file(_SILVER_ODDS) == source_digest_before, (
            "the DEFECT-2 dry run changed data/silver/odds_snapshot.parquet. This plan writes "
            "NOTHING to production silver -- Plan 31-11 owns the only silver write in this "
            "phase -- and the whole measurement is supposed to happen inside tmp_path."
        )
