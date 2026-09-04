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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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


# ---------------------------------------------------------------------------
# Plan 31-08: the rewritten ingest WRITE CONTRACT.
#
# Four separately-attributable facts change at once -- label, game-type scope, juice columns
# and snapshot semantics -- so each is named in the ratified pre-registration and proved here in
# isolation. NOTHING below writes production silver: every destination is ``tmp_path`` and the
# module-scoped digest guard asserts the production flat store is byte-identical across the whole
# module run.
# ---------------------------------------------------------------------------

_INGEST_MODULE = Path("scripts/ingest_historical_odds.py")

# The five game types the pre-registration ratified (D31-38, playoffs EVERYWHERE). Restated here
# as literals ON PURPOSE: a test that imported the same constant the code reads would pass even
# if the constant itself moved, and the point of this gate is that the ADMITTED SET IS THE
# RATIFIED SET, not merely that two modules agree with each other.
_RATIFIED_GAME_TYPES = ("REG", "WC", "DIV", "CON", "SB")


def _schedule_row(
    *,
    season: int = 2024,
    week: int = 1,
    gameday: str = "2024-09-08",
    home_team: str = "KC",
    away_team: str = "BAL",
    game_type: str = "REG",
    spread_line: float | None = -3.0,
    total_line: float | None = 46.5,
    home_moneyline: int | None = -150,
    away_moneyline: int | None = 130,
    home_spread_odds: int | None = -108,
    away_spread_odds: int | None = -112,
    over_odds: int | None = -105,
    under_odds: int | None = -115,
) -> dict:
    """One nflreadpy-shaped schedule row with ASYMMETRIC juice (never the -110 default)."""
    return {
        "season": season,
        "week": week,
        "gameday": gameday,
        "home_team": home_team,
        "away_team": away_team,
        "game_type": game_type,
        "spread_line": spread_line,
        "total_line": total_line,
        "home_moneyline": home_moneyline,
        "away_moneyline": away_moneyline,
        "home_spread_odds": home_spread_odds,
        "away_spread_odds": away_spread_odds,
        "over_odds": over_odds,
        "under_odds": under_odds,
    }


def _non_docstring_string_constants() -> set[str]:
    """Every string LITERAL in the ingest module that is not a docstring.

    Docstrings are excluded deliberately: the module must stay free to NAME the legacy mislabel
    and the playoff codes in prose while never DECLARING either as a value it uses.
    """
    import ast

    tree = ast.parse(_INGEST_MODULE.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    }


class TestTheRatifiedLabel:
    """Write-contract clause 1: the label is the one the live silver rows already carry."""

    def test_a_transformed_row_carries_the_frozen_label(self) -> None:
        from backtest.ev_chain_constants import ODDS_SPORTSBOOK_LABEL
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        out = transform_nfl_odds_to_standard_format(pd.DataFrame([_schedule_row()]))

        assert set(out["sportsbook"]) == {ODDS_SPORTSBOOK_LABEL}

    def test_the_provenance_guard_admits_a_transformed_row(self) -> None:
        """The point of clause 1: no allowlist widening is needed for the row to pass."""
        from backtest.bet_selector import assert_real_odds
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        out = transform_nfl_odds_to_standard_format(
            pd.DataFrame([_schedule_row(season=2025, gameday="2025-09-14")])
        )

        assert_real_odds(out)

    def test_the_oum_06_allowlist_is_not_widened(self) -> None:
        from backtest.ou_divergence import _ALLOWED_SPORTSBOOKS

        assert frozenset({"consensus", "draftkings"}) == _ALLOWED_SPORTSBOOKS, (
            "the OUM-06 allowlist moved. Clause 1 of the pre-registration (D31-12 branch 1) "
            "settles the label question by MATCHING the live rows, explicitly so that the "
            "allowlist stays exactly as Phase 26 left it. Widening it is the OTHER branch, and "
            "that branch was not taken."
        )

    def test_the_label_is_not_a_literal_in_the_ingest_module(self) -> None:
        constants = _non_docstring_string_constants()

        assert "nflverse_closing" not in constants, (
            "the documented LEGACY MISLABEL is still a value in the ingest module. It is named "
            "in prose and nowhere else; the written label is ODDS_SPORTSBOOK_LABEL."
        )
        assert "consensus" not in constants, (
            "the ratified label is declared locally instead of being READ from the frozen "
            "backtest/ev_chain_constants.py. A second declaration is a second rule."
        )


class TestTheRatifiedGameTypeScope:
    """Write-contract clause 4: REG plus all four playoff types, in BOTH windows (D31-38)."""

    @pytest.mark.parametrize("game_type", _RATIFIED_GAME_TYPES)
    def test_every_ratified_game_type_is_admitted_in_both_windows(
        self, game_type: str
    ) -> None:
        from scripts.ingest_historical_odds import admitted_game_types

        assert game_type in admitted_game_types(2022), (
            f"{game_type} is not admitted in a TUNE season. D31-38 is playoffs EVERYWHERE."
        )
        assert game_type in admitted_game_types(2025), (
            f"{game_type} is not admitted in the HOLD season. The 2025 hold is 285 games, "
            "not 272, and that is the ratified scope."
        )

    def test_preseason_is_excluded_in_both_windows(self) -> None:
        from scripts.ingest_historical_odds import admitted_game_types

        assert "PRE" not in admitted_game_types(2022)
        assert "PRE" not in admitted_game_types(2025)

    def test_the_admitted_set_is_read_from_the_frozen_constants(self) -> None:
        from backtest.ev_chain_constants import HOLD_GAME_TYPES, TUNE_GAME_TYPES
        from scripts.ingest_historical_odds import admitted_game_types

        assert admitted_game_types(2022) == frozenset(TUNE_GAME_TYPES)
        assert admitted_game_types(2025) == frozenset(HOLD_GAME_TYPES)

    def test_the_module_declares_no_local_playoff_type_collection(self) -> None:
        constants = _non_docstring_string_constants()
        declared = sorted(constants & set(_RATIFIED_GAME_TYPES))

        assert declared == [], (
            f"the ingest module declares the game-type literals {declared} locally. The scope "
            "is a clause of the FROZEN pre-registration; a local declaration is a second rule "
            "that can drift from the ratified one without any test noticing."
        )

    def test_a_fabricated_game_type_is_dropped_and_counted(self) -> None:
        from scripts.ingest_historical_odds import transform_nfl_odds_with_counts

        frame = pd.DataFrame(
            [
                _schedule_row(away_team="BAL"),
                _schedule_row(away_team="BUF", game_type="XFL"),
                _schedule_row(away_team="NYJ", game_type="PRE"),
            ]
        )
        report = transform_nfl_odds_with_counts(frame)

        assert report.admitted == 1
        assert report.dropped_by_game_type == {"PRE": 1, "XFL": 1}
        assert set(report.odds["game_id"]) == {"2024_W01_BAL@KC"}

    def test_the_live_2025_schedule_admits_285_and_drops_none(self) -> None:
        from scripts.audit_odds_preingest import MIN_2025_GAMES_WITH_TOTAL
        from scripts.ingest_historical_odds import transform_nfl_odds_with_counts

        report = transform_nfl_odds_with_counts(_load_live_schedule(2025))

        assert report.admitted == MIN_2025_GAMES_WITH_TOTAL
        assert report.dropped_by_game_type == {}
        assert report.dropped_no_betting_data == 0
        assert report.dropped_transform_error == 0
        assert len(report.odds) == MIN_2025_GAMES_WITH_TOTAL

    def test_the_counts_are_returned_not_only_logged(self) -> None:
        """Plan 31-11 asserts these counts; it must not have to parse a log line."""
        import dataclasses

        from scripts.ingest_historical_odds import (
            OddsTransformReport,
            transform_nfl_odds_with_counts,
        )

        report = transform_nfl_odds_with_counts(pd.DataFrame([_schedule_row()]))

        assert isinstance(report, OddsTransformReport)
        fields = {f.name for f in dataclasses.fields(report)}
        assert {
            "odds",
            "admitted",
            "dropped_by_game_type",
            "dropped_no_betting_data",
            "dropped_transform_error",
        } <= fields


class TestTheSeasonArgument:
    """2025 is ingested only when it is asked for, never by silent inclusion in a default."""

    def test_the_historical_default_does_not_include_2025(self) -> None:
        from scripts.ingest_historical_odds import (
            DEFAULT_HISTORICAL_SEASONS,
            build_parser,
        )

        assert 2025 not in DEFAULT_HISTORICAL_SEASONS
        assert build_parser().parse_args([]).seasons == list(DEFAULT_HISTORICAL_SEASONS)

    def test_2025_is_accepted_when_named_explicitly(self) -> None:
        from scripts.ingest_historical_odds import build_parser

        assert build_parser().parse_args(["--seasons", "2025"]).seasons == [2025]


# The two string shapes the live snapshot_ts column is KNOWN to hold, quoted from the
# pre-registration's clause-3 type trap. Neither is a datetime, and they do not share a format.
_LEGACY_PER_SEASON_STRING = "2021-09-19T18:00:00-04:00"
_NON_CONSENSUS_UTC_STRING = "2025-09-29 18:44:09.707942+00:00"


def _sandbox_silver_copy(tmp_path: Path) -> Path:
    """A writable copy of the production flat odds store, under ``tmp_path``.

    Every write-path proof in this module runs against this copy. Plan 31-11 owns the only
    production silver write in this phase, under CHECKPOINT 2.
    """
    sandbox_silver = tmp_path / "silver"
    sandbox_silver.mkdir(parents=True, exist_ok=True)
    sandbox_odds = sandbox_silver / _SILVER_ODDS.name
    shutil.copy2(_SILVER_ODDS, sandbox_odds)
    return sandbox_odds


def _local_utc_offset_hours() -> float:
    """This process's own UTC offset, used to show a timezone test is not vacuous."""
    from datetime import datetime as _dt

    offset = _dt.now().astimezone().utcoffset()
    assert offset is not None
    return offset.total_seconds() / 3600.0


class TestTheAdditiveJuiceColumns:
    """Write-contract clause 2: real two-sided prices, added, never overwriting a stored line."""

    def test_every_transformed_row_carries_all_four_juice_columns(self) -> None:
        from scripts.ingest_historical_odds import (
            JUICE_COLUMNS,
            transform_nfl_odds_to_standard_format,
        )

        out = transform_nfl_odds_to_standard_format(pd.DataFrame([_schedule_row()]))

        for column in JUICE_COLUMNS:
            assert column in out.columns
            assert out[column].notna().all()

    def test_the_written_juice_is_the_real_asymmetric_price(self) -> None:
        """The -110 default is what clause 2 exists to stop; a real price is asymmetric."""
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        row = transform_nfl_odds_to_standard_format(
            pd.DataFrame([_schedule_row()])
        ).iloc[0]

        assert int(row["spread_ju_home"]) == -108
        assert int(row["spread_ju_away"]) == -112
        assert int(row["total_over_ju"]) == -105
        assert int(row["total_under_ju"]) == -115

    def test_the_live_2025_transform_is_285_of_285_on_all_four_columns(self) -> None:
        from scripts.audit_odds_preingest import MIN_2025_GAMES_WITH_TOTAL
        from scripts.ingest_historical_odds import (
            JUICE_COLUMNS,
            transform_nfl_odds_with_counts,
        )

        out = transform_nfl_odds_with_counts(_load_live_schedule(2025)).odds

        assert len(out) == MIN_2025_GAMES_WITH_TOTAL
        for column in JUICE_COLUMNS:
            n_present = int(out[column].notna().sum())
            assert n_present == MIN_2025_GAMES_WITH_TOTAL, (
                f"{column} is present on {n_present} of {len(out)} admitted 2025 games. "
                "Clause 2 states both spread-side and both total-side prices are complete; "
                "a gap is a source change to investigate, not a value to default."
            )

    def test_a_missing_juice_value_is_a_hard_failure_not_a_minus_110_default(
        self,
    ) -> None:
        from scripts.ingest_historical_odds import transform_nfl_odds_with_counts
        from utils import DataIngestionError

        frame = pd.DataFrame([_schedule_row(over_odds=None)])

        with pytest.raises(DataIngestionError, match="total_over_ju"):
            transform_nfl_odds_with_counts(frame)

    def test_the_stored_lines_are_byte_unchanged_by_a_merge_into_a_temporary_copy(
        self, tmp_path: Path
    ) -> None:
        """The clause-2 proof: juice is ADDED and the four stored line values are preserved."""
        _require_silver_odds()
        _require_gold_ou()
        from scripts.ingest_historical_odds import (
            JUICE_COLUMNS,
            PROTECTED_LINE_COLUMNS,
            canonical_game_id,
            transform_nfl_odds_to_standard_format,
            write_odds_additively,
        )

        sandbox_odds = _sandbox_silver_copy(tmp_path)

        before = pd.read_parquet(sandbox_odds)
        # Compare on CANONICAL keys: the merge re-keys the stored LAR rows to LA first, so a
        # raw-key comparison would report every Rams row as vanished when it was only renamed.
        before_keyed = before.assign(
            game_id=before["game_id"].astype(str).map(canonical_game_id),
            sportsbook=before["sportsbook"].astype(str),
        ).set_index(["game_id", "sportsbook"])

        incoming = transform_nfl_odds_to_standard_format(_load_live_schedule(2024))
        report = write_odds_additively(
            incoming,
            base_path=tmp_path,
            features_ou_df=pd.read_parquet(_GOLD_OU, columns=["game_id"]),
        )

        after = pd.read_parquet(sandbox_odds)
        after_keyed = after.assign(
            game_id=after["game_id"].astype(str),
            sportsbook=after["sportsbook"].astype(str),
        ).set_index(["game_id", "sportsbook"])

        survived = before_keyed.index.intersection(after_keyed.index)
        assert len(survived) == len(before_keyed), (
            f"{len(before_keyed)} stored rows went in and only {len(survived)} keys came out. "
            "The merge is additive; it never drops a stored row."
        )

        for column in PROTECTED_LINE_COLUMNS:
            pd.testing.assert_series_equal(
                before_keyed.loc[survived, column],
                after_keyed.loc[survived, column],
                check_names=False,
                obj=f"stored {column} across the merge",
            )

        # The rows the merge actually REWROTE are the incoming keys that already existed. Those
        # are the rows where an overwrite was possible at all, so that is the count clause 2 has
        # to account for -- comparing against the whole store would pass on a merge that touched
        # nothing.
        incoming_keys = pd.MultiIndex.from_arrays(
            [incoming["game_id"].astype(str), incoming["sportsbook"].astype(str)]
        )
        touched = before_keyed.index.intersection(incoming_keys)
        assert len(touched) > 0, (
            "the merge shared no (game_id, sportsbook) key with the stored table, so the "
            "overwrite comparison would be vacuous. Either the label or the keying moved."
        )
        assert report.rows_with_lines_preserved == len(touched), (
            f"{report.rows_with_lines_preserved} rows had their stored lines carried over but "
            f"{len(touched)} incoming keys already existed. Clause 2 preserves the stored line "
            "on EVERY row the merge rewrites, not on a subset."
        )
        for column in JUICE_COLUMNS:
            assert (
                after.loc[after["game_id"].isin(incoming["game_id"]), column]
                .notna()
                .all()
            )


class TestThePerGameSnapshotInstant:
    """Write-contract clause 3: every row carries its OWN preceding-Friday freeze (D31-37)."""

    def test_snapshot_ts_is_a_tz_aware_datetime_not_an_object_column(self) -> None:
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        out = transform_nfl_odds_to_standard_format(
            pd.DataFrame(
                [
                    _schedule_row(gameday="2024-09-08"),
                    _schedule_row(gameday="2024-09-15", week=2, away_team="BUF"),
                ]
            )
        )

        assert pd.api.types.is_datetime64_any_dtype(out["snapshot_ts"]), (
            f"snapshot_ts has dtype {out['snapshot_ts'].dtype}. D31-37 re-derives it precisely "
            "so the column stops being an object column of strings; the freshness comparison "
            "needs the datetime type regardless."
        )
        assert out["snapshot_ts"].dt.tz is not None

    def test_a_thursday_and_a_sunday_in_one_week_are_seven_days_apart(self) -> None:
        """The per-game rule, not a per-week one -- D31-18's stated failure mode."""
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        out = transform_nfl_odds_to_standard_format(
            pd.DataFrame(
                [
                    _schedule_row(gameday="2024-09-05", away_team="BAL"),  # Thursday
                    _schedule_row(gameday="2024-09-08", away_team="BUF"),  # Sunday
                ]
            )
        ).set_index("game_id")

        thursday = out.loc["2024_W01_BAL@KC", "snapshot_ts"]
        sunday = out.loc["2024_W01_BUF@KC", "snapshot_ts"]

        assert (sunday - thursday) == pd.Timedelta(days=7), (
            f"the Thursday game's freeze is {thursday} and the Sunday game's is {sunday}. Under "
            "a PER-WEEK freeze the Thursday snapshot would be strictly earlier than the week's "
            "instant and would be suppressed every week as a pure calendar artifact."
        )

    def test_a_friday_kickoff_takes_the_prior_friday(self) -> None:
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        # 2024-12-20 is a Friday; the freeze is 2024-12-13 at 18:00 Eastern.
        out = transform_nfl_odds_to_standard_format(
            pd.DataFrame([_schedule_row(week=16, gameday="2024-12-20")])
        )
        freeze = out.iloc[0]["snapshot_ts"].tz_convert("America/New_York")

        assert (freeze.year, freeze.month, freeze.day) == (2024, 12, 13)
        assert (freeze.hour, freeze.minute) == (18, 0)

    def test_every_replay_row_is_exactly_at_its_own_freeze_and_therefore_fresh(
        self,
    ) -> None:
        """Clause 3's whole point: re-derivation makes the replay population fresh by fact."""
        from scripts.ingest_historical_odds import (
            is_fresh_at_freeze,
            transform_nfl_odds_with_counts,
        )

        schedule = _load_live_schedule(2025)
        out = transform_nfl_odds_with_counts(schedule).odds
        gamedays = schedule.assign(
            key=schedule["season"].astype(int).astype(str)
            + "_W"
            + schedule["week"].astype(int).map("{:02d}".format)
        )[["key", "gameday"]].set_index("key")["gameday"]

        checked = 0
        for _, row in out.iterrows():
            key = row["game_id"].rsplit("_", 1)[0]
            gameday = gamedays.loc[key]
            gameday = gameday.iloc[0] if hasattr(gameday, "iloc") else gameday
            assert is_fresh_at_freeze(row["snapshot_ts"], gameday)
            checked += 1

        assert checked == len(out)


class TestTheSharedSnapshotNormalization:
    """The ONE parse path clause 3's type trap requires -- shared with Plan 31-09's selector."""

    def test_the_legacy_per_season_string_parses(self) -> None:
        from scripts.ingest_historical_odds import normalize_snapshot_ts

        parsed = normalize_snapshot_ts(_LEGACY_PER_SEASON_STRING)

        assert parsed == pd.Timestamp(_LEGACY_PER_SEASON_STRING).to_pydatetime()
        assert parsed.utcoffset() is not None

    def test_the_space_separated_utc_string_parses(self) -> None:
        """The single non-consensus row uses a DIFFERENT format; one parse path handles both."""
        from scripts.ingest_historical_odds import normalize_snapshot_ts

        parsed = normalize_snapshot_ts(_NON_CONSENSUS_UTC_STRING)

        assert parsed == pd.Timestamp(_NON_CONSENSUS_UTC_STRING).to_pydatetime()

    def test_a_tz_aware_datetime_round_trips_to_the_same_instant(self) -> None:
        from scripts.ingest_historical_odds import (
            get_synthetic_snapshot_ts,
            normalize_snapshot_ts,
        )

        freeze = get_synthetic_snapshot_ts("2024-09-08")

        assert normalize_snapshot_ts(freeze) == freeze

    def test_a_naive_value_is_anchored_in_eastern_and_not_in_utc(self) -> None:
        """The anchor is a STATED choice: an unqualified odds wall clock is market-local."""
        from scripts.ingest_historical_odds import EASTERN, normalize_snapshot_ts

        parsed = normalize_snapshot_ts("2021-09-19 18:00:00")

        assert parsed == datetime(2021, 9, 19, 18, 0, tzinfo=EASTERN)
        assert parsed != datetime(2021, 9, 19, 18, 0, tzinfo=UTC), (
            "a naive snapshot value was read as UTC. That moves the instant four or five hours "
            "-- across the 6 PM freeze in either direction -- and silently flips verdicts."
        )

    def test_a_null_snapshot_is_never_silently_fresh(self) -> None:
        from scripts.ingest_historical_odds import normalize_snapshot_ts

        for value in (None, pd.NaT, "", "   "):
            with pytest.raises(ValueError):
                normalize_snapshot_ts(value)

    def test_at_freeze_is_fresh_and_one_second_before_is_stale(self) -> None:
        from scripts.ingest_historical_odds import (
            get_synthetic_snapshot_ts,
            is_fresh_at_freeze,
        )

        freeze = get_synthetic_snapshot_ts("2024-09-08")

        assert is_fresh_at_freeze(freeze, "2024-09-08") is True
        assert is_fresh_at_freeze(freeze - timedelta(seconds=1), "2024-09-08") is False
        assert is_fresh_at_freeze(freeze + timedelta(seconds=1), "2024-09-08") is True

    def test_one_instant_in_three_encodings_gives_one_verdict(self) -> None:
        """A string comparison would call these three different; a parse calls them one."""
        from scripts.ingest_historical_odds import (
            get_synthetic_snapshot_ts,
            is_fresh_at_freeze,
        )

        freeze = get_synthetic_snapshot_ts("2024-09-08")
        string_encodings = [
            freeze.isoformat(),  # ISO, UTC offset, T separator
            str(freeze),  # the space-separated UTC shape the odd row uses
            freeze.astimezone(
                ZoneInfo("America/New_York")
            ).isoformat(),  # Eastern offset
        ]
        encodings = [
            freeze,
            *string_encodings,
        ]  # plus the datetime this ingest now writes

        verdicts = {is_fresh_at_freeze(value, "2024-09-08") for value in encodings}

        assert verdicts == {True}
        assert len(set(string_encodings)) == len(string_encodings), (
            "the three string encodings are not textually distinct, so this test would pass "
            "even under a string comparison and would prove nothing."
        )

    def test_no_call_site_consults_the_process_local_timezone(self) -> None:
        """The structural half of the T-31-37 guard: there is no local-zone call site at all."""
        import ast

        tree = ast.parse(_INGEST_MODULE.read_text(encoding="utf-8"))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            attr = node.func.attr
            if attr == "utcnow":
                offenders.append(f"{attr}() at line {node.lineno}")
            elif (
                attr in {"astimezone", "now", "today", "fromtimestamp"}
                and not node.args
            ):
                offenders.append(f"{attr}() with no argument at line {node.lineno}")

        assert offenders == [], (
            f"the ingest module reaches for the PROCESS local timezone at {offenders}. A bare "
            "astimezone() or now() anchors the freeze wherever the run happens to execute, so "
            "the same data yields different staleness verdicts on two machines (T-31-37)."
        )

    def test_the_verdict_is_unchanged_under_a_non_eastern_process_timezone(
        self,
    ) -> None:
        """The behavioural half: the SAME verdicts from a process whose local zone is Pacific.

        Run in a subprocess because a process's local timezone is fixed at start-up and
        ``time.tzset`` does not exist on Windows, where this project runs. The subprocess reports
        both its own offset and its verdicts, so a run where TZ had no effect is visible as a
        failure rather than passing vacuously.
        """
        import json
        import os
        import subprocess
        import sys

        script = (
            "import json\n"
            "from datetime import datetime, timedelta\n"
            "from scripts.ingest_historical_odds import ("
            "get_synthetic_snapshot_ts, is_fresh_at_freeze, normalize_snapshot_ts)\n"
            "freeze = get_synthetic_snapshot_ts('2024-09-08')\n"
            "print(json.dumps({\n"
            "  'offset_hours': datetime.now().astimezone().utcoffset().total_seconds()/3600,\n"
            "  'at_freeze': is_fresh_at_freeze(freeze, '2024-09-08'),\n"
            "  'one_second_before': is_fresh_at_freeze("
            "freeze - timedelta(seconds=1), '2024-09-08'),\n"
            "  'legacy_string': normalize_snapshot_ts("
            f"{_LEGACY_PER_SEASON_STRING!r}).isoformat(),\n"
            "  'naive_string': normalize_snapshot_ts('2021-09-19 18:00:00').isoformat(),\n"
            "}))\n"
        )

        env = dict(os.environ)
        env["TZ"] = "PST8PDT"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path.cwd()),
            check=True,
        )
        remote = json.loads(completed.stdout.strip().splitlines()[-1])

        assert remote["offset_hours"] != _local_utc_offset_hours(), (
            f"the subprocess reports the same UTC offset ({remote['offset_hours']}) as this "
            "process, so TZ had no effect and this control proved nothing. Fix the harness "
            "rather than deleting the assertion."
        )

        from scripts.ingest_historical_odds import (
            get_synthetic_snapshot_ts,
            is_fresh_at_freeze,
            normalize_snapshot_ts,
        )

        freeze = get_synthetic_snapshot_ts("2024-09-08")
        assert remote["at_freeze"] is is_fresh_at_freeze(freeze, "2024-09-08")
        assert remote["one_second_before"] is is_fresh_at_freeze(
            freeze - timedelta(seconds=1), "2024-09-08"
        )
        assert pd.Timestamp(remote["legacy_string"]) == pd.Timestamp(
            normalize_snapshot_ts(_LEGACY_PER_SEASON_STRING)
        )
        assert pd.Timestamp(remote["naive_string"]) == pd.Timestamp(
            normalize_snapshot_ts("2021-09-19 18:00:00")
        )
