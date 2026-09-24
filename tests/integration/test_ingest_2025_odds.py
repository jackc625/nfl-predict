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
from typing import cast
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


DRIFT_XFAIL_REASON = (
    "DEF-31-06 (owner ruling B, 2026-09-05): the live re-score no longer reproduces the "
    "RATIFIED ATS residual constants. Two separately-attributable causes -- an nflverse "
    "upstream re-release that moved sixteen gold columns before this plan wrote anything, "
    "and the ratified clause-5 LAR -> LA key normalization reaching the 2021-2024 slice at "
    "the ruling-A full rebuild, which corrected 68 games that had carried a fabricated zero "
    "market line. The owner ruled this a DISCLOSURE item, not a tampering one: the frozen "
    "constants ARE the rule, backtest/ev_chain_constants.py reads them, and the verdict is "
    "computed with the ratified numbers. What was lost is their RE-DERIVABILITY. The "
    "assertions below are PRESERVED INTACT and strict=True, so the drift stays measurable "
    "in the terminal summary and a return to the ratified values fails loudly instead of "
    "passing unnoticed. Both value sets are recorded in "
    "tests/phase31_state.ATS_RESIDUAL_LIVE_UPSTREAM_ONLY and "
    "ATS_RESIDUAL_LIVE_AFTER_FULL_REBUILD."
)


class TestTheAppendedATSResidualConstantsStillHold:
    """The live re-score must equal the constants Plan 31-02 appended, to 17 significant digits.

    The pre-registration Plan 31-05 freezes binds to these numbers. If the deployed artifact,
    the gold matrix or the residual contract moves, this test is where that shows up -- as a
    failure naming both figures, not as a quietly different number in a later run.

    XFAIL SINCE 2026-09-05, UNDER THE OWNER'S RULING B (register entry DEF-31-06). Four of
    these cases are EXPECTED failures today and are marked ``xfail(strict=True)`` -- never
    ``skip``, never a widened tolerance, and with every assertion left byte-for-byte as it
    was. The distinction matters: a skipped case asserts nothing, and a tolerance chosen
    after seeing the drift is exactly the post-hoc threshold selection this phase forbids
    everywhere else. An xfail still RUNS the comparison, still reports both figures, and
    ``strict=True`` means the day gold returns to the ratified numbers this turns into an
    unexpected PASS -- a failure -- forcing DEF-31-06 to be closed rather than forgotten.

    ``test_the_scored_artifact_is_the_one_the_constants_name`` is NOT marked: the artifact
    identity did not drift and must keep passing, because a different artifact would mean a
    different number for a reason the ruling does not cover.
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

    @pytest.mark.xfail(strict=True, reason=DRIFT_XFAIL_REASON)
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

    @pytest.mark.xfail(strict=True, reason=DRIFT_XFAIL_REASON)
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

        THIS CASE STILL PASSES AND IS DELIBERATELY NOT MARKED. The one assertion it carried
        that DID drift -- the negative-season set -- was split into the case below rather
        than dragging these three guards into an xfail with it. Marking the whole test would
        have silently stopped asserting the pooled direction claim the entire ATS arm of the
        pre-registration rests on, which is a loss of protection the ruling did not ask for.
        """
        assert float(ats_bias_block["pooled"]["mean"]) > 0.0
        assert ats_bias_block["pooled_direction_asserted"] is True
        assert ats_bias_block["per_season_sign_asserted"] is False
        assert ats_bias_block["numeric_tolerance_used"] is None, (
            "a numeric tolerance appeared in the ATS bias measurement. Choosing a magnitude "
            "threshold after seeing the measured values is the post-hoc threshold selection "
            "this phase forbids everywhere else (T-31-08c)."
        )

    @pytest.mark.xfail(strict=True, reason=DRIFT_XFAIL_REASON)
    def test_the_negative_mean_season_set_is_still_exactly_2022(
        self, ats_bias_block: dict
    ) -> None:
        """SPLIT OUT of the case above on 2026-09-05, assertion carried over verbatim.

        This is the sixth drift failure, and it is the one the ruling's own table did not
        list -- because it did not exist yet. It appeared at the ruling-A full rebuild: the
        clause-5 correction moved the live 2022 mean from -0.03170638 to +0.04653887, so
        ``seasons_with_negative_mean`` went from ``[2022]`` to ``[]``. The ratified 2022
        figure is negative and the pre-registration's prose says so in as many words, so this
        is a statement of record that the live data no longer supports -- exactly the kind of
        thing DEF-31-06 obliges PROFITABILITY-READOUT.md to disclose.
        """
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

    def test_the_synthetic_row_is_GONE_and_the_allowlist_would_still_admit_it(
        self,
    ) -> None:
        """The premise of the whole gate: a forged game_id under a legitimate sportsbook.

        RE-EXPRESSED AFTER THE INGEST (owner ruling C, 2026-09-05). Before Plan 31-11's
        write this asserted that production silver CONTAINED exactly one
        ``2025_W01_TEST@HOME`` row. That was a correct measurement of a world this plan was
        pre-registered to change: ``remove_synthetic_stored_rows`` is the NAMED pre-ingest
        step that deleted it, so the old assertion now measures the step having succeeded.

        What the test was actually protecting is NOT the row's presence -- it is the PREMISE
        that the OUM-06 sportsbook allowlist does not discriminate, which is the whole reason
        a ``game_id`` shape-and-existence gate has to exist. That premise is asserted here
        directly, against a reconstruction of the removed row, so it keeps holding after the
        row itself is gone. The removal is asserted alongside it, so the control now proves
        both halves rather than trading one for the other.
        """
        _require_silver_odds()
        from backtest.ou_divergence import _ALLOWED_SPORTSBOOKS

        odds = pd.read_parquet(_SILVER_ODDS)
        fixture = odds[odds["game_id"] == _SYNTHETIC_FIXTURE_GAME_ID]
        assert len(fixture) == 0, (
            f"production silver still carries {len(fixture)} "
            f"{_SYNTHETIC_FIXTURE_GAME_ID} row(s). Plan 31-11 runs "
            "`remove_synthetic_stored_rows` as a NAMED pre-ingest step before the 2025 "
            "write; a surviving row means that step did not run, or ran and was undone."
        )

        # The premise, unchanged and still load-bearing: this row's provenance columns are
        # indistinguishable from a real one. Only its game_id gives it away.
        forged = {"sportsbook": "consensus", "is_live": False}
        assert forged["sportsbook"] in _ALLOWED_SPORTSBOOKS, (
            "a forged row under sportsbook 'consensus' would no longer pass the OUM-06 "
            "allowlist, so the hole DEFECT-3's game_id gate closes may have changed shape. "
            "Re-read DEFECT-3 before relaxing anything."
        )
        assert not forged["is_live"]

    def test_it_RETURNS_on_the_current_silver_2025_slice_and_still_raises_on_a_forgery(
        self,
    ) -> None:
        """The gate's verdict on the live 2025 slice, and proof it is still discriminating.

        RE-EXPRESSED AFTER THE INGEST (owner ruling C, 2026-09-05). This asserted that
        ``assert_no_synthetic_game_ids`` RAISES on the live 2025 slice, which was true while
        that slice was the single synthetic fixture row. The ingest replaced it with 285 real
        games, so the gate must now RETURN -- and asserting only that would leave a gate that
        passes for free. The second half re-injects one forged id into the SAME live slice
        and requires the raise, so the gate is proved discriminating on production data
        rather than merely quiet.
        """
        _require_silver_odds()
        _require_gold_ou()
        from scripts.audit_odds_preingest import assert_no_synthetic_game_ids

        odds = pd.read_parquet(_SILVER_ODDS)
        slice_2025 = odds[odds["game_id"].astype(str).str.startswith("2025_")]
        features_ou = pd.read_parquet(_GOLD_OU, columns=["game_id"])

        assert len(slice_2025) > 0, (
            "the live 2025 odds slice is EMPTY, so both halves of this control would be "
            "judging nothing. The ingest admitted 285 games."
        )
        assert_no_synthetic_game_ids(slice_2025, features_ou)

        forged = pd.concat(
            [
                slice_2025,
                slice_2025.head(1).assign(game_id=_SYNTHETIC_FIXTURE_GAME_ID),
            ],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match=_SYNTHETIC_FIXTURE_GAME_ID):
            assert_no_synthetic_game_ids(forged, features_ou)

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

    @staticmethod
    def _rams_2024_rows() -> pd.DataFrame:
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        transformed = transform_nfl_odds_to_standard_format(_load_live_schedule(2024))
        rams = transformed[
            transformed["game_id"].str.contains(r"_LA@|@LA$", regex=True)
        ].copy()
        assert not rams.empty, (
            "the ingest transform produced no LA-keyed Rams rows for 2024, so the dry run "
            "would measure nothing. create_standard_game_id normalizes LAR -> LA; if that "
            "changed, DEFECT-2 has changed shape."
        )
        return cast("pd.DataFrame", rams)

    def test_the_defect_still_reproduces_against_UN_normalized_stored_keys(
        self, tmp_path: Path
    ) -> None:
        """The DEFECT-2 measurement, preserved against a store that still carries the hazard.

        RE-EXPRESSED AFTER THE INGEST (owner ruling C, 2026-09-05). This measurement used to
        run against production silver, which carried 116 ``LAR``-keyed Rams rows. Plan
        31-11's ratified clause-5 ``normalize_stored_game_ids`` re-keyed all of them, so the
        hazard is no longer LYING IN the production store -- which is the fix working, not
        the hazard ceasing to exist.

        The measurement is therefore taken against a sandbox that is DE-normalized back to
        the pre-clause-5 spelling. That keeps the fact this control exists to state -- that
        ``upsert_silver`` keys on ``game_id`` and ``LAR`` and ``LA`` are DIFFERENT KEYS, so a
        re-ingest ADDS rather than replaces -- measurable forever, instead of resting on a
        transient property of a production file.
        """
        _require_silver_odds()
        from data.storage import upsert_silver
        from scripts.audit_odds_preingest import sha256_file

        source_digest_before = sha256_file(_SILVER_ODDS)
        sandbox_odds = _sandbox_silver_copy(tmp_path)

        # Re-install the pre-clause-5 hazard: every canonical LA Rams key spelled LAR.
        seeded = pd.read_parquet(sandbox_odds)
        seeded["game_id"] = (
            seeded["game_id"]
            .astype(str)
            .str.replace(r"_LA@", "_LAR@", regex=True)
            .str.replace(r"@LA$", "@LAR", regex=True)
        )
        seeded.to_parquet(sandbox_odds, index=False)

        rams = self._rams_2024_rows()
        before = pd.read_parquet(sandbox_odds)
        n_before = len(before)
        n_lar_before = int(before["game_id"].str.contains("LAR").sum())
        n_la_before = int(
            before["game_id"].str.contains(r"_LA@|@LA$", regex=True).sum()
        )
        assert n_lar_before > 0, (
            "the de-normalization seeded no LAR-keyed rows, so this control is measuring a "
            "hazard it failed to install and would pass for free."
        )

        upsert_silver(rams, "odds_snapshot", key_column="game_id", base_path=tmp_path)

        after = pd.read_parquet(sandbox_odds)
        n_after = len(after)
        n_lar_after = int(after["game_id"].str.contains("LAR").sum())
        n_la_after = int(after["game_id"].str.contains(r"_LA@|@LA$", regex=True).sum())

        measured = (
            f"MEASURED upsert behaviour (assumption A2, DEFECT-2): upserting {len(rams)} "
            f"LA-keyed Rams rows for 2024 into a DE-NORMALIZED copy of production silver "
            f"took the table from {n_before} to {n_after} rows, a delta of "
            f"{n_after - n_before}. The LAR-keyed Rams rows went {n_lar_before} -> "
            f"{n_lar_after} and the LA-keyed rows went {n_la_before} -> {n_la_after}. "
            "upsert_silver keys on game_id, and 'LAR' and 'LA' are DIFFERENT KEYS, so the "
            "re-ingest ADDS duplicate-game rows and REPLACES NONE. An all-season re-ingest "
            "must therefore either replace the odds table outright or normalize the stored "
            "LAR keys first; a plain merge silently duplicates every Rams game."
        )

        assert n_after == n_before + len(rams), measured
        assert n_lar_after == n_lar_before, measured
        assert n_la_after == n_la_before + len(rams), measured

        assert sha256_file(_SILVER_ODDS) == source_digest_before, (
            "the DEFECT-2 dry run changed data/silver/odds_snapshot.parquet. The whole "
            "measurement is supposed to happen inside tmp_path."
        )

    def test_the_ratified_normalization_CLOSED_it_in_the_production_store(
        self, tmp_path: Path
    ) -> None:
        """The other half: against the store as it now stands, the re-ingest REPLACES.

        Same 2024 Rams rows, same ``upsert_silver`` call, against a sandbox copy of
        production silver as clause 5 left it. Every Rams game already present is REPLACED
        rather than duplicated, so the only additions are ids the store did not hold at all.
        That is the fix, measured on the artifact it was applied to.
        """
        _require_silver_odds()
        from data.storage import upsert_silver
        from scripts.audit_odds_preingest import sha256_file

        source_digest_before = sha256_file(_SILVER_ODDS)
        sandbox_odds = _sandbox_silver_copy(tmp_path)

        rams = self._rams_2024_rows()
        before = pd.read_parquet(sandbox_odds)
        n_before = len(before)
        stored_ids = set(before["game_id"].astype(str))
        genuinely_new = sorted(set(rams["game_id"].astype(str)) - stored_ids)

        assert int(before["game_id"].str.contains("LAR").sum()) == 0, (
            "the production store still carries LAR-keyed rows, so the ratified clause-5 "
            "normalization did not hold. DEF-31 clause 5 re-keyed 116 rows on 2026-09-05."
        )

        upsert_silver(rams, "odds_snapshot", key_column="game_id", base_path=tmp_path)

        after = pd.read_parquet(sandbox_odds)
        n_added = len(after) - n_before
        assert n_added == len(genuinely_new), (
            f"the re-ingest added {n_added} rows against {len(genuinely_new)} genuinely new "
            f"game_id(s) {genuinely_new}. With canonical stored keys every already-present "
            "Rams game must be REPLACED, not duplicated; any excess addition means the keys "
            "still disagree."
        )
        assert int(after["game_id"].str.contains("LAR").sum()) == 0

        assert sha256_file(_SILVER_ODDS) == source_digest_before

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


class TestAStoredNullNeverErasesARealIncomingLine:
    """WR-14: clause 2 says a stored line is never OVERWRITTEN, not that a null wins.

    ``preserve_stored_lines`` copied the stored value onto every MATCHED key unconditionally,
    including when the stored value was NaN. ``PROTECTED_LINE_COLUMNS`` is
    ``(spread, total, ml_home, ml_away)`` and rows carrying a null moneyline exist BY
    CONSTRUCTION -- ``transform_nfl_odds_with_counts`` writes ``ml_home = None`` when the source is
    null. So a stored 2019 row with no moneyline erased the ``-150`` a later nflverse pull
    supplied, and "never overwritten" became "never improved, and sometimes destroyed".

    These run entirely on frames built in the test. Nothing under ``data/`` is read or written.
    """

    @staticmethod
    def _stored(**overrides: object) -> pd.DataFrame:
        row = {
            "game_id": "2019_W01_AAA@BBB",
            "sportsbook": "consensus",
            "spread": -3.0,
            "total": 44.5,
            "ml_home": None,
            "ml_away": None,
        }
        row.update(overrides)
        return pd.DataFrame([row])

    @staticmethod
    def _incoming(**overrides: object) -> pd.DataFrame:
        row = {
            "game_id": "2019_W01_AAA@BBB",
            "sportsbook": "consensus",
            "spread": -7.5,
            "total": 51.0,
            "ml_home": -150.0,
            "ml_away": 130.0,
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def test_a_stored_null_moneyline_does_not_erase_the_incoming_one(self) -> None:
        from scripts.ingest_historical_odds import preserve_stored_lines

        result, n_matched = preserve_stored_lines(self._incoming(), self._stored())

        assert n_matched == 1, "the fixture did not match, so nothing is being asserted"
        assert result.iloc[0]["ml_home"] == pytest.approx(-150.0), (
            "the stored NULL moneyline overwrote the real incoming -150; the row would be "
            "written back with its moneyline erased"
        )
        assert result.iloc[0]["ml_away"] == pytest.approx(130.0)

    def test_a_stored_line_that_IS_present_still_wins(self) -> None:
        """The control. Clause 2's actual guarantee must survive the fix."""
        from scripts.ingest_historical_odds import preserve_stored_lines

        result, _ = preserve_stored_lines(self._incoming(), self._stored())

        assert result.iloc[0]["spread"] == pytest.approx(-3.0), (
            "the stored spread was overwritten by the incoming one -- clause 2 broken in the "
            "direction it was actually written to prevent"
        )
        assert result.iloc[0]["total"] == pytest.approx(44.5)

    def test_a_stored_zero_is_a_value_and_still_wins(self) -> None:
        """0.0 is a real pick-em line, not an absence. ``notna`` distinguishes them; falsiness does not."""
        from scripts.ingest_historical_odds import preserve_stored_lines

        result, _ = preserve_stored_lines(self._incoming(), self._stored(spread=0.0))

        assert result.iloc[0]["spread"] == pytest.approx(0.0)


class TestThePerGameSnapshotInstant:
    """Write-contract clause 3: every row carries its OWN day-before-kickoff lock (D31-37).

    Plan 33.2-02 retired the preceding-Friday freeze this clause was written against; the
    per-game property it protects is unchanged, only the instant moved (D33.2-01).
    """

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

    def test_a_thursday_and_a_sunday_in_one_week_are_three_days_apart(self) -> None:
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

        assert (sunday - thursday) == pd.Timedelta(days=3), (
            f"the Thursday game's lock is {thursday} and the Sunday game's is {sunday}. Each "
            "game locks on the ET day before its OWN kickoff (Wednesday and Saturday); a "
            "PER-WEEK instant would stamp both games with one value."
        )

    def test_a_friday_kickoff_locks_the_thursday_before(self) -> None:
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        # 2024-12-20 is a Friday; the lock is Thursday 2024-12-19 at 18:00 Eastern. The retired
        # rule gave the PRIOR Friday here, which is one of the disagreements D33.2-01 removed.
        out = transform_nfl_odds_to_standard_format(
            pd.DataFrame([_schedule_row(week=16, gameday="2024-12-20")])
        )
        lock = out.iloc[0]["snapshot_ts"].tz_convert("America/New_York")

        assert (lock.year, lock.month, lock.day) == (2024, 12, 19)
        assert (lock.hour, lock.minute) == (18, 0)

    def test_every_replay_row_is_exactly_at_its_own_lock_and_therefore_admissible(
        self,
    ) -> None:
        """Clause 3's whole point: re-derivation makes the replay population admissible by fact.

        The stamp is MANUFACTURED (nflreadpy gives a closing line with no capture time) and the
        code says so; what this proves is only that the stamp and the fence agree.
        """
        from scripts.ingest_historical_odds import (
            gameday_lock,
            transform_nfl_odds_with_counts,
        )
        from utils.game_id_utils import create_standard_game_id
        from utils.game_lock import is_admissible
        from utils.team_data import normalize_team_abbreviation

        schedule = _load_live_schedule(2025)
        out = transform_nfl_odds_with_counts(schedule).odds
        # Keyed by the GAME, not by its week: under the day-before lock the games of one week
        # carry different stamps (a Thursday game locks on the Wednesday, the Sunday slate on
        # the Saturday), so a per-week lookup would compare a row against another game's day.
        gameday_by_game = {
            create_standard_game_id(
                season=int(game["season"]),
                week=int(game["week"]),
                away_team=normalize_team_abbreviation(game["away_team"]),
                home_team=normalize_team_abbreviation(game["home_team"]),
            ): game["gameday"]
            for _, game in schedule.iterrows()
        }

        checked = 0
        for _, row in out.iterrows():
            gameday = gameday_by_game[row["game_id"]]
            # Retargeted by Plan 33.2-20 at ``utils.game_lock.is_admissible``, THE one
            # rule. Was: ``is_admissible_at_lock(row["snapshot_ts"], gameday)``, a second
            # admissibility entry point in ``scripts/`` that re-derived the lock from the
            # gameday on every call and that no production code ever called. The
            # comparison, and this assertion's intent, are unchanged.
            lock = gameday_lock(gameday)
            assert row["snapshot_ts"] == lock
            assert is_admissible(row["snapshot_ts"], lock)
            checked += 1

        assert checked == len(out)
        assert checked > 0, "no replay row was checked, so this proves nothing"


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
        from scripts.ingest_historical_odds import gameday_lock, normalize_snapshot_ts

        lock = gameday_lock("2024-09-08")

        assert normalize_snapshot_ts(lock) == lock

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

    def test_at_lock_is_admissible_and_one_second_after_is_not(self) -> None:
        """THE DIRECTION IS REVERSED ON PURPOSE (Plan 33.2-02, RESEARCH 2.4).

        The retired test here asserted a STALENESS rule: at-freeze fresh, one second BEFORE
        stale. The lock rule is an ADMISSIBILITY rule over information time: at-lock
        admissible, one second AFTER inadmissible, and a quote well before the lock is old
        information -- admissible, not stale. The staleness intent dies with the rule
        (D33.2-01); choosing the freshest admissible quote is Plan 33.2-13's subject.
        """
        from scripts.ingest_historical_odds import gameday_lock
        from utils.game_lock import is_admissible

        lock = gameday_lock("2024-09-08")

        assert is_admissible(lock, lock) is True
        assert is_admissible(lock + timedelta(seconds=1), lock) is False
        assert is_admissible(lock - timedelta(seconds=1), lock) is True
        assert is_admissible(lock - timedelta(hours=72), lock) is True

    def test_one_instant_in_three_encodings_gives_one_verdict(self) -> None:
        """A string comparison would call these three different; a parse calls them one."""
        from scripts.ingest_historical_odds import gameday_lock
        from utils.game_lock import is_admissible

        lock = gameday_lock("2024-09-08").astimezone(UTC)
        string_encodings = [
            lock.isoformat(),  # ISO, UTC offset, T separator
            str(lock),  # the space-separated UTC shape the odd row uses
            lock.astimezone(ZoneInfo("America/New_York")).isoformat(),  # Eastern offset
        ]
        encodings = [
            lock,
            *string_encodings,
        ]  # plus the datetime this ingest now writes

        verdicts = {is_admissible(value, lock) for value in encodings}

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
            "astimezone() or now() anchors the lock wherever the run happens to execute, so "
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
            "gameday_lock, normalize_snapshot_ts)\n"
            "from utils.game_lock import is_admissible\n"
            "lock = gameday_lock('2024-09-08')\n"
            "print(json.dumps({\n"
            "  'offset_hours': datetime.now().astimezone().utcoffset().total_seconds()/3600,\n"
            "  'lock': lock.isoformat(),\n"
            "  'at_lock': is_admissible(lock, lock),\n"
            "  'one_second_after': is_admissible("
            "lock + timedelta(seconds=1), lock),\n"
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
            gameday_lock,
            normalize_snapshot_ts,
        )
        from utils.game_lock import is_admissible

        lock = gameday_lock("2024-09-08")
        assert pd.Timestamp(remote["lock"]) == pd.Timestamp(lock)
        assert remote["at_lock"] is is_admissible(lock, lock) is True
        assert (
            remote["one_second_after"]
            is is_admissible(lock + timedelta(seconds=1), lock)
            is False
        )
        assert pd.Timestamp(remote["legacy_string"]) == pd.Timestamp(
            normalize_snapshot_ts(_LEGACY_PER_SEASON_STRING)
        )
        assert pd.Timestamp(remote["naive_string"]) == pd.Timestamp(
            normalize_snapshot_ts("2021-09-19 18:00:00")
        )


# Every skip reason this module can emit. Written out rather than derived so that adding a new
# skip without registering its phrasing is a FAILURE here, not an invisible non-run in a green
# suite (WR-10). The trailing exception text of the nflreadpy reason is elided; the registered
# marker is the fixed prefix.
_SKIP_REASONS_THIS_MODULE_CAN_EMIT = (
    "live gold absent (features_ats) -- data/ is gitignored runtime state.",
    "production manifest not present at artifacts/latest.json",
    (
        "live silver odds not present at data/silver/odds_snapshot.parquet -- data/ is "
        "gitignored runtime state."
    ),
    "live gold absent (features_ou) -- data/ is gitignored runtime state.",
    (
        "the live nflreadpy 2025 schedule could not be loaded on this checkout (offline or "
        "upstream unavailable)"
    ),
)


@pytest.fixture(scope="module", autouse=True)
def _production_silver_is_byte_identical_across_this_module():
    """The HARD BOUNDARY, asserted by CONTENT rather than by ``git status``.

    ``.gitignore`` blankets ``data/``, so a working-tree check on it is vacuous. Every write
    proof in this module runs against a ``tmp_path`` copy; this fixture is what turns "it should
    not have written production silver" into a fact the suite checks.
    """
    if not _SILVER_ODDS.is_file():
        yield
        return

    from scripts.audit_odds_preingest import sha256_file

    before = sha256_file(_SILVER_ODDS)
    yield
    after = sha256_file(_SILVER_ODDS)
    assert after == before, (
        f"data/silver/odds_snapshot.parquet CHANGED across this test module "
        f"({before} -> {after}). Plan 31-08 writes no production silver at all; Plan 31-11 owns "
        "the only silver write in this phase, under CHECKPOINT 2."
    )


def _call_sequence(function_name: str) -> list[str]:
    """The names called inside *function_name*, in SOURCE ORDER."""
    import ast

    tree = ast.parse(_INGEST_MODULE.read_text(encoding="utf-8"))
    target = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )

    calls: list[tuple[int, int, str]] = []
    for node in ast.walk(target):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        calls.append((node.lineno, node.col_offset, name))

    return [name for _, _, name in sorted(calls)]


class TestTheWritePathInvariants:
    """Clauses 5 through 8, proven against a temporary destination. No production write."""

    def test_the_synthetic_id_gate_is_called_before_the_write_not_after(self) -> None:
        """A gate that runs after a write is a report, not a gate (clause 7)."""
        sequence = _call_sequence("write_odds_additively")

        assert "assert_no_synthetic_game_ids" in sequence, (
            "write_odds_additively no longer calls the DEFECT-3 synthetic-id gate at all. The "
            "OUM-06 sportsbook allowlist provably does not catch a forged game_id under a "
            "legitimate sportsbook."
        )
        assert sequence.index("assert_no_synthetic_game_ids") < sequence.index(
            "upsert_silver"
        ), (
            f"the synthetic-id gate runs AFTER the write. Call order was {sequence}. A row that "
            "is already on disk when its gate fires has already contaminated the population."
        )

    def test_the_completeness_gate_is_called_before_the_write_not_after(self) -> None:
        sequence = _call_sequence("write_odds_additively")

        assert sequence.index("assert_2025_odds_completeness") < sequence.index(
            "upsert_silver"
        ), (
            f"the SPEC R2 completeness hard stop runs AFTER the write. Call order was "
            f"{sequence}. The 2025 hold can only be spent once."
        )

    def test_the_canonical_key_check_and_the_key_normalization_both_precede_the_write(
        self,
    ) -> None:
        sequence = _call_sequence("write_odds_additively")
        write_at = sequence.index("upsert_silver")

        assert sequence.index("assert_canonical_game_ids") < write_at
        assert sequence.index("normalize_stored_game_ids") < write_at, (
            "the stored keys are normalized AFTER the merge. The pre-registered Rams resolution "
            "is to normalize BEFORE any merge: upsert_silver keys on game_id, so a merge run "
            "first silently duplicates every Rams game rather than replacing it."
        )
        # 33.2 review C1 WR-07: the invariant runs on the merged table BEFORE the write (a gate,
        # not a report) AND again on what actually landed on disk.
        checks = [
            i for i, name in enumerate(sequence) if name == "assert_odds_table_keys"
        ]
        assert checks and checks[0] < write_at, (
            f"the one-row-per-key invariant does not run before the write. Call order was "
            f"{sequence}; a violation found after the write has already changed silver."
        )
        assert checks[-1] > write_at, (
            "the one-row-per-key invariant is no longer checked on what actually landed on disk."
        )

    def test_the_gate_fires_before_anything_is_created_on_disk(
        self, tmp_path: Path
    ) -> None:
        """The behavioural half of the ordering claim: a refused write leaves NO file."""
        from scripts.ingest_historical_odds import write_odds_additively

        forged = pd.DataFrame(
            {
                "game_id": [_SYNTHETIC_FIXTURE_GAME_ID],
                "sportsbook": ["consensus"],
                "snapshot_ts": [pd.Timestamp("2025-09-05T18:00:00-04:00")],
                "spread": [-3.0],
                "total": [45.5],
            }
        )

        with pytest.raises(ValueError, match=_SYNTHETIC_FIXTURE_GAME_ID):
            write_odds_additively(
                forged,
                base_path=tmp_path,
                features_ou_df=pd.DataFrame({"game_id": ["2025_W01_BUF@MIA"]}),
            )

        assert not (tmp_path / "silver").exists(), (
            "the refused write still created a silver directory. The gate must run before the "
            "write path touches the filesystem at all."
        )

    def test_a_thin_2025_population_stops_the_write_and_names_the_count(
        self, tmp_path: Path
    ) -> None:
        from scripts.ingest_historical_odds import write_odds_additively

        thin = pd.DataFrame(
            {
                "game_id": [f"2025_W{i + 1:02d}_AAA@BBB" for i in range(3)],
                "sportsbook": ["consensus"] * 3,
                "snapshot_ts": [pd.Timestamp("2025-09-05T18:00:00-04:00")] * 3,
                "total": [45.5] * 3,
            }
        )

        with pytest.raises(ValueError, match="only 3 games with a total"):
            write_odds_additively(
                thin,
                base_path=tmp_path,
                features_ou_df=thin[["game_id"]],
                completeness_seasons=(2025,),
            )

        assert not (tmp_path / "silver").exists()

    def test_a_non_canonical_incoming_id_is_refused_by_name(self) -> None:
        from scripts.ingest_historical_odds import assert_canonical_game_ids

        with pytest.raises(ValueError, match="2024_W01_LAR@SF"):
            assert_canonical_game_ids(
                pd.DataFrame(
                    {"game_id": ["2024_W01_LAR@SF"], "sportsbook": ["consensus"]}
                )
            )

    def test_a_duplicated_key_raises_naming_the_offending_triple(self) -> None:
        from scripts.ingest_historical_odds import assert_one_row_per_key

        freeze = pd.Timestamp("2024-09-06T18:00:00-04:00")
        duplicated = pd.DataFrame(
            {
                "game_id": ["2024_W01_BAL@KC", "2024_W01_BAL@KC", "2024_W01_BUF@MIA"],
                "sportsbook": ["consensus"] * 3,
                "snapshot_ts": [freeze, freeze, freeze],
            }
        )

        with pytest.raises(ValueError, match="2024_W01_BAL@KC"):
            assert_one_row_per_key(duplicated, stage="test")

        # The clean frame passes, so the guard is discriminating rather than always-on.
        assert_one_row_per_key(duplicated.drop_duplicates(), stage="test")

    def test_accumulated_live_captures_do_not_trip_the_merge_and_are_kept(
        self, tmp_path: Path
    ) -> None:
        """33.2 review C1 WR-07: two live captures of one game share the historical triple.

        Every live capture carries ``snapshot_ts = lock``; they differ only in ``created_at``.
        The triple checked over the whole table made every later historical merge raise -- and
        only AFTER it had already written silver.
        """
        from scripts.ingest_historical_odds import write_odds_additively

        lock = pd.Timestamp("2026-09-19T22:00:00Z")
        live_game = "2026_W02_CAR@ATL"
        silver = tmp_path / "silver"
        silver.mkdir()
        pd.DataFrame(
            {
                "game_id": [live_game, live_game],
                "sportsbook": ["draftkings", "draftkings"],
                "snapshot_ts": [lock, lock],
                "created_at": pd.to_datetime(
                    ["2026-09-18T15:00:00Z", "2026-09-19T21:00:00Z"], utc=True
                ),
                "spread": [3.5, 4.0],
                "total": [44.5, 44.5],
            }
        ).to_parquet(silver / "odds_snapshot.parquet", index=False)

        incoming = pd.DataFrame(
            {
                "game_id": ["2025_W01_BUF@MIA"],
                "sportsbook": ["consensus"],
                "snapshot_ts": [pd.Timestamp("2025-09-06T18:00:00-04:00")],
                "created_at": [pd.Timestamp("2026-09-24T00:00:00Z")],
                "spread": [-3.0],
                "total": [45.5],
            }
        )
        write_odds_additively(
            incoming,
            base_path=tmp_path,
            features_ou_df=pd.DataFrame({"game_id": ["2025_W01_BUF@MIA"]}),
        )
        stored = pd.read_parquet(silver / "odds_snapshot.parquet")
        assert (stored["game_id"] == live_game).sum() == 2, "a live capture was lost"

    def test_a_duplicated_historical_key_refuses_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        from scripts.ingest_historical_odds import write_odds_additively

        freeze = pd.Timestamp("2025-09-06T18:00:00-04:00")
        twice = pd.DataFrame(
            {
                "game_id": ["2025_W01_BUF@MIA", "2025_W01_BUF@MIA"],
                "sportsbook": ["consensus", "consensus"],
                "snapshot_ts": [freeze, freeze],
                "spread": [-3.0, -3.5],
                "total": [45.5, 45.5],
            }
        )
        with pytest.raises(ValueError, match="pre-merge"):
            write_odds_additively(
                twice,
                base_path=tmp_path,
                features_ou_df=pd.DataFrame({"game_id": ["2025_W01_BUF@MIA"]}),
            )
        assert not (tmp_path / "silver" / "odds_snapshot.parquet").exists()

    def test_two_encodings_of_one_instant_are_ONE_key_not_two(self) -> None:
        """The invariant parses the key instant; a string key would call these two rows."""
        from scripts.ingest_historical_odds import assert_one_row_per_key

        same_instant = pd.DataFrame(
            {
                "game_id": ["2024_W01_BAL@KC", "2024_W01_BAL@KC"],
                "sportsbook": ["consensus", "consensus"],
                "snapshot_ts": [
                    "2024-09-06T18:00:00-04:00",
                    "2024-09-06 22:00:00+00:00",
                ],
            }
        )

        with pytest.raises(ValueError, match="2024_W01_BAL@KC"):
            assert_one_row_per_key(same_instant, stage="test")

    def test_a_merge_into_a_temporary_copy_holds_every_invariant(
        self, tmp_path: Path
    ) -> None:
        """The end-to-end proof: canonical keys, one row per key, and the Rams resolution.

        RE-EXPRESSED AFTER THE INGEST (owner ruling C, 2026-09-05). The premise
        ``n_lar_before > 0`` used to read production silver, which carried 116 ``LAR``-keyed
        rows. Clause 5 re-keyed them, so reading that premise off the production store now
        fails on the fix having worked.

        The hazard is INSTALLED into the sandbox instead of borrowed from production. That
        is strictly stronger: the clause-5 proof no longer depends on a production file
        happening to be dirty, so it keeps proving the same thing on a clean store and on a
        fresh checkout. The synthetic fixture row is likewise seeded rather than assumed --
        Plan 31-11 removed the real one, and the clause-7 assertion below is about what the
        merge does with a forged id in the destination, not about which files happen to
        contain one today.
        """
        _require_silver_odds()
        _require_gold_ou()
        from scripts.audit_odds_preingest import sha256_file
        from scripts.ingest_historical_odds import (
            assert_one_row_per_key,
            canonical_game_id,
            transform_nfl_odds_to_standard_format,
            write_odds_additively,
        )

        production_digest_before = sha256_file(_SILVER_ODDS)
        sandbox_odds = _sandbox_silver_copy(tmp_path)

        # Install both pre-ingest hazards into the SANDBOX: the non-canonical Rams spelling
        # clause 5 resolves, and the forged id clause 7 must leave visible.
        seeded = pd.read_parquet(sandbox_odds)
        seeded["game_id"] = (
            seeded["game_id"]
            .astype(str)
            .str.replace(r"_LA@", "_LAR@", regex=True)
            .str.replace(r"@LA$", "@LAR", regex=True)
        )
        seeded = pd.concat(
            [seeded, seeded.head(1).assign(game_id=_SYNTHETIC_FIXTURE_GAME_ID)],
            ignore_index=True,
        )
        seeded.to_parquet(sandbox_odds, index=False)

        before = pd.read_parquet(sandbox_odds)
        before_ids = before["game_id"].astype(str)
        n_lar_before = int(before_ids.str.contains("LAR").sum())
        n_non_canonical_before = int(
            (before_ids.map(canonical_game_id) != before_ids).sum()
        )
        assert n_lar_before > 0, (
            "the sandbox seeding installed no LAR-keyed Rams rows, so the clause-5 "
            "resolution has nothing to prove here and this control would pass for free."
        )

        incoming = transform_nfl_odds_to_standard_format(_load_live_schedule(2024))
        report = write_odds_additively(
            incoming,
            base_path=tmp_path,
            features_ou_df=pd.read_parquet(_GOLD_OU, columns=["game_id"]),
        )

        after = pd.read_parquet(sandbox_odds)
        after_ids = after["game_id"].astype(str)

        # Clause 6: one row per (game_id, sportsbook, snapshot_ts), on what is on DISK.
        assert_one_row_per_key(after, stage="post-merge sandbox")

        # Clause 5: the non-canonical spelling appears ZERO times in the destination.
        assert int(after_ids.str.contains("LAR").sum()) == 0, (
            f"{int(after_ids.str.contains('LAR').sum())} LAR-keyed rows survive the merge. "
            "upsert_silver keys on game_id, so a surviving LAR row is a second key for a game "
            "that already has one."
        )
        assert report.stored_ids_normalized == n_non_canonical_before

        # Every id the merge WROTE is canonical and well-formed. The one id in the destination
        # that is neither is the synthetic fixture row DEFECT-3 names, SEEDED above rather
        # than borrowed from production -- so it is asserted by name here rather than quietly
        # excluded, and the assertion states what the merge does with a forged id rather than
        # what production silver happens to contain today.
        written = set(incoming["game_id"].astype(str))
        assert all(canonical_game_id(gid) == gid for gid in written)
        from utils.game_id_utils import GAME_ID_PATTERN

        assert all(GAME_ID_PATTERN.match(gid) for gid in written)
        assert sorted(gid for gid in after_ids if not GAME_ID_PATTERN.match(gid)) == [
            _SYNTHETIC_FIXTURE_GAME_ID
        ]

        # The Rams resolution measured: with the stored keys normalized FIRST, the 2024
        # re-ingest REPLACES the regular-season rows instead of duplicating them, and only the
        # genuinely new playoff rows are added. Plan 31-02 measured the un-normalized behaviour
        # as +19 added / 0 replaced.
        n_added = report.rows_after - report.rows_before
        assert n_added < len(incoming), (
            f"the merge added {n_added} rows for {len(incoming)} incoming ones. If every "
            "incoming row is an addition, nothing was replaced and the keys still disagree."
        )
        assert report.rows_after == report.rows_before + n_added

        assert sha256_file(_SILVER_ODDS) == production_digest_before

    def test_a_stored_draftkings_row_survives_a_consensus_ingest_of_the_same_game(
        self, tmp_path: Path
    ) -> None:
        """Clause 6 / CR-03: the write is ADDITIVE across sportsbooks, not a game_id clobber.

        ``upsert_silver`` keys on ``game_id`` alone, so it deletes every stored row whose game
        appears in the incoming frame -- and every incoming row carries ``consensus``. A stored
        ``draftkings`` row for the same game was therefore erased with no warning, and
        ``assert_one_row_per_key`` could not catch it: the clobber GUARANTEES one row per key, so
        the invariant passed trivially on a table that had just lost rows.

        The module docstring records that a ``draftkings`` row WAS in production silver as
        recently as Plan 31-08, and the OUM-06 allowlist admits it, so this is a real shape.
        """
        from backtest.ev_chain_constants import ODDS_SPORTSBOOK_LABEL
        from scripts.ingest_historical_odds import (
            transform_nfl_odds_to_standard_format,
            write_odds_additively,
        )

        sandbox_odds = _sandbox_silver_copy(tmp_path)
        incoming = transform_nfl_odds_to_standard_format(_load_live_schedule(2024))
        contested_game_id = str(incoming.iloc[0]["game_id"])

        # Seed a SECOND sportsbook's row for a game this ingest is about to write.
        stored = pd.read_parquet(sandbox_odds)
        rival = stored.iloc[[0]].copy()
        rival["game_id"] = contested_game_id
        rival["sportsbook"] = "draftkings"
        rival["spread"] = -13.5
        rival["total"] = 99.5
        pd.concat([stored, rival], ignore_index=True).to_parquet(sandbox_odds)

        report = write_odds_additively(
            incoming,
            base_path=tmp_path,
            features_ou_df=pd.read_parquet(_GOLD_OU, columns=["game_id"]),
        )

        after = pd.read_parquet(sandbox_odds)
        survivor = after[
            (after["game_id"].astype(str) == contested_game_id)
            & (after["sportsbook"].astype(str) == "draftkings")
        ]

        assert len(survivor) == 1, (
            f"the stored draftkings row for {contested_game_id} did not survive a consensus "
            f"ingest of the same game ({len(survivor)} rows found). The write is declared "
            "ADDITIVE on the (game_id, sportsbook, snapshot_ts) key and upsert_silver keys on "
            "game_id alone."
        )
        # Its own lines are untouched -- it was carried through, not rewritten from the
        # consensus values.
        assert float(survivor.iloc[0]["spread"]) == -13.5
        assert float(survivor.iloc[0]["total"]) == 99.5
        assert report.rows_carried_forward >= 1, (
            "the report does not count the carried-forward row, so an operator reading the "
            "merge summary cannot tell the clobber was avoided"
        )
        # The consensus row for the same game is still there: carrying the rival forward must
        # not displace the row the ingest came to write.
        assert (
            len(
                after[
                    (after["game_id"].astype(str) == contested_game_id)
                    & (after["sportsbook"].astype(str) == ODDS_SPORTSBOOK_LABEL)
                ]
            )
            == 1
        )

    def test_the_ingest_refuses_to_write_when_its_gate_cannot_run(
        self, tmp_path: Path
    ) -> None:
        """An absent O/U matrix is a REFUSAL, never a skipped gate."""
        from scripts.ingest_historical_odds import load_features_ou
        from utils import DataIngestionError

        with pytest.raises(DataIngestionError, match="synthetic-id gate cannot run"):
            load_features_ou(tmp_path / "nope" / "features_ou.parquet")


class TestEverySkipReasonThisModuleCanEmitIsRegistered:
    """A control that did not run must be NAMED in the terminal summary (WR-10)."""

    @pytest.mark.parametrize("reason", _SKIP_REASONS_THIS_MODULE_CAN_EMIT)
    def test_the_reason_is_recognised_as_evidence_backed(self, reason: str) -> None:
        from tests.conftest import is_evidence_backed_skip

        assert is_evidence_backed_skip(reason), (
            f"the skip reason {reason!r} is emitted by a control in this module but is not "
            "recognised as evidence-backed, so that control would disappear silently into a "
            "green suite. Register its phrasing in tests/conftest._EVIDENCE_SKIP_MARKERS."
        )

    def test_the_catalogue_covers_every_skip_call_site_in_this_module(self) -> None:
        """Adding a skip without cataloguing it is a failure, not an invisible non-run."""
        import ast

        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        n_skip_calls = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "skip"
        )

        assert n_skip_calls == len(_SKIP_REASONS_THIS_MODULE_CAN_EMIT), (
            f"this module has {n_skip_calls} pytest.skip call sites but catalogues "
            f"{len(_SKIP_REASONS_THIS_MODULE_CAN_EMIT)} reasons. Every reason this module can "
            "emit must be listed in _SKIP_REASONS_THIS_MODULE_CAN_EMIT and recognised by "
            "tests/conftest.is_evidence_backed_skip."
        )
