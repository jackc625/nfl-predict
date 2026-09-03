"""Wave-0 pre-ingest gates for the Phase-31 one-shot 2025 run (Plan 31-02).

This module holds the assertions that must hold BEFORE anything writes production silver or
rebuilds the 2025 gold slice. It writes nothing itself: Plan 31-11 owns the only silver write
in this phase, and every measurement here is either read-only or confined to ``tmp_path``.

What it pins:

* **The ATS residual bias does not drift.** Plan 31-02 Task 2 re-derived the bias from the
  DEPLOYED artifact and APPENDED the result to ``tests/phase31_state.py``. This module asserts
  the live re-score still equals those constants to 17 significant digits, so a later
  disagreement is a FAILURE rather than a silently different number in a later run.

TEST CLASS: integration. Reads ``data/gold``, ``artifacts/`` and ``data/silver`` -- all
gitignored runtime state -- behind evidence-backed skip guards whose reasons are registered in
``tests/conftest._EVIDENCE_SKIP_MARKERS``, so a control that did not run is named in the
terminal summary rather than disappearing into a green suite (WR-10).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.phase31_state import (
    ATS_RESIDUAL_BY_SEASON,
    ATS_RESIDUAL_FIELDS,
    ATS_RESIDUAL_POOLED,
    ATS_RESIDUAL_POOLED_PROVENANCE,
)

_GOLD_ATS = Path("data/gold/features_ats.parquet")
_ARTIFACTS_MANIFEST = Path("artifacts/latest.json")

# The ONE explicit specifier every Phase-31 float is compared through.
_FMT = "{:.17g}"


def _require_rescore_inputs() -> None:
    """Skip with a REGISTERED reason when the re-score's gitignored inputs are absent."""
    if not _GOLD_ATS.is_file():
        pytest.skip(
            "live gold absent (features_ats) -- data/ is gitignored runtime state."
        )
    if not _ARTIFACTS_MANIFEST.is_file():
        pytest.skip("production manifest not present at artifacts/latest.json")


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
