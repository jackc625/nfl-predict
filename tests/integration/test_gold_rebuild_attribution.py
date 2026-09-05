"""The per-rung cause-attribution contract over ``compare_fingerprints`` (SPEC R1, Plan 30-04).

Phase 30 runs FOUR full gold rebuilds, one per named cause: CR-02 (rung 1), WR-06 (rung 2),
the ``line_movement`` DROP (rung 3) and the N-01 re-sync (rung 4). ``compare_fingerprints``
already reports WHICH columns moved and in which seasons. What was missing is the judge:
a predicted signature per rung, and an explicit failure when a moved column cannot be
attributed to that rung's one named cause.

The whole point is that a full rebuild reaches nflreadpy LIVE with no cache
(``scripts/fingerprint_gold.py`` module docstring), so an upstream play-by-play or
depth-chart revision can land in ANY rung and be misattributed to that rung's named fix.
At rungs 1-3 an unattributable column is therefore a FINDING and the message says so --
check the nflreadpy revision date before concluding the named fix is wrong. Rung 4 is the
deliberate exception: SPEC R2 makes an unexplained 2021-2024 move a HARD BLOCKER, and its
failure message offers no upstream escape.

TEST CLASS (Plan 30-04's phase-wide rule -- every test module this phase adds declares its
kind in its docstring):

* Every class below EXCEPT ``TestFingerprintDeterminism`` is a **plain unit test**. The
  reports are hand-built ``compare_fingerprints``-shaped dictionaries, so the contract is
  provable without running a rebuild, and the module passes on a fresh checkout with no
  ``data/``, no ``artifacts/`` and no ``outputs/``.
* ``TestFingerprintDeterminism`` is **integration / slow**: it carries
  ``@pytest.mark.integration`` and skips cleanly, with a remediation-carrying message, when
  live gold is absent.

The 15 ``line_movement`` names are NEVER re-listed here. They are derived from
``backtest.signal_lift.group_columns`` -- the ONE registry (D30-02) -- against the committed
pre-drop fixture ``tests/fixtures/gold/features_ats_pre_phase30.parquet`` frozen by Plan
30-03. A second list of the family is the 29-06 failure mode.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from backtest.signal_lift import group_columns
from scripts.fingerprint_gold import (
    BUILD_CLOCK_COLUMNS,
    GOLD_MATRICES,
    PHASE30_RUNG_DOCUMENTS,
    PHASE31_RUNG_PREFIX,
    RUNG_CAUSES,
    MissingPredecessorFingerprintError,
    _expected_signature,
    attribute_rung,
    compare_fingerprints,
    fingerprint_gold,
    fingerprint_matrix,
    require_rung_ladder,
    rung_document_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The committed pre-drop fixture (Plan 30-03). A TRACKED artifact, so its absence is a broken
# checkout rather than a legitimately-absent runtime file -- this fails, it does not skip.
_PRE_DROP_FIXTURE = (
    REPO_ROOT / "tests" / "fixtures" / "gold" / "features_ats_pre_phase30.parquet"
)

_GOLD_DIR = REPO_ROOT / "data" / "gold"


def _pre_drop_columns() -> list[str]:
    """Return the pre-drop ATS gold column names, read from the committed fixture's schema."""
    assert _PRE_DROP_FIXTURE.exists(), (
        f"The committed pre-drop fixture is missing at {_PRE_DROP_FIXTURE}. It is TRACKED "
        "(Plan 30-03, .gitignore negation '!tests/fixtures/gold/*.parquet'), so its absence "
        "means a broken checkout, not an absent runtime artifact."
    )
    return list(pq.ParquetFile(_PRE_DROP_FIXTURE).schema_arrow.names)


def _line_movement_family() -> list[str]:
    """Derive the line_movement family from the ONE registry against the pre-drop fixture."""
    frame = pd.DataFrame(columns=pd.Index(_pre_drop_columns()))
    return group_columns(frame, "line_movement")


def _before_document() -> dict:
    """A minimal BEFORE fingerprint document carrying only the pre-drop column names."""
    return {
        matrix: {"columns": {name: {} for name in _pre_drop_columns()}}
        for matrix in GOLD_MATRICES
    }


# ---------------------------------------------------------------------------
# Hand-built compare_fingerprints-shaped report builders
# ---------------------------------------------------------------------------


def _detail(
    *,
    width_before: int,
    width_after: int,
    rows_before: int = 6263,
    rows_after: int = 6263,
    added: tuple[str, ...] = (),
    removed: tuple[str, ...] = (),
    changed: dict[str, list[str]] | None = None,
    discrete: tuple[str, ...] = (),
    became_discrete: tuple[str, ...] = (),
    rows_per_season_before: dict[str, int] | None = None,
    rows_per_season_after: dict[str, int] | None = None,
) -> dict:
    """Build one matrix's entry of a compare_fingerprints report.

    ``discrete`` names columns that are indicator-valued on BOTH sides of the rung.
    ``became_discrete`` names columns that were a MEASUREMENT before and are
    indicator-valued after -- the flattening signature, which
    ``FeatureMatrixBuilder._is_discrete_indicator`` cannot distinguish from a
    genuine indicator because a constant 0.0 z-score column satisfies it (CR-03).
    """
    changed = dict(changed or {})
    details = {
        column: {
            "seasons": list(seasons),
            "dtype_before": "float64",
            "dtype_after": "float64",
            "null_count_before": 0,
            "null_count_after": 0,
            "discrete_indicator_before": column in discrete,
            "discrete_indicator_after": column in discrete or column in became_discrete,
            "reasons": ["values"] if seasons else [],
        }
        for column, seasons in changed.items()
    }
    return {
        "width_before": width_before,
        "width_after": width_after,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_per_season_before": rows_per_season_before or {"2024": 285},
        "rows_per_season_after": rows_per_season_after or {"2024": 285},
        "columns_added": sorted(added),
        "columns_removed": sorted(removed),
        "columns_changed": changed,
        "column_details": details,
    }


def _widths() -> dict[str, int]:
    """Pre-drop widths, derived so no literal width is transcribed for the ATS matrix."""
    ats_width = len(_pre_drop_columns())
    # WP and OU are one narrower than ATS (its extra target/margin columns).
    return {
        "features_wp": ats_width - 1,
        "features_ats": ats_width,
        "features_ou": ats_width - 1,
    }


def _pre_drop_report(**overrides) -> dict:
    """A three-matrix report at the real pre-drop widths."""
    widths = _widths()
    report = {}
    for matrix in GOLD_MATRICES:
        kwargs = {
            "width_before": widths[matrix],
            "width_after": widths[matrix],
            **overrides,
        }
        report[matrix] = _detail(**kwargs)
    return report


def _all_failures(verdict: dict) -> str:
    """Flatten every failure message in a verdict into one searchable string."""
    return "\n".join(verdict["failures"])


# ---------------------------------------------------------------------------
# Rung 1 -- CR-02, the discrete-indicator exemption
# ---------------------------------------------------------------------------


class TestRung1DiscreteIndicators:
    """Rung 1's one named cause is CR-02, so only discrete-indicator columns may move."""

    def test_accepts_a_diff_whose_changed_columns_are_all_discrete_indicators(self):
        report = _pre_drop_report(
            changed={"line_movement_coverage": ["2023"], "saturday_game": ["2019"]},
            discrete=("line_movement_coverage", "saturday_game"),
        )
        verdict = attribute_rung(report, 1)

        assert verdict["ok"] is True
        assert verdict["blocking"] is False
        assert verdict["cause"] == RUNG_CAUSES[1]
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["unattributed"] == []
            assert verdict["matrices"][matrix]["attributed"] == [
                "line_movement_coverage",
                "saturday_game",
            ]

    def test_rejects_a_non_indicator_column_and_names_it(self):
        report = _pre_drop_report(
            changed={"line_movement_coverage": ["2023"], "home_rest_days": ["2021"]},
            discrete=("line_movement_coverage",),
        )
        verdict = attribute_rung(report, 1)

        assert verdict["ok"] is False
        assert verdict["blocking"] is False, (
            "an unattributed column at rungs 1-3 is a FINDING, not a phase blocker"
        )
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["unattributed"] == ["home_rest_days"]
        assert "home_rest_days" in _all_failures(verdict)

    def test_failure_message_names_upstream_nflreadpy_revision_as_a_candidate(self):
        report = _pre_drop_report(changed={"home_rest_days": ["2021"]})
        verdict = attribute_rung(report, 1)

        message = _all_failures(verdict).lower()
        assert "nflreadpy" in message
        assert "revision" in message

    def test_an_added_or_removed_column_is_unattributable_at_rung_1(self):
        report = _pre_drop_report(
            width_after=_widths()["features_ats"] + 1,
            added=("some_new_column",),
            changed={"line_movement_coverage": ["2023"]},
            discrete=("line_movement_coverage",),
        )
        verdict = attribute_rung(report, 1)

        assert verdict["ok"] is False
        assert "some_new_column" in _all_failures(verdict)

    def test_a_fingerprint_without_column_metadata_cannot_attribute_and_says_so(self):
        report = _pre_drop_report(changed={"line_movement_coverage": ["2023"]})
        for matrix in GOLD_MATRICES:
            del report[matrix]["column_details"]

        verdict = attribute_rung(report, 1)

        assert verdict["ok"] is False
        assert "column_details" in _all_failures(verdict)


# ---------------------------------------------------------------------------
# Rung 2 -- WR-06, and the empty-diff edge
# ---------------------------------------------------------------------------


class TestRung2Wr06:
    """Rung 2's cause is WR-06, whose bounds change everywhere -- but SOMETHING must move."""

    def test_an_empty_diff_fails(self):
        report = _pre_drop_report(changed={})
        verdict = attribute_rung(report, 2)

        assert verdict["ok"] is False
        failures = _all_failures(verdict).lower()
        assert "moved no column" in failures or "no column moved" in failures

    def test_accepts_a_broad_changed_set(self):
        report = _pre_drop_report(
            changed={
                "home_rest_days": ["2002", "2010", "2024"],
                "away_epa_per_play": ["2015"],
            }
        )
        verdict = attribute_rung(report, 2)

        assert verdict["ok"] is True
        assert verdict["blocking"] is False
        assert verdict["cause"] == RUNG_CAUSES[2]

    def test_a_width_move_is_unattributable_at_rung_2(self):
        widths = _widths()
        report = {
            matrix: _detail(
                width_before=widths[matrix],
                width_after=widths[matrix] - 1,
                removed=("home_rest_days",),
                changed={"away_epa_per_play": ["2015"]},
            )
            for matrix in GOLD_MATRICES
        }
        verdict = attribute_rung(report, 2)

        assert verdict["ok"] is False
        assert "home_rest_days" in _all_failures(verdict)

    def test_failure_message_names_upstream_nflreadpy_revision_as_a_candidate(self):
        report = _pre_drop_report(changed={})
        assert "nflreadpy" in _all_failures(attribute_rung(report, 2)).lower()


# ---------------------------------------------------------------------------
# The flattening failure mode -- a DESTROYED column must not attribute to itself
# ---------------------------------------------------------------------------


class TestADestroyedColumnCannotAttributeToItself:
    """CR-03 / WR-11: a continuous -> constant transition is a finding, not an exemption.

    ``FeatureMatrixBuilder._is_discrete_indicator`` returns True iff every non-null
    value is one of ``-1.0 / 0.0 / 1.0``, so a CONSTANT column at any of those levels
    satisfies it -- and gold's feature columns are expanding-window z-scores, so a
    column the rebuild FLATTENS lands at a constant 0.0.

    Rung 1 used to read the two sides with ``or``:

        discrete = discrete_indicator_before or discrete_indicator_after

    which handed a flattened column the CR-02 winsorization exemption on the strength
    of the damage itself, returning ``ok: True, unattributed: []``. Rung 2 attributed
    every changed column unconditionally, so it could not fail on one at all.

    This is the codebase's own recorded failure: rung 2's first attempt destroyed 18
    columns through a degenerate ``q01 == q99`` clip while attributing perfectly
    cleanly (``GATED-REFIT-READOUT.md`` section 1). The judge is now the thing that
    says so, instead of the summary prose after the fact.

    No test previously fed the judge a ``continuous -> constant`` transition, which is
    why the arm shipped.
    """

    @pytest.mark.parametrize("rung", [1, 2])
    def test_a_column_that_became_an_indicator_is_unattributed(self, rung: int):
        report = _pre_drop_report(
            changed={"line_movement_coverage": ["2023"], "home_epa_per_play": ["2021"]},
            discrete=("line_movement_coverage",),
            became_discrete=("home_epa_per_play",),
        )
        verdict = attribute_rung(report, rung)

        assert verdict["ok"] is False, (
            f"rung {rung} accepted a column that was a measurement before the rebuild "
            "and is indicator-valued after. That is the flattening signature, and it "
            "must never be forgiven by the exemption it produced."
        )
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["unattributed"] == ["home_epa_per_play"]
            assert (
                "home_epa_per_play" not in (verdict["matrices"][matrix]["attributed"])
            )

    @pytest.mark.parametrize("rung", [1, 2])
    def test_the_failure_names_the_degenerate_clip_as_the_likely_cause(self, rung: int):
        report = _pre_drop_report(
            changed={"home_epa_per_play": ["2021"]},
            became_discrete=("home_epa_per_play",),
        )
        failures = _all_failures(attribute_rung(report, rung)).lower()

        assert "was not a discrete indicator" in failures
        assert "flatten" in failures

    def test_rung_1_still_attributes_a_column_that_was_ALREADY_an_indicator(self):
        """The exemption survives for the case it was actually written for."""
        report = _pre_drop_report(
            changed={"line_movement_coverage": ["2023"]},
            discrete=("line_movement_coverage",),
        )
        verdict = attribute_rung(report, 1)

        assert verdict["ok"] is True
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["attributed"] == [
                "line_movement_coverage"
            ]

    def test_rung_1_attributes_an_indicator_that_STOPPED_being_one(self):
        """Attribution is on the BEFORE side, so an indicator that widened still fits.

        This is the real rung-1 evidence: ``venue_high_altitude`` was discrete before
        the CR-02 rebuild and continuous after it. Reading the AFTER side would have
        turned the accepted rung-1 verdict into a failure.
        """
        report = _pre_drop_report(
            changed={"venue_high_altitude": ["2019", "2020"]},
            discrete=("venue_high_altitude",),
        )
        # discrete on BOTH sides by construction above; restate the asymmetric case.
        for matrix in GOLD_MATRICES:
            report[matrix]["column_details"]["venue_high_altitude"][
                "discrete_indicator_after"
            ] = False

        verdict = attribute_rung(report, 1)
        assert verdict["ok"] is True
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["attributed"] == ["venue_high_altitude"]

    def test_rung_4_also_refuses_a_flattened_column(self):
        """A re-sync adds rows; it does not turn a measurement into a constant."""
        widths = _widths()
        report = {
            matrix: _detail(
                width_before=widths[matrix],
                width_after=widths[matrix],
                rows_before=6263,
                rows_after=6499,
                changed={"home_epa_per_play": ["2025"]},
                became_discrete=("home_epa_per_play",),
                rows_per_season_before={"2024": 285, "2025": 49},
                rows_per_season_after={"2024": 285, "2025": 285},
            )
            for matrix in GOLD_MATRICES
        }
        verdict = attribute_rung(report, 4)

        assert verdict["ok"] is False
        assert verdict["blocking"] is True, (
            "rung 4 is SPEC R2's hard blocker; a flattened column there is not a "
            "non-blocking finding"
        )
        assert "flatten" in _all_failures(verdict).lower()


class TestABlanketAttributionDoesNotReadAsACleanVerdict:
    """WR-11: rung 2 cannot discriminate per column, and the report must say so.

    WR-06 refits every imputation and winsorization bound, so any imputed or clipped
    column may legitimately move and there is no per-column allow-list to check
    against. That is a real limit of the rung, not a defect -- but ``ok: True``,
    ``unattributed: []`` and a printed ``rung 2 (WR-06): OK`` is the shape a reader
    treats as a clean per-column verdict, and ``RUNBOOK.md`` tells the operator to
    judge by the exit code.
    """

    def test_rung_2_marks_itself_as_non_discriminating(self):
        report = _pre_drop_report(changed={"home_rest_days": ["2010"]})
        verdict = attribute_rung(report, 2)

        assert verdict["ok"] is True
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["discriminating"] is False

    @pytest.mark.parametrize("rung", [1, 3, 4])
    def test_the_per_column_rungs_stay_discriminating(self, rung: int):
        report = _pre_drop_report(
            changed={"line_movement_coverage": ["2023"]},
            discrete=("line_movement_coverage",),
        )
        verdict = attribute_rung(report, rung)

        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["discriminating"] is True

    def test_a_clean_rung_2_prints_ATTRIBUTED_NOT_HEALTH_CHECKED_not_OK(self, capsys):
        from scripts.fingerprint_gold import _print_attribution

        report = _pre_drop_report(changed={"home_rest_days": ["2010"]})
        _print_attribution(attribute_rung(report, 2))

        printed = capsys.readouterr().out.splitlines()[0]
        assert printed.endswith("ATTRIBUTED (NOT HEALTH-CHECKED)"), printed
        assert not printed.endswith(": OK")

    def test_a_clean_rung_1_still_prints_OK(self, capsys):
        from scripts.fingerprint_gold import _print_attribution

        report = _pre_drop_report(
            changed={"line_movement_coverage": ["2023"]},
            discrete=("line_movement_coverage",),
        )
        _print_attribution(attribute_rung(report, 1))

        assert capsys.readouterr().out.splitlines()[0].endswith(": OK")

    def test_a_failing_rung_2_still_prints_its_severity(self, capsys):
        from scripts.fingerprint_gold import _print_attribution

        report = _pre_drop_report(
            changed={"home_epa_per_play": ["2021"]},
            became_discrete=("home_epa_per_play",),
        )
        _print_attribution(attribute_rung(report, 2))

        assert capsys.readouterr().out.splitlines()[0].endswith("FINDING")


# ---------------------------------------------------------------------------
# Rung 3 -- the line_movement DROP
# ---------------------------------------------------------------------------


class TestRung3LineMovementDrop:
    """Rung 3 removes exactly the derived family and must move NO surviving value."""

    def _dropped_report(self, **overrides) -> dict:
        widths = _widths()
        family = _line_movement_family()
        report = {}
        for matrix in GOLD_MATRICES:
            kwargs = {
                "width_before": widths[matrix],
                "width_after": widths[matrix] - len(family),
                "removed": tuple(family),
                **overrides,
            }
            report[matrix] = _detail(**kwargs)
        return report

    def test_the_removed_set_is_exactly_the_derived_family(self):
        before = _before_document()
        verdict = attribute_rung(self._dropped_report(), 3, before=before)

        assert verdict["ok"] is True
        assert verdict["blocking"] is False
        family = _line_movement_family()
        assert len(family) == 15
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["attributed"] == sorted(family)

    def test_a_removed_column_outside_the_family_is_unattributed(self):
        family = _line_movement_family()
        report = self._dropped_report(removed=(*family, "home_rest_days"))
        for matrix in GOLD_MATRICES:
            report[matrix]["width_after"] -= 1

        verdict = attribute_rung(report, 3)

        assert verdict["ok"] is False
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["unattributed"] == ["home_rest_days"]

    def test_a_non_empty_changed_set_is_BLOCKING(self):
        verdict = attribute_rung(
            self._dropped_report(changed={"home_rest_days": ["2019"]}), 3
        )

        assert verdict["ok"] is False
        assert verdict["blocking"] is True, (
            "dropping columns must not move surviving values -- a non-empty changed set at "
            "rung 3 means the dropped columns participated in a whole-frame statistic"
        )
        assert "whole-frame" in _all_failures(verdict)

    def test_width_must_fall_by_exactly_the_removed_count(self):
        report = self._dropped_report()
        for matrix in GOLD_MATRICES:
            report[matrix]["width_after"] += 1

        verdict = attribute_rung(report, 3)

        assert verdict["ok"] is False
        assert "width" in _all_failures(verdict).lower()

    def test_failure_message_names_upstream_nflreadpy_revision_as_a_candidate(self):
        report = self._dropped_report(removed=("home_rest_days",))
        message = _all_failures(attribute_rung(report, 3)).lower()
        assert "nflreadpy" in message

    def test_a_partial_drop_FAILS_and_names_every_retained_column(self):
        """D30-DEFER-12: the family gone from two matrices and retained in a third.

        The check this exercises used to be vacuous. ``_rung3_expected_removed``
        filtered ``diff["removed"]`` by the line-movement predicate, so the expected
        set was a SUBSET of the observed set by construction and the partial-drop
        loop could never fire. The strong per-matrix expectation was derived from the
        BEFORE document, landed in ``signature.columns_removed``, and was then never
        handed to the matrix judge.
        """
        before = _before_document()
        widths = _widths()
        report = self._dropped_report()
        # features_ou keeps the whole family: nothing removed, width unmoved.
        report["features_ou"] = _detail(
            width_before=widths["features_ou"],
            width_after=widths["features_ou"],
        )

        verdict = attribute_rung(report, 3, before=before)

        assert verdict["ok"] is False, (
            "a family removed from two matrices and retained in a third is a PARTIAL "
            "drop and must fail"
        )
        failures = _all_failures(verdict)
        assert "PARTIAL" in failures
        assert "features_ou" in failures
        for column in _line_movement_family():
            assert column in failures, (
                f"the partial-drop failure must NAME the retained column {column!r}"
            )

    def test_the_expected_removed_set_is_an_EQUALITY_not_a_subset(self):
        """Removing 14 of the 15 satisfies the width arithmetic and must still fail."""
        before = _before_document()
        family = _line_movement_family()
        retained = sorted(family)[0]
        partial = tuple(name for name in family if name != retained)
        widths = _widths()
        report = {
            matrix: _detail(
                width_before=widths[matrix],
                width_after=widths[matrix] - len(partial),
                removed=partial,
            )
            for matrix in GOLD_MATRICES
        }

        verdict = attribute_rung(report, 3, before=before)

        assert verdict["ok"] is False
        failures = _all_failures(verdict)
        assert "PARTIAL" in failures
        assert retained in failures


# ---------------------------------------------------------------------------
# The build clock -- a column that is different in KIND (D30-OWNER-08)
# ---------------------------------------------------------------------------


class TestBuildClockColumn:
    """``feature_timestamp`` is a per-build ``datetime.now(UTC)`` stamp.

    It moves on EVERY rebuild by construction, so counting it as a moved value makes
    rung 3's empty-changed-set criterion structurally unsatisfiable -- no correct
    rebuild can ever satisfy it. Plan 30-06 already reported it as its own category
    at rung 1 rather than filing it under the upstream-drift escape; these tests make
    the instrument do that at every rung.
    """

    _CLOCK = BUILD_CLOCK_COLUMNS[0]

    def _dropped_report(self, **overrides) -> dict:
        widths = _widths()
        family = _line_movement_family()
        return {
            matrix: _detail(
                width_before=widths[matrix],
                width_after=widths[matrix] - len(family),
                removed=tuple(family),
                **overrides,
            )
            for matrix in GOLD_MATRICES
        }

    def test_a_clock_only_move_does_not_block_rung_3(self):
        report = self._dropped_report(changed={self._CLOCK: ["2002", "2024"]})
        verdict = attribute_rung(report, 3, before=_before_document())

        assert verdict["blocking"] is False, (
            "a per-build clock stamp cannot be evidence that a dropped column was "
            "participating in a whole-frame statistic"
        )
        assert verdict["ok"] is True
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["build_clock"] == [self._CLOCK]
            assert self._CLOCK not in verdict["matrices"][matrix]["unattributed"]
            assert self._CLOCK not in verdict["matrices"][matrix]["attributed"]

    @pytest.mark.parametrize("rung", [1, 2, 3, 4])
    def test_the_clock_is_reported_in_its_own_category_at_every_rung(self, rung: int):
        widths = _widths()
        report = {
            matrix: _detail(
                width_before=widths[matrix],
                width_after=widths[matrix],
                rows_after=6421,
                rows_per_season_before={"2024": 285, "2025": 49},
                rows_per_season_after={"2024": 285, "2025": 207},
                changed={self._CLOCK: ["2024"], "home_rest_days": ["2025"]},
            )
            for matrix in GOLD_MATRICES
        }
        verdict = attribute_rung(report, rung)

        for matrix in GOLD_MATRICES:
            detail = verdict["matrices"][matrix]
            assert detail["build_clock"] == [self._CLOCK]
            assert self._CLOCK not in detail["attributed"]
            assert self._CLOCK not in detail["unattributed"]

    @pytest.mark.parametrize("rung", [1, 2, 3])
    def test_the_clock_is_never_filed_under_the_upstream_drift_escape(self, rung: int):
        """Calling a build clock an nflreadpy revision would send a reader hunting."""
        report = self._dropped_report(changed={self._CLOCK: ["2024"]})
        verdict = attribute_rung(report, rung, before=_before_document())

        for message in verdict["failures"]:
            assert self._CLOCK not in message, (
                f"rung {rung} named the build clock in a failure: {message}"
            )

    def test_a_real_data_column_still_BLOCKS_rung_3_alongside_the_clock(self):
        report = self._dropped_report(
            changed={self._CLOCK: ["2024"], "home_rest_days": ["2019"]}
        )
        verdict = attribute_rung(report, 3, before=_before_document())

        assert verdict["blocking"] is True
        failures = _all_failures(verdict)
        assert "home_rest_days" in failures
        assert self._CLOCK not in failures


# ---------------------------------------------------------------------------
# The value-preserving dtype cause -- attribution that must be EARNED
# ---------------------------------------------------------------------------


def _dtype_documents(before_values, before_dtype, after_values, after_dtype):
    """Return (before_doc, after_doc, after_frame) for a one-column dtype move.

    Real frames fingerprinted by ``fingerprint_matrix``, so the per-season digests
    the proof must reproduce are the genuine article rather than hand-written.
    """
    games = [f"2024_W{index + 1:02d}_A@B" for index in range(len(before_values))]

    def frame(values, dtype):
        return pd.DataFrame(
            {
                "game_id": games,
                "season": [2024] * len(values),
                "home_win": pd.Series(values, dtype=dtype),
            }
        )

    before_frame = frame(before_values, before_dtype)
    after_frame = frame(after_values, after_dtype)
    before_doc = {matrix: fingerprint_matrix(before_frame) for matrix in GOLD_MATRICES}
    after_doc = {matrix: fingerprint_matrix(after_frame) for matrix in GOLD_MATRICES}
    return before_doc, after_doc, after_frame


class TestValuePreservingDtypeCause:
    """An "it is only a dtype change" claim is an assertion; this phase refuses those.

    The attribution is granted ONLY when re-encoding the new values back to the prior
    dtype reproduces the prior per-season hash EXACTLY, in every season. Plan 30-07
    performed that proof by hand for ``home_win`` in 24 of 24 seasons; these tests
    make the judge perform it, and make it refuse when the proof does not land.
    """

    _VALUES = [1, 0, 1, 1, 0]

    def _judge(self, before_doc, after_doc, after_frame, rung=3):
        report = compare_fingerprints(before_doc, after_doc)
        return attribute_rung(
            report,
            rung,
            before=before_doc,
            after=after_doc,
            frame_loader=lambda matrix: after_frame,
        )

    def test_a_reproduced_hash_earns_the_attribution_and_does_not_block(self):
        before_doc, after_doc, after_frame = _dtype_documents(
            self._VALUES, "float64", self._VALUES, "int32"
        )
        verdict = self._judge(before_doc, after_doc, after_frame)

        assert verdict["blocking"] is False
        assert verdict["ok"] is True
        for matrix in GOLD_MATRICES:
            preserved = verdict["matrices"][matrix]["value_preserving_dtype"]
            assert preserved == [
                {
                    "column": "home_win",
                    "dtype_before": "float64",
                    "dtype_after": "int32",
                }
            ]

    def test_an_unprovable_dtype_change_stays_BLOCKING(self):
        """Same dtype move, one value genuinely different: a move wearing a costume."""
        moved = [1, 0, 1, 0, 0]
        assert moved != self._VALUES
        before_doc, after_doc, after_frame = _dtype_documents(
            self._VALUES, "float64", moved, "int32"
        )
        verdict = self._judge(before_doc, after_doc, after_frame)

        assert verdict["blocking"] is True
        assert verdict["ok"] is False
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["value_preserving_dtype"] == []
        assert "home_win" in _all_failures(verdict)

    def test_a_LOSSY_re_encode_stays_BLOCKING_even_though_the_hash_reproduces(self):
        """int32 <- float64 truncation reproduces the prior hash and is still a move.

        Re-encoding ``[1.5, 0.0, 1.0, 1.0, 0.0]`` to ``int32`` yields the prior
        ``[1, 0, 1, 1, 0]`` exactly, so the per-season hash reproduces. The round-trip
        does not: casting back gives ``1.0`` where the frame holds ``1.5``. Without
        the round-trip guard this is the hole a real value change escapes through.
        """
        lossy = [1.5, 0.0, 1.0, 1.0, 0.0]
        before_doc, after_doc, after_frame = _dtype_documents(
            self._VALUES, "int32", lossy, "float64"
        )
        verdict = self._judge(before_doc, after_doc, after_frame)

        assert verdict["blocking"] is True
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["value_preserving_dtype"] == []

    def test_without_the_after_values_the_proof_cannot_run_and_it_stays_BLOCKING(self):
        """Fail-closed: an unverifiable dtype change is not a forgiven one."""
        before_doc, after_doc, _ = _dtype_documents(
            self._VALUES, "float64", self._VALUES, "int32"
        )
        report = compare_fingerprints(before_doc, after_doc)
        verdict = attribute_rung(report, 3, before=before_doc, after=after_doc)

        assert verdict["blocking"] is True
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["value_preserving_dtype"] == []

    def test_a_frame_that_is_not_the_after_document_cannot_prove_anything(self):
        """The loaded frame must BE the judged artifact, not merely resemble it."""
        before_doc, after_doc, _ = _dtype_documents(
            self._VALUES, "float64", self._VALUES, "int32"
        )
        _, _, other_frame = _dtype_documents(
            self._VALUES, "float64", [0, 0, 0, 0, 0], "int32"
        )
        report = compare_fingerprints(before_doc, after_doc)
        verdict = attribute_rung(
            report,
            3,
            before=before_doc,
            after=after_doc,
            frame_loader=lambda matrix: other_frame,
        )

        assert verdict["blocking"] is True
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["value_preserving_dtype"] == []

    def test_a_moved_data_column_is_untouched_by_the_dtype_cause(self):
        """A plain value move, no dtype change, still blocks rung 3."""
        widths = _widths()
        family = _line_movement_family()
        report = {
            matrix: _detail(
                width_before=widths[matrix],
                width_after=widths[matrix] - len(family),
                removed=tuple(family),
                changed={"home_rest_days": ["2019"]},
            )
            for matrix in GOLD_MATRICES
        }
        verdict = attribute_rung(report, 3, before=_before_document())

        assert verdict["blocking"] is True
        assert "home_rest_days" in _all_failures(verdict)


# ---------------------------------------------------------------------------
# Rung 4 -- the N-01 re-sync, SPEC R2's positive control
# ---------------------------------------------------------------------------


class TestRung4Resync:
    """Rung 4 may move 2025 and NOTHING ELSE. Any other season blocks the phase."""

    def _resync_report(self, **overrides) -> dict:
        widths = _widths()
        family = _line_movement_family()
        report = {}
        for matrix in GOLD_MATRICES:
            kwargs = {
                "width_before": widths[matrix] - len(family),
                "width_after": widths[matrix] - len(family),
                "rows_before": 6263,
                "rows_after": 6421,
                "rows_per_season_before": {"2024": 285, "2025": 49},
                "rows_per_season_after": {"2024": 285, "2025": 207},
                **overrides,
            }
            report[matrix] = _detail(**kwargs)
        return report

    def test_accepts_a_2025_only_move(self):
        verdict = attribute_rung(
            self._resync_report(changed={"home_rest_days": ["2025"]}), 4
        )

        assert verdict["ok"] is True
        assert verdict["blocking"] is False
        assert verdict["cause"] == RUNG_CAUSES[4]

    def test_a_2021_2024_move_is_blocking_and_names_the_incomplete_wr06_fix(self):
        verdict = attribute_rung(
            self._resync_report(changed={"home_rest_days": ["2023", "2025"]}), 4
        )

        assert verdict["ok"] is False
        assert verdict["blocking"] is True
        failures = _all_failures(verdict)
        assert "WR-06" in failures
        assert "incomplete" in failures.lower()
        assert "R2" in failures

    def test_the_rung_4_message_does_not_offer_the_upstream_escape(self):
        verdict = attribute_rung(
            self._resync_report(changed={"home_rest_days": ["2023"]}), 4
        )
        assert "nflreadpy" not in _all_failures(verdict).lower()

    def test_rows_must_grow(self):
        report = self._resync_report(
            changed={"home_rest_days": ["2025"]},
            rows_after=6263,
            rows_per_season_after={"2024": 285, "2025": 49},
        )
        verdict = attribute_rung(report, 4)

        assert verdict["ok"] is False
        assert verdict["blocking"] is True
        assert "rows" in _all_failures(verdict).lower()

    def test_an_added_or_removed_column_is_blocking(self):
        report = self._resync_report(
            changed={"home_rest_days": ["2025"]}, removed=("away_rest_days",)
        )
        for matrix in GOLD_MATRICES:
            report[matrix]["width_after"] -= 1

        verdict = attribute_rung(report, 4)

        assert verdict["ok"] is False
        assert verdict["blocking"] is True
        assert "away_rest_days" in _all_failures(verdict)


# ---------------------------------------------------------------------------
# Determinism of the attribution itself
# ---------------------------------------------------------------------------


class TestAttributionIsDeterministic:
    """A JSON document's key order carries no meaning; a verdict sensitive to it is wrong."""

    def _base_report(self) -> dict:
        return _pre_drop_report(
            changed={
                "line_movement_coverage": ["2023"],
                "saturday_game": ["2019"],
                "home_rest_days": ["2021"],
            },
            discrete=("line_movement_coverage", "saturday_game"),
        )

    @staticmethod
    def _shuffled(report: dict, seed: int = 7) -> dict:
        rng = random.Random(seed)
        out = copy.deepcopy(report)
        for detail in out.values():
            for key in ("columns_added", "columns_removed"):
                rng.shuffle(detail[key])
            for bucket in ("columns_changed", "column_details"):
                items = list(detail[bucket].items())
                rng.shuffle(items)
                detail[bucket] = dict(items)
        return out

    @staticmethod
    def _recased(report: dict, column: str) -> dict:
        out = copy.deepcopy(report)
        for detail in out.values():
            for bucket in ("columns_changed", "column_details"):
                detail[bucket] = {
                    (column.upper() if name == column else name): value
                    for name, value in detail[bucket].items()
                }
        return out

    def test_verdict_is_identical_under_shuffling_and_recasing(self):
        base = self._base_report()

        plain = attribute_rung(base, 1)
        shuffled = attribute_rung(self._shuffled(base), 1)
        recased = attribute_rung(self._recased(base, "home_rest_days"), 1)

        assert plain == shuffled
        assert plain == recased
        for verdict in (plain, shuffled, recased):
            for matrix in GOLD_MATRICES:
                assert verdict["matrices"][matrix]["attributed"] == [
                    "line_movement_coverage",
                    "saturday_game",
                ]
                assert verdict["matrices"][matrix]["unattributed"] == ["home_rest_days"]

    def test_a_case_only_rename_is_reported_as_a_rename_not_an_add_plus_remove(self):
        widths = _widths()
        report = {
            matrix: _detail(
                width_before=widths[matrix],
                width_after=widths[matrix],
                added=("Home_Rest_Days",),
                removed=("home_rest_days",),
                changed={"line_movement_coverage": ["2023"]},
                discrete=("line_movement_coverage",),
            )
            for matrix in GOLD_MATRICES
        }
        verdict = attribute_rung(report, 1)

        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["renamed_case_only"] == [
                ["home_rest_days", "Home_Rest_Days"]
            ]
            assert verdict["matrices"][matrix]["unattributed"] == []
        assert verdict["ok"] is False, (
            "a rename is still a finding, just not an add+remove"
        )
        assert "case-only rename" in _all_failures(verdict)

    def test_the_verdict_is_json_serializable(self):
        verdict = attribute_rung(self._base_report(), 1)
        assert json.loads(json.dumps(verdict)) == verdict


class TestExpectedSignature:
    """Every rung has a predicted signature and an unknown rung is refused."""

    @pytest.mark.parametrize("rung", [1, 2, 3, 4])
    def test_every_rung_has_a_signature_naming_its_cause(self, rung: int):
        signature = _expected_signature(rung)
        assert signature["rung"] == rung
        assert signature["cause"] == RUNG_CAUSES[rung]
        for key in (
            "columns_added",
            "columns_removed",
            "columns_changed",
            "rows",
            "width",
        ):
            assert key in signature

    def test_rung_3_signature_derives_the_family_from_the_before_document(self):
        before = _before_document()
        signature = _expected_signature(3, before=before)
        assert signature["columns_removed"]["features_ats"] == sorted(
            _line_movement_family()
        )

    @pytest.mark.parametrize("rung", [0, 5, -1])
    def test_an_unknown_rung_raises(self, rung: int):
        with pytest.raises(ValueError, match="rung"):
            _expected_signature(rung)


# ---------------------------------------------------------------------------
# fingerprint_matrix / compare_fingerprints metadata
# ---------------------------------------------------------------------------


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2024_W01_A@B", "2024_W02_C@D", "2024_W03_E@F"],
            "season": [2024, 2024, 2024],
            "home_rest_days": [7.0, 6.0, None],
            "saturday_game": [0.0, 1.0, 0.0],
        }
    )


class TestFingerprintColumnMetadata:
    """A fingerprint that records dtype / null-count can say WHY a column moved."""

    def test_fingerprint_matrix_records_dtype_null_count_and_discreteness(self):
        result = fingerprint_matrix(_tiny_frame())

        meta = result["column_meta"]
        assert meta["home_rest_days"]["dtype"] == "float64"
        assert meta["home_rest_days"]["null_count"] == 1
        assert meta["home_rest_days"]["discrete_indicator"] is False
        assert meta["saturday_game"]["discrete_indicator"] is True
        assert meta["saturday_game"]["null_count"] == 0

    def test_the_per_season_hash_structure_is_unchanged(self):
        result = fingerprint_matrix(_tiny_frame())
        assert set(result["columns"]) == set(_tiny_frame().columns)
        assert set(result["columns"]["home_rest_days"]) == {"2024"}
        assert result["rows"] == 3
        assert result["width"] == 4

    def test_a_dtype_only_move_still_counts_as_moved(self):
        before = {"features_wp": fingerprint_matrix(_tiny_frame())}
        after = copy.deepcopy(before)
        after["features_wp"]["column_meta"]["saturday_game"]["dtype"] = "int64"

        detail = compare_fingerprints(before, after)["features_wp"]

        assert "saturday_game" in detail["columns_changed"]
        assert detail["columns_changed"]["saturday_game"] == []
        assert detail["column_details"]["saturday_game"]["reasons"] == ["dtype"]

    def test_a_null_count_only_move_still_counts_as_moved(self):
        before = {"features_wp": fingerprint_matrix(_tiny_frame())}
        after = copy.deepcopy(before)
        after["features_wp"]["column_meta"]["saturday_game"]["null_count"] = 2

        detail = compare_fingerprints(before, after)["features_wp"]

        assert detail["column_details"]["saturday_game"]["reasons"] == ["null_count"]

    def test_rows_per_season_is_carried_into_the_comparison(self):
        before = {"features_wp": fingerprint_matrix(_tiny_frame())}
        detail = compare_fingerprints(before, copy.deepcopy(before))["features_wp"]
        assert detail["rows_per_season_before"] == {"2024": 3}
        assert detail["rows_per_season_after"] == {"2024": 3}

    def test_an_unchanged_pair_reports_no_moved_column(self):
        before = {"features_wp": fingerprint_matrix(_tiny_frame())}
        detail = compare_fingerprints(before, copy.deepcopy(before))["features_wp"]
        assert detail["columns_changed"] == {}
        assert detail["columns_added"] == []
        assert detail["columns_removed"] == []


# ---------------------------------------------------------------------------
# The Phase-31 rung ladder -- ordering, naming, and the move-kind report
# ---------------------------------------------------------------------------


class TestTheRungLadderCannotBeRunOutOfOrder:
    """Rung 0 must exist (T-31-12).

    D31-09 describes rung 1 as a full rebuild proving the build still reproduces
    CURRENT gold. That is only checkable against a fingerprint of current gold taken
    BEFORE rung 1 overwrites it -- and once the rebuild has run, the artifact the
    baseline was supposed to describe is gone. There is no recovering it afterwards,
    so the refusal has to arrive before the rebuild, not as a diagnosis after it.
    """

    def test_rung_1_refuses_when_rung_0_is_absent(self, tmp_path: Path):
        report = _pre_drop_report(
            changed={"line_movement_coverage": ["2023"]},
            discrete=("line_movement_coverage",),
        )

        with pytest.raises(MissingPredecessorFingerprintError) as excinfo:
            attribute_rung(
                report, 1, ladder_directory=tmp_path, rung_prefix=PHASE31_RUNG_PREFIX
            )

        message = str(excinfo.value)
        assert "rung 1" in message
        assert f"{PHASE31_RUNG_PREFIX}rung0.json" in message
        assert "--rung 0" in message, "the refusal must say how to fix itself"

    def test_the_refusal_is_a_NAMED_error_not_a_FileNotFoundError(self):
        """The two say different things: a wrong path, versus a wrong ORDER."""
        assert not issubclass(MissingPredecessorFingerprintError, FileNotFoundError)
        assert issubclass(MissingPredecessorFingerprintError, RuntimeError)

    def test_the_whole_chain_is_required_not_only_the_immediate_predecessor(
        self, tmp_path: Path
    ):
        """A rung-1 document that was never judged against rung 0 is a missing link."""
        rung_document_path(tmp_path, 1, PHASE31_RUNG_PREFIX).write_text(
            "{}", encoding="utf-8"
        )

        with pytest.raises(MissingPredecessorFingerprintError) as excinfo:
            attribute_rung(
                _pre_drop_report(changed={"home_rest_days": ["2010"]}),
                2,
                ladder_directory=tmp_path,
                rung_prefix=PHASE31_RUNG_PREFIX,
            )

        assert f"{PHASE31_RUNG_PREFIX}rung0.json" in str(excinfo.value)

    def test_a_complete_ladder_attributes_normally(self, tmp_path: Path):
        for rung in (0, 1):
            rung_document_path(tmp_path, rung, PHASE31_RUNG_PREFIX).write_text(
                "{}", encoding="utf-8"
            )

        verdict = attribute_rung(
            _pre_drop_report(changed={"home_rest_days": ["2010"]}),
            2,
            ladder_directory=tmp_path,
            rung_prefix=PHASE31_RUNG_PREFIX,
        )

        assert verdict["ok"] is True

    def test_require_rung_ladder_returns_every_document_it_verified(
        self, tmp_path: Path
    ):
        for rung in (0, 1, 2):
            rung_document_path(tmp_path, rung, PHASE31_RUNG_PREFIX).write_text(
                "{}", encoding="utf-8"
            )

        verified = require_rung_ladder(tmp_path, 3, PHASE31_RUNG_PREFIX)

        assert [path.name for path in verified] == [
            f"{PHASE31_RUNG_PREFIX}rung{rung}.json" for rung in (0, 1, 2)
        ]

    def test_the_check_is_opt_in_so_a_hand_built_report_still_attributes(self):
        """Every test above this section judges reports that have no ladder on disk."""
        verdict = attribute_rung(
            _pre_drop_report(changed={"home_rest_days": ["2010"]}), 2
        )
        assert verdict["ok"] is True


class TestTheOrderingRefusalFiresOnALadderAndNotOnAnAdHocComparison:
    """A refusal that fires on the wrong thing is the kind operators learn to override.

    ``RUNBOOK.md`` section 11 documents ``--compare before.json after.json
    --attribute-rung 2``, which is a one-off comparison that never claimed to be a
    rung of anything. Demanding a rung-0 baseline of it would refuse a correct
    command -- and on a fresh checkout, where ``outputs/`` is gitignored and empty, it
    would refuse EVERY such command. The gate therefore asks whether the run CLAIMS to
    be a rung of a ladder, and only then enforces the order.
    """

    def _args(self, tmp_path: Path, before_name: str, prefix: str = ""):
        from scripts.fingerprint_gold import build_parser

        return build_parser().parse_args(
            [
                "--compare",
                str(tmp_path / before_name),
                str(tmp_path / "after.json"),
                "--attribute-rung",
                "2",
                "--fingerprint-dir",
                str(tmp_path),
                *(["--rung-prefix", prefix] if prefix else []),
            ]
        )

    def test_an_ad_hoc_before_after_comparison_is_not_a_ladder_run(
        self, tmp_path: Path
    ):
        from scripts.fingerprint_gold import _is_ladder_run

        assert _is_ladder_run(self._args(tmp_path, "before.json")) is False

    def test_a_before_document_that_IS_rung_n_minus_1_is_a_ladder_run(
        self, tmp_path: Path
    ):
        from scripts.fingerprint_gold import _is_ladder_run

        assert _is_ladder_run(self._args(tmp_path, "rung1.json")) is True

    def test_a_prefixed_run_is_always_a_ladder_run(self, tmp_path: Path):
        from scripts.fingerprint_gold import _is_ladder_run

        assert (
            _is_ladder_run(
                self._args(tmp_path, "anything.json", prefix=PHASE31_RUNG_PREFIX)
            )
            is True
        ), (
            "--rung-prefix DECLARES that this run belongs to a named ladder, so the "
            "ordering check must fire regardless of what the documents are called"
        )


class TestPhase31DocumentsCannotOverwriteThePhase30Record:
    """T-31-11: the Phase-30 rung documents are evidence that cannot be regenerated."""

    def test_the_prefixed_path_carries_the_prefix(self):
        path = rung_document_path(Path("outputs/fingerprints"), 0, PHASE31_RUNG_PREFIX)

        assert path.name == "p31_rung0.json"
        assert path.name.startswith(PHASE31_RUNG_PREFIX)

    @pytest.mark.parametrize("rung", [0, 1, 2, 3, 4])
    def test_no_prefixed_path_collides_with_a_phase30_document(self, rung: int):
        directory = Path("outputs/fingerprints")
        prefixed = rung_document_path(directory, rung, PHASE31_RUNG_PREFIX)

        assert prefixed.name not in PHASE30_RUNG_DOCUMENTS
        assert prefixed != rung_document_path(directory, rung)

    def test_the_cli_exposes_the_rung_name_arguments_with_safe_defaults(self):
        from scripts.fingerprint_gold import FINGERPRINT_DIR, build_parser

        args = build_parser().parse_args([])

        assert args.rung is None
        assert args.rung_prefix == "", (
            "the default prefix must be the Phase-30 naming, so an existing documented "
            "command keeps writing where it always did"
        )
        assert args.fingerprint_dir == FINGERPRINT_DIR

    def test_the_cli_prefix_composes_into_the_prefixed_document_path(self):
        from scripts.fingerprint_gold import build_parser

        args = build_parser().parse_args(
            ["--rung", "2", "--rung-prefix", PHASE31_RUNG_PREFIX]
        )
        derived = rung_document_path(args.fingerprint_dir, args.rung, args.rung_prefix)

        assert derived.name == "p31_rung2.json"
        assert derived.name not in PHASE30_RUNG_DOCUMENTS

    def test_the_unprefixed_path_is_exactly_the_phase30_naming(self):
        """The Phase-30 documents are named by the same function, so the two are checked
        against each other rather than against a transcribed list."""
        directory = Path("outputs/fingerprints")
        names = [rung_document_path(directory, rung).name for rung in range(5)]

        assert tuple(names) == PHASE30_RUNG_DOCUMENTS


class TestTheRungReportCarriesTheMoveKind:
    """A rung report says WHY each column moved, not merely THAT it moved."""

    def test_every_moved_column_carries_a_kind_and_its_seasons(self):
        report = _pre_drop_report(
            changed={"line_movement_coverage": ["2023"], "home_rest_days": ["2021"]},
            discrete=("line_movement_coverage",),
        )
        verdict = attribute_rung(report, 1)

        for matrix in GOLD_MATRICES:
            kinds = verdict["matrices"][matrix]["move_kinds"]
            assert set(kinds) == {"line_movement_coverage", "home_rest_days"}
            assert kinds["home_rest_days"]["kind"] == "values"
            assert kinds["home_rest_days"]["seasons"] == ["2021"]

    def test_a_storage_move_reports_its_season_and_kind_rather_than_an_empty_list(self):
        """The reporting shape Plan 31-11's hard stop needs in order to judge at all."""
        before = {"features_wp": fingerprint_matrix(_tiny_frame())}
        after = copy.deepcopy(before)
        after["features_wp"]["column_meta_by_season"]["saturday_game"]["2024"][
            "null_count"
        ] = 1

        verdict = attribute_rung(compare_fingerprints(before, after), 2)
        kinds = verdict["matrices"]["features_wp"]["move_kinds"]

        assert kinds["saturday_game"]["kind"] == "storage"
        assert kinds["saturday_game"]["seasons"] == ["2024"]
        assert kinds["saturday_game"]["seasons_storage"] == ["2024"]
        assert kinds["saturday_game"]["seasons_values"] == []

    def test_the_two_summary_lists_are_disjoint_and_cover_the_moved_set(self):
        clock = BUILD_CLOCK_COLUMNS[0]
        report = _pre_drop_report(
            changed={clock: ["2024"], "home_rest_days": ["2021"]},
        )
        verdict = attribute_rung(report, 2)

        assert verdict["non_clock_moves"] == ["home_rest_days"]
        assert verdict["build_clock_moves"] == [clock]
        assert (
            set(verdict["non_clock_moves"]) & set(verdict["build_clock_moves"]) == set()
        )

    def test_the_clock_is_reported_in_the_summary_even_though_it_is_split_out(self):
        """Split out of the CHANGED set, never out of the REPORT."""
        clock = BUILD_CLOCK_COLUMNS[0]
        verdict = attribute_rung(_pre_drop_report(changed={clock: ["2024"]}), 2)

        assert verdict["build_clock_moves"] == [clock]
        for matrix in GOLD_MATRICES:
            assert verdict["matrices"][matrix]["move_kinds"][clock]["kind"] == (
                "build_clock"
            )

    def test_a_zero_non_clock_move_condition_is_reachable_for_a_clock_stamping_build(
        self,
    ):
        """REVIEW-CLOCK: "zero moved columns" is not reachable; this is."""
        clock = BUILD_CLOCK_COLUMNS[0]
        verdict = attribute_rung(_pre_drop_report(changed={clock: ["2024"]}), 2)

        assert verdict["non_clock_moves"] == []
        assert set(verdict["build_clock_moves"]) == {
            _canonical_name(name) for name in BUILD_CLOCK_COLUMNS
        }

    def test_the_printed_report_names_the_kind_and_the_seasons(self, capsys):
        from scripts.fingerprint_gold import _print_attribution

        report = _pre_drop_report(changed={"home_rest_days": ["2021"]})
        _print_attribution(attribute_rung(report, 2))

        printed = capsys.readouterr().out
        assert "home_rest_days [values] seasons 2021" in printed
        assert "non-clock moves: ['home_rest_days']" in printed

    def test_the_verdict_stays_json_serializable_with_the_new_keys(self):
        verdict = attribute_rung(
            _pre_drop_report(changed={"home_rest_days": ["2021"]}), 2
        )
        assert json.loads(json.dumps(verdict)) == verdict


def _canonical_name(name: str) -> str:
    """Lower-case, matching ``scripts.fingerprint_gold._canonical``."""
    return name.lower()


@pytest.mark.integration
class TestFingerprintDeterminism:
    """Re-running fingerprint_gold on unchanged gold must serialize identically."""

    def test_two_runs_on_unchanged_gold_serialize_identically(self):
        missing = [
            matrix
            for matrix in GOLD_MATRICES
            if not (_GOLD_DIR / f"{matrix}.parquet").exists()
        ]
        if missing:
            pytest.skip(
                f"live gold matrices absent ({', '.join(missing)}) -- "
                "run `python -m scripts.build_features --all` to populate data/gold, "
                "or ignore on a fresh checkout where data/ is legitimately empty"
            )

        first = json.dumps(fingerprint_gold(), indent=2, sort_keys=True)
        second = json.dumps(fingerprint_gold(), indent=2, sort_keys=True)

        assert first == second, (
            "fingerprint_gold is not deterministic on unchanged gold -- every rung's "
            "attribution rests on the assumption that a re-run hashes identically"
        )


# ---------------------------------------------------------------------------
# The Phase-31 THREE-RUNG LADDER over LIVE gold (Plan 31-11, D31-09 / D31-10)
# ---------------------------------------------------------------------------
#
# TEST CLASS: integration. Every live class below reads the ``p31_``-prefixed rung
# documents Plan 31-11 wrote under ``outputs/fingerprints/``. ``outputs/`` is gitignored,
# so on a checkout that never ran the ladder these skip with a REGISTERED reason ("not
# present at ...", see ``tests/conftest._EVIDENCE_SKIP_MARKERS``) rather than passing
# vacuously. ``TestTheProtectedSliceTripwireCanActuallyFire`` is a plain unit test and
# runs everywhere.
#
# The ladder's three rungs and what each one is FOR:
#
#   rung 0 -- the fingerprint of gold as it stood BEFORE anything in Phase 31 wrote.
#             Taken first because rung 1's claim ("today's code still reproduces today's
#             gold") is only checkable against a baseline captured before rung 1
#             overwrote it. Once the rebuild has run, that baseline cannot be recovered.
#   rung 1 -- a FULL rebuild on today's code with NO new odds. Isolates code state as a
#             cause, so any move at rung 2 is attributable to the odds and nothing else.
#   rung 2 -- the 2025 odds ingested and a SCOPED --season 2025 incremental rebuild.
#
# THE RUNG CONDITIONS ARE DELIBERATELY DIFFERENT AT THE TWO RUNGS, and the difference is
# informative rather than inconsistent:
#
# * At rung 1 the build stamps a fresh clock into ``feature_timestamp`` for EVERY row
#   (``scripts/build_features.py:570``), so the clock moves in every season. The reachable
#   condition is "zero NON-clock moves, with the moved set exactly EQUAL to the registered
#   clock set" -- NOT "zero moved columns", which is structurally unreachable
#   (REVIEW-CLOCK). Equality, never containment: a rung where the clock did NOT move is as
#   wrong as one where an unregistered column did.
# * At rung 2 the incremental latest-wins path CARRIES FORWARD the retained rows rather
#   than re-deriving them, so their clock is untouched. A build-clock move attributed to
#   any season in 2021-2024 there means those rows WERE rewritten -- the replace-mode
#   catastrophe the scoped invocation exists to avoid -- so it is a HARD STOP, not an
#   exemption.

_P31_LADDER_DIR = REPO_ROOT / "outputs" / "fingerprints"

# The strict slice the hard stop measures. 2021-2024 is the deploy gate's holdout AND the
# Phase-31 tune window; 2025 is the single unburned hold the rebuild is FOR.
_P31_PROTECTED_SEASONS = ("2021", "2022", "2023", "2024")
_P31_HOLD_SEASON = "2025"

# D31-38: playoffs are admitted in both windows, so the 2025 hold is 285 games, not 272.
_P31_HOLD_ROWS = 285

_P31_CLOCK_COLUMNS = tuple(sorted(_canonical_name(c) for c in BUILD_CLOCK_COLUMNS))


def _p31_rung_path(rung: int) -> Path:
    return rung_document_path(_P31_LADDER_DIR, rung, PHASE31_RUNG_PREFIX)


def _p31_document(rung: int) -> dict:
    """Load a Phase-31 rung document, or skip with a REGISTERED evidence reason."""
    path = _p31_rung_path(rung)
    if not path.is_file():
        pytest.skip(
            f"the Phase-31 rung-{rung} fingerprint document is not present at {path} -- "
            "outputs/ is gitignored runtime state, written by Plan 31-11's ladder run."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _p31_require_protected_seasons(document: dict, rung: int) -> None:
    """Fail (never pass vacuously) unless the document actually carries 2021-2024.

    Every assertion below is a statement ABOUT those seasons. A document that did not
    contain them would satisfy each one for free, which is the shape of a tripwire that
    cannot fire.
    """
    for matrix in GOLD_MATRICES:
        per_season = document.get(matrix, {}).get("rows_per_season", {})
        missing = [s for s in _P31_PROTECTED_SEASONS if s not in per_season]
        assert not missing, (
            f"rung-{rung} document's {matrix} carries no rows for season(s) {missing}, "
            "so every 2021-2024 assertion in this module would pass for free. The ladder "
            "is judging the wrong artifact."
        )


def _p31_moves(report: dict) -> list[tuple[str, str, str, list[str]]]:
    """Flatten a comparison report into (matrix, column, move_kind, seasons) rows."""
    rows: list[tuple[str, str, str, list[str]]] = []
    for matrix in GOLD_MATRICES:
        detail = report.get(matrix, {})
        for column, seasons in (detail.get("columns_changed") or {}).items():
            kind = ((detail.get("column_details") or {}).get(column) or {}).get(
                "move_kind", "unknown"
            )
            rows.append((matrix, column, kind, sorted(seasons)))
    return rows


def _p31_moves_touching_protected_seasons(
    report: dict,
) -> list[tuple[str, str, str, list[str]]]:
    """Return every moved column -- of ANY kind -- carrying a 2021-2024 season.

    ANY kind is the point (D31-10). Adding 285 real 2025 rows can flip an ``int64`` to a
    ``float64``, which genuinely rewrites the 2021-2024 bytes on disk even when every
    value is equal, so a storage move is judged exactly as a value move is. The build
    clock is included too: at rung 2 a clock move in a retained season means those rows
    were rewritten.
    """
    protected = set(_P31_PROTECTED_SEASONS)
    return [move for move in _p31_moves(report) if protected & set(move[3])]


def _p31_rows_per_season(document: dict, matrix: str) -> dict[str, int]:
    return {
        season: int(rows)
        for season, rows in (
            document.get(matrix, {}).get("rows_per_season", {}) or {}
        ).items()
    }


@pytest.mark.integration
class TestThePhase31Rung1ReproducesCurrentGold:
    """Rung 1: a full rebuild on today's code, with NO new odds, moves only the clock.

    This rung exists so that a move at rung 2 has exactly ONE candidate cause. Without it,
    a moved column after the 2025 ingest could be the odds OR a pre-existing reproduction
    failure in the build, and no evidence would separate them.
    """

    def test_rung_0_was_taken_before_rung_1_overwrote_gold(self) -> None:
        """The ladder is an ORDER, and rung 0 is the link that cannot be recovered."""
        rung0, rung1 = _p31_rung_path(0), _p31_rung_path(1)
        if not rung1.is_file():
            pytest.skip(
                f"the Phase-31 rung-1 fingerprint document is not present at {rung1} -- "
                "outputs/ is gitignored runtime state."
            )
        assert rung0.is_file(), (
            f"rung 1 exists at {rung1} but its rung-0 predecessor does not exist at "
            f"{rung0}. Rung 1 REBUILT gold, so the baseline it was supposed to be judged "
            "against has already been overwritten and cannot be recovered."
        )
        assert rung0.stat().st_mtime <= rung1.stat().st_mtime, (
            "the rung-0 document is NEWER than the rung-1 document, so it cannot be a "
            "fingerprint of the gold that rung 1 rebuilt over."
        )

    def test_no_non_clock_column_moved(self) -> None:
        before, after = _p31_document(0), _p31_document(1)
        _p31_require_protected_seasons(before, 0)
        report = compare_fingerprints(before, after)

        for matrix in GOLD_MATRICES:
            non_clock = report[matrix]["non_clock_moves"]
            details = report[matrix].get("column_details") or {}
            attribution = "; ".join(
                f"{c} [{(details.get(c) or {}).get('move_kind')}] seasons "
                f"{','.join((details.get(c) or {}).get('seasons') or [])}"
                for c in non_clock
            )
            assert non_clock == [], (
                f"{matrix}: rung 1 moved {len(non_clock)} NON-CLOCK column(s) on a "
                f"rebuild that changed no input: {non_clock}. Per-column attribution: "
                f"{attribution}. This is a PRE-EXISTING reproduction failure with nothing "
                "to do with the 2025 odds, and separating that cause from the odds is "
                "exactly what rung 1 exists for. HARD STOP -- do not proceed to the "
                "ingest."
            )

    def test_the_moved_set_EQUALS_the_registered_clock_set(self) -> None:
        """Set EQUALITY, not containment. A clock that did NOT move also stops the run."""
        report = compare_fingerprints(_p31_document(0), _p31_document(1))

        for matrix in GOLD_MATRICES:
            moved_clock = tuple(
                sorted(_canonical_name(c) for c in report[matrix]["build_clock_moves"])
            )
            assert moved_clock == _P31_CLOCK_COLUMNS, (
                f"{matrix}: the rung-1 build-clock moved set is {list(moved_clock)}, "
                f"which is not EQUAL to the registered set {list(_P31_CLOCK_COLUMNS)}. A "
                "strict SUBSET means the per-build clock did not move, so either the "
                "rebuild did not run or the comparison is not reading the rebuilt frame; "
                "a superset is impossible by construction, because an unregistered column "
                "lands in non_clock_moves. An exemption that only ever widens is an "
                "exemption that can hide a real move."
            )

    def test_no_column_was_added_or_removed_and_the_shape_held(self) -> None:
        report = compare_fingerprints(_p31_document(0), _p31_document(1))
        for matrix in GOLD_MATRICES:
            detail = report[matrix]
            assert detail["columns_added"] == [], (
                f"{matrix}: rung 1 ADDED {detail['columns_added']}; a reproduction "
                "rebuild adds no column"
            )
            assert detail["columns_removed"] == [], (
                f"{matrix}: rung 1 REMOVED {detail['columns_removed']}; a reproduction "
                "rebuild removes no column"
            )
            assert detail["width_before"] == detail["width_after"], (
                f"{matrix}: width moved {detail['width_before']} -> "
                f"{detail['width_after']} on a reproduction rebuild"
            )
            assert detail["rows_before"] == detail["rows_after"], (
                f"{matrix}: rows moved {detail['rows_before']} -> "
                f"{detail['rows_after']} on a reproduction rebuild"
            )

    def test_the_protected_season_row_counts_are_identical(self) -> None:
        before, after = _p31_document(0), _p31_document(1)
        _p31_require_protected_seasons(before, 0)
        for matrix in GOLD_MATRICES:
            b = _p31_rows_per_season(before, matrix)
            a = _p31_rows_per_season(after, matrix)
            for season in _P31_PROTECTED_SEASONS:
                assert b[season] == a[season], (
                    f"{matrix}: season {season} row count moved {b[season]} -> "
                    f"{a[season]} across rung 1. A reproduction rebuild adds and drops no "
                    "row in a season whose inputs did not change."
                )


@pytest.mark.integration
class TestThePhase31Rung2AttributesEveryMoveTo2025:
    """Rung 2: the scoped 2025 rebuild may move 2025 and NOTHING else.

    The hard stop measures a STRICT 2021-2024 slice INCLUDING storage-level moves (D31-10)
    and INCLUDING the build clock. A 2025-only null-count or dtype change is expected and
    is not a stop; a 2021-2024 move of any kind is.
    """

    def test_no_column_of_any_kind_moved_in_a_protected_season(self) -> None:
        before, after = _p31_document(1), _p31_document(2)
        _p31_require_protected_seasons(before, 1)
        _p31_require_protected_seasons(after, 2)
        report = compare_fingerprints(before, after)

        offenders = _p31_moves_touching_protected_seasons(report)
        attribution = "; ".join(
            f"{matrix}.{column} [{kind}] seasons {','.join(seasons)}"
            for matrix, column, kind, seasons in offenders
        )
        assert offenders == [], (
            "rung 2 moved column(s) in the PROTECTED 2021-2024 slice, which the scoped "
            f"incremental build must never touch. Per-column attribution: {attribution}. "
            "A move here means the 2025 slice was written as the whole table, or that a "
            "whole-frame statistic reached back into retained history. HARD STOP."
        )

    def test_the_build_clock_moved_in_2025_alone(self) -> None:
        """Proof that the incremental path CARRIED the retained rows' clock forward.

        The scoped build re-derives only the 2025 rows; the latest-wins merge concatenates
        them onto the retained history untouched. So the clock -- the one column that
        moves in EVERY season of a full rebuild -- must move in 2025 and in no other
        season. That asymmetry against rung 1 is the single cheapest proof that the write
        mode was the incremental one and not replace.
        """
        report = compare_fingerprints(_p31_document(1), _p31_document(2))

        seen = False
        for matrix in GOLD_MATRICES:
            detail = report[matrix]
            for column in detail["build_clock_moves"]:
                seen = True
                seasons = sorted((detail["columns_changed"] or {})[column])
                assert seasons == [_P31_HOLD_SEASON], (
                    f"{matrix}: the build clock '{column}' moved in season(s) "
                    f"{','.join(seasons)}, not in {_P31_HOLD_SEASON} alone. Every season "
                    "outside 2025 whose clock moved had its rows REWRITTEN by this build, "
                    "which is the replace-mode catastrophe the scoped invocation exists "
                    "to avoid."
                )
        assert seen, (
            "rung 2 moved no build-clock column at all, so the 2025 slice was never "
            "re-derived and the rebuild did not do what it claimed."
        )

    def test_every_moved_column_is_attributed_to_2025_with_a_kind(self) -> None:
        report = compare_fingerprints(_p31_document(1), _p31_document(2))
        moves = _p31_moves(report)

        assert moves, (
            "rung 2 moved NO column. Ingesting 285 real 2025 market anchors must move the "
            "2025 slice; an empty diff means the rebuild did not do what it claimed."
        )
        for matrix, column, kind, seasons in moves:
            assert seasons == [_P31_HOLD_SEASON], (
                f"{matrix}.{column} [{kind}] moved in season(s) {','.join(seasons)}; "
                f"every rung-2 move must be attributed to {_P31_HOLD_SEASON} alone."
            )
            assert kind in ("values", "storage", "build_clock"), (
                f"{matrix}.{column} moved with no recorded move kind ({kind!r}). A move "
                "reported without a kind cannot be judged."
            )

    def test_no_column_was_added_or_removed_and_the_width_held(self) -> None:
        report = compare_fingerprints(_p31_document(1), _p31_document(2))
        for matrix in GOLD_MATRICES:
            detail = report[matrix]
            assert detail["columns_added"] == [], (
                f"{matrix}: rung 2 ADDED {detail['columns_added']}. The scoped build's "
                "narrowing guard protects the width; an ADDED column means the schema "
                "moved on an incremental path."
            )
            assert detail["columns_removed"] == [], (
                f"{matrix}: rung 2 REMOVED {detail['columns_removed']}"
            )
            assert detail["width_before"] == detail["width_after"], (
                f"{matrix}: width moved {detail['width_before']} -> "
                f"{detail['width_after']} at rung 2"
            )

    def test_the_protected_season_row_counts_are_identical_across_all_three_rungs(
        self,
    ) -> None:
        rung0, rung1, rung2 = _p31_document(0), _p31_document(1), _p31_document(2)
        _p31_require_protected_seasons(rung0, 0)
        for matrix in GOLD_MATRICES:
            counts = [_p31_rows_per_season(d, matrix) for d in (rung0, rung1, rung2)]
            for season in _P31_PROTECTED_SEASONS:
                observed = [c[season] for c in counts]
                assert len(set(observed)) == 1, (
                    f"{matrix}: season {season} row count moved across the ladder "
                    f"(rung0/1/2 = {observed}). Only 2025 may change."
                )

    def test_the_hold_season_is_the_playoff_inclusive_285_game_population(self) -> None:
        """D31-38: playoffs are IN, so the 2025 hold is 285 games, not 272."""
        after = _p31_document(2)
        for matrix in GOLD_MATRICES:
            rows = _p31_rows_per_season(after, matrix).get(_P31_HOLD_SEASON)
            assert rows == _P31_HOLD_ROWS, (
                f"{matrix}: the {_P31_HOLD_SEASON} slice holds {rows} rows, not the "
                f"{_P31_HOLD_ROWS} the frozen pre-registration binds the hold to (REG "
                "plus all four playoff types, D31-38)."
            )


class TestTheProtectedSliceTripwireCanActuallyFire:
    """Guard self-verification: a tripwire never shown to fire carries no information.

    Both live ladder classes above are expected to report an EMPTY offender list, which is
    also exactly what a broken detector would report. These cases drive the same helper
    over hand-built reports and require it to catch each shape the hard stop exists for --
    a value move, a storage-only move and a build-clock move -- in a protected season.
    """

    @staticmethod
    def _report(move_kind: str, seasons: list[str], column: str = "elo_diff") -> dict:
        return {
            matrix: {
                "columns_changed": {column: seasons},
                "column_details": {
                    column: {"move_kind": move_kind, "seasons": seasons}
                },
            }
            for matrix in GOLD_MATRICES
        }

    @pytest.mark.parametrize("move_kind", ["values", "storage", "build_clock"])
    def test_a_protected_season_move_of_any_kind_is_caught(
        self, move_kind: str
    ) -> None:
        offenders = _p31_moves_touching_protected_seasons(
            self._report(move_kind, ["2023"])
        )
        assert len(offenders) == len(GOLD_MATRICES), (
            f"a {move_kind} move in season 2023 was NOT caught by the protected-slice "
            "helper, so the rung-2 hard stop would wave it through."
        )

    def test_a_move_spanning_2025_and_a_protected_season_is_caught(self) -> None:
        offenders = _p31_moves_touching_protected_seasons(
            self._report("values", ["2024", "2025"])
        )
        assert len(offenders) == len(GOLD_MATRICES), (
            "a move that touches 2025 AND a protected season was not caught. A 2025 "
            "component does not license the 2024 one."
        )

    def test_a_2025_only_move_is_not_caught(self) -> None:
        offenders = _p31_moves_touching_protected_seasons(
            self._report("values", ["2025"])
        )
        assert offenders == [], (
            "a 2025-only move was flagged as a protected-slice offender. A tripwire that "
            "fires on the expected outcome gets overridden, and an overridden tripwire is "
            "worse than none."
        )
