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


# ---------------------------------------------------------------------------
# RUNG 3 -- the FULL rebuild that re-derives 2025 WITH its prior-season context
# (Plan 31-11 continuation, owner ruling A of 2026-09-05)
# ---------------------------------------------------------------------------
#
# WHY A FOURTH RUNG EXISTS AT ALL. Rung 2 was a SCOPED ``--season 2025`` build. On a
# 2025-only frame ``expanding_normalize``'s prior-season bootstrap finds nothing to
# bootstrap from and falls back to a neutral 0.0, and the per-season winsorization loses
# its strictly-prior bounds; the run's own log said ``No data for prior season
# season=2024``. 139/140/140 of 2025's columns moved for that reason rather than for the
# odds, and the independent project control ``tests/unit/test_temporal_display_columns.py
# ::TestRealGold`` failed on all three matrices. 2025 is the verdict population, so it must
# be derived the way every other season was. Rung 3 is that rebuild: full scope, replace
# mode, every season re-derived from the pinned upstream input.
#
# THE RUNG CONDITION IS DIFFERENT AGAIN, AND THE DIFFERENCE IS INFORMATIVE.
#
# * Rung 1 (full rebuild, no new odds): zero NON-clock moves anywhere; the clock moves in
#   every season.
# * Rung 2 (scoped incremental): zero moves of ANY kind in 2021-2024, INCLUDING the clock,
#   because the retained rows are carried forward rather than re-derived.
# * Rung 3 (full rebuild, odds in place): the clock moves in EVERY season again, exactly as
#   at rung 1, because every row is genuinely re-derived. So the protected-slice hard stop
#   here measures NON-CLOCK moves only. Requiring a frozen clock at a full rebuild would
#   fire on the rebuild having happened, and a tripwire that fires on the expected outcome
#   gets overridden.
#
# The substantive protection is undiminished: SPEC R2 binds the 2021-2024 VALUES, and a
# value or storage move in a protected season is caught here exactly as it is at rung 2.
# An unregistered column cannot reach the clock exemption either, because
# ``compare_fingerprints`` files anything outside ``BUILD_CLOCK_COLUMNS`` into
# ``non_clock_moves`` by construction.


def _p31_non_clock_moves_touching_protected_seasons(
    report: dict,
) -> list[tuple[str, str, str, list[str]]]:
    """Return every NON-CLOCK moved column carrying a 2021-2024 season.

    The rung-2 helper ``_p31_moves_touching_protected_seasons`` includes the build clock and
    must keep doing so: at a scoped incremental build a clock move in a retained season means
    those rows were rewritten. At a FULL rebuild every row is re-derived by design, so the
    clock is separated out here and judged by ``test_the_clock_moved_in_every_season``
    instead -- reported in its own category, never suppressed.
    """
    protected = set(_P31_PROTECTED_SEASONS)
    clock = {_canonical_name(c) for c in BUILD_CLOCK_COLUMNS}
    return [
        move
        for move in _p31_moves(report)
        if _canonical_name(move[1]) not in clock and protected & set(move[3])
    ]


@pytest.mark.integration
class TestThePhase31Rung3IsTheFullRebuildOfTheVerdictPopulation:
    """Rung 3: a FULL rebuild may move 2025 and may re-stamp the clock, and nothing else."""

    def test_the_ladder_predecessors_all_exist(self) -> None:
        """The 31-03 ordering refusal, asserted on the rung this plan actually ran."""
        if not _p31_rung_path(3).is_file():
            pytest.skip(
                "the Phase-31 rung-3 fingerprint document is not present at "
                f"{_p31_rung_path(3)} -- outputs/ is gitignored runtime state."
            )
        verified = require_rung_ladder(_P31_LADDER_DIR, 3, PHASE31_RUNG_PREFIX)
        assert [path.name for path in verified] == [
            "p31_rung0.json",
            "p31_rung1.json",
            "p31_rung2.json",
        ]

    def test_no_NON_CLOCK_column_moved_in_a_protected_season(self) -> None:
        """SPEC R2's hard stop, measured across the full rebuild.

        This is the assertion the owner's ruling A asked to be PROVEN rather than assumed.
        It is left exactly as strict as it reads: if it fails, the full rebuild moved bytes
        the pre-registration binds, and that is a finding for the owner rather than a
        condition to relax.
        """
        before, after = _p31_document(2), _p31_document(3)
        _p31_require_protected_seasons(before, 2)
        _p31_require_protected_seasons(after, 3)
        report = compare_fingerprints(before, after)

        offenders = _p31_non_clock_moves_touching_protected_seasons(report)
        attribution = "; ".join(
            f"{matrix}.{column} [{kind}] seasons {','.join(seasons)}"
            for matrix, column, kind, seasons in offenders
        )
        assert offenders == [], (
            f"the FULL rebuild moved {len(offenders)} non-clock column-slot(s) in the "
            f"PROTECTED 2021-2024 window. Per-column attribution: {attribution}. The "
            "2021-2024 slice is the deploy gate's holdout AND the Phase-31 tune window, and "
            "the pre-registration binds its values. HARD STOP -- take the attribution to the "
            "owner rather than relaxing this."
        )

    def test_the_clock_moved_in_every_season(self) -> None:
        """A full rebuild re-derives every row, so the registered clock must move everywhere.

        Set EQUALITY against the registered set, as at rung 1. A clock that moved in only
        some seasons would mean some rows were carried forward, which a replace-mode full
        rebuild does not do -- and a partial re-derivation is exactly the state in which a
        protected-slice assertion could pass for the wrong reason.
        """
        before, after = _p31_document(2), _p31_document(3)
        report = compare_fingerprints(before, after)
        expected_seasons = sorted(set(_P31_PROTECTED_SEASONS) | {_P31_HOLD_SEASON})

        for matrix in GOLD_MATRICES:
            detail = report[matrix]
            moved_clock = tuple(
                sorted(_canonical_name(c) for c in detail["build_clock_moves"])
            )
            assert moved_clock == _P31_CLOCK_COLUMNS, (
                f"{matrix}: the rung-3 build-clock moved set is {list(moved_clock)}, not "
                f"EQUAL to the registered set {list(_P31_CLOCK_COLUMNS)}. A full rebuild "
                "stamps every row, so a clock that did not move means the rebuild did not "
                "run over this matrix."
            )
            for column in detail["build_clock_moves"]:
                seasons = sorted((detail["columns_changed"] or {})[column])
                missing = [s for s in expected_seasons if s not in seasons]
                assert not missing, (
                    f"{matrix}: the build clock '{column}' did NOT move in season(s) "
                    f"{missing}. Those rows were carried forward rather than re-derived, so "
                    "this was not the full rebuild it claims to be."
                )

    def test_no_column_was_added_or_removed_and_the_shape_held(self) -> None:
        report = compare_fingerprints(_p31_document(2), _p31_document(3))
        for matrix in GOLD_MATRICES:
            detail = report[matrix]
            assert detail["columns_added"] == [], (
                f"{matrix}: rung 3 ADDED {detail['columns_added']}. A full rebuild is the "
                "only mode that CAN move the schema, which is exactly why it is asserted "
                "here rather than assumed."
            )
            assert detail["columns_removed"] == [], (
                f"{matrix}: rung 3 REMOVED {detail['columns_removed']}"
            )
            assert detail["width_before"] == detail["width_after"], (
                f"{matrix}: width moved {detail['width_before']} -> "
                f"{detail['width_after']} at rung 3"
            )
            assert detail["rows_before"] == detail["rows_after"], (
                f"{matrix}: rows moved {detail['rows_before']} -> "
                f"{detail['rows_after']} at rung 3"
            )

    def test_the_protected_season_row_counts_are_identical_across_all_four_rungs(
        self,
    ) -> None:
        documents = [_p31_document(rung) for rung in range(4)]
        _p31_require_protected_seasons(documents[0], 0)
        for matrix in GOLD_MATRICES:
            counts = [_p31_rows_per_season(d, matrix) for d in documents]
            for season in _P31_PROTECTED_SEASONS:
                observed = [c[season] for c in counts]
                assert len(set(observed)) == 1, (
                    f"{matrix}: season {season} row count moved across the ladder "
                    f"(rung0/1/2/3 = {observed}). Only 2025 may change."
                )

    def test_the_hold_season_is_still_the_playoff_inclusive_285_game_population(
        self,
    ) -> None:
        after = _p31_document(3)
        for matrix in GOLD_MATRICES:
            rows = _p31_rows_per_season(after, matrix).get(_P31_HOLD_SEASON)
            assert rows == _P31_HOLD_ROWS, (
                f"{matrix}: the {_P31_HOLD_SEASON} slice holds {rows} rows after the full "
                f"rebuild, not the {_P31_HOLD_ROWS} the frozen pre-registration binds the "
                "hold to (REG plus all four playoff types, D31-38)."
            )


class TestTheRung3ProtectedSliceHelperCanActuallyFire:
    """The rung-3 helper exempts the clock. Prove the exemption is narrow, not a hole."""

    @staticmethod
    def _report(column: str, move_kind: str, seasons: list[str]) -> dict:
        return {
            matrix: {
                "columns_changed": {column: seasons},
                "column_details": {
                    column: {"move_kind": move_kind, "seasons": seasons}
                },
            }
            for matrix in GOLD_MATRICES
        }

    @pytest.mark.parametrize("move_kind", ["values", "storage"])
    def test_a_protected_season_data_move_is_still_caught(self, move_kind: str) -> None:
        offenders = _p31_non_clock_moves_touching_protected_seasons(
            self._report("elo_diff", move_kind, ["2023"])
        )
        assert len(offenders) == len(GOLD_MATRICES), (
            f"a {move_kind} move in season 2023 was NOT caught by the rung-3 helper, so the "
            "full rebuild's hard stop would wave a real protected-slice move through."
        )

    def test_the_clock_is_the_ONLY_thing_the_rung_3_helper_exempts(self) -> None:
        exempted = _p31_non_clock_moves_touching_protected_seasons(
            self._report("feature_timestamp", "build_clock", ["2022"])
        )
        assert exempted == [], (
            "the registered build clock was flagged as a rung-3 protected-slice offender. A "
            "full rebuild re-stamps every row by construction; flagging it would fire on the "
            "rebuild having happened."
        )

    def test_a_column_merely_NAMED_like_a_clock_is_not_exempt(self) -> None:
        """A name heuristic is how a real move gets waved through wearing a clock's costume."""
        offenders = _p31_non_clock_moves_touching_protected_seasons(
            self._report("snapshot_ts", "values", ["2024"])
        )
        assert len(offenders) == len(GOLD_MATRICES), (
            "'snapshot_ts' is not in BUILD_CLOCK_COLUMNS but was exempted anyway, so the "
            "rung-3 exemption is matching on appearance rather than on registration."
        )

    def test_a_move_spanning_2025_and_a_protected_season_is_caught(self) -> None:
        offenders = _p31_non_clock_moves_touching_protected_seasons(
            self._report("snapshot_spread", "values", ["2024", "2025"])
        )
        assert len(offenders) == len(GOLD_MATRICES), (
            "a move that touches 2025 AND a protected season was not caught. A 2025 "
            "component does not license the 2024 one."
        )

    def test_a_2025_only_move_is_not_caught(self) -> None:
        offenders = _p31_non_clock_moves_touching_protected_seasons(
            self._report("snapshot_spread", "values", ["2025"])
        )
        assert offenders == [], (
            "a 2025-only move was flagged. Rung 3 exists to re-derive 2025 with its "
            "prior-season context; moving 2025 is the point."
        )


# ---------------------------------------------------------------------------
# THE PHASE-33.1 RUNG -- a COMPOUND cause under its own document prefix
# (Plan 33.1-07 Task 1, Rulings N and N2)
# ---------------------------------------------------------------------------
#
# TEST CLASS (this module's phase-wide rule -- every class declares its kind in its
# docstring): ``TestPhase331WeatherRoutingRung`` and
# ``TestPhase331TheDeliberateTripwiresInTheEditedModulesAreUnchanged`` are plain
# UNIT tests -- hand-built reports and an AST scan, so both pass on a fresh checkout
# with no ``data/``. ``TestPhase331StalenessGameIdsWereMeasuredBeforeTheRebuild`` is
# INTEGRATION / slow: it re-derives the 207 ids off live gold and live silver and
# skips with a remediation-carrying message when either is absent.
#
# WHY A PREFIX AND NOT A NEW INTEGER (Ruling N). ``RUNG_CAUSES`` is a closed
# four-entry dict keyed by a bare integer. Taking integer 5 would land in
# ``_expected_signature``'s ``else`` branch, which is rung 4's "rows strictly
# increased" signature, and this rung's rows are UNCHANGED at 6,499. Taking 5 under
# a fresh prefix would make ``require_rung_ladder`` demand four documents describing
# rebuilds this phase does not run. And Plan 33-14 wants the same dict for the Elo
# rung. So the cause lookup became PREFIX-AWARE, and the negative control below
# proves the new table did not capture the old integer.


def _p331_module():
    """The fingerprint module, imported inside the test rather than at module scope.

    Deliberate: the symbols this section exercises did not exist before Plan
    33.1-07 Task 1, and a module-scope ``from ... import`` of a name that is not
    there yet is a COLLECTION error -- which reports "no tests ran" rather than
    "this test failed", and is exactly the INVALID_RED shape a TDD gate must not
    accept as evidence.
    """
    import scripts.fingerprint_gold as module

    return module


def _p331_state():
    """``tests.phase33_state``, imported lazily for the same reason as above."""
    import tests.phase33_state as state

    return state


def _phase331_report(
    *,
    added: tuple[str, ...] = ("weather_coverage",),
    removed: tuple[str, ...] = (),
    changed: dict[str, list[str]] | None = None,
    width_delta: int = 1,
    rows_before: int = 6499,
    rows_after: int = 6499,
    discrete: tuple[str, ...] = (),
    became_discrete: tuple[str, ...] = (),
) -> dict:
    """A three-matrix report in the shape the Phase-33.1 rung PREDICTS.

    The defaults ARE the prediction: one column added (the coverage flag), nothing
    removed, rows unchanged at 6,499, width +1. Every test below states only its
    departure from that, rather than restating the whole shape each time.
    """
    widths = _widths()
    return {
        matrix: _detail(
            width_before=widths[matrix],
            width_after=widths[matrix] + width_delta,
            rows_before=rows_before,
            rows_after=rows_after,
            added=added,
            removed=removed,
            changed=dict(changed or {}),
            discrete=discrete,
            became_discrete=became_discrete,
        )
        for matrix in GOLD_MATRICES
    }


def _phase331_clean_changed() -> dict[str, list[str]]:
    """A changed set carrying exactly one member of each of the three families."""
    return {
        # family 1 -- features.weather.WEATHER_FEATURE_COLUMNS
        "temp_f": ["2002", "2019", "2024"],
        # family 2 -- the contextual builder's own emitted set. Ruling H puts
        # venue_cold_climate HERE rather than in the weather family, because it is
        # derived from the stadium's geography and from no weather observation.
        "venue_cold_climate": ["2004", "2019"],
        # family 3 -- the ROW-SCOPED staleness repair. A column outside families 1
        # and 2 may move ONLY in the season the 207 absent rows belong to.
        "home_qb_adjustment": ["2025"],
    }


class TestPhase331WeatherRoutingRung:
    """The rung's cause is COMPOUND and its three families are enumerable or derived.

    TEST CLASS: plain unit test. Hand-built ``compare_fingerprints``-shaped reports,
    so the contract is provable without running a rebuild and the class passes on a
    fresh checkout with no ``data/``.
    """

    def test_the_rung_cause_is_the_compound_label_and_names_all_three(self) -> None:
        """SPEC prohibition: this rebuild must never be called "the weather rung"."""
        f = _p331_module()
        verdict = attribute_rung(
            _phase331_report(changed=_phase331_clean_changed()),
            f.PHASE331_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        cause = verdict["cause"].lower()
        for word in ("weather", "routing", "coverage"):
            assert word in cause, (
                f"the Phase-33.1 rung's declared cause does not name {word!r}: "
                f"{verdict['cause']!r}. The rung carries THREE causes -- real ERA5 "
                "weather, the all-seasons stadium_id routing correction, and the "
                "restored 2025 coverage -- and a label naming only one of them "
                "re-creates the attribution failure this phase exists to prevent."
            )

    def test_a_diff_inside_the_three_families_attributes_cleanly(self) -> None:
        f = _p331_module()
        verdict = attribute_rung(
            _phase331_report(changed=_phase331_clean_changed()),
            f.PHASE331_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is True, _all_failures(verdict)
        for matrix in GOLD_MATRICES:
            detail = verdict["matrices"][matrix]
            assert detail["unattributed"] == []
            assert detail["changed_by_family"]["weather"] == ["temp_f"]
            assert detail["changed_by_family"]["venue"] == ["venue_cold_climate"]
            assert detail["changed_by_family"]["staleness_2025"] == [
                "home_qb_adjustment"
            ]

    def test_the_negative_control_the_same_report_at_rung_1_is_still_CR_02(
        self,
    ) -> None:
        """The new table must not have captured Phase 30's integer 1."""
        verdict = attribute_rung(_phase331_report(changed=_phase331_clean_changed()), 1)
        assert verdict["cause"] == RUNG_CAUSES[1] == "CR-02"

    def test_phase_30_rung_causes_still_holds_exactly_its_four_entries(self) -> None:
        assert RUNG_CAUSES == {
            1: "CR-02",
            2: "WR-06",
            3: "line_movement drop",
            4: "N-01",
        }

    def test_an_added_column_other_than_the_coverage_flag_is_refused(self) -> None:
        f = _p331_module()
        verdict = attribute_rung(
            _phase331_report(
                added=("weather_coverage", "some_new_column"),
                width_delta=2,
                changed=_phase331_clean_changed(),
            ),
            f.PHASE331_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is False
        assert "some_new_column" in _all_failures(verdict)

    def test_the_coverage_flag_MUST_be_added_and_its_absence_is_refused(self) -> None:
        """A rebuild that moved the right columns and forgot the flag is not this rung."""
        f = _p331_module()
        verdict = attribute_rung(
            _phase331_report(
                added=(), width_delta=0, changed=_phase331_clean_changed()
            ),
            f.PHASE331_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is False
        assert "weather_coverage" in _all_failures(verdict)

    def test_a_removed_column_is_refused(self) -> None:
        f = _p331_module()
        verdict = attribute_rung(
            _phase331_report(
                removed=("home_rest_days",),
                width_delta=0,
                changed=_phase331_clean_changed(),
            ),
            f.PHASE331_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is False
        assert "home_rest_days" in _all_failures(verdict)

    def test_a_changed_row_count_is_refused(self) -> None:
        f = _p331_module()
        verdict = attribute_rung(
            _phase331_report(rows_after=6706, changed=_phase331_clean_changed()),
            f.PHASE331_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is False
        assert "rows" in _all_failures(verdict).lower()

    def test_an_empty_diff_is_refused(self) -> None:
        f = _p331_module()
        verdict = attribute_rung(
            _phase331_report(added=(), width_delta=0, changed={}),
            f.PHASE331_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is False

    @pytest.mark.parametrize(
        "column",
        [
            "home_elo",
            "elo_diff",
            "home_off_rolling_opp_adj_epa_per_play",
            "away_def_rolling_success_rate",
            "ats_edge",
        ],
    )
    def test_an_elo_or_bye_window_or_ats_edge_column_is_unattributable_at_this_rung(
        self, column: str
    ) -> None:
        """The SPEC prohibition, enforced rather than stated.

        "MUST NOT label a gold rebuild 'the weather rung' if it also carries Elo,
        bye-window or ats_edge changes." A moved column from any of those three is
        reported UNATTRIBUTED here and the message names the prohibition, so the
        rung refuses to absorb another phase's cause instead of disclosing it in
        prose afterwards.
        """
        f = _p331_module()
        changed = _phase331_clean_changed()
        changed[column] = ["2019"]
        verdict = attribute_rung(
            _phase331_report(changed=changed),
            f.PHASE331_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is False
        for matrix in GOLD_MATRICES:
            assert column in verdict["matrices"][matrix]["unattributed"]
        failures = _all_failures(verdict)
        assert column in failures
        assert "rung" in failures
        assert "weather rung" in failures, (
            "the failure must name the SPEC prohibition about mislabelling a "
            "compound rebuild, not merely report an unexplained column"
        )

    def test_the_row_scoped_predicate_DISCRIMINATES_rather_than_accepting(self) -> None:
        """Both directions, so the third family is proven able to FAIL.

        ``_attribute_rung2`` is this repository's worked example of the opposite: a
        blanket predicate whose own comment records that it "cannot FAIL on a moved
        column", and that once reported "ok, zero unattributed" while eighteen
        columns had been silently destroyed.
        """
        f = _p331_module()

        only_2025 = _phase331_clean_changed()
        only_2025["home_snap_continuity"] = ["2025"]
        accepted = attribute_rung(
            _phase331_report(changed=only_2025),
            f.PHASE331_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )
        assert accepted["ok"] is True, _all_failures(accepted)
        for matrix in GOLD_MATRICES:
            assert (
                "home_snap_continuity"
                in (accepted["matrices"][matrix]["changed_by_family"]["staleness_2025"])
            )

        also_2019 = _phase331_clean_changed()
        also_2019["home_snap_continuity"] = ["2019", "2025"]
        refused = attribute_rung(
            _phase331_report(changed=also_2019),
            f.PHASE331_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )
        assert refused["ok"] is False, (
            "a column that moved in 2019 cannot have moved because 207 rows of 2025 "
            "gained coverage, and a predicate that accepts it is not a predicate"
        )
        for matrix in GOLD_MATRICES:
            assert "home_snap_continuity" in refused["matrices"][matrix]["unattributed"]

    def test_the_signature_predicts_unchanged_rows_and_a_width_of_plus_one(
        self,
    ) -> None:
        f = _p331_module()
        signature = f._expected_signature(
            f.PHASE331_RUNG, prefix=f.PHASE331_RUNG_PREFIX
        )

        assert signature["rows"] == "unchanged"
        assert "one" in str(signature["width"]).lower()
        assert signature["cause"] == f.PHASE331_RUNG_CAUSE

    def test_the_phase_30_signatures_are_unchanged(self) -> None:
        f = _p331_module()
        assert f._expected_signature(4)["rows"] == "strictly increased"
        assert f._expected_signature(1)["cause"] == "CR-02"
        assert f._expected_signature(1)["width"] == "unchanged"

    def test_an_unknown_rung_under_the_prefix_raises_naming_both(self) -> None:
        f = _p331_module()
        with pytest.raises(ValueError) as excinfo:
            f._expected_signature(99, prefix=f.PHASE331_RUNG_PREFIX)

        message = str(excinfo.value)
        assert "99" in message
        assert f.PHASE331_RUNG_PREFIX in message

    def test_the_ladder_demands_exactly_p331_rung0(self, tmp_path: Path) -> None:
        f = _p331_module()
        with pytest.raises(MissingPredecessorFingerprintError) as excinfo:
            attribute_rung(
                _phase331_report(changed=_phase331_clean_changed()),
                f.PHASE331_RUNG,
                ladder_directory=tmp_path,
                rung_prefix=f.PHASE331_RUNG_PREFIX,
            )

        assert "p331_rung0.json" in str(excinfo.value)

        rung_document_path(tmp_path, 0, f.PHASE331_RUNG_PREFIX).write_text(
            "{}", encoding="utf-8"
        )
        verified = require_rung_ladder(
            tmp_path, f.PHASE331_RUNG, f.PHASE331_RUNG_PREFIX
        )
        assert [path.name for path in verified] == ["p331_rung0.json"]

    def test_no_p331_document_collides_with_phase_30_or_phase_31(self) -> None:
        f = _p331_module()
        directory = Path("outputs/fingerprints")
        for rung in range(5):
            prefixed = rung_document_path(directory, rung, f.PHASE331_RUNG_PREFIX)
            assert prefixed.name not in PHASE30_RUNG_DOCUMENTS
            assert prefixed != rung_document_path(directory, rung)
            assert prefixed != rung_document_path(directory, rung, PHASE31_RUNG_PREFIX)

    def test_the_rung_offers_no_upstream_drift_escape(self) -> None:
        """The rebuild reads SILVER, not nflreadpy, so that excuse would not be true."""
        f = _p331_module()
        changed = _phase331_clean_changed()
        changed["home_elo"] = ["2019"]
        verdict = attribute_rung(
            _phase331_report(changed=changed),
            f.PHASE331_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert "nflreadpy" not in _all_failures(verdict).lower()

    def test_the_unprefixed_rung_1_still_offers_the_upstream_escape(self) -> None:
        """The suppression above is scoped to this PREFIX, not to the integer."""
        report = _pre_drop_report(changed={"home_rest_days": ["2021"]})
        assert "nflreadpy" in _all_failures(attribute_rung(report, 1)).lower()

    def test_the_venue_family_is_the_contextual_builders_own_emitted_set(self) -> None:
        """Computed HERE from the builder, never compared against a second list.

        ``tests/unit/test_data_qa_gold_width.py``'s docstring already records what a
        second hand-written list of a family costs: the list and the predicate that
        actually did the work drift apart, and the assertion resting on them pins a
        fiction.
        """
        from datetime import UTC, datetime

        from features.contextual import ContextualFeaturesCalculator

        f = _p331_module()
        calculator = ContextualFeaturesCalculator()
        record = next(
            venue
            for venue in calculator.venues_data["venues"]
            if venue.get("stadium_id") and venue.get("home_teams")
        )
        team = record["home_teams"][0]
        probe = pd.DataFrame(
            [
                {
                    "game_id": "2024_W01_PROBE",
                    "season": 2024,
                    "week": 1,
                    "home_team": team,
                    "away_team": team,
                    "kickoff_et": pd.Timestamp(
                        "2024-09-08 13:00:00", tz="America/New_York"
                    ),
                    "stadium_id": record["stadium_id"],
                    "home_score": 0.0,
                    "away_score": 0.0,
                }
            ]
        )
        emitted = calculator.build_features(probe, datetime(2024, 9, 9, tzinfo=UTC))
        expected = {
            column
            for column in emitted.columns
            if column not in f.PHASE331_CONTEXTUAL_MERGE_KEYS
        }

        assert set(f.PHASE331_VENUE_FAMILY_COLUMNS) == expected
        assert f.PHASE331_VENUE_FAMILY_COLUMNS
        assert "venue_cold_climate" in f.PHASE331_VENUE_FAMILY_COLUMNS

    def test_the_staleness_seasons_are_2025_alone(self) -> None:
        f = _p331_module()
        assert f.PHASE331_STALENESS_SEASONS == (2025,)

    def test_the_declared_families_are_disjoint_from_the_prohibited_ones(self) -> None:
        """Otherwise a prohibited column could be absorbed by a declared family."""
        f = _p331_module()
        declared = {c.lower() for c in f.PHASE331_VENUE_FAMILY_COLUMNS} | {
            c.lower() for c in f.phase331_weather_family()
        }
        for label, columns in f.phase331_prohibited_families().items():
            overlap = declared & {c.lower() for c in columns}
            assert not overlap, (
                f"the prohibited {label!r} family overlaps a DECLARED family at "
                f"{sorted(overlap)}, so a column this rung must refuse could be "
                "attributed instead"
            )

    def test_assert_ladder_is_recoverable_refuses_an_absent_document(
        self, tmp_path: Path
    ) -> None:
        f = _p331_module()
        with pytest.raises(MissingPredecessorFingerprintError) as excinfo:
            f.assert_ladder_is_recoverable(
                tmp_path, f.PHASE331_RUNG, f.PHASE331_RUNG_PREFIX
            )

        message = str(excinfo.value)
        assert "p331_rung0.json" in message
        assert "--rung 0 --rung-prefix p331_" in message

    def test_assert_ladder_is_recoverable_refuses_an_UNPARSEABLE_document(
        self, tmp_path: Path
    ) -> None:
        """A truncated document passes ``exists()`` and fails at the unrecoverable moment."""
        f = _p331_module()
        rung_document_path(tmp_path, 0, f.PHASE331_RUNG_PREFIX).write_text(
            '{"features_wp": {"columns": ', encoding="utf-8"
        )

        with pytest.raises(MissingPredecessorFingerprintError) as excinfo:
            f.assert_ladder_is_recoverable(
                tmp_path, f.PHASE331_RUNG, f.PHASE331_RUNG_PREFIX
            )

        message = str(excinfo.value)
        assert "--rung 0 --rung-prefix p331_" in message
        assert "parse" in message.lower()

    def test_assert_ladder_is_recoverable_passes_on_a_complete_ladder(
        self, tmp_path: Path
    ) -> None:
        f = _p331_module()
        rung_document_path(tmp_path, 0, f.PHASE331_RUNG_PREFIX).write_text(
            "{}", encoding="utf-8"
        )
        assert (
            f.assert_ladder_is_recoverable(
                tmp_path, f.PHASE331_RUNG, f.PHASE331_RUNG_PREFIX
            )
            is None
        )

    def test_the_declaration_was_committed_with_all_three_family_mechanisms(
        self,
    ) -> None:
        """No cause-story family: every mechanism is a constant or a predicate."""
        state = _p331_state()
        declaration = state.PHASE331_RUNG_DECLARATION

        assert set(declaration["declared_families"]) == {
            "weather",
            "venue",
            "staleness_2025",
        }
        assert set(declaration["family_mechanisms"].values()) <= {
            "source-derived constant",
            "row-scoped per-season-digest predicate",
        }
        assert declaration["committed_before_rebuild"] is True
        assert declaration["ok_required_unconditionally"] is True
        assert declaration["rows"] == "unchanged"
        assert declaration["columns_added"] == ("weather_coverage",)
        assert declaration["rung_prefix"] == "p331_"

    def test_the_declaration_and_the_signature_agree_about_the_families(self) -> None:
        f = _p331_module()
        state = _p331_state()
        assert tuple(state.PHASE331_RUNG_DECLARATION["declared_families"]) == tuple(
            f.PHASE331_EXPECTED_SIGNATURE["declared_families"]
        )
        assert state.PHASE331_RUNG_DECLARATION["cause"] == f.PHASE331_RUNG_CAUSE


# ---------------------------------------------------------------------------
# THE PHASE-33.1 FOLLOW-UP RUNG -- the three residual groups, declared on
# evidence (Plan 33.1-07, Ruling N2's second legitimate move)
# ---------------------------------------------------------------------------
#
# TEST CLASS (this module's phase-wide rule): ``TestPhase331FollowupRung`` is a
# plain UNIT test -- hand-built reports plus source-derived family checks, so it
# passes on a fresh checkout with no ``data/``.
#
# WHY A FOLLOW-UP RUNG AND NOT A WIDER RUNG 1. Rung 1 returned ``ok=False`` with
# 45 unattributed columns. Ruling N2 permits exactly two responses -- STOP, or
# declare a NEW family in a FOLLOW-UP RUNG -- and forbids the third, a footnote on
# rung 1. The residual was DIAGNOSED first (eight controlled rebuilds,
# ``33.1-07-GROUP1-DIAGNOSIS.md``), and the tests below pin the property that
# makes the follow-up honest rather than cosmetic: rung 1's declaration is
# UNCHANGED and STILL REFUSES this diff.


def _phase331_followup_changed() -> dict[str, list[str]]:
    """A changed set carrying one member of each family the follow-up rung knows.

    The first three are rung 1's, recorded by the follow-up as
    ``carried_at_rung_1``; the last three are the residual groups this rung
    declares, each with the seasons its declaration restricts it to.
    """
    return {
        **_phase331_clean_changed(),
        # Group 1 -- the stale-baseline carry-forward. 2024, reaching 2025 only
        # through season 2025's normalisation bootstrap on season 2024.
        "home_off_rolling_cpoe": ["2024", "2025"],
        # Group 2 -- the mislabelling prohibition firing correctly: 2025 only.
        "home_elo": ["2025"],
        # Group 3 -- the weather-family widening. Every season, which is what
        # replacing a fabricated constant with real observations does.
        "raw_weather_severity": [str(season) for season in range(2002, 2026)],
    }


class TestPhase331FollowupRung:
    """The residual is declared in a SECOND rung, and rung 1 still refuses it.

    TEST CLASS: plain unit test. Hand-built ``compare_fingerprints``-shaped
    reports, so the contract is provable without running a rebuild.
    """

    def test_the_followup_cause_names_all_three_residual_groups(self) -> None:
        f = _p331_module()
        cause = f.PHASE331_FOLLOWUP_RUNG_CAUSE.lower()
        for phrase in ("stale-baseline", "prohibition", "widening"):
            assert phrase in cause, (
                f"the follow-up rung's declared cause does not name {phrase!r}: "
                f"{f.PHASE331_FOLLOWUP_RUNG_CAUSE!r}. It carries THREE residual "
                "groups and a label naming fewer of them hides one inside another."
            )

    def test_the_cause_records_the_trigger_as_not_established(self) -> None:
        """The unknown must be written down, not smoothed over."""
        f = _p331_module()
        signature = f._expected_signature(
            f.PHASE331_FOLLOWUP_RUNG, prefix=f.PHASE331_RUNG_PREFIX
        )
        assert signature["group1_trigger"] == "NOT ESTABLISHED"
        assert signature["group1_magnitude"] == "PERMANENTLY UNMEASURABLE"
        assert "NOT ESTABLISHED" in f.PHASE331_FOLLOWUP_RUNG_CAUSE
        assert "PERMANENTLY" in f.PHASE331_FOLLOWUP_RUNG_CAUSE

    def test_the_full_residual_attributes_cleanly_at_the_followup_rung(self) -> None:
        f = _p331_module()
        verdict = attribute_rung(
            _phase331_report(changed=_phase331_followup_changed()),
            f.PHASE331_FOLLOWUP_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is True, _all_failures(verdict)
        for matrix in GOLD_MATRICES:
            families = verdict["matrices"][matrix]["changed_by_family"]
            assert verdict["matrices"][matrix]["unattributed"] == []
            assert families["stale_baseline_2024"] == ["home_off_rolling_cpoe"]
            assert families["prohibited_family_2025"] == ["home_elo"]
            assert families["weather_widening"] == ["raw_weather_severity"]
            assert sorted(families["carried_at_rung_1"]) == [
                "home_qb_adjustment",
                "temp_f",
                "venue_cold_climate",
            ]

    def test_rung_1_STILL_REFUSES_the_same_diff(self) -> None:
        """The load-bearing property: the follow-up did NOT widen rung 1.

        A follow-up rung that silently made rung 1 pass would be the retroactive
        edit Ruling N2 forbids, wearing a rung's clothes. Rung 1's verdict on this
        diff is what it always was.
        """
        f = _p331_module()
        verdict = attribute_rung(
            _phase331_report(changed=_phase331_followup_changed()),
            f.PHASE331_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is False
        assert tuple(f.PHASE331_EXPECTED_SIGNATURE["declared_families"]) == (
            "weather",
            "venue",
            "staleness_2025",
        )

    def test_a_group_1_column_that_moved_in_2019_is_still_UNATTRIBUTED(self) -> None:
        """The season restriction is what makes an enumerated family discriminate."""
        f = _p331_module()
        changed = _phase331_followup_changed()
        changed["home_off_rolling_cpoe"] = ["2019", "2024"]
        verdict = attribute_rung(
            _phase331_report(changed=changed),
            f.PHASE331_FOLLOWUP_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is False
        failures = _all_failures(verdict)
        assert "home_off_rolling_cpoe" in failures
        assert "2019" in failures

    def test_a_group_2_column_that_moved_outside_2025_is_still_UNATTRIBUTED(
        self,
    ) -> None:
        f = _p331_module()
        changed = _phase331_followup_changed()
        changed["home_elo"] = ["2019", "2025"]
        verdict = attribute_rung(
            _phase331_report(changed=changed),
            f.PHASE331_FOLLOWUP_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is False
        assert "home_elo" in _all_failures(verdict)

    def test_an_UNDECLARED_prohibited_column_is_still_refused(self) -> None:
        """Declaring 44 names does not open the prohibited families."""
        f = _p331_module()
        changed = _phase331_followup_changed()
        changed["elo_diff"] = ["2024"]
        verdict = attribute_rung(
            _phase331_report(changed=changed),
            f.PHASE331_FOLLOWUP_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is False
        failures = _all_failures(verdict)
        assert "elo_diff" in failures
        assert "PROHIBITED" in failures

    def test_an_undeclared_out_of_family_column_is_still_refused(self) -> None:
        f = _p331_module()
        changed = _phase331_followup_changed()
        changed["home_snap_continuity"] = ["2019"]
        verdict = attribute_rung(
            _phase331_report(changed=changed),
            f.PHASE331_FOLLOWUP_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is False
        assert "home_snap_continuity" in _all_failures(verdict)

    def test_the_negative_control_rung_2_with_no_prefix_is_still_WR_06(self) -> None:
        """The follow-up entry must not have captured Phase 30's integer 2."""
        assert _expected_signature(2)["cause"] == RUNG_CAUSES[2] == "WR-06"

    def test_the_three_declared_groups_are_disjoint_and_exactly_45_columns(
        self,
    ) -> None:
        f = _p331_module()
        group1 = set(f.PHASE331_FOLLOWUP_STALE_BASELINE_COLUMNS)
        group2 = set(f.PHASE331_FOLLOWUP_PROHIBITED_2025_COLUMNS)
        group3 = set(f.PHASE331_FOLLOWUP_WEATHER_WIDENING)

        assert len(group1) == 40
        assert len(group2) == 4
        assert len(group3) == 1
        assert group1 & group2 == set()
        assert group1 & group3 == set()
        assert group2 & group3 == set()
        assert len(group1 | group2 | group3) == 45

    def test_groups_1_and_2_are_EXACTLY_what_rung_1_refused_on_the_prohibition(
        self,
    ) -> None:
        """Every enumerated name belongs to a prohibited family -- nothing else.

        This is what bounds the follow-up: it declares the columns rung 1 refused
        on the prohibited-family check and no others. A name that was NOT refused
        there would be a column smuggled into a declaration built for a different
        refusal.
        """
        f = _p331_module()
        declared = (
            *f.PHASE331_FOLLOWUP_STALE_BASELINE_COLUMNS,
            *f.PHASE331_FOLLOWUP_PROHIBITED_2025_COLUMNS,
        )
        for column in declared:
            assert f._phase331_prohibited_family(column) is not None, (
                f"{column!r} is declared by the follow-up rung but is NOT in any "
                "prohibited family, so rung 1 did not refuse it on the "
                "prohibition. The follow-up declares that refusal's residual and "
                "nothing else."
            )

    def test_the_widening_family_is_SOURCE_DERIVED_not_a_second_hand_written_list(
        self,
    ) -> None:
        """Each widening entry names the registered weather column it copies.

        And the widening column itself is NOT in that registry -- which is
        precisely why rung 1's family 1 could never reach it.
        """
        f = _p331_module()
        weather = set(f.phase331_weather_family())
        for column, source in f.PHASE331_FOLLOWUP_WEATHER_WIDENING.items():
            assert source in weather, (
                f"{column!r} is declared as an un-normalized copy of {source!r}, "
                f"but {source!r} is not in WEATHER_FEATURE_COLUMNS"
            )
            assert column not in weather, (
                f"{column!r} IS in WEATHER_FEATURE_COLUMNS, so rung 1's family 1 "
                "already reached it and the widening family is unnecessary"
            )

    def test_a_widening_entry_whose_source_is_not_a_weather_column_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The source check is live, not decorative."""
        f = _p331_module()
        monkeypatch.setattr(
            f,
            "PHASE331_FOLLOWUP_WEATHER_WIDENING",
            {"raw_weather_severity": "home_rest_days"},
        )
        verdict = attribute_rung(
            _phase331_report(changed=_phase331_followup_changed()),
            f.PHASE331_FOLLOWUP_RUNG,
            rung_prefix=f.PHASE331_RUNG_PREFIX,
        )

        assert verdict["ok"] is False
        assert "home_rest_days" in _all_failures(verdict)

    def test_the_followup_rung_declares_that_it_rebuilt_nothing(self) -> None:
        f = _p331_module()
        signature = f._expected_signature(
            f.PHASE331_FOLLOWUP_RUNG, prefix=f.PHASE331_RUNG_PREFIX
        )
        assert signature["no_new_rebuild"] is True
        assert signature["ok_required_unconditionally"] is True

    def test_the_state_declaration_agrees_with_the_judge_it_describes(self) -> None:
        """No second-list drift between the record and the predicate.

        ``tests/unit/test_data_qa_gold_width.py``'s docstring already records what
        a second hand-written list of a family costs. The state manifest records
        COUNTS and the two small enumerations a reader needs in the document; both
        are checked against the judge rather than trusted.
        """
        f = _p331_module()
        declaration = _p331_state().PHASE331_FOLLOWUP_RUNG_DECLARATION

        assert declaration["rung"] == f.PHASE331_FOLLOWUP_RUNG
        assert declaration["rung_prefix"] == f.PHASE331_RUNG_PREFIX
        assert tuple(declaration["declared_families"]) == tuple(
            f.PHASE331_FOLLOWUP_EXPECTED_SIGNATURE["declared_families"]
        )
        assert declaration["group_1"]["columns"] == len(
            f.PHASE331_FOLLOWUP_STALE_BASELINE_COLUMNS
        )
        assert tuple(declaration["group_1"]["season_restriction"]) == tuple(
            f.PHASE331_FOLLOWUP_STALE_BASELINE_SEASONS
        )
        assert tuple(declaration["group_2"]["column_names"]) == tuple(
            f.PHASE331_FOLLOWUP_PROHIBITED_2025_COLUMNS
        )
        assert tuple(declaration["group_2"]["season_restriction"]) == tuple(
            f.PHASE331_FOLLOWUP_PROHIBITED_2025_SEASONS
        )
        assert tuple(declaration["group_3"]["column_names"]) == tuple(
            f.PHASE331_FOLLOWUP_WEATHER_WIDENING
        )
        assert (
            declaration["group_3"]["source_column"]
            == f.PHASE331_FOLLOWUP_WEATHER_WIDENING["raw_weather_severity"]
        )
        assert declaration["group_1"]["trigger"] == "NOT ESTABLISHED"
        assert declaration["group_1"]["magnitude"] == "PERMANENTLY UNMEASURABLE"
        assert declaration["no_new_rebuild"] is True
        assert declaration["fingerprint_document_written"] is None

    def test_the_followup_ladder_demands_rung_0_AND_rung_1(self, tmp_path) -> None:
        """A follow-up rung is still a rung: the chain is enforced, not assumed."""
        f = _p331_module()
        with pytest.raises(MissingPredecessorFingerprintError) as absent:
            require_rung_ladder(
                tmp_path, f.PHASE331_FOLLOWUP_RUNG, f.PHASE331_RUNG_PREFIX
            )
        assert "p331_rung0.json" in str(absent.value)

        rung_document_path(tmp_path, 0, f.PHASE331_RUNG_PREFIX).write_text(
            "{}", encoding="utf-8"
        )
        with pytest.raises(MissingPredecessorFingerprintError) as still:
            require_rung_ladder(
                tmp_path, f.PHASE331_FOLLOWUP_RUNG, f.PHASE331_RUNG_PREFIX
            )
        assert "p331_rung1.json" in str(still.value)

        rung_document_path(tmp_path, 1, f.PHASE331_RUNG_PREFIX).write_text(
            "{}", encoding="utf-8"
        )
        verified = require_rung_ladder(
            tmp_path, f.PHASE331_FOLLOWUP_RUNG, f.PHASE331_RUNG_PREFIX
        )
        assert [path.name for path in verified] == [
            "p331_rung0.json",
            "p331_rung1.json",
        ]


class TestPhase331TheDeliberateTripwiresInTheEditedModulesAreUnchanged:
    """The two registered tripwires living in modules this plan edits still exist.

    TEST CLASS: plain unit test (an AST scan over two committed source files).

    A plan that appends a class to a module holding a DELIBERATE tripwire can turn
    that tripwire green by accident -- renaming the class, renaming the method, or
    deleting it while tidying. Both node ids are asserted as STRINGS against the
    registry AND resolved against the real source, so neither half can drift alone.
    """

    _EDITED_MODULE_TRIPWIRES = (
        "tests/integration/test_gold_rebuild_attribution.py::"
        "TestThePhase31Rung3IsTheFullRebuildOfTheVerdictPopulation::"
        "test_no_NON_CLOCK_column_moved_in_a_protected_season",
        "tests/integration/test_gate_baseline_byte_identity.py::"
        "TestTheRegeneratedBaselineIsByteIdenticalToTheCommittedOne::"
        "test_the_generated_block_equals_the_committed_block_byte_for_byte",
    )

    def test_both_node_ids_are_still_registered_verbatim(self) -> None:
        state = _p331_state()
        for node_id in self._EDITED_MODULE_TRIPWIRES:
            assert node_id in state.DELIBERATE_TRIPWIRE_NODE_IDS, (
                f"{node_id} is no longer in DELIBERATE_TRIPWIRE_NODE_IDS. Each of "
                "the five encodes an owner-accepted fact; a phase that turned one "
                "green erased a disclosure rather than fixing a defect."
            )

    def test_both_node_ids_still_resolve_to_a_real_class_and_method(self) -> None:
        import ast as _ast

        for node_id in self._EDITED_MODULE_TRIPWIRES:
            relative, class_name, method_name = node_id.split("::")
            path = REPO_ROOT / relative
            assert path.is_file(), f"{relative} is missing from the checkout"
            tree = _ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            classes = {
                node.name: node for node in tree.body if isinstance(node, _ast.ClassDef)
            }
            assert class_name in classes, (
                f"{relative} no longer defines {class_name}; the registered tripwire "
                "node id can no longer be collected."
            )
            methods = {
                node.name
                for node in classes[class_name].body
                if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef))
            }
            assert method_name in methods, (
                f"{relative}::{class_name} no longer defines {method_name}."
            )


@pytest.mark.integration
class TestPhase331StalenessGameIdsWereMeasuredBeforeTheRebuild:
    """The 207 ids are re-derived off the live tree, never transcribed.

    TEST CLASS: integration / slow. Reads live gold and live silver, and SKIPS with
    a remediation-carrying message when either is absent -- ``data/`` is gitignored.

    THE MEASUREMENT IS ONE-SHOT. The ids are exactly the games present in gold and
    ABSENT from the two silver feature tables; the Plan 33.1-07 rebuild is what
    closes that gap, so afterwards the set is unmeasurable. That is the same
    argument ``p331_rung0.json`` rests on, and it is why the set is COMMITTED.
    """

    _WEATHER = REPO_ROOT / "data" / "silver" / "weather_features.parquet"
    _CONTEXTUAL = REPO_ROOT / "data" / "silver" / "contextual_features.parquet"

    def _gold_ids(self) -> set[str]:
        path = _GOLD_DIR / "features_ats.parquet"
        if not path.exists():
            pytest.skip(
                f"live gold is not present at {path} -- run "
                "`uv run python scripts/build_features.py --all-seasons`, or ignore "
                "on a fresh checkout where data/ is legitimately empty"
            )
        return set(pd.read_parquet(path, columns=["game_id"])["game_id"])

    def test_the_recorded_ids_are_non_empty_and_all_belong_to_2025(self) -> None:
        state = _p331_state()
        ids = state.PHASE331_STALENESS_GAME_IDS
        assert ids
        assert len(set(ids)) == len(ids), "the recorded id set carries duplicates"
        for game_id in ids:
            assert game_id.startswith("2025_"), (
                f"{game_id} is not a 2025 game, but PHASE331_STALENESS_SEASONS "
                "declares the staleness repair is 2025 alone"
            )

    def test_every_recorded_id_is_in_gold_and_absent_from_both_silver_tables(
        self,
    ) -> None:
        state = _p331_state()
        gold = self._gold_ids()
        if not (self._WEATHER.exists() and self._CONTEXTUAL.exists()):
            pytest.skip(
                "the silver feature tables are not present at "
                f"{self._WEATHER} / {self._CONTEXTUAL} -- outputs of "
                "`uv run python scripts/build_weather.py --all-seasons` and "
                "`uv run python scripts/build_contextual.py --all-seasons`"
            )
        weather = set(pd.read_parquet(self._WEATHER, columns=["game_id"])["game_id"])
        contextual = set(
            pd.read_parquet(self._CONTEXTUAL, columns=["game_id"])["game_id"]
        )
        recorded = set(state.PHASE331_STALENESS_GAME_IDS)

        if not (recorded - weather) and not (recorded - contextual):
            pytest.skip(
                "the silver feature tables already carry every recorded id, so the "
                "staleness gap this set measures has been CLOSED by the Plan 33.1-07 "
                "rebuild. The set was measured before that rebuild ran and cannot be "
                "re-derived afterwards -- which is exactly why it is committed."
            )

        assert recorded <= gold, sorted(recorded - gold)[:5]
        assert recorded == (gold - weather), (
            "the recorded ids are not exactly the games gold has and "
            "weather_features.parquet lacks"
        )
        assert recorded == (gold - contextual), (
            "the recorded ids are not exactly the games gold has and "
            "contextual_features.parquet lacks"
        )

    def test_the_recorded_count_is_stated_against_the_research_figure(self) -> None:
        """RESEARCH section 10 says 207. Re-derived here, not transcribed."""
        state = _p331_state()
        declaration = state.PHASE331_RUNG_DECLARATION
        assert declaration["staleness_game_id_count"] == len(
            state.PHASE331_STALENESS_GAME_IDS
        )
        assert declaration["staleness_game_id_count_recorded_by_research"] == 207
