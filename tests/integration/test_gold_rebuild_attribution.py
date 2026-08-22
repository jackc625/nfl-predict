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
    RUNG_CAUSES,
    _expected_signature,
    attribute_rung,
    compare_fingerprints,
    fingerprint_gold,
    fingerprint_matrix,
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
    rows_per_season_before: dict[str, int] | None = None,
    rows_per_season_after: dict[str, int] | None = None,
) -> dict:
    """Build one matrix's entry of a compare_fingerprints report."""
    changed = dict(changed or {})
    details = {
        column: {
            "seasons": list(seasons),
            "dtype_before": "float64",
            "dtype_after": "float64",
            "null_count_before": 0,
            "null_count_after": 0,
            "discrete_indicator_before": column in discrete,
            "discrete_indicator_after": column in discrete,
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
