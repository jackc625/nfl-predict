"""Per-season storage metadata and build-clock classification in the gold fingerprint.

TEST CLASS: **plain unit test**. Every frame below is synthesized in memory and every
fingerprint document is either produced from one of those frames or hand-edited from
one. The module touches nothing under ``data/``, so it passes on a fresh checkout.

WHY THIS MODULE EXISTS (Plan 31-03, D31-10, T-31-10). ``compare_fingerprints`` reported a
whole-column dtype or null-count move as changed with an EMPTY season list. Plan 31-11's
hard stop measures a STRICT 2021-2024 slice, so a storage-level move carrying no season is
exactly the case that either trips the tripwire spuriously or slips past it. The fix is
attribution quality, not a new tolerance: a 2025-only null-count change must never read
like a 2021-2024 value move, and a 2021-2024 storage move must never read like nothing.

WHY THE STORAGE-ONLY CASES ARE BUILT AT THE DOCUMENT LEVEL AND NOT FROM TWO FRAMES.
``_column_meta``'s own docstring records the reason: ``_column_bytes`` already makes a
dtype change or a null-count change move the per-season hash, so a real pair of frames
CANNOT exhibit a storage move that the value hash does not also carry -- and a single
``DataFrame`` cannot hold two dtypes for one column, so a one-season dtype difference has
no frame-level representation at all. The storage-only case is a fact about two DOCUMENTS,
which is the level ``compare_fingerprints`` actually operates on, so that is the level the
synthetic pair is constructed at. Where a real frame pair CAN express the case (a
whole-column dtype move, a value move, a build-clock move) it is used instead.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.fingerprint_gold import (
    BUILD_CLOCK_COLUMNS,
    compare_fingerprints,
    fingerprint_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_FINGERPRINT_SOURCE = REPO_ROOT / "scripts" / "fingerprint_gold.py"
_PHASE30_FINGERPRINTS = REPO_ROOT / "outputs" / "fingerprints"

_MATRIX = "features_wp"


# ---------------------------------------------------------------------------
# Synthetic gold-shaped frames
# ---------------------------------------------------------------------------


def _frame(
    *,
    rest_days_2025: list[float] | None = None,
    saturday_dtype: str = "float64",
    clock: str = "2026-01-01T00:00:00Z",
    audit_stamp: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    """Return a two-season gold-shaped frame.

    ``feature_timestamp`` is the REGISTERED per-build clock. ``audit_timestamp`` is a
    column that merely LOOKS like one and is deliberately NOT registered, so the
    classification can be shown to read the registered set rather than the name.
    """
    rest_days_2025 = [7.0, 6.0] if rest_days_2025 is None else rest_days_2025
    return pd.DataFrame(
        {
            "game_id": [
                "2024_W01_A@B",
                "2024_W02_C@D",
                "2025_W01_E@F",
                "2025_W02_G@H",
            ],
            "season": [2024, 2024, 2025, 2025],
            "home_rest_days": [7.0, 10.0, *rest_days_2025],
            "saturday_game": pd.Series([0, 1, 0, 0], dtype=saturday_dtype),
            "feature_timestamp": [clock] * 4,
            "audit_timestamp": [audit_stamp] * 4,
        }
    )


def _document(frame: pd.DataFrame) -> dict:
    """Return a one-matrix fingerprint document over *frame*."""
    return {_MATRIX: fingerprint_matrix(frame)}


def _detail(before: dict, after: dict) -> dict:
    """Return the single matrix's comparison detail for a document pair."""
    return compare_fingerprints(before, after)[_MATRIX]


# ---------------------------------------------------------------------------
# The additive per-season map
# ---------------------------------------------------------------------------


class TestColumnMetaBySeasonIsAdditive:
    """``column_meta_by_season`` is a SIBLING map; every pre-existing key keeps its meaning."""

    def test_every_pre_existing_key_survives_unchanged(self):
        result = fingerprint_matrix(_frame())

        for key in (
            "rows",
            "width",
            "seasons",
            "rows_per_season",
            "columns",
            "column_meta",
        ):
            assert key in result, (
                f"the additive growth dropped the pre-existing key {key!r}"
            )
        assert result["rows"] == 4
        assert result["seasons"] == [2024, 2025]
        assert set(result["columns"]["home_rest_days"]) == {"2024", "2025"}

    def test_the_per_season_map_carries_dtype_and_null_count_per_season(self):
        result = fingerprint_matrix(_frame(rest_days_2025=[7.0, float("nan")]))

        by_season = result["column_meta_by_season"]["home_rest_days"]
        assert set(by_season) == {"2024", "2025"}
        assert by_season["2024"] == {"dtype": "float64", "null_count": 0}
        assert by_season["2025"] == {"dtype": "float64", "null_count": 1}

    def test_discreteness_stays_a_whole_column_fact(self):
        """Per-season discreteness would MISCLASSIFY, so it is not in the per-season map.

        ``FeatureMatrixBuilder._is_discrete_indicator`` is True for a column whose every
        non-null value is -1.0 / 0.0 / 1.0, so a continuous column that happens to be
        constant within ONE season would acquire the flag for that season alone. The
        module's own ``_column_meta`` docstring records that reasoning; this pins it.
        """
        result = fingerprint_matrix(_frame())

        assert "discrete_indicator" in result["column_meta"]["saturday_game"]
        for column, by_season in result["column_meta_by_season"].items():
            for season, meta in by_season.items():
                assert set(meta) == {"dtype", "null_count"}, (
                    f"{column}/{season} carries {sorted(meta)}; discreteness is a "
                    "whole-column fact and must not be recorded per season"
                )

    def test_the_per_season_map_covers_every_column_and_every_season(self):
        result = fingerprint_matrix(_frame())

        assert set(result["column_meta_by_season"]) == set(result["column_meta"])
        for by_season in result["column_meta_by_season"].values():
            assert set(by_season) == {"2024", "2025"}


# ---------------------------------------------------------------------------
# Storage moves are attributed to seasons and carry a move kind
# ---------------------------------------------------------------------------


class TestStorageMovesAreAttributedToASeason:
    """A dtype or null-count move names the seasons it moved in, and is kind ``storage``."""

    def test_a_one_season_dtype_move_names_exactly_that_season(self):
        before = _document(_frame())
        after = copy.deepcopy(before)
        after[_MATRIX]["column_meta_by_season"]["saturday_game"]["2025"]["dtype"] = (
            "int64"
        )

        detail = _detail(before, after)

        assert detail["columns_changed"]["saturday_game"] == ["2025"]
        assert detail["column_details"]["saturday_game"]["move_kind"] == "storage"
        assert detail["column_details"]["saturday_game"]["reasons"] == ["dtype"]

    def test_a_one_season_null_count_move_names_exactly_that_season(self):
        before = _document(_frame())
        after = copy.deepcopy(before)
        after[_MATRIX]["column_meta_by_season"]["home_rest_days"]["2025"][
            "null_count"
        ] = 1

        detail = _detail(before, after)

        assert detail["columns_changed"]["home_rest_days"] == ["2025"]
        assert detail["column_details"]["home_rest_days"]["move_kind"] == "storage"
        assert detail["column_details"]["home_rest_days"]["reasons"] == ["null_count"]

    def test_a_storage_move_is_never_reported_with_an_empty_season_list(self):
        """The reporting case Plan 31-11's hard stop cannot judge."""
        before = _document(_frame())
        after = copy.deepcopy(before)
        after[_MATRIX]["column_meta_by_season"]["saturday_game"]["2024"]["dtype"] = (
            "int64"
        )
        after[_MATRIX]["column_meta"]["saturday_game"]["dtype"] = "int64"

        detail = _detail(before, after)

        assert detail["columns_changed"]["saturday_game"] == ["2024"]

    def test_a_whole_column_dtype_move_between_real_frames_names_every_season(self):
        before = _document(_frame(saturday_dtype="float64"))
        after = _document(_frame(saturday_dtype="int64"))

        detail = _detail(before, after)

        assert detail["columns_changed"]["saturday_game"] == ["2024", "2025"]
        assert detail["column_details"]["saturday_game"]["move_kind"] == "storage"
        assert detail["column_details"]["saturday_game"]["dtype_before"] == "float64"
        assert detail["column_details"]["saturday_game"]["dtype_after"] == "int64"

    def test_a_value_move_is_still_reported_as_a_values_move(self):
        before = _document(_frame())
        after = _document(_frame(rest_days_2025=[7.0, 3.0]))

        detail = _detail(before, after)

        assert detail["columns_changed"]["home_rest_days"] == ["2025"]
        assert detail["column_details"]["home_rest_days"]["move_kind"] == "values"
        assert detail["column_details"]["home_rest_days"]["reasons"] == ["values"]

    def test_a_column_that_moved_in_both_ways_is_a_values_move(self):
        """Values dominate: a moved value is the stronger claim and must not be softened."""
        before = _document(_frame())
        after = _document(_frame(rest_days_2025=[7.0, float("nan")]))

        detail = _detail(before, after)

        details = detail["column_details"]["home_rest_days"]
        assert details["move_kind"] == "values"
        assert "values" in details["reasons"]
        assert "null_count" in details["reasons"]
        assert detail["columns_changed"]["home_rest_days"] == ["2025"]

    def test_an_unchanged_pair_reports_nothing_moved(self):
        before = _document(_frame())

        detail = _detail(before, copy.deepcopy(before))

        assert detail["columns_changed"] == {}
        assert detail["non_clock_moves"] == []
        assert detail["build_clock_moves"] == []


# ---------------------------------------------------------------------------
# Backward compatibility with documents written before Plan 31-03
# ---------------------------------------------------------------------------


class TestPrePhase31DocumentsStayComparable:
    """A document written before this change compares without raising."""

    def test_a_document_with_no_per_season_map_compares_without_raising(self):
        before = _document(_frame())
        del before[_MATRIX]["column_meta_by_season"]
        after = _document(_frame(rest_days_2025=[7.0, 3.0]))

        detail = _detail(before, after)

        assert detail["columns_changed"]["home_rest_days"] == ["2025"]

    def test_the_key_is_ABSENT_rather_than_empty_on_an_old_document(self):
        before = _document(_frame())
        del before[_MATRIX]["column_meta_by_season"]

        assert "column_meta_by_season" not in before[_MATRIX], (
            "an EMPTY per-season map would be indistinguishable from 'every season "
            "moved to nothing'; the old document simply does not carry the key"
        )
        assert _detail(before, copy.deepcopy(before))["columns_changed"] == {}

    def test_a_whole_column_move_on_an_old_document_keeps_its_old_reporting_shape(self):
        """Without a per-season map there is nothing to attribute to, and that is honest."""
        before = _document(_frame())
        after = copy.deepcopy(before)
        for document in (before, after):
            del document[_MATRIX]["column_meta_by_season"]
        after[_MATRIX]["column_meta"]["saturday_game"]["dtype"] = "int64"

        detail = _detail(before, after)

        assert detail["columns_changed"]["saturday_game"] == []
        assert detail["column_details"]["saturday_game"]["reasons"] == ["dtype"]
        assert detail["column_details"]["saturday_game"]["move_kind"] == "storage"

    @pytest.mark.parametrize("document", ["rung3.json", "rung4.json"])
    def test_the_committed_phase30_documents_still_load_and_compare(
        self, document: str
    ):
        path = _PHASE30_FINGERPRINTS / document
        if not path.exists():
            pytest.skip(
                f"{path} is absent -- outputs/ is gitignored, so a fresh checkout "
                "legitimately has no Phase-30 fingerprint documents"
            )

        loaded = json.loads(path.read_text(encoding="utf-8"))
        for matrix in loaded.values():
            if matrix.get("missing"):
                continue
            assert "column_meta_by_season" not in matrix, (
                "a Phase-30 document predates the per-season map; its absence is the "
                "backward-compatibility case, not an error"
            )

        report = compare_fingerprints(loaded, copy.deepcopy(loaded))
        for detail in report.values():
            assert detail["columns_changed"] == {}
            assert detail["non_clock_moves"] == []


# ---------------------------------------------------------------------------
# The build clock is a REGISTERED classification, never a suppression
# ---------------------------------------------------------------------------


class TestBuildClockClassification:
    """REVIEW-CLOCK: a zero-move condition must be expressible against a clock-stamping build."""

    _CLOCK = "feature_timestamp"

    def test_the_registered_set_is_exactly_the_feature_timestamp_column(self):
        assert BUILD_CLOCK_COLUMNS == ("feature_timestamp",)

    def test_the_constant_cites_the_two_facts_that_justify_it(self):
        """The comment must carry EVIDENCE, so a later reader can re-check the claim."""
        source = _FINGERPRINT_SOURCE.read_text(encoding="utf-8")
        head, _, _ = source.partition("BUILD_CLOCK_COLUMNS = ")
        comment = head.rsplit("\n\n", 1)[-1]

        assert "scripts/build_features.py:570" in comment, (
            "the comment must cite the line that stamps datetime.now(UTC) on every build"
        )
        assert "tests/phase30_state.py:768-774" in comment, (
            "the comment must cite the recorded observation that two full-history "
            "builds differ in exactly this column"
        )

    def test_a_clock_only_move_leaves_the_non_clock_list_empty(self):
        before = _document(_frame(clock="2026-01-01T00:00:00Z"))
        after = _document(_frame(clock="2026-02-02T00:00:00Z"))

        detail = _detail(before, after)

        assert detail["non_clock_moves"] == []
        assert detail["build_clock_moves"] == [self._CLOCK]
        assert detail["column_details"][self._CLOCK]["move_kind"] == "build_clock"

    def test_the_clock_classification_cannot_mask_a_real_move(self):
        before = _document(_frame(clock="2026-01-01T00:00:00Z"))
        after = _document(
            _frame(clock="2026-02-02T00:00:00Z", rest_days_2025=[7.0, 3.0])
        )

        detail = _detail(before, after)

        assert detail["non_clock_moves"] == ["home_rest_days"]
        assert detail["build_clock_moves"] == [self._CLOCK]

    def test_an_unregistered_timestamp_looking_column_is_a_NON_clock_move(self):
        """The classification reads the REGISTERED set, never a name heuristic."""
        assert "audit_timestamp" not in BUILD_CLOCK_COLUMNS
        before = _document(_frame(audit_stamp="2026-01-01T00:00:00Z"))
        after = _document(_frame(audit_stamp="2026-02-02T00:00:00Z"))

        detail = _detail(before, after)

        assert detail["non_clock_moves"] == ["audit_timestamp"]
        assert detail["build_clock_moves"] == []
        assert detail["column_details"]["audit_timestamp"]["move_kind"] == "values"

    def test_a_moved_clock_is_still_REPORTED_not_suppressed(self):
        before = _document(_frame(clock="2026-01-01T00:00:00Z"))
        after = _document(_frame(clock="2026-02-02T00:00:00Z"))

        detail = _detail(before, after)

        assert self._CLOCK in detail["columns_changed"], (
            "the clock set is a CLASSIFICATION, not a suppression -- a moved clock "
            "stays in the changed set and is merely attributed elsewhere"
        )
        assert detail["columns_changed"][self._CLOCK] == ["2024", "2025"]

    def test_the_two_lists_are_disjoint_and_their_union_is_the_moved_set(self):
        before = _document(_frame(clock="2026-01-01T00:00:00Z"))
        after = _document(
            _frame(
                clock="2026-02-02T00:00:00Z",
                audit_stamp="2026-02-02T00:00:00Z",
                rest_days_2025=[7.0, 3.0],
            )
        )

        detail = _detail(before, after)

        non_clock = set(detail["non_clock_moves"])
        clock = set(detail["build_clock_moves"])
        assert non_clock & clock == set()
        assert non_clock | clock == set(detail["columns_changed"])
        assert non_clock == {"home_rest_days", "audit_timestamp"}
        assert clock == {self._CLOCK}

    def test_a_storage_only_clock_move_is_still_classified_as_the_clock(self):
        before = _document(_frame())
        after = copy.deepcopy(before)
        after[_MATRIX]["column_meta_by_season"][self._CLOCK]["2025"]["null_count"] = 1

        detail = _detail(before, after)

        assert detail["build_clock_moves"] == [self._CLOCK]
        assert detail["non_clock_moves"] == []
        assert detail["columns_changed"][self._CLOCK] == ["2025"]
