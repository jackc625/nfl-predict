"""The realized-versus-expected tracker: strict partition, push handling, zero-row refusal.

Phase 31, plan 31-13 (SPEC R8, D31-21/22, PROD-03). ``backtest/bet_tracker.py`` is a PURE
AGGREGATION over the stored bet-list blob. It prices nothing, re-derives no EV and re-grades no
bet: every figure it reports is a count or a sum over columns another module already wrote. That
is what makes it correct under either resolution of the open per-target pricing question -- a
ruling there changes the values flowing IN, and changes nothing here.

Four things could each be got wrong in a way that a smoke test would not notice, so each has its
own section below.

1. **The partition is structural, not a display convention.** A backtest-replay row was
   reconstructed after the fact; a forward row was written to the cache before the result existed.
   Pooling them publishes reconstructions as a track record. There is no aggregate that spans two
   classes, and no default partition to fall into: the caller names its block.

2. **A push and an ungraded bet are told apart by ``grading_status``, never by a null outcome.**
   Both store ``outcome`` as SQL NULL (the retired ``materialize_ou_bet_list`` did exactly that),
   and this module must COUNT one and EXCLUDE the other. A null-outcome test cannot tell them
   apart, so no test here uses one.

3. **The flat return comes from the STORED prices.** ``sum(payout_flat) / sum(flat_stake)`` over
   the graded rows, mirroring ``backtest.ou_monetization._flat_roi_from_records``. Under
   asymmetric American prices a win at -110 and a win at +140 return different amounts, so an
   outcome-derived return would be wrong for every bet not priced at the reference juice. The
   -110/+140 test below is the one that fails if anyone re-derives it.

4. **A zero-graded block does not compute a rate.** Guarding a division is not the same as not
   performing it: the requirement asks for an empty state, and a block of zeros is a claim that
   the record was measured and came out zero.

Run this module:  .venv/Scripts/python.exe -m pytest tests/unit/test_bet_tracker.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from api.cache import (
    BET_TRACKER_BLOCK_COLUMNS,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PENDING,
    GRADING_STATUS_PUSH,
    GRADING_STATUS_WIN,
    PROVENANCE_BACKTEST_REPLAY,
    PROVENANCE_FORWARD,
    VALIDATION_TYPE_CLEAN_HOLDOUT,
    VALIDATION_TYPE_CONTAMINATED,
    VALIDATION_TYPE_FORWARD_REALIZED,
    materialize_bet_tracker_blocks,
)
from api.services import DataService, clear_cache
from backtest.bet_tracker import (
    TRACKER_BLOCK_FIGURES,
    TRACKER_REQUIRED_COLUMNS,
    EmptyTrackerBlock,
    TrackerBlock,
    aggregate_all_blocks,
    aggregate_by_provenance,
    to_tracker_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER_PATH = REPO_ROOT / "backtest" / "bet_tracker.py"

_REPLAY = (PROVENANCE_BACKTEST_REPLAY, VALIDATION_TYPE_CONTAMINATED)
_CLEAN = (PROVENANCE_BACKTEST_REPLAY, VALIDATION_TYPE_CLEAN_HOLDOUT)
_FORWARD = (PROVENANCE_FORWARD, VALIDATION_TYPE_FORWARD_REALIZED)

# The two payouts a UNIT stake returns at the two prices the asymmetry test uses. Written as
# literals rather than imported from the pricing helper, so this test would still catch a tracker
# that started pricing for itself and happened to agree with the helper.
_PAYOUT_AT_MINUS_110 = 100.0 / 110.0
_PAYOUT_AT_PLUS_140 = 1.40


def _row(
    provenance_pair: tuple[str, str],
    grading_status: str,
    *,
    flat_stake: float = 1.0,
    payout_flat: float | None = None,
    status: str = "live",
    **overrides: Any,
) -> dict[str, Any]:
    """One bet-list row carrying every column the tracker requires.

    ``outcome`` is set from ``grading_status`` the way the writer does -- SQL NULL for BOTH a push
    and a pending row -- precisely so that no test here can accidentally distinguish them by it.
    """
    provenance, validation_type = provenance_pair
    outcome: bool | None = {
        GRADING_STATUS_WIN: True,
        GRADING_STATUS_LOSS: False,
        GRADING_STATUS_PUSH: None,
        GRADING_STATUS_PENDING: None,
    }[grading_status]
    if payout_flat is None:
        payout_flat = {
            GRADING_STATUS_WIN: _PAYOUT_AT_MINUS_110 * flat_stake,
            GRADING_STATUS_LOSS: -flat_stake,
            GRADING_STATUS_PUSH: 0.0,
            GRADING_STATUS_PENDING: None,
        }[grading_status]
    row: dict[str, Any] = {
        "status": status,
        "provenance": provenance,
        "validation_type": validation_type,
        "grading_status": grading_status,
        "flat_stake": flat_stake,
        "payout_flat": payout_flat,
        "outcome": outcome,
    }
    row.update(overrides)
    return row


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. The partition is structural
# ---------------------------------------------------------------------------


class TestNoAggregateSpansTwoProvenanceClasses:
    """T-31-61: a reconstruction must never be pooled into a made-in-advance record."""

    @staticmethod
    def _mixed_frame() -> pd.DataFrame:
        """Two replay WINS and two forward LOSSES: pooled hit rate 0.5, per-block 1.0 and 0.0."""
        return _frame(
            [
                _row(_REPLAY, GRADING_STATUS_WIN),
                _row(_REPLAY, GRADING_STATUS_WIN),
                _row(_FORWARD, GRADING_STATUS_LOSS),
                _row(_FORWARD, GRADING_STATUS_LOSS),
            ]
        )

    def test_a_mixed_frame_yields_two_separate_blocks(self) -> None:
        blocks = aggregate_all_blocks(self._mixed_frame())
        assert len(blocks) == 2
        assert {(b.provenance, b.validation_type) for b in blocks} == {
            _REPLAY,
            _FORWARD,
        }

    def test_no_figure_equals_the_pooled_value(self) -> None:
        """The pooled hit rate is 0.5 and the pooled return is 0. Neither may appear."""
        blocks = aggregate_all_blocks(self._mixed_frame())
        by_class = {(b.provenance, b.validation_type): b for b in blocks}
        replay = by_class[_REPLAY]
        forward = by_class[_FORWARD]
        assert isinstance(replay, TrackerBlock)
        assert isinstance(forward, TrackerBlock)
        assert replay.hit_rate == 1.0
        assert forward.hit_rate == 0.0
        assert 0.5 not in {replay.hit_rate, forward.hit_rate}
        assert replay.bets_graded == 2
        assert forward.bets_graded == 2
        # The pooled 4-row block does not exist anywhere in the output.
        assert 4 not in {block.bets_graded for block in blocks}

    def test_the_clean_holdout_replay_rows_are_their_own_block(self) -> None:
        """A 2025 replay row is replay-PRODUCED and clean-holdout WEIGHT.

        Pooling it with the burned 2021-2024 replay rows would publish the one unspent split's
        result inside a contaminated figure, which is the reason D31-22 carries two columns
        instead of one. The partition is on the PAIR.
        """
        frame = _frame(
            [
                _row(_REPLAY, GRADING_STATUS_WIN),
                _row(_CLEAN, GRADING_STATUS_LOSS),
            ]
        )
        blocks = aggregate_all_blocks(frame)
        assert {(b.provenance, b.validation_type) for b in blocks} == {_REPLAY, _CLEAN}
        assert all(block.bets_graded == 1 for block in blocks)

    def test_the_caller_must_state_its_partition_explicitly(self) -> None:
        """Both partition arguments are keyword-only and have NO default to fall into."""
        signature = inspect.signature(aggregate_by_provenance)
        for name in ("provenance", "validation_type"):
            parameter = signature.parameters[name]
            assert parameter.default is inspect.Parameter.empty, (
                f"{name} has default {parameter.default!r}; a default partition is a pooled "
                "figure waiting to happen"
            )
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    def test_an_out_of_vocabulary_partition_raises(self) -> None:
        frame = _frame([_row(_REPLAY, GRADING_STATUS_WIN)])
        with pytest.raises(ValueError) as excinfo:
            aggregate_by_provenance(
                frame,
                provenance="simulated",
                validation_type=VALIDATION_TYPE_CONTAMINATED,
            )
        assert "simulated" in str(excinfo.value)

    def test_a_valid_but_absent_partition_returns_the_empty_marker(self) -> None:
        """Asking for the forward block before any forward week exists is not an error."""
        frame = _frame([_row(_REPLAY, GRADING_STATUS_WIN)])
        block = aggregate_by_provenance(
            frame,
            provenance=PROVENANCE_FORWARD,
            validation_type=VALIDATION_TYPE_FORWARD_REALIZED,
        )
        assert isinstance(block, EmptyTrackerBlock)

    def test_a_row_whose_labels_are_out_of_vocabulary_raises(self) -> None:
        frame = _frame(
            [_row(("made_up", VALIDATION_TYPE_CONTAMINATED), GRADING_STATUS_WIN)]
        )
        with pytest.raises(ValueError) as excinfo:
            aggregate_all_blocks(frame)
        assert "made_up" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2. Push versus pending, decided on grading_status
# ---------------------------------------------------------------------------


class TestAPushIsNeverCoercedAndNeverConfusedWithAnUngradedBet:
    """T-31-64/64b: both carry a NULL outcome, so the discriminator must be the status."""

    def test_a_push_is_counted_and_excluded_from_the_hit_rate_denominator(self) -> None:
        """Two wins, one loss, one push. Hit rate is 2/3, NOT 2/4 and NOT 3/4."""
        frame = _frame(
            [
                _row(_REPLAY, GRADING_STATUS_WIN),
                _row(_REPLAY, GRADING_STATUS_WIN),
                _row(_REPLAY, GRADING_STATUS_LOSS),
                _row(_REPLAY, GRADING_STATUS_PUSH),
            ]
        )
        block = aggregate_by_provenance(
            frame,
            provenance=PROVENANCE_BACKTEST_REPLAY,
            validation_type=VALIDATION_TYPE_CONTAMINATED,
        )
        assert isinstance(block, TrackerBlock)
        assert block.pushes == 1
        assert block.wins == 2
        assert block.losses == 1
        assert block.bets_graded == 4
        assert block.hit_rate == pytest.approx(2.0 / 3.0)
        assert block.hit_rate != pytest.approx(2.0 / 4.0)
        assert block.hit_rate != pytest.approx(3.0 / 4.0)

    def test_one_push_and_one_pending_row_are_told_apart_by_status_alone(self) -> None:
        """Both rows carry ``outcome`` SQL NULL and differ ONLY in ``grading_status``.

        The push is COUNTED (pushes 1, bets_graded 1); the pending row is EXCLUDED from every
        figure INCLUDING pushes. A null-outcome partition could not produce this result.
        """
        rows = [
            _row(_FORWARD, GRADING_STATUS_PUSH),
            _row(_FORWARD, GRADING_STATUS_PENDING),
        ]
        assert all(row["outcome"] is None for row in rows), (
            "the fixture must keep both outcomes NULL, or the test proves nothing"
        )
        block = aggregate_by_provenance(
            _frame(rows),
            provenance=PROVENANCE_FORWARD,
            validation_type=VALIDATION_TYPE_FORWARD_REALIZED,
        )
        assert isinstance(block, TrackerBlock)
        assert block.pushes == 1
        assert block.bets_graded == 1
        assert block.wins == 0
        assert block.losses == 0

    def test_a_pending_row_is_never_counted_as_a_loss(self) -> None:
        """One win and three pending. Losses is 0 and the hit rate is 1.0, not 0.25."""
        frame = _frame(
            [
                _row(_FORWARD, GRADING_STATUS_WIN),
                *[_row(_FORWARD, GRADING_STATUS_PENDING) for _ in range(3)],
            ]
        )
        block = aggregate_by_provenance(
            frame,
            provenance=PROVENANCE_FORWARD,
            validation_type=VALIDATION_TYPE_FORWARD_REALIZED,
        )
        assert isinstance(block, TrackerBlock)
        assert block.losses == 0
        assert block.bets_graded == 1
        assert block.hit_rate == 1.0

    def test_a_partially_graded_week_uses_only_its_graded_rows(self) -> None:
        """Ungraded rows appear in no denominator -- hit rate NOR flat return."""
        frame = _frame(
            [
                _row(_FORWARD, GRADING_STATUS_WIN, flat_stake=1.0),
                _row(_FORWARD, GRADING_STATUS_LOSS, flat_stake=1.0),
                _row(_FORWARD, GRADING_STATUS_PENDING, flat_stake=1.0),
                _row(_FORWARD, GRADING_STATUS_PENDING, flat_stake=1.0),
            ]
        )
        block = aggregate_by_provenance(
            frame,
            provenance=PROVENANCE_FORWARD,
            validation_type=VALIDATION_TYPE_FORWARD_REALIZED,
        )
        assert isinstance(block, TrackerBlock)
        assert block.bets_graded == 2
        assert block.hit_rate == 0.5
        # Denominator is the TWO graded stakes, not the four written stakes.
        assert block.flat_return_units == pytest.approx(
            (_PAYOUT_AT_MINUS_110 - 1.0) / 2.0
        )

    def test_a_suppressed_row_enters_no_figure(self) -> None:
        """A candidate that was never bet cannot be part of a betting record (D31-21)."""
        frame = _frame(
            [
                _row(_FORWARD, GRADING_STATUS_WIN),
                _row(_FORWARD, GRADING_STATUS_PENDING, status="suppressed"),
            ]
        )
        block = aggregate_by_provenance(
            frame,
            provenance=PROVENANCE_FORWARD,
            validation_type=VALIDATION_TYPE_FORWARD_REALIZED,
        )
        assert isinstance(block, TrackerBlock)
        assert block.bets_graded == 1

    def test_a_suppressed_row_carrying_a_terminal_grade_is_refused(self) -> None:
        """A bet that was never placed cannot have won. That is a contradiction, not a figure."""
        frame = _frame([_row(_FORWARD, GRADING_STATUS_WIN, status="suppressed")])
        with pytest.raises(ValueError) as excinfo:
            aggregate_all_blocks(frame)
        assert "suppressed" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. The flat return comes from the stored prices
# ---------------------------------------------------------------------------


class TestTheFlatReturnIsReadFromTheStoredPricesAndNeverDerivedFromTheOutcome:
    """T-31-64c: under asymmetric American prices an outcome-derived return is wrong."""

    def test_two_wins_at_different_prices_do_not_return_the_same_amount(self) -> None:
        """Identical stakes, prices -110 and +140, and a hand-computed sum-over-sum.

        An outcome-derived return would treat both wins as the reference juice and report
        ``_PAYOUT_AT_MINUS_110``. The final assertion is that it does NOT.
        """
        frame = _frame(
            [
                _row(
                    _CLEAN,
                    GRADING_STATUS_WIN,
                    flat_stake=1.0,
                    payout_flat=_PAYOUT_AT_MINUS_110,
                    selected_odds=-110.0,
                ),
                _row(
                    _CLEAN,
                    GRADING_STATUS_WIN,
                    flat_stake=1.0,
                    payout_flat=_PAYOUT_AT_PLUS_140,
                    selected_odds=140.0,
                ),
            ]
        )
        block = aggregate_by_provenance(
            frame,
            provenance=PROVENANCE_BACKTEST_REPLAY,
            validation_type=VALIDATION_TYPE_CLEAN_HOLDOUT,
        )
        assert isinstance(block, TrackerBlock)
        hand_computed = (_PAYOUT_AT_MINUS_110 + _PAYOUT_AT_PLUS_140) / 2.0
        assert block.flat_return_units == pytest.approx(hand_computed)
        outcome_derived = _PAYOUT_AT_MINUS_110
        assert block.flat_return_units != pytest.approx(outcome_derived), (
            "the return equals the reference-juice value, so it was derived from the outcome "
            "rather than read from the stored payout"
        )

    def test_the_push_stake_is_in_the_return_denominator_but_not_the_hit_rate_one(
        self,
    ) -> None:
        """The two denominators are DIFFERENT and both are stated.

        A push returns the stake, so it turns over money at zero profit: its stake belongs in the
        return denominator. It is not a contest won or lost, so it does not belong in the hit-rate
        denominator. Collapsing the two would be wrong in one direction or the other.
        """
        frame = _frame(
            [
                _row(_REPLAY, GRADING_STATUS_WIN, flat_stake=1.0),
                _row(_REPLAY, GRADING_STATUS_PUSH, flat_stake=1.0),
            ]
        )
        block = aggregate_by_provenance(
            frame,
            provenance=PROVENANCE_BACKTEST_REPLAY,
            validation_type=VALIDATION_TYPE_CONTAMINATED,
        )
        assert isinstance(block, TrackerBlock)
        assert block.hit_rate == 1.0  # 1 win / 1 contested
        assert block.flat_return_units == pytest.approx(_PAYOUT_AT_MINUS_110 / 2.0)

    def test_a_zero_total_stake_reports_an_unmeasured_return_not_a_zero_one(
        self,
    ) -> None:
        """None is not 0.0.

        An unmeasured return published as zero reads as "we broke even", which is a claim. This
        module inherits the ``_clv_report`` rule that a figure nobody measured is None.
        """
        frame = _frame(
            [_row(_REPLAY, GRADING_STATUS_PUSH, flat_stake=0.0, payout_flat=0.0)]
        )
        block = aggregate_by_provenance(
            frame,
            provenance=PROVENANCE_BACKTEST_REPLAY,
            validation_type=VALIDATION_TYPE_CONTAMINATED,
        )
        assert isinstance(block, TrackerBlock)
        assert block.flat_return_units is None
        assert block.flat_return_units != 0.0


# ---------------------------------------------------------------------------
# 4. A zero-graded block does not compute a rate
# ---------------------------------------------------------------------------


class TestAZeroGradedBlockRefusesToComputeARate:
    """T-31-66: the hit rate is not computed at all in that branch."""

    def test_an_all_pending_block_returns_the_empty_marker(self) -> None:
        """A freshly written forward week. Every row is pending, so nothing has been measured."""
        frame = _frame([_row(_FORWARD, GRADING_STATUS_PENDING) for _ in range(5)])
        block = aggregate_by_provenance(
            frame,
            provenance=PROVENANCE_FORWARD,
            validation_type=VALIDATION_TYPE_FORWARD_REALIZED,
        )
        assert isinstance(block, EmptyTrackerBlock)
        assert not isinstance(block, TrackerBlock)
        assert block.bets_graded == 0

    def test_the_empty_marker_has_no_hit_rate_field_at_all(self) -> None:
        """Absent, not zero and not null. The template renders an empty state from that."""
        frame = _frame([_row(_FORWARD, GRADING_STATUS_PENDING)])
        block = aggregate_by_provenance(
            frame,
            provenance=PROVENANCE_FORWARD,
            validation_type=VALIDATION_TYPE_FORWARD_REALIZED,
        )
        assert not hasattr(block, "hit_rate")
        assert not hasattr(block, "flat_return_units")

    def test_an_entirely_empty_frame_returns_the_empty_marker(self) -> None:
        frame = pd.DataFrame(columns=list(TRACKER_REQUIRED_COLUMNS))
        block = aggregate_by_provenance(
            frame,
            provenance=PROVENANCE_FORWARD,
            validation_type=VALIDATION_TYPE_FORWARD_REALIZED,
        )
        assert isinstance(block, EmptyTrackerBlock)
        assert aggregate_all_blocks(frame) == []

    def test_the_populated_block_carries_exactly_the_six_declared_figures(self) -> None:
        frame = _frame([_row(_REPLAY, GRADING_STATUS_WIN)])
        block = aggregate_by_provenance(
            frame,
            provenance=PROVENANCE_BACKTEST_REPLAY,
            validation_type=VALIDATION_TYPE_CONTAMINATED,
        )
        assert isinstance(block, TrackerBlock)
        # ``dataclasses.fields`` and NOT ``__dataclass_fields__``: the latter also carries the
        # ClassVar pseudo-fields, which are markers rather than per-block data.
        names = {f.name for f in dataclasses.fields(block)}
        assert names == {"provenance", "validation_type", *TRACKER_BLOCK_FIGURES}
        assert TRACKER_BLOCK_FIGURES == (
            "bets_graded",
            "wins",
            "losses",
            "pushes",
            "hit_rate",
            "flat_return_units",
        )


# ---------------------------------------------------------------------------
# 5. The persistence handoff and the required columns
# ---------------------------------------------------------------------------


class TestThePersistenceHandoff:
    """``to_tracker_frame`` produces exactly what the pure-persistence writer requires."""

    def test_the_frame_columns_are_the_writers_locked_order(self) -> None:
        """Derived from ``api.cache.BET_TRACKER_BLOCK_COLUMNS``, never a second hand-typed list."""
        frame = to_tracker_frame(
            aggregate_all_blocks(_frame([_row(_REPLAY, GRADING_STATUS_WIN)]))
        )
        assert list(frame.columns) == BET_TRACKER_BLOCK_COLUMNS

    def test_an_empty_block_persists_as_zero_counts_and_null_figures(self) -> None:
        """SQL cannot express an ABSENT column, so the two rate fields are NULL, never 0.0."""
        frame = to_tracker_frame(
            [EmptyTrackerBlock(PROVENANCE_FORWARD, VALIDATION_TYPE_FORWARD_REALIZED)]
        )
        row = frame.iloc[0]
        assert row["bets_graded"] == 0
        assert row["hit_rate"] is None
        assert row["flat_return_units"] is None

    def test_no_blocks_produce_an_empty_frame_with_the_columns_present(self) -> None:
        frame = to_tracker_frame([])
        assert len(frame) == 0
        assert list(frame.columns) == BET_TRACKER_BLOCK_COLUMNS

    def test_a_missing_required_column_raises_a_named_key_error(self) -> None:
        frame = _frame([_row(_REPLAY, GRADING_STATUS_WIN)]).drop(
            columns=["payout_flat"]
        )
        with pytest.raises(KeyError) as excinfo:
            aggregate_all_blocks(frame)
        assert "payout_flat" in str(excinfo.value)

    def test_an_out_of_vocabulary_grading_status_raises(self) -> None:
        frame = _frame([_row(_REPLAY, GRADING_STATUS_WIN)])
        frame.loc[0, "grading_status"] = "settled"
        with pytest.raises(ValueError) as excinfo:
            aggregate_all_blocks(frame)
        assert "settled" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 6. Tier placement: the tracker is analytics, and it computes nothing it was given
# ---------------------------------------------------------------------------


def _tracker_tree() -> ast.Module:
    return ast.parse(TRACKER_PATH.read_text(encoding="utf-8"))


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
    return roots


def _called_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


class TestTheTrackerLivesInTheAnalyticsTierAndPricesNothing:
    """T-31-63/64d: aggregation is computation, and computation never runs in the request path."""

    def test_the_module_lives_under_backtest_not_under_api(self) -> None:
        assert TRACKER_PATH.exists()
        assert not (REPO_ROOT / "api" / "bet_tracker.py").exists()

    def test_it_imports_no_model_feature_or_rating_module(self) -> None:
        forbidden = {"models", "features", "ratings", "api"} & _imported_roots(
            _tracker_tree()
        )
        # ``api`` is permitted for the LABEL VOCABULARY only -- asserted narrowly below.
        forbidden.discard("api")
        assert not forbidden, sorted(forbidden)

    def test_it_imports_from_api_cache_only_the_shared_vocabulary(self) -> None:
        """One source for the labels and the column order, and no behaviour crosses the seam.

        The dependency direction that matters is the OTHER one: ``api/`` must not import
        ``backtest``. Naming the vocabulary here rather than re-typing it is what stops the
        tracker and the writer disagreeing about what ``push`` is called.
        """
        imported: set[str] = set()
        for node in ast.walk(_tracker_tree()):
            if isinstance(node, ast.ImportFrom) and node.module == "api.cache":
                imported.update(alias.name for alias in node.names)
        assert imported, (
            "the tracker must take the vocabulary from api.cache, not re-type it"
        )
        assert all(
            name.isupper() or name.startswith("GRADING_") for name in imported
        ), f"only UPPER_CASE constants may cross this seam; got {sorted(imported)}"

    def test_it_opens_no_duckdb_connection_and_writes_no_file(self) -> None:
        source = TRACKER_PATH.read_text(encoding="utf-8")
        roots = _imported_roots(_tracker_tree())
        assert "duckdb" not in roots
        called = _called_names(_tracker_tree())
        assert not (
            {"connect", "open", "to_parquet", "to_csv", "write_text", "mkdir"} & called
        ), sorted(
            {"connect", "open", "to_parquet", "to_csv", "write_text", "mkdir"} & called
        )
        assert "INSERT INTO" not in source
        assert "CREATE TABLE" not in source

    def test_it_calls_no_pricing_or_ev_helper(self) -> None:
        """The tracker aggregates what was stored. It re-derives no price and no expected value.

        This is what makes it correct under either resolution of the open per-target pricing
        question: a ruling changes the values written into the blob, and changes nothing here.
        """
        called = _called_names(_tracker_tree())
        pricing = {
            "american_to_payout",
            "per_bet_ev",
            "devig",
            "implied_probability",
            "moneyline_to_probability",
            "apply_sizing_pipeline",
            "kelly_fraction",
        }
        assert not (pricing & called), sorted(pricing & called)

    def test_api_cache_still_imports_no_backtest_module(self) -> None:
        """The seam holds from the other side too, asserted here as well as in the api guard."""
        tree = ast.parse((REPO_ROOT / "api" / "cache.py").read_text(encoding="utf-8"))
        assert "backtest" not in _imported_roots(tree)


# ---------------------------------------------------------------------------
# 7. The DataService getter reads the stored aggregate and computes nothing
# ---------------------------------------------------------------------------


def _service_over_stored_blocks(blocks: list[Any]) -> Any:
    """Persist *blocks* through the REAL writer and serve them through the REAL getter."""
    conn = duckdb.connect(":memory:")
    materialize_bet_tracker_blocks(conn, to_tracker_frame(blocks))
    clear_cache()
    return DataService(conn)


class TestTheDataServiceGetterReadsAndDoesNotCompute:
    """T-31-63: the page reads stored rows; no figure is derived at request time."""

    def test_the_getter_returns_the_stored_blocks_unchanged(self) -> None:
        frame = _frame(
            [
                _row(_REPLAY, GRADING_STATUS_WIN),
                _row(_REPLAY, GRADING_STATUS_LOSS),
                _row(_REPLAY, GRADING_STATUS_PUSH),
            ]
        )
        blocks = aggregate_all_blocks(frame)
        rows = _service_over_stored_blocks(blocks).get_bet_tracker_blocks()
        assert len(rows) == 1
        row = rows[0]
        block = blocks[0]
        assert isinstance(block, TrackerBlock)
        assert row["provenance"] == block.provenance
        assert row["validation_type"] == block.validation_type
        assert row["bets_graded"] == block.bets_graded
        assert row["wins"] == block.wins
        assert row["losses"] == block.losses
        assert row["pushes"] == block.pushes
        assert row["hit_rate"] == pytest.approx(block.hit_rate)
        assert row["flat_return_units"] == pytest.approx(block.flat_return_units)

    def test_a_zero_graded_block_serves_a_null_rate_beside_a_zero_count(self) -> None:
        """The template's empty-state pair. A NULL rate is NOT a measured zero."""
        rows = _service_over_stored_blocks(
            [EmptyTrackerBlock(PROVENANCE_FORWARD, VALIDATION_TYPE_FORWARD_REALIZED)]
        ).get_bet_tracker_blocks()
        assert rows[0]["bets_graded"] == 0
        assert rows[0]["hit_rate"] is None
        assert rows[0]["flat_return_units"] is None

    def test_the_getter_pools_nothing_across_classes(self) -> None:
        """Two stored classes come back as two rows; there is no combined row to read."""
        frame = _frame(
            [
                _row(_REPLAY, GRADING_STATUS_WIN),
                _row(_FORWARD, GRADING_STATUS_LOSS),
            ]
        )
        rows = _service_over_stored_blocks(
            aggregate_all_blocks(frame)
        ).get_bet_tracker_blocks()
        assert len(rows) == 2
        assert {(r["provenance"], r["validation_type"]) for r in rows} == {
            _REPLAY,
            _FORWARD,
        }

    def test_an_absent_table_serves_an_empty_list_rather_than_raising(self) -> None:
        clear_cache()
        assert DataService(duckdb.connect(":memory:")).get_bet_tracker_blocks() == []

    def test_the_getter_performs_no_arithmetic_and_no_sql_aggregate(self) -> None:
        """Source-level: neither Python arithmetic nor a SQL aggregate function appears.

        The tracker figures were computed once, at population time, in the analytics tier. A
        ``SUM``/``COUNT``/``AVG`` here would be the same computation running again per request,
        which is the rule UIAP-01 exists to state.
        """
        tree = ast.parse(
            (REPO_ROOT / "api" / "services.py").read_text(encoding="utf-8")
        )
        target = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_get_bet_tracker_blocks_uncached"
        )
        arithmetic = [
            node
            for node in ast.walk(target)
            if isinstance(node, (ast.BinOp, ast.AugAssign))
        ]
        assert not arithmetic, [ast.dump(n) for n in arithmetic]
        source = ast.get_source_segment(
            (REPO_ROOT / "api" / "services.py").read_text(encoding="utf-8"), target
        )
        assert source is not None
        upper = source.upper()
        for aggregate in ("SUM(", "COUNT(", "AVG(", "GROUP BY", "MIN(", "MAX("):
            assert aggregate not in upper, aggregate
