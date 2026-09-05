"""The NAMED pre-ingest removal of synthetic rows already sitting in stored silver.

TEST CLASS: plain unit tests. Every table these tests touch is built inside ``tmp_path``;
no production store is read or written, and the module passes on a fresh checkout with no
``data/``.

WHY THE STEP EXISTS. Clause 7's gate (``assert_no_synthetic_game_ids``) runs on the
INCOMING frame, so it cannot see a forged row already in the destination, and
``upsert_silver`` replaces by key rather than truncating, so the merge cannot remove one
either. Production silver holds exactly one such row -- ``2025_W01_TEST@HOME``, a
hand-written fixture carrying the legitimate sportsbook ``draftkings``, which the OUM-06
allowlist therefore ADMITS. Plan 31-08's summary recorded its survival as a carry-forward
and said plainly that removing it "does not happen by itself".

WHAT IS PROVED HERE:

* the removal really removes, and the row it removes is exactly the row clause 7's gate
  would have refused -- because both read ONE predicate, ``find_synthetic_game_ids``;
* it is idempotent AND it does not rewrite a clean table at all, so a second run cannot
  move a byte;
* it leaves every legitimate row, every column and every value untouched;
* an ORPHAN (well-formed id naming no gold game) is removed too, not just a malformed one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_odds_preingest import (
    assert_no_synthetic_game_ids,
    find_synthetic_game_ids,
)
from scripts.ingest_historical_odds import (
    normalize_stored_game_ids,
    remove_synthetic_stored_rows,
)

# The exact forged row in production silver, restated here as the fixture under test.
SYNTHETIC_ID = "2025_W01_TEST@HOME"
ORPHAN_ID = "2025_W01_BAL@KC"
REAL_ID = "2024_W01_BAL@KC"


def _odds_row(game_id: str, *, spread: float = -3.0, total: float = 46.5) -> dict:
    return {
        "game_id": game_id,
        "sportsbook": "draftkings",
        "snapshot_ts": "2024-09-06 22:00:00",
        "spread": spread,
        "total": total,
        "ml_home": -150,
        "ml_away": 130,
    }


def _write_stored(tmp_path: Path, rows: list[dict]) -> Path:
    silver = tmp_path / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    path = silver / "odds_snapshot.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _gold_ou(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"game_id": ids})


class TestTheRemovalRemoves:
    def test_the_malformed_fixture_row_is_deleted_and_the_real_rows_survive(
        self, tmp_path: Path
    ) -> None:
        _write_stored(tmp_path, [_odds_row(REAL_ID), _odds_row(SYNTHETIC_ID)])

        report = remove_synthetic_stored_rows(
            base_path=tmp_path, features_ou_df=_gold_ou([REAL_ID])
        )

        assert report.rows_before == 2
        assert report.rows_after == 1
        assert report.rows_removed == 1
        assert report.removed_malformed == (SYNTHETIC_ID,)
        assert report.removed_orphans == ()

        stored = pd.read_parquet(report.path)
        assert list(stored["game_id"]) == [REAL_ID]

    def test_an_orphan_is_removed_too_not_only_a_malformed_id(
        self, tmp_path: Path
    ) -> None:
        """A well-formed id naming no gold game is the second failure class, and it is
        the one an inner join would "most likely" have dropped. Most likely is not a
        guarantee."""
        _write_stored(tmp_path, [_odds_row(REAL_ID), _odds_row(ORPHAN_ID)])

        report = remove_synthetic_stored_rows(
            base_path=tmp_path, features_ou_df=_gold_ou([REAL_ID])
        )

        assert report.removed_orphans == (ORPHAN_ID,)
        assert report.removed_malformed == ()
        assert list(pd.read_parquet(report.path)["game_id"]) == [REAL_ID]

    def test_every_surviving_value_is_untouched(self, tmp_path: Path) -> None:
        """The step deletes rows. It must not rewrite one."""
        rows = [_odds_row(REAL_ID, spread=-7.5, total=51.0), _odds_row(SYNTHETIC_ID)]
        path = _write_stored(tmp_path, rows)
        before = pd.read_parquet(path)
        expected = before[before["game_id"] == REAL_ID].reset_index(drop=True)

        remove_synthetic_stored_rows(
            base_path=tmp_path, features_ou_df=_gold_ou([REAL_ID])
        )

        after = pd.read_parquet(path).reset_index(drop=True)
        assert list(after.columns) == list(before.columns)
        pd.testing.assert_frame_equal(after, expected)


class TestTheRemovalIsIdempotentAndCheap:
    def test_a_clean_table_is_not_rewritten_at_all(self, tmp_path: Path) -> None:
        """Not merely "the same rows come back" -- the FILE must not move.

        A step that rewrites a clean production store on every invocation would show up in
        the content-digest boundary guard as a data move, and a boundary alarm that fires
        for a no-op is an alarm people learn to ignore.
        """
        path = _write_stored(tmp_path, [_odds_row(REAL_ID)])
        digest_before = path.read_bytes()

        report = remove_synthetic_stored_rows(
            base_path=tmp_path, features_ou_df=_gold_ou([REAL_ID])
        )

        assert report.rows_removed == 0
        assert path.read_bytes() == digest_before, (
            "the removal step rewrote a table it removed nothing from"
        )

    def test_a_second_run_removes_nothing_and_moves_no_byte(
        self, tmp_path: Path
    ) -> None:
        _write_stored(tmp_path, [_odds_row(REAL_ID), _odds_row(SYNTHETIC_ID)])
        gold = _gold_ou([REAL_ID])

        first = remove_synthetic_stored_rows(base_path=tmp_path, features_ou_df=gold)
        bytes_after_first = first.path.read_bytes()
        second = remove_synthetic_stored_rows(base_path=tmp_path, features_ou_df=gold)

        assert first.rows_removed == 1
        assert second.rows_removed == 0
        assert second.path.read_bytes() == bytes_after_first

    def test_an_absent_table_is_reported_rather_than_raising(
        self, tmp_path: Path
    ) -> None:
        report = remove_synthetic_stored_rows(
            base_path=tmp_path, features_ou_df=_gold_ou([REAL_ID])
        )
        assert report.rows_before == 0
        assert report.rows_removed == 0


class TestTheRemovalAndTheGateShareOnePredicate:
    """A row this step deletes is EXACTLY a row clause 7's gate would refuse.

    If the two ever came apart, the ingest would either refuse a row the cleanup left
    behind (a stall) or leave behind a row the gate would have rejected (a contaminated
    population). One registry is why neither can happen.
    """

    def test_what_the_removal_deletes_is_what_the_gate_refuses(
        self, tmp_path: Path
    ) -> None:
        stored = pd.DataFrame(
            [_odds_row(REAL_ID), _odds_row(SYNTHETIC_ID), _odds_row(ORPHAN_ID)]
        )
        gold = _gold_ou([REAL_ID])

        with pytest.raises(ValueError) as refusal:
            assert_no_synthetic_game_ids(stored, gold)
        message = str(refusal.value)
        assert SYNTHETIC_ID in message
        assert ORPHAN_ID in message

        _write_stored(tmp_path, list(stored.to_dict(orient="records")))
        report = remove_synthetic_stored_rows(base_path=tmp_path, features_ou_df=gold)

        deleted = set(report.removed_malformed) | set(report.removed_orphans)
        assert deleted == {SYNTHETIC_ID, ORPHAN_ID}

        # And the cleaned table now PASSES the gate that had refused it.
        assert_no_synthetic_game_ids(pd.read_parquet(report.path), gold)

    def test_the_gate_delegates_to_the_shared_predicate(self) -> None:
        """Held as a test so a future edit cannot re-inline a second copy of the rule."""
        import ast
        import inspect

        from scripts import audit_odds_preingest

        source = inspect.getsource(audit_odds_preingest.assert_no_synthetic_game_ids)
        tree = ast.parse(source.lstrip())
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "find_synthetic_game_ids" in called, (
            "assert_no_synthetic_game_ids no longer reads the shared predicate, so the "
            "gate and the removal step can now disagree about what 'synthetic' means."
        )

    def test_the_predicate_reports_both_classes_separately(self) -> None:
        malformed, orphans = find_synthetic_game_ids(
            pd.DataFrame(
                [_odds_row(REAL_ID), _odds_row(SYNTHETIC_ID), _odds_row(ORPHAN_ID)]
            ),
            _gold_ou([REAL_ID]),
        )
        assert malformed == [SYNTHETIC_ID]
        assert orphans == [ORPHAN_ID]

    def test_a_frame_without_a_game_id_column_is_refused_not_silently_passed(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="no 'game_id' column"):
            find_synthetic_game_ids(pd.DataFrame({"sportsbook": ["dk"]}), _gold_ou([]))


class TestTheStepRefusesToRunBeforeTheKeyNormalization:
    """The two pre-ingest steps are NOT commutative, and the wrong order is destructive.

    Measured on production silver on 2026-09-05: run before the clause-5 key
    normalization, this step classifies all 116 non-canonical ``LAR`` Rams rows as
    ORPHANS -- gold keys the Rams as canonical ``LA``, so a well-formed
    ``2018_W01_LAR@LV`` names no game in ``features_ou`` -- and would delete sixteen
    seasons of real accumulated odds history. Run after it, the orphan set is empty and
    the only row removed is the malformed fixture.

    The destructive order is the one an operator reaches for first (clean up the obvious
    forgery, then merge), so it REFUSES rather than proceeding.
    """

    NON_CANONICAL_RAMS = "2018_W01_LAR@LV"
    CANONICAL_RAMS = "2018_W01_LA@LV"

    def test_a_non_canonical_stored_key_refuses_and_names_the_fix(
        self, tmp_path: Path
    ) -> None:
        path = _write_stored(
            tmp_path,
            [_odds_row(self.NON_CANONICAL_RAMS), _odds_row(SYNTHETIC_ID)],
        )
        before = path.read_bytes()

        with pytest.raises(ValueError) as refusal:
            remove_synthetic_stored_rows(
                base_path=tmp_path, features_ou_df=_gold_ou([self.CANONICAL_RAMS])
            )

        message = str(refusal.value)
        assert "NON-CANONICAL" in message
        assert "normalize_stored_game_ids" in message, (
            "the refusal does not name the step that satisfies it, so an operator has to "
            "guess -- and the guess that looks obvious is the destructive one"
        )
        assert path.read_bytes() == before, (
            "the step refused but had already rewritten the table"
        )

    def test_after_normalizing_the_same_call_removes_only_the_forgery(
        self, tmp_path: Path
    ) -> None:
        """The proof that the refusal is about ORDER, not about the data being unusable."""
        _write_stored(
            tmp_path,
            [_odds_row(self.NON_CANONICAL_RAMS), _odds_row(SYNTHETIC_ID)],
        )
        gold = _gold_ou([self.CANONICAL_RAMS])

        rekeyed = normalize_stored_game_ids(tmp_path)
        assert rekeyed == 1

        report = remove_synthetic_stored_rows(base_path=tmp_path, features_ou_df=gold)

        assert report.removed_malformed == (SYNTHETIC_ID,)
        assert report.removed_orphans == (), (
            "the Rams row was deleted as an orphan even after normalization"
        )
        assert list(pd.read_parquet(report.path)["game_id"]) == [self.CANONICAL_RAMS]

    def test_without_the_guard_the_rams_row_would_have_been_classified_an_orphan(
        self,
    ) -> None:
        """Names the hazard directly, so the guard above cannot be read as paranoia."""
        stored = pd.DataFrame(
            [_odds_row(self.NON_CANONICAL_RAMS), _odds_row(SYNTHETIC_ID)]
        )
        malformed, orphans = find_synthetic_game_ids(
            stored, _gold_ou([self.CANONICAL_RAMS])
        )
        assert malformed == [SYNTHETIC_ID]
        assert orphans == [self.NON_CANONICAL_RAMS], (
            "a non-canonical stored key is no longer judged an orphan against canonical "
            "gold, so the ordering guard is now protecting against nothing -- check "
            "whether it can be removed rather than leaving a guard that guards nothing."
        )
