"""The section-4.2 PROMOTE: additive, keyed on game_id, and refusing rather than guessing.

WHAT IS UNDER TEST
------------------
``PROFITABILITY-PREREGISTRATION.md`` section 4.2 is FROZEN: "the historical four juice columns
are PROMOTED from the partitioned silver store, not re-ingested across the network", and clause
4.3(2) adds that the promote is ADDITIVE -- keyed on ``game_id``, with the stored ``spread``,
``total``, ``ml_home`` and ``ml_away`` NEVER overwritten.

``scripts/promote_historical_juice.py`` executes that clause. Every test below drives it against
a ``tmp_path`` store built by hand, so the write contract is proved without touching production
silver; the single LIVE control at the bottom reads the production table read-only and asserts
the promote's OUTCOME, which is the one claim a synthetic store cannot make.

WHY THE REFUSALS GET AS MUCH ATTENTION AS THE HAPPY PATH
--------------------------------------------------------
This project's real threat model is temporal integrity and honesty of record. A promote that
silently misses rows, or silently replaces a stored price with a different one, is worse than a
promote that refuses: the miss is invisible in every downstream number and the replacement is a
price change nobody ratified. So each of the module's four refusals is driven here with a store
constructed to trip exactly it.

TEST CLASS: integration. The tmp_path tests need no gitignored state and always run; the live
control reads ``data/silver/odds_snapshot.parquet`` and skips with a REGISTERED reason
(``tests/conftest._EVIDENCE_SKIP_MARKERS``) when the lake is absent, so a checkout without it
names the control that did not run rather than reporting a green suite that quietly excluded it.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from scripts.ingest_historical_odds import (
    JUICE_COLUMNS,
    PROTECTED_LINE_COLUMNS,
)
from scripts.promote_historical_juice import (
    PROMOTE_KEY_COLUMNS,
    build_promote_frame,
    promote_historical_juice,
    read_partitioned_juice,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_SOURCE = REPO_ROOT / "scripts" / "promote_historical_juice.py"
_LIVE_ODDS = REPO_ROOT / "data" / "silver" / "odds_snapshot.parquet"

# The seasons the promote covers on the live store. 2025 is DELIBERATELY absent: section 4.1
# states the partitioned store is not the 2025 source, and the live 2025 juice came from the
# nflreadpy closing lines through Plan 31-11's ingest.
_LIVE_HISTORICAL_SEASONS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)

_SPORTSBOOK = "consensus"


def _stored_row(game_id: str, *, spread: float, total: float) -> dict[str, object]:
    """One flat-table row in the production shape, with the four juice columns NULL."""
    return {
        "game_id": game_id,
        "sportsbook": _SPORTSBOOK,
        "ml_home": -150.0,
        "ml_away": 130.0,
        "spread": spread,
        "total": total,
        "is_live": False,
        "last_update": datetime(2024, 1, 1, tzinfo=UTC),
        "snapshot_ts": "2024-09-19T18:00:00-04:00",
        "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        "spread_ju_home": None,
        "spread_ju_away": None,
        "total_over_ju": None,
        "total_under_ju": None,
    }


def _partition_row(
    game_id: str,
    *,
    spread: float,
    total: float,
    juice: tuple[float, float, float, float] = (-108.0, -112.0, -105.0, -115.0),
) -> dict[str, object]:
    """One partitioned-store row. ``snapshot_ts`` is the partition key, so it is NOT a column."""
    return {
        "game_id": game_id,
        "sportsbook": _SPORTSBOOK,
        "ml_home": -150.0,
        "ml_away": 130.0,
        "spread": spread,
        "total": total,
        "spread_ju_home": juice[0],
        "spread_ju_away": juice[1],
        "total_over_ju": juice[2],
        "total_under_ju": juice[3],
        "is_live": False,
        "last_update": datetime(2024, 1, 1, tzinfo=UTC),
        "created_at": datetime(2024, 1, 1, tzinfo=UTC),
    }


def _write_store(
    base: Path,
    stored: list[dict[str, object]],
    partitions: dict[str, list[dict[str, object]]],
) -> Path:
    """Lay out a flat table plus partition directories under ``base/silver``."""
    silver = base / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    flat = silver / "odds_snapshot.parquet"
    pd.DataFrame(stored).to_parquet(flat, index=False)
    for name, rows in partitions.items():
        directory = silver / f"snapshot_ts={name}"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(directory / "part-0.parquet", index=False)
    return flat


def _features_ou(game_ids: list[str]) -> pd.DataFrame:
    """The gold O/U matrix the synthetic-id gate checks membership against."""
    return pd.DataFrame({"game_id": game_ids})


@pytest.fixture
def simple_store(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """A two-season store: 2023 and 2024 historical rows, plus one 2025 partition row.

    The 2025 row exists so "the hold partition is excluded BY CONSTRUCTION" is a fact about a
    store that actually contains one, rather than a claim about an empty case.
    """
    ids = ["2023_W01_BUF@NYJ", "2024_W01_KC@BAL"]
    stored = [
        _stored_row(ids[0], spread=-3.0, total=44.5),
        _stored_row(ids[1], spread=-2.5, total=46.0),
    ]
    partitions = {
        "2023-09-19T18%3A00%3A00-04%3A00": [
            _partition_row(ids[0], spread=-3.0, total=44.5)
        ],
        "2024-09-19T18%3A00%3A00-04%3A00": [
            _partition_row(ids[1], spread=-2.5, total=46.0)
        ],
        "2025-10-03%2018%3A00%3A00-04%3A00": [
            _partition_row("2025_W01_DAL@PHI", spread=-1.0, total=50.0)
        ],
    }
    flat = _write_store(tmp_path, stored, partitions)
    return tmp_path, flat, ids


class TestThePromoteIsAdditive:
    """Clause 4.3(2): the four juice columns are ADDED; the four stored lines are untouched."""

    @pytest.mark.integration
    def test_the_stored_lines_and_the_row_count_are_unchanged(
        self, simple_store: tuple[Path, Path, list[str]]
    ) -> None:
        base, flat, ids = simple_store
        before = pd.read_parquet(flat).set_index("game_id").sort_index()

        promote_historical_juice(base_path=base, features_ou_df=_features_ou(ids))

        after = pd.read_parquet(flat).set_index("game_id").sort_index()
        assert len(after) == len(before), (
            f"the promote took the table from {len(before)} rows to {len(after)}. A promote adds "
            "columns' VALUES, never rows."
        )
        for column in PROTECTED_LINE_COLUMNS:
            pd.testing.assert_series_equal(
                after[column], before[column], check_names=False
            )

    @pytest.mark.integration
    def test_every_matched_row_gains_all_four_juice_columns(
        self, simple_store: tuple[Path, Path, list[str]]
    ) -> None:
        base, flat, ids = simple_store
        assert pd.read_parquet(flat)[list(JUICE_COLUMNS)].notna().sum().sum() == 0

        report = promote_historical_juice(
            base_path=base, features_ou_df=_features_ou(ids)
        )

        after = pd.read_parquet(flat)
        assert report.stored_rows_matched == 2
        assert report.stored_rows_not_served == {}
        assert report.stored_rows_still_without_juice == {}
        for column in JUICE_COLUMNS:
            assert int(after[column].notna().sum()) == 2, (
                f"{column} is non-null on {int(after[column].notna().sum())} of 2 rows after the "
                "promote. A partial promote is the failure mode this test exists for."
            )

    @pytest.mark.integration
    def test_no_column_outside_the_four_juice_names_moved(
        self, simple_store: tuple[Path, Path, list[str]]
    ) -> None:
        """The strongest form of the additive claim: EVERY other column, not just the lines."""
        base, flat, ids = simple_store
        before = pd.read_parquet(flat).set_index("game_id").sort_index()

        promote_historical_juice(base_path=base, features_ou_df=_features_ou(ids))

        after = pd.read_parquet(flat).set_index("game_id").sort_index()
        assert set(after.columns) == set(before.columns)
        moved = [
            column
            for column in before.columns
            if column not in JUICE_COLUMNS and not before[column].equals(after[column])
        ]
        assert moved == [], (
            f"the promote moved {moved}, which section 4.2 does not authorize. Only "
            f"{list(JUICE_COLUMNS)} may change."
        )


class TestTheHoldPartitionIsNotASource:
    """Section 4.1: the partitioned store is NOT the 2025 source, stated before the run."""

    @pytest.mark.integration
    def test_a_hold_season_partition_row_is_dropped_before_any_value_is_read(
        self, simple_store: tuple[Path, Path, list[str]]
    ) -> None:
        base, _flat, _ids = simple_store
        promoted, rows_in, rows_after, _rekeyed = read_partitioned_juice(base)

        assert rows_in == 3
        assert rows_after == 2, (
            f"{rows_after} rows survived the hold exclusion, not 2. The 2025 partition holds "
            "nine sportsbooks of which the OUM-06 allowlist admits one and none is 'consensus'; "
            "it is excluded BY CONSTRUCTION rather than rejected at run time."
        )
        assert not promoted["game_id"].astype(str).str.startswith("2025").any()

    @pytest.mark.integration
    def test_a_stored_hold_row_is_left_without_juice_and_is_COUNTED(
        self, tmp_path: Path
    ) -> None:
        """A 2025 stored row the promote does not serve is REPORTED, never implied away."""
        ids = ["2024_W01_KC@BAL", "2025_W01_DAL@PHI"]
        stored = [
            _stored_row(ids[0], spread=-2.5, total=46.0),
            _stored_row(ids[1], spread=-1.0, total=50.0),
        ]
        partitions = {
            "2024-09-19T18%3A00%3A00-04%3A00": [
                _partition_row(ids[0], spread=-2.5, total=46.0)
            ],
            "2025-10-03%2018%3A00%3A00-04%3A00": [
                _partition_row(ids[1], spread=-1.0, total=50.0)
            ],
        }
        _write_store(tmp_path, stored, partitions)

        report = promote_historical_juice(
            base_path=tmp_path, features_ou_df=_features_ou(ids)
        )

        assert report.stored_rows_matched == 1
        assert report.stored_rows_not_served == {2025: 1}
        # ... and the SEPARATE claim: on this synthetic store the hold row carries no juice of its
        # own either, so it is ALSO a real coverage gap and is reported as one. On production the
        # two figures deliberately DIVERGE -- the 285 live 2025 rows are not served here and DO
        # carry juice, from Plan 31-11's nflreadpy ingest. Conflating the two would let a genuine
        # gap hide behind "that season was out of scope".
        assert report.stored_rows_still_without_juice == {2025: 1}


class TestTheTwoCoverageFiguresAreDifferentClaims:
    """ "Not served here" and "still has no juice" must never be read as one number."""

    @pytest.mark.integration
    def test_a_hold_row_that_already_carries_juice_is_not_served_and_is_not_a_gap(
        self, tmp_path: Path
    ) -> None:
        """The production shape, in miniature: 2025 is out of this promote's scope AND covered."""
        ids = ["2024_W01_KC@BAL", "2025_W01_DAL@PHI"]
        hold_row = _stored_row(ids[1], spread=-1.0, total=50.0)
        for column, value in zip(
            JUICE_COLUMNS, (-102.0, -118.0, -110.0, -110.0), strict=True
        ):
            hold_row[column] = value
        _write_store(
            tmp_path,
            [_stored_row(ids[0], spread=-2.5, total=46.0), hold_row],
            {
                "2024-09-19T18%3A00%3A00-04%3A00": [
                    _partition_row(ids[0], spread=-2.5, total=46.0)
                ]
            },
        )

        report = promote_historical_juice(
            base_path=tmp_path, features_ou_df=_features_ou(ids)
        )

        assert report.stored_rows_not_served == {2025: 1}, (
            "the hold row must still be reported as NOT SERVED by this promote -- that is a "
            "statement about scope, and it stays true whoever else covered the row."
        )
        assert report.stored_rows_still_without_juice == {}, (
            "the hold row already carries all four prices, so it is NOT a coverage gap. Reporting "
            "it as one would make every future gap indistinguishable from this permanent, "
            "expected exclusion."
        )


class TestTheRamsKeyIsNormalizedOnTheSourceSide:
    """Clause 4.3(5): ``LAR`` and ``LA`` are different keys, so a match needs canonical ids."""

    @pytest.mark.integration
    def test_a_LAR_keyed_partition_row_promotes_onto_the_LA_keyed_stored_row(
        self, tmp_path: Path
    ) -> None:
        stored_id = "2018_W01_LA@SF"
        stored = [_stored_row(stored_id, spread=-3.0, total=44.5)]
        partitions = {
            "2018-09-19T18%3A00%3A00-04%3A00": [
                _partition_row("2018_W01_LAR@SF", spread=-3.0, total=44.5)
            ]
        }
        flat = _write_store(tmp_path, stored, partitions)

        report = promote_historical_juice(
            base_path=tmp_path, features_ou_df=_features_ou([stored_id])
        )

        assert report.partition_rows_rekeyed == 1
        assert report.stored_rows_matched == 1, (
            "the LAR-keyed partition row did not match the LA-keyed stored row. Without the "
            "source-side canonicalization the promote silently misses every Rams game, which is "
            "exactly the silent miss this project treats as worse than a refusal."
        )
        after = pd.read_parquet(flat)
        assert float(after.loc[0, "spread_ju_home"]) == -108.0


class TestThePromoteIsIdempotent:
    """A second run must be a no-op, not a second opinion."""

    @pytest.mark.integration
    def test_running_it_twice_changes_nothing_the_second_time(
        self, simple_store: tuple[Path, Path, list[str]]
    ) -> None:
        base, flat, ids = simple_store
        promote_historical_juice(base_path=base, features_ou_df=_features_ou(ids))
        first = pd.read_parquet(flat).sort_values("game_id").reset_index(drop=True)

        report = promote_historical_juice(
            base_path=base, features_ou_df=_features_ou(ids)
        )

        second = pd.read_parquet(flat).sort_values("game_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(first, second)
        assert report.stored_rows_already_juiced == 2, (
            "the second run reported "
            f"{report.stored_rows_already_juiced} already-juiced rows, not 2. Idempotency that "
            "is not COUNTED cannot be distinguished from a promote that quietly re-wrote."
        )


class TestTheFourRefusals:
    """Each refusal is driven by a store constructed to trip exactly it."""

    @pytest.mark.integration
    def test_disagreeing_accumulated_copies_are_refused(self, tmp_path: Path) -> None:
        game_id = "2024_W01_KC@BAL"
        partitions = {
            "2024-09-19T18%3A00%3A00-04%3A00": [
                _partition_row(game_id, spread=-2.5, total=46.0),
                _partition_row(
                    game_id,
                    spread=-2.5,
                    total=46.0,
                    juice=(-120.0, -112.0, -105.0, -115.0),
                ),
            ]
        }
        _write_store(
            tmp_path, [_stored_row(game_id, spread=-2.5, total=46.0)], partitions
        )

        with pytest.raises(ValueError, match="DISAGREEING"):
            read_partitioned_juice(tmp_path)

    @pytest.mark.integration
    def test_a_null_partitioned_juice_value_is_refused(self, tmp_path: Path) -> None:
        game_id = "2024_W01_KC@BAL"
        row = _partition_row(game_id, spread=-2.5, total=46.0)
        row["total_under_ju"] = None
        _write_store(
            tmp_path,
            [_stored_row(game_id, spread=-2.5, total=46.0)],
            {"2024-09-19T18%3A00%3A00-04%3A00": [row]},
        )

        with pytest.raises(ValueError, match="NULL juice"):
            read_partitioned_juice(tmp_path)

    @pytest.mark.integration
    def test_a_line_disagreement_reopens_the_branch_rule_and_is_refused(
        self, tmp_path: Path
    ) -> None:
        """D31-39 clause 3 -- row-for-row agreement -- is re-evaluated at WRITE time."""
        game_id = "2024_W01_KC@BAL"
        _write_store(
            tmp_path,
            [_stored_row(game_id, spread=-2.5, total=46.0)],
            {
                "2024-09-19T18%3A00%3A00-04%3A00": [
                    _partition_row(game_id, spread=-7.5, total=46.0)
                ]
            },
        )

        with pytest.raises(ValueError, match="DISAGREES with the flat table"):
            promote_historical_juice(
                base_path=tmp_path, features_ou_df=_features_ou([game_id])
            )

    @pytest.mark.integration
    def test_a_stored_juice_value_is_never_replaced_with_a_different_one(
        self, tmp_path: Path
    ) -> None:
        game_id = "2024_W01_KC@BAL"
        stored = _stored_row(game_id, spread=-2.5, total=46.0)
        stored["spread_ju_home"] = -101.0
        _write_store(
            tmp_path,
            [stored],
            {
                "2024-09-19T18%3A00%3A00-04%3A00": [
                    _partition_row(game_id, spread=-2.5, total=46.0)
                ]
            },
        )

        with pytest.raises(ValueError, match="would OVERWRITE"):
            promote_historical_juice(
                base_path=tmp_path, features_ou_df=_features_ou([game_id])
            )


class TestThePromoteWritesOnlyThroughTheGatedPath:
    """The write contract is honoured because there is no other write to honour it instead."""

    @pytest.mark.integration
    def test_the_module_calls_write_odds_additively_and_nothing_else(self) -> None:
        tree = ast.parse(_MODULE_SOURCE.read_text(encoding="utf-8"))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "write_odds_additively" in called
        for forbidden in ("upsert_silver", "ingest_historical_odds_for_seasons"):
            assert forbidden not in called, (
                f"{forbidden} is called by scripts/promote_historical_juice.py. That path "
                "hard-codes the production root, runs no gate, and -- because it replaces a "
                "whole row by key -- overwrites a stored line by construction."
            )

    @pytest.mark.integration
    def test_the_synthetic_id_gate_refuses_before_the_table_is_written(
        self, simple_store: tuple[Path, Path, list[str]]
    ) -> None:
        """A gate that runs after a write is a report. Proven by the table not moving."""
        base, flat, ids = simple_store
        before = pd.read_parquet(flat)

        with pytest.raises(ValueError):
            # An O/U matrix naming only ONE of the two games makes the other an ORPHAN.
            promote_historical_juice(
                base_path=base, features_ou_df=_features_ou([ids[0]])
            )

        pd.testing.assert_frame_equal(pd.read_parquet(flat), before)


class TestTheIncomingFrameIsTheStoredRows:
    """Built as a copy of the destination, so "nothing else moved" holds BY CONSTRUCTION."""

    @pytest.mark.integration
    def test_the_incoming_frame_carries_the_stored_column_set_and_values(
        self, simple_store: tuple[Path, Path, list[str]]
    ) -> None:
        base, flat, _ids = simple_store
        stored = pd.read_parquet(flat)
        promoted, _in, _after, _rekeyed = read_partitioned_juice(base)

        incoming, _already, _without = build_promote_frame(stored, promoted)

        assert list(incoming.columns) == list(stored.columns)
        untouched = [c for c in stored.columns if c not in JUICE_COLUMNS]
        pd.testing.assert_frame_equal(
            incoming[untouched].sort_values("game_id").reset_index(drop=True),
            stored[untouched].sort_values("game_id").reset_index(drop=True),
        )
        assert list(PROMOTE_KEY_COLUMNS) == ["game_id", "sportsbook"]


class TestTheLiveOutcomeOfTheRuledPromote:
    """The one claim a tmp_path store cannot make: the PRODUCTION table carries the juice.

    Read-only. This asserts the OUTCOME of the 2026-09-05 owner-ruled promote, so a later run
    that quietly reverted it -- an odds table rebuilt from the network, say -- fails here rather
    than surfacing as a silently flat-priced tune window inside the 2025 verdict.
    """

    @pytest.mark.integration
    def test_every_historical_row_carries_all_four_juice_columns(self) -> None:
        if not _LIVE_ODDS.is_file():
            pytest.skip(
                "live silver odds not present at data/silver/odds_snapshot.parquet"
            )
        odds = pd.read_parquet(_LIVE_ODDS)
        odds = odds.assign(
            season=odds["game_id"].astype(str).str.slice(0, 4).astype(int)
        )

        for season in _LIVE_HISTORICAL_SEASONS:
            rows = odds[odds["season"] == season]
            if rows.empty:
                continue
            for column in JUICE_COLUMNS:
                missing = int(rows[column].isna().sum())
                assert missing == 0, (
                    f"season {season}: {missing} of {len(rows)} stored odds rows carry no "
                    f"'{column}'. The section-4.2 PROMOTE was executed on 2026-09-05 under an "
                    "owner ruling precisely so the DEF-31-13 devig binds across the whole "
                    "window; a gap here means the tune split is priced at the flat -110 "
                    "fallback while the 2025 hold is priced on devigged real juice."
                )

    @pytest.mark.integration
    def test_the_promoted_juice_is_real_and_asymmetric_not_a_minus_110_fill(
        self,
    ) -> None:
        """A -110 fill would satisfy the coverage test above and mean nothing."""
        if not _LIVE_ODDS.is_file():
            pytest.skip(
                "live silver odds not present at data/silver/odds_snapshot.parquet"
            )
        odds = pd.read_parquet(_LIVE_ODDS)
        historical = odds[
            odds["game_id"]
            .astype(str)
            .str.slice(0, 4)
            .astype(int)
            .isin(_LIVE_HISTORICAL_SEASONS)
        ]
        if historical.empty:
            pytest.skip(
                "live silver odds not present at data/silver/odds_snapshot.parquet"
            )
        for column in JUICE_COLUMNS:
            non_default = int((historical[column] != -110).sum())
            assert non_default > 0, (
                f"every historical '{column}' equals -110. That is the shape a fabricated fill "
                "has, and the PROMOTE branch (D31-39) was chosen BECAUSE the partitioned store "
                "carries real asymmetric prices."
            )
