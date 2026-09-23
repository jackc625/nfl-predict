"""Every tz-aware datetime column in silver is stored in ONE representation.

Plan 33.2-20 continuation (deferred-items entry found by Plan 33.2-16, not caused by it).

THE DEFECT
----------
``data/silver/weather.parquet`` held ``created_at`` as ``datetime64[us, UTC]`` while its
three siblings in the same table were ``datetime64[ns, UTC]``. The table predates Plan
33.2-15's ``1344ccf``, which taught ``data.storage.upsert_silver`` to align a mismatched
tz-aware pair to ``datetime64[ns, UTC]`` before concatenating -- so an offline
regeneration produced ``ns`` and the comparison against production failed on DTYPE before
a single value was compared, in the one test asserting that silver weather rebuilds
offline and reproduces what is deployed.

WHY THE STORE MOVED AND NOT THE COMPARISON
-------------------------------------------
Making the writer preserve whatever unit is stored would have fixed the symptom and left
the representation a function of the table's own history: the same column would be ``us``
or ``ns`` depending on when it was last written. Aligning the store gives it the unit
``upsert_silver`` already converges on, so the writer's rule and the store agree, and the
regeneration comparison means what it says. Relaxing the comparison was never an option.

A CONVERSION, NEVER A RELABEL (D33.2-01): ``us`` to ``ns`` widens the same tz-aware
instants, and the repair asserted every instant unchanged on both sides.

WHAT THIS MODULE GUARDS, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------------------
It asserts the invariant PER TABLE, over whatever silver tables the checkout has, rather
than naming ``weather`` alone -- a rule that only covers the table that broke is a rule
that waits for the next one. It also asserts the invariant WITHIN a table, which is the
form the defect actually took: one column disagreeing with its own siblings.

It does NOT assert that every silver datetime is tz-AWARE. ``games.parquet`` carries
``kickoff_et`` and ``created_at`` at ``datetime64[us, UTC]`` today -- aware, but at the
other unit -- and aligning it is a separate declared write on a table nothing in this
plan touched. That is recorded as a known divergence below and asserted as such, so the
gap is a stated fact rather than a silent exemption: if `games` is ever aligned, this
test fails and the record is updated deliberately.

FOUR CONTROLS
-------------
1. NON-VACUITY: the scan is asserted to have found tz-aware columns at all, and the
   tables it read are named in the failure message.
2. THE ASSERTION: every tz-aware column is at the target unit, except the recorded
   divergence.
3. A PLANTED VIOLATION: a frame carrying a ``us`` column is detected by the same
   predicate the assertion uses.
4. NO FALSE POSITIVE: a frame already at ``ns``, and a frame with no datetime column at
   all, are not flagged.

ASCII only, no emoji (CLAUDE.md hard constraint). Read-only: nothing here writes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.align_silver_datetime_units import TARGET_DTYPE, plan_alignment

REPO_ROOT = Path(__file__).resolve().parents[2]
SILVER = REPO_ROOT / "data" / "silver"

#: Tables whose tz-aware datetime columns are NOT yet at the target unit, recorded rather
#: than exempted. ``games`` carries kickoff_et and created_at at ``datetime64[us, UTC]``;
#: aligning it is a declared write on a table Plan 33.2-20 did not touch, and doing it
#: unbracketed inside a test would be a production write nobody declared.
KNOWN_UNALIGNED_TABLES: tuple[str, ...] = ("games",)


def _silver_tables() -> list[Path]:
    if not SILVER.is_dir():
        pytest.skip(
            f"the silver layer is not present at {SILVER.as_posix()} -- data/ is "
            "gitignored, so a checkout that has not built the lake has no store to scan"
        )
    return sorted(SILVER.glob("*.parquet"))


def _aware_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {
        column: str(dtype)
        for column, dtype in frame.dtypes.items()
        if isinstance(dtype, pd.DatetimeTZDtype)
    }


class TestSilverStoresOneDatetimeRepresentation:
    def test_the_scan_found_tz_aware_columns_to_judge(self) -> None:
        """NON-VACUITY: a scan that finds nothing would pass for the wrong reason."""
        tables = _silver_tables()
        assert tables, "silver holds no parquet tables at all"
        found = {
            path.stem: _aware_columns(pd.read_parquet(path))
            for path in tables
            if _aware_columns(pd.read_parquet(path))
        }
        assert found, (
            f"no tz-aware datetime column in any of {[p.stem for p in tables]}"
        )

    def test_every_tz_aware_column_is_at_the_one_unit(self) -> None:
        offenders: dict[str, dict[str, str]] = {}
        for path in _silver_tables():
            if path.stem in KNOWN_UNALIGNED_TABLES:
                continue
            wrong = {
                column: dtype
                for column, dtype in _aware_columns(pd.read_parquet(path)).items()
                if dtype != TARGET_DTYPE
            }
            if wrong:
                offenders[path.stem] = wrong
        assert offenders == {}, (
            f"silver tables store a tz-aware datetime at a unit other than "
            f"{TARGET_DTYPE}: {offenders}. upsert_silver converges a mismatched pair on "
            "that unit, so a store at another one cannot be rebuilt into the same frame "
            "-- which is what broke the offline weather regeneration comparison. Align "
            "it with `python -m scripts.align_silver_datetime_units --table <name> "
            "--apply` under a digest bracket."
        )

    def test_no_table_disagrees_with_itself(self) -> None:
        """The form the defect took: one column at odds with its own siblings."""
        for path in _silver_tables():
            units = set(_aware_columns(pd.read_parquet(path)).values())
            assert len(units) <= 1, (
                f"{path.stem} stores its tz-aware datetimes at {sorted(units)}. A single "
                "table cannot have two answers about how it records an instant."
            )

    def test_the_known_divergence_is_still_exactly_what_was_recorded(self) -> None:
        """The recorded gap is asserted, so it cannot widen or be forgotten."""
        for table in KNOWN_UNALIGNED_TABLES:
            path = SILVER / f"{table}.parquet"
            if not path.is_file():
                continue
            wrong = {
                column: dtype
                for column, dtype in _aware_columns(pd.read_parquet(path)).items()
                if dtype != TARGET_DTYPE
            }
            assert wrong, (
                f"{table} is now aligned to {TARGET_DTYPE}. That is the intended end "
                "state -- remove it from KNOWN_UNALIGNED_TABLES so the assertion above "
                "starts covering it."
            )


class TestThePredicateIsCapableOfJudging:
    def test_a_planted_microsecond_column_is_flagged(self) -> None:
        planted = pd.DataFrame(
            {
                "game_id": ["G"],
                "created_at": pd.to_datetime(
                    ["2026-01-01T00:00:00+00:00"], utc=True
                ).astype("datetime64[us, UTC]"),
            }
        )
        assert _aware_columns(planted) == {"created_at": "datetime64[us, UTC]"}
        plan = plan_alignment(planted)
        assert plan.aligned == ("created_at",)
        assert str(plan.after["created_at"].dtype) == TARGET_DTYPE
        assert plan.after["created_at"].iloc[0] == planted["created_at"].iloc[0]

    def test_a_frame_already_at_the_target_is_not_flagged(self) -> None:
        clean = pd.DataFrame(
            {
                "game_id": ["G"],
                "created_at": pd.to_datetime(["2026-01-01T00:00:00+00:00"], utc=True),
            }
        )
        plan = plan_alignment(clean)
        assert plan.aligned == ()
        assert plan.already_aligned == ("created_at",)

    def test_a_frame_with_no_datetime_column_is_not_flagged(self) -> None:
        plain = pd.DataFrame({"game_id": ["G"], "spread": [-3.0]})
        plan = plan_alignment(plain)
        assert plan.aligned == () and plan.already_aligned == ()

    def test_a_naive_datetime_column_is_refused_rather_than_localized(self) -> None:
        """Guessing a zone is the relabel D33.2-01 forbids, so it raises by name."""
        from scripts.align_silver_datetime_units import SilverDatetimeUnitRepairError

        naive = pd.DataFrame(
            {
                "game_id": ["G"],
                "created_at": pd.to_datetime(["2026-01-01T00:00:00"]),
            }
        )
        with pytest.raises(SilverDatetimeUnitRepairError, match="created_at"):
            plan_alignment(naive)
