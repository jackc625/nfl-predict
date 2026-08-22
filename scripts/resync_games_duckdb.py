"""Re-sync the STALE DuckDB copy of silver ``games`` from its authoritative parquet (N-01).

WHAT THIS SCRIPT DOES, AND WHY IT HAS TO EXIST
----------------------------------------------
``data.storage.load_dataframe(source="auto")`` resolves the DuckDB copy FIRST whenever the
table exists and only falls back to parquet. So a DuckDB copy that has fallen behind its
parquet makes every ``upsert_silver`` write since the divergence INVISIBLE to the whole
pipeline -- silently, with no error raised anywhere, by any component. That is not
hypothetical: at Phase-30 start the silver ``games`` parquet held 6,499 rows against 6,292 in
DuckDB, a 207-row divergence, all of them season 2025 (measured by Plan 30-04 Task 2 and pinned
in ``tests/phase30_state.py``).

This script performs the NARROWEST possible correction: the parquet is authoritative, so it
loads the parquet and replaces the DuckDB table wholesale, with parquet writing switched OFF so
the authoritative file is left byte-untouched.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-----------------------------------------
* It does NOT change the shared read path. ``load_dataframe``'s ``source="auto"`` preference
  for DuckDB is the seam every builder, trainer and backtest goes through, and changing it in
  the same phase that is trying to MEASURE a gold rebuild would be exactly the confound the
  phase is built to avoid (D30-18).
* It does NOT fix the ``upsert_silver`` / ``load_dataframe`` write-path asymmetry that let the
  two copies drift apart in the first place. The SPEC puts that out of scope explicitly, as a
  separate architectural change.

Fixing the stale copy is a data correction. Fixing the asymmetry is an architecture change.
This script is only the first.

THE RE-SYNC IS ALSO A POSITIVE CONTROL (SPEC R2)
------------------------------------------------
N-01 was held open through Phase 29 on purpose. With whole-frame-fitted imputation medians and
q01/q99 bounds, adding 2025 rows WOULD have moved 2021-2024 feature values -- that coupling IS
the WR-06 defect. Only with WR-06 already landed does "the re-sync moved nothing in 2021-2024"
mean anything, which is why the D30-17 ladder runs this rung LAST.

For that control to be a control it must be impossible to satisfy by doing nothing, so:

* the expected delta was measured and committed BEFORE this script could re-sync it away, and
* an ``--apply`` against a ZERO divergence is REFUSED rather than reported as a success.

``slice_digests`` lives here rather than in the control's test module because the BEFORE and
AFTER hashes have to be produced by the same code to mean anything at all, and because it must
reuse ``scripts.fingerprint_gold._column_bytes`` -- the one canonical byte encoding in this
repository -- rather than becoming a second hash implementation that is free to drift from it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

from conf.settings import get_settings
from data.storage import load_dataframe, save_dataframe
from scripts.fingerprint_gold import GOLD_MATRICES, _column_bytes

# The holdout-bearing seasons SPEC R2's byte-identity clause is asserted over. A 2025 re-sync
# may move 2025. Anything it moves in these four means a whole-frame statistic is still
# reaching backwards, which is the WR-06 defect, and the phase is blocked.
SLICE_SEASONS: Final[tuple[int, ...]] = (2021, 2022, 2023, 2024)


def slice_digests(base_path: Path | None = None) -> dict[str, dict[str, str]]:
    """Return ``matrix -> column -> sha256`` over each matrix's 2021-2024 rows only.

    The slice is taken by SEASON, ordered by ``game_id``, and hashed with
    ``scripts.fingerprint_gold._column_bytes`` -- the same canonical byte encoding that
    wrote every fingerprint document in this phase. Reusing it is what makes the
    comparison exact rather than approximate: a float-representation difference is
    neither tolerated nor introduced, because no float is ever re-formatted.

    Ordering by ``game_id`` alone is a TOTAL order here, and that is asserted rather than
    assumed -- a duplicated id would make the sort non-deterministic and the digests
    meaningless. There is no positional comparison anywhere: the ``game_id`` column is
    itself one of the hashed columns, so a slice whose membership or ordering moved shows
    up as a moved ``game_id`` digest and is diagnosable rather than merely wrong.
    """
    root = (
        Path(base_path)
        if base_path is not None
        else Path(get_settings().config.data.root_path)
    )
    digests: dict[str, dict[str, str]] = {}
    for matrix in GOLD_MATRICES:
        path = root / "gold" / f"{matrix}.parquet"
        frame = pd.read_parquet(path, engine="pyarrow")
        # cast: pandas types a boolean-mask __getitem__ as Series | DataFrame (a
        # duplicated label could yield a frame). Gold carries no duplicate labels.
        rows = cast("pd.DataFrame", frame[frame["season"].isin(SLICE_SEASONS)])
        subset = rows.sort_values("game_id")
        duplicated = int(subset["game_id"].duplicated().sum())
        if duplicated:
            raise ValueError(
                f"{matrix} carries {duplicated} duplicated game_id values in its "
                f"{SLICE_SEASONS[0]}-{SLICE_SEASONS[-1]} slice, so sorting by game_id is "
                "not a total order and the slice digests would not be reproducible"
            )
        digests[matrix] = {
            str(column): hashlib.sha256(_column_bytes(series)).hexdigest()
            for column, series in subset.items()
        }
    return digests


def table_content_digest(frame: pd.DataFrame) -> str:
    """Return a sha256 over *frame*'s content, independent of row and column order.

    Used to prove that a SECOND apply is a content no-op. Row order is normalized by
    sorting on ``game_id`` and column order by sorting the column names, so a re-write
    that reproduces the same table is recognised as such even if DuckDB hands the columns
    back in a different order.
    """
    ordered = frame.sort_values("game_id")
    digest = hashlib.sha256()
    for column in sorted(str(name) for name in ordered.columns):
        digest.update(column.encode("utf-8"))
        digest.update(_column_bytes(cast("pd.Series", ordered[column])))
    return digest.hexdigest()


def measure_divergence(loader=load_dataframe) -> dict[str, Any]:
    """Measure the parquet-versus-DuckDB divergence of silver ``games``.

    Read through ``data.storage.load_dataframe`` with explicit ``source=`` literals rather
    than off the files, because the question is what the PIPELINE sees, and the pipeline
    reads through that function. Note the literal is ``"db"``; ``"duckdb"`` raises
    ``ValueError`` in ``load_dataframe``.
    """
    parquet = loader("games", layer="silver", source="parquet")
    database = loader("games", layer="silver", source="db")

    parquet_ids = set(parquet["game_id"])
    db_ids = set(database["game_id"])
    only_in_parquet = sorted(parquet_ids - db_ids)
    only_in_duckdb = sorted(db_ids - parquet_ids)

    missing_rows = parquet[parquet["game_id"].isin(only_in_parquet)]
    by_season = {
        str(int(season)): int(count)
        for season, count in sorted(missing_rows["season"].value_counts().items())
    }
    by_season_week = [
        {"season": int(season), "week": int(week), "rows": int(rows)}
        for (season, week), rows in sorted(
            missing_rows.groupby(["season", "week"]).size().items()
        )
    ]

    return {
        "parquet_rows": len(parquet),
        "db_rows": len(database),
        "divergence": len(parquet) - len(database),
        "only_in_parquet_count": len(only_in_parquet),
        "only_in_duckdb_count": len(only_in_duckdb),
        "missing_by_season": by_season,
        "missing_by_season_week": by_season_week,
        "missing_game_ids": only_in_parquet,
        "missing_game_ids_sha256": hashlib.sha256(
            "\n".join(only_in_parquet).encode("utf-8")
        ).hexdigest(),
        "only_in_duckdb_game_ids": only_in_duckdb,
    }


def resync_games() -> dict[str, Any]:
    """Replace the DuckDB copy of silver ``games`` with the authoritative parquet.

    ``save_to_parquet=False`` because the parquet is ALREADY correct: leaving it
    byte-untouched keeps the phase's hard-boundary hash manifest clean for this rung, and
    rewriting a correct file is a change nobody asked for.

    ``replace_mode=True`` because the passed frame IS the table. Replace mode skips the
    append-and-dedup merge entirely (``data/storage.py`` sets ``append_mode = False`` and
    ``partition_cols = None``), which is the correct semantic for "the parquet is the
    truth" and is what makes a second apply a content no-op rather than a re-merge.

    This function does NOT refuse a zero divergence -- ``main`` does. The refusal is a
    guard on the OPERATION being meaningful; the function itself is idempotent by
    construction and the committed idempotency test drives it directly to prove that.
    """
    before = measure_divergence()
    games = load_dataframe("games", layer="silver", source="parquet")

    save_dataframe(
        games,
        table_name="games",
        layer="silver",
        save_to_db=True,
        save_to_parquet=False,
        replace_mode=True,
    )

    after = measure_divergence()
    return {"before": before, "after": after}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (extracted so argument wiring is unit-testable)."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-sync the stale DuckDB copy of silver games from its authoritative "
            "parquet (N-01). Reports the divergence and writes nothing unless --apply."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually replace the DuckDB table. Without this flag the script is a DRY "
            "RUN: it measures and prints the divergence and writes nothing."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write the measured divergence JSON to this path (never under data/)",
    )
    return parser


def _print_divergence(label: str, measured: dict[str, Any]) -> None:
    print(f"{label}:")
    print(
        f"  parquet {measured['parquet_rows']}   db {measured['db_rows']}   "
        f"divergence {measured['divergence']}"
    )
    print(
        f"  only in parquet: {measured['only_in_parquet_count']}   "
        f"only in duckdb: {measured['only_in_duckdb_count']}"
    )
    if measured["missing_by_season"]:
        print(f"  missing by season: {measured['missing_by_season']}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    measured = measure_divergence()
    _print_divergence("silver games", measured)

    if not args.apply:
        print(
            "DRY RUN -- nothing was written. Pass --apply to replace the DuckDB copy."
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(measured, indent=2), encoding="utf-8")
            print(f"Wrote {args.out}")
        return 0

    if measured["divergence"] == 0:
        print(
            "REFUSING to apply: the measured divergence is ZERO. That means someone has "
            "already re-synced this lake, and SPEC R2's positive control now has nothing "
            "to prove -- a control that can be satisfied by doing nothing is not a "
            "control. The expected delta is pinned in tests/phase30_state.py as "
            "N01_DIVERGENCE_BEFORE and cannot be re-measured after the fact.",
            file=sys.stderr,
        )
        return 2

    result = resync_games()
    _print_divergence("after re-sync", result["after"])

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result["after"], indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")

    return 0 if result["after"]["divergence"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
