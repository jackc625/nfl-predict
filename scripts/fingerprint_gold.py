"""Fingerprint the gold feature matrices per column, per season (D-Q5).

A full gold rebuild reaches nflreadpy LIVE (``features/team_form.py:35,118``;
``features/qb_tracking.py:710,776``) with no cache configured, so an upstream
play-by-play or depth-chart revision lands in the same artifact as whatever
change the rebuild was actually run for. Without a control, that upstream drift
is silently absorbed and misattributed.

This tool turns that invisible confound into a recorded observation. Run it
BEFORE a rebuild, run it again AFTER, then ``--compare BEFORE AFTER`` to get the
exact list of columns that moved and the seasons in which each moved.

It is strictly read-only with respect to ``data/`` -- the JSON output must be
written somewhere else (the quick task keeps it under the phase's gitignored
``artifacts/`` directory).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from conf.settings import get_settings

GOLD_MATRICES = ("features_wp", "features_ats", "features_ou")


def _column_bytes(series: pd.Series) -> bytes:
    """Return a deterministic byte encoding of *series* values.

    Numeric columns hash their raw IEEE-754 / integer representation, which is
    exact; anything else falls back to a repr-joined string so the encoding is
    still stable across runs.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype="uint8").tobytes()
    if pd.api.types.is_integer_dtype(series):
        return series.to_numpy(dtype="int64").tobytes()
    if pd.api.types.is_float_dtype(series):
        return series.to_numpy(dtype="float64").tobytes()
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("int64").to_numpy(dtype="int64").tobytes()
    return "\x1f".join(repr(value) for value in series.to_numpy()).encode("utf-8")


def fingerprint_matrix(df: pd.DataFrame) -> dict:
    """Return per-column, per-season hashes plus shape metadata for *df*.

    Each cell hashes the column's values ordered by ``game_id`` within the
    season, so a row-order change alone never registers as drift.
    """
    seasons = sorted(int(season) for season in df["season"].dropna().unique())
    columns: dict[str, dict[str, str]] = {}

    ordered = df.sort_values("game_id")
    for season in seasons:
        subset = ordered[ordered["season"] == season]
        for column in subset.columns:
            digest = hashlib.sha256(_column_bytes(subset[column])).hexdigest()[:16]
            columns.setdefault(column, {})[str(season)] = digest

    return {
        "rows": len(df),
        "width": int(df.shape[1]),
        "seasons": seasons,
        "rows_per_season": {
            str(season): int((df["season"] == season).sum()) for season in seasons
        },
        "columns": columns,
    }


def fingerprint_gold(base_path: Path | None = None) -> dict:
    """Fingerprint every gold feature matrix."""
    root = (
        Path(base_path)
        if base_path is not None
        else Path(get_settings().config.data.root_path)
    )
    result: dict[str, dict] = {}
    for matrix in GOLD_MATRICES:
        path = root / "gold" / f"{matrix}.parquet"
        if not path.exists():
            result[matrix] = {"missing": True}
            continue
        result[matrix] = fingerprint_matrix(pd.read_parquet(path, engine="pyarrow"))
    return result


def compare_fingerprints(before: dict, after: dict) -> dict:
    """Return, per matrix, the columns whose hash moved and in which seasons."""
    report: dict[str, dict] = {}
    for matrix in GOLD_MATRICES:
        b = before.get(matrix, {})
        a = after.get(matrix, {})
        b_cols = b.get("columns", {})
        a_cols = a.get("columns", {})

        changed: dict[str, list[str]] = {}
        for column in sorted(set(b_cols) & set(a_cols)):
            seasons = [
                season
                for season in sorted(set(b_cols[column]) | set(a_cols[column]))
                if b_cols[column].get(season) != a_cols[column].get(season)
            ]
            if seasons:
                changed[column] = seasons

        report[matrix] = {
            "width_before": b.get("width"),
            "width_after": a.get("width"),
            "rows_before": b.get("rows"),
            "rows_after": a.get("rows"),
            "columns_added": sorted(set(a_cols) - set(b_cols)),
            "columns_removed": sorted(set(b_cols) - set(a_cols)),
            "columns_changed": changed,
        }
    return report


def _print_comparison(report: dict) -> None:
    for matrix, detail in report.items():
        print(f"{matrix}:")
        print(
            f"  width {detail['width_before']} -> {detail['width_after']}   "
            f"rows {detail['rows_before']} -> {detail['rows_after']}"
        )
        if detail["columns_added"]:
            print(f"  columns ADDED:   {detail['columns_added']}")
        if detail["columns_removed"]:
            print(f"  columns REMOVED: {detail['columns_removed']}")
        changed = detail["columns_changed"]
        print(f"  columns CHANGED: {len(changed)}")
        for column, seasons in changed.items():
            print(f"    {column}: seasons {','.join(seasons)}")
        print()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (extracted so argument wiring is unit-testable)."""
    parser = argparse.ArgumentParser(
        description="Fingerprint gold feature matrices per column, per season"
    )
    parser.add_argument(
        "--out", type=Path, help="Write the fingerprint JSON to this path"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BEFORE", "AFTER"),
        help="Compare two previously written fingerprint JSON documents",
    )
    return parser


def main() -> None:
    """CLI entry point for gold fingerprinting."""
    args = build_parser().parse_args()

    if args.compare:
        before = json.loads(args.compare[0].read_text(encoding="utf-8"))
        after = json.loads(args.compare[1].read_text(encoding="utf-8"))
        report = compare_fingerprints(before, after)
        _print_comparison(report)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return

    if args.out is None:
        print("ERROR: --out is required unless --compare is given", file=sys.stderr)
        sys.exit(2)

    fingerprint = fingerprint_gold()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    for matrix, detail in fingerprint.items():
        if detail.get("missing"):
            print(f"{matrix}: MISSING")
            continue
        print(
            f"{matrix}: rows={detail['rows']} width={detail['width']} "
            f"seasons={detail['seasons'][0]}-{detail['seasons'][-1]}"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
