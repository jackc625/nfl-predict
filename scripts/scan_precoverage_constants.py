"""Re-scan the three gold matrices for pre-coverage constants (Plan 33.2-17 Task 3, SPEC R10).

WHY. Before each feature family's first covered season, gold carried a value nobody measured -- a
constant 0.0 with no flag -- and a model reads a centred 0.0 as "exactly average" (D33.2-08 item
2). This scanner classifies every MODEL INPUT column of every gold matrix by the first season in
which it takes more than one distinct value, and says whether the block of seasons before that
is honestly flagged:

* ``varies_from_corpus_start`` -- the column varies in the matrix's first season;
* ``flagged_precoverage``      -- it varies only from a later season, and its family carries a
  coverage flag in the matrix, so every unmeasured row of the block before it can say so;
* ``unflagged_precoverage``    -- it varies only from a later season, and the block before it is
  NOT flagged: the defect;
* ``never_varies``             -- it never takes two distinct values in any season. A different
  matter (a dead or held-constant column), reported separately and not counted as a block.

Every ``unflagged_precoverage`` column is ROUTED to the planned fix that closes it at rung 8 by
family; one that no planned fix closes is ``UNROUTED`` and must be fixed before rung 8 runs.

THE COVERAGE FLAG of a family is found by name, from the family the column belongs to
(:data:`FAMILY_FLAG_TEMPLATES`); a family with no flag column cannot flag a block.

Read-only: it reads gold and writes nothing. Printed lines (machine-read by the doc-drift guard):
``CLASSIFIED=``, ``UNFLAGGED=``, ``UNROUTED=``, ``CONSTANT_2002_2017=``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.temporal import _DEFAULT_ID_COLS
from utils.feature_columns import display_only_columns

#: The three gold matrices, in the order every report lists them.
GOLD_MATRICES: tuple[str, ...] = ("features_wp", "features_ats", "features_ou")

#: The four classes.
VARIES_FROM_START = "varies_from_corpus_start"
FLAGGED = "flagged_precoverage"
UNFLAGGED = "unflagged_precoverage"
NEVER_VARIES = "never_varies"
CLASSES: tuple[str, ...] = (VARIES_FROM_START, FLAGGED, UNFLAGGED, NEVER_VARIES)

#: The span the RULE_EVIDENCE census figure names ("constant over 2002-2017"). A REPORTING
#: window fixed by that figure's wording -- so this scan's number is comparable with the one it
#: replaces -- and not a coverage floor: nothing is included or excluded by it.
CENSUS_WINDOW: tuple[int, int] = (2002, 2017)

#: A column's family, by name. Checked in order; the first match wins.
FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("opp_adj", re.compile(r"^(home|away)_(off|def)_rolling_opp_adj_")),
    (
        "snap",
        re.compile(
            r"^(home|away)_(rolling_snap_share_|snap_concentration|snap_continuity)"
        ),
    ),
    ("injury", re.compile(r"^(home|away)_(qb_out_flag|backup_quality_delta)$")),
    ("availability", re.compile(r"^(home|away)_availability_fraction$")),
    # The one team-form metric the pinned play-by-play does not carry in every season
    # (features.team_form.SOURCE_LIMITED_ROLLING_COLUMNS): its own flag, its own route.
    ("team_form_source_limited", re.compile(r"^(home|away)_off_rolling_cpoe$")),
    ("team_form", re.compile(r"^(home|away)_(off|def)_rolling_")),
    ("market", re.compile(r"^(snapshot_|spread_movement|total_movement)")),
)

#: Each family's coverage flag, as a template over the column's own name parts.
FAMILY_FLAG_TEMPLATES: dict[str, str] = {
    "opp_adj": "{side}_{unit}_rolling_opp_adj_coverage",
    "snap": "{side}_snap_coverage",
    "injury": "{side}_injury_coverage",
    # Availability is weighted by prior SNAP shares; its flag says whether those exist.
    "availability": "{side}_availability_coverage",
    "team_form_source_limited": "{side}_off_rolling_cpoe_coverage",
}

#: Where each family's unflagged pre-coverage block is closed at rung 8.
FAMILY_ROUTES: dict[str, str] = {
    "team_form": (
        "Plan 33.2-17 Task 1 -- silver team form computed for 2002-2026 from the pinned "
        "play-by-play; reaches gold at Plan 33.2-18's rung 8"
    ),
    "opp_adj": (
        "Plan 33.2-18 Task 1 -- the selection-window floor move carries the per-game pool to "
        "2002 through the identity binding; reaches gold at rung 8"
    ),
    "snap": (
        "Plan 33.2-17 Task 2 -- snap_coverage added; NaN with the flag false before the first "
        "covered season; reaches gold at rung 8"
    ),
    "injury": (
        "Plan 33.2-17 Task 2 -- NaN with the injury flags false wherever nothing was admitted; "
        "reaches gold at rung 8"
    ),
    "availability": (
        "Plan 33.2-17 Task 2 -- NaN with the injury flags false wherever nothing was admitted; "
        "reaches gold at rung 8"
    ),
    "team_form_source_limited": (
        "Plan 33.2-17 Task 1 (values from 2006, where the pinned play-by-play carries cpoe) "
        "AND Task 2 (NaN beside rolling_cpoe_coverage 0.0 for 2002-2005, where it does not); "
        "reaches gold at rung 8"
    ),
}


@dataclass
class ColumnScan:
    """One column's classification in one matrix."""

    matrix: str
    column: str
    family: str | None
    first_varying_season: int | None
    klass: str
    flag_column: str | None
    route: str | None
    block_seasons: list[int] = field(default_factory=list)


def model_input_columns(frame: pd.DataFrame) -> list[str]:
    """The columns a fit can select: numeric, not an id / target / display column.

    The same exclusion ``models.temporal.WalkForwardSplitter._feature_cols`` applies, plus the
    ``*_coverage`` flags, which ARE the statements this scan checks for and are classified as
    flags, not as blocks.
    """
    excluded = set(_DEFAULT_ID_COLS) | display_only_columns()
    numeric = frame.select_dtypes(include=["number"]).columns
    return [c for c in numeric if c not in excluded and not c.endswith("_coverage")]


def column_family(column: str) -> str | None:
    """The family *column* belongs to, or None."""
    for family, pattern in FAMILY_PATTERNS:
        if pattern.match(column):
            return family
    return None


def family_flag_column(column: str, family: str | None) -> str | None:
    """The coverage flag that states *column*'s absence, by name, or None."""
    template = FAMILY_FLAG_TEMPLATES.get(family or "")
    if template is None:
        return None
    side = column.split("_", 1)[0]
    unit_match = re.match(r"^(home|away)_(off|def)_", column)
    unit = unit_match.group(2) if unit_match else ""
    return template.format(side=side, unit=unit)


def _first_varying_season(
    frame: pd.DataFrame, column: str, seasons: Iterable[int]
) -> int | None:
    for season in seasons:
        if frame.loc[frame["season"] == season, column].nunique(dropna=True) > 1:
            return int(season)
    return None


def scan_matrix(matrix: str, frame: pd.DataFrame) -> list[ColumnScan]:
    """Classify every model-input column of one gold matrix."""
    seasons = sorted(int(s) for s in frame["season"].dropna().unique())
    results: list[ColumnScan] = []
    for column in model_input_columns(frame):
        family = column_family(column)
        first = _first_varying_season(frame, column, seasons)
        flag = family_flag_column(column, family)
        if first is None:
            klass, block = NEVER_VARIES, []
        elif first == seasons[0]:
            klass, block = VARIES_FROM_START, []
        else:
            block = [s for s in seasons if s < first]
            # A block is FLAGGED when its family's coverage flag is present: an unmeasured row
            # then says so. (A row the flag marks covered carries a measured value, which may
            # legitimately not vary -- e.g. 2009's handful of dated injury reports.)
            klass = FLAGGED if flag is not None and flag in frame.columns else UNFLAGGED
        route = FAMILY_ROUTES.get(family or "") if klass == UNFLAGGED else None
        results.append(
            ColumnScan(matrix, column, family, first, klass, flag, route, block)
        )
    return results


def constant_over_census_window(frame: pd.DataFrame) -> list[str]:
    """Model-input columns holding ONE value (NaN counted as a value) over every census season."""
    first, last = CENSUS_WINDOW
    window = frame.loc[frame["season"].between(first, last)]
    return sorted(
        column
        for column in model_input_columns(frame)
        if window[column].nunique(dropna=False) == 1
    )


def scan_gold(gold_dir: Path) -> dict[str, object]:
    """Scan every gold matrix under *gold_dir*; the report the CLI prints."""
    scans: list[ColumnScan] = []
    constant: set[str] = set()
    for matrix in GOLD_MATRICES:
        frame = pd.read_parquet(gold_dir / f"{matrix}.parquet")
        scans.extend(scan_matrix(matrix, frame))
        constant |= set(constant_over_census_window(frame))
    unflagged = [s for s in scans if s.klass == UNFLAGGED]
    return {
        "scans": scans,
        "classified": len(scans),
        "unflagged": unflagged,
        "unrouted": [s for s in unflagged if s.route is None],
        "constant_census": sorted(constant),
        "counts": {
            matrix: {
                k: sum(1 for s in scans if s.matrix == matrix and s.klass == k)
                for k in CLASSES
            }
            for matrix in GOLD_MATRICES
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Print the scan. Exit 0; the verdict is read from the printed lines."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold-dir", type=Path, default=Path("data/gold"))
    parser.add_argument("--verbose", action="store_true", help="list every column")
    args = parser.parse_args(argv)

    report = scan_gold(args.gold_dir)
    for matrix, counts in report["counts"].items():  # type: ignore[union-attr]
        print(f"{matrix}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    for scan in report["unflagged"]:  # type: ignore[union-attr]
        print(
            f"UNFLAGGED_BLOCK {scan.matrix} {scan.column} family={scan.family} "
            f"first_varies={scan.first_varying_season} route={scan.route or 'NONE'}"
        )
    if args.verbose:
        for scan in report["scans"]:  # type: ignore[union-attr]
            print(
                f"COLUMN {scan.matrix} {scan.column} {scan.klass} {scan.first_varying_season}"
            )
    print(f"CLASSIFIED= {report['classified']}")
    print(f"UNFLAGGED= {len(report['unflagged'])}")  # type: ignore[arg-type]
    print(f"UNROUTED= {len(report['unrouted'])}")  # type: ignore[arg-type]
    print(f"CONSTANT_2002_2017= {len(report['constant_census'])}")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
