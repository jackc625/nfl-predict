"""Carry out the owner's ruling on the 2026 rows predicted under the OLD rule (Plan 33.2-26 Task 4).

Run via:
    uv run python -m scripts.dispose_old_rule_predictions --ruling <ruling>            # dry run
    uv run python -m scripts.dispose_old_rule_predictions --ruling <ruling> --apply    # act

WHY THIS EXISTS
---------------
Plan 33.2-26 superseded the frozen 2026 bet rule (``11761c7``) and repointed the live rule in one
corrective commit. Any 2026 prediction or bet row produced BEFORE that commit was made under the
old models and the old rule. What happens to those rows was the one question the phase could not
settle from the record, so it was put to the owner as three competing fixes, and this program
executes whichever one was ruled -- a checkpoint whose options only one of which could be carried
out would not have been a real choice.

THE THREE BRANCHES -- AND NO DEFAULT
------------------------------------
* ``delete-and-repredict``: every 2026 prediction file is removed, and every 2026 row is removed
  from the durable bet list (its tracker is re-aggregated from what remains). The next scheduled
  run predicts those games again under the corrected rule.
* ``keep-and-supersede``: the same artifacts are MOVED into a ``superseded/`` directory beside
  them, with ONE marker file recording the ruling, its date, both corrective shas and the reason.
  Nothing is destroyed.
* ``keep-and-label``: nothing is written. The no-change decision is still a checked outcome: the
  caller brackets the run over ``outputs/`` and asserts the digest comparison is empty.

A MISSING ruling and an UNRECOGNISED ruling are two distinct named refusals
(:class:`AbsentRulingError`, :class:`UnrecognisedRulingError`), never a fallback branch. A program
that silently picked a branch would reproduce, in code, the defect the checkpoint exists to
prevent.

SCOPE: BY SEASON, THROUGH THE ONE MODULE THAT DECIDES WHERE THE FILES LIVE
--------------------------------------------------------------------------
The files are ``predictions_2026_week*.csv`` under ``pipeline.steps._predictions_output_dir()``
and the 2026 rows of the durable bet list under ``pipeline.steps._bet_list_output_dir()``, both
resolved through those helpers rather than re-typed: one module already decides where these
artifacts live, and a second copy of that decision is how a disposition cleans the wrong
directory. A 2025 artifact in the same directory is never touched. Scope is by SEASON, never by
file time.

It is a ONE-TIME disposition, bound to the repoint. Every 2026 row that exists when it is run was
produced before the corrective commit, because nothing in the tree after that commit produces an
old-rule row. Run later, it would also match rows the corrected rule produced, which the ruling
does not cover -- so it is run at the repoint, bracketed, and recorded, not scheduled.

THE WEB CACHE IS NOT TOUCHED, ON PURPOSE
----------------------------------------
"Also clean the cache" is the obvious-looking next step and it is wrong. ``data/web_cache.duckdb``
is DERIVED: ``pipeline.steps.step_populate_web_cache`` rebuilds it in full from the durable
artifacts and swaps the whole file atomically, so the cache follows the artifacts by construction.
Hand-editing it would create a second answer the next population run silently overwrites. Nothing
under ``data/`` or ``artifacts/`` is written by any branch.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

RULING_DELETE = "delete-and-repredict"
RULING_SUPERSEDE = "keep-and-supersede"
RULING_LABEL = "keep-and-label"
RULINGS: tuple[str, ...] = (RULING_DELETE, RULING_SUPERSEDE, RULING_LABEL)

#: The season whose old-rule rows the ruling covers.
DISPOSED_SEASON = 2026

#: Where a superseded artifact is moved, beside the directory it came from.
SUPERSEDED_DIR_NAME = "superseded"
MARKER_NAME = "SUPERSEDED.json"
SUPERSEDED_BET_ROWS_NAME = "bet_list_{season}_old_rule.parquet"

REASON = (
    "Produced before the corrective commit that superseded the frozen 2026 bet rule (11761c7) "
    "and the EV chain (ee20773): these rows came from the retired models and the old rule."
)


class AbsentRulingError(Exception):
    """No ruling was given. The owner's answer selects the branch; there is no default."""


class UnrecognisedRulingError(Exception):
    """The ruling is not one of the three the owner was offered. It is refused, not guessed."""


def require_ruling(ruling: str | None) -> str:
    """The ruling, verbatim, or a named refusal -- absent and unrecognised are different failures.

    Raises:
        AbsentRulingError: *ruling* is None or blank.
        UnrecognisedRulingError: *ruling* is not in :data:`RULINGS`.
    """
    if ruling is None or not ruling.strip():
        msg = (
            "no ruling was given; the owner's answer selects the branch and there is NO default. "
            f"Pass --ruling with one of {list(RULINGS)}."
        )
        raise AbsentRulingError(msg)
    if ruling not in RULINGS:
        msg = f"unrecognised ruling {ruling!r}; the owner was offered exactly {list(RULINGS)}."
        raise UnrecognisedRulingError(msg)
    return ruling


@dataclass(frozen=True)
class Disposition:
    """What one ruling does to the store, enumerated file by file BEFORE anything happens.

    Attributes:
        ruling: The owner's ruling, verbatim.
        prediction_files: The season's prediction CSVs.
        bet_rows: The season's rows in the durable bet list (0 when there are none).
        touched: Every path this disposition writes, moves or removes, in order. The dry run
            prints exactly this list and ``--apply`` touches exactly this list.
    """

    ruling: str
    predictions_dir: Path
    bet_list_dir: Path
    prediction_files: tuple[Path, ...]
    bet_rows: int
    touched: tuple[Path, ...] = field(default_factory=tuple)


def _season_prediction_files(predictions_dir: Path, season: int) -> tuple[Path, ...]:
    return tuple(sorted(Path(predictions_dir).glob(f"predictions_{season}_week*.csv")))


def _season_bet_rows(bet_list_dir: Path, season: int) -> int:
    from backtest.weekly_bet_list import read_bet_list_artifact

    stored = read_bet_list_artifact(bet_list_dir)
    return int((stored["season"].astype(int) == season).sum()) if len(stored) else 0


def plan_disposition(
    ruling: str | None,
    predictions_dir: Path,
    bet_list_dir: Path,
    season: int = DISPOSED_SEASON,
) -> Disposition:
    """Enumerate what *ruling* would do, touching nothing.

    Raises:
        AbsentRulingError / UnrecognisedRulingError: see :func:`require_ruling`.
    """
    from backtest.weekly_bet_list import (
        BET_LIST_ARTIFACT_NAME,
        BET_TRACKER_ARTIFACT_NAME,
    )

    chosen = require_ruling(ruling)
    files = _season_prediction_files(predictions_dir, season)
    rows = _season_bet_rows(bet_list_dir, season)

    touched: list[Path] = []
    if chosen != RULING_LABEL:
        superseded = Path(predictions_dir) / SUPERSEDED_DIR_NAME
        for path in files:
            touched.append(path)
            if chosen == RULING_SUPERSEDE:
                touched.append(superseded / path.name)
        if rows:
            touched.append(Path(bet_list_dir) / BET_LIST_ARTIFACT_NAME)
            touched.append(Path(bet_list_dir) / BET_TRACKER_ARTIFACT_NAME)
            if chosen == RULING_SUPERSEDE:
                touched.append(
                    Path(bet_list_dir)
                    / SUPERSEDED_DIR_NAME
                    / SUPERSEDED_BET_ROWS_NAME.format(season=season)
                )
        if chosen == RULING_SUPERSEDE and (files or rows):
            touched.append(superseded / MARKER_NAME)
    return Disposition(
        ruling=chosen,
        predictions_dir=Path(predictions_dir),
        bet_list_dir=Path(bet_list_dir),
        prediction_files=files,
        bet_rows=rows,
        touched=tuple(touched),
    )


def _rewrite_bet_list(bet_list_dir: Path, season: int) -> pd.DataFrame:
    """Rewrite the durable pair WITHOUT *season*'s rows; return the rows taken out."""
    from backtest.bet_tracker import aggregate_all_blocks, to_tracker_frame
    from backtest.weekly_bet_list import (
        read_bet_list_artifact,
        write_bet_list_artifact,
        write_bet_tracker_artifact,
    )

    stored = read_bet_list_artifact(bet_list_dir)
    in_season = stored["season"].astype(int) == season
    kept = stored.loc[~in_season].reset_index(drop=True)
    write_bet_list_artifact(kept, bet_list_dir)
    write_bet_tracker_artifact(
        to_tracker_frame(aggregate_all_blocks(kept)), output_dir=bet_list_dir
    )
    return stored.loc[in_season].reset_index(drop=True)


def apply_disposition(
    disposition: Disposition,
    *,
    decided_on: str,
    season: int = DISPOSED_SEASON,
) -> tuple[Path, ...]:
    """Carry out *disposition* and return the paths it touched -- exactly ``disposition.touched``."""
    if disposition.ruling == RULING_LABEL:
        return ()

    superseded = disposition.predictions_dir / SUPERSEDED_DIR_NAME
    for path in disposition.prediction_files:
        if disposition.ruling == RULING_DELETE:
            path.unlink()
        else:
            superseded.mkdir(parents=True, exist_ok=True)
            path.replace(superseded / path.name)

    if disposition.bet_rows:
        removed = _rewrite_bet_list(disposition.bet_list_dir, season)
        if disposition.ruling == RULING_SUPERSEDE:
            target = (
                disposition.bet_list_dir
                / SUPERSEDED_DIR_NAME
                / SUPERSEDED_BET_ROWS_NAME.format(season=season)
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            removed.to_parquet(target, index=False)

    if disposition.ruling == RULING_SUPERSEDE and (
        disposition.prediction_files or disposition.bet_rows
    ):
        from tests.phase33_state import (
            COLD_START_CORRECTIVE_COMMIT_SHA,
            EV_CHAIN_CORRECTIVE_COMMIT_SHA,
        )

        marker = {
            "ruling": disposition.ruling,
            "ruled_on": decided_on,
            "season": season,
            "corrective_commits": {
                "cold_start_11761c7": COLD_START_CORRECTIVE_COMMIT_SHA,
                "ev_chain_ee20773": EV_CHAIN_CORRECTIVE_COMMIT_SHA,
            },
            "reason": REASON,
            "moved_prediction_files": [p.name for p in disposition.prediction_files],
            "moved_bet_rows": disposition.bet_rows,
        }
        superseded.mkdir(parents=True, exist_ok=True)
        (superseded / MARKER_NAME).write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
    return disposition.touched


def main(argv: Sequence[str] | None = None) -> int:
    """Print the enumerated disposition; with ``--apply``, carry it out."""
    from pipeline.steps import _bet_list_output_dir, _predictions_output_dir

    parser = argparse.ArgumentParser(
        prog="dispose_old_rule_predictions",
        description="Carry out the owner's ruling on the old-rule 2026 rows (Plan 33.2-26).",
    )
    parser.add_argument("--ruling", default=None, help=f"one of {list(RULINGS)}")
    parser.add_argument(
        "--apply", action="store_true", help="act; the default is a dry run"
    )
    parser.add_argument(
        "--ruled-on",
        default=datetime.now(tz=ZoneInfo("America/New_York")).date().isoformat(),
        help="the date the owner gave the ruling (recorded in the superseded marker)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    disposition = plan_disposition(
        args.ruling, _predictions_output_dir(), _bet_list_output_dir()
    )
    touched = [p.as_posix() for p in disposition.touched]
    print(f"RULING= {disposition.ruling}")
    print(f"SEASON= {DISPOSED_SEASON}")
    print(f"PREDICTION_FILES= {[p.as_posix() for p in disposition.prediction_files]}")
    print(f"BET_ROWS= {disposition.bet_rows}")
    if not args.apply:
        print(f"WOULD_TOUCH= {touched}")
        return 0
    done = apply_disposition(disposition, decided_on=args.ruled_on)
    print(f"TOUCHED= {[p.as_posix() for p in done]}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
