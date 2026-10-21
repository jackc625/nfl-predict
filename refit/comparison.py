"""The pre-registered comparison of the live arm against the frozen arm (Phase 37; SPEC R12, RFIT-08).

THE RULE IS ROLLING-REFIT-PREREGISTRATION.md SECTION 11
-------------------------------------------------------
Every constant this module decides with -- the log-loss clip, the minimum count, alpha, the
roll-up words, the season -- is imported from ``backtest.refit_preregistration_2026``, never
restated. The deciding run happens ONCE, at season close (Phase 38); this module is runnable at any
time so the command can be exercised before then, and the Season page shows only raw numbers.

WHAT IS SCORED (D-18)
---------------------
Each arm's OWN stored model prediction: the ledger's immutable ``model_value`` (WP ``model_prob``,
ATS ``model_spread``, O/U ``model_total``), never the market-blended value. The row-19 blend gives
WP and ATS a model weight of 0, so scoring the blend would compare two copies of the market.

THE PER-GAME LOSS AND ITS SIGN CONVENTION (D-07, D-11, D-31)
-----------------------------------------------------------
* WP: log loss of the home-win probability, clipped to ``[LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP]``,
  against y = 1 when the home score is greater than the away score and 0 otherwise -- a tie is 0,
  the gold label formula of ``forward_ledger.settle.realized_values_from_scores``.
* ATS: ``|model_value - (home - away)|``. ``model_spread`` IS a predicted home margin (positive when
  the home team is favoured), so there is no sign flip; the comparison never touches a line.
* O/U: ``|model_value - (home + away)|``.

Per game both arms predicted, gap = live loss minus frozen loss, so a NEGATIVE gap means the live
(re-fitted) arm was more accurate. A game belongs to the generation of its LIVE row
(``refit.generations.generation_of_row``); a shadow row is always generation 0.

WHAT IS A PAIR (R8)
-------------------
:func:`paired_frame` reads the ledger week by week through ``refit.aa.week_pairs``. In this order:

1. A live row whose stamped model id maps to no generation (weeks 3-4, NULL stamps) never enters
   the comparison (``excluded["no_generation"]``).
2. A week with live rows but no shadow row at all (before the shadow start, or a lost slate) is
   excluded and listed by week (``unpaired_weeks``).
3. A live row of a week that has shadow rows but no twin of its own is a hole, never a pair.
4. A pair whose twins disagree on a comparability field (``odds_snapshot_digest``,
   ``gold_generation_key``, ``decided_at_utc``) is not comparable
   (``excluded["invariant_failure"]``).
5. A pair with no stored model value (``no_model_value``) or whose game has no final score
   (``no_final_score``) is excluded and counted.

Each pair also carries both arms' stored ``status`` and their IN-FORCE grading through
``forward_ledger.corrections.in_force_outcomes``, so a grading correction is honoured by the
report-only ROI figure.

THE TEST (D-07, D-09)
---------------------
:func:`paired_test` is the one per-cell wrapper: below ``MIN_PAIRED_GAMES`` paired games it reports
the count and "too few to test" with no p-value; when every gap is exactly zero it reports no
difference with no p-value (scipy would answer NaN); otherwise it calls
``scipy.stats.ttest_1samp(gaps, 0.0)`` -- the Diebold-Mariano test in its independent-games form,
two-sided. Nothing is resampled, so no block can cross a generation boundary, and no test ever pools
two generations.

CLV APPEARS NOWHERE (must-NOT P5). No frame column, result field or printed line carries a CLV
value; the deciding figure is accuracy only.

Batch code: the comparison CLI (``scripts/compare_refit_arms.py``) and the cache build import it.
No module under ``api/`` may import it, so the request path never loads it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
from scipy import stats

from api.cache import GRADING_STATUS_PENDING
from backtest.refit_preregistration_2026 import (
    LOG_LOSS_CLIP,
    MIN_PAIRED_GAMES,
    REFIT_SEASON,
)
from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.corrections import InForce, in_force_outcomes
from forward_ledger.schema import ARM_LIVE, ARM_SHADOW
from forward_ledger.settle import realized_values_from_scores
from forward_ledger.store import LedgerEntry
from refit.aa import week_pairs
from refit.generations import Generation, generation_of_row

__all__ = [
    "EXCLUDED_REASONS",
    "FRAME_COLUMNS",
    "STATUS_ALL_ZERO",
    "STATUS_TESTED",
    "STATUS_TOO_FEW",
    "TARGETS",
    "PairedFrame",
    "cell_gaps",
    "line_loss",
    "paired_frame",
    "paired_test",
    "wp_loss",
]

# The three targets, in the order every report lists them.
TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# The per-cell statuses :func:`paired_test` answers with.
STATUS_TESTED: str = "tested"
STATUS_TOO_FEW: str = "too few to test"
STATUS_ALL_ZERO: str = "no difference (all paired gaps zero)"

# Why a live row of a paired week left the comparison, in the order the checks run.
EXCLUDED_REASONS: tuple[str, ...] = (
    "no_generation",
    "invariant_failure",
    "no_model_value",
    "no_final_score",
)

# The paired frame's columns: one row per comparable pair.
FRAME_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "target",
    "generation",
    "live_value",
    "frozen_value",
    "home_score",
    "away_score",
    "live_loss",
    "frozen_loss",
    "gap",
    "live_status",
    "live_grading_status",
    "live_payout_flat",
    "frozen_status",
    "frozen_grading_status",
    "frozen_payout_flat",
)

# The ledger row key without its arm: (game_id, season, week, target).
_TWIN_KEY: tuple[str, ...] = ("game_id", "season", "week", "target")


@dataclass(frozen=True)
class PairedFrame:
    """The comparable pairs of the season, and what was left out.

    Attributes:
        frame: One row per comparable pair, :data:`FRAME_COLUMNS`, in week then file order.
        holes: The twin keys ``(game_id, season, week, target)`` of live rows, in weeks that have
            shadow rows, with no shadow twin.
        unpaired_weeks: The weeks with live rows in a generation but no shadow row at all.
        excluded: :data:`EXCLUDED_REASONS` -> how many live rows each reason removed.
    """

    frame: pd.DataFrame
    holes: tuple[tuple[Any, ...], ...]
    unpaired_weeks: tuple[int, ...]
    excluded: dict[str, int]


def wp_loss(p: float, y: int) -> float:
    """The log loss of home-win probability *p* against the label *y* (1 home win, else 0).

    *p* is clipped to ``[LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP]`` first, so a certain wrong call costs a
    large finite loss instead of infinity.
    """
    clipped = min(max(float(p), LOG_LOSS_CLIP), 1.0 - LOG_LOSS_CLIP)
    return -math.log(clipped) if y == 1 else -math.log(1.0 - clipped)


def line_loss(value: float, realized: float) -> float:
    """The absolute error in points of a predicted margin or total against the final one."""
    return abs(float(value) - float(realized))


def _loss(target: str, value: float, realized: float) -> float:
    if target == "wp":
        return wp_loss(value, int(realized))
    if target in ("ats", "ou"):
        return line_loss(value, realized)
    msg = f"no per-game loss is registered for target {target!r}"
    raise ValueError(msg)


def paired_test(gaps: Sequence[float]) -> dict[str, Any]:
    """The pre-registered test of one cell's paired gaps (D-07, D-09).

    Returns:
        ``{n, mean_gap, t, p, status}``: ``status`` is :data:`STATUS_TOO_FEW` below
        ``MIN_PAIRED_GAMES`` games, :data:`STATUS_ALL_ZERO` when every gap is exactly zero (both
        with ``t`` and ``p`` None), else :data:`STATUS_TESTED` with the two-sided
        ``scipy.stats.ttest_1samp(gaps, 0.0)`` statistic and p-value. ``mean_gap`` is None only
        when there is no gap at all.
    """
    values = [float(gap) for gap in gaps]
    n = len(values)
    result: dict[str, Any] = {
        "n": n,
        "mean_gap": math.fsum(values) / n if n else None,
        "t": None,
        "p": None,
    }
    if n < MIN_PAIRED_GAMES:
        return {**result, "status": STATUS_TOO_FEW}
    if all(gap == 0.0 for gap in values):
        return {**result, "status": STATUS_ALL_ZERO}
    # scipy's stub types the result tuple generically; at run time both items are numpy floats.
    t_stat, p_value = cast("tuple[Any, Any]", stats.ttest_1samp(values, 0.0))
    return {**result, "t": float(t_stat), "p": float(p_value), "status": STATUS_TESTED}


def _final_scores(games: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """``game_id -> (home score, away score)`` for every game with both scores recorded."""
    scores: dict[str, tuple[int, int]] = {}
    for game_id, home, away in zip(
        games["game_id"].tolist(),
        games["home_score"].tolist(),
        games["away_score"].tolist(),
        strict=True,
    ):
        if home is None or away is None or pd.isna(home) or pd.isna(away):
            continue
        scores[str(game_id)] = (int(home), int(away))
    return scores


def _arm_columns(prefix: str, status: Any, in_force: InForce | None) -> dict[str, Any]:
    """One arm's stored status and in-force grading, as ``<prefix>_*`` frame columns."""
    return {
        f"{prefix}_status": status,
        f"{prefix}_grading_status": (
            in_force.grading_status if in_force is not None else GRADING_STATUS_PENDING
        ),
        f"{prefix}_payout_flat": in_force.payout_flat if in_force is not None else None,
    }


def paired_frame(
    entries: Sequence[LedgerEntry],
    games: pd.DataFrame,
    generations: Sequence[Generation],
) -> PairedFrame:
    """Every comparable (live, frozen) pair of the season, with each arm's loss and the gap.

    Args:
        entries: The ledger, in file order (``forward_ledger.store.read_entries``).
        games: A silver-``games``-shaped frame (``game_id``, ``home_score``, ``away_score``).
        generations: The installed generations (``refit.generations.installed_generations``).

    Raises:
        refit.generations.UnknownGenerationError: a live row's model id belongs to no generation,
            or a shadow row's id is not the pin's (never read as generation 0).
    """
    excluded = dict.fromkeys(EXCLUDED_REASONS, 0)
    generation_by_key: dict[tuple[Any, ...], int] = {}
    weeks: set[int] = set()
    for entry in entries:
        values = entry.immutable
        if (
            entry.kind != ENTRY_KIND_ROW
            or values["arm"] != ARM_LIVE
            or values["season"] != REFIT_SEASON
        ):
            continue
        generation = generation_of_row(
            values["model_artifact_id"], ARM_LIVE, generations
        )
        if generation is None:
            excluded["no_generation"] += 1
            continue
        generation_by_key[tuple(values[name] for name in _TWIN_KEY)] = generation
        weeks.add(int(values["week"]))

    in_force = in_force_outcomes(entries)
    realized = realized_values_from_scores(games)
    scores = _final_scores(games)
    records: list[dict[str, Any]] = []
    holes: list[tuple[Any, ...]] = []
    unpaired: list[int] = []
    for week in sorted(weeks):
        week_result = week_pairs(entries, REFIT_SEASON, week)
        if week_result.shadow_rows == 0:
            unpaired.append(week)
            continue
        holes.extend(key for key in week_result.holes if key in generation_by_key)
        failed = {key for key, _field in week_result.invariant_failures}
        for live, frozen in week_result.pairs:
            key = tuple(live[name] for name in _TWIN_KEY)
            if key not in generation_by_key:
                continue
            generation_of_row(frozen["model_artifact_id"], ARM_SHADOW, generations)
            if key in failed:
                excluded["invariant_failure"] += 1
                continue
            if live["model_value"] is None or frozen["model_value"] is None:
                excluded["no_model_value"] += 1
                continue
            game_id, target = str(live["game_id"]), str(live["target"])
            if game_id not in scores:
                excluded["no_final_score"] += 1
                continue
            outcome = realized[target][game_id]
            live_loss = _loss(target, live["model_value"], outcome)
            frozen_loss = _loss(target, frozen["model_value"], outcome)
            records.append(
                {
                    "game_id": game_id,
                    "season": REFIT_SEASON,
                    "week": week,
                    "target": target,
                    "generation": generation_by_key[key],
                    "live_value": float(live["model_value"]),
                    "frozen_value": float(frozen["model_value"]),
                    "home_score": scores[game_id][0],
                    "away_score": scores[game_id][1],
                    "live_loss": live_loss,
                    "frozen_loss": frozen_loss,
                    "gap": live_loss - frozen_loss,
                    **_arm_columns(
                        "live", live["status"], in_force.get((*key, ARM_LIVE))
                    ),
                    **_arm_columns(
                        "frozen", frozen["status"], in_force.get((*key, ARM_SHADOW))
                    ),
                }
            )
    return PairedFrame(
        frame=pd.DataFrame.from_records(records, columns=list(FRAME_COLUMNS)),
        holes=tuple(holes),
        unpaired_weeks=tuple(unpaired),
        excluded=excluded,
    )


def cell_gaps(paired: PairedFrame) -> Mapping[tuple[str, int], pd.DataFrame]:
    """The paired frame split into its (target, generation) cells, generations ascending."""
    frame = paired.frame
    cells: dict[tuple[str, int], pd.DataFrame] = {}
    for generation in sorted({int(value) for value in frame["generation"]}):
        for target in TARGETS:
            cell = frame.loc[
                (frame["generation"] == generation) & (frame["target"] == target)
            ]
            if not cell.empty:
                cells[(target, generation)] = cell
    return cells
