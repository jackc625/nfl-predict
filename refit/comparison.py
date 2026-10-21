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

THE DECISION (D-08, D-10, D-11, D-32)
-------------------------------------
:func:`decide` tests every (target, generation >= 1) cell. The Benjamini-Hochberg family is every
such cell with at least ``MIN_PAIRED_GAMES`` paired games, and its size ``m`` is passed explicitly
to ``backtest.profitability_2025.benjamini_hochberg_adjusted`` at ``ALPHA``: a cell in the family
whose gaps are all zero has no p-value yet still counts in ``m``. A cell is ``better`` when its
adjusted p-value is below ``ALPHA`` and its mean gap is negative, ``worse`` when it is below and
the mean gap is positive, else ``no difference`` (or ``too few to test``). Per target the
``ROLLUP_WORDS`` follow: some better and none worse is HELPED, some worse and none better HURT,
both MIXED, neither NO DETECTABLE DIFFERENCE.

Secondary, never changing a word: the POOLED figure per target is :func:`paired_test` over every
game of that target's family cells, with its own p-value, outside the family; flat ROI per arm and
generation is the mean in-force ``payout_flat`` over that arm's live-status pairs graded win, loss
or push, with no test. Generation 0 is reported (its count and its gap, exactly zero in A/A weeks)
and never tested.

CLV APPEARS NOWHERE (must-NOT P5). No frame column, result field or printed line carries a CLV
value; the deciding figure is accuracy only.

THE SEASON PAGE'S PANEL (D-12, D-13, D-14, SPEC R13)
----------------------------------------------------
:func:`panel_frames` hands the cache build the panel's RAW numbers -- never a p-value, a confidence
range or a test result: the test runs once, at season close. :func:`period_summary` gives, per
target, a ``Before the first re-fit`` row over the generation-0 pairs (when there are any), one
``Period N, from Tue Nov 3`` row per installed generation N >= 1 -- from the moment it is
installed, so a generation with no completed pair yet is a row with 0 paired games and NULL
figures, never "no re-fit yet" -- and a season-to-date row over every paired game since the first
re-fit. A period starts on its cadence date (the date its install intent names). The panel is
accuracy only: no betting return per arm, so frozen-arm bets never look like bets.

Batch code: the comparison CLI (``scripts/compare_refit_arms.py``) and the cache build
(``forward_ledger.cache_sources``) import it. No module under ``api/`` may import it, so the
request path never loads it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import pandas as pd
from scipy import stats

from api.cache import (
    BET_STATUS_LIVE,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PENDING,
    GRADING_STATUS_PUSH,
    GRADING_STATUS_WIN,
)
from api.cache_schema import (
    REFIT_AA_STATUS_COLUMNS,
    REFIT_PANEL_COLUMNS,
    REFIT_PANEL_PERIOD,
    REFIT_PANEL_SEASON_TO_DATE,
)
from backtest.profitability_2025 import benjamini_hochberg_adjusted
from backtest.refit_preregistration_2026 import (
    ALPHA,
    LOG_LOSS_CLIP,
    MIN_PAIRED_GAMES,
    REFIT_SEASON,
    ROLLUP_WORDS,
)
from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.corrections import InForce, in_force_outcomes
from forward_ledger.schema import ARM_LIVE, ARM_SHADOW
from forward_ledger.settle import realized_values_from_scores
from forward_ledger.store import LedgerEntry
from refit.aa import week_pairs
from refit.generations import (
    GENERATION_ZERO,
    Generation,
    generation_of_row,
    installed_generations,
)
from refit.records import read_aa_records, read_date_events

__all__ = [
    "AHEAD_EVEN",
    "AHEAD_FROZEN",
    "AHEAD_LIVE",
    "ARM_LABELS",
    "BEFORE_FIRST_REFIT_LABEL",
    "EXCLUDED_REASONS",
    "FRAME_COLUMNS",
    "RESULT_BETTER",
    "RESULT_NOT_TESTED",
    "RESULT_NO_DIFFERENCE",
    "RESULT_WORSE",
    "STATUS_ALL_ZERO",
    "STATUS_GENERATION_ZERO",
    "STATUS_TESTED",
    "STATUS_TOO_FEW",
    "TARGETS",
    "CellResult",
    "ComparisonResult",
    "PairedFrame",
    "PooledResult",
    "RoiResult",
    "decide",
    "line_loss",
    "paired_frame",
    "paired_test",
    "panel_frames",
    "period_summary",
    "rollup_word",
    "wp_loss",
]

# The three targets, in the order every report lists them.
TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# The per-cell statuses :func:`paired_test` answers with, and generation 0's.
STATUS_TESTED: str = "tested"
STATUS_TOO_FEW: str = "too few to test"
STATUS_ALL_ZERO: str = "no difference (all paired gaps zero)"
STATUS_GENERATION_ZERO: str = "never tested (generation 0)"

# A cell's result after the correction (D-10 reads these).
RESULT_BETTER: str = "better"
RESULT_WORSE: str = "worse"
RESULT_NO_DIFFERENCE: str = "no difference"
RESULT_NOT_TESTED: str = "not tested"

# The D-10 words, unpacked from the frozen record in its order, never re-typed.
_HELPED, _HURT, _MIXED, _NO_DETECTABLE_DIFFERENCE = ROLLUP_WORDS

# The two arms of the report-only ROI figure: the frame's column prefix is the label.
ARM_LABELS: tuple[str, ...] = ("live", "frozen")

# The grading statuses a flat ROI counts: every settled bet.
_GRADED: tuple[str, ...] = (
    GRADING_STATUS_WIN,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PUSH,
)

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


@dataclass(frozen=True)
class CellResult:
    """One (target, generation) cell: its raw figures and, for generation >= 1, its test.

    Attributes:
        target: ``wp``, ``ats`` or ``ou``.
        generation: The generation of the cell's live rows.
        n: Its paired games.
        live_loss: The live arm's mean per-game loss (None with no game).
        frozen_loss: The frozen arm's mean per-game loss (None with no game).
        mean_gap: The mean of live loss minus frozen loss (None with no game).
        t: The paired t-statistic, or None when the cell was not tested.
        p: The raw two-sided p-value, or None when the cell was not tested.
        adjusted_p: The Benjamini-Hochberg adjusted p-value, or None when it passed none.
        status: :data:`STATUS_TESTED`, :data:`STATUS_TOO_FEW`, :data:`STATUS_ALL_ZERO` or
            :data:`STATUS_GENERATION_ZERO`.
        result: :data:`RESULT_BETTER`, :data:`RESULT_WORSE`, :data:`RESULT_NO_DIFFERENCE`,
            :data:`STATUS_TOO_FEW` or :data:`RESULT_NOT_TESTED` (generation 0).
    """

    target: str
    generation: int
    n: int
    live_loss: float | None
    frozen_loss: float | None
    mean_gap: float | None
    t: float | None
    p: float | None
    adjusted_p: float | None
    status: str
    result: str


@dataclass(frozen=True)
class PooledResult:
    """The secondary pooled figure of one target: outside the family, never moving the word."""

    target: str
    n: int
    mean_gap: float | None
    t: float | None
    p: float | None
    status: str


@dataclass(frozen=True)
class RoiResult:
    """Report-only flat ROI of one arm in one generation; no test is ever run on it."""

    generation: int
    arm: str
    bets: int
    flat_roi: float | None


@dataclass(frozen=True)
class ComparisonResult:
    """The pre-registered comparison: the cells, the family, the words and the secondaries.

    Attributes:
        cells: Every reported cell, generations ascending, then in :data:`TARGETS` order.
        family_size: The BH family size ``m`` (cells of generation >= 1 with enough games).
        verdicts: Target -> its ``ROLLUP_WORDS`` word.
        pooled: The secondary pooled figure per target, in :data:`TARGETS` order.
        roi: Report-only flat ROI per generation and arm.
    """

    cells: tuple[CellResult, ...]
    family_size: int
    verdicts: dict[str, str]
    pooled: tuple[PooledResult, ...]
    roi: tuple[RoiResult, ...]

    def as_dict(self) -> dict[str, Any]:
        """The result as plain JSON-ready values."""
        return dataclasses.asdict(self)


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def rollup_word(results: Sequence[str]) -> str:
    """The D-10 word for one target's generation >= 1 cell results (after the correction)."""
    better = RESULT_BETTER in results
    worse = RESULT_WORSE in results
    if better and worse:
        return _MIXED
    if better:
        return _HELPED
    if worse:
        return _HURT
    return _NO_DETECTABLE_DIFFERENCE


def _cell_result(status: str, mean_gap: float | None, adjusted_p: float | None) -> str:
    """A tested cell is better or worse only when its adjusted p-value is below ``ALPHA``."""
    if status == STATUS_TOO_FEW:
        return STATUS_TOO_FEW
    if status != STATUS_TESTED or adjusted_p is None or mean_gap is None:
        return RESULT_NO_DIFFERENCE
    if adjusted_p < ALPHA and mean_gap < 0.0:
        return RESULT_BETTER
    if adjusted_p < ALPHA and mean_gap > 0.0:
        return RESULT_WORSE
    return RESULT_NO_DIFFERENCE


def _flat_roi(rows: Sequence[dict[Any, Any]], generation: int, arm: str) -> RoiResult:
    """Mean in-force ``payout_flat`` over *arm*'s live-status settled pairs of *generation*."""
    payouts = [
        float(row[f"{arm}_payout_flat"])
        for row in rows
        if int(row["generation"]) == generation
        and row[f"{arm}_status"] == BET_STATUS_LIVE
        and row[f"{arm}_grading_status"] in _GRADED
    ]
    return RoiResult(
        generation=generation, arm=arm, bets=len(payouts), flat_roi=_mean(payouts)
    )


def decide(paired: PairedFrame, generations: Sequence[Generation]) -> ComparisonResult:
    """The pre-registered decision over *paired* (ROLLING-REFIT-PREREGISTRATION.md section 11).

    Args:
        paired: :func:`paired_frame`'s result.
        generations: The installed generations; every generation >= 1 among them is reported,
            with a count of 0 when it has no paired game yet.
    """
    rows = paired.frame.to_dict("records")
    installed = {g.number for g in generations if g.number != GENERATION_ZERO}
    numbers = sorted({int(row["generation"]) for row in rows} | installed)

    cells_rows: list[tuple[str, int, list[dict[Any, Any]]]] = []
    for generation in numbers:
        for target in TARGETS:
            members = [
                row
                for row in rows
                if row["target"] == target and int(row["generation"]) == generation
            ]
            if generation != GENERATION_ZERO or members:
                cells_rows.append((target, generation, members))

    tests = {
        index: paired_test([float(row["gap"]) for row in members])
        for index, (_target, generation, members) in enumerate(cells_rows)
        if generation != GENERATION_ZERO
    }
    family = [index for index, test in tests.items() if test["n"] >= MIN_PAIRED_GAMES]
    tested = [index for index in family if tests[index]["status"] == STATUS_TESTED]
    adjusted: dict[int, float] = {}
    if family:
        values = benjamini_hochberg_adjusted(
            [tests[i]["p"] for i in tested], len(family)
        )
        adjusted = dict(zip(tested, values, strict=True))

    cells: list[CellResult] = []
    for index, (target, generation, members) in enumerate(cells_rows):
        live_loss = _mean([float(row["live_loss"]) for row in members])
        frozen_loss = _mean([float(row["frozen_loss"]) for row in members])
        test = tests.get(index)
        if test is None:
            cells.append(
                CellResult(
                    target=target,
                    generation=generation,
                    n=len(members),
                    live_loss=live_loss,
                    frozen_loss=frozen_loss,
                    mean_gap=_mean([float(row["gap"]) for row in members]),
                    t=None,
                    p=None,
                    adjusted_p=None,
                    status=STATUS_GENERATION_ZERO,
                    result=RESULT_NOT_TESTED,
                )
            )
            continue
        adjusted_p = adjusted.get(index)
        cells.append(
            CellResult(
                target=target,
                generation=generation,
                n=test["n"],
                live_loss=live_loss,
                frozen_loss=frozen_loss,
                mean_gap=test["mean_gap"],
                t=test["t"],
                p=test["p"],
                adjusted_p=adjusted_p,
                status=test["status"],
                result=_cell_result(test["status"], test["mean_gap"], adjusted_p),
            )
        )

    verdicts = {
        target: rollup_word(
            [
                cell.result
                for cell in cells
                if cell.target == target and cell.generation != GENERATION_ZERO
            ]
        )
        for target in TARGETS
    }
    pooled = tuple(
        PooledResult(
            target=target,
            **paired_test(
                [
                    float(row["gap"])
                    for index in family
                    if cells_rows[index][0] == target
                    for row in cells_rows[index][2]
                ]
            ),
        )
        for target in TARGETS
    )
    roi = tuple(
        _flat_roi(rows, generation, arm) for generation in numbers for arm in ARM_LABELS
    )
    return ComparisonResult(
        cells=tuple(cells),
        family_size=len(family),
        verdicts=verdicts,
        pooled=pooled,
        roi=roi,
    )


# ---------------------------------------------------------------------------
# The Season page's panel: raw numbers only (D-12, D-13, D-14)
# ---------------------------------------------------------------------------

# Which arm erred less over a panel row: the live arm (negative gap), the frozen arm (positive) or
# neither (exactly zero, as in every A/A week).
AHEAD_LIVE: str = "live"
AHEAD_FROZEN: str = "frozen"
AHEAD_EVEN: str = "even"

BEFORE_FIRST_REFIT_LABEL: str = "Before the first re-fit"

# The unit each target's per-game error is measured in (D-07).
_UNITS: dict[str, str] = {"wp": "log loss", "ats": "points", "ou": "points"}

# Fixed English abbreviations, so a label never depends on the machine's locale.
_WEEKDAYS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS: tuple[str, ...] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def _short_date(day: date) -> str:
    """``Tue Nov 3``: the weekday, the month and the day without a leading zero."""
    return f"{_WEEKDAYS[day.weekday()]} {_MONTHS[day.month - 1]} {day.day}"


def _period_starts(
    installs: Sequence[Generation], date_events: Sequence[Mapping[str, Any]]
) -> dict[int, date]:
    """Generation number -> its cadence date, read from the date events its install names.

    Raises:
        ValueError: an installed generation's cadence date has no event (never guessed).
    """
    dates: dict[tuple[int | None, int | None], str] = {
        (int(event["list_week"]), int(event["date_index"])): str(event["date"])
        for event in date_events
    }
    starts: dict[int, date] = {}
    for generation in installs:
        key = (generation.list_week, generation.date_index)
        if key not in dates:
            msg = (
                f"generation {generation.number} names cadence date {key} (list week, index), "
                "which no date event records; a period start is never guessed"
            )
            raise ValueError(msg)
        starts[generation.number] = date.fromisoformat(dates[key])
    return starts


def _ahead(gap: float | None) -> str | None:
    if gap is None:
        return None
    if gap < 0.0:
        return AHEAD_LIVE
    if gap > 0.0:
        return AHEAD_FROZEN
    return AHEAD_EVEN


def _panel_row(
    target: str,
    period_index: int | None,
    row_kind: str,
    label: str,
    start: date | None,
    members: Sequence[Mapping[Any, Any]],
) -> dict[str, Any]:
    """One panel row: both arms' mean error over *members*, the gap and who is ahead."""
    gap = _mean([float(row["gap"]) for row in members])
    return {
        "season": REFIT_SEASON,
        "target": target,
        "period_index": period_index,
        "row_kind": row_kind,
        "period_label": label,
        "period_start": None if start is None else start.isoformat(),
        "live_error": _mean([float(row["live_loss"]) for row in members]),
        "frozen_error": _mean([float(row["frozen_loss"]) for row in members]),
        "gap": gap,
        "ahead": _ahead(gap),
        "paired_games": len(members),
        "unit": _UNITS[target],
    }


def _panel_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(list(records), columns=REFIT_PANEL_COLUMNS)
    # A season-to-date row has no period index; keep the column integer, NULL there.
    frame["period_index"] = frame["period_index"].astype("Int64")
    frame["paired_games"] = frame["paired_games"].astype("int64")
    return frame


def period_summary(
    paired: PairedFrame,
    generations: Sequence[Generation],
    date_events: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """The panel's rows (``REFIT_PANEL_COLUMNS``): raw numbers per target and period, no test.

    Per target, in this order: the ``Before the first re-fit`` row over the generation-0 pairs
    (only when there are any); one ``Period N, from <date>`` row per installed generation N >= 1,
    present from its install even with no completed pair (0 paired games, NULL figures); and,
    whenever a generation >= 1 is installed, a season-to-date row over every pair of generation
    >= 1, whose label names the first re-fit's date.

    Args:
        paired: :func:`paired_frame`'s result.
        generations: The installed generations (``refit.generations.installed_generations``).
        date_events: The cadence-date events (``refit.records.read_date_events``).

    Raises:
        ValueError: an installed generation's cadence date has no event.
    """
    rows = paired.frame.to_dict("records")
    installs = sorted(
        (g for g in generations if g.number != GENERATION_ZERO),
        key=lambda g: g.number,
    )
    starts = _period_starts(installs, date_events)
    records: list[dict[str, Any]] = []
    for target in TARGETS:
        target_rows = [row for row in rows if row["target"] == target]
        before = [
            row for row in target_rows if int(row["generation"]) == GENERATION_ZERO
        ]
        if before:
            records.append(
                _panel_row(
                    target,
                    GENERATION_ZERO,
                    REFIT_PANEL_PERIOD,
                    BEFORE_FIRST_REFIT_LABEL,
                    None,
                    before,
                )
            )
        for generation in installs:
            start = starts[generation.number]
            records.append(
                _panel_row(
                    target,
                    generation.number,
                    REFIT_PANEL_PERIOD,
                    f"Period {generation.number}, from {_short_date(start)}",
                    start,
                    [
                        row
                        for row in target_rows
                        if int(row["generation"]) == generation.number
                    ],
                )
            )
        if installs:
            first = starts[installs[0].number]
            records.append(
                _panel_row(
                    target,
                    None,
                    REFIT_PANEL_SEASON_TO_DATE,
                    f"Season to date: every paired game since {_short_date(first)}",
                    first,
                    [
                        row
                        for row in target_rows
                        if int(row["generation"]) != GENERATION_ZERO
                    ],
                )
            )
    return _panel_frame(records)


def _aa_status_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """One ``REFIT_AA_STATUS_COLUMNS`` row per A/A week record, in file order."""
    rows: list[dict[str, Any]] = []
    for record in records:
        mismatch = record.get("first_mismatch") or {}
        rows.append(
            {
                "season": int(record["season"]),
                "week": int(record["week"]),
                "evaluated_at": record.get("evaluated_at_utc"),
                "clean": bool(record["clean"]),
                "reason": record.get("reason"),
                "live_rows": record.get("live_rows"),
                "twins": record.get("twins"),
                "mismatch_field": mismatch.get("field"),
                "mismatch_game": mismatch.get("game_id"),
                "mismatch_target": mismatch.get("target"),
            }
        )
    return pd.DataFrame.from_records(rows, columns=REFIT_AA_STATUS_COLUMNS)


def panel_frames(
    entries: Sequence[LedgerEntry],
    games: pd.DataFrame,
    ledger_dir: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The Season page panel's two cache frames, computed at cache build (D-12..D-14, R13).

    Args:
        entries: The ledger, in file order (``forward_ledger.store.read_entries``).
        games: A silver-``games``-shaped frame (``game_id``, ``home_score``, ``away_score``); an
            empty one leaves every pair without a final score, so every period counts 0 games.
        ledger_dir: The ledger directory holding ``refit/`` (the generations, the cadence dates
            and the A/A week records).

    Returns:
        ``(panel rows, A/A status rows)`` in ``REFIT_PANEL_COLUMNS`` and
        ``REFIT_AA_STATUS_COLUMNS``; both empty before any shadow row or install exists.

    Raises:
        refit.generations.UnknownGenerationError: a row's stamped id belongs to no generation.
        refit.records.RefitRecordFormatError: a re-fit record cannot be read.
    """
    generations = installed_generations(ledger_dir)
    paired = paired_frame(entries, games, generations)
    panel = period_summary(paired, generations, read_date_events(ledger_dir))
    return panel, _aa_status_frame(read_aa_records(ledger_dir))
