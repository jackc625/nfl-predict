"""Study 1b: does the v2 model's disagreement with the early-week line predict the line's move?

Each studied game of 2020-2024 carries three lines (``backtest.study_1b_lines``): the EARLY line
(the earliest same-source capture taken after the season's earlier weeks ended, usually Tuesday
11:55 ET), the same-source DAY-BEFORE line and the nflverse CLOSE. The v2 model scores each game
from its inputs as known at that early capture (``backtest.study_1b_inputs``); it trains
walk-forward on the day-before v2 gold exactly as in Study 1a (``models.v2_trainer``), and only
the rows it scores are built as of the early capture.

THE MOVE TEST. ``backtest.market_info.information_beyond_line(later, early, model)``: the OLS
slope of (later - early) on an intercept and (model - early), one-sided (H1: slope > 0). A slope
b means that when the model and the early line differed by d, the later line moved on average
b x d toward the model. ATS uses the home-positive spreads against the ATS prediction (the home
margin), O/U the totals against the O/U prediction. WP has no early moneyline in 2020-2024, so
each of its lines is a market log-odds from the spread through the converter's prior-only season
slope (``wp_market_log_odds``), tested against the logit of the WP prediction; a season with no
prior slope (2020) is left out.

THE PASS BAR (owner ruling 1, 2026-11-06). A target passes only if the slope is positive at
one-sided p < ``PASS_ALPHA`` against BOTH the nflverse close and the same-source day-before line.
The close comes from a different source than the early line, and a consistent difference between
two sources can make a model look as if it predicts the move. A passing target may get an
early-week arm in plan 4; a failed one gets none.

Every game left out on the way from the schedule to a test is counted by name in
``Study1bResult.counts``. Nothing is written except ``study_1b.json`` and ``report.md``, into a
directory refused if it lies under ``data/``. ASCII only, no emoji.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.special import logit

from backtest.market_info import InfoResult, information_beyond_line
from backtest.refit_preregistration_2026 import MARKET_CONVERTER_ID
from backtest.simulation import SLIPPAGE_POINTS
from backtest.study_1a import (
    TARGETS,
    _plain,
    _read_manifest,
    _table,
    live_artifacts_dir,
    refuse_data_dir,
)
from backtest.study_1b_inputs import (
    NO_GOLD_ROW,
    NOT_SETTLED,
    SITE_NOT_KNOWN_AT_EARLY_CAPTURE,
    EarlyInputs,
    build_early_inputs,
)
from backtest.study_1b_lines import (
    EARLY_WINDOW,
    EXCLUSION_REASONS,
    StudyLines,
    load_study_lines,
)
from data.upstream_pin import load_schedules
from forward_ledger.settle import load_silver_games
from models.market_probability import (
    OWNED_LINE_SEASONS,
    load_market_probability_artifact,
    oof_market_probability,
)
from models.v2_trainer import walk_forward_v2
from refit.records import code_commit
from refit.remeasure import extended_slopes

__all__ = [
    "COMPARISONS",
    "DEFAULT_OUT_DIR",
    "PASS_ALPHA",
    "STUDY_SEASONS",
    "TARGETS",
    "MoveTest",
    "Study1bResult",
    "TargetVerdict",
    "context_notes",
    "move_test",
    "passes",
    "render_report",
    "run_study_1b",
    "summary_lines",
    "target_verdict",
    "wp_market_log_odds",
    "write_report",
]

#: The seasons with time-stamped early captures.
STUDY_SEASONS: tuple[int, ...] = OWNED_LINE_SEASONS

#: A comparison clears the bar only with a positive slope below this one-sided p.
PASS_ALPHA: float = 0.05

#: The later lines each early line is compared with; a target must clear the bar on both.
COMPARISONS: tuple[str, ...] = ("close", "day_before")

DEFAULT_OUT_DIR = Path("outputs/study_1b")
JSON_FILENAME = "study_1b.json"
REPORT_FILENAME = "report.md"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The WP prediction is clipped to this distance from 0 and 1 before its logit.
_PROB_CLIP = 1e-6

#: Which of a game's lines each target is tested on (WP converts the spread).
_LINE_KIND: dict[str, str] = {"wp": "spread", "ats": "spread", "ou": "total"}

#: The reasons ``build_early_inputs`` leaves a game out, in the order it tests them.
_INPUT_EXCLUSION_REASONS: tuple[str, ...] = (
    NOT_SETTLED,
    SITE_NOT_KNOWN_AT_EARLY_CAPTURE,
    NO_GOLD_ROW,
)

_SHORT_NAME: dict[str, str] = {"wp": "WP", "ats": "ATS", "ou": "O/U"}
_COMPARISON_NAME: dict[str, str] = {
    "close": "the nflverse close",
    "day_before": "the day-before line",
}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveTest:
    """One line-move test: does (model - early) predict (later - early)?

    Attributes:
        info: The slope on (model - early), or None when the fit could not run.
        note: Why ``info`` is None; empty otherwise.
        games: The rows where the later line, the early line and the model are all finite.
        mean_abs_move: Mean |later - early| over those rows (None with no rows).
        mean_abs_departure: Mean |model - early| over those rows (None with no rows).
    """

    info: InfoResult | None
    note: str
    games: int
    mean_abs_move: float | None
    mean_abs_departure: float | None


@dataclass(frozen=True)
class TargetVerdict:
    """One target's two move tests and its verdict.

    Attributes:
        close: The test against the nflverse close.
        day_before: The test against the same-source day-before line.
        passed: Both tests clear the bar (:func:`passes`).
        share_clearing_slippage: ATS and O/U: the share of the close test's games where
            |close slope x (model - early)| >= ``SLIPPAGE_POINTS``, i.e. how often a predicted
            move would clear the half-point of slippage; None for WP or when the close test
            could not run.
    """

    close: MoveTest
    day_before: MoveTest
    passed: bool
    share_clearing_slippage: float | None


@dataclass(frozen=True)
class Study1bResult:
    """The study's verdicts and the numbers behind them.

    Attributes:
        generation_cutoff_utc: The walk-forward replay's cutoff, ISO text.
        verdicts: ``{target: TargetVerdict}``.
        counts: Games counted and left out, by name, in the order a game meets them.
        code_commit: The code's commit (``+dirty`` marks an uncommitted change), or None.
        v2_gold_manifest: The contents of the v2 gold's ``manifest.json``, or None.
    """

    generation_cutoff_utc: str
    verdicts: dict[str, TargetVerdict]
    counts: dict[str, int]
    code_commit: str | None
    v2_gold_manifest: dict[str, Any] | None


# ---------------------------------------------------------------------------
# The move test and the bar
# ---------------------------------------------------------------------------


def _finite_rows(
    later: ArrayLike, early: ArrayLike, model: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three columns as floats, keeping only the rows where all three are finite.

    Raises:
        ValueError: the inputs are not three 1-D arrays of one length.
    """
    columns = [np.asarray(values, dtype=float) for values in (later, early, model)]
    if any(c.ndim != 1 for c in columns) or len({len(c) for c in columns}) != 1:
        msg = (
            "the later line, the early line and the model must be 1-D and equally long"
        )
        raise ValueError(msg)
    keep = np.isfinite(columns[0]) & np.isfinite(columns[1]) & np.isfinite(columns[2])
    return columns[0][keep], columns[1][keep], columns[2][keep]


def move_test(later: ArrayLike, early: ArrayLike, model: ArrayLike) -> MoveTest:
    """The slope of (later - early) on (model - early), over the rows where all are finite.

    A fit that cannot run (too few rows, a singular or degenerate design) is recorded as a
    note, never raised: it clears no bar.
    """
    later_f, early_f, model_f = _finite_rows(later, early, model)
    games = len(later_f)
    try:
        info: InfoResult | None = information_beyond_line(later_f, early_f, model_f)
        note = ""
    except ValueError as exc:
        info, note = None, str(exc)
    return MoveTest(
        info=info,
        note=note,
        games=games,
        mean_abs_move=float(np.mean(np.abs(later_f - early_f))) if games else None,
        mean_abs_departure=float(np.mean(np.abs(model_f - early_f))) if games else None,
    )


def _clears(test: MoveTest) -> bool:
    """A positive slope at one-sided p < ``PASS_ALPHA``."""
    return (
        test.info is not None
        and test.info.coef > 0
        and test.info.p_one_sided < PASS_ALPHA
    )


def passes(close: MoveTest, day_before: MoveTest) -> bool:
    """True only if both comparisons clear the bar (owner ruling 1, 2026-11-06)."""
    return _clears(close) and _clears(day_before)


def wp_market_log_odds(
    frame: pd.DataFrame, spread_column: str, slopes: Mapping[int, float]
) -> np.ndarray:
    """The market log-odds of a home win implied by *frame*'s home-positive spread.

    The logit of ``oof_market_probability`` with ``frame[spread_column]`` as
    ``home_fav_margin``, i.e. each season's prior-only slope x the spread. A NaN spread gives
    NaN.

    Raises:
        NoPriorFoldError: a row's season has no prior-only slope.
    """
    seasonal: dict[int | str, float] = {
        int(season): float(slope) for season, slope in slopes.items()
    }
    margins = pd.DataFrame(
        {
            "season": frame["season"].to_numpy(),
            "home_fav_margin": frame[spread_column].to_numpy(dtype=float),
        }
    )
    return np.asarray(logit(oof_market_probability(margins, seasonal)), dtype=float)


def _share_clearing_slippage(
    close: MoveTest, later: np.ndarray, early: np.ndarray, model: np.ndarray
) -> float | None:
    """Share of the close test's games whose predicted move reaches ``SLIPPAGE_POINTS``."""
    if close.info is None:
        return None
    _, early_f, model_f = _finite_rows(later, early, model)
    if len(early_f) == 0:
        return None
    predicted_move = np.abs(close.info.coef * (model_f - early_f))
    return float(np.mean(predicted_move >= SLIPPAGE_POINTS))


def target_verdict(
    target: str, frame: pd.DataFrame, slopes: Mapping[int, float]
) -> tuple[TargetVerdict, dict[str, int]]:
    """One target's two move tests and verdict, and the games each step left out.

    Args:
        target: ``wp``, ``ats`` or ``ou``.
        frame: The target's predictions (``season``, ``prediction``) joined to the study's
            lines (``early_/day_before_/close_spread`` and ``_total``), one row per game.
        slopes: The converter's prior-only season slopes (WP only).

    Returns:
        The verdict, and the counts ``wp_no_prior_fold_converter`` (WP only) and
        ``{target}_{comparison}_missing_number`` per comparison: the rows a test dropped for
        a missing later line, early line or prediction.
    """
    counts: dict[str, int] = {}
    kind = _LINE_KIND[target]
    rows = frame
    if target == "wp":
        has_slope = frame["season"].astype(int).isin([int(s) for s in slopes])
        counts["wp_no_prior_fold_converter"] = int((~has_slope).sum())
        rows = frame.loc[has_slope.to_numpy()]
        early = wp_market_log_odds(rows, f"early_{kind}", slopes)
        probability = np.clip(
            rows["prediction"].to_numpy(dtype=float), _PROB_CLIP, 1.0 - _PROB_CLIP
        )
        model = np.asarray(logit(probability), dtype=float)
        later = {
            c: wp_market_log_odds(rows, f"{c}_{kind}", slopes) for c in COMPARISONS
        }
    else:
        early = rows[f"early_{kind}"].to_numpy(dtype=float)
        model = rows["prediction"].to_numpy(dtype=float)
        later = {c: rows[f"{c}_{kind}"].to_numpy(dtype=float) for c in COMPARISONS}

    tests = {c: move_test(later[c], early, model) for c in COMPARISONS}
    for c in COMPARISONS:
        counts[f"{target}_{c}_missing_number"] = len(rows) - tests[c].games
    share = (
        None
        if target == "wp"
        else _share_clearing_slippage(tests["close"], later["close"], early, model)
    )
    verdict = TargetVerdict(
        close=tests["close"],
        day_before=tests["day_before"],
        passed=passes(tests["close"], tests["day_before"]),
        share_clearing_slippage=share,
    )
    return verdict, counts


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _reason_counts(
    prefix: str, reasons: Iterable[object], known: tuple[str, ...]
) -> dict[str, int]:
    """``{prefix}_{reason}: games`` for every known reason (zeros too) and any other seen."""
    observed = Counter(str(reason) for reason in reasons)
    names = [*known, *sorted(set(observed) - set(known))]
    return {f"{prefix}_{name}": observed[name] for name in names}


def _line_and_input_counts(lines: StudyLines, inputs: EarlyInputs) -> dict[str, int]:
    """The games of the study seasons, and those left out at the lines and the inputs."""
    counts = {"study_season_games": len(lines.frame) + len(lines.excluded)}
    counts.update(
        _reason_counts("lines_excluded", lines.excluded["reason"], EXCLUSION_REASONS)
    )
    counts["games_with_lines"] = len(lines.frame)
    counts.update(
        _reason_counts(
            "inputs_excluded", inputs.excluded["reason"], _INPUT_EXCLUSION_REASONS
        )
    )
    counts["qb_values_clipped"] = inputs.qb_values_clipped
    return counts


def run_study_1b(
    v2_gold_dir: Path,
    silver_dir: Path,
    generation_cutoff: datetime,
    *,
    artifacts_dir: Path | None = None,
) -> Study1bResult:
    """Score every studied game as of its early capture and judge each target. Writes nothing.

    Args:
        v2_gold_dir: The v2 gold matrices (``outputs/gold_v2``); the models train on them and
            the early rows are rebuilt from them.
        silver_dir: The silver lake (``games``, ``odds_timeline``, ``odds_snapshot``).
        generation_cutoff: The replay's cutoff, tz-aware.
        artifacts_dir: Where the bound WP market converter is read from; None means the
            artifacts folder beside the live lake (``live_artifacts_dir``).
    """
    converter_dir = (
        live_artifacts_dir(silver_dir) if artifacts_dir is None else Path(artifacts_dir)
    )
    slopes = extended_slopes(
        load_market_probability_artifact(MARKET_CONVERTER_ID, converter_dir)
    )
    games = load_silver_games(silver_dir)
    lines = load_study_lines(silver_dir)
    nflverse = load_schedules(list(range(STUDY_SEASONS[0] - 1, STUDY_SEASONS[-1] + 1)))
    early_times = cast(pd.Series, lines.frame.set_index("game_id")["early_ts"])
    inputs = build_early_inputs(Path(v2_gold_dir), games, nflverse, early_times)
    predictions = walk_forward_v2(
        Path(v2_gold_dir), games, generation_cutoff, prediction_rows=inputs.rows
    )

    counts = _line_and_input_counts(lines, inputs)
    verdicts: dict[str, TargetVerdict] = {}
    for target in TARGETS:
        supplied = set(inputs.rows[target]["game_id"].astype(str))
        predicted = predictions[target].loc[:, ["game_id", "prediction"]]
        counts[f"{target}_games_with_inputs"] = len(supplied)
        # walk_forward_v2 silently drops a supplied row that falls in no fold.
        counts[f"{target}_inputs_without_prediction"] = len(
            supplied - set(predicted["game_id"])
        )
        counts[f"{target}_games_predicted"] = len(predicted)
        joined = lines.frame.merge(
            predicted, on="game_id", how="inner", validate="one_to_one"
        )
        verdicts[target], target_counts = target_verdict(target, joined, slopes)
        counts.update(target_counts)

    return Study1bResult(
        generation_cutoff_utc=generation_cutoff.isoformat(),
        verdicts=verdicts,
        counts=counts,
        code_commit=code_commit(_PROJECT_ROOT),
        v2_gold_manifest=_read_manifest(Path(v2_gold_dir)),
    )


# ---------------------------------------------------------------------------
# The summary and the notes
# ---------------------------------------------------------------------------


def _count(result: Study1bResult, name: str) -> str:
    value = result.counts.get(name)
    return "n/a" if value is None else f"{value:,}"


def _test_text(test: MoveTest) -> str:
    if test.info is None:
        return f"no result ({test.note})"
    return (
        f"slope {test.info.coef:+.3f}, one-sided p {test.info.p_one_sided:.3g}, "
        f"{test.games:,} games"
    )


def _target_line(target: str, verdict: TargetVerdict) -> str:
    """One target's verdict, both tests and, for a passing ATS or O/U, the slippage share."""
    name = _SHORT_NAME[target]
    tests = "; ".join(
        f"{'Against' if i == 0 else 'against'} {_COMPARISON_NAME[c]}: "
        f"{_test_text(getattr(verdict, c))}"
        for i, c in enumerate(COMPARISONS)
    )
    if not verdict.passed:
        failed = [
            _COMPARISON_NAME[c] for c in COMPARISONS if not _clears(getattr(verdict, c))
        ]
        reason = f" (it fails against {' and '.join(failed)})" if failed else ""
        return f"{name}: does NOT pass{reason}. {tests}."
    line = f"{name}: passes. {tests}."
    if verdict.share_clearing_slippage is not None:
        line += (
            f" A predicted move clears the {SLIPPAGE_POINTS}-point slippage in "
            f"{verdict.share_clearing_slippage:.0%} of the close test's games."
        )
    return line


def _plan_4_line(result: Study1bResult) -> str:
    """Which bet types plan 4 may build an early-week arm for."""
    passing = [_SHORT_NAME[t] for t in TARGETS if result.verdicts[t].passed]
    failing = [_SHORT_NAME[t] for t in TARGETS if not result.verdicts[t].passed]
    if not passing:
        return (
            "Plan 4 builds no early-week arm: no bet type passes, so none gets one; the "
            "daily early boards keep recording for later study."
        )
    line = f"Plan 4 may build an early-week arm for: {', '.join(passing)}."
    if failing:
        line += (
            f" {' and '.join(failing)} did not pass, so "
            f"{'it gets' if len(failing) == 1 else 'they get'} none; the daily early "
            "boards keep recording for later study."
        )
    else:
        line += " The daily early boards keep recording."
    return line


def summary_lines(result: Study1bResult) -> list[str]:
    """The plain-English summary, at most eight lines: one per target, then plan 4."""
    return [
        *[_target_line(t, result.verdicts[t]) for t in TARGETS],
        _plan_4_line(result),
    ]


def context_notes(result: Study1bResult) -> list[str]:
    """The caveats that follow the summary (not part of its eight-line limit)."""
    seasons = f"{STUDY_SEASONS[0]}-{STUDY_SEASONS[-1]}"
    not_settled = _count(result, f"inputs_excluded_{NOT_SETTLED}")
    neutral = _count(result, f"inputs_excluded_{SITE_NOT_KNOWN_AT_EARLY_CAPTURE}")
    return [
        "Reading a slope: when the model and the early line differed by d, the later line "
        "moved on average slope x d toward the model (0: the line ignored the model; 1: it "
        "moved all the way). ATS and O/U are in points, WP in market log-odds.",
        "The early line is a game's earliest same-source capture taken after every game of "
        f"the season's earlier weeks had ended and at most {EARLY_WINDOW.days} days before "
        "kickoff; its own "
        "timestamp is the information time the model's inputs are fenced to.",
        f"Why both comparisons must pass (owner ruling 1, 2026-11-06): a target passes only "
        f"if its slope is positive at one-sided p < {PASS_ALPHA} against BOTH the nflverse "
        "close and the same-source day-before line. The close comes from a different source "
        "than the early line, and a consistent difference between two sources can make a "
        "model look as if it predicts the move.",
        "Inputs as of the capture: each game is scored by the v2 model trained walk-forward "
        "on the day-before v2 gold, from inputs rebuilt as known at the game's early capture "
        "(usually Tuesday 11:55 ET): QB, the market rating and the league scoring level. "
        "Elo, success rate, opponent-adjusted EPA, pace and home field rest only on finished "
        f"games; {not_settled} games whose teams had not both finished their previous game "
        "by the capture are left out.",
        "QB is each team's latest primary passer (the passer of its last game); the untimed "
        "depth charts are never read, which is conservative: a live early arm would see "
        f"Tuesday's charts. {_count(result, 'qb_values_clipped')} rebuilt QB values were "
        "clipped to the day-before gold's range of that column in that season, as the gold "
        "build winsorizes its QB columns.",
        "Wind is unknown at the early capture, so it is left blank and the model's median "
        "fill supplies it; games in indoor stadiums keep their wind of 0.0, a fact of the "
        "stadium (a retractable roof's state is not known then, so its wind is left blank "
        "too).",
        f"WP goes through the converter: no early moneyline exists for {seasons}, so the WP "
        "early, day-before and closing market log-odds all come from the spread through the "
        "converter's prior-only season slope (log-odds = slope x spread). The first owned "
        f"season, {STUDY_SEASONS[0]}, has no earlier season to fit a slope on, so its "
        f"{_count(result, 'wp_no_prior_fold_converter')} WP games are left out, as in Study "
        "1a.",
        f"{neutral} neutral-site games other than Super Bowls are left out, because their "
        "site (which home field and indoor wind rest on) may have been set after the early "
        "capture, and silver holds no announcement time.",
        "A possible imputation residual: where the gold build filled a missing kept input, "
        "its league-wide fallback may count a same-week Thursday game's pre-game row, whose "
        "values rest only on earlier weeks. How many values this touches was not measured.",
        "The Prediction Tracker's open and midweek lines are left out (owner ruling 2, "
        "2026-11-06): they carry no posting time, so the inputs cannot be fenced to them.",
        f"The slippage share (ATS and O/U) is the share of the close test's games where the "
        f"close slope x (model - early) is at least {SLIPPAGE_POINTS} points: how often a "
        "predicted move would clear the half-point of slippage. It is reported, and decides "
        "nothing.",
    ]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _number(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _move_rows(result: Study1bResult) -> list[list[str]]:
    rows = []
    for t in TARGETS:
        for c in COMPARISONS:
            test: MoveTest = getattr(result.verdicts[t], c)
            info = test.info
            rows.append(
                [
                    _SHORT_NAME[t],
                    _COMPARISON_NAME[c],
                    f"{test.games:,}",
                    "n/a" if info is None else f"{info.coef:+.4f}",
                    "n/a" if info is None else f"{info.se:.4f}",
                    "n/a" if info is None else f"{info.p_one_sided:.3g}",
                    "yes" if _clears(test) else "no",
                    _number(test.mean_abs_move, 3),
                    _number(test.mean_abs_departure, 3),
                ]
            )
    return rows


def _share_text(target: str, verdict: TargetVerdict) -> str:
    share = verdict.share_clearing_slippage
    return f"{_SHORT_NAME[target]} {'n/a' if share is None else f'{share:.0%}'}"


def _move_section(result: Study1bResult) -> list[str]:
    notes = [
        f"- {_SHORT_NAME[t]} against {_COMPARISON_NAME[c]}: {getattr(result.verdicts[t], c).note}"
        for t in TARGETS
        for c in COMPARISONS
        if getattr(result.verdicts[t], c).info is None
    ]
    shares = ", ".join(_share_text(t, result.verdicts[t]) for t in TARGETS if t != "wp")
    return [
        "## The move tests",
        "",
        "Slope of (later line - early line) on (model - early line), one-sided (H1: slope "
        f"> 0). 'Clears' means a positive slope at one-sided p < {PASS_ALPHA}. Mean abs move "
        "is the mean absolute (later - early); mean abs model - early is the model's mean "
        "absolute disagreement with the early line (ATS and O/U in points, WP in market "
        "log-odds).",
        "",
        *_table(
            [
                "Target",
                "Against",
                "Games",
                "Slope",
                "SE",
                "One-sided p",
                "Clears",
                "Mean abs move",
                "Mean abs model - early",
            ],
            _move_rows(result),
        ),
        f"Share of the close test's games whose predicted move reaches {SLIPPAGE_POINTS} "
        f"points (reported only): {shares}.",
        "",
        *(["Tests that could not run:", "", *notes, ""] if notes else []),
    ]


def _summed(result: Study1bResult, prefix: str) -> str:
    names = [name for name in result.counts if name.startswith(prefix)]
    return f"{sum(result.counts[n] for n in names):,}" if names else "n/a"


def _funnel_section(result: Study1bResult) -> list[str]:
    rows = []
    for t in TARGETS:
        verdict = result.verdicts[t]
        missing = " / ".join(
            _count(result, f"{t}_{c}_missing_number") for c in COMPARISONS
        )
        tested = " / ".join(f"{getattr(verdict, c).games:,}" for c in COMPARISONS)
        rows.append(
            [
                _SHORT_NAME[t],
                _count(result, "study_season_games"),
                _summed(result, "lines_excluded_"),
                _count(result, "games_with_lines"),
                _summed(result, "inputs_excluded_"),
                _count(result, f"{t}_inputs_without_prediction"),
                _count(result, "wp_no_prior_fold_converter") if t == "wp" else "-",
                missing,
                tested,
            ]
        )
    return [
        f"## From the {STUDY_SEASONS[0]}-{STUDY_SEASONS[-1]} schedule to the games tested",
        "",
        "Each row reads left to right: the games with lines, less those left out at the "
        "inputs, those with no prediction, those with no prior-season slope (WP) and those "
        "missing a number, are the games tested. Pairs are close / day-before.",
        "",
        *_table(
            [
                "Target",
                "Games in the seasons",
                "Left out: lines",
                "Games with lines",
                "Left out: inputs",
                "No prediction",
                "No prior-season slope",
                "Missing a number",
                "Tested",
            ],
            rows,
        ),
    ]


def render_report(result: Study1bResult) -> str:
    """The report: the summary (at most eight lines), the notes, the tables. ASCII only."""
    lines = [
        "# Study 1b: does the v2 model's disagreement with the early-week line predict "
        "the line's move?",
        "",
        "## Summary",
        "",
        *[f"- {line}" for line in summary_lines(result)],
        "",
        "## Context",
        "",
        *[f"- {note}" for note in context_notes(result)],
        "",
        f"Replay cutoff (UTC): {result.generation_cutoff_utc}. Code commit: "
        f"{result.code_commit or 'unknown'}. Seasons {STUDY_SEASONS[0]}-"
        f"{STUDY_SEASONS[-1]}, week 1 and the postseason included.",
        "",
        *_move_section(result),
        *_funnel_section(result),
        "## Games counted and excluded",
        "",
        *[f"- {name}: {n:,}" for name, n in result.counts.items()],
    ]
    return "\n".join(lines) + "\n"


def write_report(result: Study1bResult, out_dir: Path) -> None:
    """Write ``study_1b.json`` and ``report.md`` into *out_dir* (never under ``data/``).

    Both files are written as ASCII, so a non-ASCII character raises instead of landing.
    """
    target_dir = refuse_data_dir(
        out_dir, what="the Study 1b report", default_dir=DEFAULT_OUT_DIR
    )
    report = render_report(result)
    payload = json.dumps(_plain(asdict(result)), indent=2, allow_nan=False)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / JSON_FILENAME).write_text(payload + "\n", encoding="ascii")
    (target_dir / REPORT_FILENAME).write_text(report, encoding="ascii")
