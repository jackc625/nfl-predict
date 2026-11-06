"""Study 1a: does the repaired (v2) model replace the current one, and what weight does each
target earn against the betting market?

Two model versions -- "variants" -- are replayed walk-forward on the same folds (one per season
2017-2024, then one per finished 2026 week; ``refit.remeasure.fold_specs``):

* ``current``: the live recipe, ``refit.remeasure.walk_forward_predictions`` reading the LIVE
  gold. Never the v2 gold: v2 columns would enter its frozen candidate pool (map T9).
* ``v2``: ``models.v2_trainer.walk_forward_v2`` reading the v2 gold.

THE INSTALL BAR. v2 goes in only if its pooled 2017-2024 loss (WP log loss, ATS and O/U mean
absolute error) is at least as good as the current recipe's on EVERY target, measured on the
games both variants predicted. No significance test; the 2026 weeks are reported, never a veto.

THE WEIGHT BAR. Per variant and target, the blend weight is fitted by
``MarketBlender.tune_weights`` on the decision-time tuning frames -- the corpus the live re-fit
uses (``extended_corpus``: the 2020-2024 Friday consensus plus the 2026 live captures; its
``market_spread`` is already home-positive and is never flipped again, map T6). The weight is
EARNED only if the information-beyond-market slope on those same frames is positive at
one-sided p < ``WEIGHT_ALPHA``; otherwise it is 0. A test that cannot run (a degenerate or
singular fit) earns 0 and its reason is recorded, so one target never stops the study.

THE CLOSE, REPORT ONLY. The same test against the nflverse closing line 2017-2024 (WP: the
de-vigged closing moneyline, games with both prices only). Close columns never reach a
market-probability fit (map T7) and never decide anything.

Nothing is written except ``study_1a.json`` and ``report.md``, into a directory refused if it
lies under ``data/``. ASCII only, no emoji.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from backtest.market_info import (
    InfoResult,
    information_beyond_line,
    information_beyond_probability,
)
from backtest.metrics import compute_ats_metrics, compute_ou_metrics, compute_wp_metrics
from backtest.refit_preregistration_2026 import MARKET_CONVERTER_ID, REFIT_SEASON
from data.upstream_pin import load_schedules
from features.elo_features import load_elo_snapshots
from forward_ledger.settle import load_silver_games
from models.blending import BLEND_TUNING_COLUMNS, MarketBlender
from models.blending_data import build_tuning_frames
from models.market_probability import (
    OWNED_LINE_SEASONS,
    load_market_probability_artifact,
)
from models.v2_trainer import walk_forward_v2
from refit.records import code_commit
from refit.remeasure import extended_corpus, extended_slopes, walk_forward_predictions
from utils.game_id_utils import create_standard_game_id
from utils.paths import reject_data_path
from utils.probability_utils import moneyline_to_probability

__all__ = [
    "DEFAULT_OUT_DIR",
    "INSTALL_SEASONS",
    "LIVE_GOLD_DIR",
    "TARGETS",
    "VARIANTS",
    "WEIGHT_ALPHA",
    "LossScores",
    "MarketTest",
    "Study1aResult",
    "VariantScores",
    "close_lines",
    "context_notes",
    "earned_weight",
    "install_verdict",
    "live_artifacts_dir",
    "market_test",
    "refuse_data_dir",
    "render_report",
    "run_study_1a",
    "summary_lines",
    "target_loss",
    "write_report",
]

#: The seasons whose pooled walk-forward loss decides the install.
INSTALL_SEASONS: tuple[int, ...] = tuple(range(2017, 2025))

#: A weight is earned only below this one-sided p on a positive slope.
WEIGHT_ALPHA: float = 0.05

TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

#: The two model versions compared, as named in every result and report.
VARIANTS: tuple[str, ...] = ("current", "v2")

#: The live gold, which the current-recipe variant (and only it) reads.
LIVE_GOLD_DIR = Path("data/gold")
DEFAULT_OUT_DIR = Path("outputs/study_1a")
JSON_FILENAME = "study_1a.json"
REPORT_FILENAME = "report.md"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PREDICTION_COLUMNS: list[str] = ["game_id", "season", "prediction", "actual"]
_CLOSE_LINE: dict[str, str] = {
    "wp": "close_home_prob",
    "ats": "close_spread",
    "ou": "close_total",
}
_TARGET_NAME: dict[str, str] = {
    "wp": "WP (win / moneyline)",
    "ats": "ATS (spread)",
    "ou": "O/U (total)",
}
_SHORT_NAME: dict[str, str] = {"wp": "WP", "ats": "ATS", "ou": "O/U"}
_METRIC: dict[str, str] = {"wp": "log loss", "ats": "MAE", "ou": "MAE"}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketTest:
    """One variant's information-beyond-market test against one line, with both losses.

    Attributes:
        info: The slope on (model - market), or None when the test could not run.
        note: Why ``info`` is None; empty otherwise.
        games: The rows the test was given.
        market_loss: The market line alone, scored as a prediction (None with no rows).
        model_loss: The model alone on the same rows (None with no rows).
    """

    info: InfoResult | None
    note: str
    games: int
    market_loss: float | None
    model_loss: float | None


@dataclass(frozen=True)
class LossScores:
    """One variant's walk-forward loss on one target, on the games both variants predicted.

    A residual is prediction minus the realized value: positive means the model expected more
    for the home side (WP, ATS) or more points (O/U) than happened -- the home lean.
    """

    pooled: float
    pooled_games: int
    by_season: dict[int, float]
    residual_by_season: dict[int, float]
    games_by_season: dict[int, int]
    weeks_2026: float | None
    weeks_2026_games: int


@dataclass(frozen=True)
class VariantScores:
    """Everything Study 1a measured for one variant on one target."""

    losses: LossScores
    decision: MarketTest
    tuned_weight: float
    earned_weight: float
    close: MarketTest


@dataclass(frozen=True)
class Study1aResult:
    """The study's verdicts and the numbers behind them.

    Attributes:
        generation_cutoff_utc: The replay's cutoff, ISO text.
        scores: ``{target: {variant: VariantScores}}``.
        install: True when v2 replaces the current models (the install bar).
        weights: v2's EARNED weight per target.
        current_weights: The current models' earned weight per target (the weights that apply
            when v2 does not go in).
        counts: Games counted and excluded, by name.
        applied_weights: The weights that apply after the verdict: v2's earned weights if v2
            goes in, else the current models'. ``weights`` is kept unchanged beside it.
        code_commit: The code's commit (``+dirty`` marks an uncommitted change), or None.
        v2_gold_manifest: The contents of the v2 gold's ``manifest.json``, or None.
    """

    generation_cutoff_utc: str
    scores: dict[str, dict[str, VariantScores]]
    install: bool
    weights: dict[str, float]
    current_weights: dict[str, float]
    counts: dict[str, int]
    applied_weights: dict[str, float] = field(default_factory=dict)
    code_commit: str | None = None
    v2_gold_manifest: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# The two bars
# ---------------------------------------------------------------------------


def install_verdict(current: Mapping[str, float], v2: Mapping[str, float]) -> bool:
    """True only if v2's loss is at most the current recipe's on every target.

    Raises:
        ValueError: a target's loss is missing from either side.
    """
    missing = [t for t in TARGETS if t not in current or t not in v2]
    if missing:
        msg = f"the install bar needs a loss for every target; missing {missing}"
        raise ValueError(msg)
    return all(v2[t] <= current[t] for t in TARGETS)


def earned_weight(tuned_weight: float, info: InfoResult | None) -> float:
    """*tuned_weight* if the slope is positive at one-sided p < ``WEIGHT_ALPHA``, else 0."""
    if info is not None and info.coef > 0 and info.p_one_sided < WEIGHT_ALPHA:
        return float(tuned_weight)
    return 0.0


# ---------------------------------------------------------------------------
# Losses and market tests
# ---------------------------------------------------------------------------


def target_loss(target: str, realized: ArrayLike, predicted: ArrayLike) -> float:
    """WP log loss, or ATS / O/U mean absolute error, of *predicted* against *realized*."""
    if target == "wp":
        return compute_wp_metrics(np.asarray(realized), np.asarray(predicted))[
            "log_loss"
        ]
    if target == "ats":
        return compute_ats_metrics(np.asarray(realized), np.asarray(predicted))["mae"]
    if target == "ou":
        return compute_ou_metrics(np.asarray(realized), np.asarray(predicted))["mae"]
    msg = f"unknown target {target!r}; expected one of {list(TARGETS)}"
    raise ValueError(msg)


def market_test(
    target: str, realized: ArrayLike, market: ArrayLike, model: ArrayLike
) -> MarketTest:
    """The information-beyond-market slope of *model* over *market*, and both losses.

    WP regresses the home win on the model's log-odds departure from the market probability;
    ATS and O/U regress (realized - line) on (model - line). A fit that cannot run is recorded
    as a note, never raised: it means "no information", which earns no weight.
    """
    realized_a = np.asarray(realized, dtype=float)
    market_a = np.asarray(market, dtype=float)
    model_a = np.asarray(model, dtype=float)
    if len(realized_a) == 0:
        return MarketTest(None, "no games to test", 0, None, None)
    try:
        if target == "wp":
            info = information_beyond_probability(realized_a, market_a, model_a)
        else:
            info = information_beyond_line(realized_a, market_a, model_a)
        note = ""
    except ValueError as exc:
        info, note = None, str(exc)
    return MarketTest(
        info=info,
        note=note,
        games=len(realized_a),
        market_loss=target_loss(target, realized_a, market_a),
        model_loss=target_loss(target, realized_a, model_a),
    )


def _narrow(frame: pd.DataFrame) -> pd.DataFrame:
    """A walk-forward frame reduced to ``game_id, season, prediction, actual``."""
    out = frame.loc[:, _PREDICTION_COLUMNS].copy()
    out["game_id"] = out["game_id"].astype(str)
    out["season"] = out["season"].astype(int)
    return out


def _loss_scores(target: str, frame: pd.DataFrame) -> LossScores:
    """Pooled 2017-2024, per-season and 2026-weeks loss, and the per-season mean residual."""
    pooled = frame.loc[frame["season"].isin(INSTALL_SEASONS)]
    by_season: dict[int, float] = {}
    residual: dict[int, float] = {}
    games: dict[int, int] = {}
    for season, rows in frame.groupby("season", sort=True):
        key = int(str(season))
        by_season[key] = target_loss(target, rows["actual"], rows["prediction"])
        residual[key] = float((rows["prediction"] - rows["actual"]).mean())
        games[key] = len(rows)
    weeks = frame.loc[frame["season"] == REFIT_SEASON]
    return LossScores(
        pooled=target_loss(target, pooled["actual"], pooled["prediction"]),
        pooled_games=len(pooled),
        by_season=by_season,
        residual_by_season=residual,
        games_by_season=games,
        weeks_2026=(
            None
            if weeks.empty
            else target_loss(target, weeks["actual"], weeks["prediction"])
        ),
        weeks_2026_games=len(weeks),
    )


def _decision_tests(
    corpus_frame: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    slopes: dict[int, float],
) -> tuple[dict[str, tuple[MarketTest, float]], int]:
    """Per target, the test and the fitted weight on the decision-time tuning frames.

    Returns the per-target ``(test, tuned weight)`` and the WP rows dropped for lack of a
    prior-fold converter slope (the first owned season, 2020).
    """
    frames = build_tuning_frames(corpus_frame, predictions, slopes)
    tuning = MarketBlender().tune_weights(
        frames.frames, owned_seasons=(*OWNED_LINE_SEASONS, REFIT_SEASON)
    )
    out: dict[str, tuple[MarketTest, float]] = {}
    for target in TARGETS:
        model_col, market_col, realized_col = BLEND_TUNING_COLUMNS[target]
        frame = frames.frames[target]
        test = market_test(
            target, frame[realized_col], frame[market_col], frame[model_col]
        )
        out[target] = (test, float(getattr(tuning.weights, f"{target}_model_weight")))
    return out, len(frames.excluded)


# ---------------------------------------------------------------------------
# The nflverse close (report only)
# ---------------------------------------------------------------------------


def _silver_id(season: Any, week: Any, away: Any, home: Any) -> str | None:
    try:
        return create_standard_game_id(int(season), int(week), str(away), str(home))
    except ValueError:
        return None


def _fair_home_probability(home_price: Any, away_price: Any) -> float:
    """The home side's de-vigged closing probability; NaN unless both prices are usable."""
    if pd.isna(home_price) or pd.isna(away_price):
        return math.nan
    try:
        home = moneyline_to_probability(int(home_price))
        away = moneyline_to_probability(int(away_price))
    except ValueError:
        return math.nan
    return home / (home + away)


def close_lines(schedules: pd.DataFrame) -> pd.DataFrame:
    """The nflverse close per silver game id: ``close_spread``, ``close_total``, ``close_home_prob``.

    ``close_spread`` is nflverse ``spread_line`` unchanged: positive when the HOME side is
    favoured, the scale the ATS model predicts on. ``close_home_prob`` is the proportionally
    de-vigged home moneyline, NaN unless both prices are present. Rows whose ids cannot be
    mapped are left out (the caller counts them). Report only: never a fit input (map T7).

    Raises:
        ValueError: two schedule rows map to one silver game.
    """
    frame = pd.DataFrame(
        {
            "game_id": [
                _silver_id(s, w, a, h)
                for s, w, a, h in zip(
                    schedules["season"],
                    schedules["week"],
                    schedules["away_team"],
                    schedules["home_team"],
                    strict=True,
                )
            ],
            "close_spread": pd.to_numeric(schedules["spread_line"], errors="coerce"),
            "close_total": pd.to_numeric(schedules["total_line"], errors="coerce"),
            "close_home_prob": [
                _fair_home_probability(h, a)
                for h, a in zip(
                    schedules["home_moneyline"],
                    schedules["away_moneyline"],
                    strict=True,
                )
            ],
        }
    ).dropna(subset=["game_id"])
    repeated = sorted(frame.loc[frame["game_id"].duplicated(), "game_id"])
    if repeated:
        msg = f"two nflverse schedule rows map to one silver game: {repeated[:10]}"
        raise ValueError(msg)
    return frame.reset_index(drop=True)


def _close_test(
    target: str, predictions: pd.DataFrame, close: pd.DataFrame
) -> tuple[MarketTest, int]:
    """The test against the close over 2017-2024, and the games left out for lack of a price."""
    rows = predictions.loc[predictions["season"].isin(INSTALL_SEASONS)].merge(
        close, on="game_id", how="inner", validate="one_to_one"
    )
    line = _CLOSE_LINE[target]
    priced = rows.dropna(subset=[line])
    test = market_test(target, priced["actual"], priced[line], priced["prediction"])
    return test, len(rows) - len(priced)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def live_artifacts_dir(silver_dir: Path | str) -> Path:
    """The ``artifacts`` folder of the checkout that owns *silver_dir* (read only here).

    A worktree reaches the live lake through a directory junction and has no ``artifacts`` of
    its own, so the folder is found from where the silver lake really lives:
    ``<checkout>/data/silver`` -> ``<checkout>/artifacts``.
    """
    return Path(silver_dir).resolve().parent.parent / "artifacts"


def _refuse_shared_gold(v2_gold_dir: Path, live_gold_dir: Path) -> None:
    if Path(v2_gold_dir).resolve() == Path(live_gold_dir).resolve():
        msg = (
            f"the current-recipe variant must read the live gold, never the v2 gold "
            f"({v2_gold_dir}); v2 columns would enter its frozen candidate pool (map T9)"
        )
        raise ValueError(msg)


def _read_manifest(v2_gold_dir: Path) -> dict[str, Any] | None:
    """The v2 gold's ``manifest.json`` contents, or None when the build left none."""
    path = v2_gold_dir / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_study_1a(
    v2_gold_dir: Path,
    silver_dir: Path,
    generation_cutoff: datetime,
    *,
    live_gold_dir: Path = LIVE_GOLD_DIR,
    artifacts_dir: Path | None = None,
) -> Study1aResult:
    """Replay both variants walk-forward and judge the install and the weights. Writes nothing.

    Args:
        v2_gold_dir: The v2 gold matrices (``outputs/gold_v2``).
        silver_dir: The silver lake (``games``, ``odds_timeline``, ``odds_snapshot``).
        generation_cutoff: The replay's cutoff, tz-aware; nothing at or after it is measured.
        live_gold_dir: The live gold the current recipe reads.
        artifacts_dir: Where the bound WP market converter is read from; None means the
            artifacts folder beside the live lake (:func:`live_artifacts_dir`).

    Raises:
        ValueError: the two variants would read the same gold.
    """
    _refuse_shared_gold(v2_gold_dir, live_gold_dir)
    games = load_silver_games(silver_dir)
    walk_forward = {
        "current": walk_forward_predictions(
            live_gold_dir, games, generation_cutoff, snapshots=load_elo_snapshots()
        ),
        "v2": walk_forward_v2(Path(v2_gold_dir), games, generation_cutoff),
    }
    predictions = {
        variant: {target: _narrow(frames[target]) for target in TARGETS}
        for variant, frames in walk_forward.items()
    }

    corpus = extended_corpus(silver_dir, games, generation_cutoff)
    converter_dir = (
        live_artifacts_dir(silver_dir) if artifacts_dir is None else Path(artifacts_dir)
    )
    slopes = extended_slopes(
        load_market_probability_artifact(MARKET_CONVERTER_ID, converter_dir)
    )
    schedules = load_schedules(list(INSTALL_SEASONS))
    close = close_lines(schedules)

    counts: dict[str, int] = {
        "decision_corpus_games": len(corpus.frame),
        "decision_excluded_no_prelock_line": len(corpus.excluded),
        "close_schedule_games": len(schedules),
        "close_unmapped_schedule_games": len(schedules) - len(close),
    }
    for detail, n in corpus.excluded["detail"].value_counts().sort_index().items():
        counts[f"decision_excluded_no_prelock_line_{detail}"] = int(n)

    decisions: dict[str, dict[str, tuple[MarketTest, float]]] = {}
    for variant in VARIANTS:
        decisions[variant], wp_dropped = _decision_tests(
            corpus.frame, predictions[variant], slopes
        )
        counts["decision_wp_excluded_no_prior_fold_converter"] = wp_dropped

    scores: dict[str, dict[str, VariantScores]] = {}
    for target in TARGETS:
        ids = [set(predictions[v][target]["game_id"]) for v in VARIANTS]
        common = sorted(ids[0] & ids[1])
        counts[f"{target}_games_predicted_by_one_variant_only"] = len(ids[0] ^ ids[1])
        scores[target] = {}
        for variant in VARIANTS:
            frame = predictions[variant][target]
            test, tuned = decisions[variant][target]
            close_test, unpriced = _close_test(target, frame, close)
            if target == "wp":
                counts["close_wp_games_without_both_moneylines"] = unpriced
            scores[target][variant] = VariantScores(
                losses=_loss_scores(target, frame.loc[frame["game_id"].isin(common)]),
                decision=test,
                tuned_weight=tuned,
                earned_weight=earned_weight(tuned, test.info),
                close=close_test,
            )

    pooled = {
        variant: {t: scores[t][variant].losses.pooled for t in TARGETS}
        for variant in VARIANTS
    }
    install = install_verdict(pooled["current"], pooled["v2"])
    v2_weights = {t: scores[t]["v2"].earned_weight for t in TARGETS}
    current_weights = {t: scores[t]["current"].earned_weight for t in TARGETS}
    applied = v2_weights if install else current_weights
    return Study1aResult(
        generation_cutoff_utc=generation_cutoff.isoformat(),
        scores=scores,
        install=install,
        weights=v2_weights,
        current_weights=current_weights,
        counts=counts,
        applied_weights=applied,
        code_commit=code_commit(_PROJECT_ROOT),
        v2_gold_manifest=_read_manifest(Path(v2_gold_dir)),
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _loss_text(target: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}" if target == "wp" else f"{value:.3f}"


def _residual_text(target: str, value: float) -> str:
    return f"{value:+.4f}" if target == "wp" else f"{value:+.2f}"


def _info_text(test: MarketTest) -> str:
    if test.info is None:
        return f"no test result: {test.note}"
    return (
        f"slope {test.info.coef:+.3f}, one-sided p {test.info.p_one_sided:.3g}, "
        f"{test.info.n} games"
    )


def _weight_phrase(scores: VariantScores) -> str:
    return (
        f"weight {scores.earned_weight:.2f} ({_info_text(scores.decision)}; "
        f"fitted blend weight {scores.tuned_weight:.2f})"
    )


def _tests(result: Study1aResult) -> list[tuple[str, str, VariantScores]]:
    """Every (target, variant) decision test, in report order."""
    return [(t, v, result.scores[t][v]) for t in TARGETS for v in VARIANTS]


def _weeks_line(result: Study1aResult) -> str:
    """The finished 2026 games, reported only: which variant is better, per target."""
    parts = []
    for t in TARGETS:
        cur = result.scores[t]["current"].losses.weeks_2026
        new = result.scores[t]["v2"].losses.weeks_2026
        if cur is None or new is None:
            parts.append(f"{_SHORT_NAME[t]} n/a")
            continue
        verdict = "worse" if new > cur else "better" if new < cur else "level"
        parts.append(
            f"{_SHORT_NAME[t]} {verdict} ({_loss_text(t, new)} vs {_loss_text(t, cur)})"
        )
    games = result.scores[TARGETS[0]]["v2"].losses.weeks_2026_games
    return (
        f"On the {games} finished {REFIT_SEASON} games (reported only, never a veto) v2 vs "
        "current is " + "; ".join(parts) + "."
    )


def _line_trail_line(result: Study1aResult) -> str:
    """v2's loss against the betting line's own loss on the decision-time games."""
    parts = []
    for t in TARGETS:
        test = result.scores[t]["v2"].decision
        if test.model_loss is None or test.market_loss is None:
            parts.append(f"{_SHORT_NAME[t]} n/a")
            continue
        gap = "trails" if test.model_loss > test.market_loss else "beats"
        parts.append(
            f"{_SHORT_NAME[t]} {_METRIC[t]} {_loss_text(t, test.model_loss)} for v2 vs "
            f"{_loss_text(t, test.market_loss)} for the line ({gap})"
        )
    return (
        "Against the day-before line itself, on the decision-time games: "
        + "; ".join(parts)
        + "."
    )


def summary_lines(result: Study1aResult) -> list[str]:
    """The plain-English summary, at most eight lines.

    The verdict, one line per target, the finished-2026 and against-the-line comparisons, and
    what it means for bets.
    """
    pooled = {
        t: {v: result.scores[t][v].losses.pooled for v in VARIANTS} for t in TARGETS
    }
    detail = "; ".join(
        f"{_SHORT_NAME[t]} {_METRIC[t]} v2 {_loss_text(t, pooled[t]['v2'])} vs current "
        f"{_loss_text(t, pooled[t]['current'])}"
        for t in TARGETS
    )
    if result.install:
        lines = [
            "v2 goes in: on the pooled 2017-2024 replay its loss is at least as good as the "
            f"current models' on all three targets ({detail})."
        ]
    else:
        worse = [
            _SHORT_NAME[t]
            for t in TARGETS
            if not pooled[t]["v2"] <= pooled[t]["current"]
        ]
        lines = [
            "v2 does NOT go in: on the pooled 2017-2024 replay its loss is worse than the "
            f"current models' on {', '.join(worse) or 'a target'} ({detail})."
        ]
    for t in TARGETS:
        lines.append(
            f"{_SHORT_NAME[t]}: v2 earns {_weight_phrase(result.scores[t]['v2'])}; the current "
            f"model earns {_weight_phrase(result.scores[t]['current'])}."
        )
    lines.append(_weeks_line(result))
    lines.append(_line_trail_line(result))
    applied, weights = (
        ("v2", result.weights)
        if result.install
        else ("current", result.current_weights)
    )
    outcome = "; ".join(
        f"{_SHORT_NAME[t]} places no day-before bets"
        if weights[t] <= 0.0
        else f"{_SHORT_NAME[t]} bets with the model at weight {weights[t]:.2f} against "
        "the market"
        for t in TARGETS
    )
    lines.append(
        f"For betting: after the switch the {applied} models run, so their weights apply "
        f"(a weight of 0 means no day-before bets for that bet type). So: {outcome}."
    )
    return lines


def _passing_tests_note(result: Study1aResult) -> str:
    """How many of the information tests pass, and which passes were given no weight."""
    tests = _tests(result)
    passed = [
        (t, v, s)
        for t, v, s in tests
        if s.decision.info is not None
        and s.decision.info.coef > 0
        and s.decision.info.p_one_sided < WEIGHT_ALPHA
    ]
    total = len(tests)
    if not passed:
        return f"None of the {total} information tests passes, so no weight is earned."
    names = "; ".join(
        f"the {v} {_SHORT_NAME[t]} model ({_info_text(s.decision)})"
        for t, v, s in passed
    )
    zero = [
        f"the {v} {_SHORT_NAME[t]} model's fitted blend weight was {s.tuned_weight:.2f}"
        for t, v, s in passed
        if s.tuned_weight <= 0.0
    ]
    tail = f" but {'; '.join(zero)} (the blend fit chose 0)" if zero else ""
    return (
        f"{len(passed)} of the {total} information tests passes: {names};{tail}. That is "
        f"weak evidence ({len(passed)} of {total} tests)."
    )


def _lean_note(result: Study1aResult) -> str:
    """Whether v2 still leans home on ATS: the game-weighted mean residual, both variants."""

    def pooled_residual(variant: str) -> float:
        losses = result.scores["ats"][variant].losses
        seasons = [s for s in INSTALL_SEASONS if s in losses.games_by_season]
        games = sum(losses.games_by_season[s] for s in seasons)
        total = sum(
            losses.residual_by_season[s] * losses.games_by_season[s] for s in seasons
        )
        return total / games if games else math.nan

    def weeks_residual(variant: str) -> float:
        return result.scores["ats"][variant].losses.residual_by_season.get(
            REFIT_SEASON, math.nan
        )

    return (
        "The home lean on ATS (game-weighted mean residual, prediction minus result, "
        f"{INSTALL_SEASONS[0]}-{INSTALL_SEASONS[-1]}): v2 {pooled_residual('v2'):+.2f} points "
        f"vs current {pooled_residual('current'):+.2f}; {REFIT_SEASON} weeks "
        f"{weeks_residual('v2'):+.2f} vs {weeks_residual('current'):+.2f}. Where v2 is "
        "positive it still leans home: the existing after-the-fact lean correction must not "
        "be dropped on the assumption that v2 fixed it."
    )


def context_notes(result: Study1aResult) -> list[str]:
    """The caveats that follow the summary (not part of its eight-line limit)."""
    return [
        "Caveat on the install margin: the v2 inputs were fixed before the replay but were "
        "informed by the audit's measurements on the same 2017-2024 seasons, so a small "
        "optimistic bias in v2's install margin is possible.",
        _passing_tests_note(result),
        _lean_note(result),
        "Weights are judged against the day-before line (2020-2024 Friday consensus plus the "
        "2026 live captures); the nflverse close below is for reference only.",
    ]


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return [*out, ""]


def _install_section(result: Study1aResult) -> list[str]:
    rows = []
    for t in TARGETS:
        cur = result.scores[t]["current"].losses
        new = result.scores[t]["v2"].losses
        rows.append(
            [
                _TARGET_NAME[t],
                _METRIC[t],
                _loss_text(t, cur.pooled),
                _loss_text(t, new.pooled),
                f"{new.pooled - cur.pooled:+.4f}",
                str(new.pooled_games),
            ]
        )
    return [
        "## Install bar: pooled 2017-2024 walk-forward loss (lower is better)",
        "",
        *_table(["Target", "Metric", "Current", "v2", "v2 - current", "Games"], rows),
    ]


def _weeks_section(result: Study1aResult) -> list[str]:
    rows = [
        [
            _TARGET_NAME[t],
            _METRIC[t],
            _loss_text(t, result.scores[t]["current"].losses.weeks_2026),
            _loss_text(t, result.scores[t]["v2"].losses.weeks_2026),
            str(result.scores[t]["v2"].losses.weeks_2026_games),
        ]
        for t in TARGETS
    ]
    return [
        f"## {REFIT_SEASON} weeks so far (reported; never a veto)",
        "",
        *_table(["Target", "Metric", "Current", "v2", "Games"], rows),
    ]


def _season_section(result: Study1aResult) -> list[str]:
    rows = []
    for t in TARGETS:
        cur = result.scores[t]["current"].losses
        new = result.scores[t]["v2"].losses
        for season in sorted(new.by_season):
            rows.append(
                [
                    _SHORT_NAME[t],
                    str(season),
                    _loss_text(t, cur.by_season.get(season)),
                    _loss_text(t, new.by_season[season]),
                    _residual_text(t, cur.residual_by_season.get(season, math.nan)),
                    _residual_text(t, new.residual_by_season[season]),
                    str(new.games_by_season[season]),
                ]
            )
    return [
        "## Per season: loss and mean residual (the home lean)",
        "",
        "Residual = prediction minus what happened. Positive: the model expected more for the "
        "home side (WP, ATS) or more points (O/U) than happened.",
        "",
        *_table(
            [
                "Target",
                "Season",
                "Current loss",
                "v2 loss",
                "Current residual",
                "v2 residual",
                "Games",
            ],
            rows,
        ),
    ]


def _market_rows(result: Study1aResult, which: str) -> list[list[str]]:
    rows = []
    for t in TARGETS:
        for variant in VARIANTS:
            scores = result.scores[t][variant]
            test: MarketTest = getattr(scores, which)
            info = test.info
            row = [
                _SHORT_NAME[t],
                variant,
                str(test.games),
                "n/a" if info is None else f"{info.coef:+.4f}",
                "n/a" if info is None else f"{info.se:.4f}",
                "n/a" if info is None else f"{info.p_one_sided:.3g}",
                _loss_text(t, test.market_loss),
                _loss_text(t, test.model_loss),
            ]
            if which == "decision":
                row += [f"{scores.tuned_weight:.2f}", f"{scores.earned_weight:.2f}"]
            rows.append(row)
    return rows


def _market_sections(result: Study1aResult) -> list[str]:
    header = [
        "Target",
        "Variant",
        "Games",
        "Slope",
        "SE",
        "One-sided p",
        "Market-alone loss",
        "Model-alone loss",
    ]
    notes = [
        f"- {_SHORT_NAME[t]} {variant} ({which}): {getattr(result.scores[t][variant], which).note}"
        for which in ("decision", "close")
        for t in TARGETS
        for variant in VARIANTS
        if getattr(result.scores[t][variant], which).info is None
    ]
    return [
        "## Against the day-before line (decides the weights)",
        "",
        "Slope of the realized result on the model's departure from the line (WP: log-odds "
        f"departure from the market probability). Earned only if the slope is positive at "
        f"one-sided p < {WEIGHT_ALPHA}.",
        "",
        *_table(
            [*header, "Fitted blend weight", "Earned weight"],
            _market_rows(result, "decision"),
        ),
        f"## Against the nflverse close, {INSTALL_SEASONS[0]}-{INSTALL_SEASONS[-1]} "
        "(report only)",
        "",
        "WP uses the de-vigged closing moneyline where both prices exist.",
        "",
        *_table(header, _market_rows(result, "close")),
        *(["Tests that could not run:", "", *notes, ""] if notes else []),
    ]


def render_report(result: Study1aResult) -> str:
    """The report: the summary (at most eight lines) first, then the tables. ASCII only."""
    lines = [
        "# Study 1a: does v2 replace the current models, and what does each target earn?",
        "",
        "## Summary",
        "",
        *[f"- {line}" for line in summary_lines(result)],
        "",
        *[f"{note}\n" for note in context_notes(result)],
        f"Replay cutoff (UTC): {result.generation_cutoff_utc}. Walk-forward folds: one per "
        f"season {INSTALL_SEASONS[0]}-{INSTALL_SEASONS[-1]}, then one per finished "
        f"{REFIT_SEASON} week. 'Current' is the live recipe re-fitted at every fold on the "
        "live gold; 'v2' is the fixed-input linear recipe on the repaired gold.",
        "",
        *_install_section(result),
        *_weeks_section(result),
        *_season_section(result),
        *_market_sections(result),
        "## Games counted and excluded",
        "",
        *[f"- {name}: {n}" for name, n in sorted(result.counts.items())],
    ]
    return "\n".join(lines) + "\n"


def _plain(value: Any) -> Any:
    """*value* as strict JSON: text keys, lists, and None for any non-finite number."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    return value


def refuse_data_dir(out_dir: Path) -> Path:
    """The resolved *out_dir*, refusing any path under a ``data/`` tree or the live lake.

    The path is checked as written and as resolved, against this checkout's ``data/``, the
    working directory's ``data/`` and the live lake the silver junction resolves into (a
    worktree's ``data/silver`` is a junction into another checkout's ``data/``).
    """
    resolved = reject_data_path(
        out_dir, what="the Study 1a report", suggestion=f"{DEFAULT_OUT_DIR.as_posix()}/"
    )
    written = Path(os.path.abspath(out_dir))
    lake = (_PROJECT_ROOT / "data" / "silver").resolve().parent
    for root in (_PROJECT_ROOT / "data", Path.cwd() / "data", lake):
        if written.is_relative_to(os.path.abspath(root)) or resolved.is_relative_to(
            root.resolve()
        ):
            msg = (
                f"Refusing to write the Study 1a report to '{out_dir}': it is under {root}, "
                f"the data lake. Write it under '{DEFAULT_OUT_DIR.as_posix()}/' instead."
            )
            raise ValueError(msg)
    return resolved


def write_report(result: Study1aResult, out_dir: Path) -> None:
    """Write ``study_1a.json`` and ``report.md`` into *out_dir* (never under ``data/``).

    Both files are written as ASCII, so a non-ASCII character raises instead of landing.
    """
    target_dir = refuse_data_dir(out_dir)
    report = render_report(result)
    payload = json.dumps(_plain(asdict(result)), indent=2, allow_nan=False)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / JSON_FILENAME).write_text(payload + "\n", encoding="ascii")
    (target_dir / REPORT_FILENAME).write_text(report, encoding="ascii")
