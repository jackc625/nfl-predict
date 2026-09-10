"""The ONE weekly bet-selection path (Phase 31, plan 31-17; PROD-02, SPEC R4, D31-31/32).

WHAT THIS MODULE IS
-------------------
The single entry point the Friday pipeline's recommendation step delegates to. It routes a week
through ``backtest.bet_selector.BetSelector`` -- the LOCKED-2 single bet-decision source -- maps
the decision records onto ``api.cache.BET_LIST_COLUMNS``, and persists them to a DURABLE artifact
under ``outputs/bet_list/`` that the Plan 31-18 cache population step reads.

It REPLACES a legacy step body that filtered on a confidence tier with no expected value, no
sizing and no suppression, and wrote a JSON file no code ever read.

WHY THE WRITE IS AN ARTIFACT AND NOT THE LIVE CACHE (REVIEW-CACHE)
------------------------------------------------------------------
``api.cache.populate_cache`` does NOT update the live database. It opens a FRESH temporary DuckDB,
loads its source artifacts into it, then does ``db_path.unlink()`` followed by
``tmp_path.rename(db_path)``. Anything this step wrote into the live cache would be silently
DELETED by the next population run, along with every preserved forward row. Every other cache table
already follows the working pattern -- a source artifact under ``artifacts/`` or ``outputs/`` that
the build READS -- and the bet list joins it.

WHY THE ARTIFACT IS ALSO THE DURABLE HOME OF FORWARD HISTORY
-------------------------------------------------------------
A forward row is the claim that on this Friday, before these games were played, the system
recommended this bet at this size. Regenerating it from today's models would silently rewrite what
was recommended. The cache is rebuilt from scratch on every population run, so a forward row
preserved only in the cache would not survive one. The upsert is therefore keyed on
``BET_LIST_ROW_KEY`` and split on the game's OWN freeze instant (D31-18):

  * BEFORE that game's freeze -- the latest run WINS. The games have not been priced and frozen
    yet and late odds can still land, so replacing is legitimate.
  * AT OR AFTER that game's freeze -- the STORED row wins, whole. Not merely its
    ``BET_LIST_IMMUTABLE_COLUMNS``: the entire stored row is carried forward untouched, which is
    strictly stronger and additionally stops a later run restating the decision-time CLV.

IMMUTABILITY COVERS THE RECOMMENDATION FACTS ONLY (REVIEW-FWD-GRADE)
---------------------------------------------------------------------
Whole-row-forever immutability would be self-defeating: at freeze the game has not been played, so
a row that can never change afterwards stays ``pending`` forever and the self-grading honesty loop
is silently disabled -- the tracker would show a permanently zero-graded forward block, and a
reader could not tell that from "we recommended nothing that won". So a frozen row is immutable
against RE-SELECTION and mutable ONLY through :func:`grade_pending_rows`, which transitions
``grading_status`` ONE-WAY and ONE-TIME out of ``pending``.

REPLAY rows are derived and fully regenerable, so they are rebuilt whenever inputs change; only
the FORWARD provenance carries the freeze fence.

WHY THE FROZEN FIT IS READ AND NEVER RE-FIT
--------------------------------------------
The per-target EV floor ``t`` and the frozen residual SD are PRE-REGISTERED quantities swept and
fitted on tune-only data by ``backtest.profitability_2025``. This module READS them from that run's
record and refuses to invent one. There is NO silent fallback to a 0.0 floor: admitting a target
at an unswept floor is an un-pre-registered threshold, which is the failure the whole apparatus
exists to prevent (the same rule ``BetSelector.ev_floor_for`` states for its own mapping).

Run the tests:  .venv/Scripts/python.exe -m pytest tests/unit/test_weekly_bet_list.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from api.cache import (
    BET_LIST_COLUMNS,
    BET_LIST_IMMUTABLE_COLUMNS,
    BET_STATUS_LIVE,
    BET_TRACKER_BLOCK_COLUMNS,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PENDING,
    GRADING_STATUS_PUSH,
    GRADING_STATUS_WIN,
    PROVENANCE_FORWARD,
    RUN_MODE_FORWARD,
    RUN_MODE_REPLAY,
    assert_grading_transition,
    stamp_bet_list_provenance,
)
from backtest.bet_selector import BetSelector, SelectionResult
from backtest.ev_chain_constants import assign_ev_tier
from backtest.ou_divergence import (
    HIGH_TOTAL_BOUNDARY_PREHOLD,
    dedupe_odds_by_book_preference,
)
from backtest.ou_ev_chain import american_to_payout
from backtest.selector_strategies import default_strategies
from utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "BET_LIST_ARTIFACT_NAME",
    "BET_LIST_ROW_KEY",
    "BET_TRACKER_ARTIFACT_NAME",
    "DEFAULT_BET_LIST_DIR",
    "AlreadyGradedError",
    "BetListCacheSources",
    "FrozenChainFitError",
    "WeeklyChainFit",
    "build_bet_week_schedule",
    "build_weekly_candidates",
    "generate_weekly_bet_list",
    "grade_pending_rows",
    "grade_row",
    "load_frozen_chain_fit",
    "read_bet_list_artifact",
    "read_bet_list_cache_sources",
    "read_bet_tracker_artifact",
    "records_to_bet_list_frame",
    "select_weekly_bets",
    "upsert_bet_list_rows",
    "write_bet_list_artifact",
    "write_bet_tracker_artifact",
]


# ---------------------------------------------------------------------------
# Artifact locations and the row key
# ---------------------------------------------------------------------------

# ``outputs/`` is gitignored, which is the same status every other cache SOURCE artifact has and is
# correct for derived data. The durable-history claim rests on the file surviving a cache rebuild,
# not on it being in git.
DEFAULT_BET_LIST_DIR: Path = Path("outputs") / "bet_list"
BET_LIST_ARTIFACT_NAME: str = "bet_list.parquet"
BET_TRACKER_ARTIFACT_NAME: str = "bet_tracker.json"

# The upsert key. A (game, target) pair is decided ONCE per week, so this quadruple identifies a
# row uniquely; the season and week are carried in the key rather than derived from the game_id so
# a malformed identifier cannot silently merge two weeks.
BET_LIST_ROW_KEY: tuple[str, ...] = ("game_id", "season", "week", "target")

# The pre-registered run record carrying the tune-only fit. Gitignored (it is generator output),
# which is why its absence RAISES a named error instead of defaulting.
DEFAULT_CHAIN_FIT_PATH: Path = (
    Path("outputs") / "p31" / "profitability_2025_verdict.json"
)

# 1 unit = 1% of the notional bankroll (D27-09). The selector sizes in dollars; the bet list
# publishes units, so this is the ONE conversion in the mapping and it is not a metric.
DEFAULT_BANKROLL: float = 10_000.0
UNIT_FRACTION_OF_BANKROLL: float = 0.01

# The flat stake every row is graded against. 1.0 makes ``payout_flat`` a per-unit-staked profit,
# which is exactly what ``backtest.bet_tracker._flat_return_units`` sums, and it matches
# ``backtest.profitability_2025._per_bet_frame`` field for field.
FLAT_STAKE: float = 1.0

# The gold label column carrying each target's REALIZED value, taken from ``backtest.diagnose``'s
# contract rather than restated: WP the 0/1 home win, ATS the home margin, O/U the total points.
_REALIZED_LABEL_COLUMN: dict[str, str] = {
    "wp": "home_win",
    "ats": "home_margin",
    "ou": "total_points",
}

# The model column and the market columns each target's candidate frame must carry, and the silver
# odds column each market column is renamed FROM. Same contracts as
# ``backtest.profitability_2025``, restated here because this module builds a CURRENT-week frame
# from the odds SNAPSHOT rather than a historical frame from the closing odds.
_MODEL_COLUMN: dict[str, str] = {
    "wp": "model_prob",
    "ats": "model_spread",
    "ou": "model_total",
}
_MARKET_COLUMNS: dict[str, tuple[str, ...]] = {
    "wp": ("ml_home", "ml_away"),
    "ats": ("closing_spread",),
    "ou": ("closing_total",),
}
_ODDS_SOURCE_COLUMN: dict[str, dict[str, str]] = {
    "wp": {},
    "ats": {"spread": "closing_spread"},
    "ou": {"total": "closing_total"},
}

# The LINE column each target publishes on the bet list, or None for a target that has no line. A
# moneyline is a PRICE, not a line, so WP publishes None rather than a fabricated zero -- the same
# reason ``WPStrategy.side_probability`` returns None for its slipped line.
_LINE_COLUMN: dict[str, str | None] = {
    "wp": None,
    "ats": "closing_spread",
    "ou": "closing_total",
}

CANONICAL_TARGETS: tuple[str, str, str] = ("wp", "ats", "ou")


class FrozenChainFitError(RuntimeError):
    """The pre-registered tune-only fit could not be resolved, so no week can be selected.

    Raised rather than defaulted. Every quantity it carries -- the per-target EV floor, the frozen
    residual SD, the prior-season walk-forward bias -- was fitted on tune-only data under a rule
    frozen before any 2025 number existed. Substituting a default would admit bets at a threshold
    nobody swept for.
    """


class AlreadyGradedError(ValueError):
    """A row whose ``grading_status`` is already terminal was handed to the grader again.

    The transition is ONE-WAY and ONE-TIME (REVIEW-FWD-GRADE). ``api.cache.assert_grading_transition``
    permits an idempotent re-assertion of the SAME terminal status so a re-run of a batch grader is
    safe; this error is the stricter single-row rule, so a caller that reaches a settled bet at all
    is told rather than quietly allowed to restate a result.
    """


@dataclass(frozen=True)
class WeeklyChainFit:
    """One target's PRE-REGISTERED tune-only fit, read and never re-derived.

    Attributes:
        target: The canonical target code.
        ev_floor_t: The pre-registered per-target EV floor.
        frozen_sd: The frozen residual SD, or None for WP, which fits none by design (D31-07).
        season_bias_by_season: The prior-season walk-forward bias per season.
        calibration_gate_passed: The TUNE-split calibration gate verdict, or None when no gate was
            run for this target. CARRIED (WR-12) because without it ``build_strategies`` had no way
            to know the gate had failed, so ``WPStrategy.fallback_fired`` was always False and the
            registered fallback could not fire on the weekly path AT ALL -- inverting the D31-07
            guarantee that it "cannot fire silently" into "cannot fire, silently".
        fallback_trigger: The reason string the run recorded, or None. Carried alongside the
            verdict so a reconstructed gate stamps the SAME trigger onto every weekly decision
            record that the profitability run stamped onto its own.
    """

    target: str
    ev_floor_t: float
    frozen_sd: float | None
    season_bias_by_season: dict[int, float]
    calibration_gate_passed: bool | None = None
    fallback_trigger: str | None = None


# ---------------------------------------------------------------------------
# The frozen fit
# ---------------------------------------------------------------------------


def load_frozen_chain_fit(
    path: Path | str = DEFAULT_CHAIN_FIT_PATH,
) -> dict[str, WeeklyChainFit]:
    """Read the pre-registered per-target tune-only fit from the profitability run record.

    Args:
        path: The run record written by ``backtest.profitability_2025``.

    Returns:
        ``{target -> WeeklyChainFit}`` for all three canonical targets.

    Raises:
        FrozenChainFitError: when the record is absent, unreadable, or missing a target's fit.
            Never a default: see the class docstring.
    """
    fit_path = Path(path)
    if not fit_path.exists():
        msg = (
            f"the pre-registered tune-only fit is not on disk at {fit_path.as_posix()}; the "
            "weekly bet list cannot be selected without the frozen per-target EV floor and "
            "residual SD. Regenerate it with `python -m backtest.profitability_2025`. There is NO "
            "fallback: a defaulted floor would admit bets at a threshold nobody swept for."
        )
        raise FrozenChainFitError(msg)

    try:
        record = json.loads(fit_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"the tune-only fit at {fit_path.as_posix()} could not be read: {exc}"
        raise FrozenChainFitError(msg) from exc

    tune_fit = record.get("tune_fit")
    if not isinstance(tune_fit, dict):
        msg = (
            f"{fit_path.as_posix()} carries no 'tune_fit' block; it is not a profitability run "
            "record, or it predates the block this module reads."
        )
        raise FrozenChainFitError(msg)

    fits: dict[str, WeeklyChainFit] = {}
    for target in CANONICAL_TARGETS:
        block = tune_fit.get(target)
        if not isinstance(block, dict) or "ev_floor_t" not in block:
            msg = (
                f"{fit_path.as_posix()} carries no tune-only fit for target {target!r}; every "
                "registered target needs its own pre-registered EV floor."
            )
            raise FrozenChainFitError(msg)
        frozen_sd = block.get("frozen_sd")
        gate_passed = block.get("calibration_gate_passed")
        # The trigger lives on the per-target VERDICT record, not in ``tune_fit`` -- the run writes
        # the gate verdict in one place and the reason it fired in another. Read both, so a
        # reconstructed gate stamps the same trigger the profitability run stamped (WR-12).
        target_record = record.get("targets")
        trigger = None
        if isinstance(target_record, dict) and isinstance(
            target_record.get(target), dict
        ):
            trigger = target_record[target].get("fallback_trigger")
        fits[target] = WeeklyChainFit(
            target=target,
            ev_floor_t=float(block["ev_floor_t"]),
            frozen_sd=None if frozen_sd is None else float(frozen_sd),
            season_bias_by_season={
                int(season): float(bias)
                for season, bias in (block.get("season_bias_by_season") or {}).items()
            },
            calibration_gate_passed=(
                None if gate_passed is None else bool(gate_passed)
            ),
            fallback_trigger=None if trigger is None else str(trigger),
        )
    return fits


def _require_season_covered(fits: dict[str, WeeklyChainFit], season: int) -> None:
    """Refuse a season the pre-registered walk-forward bias does not cover.

    The bias is a per-season quantity estimated from STRICTLY PRIOR realized residuals. A season
    outside the fitted mapping has no bias, and the two line strategies raise by name when asked
    for one. Refusing HERE names the whole problem once, before a week is half-selected, and says
    what to re-run -- rather than surfacing as a per-candidate ValueError deep inside a strategy.
    """
    # WP is exempt ONLY while its registered fallback is inactive (WR-12). When the gate failed,
    # ``WPStrategy._p_home`` consults ``season_bias_for`` for every candidate and raises by name
    # deep inside the strategy if the season is uncovered -- which is exactly the per-candidate
    # failure this function exists to turn into one named refusal up front.
    uncovered = [
        target
        for target, fit in fits.items()
        if (target != "wp" or wp_fallback_is_active(fits))
        and season not in fit.season_bias_by_season
    ]
    if uncovered:
        covered = sorted(
            {s for fit in fits.values() for s in fit.season_bias_by_season}
        )
        msg = (
            f"the pre-registered walk-forward bias does not cover season {season} for target(s) "
            f"{sorted(uncovered)}; the fitted seasons are {covered}. Re-run "
            "`python -m backtest.profitability_2025` so the bias is estimated for this season "
            "from STRICTLY PRIOR residuals. No bias is invented here: a raw biased total is "
            "exactly what the walk-forward correction exists to remove (D27-07)."
        )
        raise FrozenChainFitError(msg)


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _load_week_schedule(season: int, week: int, silver_dir: Path) -> pd.DataFrame:
    """The week's schedule, carrying the ``gameday`` the per-game freeze fence is measured from.

    READ ONLY. The schedule -- not the odds join -- is the universe's spine (D31-19): a game
    absent from it could not be reported as suppressed at all.
    """
    games_path = silver_dir / "games.parquet"
    if not games_path.exists():
        msg = f"cannot build the weekly universe -- schedule missing: {games_path.as_posix()}"
        raise FileNotFoundError(msg)

    games = pd.read_parquet(games_path)
    week_games = games[(games["season"] == season) & (games["week"] == week)].copy()
    if week_games.empty:
        msg = (
            f"no scheduled games for {season} week {week} in "
            f"{games_path.as_posix()}; an empty universe is refused rather than published as a "
            "week in which nothing was recommended."
        )
        raise ValueError(msg)

    kickoff = pd.to_datetime(week_games["kickoff_et"], utc=True)
    week_games["gameday"] = kickoff.dt.tz_convert("US/Eastern").dt.strftime("%Y-%m-%d")
    return week_games[["game_id", "season", "week", "gameday"]].reset_index(drop=True)


def build_weekly_candidates(
    season: int,
    week: int,
    *,
    artifacts_dir: Path = Path("artifacts"),
    gold_dir: Path = Path("data/gold"),
    silver_dir: Path = Path("data/silver"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the deployed artifacts over one week and join the stored odds snapshot. READ ONLY.

    Mirrors ``backtest.profitability_2025._load_candidate_frames`` one week at a time, with one
    deliberate difference: the market side comes from ``data/silver/odds_snapshot.parquet`` -- the
    FREEZE-time snapshot the Friday run captures -- carrying its ``snapshot_ts`` and its
    provenance columns, so the selector's freshness fence and its OUM-06 provenance hard-fail both
    have something real to judge.

    Returns:
        ``(candidates, schedule)``. ``candidates`` carries one row per (game, target) for which a
        model output exists; ``schedule`` is the week's spine for the universe build.
    """
    from backtest.diagnose import score_deployed_artifacts

    schedule = _load_week_schedule(season, week, silver_dir)
    game_ids = set(schedule["game_id"])

    odds_path = silver_dir / "odds_snapshot.parquet"
    if not odds_path.exists():
        msg = f"cannot price the weekly bet list -- odds snapshot missing: {odds_path.as_posix()}"
        raise FileNotFoundError(msg)
    odds = pd.read_parquet(odds_path)
    # The book is chosen BY NAME, not by parquet row order (WR-08). ``keep="first"`` picked
    # whichever row happened to appear first in the file, so appending a second book's row for a
    # game -- or any rewrite that changed row order -- silently changed which book's price the
    # published bet was struck at, with nothing on the record to attribute the change to.
    odds = dedupe_odds_by_book_preference(odds[odds["game_id"].isin(game_ids)])

    frames: list[pd.DataFrame] = []
    for target in CANONICAL_TARGETS:
        gold_path = gold_dir / f"features_{target}.parquet"
        if not gold_path.exists():
            msg = f"cannot select {target!r} -- gold matrix missing: {gold_path.as_posix()}"
            raise FileNotFoundError(msg)
        gold = pd.read_parquet(gold_path)
        gold = gold[(gold["season"] == season) & (gold["week"] == week)].copy()
        if gold.empty:
            msg = (
                f"gold matrix {gold_path.as_posix()} has no rows for {season} week {week}; the "
                "week cannot be selected for target " + repr(target)
            )
            raise ValueError(msg)

        scored = score_deployed_artifacts(
            target, gold_df=gold, artifacts_dir=artifacts_dir
        )
        keep = ["game_id", "snapshot_ts", "ml_home", "ml_away", "spread", "total"]
        keep += [c for c in ("sportsbook", "is_live") if c in odds.columns]
        keep += [
            c
            for c in (
                "spread_ju_home",
                "spread_ju_away",
                "total_over_ju",
                "total_under_ju",
            )
            if c in odds.columns
        ]
        merged = scored.merge(odds[keep], on="game_id", how="left")
        merged = merged.rename(columns=dict(_ODDS_SOURCE_COLUMN[target]))
        merged["target"] = target
        frames.append(merged)

    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates.merge(
        schedule[["game_id", "gameday"]], on="game_id", how="left"
    )
    logger.info(
        "Built weekly candidates",
        season=season,
        week=week,
        n_scheduled=len(schedule),
        n_candidates=len(candidates),
    )
    return candidates, schedule


def require_frozen_sd(fit: WeeklyChainFit) -> float:
    """The frozen residual SD, REFUSING an absent, non-finite or non-positive one (WR-05).

    ``load_frozen_chain_fit`` permits ``frozen_sd`` to be absent -- ``block.get("frozen_sd")``
    yields ``None`` -- and ``build_strategies`` used to write ``float(fit.frozen_sd or 0.0)``,
    which is silent in exactly the case that matters. A zero SD reaches ``calibrated_p_cover`` /
    ``calibrated_p_over``, where ``z = (line - corrected) / sd`` divides by zero, yielding
    ``+/-inf`` -> ``norm.cdf`` -> 0.0 or 1.0 -> clipped to ``P_OVER_CLIP``. Every candidate then
    prices at 0.999 or 0.001, every 0.999 clears any EV floor, and Kelly stakes it at the 5%
    per-bet cap: a week's bet list maximally staked on a model that produced no probability at
    all, with nothing raising.

    The PRICING path already refuses this input by name (``ats_ev_chain.py``: "the ATS chain
    requires a frozen residual SD ... ChainFit.frozen_sd is None"), so without this the SELECTION
    path was strictly weaker than the pricing path it is supposed to mirror. ``or 0.0`` also
    swallowed a legitimately-0.0 stored value, which is the same fatal input arriving a different
    way.

    Args:
        fit: The per-target frozen fit.

    Returns:
        The SD as a positive, finite float.

    Raises:
        FrozenChainFitError: naming the target and the offending value.
    """
    value = fit.frozen_sd
    if value is None or not math.isfinite(value) or value <= 0:
        msg = (
            f"target {fit.target!r} has no usable frozen residual SD ({value!r}); a zero, "
            "absent or non-finite SD divides by zero in the calibrated-probability converter "
            "and clips every candidate to the probability bound, which Kelly then stakes at the "
            "per-bet cap. Re-run `python -m backtest.profitability_2025` so the tune-only fit "
            "carries one."
        )
        raise FrozenChainFitError(msg)
    return float(value)


def wp_fallback_is_active(fits: Mapping[str, WeeklyChainFit]) -> bool:
    """True when the run record says WP's TUNE-split calibration gate FAILED (WR-12).

    ``None`` -- no gate was run -- is NOT a failure and is the D31-07 default path. Only an
    explicit ``False`` activates the registered fallback.
    """
    return fits["wp"].calibration_gate_passed is False


def _wp_gate_from_fit(fit: WeeklyChainFit) -> Any | None:
    """Rebuild the ``WPGateResult`` the profitability run recorded, or None on the default path.

    The weekly path reads a run RECORD, not a live gate, so the report body is not available and
    is reconstructed as an explicit marker rather than an empty dict pretending to be one. Only
    the three fields ``WPStrategy`` actually reads -- ``passed``, ``fallback_fired`` and
    ``fallback_trigger`` -- are load-bearing here.
    """
    if fit.calibration_gate_passed is None:
        return None
    from backtest.wp_ev_chain import WPGateResult

    passed = bool(fit.calibration_gate_passed)
    return WPGateResult(
        passed=passed,
        report={
            "source": "reconstructed from the profitability run record; the bin table lives on "
            "that artifact, not here"
        },
        fallback_fired=not passed,
        fallback_trigger=fit.fallback_trigger,
    )


def build_strategies(fits: dict[str, WeeklyChainFit]) -> list[Any]:
    """The three registered strategies, built ONCE from the frozen fit.

    Built once and shared between selection and grading, so a bet is graded under exactly the rule
    it was selected under. Two independent constructions could drift apart on a parameter and the
    disagreement would show up as a mis-graded result rather than as an error.

    The two line targets' residual SDs go through :func:`require_frozen_sd`, which REFUSES an
    absent or non-positive value rather than substituting 0.0 (WR-05).
    """
    return default_strategies(
        ou_frozen_sd=require_frozen_sd(fits["ou"]),
        ou_season_bias_by_season=fits["ou"].season_bias_by_season,
        ats_frozen_sd=require_frozen_sd(fits["ats"]),
        ats_season_bias_by_season=fits["ats"].season_bias_by_season,
        wp_season_bias_by_season=fits["wp"].season_bias_by_season,
        # The gate verdict the run RECORDED, carried through (WR-12). Without it ``wp_gate``
        # defaulted to None, so ``WPStrategy.fallback_fired`` was always False, ``_p_home`` always
        # returned the deployed probability unchanged, and every weekly row was stamped
        # ``fallback_fired = False`` -- a registered fallback that could not fire at all, silently,
        # which is the inverse of the D31-07 guarantee.
        wp_gate=_wp_gate_from_fit(fits["wp"]),
        high_total_boundary=float(HIGH_TOTAL_BOUNDARY_PREHOLD),
    )


def select_weekly_bets(
    candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    fits: dict[str, WeeklyChainFit],
    *,
    strategies: list[Any] | None = None,
    bankroll: float = DEFAULT_BANKROLL,
) -> SelectionResult:
    """Route the week through the SINGLE bet-decision source (BET-01, LOCKED-2).

    Every pricing, admission, sizing and grading decision happens inside ``BetSelector.select``.
    This function supplies the frozen inputs and the schedule that makes the universe complete;
    it prices nothing of its own, which is what keeps exactly one bet-decision path in the tree.
    """
    if strategies is None:
        strategies = build_strategies(fits)
    selector = BetSelector(
        frozen_sd=float(fits["ou"].frozen_sd or 0.0),
        season_bias_by_season={},
        ev_floor_t={target: fit.ev_floor_t for target, fit in fits.items()},
        bankroll=bankroll,
        high_total_boundary=float(HIGH_TOTAL_BOUNDARY_PREHOLD),
        strategies=strategies,
    )
    return selector.select(candidates, scheduled_games=schedule)


# ---------------------------------------------------------------------------
# The selector-record -> bet_list mapping
# ---------------------------------------------------------------------------


def _instant_text(value: Any) -> str | None:
    """Render a timestamp for the bet list's VARCHAR instant columns without re-deriving it."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value)
    return text or None


def _row_from_record(
    record: dict[str, Any],
    *,
    status: str,
    rejection_reason: str | None,
    ev_floor_t: float,
    unit: float,
) -> dict[str, Any]:
    """Map ONE selector decision record onto the locked 28-column bet_list schema.

    The only transforms are a UNIT conversion (dollars -> units) and the two honesty labels, which
    are stamped separately by ``stamp_bet_list_provenance``. NO metric is re-derived: the EV, the
    calibrated P(side), the line, the slipped line and the price are carried through unchanged.

    A SUPPRESSED row was never priced, so its ``ev_tier`` is None -- ``assign_ev_tier`` raises on a
    non-finite or below-floor EV by design, and tiering a bet nobody made would claim a band it was
    never in.
    """
    target = str(record["target"])
    line_column = _LINE_COLUMN[target]
    per_bet_ev = record.get("per_bet_ev")
    kelly_stake = record.get("kelly_stake")

    ev_tier: str | None = None
    if status == BET_STATUS_LIVE and per_bet_ev is not None:
        ev_tier = assign_ev_tier(float(per_bet_ev), ev_floor_t)

    return {
        "game_id": record["game_id"],
        "season": int(record["season"]),
        "week": int(record["week"]),
        "target": target,
        "bet_side": record.get("bet_side"),
        "model_value": record.get(_MODEL_COLUMN[target]),
        "market_value": None if line_column is None else record.get(line_column),
        "line": None if line_column is None else record.get(line_column),
        "slipped_line": record.get("slipped_line"),
        "calibrated_p_side": record.get("calibrated_p_side"),
        "per_bet_ev": per_bet_ev,
        "stake_units": None if kelly_stake is None else float(kelly_stake) / unit,
        "ev_tier": ev_tier,
        "status": status,
        "rejection_reason": rejection_reason,
        "eligibility_label": record.get("subpop_label"),
        "snapshot_ts": _instant_text(record.get("snapshot_ts")),
        "freeze_ts": _instant_text(record.get("freeze_ts")),
        "selected_odds": record.get("selected_odds"),
        "flat_stake": FLAT_STAKE if status == BET_STATUS_LIVE else None,
        "provenance": None,  # stamped by stamp_bet_list_provenance
        "validation_type": None,  # stamped by stamp_bet_list_provenance
        "grading_status": GRADING_STATUS_PENDING,
        # The decision-time CLV the selector measured, carried onto the row unchanged. It is the
        # REPORT-ONLY model-edge CLV (D27-06) and it is NOT the forward freeze-vs-close metric --
        # the silver odds store holds exactly ONE snapshot per game, so no closing line exists to
        # compare a freeze price against. Recorded as a deferred item rather than fabricated.
        "clv": record.get("clv"),
        "outcome": None,
        "payout_flat": None,
        "realized_units": None,
        "graded_at": None,
    }


def records_to_bet_list_frame(
    result: SelectionResult,
    fits: dict[str, WeeklyChainFit],
    *,
    run_mode: str,
    bankroll: float = DEFAULT_BANKROLL,
) -> pd.DataFrame:
    """Turn a whole ``SelectionResult`` into the stamped ``BET_LIST_COLUMNS`` frame.

    Selected records become ``status='live'``; every rejected record becomes a SUPPRESSED row
    carrying its own ``rejection_reason`` from the selector's own taxonomy. The two sets are the
    EXACT partition of the universe, so a candidate is never silently dropped (SPEC R6).
    """
    unit = bankroll * UNIT_FRACTION_OF_BANKROLL
    rows: list[dict[str, Any]] = [
        _row_from_record(
            record,
            status=BET_STATUS_LIVE,
            rejection_reason=None,
            ev_floor_t=fits[str(record["target"])].ev_floor_t,
            unit=unit,
        )
        for record in result.selected
    ]
    rows += [
        _row_from_record(
            record,
            status="suppressed",
            rejection_reason=str(record["rejection_reason"]),
            ev_floor_t=fits[str(record["target"])].ev_floor_t,
            unit=unit,
        )
        for record in result.rejected
    ]

    frame = pd.DataFrame(rows, columns=pd.Index(BET_LIST_COLUMNS))
    stamped = stamp_bet_list_provenance(frame, run_mode)
    return stamped[BET_LIST_COLUMNS]


# ---------------------------------------------------------------------------
# The durable artifact: read, upsert, write
# ---------------------------------------------------------------------------


def read_bet_list_artifact(output_dir: Path = DEFAULT_BET_LIST_DIR) -> pd.DataFrame:
    """The stored bet list, or an EMPTY frame with the locked columns when none exists yet."""
    path = Path(output_dir) / BET_LIST_ARTIFACT_NAME
    if not path.exists():
        return pd.DataFrame(columns=pd.Index(BET_LIST_COLUMNS))
    stored = pd.read_parquet(path)
    missing = [c for c in BET_LIST_COLUMNS if c not in stored.columns]
    if missing:
        msg = (
            f"the stored bet list at {path.as_posix()} is missing column(s) {missing}; it was "
            "written by a different schema and is refused rather than merged."
        )
        raise ValueError(msg)
    return stored[BET_LIST_COLUMNS]


def read_bet_tracker_artifact(
    output_dir: Path = DEFAULT_BET_LIST_DIR,
) -> pd.DataFrame:
    """The stored tracker blocks, or an EMPTY frame with the locked columns when none exists yet.

    The mirror of :func:`write_bet_tracker_artifact`. JSON ``null`` is read back as ``None`` and
    left alone: a NULL ``hit_rate`` beside ``bets_graded = 0`` means the rate was NOT COMPUTED,
    which is a different claim from a measured zero, and coercing it to NaN here would erase the
    distinction the tracker exists to preserve.
    """
    path = Path(output_dir) / BET_TRACKER_ARTIFACT_NAME
    if not path.exists():
        return pd.DataFrame(columns=pd.Index(BET_TRACKER_BLOCK_COLUMNS))

    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        return pd.DataFrame(columns=pd.Index(BET_TRACKER_BLOCK_COLUMNS))

    stored = pd.DataFrame(records)
    missing = [c for c in BET_TRACKER_BLOCK_COLUMNS if c not in stored.columns]
    if missing:
        msg = (
            f"the stored bet tracker at {path.as_posix()} is missing column(s) {missing}; it was "
            "written by a different schema and is refused rather than merged."
        )
        raise ValueError(msg)
    return stored[BET_TRACKER_BLOCK_COLUMNS]


def build_bet_week_schedule(silver_dir: Path = Path("data/silver")) -> pd.DataFrame:
    """The full schedule with each game's OWN Friday-6PM-ET freeze instant. READ ONLY.

    The source of BOTH schedule-derived cache tables: ``available_bet_weeks`` (navigation) and
    ``bet_week_freeze`` (the threshold the stale-cache hard-block compares the populated-at marker
    against). Deriving them from the SCHEDULE rather than from bet rows is load-bearing in two
    separate ways:

    * a scheduled week carrying no prediction row is still reachable in the week selector; and
    * the freeze threshold exists even when the week has NO bet rows at all -- which is precisely
      the state a failed bet-list insertion leaves behind, and the state the hard-block was built
      for (REVIEW-STALE). A guard reading its own threshold from the data it is checking could not
      fire in that case.

    THE FREEZE IS PER-GAME, NOT PER-WEEK (D31-18), and it is not re-derived here: every value comes
    from ``scripts.ingest_historical_odds.get_synthetic_snapshot_ts``, the ONE source of the rule.
    A Thursday game's preceding Friday is seven days before the Friday preceding that week's Sunday
    games, so a second implementation that rounded to a week would be wrong for every Thursday
    game. ``api.cache.materialize_bet_week_freeze`` takes the per-week MAXIMUM of these values.

    The import is deferred because ``scripts.ingest_historical_odds`` imports
    ``backtest.ev_chain_constants``, which imports ``backtest.ou_monetization``, which imports
    ``backtest.bet_selector`` -- which THIS module imports at module scope. A module-level import
    would close that cycle and fail at collection. This is the same cycle break
    ``backtest.bet_selector._freshness_context`` documents, for the same reason.

    Args:
        silver_dir: The silver layer holding ``games.parquet``.

    Returns:
        A frame of ``game_id``, ``season``, ``week``, ``game_freeze_ts`` -- one row per scheduled
        game. An EMPTY frame with those columns when the schedule is absent, so a checkout without
        a silver layer still builds a cache whose ``/bets`` renders its no-current-week state
        rather than failing the population step.
    """
    from scripts.ingest_historical_odds import get_synthetic_snapshot_ts

    games_path = Path(silver_dir) / "games.parquet"
    columns = ["game_id", "season", "week", "game_freeze_ts"]
    if not games_path.exists():
        logger.warning(
            "Schedule absent -- bet navigation and freeze tables will be empty",
            path=games_path.as_posix(),
        )
        return pd.DataFrame(columns=pd.Index(columns))

    games = pd.read_parquet(
        games_path, columns=["game_id", "season", "week", "kickoff_et"]
    )
    if games.empty:
        return pd.DataFrame(columns=pd.Index(columns))

    kickoff = pd.to_datetime(games["kickoff_et"], utc=True)
    gameday = kickoff.dt.tz_convert("US/Eastern").dt.strftime("%Y-%m-%d")

    # One derivation per DISTINCT gameday rather than per game (about 1,300 against 6,500), then
    # mapped back. Same values, and it keeps the single-source rule affordable over full history.
    freeze_by_gameday = {
        day: get_synthetic_snapshot_ts(day) for day in sorted(gameday.dropna().unique())
    }

    schedule = games[["game_id", "season", "week"]].copy()
    schedule["game_freeze_ts"] = gameday.map(freeze_by_gameday)
    return schedule.dropna(subset=["game_freeze_ts"]).reset_index(drop=True)[columns]


@dataclass(frozen=True)
class BetListCacheSources:
    """The three frames ``api.cache.populate_cache`` loads into its TEMPORARY database.

    Bundled into one object so the scheduled step (``pipeline/steps.py``) and the manual recovery
    command (``scripts/populate_cache.py``) cannot read a DIFFERENT set of sources -- the manual
    command is the one the ``/bets`` refusal text tells the reader to run, so a cache it builds
    that lacks the freeze table would leave the refusal permanently unrecoverable.
    """

    bet_list: pd.DataFrame
    tracker: pd.DataFrame
    schedule: pd.DataFrame


def read_bet_list_cache_sources(
    output_dir: Path = DEFAULT_BET_LIST_DIR,
    silver_dir: Path = Path("data/silver"),
) -> BetListCacheSources:
    """Read every bet-list cache source: the durable rows, the tracker blocks and the schedule.

    THE SEAM (REVIEW-IMPORT). ``api/cache.py`` may import no ``backtest`` module, so it cannot know
    these filenames or derive a per-game freeze. This function -- called from ``pipeline/steps.py``
    and ``scripts/populate_cache.py``, both of which may -- reads them and hands over frames.

    Each reader degrades to an EMPTY frame rather than raising when its source is absent, because
    the cache population step is registered NON-CRITICAL: a raise here would degrade the whole
    Friday run for a first-ever build that simply has nothing to load yet. An absent bet list is
    then represented HONESTLY downstream -- zero rows and no populated-at marker -- which is what
    makes ``/bets`` refuse the week rather than render it as one in which nothing was recommended.
    """
    return BetListCacheSources(
        bet_list=read_bet_list_artifact(output_dir),
        tracker=read_bet_tracker_artifact(output_dir),
        schedule=build_bet_week_schedule(silver_dir),
    )


def write_bet_list_artifact(
    frame: pd.DataFrame, output_dir: Path = DEFAULT_BET_LIST_DIR
) -> Path:
    """Persist the bet list to its durable parquet artifact and return the path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / BET_LIST_ARTIFACT_NAME
    frame[BET_LIST_COLUMNS].to_parquet(path, index=False)
    logger.info("Wrote bet-list artifact", path=str(path), n_rows=len(frame))
    return path


def write_bet_tracker_artifact(
    tracker_frame: pd.DataFrame, output_dir: Path = DEFAULT_BET_LIST_DIR
) -> Path:
    """Persist the PRECOMPUTED tracker blocks as JSON records and return the path.

    JSON rather than parquet because the frame is a handful of aggregate rows whose ``None`` rate
    fields must survive the round trip as null -- the not-measured / measured-zero distinction the
    tracker exists to preserve. Parquet would promote them to NaN.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / BET_TRACKER_ARTIFACT_NAME
    records = tracker_frame.astype(object).where(tracker_frame.notna(), None)
    path.write_text(
        json.dumps(records.to_dict("records"), indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "Wrote bet-tracker artifact", path=str(path), n_blocks=len(tracker_frame)
    )
    return path


def _key_frame(frame: pd.DataFrame) -> pd.Series:
    """The row key as a single joined string, so a merge/index cannot mis-pair the parts."""
    if frame.empty:
        return pd.Series([], dtype=object)
    return frame[list(BET_LIST_ROW_KEY)].astype(str).agg("|".join, axis=1)


def _is_frozen(row: pd.Series, now: datetime) -> bool:
    """True when THIS row's own game freeze has passed, so its recommendation facts are settled.

    The fence is PER-GAME (D31-18) and applies to FORWARD rows only: a replay row is derived and
    fully regenerable, so freezing it would pin a reproduction rather than a record.
    """
    if row.get("provenance") != PROVENANCE_FORWARD:
        return False
    freeze_text = row.get("freeze_ts")
    if freeze_text is None or (
        isinstance(freeze_text, float) and math.isnan(freeze_text)
    ):
        return False
    freeze = pd.to_datetime(freeze_text, errors="coerce")
    if pd.isna(freeze):
        return False
    freeze_dt = freeze.to_pydatetime()
    if freeze_dt.tzinfo is None:
        freeze_dt = freeze_dt.replace(tzinfo=UTC)
    return now >= freeze_dt


def upsert_bet_list_rows(
    stored: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    now: datetime,
) -> pd.DataFrame:
    """Merge a fresh selection into the stored bet list under the D31-18 per-game freeze fence.

    * A stored FORWARD row whose own game freeze has PASSED wins WHOLE -- the incoming row for
      that key is discarded. Not merely its ``BET_LIST_IMMUTABLE_COLUMNS``: carrying the entire
      stored row forward is strictly stronger and additionally stops a later run restating the
      decision-time CLV, which is a grading-half column and would otherwise be overwritten.
    * Every other key takes the incoming row, because before the freeze the games have not been
      priced and frozen yet and late odds can still land.
    * A stored key the incoming selection does not mention is KEPT. A week the run did not touch
      is history, not an absence.

    Args:
        stored: The bet list already on disk (possibly empty).
        incoming: This run's freshly selected rows.
        now: The instant the freeze is judged against. Injected rather than read from the clock so
            the before-freeze and after-freeze behaviours are both testable.

    Returns:
        The merged frame in ``BET_LIST_COLUMNS`` order.
    """
    if stored.empty:
        return incoming[BET_LIST_COLUMNS].reset_index(drop=True)
    if incoming.empty:
        return stored[BET_LIST_COLUMNS].reset_index(drop=True)

    stored = stored.reset_index(drop=True)
    incoming = incoming.reset_index(drop=True)
    stored_keys = _key_frame(stored)
    incoming_keys = _key_frame(incoming)

    frozen_keys = {
        key
        for key, (_, row) in zip(stored_keys, stored.iterrows(), strict=True)
        if _is_frozen(row, now)
    }

    superseded = set(incoming_keys) - frozen_keys
    kept_stored = stored[~stored_keys.isin(superseded)]
    accepted_incoming = incoming[~incoming_keys.isin(frozen_keys)]

    merged = pd.concat(
        [kept_stored[BET_LIST_COLUMNS], accepted_incoming[BET_LIST_COLUMNS]],
        ignore_index=True,
    )
    logger.info(
        "Upserted bet-list rows",
        n_stored=len(stored),
        n_incoming=len(incoming),
        n_frozen_preserved=len(frozen_keys & set(incoming_keys)),
        n_merged=len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# The grading pass
# ---------------------------------------------------------------------------


def _payout_multiple(outcome: bool | None, selected_odds: float | None) -> float:
    """The per-unit-staked profit of a settled bet, at the price the bet was ACTUALLY struck at.

    Identical in form to ``backtest.profitability_2025._per_bet_frame``: a win pays
    ``american_to_payout(selected_odds)``, a loss pays -1.0, a push pays 0.0. A flat -110 payout on
    a -320 favourite would turn a losing bet into a winning one on paper.
    """
    if outcome is True:
        if selected_odds is None:
            msg = (
                "cannot settle a winning bet with no stored price; ``selected_odds`` is written "
                "at selection and a settled row without one is a corrupt record, not a default."
            )
            raise ValueError(msg)
        return float(american_to_payout(int(selected_odds)))
    if outcome is False:
        return -1.0
    return 0.0


def grade_row(
    row: dict[str, Any],
    outcome: bool | None,
    *,
    graded_at: datetime,
) -> dict[str, Any]:
    """Settle ONE pending row, ONE-WAY and ONE-TIME (REVIEW-FWD-GRADE, T-31-86b).

    Every ``BET_LIST_IMMUTABLE_COLUMNS`` value is carried through byte-identical; only the six
    grading columns move.

    Args:
        row: The stored bet-list row.
        outcome: True (win), False (loss) or None (push). A row whose realized value is UNKNOWN
            must not be passed here at all -- it stays ``pending``.
        graded_at: The settlement instant.

    Returns:
        A NEW row dict; the caller's row is not mutated.

    Raises:
        AlreadyGradedError: when the row's ``grading_status`` is already terminal. A later run may
            not quietly restate a result.
    """
    current = str(row.get("grading_status") or GRADING_STATUS_PENDING)
    if current != GRADING_STATUS_PENDING:
        msg = (
            f"refusing to re-grade {row.get('game_id')!r}/{row.get('target')!r}: its "
            f"grading_status is already {current!r}. The transition out of "
            f"{GRADING_STATUS_PENDING!r} is one-way and one-time, so a later run cannot restate a "
            "settled result."
        )
        raise AlreadyGradedError(msg)

    if outcome is True:
        new_status = GRADING_STATUS_WIN
    elif outcome is False:
        new_status = GRADING_STATUS_LOSS
    else:
        new_status = GRADING_STATUS_PUSH
    assert_grading_transition(current, new_status)

    multiple = _payout_multiple(outcome, row.get("selected_odds"))
    stake_units = row.get("stake_units")
    graded = dict(row)
    graded.update(
        {
            "grading_status": new_status,
            "outcome": outcome,
            "payout_flat": FLAT_STAKE * multiple,
            "realized_units": (
                None if stake_units is None else float(stake_units) * multiple
            ),
            "graded_at": graded_at,
        }
    )
    return graded


def _realized_values(
    frame: pd.DataFrame, gold_dir: Path
) -> dict[str, dict[str, float]]:
    """The realized label per (target, game_id) for every game the frame mentions. READ ONLY."""
    wanted = set(frame["game_id"].astype(str))
    realized: dict[str, dict[str, float]] = {}
    for target in CANONICAL_TARGETS:
        path = Path(gold_dir) / f"features_{target}.parquet"
        label = _REALIZED_LABEL_COLUMN[target]
        if not path.exists():
            realized[target] = {}
            continue
        gold = pd.read_parquet(path, columns=["game_id", label])
        gold = gold[gold["game_id"].astype(str).isin(wanted)]
        gold = gold[gold[label].notna()]
        realized[target] = {
            str(game_id): float(value)
            for game_id, value in zip(gold["game_id"], gold[label], strict=True)
        }
    return realized


def grade_pending_rows(
    frame: pd.DataFrame,
    strategies: dict[str, Any],
    *,
    gold_dir: Path = Path("data/gold"),
    graded_at: datetime | None = None,
    realized: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Settle every LIVE, PENDING row whose game result is now known.

    The outcome is resolved by the row's OWN target strategy -- the LOCKED
    ``_resolve_wp_outcome`` / ``_resolve_ats_outcome`` / ``_resolve_ou_outcome`` convention -- and
    is never re-implemented here. A row whose realized value is still unknown stays ``pending``,
    which is what lets the tracker say "not measured" instead of "measured zero".

    The pass FILTERS to pending rows, so re-running it is idempotent; :func:`grade_row` is the
    enforcement point that refuses a second attempt on a settled row.

    Args:
        frame: The merged bet list.
        strategies: ``{target -> TargetStrategy}``, taken from the selector's own registry.
        gold_dir: Where the realized labels are read from.
        graded_at: The settlement instant. Defaults to now (UTC).
        realized: Pre-resolved ``{target -> {game_id -> realized value}}``, used INSTEAD of reading
            gold. The injection seam the tests drive.

    Returns:
        A NEW frame; the caller's frame is not mutated.
    """
    if frame.empty:
        return frame

    settled_at = graded_at or datetime.now(tz=UTC)
    lookup = realized if realized is not None else _realized_values(frame, gold_dir)

    graded_rows: list[dict[str, Any]] = []
    n_graded = 0
    for row in frame.to_dict("records"):
        pending = str(row.get("grading_status") or "") == GRADING_STATUS_PENDING
        live = row.get("status") == BET_STATUS_LIVE
        target = str(row.get("target"))
        value = lookup.get(target, {}).get(str(row.get("game_id")))
        if not (pending and live) or value is None:
            graded_rows.append(row)
            continue

        strategy = strategies.get(target)
        if strategy is None:
            msg = (
                f"no strategy registered for target {target!r}; a stored row cannot be graded by "
                "a rule that is not present, and guessing one would book a result under the "
                "wrong target's convention."
            )
            raise ValueError(msg)

        outcome = strategy.grade(
            {
                "bet_side": row.get("bet_side"),
                "slipped_line": row.get("slipped_line"),
                "_actual_total": value,
            }
        )
        graded_rows.append(grade_row(row, outcome, graded_at=settled_at))
        n_graded += 1

    logger.info("Graded pending bet-list rows", n_rows=len(frame), n_graded=n_graded)
    return pd.DataFrame(graded_rows, columns=pd.Index(BET_LIST_COLUMNS))


# ---------------------------------------------------------------------------
# The ONE weekly entry point
# ---------------------------------------------------------------------------


def generate_weekly_bet_list(
    season: int,
    week: int,
    *,
    output_dir: Path = DEFAULT_BET_LIST_DIR,
    artifacts_dir: Path = Path("artifacts"),
    gold_dir: Path = Path("data/gold"),
    silver_dir: Path = Path("data/silver"),
    chain_fit_path: Path | str = DEFAULT_CHAIN_FIT_PATH,
    run_mode: str = RUN_MODE_FORWARD,
    bankroll: float = DEFAULT_BANKROLL,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Select one week, merge it into the durable artifact, grade what is settled, and persist.

    THE single weekly selection entry point (SPEC R4, D31-31). It writes NOTHING under ``data/``
    and opens NO connection to the live cache: the write is the artifact under ``output_dir``,
    which the Plan 31-18 population step loads into the temp cache build before the atomic swap.

    Returns:
        The merged, graded bet list -- the same frame that was written.
    """
    if run_mode not in (RUN_MODE_FORWARD, RUN_MODE_REPLAY):
        msg = (
            f"run_mode {run_mode!r} is outside the vocabulary "
            f"('{RUN_MODE_FORWARD}', '{RUN_MODE_REPLAY}')."
        )
        raise ValueError(msg)

    fits = load_frozen_chain_fit(chain_fit_path)
    _require_season_covered(fits, season)

    candidates, schedule = build_weekly_candidates(
        season,
        week,
        artifacts_dir=artifacts_dir,
        gold_dir=gold_dir,
        silver_dir=silver_dir,
    )
    # ONE registry, shared by selection and grading, so a bet is graded under exactly the rule it
    # was selected under.
    strategies = build_strategies(fits)
    result = select_weekly_bets(
        candidates, schedule, fits, strategies=strategies, bankroll=bankroll
    )
    incoming = records_to_bet_list_frame(
        result, fits, run_mode=run_mode, bankroll=bankroll
    )

    stored = read_bet_list_artifact(output_dir)
    merged = upsert_bet_list_rows(stored, incoming, now=now or datetime.now(tz=UTC))

    graded = grade_pending_rows(
        merged,
        {strategy.target: strategy for strategy in strategies},
        gold_dir=gold_dir,
    )
    write_bet_list_artifact(graded, output_dir)

    logger.info(
        "Weekly bet list generated",
        season=season,
        week=week,
        run_mode=run_mode,
        n_live=int((graded["status"] == BET_STATUS_LIVE).sum()),
        n_rows=len(graded),
    )
    return graded


# The ONE assertion this module makes about the schema it writes: the immutable half named by
# ``api.cache`` is exactly the half the freeze fence protects. Stated as an import-time check so a
# column added to one list and not the other cannot ship silently.
if set(BET_LIST_IMMUTABLE_COLUMNS) - set(BET_LIST_COLUMNS):  # pragma: no cover
    msg = "BET_LIST_IMMUTABLE_COLUMNS names a column absent from BET_LIST_COLUMNS"
    raise RuntimeError(msg)
