"""Market blending module for combining model predictions with market odds.

Provides:
- BlendWeights: Per-target model weight configuration (WP, ATS, O/U)
- BlendConfig: Full blending configuration with probability clipping
- TuningResult: Output of the fixed-weight tuning grid search
- BlendProvenance: What a blend artifact was fitted on, recorded inside its payload
- MarketBlender: Core blending logic with three strategies:
    - WP: Log-odds space blending via logit/expit (D-01)
    - ATS: Linear interpolation in spread-point space
    - O/U: Linear interpolation in total-point space
  Plus weight tuning, the weekly edge-rate diagnostic, and artifact management.

The log-odds approach for WP ensures that blending respects the
non-linear nature of probabilities -- a 50/50 blend of 0.9 and 0.1
should yield 0.5 (which log-odds gives), not 0.5 (which linear also
gives in this symmetric case, but deviates in asymmetric cases).

For ATS and O/U, linear interpolation is appropriate because spreads
and totals live in a linear point space.

ONE FIXED WEIGHT PER TARGET. THE WEEK-VARYING BLEND IS RETIRED (D33.2-10, Plan 33.2-24)
---------------------------------------------------------------------------------------
Until Plan 33.2-24 a blender could carry a week-varying "dynamic" schedule: a per-target
sigmoid in the week of the season (six tuned parameters), auto-detected from a ``dynamic``
section of ``blend_weights.json`` and applied through a ``_dynamic_weights`` attribute on
every blend and edge path. That schedule was fitted on SYNTHETIC predictions over 2010-2017
nflverse CLOSING lines and gated on a closing-line CLV comparison. Both inputs are dead: a
closing line did not exist at a game's lock (D33.2-03), and results built on the old inputs
are not evidence (D33.2-07, the standing ruling that old baselines are dead).

So the schedule is REMOVED, not disabled. There is no constructor parameter for it, no
attribute, no branch, no CLI flag, and ``from_artifacts`` REFUSES a payload that still carries
a ``dynamic`` section (:class:`RetiredDynamicBlendError`) rather than silently reading it as
static. A week-varying shape may return later ONLY with evidence measured under the new rule
-- which is why nothing here leaves a hook for it: a dormant branch is an invitation to
restore it without the evidence.

The one weight per target is fitted on the owned PRE-LOCK lines (``models.blending_data``)
against each model's own out-of-sample outcome loss -- WP log loss, ATS and O/U absolute
error -- never on a closing-line objective (:meth:`MarketBlender.tune_weights`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from models.market_probability import (
    OWNED_LINE_SEASONS,
    MarketProbabilityArtifactError,
    load_market_probability_artifact,
    market_home_win_probability,
    oof_market_probability,
)
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class MarketProbabilityUnavailable(Exception):
    """The WP blend has no market opinion for a game, so it refuses to pretend it has one.

    WHY THIS EXISTS. ``_blend_wp_predictions`` used to log a warning and RETURN when the
    market columns were absent, leaving ``model_prob`` untouched. A silent no-blend is
    indistinguishable from a blend with weight zero, so a blend that quietly stops blending
    is a model change nobody can see -- in a system whose predictions are published.

    WHY THIS BASE CLASS. It inherits ``Exception`` and NOT ``ValueError`` / ``KeyError`` /
    ``RuntimeError`` / ``ImportError``, for the reason
    ``data.sealed_probe_log.SealedProbeLogCorrupt`` records: several call sites in this
    repository catch that tuple and degrade quietly, and this refusal degraded into "carry
    on" would restore the exact silent fallback it replaced.

    A caller that genuinely wants the model's own unblended probability asks for it by name
    through :meth:`MarketBlender.unblended_wp_predictions`, so the intent is in the call
    rather than in an absence.

    The ATS and O/U frame blends raise it too, for the same reason: a game with no market
    line has no market opinion, and a column holding blended values for some games and raw
    model values for others holds two models' outputs (A33.2-review IN-09).
    """


class DuplicateMarketRowError(ValueError):
    """A market frame handed to a blend carries more than one row for a game.

    A repeated ``game_id`` makes the left merge in :meth:`MarketBlender.blend_predictions`
    return more rows than the predictions it blends, so the blended values no longer line up
    with the games they were computed for (A33.2-review IN-09). A ``ValueError`` so the
    serving paths' ``(KeyError, FileNotFoundError)`` fallback cannot swallow it.
    """


class MarketProbabilityBindingError(Exception):
    """A blend's bound converter does not match the converter directory it names.

    Raised when ``blend_weights.json`` names a ``market_probability_*`` artifact that is
    absent, records a slope the directory disagrees with, or carries only half the binding
    -- and, at write time, when a blender with no converter bound is asked to save an
    artifact that could therefore never serve WP.

    Without the binding, nothing guarantees that the converter a blend was TUNED against is
    the converter it is SERVED with: ``artifacts/latest.json`` names four artifacts under
    SPEC R13 and the converter is deliberately not a fifth, while :meth:`from_artifacts`
    resolves only the blend directory (reviews round ``f924749``, Codex HIGH).

    Base class chosen for the same reason as :class:`MarketProbabilityUnavailable`.
    """


class RetiredDynamicBlendError(ValueError):
    """A blend payload still carries the retired week-varying ``dynamic`` section.

    Raised by :meth:`MarketBlender.from_artifacts` naming the artifact. Reading such a payload
    as static would publish a number that neither the retired rule nor the new one chose.

    WHY A ``ValueError`` AND NOT A ``KeyError`` OR ``FileNotFoundError``. Both serving paths
    catch exactly ``(KeyError, FileNotFoundError)`` around the blend load and fall back
    SILENTLY -- ``scripts/generate_current_week_predictions.apply_blending`` to unblended
    predictions, ``api/cache._load_predictions`` to NULL blended columns. A refusal raised as
    either type would be swallowed into the silent no-blend a refusal exists to prevent. A
    ``ValueError`` escapes both, so the window between this plan and the production swap
    (Plan 33.2-25), while the live blend is still the dynamic incumbent, fails LOUDLY.
    """


class BlendTuningError(Exception):
    """The blend weights cannot be tuned honestly on the frames given, so they are not tuned.

    Inherits ``Exception`` for the same reason as :class:`MarketProbabilityUnavailable`.
    """


class IncompleteBlendProvenanceError(Exception):
    """A blend payload carries some provenance keys but not all of them."""


#: The market column the WP blend reads. It is a PRE-LOCK spread, not a closing one: under
#: D33.2-03 no betting line of any timing is a MODEL input, and this is a BLEND input --
#: the market's own opinion, applied after the model has predicted.
_PRELOCK_SPREAD_COLUMN: str = "spread"

#: The column a HISTORICAL WP row carries its market side in: each game's own season's
#: PRIOR-ONLY converter slope applied to its owned pre-lock spread. The tuner, the historical
#: blend (:meth:`MarketBlender.blend_historical_wp_predictions`) and the edge diagnostic all
#: read it under this one name.
MARKET_PROB_OOF_COLUMN: str = "market_prob_oof"

#: The (model column, market column) each target's frame blend reads.
_FRAME_BLEND_COLUMNS: dict[str, tuple[str, str]] = {
    "ats": ("model_spread", "spread"),
    "ou": ("model_total", "total"),
}


def _require_one_market_row_per_game(market_df: pd.DataFrame) -> None:
    """Refuse a frame that repeats a ``game_id`` (A33.2-review IN-09)."""
    if "game_id" not in market_df.columns:
        return
    repeated = sorted(
        {str(g) for g in market_df.loc[market_df["game_id"].duplicated(), "game_id"]}
    )
    if repeated:
        msg = (
            f"the frame carries more than one row for {len(repeated)} game(s), "
            f"e.g. {repeated[:10]}. A blend merges one market row per game; a repeat "
            "would return more rows than predictions and misalign the blended values. "
            "Select one row per game before blending."
        )
        raise DuplicateMarketRowError(msg)


def home_fav_margin_from_prelock_spread(spread: np.ndarray | pd.Series) -> np.ndarray:
    """The stored pre-lock spread, on the converter's documented ``home_fav_margin`` scale.

    WHICH STORE'S CONVENTION THIS EXPECTS, stated once so no call site has to guess: the
    ``spread`` column of silver ``odds_snapshot`` and of the live ingest, which is on the
    HOME-MARGIN scale -- POSITIVE when the home team is favoured. That was MEASURED, not
    assumed (DEF-31-01): ``ml_home <= -300`` gives mean spread +10.18, ``ml_away <= -300``
    gives -9.59, and corr(spread, realized home margin) = +0.44.

    So the conversion is the identity, and this function exists anyway -- because the one
    place in the tree that knows the convention should be a NAMED place. The owned
    ``odds_timeline`` stores the OPPOSITE sign (D33.2-23, corr -0.9867) and is flipped once,
    by ``models.market_probability.timeline_spread_to_home_fav_margin``, at its two readers;
    a frame arriving here has already been through that flip or was never on that scale to
    begin with. Flipping again here would be the double flip the converter's plausibility
    band exists to catch.
    """
    return np.asarray(spread, dtype=float)


# ---------------------------------------------------------------------------
# The fixed-weight shape, decided in one place
# ---------------------------------------------------------------------------

#: Every blend artifact directory is ``{BLEND_ARTIFACT_PREFIX}_{timestamp}``. ONE shape, ONE
#: prefix, ONE place it is decided: the retired week-varying blend wrote a second prefix
#: (``blend_dynamic_*``) from a branch on its attribute, and that branch is gone with it.
BLEND_ARTIFACT_PREFIX: str = "blend"

#: The blend's one payload file, and the only file :meth:`MarketBlender.from_artifacts` opens.
BLEND_PAYLOAD_FILENAME: str = "blend_weights.json"

#: The payload schema version. "1.0" was the original static blend and "2.0" the retired
#: week-varying one; "3.0" is one fixed weight per target tuned on owned pre-lock lines, with
#: its provenance and converter binding inside the payload.
BLENDER_VERSION: str = "3.0"

#: The weights the grid search considers: 0.00 to 1.00 in steps of 0.01. BOTH ENDS ARE IN
#: THE GRID ON PURPOSE: a fitted weight of exactly 0 ("the model adds nothing over the
#: market") or exactly 1 ("the market adds nothing over the model") is a FINDING, reported
#: rather than excluded by a narrower range. The retired tuner searched [0.50, 0.70] only.
BLEND_WEIGHT_GRID: np.ndarray = np.round(np.arange(0, 101) / 100.0, 2)

#: Each target's tuning columns: (model prediction, market opinion, realized outcome). The
#: outcome columns are the trainers' own target columns, so the objective below is each
#: model's own primary metric measured on the blend of its prediction with the market's.
BLEND_TUNING_COLUMNS: dict[str, tuple[str, str, str]] = {
    "wp": ("model_prob", "market_prob_oof", "home_win"),
    "ats": ("model_spread", "market_spread", "home_margin"),
    "ou": ("model_total", "market_total", "total_points"),
}

#: The loss each target's weight minimises -- the same metric its trainer is scored on.
#: None of them reads a closing line: the market side is the pre-lock opinion and the
#: yardstick is the game's real outcome.
BLEND_OBJECTIVE_BY_TARGET: dict[str, str] = {
    "wp": "log_loss",
    "ats": "mean_absolute_error",
    "ou": "mean_absolute_error",
}

_WEIGHT_ATTR_BY_TARGET: dict[str, str] = {
    "wp": "wp_model_weight",
    "ats": "ats_model_weight",
    "ou": "ou_model_weight",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BlendWeights:
    """Per-target model weights for market blending.

    Each weight controls how much the model's prediction is trusted
    relative to the market. A weight of 1.0 means "trust the model
    completely"; 0.0 means "trust the market completely".

    Attributes:
        wp_model_weight: Model weight for Win Probability blending.
        ats_model_weight: Model weight for Against the Spread blending.
        ou_model_weight: Model weight for Over/Under blending.
    """

    wp_model_weight: float = 0.60
    ats_model_weight: float = 0.60
    ou_model_weight: float = 0.60

    def __post_init__(self) -> None:
        """Validate all weights are in [0.0, 1.0]."""
        for name in ("wp_model_weight", "ats_model_weight", "ou_model_weight"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                msg = f"{name}={value} must be in [0.0, 1.0]"
                raise ValueError(msg)


@dataclass
class EdgeThresholds:
    """Per-target edge thresholds for bet flagging.

    Edges below the threshold are not flagged. The 30% cap on
    mean per-week flagging is a DIAGNOSTIC WARNING, not a hard filter --
    games are never removed (per D-09).

    Attributes:
        wp_threshold: WP edge threshold in probability units.
        ats_threshold: ATS edge threshold in spread points.
        ou_threshold: O/U edge threshold in total points.
    """

    wp_threshold: float = 0.03
    ats_threshold: float = 1.5
    ou_threshold: float = 1.5


@dataclass
class BlendConfig:
    """Full configuration for market blending.

    Attributes:
        weights: Per-target model weights.
        clip_min: Minimum probability for logit clipping (prevents -inf).
        clip_max: Maximum probability for logit clipping (prevents +inf).
        edge_thresholds: Per-target edge thresholds for bet flagging.
    """

    weights: BlendWeights = field(default_factory=BlendWeights)
    clip_min: float = 0.001
    clip_max: float = 0.999
    edge_thresholds: EdgeThresholds = field(default_factory=EdgeThresholds)


@dataclass(frozen=True)
class TuningResult:
    """Output of the fixed-weight tuning grid search (Plan 33.2-24).

    Attributes:
        weights: The fitted weight per target -- the grid point minimising that target's
            outcome loss on its tuning frame.
        objective_by_target: The loss each weight minimised (:data:`BLEND_OBJECTIVE_BY_TARGET`).
        loss_by_target: The loss at the fitted weight.
        market_only_loss_by_target: The loss at weight 0 -- the pre-lock market alone.
        model_only_loss_by_target: The loss at weight 1 -- the model alone.
        grid_by_target: The full ``(weight, loss)`` curve per target.
        seasons_by_target: The seasons each target was tuned over.
        n_games: The tuning rows per target.
        season_best_weight_by_target: Per target, the weight each season would have chosen
            ON ITS OWN -- a stability record, not a second fit.
    """

    weights: BlendWeights
    objective_by_target: dict[str, str]
    loss_by_target: dict[str, float]
    market_only_loss_by_target: dict[str, float]
    model_only_loss_by_target: dict[str, float]
    grid_by_target: dict[str, list[tuple[float, float]]]
    seasons_by_target: dict[str, list[int]]
    n_games: dict[str, int]
    season_best_weight_by_target: dict[str, dict[int, float]]

    @property
    def boundary_targets(self) -> list[str]:
        """The tuned targets whose fitted weight is EXACTLY 0 or 1 -- a finding, reported."""
        return [
            target
            for target in self.objective_by_target
            if getattr(self.weights, _WEIGHT_ATTR_BY_TARGET[target]) in (0.0, 1.0)
        ]


@dataclass(frozen=True)
class BlendProvenance:
    """What a blend artifact was fitted ON, recorded INSIDE ``blend_weights.json``.

    One payload file, one provenance answer: a separate ``metadata.json`` would be a second
    answer about the same artifact. Plan 33.2-25's bundle validator reads these through
    :meth:`MarketBlender.from_artifacts`, the same loader the serving path uses, and its
    R13 check asserts ``gold_generation_digest`` across all four production artifacts.

    The converter binding (``market_probability_artifact_id`` / ``..._slope_beta``) is NOT a
    field here: it lives on the blender itself, is written from there, and is read back and
    cross-checked by :meth:`MarketBlender._read_converter_binding` (Plan 33.2-21).

    Attributes:
        gold_generation_digest: The gold generation the three source models were fitted on
            and the walk-forward predictions were computed from.
        source_artifact_ids: ``{wp, ats, ou}`` -> the model artifact whose recipe produced
            that target's tuning predictions.
        tuning_corpus_rows: The owned pre-lock games the corpus carried.
        excluded_counts: ``{reason: games}`` for every exclusion class.
        thread_limit: The OpenMP thread count every fit was pinned to.
    """

    gold_generation_digest: str
    source_artifact_ids: dict[str, str]
    tuning_corpus_rows: int
    excluded_counts: dict[str, int]
    thread_limit: int

    def to_payload(self) -> dict[str, Any]:
        """The payload keys, as written into ``blend_weights.json``."""
        return {
            "gold_generation_digest": self.gold_generation_digest,
            "source_artifact_ids": dict(self.source_artifact_ids),
            "tuning_corpus": {
                "table": "odds_timeline",
                "rows": int(self.tuning_corpus_rows),
                "rule": "latest snapshot at or before each game's own lock",
            },
            "excluded_counts": dict(self.excluded_counts),
            "thread_limit": int(self.thread_limit),
        }

    @classmethod
    def from_payload(cls, data: Mapping[str, Any]) -> BlendProvenance | None:
        """The provenance recorded in *data*, or None when the payload predates it.

        Raises:
            IncompleteBlendProvenanceError: when some provenance keys are present and some
                are not -- half a provenance record answers nothing reliably.
        """
        keys = (
            "gold_generation_digest",
            "source_artifact_ids",
            "tuning_corpus",
            "excluded_counts",
            "thread_limit",
        )
        present = [key for key in keys if key in data]
        if not present:
            return None
        if len(present) != len(keys):
            missing = sorted(set(keys) - set(present))
            msg = (
                f"blend payload carries provenance keys {sorted(present)} but not {missing}; "
                "a blend's provenance is all of it or none of it."
            )
            raise IncompleteBlendProvenanceError(msg)
        return cls(
            gold_generation_digest=str(data["gold_generation_digest"]),
            source_artifact_ids={
                str(k): str(v) for k, v in dict(data["source_artifact_ids"]).items()
            },
            tuning_corpus_rows=int(dict(data["tuning_corpus"])["rows"]),
            excluded_counts={
                str(k): int(v) for k, v in dict(data["excluded_counts"]).items()
            },
            thread_limit=int(data["thread_limit"]),
        )


# ---------------------------------------------------------------------------
# The blend arithmetic, one implementation
# ---------------------------------------------------------------------------


def _blend_values(
    target: str,
    model: np.ndarray,
    market: np.ndarray,
    weight: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """The blend of *model* and *market* at *weight* -- the ONE place the arithmetic lives.

    WP blends in log-odds space after clipping both sides to ``[clip_min, clip_max]`` so the
    logit stays finite; ATS and O/U interpolate linearly in point space. The serving methods
    and the tuner's grid both call this, so the weight the tuner chooses is scored with
    exactly the arithmetic the served blend applies.
    """
    model = np.asarray(model, dtype=np.float64)
    market = np.asarray(market, dtype=np.float64)
    if target == "wp":
        model_clipped = np.clip(model, clip_min, clip_max)
        market_clipped = np.clip(market, clip_min, clip_max)
        return expit(
            weight * logit(model_clipped) + (1 - weight) * logit(market_clipped)
        )
    if target in ("ats", "ou"):
        return weight * model + (1 - weight) * market
    msg = f"Unknown target: {target}. Must be 'wp', 'ats', or 'ou'."
    raise ValueError(msg)


#: Two grid losses within this RELATIVE distance of each other are a tie. Exact arithmetic
#: would score them equal; floating-point rounding separates them in the last bit.
_TIE_RELATIVE_TOLERANCE: float = 1e-12


def _first_minimum(losses: list[float]) -> int:
    """The index of the FIRST loss within :data:`_TIE_RELATIVE_TOLERANCE` of the minimum.

    The grid is ordered from weight 0 upward, so a tie resolves to the lowest weight -- the
    one that trusts the market most -- deterministically rather than by rounding noise.
    """
    values = np.asarray(losses, dtype=np.float64)
    minimum = float(values.min())
    tolerance = _TIE_RELATIVE_TOLERANCE * max(1.0, abs(minimum))
    return int(np.flatnonzero(values <= minimum + tolerance)[0])


def _outcome_loss(target: str, blended: np.ndarray, outcome: np.ndarray) -> float:
    """The target's own primary metric of *blended* against the realized *outcome*.

    WP: log loss (the WP blend's output is already inside the clip, so it is finite).
    ATS / O/U: mean absolute error, in points.
    """
    blended = np.asarray(blended, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    if target == "wp":
        return float(
            -np.mean(
                outcome * np.log(blended) + (1.0 - outcome) * np.log(1.0 - blended)
            )
        )
    return float(np.mean(np.abs(blended - outcome)))


# ---------------------------------------------------------------------------
# MarketBlender
# ---------------------------------------------------------------------------


class MarketBlender:
    """Blends model predictions with market odds at ONE fixed weight per target.

    Uses log-odds space for WP (probabilities are non-linear) and
    linear interpolation for ATS/O/U (spreads and totals are linear).

    Usage::

        blender = MarketBlender()
        blended_wp = blender.blend_wp(model_probs, market_probs)
        blended_spread = blender.blend_ats(model_spreads, market_spreads)
        blended_total = blender.blend_ou(model_totals, market_totals)

        # DataFrame-level blending:
        result_df = blender.blend_predictions(preds_df, market_df, target="wp")
    """

    def __init__(
        self,
        config: BlendConfig | None = None,
        market_probability_artifact_id: str | None = None,
        market_probability_slope_beta: float | None = None,
        provenance: BlendProvenance | None = None,
    ) -> None:
        """Construct a blender, optionally with a converter BOUND to it.

        Args:
            config: Per-target weights, clipping and edge thresholds.
            market_probability_artifact_id: The ``market_probability_*`` directory this
                blend was tuned against. Carried for provenance and cross-checked at load.
            market_probability_slope_beta: That converter's ``slope_beta``. The WP blend
                converts with THIS number and nothing else -- no lookup of "the newest
                converter directory", no module default, no fallback to a moneyline.
            provenance: What a LOADED blend artifact was fitted on, exposed so the swap's
                validator reads it through the same loader the serving path uses. None for a
                blender built in memory.
        """
        self.config = config or BlendConfig()
        self.market_probability_artifact_id = market_probability_artifact_id
        self.market_probability_slope_beta = (
            None
            if market_probability_slope_beta is None
            else float(market_probability_slope_beta)
        )
        self.provenance = provenance
        self.logger = get_logger(__name__)

    def blend_wp(
        self,
        model_prob: np.ndarray,
        market_prob: np.ndarray,
    ) -> np.ndarray:
        """Blend WP predictions in log-odds space at the fixed WP weight.

        There is no week or season parameter: the weight does not vary through the season
        (D33.2-10), so a parameter that selected one would be a hook for a retired shape.

        Clips both inputs to [clip_min, clip_max] before applying logit
        to prevent NaN/inf from boundary probabilities. The blended
        logit is converted back to probability via expit.

        Args:
            model_prob: Model's predicted win probabilities.
            market_prob: Market's fair win probabilities.

        Returns:
            Blended win probabilities in [0, 1].
        """
        return _blend_values(
            "wp",
            model_prob,
            market_prob,
            self.config.weights.wp_model_weight,
            self.config.clip_min,
            self.config.clip_max,
        )

    def blend_ats(
        self,
        model_spread: np.ndarray,
        market_spread: np.ndarray,
    ) -> np.ndarray:
        """Blend ATS predictions in spread-point space (linear interpolation).

        The arithmetic is SCALE-AGNOSTIC -- a weighted average of two numbers on the same scale --
        so it is correct under either sign convention and nothing about it changes here. Only the
        documentation was wrong (DEF-31-03, cosmetic half, corrected in plan 31-17).

        Args:
            model_spread: Model's predicted home MARGIN (positive = home wins by that much).
                The docstring previously said "negative = home favored", which is the OPPOSITE of
                the convention DEF-31-01 measured and the owner ruled on: the ATS trainer's target
                column is ``home_margin`` (``models/trainers/ats_trainer.py``).
            market_spread: Market's spreads on the SAME home-margin scale, POSITIVE when the
                home team is favored (corr with ml_home -0.9506, corr with realized home margin
                +0.4517, measured over 2140 stored rows).

        Returns:
            Blended spreads.
        """
        return _blend_values(
            "ats",
            model_spread,
            market_spread,
            self.config.weights.ats_model_weight,
            self.config.clip_min,
            self.config.clip_max,
        )

    def blend_ou(
        self,
        model_total: np.ndarray,
        market_total: np.ndarray,
    ) -> np.ndarray:
        """Blend O/U predictions in total-point space (linear interpolation).

        Args:
            model_total: Model's predicted game totals.
            market_total: Market's totals.

        Returns:
            Blended totals.
        """
        return _blend_values(
            "ou",
            model_total,
            market_total,
            self.config.weights.ou_model_weight,
            self.config.clip_min,
            self.config.clip_max,
        )

    def blend_predictions(
        self,
        predictions_df: pd.DataFrame,
        market_df: pd.DataFrame,
        target: str,
    ) -> pd.DataFrame:
        """Blend a predictions DataFrame with market data.

        Merges predictions with market data on game_id, then applies
        the appropriate blending method based on target type.

        For WP: converts the pre-lock spread to a market win probability through the BOUND
        converter, then blends model_prob with it in log-odds space.

        For ATS: Blends model_spread with market spread linearly.

        For O/U: Blends model_total with market total linearly.

        Args:
            predictions_df: DataFrame with model predictions. Must contain
                game_id and the target-specific column (model_prob, model_spread,
                or model_total).
            market_df: DataFrame with market odds. Must contain game_id and
                the relevant market columns (spread for WP and ATS, total for O/U).
            target: One of "wp", "ats", "ou".

        Returns:
            Copy of predictions_df with blended values replacing originals.
            All other columns preserved unchanged.

        Raises:
            DuplicateMarketRowError: when *market_df* repeats a ``game_id``.
            MarketProbabilityUnavailable: when ANY game lacks its market line, for every
                target. The whole frame is refused rather than half-blended in place
                (A33.2-review IN-09); a caller blending history selects the games that
                have a line first (``models.blending_data.blend_historical_predictions``).

        A HISTORICAL WP frame must not come through here -- this converts with the bound
        SERVING slope; see :meth:`blend_historical_wp_predictions`.
        """
        result = predictions_df.copy()

        if result.empty:
            return result

        _require_one_market_row_per_game(market_df)

        # Merge on game_id to get market data alongside predictions. With one market row
        # per game the left merge returns exactly one row per prediction, in order.
        merged = result.merge(
            market_df, on="game_id", how="left", suffixes=("", "_market")
        )

        if target == "wp":
            self._blend_wp_predictions(result, merged)
        elif target in _FRAME_BLEND_COLUMNS:
            self._blend_line_predictions(target, result, merged)
        else:
            msg = f"Unknown target: {target}. Must be 'wp', 'ats', or 'ou'."
            raise ValueError(msg)

        return result

    def blend_historical_wp_predictions(
        self,
        predictions_df: pd.DataFrame,
        prelock_lines: pd.DataFrame,
        walk_forward_slopes: Mapping[int, float] | Mapping[str, float],
    ) -> pd.DataFrame:
        """Blend HISTORICAL WP predictions against the OUT-OF-FOLD pre-lock market.

        A33.2-review WR-04/WR-05. :meth:`blend_predictions` converts a spread with the bound
        SERVING slope, which was fitted on 2020-2024 outcomes -- so a 2021-2024 game blended
        through it is priced by a slope fitted partly on its own result, and a diagnostic fed
        CLOSING spreads adds a line that did not exist at the lock. Every historical WP
        consumer (the diagnosis blended cut, the gate's blend re-score, the backtest's
        ``--blend``) comes through here instead: the market side of each game is its OWNED
        pre-lock spread converted with its own season's PRIOR-ONLY slope
        (``models.market_probability.oof_market_probability``), the same column the tuner
        fitted the weight on.

        Args:
            predictions_df: One row per game, carrying ``game_id`` and ``model_prob``.
            prelock_lines: One row per game with ``game_id``, ``season`` and
                ``market_spread`` on the home-margin scale --
                ``models.blending_data.PrelockTuningCorpus.frame``.
            walk_forward_slopes: The BOUND converter artifact's ``walk_forward_slopes``.

        Returns:
            The rows of *predictions_df* whose game has an owned pre-lock line in a season
            with a prior-fold slope, with ``model_prob`` blended and
            :data:`MARKET_PROB_OOF_COLUMN` carrying the market side. Every other row is
            LEFT OUT and counted in the log: never converted with the serving slope, never
            kept unblended in a blended column.

        Raises:
            DuplicateMarketRowError: when either frame repeats a ``game_id``.
        """
        _require_one_market_row_per_game(prelock_lines)
        _require_one_market_row_per_game(predictions_df)

        covered_seasons = {int(season) for season in walk_forward_slopes}
        lines = prelock_lines.loc[:, ["game_id", "season", "market_spread"]].copy()
        lines = lines[lines["season"].astype(int).isin(covered_seasons)]
        lines[MARKET_PROB_OOF_COLUMN] = oof_market_probability(
            lines.assign(home_fav_margin=lines["market_spread"]), walk_forward_slopes
        )

        base = predictions_df.drop(
            columns=[
                c for c in (MARKET_PROB_OOF_COLUMN,) if c in predictions_df.columns
            ]
        )
        merged = base.merge(
            lines[["game_id", MARKET_PROB_OOF_COLUMN]], on="game_id", how="left"
        )
        covered = merged[MARKET_PROB_OOF_COLUMN].notna().to_numpy()
        result = merged.loc[covered].reset_index(drop=True)
        if not result.empty:
            result["model_prob"] = self.blend_wp(
                result["model_prob"].to_numpy(dtype=np.float64),
                result[MARKET_PROB_OOF_COLUMN].to_numpy(dtype=np.float64),
            )

        self.logger.info(
            "Blended historical WP predictions out of fold",
            n_blended=len(result),
            n_total=len(merged),
            n_excluded_no_prelock_line_or_prior_fold=int((~covered).sum()),
        )
        return result

    def unblended_wp_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """The model's OWN win probabilities, explicitly unblended.

        This is the named path a caller takes when it deliberately wants the model half on
        its own -- a diagnostic, a before/after comparison, an ablation. It exists so that
        "no blending happened" is something a call site SAYS rather than something that
        happens when a market opinion is quietly missing.

        Returns:
            A copy of *predictions_df*, values untouched.
        """
        return predictions_df.copy()

    def _blend_wp_predictions(
        self,
        result: pd.DataFrame,
        merged: pd.DataFrame,
    ) -> None:
        """Blend WP predictions in-place using the BOUND converter and the pre-lock spread.

        What changed in Plan 33.2-21, and why (D33.2-09):

        * the market opinion is ``market_home_win_probability(pre-lock spread, bound
          slope)``, not a devigged two-sided CLOSING moneyline. A closing price did not
          exist at the game's lock, and the owned line history carries no moneyline at all;
        * an ABSENT market opinion RAISES :class:`MarketProbabilityUnavailable` instead of
          returning the unblended model probability. There is no longer any path through
          this method that leaves ``model_prob`` untouched;
        * the conversion uses ``self.market_probability_slope_beta`` and NOTHING else --
          no "newest converter directory" lookup, no module default. A blender with no
          bound converter refuses, naming the missing binding.

        This is a SERVING path: it converts with the bound serving slope. A HISTORICAL tuning
        row must never come through here -- the tuner reads the out-of-fold market column
        instead (Plan 33.2-24).

        Raises:
            MarketProbabilityUnavailable: when no converter is bound, when the pre-lock
                spread column is absent, or when any game lacks a pre-lock spread.
        """
        games = (
            [str(game_id) for game_id in merged["game_id"]]
            if "game_id" in merged.columns
            else []
        )

        if self.market_probability_slope_beta is None:
            msg = (
                "no market opinion: this blender has no bound converter, so "
                "market_probability_slope_beta is None and there is nothing to convert "
                f"the pre-lock spread with. Games affected: {games[:10]}. A blend "
                "artifact binds a converter by writing market_probability_artifact_id "
                "and market_probability_slope_beta into blend_weights.json (Plan "
                "33.2-24); the live incumbent carries neither, and refusing here is the "
                "point -- a silent no-blend is indistinguishable from a blend with "
                "weight zero."
            )
            raise MarketProbabilityUnavailable(msg)

        if _PRELOCK_SPREAD_COLUMN not in merged.columns:
            msg = (
                f"no market opinion: the market frame has no {_PRELOCK_SPREAD_COLUMN!r} "
                f"column, so no pre-lock spread can be converted. Games affected: "
                f"{games[:10]}. Since D33.2-09 the WP blend's market half comes from the "
                "pre-lock SPREAD, never from a closing moneyline."
            )
            raise MarketProbabilityUnavailable(msg)

        missing_mask = merged[_PRELOCK_SPREAD_COLUMN].isna()
        if bool(missing_mask.any()):
            offenders = sorted(
                {
                    str(game_id)
                    for game_id in merged.loc[missing_mask, "game_id"]
                    if "game_id" in merged.columns
                }
            )
            msg = (
                f"no market opinion for {int(missing_mask.sum())} game(s): they carry no "
                f"pre-lock spread, e.g. {offenders[:10]}. Refusing the whole WP blend "
                "rather than blending some games and silently leaving the rest "
                "unblended, which would put two different models' outputs in one column."
            )
            raise MarketProbabilityUnavailable(msg)

        market_prob = np.asarray(
            market_home_win_probability(
                home_fav_margin_from_prelock_spread(
                    merged[_PRELOCK_SPREAD_COLUMN].to_numpy()
                ),
                self.market_probability_slope_beta,
            ),
            dtype=np.float64,
        )
        model_prob = np.asarray(result["model_prob"].to_numpy(), dtype=np.float64)

        result["model_prob"] = self.blend_wp(model_prob, market_prob)

        self.logger.info(
            "Blended WP predictions",
            n_blended=len(result),
            n_total=len(result),
            market_probability_artifact_id=self.market_probability_artifact_id,
            market_probability_slope_beta=self.market_probability_slope_beta,
        )

    def _blend_line_predictions(
        self,
        target: str,
        result: pd.DataFrame,
        merged: pd.DataFrame,
    ) -> None:
        """Blend ATS or O/U predictions in place against the market line -- ALL OR NOTHING.

        A33.2-review IN-09: rows without a line used to keep their raw model value in the
        same column as the blended rows, and a missing column returned with a warning -- the
        silent no-blend the WP path already refuses. Both now refuse, and the assignment is
        POSITIONAL on the one-row-per-game merge, so it cannot misalign on a non-default
        index.

        Raises:
            MarketProbabilityUnavailable: when the line column is absent or any game has
                no line.
        """
        model_col, line_col = _FRAME_BLEND_COLUMNS[target]
        games = (
            [str(game_id) for game_id in merged["game_id"]]
            if "game_id" in merged.columns
            else []
        )
        if line_col not in merged.columns:
            msg = (
                f"no market opinion: the market frame has no {line_col!r} column, so no "
                f"{target} game can be blended. Games affected: {games[:10]}."
            )
            raise MarketProbabilityUnavailable(msg)

        missing = merged[line_col].isna().to_numpy()
        if bool(missing.any()):
            offenders = sorted(
                {games[i] for i in np.flatnonzero(missing)} if games else []
            )
            msg = (
                f"no market opinion for {int(missing.sum())} {target} game(s): they carry "
                f"no {line_col!r}, e.g. {offenders[:10]}. Refusing the whole blend rather "
                "than leaving those games' raw model values in a column of blended ones."
            )
            raise MarketProbabilityUnavailable(msg)

        result[model_col] = _blend_values(
            target,
            result[model_col].to_numpy(dtype=np.float64),
            merged[line_col].to_numpy(dtype=np.float64),
            getattr(self.config.weights, _WEIGHT_ATTR_BY_TARGET[target]),
            self.config.clip_min,
            self.config.clip_max,
        )

        self.logger.info(
            "Blended line predictions",
            target=target,
            n_blended=len(result),
            n_total=len(result),
        )

    # -----------------------------------------------------------------------
    # Weight tuning: one fixed weight per target, on each model's own outcome loss
    # -----------------------------------------------------------------------

    def tune_weights(
        self,
        tuning_frames: Mapping[str, pd.DataFrame],
        weight_grid: np.ndarray = BLEND_WEIGHT_GRID,
    ) -> TuningResult:
        """Fit ONE fixed weight per target by grid search on each model's own outcome loss.

        THE OBJECTIVE IS THE GAME, NOT A CLOSING LINE. For every candidate weight the blend of
        the model's out-of-sample prediction with the market's PRE-LOCK opinion is scored
        against the realized outcome in the target's own primary metric
        (:data:`BLEND_OBJECTIVE_BY_TARGET`), and the weight with the lowest loss wins. The
        retired tuner maximised closing-line value instead: a closing line did not exist at
        the lock (D33.2-03), and scoring a blend against the very line it blends with rewards
        whichever end of the range the search is allowed to reach.

        THE WP MARKET SIDE IS READ, NEVER CONVERTED HERE. Each WP row carries
        ``market_prob_oof`` -- its season's prior-only converter slope applied through
        ``models.market_probability.oof_market_probability`` -- and this method reads that
        column. It does not convert a spread, does not call ``market_home_win_probability``
        and does not route a row through :meth:`blend_predictions`, because those use the
        BOUND serving slope, which was fitted partly on these games' own outcomes.

        Ties between grid points resolve to the FIRST (lowest) weight, deterministically --
        where a "tie" is judged within a relative tolerance of 1e-12, because two weights that
        score identically in exact arithmetic can differ in the last bit after rounding, and a
        choice decided by rounding noise is not a choice (:func:`_first_minimum`).

        Args:
            tuning_frames: Target -> one row per game carrying ``season`` and that target's
                :data:`BLEND_TUNING_COLUMNS`. Built by ``models.blending_data.
                build_tuning_frames``.
            weight_grid: The candidate weights, in [0, 1].

        Returns:
            The :class:`TuningResult`. ``self.config.weights`` is updated to the fit.

        Raises:
            BlendTuningError: when a frame is empty, lacks a column, carries a null, or holds
                a season no owned pre-lock line covers.
        """
        candidates = np.round(np.asarray(weight_grid, dtype=float), 4)
        objective_by_target: dict[str, str] = {}
        loss_by_target: dict[str, float] = {}
        market_only: dict[str, float] = {}
        model_only: dict[str, float] = {}
        grid_by_target: dict[str, list[tuple[float, float]]] = {}
        seasons_by_target: dict[str, list[int]] = {}
        n_games: dict[str, int] = {}
        season_best: dict[str, dict[int, float]] = {}
        fitted: dict[str, float] = {}

        for target, frame in tuning_frames.items():
            model_col, market_col, outcome_col = self._require_tuning_frame(
                target, frame
            )
            model = frame[model_col].to_numpy(dtype=float)
            market = frame[market_col].to_numpy(dtype=float)
            outcome = frame[outcome_col].to_numpy(dtype=float)

            grid = [
                (
                    float(weight),
                    _outcome_loss(
                        target,
                        _blend_values(
                            target,
                            model,
                            market,
                            float(weight),
                            self.config.clip_min,
                            self.config.clip_max,
                        ),
                        outcome,
                    ),
                )
                for weight in candidates
            ]
            best_index = _first_minimum([loss for _, loss in grid])
            best_weight, best_loss = grid[best_index]

            fitted[target] = best_weight
            objective_by_target[target] = BLEND_OBJECTIVE_BY_TARGET[target]
            loss_by_target[target] = best_loss
            market_only[target] = _outcome_loss(
                target,
                _blend_values(
                    target,
                    model,
                    market,
                    0.0,
                    self.config.clip_min,
                    self.config.clip_max,
                ),
                outcome,
            )
            model_only[target] = _outcome_loss(
                target,
                _blend_values(
                    target,
                    model,
                    market,
                    1.0,
                    self.config.clip_min,
                    self.config.clip_max,
                ),
                outcome,
            )
            grid_by_target[target] = grid
            seasons = sorted({int(season) for season in frame["season"]})
            seasons_by_target[target] = seasons
            n_games[target] = len(frame)
            season_best[target] = {
                season: self._best_weight_on(
                    target, frame[frame["season"] == season], candidates
                )
                for season in seasons
            }

            self.logger.info(
                "Fixed blend weight fitted",
                target=target,
                weight=best_weight,
                objective=BLEND_OBJECTIVE_BY_TARGET[target],
                loss=best_loss,
                market_only_loss=market_only[target],
                model_only_loss=model_only[target],
                n_games=len(frame),
                seasons=seasons,
            )

        tuned = BlendWeights(
            wp_model_weight=fitted.get("wp", self.config.weights.wp_model_weight),
            ats_model_weight=fitted.get("ats", self.config.weights.ats_model_weight),
            ou_model_weight=fitted.get("ou", self.config.weights.ou_model_weight),
        )
        self.config.weights = tuned

        result = TuningResult(
            weights=tuned,
            objective_by_target=objective_by_target,
            loss_by_target=loss_by_target,
            market_only_loss_by_target=market_only,
            model_only_loss_by_target=model_only,
            grid_by_target=grid_by_target,
            seasons_by_target=seasons_by_target,
            n_games=n_games,
            season_best_weight_by_target=season_best,
        )
        for target in result.boundary_targets:
            # A FINDING, stated loudly rather than clipped away by a narrower grid.
            self.logger.warning(
                "Blend weight fitted at a BOUNDARY: one side of the blend adds nothing "
                "on this corpus",
                target=target,
                weight=fitted[target],
            )
        return result

    def _require_tuning_frame(
        self, target: str, frame: pd.DataFrame
    ) -> tuple[str, str, str]:
        """Refuse a tuning frame that cannot be scored honestly; return its column triple."""
        if target not in BLEND_TUNING_COLUMNS:
            msg = f"Unknown target: {target}. Must be 'wp', 'ats', or 'ou'."
            raise BlendTuningError(msg)
        columns = BLEND_TUNING_COLUMNS[target]
        missing = [c for c in ("game_id", "season", *columns) if c not in frame.columns]
        if missing:
            msg = f"the {target} tuning frame is missing {missing}"
            raise BlendTuningError(msg)
        if frame.empty:
            msg = (
                f"the {target} tuning frame is EMPTY; a weight fitted on no games is a "
                "number that means nothing, so none is fitted."
            )
            raise BlendTuningError(msg)
        nulls = [c for c in columns if bool(frame[c].isna().any())]
        if nulls:
            msg = (
                f"the {target} tuning frame carries nulls in {nulls}; a tuning row must be "
                "complete -- a missing line or outcome is excluded upstream, never filled."
            )
            raise BlendTuningError(msg)
        foreign = sorted(
            {int(s) for s in frame["season"]} - {int(s) for s in OWNED_LINE_SEASONS}
        )
        if foreign:
            msg = (
                f"the {target} tuning frame holds season(s) {foreign}, which no owned "
                f"pre-lock line covers (owned: {list(OWNED_LINE_SEASONS)}). A row from such "
                "a season can only have come from another store -- refusing rather than "
                "tuning on it."
            )
            raise BlendTuningError(msg)
        return columns

    def _best_weight_on(
        self, target: str, frame: pd.DataFrame, candidates: np.ndarray
    ) -> float:
        """The grid weight minimising *target*'s loss on *frame* alone."""
        model_col, market_col, outcome_col = BLEND_TUNING_COLUMNS[target]
        model = frame[model_col].to_numpy(dtype=float)
        market = frame[market_col].to_numpy(dtype=float)
        outcome = frame[outcome_col].to_numpy(dtype=float)
        losses = [
            _outcome_loss(
                target,
                _blend_values(
                    target,
                    model,
                    market,
                    float(weight),
                    self.config.clip_min,
                    self.config.clip_max,
                ),
                outcome,
            )
            for weight in candidates
        ]
        return float(candidates[_first_minimum(losses)])

    # -----------------------------------------------------------------------
    # Weekly edge-rate diagnostic
    #
    # ``calibrate_edge_thresholds`` is DELETED (A33.2-review WR-05): it had no production
    # caller left, and its WP edges were measured against a devigged CLOSING moneyline.
    # The 2026 thresholds are derived by Plan 33.2-26 on the pre-lock converter.
    # -----------------------------------------------------------------------

    def check_weekly_edge_rate(
        self,
        predictions_df: pd.DataFrame,
        market_df: pd.DataFrame,
        target: str,
    ) -> dict:
        """Check per-week edge flagging rate and emit diagnostic warnings.

        Computes edges for each game, groups by (season, week), and
        logs a WARNING for any week where > 30% of games are flagged.
        Per D-09: This is a diagnostic WARNING, not a filter -- games
        are NOT removed.

        Args:
            predictions_df: Predictions DataFrame with model values.
            market_df: Market odds DataFrame.
            target: One of "wp", "ats", "ou".

        Returns:
            Dict with per_week_rates, mean_rate, warnings.
        """
        merged = predictions_df.merge(
            market_df, on="game_id", how="inner", suffixes=("", "_mkt")
        )
        if merged.empty:
            return {"per_week_rates": [], "mean_rate": 0.0, "warnings": []}

        # Ensure season/week columns exist for grouping.
        # Predictions may not carry week; extract from game_id (e.g. 2021_W01_ATL@PHI).
        if "season" not in merged.columns:
            merged["season"] = merged["game_id"].str.split("_").str[0].astype(int)
        if "week" not in merged.columns:
            merged["week"] = (
                merged["game_id"].str.split("_").str[1].str.lstrip("W").astype(int)
            )

        edges = self._compute_edges(target, merged)
        merged["_edge"] = edges

        # Get the threshold for this target
        threshold_map = {
            "wp": self.config.edge_thresholds.wp_threshold,
            "ats": self.config.edge_thresholds.ats_threshold,
            "ou": self.config.edge_thresholds.ou_threshold,
        }
        threshold = threshold_map.get(target, 0.03)

        # Compute per-week flagging rate
        per_week_rates: list[float] = []
        warnings: list[str] = []

        for (season, week), group in merged.groupby(["season", "week"]):
            rate = float((group["_edge"] > threshold).mean())
            per_week_rates.append(rate)

            if rate > 0.30:
                msg = (
                    f"Edge threshold diagnostic: {rate:.0%} of games flagged "
                    f"in season {season} week {week} for {target}"
                )
                warnings.append(msg)
                self.logger.warning(msg)

        mean_rate = float(np.mean(per_week_rates)) if per_week_rates else 0.0

        return {
            "per_week_rates": per_week_rates,
            "mean_rate": mean_rate,
            "warnings": warnings,
        }

    def _get_edge_weight(
        self,
        target: str,
        merged: pd.DataFrame,
    ) -> np.ndarray:
        """The fixed per-target blend weight, one entry per row of *merged*."""
        static_weight = getattr(self.config.weights, _WEIGHT_ATTR_BY_TARGET[target])
        return np.full(len(merged), static_weight)

    def _compute_edges(
        self,
        target: str,
        merged: pd.DataFrame,
    ) -> np.ndarray:
        """Compute edge magnitudes for a target on merged predictions+odds.

        THE WP MARKET SIDE IS NEVER A CLOSING MONEYLINE (A33.2-review WR-05). It is the
        historical row's :data:`MARKET_PROB_OOF_COLUMN` when the frame carries one (every
        historical blend attaches it), and otherwise the pre-lock ``spread`` converted
        through the BOUND converter -- the same conversion the serving blend applies. A
        blender with neither refuses rather than inventing a market opinion.

        Args:
            target: "wp", "ats", or "ou".
            merged: Predictions merged with odds DataFrame.

        Returns:
            Array of absolute edge values, one per row of *merged* for WP (0.0 where a row
            has no market opinion).

        Raises:
            MarketProbabilityUnavailable: for WP, when the frame carries no out-of-fold
                market column and the blender has no bound converter or no spread column.
        """
        if target == "wp":
            if MARKET_PROB_OOF_COLUMN in merged.columns:
                market = merged[MARKET_PROB_OOF_COLUMN].to_numpy(dtype=np.float64)
            else:
                if self.market_probability_slope_beta is None:
                    msg = (
                        "no WP market opinion for the edge diagnostic: the frame carries no "
                        f"{MARKET_PROB_OOF_COLUMN!r} column and this blender has no bound "
                        "converter to convert a pre-lock spread with."
                    )
                    raise MarketProbabilityUnavailable(msg)
                if _PRELOCK_SPREAD_COLUMN not in merged.columns:
                    msg = (
                        "no WP market opinion for the edge diagnostic: the frame carries "
                        f"neither {MARKET_PROB_OOF_COLUMN!r} nor {_PRELOCK_SPREAD_COLUMN!r}."
                    )
                    raise MarketProbabilityUnavailable(msg)
                market = np.asarray(
                    market_home_win_probability(
                        home_fav_margin_from_prelock_spread(
                            merged[_PRELOCK_SPREAD_COLUMN].to_numpy(dtype=np.float64)
                        ),
                        self.market_probability_slope_beta,
                    ),
                    dtype=np.float64,
                )
            model = merged["model_prob"].to_numpy(dtype=np.float64)
            valid = ~(np.isnan(model) | np.isnan(market))
            result = np.zeros(len(merged))
            if bool(valid.any()):
                blended = self.blend_wp(model[valid], market[valid])
                result[valid] = np.abs(blended - market[valid])
            return result

        if target == "ats":
            valid = merged.dropna(subset=["spread", "model_spread"])
            if valid.empty:
                return np.array([])
            weights = self._get_edge_weight("ats", valid)
            blended = (
                weights * valid["model_spread"].values
                + (1 - weights) * valid["spread"].values
            )
            edges = np.abs(blended - valid["spread"].values)
            result = np.zeros(len(merged))
            result[valid.index.to_numpy() - merged.index[0]] = edges
            return result

        if target == "ou":
            valid = merged.dropna(subset=["total", "model_total"])
            if valid.empty:
                return np.array([])
            weights = self._get_edge_weight("ou", valid)
            blended = (
                weights * valid["model_total"].values
                + (1 - weights) * valid["total"].values
            )
            edges = np.abs(blended - valid["total"].values)
            result = np.zeros(len(merged))
            result[valid.index.to_numpy() - merged.index[0]] = edges
            return result

        return np.zeros(len(merged))

    # -----------------------------------------------------------------------
    # Artifact persistence
    # -----------------------------------------------------------------------

    def save_blend_artifacts(
        self,
        tuning_result: TuningResult,
        artifacts_dir: Path = Path("artifacts"),
        *,
        provenance: BlendProvenance,
        update_latest: bool = False,
    ) -> Path:
        """Write ONE new ``blend_{timestamp}/blend_weights.json`` carrying the whole record.

        The payload holds the fitted weights, the tuning record, the provenance
        (:class:`BlendProvenance`) and the converter binding this blender carries -- all in
        the one file :meth:`from_artifacts` opens. No ``metadata.json`` is written beside it:
        two files would be two answers about one artifact's provenance.

        ``artifacts/latest.json`` IS NOT TOUCHED BY DEFAULT. It is the production swap
        surface, and until Plan 33.2-24 this method rewrote it unconditionally after writing
        the payload -- so a fit meant to produce a CANDIDATE silently deployed it. The flag
        mirrors ``models.artifacts.save_model_artifact``'s D24-08 ``update_latest`` exactly:
        the manifest is rewritten only when the caller says so. Plan 33.2-25's batched swap
        is the one writer of the ``blend`` pointer this phase.

        Args:
            tuning_result: The :class:`TuningResult` from :meth:`tune_weights`.
            artifacts_dir: Root directory for artifacts.
            provenance: What the blend was fitted on. REQUIRED: a blend artifact that cannot
                name its gold, its source models and its corpus is one nobody can audit.
            update_latest: Rewrite ``latest.json``'s ``blend`` pointer to the new directory.

        Returns:
            Path to the created artifact directory.

        Raises:
            MarketProbabilityBindingError: when no converter is bound -- such a blend could
                never convert a spread, so it could never serve WP.
            FileExistsError: when the timestamped directory already exists; an artifact id
                names exactly one payload.
        """
        if (
            self.market_probability_artifact_id is None
            or self.market_probability_slope_beta is None
        ):
            msg = (
                "refusing to write a blend artifact with no converter bound: without "
                "market_probability_artifact_id and market_probability_slope_beta the "
                "blend could never convert a pre-lock spread, so it could never serve WP."
            )
            raise MarketProbabilityBindingError(msg)

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        artifact_dir = artifacts_dir / f"{BLEND_ARTIFACT_PREFIX}_{timestamp}"
        artifact_dir.mkdir(parents=True, exist_ok=False)

        weights = {
            target: getattr(tuning_result.weights, attr)
            for target, attr in _WEIGHT_ATTR_BY_TARGET.items()
        }
        payload: dict[str, Any] = {
            "blender_version": BLENDER_VERSION,
            "blend_shape": "one fixed model weight per target",
            "weights": weights,
            # Carried from this blender's config so from_artifacts round-trips them, exactly
            # as before. The fixed-weight fit does NOT calibrate them: the threshold
            # derivation is Plan 33.2-26's, and the source is stated rather than implied.
            "edge_thresholds": {
                "wp": self.config.edge_thresholds.wp_threshold,
                "ats": self.config.edge_thresholds.ats_threshold,
                "ou": self.config.edge_thresholds.ou_threshold,
            },
            "edge_thresholds_source": (
                "the blender's configured thresholds; not calibrated by the fixed-weight "
                "fit -- the 2026 threshold derivation is Plan 33.2-26's"
            ),
            "objective": dict(tuning_result.objective_by_target),
            "loss_at_weight": dict(tuning_result.loss_by_target),
            "loss_market_only": dict(tuning_result.market_only_loss_by_target),
            "loss_model_only": dict(tuning_result.model_only_loss_by_target),
            "boundary_weights": list(tuning_result.boundary_targets),
            "weight_grid": {
                "min": float(BLEND_WEIGHT_GRID[0]),
                "max": float(BLEND_WEIGHT_GRID[-1]),
                "step": 0.01,
            },
            "tuning_seasons": {
                t: list(seasons)
                for t, seasons in tuning_result.seasons_by_target.items()
            },
            "n_games": dict(tuning_result.n_games),
            "season_best_weight": {
                t: {str(season): weight for season, weight in best.items()}
                for t, best in tuning_result.season_best_weight_by_target.items()
            },
            "tuned_at": datetime.now(tz=UTC).isoformat(),
            **provenance.to_payload(),
            "market_probability_artifact_id": self.market_probability_artifact_id,
            "market_probability_slope_beta": self.market_probability_slope_beta,
        }

        # The ONE atomic-write helper the manifest writers use, so there is one
        # implementation of an atomic JSON write in this repository.
        from models.artifacts import _atomic_write_json

        _atomic_write_json(artifact_dir / BLEND_PAYLOAD_FILENAME, payload)

        if update_latest:
            latest_path = artifacts_dir / "latest.json"
            manifest = (
                json.loads(latest_path.read_text()) if latest_path.exists() else {}
            )
            manifest["blend"] = artifact_dir.name
            _atomic_write_json(latest_path, manifest)

        self.logger.info(
            "Saved blend artifact",
            artifact_dir=str(artifact_dir),
            weights=weights,
            update_latest=update_latest,
        )
        return artifact_dir

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Path = Path("artifacts"),
        version: str | None = None,
    ) -> MarketBlender:
        """Load a MarketBlender from saved artifacts.

        Reads blend_weights.json from the artifact directory (latest or
        specific version) and returns a new MarketBlender configured
        with the loaded weights, its converter binding and its provenance.

        Args:
            artifacts_dir: Root directory for artifacts.
            version: Specific artifact directory name. If None, uses latest.

        Returns:
            MarketBlender configured with loaded weights.

        Raises:
            FileNotFoundError: If artifacts not found.
            KeyError: If 'blend' not in latest.json.
            RetiredDynamicBlendError: If the payload carries the retired ``dynamic`` section.
        """
        if version is None:
            latest_path = artifacts_dir / "latest.json"
            if not latest_path.exists():
                msg = f"latest.json not found in {artifacts_dir}"
                raise FileNotFoundError(msg)
            manifest = json.loads(latest_path.read_text())
            if "blend" not in manifest:
                msg = "'blend' not found in latest.json manifest"
                raise KeyError(msg)
            version = manifest["blend"]

        artifact_dir = artifacts_dir / version
        weights_path = artifact_dir / BLEND_PAYLOAD_FILENAME

        if not weights_path.exists():
            msg = f"blend_weights.json not found in {artifact_dir}"
            raise FileNotFoundError(msg)

        data = json.loads(weights_path.read_text())

        if "dynamic" in data:
            msg = (
                f"blend artifact {version!r} carries a 'dynamic' section: it is the retired "
                "week-varying blend (D33.2-10), fitted on synthetic predictions over "
                "2010-2017 closing lines. Refusing it rather than reading it as a static "
                "blend, which would publish a number neither the retired rule nor the new "
                "one chose. The replacement is a fixed-weight 'blend_*' artifact, installed "
                "by the production swap (Plan 33.2-25)."
            )
            raise RetiredDynamicBlendError(msg)

        weights_data = data["weights"]
        blend_weights = BlendWeights(
            wp_model_weight=weights_data["wp"],
            ats_model_weight=weights_data["ats"],
            ou_model_weight=weights_data["ou"],
        )

        # Load edge thresholds if present
        edge_data = data.get("edge_thresholds")
        if edge_data:
            edge_thresholds = EdgeThresholds(
                wp_threshold=edge_data["wp"],
                ats_threshold=edge_data["ats"],
                ou_threshold=edge_data["ou"],
            )
        else:
            edge_thresholds = EdgeThresholds()

        config = BlendConfig(
            weights=blend_weights,
            edge_thresholds=edge_thresholds,
        )

        converter_id, converter_slope = cls._read_converter_binding(data, artifacts_dir)

        return cls(
            config=config,
            market_probability_artifact_id=converter_id,
            market_probability_slope_beta=converter_slope,
            provenance=BlendProvenance.from_payload(data),
        )

    @staticmethod
    def _read_converter_binding(
        data: dict,
        artifacts_dir: Path,
    ) -> tuple[str | None, float | None]:
        """The converter this blend is BOUND to, cross-checked against its directory.

        ``blend_weights.json`` is the blend's one payload file and the only file
        :meth:`from_artifacts` opens, which makes it the one place a converter binding can
        live without adding a fifth ``latest.json`` pointer (SPEC R13 fixes it at four).
        Plan 33.2-24 writes the two keys read here.

        An ABSENT binding loads unchanged and returns ``(None, None)``: nothing that loads a
        converter-less blend may break at load time. The refusal happens later, at the blend
        itself, where a market opinion would have had to be invented.

        Raises:
            MarketProbabilityBindingError: when only half the binding is present, when the
                named converter directory is absent, or when its ``slope_beta`` differs
                from the recorded one.
        """
        recorded_id = data.get("market_probability_artifact_id")
        recorded_slope = data.get("market_probability_slope_beta")

        if recorded_id is None and recorded_slope is None:
            return None, None

        if recorded_id is None or recorded_slope is None:
            msg = (
                "half a converter binding in blend_weights.json: got "
                f"market_probability_artifact_id={recorded_id!r} and "
                f"market_probability_slope_beta={recorded_slope!r}. A binding is both "
                "keys or neither -- an id with no slope cannot be served, and a slope "
                "with no id cannot be audited."
            )
            raise MarketProbabilityBindingError(msg)

        try:
            payload = load_market_probability_artifact(str(recorded_id), artifacts_dir)
        except MarketProbabilityArtifactError as exc:
            msg = (
                f"this blend is bound to converter {recorded_id!r}, which is not present "
                f"under {artifacts_dir}. A converter artifact is immutable and an id "
                "names one payload forever, so an absent directory means the binding "
                "cannot be honoured -- not that a different converter should be used."
            )
            raise MarketProbabilityBindingError(msg) from exc

        on_disk = float(payload["slope_beta"])
        recorded = float(recorded_slope)
        if on_disk != recorded:
            msg = (
                f"converter binding mismatch: the blend records slope_beta={recorded!r} "
                f"for {recorded_id!r}, but that directory records {on_disk!r}. The blend "
                "was tuned against one converter and would be served with another, which "
                "is exactly what the binding exists to make impossible."
            )
            raise MarketProbabilityBindingError(msg)

        return str(recorded_id), recorded
