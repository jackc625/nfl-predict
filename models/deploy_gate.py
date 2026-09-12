"""The single, frozen, significance-tested per-target model deploy gate (Phase 24).

This module is THE JUDGE for the per-target deploy decision (ACTV-01, ACTV-02). It
decides, per target, whether a freshly-fit candidate model may replace the production
model in ``artifacts/latest.json``. It is import-parity-locked to ``backtest/diagnose.py``
(D24-13): it IMPORTS the per-target CLV column map (``CLV_COLUMN_FOR``), the significance
constant (``SIGNIFICANCE_ALPHA``), and the t-test (``clv_significance``) directly from the
diagnosis, and NEVER re-declares them. The gate and the honest diagnosis therefore cannot
silently diverge onto different metrics (Pitfall 2 -- the gate must judge on the SAME CLV
column the diagnosis reports: ``probability_clv`` for WP, ``line_clv`` for ATS/OU).

Why a HARD block: in v2.0 an ungated swap shipped a regression -- the deployed v1.0 pre-Elo
models were retained only because per-target gating caught the regression after the fact
(the D-17 precedent). v3.0 deliberately re-fits (lifting the v2.1 D-01 boundary), so this
gate is the safety rail: nothing ships that regresses. The frozen thresholds + baseline live
in the git-tracked ``config/gate.toml``; a threshold change is its own reviewed config edit.

The CLV floor has TWO modes, selected by ``cfg["gate"]["floor_mode"]`` (D25-01):

  * ``"non_regression"`` (the Phase-25 DEFAULT): a target PASSES a slice unless the candidate
    is SIGNIFICANTLY WORSE than the re-scored v1.0 baseline on the same gold -- a PAIRED
    per-game CLV delta (``candidate_clv - baseline_clv``) tested with ``clv_significance`` and
    read on its negative tail (``mean < 0 and p < alpha`` -> FAIL). This is the replace-or-
    retain decision: an absolute-vs-zero floor would keep a significantly-WORSE incumbent in
    production over a technicality (WP serves -0.0567 while the -0.0443 candidate is blocked),
    while where v1.0 is already positive (OU pooled +1.11) non-regression is STRICTER than
    absolute (the edge cannot regress). See D25-01 / D25-15 + ``config/gate.toml`` floor_mode.
  * ``"absolute"`` (the legacy D24-01 floor, retained): a target PASSES a slice unless its raw
    per-game CLV is significantly negative-vs-zero -- the LOGICAL COMPLEMENT of
    ``diagnose.clv_verdict``'s "systematically negative" branch
    (``mean < 0 and p < SIGNIFICANCE_ALPHA``). Kept for backward-compatible unit fixtures and
    as the bettable-bar verdict computation below.

REGARDLESS of mode, the absolute-vs-zero ``clv_significance`` of the RAW candidate CLV is
ALWAYS computed and attached to the result (``absolute_verdict`` + ``absolute_per_season``):
it is the "bettable bar" readout for Phases 26-27 (removing a CLV leak is NOT a positive
market edge -- D25-01, Pitfall 3), never removed.

Per-season-must-pass (D24-04) applies the active floor to every holdout season individually so
one lucky season cannot carry a model whose other seasons are significantly worse. The
secondary non-regression gates (accuracy/MAE, and for WP the calibration ECE/Brier per D24-05)
are evaluated POOLED (Open Question A: ~270-game per-season secondary slices fire on sampling
noise; per-season secondary deltas are a readout, not a blocker). diagnose.py is pooled-only,
so per-season CLV slicing is the one genuinely-new piece here -- it reuses the season-agnostic
``clv_significance`` on per-season array slices (raw CLV under absolute, delta under
non_regression).

The candidate/baseline bundle contract (the output of ``build_candidate_bundle``). The PINNED
paired-delta keys (D25-01, Codex HIGH) are DEFINED here in Plan 25-01 and POPULATED with the
real merge-on-game_id pairing by Plan 25-02 in ``scripts/promote_models.py`` -- this plan ships
them as ``None`` placeholders so a shape/key mismatch when 25-02 wires the pairing fails loudly:

    {
        # Pooled per-game arrays (all the SAME length, aligned by game_id):
        "clv_values": np.ndarray,                 # raw candidate per-game CLV (absolute-verdict
                                                  #   input + legacy absolute floor input)
        "baseline_clv_values": np.ndarray | None, # raw re-scored v1.0 per-game CLV, game_id order
        "clv_delta_values": np.ndarray | None,    # candidate-minus-baseline per game (the
                                                  #   non_regression floor input); equals
                                                  #   clv_values - baseline_clv_values
        "mean": float | None,                     # pooled raw-candidate CLV mean
        "t": float | None,                        # pooled raw-candidate CLV t-stat
        "p": float | None,                        # pooled raw-candidate CLV two-sided p-value
        "per_season": {int: {...}, ...},          # {season -> raw-candidate clv_significance}
        "per_season_clv_values": {int: ndarray},  # {season -> raw candidate per-game CLV array}
        "per_season_baseline_clv_values": {int: ndarray | None},  # {season -> baseline array}
        "per_season_clv_delta_values": {int: ndarray | None},     # {season -> delta array}
        # WP only:
        "accuracy": float, "ece": float, "brier_score": float,
        # ATS/OU only:
        "mae": float,
    }

``build_candidate_bundle`` is the SINGLE source of this bundle shape. ``scripts/promote_models.py``,
the rewired ``scripts/retrain_models.py``, and the gate tests MUST all construct candidate
bundles through it, so the three call sites cannot drift into subtly different bundle shapes
and the forced-FAIL/PASS tests have ONE well-defined monkeypatch seam (mirrors
``diagnose._measure_target``; because ``compute_clv_for_predictions`` RECOMPUTES the CLV column
from ``model_prob``, a pre-filled CLV column on the scored frame would be overwritten -- so the
bundle builder, not ``score_deployed_artifacts``, is the correct seam to force a CLV value).

The ``{passed, reasons, v1_metrics, v2_metrics}`` reason-dict shape is kept compatible with
``scripts.retrain_models.print_gating_summary`` so the Plan 24-04 rewire is drop-in.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import inspect
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Import-the-diagnosis parity seam (D24-13). These three symbols are IMPORTED, never
# re-declared here: the gate must judge on the exact same CLV column + significance test
# the honest diagnosis uses, or the two could silently diverge (Pitfall 2).
from backtest.diagnose import (
    CLV_COLUMN_FOR,
    SIGNIFICANCE_ALPHA,
    clv_significance,
)
from backtest.metrics import compute_wp_metrics
from models.clv import compute_clv_for_predictions
from utils import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# Re-export the imported names so callers (and the parity test) can reference
# ``gate.CLV_COLUMN_FOR`` / ``gate.SIGNIFICANCE_ALPHA``. These are the SAME objects as in
# diagnose.py (identity, not copies): ``gate.CLV_COLUMN_FOR is diagnose.CLV_COLUMN_FOR``.
__all__ = [
    "CLV_COLUMN_FOR",
    "JUDGE_DIGEST_FUNCTIONS",
    "JUDGE_VERSION",
    "LIVE_RESCORE_PROVENANCE",
    "SECONDARY_METRICS_FOR",
    "SECONDARY_SCALAR_NAMES",
    "SIGNIFICANCE_ALPHA",
    "DuplicateGameIdError",
    "EligibilityIndex",
    "build_candidate_bundle",
    "build_eligibility_index",
    "clv_floor_passes",
    "clv_non_regression_passes",
    # Re-exported from backtest.diagnose as part of the D24-13 parity surface; tests and
    # callers reference it as gate.clv_significance (IN-01).
    "clv_significance",
    "comparator_provenance",
    "evaluate_target",
    "judge_code_digest",
    "live_secondary_metrics",
    "load_gate_config",
    "per_season_clv",
    "validate_gate_config",
]

# The frozen holdout window (matches BacktestConfig + diagnose.py). per_season_clv slices
# the per-game CLV arrays by these seasons; load_gate_config asserts a populated baseline
# season table matches this set exactly.
HOLDOUT_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024)

# CLV/odds columns dropped from a scored frame before recomputing CLV, so the left-merge in
# compute_clv_for_predictions does not produce duplicate odds columns (mirrors the
# diagnose._measure_target "raw_drop" discipline at diagnose.py:416-417).
_CLV_ODDS_COLS = (
    "probability_clv",
    "fair_closing_prob",
    "has_closing_odds",
    "line_clv",
    "ml_home",
    "ml_away",
    "spread",
    "total",
)

# ---------------------------------------------------------------------------
# (0) D33-11 -- THE SECONDARY COMPARATOR IS A LIVE PAIRED RE-SCORE
#
# Owner ruling, 2026-09-12, option `live-rescore`:
#
#     Extend the live paired re-score to the five secondary comparator scalars,
#     making the whole deploy gate gold-invariant by construction rather than by
#     re-freezing after every rebuild. config/gate.toml's [baseline.*] block stays
#     byte-untouched as a historical record; what changes permanently is where the
#     comparator comes from. Both gate-baseline tripwires stay RED.
#
# WHY THIS IS A CONVERSION AND NOT AN INVENTION. The PRIMARY significance-tested CLV
# gate was ALREADY gold-invariant: `_pooled_floor_reasons` consumes
# `candidate["clv_delta_values"]` -- a live paired re-score of the deployed incumbent
# on the same gold -- and touches the frozen block NOWHERE. Only the five secondary
# scalars read it. Making those two functions look like `_pooled_floor_reasons` is
# the whole change.
#
# WHAT THIS DOES NOT DO. It does not explain WHY the frozen baseline diverges from a
# fresh re-score in 47 of 68 fields. That disclosure is PRESERVED, not discharged,
# and remains owed. A live sample measured at ruling time: `baseline.wp.pooled.t` is
# committed as -15.52461299 while a fresh re-score returns -17.93777193561613, a
# difference of 2.4131589456161304 against a _FRESHNESS_TOL of 0.005.
# ---------------------------------------------------------------------------

# The FIVE secondary scalars, flat and in target order. FIVE, not four: an earlier
# draft of Plan 33-08 said four, and a completeness check written against four would
# have PASSED while one scalar went unchecked. Every count assertion in this phase
# reads this constant rather than a literal.
SECONDARY_SCALAR_NAMES: tuple[str, ...] = (
    "wp.accuracy",
    "wp.ece",
    "wp.brier_score",
    "ats.mae",
    "ou.mae",
)

# The same five, keyed by target -- the form the scorers and the reason builders
# consume. `SECONDARY_SCALAR_NAMES` is the flattening of this map and a test asserts
# the two close against each other, so they are one declaration in two shapes rather
# than two declarations that can drift.
SECONDARY_METRICS_FOR: dict[str, tuple[str, ...]] = {
    "wp": ("accuracy", "ece", "brier_score"),
    "ats": ("mae",),
    "ou": ("mae",),
}

# The judge that renders a verdict. A permanent semantic change to deployment policy
# needs a name: a verdict that cannot say which judge produced it cannot be compared
# against a later one.
JUDGE_VERSION: str = "phase33-live-secondary-rescore-1"

# The functions whose source the judge digest covers. A judging function OUTSIDE this
# tuple is a change no verdict record can see.
JUDGE_DIGEST_FUNCTIONS: tuple[str, ...] = (
    "_secondary_reasons",
    "_calibration_reasons",
    "_pooled_floor_reasons",
    "clv_non_regression_passes",
    "live_secondary_metrics",
    "build_eligibility_index",
)

# The marker `live_secondary_metrics` stamps onto every comparator it produces, and
# the key it lives under. A comparator WITHOUT the marker is not rejected -- the
# legacy Phase-24/25/30 `scripts/promote_models.py` path still hands the gate a
# frozen-toml bundle and must keep working -- but it reads as `unattributed`, so a
# verdict record can never imply a live re-score it did not have.
COMPARATOR_PROVENANCE_KEY: str = "comparator_provenance"
LIVE_RESCORE_PROVENANCE: str = "live_paired_rescore"
UNATTRIBUTED_PROVENANCE: str = "unattributed"

# Which column carries the prediction whose presence makes a game ELIGIBLE. For
# ATS/OU this is the EXPLICIT line column, never the overloaded `model_prob` -- the
# same CR-01 discipline `build_candidate_bundle` applies to the MAE.
_PREDICTION_COLUMN_FOR: dict[str, str] = {
    "wp": "model_prob",
    "ats": "model_spread",
    "ou": "model_total",
}


class DuplicateGameIdError(ValueError):
    """A scored frame carries the same ``game_id`` twice, on a named side.

    Raised BEFORE any scoring. A duplicate is a many-to-many join waiting to happen:
    it would silently inflate one side of a paired comparison and the resulting
    delta would look like a real number.
    """


@dataclass(frozen=True)
class EligibilityIndex:
    """The ONE materialized set of ``game_id``s both scorers are handed for a target.

    "The same nominal gold file" is not the same eligible rows (T-33-41c). Two
    scorers reading one file can still drop different rows for different reasons -- a
    null in a feature one model uses and the other does not, a missing prediction on
    one side -- and produce a comparison that is plausible, paired-looking and wrong.
    This object is the answer: it is built once per target and passed to BOTH sides,
    neither of which re-derives its own eligible set.

    ``game_ids`` is SORTED, which is what makes the comparison order-invariant by
    construction rather than by both callers happening to be careful.

    The three exclusion counts are carried so an asymmetric drop is VISIBLE in the
    verdict record rather than absorbed into a smaller n nobody notices.
    """

    target: str
    game_ids: tuple[str, ...]
    excluded_incumbent_only: int
    excluded_candidate_only: int
    excluded_not_in_gold: int

    @property
    def n(self) -> int:
        """How many games survived into the paired comparison."""
        return len(self.game_ids)

    def as_record(self) -> dict[str, Any]:
        """The verdict-record view of this index (counts, not the id list)."""
        return {
            "target": self.target,
            "n_eligible": self.n,
            "excluded_incumbent_only": self.excluded_incumbent_only,
            "excluded_candidate_only": self.excluded_candidate_only,
            "excluded_not_in_gold": self.excluded_not_in_gold,
        }


def _predicted_game_ids(frame: Any, target: str, side: str) -> set[str]:
    """The set of ``game_id``s for which *frame* actually produced a prediction.

    "Produced a prediction" means a NON-NULL value in the target's prediction column,
    not merely a present column: a NaN row is a row the model did not score, and
    counting it as eligible is how one side ends up compared on rows the other never
    saw.

    Args:
        frame: A scored frame carrying ``game_id`` and the target's prediction column.
        target: One of "wp", "ats", "ou".
        side: "incumbent" or "candidate" -- used only to name a duplicate in the error.

    Returns:
        The set of game_ids with a non-null prediction.

    Raises:
        DuplicateGameIdError: If ``game_id`` repeats anywhere in *frame*.
    """
    ids = frame["game_id"].astype(str)
    duplicated = sorted(set(ids[ids.duplicated()]))
    if duplicated:
        msg = (
            f"The {side} frame for target '{target}' carries duplicate game_id values "
            f"{duplicated}. A duplicate is a many-to-many join waiting to happen, so the "
            "eligibility index refuses it before any scoring rather than producing a "
            "plausible but inflated paired comparison."
        )
        raise DuplicateGameIdError(msg)

    column = _PREDICTION_COLUMN_FOR[target]
    if column not in frame.columns:
        return set()
    predicted = frame.loc[frame[column].notna(), "game_id"].astype(str)
    return set(predicted)


def build_eligibility_index(
    target: str,
    gold: Any,
    incumbent: Any,
    candidate: Any,
) -> EligibilityIndex:
    """Materialize the ONE per-target eligible ``game_id`` set both scorers share.

    The index is the INTERSECTION of the games for which BOTH sides produced a
    prediction, further restricted to the games gold carries truth for. Rows dropped
    at each of those three boundaries are COUNTED by side, so an asymmetric exclusion
    shows up in the verdict record instead of quietly shrinking n.

    Args:
        target: One of "wp", "ats", "ou".
        gold: The truth frame carrying ``game_id``. ``None`` means "impose no truth
            restriction" -- used by synthetic fixtures that carry ``actual`` on the
            scored frames themselves.
        incumbent: The DEPLOYED incumbent's scored frame.
        candidate: The candidate's scored frame.

    Returns:
        An :class:`EligibilityIndex` with SORTED ``game_ids``.

    Raises:
        ValueError: If *target* is not a known target.
        DuplicateGameIdError: If either side repeats a ``game_id``.
    """
    if target not in _PREDICTION_COLUMN_FOR:
        msg = f"Unknown target: '{target}'. Must be one of {sorted(_PREDICTION_COLUMN_FOR)}."
        raise ValueError(msg)

    incumbent_ids = _predicted_game_ids(incumbent, target, "incumbent")
    candidate_ids = _predicted_game_ids(candidate, target, "candidate")

    paired = incumbent_ids & candidate_ids
    if gold is not None:
        gold_ids = set(gold["game_id"].astype(str))
        eligible = paired & gold_ids
        excluded_not_in_gold = len(paired - gold_ids)
    else:
        eligible = paired
        excluded_not_in_gold = 0

    return EligibilityIndex(
        target=target,
        game_ids=tuple(sorted(eligible)),
        # "incumbent_only" = the incumbent predicted it and the candidate did not, so
        # the CANDIDATE is why it is out. Named by which side HAS the row, which is
        # the side a reader would go looking at.
        excluded_incumbent_only=len(incumbent_ids - candidate_ids),
        excluded_candidate_only=len(candidate_ids - incumbent_ids),
        excluded_not_in_gold=excluded_not_in_gold,
    )


def _index_game_ids(eligibility_index: Any) -> tuple[str, ...]:
    """Normalize an :class:`EligibilityIndex` OR a bare sequence of ids to a tuple.

    Plan 33-15 may hand a scorer the ids alone; that must not be a second code path.
    """
    if isinstance(eligibility_index, EligibilityIndex):
        return eligibility_index.game_ids
    return tuple(str(g) for g in eligibility_index)


def live_secondary_metrics(
    target: str,
    incumbent: Any,
    gold: Any,
    eligibility_index: Any,
) -> dict[str, Any]:
    """Re-score the DEPLOYED INCUMBENT's secondary scalars on the shared index (D33-11).

    This is the comparator ``_secondary_reasons`` and ``_calibration_reasons`` now
    read. It is the same live-re-score idea ``build_candidate_bundle`` already applies
    to ``baseline_clv_values``, extended to the FIVE secondary scalars so the whole
    judge is gold-invariant BY CONSTRUCTION rather than by re-freezing
    ``config/gate.toml`` after every rebuild.

    BOTH sides are REINDEXED onto ``eligibility_index`` before anything is computed.
    That is what makes the comparison order-invariant and genuinely paired: a scorer
    that derived its own eligible set from the gold file could silently score the two
    models on different rows.

    Args:
        target: One of "wp", "ats", "ou".
        incumbent: The deployed incumbent's scored frame (``game_id`` plus the
            target's prediction column, and ``actual`` when *gold* is not supplied).
        gold: The truth frame (``game_id`` + ``actual``). It is the AUTHORITATIVE
            source of ``actual`` when present, so both sides are graded against one
            truth; ``None`` falls back to the scored frame's own ``actual`` column.
        eligibility_index: An :class:`EligibilityIndex` or a bare sequence of
            ``game_id``s. The index is a RESTRICTION, not a suggestion.

    Returns:
        ``{metric: value, ..., "n": int, COMPARATOR_PROVENANCE_KEY: ...,
        "judge_version": ..., "target": ...}``. Over an EMPTY index every scalar is
        ``None`` rather than 0.0 -- a metric over no rows is UNKNOWN, and a 0.0 MAE
        would read as a perfect model.
    """
    if target not in SECONDARY_METRICS_FOR:
        msg = f"Unknown target: '{target}'. Must be one of {sorted(SECONDARY_METRICS_FOR)}."
        raise ValueError(msg)

    game_ids = _index_game_ids(eligibility_index)
    metrics: dict[str, Any] = {
        "target": target,
        "n": len(game_ids),
        "judge_version": JUDGE_VERSION,
        COMPARATOR_PROVENANCE_KEY: LIVE_RESCORE_PROVENANCE,
    }
    if isinstance(eligibility_index, EligibilityIndex):
        metrics["eligibility"] = eligibility_index.as_record()

    if not game_ids:
        # No rows -> no measurement. Null, never a fabricated zero.
        for metric in SECONDARY_METRICS_FOR[target]:
            metrics[metric] = None
        return metrics

    scored = incumbent.set_index(incumbent["game_id"].astype(str)).reindex(game_ids)
    if gold is not None and "actual" in getattr(gold, "columns", ()):
        actual = (
            gold.set_index(gold["game_id"].astype(str))
            .reindex(game_ids)["actual"]
            .to_numpy(dtype=float)
        )
    else:
        actual = scored["actual"].to_numpy(dtype=float)

    if target == "wp":
        wp_metrics = compute_wp_metrics(actual, scored["model_prob"].to_numpy(float))
        for metric in SECONDARY_METRICS_FOR["wp"]:
            metrics[metric] = float(wp_metrics[metric])
    else:
        line_col = _PREDICTION_COLUMN_FOR[target]
        metrics["mae"] = float(
            np.mean(np.abs(actual - scored[line_col].to_numpy(dtype=float)))
        )

    return metrics


def comparator_provenance(comparator: dict[str, Any] | None) -> str:
    """Where a comparator bundle came from -- ``live_paired_rescore`` or ``unattributed``.

    A hand-built dict (the legacy ``scripts/promote_models.py`` frozen-toml path) is
    NOT rejected, but it must never read as a live re-score: a verdict record that
    claimed a live comparator it did not have would be worse than one that admits it.
    """
    if not comparator:
        return UNATTRIBUTED_PROVENANCE
    value = comparator.get(COMPARATOR_PROVENANCE_KEY)
    return value if isinstance(value, str) and value else UNATTRIBUTED_PROVENANCE


def judge_code_digest(sources: list[str] | None = None) -> str:
    """A newline-normalized sha256 over the source of every judging function.

    Every verdict record carries this alongside :data:`JUDGE_VERSION`. The version
    names the policy; the digest catches a change to the policy that forgot to move
    the version. Computed from source at call time, never a transcribed constant.

    Normalization follows the idiom stated at
    ``tests/unit/test_preregistration_ancestry.py:32-39``: this repository has
    ``core.autocrlf=true`` and no ``.gitattributes``, so a digest over raw bytes would
    pin a value that holds only on the machine that measured it.

    Args:
        sources: Override the source list (the mutation control uses this). Defaults
            to the live source of every name in :data:`JUDGE_DIGEST_FUNCTIONS`.

    Returns:
        A 64-character lowercase hex digest.
    """
    if sources is None:
        sources = [
            inspect.getsource(globals()[name]) for name in JUDGE_DIGEST_FUNCTIONS
        ]
    normalized = "\n".join(s.replace("\r\n", "\n").replace("\r", "\n") for s in sources)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# (1) The CLV floor: the logical complement of clv_verdict's negative branch
# ---------------------------------------------------------------------------


def clv_floor_passes(clv_values: Any, alpha: float = SIGNIFICANCE_ALPHA) -> bool:
    """Return True unless the CLV array is SIGNIFICANTLY NEGATIVE (the D24-01 floor).

    The logical complement of ``diagnose.clv_verdict``'s "systematically negative" branch
    (``mean < 0 and p < SIGNIFICANCE_ALPHA``). A target passes the floor when its per-game
    CLV is NOT significantly negative -- i.e. a non-significant negative, a zero, or any
    positive CLV all pass. Insufficient sample (below ``MIN_CLV_SAMPLE``, where the t-test
    cannot run) is a STRICT FAIL: the gate refuses to deploy on an untestable CLV.

    Args:
        clv_values: 1-D array-like of per-game CLV values (``probability_clv`` for WP,
            ``line_clv`` for ATS/OU), already filtered to games with closing odds.
        alpha: Significance level for the negative-tail test. Defaults to the imported
            ``SIGNIFICANCE_ALPHA`` (0.05) so the floor and the diagnosis share one alpha.

    Returns:
        True if the CLV is not significantly negative (passes the floor); False if it is
        significantly negative OR the sample is too small to test.
    """
    sig = clv_significance(clv_values)
    if sig["t"] is None:
        # n < MIN_CLV_SAMPLE: the t-test cannot run, so the CLV is untestable. A model whose
        # CLV cannot be shown non-negative does not clear the floor (strict fail).
        return False
    return not (sig["mean"] < 0 and sig["p"] < alpha)


def clv_non_regression_passes(
    delta_values: Any, alpha: float = SIGNIFICANCE_ALPHA
) -> bool:
    """Return True unless the PAIRED CLV delta is SIGNIFICANTLY WORSE (the D25-01/D25-15 floor).

    The non-regression complement of ``clv_floor_passes``: instead of testing the raw candidate
    CLV against ZERO, this tests the per-game ``candidate_clv - baseline_clv`` delta (same
    game_ids, same gold) against zero. A target passes the slice unless the candidate is
    SIGNIFICANTLY WORSE than the re-scored v1.0 baseline -- i.e. ``mean(delta) < 0 AND
    p < alpha`` is the only FAIL. A near-zero delta (candidate ~= baseline), a positive delta
    (candidate better), or a non-significant negative delta all PASS, regardless of whether the
    candidate's ABSOLUTE CLV is negative. This is the replace-or-retain reading (D25-01): the
    gate must not keep a significantly-worse incumbent in production on a technicality, and it
    must not block a leak-free re-fit that is merely sub-floor-vs-zero where v1.0 was too.

    Reuses the SHARED ``clv_significance`` (the D24-13 parity seam, imported from
    ``backtest.diagnose``, NEVER a new local ttest wrapper) and the one-sided negative-tail
    reading convention of ``clv_floor_passes`` (D24-02). An untestable delta (below
    ``MIN_CLV_SAMPLE``, where ``t is None``) is a STRICT FAIL, matching ``clv_floor_passes``:
    the gate refuses to deploy on an untestable delta.

    Args:
        delta_values: 1-D array-like of per-game ``candidate_clv - baseline_clv`` deltas,
            aligned by game_id (the merge-on-game_id pairing is Plan 25-02's job; this helper
            consumes the already-paired delta array).
        alpha: Significance level for the negative-tail test. Defaults to the imported
            ``SIGNIFICANCE_ALPHA`` (0.05) so the floor and the diagnosis share one alpha.

    Returns:
        True if the delta is not significantly negative (passes non-regression); False if it is
        significantly negative (candidate significantly worse) OR the sample is too small.
    """
    sig = clv_significance(delta_values)
    if sig["t"] is None:
        # n < MIN_CLV_SAMPLE: the paired delta is untestable -> strict fail (same convention as
        # clv_floor_passes: the gate refuses to deploy on an untestable CLV delta).
        return False
    return not (sig["mean"] < 0 and sig["p"] < alpha)


# ---------------------------------------------------------------------------
# (2) Per-season CLV slicing (the one genuinely-new piece; diagnose.py is pooled-only)
# ---------------------------------------------------------------------------


def per_season_clv(
    valid_preds: pd.DataFrame,
    target: str,
    seasons: tuple[int, ...] = HOLDOUT_SEASONS,
) -> dict[int, dict[str, Any]]:
    """Slice per-game CLV by season and run ``clv_significance`` on each slice.

    diagnose.py computes CLV significance POOLED only; this reuses the same season-agnostic
    ``clv_significance`` on per-season slices of the per-game CLV array (the D24-04
    per-season-must-pass input). The returned keys are INTEGER seasons.

    Args:
        valid_preds: A predictions frame already filtered to ``has_closing_odds`` and
            carrying the integer ``season`` column plus the target's CLV column
            (``CLV_COLUMN_FOR[target]``) -- i.e. the post-merge frame from
            ``compute_clv_for_predictions``.
        target: One of "wp", "ats", "ou".
        seasons: Seasons to slice. Defaults to the 2021-2024 holdout.

    Returns:
        ``{season: clv_significance_bundle}`` with one entry per requested season (integer
        keys). A season with no rows yields a bundle with ``n == 0`` and ``t is None``.
    """
    col = CLV_COLUMN_FOR[target]
    return {
        int(s): clv_significance(
            valid_preds.loc[valid_preds["season"] == s, col].to_numpy()
        )
        for s in seasons
    }


# ---------------------------------------------------------------------------
# (3) The single bundle builder -- the ONE source of the evaluate_target input
# ---------------------------------------------------------------------------


def build_candidate_bundle(
    target: str,
    scored_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Turn a scored frame + closing odds + cfg into the canonical ``evaluate_target`` bundle.

    This mirrors ``diagnose._measure_target`` (diagnose.py:416-446) and is the SINGLE source
    of the bundle shape ``evaluate_target`` consumes. ``promote_models.py``, the rewired
    ``retrain_models.py``, and the gate tests all build candidate bundles through this one
    function, so the three call sites cannot drift into different bundle shapes and the
    forced-FAIL/PASS tests have one monkeypatch seam.

    Because ``compute_clv_for_predictions`` RECOMPUTES the CLV column from ``model_prob`` /
    ``model_spread`` / ``model_total``, any pre-existing CLV/odds columns on ``scored_df`` are
    dropped first (same discipline as ``_measure_target``'s ``raw_drop``) to avoid duplicate
    merged odds columns -- and a CLV value pre-baked onto ``scored_df`` would be overwritten,
    which is why this builder (not ``score_deployed_artifacts``) is the seam tests monkeypatch.

    Args:
        target: One of "wp", "ats", "ou".
        scored_df: Predictions in the backtest contract -- ``game_id``, ``season``,
            ``model_prob`` (+ ``model_spread`` for ATS / ``model_total`` for OU), ``actual``.
        odds_df: Normalized closing odds (``game_id``, ``ml_home``, ``ml_away``, and
            ``spread`` / ``total`` as needed by the target's line-CLV computation).
        cfg: The loaded gate config (currently unused for the bundle math; threaded for
            forward-compatibility and a uniform call signature across the three call sites).

    Returns:
        The candidate bundle (see the module docstring for the full contract): ``clv_values``,
        pooled ``mean`` / ``t`` / ``p``, ``per_season`` (int season keys), and either
        ``accuracy`` / ``ece`` / ``brier_score`` (WP) or ``mae`` (ATS/OU).
    """
    _ = cfg  # threaded for a uniform signature across promote / retrain / tests
    if target not in CLV_COLUMN_FOR:
        msg = f"Unknown target: '{target}'. Must be one of {sorted(CLV_COLUMN_FOR)}."
        raise ValueError(msg)

    # CR-01 (fail early with a clear, named error): for ATS/OU the regression MAE is measured
    # against the EXPLICIT line column (model_spread / model_total). This presence check MUST
    # run BEFORE compute_clv_for_predictions -- when the line column is absent, models/clv.py's
    # line_clv recompute silently no-ops, so the CLV_COLUMN_FOR[target] read below would
    # otherwise raise an opaque KeyError('line_clv') and the documented ValueError would be
    # unreachable dead code (caught by tests/unit/test_deploy_gate.py
    # ::test_build_candidate_bundle_missing_line_column_raises).
    if target in ("ats", "ou"):
        line_col = "model_spread" if target == "ats" else "model_total"
        if line_col not in scored_df.columns:
            msg = (
                f"{target} candidate frame missing required '{line_col}' column for the "
                "regression MAE (the line value must be carried explicitly, not via model_prob)"
            )
            raise ValueError(msg)

    # Drop any pre-existing CLV/odds columns before the recompute-merge (mirrors
    # diagnose._measure_target raw_drop) so compute_clv_for_predictions's left-merge does not
    # duplicate odds columns.
    pre_drop = [c for c in _CLV_ODDS_COLS if c in scored_df.columns]
    base = scored_df.drop(columns=pre_drop) if pre_drop else scored_df

    clv_df = compute_clv_for_predictions(base, odds_df, target)
    valid = clv_df.loc[clv_df["has_closing_odds"]]

    col = CLV_COLUMN_FOR[target]
    clv_values = valid[col].to_numpy()
    pooled = clv_significance(clv_values)
    per_season = per_season_clv(valid, target)

    # Per-season RAW candidate CLV arrays (keyed by int season), aligned with `per_season`
    # significance bundles. These are the absolute-floor / absolute-verdict per-season input
    # and the populate-target for the per-season delta keys below.
    per_season_clv_values = {
        int(s): valid.loc[valid["season"] == s, col].to_numpy() for s in HOLDOUT_SEASONS
    }

    bundle: dict[str, Any] = {
        "clv_values": clv_values,
        # PINNED non-regression delta keys (D25-01, Codex HIGH). Plan 25-01 DEFINES them here as
        # None/placeholder; Plan 25-02's merge-on-game_id pairing in promote_models.py POPULATES
        # baseline_clv_values + clv_delta_values (and the per-season equivalents). Shipping them
        # as None now makes a shape/key mismatch when 25-02 wires the real pairing fail loudly
        # (KeyError-free contract); the internal-consistency invariant Plan 25-02 must keep is
        # clv_delta_values == clv_values - baseline_clv_values element-wise.
        "baseline_clv_values": None,
        "clv_delta_values": None,
        "mean": pooled["mean"],
        "t": pooled["t"],
        "p": pooled["p"],
        "n": pooled["n"],
        "per_season": per_season,
        "per_season_clv_values": per_season_clv_values,
        "per_season_baseline_clv_values": dict.fromkeys(
            (int(s) for s in HOLDOUT_SEASONS), None
        ),
        "per_season_clv_delta_values": dict.fromkeys(
            (int(s) for s in HOLDOUT_SEASONS), None
        ),
    }

    if target == "wp":
        wp_metrics = compute_wp_metrics(
            valid["actual"].to_numpy(), valid["model_prob"].to_numpy()
        )
        bundle["accuracy"] = wp_metrics["accuracy"]
        bundle["ece"] = wp_metrics["ece"]
        bundle["brier_score"] = wp_metrics["brier_score"]
    else:
        # Regression MAE is measured against the EXPLICIT line column (margin for ATS, total
        # for OU), NEVER the overloaded "model_prob" (CR-01). model_prob is a convention-only
        # alias the three producers (score_deployed_artifacts, ats_trainer, ou_trainer) happen
        # to set equal to the line value today; if a future producer set model_prob to a
        # cover/over PROBABILITY while leaving the line value in model_spread/model_total, a
        # model_prob-based MAE would be a meaningless pass/fail. The line column was validated
        # present at the top of this function, so this read is safe and unit-explicit.
        line_col = "model_spread" if target == "ats" else "model_total"
        bundle["mae"] = float(
            np.mean(np.abs(valid["actual"].to_numpy() - valid[line_col].to_numpy()))
        )

    return bundle


# ---------------------------------------------------------------------------
# (4) Config loader + validator (git-tracked config/gate.toml, season-key int-normalized)
# ---------------------------------------------------------------------------


def load_gate_config(path: Path = Path("config/gate.toml")) -> dict[str, Any]:
    """Load ``config/gate.toml`` and int-normalize the baseline season keys.

    tomllib requires binary mode. Crucially, tomllib parses the dotted-integer season keys
    (``[baseline.wp.season.2021]``) as STRING keys ("2021"); this normalizes every
    ``baseline.<target>.season`` sub-table to INTEGER keys so the rest of the gate can index
    seasons as ints (matching ``per_season_clv``'s integer keys).

    Args:
        path: Path to the gate config. Defaults to the committed ``config/gate.toml``.

    Returns:
        The parsed config dict with baseline season keys normalized to int.
    """
    with path.open("rb") as f:
        cfg = tomllib.load(f)

    for target_cfg in cfg.get("baseline", {}).values():
        season = target_cfg.get("season")
        if isinstance(season, dict):
            target_cfg["season"] = {int(k): v for k, v in season.items()}

    return cfg


def validate_gate_config(cfg: dict[str, Any]) -> None:
    """Validate the gate config structure, raising ``ValueError`` on a malformed config.

    Input-validation mitigation (T-24-INTEGRITY / T-25-01-validate): a malformed or partial gate
    config could silently change the deploy decision, so the structure is checked explicitly.
    Required: ``gate.alpha``, ``gate.floor_mode`` (one of "non_regression"/"absolute", D25-01),
    the four ``gate.secondary`` tolerance keys (including the D25-02 calibration-band keys
    ``wp_ece_max_increase`` / ``wp_brier_max_increase``), and the ``baseline.{wp,ats,ou}`` TABLES.
    The baseline tables may be EMPTY in Wave 1 (Plan 24-02 ships empty baselines; Plan 24-03 fills
    the numbers), so a present-but-empty ``baseline.<target>`` is NOT an error here. For any target
    whose ``baseline.<target>.season`` table IS populated, its (int-normalized) key set must equal
    the 2021-2024 holdout exactly.

    Args:
        cfg: A loaded gate config (ideally from ``load_gate_config`` so season keys are ints).

    Raises:
        ValueError: If a required key/table is missing, ``gate.floor_mode`` is not a recognized
            value, or a populated season table has the wrong key set.
    """
    gate = cfg.get("gate")
    if not isinstance(gate, dict):
        msg = "gate config missing required [gate] table"
        raise ValueError(msg)

    if "alpha" not in gate:
        msg = "gate config missing required key gate.alpha"
        raise ValueError(msg)

    # D25-01 (T-25-01-validate): floor_mode is REQUIRED and must be a recognized value. A partial
    # config that omits it -- or names an unrecognized mode -- must be rejected BEFORE any deploy
    # decision, so a typo/loosening cannot silently change which floor the gate applies.
    floor_mode = gate.get("floor_mode")
    if floor_mode is None:
        msg = "gate config missing required key gate.floor_mode"
        raise ValueError(msg)
    if floor_mode not in ("non_regression", "absolute"):
        msg = (
            f"gate config has invalid gate.floor_mode {floor_mode!r}; "
            "must be one of 'non_regression' or 'absolute'"
        )
        raise ValueError(msg)

    secondary = gate.get("secondary")
    if not isinstance(secondary, dict):
        msg = "gate config missing required [gate.secondary] table"
        raise ValueError(msg)

    required_secondary = (
        "wp_accuracy_max_drop",
        "regression_mae_max_increase",
        "wp_ece_max_increase",
        "wp_brier_max_increase",
    )
    missing_secondary = [k for k in required_secondary if k not in secondary]
    if missing_secondary:
        msg = (
            "gate config missing required gate.secondary tolerance keys: "
            f"{missing_secondary}"
        )
        raise ValueError(msg)

    baseline = cfg.get("baseline")
    if not isinstance(baseline, dict):
        msg = "gate config missing required [baseline] table"
        raise ValueError(msg)

    for target in ("wp", "ats", "ou"):
        if target not in baseline:
            msg = f"gate config missing required baseline table baseline.{target}"
            raise ValueError(msg)
        # Tolerate an empty Wave-1 baseline table (numeric values are Plan 24-03's job).
        season = baseline[target].get("season")
        if isinstance(season, dict) and season:
            season_keys = {int(k) for k in season}
            if season_keys != set(HOLDOUT_SEASONS):
                msg = (
                    f"baseline.{target}.season has wrong holdout key set {sorted(season_keys)}; "
                    f"expected {sorted(HOLDOUT_SEASONS)}"
                )
                raise ValueError(msg)


# ---------------------------------------------------------------------------
# (5) The per-target deploy decision
# ---------------------------------------------------------------------------


def _secondary_reasons(
    target: str,
    candidate: dict[str, Any],
    comparator: dict[str, Any],
    secondary: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Apply the POOLED secondary non-regression gates; return (passed, reasons).

    WP gates on accuracy (must not drop more than ``wp_accuracy_max_drop``); ATS/OU gate on
    MAE (must not increase more than ``regression_mae_max_increase``).

    D33-11 (owner ruling 2026-09-12): ``comparator`` is a LIVE PAIRED RE-SCORE of the
    deployed incumbent on the SAME gold the candidate was scored on -- the output of
    :func:`live_secondary_metrics`, computed over one shared
    :class:`EligibilityIndex`. It is NOT ``config/gate.toml``'s ``[baseline.*]`` block,
    which stays byte-untouched as a historical record. This is the shape
    ``_pooled_floor_reasons`` has always had; the two secondary readers were converted
    to match it rather than a third pattern being invented.

    A secondary check whose comparator value is absent is recorded as skipped and does
    NOT fail the target (an empty eligibility index yields null scalars, and a
    synthetic bundle may omit a metric).
    """
    reasons: list[str] = []
    passed = True

    if target == "wp":
        cand_acc = candidate.get("accuracy")
        base_acc = comparator.get("accuracy")
        if cand_acc is None or base_acc is None:
            reasons.append("Accuracy comparison skipped (metric not available)")
        else:
            drop = base_acc - cand_acc
            max_drop = secondary["wp_accuracy_max_drop"]
            if drop > max_drop:
                passed = False
                reasons.append(
                    f"Accuracy dropped by {drop:.4f} (max allowed {max_drop})"
                )
            else:
                reasons.append(
                    f"Accuracy delta {cand_acc - base_acc:+.4f} (within {max_drop})"
                )
    else:
        cand_mae = candidate.get("mae")
        base_mae = comparator.get("mae")
        if cand_mae is None or base_mae is None:
            reasons.append("MAE comparison skipped (metric not available)")
        else:
            increase = cand_mae - base_mae
            max_increase = secondary["regression_mae_max_increase"]
            if increase > max_increase:
                passed = False
                reasons.append(
                    f"MAE increased by {increase:.4f} (max allowed {max_increase})"
                )
            else:
                reasons.append(f"MAE delta {increase:+.4f} (within {max_increase})")

    return passed, reasons


def _calibration_reasons(
    candidate: dict[str, Any],
    comparator: dict[str, Any],
    secondary: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Apply the WP calibration non-regression gates (D24-05); return (passed, reasons).

    WP-only: ECE and Brier must not exceed the comparator by more than their
    tolerances. A missing comparator calibration metric is recorded as skipped (does
    not fail).

    D33-11: like :func:`_secondary_reasons`, ``comparator`` is the LIVE PAIRED
    RE-SCORE from :func:`live_secondary_metrics`, never ``config/gate.toml``'s frozen
    ``[baseline.*]`` block.
    """
    reasons: list[str] = []
    passed = True

    for metric, tol_key in (
        ("ece", "wp_ece_max_increase"),
        ("brier_score", "wp_brier_max_increase"),
    ):
        cand_val = candidate.get(metric)
        base_val = comparator.get(metric)
        if cand_val is None or base_val is None:
            reasons.append(f"{metric} comparison skipped (metric not available)")
            continue
        increase = cand_val - base_val
        max_increase = secondary[tol_key]
        if increase > max_increase:
            passed = False
            reasons.append(
                f"{metric} increased by {increase:.4f} (max allowed {max_increase})"
            )
        else:
            reasons.append(f"{metric} delta {increase:+.4f} (within {max_increase})")

    return passed, reasons


def _pooled_floor_reasons(
    candidate: dict[str, Any], floor_mode: str, alpha: float
) -> tuple[bool, str]:
    """Apply the POOLED CLV floor in the active ``floor_mode``; return (passed, reason).

    Under ``"non_regression"`` (D25-01) the floor runs on the paired
    ``candidate["clv_delta_values"]`` (candidate-minus-baseline per-game delta) via
    ``clv_non_regression_passes`` -- fail only if significantly WORSE than the frozen v1.0
    baseline. Under ``"absolute"`` (legacy D24-01) it runs on the raw
    ``candidate["clv_values"]`` via ``clv_floor_passes`` -- fail if significantly negative-vs-zero.
    """
    mean = candidate.get("mean")
    p = candidate.get("p")
    if floor_mode == "non_regression":
        delta = candidate.get("clv_delta_values")
        if clv_non_regression_passes(delta, alpha=alpha):
            return (
                True,
                "Pooled CLV non-regression floor PASS (not significantly worse than v1.0)",
            )
        return False, (
            "Pooled CLV significantly WORSE than v1.0 baseline or untestable "
            "(paired candidate-minus-baseline delta significantly negative)"
        )
    # Legacy absolute-vs-zero floor (floor_mode is "absolute").
    if clv_floor_passes(candidate.get("clv_values"), alpha=alpha):
        return True, f"Pooled CLV floor PASS (mean={mean}, p={p})"
    return (
        False,
        f"Pooled CLV significantly negative or untestable (mean={mean}, p={p})",
    )


def _per_season_floor_reasons(
    candidate: dict[str, Any], floor_mode: str, alpha: float
) -> tuple[bool, list[str]]:
    """Apply the per-season CLV floor in the active ``floor_mode``; return (passed, reasons).

    Fail-closed (WR-04): an empty/absent per-season map provides NO evidence and must NOT be
    reported as "all holdout seasons passed". Under ``"non_regression"`` each season's paired
    DELTA slice (``per_season_clv_delta_values[season]``) is tested via
    ``clv_non_regression_passes``; under ``"absolute"`` each season's raw CLV slice (the
    ``per_season`` clv_significance bundle, or a raw ``clv_values`` array on a synthetic bundle)
    is tested via the legacy absolute reading.
    """
    reasons: list[str] = []

    if floor_mode == "non_regression":
        per_season_delta = candidate.get("per_season_clv_delta_values", {})
        if not per_season_delta:
            return False, [
                "Per-season-must-pass enabled but no per-season CLV slices provided"
            ]
        passed = True
        for season in sorted(per_season_delta):
            delta_arr = per_season_delta[season]
            season_pass = clv_non_regression_passes(delta_arr, alpha=alpha)
            if not season_pass:
                passed = False
                reasons.append(
                    f"Season {season} CLV non-regression floor FAIL "
                    "(paired delta significantly worse than v1.0 or untestable)"
                )
        if passed:
            reasons.append(
                "Per-season CLV non-regression floor PASS (all holdout seasons)"
            )
        return passed, reasons

    # floor_mode == "absolute" (legacy D24-01 per-season floor).
    per_season = candidate.get("per_season", {})
    if not per_season:
        return False, [
            "Per-season-must-pass enabled but no per-season CLV slices provided"
        ]
    passed = True
    for season in sorted(per_season):
        season_sig = per_season[season]
        season_arr = season_sig.get("clv_values")
        # Synthetic test bundles may carry the raw array under "clv_values"; the real
        # build_candidate_bundle stores clv_significance bundles. Re-test from the array when
        # present, else re-derive the pass/fail from the stored {mean, t, p}.
        if season_arr is not None:
            season_pass = clv_floor_passes(season_arr, alpha=alpha)
        elif season_sig.get("t") is None:
            season_pass = False
        else:
            season_pass = not (season_sig["mean"] < 0 and season_sig["p"] < alpha)
        if not season_pass:
            passed = False
            reasons.append(
                f"Season {season} CLV floor FAIL "
                f"(mean={season_sig.get('mean')}, p={season_sig.get('p')}, "
                f"n={season_sig.get('n')})"
            )
    if passed:
        reasons.append("Per-season CLV floor PASS (all holdout seasons)")
    return passed, reasons


def _absolute_verdict(
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, Any]]:
    """Compute the always-on absolute-vs-zero verdict on the RAW candidate CLV (D25-01).

    REGARDLESS of floor_mode, the gate records the absolute-vs-zero ``clv_significance`` of the
    raw candidate CLV (``clv_values``) -- this is the bettable-bar readout for Phases 26-27
    (removing a CLV leak is NOT a positive market edge -- Pitfall 3), never the deploy decision
    under non_regression. Computed from the raw array when present; falls back to the bundle's
    stored pooled {mean, t, p} when only those are supplied (synthetic fixtures).

    Returns:
        ``(pooled_absolute_verdict, {season: absolute_verdict})`` -- the pooled raw-candidate
        CLV significance plus a per-season raw-candidate significance dict (empty if the raw
        per-season arrays are not on the bundle).
    """
    clv_values = candidate.get("clv_values")
    if clv_values is not None:
        pooled = clv_significance(clv_values)
    else:
        pooled = {
            "n": candidate.get("n"),
            "mean": candidate.get("mean"),
            "t": candidate.get("t"),
            "p": candidate.get("p"),
            "ci95": None,
        }

    # Per-season absolute verdict. The real build_candidate_bundle stores the raw-candidate
    # per-season clv_significance under `per_season` (per_season_clv runs clv_significance on the
    # raw CLV slice). That IS the per-season absolute-vs-zero verdict. When raw per-season arrays
    # are supplied under per_season_clv_values, recompute from them (the authoritative array);
    # otherwise fall back to the stored `per_season` significance bundles.
    absolute_per_season: dict[int, Any] = {}
    per_season_raw = candidate.get("per_season_clv_values", {})
    for season, arr in per_season_raw.items():
        if arr is not None:
            absolute_per_season[int(season)] = clv_significance(arr)
    if not absolute_per_season:
        for season, sig in candidate.get("per_season", {}).items():
            absolute_per_season[int(season)] = sig
    return pooled, absolute_per_season


def evaluate_target(
    target: str,
    candidate: dict[str, Any],
    comparator: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether a candidate model may ship for one target (the per-target deploy gate).

    The decision is the AND of all applicable checks:
      1. POOLED CLV floor in ``gate.floor_mode`` (D25-01): under ``"non_regression"`` the paired
         candidate-minus-baseline delta must not be significantly WORSE than the frozen v1.0
         baseline; under ``"absolute"`` (legacy D24-01) the raw candidate CLV must not be
         significantly negative-vs-zero.
      2. If ``gate.per_season_must_pass``: the same floor on EVERY holdout season's slice (D24-04
         -- one significantly-worse/negative season fails the target; an insufficient-sample
         season also fails per the strict floor rule; an absent per-season map fails closed).
      3. POOLED secondary non-regression vs baseline (D24-05 secondary: WP accuracy / ATS-OU MAE).
      4. For WP, if ``gate.calibration_in_gate``: ECE + Brier non-regression vs baseline.

    REGARDLESS of floor_mode the absolute-vs-zero verdict on the RAW candidate CLV is ALWAYS
    computed and attached (``absolute_verdict`` pooled + ``absolute_per_season``) for the
    bettable-bar readout (Phases 26-27, Pitfall 3) -- it is NEVER the deploy decision under
    non_regression.

    WHERE THE COMPARATOR COMES FROM (D33-11, owner ruling 2026-09-12). It is a LIVE
    PAIRED RE-SCORE of the deployed incumbent on the SAME gold the candidate was
    scored on, produced by :func:`live_secondary_metrics` over one shared
    :class:`EligibilityIndex`. It is NOT ``config/gate.toml``'s ``[baseline.*]``
    block. That block stays byte-untouched as a HISTORICAL RECORD: two of Phase 33's
    five deliberate tripwires assert it reproduces, so re-freezing it after a gold
    rebuild would clear a disclosure by making it pass. Re-scoring live makes the
    whole judge gold-invariant by construction instead.

    The frozen block's divergence from a fresh re-score in 47 of 68 fields is
    PRESERVED, not discharged, by that ruling, and remains an open disclosure.

    Callers that hand this a bundle without :data:`LIVE_RESCORE_PROVENANCE` are not
    rejected -- the legacy ``scripts/promote_models.py`` path still supplies a
    frozen-toml bundle -- but :func:`comparator_provenance` reports them as
    ``unattributed`` so a verdict record cannot imply a live re-score it did not have.

    Args:
        target: One of "wp", "ats", "ou".
        candidate: The candidate bundle from ``build_candidate_bundle``.
        comparator: The live re-scored incumbent bundle for this target (same key
            shape as the candidate -- ``accuracy``/``ece``/``brier_score`` for WP or
            ``mae`` for ATS/OU). Returned unchanged under the ``baseline`` /
            ``v1_metrics`` result keys, which downstream readers still use.
        cfg: The loaded gate config (reads ``gate.alpha``, ``gate.floor_mode``,
            ``gate.per_season_must_pass``, ``gate.calibration_in_gate``, ``gate.secondary``).

    Returns:
        ``{"passed": bool, "reasons": list[str], "candidate": {...}, "baseline": {...},
        "per_season": {...}, "absolute_verdict": {...}, "absolute_per_season": {...},
        "v1_metrics": {...}, "v2_metrics": {...}}``. The ``v1_metrics``/``v2_metrics`` aliases
        keep the shape compatible with ``scripts.retrain_models.print_gating_summary``.
    """
    gate = cfg["gate"]
    alpha = gate["alpha"]
    # floor_mode is a REQUIRED key (validate_gate_config enforces it); default to the D25-01
    # non_regression mode if a raw in-memory dict omits it, but the committed config always
    # carries it explicitly.
    floor_mode = gate.get("floor_mode", "non_regression")
    secondary = gate["secondary"]
    reasons: list[str] = []
    passed = True

    # Always-on absolute-vs-zero verdict on the RAW candidate CLV (the bettable bar, D25-01 /
    # Pitfall 3). Computed in EVERY mode; never the deploy decision under non_regression.
    absolute_verdict, absolute_per_season = _absolute_verdict(candidate)

    # (1) Pooled CLV floor (mode-aware).
    pooled_pass, pooled_reason = _pooled_floor_reasons(candidate, floor_mode, alpha)
    passed = passed and pooled_pass
    reasons.append(pooled_reason)

    # (2) Per-season-must-pass CLV floor (mode-aware; fail-closed on an absent map -- WR-04).
    # WR-01/D25-01: under floor_mode=non_regression this per-season floor is a NON-REGRESSION
    # floor -- each holdout season's PAIRED candidate-minus-baseline CLV delta must not be
    # significantly WORSE than the frozen v1.0 baseline for that season (NOT an absolute-vs-zero
    # floor). This is the deliberate, reviewed Phase-25 policy change (D25-01): an absolute floor
    # would keep a significantly-worse incumbent in production over a technicality, and would
    # block a leak-free re-fit on the line-CLV targets (ATS/OU) whose frozen v1.0 baseline is
    # itself significantly negative in some seasons. The absolute-vs-zero verdict is still
    # computed above (the bettable bar) -- it is recorded, not used to gate, under non_regression.
    # Under floor_mode=absolute the legacy per-season absolute-vs-zero floor applies unchanged.
    if gate.get("per_season_must_pass"):
        ps_pass, ps_reasons = _per_season_floor_reasons(candidate, floor_mode, alpha)
        passed = passed and ps_pass
        reasons.extend(ps_reasons)

    # (3) Pooled secondary non-regression, against the LIVE re-scored comparator (D33-11).
    sec_passed, sec_reasons = _secondary_reasons(
        target, candidate, comparator, secondary
    )
    passed = passed and sec_passed
    reasons.extend(sec_reasons)

    # (4) WP calibration-in-gate (D24-05), same comparator source.
    if target == "wp" and gate.get("calibration_in_gate"):
        cal_passed, cal_reasons = _calibration_reasons(candidate, comparator, secondary)
        passed = passed and cal_passed
        reasons.extend(cal_reasons)

    return {
        "passed": passed,
        "reasons": reasons,
        "candidate": candidate,
        # Key name UNCHANGED so downstream readers do not move (print_gating_summary,
        # the 2x2 readout, the phase readouts). What it CONTAINS changed: under D33-11
        # this is the live re-scored comparator, not the frozen gate.toml block.
        "baseline": comparator,
        "comparator_provenance": comparator_provenance(comparator),
        "per_season": candidate.get("per_season", {}),
        # Always-on absolute-vs-zero verdict on the raw candidate CLV (D25-01 bettable bar).
        "absolute_verdict": absolute_verdict,
        "absolute_per_season": absolute_per_season,
        # Aliases kept for print_gating_summary compatibility (Plan 24-04 rewire).
        "v1_metrics": comparator,
        "v2_metrics": candidate,
    }
