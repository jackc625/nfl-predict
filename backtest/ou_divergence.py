"""Thin O/U CLV-to-ROI divergence harness (Phase 26, OUM-01).

This is the "thin entry where none fits" allowance (the Phase-22 ``backtest/diagnose.py``
precedent): a diagnosis tool that CALLS existing ``backtest/`` + ``models/`` functions on the
canonical Phase-20 gold and adds only genuinely-new MEASUREMENTS. It answers the milestone's
load-bearing question -- why does the deployed O/U model's strong, significant line_clv edge
(+1.11) sit on a below-breakeven hit-rate? -- with evidence, before any monetization code.

Genuinely-new measurements (vs the consumed scorers):
  - the odds-integrity preamble (D26-04): provenance vs disclosed synthetic-timestamp reality;
  - the bias-vs-anticipation decomposition (D26-05, the LEAD section): the model-picks-over
    share and the directional reading of the +1.11 as systematic upward total bias;
  - the prior-season bias-adjusted re-score (D26-18, this module's ``debiased_rescore``);
  - [Plan 26-03, NOT here] the extended bucket sweep (D26-06), the trial registry + BH-FDR
    (D26-10), and the throwaway EV preview (D26-07).

HARD BOUNDARY (carried from the Phase-22 harness it mirrors): this module imports NO ``train_*``
module and NEVER writes ``data/`` -- it LOADS the deployed artifact via ``load_model_artifact``
and runs inference only, NO re-fit, NO gold rebuild. It does NOT edit its own judges -- production
``models/clv.py``, ``config/gate.toml``, and ``backtest/simulation.py`` are CONSUMED verbatim,
never modified (the bias-signed CLV lives ONLY in this harness, D26-05; the self-judging boundary
is proven by a sha256 byte-identity test). The deployed-artifact identity is asserted via the
loaded artifact's directory name / ``artifacts/latest.json``, NEVER a metadata version key (the
OU metadata.json has no ``version`` key).

Data reality this harness honestly discloses (RESEARCH.md Finding 1 / Pitfall 1): the stored
silver odds carry ONE synthetic-Friday line per game (freeze == closing; 8 distinct synthetic
``snapshot_ts``). A true freeze-vs-closing CLV delta is therefore UNMEASURABLE from stored data;
the bias decomposition stands on the over-share + directional analysis, which IS measurable. The
"anticipation" component a bettor could capture is unmeasurable from a single stored snapshot --
which itself strengthens the bias reading (there is no pre-close line to anticipate from).

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backtest.diagnose import (
    CLV_COLUMN_FOR,
    SIGNIFICANCE_ALPHA,
    apply_blended_cut,
    clv_significance,
    score_deployed_artifacts,
)
from models.artifacts import load_model_artifact
from models.clv import compute_clv_for_predictions
from utils import get_logger

logger = get_logger(__name__)

# Re-export the consumed significance constants so callers/tests do not redefine them.
__all__ = [
    "CLV_COLUMN_FOR",
    "DEBIAS_RESIDUAL_THRESHOLD",
    "DEPLOYED_OU_ARTIFACT",
    "SIGNIFICANCE_ALPHA",
    "bias_vs_anticipation",
    "debiased_rescore",
    "integrity_preamble",
]

# Backtest holdout window (walk-forward, 2021-2024). Matches BacktestConfig defaults.
HOLDOUT_FIRST_SEASON = 2021
HOLDOUT_LAST_SEASON = 2024

# The expected resolved deployed OU artifact identity (the RETAINED v1.0, D25-14). The harness
# asserts it scores THIS artifact (Pitfall 9: prove on the path that serves).
DEPLOYED_OU_ARTIFACT = "ou_20260326_163930"

# Raw silver odds path -- read ONLY for the provenance/timestamp disclosure columns (sportsbook,
# is_live, snapshot_ts) that the normalized BacktestEngine loader strips. Never written.
_RAW_SILVER_ODDS_PATH = Path("data/silver/odds_snapshot.parquet")

# Allowed real-odds sportsbook labels (the OUM-06 provenance spirit pulled forward, D26-04(i)).
_ALLOWED_SPORTSBOOKS = frozenset({"consensus", "draftkings"})

# ---------------------------------------------------------------------------
# Pre-registered analysis bands (D26-08 -- LOCKED; Plan 26-03 consumes these as the sweep inputs).
# Resolved from EMPIRICAL 2021-2024 closing-totals data (RESEARCH.md Pre-Registration Inputs).
# Defined here so the harness skeleton carries them from the start; NO adaptive adjustment after
# seeing results (forking-paths guard).
# ---------------------------------------------------------------------------

# Distance from the nearest INTEGER total (half the totals are .5, so on-integer-vs-off-it is the
# meaningful cut for totals -- NFL totals have WEAK key numbers, a broad hump ~43-44).
KEY_TOTAL_DISTANCE_BANDS = [0.0, 0.5, 1.0, 1.5]

# The 43-44 region is the empirical key-total cluster (weak; the weakness is itself a finding).
KEY_TOTAL_CLUSTER = (43.0, 44.0)

# Totals-regime boundaries (empirical tertiles): low < 42.0, mid [42.0, 46.5], high > 46.5.
TOTALS_REGIME_BOUNDARIES = {
    "low_max": 42.0,
    "mid_min": 42.0,
    "mid_max": 46.5,
    "high_min": 46.5,
}

# Week groupings (the dynamic blend varies by week, D26-06).
WEEK_GROUPINGS = {
    "early": (1, 6),
    "mid": (7, 13),
    "late": (14, 18),
    "playoffs": (19, 22),
}

# Structural bar for NAMING a sub-population (D26-11): N floor across the window AND same-direction
# in at least MIN_SEASONS_SAME_DIRECTION of 4 seasons individually. (The per-season DIRECTION
# metric is pre-registered in Plan 26-03 -- graded edge vs breakeven / EV sign, NOT line_clv sign.)
N_FLOOR = 175
MIN_SEASONS_SAME_DIRECTION = 3

# Coverage floor (D26-17): below this a section is downgraded to "partial evidence."
COVERAGE_FLOOR = 0.80

# SD sensitivity band for the EV preview (D26-16): the reproduced pooled residual SD ~12.95 is
# centered in-band. (Plan 26-03 consumes this in the EV preview.)
SD_SENSITIVITY_BAND = (12.5, 14.5)

# Pre-registered residual threshold for the de-biased re-score interpretation (D26-18): if the
# pooled prior-season bias-adjusted line_clv collapses below this magnitude (in total points), the
# +1.11 was "nothing underneath" the bias (strong no-go evidence); otherwise "residual
# anticipation" remains (previews a Phase-27 bias-correction design). A mechanical threshold the
# harness applies -- NOT adjusted after seeing results (forking-paths guard).
DEBIAS_RESIDUAL_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deployed_ou_preds(preds: pd.DataFrame | None) -> pd.DataFrame:
    """Return the deployed-artifact OU prediction frame (D26-01), scoring if not supplied.

    Mirrors ``run_diagnosis``: ``score_deployed_artifacts("ou")`` LOADS the deployed v1.0 OU
    artifact via ``load_model_artifact`` and runs inference single-pass over 2021-2024 gold. No
    re-fit, no gold write.
    """
    if preds is not None:
        return preds
    return score_deployed_artifacts("ou")


def _ou_clv_valid(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Compute production O/U CLV and return the with-closing-line population (n~1087).

    Consumes ``compute_clv_for_predictions`` verbatim (line_clv = model_total - closing_total,
    unconditioned on bet direction -- the definitional root of the bias question). Never
    redefines ``compute_line_clv``.
    """
    clv = compute_clv_for_predictions(preds, odds, "ou")
    return clv[clv["has_closing_odds"]].copy()


def _resolve_deployed_ou_artifact() -> str:
    """Resolve the deployed OU artifact identity via a STABLE provenance source.

    Uses the loaded artifact's directory NAME (which mirrors ``artifacts/latest.json``'s ou
    pointer) -- NOT a metadata.json ``version`` key, because the OU metadata.json has no such key
    (Codex LOW, verification-confirmed). Returns e.g. "ou_20260326_163930".
    """
    artifact = load_model_artifact("ou")
    return Path(artifact["artifact_dir"]).name


# ---------------------------------------------------------------------------
# (0) Odds-integrity preamble (D26-04)
# ---------------------------------------------------------------------------


def integrity_preamble(odds_df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Build the odds-integrity preamble that OPENS the diagnosis (D26-04).

    Carries TWO DISTINCT provenance flags (the Codex HIGH terminology split -- do NOT conflate
    these concepts):
      - ``mock_free_odds`` (bool): no mock/synthetic ODDS contamination. True iff every odds row
        has a sportsbook in {consensus, draftkings} AND is_live is all False. Hard-asserts True
        (raising an actionable ValueError naming offending rows otherwise) -- this pulls the
        OUM-06 provenance spirit forward one phase (D26-04(i)).
      - ``synthetic_snapshot_ts`` (bool): the disclosed fabricated single-stamp TIMESTAMP reality.
        True iff the snapshot_ts has a small distinct count AND each game has exactly one snapshot
        (freeze == closing). This is a data-reality DISCLOSURE, not a tampering risk -- a SEPARATE
        concept from mock-odds contamination (D26-04(ii)).

    Also characterizes the 52 closing-snapshot-missing excluded games for selection bias
    (D26-04(iii)) and reports coverage/exclusion counts (D26-04(iv)).

    Args:
        odds_df: Optional normalized closing-odds frame (from ``BacktestEngine._load_closing_odds``).
            When None, loaded via the engine loader. The raw silver provenance columns
            (sportsbook, is_live, snapshot_ts) are read read-only from the silver parquet
            regardless, since the normalized loader strips them.

    Returns:
        Dict with at least: ``mock_free_odds``, ``synthetic_snapshot_ts``, ``snapshot_ts_distinct``,
        ``snapshots_per_game_max``, ``deployed_artifact``, ``n_total``, ``n_with_line``,
        ``n_excluded``, ``excluded_characterization``, ``coverage``.

    Raises:
        ValueError: If any odds row is is_live=True or has a sportsbook outside the allowed set
            (a mock/synthetic-odds contamination signal).
    """
    if odds_df is None:
        from backtest.engine import BacktestEngine

        odds_df = BacktestEngine()._load_closing_odds()

    # -- (i) PROVENANCE: no mock/synthetic ODDS (read raw silver provenance columns, read-only) --
    raw = pd.read_parquet(_RAW_SILVER_ODDS_PATH)
    sportsbooks = set(raw["sportsbook"].dropna().unique())
    bad_books = sportsbooks - _ALLOWED_SPORTSBOOKS
    live_rows = raw[raw["is_live"]] if "is_live" in raw.columns else raw.iloc[0:0]
    if bad_books or len(live_rows) > 0:
        offenders = (
            raw[(~raw["sportsbook"].isin(_ALLOWED_SPORTSBOOKS)) | raw["is_live"]][
                "game_id"
            ]
            .head(10)
            .tolist()
        )
        msg = (
            "Odds provenance check FAILED (mock/synthetic-odds contamination): "
            f"unexpected sportsbooks={sorted(bad_books)}, is_live rows={len(live_rows)}; "
            f"first offending game_ids={offenders}"
        )
        raise ValueError(msg)
    mock_free_odds = True

    # -- (ii) TIMESTAMP DISCLOSURE: the synthetic single-snapshot reality (freeze == closing) --
    snapshot_ts_distinct = int(raw["snapshot_ts"].astype(str).nunique())
    snapshots_per_game_max = int(raw.groupby("game_id").size().max())
    # Synthetic iff few distinct stamps AND exactly one line per game (a fabricated single stamp).
    synthetic_snapshot_ts = bool(
        snapshot_ts_distinct <= 12 and snapshots_per_game_max == 1
    )

    # -- Deployed-artifact identity (stable provenance source, not a metadata version key) --
    deployed_artifact = _resolve_deployed_ou_artifact()

    # -- (iii)/(iv) Coverage + the 52-excluded characterization --
    preds = _deployed_ou_preds(None)
    clv = compute_clv_for_predictions(preds, odds_df, "ou")
    n_total = len(clv)
    with_line_mask = clv["has_closing_odds"]
    n_with_line = int(with_line_mask.sum())
    n_excluded = int((~with_line_mask).sum())

    excluded = clv[~with_line_mask]
    included = clv[with_line_mask]
    excluded_characterization = {
        "n": n_excluded,
        "by_season": {
            int(s): int(c)
            for s, c in excluded["season"].value_counts().sort_index().items()
        },
        "by_week_group": _week_group_counts(excluded),
        "excluded_mean_actual_total": (
            float(excluded["actual"].mean()) if n_excluded else None
        ),
        "included_mean_actual_total": (
            float(included["actual"].mean()) if n_with_line else None
        ),
    }

    coverage = float(n_with_line / n_total) if n_total else 0.0

    return {
        "mock_free_odds": mock_free_odds,
        "synthetic_snapshot_ts": synthetic_snapshot_ts,
        "snapshot_ts_distinct": snapshot_ts_distinct,
        "snapshots_per_game_max": snapshots_per_game_max,
        "deployed_artifact": deployed_artifact,
        "n_total": n_total,
        "n_with_line": n_with_line,
        "n_excluded": n_excluded,
        "coverage": coverage,
        "coverage_floor": COVERAGE_FLOOR,
        "excluded_characterization": excluded_characterization,
    }


def _week_group_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count games per pre-registered week grouping (coverage companion, no orphan metric)."""
    counts: dict[str, int] = {}
    weeks = frame["week"]
    for label, (lo, hi) in WEEK_GROUPINGS.items():
        counts[label] = int(((weeks >= lo) & (weeks <= hi)).sum())
    return counts


# ---------------------------------------------------------------------------
# (1) Bias-vs-anticipation decomposition (D26-05) -- the LEAD section
# ---------------------------------------------------------------------------


def bias_vs_anticipation(
    preds: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Decompose the deployed O/U +1.11 line_clv into a directional-bias reading (D26-05).

    The production O/U ``line_clv = model_total - closing_total`` is UNCONDITIONED on bet
    direction and never references a freeze line, so a model that systematically predicts high
    totals scores positive CLV whenever scoring rose (2022-2024) -- this is the BIAS component.
    The "anticipation" component a bettor could capture is UNMEASURABLE from the stored
    single-snapshot data (freeze == closing; Pitfall 1) -- we disclose this rather than fabricate
    a freeze line. The reproduced model-picks-over share (~0.727 pooled, with the 2021 sign-flip)
    is the load-bearing evidence for the bias reading.

    Composes ``score_deployed_artifacts`` -> ``compute_clv_for_predictions`` -> the with-line
    filter; reuses ``clv_significance`` for the pooled t/p/CI (never re-derived). Also computes
    the blended-stream line_clv via ``apply_blended_cut`` for the D26-02 side-by-side
    (raw +1.11 vs blended +0.70).

    Args:
        preds: Optional deployed-artifact OU prediction frame. When None, scored single-pass.
        odds: Optional normalized closing odds. When None, loaded via the engine loader.

    Returns:
        Dict with: ``n``, ``n_excluded``, ``pooled_line_clv``, ``pooled_over_share``,
        ``significance`` (the clv_significance bundle), ``blended_pooled_line_clv``,
        ``per_season`` (dict[int season -> {n, over_share, line_clv}]), ``per_game`` (a frame with
        game_id/season/model_total/total/line_clv/model_over for determinism + downstream slicing),
        ``bias_reading`` (str), and ``anticipation_measurable`` (False -- the disclosed reality).
    """
    if odds is None:
        from backtest.engine import BacktestEngine

        odds = BacktestEngine()._load_closing_odds()

    preds = _deployed_ou_preds(preds)
    clv = compute_clv_for_predictions(preds, odds, "ou")
    n_total = len(clv)
    valid = clv[clv["has_closing_odds"]].copy()
    n = len(valid)
    n_excluded = n_total - n

    valid["model_over"] = valid["model_total"] > valid["total"]

    significance = clv_significance(valid["line_clv"].to_numpy())
    pooled_line_clv = float(significance["mean"])
    pooled_over_share = float(valid["model_over"].mean())

    per_season: dict[int, dict[str, Any]] = {}
    for season in sorted(int(s) for s in valid["season"].unique()):
        sub = valid[valid["season"] == season]
        per_season[season] = {
            "n": len(sub),
            "over_share": float(sub["model_over"].mean()),
            "line_clv": float(sub["line_clv"].mean()),
        }

    # Blended stream (D26-02): mirrors the LIVE dynamic blend (O/U weight 0.604).
    blended = apply_blended_cut(preds, odds, "ou")
    blended_valid = blended[blended["has_closing_odds"]]
    blended_pooled_line_clv = (
        float(blended_valid["line_clv"].mean()) if not blended_valid.empty else None
    )

    per_game = valid[
        ["game_id", "season", "week", "model_total", "total", "line_clv", "model_over"]
    ].copy()

    bias_reading = (
        "The deployed O/U line_clv is unconditioned on bet direction and never references a "
        "freeze line, so a model that systematically predicts high totals scores positive CLV "
        "whenever scoring rose. The pooled model-picks-over share is "
        f"{pooled_over_share:.3f} (2022-2024 strongly over, 2021 sign-flipped), so the +"
        f"{pooled_line_clv:.4f} pooled line_clv reads as a systematic upward total bias rather "
        "than market-beating close-anticipation. The anticipation component is unmeasurable "
        "from the single stored snapshot (freeze == closing)."
    )

    return {
        "n": n,
        "n_excluded": n_excluded,
        "pooled_line_clv": pooled_line_clv,
        "pooled_over_share": pooled_over_share,
        "significance": significance,
        "blended_pooled_line_clv": blended_pooled_line_clv,
        "per_season": per_season,
        "per_game": per_game,
        "bias_reading": bias_reading,
        "anticipation_measurable": False,
    }


# ---------------------------------------------------------------------------
# (2) Prior-season bias-adjusted re-score (D26-18)
# ---------------------------------------------------------------------------


def debiased_rescore(
    preds: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Prior-season bias-adjusted deployed-model read (D26-18, walk-forward estimation only).

    LOCKED decision D26-18: "walk-forward estimation only" -- the correction for season N is
    estimated ONLY on seasons < N. This is NOT a true walk-forward model RE-FIT (the deployed
    single-pass frame is corrected, not re-trained); per the Codex review the output is labeled a
    "prior-season bias-adjusted deployed-model read." Analysis-layer only: no model artifact is
    touched, no ``data/`` is written, and production ``models/clv.py`` is byte-identical.

    For each season N in 2022-2024: estimate the model's mean total bias as the prior-seasons
    mean of (predicted total minus actual total) (seasons < N only), construct a CORRECTED
    prediction-frame copy whose predicted total has the prior-season bias subtracted off for
    season N, then recompute that season's line_clv by CALLING the production CLV function
    ``compute_clv_for_predictions`` on the corrected copy (the production definition is reused
    verbatim -- the harness never re-derives the CLV subtraction by hand). Season 2021 has no
    prior seasons and is EXCLUDED from the de-biased read (the no-prior-seasons caveat is recorded).

    The ``interpretation`` is a MECHANICAL pre-registered threshold (D26-18): if
    ``abs(pooled_debiased_line_clv) < DEBIAS_RESIDUAL_THRESHOLD`` (0.25 points) the +1.11 was
    "nothing underneath" the bias (strong no-go evidence); otherwise "residual anticipation"
    remains (previews a Phase-27 bias-correction design).

    Args:
        preds: Optional deployed-artifact OU prediction frame. When None, scored single-pass.
        odds: Optional normalized closing odds. When None, loaded via the engine loader.

    Returns:
        Dict with: ``per_season_debiased_line_clv`` (dict[int season -> {n, debiased_line_clv,
        bias_subtracted}] for 2022-2024 ONLY), ``pooled_debiased_line_clv`` (float over 2022-2024),
        ``per_season_bias_subtracted`` (dict[int season -> float]), ``caveat`` (str mentioning the
        no-prior-seasons 2021 exclusion), ``interpretation`` ("nothing underneath" |
        "residual anticipation"), ``residual_threshold`` (the pre-registered cut), and
        ``estimation`` (the D26-18 "walk-forward estimation only" decision language, verbatim).
    """
    if odds is None:
        from backtest.engine import BacktestEngine

        odds = BacktestEngine()._load_closing_odds()

    preds = _deployed_ou_preds(preds)
    valid = _ou_clv_valid(preds, odds)

    # The de-biased read covers seasons WITH at least one prior season (2022-2024); 2021 excluded.
    debias_seasons = [
        s
        for s in sorted(int(s) for s in valid["season"].unique())
        if s > HOLDOUT_FIRST_SEASON
    ]

    per_season_debiased: dict[int, dict[str, Any]] = {}
    per_season_bias_subtracted: dict[int, float] = {}
    pooled_debiased_values: list[float] = []

    for season in debias_seasons:
        # Bias estimated on PRIOR seasons ONLY (walk-forward; no season N row in its own estimate).
        prior = valid[valid["season"] < season]
        bias_n = float((prior["model_total"] - prior["actual"]).mean())

        # CORRECTED prediction-frame copy for season N: predicted total with bias_N subtracted off.
        # Recompute line_clv by CALLING the production CLV function (the subtraction is never
        # re-derived by hand here -- the production definition is reused verbatim).
        season_game_ids = valid[valid["season"] == season]["game_id"]
        corrected = preds[preds["game_id"].isin(season_game_ids)].copy()
        corrected["model_total"] = corrected["model_total"] - bias_n
        rescored = compute_clv_for_predictions(corrected, odds, "ou")
        rescored_valid = rescored[rescored["has_closing_odds"]]

        season_clv = float(rescored_valid["line_clv"].mean())
        per_season_debiased[season] = {
            "n": len(rescored_valid),
            "debiased_line_clv": season_clv,
            "bias_subtracted": bias_n,
        }
        per_season_bias_subtracted[season] = bias_n
        pooled_debiased_values.extend(rescored_valid["line_clv"].tolist())

    pooled_debiased_line_clv = (
        float(sum(pooled_debiased_values) / len(pooled_debiased_values))
        if pooled_debiased_values
        else None
    )

    interpretation = (
        "nothing underneath"
        if pooled_debiased_line_clv is not None
        and abs(pooled_debiased_line_clv) < DEBIAS_RESIDUAL_THRESHOLD
        else "residual anticipation"
    )

    caveat = (
        "Season 2021 is excluded from the de-biased read: it has no prior seasons to estimate a "
        "bias from (walk-forward estimation only, D26-18). The de-biased read covers 2022-2024."
    )

    return {
        "per_season_debiased_line_clv": per_season_debiased,
        "pooled_debiased_line_clv": pooled_debiased_line_clv,
        "per_season_bias_subtracted": per_season_bias_subtracted,
        "caveat": caveat,
        "interpretation": interpretation,
        "residual_threshold": DEBIAS_RESIDUAL_THRESHOLD,
        "estimation": "walk-forward estimation only",
    }
