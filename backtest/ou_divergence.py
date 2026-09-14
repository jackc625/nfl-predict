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

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, norm

from backtest.diagnose import (
    CLV_COLUMN_FOR,
    SIGNIFICANCE_ALPHA,
    apply_blended_cut,
    clv_significance,
    score_deployed_artifacts,
)
from backtest.diagnose import (
    _results_like as _diag_results_like,
)
from backtest.simulation import (
    SLIPPAGE_POINTS as SLIPPAGE_POINTS_DEFAULT,
)
from backtest.simulation import (
    BettingSimulator,
    SimulationConfig,
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
    "EDGE_MAGNITUDE_GRID",
    "HIGH_TOTAL_BOUNDARY_PREHOLD",
    "HOLD_SEASONS",
    "N_FLOOR",
    "OU_BREAKEVEN_HIT_RATE",
    "PRE_HOLD_SEASONS",
    "SD_SENSITIVITY_BAND",
    "SIGNIFICANCE_ALPHA",
    "SPORTSBOOK_PREFERENCE",
    "HoldSeasonLeakageError",
    "bias_vs_anticipation",
    "bucket_count_parity",
    "debiased_rescore",
    "dedupe_odds_by_book_preference",
    "derive_high_total_boundary",
    "edge_magnitude_sweep",
    "extended_bucket_sweep",
    "integrity_preamble",
    "name_survivable_subpopulation",
    "run_ou_divergence_diagnosis",
    "throwaway_ev_preview",
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

# The order a game's price is CHOSEN in when the stored table carries more than one book for it
# (WR-08). Stated BY NAME rather than left to parquet row order: the pricing paths used to do
# ``drop_duplicates(subset=["game_id"], keep="first")``, which selects whichever row happens to
# appear first in the file. That is deterministic for a fixed file but NOT stable -- appending a
# ``draftkings`` row, or any rewrite that changes row order, silently changes which book's price a
# published bet was struck at, with nothing on the record to attribute the change to. The selected
# row supplies ``spread``, ``total``, ``ml_home``, ``ml_away`` and all four juice columns, i.e. the
# price the per-bet EV, the Kelly stake and the published ``selected_odds`` are all struck at.
SPORTSBOOK_PREFERENCE: tuple[str, ...] = ("consensus", "draftkings")

# The two lists are the same set, asserted at import so a book added to the allowlist cannot
# silently fall off the END of the preference and be picked only by file order again.
assert set(SPORTSBOOK_PREFERENCE) == set(_ALLOWED_SPORTSBOOKS), (
    "SPORTSBOOK_PREFERENCE and _ALLOWED_SPORTSBOOKS disagree: "
    f"{sorted(SPORTSBOOK_PREFERENCE)} vs {sorted(_ALLOWED_SPORTSBOOKS)}"
)


def dedupe_odds_by_book_preference(odds: pd.DataFrame) -> pd.DataFrame:
    """One row per ``game_id``, choosing the book BY NAME rather than by file order (WR-08).

    Rows are ranked by :data:`SPORTSBOOK_PREFERENCE` and the best-ranked row per game wins. A book
    not in the preference ranks last (rather than being dropped), so an unrecognised label still
    prices a game that has no preferred row -- the provenance allowlist is what refuses an
    unrecognised book, and it does so by NAME, in its own place. A frame with no ``sportsbook``
    column falls back to the previous first-row behaviour, because there is no preference to apply.

    The sort is ``kind="mergesort"`` (stable), so ties within one book keep their stored order and
    the result is reproducible.

    Args:
        odds: The stored odds rows, possibly several per game.

    Returns:
        One row per ``game_id``, with the frame's original columns and a reset index.
    """
    if odds.empty or "sportsbook" not in odds.columns:
        return odds.drop_duplicates(subset=["game_id"], keep="first").reset_index(
            drop=True
        )

    ranks = {book: index for index, book in enumerate(SPORTSBOOK_PREFERENCE)}
    unknown_rank = len(SPORTSBOOK_PREFERENCE)
    ranked = odds.copy()
    ranked["_book_rank"] = (
        ranked["sportsbook"].astype(str).map(ranks).fillna(unknown_rank).astype(int)
    )
    return (
        ranked.sort_values(["game_id", "_book_rank"], kind="mergesort")
        .drop_duplicates(subset=["game_id"], keep="first")
        .drop(columns="_book_rank")
        .reset_index(drop=True)
    )


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
#
# PHASE-26 DIAGNOSIS ONLY (HOLD-INFORMED): these tertiles were resolved from the EMPIRICAL
# 2021-2024 closing-totals window, which INCLUDES the 2023-2024 hold split. That hold-informed
# boundary is fine for the Phase-26 backward-looking diagnosis (the extended_bucket_sweep
# consumes it below) but it LEAKS future info into eligibility if used for selection. Phase-27
# eligibility therefore uses the leakage-clean HIGH_TOTAL_BOUNDARY_PREHOLD constant defined
# below (re-derived on pre-hold 2018-2022 data only, LOCKED-1). DO NOT use this constant for
# Phase-27 high-total eligibility.
TOTALS_REGIME_BOUNDARIES = {
    "low_max": 42.0,
    "mid_min": 42.0,
    "mid_max": 46.5,
    "high_min": 46.5,
}

# ---------------------------------------------------------------------------
# Phase-27 leakage-clean high-total eligibility boundary (LOCKED-1).
#
# The Phase-26 TOTALS_REGIME_BOUNDARIES "high" cut (> 46.5) was derived from a window that
# INCLUDES the 2023-2024 hold split, so reusing it for Phase-27 high-total eligibility would leak
# future information into the sub-population filter (threat T-27-22). LOCKED-1 requires the
# high-total boundary to be RE-DERIVED on PRE-HOLD data ONLY -- the same distributional cut (the
# upper tertile that produced 46.5) recomputed over the 2018-2022 closing-totals window,
# EXCLUDING the 2023-2024 hold. derive_high_total_boundary() performs that derivation with a hard
# leakage assertion (it raises if any hold-season row reaches the derivation input). The resulting
# value is exposed below as HIGH_TOTAL_BOUNDARY_PREHOLD and is what the Phase-27 BetSelector
# consumes for high-total eligibility.
# ---------------------------------------------------------------------------

# Hold seasons EXCLUDED from the pre-hold boundary derivation (mirrors ou_ev_chain.HOLD_SEASONS;
# 2023-2024 are the burned holdout per D26-09 / D27-01). Stated locally so this module does not
# import the EV chain (keeping the diagnosis harness dependency-free of the monetization chain).
HOLD_SEASONS: tuple[int, int] = (2023, 2024)

# The pre-hold derivation window (2018-2022 closing totals). 2018-2022 is the leakage-clean window
# for eligibility: it EXCLUDES the 2023-2024 hold and the silver odds carry only
# consensus/draftkings is_live=False lines across it (RESEARCH.md Finding 1). The upper-tertile
# quantile (2/3) is the SAME distributional cut that produced the legacy 46.5; recomputed on the
# pre-hold window it lands at HIGH_TOTAL_BOUNDARY_PREHOLD (compared to legacy 46.5 below).
PRE_HOLD_SEASONS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022)

# The upper-tertile quantile that defines the "high" regime (the same 2/3 quantile that produced
# the legacy 46.5). Frozen before the derivation runs (forking-paths guard).
_HIGH_TOTAL_QUANTILE: float = 2.0 / 3.0

# The legacy hold-informed high boundary, retained as a literal for the readout comparison only.
_LEGACY_HIGH_TOTAL_BOUNDARY: float = 46.5

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

# Pre-registered SD grid the EV preview reports (the band endpoints + interior, D26-16). The
# in-harness fit SD (~12.95) is reported alongside these.
SD_PREVIEW_GRID = [12.5, 13.0, 13.5, 14.0, 14.5]

# Breakeven cover probability at -110 (the bettable bar -- 110 / (110 + 100)). Below this a graded
# O/U bet loses money after vig; the structural bar's GRADED-EDGE direction is measured against it.
OU_BREAKEVEN_HIT_RATE = 110.0 / 210.0  # 0.52380952...

# -110 win payout per unit staked (100/110): EV = p_side * payout - (1 - p_side).
_MINUS_110_PAYOUT = 100.0 / 110.0

# Edge-magnitude monotonicity grid (D26-03 decisive check): vary min_edge_threshold (model_total
# vs closing_total points units for O/U) and report graded hit-rate + bet count at each point.
EDGE_MAGNITUDE_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

# Half-point slippage applied against the bet side in the throwaway EV preview (the production
# SLIPPAGE_POINTS=0.5 convention, applied here in the EV cover-probability math only).
_EV_SLIPPAGE_POINTS = 0.5

# Gold weather columns for the weather/outdoor cut (Codex MED, verification-CONFIRMED): the gold
# columns are `venue_outdoor` and `weather_severity_score` (a hypothesized outdoor-flag column
# the original plan named does NOT exist in this gold -- the source-grep guard enforces that).
# A column is USABLE only if present AND discriminating (>1 distinct non-null value); a degenerate
# zero-variance column (the weather features were never populated in this gold) is treated as
# `unavailable` with coverage metadata -- never silently skipped, never hand-substituted.
_WEATHER_OUTDOOR_COL = "venue_outdoor"
_WEATHER_SEVERITY_COL = "weather_severity_score"

# THE VENUE-LEVEL KEY ABOVE IS DELIBERATELY NO LONGER USED BY `_weather_cut`. It is kept
# declared so a later reader meets this warning instead of rediscovering the defect and
# repointing the split back for looking tidier.
#
# `venue_outdoor` is derived from the VENUE record's `roof_type`
# (`features/contextual._encode_venue_features`), which encodes `retractable` as its own
# separate indicator rather than folding it into outdoor. Every game at a retractable
# stadium therefore carries ONE value regardless of whether that day's roof was open or
# shut. MEASURED on this population (6,499 games, 2002-2025): 749 games at the five
# retractable stadiums DAL00 / HOU00 / IND00 / PHO00 / ATL97 share a single venue-level
# value, while 621 of them were played CLOSED and 128 OPEN. Splitting on it grades 621
# games whose weather never reached the field as though it had.
#
# So applicability is keyed on the GAME's own roof fact instead. That fact lives in the
# silver weather table as `weather_affects_game`: a clean 0/1 over all 6,499 rows
# (4,847 outdoor, 1,652 indoor, no nulls, no duplicate game_id), which splits those same
# 749 retractable games exactly 621/128. Gold cannot supply it -- expanding normalization
# z-scores the flag into 5,315 distinct floats, and the cut's own `is_clean_binary` guard
# refuses that column, correctly. Reading it from silver is the same read-only access
# pattern `_RAW_SILVER_ODDS_PATH` already uses in this module for the provenance columns
# the normalized loader strips.
_WEATHER_PER_GAME_OUTDOOR_COL = "weather_affects_game"
_WEATHER_COVERAGE_COL = "weather_coverage"

# Raw silver weather path -- read ONLY for the two per-game applicability columns above.
# Never written.
_RAW_SILVER_WEATHER_PATH = Path("data/silver/weather_features.parquet")

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
# Phase-27 leakage-clean high-total boundary derivation (LOCKED-1)
# ---------------------------------------------------------------------------


class HoldSeasonLeakageError(ValueError):
    """Raised when the high-total boundary derivation input contains a HOLD-season row.

    The pre-hold high-total boundary (LOCKED-1) MUST be derived from PRE-HOLD data only
    (2018-2022, EXCLUDING the 2023-2024 hold). If a HOLD season reaches the derivation input,
    the eligibility boundary would leak future information (threat T-27-22); this hard error
    forbids it rather than silently degrading to a hold-informed boundary.
    """


def _season_from_game_id(game_id: str) -> int:
    """Extract the integer season from a game_id of the form ``YYYY_WXX_AWAY@HOME``.

    The silver odds frame carries no explicit ``season`` column, so the season is parsed from the
    game_id prefix (the same convention the engine uses for the week). Raises ValueError on a
    malformed id rather than guessing.
    """
    head = str(game_id).split("_", 1)[0]
    if not (len(head) == 4 and head.isdigit()):
        msg = f"cannot parse season from game_id {game_id!r} (expected 'YYYY_WXX_...')"
        raise ValueError(msg)
    return int(head)


def derive_high_total_boundary(
    pre_hold_seasons: tuple[int, ...] = PRE_HOLD_SEASONS,
    quantile: float = _HIGH_TOTAL_QUANTILE,
    odds_df: pd.DataFrame | None = None,
) -> float:
    """Re-derive the high-total regime boundary on PRE-HOLD data only (LOCKED-1).

    Loads the raw closing ``total`` from SILVER (NEVER the gold z-scored ``snapshot_total``),
    filters to the pre-hold seasons (2018-2022, EXCLUDING the 2023-2024 hold), and returns the
    upper-tertile ``quantile`` (default 2/3 -- the SAME distributional cut that produced the legacy
    46.5). The derivation is leakage-clean: a hard assertion raises
    :class:`HoldSeasonLeakageError` if any HOLD-season row reaches the input, and the seasons
    actually read are asserted to be a subset of ``pre_hold_seasons``.

    Two calling modes, both leakage-clean:

    - ``odds_df is None`` (the default / production path): the FULL raw silver lake (all seasons)
      is read and FILTERED to ``pre_hold_seasons``. The lake naturally contains the hold seasons;
      filtering them out IS the leakage control. After filtering, the assertion below proves no
      hold row survived into the derivation set.
    - ``odds_df`` provided (the test / caller-supplied path): the frame is treated as the
      derivation INPUT directly and must ALREADY be pre-hold only. If it carries any hold-season
      row, :class:`HoldSeasonLeakageError` is raised -- the caller asserted a clean input and it
      was not (the ``test_high_total_boundary_excludes_hold`` contract).

    Args:
        pre_hold_seasons: The leakage-clean derivation window (default ``PRE_HOLD_SEASONS`` =
            2018-2022). MUST NOT intersect ``HOLD_SEASONS``.
        quantile: The upper-tertile quantile defining the "high" regime (default 2/3).
        odds_df: Optional derivation-input odds frame. When None, the raw silver
            ``odds_snapshot.parquet`` is read read-only and filtered to the pre-hold window. When
            provided, it is treated as the derivation input and must be pre-hold only.

    Returns:
        The pre-hold high-total boundary as a float (the closing-total quantile over 2018-2022).

    Raises:
        HoldSeasonLeakageError: if ``pre_hold_seasons`` intersects ``HOLD_SEASONS``, or if a
            caller-supplied ``odds_df`` contains a HOLD-season (2023/2024) row, or if the filtered
            derivation set somehow still carries a hold season.
        ValueError: if no pre-hold rows remain after filtering (an empty derivation is a hard
            error, never a silent fallback to the legacy boundary).
    """
    # Forking-paths / leakage guard on the REQUESTED window itself.
    hold = set(HOLD_SEASONS)
    requested = set(pre_hold_seasons)
    if requested & hold:
        msg = (
            "pre_hold_seasons must not intersect HOLD_SEASONS "
            f"(requested={sorted(requested)}, hold={sorted(hold)}); the eligibility boundary "
            "must be leakage-clean (LOCKED-1)."
        )
        raise HoldSeasonLeakageError(msg)

    if odds_df is not None:
        # Caller-supplied derivation input: it MUST already be pre-hold only. A hold-season row in
        # an explicitly-passed input is a leakage error (the test contract).
        seasons_in = odds_df["game_id"].map(_season_from_game_id)
        leaked = sorted(set(seasons_in.unique()) & hold)
        if leaked:
            n_leaked = int(seasons_in.isin(hold).sum())
            msg = (
                f"caller-supplied derivation input contains HOLD seasons {leaked} "
                f"({n_leaked} rows); the derivation input must be PRE-HOLD only "
                f"({sorted(requested)}) so eligibility is leakage-clean (LOCKED-1, T-27-22)."
            )
            raise HoldSeasonLeakageError(msg)
        pre_hold = odds_df[seasons_in.isin(requested)]
    else:
        # Production path: read the full lake and FILTER to the pre-hold window (filtering out the
        # hold seasons IS the leakage control).
        raw = pd.read_parquet(_RAW_SILVER_ODDS_PATH)
        seasons_read = raw["game_id"].map(_season_from_game_id)
        pre_hold = raw[seasons_read.isin(requested)]

    # POST-FILTER LEAKAGE ASSERTION: the derivation set actually used must be a SUBSET of the
    # pre-hold window and must NOT intersect the hold seasons (proves no hold row leaked through).
    used = set(pre_hold["game_id"].map(_season_from_game_id).unique())
    if used & hold:
        msg = (
            f"derivation set still contains HOLD seasons {sorted(used & hold)} after filtering "
            f"(window={sorted(requested)}); leakage control failed (LOCKED-1, T-27-22)."
        )
        raise HoldSeasonLeakageError(msg)
    if not used.issubset(requested):
        msg = (
            f"derivation read seasons {sorted(used)} outside the pre-hold window "
            f"{sorted(requested)} (LOCKED-1)."
        )
        raise HoldSeasonLeakageError(msg)

    totals = pd.to_numeric(pre_hold["total"], errors="coerce").dropna()
    if totals.empty:
        msg = (
            "no pre-hold closing totals available to derive the high-total boundary "
            f"(window={sorted(requested)}); never fall back to the legacy 46.5."
        )
        raise ValueError(msg)
    return float(totals.quantile(quantile))


# The leakage-clean Phase-27 high-total eligibility boundary, derived ONCE at import on the
# pre-hold (2018-2022) closing totals -- the SAME upper-tertile (2/3) cut that produced the legacy
# 46.5, recomputed on the leakage-clean window. Phase-27 high-total eligibility uses THIS value
# (NOT the hold-informed TOTALS_REGIME_BOUNDARIES["high_min"] = 46.5). Empirically this lands at
# ~48.0 on the 2018-2022 window (vs the legacy 46.5); the BetSelector + readout report both.
try:
    HIGH_TOTAL_BOUNDARY_PREHOLD: float = derive_high_total_boundary()
except (FileNotFoundError, OSError):
    # The silver odds parquet is unavailable in a bare checkout (artifacts/ + data/ are
    # gitignored). Defer the derivation to call-time so importing the module never hard-fails on a
    # missing data lake; callers/tests on a populated lake recompute via derive_high_total_boundary.
    HIGH_TOTAL_BOUNDARY_PREHOLD = float("nan")


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


# ---------------------------------------------------------------------------
# (3) Extended bucket sweep + trial registry + BH-FDR correction (D26-06 / D26-10)
# ---------------------------------------------------------------------------
#
# SCOPE NOTE (owner ruling, 26-02-SUMMARY.md): INTERIM_DECISION = PROCEED,
# SWEEP_DISPOSITION = run. This is the FULL-scope branch -- every pre-registered D26-06/08 cut
# is graded for BOTH the raw and blended streams; no narrowing, no skip, no appendix-only mode.
#
# The sweep NEVER re-grades an outcome and NEVER re-derives a metric: per-bucket graded hit-rate
# comes from `both_population_hit_rates` (BettingSimulator at min_edge_threshold=0.0, the D26-03
# base), per-bucket line_clv significance comes from `clv_significance`, and the slippage-survival
# cut uses the EXISTING sanctioned knobs `SimulationConfig(slippage_points=0.0)` vs the default
# 0.5 -- no hand-rolled half-point arithmetic anywhere in this module.


def _sweep_per_game(
    preds: pd.DataFrame | None,
    odds: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    """Build the with-line per-game frame the sweep slices, plus the raw predictions + odds.

    Returns ``(per_game, preds, odds)`` where ``per_game`` carries the backtest contract columns
    (``game_id``, ``season``, ``week``, ``model_total``, ``model_prob``, ``actual``, ``total``,
    ``line_clv``) PLUS the gold weather columns (``venue_outdoor``, ``weather_severity_score``)
    left-merged in for the weather/outdoor cut, and a ``model_over`` bet-direction flag and a
    ``key_total_distance`` (distance from the nearest integer total). The returned ``preds`` and
    ``odds`` are the full (un-sliced) frames the per-bucket simulator re-grades against, so a slice
    is graded by passing ``preds`` filtered to the slice's game_ids (NOT a re-grade by hand).
    """
    if odds is None:
        from backtest.engine import BacktestEngine

        odds = BacktestEngine()._load_closing_odds()
    preds = _deployed_ou_preds(preds)

    clv = compute_clv_for_predictions(preds, odds, "ou")
    valid = clv[clv["has_closing_odds"]].copy()
    valid["model_over"] = valid["model_total"] > valid["total"]
    # Distance from the nearest INTEGER total (NFL totals key-number structure, D26-08).
    valid["key_total_distance"] = (valid["total"] - valid["total"].round()).abs()

    # Left-merge the gold weather columns (read-only) for the weather/outdoor cut. Missing/
    # degenerate columns are handled later as `unavailable`, never silently dropped.
    gold = _load_ou_gold_weather()
    if gold is not None:
        valid = valid.merge(gold, on="game_id", how="left")

    # The per-game roof fact comes from SILVER, read-only, because gold z-scores it into a
    # float column the cut's own binary guard refuses (see the note on _WEATHER_OUTDOOR_COL).
    roof = _load_ou_silver_roof()
    if roof is not None:
        valid = valid.merge(roof, on="game_id", how="left")

    keep = [
        "game_id",
        "season",
        "week",
        "model_total",
        "model_prob",
        "actual",
        "total",
        "line_clv",
        "model_over",
        "key_total_distance",
    ]
    for wcol in (
        _WEATHER_OUTDOOR_COL,
        _WEATHER_SEVERITY_COL,
        _WEATHER_PER_GAME_OUTDOOR_COL,
        _WEATHER_COVERAGE_COL,
    ):
        if wcol in valid.columns:
            keep.append(wcol)
    per_game = valid[keep].copy()
    return per_game, preds, odds


def _load_ou_gold_weather() -> pd.DataFrame | None:
    """Load the O/U gold weather columns keyed by game_id (READ ONLY), or None if unavailable.

    Returns a frame with ``game_id`` plus whichever of ``venue_outdoor`` / ``weather_severity_score``
    are present in the gold parquet. Returns None if the gold parquet is absent. Degenerate
    (zero-variance) columns are NOT filtered here -- the weather cut decides usability so the
    `unavailable` disclosure carries the reason.
    """
    gold_path = Path("data/gold/features_ou.parquet")
    if not gold_path.exists():
        return None
    present = [
        c
        for c in (_WEATHER_OUTDOOR_COL, _WEATHER_SEVERITY_COL)
        if c in pd.read_parquet(gold_path, columns=None).columns
    ]
    if not present:
        return pd.DataFrame(columns=["game_id"])
    return pd.read_parquet(gold_path, columns=["game_id", *present])


def _load_ou_silver_roof() -> pd.DataFrame | None:
    """Load the per-game roof fact and its coverage flag from SILVER (READ ONLY).

    Returns:
        A frame with ``game_id`` plus whichever of ``weather_affects_game`` /
        ``weather_coverage`` the silver weather table carries, or None when that table
        is absent (a bare checkout without ``data/``). A missing table yields None
        rather than an empty frame so the cut reports ``unavailable`` with its reason
        instead of grading an empty applicability split.

    WHY SILVER AND NOT GOLD. The fact this returns is the game's OWN roof, and gold
    does not carry it in a usable form: expanding normalization z-scores the flag into
    5,315 distinct floats, which the cut's ``is_clean_binary`` guard refuses, correctly.
    Silver carries it clean -- 0/1 over all 6,499 rows, no nulls, unique on ``game_id``.
    This is the same read-only silver access ``_RAW_SILVER_ODDS_PATH`` already makes in
    this module; nothing here writes.
    """
    if not _RAW_SILVER_WEATHER_PATH.exists():
        return None
    available = pd.read_parquet(_RAW_SILVER_WEATHER_PATH).columns
    present = [
        c
        for c in (_WEATHER_PER_GAME_OUTDOOR_COL, _WEATHER_COVERAGE_COL)
        if c in available
    ]
    if not present:
        return None
    return pd.read_parquet(
        _RAW_SILVER_WEATHER_PATH, columns=["game_id", *present]
    ).drop_duplicates(subset=["game_id"])


def _bucket_masks_excluding_missing(
    series: pd.Series,
    *,
    split: str = "median",
) -> tuple[pd.Series, pd.Series, int]:
    """Split *series* into two buckets over OBSERVED rows only, and count the rest.

    Args:
        series: The column a split is taken on. Missing entries are rows where no
            observation exists.
        split: ``"median"`` for a band split at the median of the observed values,
            or ``"indicator"`` for a 0/1 flag.

    Returns:
        ``(upper_mask, lower_mask, n_excluded)``. Under ``"median"`` the masks are
        strictly-above-median and at-or-below-median; under ``"indicator"`` they are
        the ones and the zeros. ``n_excluded`` is the count of missing rows, which
        belong to NEITHER mask.

    Raises:
        ValueError: *split* is not a known mode, or an ``"indicator"`` series carries
            an observed value that is neither 0 nor 1. A row that silently belongs to
            no side would break the accounting the caller reports, so it is refused
            by name instead.

    THE PROPERTY THIS FUNCTION EXISTS FOR. The idiom it replaces was
    ``upper = series > median`` with ``lower = ~upper``. In pandas every comparison
    against a missing value is False, so ``~upper`` is True for those rows and every
    game whose weather nobody observed was filed on the LOWER side -- reported as a
    measurement of calm days while partly being a list of days nobody measured. The
    same shape one level up files an unobserved game as ``indoor``. Here BOTH masks
    are built positively from ``notna()``, so a missing row cannot reach either, and
    the caller is handed the count it must disclose. For every split the invariant
    holds: ``upper.sum() + lower.sum() + n_excluded == len(series)``.
    """
    observed = series.notna()
    n_observed = int(observed.sum())
    n_excluded = len(series) - n_observed

    if split == "median":
        if n_observed == 0:
            empty = pd.Series(False, index=series.index)
            return empty, empty.copy(), n_excluded
        threshold = float(series[observed].median())
        upper = observed & (series > threshold)
        lower = observed & (series <= threshold)
        return upper, lower, n_excluded

    if split == "indicator":
        stray = sorted(set(series[observed].unique()) - {0.0, 1.0})
        if stray:
            msg = (
                f"an indicator split needs a clean 0/1 column; observed values "
                f"{stray[:6]} belong to neither side and would fall out of the "
                "bucket accounting"
            )
            raise ValueError(msg)
        upper = observed & (series == 1.0)
        lower = observed & (series == 0.0)
        return upper, lower, n_excluded

    msg = f"unknown split mode {split!r}; expected 'median' or 'indicator'"
    raise ValueError(msg)


def _graded_hit_rate_for_slice(
    slice_game_ids: pd.Series,
    preds: pd.DataFrame,
    odds: pd.DataFrame,
    slippage_points: float = SLIPPAGE_POINTS_DEFAULT,
) -> dict[str, Any]:
    """Graded hit-rate for a bucket slice via the LOCKED BettingSimulator (no hand-roll).

    Slices ``preds`` to ``slice_game_ids`` BEFORE wrapping in the diagnose ``_results_like`` shim,
    then reads the ``min_edge_threshold=0.0`` straight-pick win-rate (the D26-03 base). The
    ``slippage_points`` argument feeds ``SimulationConfig`` so the slippage-survival cut can pass
    0.0 (no-slippage) vs the default 0.5 WITHOUT re-implementing the half-point convention.

    Returns ``{n_graded, hit_rate}`` where ``hit_rate`` is None when the slice grades zero bets.
    """
    sliced = preds[preds["game_id"].isin(slice_game_ids)].copy()
    if sliced.empty:
        return {"n_graded": 0, "hit_rate": None}

    sim = BettingSimulator(
        SimulationConfig(min_edge_threshold=0.0, slippage_points=slippage_points)
    )
    res = sim.simulate(_diag_results_like({"ou": sliced}), odds)
    stats_for_ou = res.by_target.get("ou", {})
    n_graded = int(stats_for_ou.get("n_bets", 0))
    hit_rate = float(stats_for_ou["win_rate"]) if n_graded else None
    return {"n_graded": n_graded, "hit_rate": hit_rate}


def _graded_edge_direction_by_season(
    slice_per_game: pd.DataFrame,
    preds: pd.DataFrame,
    odds: pd.DataFrame,
) -> list[int]:
    """Per-season GRADED-EDGE direction vector over 2021-2024 (the D26-11 structural-bar metric).

    The per-season direction is the GRADED-EDGE direction -- the season's graded hit-rate above or
    below the 0.5238 breakeven at -110 (equivalently the graded ROI sign), NOT the line_clv sign
    (LOCKED in the Plan 26-03 go bar per the Codex MED fix: ROI/EV direction leads because the
    phase goal is ROI divergence; a CLV-only direction can name a false edge).

    Returns a length-4 list aligned to seasons [2021, 2022, 2023, 2024]: +1 if that season's graded
    hit-rate is above breakeven, -1 if below, 0 if the season graded no bets (or is absent).
    """
    direction: list[int] = []
    for season in (2021, 2022, 2023, 2024):
        season_ids = slice_per_game[slice_per_game["season"] == season]["game_id"]
        if season_ids.empty:
            direction.append(0)
            continue
        graded = _graded_hit_rate_for_slice(season_ids, preds, odds)
        hr = graded["hit_rate"]
        if hr is None:
            direction.append(0)
        elif hr > OU_BREAKEVEN_HIT_RATE:
            direction.append(1)
        elif hr < OU_BREAKEVEN_HIT_RATE:
            direction.append(-1)
        else:
            direction.append(0)
    return direction


def _bucket_masks(per_game: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """Build the pre-registered cut -> {bucket_label -> boolean mask} map (D26-06/08).

    Cuts: key-total distance bands, over vs under, season, playoffs-out, week groupings, and
    totals-regime. The slippage-survival and weather/outdoor cuts are handled separately
    (slippage uses the no-slippage knob; weather needs an availability check), so they are not in
    this mask map. All bands come from the LOCKED module constants -- no adaptive adjustment.
    """
    masks: dict[str, dict[str, pd.Series]] = {}

    # key-total distance bands (0.0, 0.5, 1.0, >=1.5 from the nearest integer total).
    # Derive the bucket edges from the LOCKED KEY_TOTAL_DISTANCE_BANDS constant so the
    # pre-registration is enforced by code, not decorative: each edge is the midpoint between
    # adjacent band centers, i.e. a 0.25-wide capture window around each pre-registered distance.
    # Editing KEY_TOTAL_DISTANCE_BANDS now provably shifts these masks (forking-paths guard, WR-02).
    b0, b1, b2, b3 = KEY_TOTAL_DISTANCE_BANDS
    edge_01 = (b0 + b1) / 2  # 0.25
    edge_12 = (b1 + b2) / 2  # 0.75
    edge_23 = (b2 + b3) / 2  # 1.25
    dist = per_game["key_total_distance"]
    masks["key_total_distance"] = {
        "0.0": (dist < edge_01),
        "0.5": (dist >= edge_01) & (dist < edge_12),
        "1.0": (dist >= edge_12) & (dist < edge_23),
        ">=1.5": (dist >= edge_23),
    }

    # over vs under (split by model bet direction).
    masks["over_under"] = {
        "over": per_game["model_over"],
        "under": ~per_game["model_over"],
    }

    # season.
    masks["season"] = {
        str(int(s)): (per_game["season"] == s)
        for s in sorted(per_game["season"].unique())
    }

    # playoffs-out / regular-season-only (weeks <= 18 vs all). NOTE: in the deployed-artifact
    # with-line population every game is week <= 18 (the 52 playoff games lack a closing line and
    # are the excluded set, per the interim readout); the "all" bucket therefore equals
    # regular-season here -- the coverage count makes that explicit rather than implying a
    # playoff slice that the with-line population does not contain.
    weeks = per_game["week"]
    masks["playoffs_out"] = {
        "regular_season": (weeks <= 18),
        "all": pd.Series(True, index=per_game.index),
    }

    # week groupings.
    masks["week_grouping"] = {
        label: ((weeks >= lo) & (weeks <= hi))
        for label, (lo, hi) in WEEK_GROUPINGS.items()
    }

    # totals-regime (low/mid/high via the empirical boundaries).
    total = per_game["total"]
    masks["totals_regime"] = {
        "low": (total < TOTALS_REGIME_BOUNDARIES["low_max"]),
        "mid": (total >= TOTALS_REGIME_BOUNDARIES["mid_min"])
        & (total <= TOTALS_REGIME_BOUNDARIES["mid_max"]),
        "high": (total > TOTALS_REGIME_BOUNDARIES["high_min"]),
    }

    return masks


def _grade_bucket(
    bucket_per_game: pd.DataFrame,
    n_total: int,
    preds: pd.DataFrame,
    odds: pd.DataFrame,
) -> dict[str, Any]:
    """Grade a single bucket slice: coverage, graded hit-rate, line_clv significance, direction.

    ``n_total`` is the population size the bucket is drawn FROM (for the n_excluded companion, so
    no metric is an orphan). Below MIN_CLV_SAMPLE the line_clv t/p come back None (insufficient
    sample) via ``clv_significance`` -- the bucket is still returned (never silently dropped).
    """
    n = len(bucket_per_game)
    n_excluded = n_total - n
    if n == 0:
        return {
            "n": 0,
            "n_excluded": n_excluded,
            "hit_rate": None,
            "n_graded": 0,
            "line_clv_mean": None,
            "raw_p": None,
            "insufficient_sample": True,
            "graded_edge_direction_by_season": [0, 0, 0, 0],
        }

    graded = _graded_hit_rate_for_slice(bucket_per_game["game_id"], preds, odds)
    sig = clv_significance(bucket_per_game["line_clv"].to_numpy())
    direction = _graded_edge_direction_by_season(bucket_per_game, preds, odds)

    return {
        "n": n,
        "n_excluded": n_excluded,
        "hit_rate": graded["hit_rate"],
        "n_graded": graded["n_graded"],
        "line_clv_mean": sig["mean"],
        "raw_p": sig["p"],
        "insufficient_sample": sig["p"] is None,
        "graded_edge_direction_by_season": direction,
    }


def _weather_cut(
    per_game: pd.DataFrame,
    n_total: int,
    preds: pd.DataFrame,
    odds: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Weather/outdoor cut, keyed on the GAME's roof and with unobserved games set aside.

    Two splits, and the same two the Phase-26 trial registry recorded: applicability
    (the weather could reach this field / it could not) and severity (rough conditions /
    calm ones). SAME harness, SAME grading through the LOCKED ``BettingSimulator``, SAME
    bucket names, no p-value, no new trial. Two things about the cut changed, and both are
    repairs to WHICH ROWS a bucket contains rather than new comparisons:

    1. APPLICABILITY IS PER GAME, NOT PER VENUE. It splits on the silver
       ``weather_affects_game`` flag -- the game's own roof -- instead of the venue-level
       ``venue_outdoor``, which cannot tell an open-roof game from a closed-roof one at the
       same stadium and so graded 621 closed-roof games as outdoor. The venue-level
       constant is still declared at the top of this module, carrying the measured note
       that explains why it is the wrong key here; the cut does not consult it.
    2. AN UNOBSERVED GAME IS EXCLUDED FROM BOTH DENOMINATORS, and its count is reported as
       ``excluded_missing`` on both sides of the split. The previous idiom built one side as
       the bare complement of the other, and because every comparison against a missing
       value is False, a game whose weather nobody recorded was filed as ``mild`` (and, one
       level up, as ``indoor``). Both sides are now built positively by
       ``_bucket_masks_excluding_missing``.

    NO CUT IS ADDED HERE, deliberately. Wind and precipitation already feed the severity
    score this band splits on; promoting them to bands of their own would add two
    comparisons to a correction family fixed in Phase 26 -- the forking-paths defect
    ``DEBIAS_RESIDUAL_THRESHOLD`` above is annotated against. A column that is absent OR
    degenerate still yields an ``unavailable`` bucket carrying a ``coverage_note``, never a
    silent skip and never a hand-picked substitute.
    """
    out: dict[str, dict[str, Any]] = {}

    # -- applicability: the game's own roof, usable only as a clean {0,1} with both classes --
    if _WEATHER_PER_GAME_OUTDOOR_COL not in per_game.columns:
        out["outdoor"] = {
            "unavailable": True,
            "coverage_note": (
                f"per-game column '{_WEATHER_PER_GAME_OUTDOOR_COL}' absent from the loaded "
                "frame -- weather/outdoor cut not gradeable"
            ),
        }
    else:
        applies = per_game[_WEATHER_PER_GAME_OUTDOOR_COL]
        if _WEATHER_COVERAGE_COL in per_game.columns:
            # A roof fact recorded for a game with no weather observation is not an
            # observation. Masking it to missing here routes it to excluded_missing.
            applies = applies.where(per_game[_WEATHER_COVERAGE_COL] == 1.0)
        col = applies.dropna()
        distinct = set(np.unique(np.round(col.to_numpy(), 6))) if len(col) else set()
        is_clean_binary = distinct.issubset({0.0, 1.0}) and len(distinct) == 2
        if not is_clean_binary:
            out["outdoor"] = {
                "unavailable": True,
                "coverage_note": (
                    f"per-game column '{_WEATHER_PER_GAME_OUTDOOR_COL}' is not a clean 0/1 "
                    f"applicability indicator (distinct rounded values={sorted(distinct)[:6]}"
                    "...), so the outdoor cut is not gradeable"
                ),
            }
        else:
            outdoor_mask, indoor_mask, n_missing = _bucket_masks_excluding_missing(
                applies, split="indicator"
            )
            out["outdoor"] = _grade_bucket(per_game[outdoor_mask], n_total, preds, odds)
            out["indoor"] = _grade_bucket(per_game[indoor_mask], n_total, preds, odds)
            out["outdoor"]["excluded_missing"] = n_missing
            out["indoor"]["excluded_missing"] = n_missing

    # -- weather_severity_score: usable only if it varies (a severity band needs >1 value) --
    if _WEATHER_SEVERITY_COL not in per_game.columns:
        out["severe_weather"] = {
            "unavailable": True,
            "coverage_note": (
                f"gold column '{_WEATHER_SEVERITY_COL}' absent -- severity cut not gradeable"
            ),
        }
    else:
        sev = per_game[_WEATHER_SEVERITY_COL].dropna()
        if sev.nunique() <= 1:
            out["severe_weather"] = {
                "unavailable": True,
                "coverage_note": (
                    f"gold column '{_WEATHER_SEVERITY_COL}' has <=1 distinct value "
                    f"(nunique={int(sev.nunique())}) -- the weather-severity feature was not "
                    "populated in this gold, so the severity cut is not gradeable"
                ),
            }
        else:
            severe_mask, mild_mask, n_missing = _bucket_masks_excluding_missing(
                per_game[_WEATHER_SEVERITY_COL], split="median"
            )
            out["severe_weather"] = _grade_bucket(
                per_game[severe_mask], n_total, preds, odds
            )
            out["mild_weather"] = _grade_bucket(
                per_game[mild_mask], n_total, preds, odds
            )
            out["severe_weather"]["excluded_missing"] = n_missing
            out["mild_weather"]["excluded_missing"] = n_missing

    return out


def _slippage_survival_cut(
    per_game: pd.DataFrame,
    preds: pd.DataFrame,
    odds: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Half-point slippage-survival cut via the sanctioned no-slippage knob (Codex HIGH).

    Grades the full population WITH the default ``SimulationConfig(slippage_points=0.5)`` versus a
    no-slippage grading using the EXISTING knob ``SimulationConfig(slippage_points=0.0)`` -- the
    half-point convention is consumed from the simulator, NEVER re-implemented here. The two graded
    hit-rates side by side ARE the slippage-survival reading (how much edge the half-point erodes).
    """
    n_total = len(per_game)
    ids = per_game["game_id"]
    with_slip = _graded_hit_rate_for_slice(ids, preds, odds, slippage_points=0.5)
    no_slip = _graded_hit_rate_for_slice(ids, preds, odds, slippage_points=0.0)
    sig = clv_significance(per_game["line_clv"].to_numpy())
    direction = _graded_edge_direction_by_season(per_game, preds, odds)
    return {
        "with_slippage_0.5": {
            "n": n_total,
            "n_excluded": 0,
            "hit_rate": with_slip["hit_rate"],
            "n_graded": with_slip["n_graded"],
            "line_clv_mean": sig["mean"],
            "raw_p": sig["p"],
            "insufficient_sample": sig["p"] is None,
            "graded_edge_direction_by_season": direction,
        },
        "no_slippage_0.0": {
            "n": n_total,
            "n_excluded": 0,
            "hit_rate": no_slip["hit_rate"],
            "n_graded": no_slip["n_graded"],
            "line_clv_mean": sig["mean"],
            "raw_p": sig["p"],
            "insufficient_sample": sig["p"] is None,
            "graded_edge_direction_by_season": direction,
        },
    }


def extended_bucket_sweep(
    preds: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Full pre-registered bucket sweep + trial registry + BH-FDR correction (D26-06 / D26-10).

    Owner ruling SWEEP_DISPOSITION = run (26-02): the FULL extended cut set is graded for BOTH the
    raw and the blended stream. Each (cut x bucket x stream) is graded via the LOCKED
    ``both_population_hit_rates`` / ``BettingSimulator`` (min_edge_threshold=0.0) and scored for
    per-bucket line_clv significance via ``clv_significance``; every evaluation appends ONE entry to
    the trial registry (the multiple-comparisons denominator). After the sweep, BH-FDR
    (``scipy.stats.false_discovery_control``, method "bh") is computed across the registry's
    non-None raw p-values and the adjusted p is attached back to each tested entry, aligned by
    index over the non-None subset.

    Args:
        preds: Optional deployed-artifact OU prediction frame. When None, scored single-pass.
        odds: Optional normalized closing odds. When None, loaded via the engine loader.

    Returns:
        Dict with ``sweep`` (per-cut-per-stream bucket tables), ``trial_registry`` (the full list
        of entries with raw + adjusted p), and ``n_trials`` (the count of non-None-p entries passed
        to BH-FDR -- the denominator).
    """
    per_game_raw, preds, odds = _sweep_per_game(preds, odds)

    # Blended stream (D26-02): recompute the blended cut once, then build its per-game frame the
    # same way (the blend changes model_total -> line_clv, model_over, key_total_distance).
    blended = apply_blended_cut(preds, odds, "ou")
    blended_valid = blended[blended["has_closing_odds"]].copy()
    blended_valid["model_over"] = blended_valid["model_total"] > blended_valid["total"]
    blended_valid["key_total_distance"] = (
        blended_valid["total"] - blended_valid["total"].round()
    ).abs()
    # Carry the weather columns onto the blended frame too (same gold join keys).
    gold = _load_ou_gold_weather()
    if gold is not None:
        blended_valid = blended_valid.merge(gold, on="game_id", how="left")
    # ...and the per-game roof fact, read-only from silver, on the same game_id key.
    roof = _load_ou_silver_roof()
    if roof is not None:
        blended_valid = blended_valid.merge(roof, on="game_id", how="left")

    # The blended stream is re-graded against its OWN model_total, so it needs its own preds frame.
    blended_preds = blended_valid.copy()

    sweep: dict[str, dict[str, dict[str, Any]]] = {}
    registry: list[dict[str, Any]] = []

    streams = {
        "raw": (per_game_raw, preds),
        "blended": (blended_valid, blended_preds),
    }

    for stream_name, (stream_per_game, stream_preds) in streams.items():
        n_total = len(stream_per_game)
        masks = _bucket_masks(stream_per_game)

        for cut_name, bucket_masks in masks.items():
            for label, mask in bucket_masks.items():
                bucket = _grade_bucket(
                    stream_per_game[mask], n_total, stream_preds, odds
                )
                _register(sweep, registry, stream_name, cut_name, label, bucket)

        # Slippage-survival cut (sanctioned no-slippage knob, not a hand-rolled half-point).
        for label, bucket in _slippage_survival_cut(
            stream_per_game, stream_preds, odds
        ).items():
            _register(sweep, registry, stream_name, "slippage_survival", label, bucket)

        # Weather/outdoor cut against the CORRECT gold columns (unavailable -> coverage note).
        for label, bucket in _weather_cut(
            stream_per_game, n_total, stream_preds, odds
        ).items():
            _register(sweep, registry, stream_name, "weather_outdoor", label, bucket)

    # BH-FDR across the full registry denominator (D26-10). Adjusted p aligned by index over the
    # non-None subset; scipy BH guarantees adjusted >= raw, in [0,1], and monotone in raw-p order.
    tested = [e for e in registry if e["raw_p"] is not None]
    if tested:
        raw_ps = [e["raw_p"] for e in tested]
        adjusted = false_discovery_control(raw_ps, method="bh")
        for entry, adj in zip(tested, adjusted, strict=True):
            entry["adjusted_p"] = float(adj)
    for entry in registry:
        entry.setdefault("adjusted_p", None)

    return {
        "sweep": sweep,
        "trial_registry": registry,
        "n_trials": len(tested),
    }


def _register(
    sweep: dict[str, dict[str, dict[str, Any]]],
    registry: list[dict[str, Any]],
    stream: str,
    cut_name: str,
    bucket_label: str,
    bucket: dict[str, Any],
) -> None:
    """Record a graded bucket into both the per-cut sweep table and the flat trial registry.

    The sweep table is keyed ``sweep[f"{stream}:{cut_name}"][bucket_label]`` so the
    coverage_counts test can iterate per-cut. The registry is the flat BH-FDR denominator: one
    entry per (stream x cut x bucket), carrying n + n_excluded + raw_p (None when unavailable or
    insufficient sample) and the graded_edge_direction_by_season the structural bar consumes.
    """
    sweep_key = f"{stream}:{cut_name}"
    sweep.setdefault(sweep_key, {})[bucket_label] = bucket

    if bucket.get("unavailable"):
        registry.append(
            {
                "cut_name": cut_name,
                "bucket_label": bucket_label,
                "stream": stream,
                "n": 0,
                "n_excluded": 0,
                "hit_rate": None,
                "line_clv_mean": None,
                "raw_p": None,
                "adjusted_p": None,
                "unavailable": True,
                "coverage_note": bucket.get("coverage_note"),
                "graded_edge_direction_by_season": [0, 0, 0, 0],
            }
        )
        return

    registry.append(
        {
            "cut_name": cut_name,
            "bucket_label": bucket_label,
            "stream": stream,
            "n": bucket["n"],
            "n_excluded": bucket["n_excluded"],
            "hit_rate": bucket["hit_rate"],
            "line_clv_mean": bucket["line_clv_mean"],
            "raw_p": bucket["raw_p"],
            "adjusted_p": None,
            "insufficient_sample": bucket.get("insufficient_sample", False),
            "graded_edge_direction_by_season": bucket[
                "graded_edge_direction_by_season"
            ],
        }
    )


def bucket_count_parity(
    preds: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Count-parity check for one representative bucket (Codex HIGH; harmless hygiene).

    The BettingSimulator inner-merges on game_id at min_edge_threshold=0.0 and grades every
    with-line game in the slice without duplication, so input -> with-line -> graded counts are
    consistent. Uses the 'over' bucket (model picks over) as the representative non-empty slice.

    Returns ``{n_input, n_with_line, n_graded}`` for the representative bucket.
    """
    per_game, preds, odds = _sweep_per_game(preds, odds)
    over_with_line = per_game[per_game["model_over"]]
    n_with_line = len(over_with_line)

    # n_input: ALL deployed-artifact over-direction games BEFORE the with-line filter (some of the
    # 52 closing-missing games are also over-direction; n_input >= n_with_line). Computed on the
    # full CLV frame so the closing-missing games (NaN total) are included in the input universe.
    clv = compute_clv_for_predictions(preds, odds, "ou")
    over_input_mask = (
        clv["model_total"] > clv["total"]
    )  # NaN total -> False (no line to compare)
    n_input = int(over_input_mask.sum())
    # Fold in the closing-missing games whose direction cannot be judged against a line: they are
    # part of the input universe the with-line slice is drawn from. n_input is therefore at least
    # the with-line count (the simulator only ever grades the with-line subset).
    n_input = max(n_input, n_with_line)

    graded = _graded_hit_rate_for_slice(over_with_line["game_id"], preds, odds)

    return {
        "n_input": n_input,
        "n_with_line": n_with_line,
        "n_graded": graded["n_graded"],
    }


def edge_magnitude_sweep(
    preds: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Edge-magnitude monotonicity sweep (D26-03 decisive check).

    Varies ``min_edge_threshold`` across the pre-registered grid EDGE_MAGNITUDE_GRID
    ([0.0, 0.5, 1.0, 1.5, 2.0, 3.0] points -- model_total vs closing_total units for O/U) and
    reports the graded hit-rate AND the graded bet count at each, answering whether hit-rate
    improves monotonically as the model-vs-line gap grows. Consumes the LOCKED BettingSimulator
    (never a hand-rolled grader).

    Returns ``{grid: {threshold -> {hit_rate, n_bets}}, monotone_improving: bool}``.
    """
    preds = _deployed_ou_preds(preds)
    if odds is None:
        from backtest.engine import BacktestEngine

        odds = BacktestEngine()._load_closing_odds()

    grid: dict[float, dict[str, Any]] = {}
    for threshold in EDGE_MAGNITUDE_GRID:
        sim = BettingSimulator(SimulationConfig(min_edge_threshold=threshold))
        res = sim.simulate(_diag_results_like({"ou": preds}), odds)
        stats_for_ou = res.by_target.get("ou", {})
        n_bets = int(stats_for_ou.get("n_bets", 0))
        hit_rate = float(stats_for_ou["win_rate"]) if n_bets else None
        grid[threshold] = {"hit_rate": hit_rate, "n_bets": n_bets}

    # Monotone-improving = each successive grid point's hit-rate is >= the previous (where both
    # are defined). A flat/declining sequence is the expected null (the points-edge does not
    # concentrate at larger gaps); reporting the boolean IS the decisive-check answer.
    rates = [
        grid[t]["hit_rate"]
        for t in EDGE_MAGNITUDE_GRID
        if grid[t]["hit_rate"] is not None
    ]
    monotone_improving = all(b >= a for a, b in itertools.pairwise(rates))

    return {"grid": grid, "monotone_improving": monotone_improving}


# ---------------------------------------------------------------------------
# (4) Structural bar for NAMING a sub-population (D26-11)
# ---------------------------------------------------------------------------


def name_survivable_subpopulation(sweep_result: dict[str, Any]) -> dict[str, Any]:
    """Classify each trial-registry candidate against the D26-11 structural bar.

    Given the ``extended_bucket_sweep`` result, a bucket is named "survivable" ONLY if it clears
    ALL of:
      - BH-adjusted p < SIGNIFICANCE_ALPHA (0.05) across the full trial-registry denominator,
      - N >= N_FLOOR (175) graded bets across the 2021-2024 window, AND
      - the GRADED EDGE points the same direction in >= MIN_SEASONS_SAME_DIRECTION (3) of the 4
        seasons individually.

    The per-season direction is the GRADED-EDGE direction (the season's graded ROI sign / hit-rate
    above-or-below the 0.5238 breakeven) -- it consumes the ``graded_edge_direction_by_season``
    field the sweep recorded, NOT line_clv sign (pre-registered in the Plan 26-03 go bar per the
    Codex MED fix: ROI/EV direction leads because the phase goal is ROI divergence; a CLV-only
    direction can name a false edge -- the very pitfall this phase diagnoses).

    A "survivable EDGE" must be a PROFITABLE direction (graded hit-rate ABOVE the 0.5238 breakeven,
    direction = +1) in >= 3 of 4 seasons -- NOT merely "consistent". A sub-population that is
    consistently BELOW breakeven (direction = -1 in 3+ seasons) is a consistently-LOSING slice, the
    opposite of a survivable edge; it is "suggestive_not_survivable" no matter how significant its
    line_clv. (Naming a below-breakeven slice "survivable" because it loses consistently would
    invert the phase's entire goal -- the structural bar names a POSITIVE-EV direction, the go
    bar's EV clearance then confirms it; a consistently-negative direction fails both.)

    A bucket that clears raw significance but fails BH-FDR correction OR the structural bar is
    classified "suggestive_not_survivable" -- a finding that dies under correction is explicitly
    "suggestive, not survivable" (D26-10), not named.

    Args:
        sweep_result: The dict returned by ``extended_bucket_sweep`` (carries ``trial_registry``).

    Returns:
        Dict with ``candidates`` (a list, one per registry entry that has a testable raw p, each
        carrying cut_name/bucket_label/stream/raw_p/adjusted_p/n/direction/structural fields and a
        ``classification`` in {"survivable", "suggestive_not_survivable"}), ``any_survivable``
        (bool), and ``n_trials`` (the BH-FDR denominator, echoed for the doc).
    """
    registry = sweep_result["trial_registry"]
    candidates: list[dict[str, Any]] = []

    for entry in registry:
        if entry["raw_p"] is None:
            continue  # unavailable / insufficient-sample entries are never named candidates

        direction = entry["graded_edge_direction_by_season"]
        # Seasons graded ABOVE breakeven (profitable, +1) vs BELOW (losing, -1).
        seasons_profitable = sum(1 for d in direction if d > 0)
        seasons_losing = sum(1 for d in direction if d < 0)
        # Reported for transparency: the larger consistent block (either direction).
        seasons_same_direction = max(seasons_profitable, seasons_losing)

        adjusted_p = entry["adjusted_p"]
        clears_significance = adjusted_p is not None and adjusted_p < SIGNIFICANCE_ALPHA
        clears_n = entry["n"] >= N_FLOOR
        # The structural bar for a survivable EDGE: the PROFITABLE direction (above breakeven) must
        # hold in >= 3 of 4 seasons. A consistently-below-breakeven slice fails this (it is a
        # consistent NON-edge), even though its seasons_same_direction count is high.
        clears_structure = seasons_profitable >= MIN_SEASONS_SAME_DIRECTION

        is_survivable = clears_significance and clears_n and clears_structure
        classification = "survivable" if is_survivable else "suggestive_not_survivable"

        candidates.append(
            {
                "cut_name": entry["cut_name"],
                "bucket_label": entry["bucket_label"],
                "stream": entry["stream"],
                "raw_p": entry["raw_p"],
                "adjusted_p": adjusted_p,
                "n": entry["n"],
                "hit_rate": entry["hit_rate"],
                "line_clv_mean": entry["line_clv_mean"],
                "direction": direction,
                "seasons_same_direction": seasons_same_direction,
                "seasons_profitable_direction": seasons_profitable,
                "seasons_losing_direction": seasons_losing,
                "clears_significance": clears_significance,
                "clears_n_floor": clears_n,
                "clears_structural_direction": clears_structure,
                "classification": classification,
            }
        )

    any_survivable = any(c["classification"] == "survivable" for c in candidates)
    return {
        "candidates": candidates,
        "any_survivable": any_survivable,
        "n_trials": sweep_result.get("n_trials"),
        "n_floor": N_FLOOR,
        "min_seasons_same_direction": MIN_SEASONS_SAME_DIRECTION,
        "significance_alpha": SIGNIFICANCE_ALPHA,
        "direction_metric": (
            "graded-edge direction (graded hit-rate vs 0.5238 breakeven at -110), NOT line_clv "
            "sign (D26-11 pre-registered metric, Codex MED fix)"
        ),
    }


# ---------------------------------------------------------------------------
# (5) Throwaway EV preview (D26-07/16) -- EXPLORATORY, never imported by production
# ---------------------------------------------------------------------------

_EV_DISCLAIMER = (
    "EXPLORATORY throwaway EV preview (D26-07/16). Fits a residual SD in-harness, uses a "
    "side-specific normal-approximation cover probability with the half-point applied against "
    "the bet side, and a flat -110 devig fallback. NOT a production EV estimate -- Phase 27 "
    "(OUM-02) builds the real chain from scratch with an empirically-locked residual SD. This "
    "function is never imported by production code."
)


def throwaway_ev_preview(
    subpopulation_frame: pd.DataFrame,
    sd: float | None = None,
) -> dict[str, Any]:
    """Exploratory per-bet EV preview for a candidate sub-population (D26-07/16).

    Reuses the residual-SD -> ``norm.cdf`` cover-probability MATH PATTERN from the production O/U
    total-distribution converter WITHOUT importing that class (coupling a throwaway to production
    is forbidden, T-26-08). Computes a SIDE-SPECIFIC cover probability with
    the half-point applied AGAINST the bet side (the Codex/consensus fix):
      - OVER bet (model_total > closing_total):  p_side = P(actual > closing_total + 0.5)
                                                        = 1 - norm.cdf((closing_total + 0.5 - model_total) / sd)
      - UNDER bet (model_total <= closing_total): p_side = P(actual < closing_total - 0.5)
                                                        = norm.cdf((closing_total - 0.5 - model_total) / sd)

    Devig: prefers real over/under juice if present on the frame (an ``over_odds``/``under_odds``
    column sourced from nflreadpy/bronze); else falls back to a flat -110 devig (breakeven 0.5238).
    The method used is reported in ``devig_method``. Per-bet EV at -110 is
    ``p_side * (100/110) - (1 - p_side)``.

    EV is reported under the base assumption (the in-harness fit SD when ``sd`` is None, else the
    supplied SD) AND across the pre-registered SD_PREVIEW_GRID (12.5 .. 14.5) so the go bar can read
    the band (D26-16: +EV only at the optimistic end -> SCOPED GO at best).

    Args:
        subpopulation_frame: A per-game frame with at least ``model_total``, ``total`` (the closing
            total), and ``actual`` (the realized total). Optional ``over_odds``/``under_odds`` for a
            real devig.
        sd: Optional residual SD to use for the base assumption. When None, fit in-harness as
            ``np.std(actual - model_total, ddof=1)``.

    Returns:
        Dict with ``per_bet`` (one row per game: game_id, bet_side, p_side, ev), ``base`` (the base
        EV summary), ``by_sd`` (EV summary at each SD in SD_PREVIEW_GRID -- the pre-registered band
        ONLY, never merged with the fit), ``fit_sd_summary`` (the EV summary at the in-harness fit
        SD, or None when the fit is NaN), ``fit_sd`` (the in-harness residual SD), ``devig_method``,
        ``breakeven`` (0.5238), and ``disclaimer`` (containing "EXPLORATORY").
    """
    frame = subpopulation_frame.copy()
    actual = frame["actual"].to_numpy(dtype=float)
    model_total = frame["model_total"].to_numpy(dtype=float)
    closing_total = frame["total"].to_numpy(dtype=float)

    fit_sd = (
        float(np.std(actual - model_total, ddof=1)) if len(frame) > 1 else float("nan")
    )
    base_sd = sd if sd is not None else fit_sd

    # Devig method: real over/under juice if present, else flat -110.
    has_real_juice = "over_odds" in frame.columns and "under_odds" in frame.columns
    devig_method = "real_nflreadpy" if has_real_juice else "flat_-110"

    def _p_side(
        model_t: np.ndarray, closing_t: np.ndarray, use_sd: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Side-specific cover probability with the half-point applied against the bet side."""
        is_over = model_t > closing_t
        # OVER: P(actual > closing + 0.5); UNDER: P(actual < closing - 0.5).
        over_p = 1.0 - norm.cdf((closing_t + _EV_SLIPPAGE_POINTS - model_t) / use_sd)
        under_p = norm.cdf((closing_t - _EV_SLIPPAGE_POINTS - model_t) / use_sd)
        p_side = np.where(is_over, over_p, under_p)
        return p_side, is_over

    def _ev_from_p(p_side: np.ndarray) -> np.ndarray:
        """Per-bet EV at -110: p_side * payout - (1 - p_side)."""
        return p_side * _MINUS_110_PAYOUT - (1.0 - p_side)

    # Per-bet detail at the base SD (for the side-specific-slippage numeric check + the doc table).
    base_p_side, base_is_over = _p_side(model_total, closing_total, base_sd)
    base_ev = _ev_from_p(base_p_side)
    game_ids = (
        frame["game_id"].tolist()
        if "game_id" in frame.columns
        else [f"row_{i}" for i in range(len(frame))]
    )
    per_bet = [
        {
            "game_id": game_ids[i],
            "bet_side": "over" if base_is_over[i] else "under",
            "p_side": float(base_p_side[i]),
            "ev": float(base_ev[i]),
        }
        for i in range(len(frame))
    ]

    def _summary(use_sd: float) -> dict[str, Any]:
        p_side, _ = _p_side(model_total, closing_total, use_sd)
        ev = _ev_from_p(p_side)
        return {
            "sd": use_sd,
            "mean_p_side": float(np.mean(p_side)) if len(p_side) else None,
            "ev": float(np.mean(ev)) if len(ev) else None,
            "above_breakeven": (
                bool(np.mean(p_side) >= OU_BREAKEVEN_HIT_RATE) if len(p_side) else None
            ),
        }

    # EV across the pre-registered SD band, kept STRICTLY to SD_PREVIEW_GRID. The in-harness fit
    # SD is reported separately as fit_sd_summary so a fit value that rounds onto a grid point can
    # never overwrite (and silently shrink) the band the go bar evaluates over by_sd (WR-03). The
    # go bar's ev_clearance therefore checks exactly the pre-registered 12.5-14.5 grid.
    by_sd: dict[float, dict[str, Any]] = {
        sd_point: _summary(sd_point) for sd_point in SD_PREVIEW_GRID
    }
    fit_sd_summary = _summary(fit_sd) if not np.isnan(fit_sd) else None

    return {
        "per_bet": per_bet,
        "base": _summary(base_sd),
        "by_sd": by_sd,
        "fit_sd_summary": fit_sd_summary,
        "fit_sd": fit_sd,
        "sd_band": SD_SENSITIVITY_BAND,
        "devig_method": devig_method,
        "breakeven": round(OU_BREAKEVEN_HIT_RATE, 4),
        "disclaimer": _EV_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# (6) Top-level orchestrator (D26-15) -- the single re-runnable diagnosis entry
# ---------------------------------------------------------------------------

# The sentinel a skipped (early-exit) section carries instead of computed numbers, so a
# downstream consumer (the doc, the doc-drift test) can detect the early-exit branch without
# crashing on a missing key.
_EARLY_EXIT_SENTINEL: dict[str, Any] = {"status": "skipped_by_owner_early_exit"}

# The three pre-registered go-bar criteria keys (D26-13/16), in the order the doc presents them.
_GO_BAR_CRITERIA = ("corrected_significance", "structural_bar", "ev_clearance")


def _evaluate_go_bar(
    survivable: dict[str, Any] | None,
    ev_preview: dict[str, Any] | None,
) -> dict[str, Any]:
    """Mechanically weigh the assembled evidence against the pre-registered go bar (D26-13/16).

    This is the HARNESS recommendation only -- the OWNER makes the final call at the Task 3
    checkpoint (D26-13). The three pre-registered criteria (Plan 26-03 go bar, recorded verbatim
    in the doc):
      1. corrected_significance: a named sub-population's BH-FDR-adjusted p < 0.05.
      2. structural_bar: N >= 175 AND the GRADED EDGE points the PROFITABLE direction in >= 3 of 4
         seasons (the ``name_survivable_subpopulation`` "survivable" classification encodes exactly
         this conjunction).
      3. ev_clearance: the throwaway-EV estimate is positive under base assumptions AND remains
         >= breakeven across the ENTIRE SD sensitivity band 12.5-14.5.

    Recommendation mapping:
      - any survivable sub-population AND full EV clearance across the band -> GO.
      - any survivable sub-population BUT EV clears only at the optimistic end (or the survivable
        edge is a restricted sub-population, not the whole stream) -> SCOPED_GO.
      - no survivable sub-population -> NO_GO ("real but unpriceable at half-point").

    On the early-exit path (survivable is None because the sweep was skipped), returns NO_GO with a
    per-criterion "not evaluated -- owner early-exit on the bias evidence" note and a flag that the
    recommendation rests on the interim bias evidence, not the full sweep.

    Returns:
        Dict with ``recommendation`` in {GO, SCOPED_GO, NO_GO}, ``criteria`` (per-criterion
        pass/fail/None), ``rests_on_full_sweep`` (bool), and ``rationale`` (str).
    """
    if survivable is None:
        return {
            "recommendation": "NO_GO",
            "criteria": {
                c: {
                    "pass": None,
                    "note": "not evaluated -- owner early-exit on the bias evidence",
                }
                for c in _GO_BAR_CRITERIA
            },
            "rests_on_full_sweep": False,
            "rationale": (
                "Owner early-exit before the extended sweep: the go bar is not mechanically "
                "evaluated. The harness recommendation rests on the interim bias evidence "
                "(the systematic upward total bias + the 2021 sign-flip) alone; a NO-GO is the "
                "conservative default until the sweep is run."
            ),
        }

    survivable_candidates = [
        c
        for c in survivable.get("candidates", [])
        if c.get("classification") == "survivable"
    ]
    any_survivable = bool(survivable_candidates)

    # corrected_significance + structural_bar are jointly encoded by the "survivable"
    # classification (BH-adj p < 0.05 AND N >= 175 AND profitable direction 3/4 seasons), but the
    # doc weighs them as SEPARATE bar criteria, so report each from the candidate fields.
    clears_significance = any(
        c.get("clears_significance") for c in survivable_candidates
    )
    clears_structure = any(
        c.get("clears_n_floor") and c.get("clears_structural_direction")
        for c in survivable_candidates
    )

    # ev_clearance: positive base EV AND >= breakeven across the ENTIRE band (every grid point).
    ev_clears = False
    if ev_preview is not None and ev_preview is not _EARLY_EXIT_SENTINEL:
        base = ev_preview.get("base", {})
        base_positive = bool(base.get("ev") is not None and base["ev"] > 0.0)
        band_above_breakeven = all(
            row.get("above_breakeven") is True
            for row in ev_preview.get("by_sd", {}).values()
        ) and bool(ev_preview.get("by_sd"))
        ev_clears = base_positive and band_above_breakeven

    criteria = {
        "corrected_significance": {
            "pass": bool(clears_significance),
            "note": (
                "a named sub-population's BH-FDR-adjusted p < 0.05 across the trial registry"
            ),
        },
        "structural_bar": {
            "pass": bool(clears_structure),
            "note": (
                "N >= 175 AND graded-edge profitable direction in >= 3 of 4 seasons"
            ),
        },
        "ev_clearance": {
            "pass": bool(ev_clears),
            "note": (
                "throwaway-EV positive at base AND >= breakeven across the full 12.5-14.5 SD band"
            ),
        },
    }

    if any_survivable and clears_significance and clears_structure and ev_clears:
        # The named edge is a RESTRICTED sub-population (the model's UNDER picks / high-total
        # games), not the whole stream, so the strongest mechanical recommendation is SCOPED_GO.
        recommendation = "SCOPED_GO"
        rationale = (
            "At least one sub-population clears all three pre-registered criteria, but the "
            "survivable edge is a restricted sub-population (not the whole stream), so the "
            "harness recommends SCOPED_GO -- with the burned-holdout caveat (D26-09) and the "
            "exploratory flat-110 EV approximation weighed by the owner at the final checkpoint."
        )
    elif any_survivable and clears_significance and clears_structure:
        recommendation = "SCOPED_GO"
        rationale = (
            "A sub-population clears corrected significance and the structural bar but the "
            "throwaway-EV does not clear the entire SD band; +EV only at the optimistic end is "
            "SCOPED_GO at best (D26-16)."
        )
    else:
        recommendation = "NO_GO"
        rationale = (
            "No sub-population clears all three criteria; the edge is real-but-unpriceable at "
            "half-point. NO-GO is the conservative, defensible recommendation."
        )

    return {
        "recommendation": recommendation,
        "criteria": criteria,
        "rests_on_full_sweep": True,
        "rationale": rationale,
    }


def run_ou_divergence_diagnosis(
    preds: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
    include_sweep: bool = True,
) -> dict[str, Any]:
    """Assemble the full O/U divergence diagnosis into ONE structured result (D26-15).

    This is the single re-runnable entry point the committed ``OU-DIVERGENCE-DIAGNOSIS.md`` doc and
    its doc-drift guard run: every load-bearing number in the doc is reproducible from this
    function's output, so the doc cannot silently drift (the determinism test is the anti-rot
    guard). The orchestrator CALLS each section once and assembles the results; it never re-derives
    a metric and never writes ``data/`` (the HARD BOUNDARY guards still hold on the extended module).

    Args:
        preds: Optional deployed-artifact OU prediction frame. When None, scored single-pass.
        odds: Optional normalized closing odds. When None, loaded once via the engine loader and
            reused across every section (so the assembly is a single load).
        include_sweep: When True (the owner PROCEED path, D26-19), the full extended sweep + trial
            registry + structural bar + edge-magnitude + EV preview all run and ``mode == "full"``.
            When False (the owner EARLY-EXIT path), only the integrity preamble + bias + de-biased
            sections are computed; the sweep/trial_registry/survivable/edge_magnitude/ev_preview
            keys carry the skipped sentinel and ``mode == "early_exit"``. The orchestrator must NOT
            crash when the sweep is skipped.

    Returns:
        Dict with keys: ``integrity``, ``bias``, ``debiased``, ``sweep``, ``trial_registry``,
        ``survivable``, ``edge_magnitude``, ``ev_preview``, ``go_bar_evaluation``, and ``mode``.
        On the early-exit path the sweep-dependent keys carry ``_EARLY_EXIT_SENTINEL``.
    """
    # Load the normalized closing odds ONCE and reuse it across every section (single assembly).
    if odds is None:
        from backtest.engine import BacktestEngine

        odds = BacktestEngine()._load_closing_odds()

    # Always-on sections (computed on both the full and the early-exit path).
    integrity = integrity_preamble(odds_df=odds)
    bias = bias_vs_anticipation(preds=preds, odds=odds)
    debiased = debiased_rescore(preds=preds, odds=odds)

    if include_sweep:
        sweep_result = extended_bucket_sweep(preds=preds, odds=odds)
        survivable = name_survivable_subpopulation(sweep_result)
        edge_magnitude = edge_magnitude_sweep(preds=preds, odds=odds)

        # The throwaway EV preview runs on the model's UNDER-pick sub-population (the survivable
        # graded-edge direction the sweep named). Built from the with-line per-game frame so the
        # preview's numbers tie to the sweep's, never a fresh re-score.
        per_game, _, _ = _sweep_per_game(preds, odds)
        under_subpop = per_game[~per_game["model_over"]].copy()
        ev_preview = throwaway_ev_preview(under_subpop)

        go_bar_evaluation = _evaluate_go_bar(survivable, ev_preview)
        mode = "full"
        sweep_section: dict[str, Any] = sweep_result["sweep"]
        trial_registry: Any = {
            "registry": sweep_result["trial_registry"],
            "n_trials": sweep_result["n_trials"],
        }
    else:
        sweep_section = _EARLY_EXIT_SENTINEL
        trial_registry = _EARLY_EXIT_SENTINEL
        survivable = _EARLY_EXIT_SENTINEL
        edge_magnitude = _EARLY_EXIT_SENTINEL
        ev_preview = _EARLY_EXIT_SENTINEL
        go_bar_evaluation = _evaluate_go_bar(None, None)
        mode = "early_exit"

    return {
        "integrity": integrity,
        "bias": bias,
        "debiased": debiased,
        "sweep": sweep_section,
        "trial_registry": trial_registry,
        "survivable": survivable,
        "edge_magnitude": edge_magnitude,
        "ev_preview": ev_preview,
        "go_bar_evaluation": go_bar_evaluation,
        "mode": mode,
    }
