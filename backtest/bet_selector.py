"""Single-source O/U bet-decision engine (Phase 27, plan 27-03; BET-01/02, OUM-04/06).

``BetSelector.select()`` is the ONE place O/U bet decisions are made (BET-01, LOCKED-2). It is a
THIN HARNESS in the house style of ``backtest/diagnose.py`` and ``backtest/ou_divergence.py``: it
CALLS the LOCKED scorers and never re-derives a metric. Responsibilities, in order:

  1. Provenance hard-fail (OUM-06, T-27-07): ``assert_real_odds`` rejects any odds row whose
     sportsbook is outside {consensus, draftkings} or whose ``is_live`` is True, raising a
     ValueError naming the offending game_ids. Called BEFORE any selection.
  2. Sub-pop UNION filter (D27-04/05): a candidate is eligible iff its bet_side is "under" OR its
     totals_regime is "high". The bet_side comes from the LOCKED
     ``BettingSimulator._determine_bet_side_ou``; the totals_regime comes from the leakage-clean
     PRE-HOLD boundary ``ou_divergence.HIGH_TOTAL_BOUNDARY_PREHOLD`` (LOCKED-1), NOT the legacy
     hold-informed 46.5.
  3. EV admission (D27-14): within the eligible set, a bet is admitted iff its per-bet EV is at or
     above the EV-floor scalar ``t``. The EV uses the calibrated P(side) from the Plan-01 EV chain
     (``ou_ev_chain.calibrated_p_over``) evaluated against the half-point-slipped line (the line
     moves against the bettor, via the LOCKED ``apply_slippage_total``), so the high-total OVER
     over-bias pocket (graded below breakeven) is dropped (T-27-08 / D27-05).
  4. Sizing (BET-02 fix, T-27-08): Kelly consumes the CALIBRATED P(side) -- never the points
     distance ``implied + abs(model_total - closing_total)`` -- through
     ``KellyCalculator.calculate_optimal_bet_size(model_prob=p_side, ...)``, then the Plan-02
     LOCKED-order ``apply_sizing_pipeline`` (kelly -> 5% per-bet -> same-side de-weight -> 10%
     weekly cap), per week. Unit = 1% of bankroll (D27-09).
  5. Push handling (#8, T-27-23): grading uses the LOCKED ``_resolve_ou_outcome``; a push is carried
     as ``outcome=None`` on the record, never coerced to win/loss.
  6. CLV reporting (D27-06/12, REPORT-ONLY, T-27-10): per bet ``compute_line_clv(model_total,
     closing_total, direction="total")`` -- the MODEL EDGE vs the line (the non-zero line_clv,
     ~+1.11 pooled) -- is attached and summarized via ``diagnose.clv_significance``. This is
     DISTINCT from the separate freeze-vs-close forward metric, which is structurally ~0 here
     because stored silver carries freeze == close (one synthetic snapshot per game); the backtest
     CLV-vs-close is a LIVE-ONLY FORWARD metric (D27-12). CLV NEVER gates a bet -- it is computed
     and reported only. Do not imply the backtest line_clv is meaningful historical forward
     evidence.

``select()`` returns BOTH the FILTERED decision set (the acceptance basis) and the UNFILTERED
whole-population cross-check (D27-04), plus the REJECTED eligible candidates with rejection reasons
in {not_subpop, ev_below_floor, real_odds_failed}.

It WRAPS -- never re-implements -- the LOCKED ``BettingSimulator`` side/slippage/outcome convention
(D-18). It makes NO change to ``models/clv.py``, ``config/gate.toml``, or any production artifact
(the LOCKED self-judge boundary). It does NOT import the exploratory Phase-26 EV preview from the
divergence harness (the no-leak guard, T-26-08); it consumes the production-grade Plan-01 EV chain.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backtest.diagnose import clv_significance
from backtest.ou_divergence import (
    _ALLOWED_SPORTSBOOKS,
    HIGH_TOTAL_BOUNDARY_PREHOLD,
)
from backtest.ou_ev_chain import (
    MINUS_110_PAYOUT,
    calibrated_p_over,
    per_bet_ev,
)
from backtest.simulation import (
    SLIPPAGE_POINTS,
    STANDARD_VIG_ODDS,
    BettingSimulator,
    SimulationConfig,
    apply_slippage_total,
)
from models.clv import compute_line_clv
from utils import get_logger
from utils.kelly_criterion import KellyCalculator, KellyMode, apply_sizing_pipeline

logger = get_logger(__name__)

# The rejection-reason taxonomy surfaced on rejected records (BET-01). An eligible candidate that
# is not bet is tagged with exactly one of these; the taxonomy is exported so callers/tests do not
# redefine the strings.
REJECTION_REASONS: tuple[str, ...] = (
    "not_subpop",  # outside the UNION {under} OR {high-total} (D27-04/05)
    "ev_below_floor",  # eligible but per-bet EV < the EV-floor t (D27-14)
    "real_odds_failed",  # provenance hard-fail (OUM-06) -- raised before selection
)

__all__ = [
    "REJECTION_REASONS",
    "BetSelector",
    "SelectionResult",
    "assert_real_odds",
]


def assert_real_odds(raw_odds_df: pd.DataFrame) -> None:
    """Hard-fail on mock/synthetic odds BEFORE any selection (OUM-06, T-27-07).

    Replicates the LOCKED ``ou_divergence.integrity_preamble`` provenance guard logic (the
    allowlist + is_live check) directly on a passed frame, so the selector can validate an
    arbitrary candidate/odds frame without the harness's silver-parquet read. A row is offending if
    its ``sportsbook`` is outside {consensus, draftkings} OR its ``is_live`` is True. Raises a
    ValueError naming the offending game_ids; a clean frame passes silently.

    Args:
        raw_odds_df: A frame carrying at least ``game_id`` and ``sportsbook`` (and optionally
            ``is_live``). When ``sportsbook`` is absent the frame is treated as not carrying
            provenance (no check possible) and passes -- callers wanting the guard must supply it.

    Raises:
        ValueError: naming the offending game_ids when a mock/synthetic-odds contamination signal
            is present.
    """
    if (
        raw_odds_df is None
        or raw_odds_df.empty
        or "sportsbook" not in raw_odds_df.columns
    ):
        return

    bad_book_mask = ~raw_odds_df["sportsbook"].isin(_ALLOWED_SPORTSBOOKS)
    if "is_live" in raw_odds_df.columns:
        live_mask = raw_odds_df["is_live"].fillna(False).astype(bool)
    else:
        live_mask = pd.Series(False, index=raw_odds_df.index)

    offending_mask = bad_book_mask | live_mask
    if not offending_mask.any():
        return

    offenders = raw_odds_df.loc[offending_mask, "game_id"].head(10).tolist()
    bad_books = sorted(
        set(raw_odds_df.loc[bad_book_mask, "sportsbook"].dropna().unique())
    )
    msg = (
        "Odds provenance check FAILED (mock/synthetic-odds contamination, OUM-06): "
        f"unexpected sportsbooks={bad_books}, is_live rows={int(live_mask.sum())}; "
        f"offending game_ids={offenders}"
    )
    raise ValueError(msg)


@dataclass
class SelectionResult:
    """The BetSelector output (BET-01).

    Attributes:
        selected: Per-bet records the selector BET (eligible AND EV >= floor).
        rejected: Per-candidate records NOT bet, each carrying a ``rejection_reason`` in
            ``REJECTION_REASONS`` (not_subpop / ev_below_floor).
        filtered: The eligible acceptance-basis records (the sub-pop UNION; both selected and
            EV-floor-rejected eligible candidates carry their decision metadata here).
        unfiltered: The whole-population cross-check (every candidate, eligible or not) -- a
            REPORTED cross-check only (D27-04), never the acceptance basis.
        clv_report: A report-only CLV summary over the SELECTED bets (mean / t / p / 95% CI via
            ``clv_significance``). NEVER used to gate (D27-06). None when no bets were selected.
    """

    selected: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    filtered: list[dict[str, Any]] = field(default_factory=list)
    unfiltered: list[dict[str, Any]] = field(default_factory=list)
    clv_report: dict[str, Any] | None = None


class BetSelector:
    """Single-source O/U bet-decision engine (BET-01/02, OUM-04/06, LOCKED-1/2).

    Construct with the frozen residual SD and the per-season prior-season-walk-forward bias (both
    fit on tune-only data by the caller -- the no-leak fence is the caller's responsibility, per the
    Plan-01 EV-chain contract), the EV-floor scalar ``ev_floor_t`` (Plan 04 tunes it), the bankroll,
    and the leakage-clean pre-hold high-total boundary. ``select()`` is the ONE source of O/U bet
    decisions (LOCKED-2).
    """

    def __init__(
        self,
        frozen_sd: float,
        season_bias_by_season: dict[int, float],
        ev_floor_t: float = 0.0,
        bankroll: float = 10_000.0,
        high_total_boundary: float = HIGH_TOTAL_BOUNDARY_PREHOLD,
        slippage_points: float = SLIPPAGE_POINTS,
        odds: int = STANDARD_VIG_ODDS,
    ) -> None:
        self.frozen_sd = float(frozen_sd)
        self.season_bias_by_season = dict(season_bias_by_season)
        self.ev_floor_t = float(ev_floor_t)
        self.bankroll = float(bankroll)
        self.high_total_boundary = float(high_total_boundary)
        self.slippage_points = float(slippage_points)
        self.odds = int(odds)

        # Wrap -- never re-implement -- the LOCKED side/slippage/outcome convention (D-18). The
        # simulator is used ONLY for its grading helpers; the BetSelector owns the decision.
        self._sim = BettingSimulator(SimulationConfig())

        # Kelly sizing (BET-02): the calibrated P(side) is fed to the EXISTING calculator. unit = 1%
        # of bankroll (D27-09); quarter-Kelly via default_kelly_fraction=0.25; 5% per-bet cap.
        self._kelly = KellyCalculator(
            starting_bankroll=self.bankroll,
            max_bet_pct=0.05,
            base_unit_size=self.bankroll * 0.01,
            default_kelly_fraction=0.25,
            # The EV chain owns admission; do not double-gate sizing on a points-edge threshold.
            confidence_threshold=0.0,
        )

    # -- side / regime / EV helpers (wrap the LOCKED scorers) -----------------

    def _bet_side(self, model_total: float, closing_total: float) -> str | None:
        """Determine the O/U bet side via the LOCKED ``_determine_bet_side_ou`` (D-18)."""
        return self._sim._determine_bet_side_ou(model_total, closing_total)

    def _totals_regime(self, closing_total: float) -> str:
        """High iff the closing total exceeds the leakage-clean PRE-HOLD boundary (LOCKED-1)."""
        return "high" if closing_total > self.high_total_boundary else "not_high"

    def _season_bias(self, season: int) -> float:
        """Prior-season walk-forward bias for ``season`` (NEGATIVE for an over-biased model)."""
        if season not in self.season_bias_by_season:
            msg = (
                f"no prior-season bias provided for season {season}; the caller must supply a "
                "walk-forward bias (ou_ev_chain.estimate_prior_season_bias) for every candidate "
                "season (no silent fallback to the raw biased total, D27-07)."
            )
            raise ValueError(msg)
        return float(self.season_bias_by_season[season])

    def _calibrated_p_side(
        self, bet_side: str, model_total: float, closing_total: float, season: int
    ) -> tuple[float, float]:
        """Calibrated P(side) and the slipped line for one candidate (BET-02 input).

        The P(side) is evaluated against the HALF-POINT-SLIPPED line (the line moves against the
        bettor, the LOCKED ``apply_slippage_total``), so a high-total OVER's over-bias is not
        rewarded: the slipped line + the bias correction pull the calibrated P(over) down. Returns
        ``(p_side, slipped_line)``.
        """
        slipped_line = apply_slippage_total(
            closing_total, bet_side, self.slippage_points
        )
        season_bias = self._season_bias(season)
        p_over = float(
            calibrated_p_over(model_total, slipped_line, self.frozen_sd, season_bias)
        )
        p_side = p_over if bet_side == "over" else (1.0 - p_over)
        return p_side, slipped_line

    # -- main entry point -----------------------------------------------------

    def select(
        self,
        candidates: pd.DataFrame | list[dict[str, Any]],
        raw_odds_df: pd.DataFrame | None = None,
    ) -> SelectionResult:
        """The single O/U bet-decision source (BET-01/02, LOCKED-2).

        Args:
            candidates: Per-game O/U candidate rows (a DataFrame or list of dicts) carrying at least
                ``game_id``, ``season``, ``week``, ``model_total``, ``closing_total``, and
                (for grading) ``actual``. May also carry ``sportsbook`` / ``is_live`` provenance
                columns, which are validated when present.
            raw_odds_df: Optional raw odds provenance frame. When supplied, it is validated FIRST
                via ``assert_real_odds`` (OUM-06) before any selection. When None, the candidates
                frame itself is checked for provenance columns.

        Returns:
            A :class:`SelectionResult` with selected / rejected / filtered / unfiltered records and
            a report-only ``clv_report``.
        """
        rows = self._to_records(candidates)

        # (1) Provenance hard-fail BEFORE any selection (OUM-06, T-27-07). Validate the explicit
        # raw odds frame if supplied; otherwise validate any provenance columns on the candidates.
        if raw_odds_df is not None:
            assert_real_odds(raw_odds_df)
        else:
            assert_real_odds(pd.DataFrame(rows))

        selected: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        filtered: list[dict[str, Any]] = []
        unfiltered: list[dict[str, Any]] = []

        # Eligible candidates needing sizing, grouped by week so the per-week caps apply per week.
        eligible_by_week: dict[tuple[int, int], list[dict[str, Any]]] = {}

        for row in rows:
            record = self._build_decision_record(row)
            unfiltered.append(record)

            if not record["eligible"]:
                rejected.append({**record, "rejection_reason": "not_subpop"})
                continue

            # Eligible: it is part of the acceptance basis regardless of the EV decision.
            filtered.append(record)
            key = (int(record["season"]), int(record["week"]))
            eligible_by_week.setdefault(key, []).append(record)

        # (3)/(4) EV admission + LOCKED-order sizing, per week.
        for (_season, _week), week_records in eligible_by_week.items():
            self._admit_and_size_week(week_records, selected, rejected)

        clv_report = self._clv_report(selected)

        logger.info(
            "BetSelector.select complete",
            n_candidates=len(rows),
            n_eligible=len(filtered),
            n_selected=len(selected),
            n_rejected=len(rejected),
            ev_floor_t=self.ev_floor_t,
        )

        return SelectionResult(
            selected=selected,
            rejected=rejected,
            filtered=filtered,
            unfiltered=unfiltered,
            clv_report=clv_report,
        )

    # -- decision-record construction -----------------------------------------

    def _build_decision_record(self, row: dict[str, Any]) -> dict[str, Any]:
        """Build the per-candidate decision record (side, regime, eligibility, P(side), EV, CLV)."""
        game_id = row["game_id"]
        season = int(row["season"])
        week = int(row["week"])
        model_total = float(row["model_total"])
        closing_total = float(row["closing_total"])

        bet_side = self._bet_side(model_total, closing_total)
        totals_regime = self._totals_regime(closing_total)

        # Sub-pop UNION (D27-04/05): eligible iff under-pick OR high-total. A None side (the model
        # agrees with the market within the side threshold) is not a bet of either side.
        is_under = bet_side == "under"
        is_high = totals_regime == "high"
        eligible = bool(bet_side is not None and (is_under or is_high))

        record: dict[str, Any] = {
            "game_id": game_id,
            "season": season,
            "week": week,
            "bet_side": bet_side,
            "totals_regime": totals_regime,
            "subpop_label": self._subpop_label(is_under, is_high),
            "model_total": model_total,
            "closing_total": closing_total,
            "eligible": eligible,
            "calibrated_p_side": None,
            "per_bet_ev": None,
            "slipped_line": None,
            "kelly_stake": 0.0,
            "outcome": None,
            # CLV (D27-06, REPORT-ONLY): the model-edge line_clv (model_total - closing_total),
            # DISTINCT from the freeze-vs-close forward metric (~0); never a selection gate.
            "clv": compute_line_clv(model_total, closing_total, direction="total"),
        }

        if eligible:
            p_side, slipped_line = self._calibrated_p_side(
                bet_side, model_total, closing_total, season
            )
            record["calibrated_p_side"] = p_side
            record["slipped_line"] = slipped_line
            record["per_bet_ev"] = per_bet_ev(p_side, MINUS_110_PAYOUT)

        return record

    @staticmethod
    def _subpop_label(is_under: bool, is_high: bool) -> str:
        """A human-readable sub-pop label for the UNION arms a candidate satisfies."""
        if is_under and is_high:
            return "under+high_total"
        if is_under:
            return "under"
        if is_high:
            return "high_total"
        return "none"

    # -- EV admission + sizing (per week) -------------------------------------

    def _admit_and_size_week(
        self,
        week_records: list[dict[str, Any]],
        selected: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
    ) -> None:
        """Admit eligible records by the EV floor, then size the admitted set via the LOCKED order.

        EV admission: bet iff ``per_bet_ev(p_side) >= ev_floor_t`` (D27-14). Sizing: Kelly on the
        CALIBRATED P(side) (BET-02 fix) -> the Plan-02 ``apply_sizing_pipeline`` LOCKED cap order.
        Grading uses the LOCKED ``_resolve_ou_outcome`` (push-aware).
        """
        admitted: list[dict[str, Any]] = []
        for record in week_records:
            if record["per_bet_ev"] is None or record["per_bet_ev"] < self.ev_floor_t:
                rejected.append({**record, "rejection_reason": "ev_below_floor"})
                continue
            admitted.append(record)

        if not admitted:
            return

        # Kelly stake on the CALIBRATED P(side) (BET-02 fix -- NEVER implied + points_edge).
        kelly_inputs: list[dict[str, Any]] = []
        for record in admitted:
            kelly_result = self._kelly.calculate_optimal_bet_size(
                model_prob=record["calibrated_p_side"],
                market_odds=self.odds,
                mode=KellyMode.FRACTIONAL,
            )
            kelly_inputs.append(
                {"bet_side": record["bet_side"], "stake": kelly_result.recommended_bet}
            )

        # LOCKED-order sizing pipeline (Plan-02): kelly -> 5% per-bet -> same-side de-weight ->
        # 10% weekly cap. The BetSelector consumes this helper and does NOT re-order the steps.
        sized = apply_sizing_pipeline(kelly_inputs, self.bankroll)

        for record, sized_rec in zip(admitted, sized, strict=True):
            record["kelly_stake"] = sized_rec["weekly_scaled_stake"]
            record["outcome"] = self._grade(record)
            selected.append(record)

    def _grade(self, record: dict[str, Any]) -> bool | None:
        """Grade a selected bet via the LOCKED ``_resolve_ou_outcome`` (push-aware, T-27-23).

        Returns True (win), False (loss), or None (push). The push (actual == slipped line) is
        carried as None, never coerced. When the candidate carries no ``actual`` (a forward,
        not-yet-played game), the outcome is None (ungraded) -- distinct from a push but both
        represented by None here; downstream stores both as SQL NULL (Plan 04).
        """
        actual = record.get("_actual_total")
        if actual is None:
            return None
        return self._sim._resolve_ou_outcome(
            record["bet_side"], float(actual), record["slipped_line"]
        )

    # -- report-only CLV ------------------------------------------------------

    @staticmethod
    def _clv_report(selected: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Report-only CLV summary over the selected bets (D27-06/12, NEVER a gate).

        Computes the mean / t / p / 95% CI of the per-bet model-edge line_clv via the LOCKED
        ``clv_significance``. This is the model-edge line_clv, DISTINCT from the freeze-vs-close
        forward metric (~0 in backtest because freeze == close); CLV-vs-close is a live-only
        forward metric (D27-12). Returns None when no bets were selected.
        """
        if not selected:
            return None
        clv_values = [r["clv"] for r in selected if r["clv"] is not None]
        if not clv_values:
            return None
        report = dict(clv_significance(clv_values))
        report["metric"] = (
            "model_edge_line_clv (model_total - closing_total); REPORT-ONLY (D27-06)"
        )
        return report

    # -- input normalization --------------------------------------------------

    @staticmethod
    def _to_records(
        candidates: pd.DataFrame | list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Normalize candidates to a list of dicts, stashing the grading ``actual`` privately.

        The ``actual`` total (when present) is kept under ``_actual_total`` so the public record
        never confuses the model/closing totals with the realized outcome.
        """
        if isinstance(candidates, pd.DataFrame):
            raw = candidates.to_dict("records")
        else:
            raw = [dict(c) for c in candidates]

        records: list[dict[str, Any]] = []
        for row in raw:
            rec = dict(row)
            if "actual" in rec:
                rec["_actual_total"] = rec.get("actual")
            records.append(rec)
        return records
