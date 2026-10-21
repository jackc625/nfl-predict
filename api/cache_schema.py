"""The web cache's names and pure parsers, in a module that imports only the standard library.

WHY THIS MODULE EXISTS (Phase 36, D-11). The request path -- ``api/services.py`` and
``api/routes/pages.py`` -- reads values the cache writer defines: column orders, ``cache_meta``
keys, the closed vocabularies, and the parsers for stamped values. It used to import them from
``api/cache.py``, the writer, which loads numpy, pandas and scipy at import. That put the whole
scientific stack into every page request and into the hosted bundle.

They live here now, and ``api/cache.py`` imports every one of them back, so each keeps ONE
definition (research PITFALLS E2): ``api.cache.X is api.cache_schema.X``, and every existing
``from api.cache import X`` keeps working. Nothing in this module may need numpy, pandas, scipy or
duckdb; ``tests/unit/test_cache_data_format.py`` checks its imports.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

# ---------------------------------------------------------------------------
# The bet-list column classes (Phase 31, plan 31-01; Phase 34)
# ---------------------------------------------------------------------------

# The LOCKED column order, spelled in named DISJOINT classes whose concatenation IS
# BET_LIST_COLUMNS -- immutable and grading here, with the Phase-34 fill and closing classes after
# them. WHY the first split (REVIEW-FWD-GRADE): a forward row is written at freeze, when the
# result does not yet exist. If the WHOLE row were immutable the row could never be graded and the
# self-grading honesty loop would be silently disabled. So the RECOMMENDATION facts below are
# immutable -- once written they are the record of what was recommended and when -- and the GRADING
# facts are the ONLY ones permitted to transition, and only out of ``pending``.
BET_LIST_IMMUTABLE_COLUMNS: list[str] = [
    "game_id",
    "season",
    "week",
    "target",
    "bet_side",
    "model_value",
    "market_value",
    "line",
    "slipped_line",
    "calibrated_p_side",
    "per_bet_ev",
    "stake_units",
    "ev_tier",
    "status",
    "rejection_reason",
    "eligibility_label",
    "snapshot_ts",
    # VALUE is each game's day-before-kickoff lock (D33.2-01); the NAME keeps the retired
    # 'freeze' wording by ruling: renaming a published cache column is HOST-07's schema change.
    "freeze_ts",
    "selected_odds",
    "flat_stake",
    "provenance",
    "validation_type",
    # THE ROW'S OWN OBSERVATION TIME (Phase 33, D33-27; OWNER RULING 2026-09-12). An ISO-8601
    # string with an explicit UTC offset saying when THIS run decided THIS bet. It is an
    # IMMUTABLE recommendation fact, not a grading one: a claim about when a pick was made must
    # never transition, or the forward ledger's central guarantee -- that no pick was made after
    # its own game froze -- becomes unfalsifiable.
    #
    # VARCHAR, NOT ``graded_at``'s TIMESTAMP, and that is the ruling rather than a preference. It
    # is compared against ``snapshot_ts`` and ``freeze_ts``, which are offset-carrying strings, so
    # one representation is what lets a single strict parse helper serve every comparison
    # (``scripts.ingest_historical_odds.require_aware_snapshot_ts``). A second representation
    # would need a second code path.
    #
    # LAST IN THE IMMUTABLE HALF, at index 22 of the locked order, so Phase 34's own bump for
    # ``arm``, the artifact/recipe/fill-convention stamps, the real-fill columns and the
    # reproduction key (D33-06) appends against a base that was written down.
    #
    # The 234 stored ``backtest_replay`` rows take NULL and are NEVER backfilled: every one is
    # already past its freeze, and filling them from ``snapshot_ts`` would stamp an observation
    # time nobody observed.
    "decided_at_utc",
    # ---- Phase 34 (forward ledger) stamps, appended after ``decided_at_utc`` as D33-06 planned.
    # IMMUTABLE because each is a claim about the moment of decision; a stamp that could change
    # later is no evidence of what decided the row. Every one is NULL on a row written before the
    # ledger went live and is NEVER reconstructed (SPEC must-not). This list must stay equal to
    # ``forward_ledger.canonical.IMMUTABLE_COLUMNS_V1`` -- the chain's frozen v1 column list --
    # which ``forward_ledger.schema`` checks at import.
    #
    # LDGR-02: which arm decided the row (``ARMS``). Part of the ledger row key, so a live and a
    # shadow row for the same game/week/target are two rows, never one overwriting the other.
    "arm",
    # LDGR-03: the production model artifact and blend that SCORED the row -- carried from scoring,
    # never re-read at write time, so an ``artifacts/latest.json`` swap mid-run cannot mis-stamp.
    "model_artifact_id",
    "blend_id",
    # LDGR-03: the committed recipe-registry entry (training recipe + bet rule) in force.
    "recipe_id",
    # LDGR-11: the pre-registered pricing/sizing convention the row was priced under (fill-v1).
    "fill_convention_id",
    # LDGR-09: the reproduction key -- the Phase-32 live upstream capture identity, the gold build
    # the inputs came from, the odds snapshot's digest, and the stored decision-input snapshot's
    # digest. Inside the immutable half, so the chain covers them.
    "upstream_capture_key",
    "gold_generation_key",
    "odds_snapshot_digest",
    "decision_snapshot_digest",
    # LDGR-10: whether the row counts toward the pre-registered 2026 verdict (``VERDICT_SCOPES``),
    # and the D40-08 regime label weeks 2-4 carry (``REGIME_LABEL_BOOTSTRAP``).
    "verdict_scope",
    "regime_label",
]

# WHY these six, and not ``outcome`` alone (REVIEW-SCHEMA):
#   * ``outcome`` alone cannot carry the record. The retired ``materialize_ou_bet_list`` stored BOTH
#     a push and an ungraded forward game as SQL NULL, which the tracker must tell apart -- so
#     ``grading_status`` carries an explicit four-state vocabulary (pending / win / loss / push) and
#     ``outcome`` is retained only as the boolean the existing consumers expect.
#   * ``outcome`` also cannot yield a flat return under ASYMMETRIC American prices, which is why
#     ``selected_odds`` / ``flat_stake`` (immutable) and ``payout_flat`` / ``realized_units``
#     (grading) are stored per row -- exactly as the shipped ``betting_bets`` ledger already does.
BET_LIST_GRADING_COLUMNS: list[str] = [
    "grading_status",
    "outcome",
    "clv",
    "payout_flat",
    "realized_units",
    "graded_at",
]

# The THIRD mutability class (LDGR-04, D40-05): what was ACTUALLY bet -- book, line, price, stake
# and fill time. NULL on every paper row (the default), outside the hash chain, and written only
# by the dedicated owner-run fill path, which moves each column from NULL to a value once (D-17).
# Present from the first ledger row so a real-money week needs no schema change.
BET_LIST_FILL_COLUMNS: list[str] = [
    "fill_sportsbook",
    "fill_line",
    "fill_odds",
    "fill_stake_dollars",
    "fill_at_utc",
]

# The FOURTH class (LDGR-07, D-09): the closing line read from a near-kickoff capture, and the
# forward closing-line value measured against it. REPORT-ONLY -- it feeds no decision -- and
# DISTINCT from the decision-time ``clv`` grading column, which keeps its published name. Set once
# from an in-window capture or left NULL with a ``closing_null_reason`` (``CLOSING_NULL_REASONS``);
# never filled from the decision-time line.
BET_LIST_CLOSING_COLUMNS: list[str] = [
    "closing_line",
    "closing_odds",
    "closing_sportsbook",
    "closing_captured_at",
    "forward_clv",
    "closing_null_reason",
]

# The single source of the column order shared by the schema, the explicit-column INSERT and the
# contract tests: four pairwise-disjoint classes, 34 immutable + 6 grading + 5 fill + 6 closing =
# 51 (Phase 34). The width history is why ``backtest.weekly_bet_list.read_bet_list_with_schema_shim``
# exists: 22 + 6 = 28 before Phase 33 added ``decided_at_utc``, then 29 until Phase 34 -- and a
# stored parquet still carries whichever width last wrote it.
BET_LIST_COLUMNS: list[str] = [
    *BET_LIST_IMMUTABLE_COLUMNS,
    *BET_LIST_GRADING_COLUMNS,
    *BET_LIST_FILL_COLUMNS,
    *BET_LIST_CLOSING_COLUMNS,
]

# ---------------------------------------------------------------------------
# The closed vocabularies the request path filters on
# ---------------------------------------------------------------------------

# The two orthogonal D31-22 honesty axes.
PROVENANCE_BACKTEST_REPLAY = "backtest_replay"
PROVENANCE_FORWARD = "forward"
VALIDATION_TYPE_CONTAMINATED = "contaminated"
VALIDATION_TYPE_CLEAN_HOLDOUT = "clean_holdout"
VALIDATION_TYPE_FORWARD_REALIZED = "forward_realized"
# A DISPLAY-ONLY class, NEVER stamped on a row (research Pitfall 12, D-16). A 2026 forward row
# decided before the declared counting start week keeps its stored ``forward_realized`` -- it is
# immutable and chained -- and is grouped under this class at CACHE BUILD from its
# ``verdict_scope``, so it is kept on the record and never counted in a verdict figure.
VALIDATION_TYPE_PRE_VERDICT = "pre_verdict"

# The CLOSED ``arm`` vocabulary (LDGR-02). Every row this phase writes is ``ARM_LIVE``; only the
# Phase-37 frozen shadow arm writes ``ARM_SHADOW``.
#
# NAMED ``ARM_*`` AND NOT ``*_LIVE`` ON PURPOSE: ``BET_STATUS_LIVE`` (below) is ALSO the string
# ``"live"`` but means something unrelated -- "a bet was placed" in the ``status`` column. A
# suppressed row of the live arm is ``arm='live'`` with ``status='suppressed'``; a constant named
# ``LIVE`` alone would invite exactly that confusion (34-RESEARCH Pitfall 1).
ARM_LIVE = "live"
ARM_SHADOW = "shadow"
ARMS: tuple[str, ...] = (ARM_LIVE, ARM_SHADOW)

# The contaminated replay window (2021-2024 were burned across Phases 26/27) and the single
# unburned clean-holdout season this milestone spends exactly once.
_REPLAY_CONTAMINATED_SEASONS: frozenset[int] = frozenset({2021, 2022, 2023, 2024})
_REPLAY_CLEAN_HOLDOUT_SEASON = 2025

# The seasons each backtest-replay evidence class is drawn from, read off the two constants above.
# The /bets replay tracker sections carry no season column of their own, so the website's dated
# old-rule label (R16 / D33.2-07) scopes each section through this map. A class missing from it is
# an unknown span, which labels.
REPLAY_VALIDATION_TYPE_SEASONS: dict[str, frozenset[int]] = {
    VALIDATION_TYPE_CONTAMINATED: _REPLAY_CONTAMINATED_SEASONS,
    VALIDATION_TYPE_CLEAN_HOLDOUT: frozenset({_REPLAY_CLEAN_HOLDOUT_SEASON}),
}

# The ``status`` value that means a bet was actually PLACED. ``DataService.get_bet_list`` selects
# rows equal to it and ``DataService.get_suppressed_bets`` selects the exact COMPLEMENT
# (``IS DISTINCT FROM``, so a NULL status lands in the suppressed list rather than vanishing from
# both). Deriving both readers from ONE constant is what makes "never silently dropped" (SPEC R6)
# a partition of the table rather than two whitelists that can drift apart -- and drifting
# whitelists are the failure mode this repo has already had once.
BET_STATUS_LIVE = "live"

# ---------------------------------------------------------------------------
# The column orders of the precomputed tables the request path selects from
# ---------------------------------------------------------------------------

BET_TRACKER_BLOCK_COLUMNS: list[str] = [
    "provenance",
    "validation_type",
    "bets_graded",
    "wins",
    "losses",
    "pushes",
    "hit_rate",
    "flat_return_units",
]

# The forward ledger's IN-FORCE corrections (Phase 34, D-05/D-06): one row per corrected key, the
# latest correction entry beside the row's original grade. THREE SITES like ``bet_list``: the
# CACHE_SCHEMA literal, the standalone CREATE, and the explicit-column INSERT in
# ``api.cache.materialize_bet_list_corrections`` -- ``tests/api/test_cache_betting.py`` builds and
# compares all three.
BET_LIST_CORRECTIONS_COLUMNS: list[str] = [
    "game_id",
    "season",
    "week",
    "target",
    "arm",
    "original_grading_status",
    "corrected_grading_status",
    "corrected_outcome",
    "corrected_payout_flat",
    "corrected_realized_units",
    "realized_value",
    "detected_at_utc",
    "corrected_at_utc",
]

# The /bets result strip's rows (review finding 4), materialized at cache build from the SAME
# prepared frame the tracker blocks aggregate (``backtest.bet_tracker.graded_outcome_rows``):
# ``validation_type`` is the display class and ``grading_status`` the in-force grade. Three sites,
# as above; the writer is ``api.cache.materialize_bet_graded_outcomes``.
BET_GRADED_OUTCOMES_COLUMNS: list[str] = [
    "provenance",
    "validation_type",
    "season",
    "week",
    "game_id",
    "target",
    "arm",
    "grading_status",
    "corrected",
]

# The one declared display order of the counting classes, read by the per-week aggregation
# (``backtest.bet_week_results``) and the getter (``DataService.get_bet_week_results``) alike, so
# neither falls back to alphabetical order, which agrees with this only by accident of two labels.
BET_WEEK_CLASS_ORDER: tuple[str, ...] = (
    VALIDATION_TYPE_FORWARD_REALIZED,
    VALIDATION_TYPE_PRE_VERDICT,
)

# Last week's actual versus expected (SPEC R5), one row per (season, week, counting class),
# PRECOMPUTED at cache build by ``backtest.bet_week_results.aggregate_bet_weeks``. A NULL return is
# "not measured", never zero. THREE SITES like ``bet_list``: the CACHE_SCHEMA literal, the
# standalone CREATE, and the explicit-column INSERT in ``api.cache.materialize_bet_week_results``
# -- ``tests/api/test_cache_betting.py`` builds and compares all three.
BET_WEEK_RESULTS_COLUMNS: list[str] = [
    "season",
    "week",
    "counting_class",
    "wins",
    "losses",
    "pushes",
    "bets_graded",
    "bets_ungraded",
    "actual_return_units",
    "expected_return_units",
]

# The re-fitting vs standing-still panel on /season (Phase 37, D-12..D-14, SPEC R13), PRECOMPUTED at
# cache build by ``refit.comparison.panel_frames`` from the ledger, the silver scores and the re-fit
# records. RAW NUMBERS ONLY (D-13): no p-value, no confidence range and no test result is ever
# stored here -- the pre-registered test runs once, at season close. Per target, one ``period`` row
# per period (``period_index`` 0 is the stretch before the first re-fit, N is re-fit generation N,
# present from its install even with no completed pair) and one ``season_to_date`` row over every
# paired game since the first re-fit (``period_index`` NULL). ``gap`` is live minus frozen, so a
# negative gap means the live arm erred less; ``ahead`` names the arm with the lower error
# (``live``, ``frozen`` or ``even``). A NULL error, gap or ``ahead`` means "not measured yet",
# never 0.0. THREE SITES like ``bet_week_results``: the CACHE_SCHEMA literal, the standalone CREATE,
# and the explicit-column INSERT in ``api.cache.materialize_refit_panel``.
REFIT_PANEL_COLUMNS: list[str] = [
    "season",
    "target",
    "period_index",
    "row_kind",
    "period_label",
    "period_start",
    "live_error",
    "frozen_error",
    "gap",
    "ahead",
    "paired_games",
    "unit",
]

# The two ``row_kind`` values of ``refit_panel``.
REFIT_PANEL_PERIOD = "period"
REFIT_PANEL_SEASON_TO_DATE = "season_to_date"

# The A/A status beside the panel (D-14: before the first re-fit the panel shows which weeks were
# checked and whether each was clean), one row per evaluated A/A week record in
# ``ledger/refit/aa_weeks.jsonl``, with the first mismatched field, game and target of an unclean
# week (NULL for a clean one). Materialized with ``refit_panel`` by the same writer.
REFIT_AA_STATUS_COLUMNS: list[str] = [
    "season",
    "week",
    "evaluated_at",
    "clean",
    "reason",
    "live_rows",
    "twins",
    "mismatch_field",
    "mismatch_game",
    "mismatch_target",
]

# The one season the comparison panel belongs to. The page gates on it (Plan 37-19) because api/
# may not import the pre-registration record; ``tests/unit/test_refit_panel_cache.py`` pins it to
# ``backtest.refit_preregistration_2026.REFIT_SEASON``.
REFIT_PANEL_SEASON: int = 2026

# The ``predictions`` table's columns in CACHE_SCHEMA order. Both loaders in ``api.cache`` select
# exactly this list, so the positional ``INSERT ... SELECT *`` lines up with the schema for either
# source.
PREDICTIONS_TABLE_COLUMNS: list[str] = [
    "game_id",
    "season",
    "week",
    "game_date",
    "home_team",
    "away_team",
    "status",
    "home_score",
    "away_score",
    "wp_prob",
    "wp_confidence",
    "ats_prediction",
    "ats_confidence",
    "ou_prediction",
    "ou_confidence",
    "market_spread",
    "market_total",
    "market_ml_home",
    "market_ml_away",
    "wp_edge",
    "ats_edge",
    "ou_edge",
    "blended_wp",
    "blended_ats",
    "blended_ou",
    "market_wp",
]

# ---------------------------------------------------------------------------
# cache_meta keys and the parsers for their stamped values
# ---------------------------------------------------------------------------

# The forward verdict context /bets needs to name its counting start week (D-15, D-16, LDGR-08),
# stamped at cache build so the page computes nothing. ``declared`` is ``"true"`` / ``"false"``;
# the season and start week are NULL while nothing is declared.
FORWARD_VERDICT_DECLARED_KEY = "forward_verdict_declared"
FORWARD_VERDICT_SEASON_KEY = "forward_verdict_season"
FORWARD_VERDICT_START_WEEK_KEY = "forward_verdict_start_week"

# The honesty pair of Track Record's betting simulation (LDGR-08), derived from the seasons the
# simulation itself covers. Absent when the seasons map to no single evidence class.
BETTING_SIM_PROVENANCE_KEY = "betting_sim_provenance"
BETTING_SIM_VALIDATION_TYPE_KEY = "betting_sim_validation_type"

# The PREFIX of the cache_meta keys stamping when a week's bet list was last materialized. It is
# a prefix and not a key: see ``bet_list_populated_at_key``.
BET_LIST_POPULATED_AT_KEY = "bet_list_populated_at"

# The separator between the prefix and the (season, week) pair. Named so a reader of a raw
# cache_meta dump can decompose a key, and so the two sides -- the writer in ``api.cache`` and
# ``DataService.get_bet_list_populated_at`` -- cannot spell it differently.
BET_LIST_POPULATED_AT_SEPARATOR = ":"


def bet_list_populated_at_key(season: int, week: int) -> str:
    """The cache_meta key stamping when THAT week's bet list was materialized (D31-29).

    PER-WEEK, not global, and that is the whole design. ``/bets`` hard-blocks a week whose bet list
    was populated BEFORE that week's latest per-game line freeze, and the value it compares is this
    marker. Reading freshness off the generic ``last_updated`` timestamp was REJECTED because it
    advances whenever ANY cache table is repopulated: a run that populated predictions and then
    failed on the bet list would look fresh, and the guard would be defeated by the exact failure
    it exists to catch. Keying by week closes the second half of the same hole -- a run that
    populated week 2 must not vouch for a week 1 it never touched.

    Args:
        season: NFL season. Accepts a numpy integer (what a DataFrame column yields) as well as a
            Python one; both key the same cell.
        week: NFL week, same.

    Returns:
        ``bet_list_populated_at:<season>:<week>``.
    """
    return (
        f"{BET_LIST_POPULATED_AT_KEY}"
        f"{BET_LIST_POPULATED_AT_SEPARATOR}{int(season)}"
        f"{BET_LIST_POPULATED_AT_SEPARATOR}{int(week)}"
    )


# The season spans of the pre-rendered blocks (Phase 33.2, R16 / D33.2-07). The writer stamps them
# (``api.cache.stamp_old_rule_season_ranges``); the website reads them back through
# ``DataService.cached_span_old_rule_scope``.

# The backtest corpus: Track Record's all-time summary and model-vs-market section
# (/track-record) and every How It Works chart (/how-it-works) -- every table their charts and
# figures are drawn from. The retired /backtest, /insights and /performance URLs redirect there.
BACKTEST_SEASON_RANGE_KEY = "backtest_season_range"

# Track Record's betting simulation (/track-record#betting-sim; the retired /betting URL
# redirects there): the per-bet simulation ledger its KPI strip, charts and ROI table are drawn
# from.
BETTING_SEASON_RANGE_KEY = "betting_season_range"

# The value stamped for a source with no rows: a known-empty span, so no label.
NO_SEASONS = "none"


def format_season_range(low: int | None, high: int | None) -> str:
    """Render a season span as ``"MIN-MAX"``, or ``"none"`` for a source with no rows."""
    if low is None or high is None:
        return NO_SEASONS
    return f"{int(low)}-{int(high)}"


def parse_season_range(value: str | None) -> list[int] | None:
    """Read a stamped span back: ``[MIN, MAX]``, ``[]`` for ``"none"``, None if unreadable.

    None means UNKNOWN -- an absent key or a value this format does not describe -- and the label
    partial labels an unknown span. Returning an empty list for garbage instead would turn a
    malformed stamp into a silent "no pre-fix numbers here".
    """
    if value == NO_SEASONS:
        return []
    if not value or value.count("-") != 1:
        return None
    low, high = value.split("-")
    if not (low.isdigit() and high.isdigit()):
        return None
    return [int(low), int(high)]


# The current slate, stamped for /bets' default week (33.2 review C2 WR-01). The writer resolves
# and stamps it (``api.cache.stamp_current_slate``); /bets reads it back.
CURRENT_SLATE_KEY = "current_slate"
_CURRENT_SLATE_SEPARATOR = ":"


def format_current_slate(season: int, week: int) -> str:
    """Render a slate as ``"SEASON:WEEK"``."""
    return f"{int(season)}{_CURRENT_SLATE_SEPARATOR}{int(week)}"


def parse_current_slate(value: str | None) -> tuple[int, int] | None:
    """Read a stamped slate back as ``(season, week)``; None when absent or unreadable."""
    if not value or value.count(_CURRENT_SLATE_SEPARATOR) != 1:
        return None
    season, week = value.split(_CURRENT_SLATE_SEPARATOR)
    if not (season.isdecimal() and week.isdecimal()):
        return None
    return int(season), int(week)


# ---------------------------------------------------------------------------
# The cache's data-format contract (Phase 36, D-20, D-04, D-14)
# ---------------------------------------------------------------------------

# The data-format version the pages expect, stamped into cache_meta by ``populate_cache`` and read
# back by ``api.cache_reader.read_data_format``. RAISE IT whenever the pages need something an
# older cache lacks. ``tests/unit/test_cache_data_format.py`` pins every table and column name the
# writer builds beside this number, so a layout change without a bump fails there. KNOWN LIMIT:
# that test sees table and column names only. A change inside a stored blob -- the pre-rendered
# chart HTML, a JSON payload -- is not detected and must be judged by hand. Version 2 (Phase 37,
# D-14): the ``refit_panel`` and ``refit_aa_status`` tables were added.
DATA_FORMAT_VERSION: int = 2
DATA_FORMAT_VERSION_KEY = "data_format_version"

# The publish stamp the push writes into cache_meta on a copy of the cache (D-14,
# ``api.cache.stamp_publish``): this attempt's id and the UTC push time as ISO text.
PUBLISH_ID_KEY = "publish_id"
PUSH_TIME_KEY = "push_time"

# A stamped version is a plain integer and nothing else: no decimal point, no whitespace.
_INTEGER_TEXT = re.compile(r"-?[0-9]+")


@dataclass(frozen=True)
class DataFormatState:
    """What the cache says its data format is, against what the code expects.

    Attributes:
        raw: The stored ``data_format_version`` text, or None when the cache has none.
        expected: The version this code expects.
        matches: True only when ``raw`` is an integer equal to ``expected``.
    """

    raw: str | None
    expected: int
    matches: bool

    @property
    def found_label(self) -> str:
        """The stored version as shown to a reader: its integer text, "missing" or "non_integer"."""
        if not self.raw:
            return "missing"
        if _INTEGER_TEXT.fullmatch(self.raw) is None:
            return "non_integer"
        return self.raw


def data_format_state(
    raw: str | None, expected: int = DATA_FORMAT_VERSION
) -> DataFormatState:
    """Judge a stored data-format value strictly: only the exact expected integer matches."""
    matches = (
        raw is not None
        and _INTEGER_TEXT.fullmatch(raw) is not None
        and int(raw) == expected
    )
    return DataFormatState(raw=raw, expected=expected, matches=matches)


# How long the site may go without a push before it says it is out of date (SPEC R8). The daily
# run pushes once a day, so 36 hours is a missed run plus slack.
STALE_AFTER = timedelta(hours=36)


def push_is_stale(push_time_raw: str | None, now: datetime) -> bool:
    """Whether the last push is too old to trust -- THE ONE RULE the bar and /health share.

    Args:
        push_time_raw: The stamped ``push_time`` text, or None when the cache carries none.
        now: The current instant, timezone-aware.

    Returns:
        True when more than 36 hours separate the push from *now* (exactly 36 hours is still
        fresh), and True when there is no push time or it cannot be read -- absent is never
        fresh. A naive stamp is read as UTC.
    """
    if not push_time_raw:
        return True
    try:
        push_time = datetime.fromisoformat(push_time_raw)
    except ValueError:
        return True
    if push_time.tzinfo is None:
        push_time = push_time.replace(tzinfo=UTC)
    return now - push_time > STALE_AFTER
