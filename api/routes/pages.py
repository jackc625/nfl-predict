"""HTML page route handlers for the NFL Prediction System.

Serves full HTML pages using Jinja2Blocks templates. When an HX-Request
header is present, only the relevant block is returned (HTMX fragment).

Routes:
    GET /            -- This Week's predictions dashboard (landing page)
    GET /performance -- Historical performance view
    GET /backtest    -- Backtest analysis with Plotly charts
    GET /insights    -- Model insights (calibration, feature importance, vs market)
    GET /betting     -- Betting dashboard (KPI strip, equity, ROI, edge; scope toggle)
    GET /bets        -- Weekly bet list (ranked +EV bets, units, EV band; week selector)
    GET /games/{id}  -- Game detail drill-down (feature importance, market comparison)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, NamedTuple

from fastapi import APIRouter, Depends, Query, Request

from api.cache import (
    BACKTEST_SEASON_RANGE_KEY,
    BETTING_SEASON_RANGE_KEY,
    PROVENANCE_BACKTEST_REPLAY,
    REPLAY_VALIDATION_TYPE_SEASONS,
)
from api.charts import BETTING_CHART_IDS, INSIGHTS_CHART_IDS
from api.dependencies import get_data_service, templates
from api.season_metrics import (
    _ats_outcome,
    _ou_outcome,
    _wp_outcome,
    season_has_graded_games,
)
from api.services import DataService
from utils import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["pages"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pivot_season_metrics(raw_metrics: list[dict]) -> list[dict]:
    """Pivot long-format (season, target, metric_name, metric_value) rows
    into one row per (season, target) with named metric columns.

    Returns list of dicts with keys: season, target, games, accuracy, mae, rmse, r2.
    """
    grouped: dict[tuple[int, str], dict[str, Any]] = {}

    for m in raw_metrics:
        season = m.get("season", 0)
        target = m.get("target", "")
        metric_name = m.get("metric_name", "")
        metric_value = m.get("metric_value")

        if season == 0:
            continue

        key = (season, target)
        if key not in grouped:
            grouped[key] = {
                "season": season,
                "target": target,
                "games": None,
                "accuracy": None,
                "mae": None,
                "rmse": None,
                "r2": None,
            }

        if metric_name in ("n_games", "n_predictions"):
            grouped[key]["games"] = int(metric_value) if metric_value else None
        elif metric_name == "accuracy":
            grouped[key]["accuracy"] = (
                float(metric_value) * 100 if metric_value else None
            )
        elif metric_name == "mae":
            grouped[key]["mae"] = float(metric_value) if metric_value else None
        elif metric_name == "rmse":
            grouped[key]["rmse"] = float(metric_value) if metric_value else None
        elif metric_name == "r2":
            grouped[key]["r2"] = float(metric_value) if metric_value else None

    rows = sorted(grouped.values(), key=lambda r: (r["season"], r["target"]))
    return rows


def _compute_week_summary(games: list[dict]) -> dict[str, Any]:
    """Compute per-target accuracy for completed games in a week.

    For each target (WP, ATS, O/U), counts correct predictions and
    returns counts plus percentages.

    Every per-game hit/miss/excluded decision is delegated to the authoritative
    classifiers in :mod:`api.season_metrics` (``_wp_outcome`` / ``_ats_outcome``
    / ``_ou_outcome``) so this This-Week banner reproduces the ``/betting`` and
    ``/season`` numbers EXACTLY -- a single source of truth for the locked
    simulator sign convention (CR-01/CR-02/WR-03). Those classifiers fold in:

    * the home-margin ATS convention (``ats_prediction > market_spread`` ->
      home_cover, DEF-31-01) and the O/U side rule, WITH 0.5-pt slippage,
    * push/tie exclusion (a game landing on the slipped line, or a WP tie, is
      EXCLUDED from the denominator rather than silently scored), and
    * NaN-safe score/line coercion.

    Args:
        games: List of game prediction dicts.

    Returns:
        Dict with total_games, wp_correct/wp_total/wp_pct,
        ats_correct/ats_total/ats_pct, ou_correct/ou_total/ou_pct.
        Empty dict if no completed games. Each ``*_total`` is the count of
        DECIDED games for that target (pushes/ties excluded).
    """
    completed = [g for g in games if g.get("status") == "completed"]
    if not completed:
        return {}

    def _tally(outcome_fn: Any) -> tuple[int, int]:
        """Return (correct, decided) over completed games for one classifier.

        ``None`` from the classifier means push / tie / missing data -> the game
        is excluded from BOTH the numerator and the denominator.
        """
        correct = 0
        decided = 0
        for g in completed:
            outcome = outcome_fn(g)
            if outcome is None:
                continue
            decided += 1
            if outcome:
                correct += 1
        return correct, decided

    wp_correct, wp_total = _tally(_wp_outcome)
    ats_correct, ats_total = _tally(_ats_outcome)
    ou_correct, ou_total = _tally(_ou_outcome)

    return {
        "total_games": len(completed),
        "wp_correct": wp_correct,
        "wp_total": wp_total,
        "wp_pct": round(wp_correct / wp_total * 100) if wp_total else 0,
        "ats_correct": ats_correct,
        "ats_total": ats_total,
        "ats_pct": round(ats_correct / ats_total * 100) if ats_total else 0,
        "ou_correct": ou_correct,
        "ou_total": ou_total,
        "ou_pct": round(ou_correct / ou_total * 100) if ou_total else 0,
    }


PAGE_CACHE_CONTROL = "public, max-age=60"

# Betting-scope whitelist (Security V5 / T-V5-01). Any value outside this set
# falls back to the default below before it ever reaches a ``betting_*_{scope}``
# chart_id, so untrusted query input never flows into a cache lookup key.
_BETTING_SCOPES: frozenset[str] = frozenset({"all", "recommended"})
_DEFAULT_BETTING_SCOPE = "recommended"  # D-17 honest default


def _normalize_betting_scope(scope: str) -> str:
    """Whitelist *scope* to {"all", "recommended"}, defaulting to recommended.

    The only untrusted input on the betting page is the ``scope`` query param;
    this is the single chokepoint that constrains it to the two literal scope
    variants before any cached ``betting_*_{scope}`` id is built (T-V5-01).
    """
    return scope if scope in _BETTING_SCOPES else _DEFAULT_BETTING_SCOPE


# ---------------------------------------------------------------------------
# Old-rule label scopes (Phase 33.2, R16 / D33.2-07)
# ---------------------------------------------------------------------------
# Every page context carries one scope per block that can show numbers, under a key ending in
# ``old_rule_scope``. The shared ``components/_old_rule_label.html`` partial reads it and renders
# the dated label when the block shows a 2021-2025 season. Each scope comes from the block's OWN
# data -- its rows' seasons, or the span population stamped for a pre-rendered chart -- and never
# from a request parameter such as the selected season, which on /performance does not even
# describe the all-history block it would be gating.


def _rows_seasons(rows: list[dict]) -> list[int]:
    """The seasons a block's rows carry. An empty list means the block renders no numbers."""
    return [row["season"] for row in rows if row.get("season") is not None]


def _replay_tracker_seasons(blocks: list[dict]) -> list[int] | None:
    """The seasons behind every backtest-replay tracker section that shows graded figures.

    A replay section carries no season column, so its seasons come from its evidence class through
    ``REPLAY_VALIDATION_TYPE_SEASONS``. A section with no graded bets renders an empty state and
    contributes nothing. A class the map does not know makes the whole span UNKNOWN (None), which
    labels.
    """
    seasons: list[int] = []
    for block in blocks:
        if block.get("provenance") != PROVENANCE_BACKTEST_REPLAY:
            continue
        if not block.get("bets_graded"):
            continue
        class_seasons = REPLAY_VALIDATION_TYPE_SEASONS.get(
            block.get("validation_type", "")
        )
        if class_seasons is None:
            return None
        seasons.extend(class_seasons)
    return seasons


def _build_betting_context(
    service: DataService, scope: str, request: Request
) -> dict[str, Any]:
    """Assemble the betting-page template context from cached data only.

    Reads ONLY pre-rendered HTML / JSON for the (already-whitelisted) *scope*:
    the per-scope chart HTML blobs from ``BETTING_CHART_IDS`` and the two
    JSON-blob accessors (KPI strip + ROI table). ZERO betting metric logic runs
    here -- every statistic was computed during cache population (D-20). The
    ``charts`` dict is keyed by bare chart_id (e.g. ``betting_equity_recommended``)
    so the template references each slot via ``current_scope`` and the fragment
    swap re-renders exactly the active scope's set.

    Shared by both ``betting_page`` and ``betting_fragment`` so the cached-read
    contract lives in one place and cannot drift between the two handlers.
    """
    charts: dict[str, str | None] = {
        chart_id: service.get_chart_html(chart_id)
        for chart_id in BETTING_CHART_IDS
        if chart_id.endswith(f"_{scope}")
    }
    return {
        "request": request,
        "charts": charts,
        "kpis": service.get_betting_kpis(scope),
        "roi_table": service.get_betting_roi_table(scope),
        "current_scope": scope,
        "current_path": "/betting",
        "cache_meta": service.get_cache_meta(),
        # One block: every KPI, chart and ROI row is drawn from the betting simulation ledger.
        # Built here so betting_page and betting_fragment receive it from one place.
        "old_rule_scope": service.cached_span_old_rule_scope(BETTING_SEASON_RANGE_KEY),
    }


def _normalize_season(season: int | None, available: list[int]) -> int | None:
    """Whitelist *season* to the seasons present in ``predictions`` (T-V5-01).

    The only untrusted input on the season page is the ``season`` query param;
    this is the single chokepoint that constrains it before any cached
    ``season_*_{season}`` id is built. Mirrors ``_normalize_betting_scope`` but
    the valid set is data-dependent rather than two literals:

    * ``available`` is ``service.get_prediction_seasons()`` -- DESC-ordered, so
      ``available[0]`` is the dynamically-resolved latest season (D-01).
    * In-range seasons pass through unchanged.
    * ``None`` (no param) or any out-of-range value falls back to the latest
      available season -- never a hardcoded year (D-01 / D-04 / T-V5-01).
    * If there are no seasons at all (empty DB), returns ``None`` so the page
      renders its whole-season empty state instead of 500ing.
    """
    if not available:
        return None
    if season in available:
        return season
    return available[0]


def _build_season_context(
    service: DataService, season: int | None, request: Request
) -> dict[str, Any]:
    """Assemble the season-tracking template context from cached data only.

    Reads ONLY pre-rendered HTML / JSON for the (already-whitelisted) *season*:
    the per-season cumulative + weekly chart HTML (via ``get_chart_html``) and
    the per-season KPI JSON blob (via ``get_season_kpis``). ZERO hit-rate metric
    logic runs here -- every statistic was computed during cache population
    (D-12). The ``charts`` dict is keyed by bare chart_id (e.g.
    ``season_cumulative_<year>``) so the template references each slot via
    ``current_season`` and the fragment swap re-renders exactly the active
    season's set.

    Shared by both ``season_tracking_page`` and ``season_fragment`` so the
    cached-read contract lives in one place and cannot drift between the two
    handlers.
    """
    charts: dict[str, str | None] = {}
    if season is not None:
        charts[f"season_cumulative_{season}"] = service.get_chart_html(
            f"season_cumulative_{season}"
        )
        charts[f"season_weekly_{season}"] = service.get_chart_html(
            f"season_weekly_{season}"
        )
    kpis = service.get_season_kpis(season) if season is not None else {}
    # A season with NOTHING GRADED shows the empty state, not "0.0%" and "0-0" (33.2 review C2
    # WR-02). Its KPI blob exists -- population builds one for every season in ``predictions``,
    # live weeks included -- so the test is on the decided counts, and the charts, which always
    # render an empty-chart div, are not consulted. The old-rule scope is known-empty in exactly
    # that case and the season otherwise.
    if not season_has_graded_games(kpis):
        kpis = {}
    shows_numbers = season is not None and bool(kpis)
    return {
        "request": request,
        "charts": charts,
        "kpis": kpis,
        "available_seasons": service.get_prediction_seasons(),
        "current_season": season,
        "current_path": "/season",
        "cache_meta": service.get_cache_meta(),
        "old_rule_scope": DataService.old_rule_scope([season] if shows_numbers else []),
    }


def _parse_int_param(raw: str | None) -> int | None:
    """Parse a raw query-string integer defensively, returning None on anything else.

    Mirrors the ``season_tracking_page`` convention: an unparseable value (``?week=abc``)
    degrades to the dynamic default rather than raising a 422 (WR-02). The whitelist in
    :func:`_normalize_week` still holds -- a non-int can never reach a SQL parameter.

    PARSE FIRST, VALIDATE SECOND (WR-07). The previous guard was
    ``int(raw) if raw.strip().lstrip("-").isdigit() else None``, which contradicted the very
    contract this docstring states, in two ways. ``lstrip("-")`` strips ALL leading hyphens, so
    ``"--5"`` passed the guard as ``"5".isdigit()`` and then ``int("--5")`` raised. And
    ``str.isdigit()`` is True for superscripts -- ``"2".isdigit()`` is True while ``int("2")``
    raises. Either one produced an unhandled ``ValueError`` and an HTTP 500 with a stack trace on
    a public page, from a two-character query string. Letting ``int`` decide what an int is
    removes the whole class.
    """
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return None


def _normalize_week(
    service: DataService, season: int | None, week: int | None
) -> tuple[int | None, int | None]:
    """Whitelist ``(season, week)`` against the SCHEDULE-derived bet tables (T-31-01, REVIEW-NAV).

    The SINGLE chokepoint for the only untrusted input on ``/bets``. Both values are constrained
    to what ``get_bet_seasons`` / ``get_available_bet_weeks`` returned BEFORE any cache key or SQL
    parameter is built, so a tampered query string can only ever select a real scheduled week.

    Resolution order:

    * ``season`` in the available set passes through. Anything else (including ``None``) falls
      back to the CURRENT SLATE's season when population stamped one, else to the latest
      available season -- never a hardcoded year.
    * ``week`` in that season's available set passes through. Anything else falls back to the
      current slate's week when it is in that season (33.2 review C2 WR-01: the default used to be
      the season's LAST scheduled week, a January week with nothing in it), else to the latest
      week of the season that carries a bet list, else to the latest scheduled week.
    * When ``available_bet_weeks`` is EMPTY the pair resolves to ``(None, None)`` so the page
      renders its no-current-week empty state. It deliberately does NOT fall back to
      ``get_available_weeks`` / ``get_available_seasons``: those read ``predictions`` and
      ``backtest_metrics``, which would reintroduce exactly the dependency the schedule-derived
      tables exist to remove (REVIEW-NAV).
    """
    available_seasons = service.get_bet_seasons()
    if not available_seasons:
        return None, None

    slate = service.get_current_slate()
    if season in available_seasons:
        resolved_season = season
    elif slate is not None and slate[0] in available_seasons:
        resolved_season = slate[0]
    else:
        resolved_season = available_seasons[0]

    weeks = service.get_available_bet_weeks(season=resolved_season)
    if not weeks:
        return resolved_season, None

    valid_weeks = {row["week"] for row in weeks}
    if week in valid_weeks:
        return resolved_season, week
    if slate is not None and slate[0] == resolved_season and slate[1] in valid_weeks:
        return resolved_season, slate[1]
    latest_built = service.get_latest_built_bet_week(resolved_season)
    if latest_built in valid_weeks:
        return resolved_season, latest_built
    return resolved_season, weeks[0]["week"]


def _as_utc(value: Any) -> datetime | None:
    """Coerce a cache timestamp to an aware UTC datetime, or None if it cannot be read.

    Both sides of the freshness comparison are written in UTC -- ``populate_cache`` stamps
    ``datetime.now(tz=UTC).isoformat()`` and ``bet_week_freeze`` persists an upstream-computed
    instant -- but they arrive in different shapes (a string from ``cache_meta``, a DuckDB
    ``TIMESTAMP WITH TIME ZONE`` from the freeze table).

    A NAIVE VALUE IS REFUSED RATHER THAN ASSUMED TO BE UTC (CR-01). The previous
    ``value.replace(tzinfo=UTC)`` is what made this reader agree with a freeze DuckDB had already
    shifted: the freeze column was a naive ``TIMESTAMP``, so a tz-aware 22:00Z instant was cast
    through the session ``TimeZone`` and came back as a naive 18:00, which this function then
    re-labelled as 18:00Z. Both halves looked self-consistent and the threshold was silently wrong
    by the server's own offset. The zone is now pinned at the WRITE side
    (``api.cache.materialize_bet_week_freeze``), so every value reaching here carries one; a naive
    one means the cache was written by a build that predates that fix and its instant cannot be
    read without guessing a zone.

    Returns None on anything unparseable -- including a naive datetime -- so the caller can
    decline to make a claim.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        logger.warning(
            "A /bets freshness instant is naive; refusing to guess a zone for it",
            value=str(value),
        )
        return None
    return value.astimezone(UTC)


def _lock_has_passed(lock: Any, now: datetime) -> bool | None:
    """Whether a game's lock is at or before *now*; None when the lock cannot be read.

    At-lock counts as passed: information timed at the lock is admissible (D33.2-01), so the run
    that builds a game's list must finish before it. A TIMESTAMP COMPARISON, not a metric.
    """
    instant = _as_utc(lock)
    if instant is None:
        return None
    return instant <= now


class BetWeekCoverage(NamedTuple):
    """Which of a week's games have a bet list, judged game by game against each game's own lock.

    Returned as one object so the verdict and the values the page shows beside it come from the
    SAME reads -- a second, separate lookup for display could disagree with the one the verdict
    was decided on.

    Attributes:
        built: At least one game of the week has a bet-list row.
        missing_games: Games whose lock has passed with no bet-list row. Permanent -- a game's
            list cannot be built after its lock -- and named on the page, never dropped.
        pending_games: Games whose lock is still ahead with no row yet: not evaluated YET.
        populated_at: This week's populated-at marker, shown for the reader to check.
        expected_freeze: The week's latest game lock, shown beside it.
    """

    built: bool
    missing_games: list[str]
    pending_games: list[str]
    populated_at: Any
    expected_freeze: Any

    @property
    def blocked(self) -> bool:
        """Games have locked and NONE has a list: the failed-insertion state, nothing to show."""
        return bool(self.missing_games) and not self.built

    @property
    def not_evaluated(self) -> bool:
        """Nothing is built and nothing is missing: every game's lock is still ahead."""
        return bool(self.pending_games) and not self.missing_games and not self.built


def _bet_week_coverage(
    service: DataService,
    season: int | None,
    week: int | None,
    now: datetime | None = None,
) -> BetWeekCoverage:
    """THE per-game coverage verdict for ``/bets`` (33.2 review C2 CR-02).

    WHY PER GAME. Under the daily run each game's list is built the day before its own kickoff,
    before its 18:00 ET lock (D33.2-01). The previous verdict compared the per-week populated-at
    marker against the week's LATEST lock -- and every population finishes before that lock by
    design, so the current week read "withheld" from its first built game until the following
    week, after every one of its games had kicked off. Now each game is judged on its own:

    * a game with a bet-list row (live or suppressed -- every evaluated game yields one per bet
      type) is BUILT and shown;
    * a game whose lock has passed with no row is MISSING, and is named;
    * a game whose lock is still ahead with no row is PENDING -- not evaluated yet, not stale.

    The week is BLOCKED (nothing to show) only when games have locked and none has a row: the
    failed-insertion state the refusal exists for, in which there may be no bet rows at all.

    THE SOURCES. The locks come from the SCHEDULE-derived ``bet_game_lock`` table, never from bet
    rows: a guard reading its threshold from the rows it checks could not fire in the zero-row
    case (REVIEW-STALE). Coverage comes from the rows' game ids -- a keyed existence probe, not a
    metric. The generic ``last_updated`` stamp is not consulted at all (D31-29); the per-week
    marker and the week-level lock are read for DISPLAY only. ``tests/api/test_bets_page.py``
    asserts these sources structurally, by walking this function's AST.

    Args:
        service: The request's data service.
        season: The resolved season, or None.
        week: The resolved week, or None.
        now: The instant to judge at; the current UTC time when omitted.

    Returns:
        The :class:`BetWeekCoverage`.
    """
    instant = now if now is not None else datetime.now(tz=UTC)
    built = set(service.get_bet_game_ids(season, week))
    missing: list[str] = []
    pending: list[str] = []
    for game in service.get_bet_game_locks(season, week):
        if game["game_id"] in built:
            continue
        passed = _lock_has_passed(game["lock_ts"], instant)
        if passed is True:
            missing.append(game["game_id"])
        elif passed is False:
            pending.append(game["game_id"])
    # The refusal shows the instants normalized to UTC (CR-01): DuckDB hands a TIMESTAMPTZ back in
    # the SESSION's own zone, and an UNREADABLE value falls back to the raw stored one.
    expected_freeze = service.get_bet_week_freeze(season, week)
    freeze_utc = _as_utc(expected_freeze)
    return BetWeekCoverage(
        built=bool(built),
        missing_games=missing,
        pending_games=pending,
        populated_at=service.get_bet_list_populated_at(season, week),
        expected_freeze=expected_freeze if freeze_utc is None else freeze_utc,
    )


def _build_bets_context(
    service: DataService,
    season: int | None,
    week: int | None,
    request: Request,
) -> dict[str, Any]:
    """Assemble the ``/bets`` template context from cached blobs ONLY.

    Every value is read straight out of the DuckDB cache: the ranked bet rows, the
    schedule-derived navigation lists, the per-week freeze, and the cache stamp. ZERO metric
    logic runs here -- the EV, the stake in units and the EV band were all computed by the
    selector and written by ``api.cache.materialize_bet_list`` (UIAP-01, SPEC R5). Shared by the
    page handler and (from Plan 31-15) the fragment handler, so the cached-read contract lives in
    one place and cannot drift between them.
    """
    cache_meta = service.get_cache_meta()
    # The per-game coverage verdict and the two timestamps the refusal interpolates, from ONE set
    # of reads (33.2 review C2 CR-02). See _bet_week_coverage for why each game is judged against
    # its own lock rather than the week against its latest one.
    freshness = _bet_week_coverage(service, season, week)
    bet_list_available = service.bet_list_table_exists()
    tracker_blocks = service.get_bet_tracker_blocks()

    return {
        "request": request,
        "bets": service.get_bet_list(season, week),
        # The declined half of the SAME candidate universe. It is READ here rather than derived
        # from the live list, because the live list is the complement -- deriving one from the
        # other would put the R6 "never silently dropped" guarantee in the request path instead of
        # in the partition the two getters share (plan 31-15).
        "suppressed_bets": service.get_suppressed_bets(season, week),
        "available_bet_weeks": service.get_available_bet_weeks(season=season),
        "bet_seasons": service.get_bet_seasons(),
        "current_season": season,
        "current_week": week,
        "bet_week_freeze": freshness.expected_freeze,
        "current_path": "/bets",
        "cache_meta": cache_meta,
        # A cache that predates Phase 31 has no bet_list table. That is a DIFFERENT absence from a
        # week that admitted nothing, and the page says so rather than reporting a missing table
        # as a modelling result (plan 31-15, UI-SPEC E2 empty).
        "bet_list_available": bet_list_available,
        "bet_list_populated_at": freshness.populated_at,
        # The hard-block (D31-27), SCOPED and now PER GAME (33.2 review C2 CR-02). It fires only
        # when games of the week have passed their lock and NONE has a list -- the failed
        # insertion -- and then withholds the week's list, its suppressed disclosure and the
        # forward tracker block. Replay weeks and the replay tracker render normally in the SAME
        # response. A week with SOME games built shows them, and names the missing ones.
        "bets_blocked": freshness.blocked,
        # Games past their lock with no list (named, never dropped), and games whose lock is still
        # ahead (not evaluated yet). A week where every game is pending is not evaluated yet.
        "missing_games": freshness.missing_games,
        "pending_games": freshness.pending_games,
        "week_not_evaluated": freshness.not_evaluated,
        # The PRECOMPUTED realized-vs-expected tracker blocks (SPEC R8, D31-22, plan 31-16). One
        # stored row per (provenance, validation_type) class, aggregated at population time by
        # ``backtest.bet_tracker`` and read here without a single arithmetic operation -- no count,
        # no rate, no return and no SQL aggregate (UIAP-01). The template partitions the rows into
        # its sections by matching the two stored labels; it never pools two classes into one
        # figure, because the pooled figure does not exist to render.
        "tracker_blocks": tracker_blocks,
        # Two blocks, two scopes. The selected week's list (a 2021-2025 week is a replay the old
        # models reconstructed), and the backtest-replay tracker sections. The forward tracker
        # section is 2026 by construction and carries no label.
        "week_old_rule_scope": DataService.old_rule_scope(
            [season] if season is not None and bet_list_available else []
        ),
        "replay_old_rule_scope": DataService.old_rule_scope(
            _replay_tracker_seasons(tracker_blocks)
        ),
    }


def _annotate_wp_correct(games: list[dict]) -> list[dict]:
    """Return a new list of games with wp_correct annotated.

    Builds fresh shallow copies of each dict so the source list (which may
    come from the DataService TTLCache in plan 15-02) is never mutated.
    A shallow copy is sufficient because wp_correct is a scalar -- no
    nested structures are touched.

    wp_correct is True if the WP prediction was correct, False if
    incorrect, or None if the game is not completed, data is missing, or the
    game was a tie. The hit/miss/tie decision is delegated to
    ``api.season_metrics._wp_outcome`` so the per-game badge uses the SAME
    locked WP convention as the This-Week banner and the season page (WR-03) --
    notably it now EXCLUDES ties (``margin == 0`` -> ``None``) instead of
    scoring them. The explicit ``status == 'completed'`` guard is retained so a
    scheduled game carrying stale scores never receives a hit/miss badge.
    """
    annotated: list[dict] = []
    for game in games:
        new_game = dict(game)
        if new_game.get("status") == "completed":
            new_game["wp_correct"] = _wp_outcome(new_game)
        else:
            new_game["wp_correct"] = None
        annotated.append(new_game)
    return annotated


def _compute_summary(service: Any) -> dict[str, Any]:
    """Aggregate all-time summary metrics from backtest data.

    Computes: total_games, overall_clv (mean wp CLV), wp_accuracy, brier_score.

    Args:
        service: DataService instance.

    Returns:
        Dict with summary metric values.
    """
    all_metrics = service.get_backtest_metrics()

    # ``brier_score`` is None until a Brier score is measured: the card renders "--", never a
    # zero and never another metric under the Brier label (33.2 review C2 WR-06).
    summary: dict[str, Any] = {
        "total_games": 0,
        "overall_clv": 0.0,
        "wp_accuracy": 0.0,
        "brier_score": None,
    }

    if not all_metrics:
        return summary

    # Gather WP-specific metrics across seasons
    wp_accuracy_values: list[float] = []
    wp_brier_values: list[float] = []
    wp_clv_values: list[float] = []

    for m in all_metrics:
        target = m.get("target", "")
        metric_name = m.get("metric_name", "")
        metric_value = m.get("metric_value", 0.0)
        season = m.get("season", 0)

        if season == 0:
            # Overall aggregate metrics
            if target == "overall" and metric_name == "total_games":
                summary["total_games"] = int(metric_value)
            continue

        if target == "wp":
            if metric_name == "accuracy":
                wp_accuracy_values.append(float(metric_value))
            elif metric_name == "brier_score":
                # The Brier score ONLY. The MAE used to be averaged in "as a proxy", so the card
                # labelled "WP Brier Score" printed the mean absolute error (33.2 review C2
                # WR-06); population now derives the real one (api.cache._load_wp_brier_scores).
                wp_brier_values.append(float(metric_value))

    # Compute CLV from predictions. Require has_closing_odds is explicitly
    # True so rows where the closing-line capture status is unknown
    # (None / NULL) do not pollute the overall CLV summary. Mirrors the
    # contract in api/charts/core.py::generate_dashboard_clv_chart.
    predictions = service.get_backtest_predictions()
    wp_preds = [
        p
        for p in predictions
        if p.get("target") == "wp"
        and p.get("probability_clv") is not None
        and p.get("has_closing_odds") is True
    ]

    if wp_preds:
        wp_clv_values = [float(p["probability_clv"]) for p in wp_preds]
        summary["overall_clv"] = sum(wp_clv_values) / len(wp_clv_values) * 100

    if not summary["total_games"]:
        # Count from predictions if not in aggregate metrics
        summary["total_games"] = len(
            {p["game_id"] for p in predictions if p.get("game_id")}
        )

    if wp_accuracy_values:
        summary["wp_accuracy"] = sum(wp_accuracy_values) / len(wp_accuracy_values) * 100
    if wp_brier_values:
        summary["brier_score"] = sum(wp_brier_values) / len(wp_brier_values)

    return summary


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/")
def this_week_page(
    request: Request,
    week: int | None = Query(None),
    season: int | None = Query(None),
    sort: str = Query("time"),
    service: DataService = Depends(get_data_service),
):
    """Serve the predictions dashboard (landing page).

    Fetches predictions for the selected (or latest) week and renders
    the full page. If the request comes from HTMX, returns only the
    game_grid block.

    Defaults to the most recent season/week with prediction data (D-01, D-02).
    Computes wp_correct for correct/incorrect indicators (D-03) and
    week_summary for per-target accuracy banner (D-04).
    """
    available_seasons = service.get_prediction_seasons()
    cache_meta = service.get_cache_meta()

    # Default to latest season with predictions (D-01)
    if season is None and available_seasons:
        season = available_seasons[0]

    available_weeks = service.get_available_weeks(season=season)

    # Default to latest week (not Week 1) per D-02
    if week is None and available_weeks:
        week = available_weeks[0]["week"]
        season = available_weeks[0]["season"]

    games = service.get_predictions(season=season, week=week, sort=sort)

    # Compute wp_correct on a fresh list so the DataService TTLCache source
    # is never mutated (plan 15-02 review item #4).
    games = _annotate_wp_correct(games)

    context = {
        "request": request,
        "games": games,
        "available_weeks": available_weeks,
        "available_seasons": available_seasons,
        "current_week": week,
        "current_season": season,
        "current_sort": sort,
        "current_path": "/",
        "cache_meta": cache_meta,
        "week_summary": _compute_week_summary(games),
        # One block: the week summary and the game grid, scoped to the games actually shown.
        "old_rule_scope": DataService.old_rule_scope(_rows_seasons(games)),
    }

    block_name = "game_grid" if request.headers.get("HX-Request") else None
    template_response = templates.TemplateResponse(
        request, "pages/this_week.html", context, block_name=block_name
    )
    # Cache-Control is set directly on the returned TemplateResponse because
    # headers set on an injected ``response: Response`` parameter are NOT
    # propagated when the handler returns its own response object (a known
    # FastAPI/Starlette behaviour).
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/performance")
def performance_page(
    request: Request,
    season: int | None = Query(None),
    service: DataService = Depends(get_data_service),
):
    """Serve the historical performance page.

    Shows all-time summary metrics with a season selector that swaps
    season-specific metrics via HTMX.
    """
    available_seasons = service.get_available_seasons()
    raw_metrics = service.get_backtest_metrics(season=season)
    season_metrics = _pivot_season_metrics(raw_metrics)
    summary = _compute_summary(service)
    cache_meta = service.get_cache_meta()

    context = {
        "request": request,
        "available_seasons": available_seasons,
        "current_season": season,
        "season_metrics": season_metrics,
        "summary": summary,
        "current_path": "/performance",
        "cache_meta": cache_meta,
        # Two blocks, two scopes. The summary aggregates the WHOLE backtest corpus whatever season
        # is selected (_compute_summary reads every metric), so its scope is that corpus's span and
        # never the season query parameter. The table is scoped to the rows it shows.
        "summary_old_rule_scope": service.cached_span_old_rule_scope(
            BACKTEST_SEASON_RANGE_KEY
        ),
        "season_metrics_old_rule_scope": DataService.old_rule_scope(
            _rows_seasons(season_metrics)
        ),
    }

    # If HTMX request, return only the performance_content block
    block = "performance_content" if request.headers.get("HX-Request") else None
    template_response = templates.TemplateResponse(
        request, "pages/performance.html", context, block_name=block
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/backtest")
def backtest_page(
    request: Request,
    service: DataService = Depends(get_data_service),
):
    """Serve the backtest results page with 4 Plotly charts.

    Charts are pre-rendered in the DuckDB chart_cache for fast serving.
    Falls back to empty state components if charts are not available.
    """
    charts = {
        "calibration": service.get_chart_html("calibration"),
        "clv": service.get_chart_html("clv"),
        "heatmap": service.get_chart_html("heatmap"),
        "equity": service.get_chart_html("equity"),
    }
    cache_meta = service.get_cache_meta()

    context = {
        "request": request,
        "charts": charts,
        "current_path": "/backtest",
        "cache_meta": cache_meta,
        # One block: the four charts, pre-rendered over the whole backtest corpus.
        "old_rule_scope": service.cached_span_old_rule_scope(BACKTEST_SEASON_RANGE_KEY),
    }
    template_response = templates.TemplateResponse(
        request, "pages/backtest.html", context
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/insights")
def insights_page(
    request: Request,
    service: DataService = Depends(get_data_service),
):
    """Serve the Model Insights page.

    All charts are pre-rendered during cache population (Plan 16-02) and
    stored in the DuckDB chart_cache. The aggregate Model-vs-Market table
    is also precomputed during pre-render and stored under chart_id
    ``insights_aggregate_table``. This handler contains NO statistical
    logic -- Plan 16-02 is the single source of truth for metric formulas
    (REVIEWS Codex HIGH #1).

    Page is static with no filters (D-19). Cache-Control header matches
    the other static-artifact pages (Phase 15 D-07 / REVIEWS Codex MEDIUM
    #12).
    """
    # Fetch the 9 new insights chart HTML blobs.
    charts: dict[str, str | None] = {
        chart_id: service.get_chart_html(chart_id) for chart_id in INSIGHTS_CHART_IDS
    }
    # Reuse the existing WP reliability chart (not part of INSIGHTS_CHART_IDS per D-22).
    charts["calibration"] = service.get_chart_html("calibration")

    # Precomputed aggregate table (list of dicts). Falls back to [] if missing/malformed.
    aggregate_table = service.get_insights_aggregate_table()

    cache_meta = service.get_cache_meta()
    context = {
        "request": request,
        "charts": charts,
        "aggregate_table": aggregate_table,
        "current_path": "/insights",
        "cache_meta": cache_meta,
        # One block: all three sections are drawn from the same backtest corpus.
        "old_rule_scope": service.cached_span_old_rule_scope(BACKTEST_SEASON_RANGE_KEY),
    }
    template_response = templates.TemplateResponse(
        request, "pages/insights.html", context
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/betting")
def betting_page(
    request: Request,
    scope: str = Query(_DEFAULT_BETTING_SCOPE),
    service: DataService = Depends(get_data_service),
):
    """Serve the Betting Dashboard page.

    Renders the 7-card KPI strip, the flat-vs-Kelly equity curve plus three
    per-bet-type mini equities, the three ROI grouped-bar charts plus a ROI
    summary table, and the three per-type edge histograms -- all in D-04 order
    (KPI -> Equity -> ROI -> Edge). A page-level All/Recommended toggle (D-05/
    D-17) re-renders the swappable ``betting_content`` block via HTMX.

    All charts and the KPI/ROI JSON blobs are pre-rendered for BOTH scope
    variants during cache population (Plan 17-03); this handler reads cached
    HTML/JSON only and contains NO betting metric logic (D-20). ``scope`` is
    whitelisted to {"all", "recommended"}, defaulting to ``"recommended"`` on
    anything else (Security V5 / T-V5-01).

    On an HX-Request the handler returns only the ``betting_content`` block so a
    full navigation to ``/betting?scope=`` and the toggle's fragment swap share
    one code path. Cache-Control is set on the returned TemplateResponse
    (Phase 15 D-07).
    """
    scope = _normalize_betting_scope(scope)
    context = _build_betting_context(service, scope, request)

    block_name = "betting_content" if request.headers.get("HX-Request") else None
    template_response = templates.TemplateResponse(
        request, "pages/betting.html", context, block_name=block_name
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/bets")
def bets_page(
    request: Request,
    season: str | None = Query(None),
    week: str | None = Query(None),
    service: DataService = Depends(get_data_service),
):
    """Serve the Weekly Bet List page (D31-25, SPEC R5).

    ONE page carrying the not-advice banner and the ranked live bet list, reached by ONE new nav
    item. Rows are rendered exactly as the selector produced them -- per-bet EV descending, ties
    broken ``(season, week, game_id, target)`` -- with the stake in UNITS and the pre-registered
    EV band. There is NO computation on the request path: this handler reads cached blobs only,
    and ``tests/api/test_import_guard_bets.py`` makes that mechanically checkable by forbidding
    any new ``backtest`` import under ``api/``.

    ``season`` and ``week`` are the only untrusted input and are whitelisted at the single
    ``_normalize_week`` chokepoint against the SCHEDULE-derived ``available_bet_weeks`` table
    before any SQL parameter is built (T-31-01). They are accepted as raw strings, mirroring
    ``season_tracking_page``, so an unparseable value degrades to the dynamic default rather than
    raising a 422.

    On an HX-Request the handler returns only the ``bets_content`` block, so a full navigation to
    ``/bets?season=&week=`` and a fragment swap share one code path. Cache-Control is set on the
    returned TemplateResponse (Phase 15 D-07).
    """
    season_resolved, week_resolved = _normalize_week(
        service, _parse_int_param(season), _parse_int_param(week)
    )
    context = _build_bets_context(service, season_resolved, week_resolved, request)

    block_name = "bets_content" if request.headers.get("HX-Request") else None
    template_response = templates.TemplateResponse(
        request, "pages/bets.html", context, block_name=block_name
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/season")
def season_tracking_page(
    request: Request,
    season: str | None = Query(None),
    service: DataService = Depends(get_data_service),
):
    """Serve the Season Tracking page.

    Renders the season-to-date KPI strip, the cumulative-accuracy chart
    (running per-target hit rate, DASH-07), and the weekly-performance chart
    (per-week hit rate + rolling overlay, DASH-08) for the selected season.
    A season selector re-renders the swappable ``season_tracking_content``
    block via HTMX (D-11).

    The default season is the dynamically-resolved latest season present in
    ``predictions`` (D-01) -- never a hardcoded year. ``season`` is whitelisted
    to ``service.get_prediction_seasons()`` before any cached
    ``season_*_{season}`` id is built; an out-of-range value falls back to the
    latest available season (Security V5 / T-V5-01).

    All charts and the KPI JSON blob are pre-rendered for every season during
    cache population (Plan 18-02); this handler reads cached HTML/JSON only and
    contains NO hit-rate metric logic (D-12). On an HX-Request the handler
    returns only the ``season_tracking_content`` block so a full navigation to
    ``/season?season=`` and the selector's fragment swap share one code path.
    Cache-Control is set on the returned TemplateResponse (Phase 15 D-07).

    ``season`` is accepted as a raw string (mirroring the sibling fragment
    routes ``/fragments/games`` and ``/fragments/performance``) and parsed
    defensively: an unparseable value (e.g. ``?season=abc``) degrades to the
    dynamic latest-season default via ``_normalize_season`` rather than raising
    a 422 (WR-02). The T-V5-01 whitelist still holds -- a non-int can never reach
    a ``season_*_{season}`` cache id because ``_normalize_season`` rejects it.
    """
    available = service.get_prediction_seasons()
    # Through the SHARED parser (WR-07). This used to inline the same broken
    # ``lstrip("-").isdigit()`` guard, so ``/season?season=--5`` and ``/season?season=<superscript>``
    # both 500'd exactly as ``/bets?week=`` did. One parser means one behaviour.
    season_int = _parse_int_param(season)
    season_resolved = _normalize_season(season_int, available)
    context = _build_season_context(service, season_resolved, request)

    block_name = (
        "season_tracking_content" if request.headers.get("HX-Request") else None
    )
    template_response = templates.TemplateResponse(
        request, "pages/season.html", context, block_name=block_name
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/games/{game_id}")
def game_detail_page(
    request: Request,
    game_id: str,
    service: DataService = Depends(get_data_service),
):
    """Serve the game detail drill-down page.

    Shows full prediction breakdown including feature importance bars,
    prediction vs market comparison, team context (Elo, form, H2H),
    venue/weather, and result overlay for completed games (D-13 to D-15).
    """
    game = service.get_game_detail(game_id)
    cache_meta = service.get_cache_meta()
    context = {
        "request": request,
        "game": game,
        "current_path": "",
        "cache_meta": cache_meta,
        # One block: the game's prediction, result badge and CLV, scoped to its own season.
        "old_rule_scope": DataService.old_rule_scope([game["season"]] if game else []),
    }
    template_response = templates.TemplateResponse(
        request, "pages/game_detail.html", context
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response
