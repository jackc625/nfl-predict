"""The live 2026 prediction set, recomputed from the snapshot table (COLD-01, SPEC R14).

WHAT THIS MODULE PROVES
-----------------------
Phase 33 exists because a GREEN RUN once served imputed Elo. A zero exit code is therefore not
the evidence; the numbers are. This module follows one live week from
``data/silver/elo_game_snapshots.parquet`` through the three gold matrices to the published
prediction files and the stored forward bet rows, and asserts VALUES:

* every Elo cell gold carries for a 2026 game of week ``LIVE_ACCEPTANCE_WEEK`` onward equals,
  within ``LIVE_ACCEPTANCE_TOLERANCE``, the value an INDEPENDENT recomputation derives from the
  snapshot table alone;
* every 2026 forward bet row was decided at or before its own game's lock, and its
  ``freeze_ts`` IS that lock;
* every 2026 prediction row carries its game's lock as ``information_cutoff_utc`` and was
  captured and computed at or before it.

WHY THE COMPARISON IS AGAINST A RECOMPUTED Z-SCORE, NEVER THE RAW RATING
-------------------------------------------------------------------------
Gold's Elo columns are Z-SCORES. ``scripts/build_features.py`` first winsorizes every numeric
column per season on bounds fitted over STRICTLY EARLIER seasons
(``handle_missing_data_and_outliers`` with ``_season_fit_source``; the 1st/99th percentiles,
``outlier_percentiles = (1, 99)``; no clip when the prior slice holds ``_MIN_FIT_POINTS = 10``
or fewer values, or when the bounds are degenerate), then z-scores it with
``features.normalization.expanding_normalize`` called with ``row_locks`` and ``min_periods=4``:
a row's mean and standard deviation are taken over every row of its season whose LOCK is at or
before its own (p332_ extra step 8c). A raw equality between gold ``home_elo`` (a small fraction of one)
and snapshot ``home_elo_pre`` (about 1500) can never hold, and neither can
``elo_diff == home_elo - away_elo`` on gold values, because each column is z-scored with its
own statistic. So the two derived columns are formed BY FORMULA on the RAW joined values
(``elo_diff`` from ``home_elo_pre`` minus ``away_elo_pre``, ``elo_prob_away`` from one minus
``elo_prob_home``) and then put through the same transform as every other column.

THE RECOMPUTATION IS INDEPENDENT OF THE BUILD. It imports neither ``scripts.build_features``
nor ``features.normalization`` -- a build that checked itself would prove nothing. It is plain
pandas and ``math.fsum``: exactly-rounded sums make every window statistic independent of row
order, which is what lets the order-invariance test demand IDENTICAL cells. A row whose window
holds fewer than four values, or whose standard deviation is below 1e-8, would take the build's
prior-season bootstrap, which this helper deliberately does not reimplement; such a row is OUT
OF SCOPE and is reported BY NAME, never silently dropped.

WHY NO ZERO SENTINEL
--------------------
A z-score of exactly zero is a real reading: the game's value equals its window mean. A
comparison that treated zero as "missing" would pass a blanked cell and fail a legitimate one,
so absence is only ever read as NaN, and a constructed exact-zero z-score is shown to pass.

WHY RANK, PERCENTILE AND MOMENTUM ARE PRESENCE-ONLY
---------------------------------------------------
Those six columns are not joined per game: ``features/elo_features.py`` derives them from the
week's snapshot POPULATION (``_add_rank_features`` ranks every team's latest pre-game rating at
``week <= W``; ``_add_momentum_features`` reads a team's last four prior-week ratings). Their
value depends on WHEN the build ran -- a provisional row is replaced in place once results land
-- which is the reproducibility hazard ``_add_rank_features``' docstring hands to Phase 34. They
are asserted present and non-null here, not recomputed.

WHY WEEK >= LIVE_ACCEPTANCE_WEEK AND NOT ONLY THE RECORDED WEEK
---------------------------------------------------------------
``LIVE_ACCEPTANCE_CHECKED_GAME_IDS`` is the checked generation's fact: the fifteen week-3 rows
of the gold the 2026-09-26 run built (``LIVE_ACCEPTANCE_GOLD_BUILD_CLOCK_UTC``). The value check
covers every 2026 gold row from that week on, so it keeps holding -- and keeps checking --
after each nightly rebuild adds a week. Weeks 1 and 2 feed every later window, so an imputed
early value would surface in the week-3 cells.

Parquet is read DIRECTLY with pandas; nothing here goes through ``data.storage``, which opens
the DuckDB store first. ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.cache import PROVENANCE_FORWARD
from tests.phase33_state import (
    LIVE_ACCEPTANCE_CHECKED_GAME_IDS,
    LIVE_ACCEPTANCE_DERIVED_ELO_COLUMNS,
    LIVE_ACCEPTANCE_ELO_COLUMN_MAP,
    LIVE_ACCEPTANCE_PRESENCE_ONLY_ELO_COLUMNS,
    LIVE_ACCEPTANCE_SEASON,
    LIVE_ACCEPTANCE_TOLERANCE,
    LIVE_ACCEPTANCE_WEEK,
)
from utils.game_lock import is_admissible, lock_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRICES: tuple[str, ...] = ("wp", "ats", "ou")
GOLD_PATHS: dict[str, Path] = {
    matrix: REPO_ROOT / "data" / "gold" / f"features_{matrix}.parquet"
    for matrix in MATRICES
}
SNAPSHOT_PATH = REPO_ROOT / "data" / "silver" / "elo_game_snapshots.parquet"
GAMES_PATH = REPO_ROOT / "data" / "silver" / "games.parquet"
BET_LIST_PATH = REPO_ROOT / "outputs" / "bet_list" / "bet_list.parquet"
PREDICTIONS_DIR = REPO_ROOT / "outputs" / "predictions"

NO_LIVE_2026_GOLD_SKIP = (
    "the live 2026 prediction-set check cannot run on this checkout: a gold matrix, the "
    "silver Elo snapshot table or silver games is absent, or gold carries no row of the live "
    "season. This control did NOT run here; it did not pass."
)

# The build's documented transform, restated as the numbers it uses (see the module docstring
# for the source of each). Restated, never imported: the recomputation must not share code
# with the build it checks.
_CLIP_QUANTILES: tuple[float, float] = (1e-2, 99e-2)
_MIN_FIT_POINTS = 10
_MIN_WINDOW = 4
_DEGENERATE_STD = 1e-8

#: The eight gold columns the value check covers: the six joined, then the two derived.
CHECKED_GOLD_COLUMNS: tuple[str, ...] = tuple(
    gold for _snapshot, gold in LIVE_ACCEPTANCE_ELO_COLUMN_MAP
) + tuple(LIVE_ACCEPTANCE_DERIVED_ELO_COLUMNS)


# ---------------------------------------------------------------------------
# The skip guard: one resolver, one pinned message.
# ---------------------------------------------------------------------------


def _require_live_gold() -> None:
    """Step aside BY NAME when this checkout cannot run the check.

    Raises:
        Skipped: when a gold matrix, the snapshot table or silver games is absent, or gold
            carries no row of ``LIVE_ACCEPTANCE_SEASON``.
    """
    required = (*GOLD_PATHS.values(), SNAPSHOT_PATH, GAMES_PATH)
    if not all(path.is_file() for path in required):
        pytest.skip(NO_LIVE_2026_GOLD_SKIP)
    for path in GOLD_PATHS.values():
        seasons = pd.read_parquet(path, columns=["season"])["season"]
        if not bool((seasons == LIVE_ACCEPTANCE_SEASON).any()):
            pytest.skip(NO_LIVE_2026_GOLD_SKIP)


# ---------------------------------------------------------------------------
# The independent recomputation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recomputation:
    """The target season's recomputed z-scores and the rows no window could score.

    Attributes:
        values: ``game_id``-indexed frame, one column per ``CHECKED_GOLD_COLUMNS`` entry; NaN
            where the row is out of scope for that column.
        out_of_scope: column -> the ``game_id``s whose window was too small or degenerate.
    """

    values: pd.DataFrame
    out_of_scope: dict[str, tuple[str, ...]]


def _snapshot_side(snapshots: pd.DataFrame) -> pd.DataFrame:
    """The snapshot columns under their GOLD names, selected and renamed BEFORE any merge."""
    rename = dict(LIVE_ACCEPTANCE_ELO_COLUMN_MAP)
    return snapshots[["game_id", *rename]].rename(columns=rename)


def _window_z_score(value: float, window: np.ndarray) -> float:
    """``value`` against its window's mean and ddof=1 standard deviation (exactly rounded)."""
    count = len(window)
    mean = math.fsum(window) / count
    variance = math.fsum((window - mean) ** 2) / (count - 1)
    return (value - mean) / math.sqrt(variance)


def recompute_live_elo(
    gold: pd.DataFrame,
    snapshots: pd.DataFrame,
    locks: pd.Series,
    season: int,
) -> Recomputation:
    """Recompute *season*'s gold Elo z-scores from the snapshot table alone.

    Args:
        gold: A gold matrix (every season: the clip bounds are fitted on earlier seasons).
        snapshots: The Elo snapshot table.
        locks: Each *season* game's lock, ``game_id``-indexed (``utils.game_lock.lock_frame``).
        season: The season to recompute.

    Returns:
        The recomputed z-scores and the out-of-scope rows, by column.
    """
    joined = gold[["game_id", "season"]].merge(
        _snapshot_side(snapshots), on="game_id", how="left", validate="one_to_one"
    )
    joined["elo_diff"] = joined["home_elo"] - joined["away_elo"]
    joined["elo_prob_away"] = 1.0 - joined["elo_prob_home"]

    target = joined.loc[joined["season"] == season]
    prior = joined.loc[joined["season"] < season]
    game_ids = target["game_id"].astype(str).to_numpy()
    row_locks = pd.DatetimeIndex(locks.reindex(game_ids)).asi8

    values: dict[str, np.ndarray] = {}
    out_of_scope: dict[str, tuple[str, ...]] = {}
    for column in CHECKED_GOLD_COLUMNS:
        column_values = target[column].astype(float)
        fit = prior[column]
        if fit.notna().sum() > _MIN_FIT_POINTS:
            lower = fit.quantile(_CLIP_QUANTILES[0])
            upper = fit.quantile(_CLIP_QUANTILES[1])
            if lower < upper:
                column_values = column_values.clip(lower=lower, upper=upper)
        raw = column_values.to_numpy()

        scored = np.full(len(raw), np.nan)
        unscorable: list[str] = []
        for position, lock in enumerate(row_locks):
            window = raw[row_locks <= lock]
            window = window[~np.isnan(window)]
            if len(window) < _MIN_WINDOW or np.std(window, ddof=1) < _DEGENERATE_STD:
                unscorable.append(str(game_ids[position]))
                continue
            scored[position] = _window_z_score(raw[position], window)
        values[column] = scored
        out_of_scope[column] = tuple(sorted(unscorable))

    frame = pd.DataFrame(values, index=pd.Index(game_ids, name="game_id"))
    return Recomputation(values=frame, out_of_scope=out_of_scope)


def worst_deviation(
    gold: pd.DataFrame, recomputed: Recomputation, column: str, game_ids: list[str]
) -> tuple[float, str]:
    """The largest ``abs(gold - recomputed)`` over *game_ids* for *column*, and its game.

    A NaN on either side of an in-scope cell is reported as an infinite deviation rather
    than skipped: an absent cell is exactly what this comparison exists to catch.
    """
    gold_values = gold.set_index("game_id").loc[game_ids, column].astype(float)
    recomputed_values = recomputed.values.loc[game_ids, column]
    deviation = (gold_values - recomputed_values).abs().fillna(np.inf)
    worst_game = str(deviation.idxmax())
    return float(deviation.max()), worst_game


# ---------------------------------------------------------------------------
# Fixtures: read once per module, only after the skip guard.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gold() -> dict[str, pd.DataFrame]:
    _require_live_gold()
    return {matrix: pd.read_parquet(path) for matrix, path in GOLD_PATHS.items()}


@pytest.fixture(scope="module")
def snapshots() -> pd.DataFrame:
    _require_live_gold()
    return pd.read_parquet(SNAPSHOT_PATH)


@pytest.fixture(scope="module")
def season_locks() -> pd.Series:
    _require_live_gold()
    games = pd.read_parquet(GAMES_PATH, columns=["game_id", "season", "kickoff_et"])
    season_games = games.loc[games["season"] == LIVE_ACCEPTANCE_SEASON]
    return lock_frame(season_games[["game_id", "kickoff_et"]])


@pytest.fixture(scope="module")
def recomputed(
    gold: dict[str, pd.DataFrame], snapshots: pd.DataFrame, season_locks: pd.Series
) -> dict[str, Recomputation]:
    return {
        matrix: recompute_live_elo(
            frame, snapshots, season_locks, LIVE_ACCEPTANCE_SEASON
        )
        for matrix, frame in gold.items()
    }


def _checked_rows(frame: pd.DataFrame) -> list[str]:
    """Every live-season gold row from the recorded week on."""
    mask = (frame["season"] == LIVE_ACCEPTANCE_SEASON) & (
        frame["week"] >= LIVE_ACCEPTANCE_WEEK
    )
    return frame.loc[mask, "game_id"].astype(str).tolist()


# ---------------------------------------------------------------------------
# 1. The map, 2. the count before any value, 3. the recorded games are in scope.
# ---------------------------------------------------------------------------


def test_the_column_map_is_six_explicit_pairs(
    gold: dict[str, pd.DataFrame], snapshots: pd.DataFrame
) -> None:
    assert len(LIVE_ACCEPTANCE_ELO_COLUMN_MAP) == 6
    snapshot_side = [snapshot for snapshot, _gold in LIVE_ACCEPTANCE_ELO_COLUMN_MAP]
    gold_side = [gold_name for _snapshot, gold_name in LIVE_ACCEPTANCE_ELO_COLUMN_MAP]
    assert len(set(snapshot_side)) == 6 and len(set(gold_side)) == 6
    assert set(snapshot_side) <= set(snapshots.columns), sorted(
        set(snapshot_side) - set(snapshots.columns)
    )
    for matrix, frame in gold.items():
        expected = (
            set(gold_side)
            | set(LIVE_ACCEPTANCE_DERIVED_ELO_COLUMNS)
            | set(LIVE_ACCEPTANCE_PRESENCE_ONLY_ELO_COLUMNS)
        )
        missing = sorted(expected - set(frame.columns))
        assert not missing, f"gold {matrix} lacks Elo columns {missing}"


@pytest.mark.parametrize("matrix", MATRICES)
def test_every_live_gold_row_joins_exactly_one_snapshot_row(
    gold: dict[str, pd.DataFrame], snapshots: pd.DataFrame, matrix: str
) -> None:
    """COUNT BEFORE VALUES: a join that matched nothing fails here, not in a value check."""
    live = gold[matrix].loc[gold[matrix]["season"] == LIVE_ACCEPTANCE_SEASON]
    assert len(live) > 0
    per_game = snapshots["game_id"].value_counts()
    counts = live["game_id"].map(per_game)
    not_one = sorted(live["game_id"][counts.ne(1)])
    assert not not_one, (
        f"{matrix}: live gold rows without exactly one snapshot row: {not_one}"
    )
    joined = live[["game_id"]].merge(
        _snapshot_side(snapshots), on="game_id", how="inner", validate="one_to_one"
    )
    assert len(joined) == len(live)


@pytest.mark.parametrize("matrix", MATRICES)
def test_every_recorded_game_is_in_gold_and_in_scope(
    gold: dict[str, pd.DataFrame], recomputed: dict[str, Recomputation], matrix: str
) -> None:
    checked = set(LIVE_ACCEPTANCE_CHECKED_GAME_IDS)
    assert len(checked) == len(LIVE_ACCEPTANCE_CHECKED_GAME_IDS) > 0
    absent = sorted(checked - set(gold[matrix]["game_id"].astype(str)))
    assert not absent, f"{matrix}: recorded games missing from gold: {absent}"
    for column, unscorable in recomputed[matrix].out_of_scope.items():
        overlap = sorted(checked & set(unscorable))
        assert not overlap, f"{matrix}.{column}: recorded games out of scope: {overlap}"


# ---------------------------------------------------------------------------
# 4. Value by value: 3 matrices x 8 columns.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("matrix", "column"),
    [(matrix, column) for matrix in MATRICES for column in CHECKED_GOLD_COLUMNS],
    ids=[
        f"{matrix}-{column}" for matrix in MATRICES for column in CHECKED_GOLD_COLUMNS
    ],
)
def test_gold_equals_the_recomputed_value(
    gold: dict[str, pd.DataFrame],
    recomputed: dict[str, Recomputation],
    matrix: str,
    column: str,
) -> None:
    out_of_scope = set(recomputed[matrix].out_of_scope[column])
    rows = [g for g in _checked_rows(gold[matrix]) if g not in out_of_scope]
    assert rows, f"{matrix}.{column}: no in-scope live row to compare"
    worst, game = worst_deviation(gold[matrix], recomputed[matrix], column, rows)
    assert worst <= LIVE_ACCEPTANCE_TOLERANCE, (
        f"{matrix}.{column}: gold differs from the snapshot-derived value by {worst!r} at "
        f"{game} (tolerance {LIVE_ACCEPTANCE_TOLERANCE!r})"
    )


# ---------------------------------------------------------------------------
# 5. The population-derived columns: presence only.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix", MATRICES)
def test_rank_percentile_and_momentum_are_present_and_non_null(
    gold: dict[str, pd.DataFrame], matrix: str
) -> None:
    frame = gold[matrix].set_index("game_id")
    rows = _checked_rows(gold[matrix])
    assert rows
    cells = frame.loc[rows, list(LIVE_ACCEPTANCE_PRESENCE_ONLY_ELO_COLUMNS)]
    nulls = {c: sorted(cells.index[cells[c].isna()]) for c in cells.columns}
    assert not any(nulls.values()), f"{matrix}: null population-derived cells {nulls}"


# ---------------------------------------------------------------------------
# 6. Order invariance, 7. non-vacuity, 8. adjacency.
# ---------------------------------------------------------------------------


def test_the_comparison_is_invariant_to_the_row_order_of_both_frames(
    gold: dict[str, pd.DataFrame],
    snapshots: pd.DataFrame,
    season_locks: pd.Series,
    recomputed: dict[str, Recomputation],
) -> None:
    shuffled_gold = gold["wp"].sample(frac=1.0, random_state=20260928)
    shuffled_snapshots = snapshots.sample(frac=1.0, random_state=20260926)
    again = recompute_live_elo(
        shuffled_gold, shuffled_snapshots, season_locks, LIVE_ACCEPTANCE_SEASON
    )
    pd.testing.assert_frame_equal(
        again.values.sort_index(),
        recomputed["wp"].values.sort_index(),
        check_exact=True,
    )
    assert again.out_of_scope == recomputed["wp"].out_of_scope
    rows = _checked_rows(gold["wp"])
    for column in CHECKED_GOLD_COLUMNS:
        in_scope = [g for g in rows if g not in set(again.out_of_scope[column])]
        assert worst_deviation(
            shuffled_gold, again, column, in_scope
        ) == worst_deviation(gold["wp"], recomputed["wp"], column, in_scope)


def test_a_planted_one_point_elo_shift_fails_the_comparison(
    gold: dict[str, pd.DataFrame], snapshots: pd.DataFrame, season_locks: pd.Series
) -> None:
    planted_game = sorted(LIVE_ACCEPTANCE_CHECKED_GAME_IDS)[0]
    planted = snapshots.copy()
    planted.loc[planted["game_id"] == planted_game, "home_elo_pre"] += 1.0
    assert int((planted["home_elo_pre"] != snapshots["home_elo_pre"]).sum()) == 1
    shifted = recompute_live_elo(
        gold["wp"], planted, season_locks, LIVE_ACCEPTANCE_SEASON
    )
    worst, _game = worst_deviation(gold["wp"], shifted, "home_elo", [planted_game])
    assert worst > LIVE_ACCEPTANCE_TOLERANCE, (
        f"a +1 Elo shift on {planted_game} moved its recomputed z-score by only {worst!r}: "
        "the comparison cannot fail"
    )


def test_a_legitimately_zero_z_score_passes() -> None:
    """A value equal to its window mean scores exactly zero, and that is a pass, not absence."""
    season = 2030
    prior = pd.DataFrame(
        {"game_id": [f"P{i}" for i in range(3)], "season": [season - 1] * 3}
    )
    target_ids = [f"T{i}" for i in range(5)]
    # Integers on purpose: the window mean of these five is 1500 exactly.
    home = [1490, 1500, 1510, 1495, 1505]
    gold_frame = pd.concat(
        [prior, pd.DataFrame({"game_id": target_ids, "season": [season] * 5})],
        ignore_index=True,
    )
    snapshot_frame = pd.DataFrame(
        {
            "game_id": gold_frame["game_id"],
            "home_elo_pre": [1500, 1501, 1502, *home],
            "away_elo_pre": [1480, 1481, 1482, 1470, 1475, 1480, 1485, 1490],
            "home_elo_uncertainty": [90, 91, 92, 60, 61, 62, 63, 64],
            "away_elo_uncertainty": [80, 81, 82, 50, 52, 54, 56, 58],
            "elo_prob_home": [0.5, 0.51, 0.52, 0.55, 0.6, 0.65, 0.7, 0.75],
            "hfa_used": [30, 30, 30, 20, 25, 30, 35, 40],
        }
    )
    one_lock = pd.Timestamp("2030-09-14T22:00:00+00:00")
    locks = pd.Series([one_lock] * 5, index=pd.Index(target_ids, name="game_id"))

    result = recompute_live_elo(gold_frame, snapshot_frame, locks, season)
    zero_row = target_ids[home.index(1500)]
    assert result.values.loc[zero_row, "home_elo"] == 0
    assert not result.out_of_scope["home_elo"]

    expected = [(v - 1500) / float(np.std(home, ddof=1)) for v in home]
    gold_frame["home_elo"] = [float("nan")] * 3 + expected
    worst, game = worst_deviation(gold_frame, result, "home_elo", target_ids)
    assert worst <= LIVE_ACCEPTANCE_TOLERANCE, (worst, game)


# ---------------------------------------------------------------------------
# 9. Forward bet rows, 10. prediction rows: stamped at or before each game's own lock.
# ---------------------------------------------------------------------------


def test_every_live_forward_row_was_decided_at_or_before_its_lock(
    season_locks: pd.Series,
) -> None:
    bets = pd.read_parquet(BET_LIST_PATH)
    forward = bets.loc[
        (bets["provenance"] == PROVENANCE_FORWARD)
        & (bets["season"] == LIVE_ACCEPTANCE_SEASON)
    ]
    assert len(forward) > 0, "no live forward row to check"
    nulls = forward.loc[forward["decided_at_utc"].isna(), ["game_id", "target"]]
    assert nulls.empty, f"forward rows with no decided_at_utc: {nulls.values.tolist()}"
    late = [
        (game, target)
        for game, target, decided in zip(
            forward["game_id"],
            forward["target"],
            forward["decided_at_utc"],
            strict=True,
        )
        if not is_admissible(decided, season_locks[game])
    ]
    assert not late, f"forward rows decided after their lock: {late}"
    wrong_freeze = sorted(
        {
            game
            for game, freeze in zip(
                forward["game_id"], forward["freeze_ts"], strict=True
            )
            if pd.Timestamp(freeze) != season_locks[game]
        }
    )
    assert not wrong_freeze, (
        f"forward rows whose freeze_ts is not the lock: {wrong_freeze}"
    )


def test_every_live_prediction_row_is_stamped_against_its_lock(
    season_locks: pd.Series,
) -> None:
    files = sorted(
        PREDICTIONS_DIR.glob(f"predictions_{LIVE_ACCEPTANCE_SEASON}_week*.csv")
    )
    assert files, "no live prediction file to check"
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    assert len(frame) > 0
    offenders: list[tuple[str, str]] = []
    for row in frame.itertuples(index=False):
        lock = season_locks[row.game_id]
        if pd.Timestamp(row.information_cutoff_utc) != lock:
            offenders.append((row.game_id, "information_cutoff_utc"))
        if not is_admissible(row.captured_at_utc, lock):
            offenders.append((row.game_id, "captured_at_utc"))
        if not is_admissible(row.computed_at_utc, lock):
            offenders.append((row.game_id, "computed_at_utc"))
    assert not offenders, f"prediction stamps not at or before the lock: {offenders}"


# ---------------------------------------------------------------------------
# 11. The skip guard fails closed.
# ---------------------------------------------------------------------------


def test_the_absent_store_skip_guard_actually_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guard only ever observed NOT firing is indistinguishable from an unwired one."""
    monkeypatch.setattr(
        "tests.integration.test_live_2026_prediction_set.SNAPSHOT_PATH",
        REPO_ROOT / "data" / "silver" / "snapshot_table_that_does_not_exist.parquet",
    )
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _require_live_gold()
    assert str(excinfo.value) == NO_LIVE_2026_GOLD_SKIP
