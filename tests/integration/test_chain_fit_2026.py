"""A 2026 week is no longer refused for season coverage (Plan 33-17 Task 2; COLD-07, D33-21).

WHAT THIS ASSERTS, AND WHERE THE REFUSAL ACTUALLY LIVED
--------------------------------------------------------
A CORRECTION, recorded here rather than quietly absorbed. The plan states that
``build_weekly_candidates(2026, W)`` raises ``FrozenChainFitError`` because the frozen fit covers
2021-2025. Read against the source, that is not where the refusal lives:
``build_weekly_candidates`` scores the deployed artifacts and joins the odds snapshot, and it never
consults the fit at all. The season-coverage gate is ``_require_season_covered``, reached from
``build_weekly_decision_frame`` (before any candidate is built) and from
``generate_weekly_bet_list``. So the refusal fired one level ABOVE the function the plan named --
which is why a 2026 call failed before it ever opened a gold matrix.

Both are asserted below: the gate itself now passes for 2026, and the decision-frame entry point
gets PAST it. Recording the correction is the point; a test quietly re-scoped to whatever happened
to be true would leave the next reader with the plan's version of the fact.

WHY IT IS DRIVEN THROUGH INJECTED TEMPORARY STORES (Codex MEDIUM, folded into the plan)
----------------------------------------------------------------------------------------
``gold_dir``, ``silver_dir`` and ``artifacts_dir`` are all keyword parameters, and every one of
them is pointed at a ``tmp_path`` tree. The module therefore reads and writes NO production store:
it cannot trip Plan 33-01's armed write guard, and its result cannot depend on whether Plan 33-18's
acceptance run has happened yet. Season-coverage behaviour is a property of the LOADER and the
OVERLAY, not of the production lake's current contents.

HOW FAR THE CALL IS DRIVEN, AND WHY THAT IS THE ASSERTION
-----------------------------------------------------------
The tmp tree carries a real 2026 week-2 schedule and deliberately NO odds snapshot. So a call that
passes the season gate proceeds into candidate construction and stops on a named
``FileNotFoundError`` for ``odds_snapshot.parquet``. That exact failure pins how far execution got:
past the fit load, past the season gate, past the schedule read. A ``FrozenChainFitError`` instead
would mean the 2026 refusal survives, and it is reported as such rather than swallowed by a broad
``pytest.raises``.

THE CONTROL MATTERS AS MUCH AS THE CLAIM. A gate that stopped refusing EVERYTHING would satisfy
"2026 is no longer refused" perfectly. So 2027 -- one season past the frozen overlay -- must still
refuse by name.

Run this module:  uv run pytest tests/integration/test_chain_fit_2026.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtest.weekly_bet_list import (
    FrozenChainFitError,
    _require_season_covered,
    build_weekly_candidates,
    build_weekly_decision_frame,
    frozen_overlay_season,
    load_frozen_chain_fit,
)

SEASON = 2026
WEEK = 2
UNCOVERED_SEASON = 2027


@pytest.fixture
def injected_stores(tmp_path: Path) -> dict[str, Path]:
    """A tmp tree carrying a real 2026 week-2 schedule and nothing else.

    The odds snapshot is deliberately ABSENT: its named FileNotFoundError is the marker proving
    the season gate was passed. Nothing under ``data/``, ``outputs/`` or ``artifacts/`` is read.
    """
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    artifacts = tmp_path / "artifacts"
    for directory in (silver, gold, artifacts):
        directory.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "game_id": f"{SEASON}_W02_DAL@PHI",
                "season": SEASON,
                "week": WEEK,
                "home_team": "PHI",
                "away_team": "DAL",
                "kickoff_et": pd.Timestamp("2026-09-13 20:20:00", tz="UTC"),
            },
            {
                "game_id": f"{SEASON}_W02_KC@BUF",
                "season": SEASON,
                "week": WEEK,
                "home_team": "BUF",
                "away_team": "KC",
                "kickoff_et": pd.Timestamp("2026-09-13 17:00:00", tz="UTC"),
            },
        ]
    ).to_parquet(silver / "games.parquet")

    return {"silver_dir": silver, "gold_dir": gold, "artifacts_dir": artifacts}


# ---------------------------------------------------------------------------
# 1. The gate itself, on the REAL production run record
# ---------------------------------------------------------------------------


def test_the_production_run_record_now_covers_2026_for_every_target() -> None:
    """The overlay reaches the loader's default path, which is what every caller uses.

    READ ONLY, and the one place this module touches a real artifact -- the season-coverage claim
    is about the record operators actually have, so asserting it on a fixture would not be the
    claim.
    """
    fits = load_frozen_chain_fit()

    assert sorted(fits) == ["ats", "ou", "wp"], (
        "the loader is TARGET-keyed; an int() over these outer keys would raise"
    )
    for target, fit in fits.items():
        assert SEASON in fit.season_bias_by_season, target
        assert set(range(2021, 2026)) <= set(fit.season_bias_by_season), target


def test_the_season_gate_no_longer_refuses_2026() -> None:
    """THE criterion, at the site the refusal actually lives."""
    _require_season_covered(load_frozen_chain_fit(), SEASON)


def test_the_season_gate_still_refuses_the_season_after_the_overlay() -> None:
    """The control. "2026 passes" must not mean "the gate was removed"."""
    with pytest.raises(FrozenChainFitError) as excinfo:
        _require_season_covered(load_frozen_chain_fit(), UNCOVERED_SEASON)
    assert str(UNCOVERED_SEASON) in str(excinfo.value)


def test_the_overlay_season_is_the_season_under_test() -> None:
    """Binds this module's 2026 to the DERIVED overlay season rather than to a literal."""
    assert frozen_overlay_season() == SEASON


# ---------------------------------------------------------------------------
# 2. The entry point gets past the gate, driven on a real 2026 week
# ---------------------------------------------------------------------------


def test_the_decision_frame_gets_past_the_season_gate_for_2026(
    injected_stores: dict[str, Path],
) -> None:
    """``build_weekly_decision_frame`` runs the gate BEFORE building any candidate.

    So a 2026 call that reaches the absent odds snapshot has demonstrably passed it. The
    ``FrozenChainFitError`` arm is caught and re-reported rather than allowed to surface as a bare
    error, so a regression here reads as the finding it is.
    """
    try:
        build_weekly_decision_frame(SEASON, WEEK, **injected_stores)
    except FrozenChainFitError as exc:  # pragma: no cover -- the regression arm
        pytest.fail(f"the 2026 season-coverage refusal survives: {exc}")
    except FileNotFoundError as exc:
        assert "odds_snapshot.parquet" in str(exc), exc
    else:  # pragma: no cover -- the injected tree carries no odds, so this cannot complete
        pytest.fail("the injected stores somehow produced a full decision frame")


def test_the_decision_frame_still_refuses_the_season_after_the_overlay(
    injected_stores: dict[str, Path],
) -> None:
    """The matching control at the entry point, not only at the bare gate."""
    with pytest.raises(FrozenChainFitError):
        build_weekly_decision_frame(UNCOVERED_SEASON, WEEK, **injected_stores)


def test_the_candidate_builder_never_consulted_the_fit_in_the_first_place(
    injected_stores: dict[str, Path],
) -> None:
    """The recorded correction, asserted rather than described.

    ``build_weekly_candidates`` reads the schedule, the odds snapshot and the gold matrices. It
    takes no ``chain_fit_path`` and raises no ``FrozenChainFitError``, for 2026 or for any other
    season -- which is why the refusal the plan attributed to it was really its caller's.
    """
    for season in (SEASON, UNCOVERED_SEASON):
        with pytest.raises(FileNotFoundError) as excinfo:
            build_weekly_candidates(season, WEEK, **injected_stores)
        assert not isinstance(excinfo.value, FrozenChainFitError)


# ---------------------------------------------------------------------------
# 3. The boundary: no production store was read or written
# ---------------------------------------------------------------------------


def test_the_injected_tree_is_the_only_lake_this_module_writes(
    injected_stores: dict[str, Path],
) -> None:
    """The tmp tree is where everything lands, so Plan 33-01's armed write guard is untouched.

    Asserted on the fixture itself: all three injected roots are under ``tmp_path`` and none of
    them resolves to a repository store.
    """
    for name, directory in injected_stores.items():
        resolved = directory.resolve()
        assert resolved.is_dir(), name
        for guarded in ("data", "outputs", "artifacts"):
            assert not resolved.is_relative_to(Path.cwd() / guarded), (name, resolved)
