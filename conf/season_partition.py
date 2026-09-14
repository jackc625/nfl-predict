"""THE season partition rule: one deterministic rule, with its evidence beside it.

ONE-WAY BY CONSTRUCTION, in the shape of ``backtest/ev_chain_constants.py`` and
``backtest/group_gate_constants.py``. This module IS the rule (SPEC R6, D33.1-03), not a
description of one. Its LAST-MODIFYING COMMIT is the git-ancestry anchor:
``tests/phase33_state.SEASON_PARTITION_RULE_COMMIT`` records it in a STRICTLY LATER commit,
and ``tests/unit/test_preregistration_ancestry.py::TestThePhase331SeasonPartitionRule``
recomputes and compares.

Stated plainly because it is easy to forget several plans later: EDITING THIS FILE AFTER A
WINDOW HAS BEEN SCORED DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. A value that is wrong
here is wrong for the remainder of the phase, and the only legitimate response is to say so in
the readout, not to amend the file.

THE PROHIBITION IS SATISFIED VACUOUSLY, AND THE ANCESTRY CHECK EXISTS ANYWAY
----------------------------------------------------------------------------
The SPEC's third prohibition reads: "MUST NOT select the training window after observing its
scores -- the rule's commit is a strict git ancestor of any run that scores a candidate
window." Under D33.1-03 this rule is DETERMINISTIC and SCORES NOTHING: no candidate window is
ever fitted, ranked or compared, so there is no observation to select after. The prohibition is
therefore satisfied vacuously rather than by discipline. The ancestry check is written anyway,
so that the property stays CHECKABLE if a future phase ever does score a window.

THIS FILE RECORDS NO HASH OF ITSELF
-----------------------------------
A file that must CONTAIN and exactly REPRODUCE its own whole-file hash is self-referential:
writing the hash changes the bytes the hash was computed over, so no fixed point exists without
a canonical exclusion rule nobody has defined. The witness lives OUTSIDE the witnessed file, in
``tests/phase33_state.py`` under that module's APPEND PROTOCOL.

WHY THIS MODULE LIVES IN ``conf/`` AND NOT IN ``models/temporal.py``
-------------------------------------------------------------------
The obvious home is ``models/temporal.py``, beside ``TemporalSplitConfig``. It cannot go there.
``conf/settings.py`` is one of the consumers, and ``models/__init__.py`` pulls in the whole
model stack -- so a rule living under ``models/`` would make importing ``conf.settings`` import
pandas, sklearn, xgboost and every trainer, and would create an import cycle the moment
anything under ``models/`` read a setting. ``conf/`` imports nothing from this project, which is
exactly what lets ``models/``, ``backtest/``, ``scripts/``, ``features/`` and ``conf.settings``
itself all read the same rule with no cycle. THIS MODULE MUST KEEP THAT PROPERTY: it imports
only the standard library, and a test asserts that importing it drags in no project package.

THE RULE, IN WORDS
------------------
Given the completed seasons:

* the HOLDOUT is the ``HOLDOUT_SEASON_COUNT`` most recent completed seasons;
* the HP_VAL fold is the ``HP_VAL_SEASON_COUNT`` seasons immediately before those;
* the SELECTION window runs from ``SELECTION_WINDOW_FIRST_SEASON`` through the season before
  hp_val;
* the FINAL FIT covers every completed season from ``CORPUS_FIRST_SEASON`` (D33.1-02).

On the seasons 2002 through 2025 that yields selection 2018-2022, hp_val 2023, holdout
2024-2025 and a final fit over 2002-2025. The three evaluated sets are pairwise disjoint, each
non-empty and strictly ordered, so ``models.temporal.TemporalSplitConfig.validate()`` accepts
them -- and 2025 is present, which is the defect SPEC R6 exists to remove.

THE EVIDENCE FOR THE SELECTION WINDOW'S 2018 BOUNDARY (Ruling Q)
----------------------------------------------------------------
R6 requires the rule AND its evidence to be readable in committed source. This is where
"readable" means read. The choice rests on a STATED PRIOR, not on a measured result: nothing
was scored to pick it, and both of the alternatives below are defensible.

**Why 534 rows is too small -- MEASURED, and already committed.**
``GATED-REFIT-READOUT.md:401-427`` records that feature selection on the 534-row 2018-2019
training window admits synthetic noise columns over real ones. The production selection path
was run with N synthetic unit-variance Gaussian columns appended, drawn independently of the
target. At N=50, SIX of ATS's 25 and NINE of O/U's 25 selected features were pure noise. The
same readout names window widening as the explicitly untried lever: "Widening the window,
changing K and changing the estimator were all explicitly out of scope."

**Why 2002 is too generous -- MEASURED this phase.**
NINETY gold columns are a flat imputed constant for every season from 2002 to 2017, because
their silver sources begin late:

* ``data/silver/elo_game_snapshots.parquet`` covers 2018-2025;
* ``data/silver/odds_snapshot.parquet`` covers 2018-2025;
* ``data/silver/team_game_stats.parquet`` covers 2020-2025.

The set includes every Elo column, every rolling opponent-adjusted EPA / success-rate / CPOE /
pace / red-zone / third-down column, all three market snapshot columns and the situational
spots. Deployed exposure: SEVEN of WP's twenty selected features, SEVENTEEN of ATS's
twenty-five and SEVENTEEN of O/U's twenty-five sit in that set, including O/U's single
highest-importance feature. A 2002-start selection window trades a 534-row frame for a
5,679-row frame in which 90 candidates are placeholders over 16 of 21 seasons -- and
``informative_columns`` will NOT withhold them, because they DO vary across the 2018 boundary.

**Why 2018 is the right boundary and not an arbitrary one.**
It is the measured coverage floor of three silver sources. It is argued from DATA COVERAGE,
never from a story about the game changing: there is no committed evidence for a regime break
anywhere in this repository, so any boundary argued that way would be an unsourced prior.
``p/n`` falls from 0.36 to 0.12, and 45 weather columns are entering the candidate pool for the
first time this phase -- adding live candidates to a selector measured choosing noise at 0.36
makes the small window worse, not merely unchanged.

**The cost, recorded rather than hidden.**
The selection window (2018-2022) is NARROWER than the fit window (2002-2025). That mismatch
already exists today and is not introduced here: the WP scaler is fitted on ``train_seasons``
while the model is fitted on everything before the last holdout season. D33.1-02 widens the fit
window, so the gap widens with it. It is named in the readout rather than presented as new.

WHAT THIS RULE DELIBERATELY DOES NOT DECIDE
-------------------------------------------
It does not decide whether a gate verdict computed on this partition is out-of-sample. Under
D33.1-01 the shipped artifact is fitted on the holdout seasons, so a re-score of it on those
seasons is IN-SAMPLE. The owner accepted that cost after it was stated twice; the readout must
label Wave 15's verdict in-sample rather than present it as a clean gate pass. That is a
property of the DECISION, not of this rule, and no edit here can change it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "CORPUS_FIRST_SEASON",
    "HOLDOUT_SEASON_COUNT",
    "HP_VAL_SEASON_COUNT",
    "LATEST_COMPLETED_SEASON",
    "PARTITION_RULE_PATH",
    "RULE_EVIDENCE",
    "SELECTION_WINDOW_FIRST_SEASON",
    "SeasonPartition",
    "backtest_seasons",
    "completed_seasons_from",
    "default_completed_seasons",
    "default_season_partition",
    "derive_season_partition",
]

# ---------------------------------------------------------------------------
# The rule's clauses, each a NAMED constant rather than a literal inside the
# function. A literal inside a function is what eleven separate sites already
# were; naming them is the point of the consolidation, not decoration.
# ---------------------------------------------------------------------------

#: How many of the most recent completed seasons are the walk-forward holdout.
HOLDOUT_SEASON_COUNT: int = 2

#: How many seasons immediately before the holdout form the hp-val fold. It must stay at least
#: one: ``models/temporal.py:88-103`` REFUSES an empty hp_val by name, because all three
#: concrete trainers fit a calibration/conversion component on that fold OUTSIDE the tuning
#: branch -- an empty fold crashes WP inside StandardScaler and gives ATS/OU a NaN-scale
#: converter, i.e. one hard crash and two silently degenerate models.
HP_VAL_SEASON_COUNT: int = 1

#: The first season of the FEATURE-SELECTION window (Ruling Q). Measured: the coverage floor of
#: the Elo, odds and team-stat silver sources. See the module docstring for the full argument
#: and for the two alternatives that were weighed and not taken.
SELECTION_WINDOW_FIRST_SEASON: int = 2018

#: The first season of the CORPUS the shipped artifact is finally fitted on (D33.1-02). The
#: pinned nflverse schedules reach back to 2002 with full coverage of the 6,499-game
#: population; gold carries exactly those seasons.
CORPUS_FIRST_SEASON: int = 2002

# ---------------------------------------------------------------------------
# THE ONE REMAINING LITERAL, AND WHY IT CANNOT BE DERIVED HERE.
#
# Every site that HOLDS DATA derives its completed seasons from that data via
# `completed_seasons_from`. But several consumers need a partition at IMPORT
# time, with no frame in hand -- `conf.settings.BacktestConfig`'s field default,
# `models.temporal.TemporalSplitConfig.default()`, `models/train.py`'s argparse
# defaults, `backtest.engine.BacktestConfig`'s field defaults. Reading
# `data/gold/*.parquet` at import time to answer them would make importing a
# settings module do disk I/O, and would make the partition depend on whether a
# gold rebuild had run -- a partition that changes with the state of a build
# artifact is not a rule.
#
# Deriving it from the calendar was also rejected: the partition would then roll
# forward silently, on a date, with no commit recording that it moved. A rule
# that changes without a commit cannot be witnessed, which is the whole
# mechanism this module rests on.
#
# So this ONE number is a dated literal, and it is the only one. Bumping it when
# a season completes is a deliberate, reviewable, one-line commit -- which is
# what replaced eleven independently-drifting literals.
# ---------------------------------------------------------------------------

#: The most recent COMPLETED NFL season, as of 2026-09-14. The 2026 season is underway and is
#: NOT complete: gold holds zero 2026 rows. Bump this, in its own commit, when a season ends.
LATEST_COMPLETED_SEASON: int = 2025

#: This module's own path, so a consumer or a test names it from one place.
PARTITION_RULE_PATH: str = "conf/season_partition.py"

#: The citations behind the rule, as strings, so a readout can quote them without re-deriving
#: the argument. The prose form is the module docstring; this is the machine-readable index of
#: it. R6 asks for the rule AND its evidence in committed source; both are here.
RULE_EVIDENCE: tuple[str, ...] = (
    "GATED-REFIT-READOUT.md:401-427 -- MEASURED: feature selection on the 534-row "
    "2018-2019 window selected 6 of ATS's 25 and 9 of O/U's 25 features from pure "
    "synthetic noise columns at N=50. The same section names window widening as the "
    "explicitly untried lever.",
    "data/silver/elo_game_snapshots.parquet covers 2018-2025 only -- every Elo column is a "
    "flat imputed constant before 2018.",
    "data/silver/odds_snapshot.parquet covers 2018-2025 only -- all three market snapshot "
    "columns are a flat imputed constant before 2018.",
    "data/silver/team_game_stats.parquet covers 2020-2025 only -- every rolling "
    "opponent-adjusted EPA / success-rate / CPOE / pace / red-zone / third-down column is a "
    "flat imputed constant before 2018.",
    "MEASURED this phase: 90 gold columns are constant over 2002-2017 and vary from 2018 "
    "onward; 7 of WP's 20, 17 of ATS's 25 and 17 of O/U's 25 deployed features are in that "
    "set, including O/U's highest-importance feature (snapshot_total).",
    "THE CHOICE RESTS ON A STATED PRIOR, NOT ON A SCORE. Nothing was fitted, ranked or "
    "compared to pick this window. Both alternatives -- D33.1-03's literal default of every "
    "season before hp_val, and the 534-row status quo -- are defensible; only the 2002 start "
    "rests on a premise the placeholder measurement contradicts.",
    "There is NO committed evidence for a regime break (a rule change, a scoring-environment "
    "shift) anywhere in this repository, so the 2018 boundary is argued from measured DATA "
    "COVERAGE and never from a story about the game changing.",
    "THE RECORDED COST: the selection window is narrower than the fit window. That mismatch "
    "already exists between the WP scaler (fitted on train_seasons) and the model (fitted on "
    "everything before the last holdout season); D33.1-02 widens the fit window, so the gap "
    "widens with it.",
)


@dataclass(frozen=True)
class SeasonPartition:
    """The four season sets the rule produces, as immutable tuples.

    Attributes:
        selection: The feature-selection / initial-fit window (``train_seasons``).
        hp_val: The hyperparameter-validation fold.
        holdout: The walk-forward reporting holdout.
        final_fit: Every completed season from ``CORPUS_FIRST_SEASON`` -- the rows the SHIPPED
            artifact is finally fitted on under D33.1-02. It DELIBERATELY OVERLAPS the other
            three: it is not a fourth fold, it is the union the final fit runs over. Only
            ``selection``, ``hp_val`` and ``holdout`` are the pairwise-disjoint evaluated sets
            ``TemporalSplitConfig.validate()`` checks.
    """

    selection: tuple[int, ...]
    hp_val: tuple[int, ...]
    holdout: tuple[int, ...]
    final_fit: tuple[int, ...]

    @property
    def latest_completed_season(self) -> int:
        """The most recent completed season in this partition."""
        return self.final_fit[-1]

    @property
    def evaluated_seasons(self) -> tuple[int, ...]:
        """The union of the three PAIRWISE-DISJOINT evaluated sets, sorted."""
        return tuple(sorted({*self.selection, *self.hp_val, *self.holdout}))


def completed_seasons_from(gold_seasons: Iterable[object]) -> tuple[int, ...]:
    """Normalize any season-bearing iterable into the completed-season tuple the rule takes.

    This is the derivation for every caller that HOLDS DATA: pass a gold frame's ``season``
    column, a list, a set, anything whose members coerce to ``int``. Duplicates collapse,
    which is what lets a per-game frame be passed straight in. Seasons before
    ``CORPUS_FIRST_SEASON`` are DROPPED rather than raising, because the corpus floor is a
    property of the rule and not of the caller's frame.

    Args:
        gold_seasons: Any iterable of season labels.

    Returns:
        The sorted, de-duplicated seasons at or above ``CORPUS_FIRST_SEASON``.
    """
    return tuple(
        sorted(
            {
                int(season)  # type: ignore[call-overload]
                for season in gold_seasons
                if int(season) >= CORPUS_FIRST_SEASON  # type: ignore[call-overload]
            }
        )
    )


def default_completed_seasons() -> tuple[int, ...]:
    """The completed seasons an IMPORT-TIME consumer gets, with no frame in hand.

    ``CORPUS_FIRST_SEASON`` through ``LATEST_COMPLETED_SEASON`` inclusive. See the block
    comment on ``LATEST_COMPLETED_SEASON`` for why that bound is a dated literal rather than a
    derivation.
    """
    return tuple(range(CORPUS_FIRST_SEASON, LATEST_COMPLETED_SEASON + 1))


def derive_season_partition(completed_seasons: Iterable[object]) -> SeasonPartition:
    """Apply THE rule to *completed_seasons*.

    Deterministic by construction: it ranks nothing, fits nothing and reads no data. Given the
    same completed seasons it returns the same partition on every machine and in every phase.

    Args:
        completed_seasons: Any iterable of completed season labels. Normalized through
            :func:`completed_seasons_from`, so a gold frame's ``season`` column is accepted
            directly.

    Returns:
        The :class:`SeasonPartition`.

    Raises:
        ValueError: If there are too few completed seasons to fill the holdout and hp-val folds,
            or if the selection window would be EMPTY. Both name the shortfall. Returning an
            empty hp_val instead would be refused by ``TemporalSplitConfig.validate()`` several
            layers later, with a message about a component this function never mentions.
    """
    completed = completed_seasons_from(completed_seasons)

    minimum = HOLDOUT_SEASON_COUNT + HP_VAL_SEASON_COUNT + 1
    if len(completed) < minimum:
        msg = (
            f"the season partition rule needs at least {minimum} completed seasons at or "
            f"above {CORPUS_FIRST_SEASON} ({HOLDOUT_SEASON_COUNT} holdout + "
            f"{HP_VAL_SEASON_COUNT} hp_val + at least 1 selection season), but got "
            f"{len(completed)}: {list(completed)}. Returning an empty hp_val instead would be "
            "refused by TemporalSplitConfig.validate() much later, with a message about a "
            "calibration component this rule never mentions."
        )
        raise ValueError(msg)

    holdout = completed[-HOLDOUT_SEASON_COUNT:]
    hp_val = completed[
        -(HOLDOUT_SEASON_COUNT + HP_VAL_SEASON_COUNT) : -HOLDOUT_SEASON_COUNT
    ]
    selection = tuple(
        season
        for season in completed
        if SELECTION_WINDOW_FIRST_SEASON <= season < hp_val[0]
    )

    if not selection:
        msg = (
            f"the selection window is EMPTY: it runs from "
            f"SELECTION_WINDOW_FIRST_SEASON={SELECTION_WINDOW_FIRST_SEASON} through the "
            f"season before hp_val={list(hp_val)}, and no completed season falls in that "
            f"range (completed seasons: {list(completed)}). Either the corpus does not yet "
            "reach the selection floor, or the floor was raised past the data."
        )
        raise ValueError(msg)

    return SeasonPartition(
        selection=selection,
        hp_val=hp_val,
        holdout=holdout,
        final_fit=completed,
    )


def default_season_partition() -> SeasonPartition:
    """The partition an IMPORT-TIME consumer gets: the rule applied to the default corpus."""
    return derive_season_partition(default_completed_seasons())


def backtest_seasons(partition: SeasonPartition) -> tuple[int, ...]:
    """The seasons a BACKTEST walks: ``SELECTION_WINDOW_FIRST_SEASON`` through the holdout.

    ``conf.settings.BacktestConfig.seasons`` and ``backtest.engine.BacktestConfig``'s
    ``first_data_season`` / ``max_backtest_season`` pair all describe this same span, and all
    three used to carry it as their own literal. The lower bound is the selection floor because
    the engine's own expanding rule (``backtest/engine.py:211``) trains from
    ``first_data_season``; the upper bound is the last holdout season, which under this rule IS
    the most recent completed season -- which is exactly how 2025 stopped being visible.

    Args:
        partition: A partition from :func:`derive_season_partition`.

    Returns:
        The inclusive season span as a tuple.
    """
    return tuple(range(SELECTION_WINDOW_FIRST_SEASON, partition.holdout[-1] + 1))
