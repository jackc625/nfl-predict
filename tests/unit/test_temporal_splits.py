"""Tests for temporal split infrastructure (MODL-01, MODL-13).

Verifies:
- Three-fold temporal split with no overlap (MODL-13)
- Walk-forward expanding-window splits for holdout (MODL-01)
- Temporal CV splits for hyperparameter tuning
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.temporal import (
    TemporalSplitConfig,
    WalkForwardSplitter,
    make_temporal_cv_splits,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_features_df() -> pd.DataFrame:
    """Create a synthetic feature DataFrame with seasons 2018-2024, 10 rows per season."""
    rows = []
    for season in range(2018, 2025):
        for i in range(10):
            rows.append(
                {
                    "game_id": f"{season}_{i:02d}",
                    "season": season,
                    "week": (i % 18) + 1,
                    "home_team": "KC",
                    "away_team": "BUF",
                    "feature_1": np.random.default_rng(season * 100 + i).random(),
                    "feature_2": np.random.default_rng(season * 200 + i).random(),
                    "home_win": int(
                        np.random.default_rng(season * 300 + i).random() > 0.5
                    ),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def default_config() -> TemporalSplitConfig:
    """Default temporal split configuration."""
    return TemporalSplitConfig.default()


# ---------------------------------------------------------------------------
# Test 1: Three-fold no overlap
# ---------------------------------------------------------------------------


def test_three_fold_no_overlap():
    """TemporalSplitConfig ensures no season overlap across the three groups."""
    config = TemporalSplitConfig(
        train_seasons=[2018, 2019],
        hp_val_seasons=[2020],
        holdout_seasons=[2021, 2022, 2023, 2024],
    )
    # Validation should pass without error
    config.validate()

    # Verify no overlap
    train_set = set(config.train_seasons)
    hp_val_set = set(config.hp_val_seasons)
    holdout_set = set(config.holdout_seasons)

    assert train_set & hp_val_set == set(), "Train and HP-val overlap"
    assert train_set & holdout_set == set(), "Train and holdout overlap"
    assert hp_val_set & holdout_set == set(), "HP-val and holdout overlap"


# ---------------------------------------------------------------------------
# Test 2: Three-fold temporal ordering
# ---------------------------------------------------------------------------


def test_three_fold_temporal_ordering():
    """validate() raises ValueError if temporal ordering is violated."""
    # Train contains season >= min(hp_val)
    bad_config_1 = TemporalSplitConfig(
        train_seasons=[2018, 2020],
        hp_val_seasons=[2019],
        holdout_seasons=[2021, 2022],
    )
    with pytest.raises(ValueError, match=r"train.*precede.*hp_val|temporal order"):
        bad_config_1.validate()

    # HP-val contains season >= min(holdout)
    bad_config_2 = TemporalSplitConfig(
        train_seasons=[2018, 2019],
        hp_val_seasons=[2021],
        holdout_seasons=[2020, 2022],
    )
    with pytest.raises(ValueError, match=r"hp_val.*precede.*holdout|temporal order"):
        bad_config_2.validate()


# ---------------------------------------------------------------------------
# Test 3: Walk-forward splits
# ---------------------------------------------------------------------------


def test_walk_forward_splits(synthetic_features_df, default_config):
    """WalkForwardSplitter generates 4 splits for 4 holdout seasons."""
    splitter = WalkForwardSplitter(config=default_config, target_col="home_win")
    splits = list(splitter.generate_splits(synthetic_features_df))

    # 4 holdout seasons -> 4 splits
    assert len(splits) == 4, f"Expected 4 splits, got {len(splits)}"

    # First split: train on [2018, 2019, 2020], test on 2021
    assert splits[0].train_seasons == [2018, 2019, 2020]
    assert splits[0].test_season == 2021


# ---------------------------------------------------------------------------
# Test 4: Walk-forward expanding window
# ---------------------------------------------------------------------------


def test_walk_forward_expanding_window(synthetic_features_df, default_config):
    """For holdout season 2023, train data includes seasons 2018-2022 (expanding)."""
    splitter = WalkForwardSplitter(config=default_config, target_col="home_win")
    splits = list(splitter.generate_splits(synthetic_features_df))

    # Find the split for season 2023 (index 2 in holdout [2021,2022,2023,2024])
    split_2023 = next(s for s in splits if s.test_season == 2023)
    assert split_2023.train_seasons == [2018, 2019, 2020, 2021, 2022]


# ---------------------------------------------------------------------------
# Test 5: Walk-forward no leakage
# ---------------------------------------------------------------------------


def test_walk_forward_no_leakage(synthetic_features_df, default_config):
    """For each split, max train season is strictly less than test season."""
    splitter = WalkForwardSplitter(config=default_config, target_col="home_win")
    splits = list(splitter.generate_splits(synthetic_features_df))

    for split in splits:
        max_train_season = max(split.train_seasons)
        assert max_train_season < split.test_season, (
            f"Leakage detected: max train season {max_train_season} "
            f">= test season {split.test_season}"
        )


# ---------------------------------------------------------------------------
# Test 6: make_temporal_cv_splits
# ---------------------------------------------------------------------------


def test_make_temporal_cv_splits(synthetic_features_df):
    """make_temporal_cv_splits produces 3 folds where test is after train."""
    folds = make_temporal_cv_splits(synthetic_features_df, n_splits=3)

    assert len(folds) == 3, f"Expected 3 folds, got {len(folds)}"

    for train_indices, test_indices in folds:
        # Ensure indices are valid numpy arrays
        assert len(train_indices) > 0
        assert len(test_indices) > 0

        # Get max train season and min test season from original df
        sorted_df = synthetic_features_df.sort_values(["season", "week"]).reset_index(
            drop=True
        )
        train_max_season = sorted_df.iloc[train_indices]["season"].max()
        test_min_season = sorted_df.iloc[test_indices]["season"].min()

        assert train_max_season <= test_min_season, (
            f"Temporal violation: max train season {train_max_season} "
            f"> min test season {test_min_season}"
        )


# ---------------------------------------------------------------------------
# Test 7: Default config
# ---------------------------------------------------------------------------


def test_default_config():
    """TemporalSplitConfig.default() returns the expected default split."""
    config = TemporalSplitConfig.default()

    assert config.train_seasons == [2018, 2019]
    assert config.hp_val_seasons == [2020]
    assert config.holdout_seasons == [2021, 2022, 2023, 2024]

    # Should validate cleanly
    config.validate()


# ---------------------------------------------------------------------------
# Test 8: an EMPTY hp_val fold is rejected by name (quick task 260816-u0e, D-Q2)
# ---------------------------------------------------------------------------
#
# The covered selection window was originally designed with an empty hp_val, on
# the reasoning that the fold is unused under tune=False. That reasoning holds
# for BaseTrainer and fails for all three concrete trainers, each of which fits a
# calibration/conversion component on the hp-val fold outside the tuning branch.
# The empty case therefore produces one hard crash (WP) and two silently
# degenerate models (ATS/OU), so validate() now rejects it with a message that
# names the cause.
#
# These tests pin two things: (a) every config shape that exists today validates
# or raises exactly as it always has, with the same messages -- the change is
# additive; (b) the empty case raises an ACTIONABLE error rather than the bare
# "min() iterable argument is empty" it used to produce from the ordering check.

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every config shape that exists today, paired with the behaviour it must keep.
# The four raising variants are the four ways validate() can currently fail on
# ordering/overlap with a non-empty hp_val.
_NON_EMPTY_HP_VAL_CASES = {
    "canonical_default": (TemporalSplitConfig.default(), None),
    "coverage_window": (
        TemporalSplitConfig(
            train_seasons=[2021, 2022],
            hp_val_seasons=[2023],
            holdout_seasons=[2024],
        ),
        None,
    ),
    "walkforward_engine_style": (
        TemporalSplitConfig(
            train_seasons=[2002, 2003, 2004],
            hp_val_seasons=[2005],
            holdout_seasons=[2006],
        ),
        None,
    ),
    "train_hp_val_overlap": (
        TemporalSplitConfig(
            train_seasons=[2018, 2019],
            hp_val_seasons=[2019],
            holdout_seasons=[2021],
        ),
        "overlap between train and hp_val",
    ),
    "hp_val_holdout_overlap": (
        TemporalSplitConfig(
            train_seasons=[2018],
            hp_val_seasons=[2020],
            holdout_seasons=[2020, 2021],
        ),
        "overlap between hp_val and holdout",
    ),
    "train_not_before_hp_val": (
        TemporalSplitConfig(
            train_seasons=[2018, 2020],
            hp_val_seasons=[2019],
            holdout_seasons=[2021],
        ),
        "train seasons must precede hp_val seasons",
    ),
    "hp_val_not_before_holdout": (
        TemporalSplitConfig(
            train_seasons=[2018],
            hp_val_seasons=[2022],
            holdout_seasons=[2021, 2023],
        ),
        "hp_val seasons must precede holdout seasons",
    ),
}


@pytest.mark.parametrize("case", sorted(_NON_EMPTY_HP_VAL_CASES))
def test_non_empty_hp_val_configs_behave_identically(case: str) -> None:
    """Every existing config shape validates or raises exactly as it did before."""
    config, expected_error = _NON_EMPTY_HP_VAL_CASES[case]
    if expected_error is None:
        config.validate()
        return
    with pytest.raises(ValueError, match=expected_error):
        config.validate()


def test_no_shipped_config_has_an_empty_hp_val_list() -> None:
    """No production config constructs an empty hp_val -- so nothing regresses.

    The two runtime-constructed configs build hp_val from an expression that is
    non-empty by construction (``[holdout_season - 1]`` in backtest/engine.py and
    a parsed CLI list defaulting to 2020 in models/train.py), and no module-level
    config declares one literally.
    """
    from backtest.signal_lift import (
        COVERAGE_WINDOW_CONFIG,
        COVERED_SELECTION_WINDOW_CONFIG,
    )

    assert TemporalSplitConfig.default().hp_val_seasons
    assert COVERAGE_WINDOW_CONFIG.hp_val_seasons
    assert COVERED_SELECTION_WINDOW_CONFIG.hp_val_seasons

    literal_empty = []
    scanned = 0
    for path in REPO_ROOT.glob("**/*.py"):
        parts = set(path.parts)
        if parts & {".venv", "tests", "__pycache__"}:
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8")
        if "hp_val_seasons=[]" in text or "hp_val_seasons = []" in text:
            literal_empty.append(path.relative_to(REPO_ROOT).as_posix())

    # Self-check (WR-11 hardening). A scan that silently matches NOTHING passes
    # this test vacuously and would keep passing forever. The glob pattern above
    # is the correct portable forward-slash form and visits ~142 production files
    # today; the floor is set far below that so it is a real tripwire against the
    # scan degrading, not a count that has to be maintained.
    assert scanned > 50, (
        f"the production scan visited only {scanned} files -- it is matching "
        f"almost nothing, so the assertion below would pass vacuously"
    )

    assert not literal_empty, (
        f"an empty hp_val_seasons list appeared in production code: "
        f"{literal_empty}. Every trainer fits a calibration/conversion component "
        "on the hp-val fold, so an empty fold crashes WP and silently gives "
        "ATS/OU a NaN-scale converter."
    )


def test_empty_hp_val_is_rejected_by_name() -> None:
    """The empty fold raises an ACTIONABLE error naming the real cause.

    Before this guard the same config raised ``min() iterable argument is empty``
    from the ordering check -- a message that says nothing about why an empty
    hp-val fold is unusable, and which sent this task's design down a path that
    only failed later, inside StandardScaler.
    """
    with pytest.raises(ValueError, match="hp_val_seasons must not be empty"):
        TemporalSplitConfig(
            train_seasons=[2018, 2019, 2020],
            hp_val_seasons=[],
            holdout_seasons=[2021, 2022, 2023, 2024],
        ).validate()


def test_empty_hp_val_error_names_the_trainer_consumers() -> None:
    """The message points at the code that makes the fold load-bearing.

    A guard whose message does not name its cause gets deleted by the next person
    who hits it.
    """
    with pytest.raises(ValueError) as excinfo:
        TemporalSplitConfig(
            train_seasons=[2018],
            hp_val_seasons=[],
            holdout_seasons=[2021],
        ).validate()

    message = str(excinfo.value)
    for citation in ("wp_trainer.py", "ats_trainer.py", "ou_trainer.py"):
        assert citation in message


def test_overlap_checks_run_before_the_empty_hp_val_guard() -> None:
    """A structurally broken config still reports its overlap first."""
    with pytest.raises(ValueError, match="overlap between train and holdout"):
        TemporalSplitConfig(
            train_seasons=[2018, 2021],
            hp_val_seasons=[],
            holdout_seasons=[2021],
        ).validate()
