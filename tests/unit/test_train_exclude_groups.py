"""The D30-01 ``--exclude-groups`` CLI seam contract (Plan 30-01, PROD-01).

``models.train`` gained a ``--exclude-groups`` flag so a Stage-2 re-fit can train on a
Stage-1-selected feature set WITHOUT producing a second gold-shaped artifact: the exclusion
is applied IN MEMORY to the frame handed to ``train_target`` via
``backtest.signal_lift.select_group_columns`` (D30-01/D30-02). This module pins the seam's
contract:

  1. PARSING -- the empty default yields an EMPTY exclusion tuple. This is the load-bearing
     case: the naive ``"".split(",")`` yields ``[""]``, a one-element list holding the empty
     string, which reaches ``group_columns`` as an unregistered name and raises. Under the
     naive form the DEFAULT invocation -- the one that must reproduce today's behaviour
     exactly -- would hard-fail.
  2. DEFAULT PATH -- an empty exclusion tuple leaves the frame width unchanged.
  3. EXCLUSION PATH -- excluding a group removes exactly that group's columns and no others.
  4. UNKNOWN NAME -- hard-fails with a ``ValueError`` naming the valid group set; no silent
     coercion (T-30-13).
  5. ABSENT FAMILY -- excluding a registered group with zero columns present is a legal no-op.
     This is what keeps the promote invocation stable across the Plan 30-07 line-movement drop.
  6. READ-ONLY -- ``select_group_columns`` returns a fresh copy; ``data/gold`` is never touched.

The group vocabulary is IMPORTED from ``backtest.signal_lift``, never re-declared here: a
second group list is exactly what silently broke the Phase-28 baseline in 29-06 (T-30-15).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from backtest.signal_lift import (
    _INJURY_COLUMN_BASENAMES,
    _LINE_MV_SUFFIXES,
    _SITUATIONAL_SUFFIXES,
    ALL_REGISTERED_GROUPS,
    group_columns,
    select_group_columns,
)
from models.train import build_parser, parse_exclude_groups

GOLD_DIR = Path(__file__).resolve().parents[2] / "data" / "gold"


# ---------------------------------------------------------------------------
# Synthetic frame builders -- column names DERIVED from the live predicates'
# own basename/suffix tuples, never a second hardcoded list.
# ---------------------------------------------------------------------------


def _injury_columns() -> list[str]:
    """Every injury column name a synthetic frame needs (home_/away_ x basenames)."""
    return [
        f"{side}_{base}"
        for side in ("home", "away")
        for base in _INJURY_COLUMN_BASENAMES
    ]


def _situational_columns() -> list[str]:
    """Every situational spot column name (home_/away_ x the new spot suffixes)."""
    return [
        f"{side}_{suffix}"
        for side in ("home", "away")
        for suffix in _SITUATIONAL_SUFFIXES
    ]


def _snap_columns() -> list[str]:
    """Snap columns, one per matched substring family (the predicate is substring-based)."""
    return [
        f"{side}_{stem}"
        for side in ("home", "away")
        for stem in ("snap_continuity", "snap_concentration", "rolling_snap_share_qb")
    ]


def _line_movement_columns() -> list[str]:
    """Line-movement columns -- game-level, NO home_/away_ prefix (exact suffix match)."""
    return list(_LINE_MV_SUFFIXES)


_BASELINE_COLUMNS = (
    "game_id",
    "season",
    "week",
    "home_elo_pre_z",
    "away_elo_pre_z",
    "snapshot_spread",
    "snapshot_total",
    "total_points",
    "home_win",
)


def _synthetic_gold(*, include_line_movement: bool = True) -> pd.DataFrame:
    """Build a small gold-shaped frame carrying every registered group's columns.

    Args:
        include_line_movement: When False the line-movement family is ABSENT, which is the
            post-Plan-30-07 shape the promote invocation must still tolerate.

    Returns:
        A 4-row frame with baseline columns plus each group's columns.
    """
    columns = [
        *_BASELINE_COLUMNS,
        *_snap_columns(),
        *_injury_columns(),
        *_situational_columns(),
    ]
    if include_line_movement:
        columns.extend(_line_movement_columns())
    n_rows = 4
    return pd.DataFrame(
        {col: list(range(n_rows)) for col in columns},
    )


# ---------------------------------------------------------------------------
# (1) Parsing -- the empty default is the case the naive comma-split breaks
# ---------------------------------------------------------------------------


def test_empty_default_parses_to_empty_tuple() -> None:
    """The DEFAULT empty value yields an EMPTY tuple, not ``('',)``.

    ``"".split(",")`` returns ``[""]``. Without the ``if g.strip()`` filter that empty
    string reaches ``group_columns`` as an unregistered name and raises ValueError, so the
    default invocation -- the one that must reproduce today's behaviour byte-for-byte --
    would hard-fail. Remediation if this goes red: restore the filtering comprehension
    ``[g.strip() for g in s.split(",") if g.strip()]`` in models.train.parse_exclude_groups.
    """
    parsed = parse_exclude_groups("")
    assert parsed == (), (
        f"parse_exclude_groups('') returned {parsed!r}, expected an empty tuple. The "
        "empty-token filter was dropped; restore "
        "'[g.strip() for g in s.split(\",\") if g.strip()]' in models.train."
    )


def test_default_parser_value_is_the_empty_string() -> None:
    """``--exclude-groups`` defaults to the empty string, so omitting it excludes nothing.

    Remediation if this goes red: the argparse default was changed; set it back to "" so a
    bare ``python -m models.train`` reproduces today's feature set exactly (D30-01).
    """
    args = build_parser().parse_args([])
    assert args.exclude_groups == "", (
        f"--exclude-groups default is {args.exclude_groups!r}, expected ''. A non-empty "
        "default silently changes what every existing caller trains on."
    )
    assert parse_exclude_groups(args.exclude_groups) == (), (
        "The parser default does not round-trip to an empty exclusion tuple; the default "
        "invocation would no longer reproduce today's behaviour."
    )


def test_comma_separated_names_parse_in_order() -> None:
    """A comma-separated scalar token parses to the named groups, in argv order.

    Remediation if this goes red: the flag was switched to nargs='+' or the split changed;
    keep it a single scalar token (nargs='+' is an argv foot-gun under PowerShell when
    followed by another flag).
    """
    parsed = parse_exclude_groups("injury,snap")
    assert parsed == ("injury", "snap"), (
        f"parse_exclude_groups('injury,snap') returned {parsed!r}, expected "
        "('injury', 'snap'). Restore the comma-split comprehension in models.train."
    )


def test_whitespace_and_trailing_comma_are_filtered_out() -> None:
    """Surrounding whitespace, an all-whitespace token and a trailing comma never become names.

    Each of those would otherwise reach ``group_columns`` as an unregistered name and abort a
    train that the operator meant to run. Remediation if this goes red: restore both the
    ``.strip()`` and the ``if g.strip()`` filter in models.train.parse_exclude_groups.
    """
    parsed = parse_exclude_groups("  injury , ,  snap  ,")
    assert parsed == ("injury", "snap"), (
        f"parse_exclude_groups('  injury , ,  snap  ,') returned {parsed!r}, expected "
        "('injury', 'snap'). Whitespace/empty tokens must be stripped and dropped, not "
        "passed through as group names."
    )


def test_parsed_names_are_a_subset_of_the_registered_vocabulary() -> None:
    """Every name this module exercises is a LIVE registered group, imported not hardcoded.

    Remediation if this goes red: a group was renamed or de-registered in
    backtest.signal_lift._GROUP_PREDICATE; update the seam and this contract together.
    """
    for name in ("injury", "snap", "situational", "line_movement"):
        assert name in ALL_REGISTERED_GROUPS, (
            f"'{name}' is not in the live ALL_REGISTERED_GROUPS "
            f"{ALL_REGISTERED_GROUPS}. This test derives from the live registry; if the "
            "group was renamed, rename it here and in the promote invocation too."
        )


# ---------------------------------------------------------------------------
# (2) Default path -- an empty exclusion tuple is a true no-op
# ---------------------------------------------------------------------------


def test_empty_exclusion_leaves_the_frame_width_unchanged() -> None:
    """With nothing excluded, the frame handed to train_target keeps every column.

    Remediation if this goes red: models.train must SKIP the select_group_columns call
    entirely when the parsed tuple is empty (D30-01) -- calling it with an empty
    exclude_groups is not equivalent, because the default of select_group_columns is the
    Phase-28 GROUPS deny-list, which would silently strip three families.
    """
    df = _synthetic_gold()
    kept = select_group_columns(df, None, exclude_groups=())
    assert list(kept.columns) == list(df.columns), (
        f"An empty exclusion changed the frame from {len(df.columns)} to "
        f"{len(kept.columns)} columns. Skip the call when the parsed tuple is empty."
    )


# ---------------------------------------------------------------------------
# (3) Exclusion path -- exactly the group's columns, and no others
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group", ["injury", "snap", "situational", "line_movement"])
def test_excluding_a_group_removes_exactly_that_groups_columns(group: str) -> None:
    """Excluding one group removes exactly ``group_columns(df, group)`` -- no collateral.

    Remediation if this goes red: a predicate in backtest.signal_lift widened (most likely a
    bare substring guard was added). Never add a bare 'snap' or 'total' substring test --
    they collide with rolling_snap_* / snapshot_total / total_points (RESEARCH D-13).
    """
    df = _synthetic_gold()
    expected_removed = set(group_columns(df, group))
    assert expected_removed, (
        f"The synthetic frame carries no '{group}' columns, so this assertion would be "
        "vacuous. Rebuild the synthetic frame from the live predicate suffixes."
    )

    kept = select_group_columns(df, None, exclude_groups=(group,))
    removed = set(df.columns) - set(kept.columns)
    assert removed == expected_removed, (
        f"Excluding '{group}' removed {sorted(removed)}, expected exactly "
        f"{sorted(expected_removed)}. A mismatch means the group predicate widened or "
        "narrowed; fix the predicate in backtest.signal_lift, not this test."
    )


def test_injury_exclusion_leaves_every_other_group_intact() -> None:
    """Excluding injury does not disturb snap, situational or line-movement columns.

    Remediation if this goes red: the exclusion is over-reaching; inspect
    backtest.signal_lift.excluded_columns for a predicate that matches across families.
    """
    df = _synthetic_gold()
    kept = select_group_columns(df, None, exclude_groups=("injury",))
    for other in ("snap", "situational", "line_movement"):
        for col in group_columns(df, other):
            assert col in kept.columns, (
                f"Excluding 'injury' also dropped '{col}', which belongs to '{other}'. "
                "The injury predicate is matching outside its own family."
            )


# ---------------------------------------------------------------------------
# (4) Unknown name -- hard-fail, never silent coercion (T-30-13)
# ---------------------------------------------------------------------------


def test_unknown_group_name_raises_valueerror_listing_valid_names() -> None:
    """An unregistered group name aborts the train and the message names the valid set.

    A typo'd group name must never silently exclude nothing: that would produce a candidate
    trained on a feature set nobody chose, gate-scored as if it were the selected one.
    Remediation if this goes red: restore the ``group not in _GROUP_PREDICATE`` guard in
    backtest.signal_lift.group_columns -- do NOT add a coercion or a fuzzy match.
    """
    df = _synthetic_gold()
    with pytest.raises(ValueError, match="Unknown group") as excinfo:
        select_group_columns(df, None, exclude_groups=("notagroup",))

    message = str(excinfo.value)
    for name in sorted(ALL_REGISTERED_GROUPS):
        assert name in message, (
            f"The ValueError message does not list the valid group '{name}': {message!r}. "
            "The operator needs the valid vocabulary in the failure, not just the rejection."
        )


# ---------------------------------------------------------------------------
# (5) Absent family -- a registered group with zero columns is a legal no-op
# ---------------------------------------------------------------------------


def test_excluding_an_absent_family_is_a_no_op() -> None:
    """Excluding line_movement from a frame that has none returns an equal-width frame.

    This is what keeps ``--exclude-groups line_movement`` STABLE across the Plan 30-07 drop:
    once the family is gone from gold the same promote invocation must still run, not abort.
    Remediation if this goes red: group_columns must return [] (not raise) for a REGISTERED
    group with no matching columns; only an UNREGISTERED name raises.
    """
    df = _synthetic_gold(include_line_movement=False)
    assert group_columns(df, "line_movement") == [], (
        "The no-line-movement synthetic frame still matches line-movement columns; the "
        "frame builder is wrong, so this test would not exercise the absent-family case."
    )

    kept = select_group_columns(df, None, exclude_groups=("line_movement",))
    assert list(kept.columns) == list(df.columns), (
        f"Excluding an absent family changed the width from {len(df.columns)} to "
        f"{len(kept.columns)}. A zero-column registered group must be a pure no-op."
    )


# ---------------------------------------------------------------------------
# (6) Read-only -- the input frame and data/gold are never mutated
# ---------------------------------------------------------------------------


def test_select_group_columns_returns_a_fresh_object() -> None:
    """The input frame is not mutated and the result is not a view onto it.

    Remediation if this goes red: select_group_columns must end in ``gold_df[keep].copy()``;
    dropping the ``.copy()`` hands back a view whose later mutation writes through.
    """
    df = _synthetic_gold()
    before = list(df.columns)
    kept = select_group_columns(df, None, exclude_groups=("injury",))

    assert list(df.columns) == before, (
        "select_group_columns mutated its input frame's columns. It must return a fresh "
        "copy and leave the caller's frame untouched (HARD BOUNDARY)."
    )
    assert kept is not df, "select_group_columns returned the input object itself."


@pytest.mark.parametrize("target", ["wp", "ats", "ou"])
def test_gold_parquet_is_byte_unchanged_across_a_selection(target: str) -> None:
    """Reading gold and applying an exclusion leaves the on-disk parquet byte-identical.

    Asserted by sha256 over the file, NOT by ``git status``: ``data/`` is gitignored
    (.gitignore line 22), so a git-status assertion here would be vacuously green.
    Remediation if this goes red: the exclusion path is writing a second gold-shaped
    artifact -- D30-01 explicitly rejected that alternative; keep the selection in memory.
    """
    path = GOLD_DIR / f"features_{target}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built yet -- run scripts.build_features first")

    digest_before = hashlib.sha256(path.read_bytes()).hexdigest()
    gold_df = pd.read_parquet(path)
    select_group_columns(gold_df, None, exclude_groups=("line_movement",))
    digest_after = hashlib.sha256(path.read_bytes()).hexdigest()

    assert digest_after == digest_before, (
        f"{path} changed across a select_group_columns call "
        f"({digest_before[:12]} -> {digest_after[:12]}). The in-memory selection must never "
        "write under data/ (Plan 30-01 prohibition)."
    )


def test_the_summary_brier_column_is_the_brier_score_not_the_ece(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A33.2-review IN-03: the Brier column used to print metadata['ece']."""
    from models.train import print_summary

    holdout = pd.DataFrame({"prediction": [0.8, 0.3], "actual": [1.0, 0.0]})
    print_summary(
        {
            "wp": {
                "model_metrics": {
                    "season_results": [{"season": 2024, "accuracy": 1.0, "n_games": 2}],
                    "metadata": {"ece": 0.777},
                    "holdout_predictions": holdout,
                },
                "market_baseline": None,
            }
        }
    )
    row = next(
        line for line in capsys.readouterr().out.splitlines() if line.startswith("WP")
    )
    brier = (0.2**2 + 0.3**2) / 2
    assert f"{brier:.3f}" in row
    assert row.count("0.777") == 1, "the ECE must appear once, in its own column"
