"""p332_ extra step 8e: the eight never-populated defensive copies leave the gold layout.

OWNER RULING 2026-09-22 (deferred-items.md, "Eight gold team-form columns are a constant
0.0 in EVERY season: the defensive copies of the four offense-only metrics", option (a) --
"REMOVE the eight columns from gold"). ``{home,away}_def_rolling_cpoe``,
``{home,away}_def_rolling_avg_drive_start_yardline``,
``{home,away}_def_rolling_neutral_pace`` and ``{home,away}_def_rolling_neutral_pass_rate``
are removed from the gold team-form layout.

WHY. The four metrics are offense-only BY DESIGN: ``TeamFormCalculator`` writes NaN for
their defensive side (``calculate_rolling_averages``'s ``offense_only_metrics``), and the
per-game frame does the same (``:516-523``, "Not applicable for defense"). Gold copied
every ``rolling_*`` column of the defence row regardless, and normalization turned the
resulting all-NaN block into a flat 0.0 -- a value nobody measured, reading as "exactly
average", in all 6,499 rows of every season. Measured on rung 8's gold:
``test_no_precoverage_constants.py::test_gold_team_form_and_opp_adj_not_constant_within_season``
fails on exactly these eight (576 matrix-season-column triples), and they are 8 of the 15
``CONSTANT_2002_2017_AFTER_RUNG8`` columns.

REJECTED, and recorded so neither is mistaken for an oversight: keeping them blank (eight
columns that can never hold anything), and computing real defence-allowed versions (a NEW
signal, which belongs with new-signal work and not inside this phase's correction ladder).

ONE REGISTRY, NOT A SECOND HAND-WRITTEN LIST. The removal reads the SAME set the
calculator skips, so the layout and the calculator cannot drift apart: a fifth offence-only
metric added to the calculator leaves gold automatically, and a metric that stops being
offence-only returns automatically.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pandas as pd

from features import team_form
from scripts.build_features import FeatureMatrixBuilder

#: The four rolling metrics the calculator never populates for the defence side.
EXPECTED_OFFENSE_ONLY_ROLLING = (
    "rolling_avg_drive_start_yardline",
    "rolling_cpoe",
    "rolling_neutral_pace",
    "rolling_neutral_pass_rate",
)

#: The eight gold columns the ruling removes.
EXPECTED_REMOVED_GOLD_COLUMNS = tuple(
    sorted(
        f"{prefix}_def_{column}"
        for prefix in ("home", "away")
        for column in EXPECTED_OFFENSE_ONLY_ROLLING
    )
)


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2024_W01_DET@KC",
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "DET",
            }
        ]
    )


def _team_form_rows() -> pd.DataFrame:
    """One offence row and one defence row per team, carrying every rolling column.

    The defence rows carry a VALUE in the offence-only columns -- not the NaN the real
    calculator writes -- so the assertion below is about the LAYOUT rule and cannot pass
    by accident on a frame that happened to be empty there.
    """
    rows = []
    for team in ("KC", "DET"):
        for side in ("offense", "defense"):
            row = {
                "team": team,
                "side": side,
                "target_season": 2024,
                "target_week": 1,
            }
            for column in team_form.ROLLING_COLUMNS:
                row[column] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


class TestTheOffenceOnlySetIsOneRegistry:
    def test_the_module_declares_the_offence_only_rolling_columns(self) -> None:
        declared = getattr(team_form, "OFFENSE_ONLY_ROLLING_COLUMNS", None)
        assert declared is not None, (
            "features.team_form does not declare OFFENSE_ONLY_ROLLING_COLUMNS, so the "
            "gold layout would need a second hand-written list of the same four metrics"
        )
        assert tuple(sorted(declared)) == EXPECTED_OFFENSE_ONLY_ROLLING

    def test_the_calculator_skips_exactly_that_set(self) -> None:
        """The registry IS what ``calculate_rolling_averages`` skips, read off its source.

        Non-vacuity control for the registry: a module-level tuple that the method
        ignored would satisfy the test above while the two drifted apart.
        """
        source = textwrap.dedent(
            inspect.getsource(team_form.TeamFormCalculator.calculate_rolling_averages)
        )
        tree = ast.parse(source)
        skipped: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and isinstance(node.ops[0], ast.In):
                target = node.comparators[0]
                if isinstance(target, ast.Name | ast.Attribute):
                    skipped.add(
                        target.attr if isinstance(target, ast.Attribute) else target.id
                    )
        assert "OFFENSE_ONLY_METRICS" in skipped, (
            "the defence-side skip does not read the module-level registry, so the "
            f"layout rule and the calculator can drift apart (found: {sorted(skipped)})"
        )

    def test_the_gold_column_names_are_derived_not_retyped(self) -> None:
        resolver = getattr(team_form, "offense_only_gold_columns", None)
        assert resolver is not None, (
            "features.team_form does not expose offense_only_gold_columns()"
        )
        assert tuple(sorted(resolver())) == EXPECTED_REMOVED_GOLD_COLUMNS


class TestTheGoldLayoutDropsThem:
    def test_the_defence_side_carries_none_of_the_four(self) -> None:
        laid_out = FeatureMatrixBuilder._get_team_features(
            _team_form_rows(), _games(), "home_team", "home"
        )
        for column in EXPECTED_REMOVED_GOLD_COLUMNS:
            if column.startswith("home_"):
                assert column not in laid_out.columns, column

    def test_the_offence_side_keeps_all_twelve(self) -> None:
        """No false positive: the OFFENSIVE copies of the same four metrics stay.

        The four are offence-only, not absent -- removing the offensive copy as well
        would delete four real measurements, which is the opposite of this ruling.
        """
        laid_out = FeatureMatrixBuilder._get_team_features(
            _team_form_rows(), _games(), "home_team", "home"
        )
        for column in team_form.ROLLING_COLUMNS:
            assert f"home_off_{column}" in laid_out.columns, column

    def test_the_other_eight_defensive_copies_stay(self) -> None:
        """No over-reach: only the four offence-only metrics lose their defence copy."""
        laid_out = FeatureMatrixBuilder._get_team_features(
            _team_form_rows(), _games(), "away_team", "away"
        )
        kept = [
            column
            for column in team_form.ROLLING_COLUMNS
            if column not in EXPECTED_OFFENSE_ONLY_ROLLING
        ]
        assert len(kept) == 8, "non-vacuity: eight defensive copies must survive"
        for column in kept:
            assert f"away_def_{column}" in laid_out.columns, column

    def test_a_planted_defensive_value_is_still_dropped(self) -> None:
        """The drop is a LAYOUT rule, not a consequence of the value being absent.

        The fixture already plants 1.0 in every defence-side offence-only column; this
        node names that as the control it is, so a future reader cannot mistake the
        removal for "the column happened to be empty".
        """
        rows = _team_form_rows()
        assert (
            rows.loc[rows["side"] == "defense", list(EXPECTED_OFFENSE_ONLY_ROLLING)]
            .notna()
            .all()
            .all()
        )
        laid_out = FeatureMatrixBuilder._get_team_features(
            rows, _games(), "home_team", "home"
        )
        assert "home_def_rolling_cpoe" not in laid_out.columns
