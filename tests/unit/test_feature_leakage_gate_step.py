"""The Friday temporal-safety gate can actually FAIL (CR-02).

WHAT THIS PINS
--------------
``pipeline.steps.step_validate_features`` is registered ``critical=True`` as the run's
temporal-safety gate. It was inert in three separate ways, and every one of them made it report
success for a check it had not performed:

1. it read ``leakage_result["has_leakage"]`` -- a key ``FeatureValidator.check_data_leakage`` has
   never returned (the verdict is ``passed``, the names are ``leakage_columns``), so the
   ``raise`` was unreachable and a post-game column reached predictions unremarked;
2. the loop's ``break`` sat inside the ``try``, so only ``features_wp`` was ever scanned and
   ``features_ats`` / ``features_ou`` were never leak-checked by this gate at all;
3. loading none of the three matrices skipped the check silently.

A gate that cannot fail is worth nothing, so each of the three is asserted by making it fail.

WHAT IT DELIBERATELY DOES NOT CLAIM
------------------------------------
This is the DEFENSE-IN-DEPTH sibling, not the binding gate. The binding one is
``LeakageGate.validate_combined_matrix`` on the canonical gold build path
(``scripts/build_features.py``). This step re-checks what actually landed on disk.

Nothing here reads or writes the real ``data/`` layer -- ``load_dataframe`` is patched.

Selectors (``-k``): fails, per_table, labels, no_matrix, passes.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline import steps

_CLEAN_COLUMNS = {
    "game_id": ["2025_W01_AAA@BBB"],
    "season": [2025],
    "week": [1],
    "elo_home": [1500.0],
    "rolling_epa_home": [0.05],
}

# The three LABEL columns a gold matrix carries by construction. Every one of them matches a
# leakage keyword by substring ("margin", "total_score"/"total", "outcome"-adjacent), which is why
# the gate must exclude them -- scanning them would hard-fail every Friday run on the target
# column itself.
_LABELS = {"home_win": [1], "home_margin": [7.0], "total_points": [44.0]}


def _matrix(**extra: object) -> pd.DataFrame:
    return pd.DataFrame({**_CLEAN_COLUMNS, **_LABELS, **extra})


def _patch_gold(
    monkeypatch: pytest.MonkeyPatch, tables: dict[str, pd.DataFrame]
) -> None:
    """Serve *tables* as the gold layer; anything else raises FileNotFoundError."""

    def fake_load(table: str, layer: str = "silver", **_: object) -> pd.DataFrame:
        assert layer == "gold", f"the gate read the {layer} layer, not gold"
        if table not in tables:
            raise FileNotFoundError(table)
        return tables[table]

    monkeypatch.setattr("data.storage.load_dataframe", fake_load)


def test_the_gate_fails_on_a_post_game_feature_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``final_score_diff`` feature in gold RAISES -- the scenario CR-02 named."""
    _patch_gold(
        monkeypatch,
        {
            "features_wp": _matrix(final_score_diff=[7.0]),
            "features_ats": _matrix(),
            "features_ou": _matrix(),
        },
    )
    with pytest.raises(RuntimeError, match="final_score_diff"):
        steps.step_validate_features()


@pytest.mark.parametrize("leaky_table", ["features_wp", "features_ats", "features_ou"])
def test_the_gate_scans_every_gold_matrix_not_just_the_first(
    monkeypatch: pytest.MonkeyPatch, leaky_table: str
) -> None:
    """Leakage in ANY of the three raises. The old ``break`` only ever reached the first."""
    tables = {
        name: _matrix() for name in ("features_wp", "features_ats", "features_ou")
    }
    tables[leaky_table] = _matrix(closing_line_total=[44.5])

    with pytest.raises(RuntimeError, match=leaky_table):
        _patch_gold(monkeypatch, tables)
        steps.step_validate_features()


def test_the_gate_does_not_fire_on_the_target_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``home_margin`` is the ATS LABEL, not a leak -- flagging it would break every run.

    This is the control that keeps the fix above from being a pipeline-wide hard-fail: the live
    ``features_ats`` matrix carries ``home_margin``, and the keyword list matches "margin".
    """
    _patch_gold(
        monkeypatch,
        {
            "features_wp": _matrix(),
            "features_ats": _matrix(),
            "features_ou": _matrix(),
        },
    )
    steps.step_validate_features()


def test_a_run_with_no_loadable_gold_matrix_raises_rather_than_passing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reporting success for a gate that never ran is the failure mode, not the safe default."""
    _patch_gold(monkeypatch, {})
    with pytest.raises(RuntimeError, match="did not run"):
        steps.step_validate_features()
