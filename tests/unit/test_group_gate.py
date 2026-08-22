"""Contract tests for the FROZEN Stage-1 pre-registration (Phase 30, SPEC R4, PROD-01).

These tests exist BEFORE the owner ratifies the pre-registration, and that ordering is
deliberate. ``backtest/group_gate_constants.py`` is one-way by construction (D30-05): once the
pre-registration commit lands, the git-ancestry assertion in Plan 30-13 fixes its content for the
remainder of the phase, so a bug found later cannot be fixed without destroying the evidence the
module exists to create. The owner therefore ratifies a rule whose stated behaviour has already
been demonstrated to be its actual behaviour.

Every assertion message carries its own remediation.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import itertools
import math
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
from scipy import stats
from scipy.stats import false_discovery_control

from backtest import group_gate, signal_lift
from backtest.diagnose import (
    CLV_COLUMN_FOR,
    MIN_CLV_SAMPLE,
    SIGNIFICANCE_ALPHA,
    clv_significance,
)
from backtest.group_gate import (
    build_parser,
    decide_group_verdicts,
    main,
    preregistration_commit,
    render_verdict_toml,
    run_group_gate,
)
from backtest.group_gate_constants import (
    ALPHA,
    BH_DENOMINATOR,
    CORRECTION_METHOD,
    EXCLUSION_INSUFFICIENT_PAIRED_SAMPLE,
    EXCLUSION_NO_PVALUE,
    EXCLUSION_NO_SELECTED_COLUMN,
    EXCLUSION_ZERO_COLUMNS_IN_GOLD,
    FIX_CYCLE_LEVERS,
    GRID_GROUPS,
    GRID_TARGETS,
    MDE_POWER,
    VERDICT_DROP,
    VERDICT_KEEP,
    VERDICT_NOT_MEASURED,
    VERDICT_UNDETERMINED,
    bh_family_exclusion_reason,
    bh_rank_order,
    bh_reject,
    mde_for_cell,
)
from backtest.signal_lift import ALL_REGISTERED_GROUPS

# The nine keys of the pre-registered grid, in the deterministic (group, target) order the
# tie-break uses. Built from the frozen vocabulary so a change to either tuple is caught here.
GRID_KEYS: list[tuple[str, str]] = sorted(itertools.product(GRID_GROUPS, GRID_TARGETS))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONSTANTS_PATH = _REPO_ROOT / "backtest" / "group_gate_constants.py"
_GATE_PATH = _REPO_ROOT / "backtest" / "group_gate.py"
_THIS_TEST_PATH = Path(__file__).resolve()


# ---------------------------------------------------------------------------
# (1) The frozen constants themselves
# ---------------------------------------------------------------------------


def test_alpha_is_the_imported_significance_constant_not_a_new_literal() -> None:
    """D30-08: alpha is backtest.diagnose.SIGNIFICANCE_ALPHA, imported, never re-declared."""
    assert ALPHA is SIGNIFICANCE_ALPHA, (
        "group_gate_constants.ALPHA must BE backtest.diagnose.SIGNIFICANCE_ALPHA (identity, not "
        "equality). Remediation: bind ALPHA to the imported symbol; do not write a float literal."
    )


def test_frozen_scalars_are_the_preregistered_values() -> None:
    assert MDE_POWER == 0.80, (
        "MDE_POWER is pre-registered at 0.80 (D30-07). Changing it after the measurement commit "
        "breaks the git-ancestry proof; changing it before requires owner re-ratification."
    )
    assert CORRECTION_METHOD == "benjamini-hochberg", (
        "CORRECTION_METHOD is pre-registered as benjamini-hochberg (SPEC R4)."
    )
    assert BH_DENOMINATOR == "full-grid", (
        "BH_DENOMINATOR is pre-registered as 'full-grid': the family is the full 9-cell grid "
        "(3 groups x 3 targets) and m is the count of MEASURED cells within it."
    )


def test_grid_vocabulary_is_the_preregistered_three_by_three() -> None:
    assert GRID_GROUPS == ("injury", "snap", "situational"), (
        "GRID_GROUPS is the three surviving Phase-28 groups. line_movement is deliberately absent "
        "(dropped from gold by Plan 30-07)."
    )
    assert GRID_TARGETS == ("wp", "ats", "ou"), (
        "GRID_TARGETS is the three model targets."
    )
    assert len(GRID_KEYS) == 9, (
        "The pre-registered grid is 3 groups x 3 targets = 9 cells; the SPEC's R4 text says "
        "'across the full 9-cell grid'."
    )


def test_verdict_vocabulary_is_four_valued_and_distinct() -> None:
    verdicts = (VERDICT_KEEP, VERDICT_DROP, VERDICT_UNDETERMINED, VERDICT_NOT_MEASURED)
    assert len(set(verdicts)) == 4, (
        "The verdict vocabulary must be four DISTINCT values. UNDETERMINED resolves to DROP for "
        "the deploy decision but is REPORTED as UNDETERMINED and never collapsed (SPEC R4/R8)."
    )


def test_fix_cycle_levers_names_exactly_one_lever_the_d25_05_window_widening() -> None:
    """D30-11: one permitted lever, named before anyone knows which target fails."""
    assert len(FIX_CYCLE_LEVERS) == 1, (
        f"Exactly ONE fix-cycle lever is pre-registered (D30-11); found {len(FIX_CYCLE_LEVERS)}. "
        "Adding a lever after seeing which target failed is precisely what the git-ancestry "
        "assertion detects."
    )
    lever = FIX_CYCLE_LEVERS[0].lower()
    assert "train-window" in lever or "train window" in lever, (
        "The single permitted lever is the D25-05 feature-selection TRAIN-WINDOW widening; the "
        f"lever string must name it. Found: {FIX_CYCLE_LEVERS[0]!r}"
    )
    assert "d25-05" in lever, (
        "The lever string must cite D25-05 so the readout can state exactly what was permitted. "
        f"Found: {FIX_CYCLE_LEVERS[0]!r}"
    )


def test_the_sample_size_floor_is_never_restated_in_this_phase() -> None:
    """The floor lives ONLY in backtest/diagnose.py and is imported (D24-13 parity).

    Assembled from fragments so this test's own source cannot match its own needle.
    """
    needles = ["MIN_CLV_SAMPLE" + " =", "MIN_CLV_SAMPLE" + ":"]
    for path in (_CONSTANTS_PATH, _THIS_TEST_PATH):
        source = path.read_text(encoding="utf-8")
        for needle in needles:
            assert needle not in source, (
                f"{path.name} re-declares the CLV sample-size floor ({needle!r}). Remediation: "
                "import MIN_CLV_SAMPLE from backtest.diagnose and read its live value; a restated "
                "literal silently forks from the primitive that actually enforces it."
            )


# ---------------------------------------------------------------------------
# (2) The vendored inclusive BH step-up
# ---------------------------------------------------------------------------


def _boundary_vector(m: int = 9, alpha: float = ALPHA) -> list[float]:
    """p_(i) sitting EXACTLY on each rank's inclusive boundary i*alpha/m."""
    return [(i + 1) * alpha / m for i in range(m)]


def test_bh_rejects_inclusively_at_every_rank_of_a_boundary_vector() -> None:
    """SPEC R4's literal form: reject at p_i <= i*alpha/m, INCLUSIVE.

    Every one of the nine p-values sits exactly on its own rank's boundary, so the inclusive
    comparison is exercised at all nine ranks. A '<' implementation rejects nothing here.
    """
    pvalues = _boundary_vector()
    rejected = bh_reject(pvalues, GRID_KEYS)
    assert all(rejected.values()), (
        "The vendored BH step-up must reject at EVERY rank when each p sits exactly on "
        f"i*alpha/m. Rejected={rejected}. Remediation: the comparison is '<=', not '<'."
    )


def test_scipy_adjusted_q_masks_the_boundary_artifact_on_this_particular_vector() -> (
    None
):
    """Records the CORRECTED account of the scipy divergence (verified live, scipy 1.17.1).

    The float artifact lives in scipy's ``ps *= m / i`` scaling: at m=9, alpha=0.05 the products
    ``p_3 * (9/3)`` and ``p_6 * (9/6)`` evaluate to 0.05000000000000001, i.e. > alpha, while the
    pre-registered ``p_i <= i*alpha/m`` is True. But scipy then applies a REVERSE CUMULATIVE
    MINIMUM, and on an all-ranks-on-the-boundary vector rank 5's smaller adjusted value
    (0.049999999999999996) propagates back over ranks 3 and 4, masking the artifact. So on THIS
    vector scipy happens to agree.

    The 30-RESEARCH.md Q5 / Pitfall-3 write-up asserted a disagreement here; that claim is about
    the pre-cumulative-minimum scaling, not about the value scipy returns. It is corrected in
    place rather than repeated, because an inaccurate sentence in a one-way module cannot be
    honestly fixed after the measurement commit. The divergence is REAL -- see the next test for
    the construction that surfaces it.
    """
    pvalues = _boundary_vector()
    for rank in (3, 6):
        scaled = pvalues[rank - 1] * (9 / rank)
        assert scaled > ALPHA, (
            f"The pre-cumulative-minimum scaling p_{rank} * (m/{rank}) is expected to exceed "
            f"alpha on float representation alone; got {scaled!r}."
        )
    qvalues = [float(q) for q in false_discovery_control(pvalues, method="bh")]
    assert all(q <= ALPHA for q in qvalues), (
        "On the all-ranks-on-the-boundary vector scipy's reverse cumulative minimum masks the "
        f"rank-3/rank-6 artifact and every q lands at or below alpha. Got {qvalues}."
    )


@pytest.mark.parametrize("boundary_rank", [3, 6])
def test_scipy_display_route_really_does_diverge_when_the_boundary_rank_is_last(
    boundary_rank: int,
) -> None:
    """The construction that surfaces the divergence (verified live, scipy 1.17.1).

    When the boundary cell is the LARGEST rank satisfying the inclusive comparison, no later rank
    has a smaller adjusted value, so the reverse cumulative minimum cannot rescue it. scipy then
    rejects NOTHING while the pre-registered rule rejects ranks 1..boundary_rank. This is why the
    binding decision is vendored and scipy is display-only.
    """
    m = 9
    boundary = boundary_rank * ALPHA / m
    pvalues = [
        boundary * (1.0 - 1e-9 * (boundary_rank - i)) for i in range(1, boundary_rank)
    ]
    pvalues.append(boundary)
    pvalues.extend([0.9] * (m - boundary_rank))
    pvalues.sort()

    rejected = bh_reject(pvalues, GRID_KEYS)
    n_rejected = sum(rejected.values())
    assert n_rejected == boundary_rank, (
        f"The pre-registered inclusive step-up must reject ranks 1..{boundary_rank}; got "
        f"{n_rejected}."
    )

    qvalues = [float(q) for q in false_discovery_control(pvalues, method="bh")]
    n_scipy = sum(q <= ALPHA for q in qvalues)
    assert n_scipy == 0, (
        f"scipy's adjusted-q route is expected to reject NOTHING here (q = {qvalues[0]!r} at "
        f"rank 1, just above alpha) while the pre-registered rule rejects {boundary_rank}. If "
        "this assertion fails, scipy's implementation changed -- re-verify the divergence claim "
        "recorded in backtest/group_gate_constants.py before touching anything else."
    )


def test_bh_is_a_step_up_not_a_step_down() -> None:
    """A failing rank 2 must NOT stop the walk: rank 5 passing rejects ranks 1-5."""
    pvalues = [0.0001, 0.02, 0.02, 0.02, 0.02, 0.9, 0.9, 0.9, 0.9]
    rejected = bh_reject(pvalues, GRID_KEYS)
    expected = {key: index < 5 for index, key in enumerate(GRID_KEYS)}
    assert rejected == expected, (
        "BH is a STEP-UP: record the LARGEST rank satisfying p_i <= i*alpha/m and reject it and "
        f"every rank below. Ranks 1-5 must be rejected here. Got {rejected}."
    )


def test_tied_pvalues_rank_by_the_group_target_lexicographic_key() -> None:
    """SPEC R4 adjacency: ties break on the deterministic (group, target) secondary key."""
    shuffled_keys = list(reversed(GRID_KEYS))
    pvalues = [0.01] * 9
    for _ in range(5):
        order = bh_rank_order(pvalues, shuffled_keys)
        assert order == GRID_KEYS, (
            "Tied p-values must rank in (group, target) lexicographic order on EVERY run. "
            "numpy's default argsort is unstable, so the secondary key must be explicit in the "
            f"sort key. Got {order}."
        )


def test_bh_rank_order_is_stable_and_deterministic_under_repetition() -> None:
    """SPEC R4 ordering: the sort is deterministic across runs on the same input."""
    pvalues = [0.04, 0.01, 0.04, 0.20, 0.01, 0.50, 0.20, 0.60, 0.04]
    orders = [bh_rank_order(pvalues, GRID_KEYS) for _ in range(5)]
    assert all(order == orders[0] for order in orders), (
        f"The BH rank order must be identical on repeated runs; got {orders}."
    )


def test_bh_agrees_with_the_scipy_display_route_away_from_every_boundary() -> None:
    """Positive control (T-30-17): a divergence away from a boundary is a REAL bug."""
    pvalues = [0.0001, 0.002, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rejected = bh_reject(pvalues, GRID_KEYS)
    qvalues = [float(q) for q in false_discovery_control(pvalues, method="bh")]
    m = len(pvalues)
    for index, (key, pvalue) in enumerate(zip(GRID_KEYS, pvalues)):
        rank = index + 1
        if abs(pvalue - rank * ALPHA / m) <= 1e-12:
            continue
        assert rejected[key] == (qvalues[index] <= ALPHA), (
            f"Away from the boundary the vendored decision and the scipy display route MUST "
            f"agree. They disagree at rank {rank} (p={pvalue!r}, q={qvalues[index]!r}). That is "
            "a real bug in the vendored step-up, not a rounding artifact."
        )


def test_bh_raises_on_a_nan_pvalue_rather_than_silently_never_rejecting() -> None:
    """T-30-18: a NaN must never reach the family. Fail loud, never coerce to 1.0."""
    pvalues = [0.001, float("nan"), 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    with pytest.raises(ValueError, match="NaN"):
        bh_reject(pvalues, GRID_KEYS)


def test_bh_on_an_empty_family_returns_an_empty_mapping() -> None:
    """Every cell excluded is a legal state (a grid where nothing was measurable)."""
    assert bh_reject([], []) == {}, (
        "An empty BH family must return an empty mapping, not raise -- a grid in which every "
        "cell is NOT MEASURED is a real outcome and must be reportable."
    )


def test_a_lone_nominally_significant_cell_does_not_survive_the_full_grid_denominator() -> (
    None
):
    """The tightening the owner ratifies at checkpoint 1, demonstrated rather than described.

    A single cell at p = 0.03 -- nominally significant, and a KEEP under the permissive Phase-28
    rule -- clears no rank once m = 9, because rank 1 requires p <= 1 * alpha / 9 = 0.00556.
    """
    pvalues = [0.03, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    rejected = bh_reject(pvalues, GRID_KEYS)
    assert not any(rejected.values()), (
        "With m=9 a lone p=0.03 must reject nothing: rank 1 needs p <= 0.00556 and every later "
        f"rank is worse. Got {rejected}."
    )


def test_bh_rejects_the_whole_grid_when_every_rank_clears_its_own_boundary() -> None:
    """Step-up, not per-rank: a uniform grid of p = 0.03 clears at rank 9 (0.03 <= 0.05).

    Recorded explicitly because it is counter-intuitive next to the test above -- the SAME
    p-value rejects nothing alone and rejects everything when all nine cells share it. That is
    the step-up rule working, not a bug.
    """
    rejected = bh_reject([0.03] * 9, GRID_KEYS)
    assert all(rejected.values()), (
        "Rank 9's boundary is 9 * alpha / 9 = alpha, so a uniform grid at p=0.03 clears at rank 9 "
        f"and the step-up rejects every rank below it. Got {rejected}."
    )


def test_bh_rejects_length_mismatch_between_pvalues_and_keys() -> None:
    with pytest.raises(ValueError, match="same length"):
        bh_reject([0.01, 0.02], [("injury", "wp")])


# ---------------------------------------------------------------------------
# (3) The MDE formula (D30-07)
# ---------------------------------------------------------------------------


def test_mde_returns_none_when_ci95_is_none() -> None:
    sig = {"n": 0, "mean": None, "t": None, "p": None, "ci95": None}
    assert mde_for_cell(sig) is None, (
        "A cell with no ci95 has no recoverable SE, so it has no MDE. Return None; never "
        "substitute a placeholder."
    )


def test_mde_returns_none_below_the_imported_sample_size_floor() -> None:
    """Below the floor clv_significance returns p/t/ci95 all None -- mde must not raise.

    The floor is read live from backtest.diagnose, never restated here.
    """
    short_series = [float(value) for value in range(MIN_CLV_SAMPLE - 1)]
    sig = clv_significance(short_series)
    assert sig["ci95"] is None, (
        "Contract drift: clv_significance is expected to return ci95=None below the floor. If "
        "this fails, the primitive changed and the MDE's None-branch no longer catches small n."
    )
    assert mde_for_cell(sig) is None, (
        "mde_for_cell must RETURN None (not raise) for a below-floor cell; that cell is routed "
        "to NOT MEASURED before any MDE is needed."
    )


def test_mde_agrees_with_the_mean_over_t_route_to_better_than_1e_9() -> None:
    """Route A (SE recovered from ci95) vs Route B (SE = mean / t), D30-07 / RESEARCH Q6."""
    series = [float(value) for value in range(20)]
    sig = clv_significance(series)
    assert sig["ci95"] is not None and sig["t"] is not None

    mde = mde_for_cell(sig)
    assert mde is not None

    degrees_of_freedom = sig["n"] - 1
    t_crit = float(stats.t.ppf(0.975, degrees_of_freedom))
    t_power = float(stats.t.ppf(MDE_POWER, degrees_of_freedom))
    route_b = (t_crit + t_power) * (sig["mean"] / sig["t"])

    assert abs(mde - route_b) < 1e-9, (
        f"The ci95-recovered SE and the mean/t SE must agree to better than 1e-9; got "
        f"mde={mde!r} vs route_b={route_b!r}. A disagreement means clv_significance's ci95 "
        "contract drifted away from its own t-statistic."
    )


def test_mde_is_positive_and_scales_with_the_ci_half_width() -> None:
    """The formula self-scales per cell -- the reason a single scalar MDE is unusable."""
    narrow = clv_significance([0.0100, 0.0101, 0.0099] * 7)
    wide = clv_significance([1.0, 2.0, 3.0] * 7)
    mde_narrow = mde_for_cell(narrow)
    mde_wide = mde_for_cell(wide)
    assert mde_narrow is not None and mde_wide is not None
    assert 0.0 < mde_narrow < mde_wide, (
        "A tighter CI must produce a smaller MDE. WP's probability_clv and ATS/OU's line_clv "
        "differ by roughly two orders of magnitude, so a single scalar MDE would make WP "
        f"unfalsifiable and ATS/OU untouchable. Got {mde_narrow!r} vs {mde_wide!r}."
    )


def test_mde_on_a_zero_variance_cell_is_zero_and_does_not_raise() -> None:
    """The realistic all-columns-discarded case: both legs identical, every delta exactly 0.0."""
    sig = clv_significance([0.0] * 20)
    assert math.isnan(sig["p"]), (
        "Contract drift: ttest_1samp on a zero-variance array is expected to return NaN p with "
        "no warning. If this fails, re-verify T-30-18's premise."
    )
    assert mde_for_cell(sig) == 0.0, (
        "A zero-variance cell has SE 0, so its MDE is 0.0. mde_for_cell must return it rather "
        "than raising -- the cell is separately routed to NOT MEASURED on its NaN p."
    )


# ---------------------------------------------------------------------------
# (4) The pure verdict rule -- synthetic grids matching signal_lift's cell contract
# ---------------------------------------------------------------------------

_UNSET = object()


def make_cell(
    target: str,
    delta_mean: float | None,
    delta_p: float | None,
    *,
    n_paired: int = 764,
    n_group_columns: int = 5,
    n_group_columns_selected: int = 3,
    half: float = 0.10,
    delta_ci95: object = _UNSET,
) -> dict:
    """Build one synthetic ``_screen_target`` cell, field for field.

    Mirrors ``backtest/signal_lift.py`` lines 557-572 exactly, including the derived
    ``keep_target`` / ``veto`` / ``paired_sufficient`` / ``measurable`` flags, so the verdict rule
    is driven by the same shape the real screen emits.
    """
    paired_sufficient = n_paired >= MIN_CLV_SAMPLE
    if delta_ci95 is _UNSET:
        delta_ci95 = (
            None
            if delta_mean is None or not paired_sufficient
            else (delta_mean - half, delta_mean + half)
        )
    has_p = delta_p is not None and not math.isnan(delta_p)
    return {
        "target": target,
        "clv_column": CLV_COLUMN_FOR[target],
        "n_paired": int(n_paired),
        "delta_mean": delta_mean,
        "delta_t": None if delta_mean is None else 3.0,
        "delta_p": delta_p,
        "delta_ci95": delta_ci95,
        "keep_target": bool(delta_mean is not None and delta_mean > 0),
        "veto": bool(
            delta_mean is not None
            and delta_mean < 0
            and has_p
            and delta_p < SIGNIFICANCE_ALPHA
        ),
        "n_group_columns": n_group_columns,
        "n_group_columns_selected": n_group_columns_selected,
        "group_columns_selected": [
            f"col_{index}" for index in range(n_group_columns_selected)
        ],
        "paired_sufficient": paired_sufficient,
        "measurable": bool(n_group_columns_selected) and paired_sufficient,
    }


def make_screen(cells_by_group: dict[str, dict[str, dict]]) -> dict:
    """Wrap per-group cells in the ``run_signal_lift_screen`` return shape."""
    groups_out: dict[str, dict] = {}
    for group, per_target in cells_by_group.items():
        measurable_targets = [t for t, cell in per_target.items() if cell["measurable"]]
        groups_out[group] = {
            "coverage_span": "full history",
            "per_target": per_target,
            "decision": {"keep": True, "reason": "synthetic permissive D-05 ruling"},
            "measurability": {
                "measurable": bool(measurable_targets),
                "measurable_targets": measurable_targets,
                "note": ""
                if measurable_targets
                else (
                    "NOT MEASURED: no column of this group was selected by ANY target's feature "
                    "selector, so no candidate model ever saw the group."
                ),
            },
        }
    return {
        "measure_window": "2021-2024",
        "alpha": SIGNIFICANCE_ALPHA,
        "anchor": "BaseTrainer.train_and_evaluate(tune=False)",
        "baseline_excludes": list(ALL_REGISTERED_GROUPS),
        "multiplicity_note": "synthetic",
        "requirement": "PROD-01",
        "targets": list(GRID_TARGETS),
        "groups": groups_out,
    }


def uniform_grid(**cell_kwargs) -> dict:
    """A full 3x3 grid in which every cell shares the same construction."""
    return make_screen(
        {
            group: {target: make_cell(target, **cell_kwargs) for target in GRID_TARGETS}
            for group in GRID_GROUPS
        }
    )


def grid_with_injury_override(injury_overrides: dict[str, dict], **base_kwargs) -> dict:
    """A full grid whose ``injury`` row is overridden per target."""
    cells_by_group: dict[str, dict[str, dict]] = {}
    for group in GRID_GROUPS:
        per_target: dict[str, dict] = {}
        for target in GRID_TARGETS:
            kwargs = dict(base_kwargs)
            if group == "injury":
                kwargs.update(injury_overrides.get(target, {}))
            per_target[target] = make_cell(target, **kwargs)
        cells_by_group[group] = per_target
    return make_screen(cells_by_group)


def all_targets(**overrides) -> dict[str, dict]:
    """The same override applied to every target of the injury row."""
    return {target: dict(overrides) for target in GRID_TARGETS}


# --- denominator -----------------------------------------------------------


def test_bh_denominator_is_nine_when_no_cell_is_excluded() -> None:
    result = decide_group_verdicts(uniform_grid(delta_mean=0.01, delta_p=0.9))
    assert result["bh_denominator"] == 9, (
        "The family is the full 9-cell grid and m is the count of MEASURED cells within it, "
        f"which is 9 when nothing is excluded. Got {result['bh_denominator']}."
    )
    for group, verdict in result["verdicts"].items():
        assert verdict["bh_denominator"] == 9, (
            f"Group {group} must report the FAMILY denominator (the grid-wide m), not a "
            "per-group count -- the correction is applied across the full grid."
        )


def test_bh_denominator_equals_the_measured_cell_count_when_cells_are_excluded() -> (
    None
):
    screen = grid_with_injury_override(
        all_targets(n_group_columns=0), delta_mean=0.01, delta_p=0.9
    )
    result = decide_group_verdicts(screen)
    assert result["bh_denominator"] == 6, (
        "m is the count of MEASURED cells in the full-grid family. Three injury cells have zero "
        f"columns in gold, so m must be 6. Got {result['bh_denominator']}."
    )


def test_every_group_reports_the_frozen_alpha_and_correction_method() -> None:
    result = decide_group_verdicts(uniform_grid(delta_mean=0.01, delta_p=0.9))
    assert result["alpha"] is ALPHA
    assert result["correction_method"] == CORRECTION_METHOD
    assert result["bh_denominator_rule"] == BH_DENOMINATOR


# --- the NOT MEASURED arm --------------------------------------------------


def test_zero_columns_short_circuits_to_not_measured_before_p_is_inspected() -> None:
    """SPEC R4: a group with zero columns present in gold reports NOT MEASURED, never DROP.

    The p-value here (1e-9) would otherwise clear rank 1 and produce a KEEP, so this pins the
    CHECK ORDER, not merely the outcome.
    """
    screen = grid_with_injury_override(
        all_targets(n_group_columns=0, delta_p=1e-9, delta_mean=0.5),
        delta_mean=0.01,
        delta_p=0.9,
    )
    result = decide_group_verdicts(screen)
    injury = result["verdicts"]["injury"]
    assert injury["verdict"] == VERDICT_NOT_MEASURED, (
        "SPEC R4: a group with zero columns present in gold reports NOT MEASURED, never DROP and "
        f"never KEEP. Got {injury['verdict']}."
    )
    for target in GRID_TARGETS:
        cell = injury["cells"][target]
        assert cell["exclusion_reason"] == EXCLUSION_ZERO_COLUMNS_IN_GOLD, (
            "The zero-columns check must run FIRST, before the p-value is inspected at all. Got "
            f"{cell['exclusion_reason']!r} for {target}."
        )
        assert cell["bh_rejected"] is False


def test_insufficient_paired_sample_is_not_measured() -> None:
    screen = grid_with_injury_override(
        all_targets(n_paired=MIN_CLV_SAMPLE - 1), delta_mean=0.01, delta_p=0.9
    )
    injury = decide_group_verdicts(screen)["verdicts"]["injury"]
    assert injury["verdict"] == VERDICT_NOT_MEASURED
    for target in GRID_TARGETS:
        assert (
            injury["cells"][target]["exclusion_reason"]
            == EXCLUSION_INSUFFICIENT_PAIRED_SAMPLE
        )


def test_nan_p_is_not_measured_and_is_never_coerced() -> None:
    """T-30-18: the zero-variance case is REAL, not theoretical."""
    screen = grid_with_injury_override(
        all_targets(delta_mean=0.0, delta_p=float("nan")), delta_mean=0.01, delta_p=0.9
    )
    result = decide_group_verdicts(screen)
    injury = result["verdicts"]["injury"]
    assert injury["verdict"] == VERDICT_NOT_MEASURED
    for target in GRID_TARGETS:
        cell = injury["cells"][target]
        assert cell["exclusion_reason"] == EXCLUSION_NO_PVALUE
        assert cell["delta_p"] is None or math.isnan(cell["delta_p"]), (
            "A NaN p must be reported as-is or as None -- NEVER coerced to 1.0, which would "
            f"manufacture evidence for a cell that has none. Got {cell['delta_p']!r}."
        )
    assert result["bh_denominator"] == 6


def test_selection_churn_cell_is_excluded_from_the_verdict_but_retained_in_the_output() -> (
    None
):
    screen = grid_with_injury_override(
        {"wp": {"n_group_columns_selected": 0, "delta_mean": 0.42, "delta_p": 1e-9}},
        delta_mean=0.01,
        delta_p=0.9,
    )
    result = decide_group_verdicts(screen)
    cell = result["verdicts"]["injury"]["cells"]["wp"]
    assert cell["exclusion_reason"] == EXCLUSION_NO_SELECTED_COLUMN
    assert cell["selection_churn"] is True, (
        "A cell whose group contributed no SELECTED column is selection churn among the other "
        "features, not this group's lift. It must be labelled as such."
    )
    assert cell["delta_mean"] == 0.42, (
        "The churn cell's delta is RETAINED in the output (labelled as churn) so the readout can "
        "publish the raw grid beside the corrected one."
    )
    assert cell["bh_rejected"] is False
    assert result["bh_denominator"] == 8


def test_group_with_every_cell_churn_returns_not_measured_without_raising() -> None:
    """The realistic cap-saturated-trainer shape (F10). Assert on the VERDICT, never on a raise.

    ``pytest.raises`` is deliberately NOT used: if an exception escapes ``mde_for_cell`` or the BH
    assembly, this test must fail loudly rather than pass by catching it.
    """
    screen = grid_with_injury_override(
        all_targets(n_group_columns_selected=0, delta_mean=0.3, delta_p=0.4),
        delta_mean=0.01,
        delta_p=0.9,
    )
    injury = decide_group_verdicts(screen)["verdicts"]["injury"]
    assert injury["verdict"] == VERDICT_NOT_MEASURED, (
        "A group whose every cell had zero SELECTED columns saw no candidate model at all. The "
        f"call must RETURN NOT MEASURED, not raise and not rule. Got {injury['verdict']}."
    )


def test_group_with_no_measured_cell_is_not_measured_and_never_drop() -> None:
    """SPEC R4: NOT MEASURED is a refusal to rule; DROP is a substantive negative finding."""
    screen = grid_with_injury_override(
        all_targets(n_group_columns=0), delta_mean=0.01, delta_p=0.9
    )
    injury = decide_group_verdicts(screen)["verdicts"]["injury"]
    assert injury["verdict"] != VERDICT_DROP, (
        "SPEC R4 forbids publishing DROP for a group that measured nothing -- that would report "
        "a substantive negative finding for an empty measurement."
    )
    assert injury["verdict"] == VERDICT_NOT_MEASURED


# --- the KEEP / DROP / UNDETERMINED arms -----------------------------------


def test_bh_rejected_positive_is_keep() -> None:
    screen = grid_with_injury_override(
        {"wp": {"delta_mean": 0.5, "delta_p": 0.0001}}, delta_mean=0.01, delta_p=0.9
    )
    result = decide_group_verdicts(screen)
    assert result["verdicts"]["injury"]["verdict"] == VERDICT_KEEP
    assert result["verdicts"]["injury"]["cells"]["wp"]["bh_rejected"] is True


def test_bh_rejected_negative_is_drop() -> None:
    screen = grid_with_injury_override(
        {"wp": {"delta_mean": -0.5, "delta_p": 0.0001}}, delta_mean=0.01, delta_p=0.9
    )
    result = decide_group_verdicts(screen)
    assert result["verdicts"]["injury"]["verdict"] == VERDICT_DROP
    assert result["verdicts"]["injury"]["cells"]["wp"]["bh_rejected"] is True


def test_keep_takes_precedence_when_a_group_is_rejected_in_both_directions() -> None:
    """The pre-registered arm ORDER is KEEP, DROP, UNDETERMINED, NOT MEASURED.

    Pinned explicitly because it REVERSES the permissive Phase-28 rule, under which a
    significantly-negative target vetoed the whole group regardless of any positive cell.
    """
    screen = grid_with_injury_override(
        {
            "wp": {"delta_mean": 0.5, "delta_p": 0.0001},
            "ats": {"delta_mean": -0.5, "delta_p": 0.001},
        },
        delta_mean=0.01,
        delta_p=0.9,
    )
    injury = decide_group_verdicts(screen)["verdicts"]["injury"]
    assert injury["cells"]["wp"]["bh_rejected"] is True
    assert injury["cells"]["ats"]["bh_rejected"] is True
    assert injury["verdict"] == VERDICT_KEEP, (
        "The pre-registered arm order puts KEEP first, so a group rejected in both directions is "
        f"KEEP. Got {injury['verdict']}."
    )


def test_no_positive_point_estimate_anywhere_is_drop() -> None:
    screen = grid_with_injury_override(
        all_targets(delta_mean=-0.01, delta_p=0.9), delta_mean=0.01, delta_p=0.9
    )
    injury = decide_group_verdicts(screen)["verdicts"]["injury"]
    assert injury["verdict"] == VERDICT_DROP
    assert "positive" in injury["reason"].lower()


def test_delta_exactly_zero_is_not_positive() -> None:
    """SPEC R4 boundary: a delta of exactly 0.0 is not positive. No epsilon anywhere."""
    screen = grid_with_injury_override(
        all_targets(delta_mean=0.0, delta_p=0.9), delta_mean=0.01, delta_p=0.9
    )
    injury = decide_group_verdicts(screen)["verdicts"]["injury"]
    assert injury["verdict"] == VERDICT_DROP, (
        "delta_mean == 0.0 must fail the strict `> 0.0` positivity test, so a flat group is DROP "
        f"and not UNDETERMINED. Got {injury['verdict']}."
    )
    assert injury["largest_positive_effect"] is None


def test_positive_but_unrejected_is_undetermined() -> None:
    screen = grid_with_injury_override(
        all_targets(delta_mean=0.01, delta_p=0.20), delta_mean=-0.01, delta_p=0.9
    )
    injury = decide_group_verdicts(screen)["verdicts"]["injury"]
    assert injury["verdict"] == VERDICT_UNDETERMINED, (
        "A positive point estimate that survives no BH rejection is UNDETERMINED -- 'we could "
        f"not tell', not 'it did not help'. Got {injury['verdict']}."
    )


def test_effect_exactly_at_the_mde_is_undetermined() -> None:
    """SPEC R4 boundary: the largest effect exactly at its cell's MDE lands in UNDETERMINED."""
    n_paired = 100
    half = 0.05
    t_crit = float(stats.t.ppf(0.975, n_paired - 1))
    standard_error = half / t_crit
    exact_mde = (t_crit + float(stats.t.ppf(MDE_POWER, n_paired - 1))) * standard_error

    screen = grid_with_injury_override(
        {
            "wp": {
                "delta_mean": exact_mde,
                "delta_p": 0.9,
                "n_paired": n_paired,
                "delta_ci95": (exact_mde - half, exact_mde + half),
            }
        },
        delta_mean=-0.01,
        delta_p=0.9,
    )
    injury = decide_group_verdicts(screen)["verdicts"]["injury"]
    cell = injury["cells"]["wp"]
    assert cell["mde"] == pytest.approx(exact_mde, rel=1e-12), (
        f"The reported MDE must be the frozen formula's value; got {cell['mde']!r} vs "
        f"{exact_mde!r}."
    )
    assert injury["verdict"] == VERDICT_UNDETERMINED
    assert injury["largest_positive_effect"]["at_or_below_mde"] is True, (
        "The UNDETERMINED band is written so that an effect EXACTLY equal to its cell's MDE "
        "lands inside it (inclusive at the MDE), never above it."
    )


# --- structure, determinism and the no-re-derivation invariant -------------


def test_screen_is_passed_through_unmodified() -> None:
    screen = uniform_grid(delta_mean=0.01, delta_p=0.9)
    snapshot = repr(screen)
    result = decide_group_verdicts(screen)
    assert result["screen"] is screen, (
        "The judge passes the raw screen output through unmodified alongside the verdicts, so "
        "the readout can publish the raw grid beside the corrected one (the evaluate_target "
        "discipline: every metric field was produced upstream)."
    )
    assert repr(screen) == snapshot, (
        "decide_group_verdicts must not mutate the screen dict."
    )


def test_display_q_values_are_present_for_measured_cells_and_labelled_display_only() -> (
    None
):
    result = decide_group_verdicts(uniform_grid(delta_mean=0.01, delta_p=0.9))
    assert result["q_values_are_display_only"] is True, (
        "scipy's adjusted q-values are a DISPLAY column. The binding decision is the vendored "
        "inclusive step-up; the structure must say so."
    )
    for verdict in result["verdicts"].values():
        for cell in verdict["cells"].values():
            assert cell["q_display"] is not None


def test_verdicts_are_deterministic_across_repeated_calls() -> None:
    screen = grid_with_injury_override(
        {"wp": {"delta_mean": 0.5, "delta_p": 0.0001}}, delta_mean=0.01, delta_p=0.9
    )
    first = decide_group_verdicts(screen)["verdicts"]
    second = decide_group_verdicts(screen)["verdicts"]
    assert first == second, (
        "The verdict rule is pure and must be reproducible run to run."
    )


def test_the_judge_redeclares_no_primitive() -> None:
    """T-30-15 / D24-13: every vocabulary symbol is imported, never re-declared."""
    source = _GATE_PATH.read_text(encoding="utf-8")
    needles = [
        "def clv_significance",
        "def _screen_target",
        "SIGNIFICANCE_ALPHA" + " = ",
    ]
    for needle in needles:
        assert needle not in source, (
            f"backtest/group_gate.py re-declares {needle!r}. The judge owns no statistic of its "
            "own: import the primitive from backtest.diagnose instead."
        )


def test_exclusion_predicate_checks_zero_columns_before_everything_else() -> None:
    """Direct predicate test -- the check order is itself pre-registered."""
    cell = make_cell(
        "wp",
        0.0,
        float("nan"),
        n_paired=MIN_CLV_SAMPLE - 1,
        n_group_columns=0,
        n_group_columns_selected=0,
    )
    assert bh_family_exclusion_reason(cell) == EXCLUSION_ZERO_COLUMNS_IN_GOLD, (
        "A cell that trips every exclusion at once must report the ZERO-COLUMNS reason, because "
        "that is the branch SPEC R4 names and the one the readout must publish."
    )
    assert bh_family_exclusion_reason(make_cell("wp", 0.01, 0.5)) is None


# ---------------------------------------------------------------------------
# (5) The Stage-1 ORCHESTRATOR: the baseline PIN and the ratified-verdict block
# ---------------------------------------------------------------------------
#
# These cover Plan 30-05's additions, which live OUTSIDE the frozen pre-registration on
# purpose: a bug in the mechanics must be fixable without touching the module whose
# last-modifying commit is the git-ancestry anchor.


def _mixed_verdict_screen(snap_zero_columns: bool = False) -> dict:
    """A synthetic 3x3 screen carrying three DIFFERENT verdicts at once.

    ``injury`` is KEEP (its WP cell survives the correction), ``situational`` is UNDETERMINED
    (positive everywhere, nothing survives), and ``snap`` is either DROP (no positive point
    estimate anywhere) or NOT MEASURED (zero columns of the group are present in gold).
    """
    cells_by_group: dict[str, dict[str, dict]] = {
        "injury": {
            "wp": make_cell("wp", 0.5, 0.0001),
            "ats": make_cell("ats", 0.01, 0.9),
            "ou": make_cell("ou", 0.01, 0.9),
        },
        "situational": {
            target: make_cell(target, 0.01, 0.9) for target in GRID_TARGETS
        },
    }
    if snap_zero_columns:
        cells_by_group["snap"] = {
            target: make_cell(
                target,
                0.0,
                float("nan"),
                n_group_columns=0,
                n_group_columns_selected=0,
            )
            for target in GRID_TARGETS
        }
    else:
        cells_by_group["snap"] = {
            target: make_cell(target, -0.01, 0.9) for target in GRID_TARGETS
        }
    return make_screen(cells_by_group)


_FAKE_COMMIT = "0" * 40


def _gate_result(snap_zero_columns: bool = False) -> dict:
    """A judged result with a FIXED pre-registration SHA, so rendering tests stay hermetic."""
    result = decide_group_verdicts(_mixed_verdict_screen(snap_zero_columns))
    result["preregistration_commit"] = _FAKE_COMMIT
    return result


# --- the baseline PIN ------------------------------------------------------


def test_run_group_gate_pins_the_baseline_to_every_registered_group(
    monkeypatch,
) -> None:
    """T-30-15: the baseline leg excludes EVERY registered group, not the three-name default.

    Written as MEMBERSHIP against the live registry rather than as a literal tuple. A literal
    would still pass after a Phase-31 group is registered while silently no longer meaning
    "gold minus every signal column we know about" -- which is exactly the failure mode this
    pin exists to prevent (backtest/signal_lift.py:345-367, the 29-06 incident).
    """
    captured: dict[str, object] = {}
    sentinel_screen = _mixed_verdict_screen()

    def _spy(**kwargs) -> dict:
        captured.update(kwargs)
        return sentinel_screen

    monkeypatch.setattr(group_gate, "run_signal_lift_screen", _spy)
    run_group_gate()

    pinned = captured["baseline_exclude_groups"]
    assert set(pinned) == set(signal_lift._GROUP_PREDICATE), (
        "run_group_gate must pass baseline_exclude_groups=ALL_REGISTERED_GROUPS -- the FULL "
        "registered set derived from _GROUP_PREDICATE. The module default is GROUPS, a "
        "three-name deny-list, and a deny-list cannot name a group that does not exist yet."
    )
    assert set(signal_lift.GROUPS) < set(pinned), (
        "The pin must be a STRICT superset of the three screened Phase-28 groups; a later "
        "phase's group must be excluded from the baseline automatically."
    )
    assert "line_movement" in pinned, (
        "line_movement stays in the pin even after Plan 30-07 removes its columns from gold: "
        "group_columns then returns [] and the family contributes nothing, which is correct."
    )


def test_run_group_gate_measures_the_frozen_grid(monkeypatch) -> None:
    """The targets and groups come from the FROZEN constants, never from the module defaults."""
    captured: dict[str, object] = {}

    def _spy(**kwargs) -> dict:
        captured.update(kwargs)
        return _mixed_verdict_screen()

    monkeypatch.setattr(group_gate, "run_signal_lift_screen", _spy)
    run_group_gate()

    assert tuple(captured["targets"]) == GRID_TARGETS, (
        "run_group_gate must screen the pre-registered GRID_TARGETS."
    )
    assert tuple(captured["groups"]) == GRID_GROUPS, (
        "run_group_gate must screen the pre-registered GRID_GROUPS."
    )


def test_run_group_gate_passes_the_screen_through_unmodified(monkeypatch) -> None:
    """The raw screen output survives the orchestrator untouched (SPEC R8 raw-beside-corrected)."""
    sentinel_screen = _mixed_verdict_screen()
    monkeypatch.setattr(
        group_gate, "run_signal_lift_screen", lambda **_kwargs: sentinel_screen
    )
    result = run_group_gate()

    assert result["screen"] is sentinel_screen, (
        "run_group_gate must return the screen dict it received, unmodified, so the readout "
        "can publish the raw grid beside the corrected one."
    )
    assert set(result["verdicts"]) == set(GRID_GROUPS), (
        "run_group_gate must return one verdict per screened group."
    )
    assert result["verdicts"]["injury"]["verdict"] == VERDICT_KEEP


def test_run_group_gate_records_the_preregistration_commit(monkeypatch) -> None:
    """The result carries the frozen rule's anchor SHA, so the JSON record is self-describing."""
    monkeypatch.setattr(
        group_gate, "run_signal_lift_screen", lambda **_kwargs: _mixed_verdict_screen()
    )
    result = run_group_gate()
    assert re.fullmatch(r"[0-9a-f]{40}", result["preregistration_commit"]), (
        "run_group_gate must record the last-modifying commit of "
        "backtest/group_gate_constants.py, the SPEC R4 ancestry anchor."
    )


def test_the_pin_is_written_as_the_derived_symbol_not_the_module_default() -> None:
    """Source-scan: the pin is spelled with the DERIVED symbol, never re-typed as a literal."""
    source = _GATE_PATH.read_text(encoding="utf-8")
    needle = "baseline_exclude_groups=" + "ALL_REGISTERED_GROUPS"
    assert source.count(needle) >= 1, (
        f"backtest/group_gate.py must pass {needle} explicitly. Relying on the module default "
        "silently measures each group against a baseline that already contains every group "
        "registered after Phase 28."
    )


def test_the_preregistration_commit_resolves_from_git() -> None:
    """The anchor SHA is RESOLVED from git, never transcribed into the source."""
    resolved = preregistration_commit()
    expected = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", "backtest/group_gate_constants.py"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert resolved == expected, (
        "preregistration_commit must return git's own answer for the frozen rule module's "
        f"last-modifying commit. Got {resolved}, git says {expected}."
    )
    assert re.fullmatch(r"[0-9a-f]{40}", resolved), (
        "The anchor must be a full 40-character SHA so Plan 30-13's strict-ancestor assertion "
        "has an unambiguous target."
    )


# --- the ratified-verdict block --------------------------------------------


def test_rendered_block_parses_as_toml() -> None:
    parsed = tomllib.loads(render_verdict_toml(_gate_result()))
    assert "excluded_groups" in parsed, (
        "excluded_groups must be a TOP-LEVEL key: scripts/promote_models.py reads "
        "verdict.get('excluded_groups', []) off the parsed document root."
    )
    assert parsed["stage1"]["bh_denominator"] == 9
    assert parsed["stage1"]["correction_method"] == CORRECTION_METHOD
    assert parsed["stage1"]["bh_denominator_rule"] == BH_DENOMINATOR
    assert parsed["stage1"]["preregistration_commit"] == _FAKE_COMMIT


def test_excluded_groups_is_drop_plus_undetermined() -> None:
    result = _gate_result()
    parsed = tomllib.loads(render_verdict_toml(result))
    verdicts = result["verdicts"]
    expected = sorted(
        group
        for group, entry in verdicts.items()
        if entry["verdict"]
        in (VERDICT_DROP, VERDICT_UNDETERMINED, VERDICT_NOT_MEASURED)
    )
    assert verdicts["snap"]["verdict"] == VERDICT_DROP
    assert verdicts["situational"]["verdict"] == VERDICT_UNDETERMINED
    assert parsed["excluded_groups"] == expected == ["situational", "snap"], (
        "excluded_groups is DERIVED as DROP + UNDETERMINED + NOT MEASURED, so the Stage-2 "
        "exclusion list can never be a transcription (T-30-26)."
    )
    assert "injury" not in parsed["excluded_groups"], (
        "A KEEP group must never appear in the exclusion list."
    )


def test_excluded_groups_includes_a_not_measured_group() -> None:
    result = _gate_result(snap_zero_columns=True)
    parsed = tomllib.loads(render_verdict_toml(result))
    assert result["verdicts"]["snap"]["verdict"] == VERDICT_NOT_MEASURED
    assert parsed["excluded_groups"] == ["situational", "snap"], (
        "A NOT MEASURED group produced no evidence for carrying it, so it is excluded -- and "
        "is REPORTED as NOT MEASURED, never as DROP."
    )


def test_each_excluded_group_carries_its_verdict_word_in_the_comment() -> None:
    for zero_columns in (False, True):
        result = _gate_result(snap_zero_columns=zero_columns)
        block = render_verdict_toml(result)
        comment = block.split("excluded_groups =")[0]
        for group in tomllib.loads(block)["excluded_groups"]:
            word = result["verdicts"][group]["verdict"]
            assert re.search(rf"^#.*\b{group}\b.*{re.escape(word)}", comment, re.M), (
                f"The comment above excluded_groups must name '{group}' with its own verdict "
                f"word '{word}', so a reader cannot mistake the exclusion list for a collapse "
                "of the three-valued vocabulary (SPEC R4/R8, T-30-19)."
            )


def test_the_undetermined_token_survives_into_the_rendered_block() -> None:
    block = render_verdict_toml(_gate_result())
    assert VERDICT_UNDETERMINED in block, (
        "A group carrying the UNDETERMINED verdict must have that literal word in the rendered "
        "block. UNDETERMINED resolves to DROP for the DEPLOY decision and is nonetheless "
        "REPORTED as UNDETERMINED (SPEC R4/R8)."
    )
    assert tomllib.loads(block)["stage1"]["verdicts"]["situational"]["verdict"] == (
        VERDICT_UNDETERMINED
    )
    assert "resolves to DROP" in block, (
        "The block must carry the prose resolution rule in its own comment, so the config file "
        "explains itself without the readout."
    )


def test_every_cell_reports_its_delta_p_rejection_rank_q_and_mde() -> None:
    parsed = tomllib.loads(render_verdict_toml(_gate_result()))
    cell = parsed["stage1"]["cells"]["injury"]["wp"]
    for key in ("delta_mean", "delta_p", "bh_rejected", "bh_rank", "q_display", "mde"):
        assert key in cell, f"Cell block is missing '{key}'."
    assert cell["bh_rejected"] is True
    assert cell["bh_rank"] == 1
    assert cell["delta_mean"] == pytest.approx(0.5)
    assert cell["delta_p"] == pytest.approx(0.0001)


def test_alpha_mde_power_and_denominator_are_all_emitted() -> None:
    parsed = tomllib.loads(render_verdict_toml(_gate_result()))["stage1"]
    assert parsed["alpha"] == pytest.approx(ALPHA)
    assert parsed["mde_power"] == pytest.approx(MDE_POWER)
    assert parsed["bh_denominator"] == 9, (
        "The BH denominator ACTUALLY USED must be recorded, not just the rule that produced it."
    )


_NON_COMMENT_FLOAT = re.compile(
    r"(?<![\w.])-?(?:\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)"
)


def test_every_float_is_emitted_through_the_fixed_precision_specifier() -> None:
    """T-30-54: no float reaches the block through Python's default repr.

    The default float repr is shortest-round-trip and its output is not contractually stable
    across platforms or patch releases. Plan 30-10 asserts config/group_gate_verdict.toml is
    byte-identical to this generator's output, so rendering drift there would read as
    tampering with a committed measurement artifact.
    """
    assert group_gate._FLOAT_FORMAT == ".17g", (
        "The precision must be an EXPLICIT specifier constant. .17g is the precision at which "
        "no two distinct IEEE-754 doubles can collide, so a fixed number of decimal places "
        "(which would flatten a 1e-30 p-value to 0.000000000000) is not an option here."
    )
    block = render_verdict_toml(_gate_result())
    tokens = [
        token
        for line in block.splitlines()
        if not line.lstrip().startswith("#")
        for token in _NON_COMMENT_FLOAT.findall(line)
    ]
    assert tokens, "The block must contain at least one float."
    for token in tokens:
        assert token == group_gate._fmt_float(float(token)), (
            f"Float token {token} is not the fixed-precision rendering of its own value. "
            "Every float must go through _fmt_float; never interpolate a bare float."
        )


def test_rendering_the_same_structure_twice_is_byte_identical() -> None:
    result = _gate_result()
    assert render_verdict_toml(result) == render_verdict_toml(result), (
        "render_verdict_toml must be deterministic: same structure in, byte-identical text out."
    )


def test_the_rendering_path_uses_no_repr() -> None:
    source = _GATE_PATH.read_text(encoding="utf-8")
    needle = "re" + "pr("
    assert source.count(needle) == 0, (
        f"backtest/group_gate.py must not call {needle} anywhere in the rendering path "
        "(T-30-54)."
    )


def test_render_verdict_toml_writes_no_file(monkeypatch) -> None:
    """The generator PRINTS; the human block-pastes. It never writes config/ (D24-07)."""

    def _forbidden(*_args, **_kwargs):
        message = "render_verdict_toml must not open or write any file"
        raise AssertionError(message)

    monkeypatch.setattr(Path, "write_text", _forbidden)
    monkeypatch.setattr(Path, "write_bytes", _forbidden)
    monkeypatch.setattr(Path, "open", _forbidden)
    monkeypatch.setattr("builtins.open", _forbidden)

    block = render_verdict_toml(_gate_result())
    assert isinstance(block, str)
    assert block.endswith("\n")


def test_the_block_says_it_is_generator_output_and_names_the_anchor() -> None:
    block = render_verdict_toml(_gate_result())
    lowered = block.lower()
    assert "generator output" in lowered and "hand-edit" in lowered, (
        "The block must state that it is generator output and must never be hand-edited "
        "(D24-07 block-paste discipline)."
    )
    assert _FAKE_COMMIT in block, (
        "The block must carry the frozen rule module's last-modifying commit SHA."
    )
    assert block.isascii(), "ASCII only, no emoji (CLAUDE.md hard constraint)."


def test_the_block_is_a_complete_paste_covering_every_screened_group() -> None:
    parsed = tomllib.loads(render_verdict_toml(_gate_result()))
    assert set(parsed["stage1"]["verdicts"]) == set(GRID_GROUPS), (
        "The block covers every screened group in ONE paste, so a transcription cannot drop a "
        "group (D24-07)."
    )
    assert set(parsed["stage1"]["cells"]) == set(GRID_GROUPS)


# --- the CLI ---------------------------------------------------------------


def test_build_parser_defaults_the_output_under_outputs() -> None:
    args = build_parser().parse_args([])
    assert Path(args.output).parts[0] == "outputs", (
        "The CLI's default result path must sit under outputs/, never under data/."
    )
    assert "group_gate" in Path(args.output).parts


def test_the_cli_refuses_an_output_path_under_data() -> None:
    """T-30-14: the no-writes-under-data prohibition is enforced BEFORE anything runs."""
    with pytest.raises(ValueError, match="data/") as excinfo:
        main(["--output", "data/gold/group_gate_result.json"])
    message = str(excinfo.value)
    assert "data/" in message, "The error must name the prohibited location."
    assert "outputs/" in message, (
        "The error must name where the result belongs instead."
    )


def test_the_data_path_refusal_accepts_a_path_outside_data(tmp_path) -> None:
    accepted = group_gate._reject_data_path(tmp_path / "result.json")
    assert accepted == (tmp_path / "result.json").resolve()
