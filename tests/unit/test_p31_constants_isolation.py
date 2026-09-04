"""Phase-31 code cannot silently inherit the Phase-27 window, or declare a second alpha.

WHAT THIS GUARDS, AND WHY IT IS AN AST SCAN RATHER THAN A RUNTIME CHECK
----------------------------------------------------------------------
D31-14 puts the widened Phase-31 window in a NEW frozen module and leaves
``backtest/ou_ev_chain.TUNE_SEASONS`` / ``HOLD_SEASONS`` untouched as the Phase-27 historical
record. Two window definitions therefore live in one tree, which is the 29-06 second-list risk
this project has already paid for once.

The concrete mechanism, named by ``31-RESEARCH.md``: ``backtest/ou_monetization._assert_fit_window``
READS the Phase-27 constants. A Phase-31 runner that reuses it inherits the Phase-27 window
SILENTLY -- the fence still passes, and the fence report names the WRONG hold seasons. Nothing at
run time raises. That is why the check is structural: a passing test suite is exactly what the
failure mode produces.

The scan is deliberately conservative about what counts as a reference. A ``from
backtest.ou_ev_chain import HOLD_SEASONS`` is caught, and so is an aliased-module attribute access
(``import backtest.ou_ev_chain as chain`` then ``chain.HOLD_SEASONS``), because the second form is
the one a careful author reaches for when the first is blocked.

THE MODULE LIST GROWS AS THE PHASE LANDS
----------------------------------------
Several Phase-31 modules do not exist yet -- they land in Waves 3 and later. The registry below
names all of them; the scan visits the ones present on this checkout and REPORTS the list it
visited. Two entries are marked REQUIRED and their absence is a failure, so the scan can never
pass vacuously by finding nothing: a scan over an empty list is a green test that asserts
nothing, which is worse than no test.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import backtest.ev_chain_constants as p31
from tests import phase31_state

REPO_ROOT = Path(__file__).resolve().parents[2]

# The module whose window constants Phase-31 code must never read.
PHASE_27_WINDOW_MODULE = "backtest.ou_ev_chain"
PHASE_27_WINDOW_SYMBOLS = frozenset({"TUNE_SEASONS", "HOLD_SEASONS"})

# The Phase-27 fence helper that reads those constants. Calling it from Phase-31 code inherits the
# Phase-27 window with no error and a wrong-but-plausible fence report.
PHASE_27_FENCE_HELPER = "_assert_fit_window"

# Every Phase-31 module, whether or not it exists on this checkout. `required` marks the two that
# must be present for the scan to be meaningful today; the rest land in later waves.
P31_MODULES: tuple[tuple[str, bool], ...] = (
    ("backtest/ev_chain_constants.py", True),  # Plan 31-01 / 31-05, the frozen rule
    ("backtest/bet_selector.py", True),  # Plan 31-06, the single-source selector facade
    (
        "backtest/selector_strategies.py",
        False,
    ),  # Plan 31-06, the three per-target strategies
    ("backtest/wp_ev_chain.py", False),  # Plan 31-07
    ("backtest/ats_ev_chain.py", False),  # Plan 31-07
    ("backtest/roi_significance.py", False),  # Plan 31-12, the ROI hypothesis test
    ("backtest/profitability_2025.py", False),  # Plan 31-12, the one-shot runner
    ("backtest/bet_tracker.py", False),  # Plan 31-13, the realized-vs-expected tracker
)

# An assignment whose TARGET looks like a significance level. T-31-22: a second alpha is a second
# answer, so Phase-31 code must IMPORT SIGNIFICANCE_ALPHA and never write a float of its own. A
# name-to-name binding (``ALPHA = SIGNIFICANCE_ALPHA``) is fine and is what the frozen module does;
# a numeric literal is not.
_ALPHA_NAME_RE = re.compile(r"^(SIGNIFICANCE_)?ALPHA$")


def _existing_modules() -> list[str]:
    """Return the Phase-31 module paths present on this checkout, asserting the required ones."""
    present: list[str] = []
    for relative_path, required in P31_MODULES:
        if (REPO_ROOT / relative_path).is_file():
            present.append(relative_path)
        elif required:
            pytest.fail(
                f"the REQUIRED Phase-31 module {relative_path} is missing from this checkout. "
                "The isolation scan would then visit a shorter list and could pass while "
                "asserting less than it claims -- a scan over nothing is a green test that "
                "proves nothing."
            )
    return present


def _violations(source: str, label: str) -> list[str]:
    """Return every Phase-27-window / fence / second-alpha violation in *source*.

    Args:
        source: Python source text.
        label: How to name this source in a failure message (a path, or a fixture name).

    Returns:
        A list of human-readable violation strings, each naming the symbol and the line.
    """
    tree = ast.parse(source, filename=label)
    found: list[str] = []

    # Aliases under which the Phase-27 chain module was imported, so an attribute access through
    # any of them is recognised. `import backtest.ou_ev_chain` with no `as` binds the top-level
    # name `backtest`, which is why that case is tracked separately below.
    module_aliases: set[str] = set()
    plain_package_import = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == PHASE_27_WINDOW_MODULE:
            for alias in node.names:
                if alias.name in PHASE_27_WINDOW_SYMBOLS:
                    found.append(
                        f"{label}:{node.lineno}: imports {alias.name} from "
                        f"{PHASE_27_WINDOW_MODULE} -- that is the PHASE-27 window (D31-14). "
                        f"Read the Phase-31 window from backtest.ev_chain_constants "
                        f"(TUNE_SEASONS_P31 / HOLD_SEASONS_P31) instead."
                    )
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == PHASE_27_FENCE_HELPER:
                    found.append(
                        f"{label}:{node.lineno}: imports {PHASE_27_FENCE_HELPER} from "
                        f"{node.module} -- that helper READS the Phase-27 window constants, so "
                        "importing it is enough to inherit the wrong window. The import is "
                        "flagged separately from a call because an import that is never called "
                        "produces no Name node to catch."
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PHASE_27_WINDOW_MODULE:
                    if alias.asname:
                        module_aliases.add(alias.asname)
                    else:
                        plain_package_import = True

    for node in ast.walk(tree):
        # Aliased or dotted attribute access: chain.HOLD_SEASONS, backtest.ou_ev_chain.HOLD_SEASONS
        if isinstance(node, ast.Attribute) and node.attr in PHASE_27_WINDOW_SYMBOLS:
            base = node.value
            reached_phase27 = (
                isinstance(base, ast.Name) and base.id in module_aliases
            ) or (
                plain_package_import
                and isinstance(base, ast.Attribute)
                and base.attr == "ou_ev_chain"
            )
            if reached_phase27:
                found.append(
                    f"{label}:{node.lineno}: reads {node.attr} off {PHASE_27_WINDOW_MODULE} "
                    f"through a module reference -- that is the PHASE-27 window (D31-14)."
                )

        # The Phase-27 fence helper, by bare name or by attribute.
        if isinstance(node, ast.Name) and node.id == PHASE_27_FENCE_HELPER:
            found.append(
                f"{label}:{node.lineno}: references {PHASE_27_FENCE_HELPER}, which READS the "
                "Phase-27 window constants. A Phase-31 runner reusing it inherits the Phase-27 "
                "window silently and its fence report names the wrong hold seasons."
            )
        elif isinstance(node, ast.Attribute) and node.attr == PHASE_27_FENCE_HELPER:
            found.append(
                f"{label}:{node.lineno}: references {PHASE_27_FENCE_HELPER} through a module "
                "reference, which READS the Phase-27 window constants (see above)."
            )

        # A second alpha, declared as a numeric literal.
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and _ALPHA_NAME_RE.match(target.id)
                and isinstance(getattr(node, "value", None), ast.Constant)
                and isinstance(node.value.value, (int, float))  # type: ignore[union-attr]
                and not isinstance(node.value.value, bool)  # type: ignore[union-attr]
            ):
                found.append(
                    f"{label}:{node.lineno}: declares {target.id} as a numeric literal. alpha "
                    "is IMPORTED from backtest.diagnose (SIGNIFICANCE_ALPHA) and never "
                    "re-declared -- a second alpha is a second answer, even when it happens to "
                    "read 0.05 today (T-31-22)."
                )

    return found


def test_the_scan_visits_a_non_empty_module_list() -> None:
    """The registry resolves to at least the two modules that exist today.

    Anti-vacuity. Every other assertion in this module is of the form "no violation was found",
    and that is trivially satisfiable by scanning nothing. This test is what makes the others
    mean something.
    """
    present = _existing_modules()
    assert present, (
        "the Phase-31 isolation scan visited ZERO modules. Every no-violation assertion below "
        "would then pass while proving nothing."
    )
    assert "backtest/ev_chain_constants.py" in present
    assert "backtest/bet_selector.py" in present


def test_no_phase31_module_reads_the_phase27_window_or_fence() -> None:
    """D31-14 / T-31-19: no Phase-31 module imports or references the Phase-27 window.

    Also covers T-31-22 (a second alpha) in the same pass, because both failures have the same
    shape: a value that looks right, arrived at through the wrong source.
    """
    present = _existing_modules()
    violations: list[str] = []
    for relative_path in present:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        violations.extend(_violations(source, relative_path))

    assert not violations, (
        "Phase-31 isolation violations found (D31-14, T-31-19, T-31-22).\n"
        + "\n".join(f"  - {line}" for line in violations)
        + f"\n\nModules scanned ({len(present)}): {present}"
    )


def test_the_scan_catches_a_planted_phase27_window_import() -> None:
    """Fail-closed control: the scan REPORTS a violation rather than only ever finding none.

    A check that has only been observed passing is indistinguishable from a check that cannot
    fail. This plants each of the three violation shapes in a source fixture and asserts each is
    reported, so the real scan above is known to be live rather than merely green.
    """
    planted = (
        "from backtest.ou_ev_chain import HOLD_SEASONS\n"
        "import backtest.ou_ev_chain as chain\n"
        "from backtest.ou_monetization import _assert_fit_window\n"
        "ALPHA = 0.05\n"
        "WINDOW = chain.TUNE_SEASONS\n"
    )
    reported = _violations(planted, "<planted>")
    joined = "\n".join(reported)

    assert "imports HOLD_SEASONS" in joined, reported
    assert "reads TUNE_SEASONS" in joined, reported
    assert "_assert_fit_window" in joined, reported
    assert "declares ALPHA as a numeric literal" in joined, reported


def test_the_scan_does_not_flag_the_legitimate_phase27_reuse() -> None:
    """Fail-open control: reusing the Phase-27 CHAIN PRIMITIVES is legitimate and stays clean.

    D31-14 forbids inheriting the Phase-27 WINDOW, not the Phase-27 mathematics. The O/U chain is
    reused verbatim by design (D31-06), and WP reuses ``devig``. A scan that reddened on those
    imports would be one a future author had to weaken, and a weakened guard asserts nothing.
    """
    legitimate = (
        "from backtest.ou_ev_chain import calibrated_p_over, devig, per_bet_ev\n"
        "from backtest.ou_monetization import BOOTSTRAP_B, BOOTSTRAP_SEED\n"
        "from backtest.diagnose import SIGNIFICANCE_ALPHA\n"
        "ALPHA = SIGNIFICANCE_ALPHA\n"
    )
    assert _violations(legitimate, "<legitimate>") == []


def test_alpha_is_the_imported_primitive_by_identity() -> None:
    """T-31-22: the frozen module's ALPHA IS backtest.diagnose.SIGNIFICANCE_ALPHA.

    The AST scan proves no NEW literal was written. This proves the binding actually resolves to
    the one alpha the whole repository already judges at, rather than to some other name that
    happens to hold 0.05.
    """
    from backtest.diagnose import SIGNIFICANCE_ALPHA

    assert p31.ALPHA is SIGNIFICANCE_ALPHA
    assert p31.ROI_SIGNIFICANCE_SPEC["alpha"] is SIGNIFICANCE_ALPHA


def test_the_ats_residual_figures_have_exactly_one_source_of_truth() -> None:
    """The frozen rule's ATS means EQUAL the Plan 31-02 measurement, to the last digit.

    The pre-registration restates the per-season and pooled MEANS because its prose makes a
    direction and magnitude claim about them, while ``tests/phase31_state.py`` holds the full
    measured (n, mean, sd, t, p) tuples. Two homes for one number is the second-list failure this
    phase guards against everywhere else, so the agreement is asserted rather than assumed: a
    drift between them is a test failure, not an argument about which number was right.
    """
    measured_means = {
        season: values[1]
        for season, values in phase31_state.ATS_RESIDUAL_BY_SEASON.items()
    }
    assert measured_means == p31.ATS_RESIDUAL_BY_SEASON_P31, (
        "the frozen ATS per-season means disagree with the Plan 31-02 measurement in "
        "tests/phase31_state.ATS_RESIDUAL_BY_SEASON. The measurement is the source of truth; "
        "the frozen rule must not be edited to match a re-measurement, and a re-measurement must "
        "not be silently accepted -- report the disagreement."
    )
    assert phase31_state.ATS_RESIDUAL_POOLED[1] == p31.ATS_RESIDUAL_POOLED_P31

    # The contract's own prose must NAME the negative season, not merely be consistent with it.
    assert p31.ATS_RESIDUAL_BY_SEASON_P31[2022] < 0.0
    assert "2022" in p31.ATS_RESIDUAL_CONTRACT
    assert "NEGATIVE" in p31.ATS_RESIDUAL_CONTRACT


def test_the_rehearsal_proxy_split_is_disjoint_and_is_not_the_rule() -> None:
    """REVIEW-REHEARSAL: the rehearsal narrows the TUNE side too, and says it is not the rule.

    Overriding only the hold to 2024 would put a TUNE season in the HOLD, so the rehearsal would
    either trip the fit-window fence or require weakening it -- in a phase whose entire value is
    temporal honesty. Disjointness is the property that makes the rehearsal exercise the fence at
    FULL strength, so it is asserted rather than described.
    """
    tune = set(p31.REHEARSAL_PROXY_SPLIT["tune_seasons"])  # type: ignore[arg-type]
    hold = set(p31.REHEARSAL_PROXY_SPLIT["hold_seasons"])  # type: ignore[arg-type]

    assert tune == {2021, 2022, 2023}
    assert hold == {2024}
    assert not (tune & hold), (
        f"the rehearsal split is not disjoint: {sorted(tune & hold)}"
    )
    assert p31.REHEARSAL_PROXY_SPLIT["is_the_preregistered_rule"] is False
    assert p31.REHEARSAL_PROXY_SPLIT["consumable_by_the_armed_run"] is False

    # And it is NOT the frozen rule, which is the thing it must never be mistaken for.
    assert tune != set(p31.TUNE_SEASONS_P31)
    assert hold != set(p31.HOLD_SEASONS_P31)


def test_the_frozen_windows_and_scope_are_what_the_owner_ratified() -> None:
    """D31-13 and D31-38, pinned so a later edit to the frozen rule turns the suite red."""
    assert p31.TUNE_SEASONS_P31 == (2021, 2022, 2023, 2024)
    assert p31.HOLD_SEASONS_P31 == (2025,)
    assert 2025 not in p31.TUNE_SEASONS_P31
    assert p31.PRIOR_RESIDUAL_SEASONS_P31 == (2018, 2019, 2020)
    # Playoffs EVERYWHERE (D31-38): 285 games in the hold, not 272.
    assert set(p31.TUNE_GAME_TYPES) == {"REG", "WC", "DIV", "CON", "SB"}
    assert p31.TUNE_GAME_TYPES == p31.HOLD_GAME_TYPES


def test_the_roi_test_is_fully_specified_and_clv_is_report_only() -> None:
    """REVIEW-ROI: the ROI hypothesis test is defined BEFORE any 2025 number exists.

    Each clause is asserted present because the whole point of registering the test here is that
    a reader can check afterwards that the profitability p-value was defined before the answer was
    seen. A spec missing its reference distribution or its finite-sample rule would let the
    p-value be finished after the fact.
    """
    spec = p31.ROI_SIGNIFICANCE_SPEC
    for clause in (
        "null",
        "statistic",
        "reference_distribution",
        "one_sided_rule",
        "finite_sample_rule",
        "min_attainable_p",
        "min_attainable_p_consequence",
        "resolution_disclosure",
    ):
        assert clause in spec, f"ROI_SIGNIFICANCE_SPEC is missing the {clause!r} clause"

    assert p31.ROI_MIN_ATTAINABLE_P == 1.0 / (p31.BOOTSTRAP_B + 1)
    # The registered consequence only binds if the minimum EXCEEDS alpha. On this configuration
    # it does not, and the spec says so explicitly rather than leaving the reader to compute it.
    assert p31.ROI_MIN_ATTAINABLE_P < p31.ALPHA

    assert p31.CLV_P_VALUE_IS_REPORT_ONLY is True
    # The reference distribution must resample the HOLD bets only and must be RECENTRED at the
    # null; a non-recentred bootstrap yields an interval, not a p-value.
    reference = str(spec["reference_distribution"])
    assert "HOLD" in reference
    assert "RECENTRED" in reference

    # The duplicate robustness cut counts ONCE: the two named cuts compute the identical frame.
    assert p31.ROBUSTNESS_CUTS_P31 == ("regular_season_only", "playoffs_excluded")
    assert p31.ROBUSTNESS_CUT_BH_COUNT == 1
    assert p31.BH_FAMILY_SPEC["denominator_no_fallback"] == 6


def test_the_run_ledger_states_and_transition_rule_are_frozen() -> None:
    """REVIEW-ONESHOT: four states, and started/failed are NOT automatically rerunnable."""
    assert p31.RUN_LEDGER_STATES == ("armed", "started", "completed", "failed")
    assert p31.RUN_LEDGER_PATH == "config/profitability_2025_run_ledger.toml"
    rule = p31.RUN_LEDGER_TRANSITION_RULE
    assert "OWNER RULING" in rule
    assert "NO force flag" in rule
    assert "BEFORE the first 2025 read" in rule


def test_the_readout_allowlist_names_five_figures_each_with_a_source() -> None:
    """REVIEW-READOUT: an allowlist replaces a blanket rule that could never have been satisfied.

    Five entries, each with a non-empty authoritative source path. The blanket "restate no
    prior-document number" form was contradicted by the readout's own required content, and an
    unsatisfiable guard gets weakened during execution until it asserts nothing.
    """
    permitted = p31.READOUT_PERMITTED_FIGURES
    assert len(permitted) == 5, sorted(permitted)
    for figure, source in permitted.items():
        assert source.strip(), (
            f"permitted figure {figure!r} has no authoritative source"
        )
    assert permitted["every_2025_figure"] == "config/profitability_2025_verdict.toml"
    assert (
        permitted["per_target_absolute_pooled_clv_with_t_and_p"] == "config/gate.toml"
    )

    # The verdict vocabulary makes "no bets selected" a first-class outcome with a stated meaning.
    assert set(p31.VERDICT_TOKEN_MEANINGS) == set(p31.VERDICT_TOKENS)
    assert p31.UNDISCHARGEABLE_NO_BETS in p31.VERDICT_TOKENS
    assert "PASS" in p31.VERDICT_TOKEN_MEANINGS[p31.UNDISCHARGEABLE_NO_BETS]
