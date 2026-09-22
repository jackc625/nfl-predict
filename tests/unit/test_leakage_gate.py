"""Tests for LeakageGate hard-fail validation.

Covers:
- The per-builder time fence is GONE (Phase 33.2, D33.2-01). Its tests moved to
  tests/unit/test_information_time_gate.py, each intent re-asked as the question the
  information-time gate answers: "raises on future data" -> a value timed one second
  after its game's lock raises naming the game; "passes clean data" -> a value timed
  exactly at the lock passes; "an inert check announces itself (WR-01)" -> a zero-row
  source is reported empty-unchecked by name and never counted as checked.
- Combined matrix validation (validate_combined_matrix)
- Elo chronological ordering (check_elo_ordering)
- Diagnostic JSON report generation (write_diagnostic_report)
"""

import json
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import pytest

from features.validation import LeakageGate, LeakageViolation

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gate():
    """Create a LeakageGate instance."""
    return LeakageGate()


@pytest.fixture
def as_of_dt():
    """Standard as-of datetime for tests: Friday 2024-10-04 18:00 ET."""
    return datetime(2024, 10, 4, 18, 0, 0)


@pytest.fixture
def clean_features(as_of_dt):
    """Feature DataFrame where all data is before as_of_datetime."""
    return pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "game_date": [
                as_of_dt - timedelta(days=21),
                as_of_dt - timedelta(days=14),
                as_of_dt - timedelta(days=7),
            ],
            "elo_home": [1520.0, 1530.0, 1525.0],
            "elo_away": [1480.0, 1470.0, 1475.0],
            "rolling_epa_home": [0.05, 0.08, 0.06],
            "rolling_epa_away": [-0.02, 0.01, -0.01],
        }
    )


# ---------------------------------------------------------------------------
# The kickoff-versus-now fence no longer exists as a decision path
# ---------------------------------------------------------------------------


def test_check_time_fence_is_gone_not_disabled():
    """D33.2-01 retires the fence; no deprecated shim is left as a second answer."""
    assert not hasattr(LeakageGate, "check_time_fence")


# ---------------------------------------------------------------------------
# Test 3: validate_combined_matrix raises on leakage keyword columns
# ---------------------------------------------------------------------------


def test_validate_combined_matrix_raises_on_closing_keyword(
    gate, clean_features, as_of_dt
):
    """validate_combined_matrix raises LeakageViolation when columns contain 'closing' keyword."""
    # Add a column with leakage keyword
    df = clean_features.copy()
    df["closing_spread"] = [3.5, 4.0, 3.0]

    with pytest.raises(LeakageViolation) as exc_info:
        gate.validate_combined_matrix(df, as_of_dt)

    assert (
        "closing" in str(exc_info.value).lower()
        or exc_info.value.details.get("violation_type") == "leakage_keyword"
    )


# ---------------------------------------------------------------------------
# Test 4: validate_combined_matrix raises when required feature groups missing
# ---------------------------------------------------------------------------


def test_validate_combined_matrix_raises_missing_required_groups(gate, as_of_dt):
    """validate_combined_matrix raises LeakageViolation when required feature groups
    (Elo, team form) are completely missing."""
    # DataFrame with NO elo_ or rolling_ features
    df = pd.DataFrame(
        {
            "game_id": ["G1", "G2"],
            "season": [2024, 2024],
            "week": [1, 2],
            "game_date": [
                as_of_dt - timedelta(days=14),
                as_of_dt - timedelta(days=7),
            ],
            "wind_mph": [10.0, 12.0],
            "snapshot_spread": [-3.5, 2.5],
        }
    )

    with pytest.raises(LeakageViolation) as exc_info:
        gate.validate_combined_matrix(df, as_of_dt)

    assert exc_info.value.details["violation_type"] == "missing_required_group"
    assert len(exc_info.value.details["missing_groups"]) > 0


# ---------------------------------------------------------------------------
# Test 5: validate_combined_matrix does NOT raise for distribution warnings
# ---------------------------------------------------------------------------


def test_validate_combined_matrix_no_raise_on_distribution_issues(gate, as_of_dt):
    """validate_combined_matrix does NOT raise for distribution/correlation warnings.
    Only leakage keywords and missing required groups are hard failures."""
    df = pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "game_date": [
                as_of_dt - timedelta(days=21),
                as_of_dt - timedelta(days=14),
                as_of_dt - timedelta(days=7),
            ],
            "elo_home": [1520.0, 1530.0, 1525.0],
            "elo_away": [1480.0, 1470.0, 1475.0],
            "rolling_epa_home": [0.05, 0.08, 0.06],
            "rolling_epa_away": [-0.02, 0.01, -0.01],
            # Extremely skewed distribution -- should warn, not fail
            "some_feature": [0.0, 0.0, 100.0],
        }
    )

    # Should NOT raise -- distribution warnings are not hard failures
    gate.validate_combined_matrix(df, as_of_dt)


# ---------------------------------------------------------------------------
# Test 6: write_diagnostic_report produces valid JSON
# ---------------------------------------------------------------------------


def test_write_diagnostic_report_produces_valid_json(gate):
    """write_diagnostic_report produces valid JSON with required keys."""
    violation = LeakageViolation(
        "Test violation",
        details={
            "violation_type": "time_fence",
            "affected_features": ["elo_home", "elo_away"],
            "affected_games": ["G1", "G2"],
            "timestamp": "2024-10-04T18:00:00",
        },
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = gate.write_diagnostic_report(violation, output_dir=tmpdir)

        # Verify file was created
        assert report_path is not None

        # Verify valid JSON
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)

        # Required keys
        assert "violation_type" in report
        assert "timestamp" in report
        assert "details" in report


# ---------------------------------------------------------------------------
# Test 6b (Plan 33.2-20): the writer has NO production default
# ---------------------------------------------------------------------------


def test_write_diagnostic_report_requires_an_output_directory(gate):
    """A caller that does not say where the report goes must not get production.

    The parameter used to default to ``outputs/diagnostics/``, so any test that drove a
    build into a Stage-2 refusal wrote a file into the production tree -- which is how
    ``outputs/diagnostics/leakage_20260921_221559.json`` appeared during Plan 33.2-13.
    The default is gone; omitting the argument is now a TypeError, not a silent
    production write.
    """
    import inspect

    signature = inspect.signature(gate.write_diagnostic_report)
    parameter = signature.parameters["output_dir"]
    assert parameter.default is inspect.Parameter.empty, (
        "write_diagnostic_report.output_dir carries a default again "
        f"({parameter.default!r}). A default here is a production write nobody asked "
        "for; the one production caller names the tree explicitly."
    )

    violation = LeakageViolation("Test violation", details={"violation_type": "x"})
    with pytest.raises(TypeError):
        gate.write_diagnostic_report(violation)


def test_the_module_names_no_production_diagnostics_directory(gate):
    """Structural: no call in features/validation.py builds outputs/diagnostics."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "features" / "validation.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # DOCSTRINGS AND OTHER BARE STRING STATEMENTS ARE OUT OF THE SUBJECT, deliberately.
    # The docstring above explains why the default was removed and has to be able to say
    # the words "outputs/diagnostics"; a scan that could not tell that apart from a live
    # path constant would force the explanation out of the file. A bare string statement
    # is an ``ast.Expr`` whose value is the constant, and it can never build a path.
    # Every constant that COULD -- an assignment value, a call argument, a ``/`` operand
    # -- is still read.
    narrating = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "diagnostics" in node.value
        and id(node) not in narrating
    ]
    assert offenders == [], (
        f"features/validation.py names a diagnostics directory at line(s) {offenders}. "
        "The destination is the CALLER's decision; a path constant here is the "
        "production default coming back under another name."
    )


def test_the_diagnostics_scan_flags_a_planted_path_constant() -> None:
    """Non-vacuity: the restriction must not have blinded the scan to a real default."""
    import ast

    planted = ast.parse(
        '"""A docstring that talks about outputs/diagnostics freely."""\n'
        "import pathlib\n"
        'DEFAULT_DIR = "outputs/diagnostics"\n'
        'OTHER = pathlib.Path("outputs/diagnostics")\n'
    )
    narrating = {
        id(node.value)
        for node in ast.walk(planted)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    hits = [
        node.lineno
        for node in ast.walk(planted)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "diagnostics" in node.value
        and id(node) not in narrating
    ]
    assert hits == [3, 4], (
        f"the scan found {hits!r}. It must flag the assignment value (line 3) and the "
        "call argument (line 4) and must not flag the docstring (line 1)."
    )


# ---------------------------------------------------------------------------
# Test 7: check_elo_ordering raises on out-of-order updates
# ---------------------------------------------------------------------------


def test_check_elo_ordering_raises_on_out_of_order(gate):
    """check_elo_ordering raises LeakageViolation when Elo updates are not chronological."""
    elo_history = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "game_date": [
                datetime(2024, 9, 8),
                datetime(2024, 9, 22),  # Week 3 before Week 2
                datetime(2024, 9, 15),  # Week 2 after Week 3 (out of order)
                datetime(2024, 9, 29),
            ],
            "team": ["BUF", "BUF", "BUF", "BUF"],
            "elo_after": [1520.0, 1535.0, 1530.0, 1540.0],
        }
    )

    with pytest.raises(LeakageViolation) as exc_info:
        gate.check_elo_ordering(elo_history)

    assert exc_info.value.details["violation_type"] == "elo_ordering"
    assert len(exc_info.value.details["out_of_order_games"]) > 0


# ---------------------------------------------------------------------------
# Test 8: check_elo_ordering passes on chronological data
# ---------------------------------------------------------------------------


def test_check_elo_ordering_passes_chronological(gate):
    """check_elo_ordering passes when Elo updates are strictly chronological."""
    elo_history = pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "game_date": [
                datetime(2024, 9, 8),
                datetime(2024, 9, 15),
                datetime(2024, 9, 22),
                datetime(2024, 9, 29),
            ],
            "team": ["BUF", "BUF", "BUF", "BUF"],
            "elo_after": [1520.0, 1530.0, 1535.0, 1540.0],
        }
    )

    # Should not raise
    gate.check_elo_ordering(elo_history)
