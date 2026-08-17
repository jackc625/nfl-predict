"""Tests for LeakageGate hard-fail validation.

Covers:
- Time-fence enforcement (check_time_fence)
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


@pytest.fixture
def leaked_features(as_of_dt):
    """Feature DataFrame with data AFTER as_of_datetime (leakage)."""
    return pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "season": [2024, 2024, 2024],
            "week": [4, 5, 6],
            "game_date": [
                as_of_dt - timedelta(days=1),
                as_of_dt + timedelta(days=6),  # FUTURE
                as_of_dt + timedelta(days=13),  # FUTURE
            ],
            "elo_home": [1520.0, 1530.0, 1525.0],
            "elo_away": [1480.0, 1470.0, 1475.0],
        }
    )


# ---------------------------------------------------------------------------
# Test 1: check_time_fence raises on future data
# ---------------------------------------------------------------------------


def test_check_time_fence_raises_on_future_data(gate, leaked_features, as_of_dt):
    """check_time_fence raises LeakageViolation when data is after as_of_datetime."""
    with pytest.raises(LeakageViolation) as exc_info:
        gate.check_time_fence(leaked_features, as_of_dt, "test_builder")

    assert exc_info.value.details["builder"] == "test_builder"
    assert exc_info.value.details["violation_type"] == "time_fence"
    assert exc_info.value.details["affected_rows"] >= 1


# ---------------------------------------------------------------------------
# Test 2: check_time_fence passes when all data is before cutoff
# ---------------------------------------------------------------------------


def test_check_time_fence_passes_clean_data(gate, clean_features, as_of_dt):
    """check_time_fence does NOT raise when all data is before as_of_datetime."""
    # Should not raise
    gate.check_time_fence(clean_features, as_of_dt, "test_builder")


# ---------------------------------------------------------------------------
# WR-01: an inert check must announce itself, not read as a pass
# ---------------------------------------------------------------------------


class _RecordingLogger:
    """Captures logger.warning calls without depending on log routing."""

    def __init__(self):
        self.warnings = []

    def warning(self, message, **kwargs):
        self.warnings.append((message, kwargs))

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


@pytest.fixture
def line_movement_shaped_output():
    """The real LineMovementBuilder output shape: game_id plus fifteen floats.

    No ``game_date``, no ``kickoff_et``, no ``snapshot_ts`` -- which is exactly why
    ``check_time_fence`` never entered its loop body for this builder and was a
    guaranteed pass, while build_features.py claimed registration routed the source
    "through the LeakageGate".
    """
    return pd.DataFrame(
        {
            "game_id": ["G1", "G2"],
            "opening_total": [44.0, 41.5],
            "total_drift": [-0.5, 1.0],
            "total_drift_dir": [-1.0, 1.0],
            "total_late_drift": [0.0, 0.5],
            "total_abs_travel": [1.5, 2.0],
            "total_reversals": [1.0, 0.0],
            "total_range": [1.5, 2.0],
            "line_movement_coverage": [1.0, 1.0],
            "opening_spread": [-2.5, 3.0],
            "spread_drift": [1.0, -0.5],
            "spread_drift_dir": [1.0, -1.0],
            "spread_late_drift": [0.5, 0.0],
            "spread_abs_travel": [1.0, 0.5],
            "spread_reversals": [0.0, 0.0],
            "spread_range": [1.0, 0.5],
        }
    )


def test_time_fence_warns_when_it_inspects_nothing(
    gate, line_movement_shaped_output, as_of_dt
):
    """WR-01: a structural no-op is recorded, not mistaken for a pass."""
    recorder = _RecordingLogger()
    gate.logger = recorder

    gate.check_time_fence(line_movement_shaped_output, as_of_dt, "line_movement")

    assert len(recorder.warnings) == 1
    message, fields = recorder.warnings[0]
    assert "inspected NOTHING" in message
    assert fields["builder"] == "line_movement"
    assert set(fields["expected_any_of"]) == {"game_date", "kickoff_et", "snapshot_ts"}


def test_time_fence_stays_silent_when_it_has_a_timestamp_to_inspect(
    gate, line_movement_shaped_output, as_of_dt
):
    """A frame carrying snapshot_ts is genuinely checked, so no warning fires."""
    frame = line_movement_shaped_output.copy()
    frame["snapshot_ts"] = [
        as_of_dt - timedelta(hours=2),
        as_of_dt - timedelta(days=1),
    ]

    recorder = _RecordingLogger()
    gate.logger = recorder

    gate.check_time_fence(frame, as_of_dt, "line_movement")

    assert recorder.warnings == []


def test_the_inert_check_still_returns_without_raising(
    gate, line_movement_shaped_output, as_of_dt
):
    """The warning changes visibility, not behaviour: no new hard failure."""
    gate.check_time_fence(line_movement_shaped_output, as_of_dt, "line_movement")


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
