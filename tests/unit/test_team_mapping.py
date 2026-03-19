"""Unit tests for team abbreviation normalization and game ID creation.

Tests that utils/team_data.py is the single source of truth for team abbreviation
mapping, and that utils/game_id_utils.py delegates to it (no duplicate mapping).
"""

import pytest

from utils.exceptions import DataValidationError
from utils.game_id_utils import create_standard_game_id
from utils.team_data import (
    ALL_TEAMS,
    get_all_teams,
    normalize_team_abbreviation,
    validate_team_abbreviation,
)

# --- Canonical abbreviation tests ---


class TestCanonicalAbbreviations:
    """All 32 canonical abbreviations must normalize to themselves."""

    CANONICAL_32 = [
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LA",
        "LAC",
        "LV",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WAS",
    ]

    @pytest.mark.parametrize("abbr", CANONICAL_32)
    def test_canonical_normalizes_to_self(self, abbr):
        assert normalize_team_abbreviation(abbr) == abbr

    def test_get_all_teams_returns_exactly_32(self):
        teams = get_all_teams()
        assert len(teams) == 32, f"Expected 32 teams, got {len(teams)}: {teams}"

    def test_la_is_in_all_teams(self):
        teams = get_all_teams()
        assert "LA" in teams, f"'LA' not found in all teams: {teams}"

    def test_lar_is_not_in_all_teams(self):
        """LAR should NOT be a canonical key -- it is a variant that maps to LA."""
        teams = get_all_teams()
        assert "LAR" not in teams, "'LAR' should not be a canonical team key"

    def test_all_teams_module_constant(self):
        assert len(ALL_TEAMS) == 32


# --- Historical variant mapping tests ---


class TestHistoricalVariants:
    """Historical team abbreviation variants must resolve to current canonical form."""

    def test_la_returns_la(self):
        assert normalize_team_abbreviation("LA") == "LA"

    def test_lar_returns_la(self):
        assert normalize_team_abbreviation("LAR") == "LA"

    def test_stl_returns_la(self):
        assert normalize_team_abbreviation("STL") == "LA"

    def test_sl_returns_la(self):
        """SL is a St. Louis Rams variant."""
        assert normalize_team_abbreviation("SL") == "LA"

    def test_oak_returns_lv(self):
        assert normalize_team_abbreviation("OAK") == "LV"

    def test_sd_returns_lac(self):
        assert normalize_team_abbreviation("SD") == "LAC"

    def test_jac_returns_jax(self):
        assert normalize_team_abbreviation("JAC") == "JAX"

    def test_wsh_returns_was(self):
        assert normalize_team_abbreviation("WSH") == "WAS"

    def test_gnb_returns_gb(self):
        assert normalize_team_abbreviation("GNB") == "GB"

    def test_nwe_returns_ne(self):
        assert normalize_team_abbreviation("NWE") == "NE"

    def test_nor_returns_no(self):
        assert normalize_team_abbreviation("NOR") == "NO"

    def test_sfo_returns_sf(self):
        assert normalize_team_abbreviation("SFO") == "SF"

    def test_tam_returns_tb(self):
        assert normalize_team_abbreviation("TAM") == "TB"

    def test_hst_returns_hou(self):
        assert normalize_team_abbreviation("HST") == "HOU"

    def test_clv_returns_cle(self):
        assert normalize_team_abbreviation("CLV") == "CLE"

    def test_lvr_returns_lv(self):
        assert normalize_team_abbreviation("LVR") == "LV"


# --- Hard-fail on unknown abbreviations ---


class TestUnknownAbbreviationHardFail:
    """Unknown abbreviations must raise DataValidationError with closest-match suggestion."""

    def test_unknown_raises_data_validation_error(self):
        with pytest.raises(DataValidationError, match="Unknown team abbreviation"):
            normalize_team_abbreviation("XYZ")

    def test_unknown_includes_suggestion(self):
        """Unknown abbreviation 'KCC' should suggest 'KC'."""
        with pytest.raises(DataValidationError, match="Did you mean"):
            normalize_team_abbreviation("KCC")

    def test_empty_string_raises(self):
        with pytest.raises(DataValidationError, match="cannot be empty"):
            normalize_team_abbreviation("")

    def test_whitespace_only_raises(self):
        with pytest.raises(DataValidationError):
            normalize_team_abbreviation("   ")

    def test_validate_team_abbreviation_true_for_valid(self):
        assert validate_team_abbreviation("KC") is True

    def test_validate_team_abbreviation_false_for_invalid(self):
        assert validate_team_abbreviation("XYZ") is False

    def test_validate_team_abbreviation_false_for_empty(self):
        assert validate_team_abbreviation("") is False


# --- Case insensitivity ---


class TestCaseInsensitivity:
    """Normalization should be case-insensitive."""

    def test_lowercase(self):
        assert normalize_team_abbreviation("kc") == "KC"

    def test_mixed_case(self):
        assert normalize_team_abbreviation("Kc") == "KC"

    def test_lowercase_variant(self):
        assert normalize_team_abbreviation("oak") == "LV"


# --- Game ID creation with normalization ---


class TestGameIdNormalization:
    """Game IDs must use normalized team abbreviations via team_data.py."""

    def test_canonical_teams(self):
        result = create_standard_game_id(2024, 1, "LA", "KC")
        assert result == "2024_W01_LA@KC"

    def test_variant_normalized_in_game_id(self):
        """LAR should be normalized to LA in game IDs."""
        result = create_standard_game_id(2024, 1, "LAR", "KC")
        assert result == "2024_W01_LA@KC"

    def test_historical_normalized_in_game_id(self):
        """OAK should be normalized to LV in game IDs."""
        result = create_standard_game_id(2024, 1, "OAK", "KC")
        assert result == "2024_W01_LV@KC"

    def test_both_teams_normalized(self):
        result = create_standard_game_id(2024, 1, "STL", "SD")
        assert result == "2024_W01_LA@LAC"
