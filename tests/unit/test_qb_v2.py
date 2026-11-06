"""Unit tests for the v2 QB tracker's replacement-level value for unknown quarterbacks."""

import pandas as pd
import pytest

from features.qb_v2 import QBTrackerV2


def test_unknown_qb_is_below_average_not_league_average():
    rolling = pd.DataFrame({"passer_player_id": ["00-1"], "qb_quality": [1.2]})
    assert QBTrackerV2._quality_of(rolling, None) == -0.5
    assert QBTrackerV2._quality_of(rolling.iloc[0:0], "00-1") == -0.5
    assert QBTrackerV2._quality_of(rolling, "00-9") == -0.5


def test_known_qb_keeps_his_rolling_quality():
    rolling = pd.DataFrame({"passer_player_id": ["00-1"], "qb_quality": [1.2]})
    assert QBTrackerV2._quality_of(rolling, "00-1") == pytest.approx(1.2)


def test_no_information_value_matches_the_replacement_level():
    assert QBTrackerV2().no_information_signature() == {"qb_adjustment": -0.5}
