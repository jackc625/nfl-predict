"""Tests for the refusals ``check_supplied_locks`` makes on naive instants."""

from __future__ import annotations

import pandas as pd
import pytest

from features.supplied_locks import check_supplied_locks


def test_a_naive_lock_is_refused():
    locks = pd.Series([pd.Timestamp("2026-11-17 17:00")], index=["g1"])
    with pytest.raises(ValueError, match="no timezone"):
        check_supplied_locks(locks, ["g1"])


def test_a_naive_lock_among_aware_ones_is_named():
    aware = pd.Timestamp("2026-11-17 17:00", tz="America/New_York")
    locks = pd.Series(
        [aware, pd.Timestamp("2026-11-17 17:00")], index=["g1", "g2"], dtype=object
    )
    with pytest.raises(ValueError, match="g2"):
        check_supplied_locks(locks, ["g1", "g2"])


def test_aware_locks_pass():
    locks = pd.Series([pd.Timestamp("2026-11-17 17:00", tz="UTC")], index=["g1"])
    check_supplied_locks(locks, ["g1"])
