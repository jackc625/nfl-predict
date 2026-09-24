"""The live 2026 bet rule moved WHOLE, in ONE commit (Plan 33.2-26 Task 3, T-33.2-26-08).

WHY ONE COMMIT, AND WHY A PER-SITE CHECK IS NOT ENOUGH
-------------------------------------------------------
``load_frozen_chain_fit`` resolves the per-target EV floors from the record named by
``DEFAULT_CHAIN_FIT_PATH`` and then, at the end of the SAME call, overlays the 2026 chain-fit bias
imported at module level. So the two halves of the live rule are resolved together, and a split
repoint produces a rule nobody chose -- corrected floors judged against the superseded bias, or
the reverse -- that loads cleanly and prints plausible numbers. This module asserts the property
a per-site check cannot express: ONE ``load_frozen_chain_fit()`` call yields floors from the
corrected record AND the 2026 bias from the corrected module, with a planted control in each
failure direction.

It also asserts the history: the corrective commit (resolved from the witness slot
``COLD_START_CORRECTIVE_COMMIT_SHA``, never re-derived) names ``11761c7`` and touches BOTH
``backtest/weekly_bet_list.py`` and ``utils/edge_tier.py``; and BOTH corrective commits of the 2026
bet rule -- this plan's and Plan 33.2-29's ``EV_CHAIN_CORRECTIVE_COMMIT_SHA`` -- are git
ancestors of HEAD, discovered by the token ``CORRECTIVE`` in the witness names.

The chain-fit record is gitignored generator output; a checkout without it skips the end-to-end
half by name rather than passing it vacuously.

Run this module:  uv run python -m pytest tests/unit/test_atomic_bet_rule_repoint.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import backtest.cold_start_constants as original
import backtest.weekly_bet_list as weekly
from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
SUPERSEDED_SHA = "11761c7"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def _corrected() -> Any:
    import backtest.corrected_cold_start_constants as corrected

    return corrected


def _record_floors() -> dict[str, float | None]:
    path = REPO_ROOT / weekly.DEFAULT_CHAIN_FIT_PATH
    if not path.exists():
        pytest.skip(
            f"{weekly.DEFAULT_CHAIN_FIT_PATH.as_posix()} is gitignored generator output and is "
            "absent on this checkout; the end-to-end half cannot run here."
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    return {target: block["ev_floor_t"] for target, block in record["tune_fit"].items()}


def _halves(fits: dict[str, Any]) -> tuple[bool, bool]:
    """``(floors from the corrected record, 2026 bias from the corrected module)``."""
    season = weekly.frozen_overlay_season()
    floors_ok = {t: f.ev_floor_t for t, f in fits.items()} == _record_floors()
    corrected_bias = dict(_corrected().CHAIN_FIT_BIAS_2026)
    bias_ok = bool(corrected_bias) and all(
        fits[target].season_bias_by_season.get(season) == value
        for target, value in corrected_bias.items()
    )
    return floors_ok, bias_ok


# ---------------------------------------------------------------------------
# 1. ONE load, BOTH halves corrected -- and each failure direction reachable
# ---------------------------------------------------------------------------


def test_one_load_yields_corrected_floors_and_the_corrected_2026_bias() -> None:
    fits = weekly.load_frozen_chain_fit()
    assert weekly.frozen_overlay_season() == 2026
    assert _halves(fits) == (True, True)


def test_corrected_bias_over_the_superseded_floors_is_caught() -> None:
    """Planted control: the floors half fails when the floors are the Phase-31 ones."""
    from backtest.corrected_ev_chain_constants import SUPERSEDED_EV_FLOOR_BY_TARGET

    fits = weekly.load_frozen_chain_fit()
    planted = {
        t: replace(f, ev_floor_t=SUPERSEDED_EV_FLOOR_BY_TARGET[t])
        for t, f in fits.items()
    }
    assert _halves(planted) == (False, True)


def test_corrected_floors_under_the_superseded_bias_are_caught() -> None:
    """Planted control: the bias half fails when the 2026 bias is the 11761c7 one."""
    fits = weekly.load_frozen_chain_fit()
    season = weekly.frozen_overlay_season()
    planted = {
        t: replace(
            f,
            season_bias_by_season={
                **f.season_bias_by_season,
                season: original.CHAIN_FIT_BIAS_2026[t],
            },
        )
        for t, f in fits.items()
    }
    assert _halves(planted) == (True, False)


def test_the_corrected_state_is_distinguishable_from_the_original() -> None:
    """Non-vacuity: both corrected sources exist and differ from the superseded ones."""
    from backtest.corrected_ev_chain_constants import (
        CORRECTED_EV_FLOOR_BY_TARGET,
        SUPERSEDED_EV_FLOOR_BY_TARGET,
    )

    assert CORRECTED_EV_FLOOR_BY_TARGET != SUPERSEDED_EV_FLOOR_BY_TARGET
    assert dict(_corrected().CHAIN_FIT_BIAS_2026) != dict(original.CHAIN_FIT_BIAS_2026)


# ---------------------------------------------------------------------------
# 2. The corrective commit, resolved from the witness
# ---------------------------------------------------------------------------


def _corrective_sha() -> str:
    sha = getattr(phase33_state, "COLD_START_CORRECTIVE_COMMIT_SHA", None)
    assert sha, "tests/phase33_state.py carries no COLD_START_CORRECTIVE_COMMIT_SHA yet"
    return str(sha)


def test_the_corrective_commit_touches_both_repointed_modules() -> None:
    sha = _corrective_sha()
    touched = _git("show", "--name-only", "--format=", sha).stdout.split()
    assert touched, f"the corrective commit {sha} resolves to no files"
    assert "backtest/weekly_bet_list.py" in touched
    assert "utils/edge_tier.py" in touched


def test_the_corrective_and_witness_messages_name_11761c7() -> None:
    sha = _corrective_sha()
    corrective = _git("log", "-1", "--format=%B", sha).stdout
    assert SUPERSEDED_SHA in corrective
    assert phase33_state.EV_CHAIN_CORRECTIVE_COMMIT_SHA[:7] in corrective, (
        "the corrective commit names the 33.2-29 staging commit it completes"
    )
    witness = _git(
        "log",
        "-1",
        "--format=%B",
        "-S",
        "COLD_START_CORRECTIVE_COMMIT_SHA",
        "--",
        "tests/phase33_state.py",
    ).stdout
    assert SUPERSEDED_SHA in witness
    assert sha[:12] in witness


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return _git("merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def test_both_corrective_commits_are_ancestors_of_head() -> None:
    """Discovered by the token CORRECTIVE, so both writers' names are part of the contract."""
    names = sorted(
        name
        for name in dir(phase33_state)
        if "CORRECTIVE" in name and isinstance(getattr(phase33_state, name), str)
    )
    assert "EV_CHAIN_CORRECTIVE_COMMIT_SHA" in names
    assert "COLD_START_CORRECTIVE_COMMIT_SHA" in names
    head = _git("rev-parse", "HEAD").stdout.strip()
    for name in names:
        assert _is_ancestor(getattr(phase33_state, name), head), name


def test_a_stale_witness_is_not_an_ancestor() -> None:
    """Control: the ancestry check can fail -- HEAD is not an ancestor of the corrective commit."""
    head = _git("rev-parse", "HEAD").stdout.strip()
    sha = _corrective_sha()
    if (
        head == sha
    ):  # pragma: no cover - the witness commit always follows the corrective one
        pytest.fail("HEAD is the corrective commit; the witness commit is missing")
    assert not _is_ancestor(head, sha)
