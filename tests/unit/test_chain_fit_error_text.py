"""The three chain-fit refusals name a command that EXISTS (Plan 33-17 Task 2; NF-03, T-33-87).

THE DEFECT
----------
``backtest/weekly_bet_list.py`` refuses in three places when the pre-registered tune-only fit
cannot be resolved, does not cover a season, or carries no usable residual SD. All three told the
operator to re-run the measurement module that produced the fit. That module cannot be re-run:
``config/profitability_2025_run_ledger.toml`` records its single-use 2025 hold split as
``state = "completed"``, created by an exclusive file creation with NO force flag, and a new
attempt requires an owner ruling written into the ledger first. The prescribed command therefore
REFUSES rather than rebuilding anything.

This repository's convention is that a refusal carries its recovery command. The corollary, stated
in the phase context, is that a refusal naming an UNAVAILABLE command is worse than no text at
all -- it sends the reader to a locked door and charges them the time it takes to discover that.

WHY THIS IS PROVEN BY RAISING AND NEVER BY GREP
------------------------------------------------
Both surviving v3.0 audit warnings exist because a test grepped a file instead of checking written
output, and the milestone invariant is that a criterion asserts the EFFECT. So every assertion
below DRIVES a site into its refusal branch and reads the message it ACTUALLY RAISED. A grep over
``backtest/weekly_bet_list.py`` would pass on a module whose messages were fixed in a docstring and
broken in the ``raise``, and would fail on one that merely mentions the module in a comment.

WHY THE PROBE PATH IS NEUTRAL, SAID PLAINLY
---------------------------------------------
``load_frozen_chain_fit``'s message interpolates the path it was HANDED, and the production default
path is ``outputs/p31/profitability_2025_verdict.json`` -- which contains the forbidden token as
part of a FILE NAME. Naming the file that is missing is necessary and correct; prescribing a
command that cannot run is the defect. So the probe supplies a neutral absent path, and what is
asserted is the MODULE'S OWN WORDS rather than the caller's data. That distinction is recorded here
rather than left for a reader to infer from a passing test.

Run this module:  uv run pytest tests/unit/test_chain_fit_error_text.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from backtest.weekly_bet_list import (
    FrozenChainFitError,
    WeeklyChainFit,
    _require_season_covered,
    load_frozen_chain_fit,
    require_frozen_sd,
)
from tests.phase33_state import (
    CHAIN_FIT_ERROR_PROBES,
    CHAIN_FIT_ERROR_SITES,
    SPENT_MEASUREMENT_TOKEN,
)

_EXPECTED_EXCEPTIONS: dict[str, type[Exception]] = {
    "FrozenChainFitError": FrozenChainFitError,
}


def _fit_from_block(target: str, block: dict[str, Any]) -> WeeklyChainFit:
    """One ``WeeklyChainFit`` from a plain-data probe block."""
    return WeeklyChainFit(
        target=block.get("target", target),
        ev_floor_t=float(block["ev_floor_t"]),
        frozen_sd=block["frozen_sd"],
        season_bias_by_season=dict(block["season_bias_by_season"]),
        calibration_gate_passed=block.get("calibration_gate_passed"),
    )


def build_chain_fit_error_probe(name: str) -> tuple[Any, ...]:
    """The argument tuple that drives refusal site *name* into its refusal branch.

    PUBLIC on purpose. ``tests.phase33_state`` may hold no project imports, so its probes are
    plain-data blocks rather than ready-made argument tuples; this is the ONE materializer that
    turns a block into arguments, and being public keeps a probe re-usable from a command line::

        uv run python -c "from tests.unit.test_chain_fit_error_text import
        build_chain_fit_error_probe as b; import backtest.weekly_bet_list as w;
        w._require_season_covered(*b('_require_season_covered'))"

    Args:
        name: One of :data:`tests.phase33_state.CHAIN_FIT_ERROR_SITES`.

    Returns:
        The positional arguments for that site.
    """
    probe = CHAIN_FIT_ERROR_PROBES[name]
    kind = probe["kind"]
    if kind == "absent_path":
        return (probe["path"],)
    if kind == "fits_and_season":
        blocks: dict[str, Any] = probe["fit_blocks"]  # type: ignore[assignment]
        fits = {
            target: _fit_from_block(target, block) for target, block in blocks.items()
        }
        return (fits, probe["season"])
    if kind == "fit":
        block = probe["fit_block"]  # type: ignore[assignment]
        return (_fit_from_block(str(block["target"]), block),)  # type: ignore[index]
    msg = f"unknown probe kind {kind!r} for site {name!r}"
    raise AssertionError(msg)


_SITE_CALLABLES: dict[str, Any] = {
    "load_frozen_chain_fit": load_frozen_chain_fit,
    "_require_season_covered": _require_season_covered,
    "require_frozen_sd": require_frozen_sd,
}


def _raised_message(name: str) -> str:
    """Drive site *name* into its refusal and return the message it actually raised."""
    expected = _EXPECTED_EXCEPTIONS[str(CHAIN_FIT_ERROR_PROBES[name]["expect"])]
    with pytest.raises(expected) as excinfo:
        _SITE_CALLABLES[name](*build_chain_fit_error_probe(name))
    return str(excinfo.value)


# ---------------------------------------------------------------------------
# 1. Every probe actually reaches its refusal
# ---------------------------------------------------------------------------


def test_the_three_recorded_sites_are_the_three_that_exist() -> None:
    """Anti-vacuity. A committed set of two would let a third message stay broken unseen."""
    assert len(CHAIN_FIT_ERROR_SITES) == 3
    assert sorted(CHAIN_FIT_ERROR_SITES) == sorted(CHAIN_FIT_ERROR_PROBES)
    assert sorted(CHAIN_FIT_ERROR_SITES) == sorted(_SITE_CALLABLES)


@pytest.mark.parametrize("name", CHAIN_FIT_ERROR_SITES)
def test_each_probe_drives_its_site_into_a_refusal(name: str) -> None:
    """A probe that stopped refusing would make every text assertion below vacuous."""
    message = _raised_message(name)
    assert message.strip(), f"{name} raised an EMPTY message"


# ---------------------------------------------------------------------------
# 2. No raised message prescribes the spent measurement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CHAIN_FIT_ERROR_SITES)
def test_no_raised_refusal_names_the_spent_measurement(name: str) -> None:
    """NF-03. The message is READ FROM THE RAISED EXCEPTION, never from the source file."""
    message = _raised_message(name)
    assert SPENT_MEASUREMENT_TOKEN not in message, (
        f"{name} still sends the reader to the spent single-use measurement:\n{message}"
    )


@pytest.mark.parametrize("name", CHAIN_FIT_ERROR_SITES)
def test_no_raised_refusal_prescribes_re_running_anything_unavailable(
    name: str,
) -> None:
    """The broader shape, not just the one token.

    "Re-run", "regenerate" and "rebuild" were the three verbs the old messages used to point at
    a measurement that cannot be re-run. A message may still SAY the record cannot be
    regenerated -- that is the honest part -- so the check is on the IMPERATIVE forms only.
    """
    message = _raised_message(name).lower()
    for imperative in ("re-run `", "regenerate it with", "rebuild it with"):
        assert imperative not in message, (
            f"{name} prescribes an unavailable action ({imperative!r}):\n{message}"
        )


# ---------------------------------------------------------------------------
# 3. Every raised message carries a recovery that a reader can actually run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CHAIN_FIT_ERROR_SITES)
def test_each_raised_refusal_carries_a_recovery_command(name: str) -> None:
    """The convention is that a refusal carries its recovery. Removing the bad one is half a fix.

    Asserted structurally -- the message must contain a runnable ``uv run python`` invocation --
    because a message that only deleted the misleading sentence would satisfy every check above
    while leaving the reader with less than they started with.
    """
    message = _raised_message(name)
    assert "uv run python" in message, (
        f"{name} names no command a reader can run:\n{message}"
    )


@pytest.mark.parametrize("name", CHAIN_FIT_ERROR_SITES)
def test_the_recovery_command_each_refusal_names_actually_runs(name: str) -> None:
    """THE criterion, asserted as an EFFECT rather than as a shape.

    The whole defect was a message naming a command that refuses when you run it. Checking that
    the replacement merely LOOKS like a command would repeat the mistake one level up, so each
    embedded ``uv run python -c "..."`` is EXTRACTED FROM THE RAISED MESSAGE and EXECUTED, and its
    exit status is the assertion. Read only: every one of them prints, none writes.
    """
    message = _raised_message(name)
    script = _extract_python_c_script(message)
    assert script, (
        f"{name}'s recovery is not an extractable python -c script:\n{message}"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        f"{name}'s recovery command FAILED, which is the defect this plan removed:\n"
        f"script: {script}\nstderr: {completed.stderr}"
    )
    assert completed.stdout.strip(), (
        f"{name}'s recovery command ran but printed nothing, so it tells the reader nothing"
    )


def _extract_python_c_script(message: str) -> str | None:
    """The body of the first ``uv run python -c "..."`` in *message*, or None.

    The messages wrap the script in DOUBLE quotes and use single quotes inside, which is the
    shell-safe spelling on Windows and the one the repository uses elsewhere.
    """
    marker = 'uv run python -c "'
    start = message.find(marker)
    if start == -1:
        return None
    start += len(marker)
    end = message.find('"', start)
    if end == -1:
        return None
    return message[start:end]


# ---------------------------------------------------------------------------
# 4. The messages still carry the claims other tests bind to
# ---------------------------------------------------------------------------


def test_the_absent_record_refusal_still_states_that_there_is_no_fallback() -> None:
    """A pre-existing pin (``tests/unit/test_weekly_bet_list.py``) binds to this phrase.

    It is the substantive half of the message: the reason a missing fit is refused rather than
    defaulted is that a defaulted floor admits bets at a threshold nobody swept for.
    """
    assert "NO fallback" in _raised_message("load_frozen_chain_fit")


def test_the_season_refusal_still_names_the_uncovered_targets_and_the_fitted_seasons() -> (
    None
):
    """Correcting the recovery must not cost the message its diagnosis."""
    message = _raised_message("_require_season_covered")
    assert "2099" in message
    for target in ("'ats'", "'ou'", "'wp'"):
        assert target in message, message


def test_the_season_refusal_points_at_the_committed_pre_registration() -> None:
    """The recovery the plan specifies for THIS site: the 2026 bias is already committed."""
    message = _raised_message("_require_season_covered")
    assert "CHAIN_FIT_BIAS_2026" in message
    assert "cold_start_constants" in message


def test_the_residual_sd_refusal_still_names_the_target_and_the_offending_value() -> (
    None
):
    """A pre-existing pin binds to ``repr(target)``; the value is what makes it diagnosable."""
    message = _raised_message("require_frozen_sd")
    assert "'ou'" in message
    assert "None" in message


# ---------------------------------------------------------------------------
# 5. The message is RECORD-AWARE (Plan 33.2-29)
# ---------------------------------------------------------------------------


def test_a_missing_corrected_record_names_the_command_that_regenerates_it(
    tmp_path: Any,
) -> None:
    """The corrected record IS regenerable, so the locked-door text would be false of it."""
    from backtest.corrected_ev_chain_constants import CORRECTED_CHAIN_FIT_RECORD_PATH

    absent = tmp_path / CORRECTED_CHAIN_FIT_RECORD_PATH.rsplit("/", 1)[-1]
    with pytest.raises(FrozenChainFitError) as excinfo:
        load_frozen_chain_fit(absent)
    message = str(excinfo.value)
    assert "scripts.derive_corrected_ev_chain" in message
    assert "NO fallback" in message
    assert "CANNOT BE REGENERATED" not in message


def test_the_phase31_record_keeps_its_one_shot_ledger_text() -> None:
    """The Phase-31 branch is unchanged: that record really is a spent single-use split."""
    assert "CANNOT BE REGENERATED" in _raised_message("load_frozen_chain_fit")
