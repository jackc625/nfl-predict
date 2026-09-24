"""The MOS decode bounds were committed before the comparison, and its readout holds (Plan 33.2-11).

TWO GUARDS IN ONE MODULE
------------------------
1. ANCESTRY (the ``tests/unit/test_preregistration_ancestry.py`` pattern). ``config/mos_tolerance.py``
   is a pre-registration: its last-modifying commit must be a STRICT git ancestor of the commit that
   adds ``MOS-DECODE-COMPARISON.md``, and that tolerance commit must touch no other file. This is
   the only mechanical evidence that the six detection bounds existed before any comparison number
   did. It cannot prove intent; the owner's 2026-09-21 rulings discharge that.
2. DOC DRIFT (the ``tests/unit/test_signal_lift_readout_md.py`` shape). The committed readout keeps
   its required sections, its detection-bounds framing, ASCII, and never says "proven". It asserts
   the RULING -- each bound PASS, verdict PASS -- and never a point estimate: pinning a point
   estimate to moving data is the mistake this repository already made once.

The git-history checks skip on a shallow or non-git checkout BEFORE any ancestry call, because
there ``git merge-base --is-ancestor`` fails for want of history rather than for want of ancestry.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import config.mos_tolerance as tol

REPO_ROOT = Path(__file__).resolve().parents[2]
TOLERANCE_PATH = "config/mos_tolerance.py"
READOUT_PATH = "MOS-DECODE-COMPARISON.md"

SHALLOW_SKIP_MESSAGE = (
    "git history is unavailable (shallow clone or not a git checkout); skipping BEFORE any "
    "ancestry call rather than reporting a false ancestry violation."
)

_REQUIRED_SECTION_MARKERS = (
    "## What this is",
    "## Pre-registration",
    "## Method",
    "## Coverage",
    "## Bounds and rulings",
    "## The six statistics",
    "## The kickoff-hour subset",
    "## By station",
    "## Live versus history: a measured, accepted difference",
)
_REQUIRED_PHRASES = (
    "DETECTION BOUNDS ON OUR DECODING, not a claim that the forecast was",
    "UNVERIFIED",
    "no stand-in station is ever used",
)
_FORBIDDEN_WORD = "proven"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def _history_unavailable() -> bool:
    if _git("rev-parse", "--git-dir").returncode != 0:
        return True
    shallow = _git("rev-parse", "--is-shallow-repository")
    return shallow.returncode != 0 or shallow.stdout.strip() == "true"


def _tolerance_commit() -> str:
    return _git("log", "-1", "--format=%H", "--", TOLERANCE_PATH).stdout.strip()


def _readout_add_commit() -> str:
    added = _git(
        "log", "--diff-filter=A", "--format=%H", "--", READOUT_PATH
    ).stdout.split()
    return added[-1] if added else ""


@pytest.fixture
def history() -> None:
    if _history_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)


def _readout() -> str:
    return (REPO_ROOT / READOUT_PATH).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Ancestry
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("history")
class TestTheBoundsWereCommittedFirst:
    def test_the_tolerance_is_tracked_and_committed(self) -> None:
        assert TOLERANCE_PATH in _git("ls-files", TOLERANCE_PATH).stdout.split()
        assert re.fullmatch(r"[0-9a-f]{40}", _tolerance_commit())

    def test_the_tolerance_commit_touches_no_other_file(self) -> None:
        touched = _git(
            "show", "--name-only", "--format=", _tolerance_commit()
        ).stdout.split()
        assert touched == [TOLERANCE_PATH]

    def test_the_tolerance_commit_is_a_strict_ancestor_of_the_readout_commit(
        self,
    ) -> None:
        tolerance, readout = _tolerance_commit(), _readout_add_commit()
        assert readout, f"{READOUT_PATH} has not been committed"
        assert tolerance != readout, "the bounds and the readout share a commit"
        assert (
            _git("merge-base", "--is-ancestor", tolerance, readout).returncode == 0
        ), (
            f"{TOLERANCE_PATH} ({tolerance}) is not an ancestor of the commit adding "
            f"{READOUT_PATH} ({readout}). A bound committed after the comparison is not a "
            "pre-registration."
        )

    def test_the_readout_names_the_tolerance_commit(self) -> None:
        assert _tolerance_commit() in _readout()

    def test_the_state_manifest_witnesses_the_same_two_commits(self) -> None:
        """The appended slot names the commits git resolves, and the order holds."""
        from tests import phase33_state

        assert _tolerance_commit() == phase33_state.P332_11_MOS_TOLERANCE_COMMIT
        assert _readout_add_commit() == phase33_state.P332_11_MOS_READOUT_COMMIT


def test_the_witnessed_schema_width_is_the_live_one() -> None:
    """The slot's WeatherSchema width is measured, not transcribed from a guess.

    Was: the live width against ``P332_11_WEATHER_SCHEMA_FIELDS_AFTER_MOS`` (30). Plan 33.2-27
    Task 2 declared ONE more column, ``model_run_available_at``, and appended the new width in
    its own slot; the MOS slot is still asserted, now as the width before that one column.
    """
    from data.schemas import WeatherSchema
    from tests import phase33_state

    live = len(WeatherSchema.model_fields)
    assert live == phase33_state.P332_27_WEATHER_SCHEMA_FIELDS_AFTER_MODEL_STAMP
    assert set(phase33_state.P332_27_WEATHER_SCHEMA_NEW_FIELDS) <= set(
        WeatherSchema.model_fields
    )
    assert live - len(phase33_state.P332_27_WEATHER_SCHEMA_NEW_FIELDS) == (
        phase33_state.P332_11_WEATHER_SCHEMA_FIELDS_AFTER_MOS
    )


# ---------------------------------------------------------------------------
# Doc drift
# ---------------------------------------------------------------------------


class TestTheReadoutHolds:
    def test_the_readout_is_ascii(self) -> None:
        assert _readout().isascii()

    @pytest.mark.parametrize("marker", _REQUIRED_SECTION_MARKERS)
    def test_every_required_section_is_present(self, marker: str) -> None:
        assert marker in _readout()

    @pytest.mark.parametrize("phrase", _REQUIRED_PHRASES)
    def test_every_required_phrase_is_present(self, phrase: str) -> None:
        assert phrase in _readout()

    def test_the_over_claim_word_never_appears(self) -> None:
        assert _FORBIDDEN_WORD not in _readout().lower()

    def test_the_owner_rulings_are_recorded_verbatim(self) -> None:
        text = _readout()
        for key, answer in tol.OWNER_RULINGS.items():
            assert f'`{key}`: "{answer}"' in text

    @pytest.mark.parametrize("bound", tol.BOUND_NAMES)
    def test_each_bound_is_ruled_pass_with_and_without_the_subset(
        self, bound: str
    ) -> None:
        rows = [
            line for line in _readout().splitlines() if line.startswith(f"| {bound} |")
        ]
        assert len(rows) == 1, bound
        cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
        assert cells[3] == "PASS" and cells[5] == "PASS", rows[0]

    def test_the_verdict_is_pass(self) -> None:
        assert "**VERDICT: PASS**" in _readout()
