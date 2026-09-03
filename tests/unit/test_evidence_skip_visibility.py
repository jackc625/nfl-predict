"""WR-10: a control that did not run must not be invisible inside a green suite.

Several of this project's load-bearing controls read run records that are deliberately
gitignored -- fingerprint documents under ``outputs/``, the gold matrices under
``data/``, the deployed artifacts under ``artifacts/`` -- or need git history a shallow
clone does not have. Each guard is individually well-reasoned: a control that cannot run
must not report a false verdict.

The aggregate is what was unstated. ``GATED-REFIT-READOUT.md`` reconciles
"2282 passed / 7 skipped / 7 xfailed / 0 failed" against "2258 / 7 / 7 / 0" and treats
the skip count as a constant. That holds only on a machine that has just run the phase.
On any other checkout the skip count is materially higher and several of the phase's
central controls do not execute -- while the suite still reports zero failures.

``tests/conftest.py`` now prints a terminal-summary note naming exactly which
evidence-backed controls did not run. These tests pin the discrimination it rests on:
counting EVERY skip would make the note noise, and counting none would restore the
silence.

TEST CLASS: plain unit test. No data/, no artifacts/, no outputs/, no nested pytest run.
"""

from __future__ import annotations

import pytest

from tests.conftest import (
    _EVIDENCE_SKIP_MARKERS,
    _skip_reason,
    is_evidence_backed_skip,
    pytest_terminal_summary,
)

# The real skip reasons this project's controls emit, quoted from their call sites.
_REAL_EVIDENCE_SKIP_REASONS = [
    (
        "fingerprint document(s) absent: outputs/fingerprints/rung4.json. These are "
        "gitignored run records (.gitignore:26)."
    ),
    (
        "live silver games not present at data/silver/games.parquet -- data/ is "
        "gitignored runtime state."
    ),
    "live gold absent (features_wp) -- data/ is gitignored runtime state.",
    "production manifest not present at artifacts/latest.json",
    "data baselines tree not present at data/baselines",
    (
        "git history is unavailable (shallow clone or not a git checkout), so "
        "`git merge-base --is-ancestor` would fail for want of history."
    ),
    "data/gold not built on this checkout",
    (
        "artifacts/ is gitignored and absent on this checkout, so the deployed-artifact "
        "residue control did not run"
    ),
    # Plan 31-02's Wave-0 pre-ingest gates.
    "live silver odds not present at data/silver/odds_snapshot.parquet",
    "live gold absent (features_ou) -- data/ is gitignored runtime state.",
    (
        "the live nflreadpy 2025 schedule could not be loaded on this checkout "
        "(offline or upstream unavailable)"
    ),
]

# Skips that are NOT about absent evidence. A platform guard or an optional dependency
# is a legitimate, permanent skip and must stay out of the count.
_ORDINARY_SKIP_REASONS = [
    "requires Windows",
    "optional dependency not installed",
    "slow test deselected by default",
    "known upstream bug, tracked separately",
]


class TestTheDiscriminationIsReal:
    """The note must name evidence skips and only evidence skips."""

    @pytest.mark.parametrize("reason", _REAL_EVIDENCE_SKIP_REASONS)
    def test_every_real_evidence_skip_reason_is_recognised(self, reason: str) -> None:
        assert is_evidence_backed_skip(reason), (
            f"The skip reason {reason!r} is emitted by one of this project's "
            "evidence-backed controls but is not recognised as one, so that control "
            "would disappear silently into a green suite. Add its phrasing to "
            "tests/conftest._EVIDENCE_SKIP_MARKERS, or reword the guard to use an "
            "existing marker."
        )

    @pytest.mark.parametrize("reason", _ORDINARY_SKIP_REASONS)
    def test_an_ordinary_skip_is_not_counted(self, reason: str) -> None:
        assert not is_evidence_backed_skip(reason), (
            f"The ordinary skip reason {reason!r} was counted as an evidence-backed "
            "control. Counting every skip turns the note into noise and it stops "
            "being read, which is the same failure in a different costume."
        )

    def test_matching_is_case_insensitive(self) -> None:
        assert is_evidence_backed_skip("LIVE GOLD ABSENT")

    def test_an_empty_reason_is_not_counted(self) -> None:
        assert not is_evidence_backed_skip("")


class TestTheReasonIsExtractedFromPytestsShape:
    """``_skip_reason`` must read the shape pytest actually produces."""

    def test_the_three_tuple_longrepr_is_unwrapped(self) -> None:
        report = type(
            "R",
            (),
            {"longrepr": ("tests/x.py", 12, "Skipped: live gold absent -- gitignored")},
        )()
        assert _skip_reason(report) == "live gold absent -- gitignored"

    def test_a_reason_without_the_skipped_prefix_survives_intact(self) -> None:
        report = type("R", (), {"longrepr": ("tests/x.py", 12, "live gold absent")})()
        assert _skip_reason(report) == "live gold absent"

    def test_an_absent_longrepr_yields_an_empty_reason(self) -> None:
        assert _skip_reason(type("R", (), {"longrepr": None})()) == ""


class TestTheSummaryHookReports:
    """The hook itself, driven over a stub reporter."""

    class _Reporter:
        def __init__(self, stats: dict) -> None:
            self.stats = stats
            self.lines: list[str] = []

        def write_sep(self, _sep: str, title: str) -> None:
            self.lines.append(title)

        def write_line(self, line: str) -> None:
            self.lines.append(line)

    @staticmethod
    def _report(nodeid: str, reason: str):
        return type(
            "R", (), {"nodeid": nodeid, "longrepr": ("f.py", 1, f"Skipped: {reason}")}
        )()

    def test_it_names_each_control_that_did_not_run(self) -> None:
        reporter = self._Reporter(
            {
                "skipped": [
                    self._report("tests/a.py::test_one", "live gold absent"),
                    self._report("tests/b.py::test_two", "requires Windows"),
                ]
            }
        )
        pytest_terminal_summary(reporter)

        printed = "\n".join(reporter.lines)
        assert "1 of 2 skipped test(s)" in printed
        assert "tests/a.py::test_one" in printed
        assert "tests/b.py::test_two" not in printed
        assert "does not include them" in printed

    def test_it_stays_silent_when_every_control_ran(self) -> None:
        reporter = self._Reporter(
            {"skipped": [self._report("tests/b.py::test_two", "requires Windows")]}
        )
        pytest_terminal_summary(reporter)

        assert reporter.lines == [], (
            "The note fired on a run where no evidence-backed control was skipped. "
            "It must appear only when it has something to say."
        )

    def test_it_stays_silent_on_a_run_with_no_skips_at_all(self) -> None:
        reporter = self._Reporter({})
        pytest_terminal_summary(reporter)
        assert reporter.lines == []


def test_the_marker_vocabulary_is_not_empty() -> None:
    """A cleared marker list would silently disable the whole control."""
    assert _EVIDENCE_SKIP_MARKERS
    assert "gitignored" in _EVIDENCE_SKIP_MARKERS
