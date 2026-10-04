"""Permanent doc-drift guard for the dated old-rule label on repo-root readouts (Phase 33.2, R16).

D33.2-07: only the 2026 season, recorded live under the day-before-kickoff lock, is evidence of
whether the system works. Every older model or betting result stays in the record, but a number that
stays unlabelled reads as current evidence. So each repo-root readout that reports a pre-fix result
carries an ADDENDUM -- appended, never a rewrite of its original numbers -- that reads "built under
the old rule on inputs later found defective; not evidence".

This is a PERMANENT committed test, not a throwaway ``scripts/check_*.py``, following
``tests/unit/test_signal_lift_readout_md.py``. It guards against these failure modes:

  - a repo-root readout reporting pre-fix results is simply not on the list, so it goes unlabelled
    while a hand-maintained name list still passes. The set is therefore PARTITIONED against
    ``git ls-files '*.md'`` at the repo root: every tracked file is either labelled or carries a
    written reason why it needs no label, and a readout committed later fails here until someone
    decides which side it belongs on;
  - a labelled readout loses its label, its date, or its opening sentinel line;
  - non-ASCII enters a labelled file (CLAUDE.md hard constraint);
  - the addendum picks up a word from the repo's over-claim dictionary
    (``backtest.ev_chain_constants.READOUT_FORBIDDEN_WORDS``, imported rather than retyped). The
    check is scoped to the text AFTER the sentinel line, never the whole file: several labelled
    readouts already use ``proven`` or ``validated`` in preserved original content that R16
    forbids editing.

It asserts the RULING (the label is present and dated), never a point estimate from any document --
pinning a number to moving gold is a mistake this repo already made once (D29-06-02).

The tracked listing is used rather than a directory glob so untracked working notes lying in the
tree cannot change what this test checks.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# Repo root resolved from this file: tests/unit/test_old_rule_labels.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The verbatim label phrase every labelled readout carries (matched case-insensitively).
LABEL_PHRASE = "built under the old rule on inputs later found defective; not evidence"

# The date the addendum carries in its heading (the day D33.2-07 was ruled).
ADDENDUM_DATE = "2026-09-15"

# The exact opening line of every addendum. It lets the addendum be located by splitting on it
# rather than guessing at a byte offset, which is what scopes the forbidden-word check to the text
# this phase wrote.
ADDENDUM_SENTINEL = "<!-- old-rule-addendum-2026-09-15 -->"

# Repo-root readouts that report a pre-fix model or betting result and therefore carry the addendum.
LABELLED_READOUTS: tuple[str, ...] = (
    "ACTIVATION-READOUT.md",
    "AUDIT-REPORT.md",
    "GATED-REFIT-READOUT.md",
    "HISTORICAL-WEATHER-READOUT.md",
    "LINE-MOVEMENT-READOUT.md",
    "METHODOLOGY.md",
    "MODEL-DIAGNOSIS.md",
    "OU-DIVERGENCE-DIAGNOSIS.md",
    "PROFITABILITY-READOUT.md",
    "README.md",
    "RUNBOOK.md",
    "SELECTION-CENSUS.md",
    "SIGNAL-LIFT-READOUT.md",
    "STATE-OF-SYSTEM.md",
)

# Tracked repo-root markdown that carries no addendum, each with the reason found on reading it.
NO_PREFIX_NUMBERS_REASONS: dict[str, str] = {
    "BLEND-TUNING-READOUT.md": (
        "Plan 33.2-24's blend re-tune, written under the day-before lock on the corrected models "
        "and on lines owned before each game's lock. It reports blend weights and their losses "
        "on re-measured 2020-2024 seasons and carries its own 'not clean evidence' label "
        "(D33.2-07); it rests on no pre-fix model and no closing line, so the old-rule addendum "
        "does not describe it."
    ),
    "CLOSING-LINE-AUDIT.md": (
        "Plan 33.2-21's audit of every place a closing line feeds a fit, with one disposition "
        "each. It lists code sites and their fates; it reports no model accuracy or betting "
        "result and rests on no pre-fix model."
    ),
    "GROUP-VERDICT-READOUT.md": (
        "Plan 33.2-22's re-measured feature-group verdict, written under the day-before lock on "
        "corrected gold. It decides which feature families a model may be fitted on and carries "
        "its own 'not clean evidence' label (D33.2-07); it reports no accuracy or betting result "
        "and rests on no pre-fix model."
    ),
    "AUTOMATION.md": (
        "Operational explanation of the scheduled run: what fires, the orchestrator steps, where "
        "logs land and how to tell a run succeeded. Its only figures describe a smoke check of the "
        "pipeline plumbing, not model accuracy or betting results."
    ),
    "CLAUDE.md": (
        "The agent instruction file every session loads, not a readout. It does restate the "
        "Phase-30/31 results as project status, but changing the instructions an agent runs under "
        "is the owner's call, not an executor's; flagged for the owner in the 33.2-07 summary "
        "instead of being edited here."
    ),
    "COLD-START-PREREGISTRATION.md": (
        "A sealed pre-registration. SPEC R14 forbids editing it in place; Plan 33.2-26 supersedes "
        "it by a later corrective commit that names its anchor instead."
    ),
    "MOS-DECODE-COMPARISON.md": (
        "Plan 33.2-11's decode check of the archived day-before forecast bulletins against the "
        "on-disk observations, written under the day-before lock. Its figures are detection "
        "bounds on data decoding, not a model accuracy or betting result, and it rests on no "
        "pre-fix model."
    ),
    "WEATHER-NULL-LIST.md": (
        "Plan 33.2-12's list of the 2002-2025 games that carry no day-before forecast (NULL "
        "weather plus a coverage flag), with the reason for each, written under the day-before "
        "lock. It counts games and names venues; it reports no model accuracy or betting result "
        "and rests on no pre-fix model."
    ),
    "PRECOVERAGE-SCAN.md": (
        "Plan 33.2-17's re-scan of the three gold matrices for constant blocks before each "
        "family's first covered season, written under the day-before lock. It classifies and "
        "counts gold columns and names the planned fix for each; it reports no model accuracy "
        "or betting result and rests on no pre-fix model."
    ),
    "PIPELINE.md": (
        "Canonical run-sequence instructions (commands, entry points, expected outputs). It "
        "reports no model or betting result."
    ),
    "REFIT-READOUT.md": (
        "Plan 33.2-25's record of the corrected models and blend the production swap installs, "
        "written under the day-before lock on corrected gold. It reports re-measured past "
        "seasons from folds that fit only on earlier seasons and labels every results section "
        "'not clean evidence' (D33.2-07); it names no pre-fix model and no gate baseline, so "
        "the old-rule addendum does not describe it."
    ),
    "COLD-START-CORRECTION.md": (
        "Plan 33.2-26's superseding correction of the 2026 cold-start rule frozen at 11761c7, "
        "re-derived on the three corrected models and on lines owned before each game's lock. It "
        "names the superseded record but reports no result from it, and labels its figures 'not "
        "clean evidence' (D33.2-07); it rests on no pre-fix model, so the old-rule addendum does "
        "not describe it."
    ),
    "EV-CHAIN-CORRECTION.md": (
        "Plan 33.2-29's re-derivation of the EV floor and frozen residual SD on the three "
        "corrected models and on lines owned before each game's lock. It lists the superseded "
        "ee20773 values only as the numbers being replaced and labels its figures 'not clean "
        "evidence' (D33.2-07); it rests on no pre-fix model, so the old-rule addendum does not "
        "describe it."
    ),
    "NEUTRAL-HFA-BET-RULE-CORRECTION.md": (
        "Quick task 261003-vke's superseding correction of the 2026 bet rule frozen at 9bb7568, "
        "re-derived by the same recipe on the three models re-fit after the neutral-site Elo "
        "fix (WINDOWS row 19) and on lines owned before each game's lock. It names the "
        "superseded values only as the numbers being replaced and labels its figures 'not clean "
        "evidence' (D33.2-07); it rests on no pre-fix model, so the old-rule addendum does not "
        "describe it."
    ),
    "LIVE-COLD-START-READOUT.md": (
        "Plan 33-18's closing record of Phase 33, written under the day-before lock. Its "
        "measurements are the 2026 live scheduled runs'; it names Phase 33's superseded models, "
        "gate verdicts and frozen numbers only as superseded records beside their replacements "
        "and restates no result from them, so the old-rule addendum does not describe it."
    ),
    "PROFITABILITY-PREREGISTRATION.md": (
        "A sealed pre-registration whose ancestry anchor is checked by "
        "tests/unit/test_preregistration_ancestry.py; editing it in place would destroy that "
        "evidence. The 2025 results it governed are labelled in PROFITABILITY-READOUT.md."
    ),
}


def tracked_root_markdown() -> list[str]:
    """Return every TRACKED markdown file at the repo root (no subdirectories), sorted."""
    listing = subprocess.run(
        ["git", "ls-files", "*.md"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.split()
    return sorted(path for path in listing if "/" not in path)


def _partition_problems() -> dict[str, list[str]]:
    """Describe every way the two sets fail to partition the tracked repo-root listing."""
    tracked = set(tracked_root_markdown())
    labelled = set(LABELLED_READOUTS)
    reasoned = set(NO_PREFIX_NUMBERS_REASONS)
    return {
        "unaccounted": sorted(tracked - labelled - reasoned),
        "in_both_sets": sorted(labelled & reasoned),
        "not_tracked": sorted((labelled | reasoned) - tracked),
        "empty_reason": sorted(
            k for k, v in NO_PREFIX_NUMBERS_REASONS.items() if not v.strip()
        ),
    }


# Collection-time partition check: a readout committed later fails the module at import, before any
# individual test runs, rather than going uncovered.
_PROBLEMS = {name: items for name, items in _partition_problems().items() if items}
if _PROBLEMS:
    raise AssertionError(
        "LABELLED_READOUTS and NO_PREFIX_NUMBERS_REASONS must partition the tracked repo-root "
        f"*.md listing exactly: {_PROBLEMS}"
    )


def _read(name: str) -> str:
    """Read one repo-root readout as UTF-8 text."""
    return (REPO_ROOT / name).read_text(encoding="utf-8")


class TestTheReadoutSetIsAPartition:
    """Every tracked repo-root markdown file is labelled or carries a written reason -- never both."""

    def test_every_tracked_root_markdown_file_is_accounted_for(self) -> None:
        """No tracked repo-root *.md is in neither set, none is in both, and none is untracked."""
        problems = _partition_problems()
        assert problems == {
            "unaccounted": [],
            "in_both_sets": [],
            "not_tracked": [],
            "empty_reason": [],
        }, problems

    def test_the_listing_is_not_empty(self) -> None:
        """Non-vacuity control: a failed git listing must not make the partition trivially pass."""
        assert tracked_root_markdown(), "git ls-files returned no repo-root markdown"

    @pytest.mark.parametrize(
        "sealed", ["COLD-START-PREREGISTRATION.md", "PROFITABILITY-PREREGISTRATION.md"]
    )
    def test_sealed_preregistrations_are_not_labelled(self, sealed: str) -> None:
        """Sealed pre-registrations are never edited in place (SPEC R14), so never labelled."""
        assert sealed not in LABELLED_READOUTS


@pytest.mark.parametrize("name", LABELLED_READOUTS)
class TestEachLabelledReadout:
    """Each labelled readout carries the dated label and stays ASCII."""

    def test_carries_the_label_phrase(self, name: str) -> None:
        """The verbatim label phrase is present (case-insensitive)."""
        assert LABEL_PHRASE in _read(name).lower(), (
            f"{name} is missing the old-rule label"
        )

    def test_carries_the_addendum_date(self, name: str) -> None:
        """The addendum is dated."""
        assert ADDENDUM_DATE in _read(name), (
            f"{name} does not carry the date {ADDENDUM_DATE}"
        )

    def test_is_ascii(self, name: str) -> None:
        """Content is ASCII-only (CLAUDE.md)."""
        assert _read(name).isascii(), f"{name} contains non-ASCII characters"


def _addendum(name: str) -> str:
    """The text AFTER the addendum sentinel -- the part of the file this phase wrote."""
    text = _read(name)
    assert ADDENDUM_SENTINEL in text, f"{name} carries no old-rule addendum sentinel"
    return text.split(ADDENDUM_SENTINEL, 1)[1]


@pytest.mark.parametrize("name", LABELLED_READOUTS)
class TestEachAddendum:
    """The appended addendum itself: located by its sentinel, checked on its own text only."""

    def test_opens_with_exactly_one_sentinel(self, name: str) -> None:
        assert _read(name).count(ADDENDUM_SENTINEL) == 1

    def test_the_addendum_carries_the_label(self, name: str) -> None:
        assert LABEL_PHRASE in _addendum(name).lower()

    def test_the_addendum_heading_is_dated(self, name: str) -> None:
        """The first level-2 heading after the sentinel carries the date, so the label is dated."""
        headings = [
            line for line in _addendum(name).splitlines() if line.startswith("## ")
        ]
        assert headings, f"{name}'s addendum has no level-2 heading"
        assert ADDENDUM_DATE in headings[0], headings[0]

    def test_the_addendum_carries_no_over_claim_word(self, name: str) -> None:
        """Scoped to the addendum: several originals use these words in preserved content."""
        from backtest.ev_chain_constants import READOUT_FORBIDDEN_WORDS

        assert READOUT_FORBIDDEN_WORDS, "the imported over-claim dictionary is empty"
        suffix = _addendum(name).lower()
        present = [word for word in READOUT_FORBIDDEN_WORDS if word in suffix]
        assert not present, f"{name}'s addendum uses over-claim words {present}"

    def test_the_addendum_says_deployed_nowhere(self, name: str) -> None:
        """The per-readout guards forbid ``deployed`` in scoped sections; the addendum avoids it."""
        assert "deployed" not in _addendum(name).lower()

    def test_the_addendum_is_the_end_of_the_file(self, name: str) -> None:
        """Appended, never inserted: no original heading follows the addendum's own."""
        headings = [
            line for line in _addendum(name).splitlines() if line.startswith("## ")
        ]
        assert len(headings) == 1, (
            f"{name} has content headings after its addendum: {headings}"
        )


LABELLING_COMMIT = "7b1928f965ba5e3cbc4a1a4f918015df34aec8aa"


def _git_history_is_unavailable() -> bool:
    """True when this is not a git checkout, or a shallow one, or the commit is absent."""

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )

    if run("rev-parse", "--git-dir").returncode != 0:
        return True
    shallow = run("rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0 or shallow.stdout.strip() == "true":
        return True
    return run("cat-file", "-e", f"{LABELLING_COMMIT}^{{commit}}").returncode != 0


def _labelling_numstat(commit: str = LABELLING_COMMIT) -> dict[str, tuple[int, int]]:
    """{path: (added, deleted)} for every repo-root markdown file the labelling commit touched."""
    out = subprocess.run(
        ["git", "show", "--numstat", "--format=", commit],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    stats: dict[str, tuple[int, int]] = {}
    for line in out.splitlines():
        added, deleted, path = line.split("\t", 2)
        # The commit also carried this guard and a readout-test edit; only repo-root markdown
        # is a readout.
        if "/" not in path and path.endswith(".md"):
            stats[path] = (int(added), int(deleted))
    return stats


def _files_with_deletions(stats: dict[str, tuple[int, int]]) -> list[str]:
    return sorted(path for path, (_, deleted) in stats.items() if deleted != 0)


@pytest.mark.skipif(
    _git_history_is_unavailable(),
    reason=(
        "git history is unavailable (shallow clone, not a git checkout, or the labelling "
        "commit is absent), so its numstat cannot be read; skipping rather than reporting a "
        "false violation."
    ),
)
class TestTheLabellingCommitOnlyAppended:
    """R16: 'original numbers unchanged' -- pinned on the labelling commit itself.

    Later edits to these files by OTHER commits are out of scope; only commit 7b1928f is read.
    """

    def test_it_deleted_zero_lines_from_every_readout_it_touched(self) -> None:
        stats = _labelling_numstat()
        assert stats, "the labelling commit touched nothing; the check is vacuous"
        assert _files_with_deletions(stats) == [], (
            "the labelling commit deleted original lines: "
            f"{ {p: stats[p] for p in _files_with_deletions(stats)} }"
        )

    def test_it_touched_exactly_the_declared_labelled_set(self) -> None:
        assert set(_labelling_numstat()) == set(LABELLED_READOUTS)

    def test_a_planted_deletion_would_be_caught(self) -> None:
        planted = {"README.md": (22, 0), "RUNBOOK.md": (21, 1)}
        assert _files_with_deletions(planted) == ["RUNBOOK.md"]
