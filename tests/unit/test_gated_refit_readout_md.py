"""Permanent doc-drift guard for the repo-root GATED-REFIT-READOUT.md (Phase 30, PROD-01).

Modelled on ``tests/unit/test_signal_lift_readout_md.py`` -- the committed doc-drift-guard pattern
in this repository -- with TWO deliberate divergences from it, both recorded here because copying
the Phase-28 guard verbatim would have produced a guard that is either unsatisfiable or
meaningless.

DIVERGENCE 1 -- THE VOCABULARY IS SCOPED, NOT BLANKET.
The Phase-28 guard bans the over-claim words with a blanket substring check over the whole lowered
document. That is exactly right for a SCREEN phase: Phase 28 screened three feature groups and
deployed nothing, so the word "deployed" could not honestly appear anywhere in its readout. Phase
30 is a DEPLOY phase. Its entire purpose was to promote gate-passing targets, one target actually
swapped, and a document that may not say so cannot describe what happened. A blanket ban here
would be unsatisfiable -- and an unsatisfiable guard gets relaxed until it means nothing, which is
worse than no guard at all.

So the ban follows the CLAIM rather than the TOKEN. The over-claim words are forbidden INSIDE the
group-verdict section and INSIDE the retained-target section, which are parsed out of the document
first: a feature group is never "deployed" (a group is kept or dropped, and a KEEP only carries it
into a candidate), and a target the gate REFUSED is never "deployed" either. The two strongest
words are additionally forbidden GLOBALLY, matching the fixed contaminated vocabulary that
``backtest/ou_monetization.py`` already freezes -- no number in this phase may be called those
things regardless of what was promoted.

Matching is by WORD BOUNDARY, never by naive substring. That is a fix for a known defect class in
this repository's guards (D30-DEFER-05): a substring check for "proven" false-positives on
"PROVENANCE", and one for "validated" false-positives on "re-validated". A guard that reddens on
an innocent word teaches a maintainer to ignore it.

DIVERGENCE 2 -- THE ANCESTRY ASSERTION IS BETWEEN TWO SPECIFIC COMMITS.
The Phase-29 guard asserts each recorded pre-registration commit is an ancestor of the CURRENT
HEAD. That is weaker than SPEC R4 asks for: everything in history is an ancestor of HEAD, so it
cannot distinguish a rule written before the measurement from one written after it and merged
later. SPEC R4's claim is an ordering between the RULE commit and the MEASUREMENT commit
specifically. This guard therefore resolves the rule commit with ``git log`` against the frozen
constants module, reads the measurement commit from the readout's own marker line, asserts
ancestry between exactly those two, and additionally asserts they are DIFFERENT commits -- a rule
and the results it produced landing in one commit is not a pre-registration, it is only a claim of
one.

THE SHALLOW-CHECKOUT GUARD IS VERIFIED, NOT ASSUMED.
In a ``--depth=1`` clone ``git merge-base --is-ancestor`` fails because the history is not
present, and from the exit code alone that is INDISTINGUISHABLE from a genuine ancestry violation.
Left unguarded it would turn the phase's central honesty assertion into a false red in CI, and
would train a future maintainer to ignore the one assertion that must never be ignored. The probe
consults ``git rev-parse --is-shallow-repository`` and the presence of a git directory and skips
BEFORE any ancestry call is issued -- and ``test_the_shallow_checkout_guard_actually_fires``
monkeypatches the probe to report shallow and asserts the documented skip, so the wiring is proved
rather than trusted.

WHAT THE HARNESS-REPRODUCTION CLASS PINS, AND WHAT IT DELIBERATELY DOES NOT.
It asserts each group's VERDICT STRING still reproduces from ``run_group_gate``, and separately
asserts the document still CONTAINS its published point estimates as RECORDED values. It does NOT
assert that a point estimate still reproduces. v3.0 rebuilds gold on purpose -- this phase rebuilt
it four times -- so pinning a point estimate against moving gold guarantees a red for a reason that
is not drift. D29-06-02 already corrected that mistake once in this repository, and re-anchoring a
published number to each new measurement was considered there and rejected: it rewrites a
published record to match a moving input and drifts again on the next rebuild. What IS permanent
is the RULING. If a numeric tripwire is ever wanted here, pin a corrected cell's REJECTION STATUS,
never its q-value.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from backtest.group_gate_constants import (
    VERDICT_DROP,
    VERDICT_KEEP,
    VERDICT_NOT_MEASURED,
    VERDICT_UNDETERMINED,
)
from backtest.ou_monetization import CONTAMINATED_VOCAB
from tests.phase30_state import GROUP_VERDICT_FILE_SHA256, MEASUREMENT_COMMIT

# Repo root resolved from this file: tests/unit/test_gated_refit_readout_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
READOUT_MD = REPO_ROOT / "docs" / "records" / "GATED-REFIT-READOUT.md"

# The FROZEN pre-registration module whose last-modifying commit IS the rule commit. Named as a
# path rather than imported, because what is resolved from it is a git fact, not a Python value.
FROZEN_RULE_MODULE = "backtest/group_gate_constants.py"

# The ratified Stage-1 verdict, committed as the measurement commit. Read rather than re-derived:
# it is the record the readout publishes, so it is the right thing to check the readout against
# without paying for a full harness run.
VERDICT_TOML = REPO_ROOT / "config" / "group_gate_verdict.toml"

# Gold / odds presence skip-guard for the harness-reproduction class only.
_GOLD_PATHS = {
    target: REPO_ROOT / "data" / "gold" / f"features_{target}.parquet"
    for target in ("wp", "ats", "ou")
}
_ODDS_PATH = REPO_ROOT / "data" / "silver" / "odds_snapshot.parquet"

# Required section markers. Each is a FRAGMENT of a section rather than a whole heading, so a
# benign reword passes and a dropped section fails.
_REQUIRED_SECTION_MARKERS = (
    "The four-rung rebuild",  # 1
    "The N-01 positive control",  # 2
    "The `line_movement` DROP",  # 3
    "The pre-registration",  # 4
    "The corrected 9-cell grid",  # 5
    "DIRECTIONAL EVIDENCE",  # 5d -- the D30-OWNER-13 honesty requirement
    "The per-target deploy outcome",  # 6
    "The O/U monetization chain",  # 7
    "re-frozen TWICE",  # 8
    "The dynamic blend was re-checked",  # 9
    "Scope notes",  # 10
    "The Phase-30 state manifest",  # 11
    "null result is a complete result",  # the SPEC R8 publish-anyway rule, stated in words
)

# The headings the two vocabulary-scoped sections are parsed out by, with the pattern that ends
# each. Section 5 runs to the next TOP-LEVEL heading so its subsections are included; 6b runs to
# the next heading of any level so it is exactly the retained-target section.
_GROUP_VERDICT_SECTION_START = "## 5. The corrected 9-cell grid"
_RETAINED_TARGET_SECTION_START = "### 6b. The two REFUSED targets"
_TOP_LEVEL_HEADING = r"^## "
_ANY_SECTION_HEADING = r"^#{2,3} "

# The over-claim words, in two tiers.
#
# GLOBAL: no number produced by this phase may be called these, whatever was promoted. They are
# the same two the frozen contaminated vocabulary in backtest/ou_monetization.py forbids of the
# 2023-2024 hold numbers, and the reason is the same -- they assert a standard of evidence that a
# non-regression gate and a burned holdout do not supply.
_GLOBALLY_FORBIDDEN_WORDS = ("proven", "validated")

# SCOPED: additionally forbidden inside the group-verdict and retained-target sections. "deployed"
# is NOT banned document-wide, because one target genuinely was.
_SECTION_SCOPED_FORBIDDEN_WORDS = ("deployed", *_GLOBALLY_FORBIDDEN_WORDS)

# The frozen four-value vocabulary, imported rather than re-typed. Hand-written string literals in
# acceptance checks are exactly what importing the frozen constants prevents: Plan 30-10 recorded
# TWO plan-level defects of that shape ('NOT_MEASURED' with an underscore against the frozen
# "NOT MEASURED" with a space, and an underscore inside preregistration_commit), and both passed or
# failed for reasons unrelated to the measurement.
_FROZEN_VERDICT_VOCABULARY = (
    VERDICT_KEEP,
    VERDICT_DROP,
    VERDICT_UNDETERMINED,
    VERDICT_NOT_MEASURED,
)

# The published point estimates, pinned as RECORDED values -- NOT as reproducing ones. See the
# module docstring.
_RECORDED_POINT_ESTIMATES = (
    "+0.004297",  # injury/wp   (excluded from the family)
    "-0.295307",  # injury/ats  (BH rejected, negative)
    "-0.156578",  # injury/ou
    "+0.013158",  # snap/wp     (BH rejected, positive -- rank 1)
    "-0.240678",  # snap/ats    (BH rejected, negative)
    "-0.227069",  # snap/ou     (BH rejected, negative)
    "+0.001806",  # situational/wp  (excluded)
    "-0.024307",  # situational/ats (excluded)
    "+0.361631",  # situational/ou  (BH rejected, positive -- rank 2)
)

# The machine-readable ancestry markers the readout carries in section 4.
_PRE_REGISTRATION_MARKER_RE = re.compile(
    r"^pre_registration_commit:\s*([0-9a-f]{40})\s*$", re.MULTILINE
)
_MEASUREMENT_MARKER_RE = re.compile(
    r"^measurement_commit:\s*([0-9a-f]{40})\s*$", re.MULTILINE
)

# The documented message the shallow / non-git guard skips with. Pinned as a constant so the test
# that proves the guard fires matches the SAME text the guard emits, rather than a paraphrase that
# could drift away from it.
SHALLOW_SKIP_MESSAGE = (
    "git history is unavailable (shallow clone or not a git checkout), so "
    "`git merge-base --is-ancestor` would fail for want of history rather than for want of "
    "ancestry -- and the two are indistinguishable from the exit code alone. Skipping BEFORE "
    "any ancestry call rather than reporting a false ancestry violation."
)


def _read_readout() -> str:
    """Read GATED-REFIT-READOUT.md from the repo root."""
    return READOUT_MD.read_text(encoding="utf-8")


def _section(content: str, start_marker: str, stop_pattern: str) -> str:
    """Return the slice of ``content`` from ``start_marker`` to the next heading matching it.

    Args:
        content: The whole readout.
        start_marker: A literal substring that begins the section (its heading).
        stop_pattern: A MULTILINE regex matching the heading that ENDS the section.

    Returns:
        The section text, heading included.
    """
    start = content.find(start_marker)
    assert start != -1, (
        f"the readout has no section beginning {start_marker!r}. The scoped vocabulary check "
        "cannot run against a section that is not there, and a guard that silently checks "
        "nothing is worse than no guard -- restore the section or update this marker."
    )
    tail = content[start + len(start_marker) :]
    stop = re.search(stop_pattern, tail, re.MULTILINE)
    return start_marker + (tail[: stop.start()] if stop else tail)


def _whole_word_hits(text: str, word: str) -> int:
    """Count WORD-BOUNDARY occurrences of ``word`` in ``text``, case-insensitively.

    Word boundaries rather than substrings, deliberately: a substring check for "proven"
    false-positives on "provenance" and one for "validated" false-positives on "re-validated"
    (D30-DEFER-05). A guard that reddens on an innocent word gets ignored.
    """
    return len(re.findall(rf"\b{re.escape(word)}\b", text, re.IGNORECASE))


def _verdict_cells_for(content: str, group: str) -> list[str]:
    """Return the SECOND cell of every markdown table row whose FIRST cell names ``group``.

    Cell-level rather than row-level, and that distinction is load-bearing rather than tidy. A
    row-level substring check passes as long as the verdict word appears ANYWHERE in the row --
    and every row of the readout's verdict table carries the generator's own reason string in its
    third cell, which begins with that same word ("DROP: significantly NEGATIVE after
    correction..."). So a row-level check stays green when the verdict CELL is flipped from DROP
    to KEEP while the reason beside it still says DROP, which is exactly the drift this guard
    exists to catch. Measured: that mutation passed a row-level check and fails this one.

    Emphasis markers are stripped, so a verdict written ``**DROP**`` for typographic reasons is
    read as ``DROP``.
    """
    cells: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [
            cell.strip().strip("*").strip() for cell in stripped.strip("|").split("|")
        ]
        if len(parts) >= 2 and parts[0] == group:
            cells.append(parts[1])
    return cells


def _ratified_verdicts() -> dict[str, str]:
    """Return {group -> verdict} from the RATIFIED Phase-30 Stage-1 verdict document.

    READ FROM ITS IMMUTABLE GIT BLOB (Plan 33.2-22, D33.2-15), at
    ``tests.phase30_state.MEASUREMENT_COMMIT``, and digest-checked against
    ``GROUP_VERDICT_FILE_SHA256`` over newline-normalized bytes.

    It used to read the WORKING-TREE ``config/group_gate_verdict.toml``, which was the right
    place while that file WAS the Phase-30 ratified document. From Phase 33.2 the working-tree
    file is the RE-MEASURED verdict -- the same frozen rule re-run on corrected gold under an
    outcome-loss objective, anchored by its own ``P332_22_*`` witness. GATED-REFIT-READOUT.md
    is the PHASE-30 record and must keep being checked against the PHASE-30 document; the
    re-measured verdict is published in ``GROUP-VERDICT-READOUT.md`` and guarded there.

    ``VERDICT_TOML`` is retained above and is still read by the tests that are about the LIVE
    file. Nothing here was relaxed: the same bytes are parsed, from the one place they cannot
    change.
    """
    if _git_history_is_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)
    blob = _git(
        "cat-file", "blob", f"{MEASUREMENT_COMMIT}:config/group_gate_verdict.toml"
    )
    assert blob.returncode == 0, (
        f"git could not read the ratified verdict document at {MEASUREMENT_COMMIT}: "
        f"{blob.stderr}. It is the Phase-30 measurement commit's only path, so this means the "
        "checkout is broken, not that the check should be skipped."
    )
    raw = blob.stdout.encode("utf-8") if isinstance(blob.stdout, str) else blob.stdout
    normalized = raw.replace(b"\r\n", b"\n")
    assert hashlib.sha256(normalized).hexdigest() == GROUP_VERDICT_FILE_SHA256, (
        "the ratified Phase-30 verdict at MEASUREMENT_COMMIT no longer matches its anchor. "
        "That document is history and cannot change."
    )
    document = tomllib.loads(normalized.decode("utf-8"))
    return {
        group: entry["verdict"]
        for group, entry in document["stage1"]["verdicts"].items()
    }


# ---------------------------------------------------------------------------
# git helpers, and the shallow / non-git probe the ancestry class consults FIRST
# ---------------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run one git command in the repo root, never raising on a non-zero exit."""
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_history_is_unavailable() -> bool:
    """Return True when git history cannot support an ancestry question.

    Two cases, both of which make `git merge-base --is-ancestor` fail for a reason that has
    nothing to do with ancestry: this is not a git checkout at all, or it is a shallow one whose
    history was truncated by `--depth`.

    This is a module-level function precisely so it can be monkeypatched -- see
    ``test_the_shallow_checkout_guard_actually_fires``, which proves the guard fires rather than
    trusting that the probe was wired correctly.
    """
    if _git("rev-parse", "--git-dir").returncode != 0:
        return True
    shallow = _git("rev-parse", "--is-shallow-repository")
    return shallow.returncode != 0 or shallow.stdout.strip() == "true"


def _rule_commit() -> str:
    """Resolve the LAST commit to touch the frozen pre-registration module."""
    result = _git("log", "-1", "--format=%H", "--", FROZEN_RULE_MODULE)
    return result.stdout.strip()


def _assert_rule_precedes_measurement() -> None:
    """Assert the RULE commit is a strict git ancestor of the MEASUREMENT commit.

    Factored out of its test so the shallow-guard proof can invoke exactly this code path under a
    monkeypatched probe. The skip is issued BEFORE any `merge-base` call, which is the whole
    point: in a `--depth=1` clone that call fails for want of history, and the exit code cannot
    tell that apart from a genuine ancestry violation.
    """
    if _git_history_is_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)

    content = _read_readout()
    measurement_shas = _MEASUREMENT_MARKER_RE.findall(content)
    assert len(measurement_shas) == 1, (
        "the readout must carry exactly ONE machine-readable `measurement_commit:` marker line "
        f"(found {len(measurement_shas)}). SPEC R4's ancestry claim is between two SPECIFIC "
        "commits, so the measurement commit has to be readable from the document itself rather "
        "than inferred."
    )
    measurement_sha = measurement_shas[0]

    rule_sha = _rule_commit()
    assert rule_sha, (
        f"git could not resolve the last commit to touch {FROZEN_RULE_MODULE}. Without it there "
        "is no rule commit to assert ancestry from."
    )

    for sha, label in ((rule_sha, "rule"), (measurement_sha, "measurement")):
        assert _git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0, (
            f"the {label} commit {sha} is not present in this checkout, yet the repository is "
            "neither shallow nor non-git. That is a broken checkout, not a reason to skip."
        )

    assert rule_sha != measurement_sha, (
        f"the frozen rule and the measurement it produced are the SAME commit ({rule_sha}). "
        "A rule and its results landing in one commit is not a pre-registration; it is only a "
        "claim of one. The remedy is never to relax this assertion -- it is that the rule must "
        "be committed, alone, before the measurement runs."
    )

    ancestry = _git("merge-base", "--is-ancestor", rule_sha, measurement_sha)
    assert ancestry.returncode == 0, (
        f"the frozen rule commit {rule_sha} (last to touch {FROZEN_RULE_MODULE}) is NOT an "
        f"ancestor of the measurement commit {measurement_sha} recorded in the readout. The rule "
        "cannot be shown to have been fixed before the numbers existed, which is the single "
        "mechanical anti-rule-shopping proof this phase has. Do NOT edit the frozen rule module "
        "to reconcile this: editing a pre-registration after the measurement destroys the "
        "evidence rather than fixing it."
    )


class TestReadoutExists:
    """GATED-REFIT-READOUT.md must exist at the repo root and be ASCII-only."""

    def test_file_exists_at_repo_root(self) -> None:
        """The PROD-01 deliverable is at the repo root, NOT under the gitignored .planning/."""
        assert READOUT_MD.is_file(), (
            f"missing: {READOUT_MD}. The readout is one of only two TRACKED homes for the "
            "Phase-30 state manifest (the other is tests/phase30_state.py); outputs/, "
            "artifacts/, data/ and .planning/ are all gitignored."
        )

    def test_content_is_ascii(self) -> None:
        """Content is ASCII-only (no emoji / non-ASCII, per CLAUDE.md)."""
        content = _read_readout()
        offenders = sorted({ch for ch in content if not ch.isascii()})
        assert not offenders, (
            f"GATED-REFIT-READOUT.md contains non-ASCII characters: {offenders}. Use `->` for "
            "arrows and `--` for dashes."
        )


class TestReadoutSections:
    """Every required section must still be present."""

    def test_all_required_section_markers_present(self) -> None:
        """Names any missing section so a silent drop is actionable.

        An absent section is indistinguishable from an unasked question, which is why the readout
        publishes all eleven even where the answer is "none" or "unchanged".
        """
        content = _read_readout()
        missing = [m for m in _REQUIRED_SECTION_MARKERS if m not in content]
        assert not missing, (
            f"GATED-REFIT-READOUT.md is missing required sections: {missing}"
        )

    def test_all_three_targets_get_a_before_after_table(self) -> None:
        """A REFUSED target gets the same table shape as a promoted one.

        The value of a refusal record is the numbers behind it -- a reader must be able to see
        what the refused candidate actually scored, not merely that it was refused.
        """
        content = _read_readout()
        for version in (
            "wp_20260824_113325",  # promoted
            "ats_20260605_220128",  # retained incumbent
            "ats_20260824_113440",  # refused candidate
            "ou_20260326_163930",  # retained incumbent
            "ou_20260824_113701",  # refused candidate
        ):
            assert version in content, (
                f"the per-target section must name {version!r}. Every target gets a before/after "
                "comparison, including the two the gate refused."
            )

    def test_the_optuna_search_is_evidenced_by_a_trial_count(self) -> None:
        """Study identity, storage path and NEW trial count -- not hyperparameter inequality.

        Differing hyperparameters are neither necessary nor sufficient evidence that a search
        ran: identical parameters can follow a genuine search that reconverged, and different
        parameters can follow a resumed at-budget study that searched nothing.
        """
        content = _read_readout()
        for marker in ("_tuning_p30s2", "NEW completed trials"):
            assert marker in content, (
                f"the per-target section must record {marker!r} as part of the three diagnostic "
                "facts that actually evidence a real search"
            )


class TestScopedHonestyVocabulary:
    """The over-claim ban follows the CLAIM, not the token. See the module docstring."""

    def test_the_two_strongest_words_are_absent_document_wide(self) -> None:
        """No number this phase produced may be called these, whatever was promoted."""
        content = _read_readout()
        present = {
            word: _whole_word_hits(content, word)
            for word in _GLOBALLY_FORBIDDEN_WORDS
            if _whole_word_hits(content, word)
        }
        assert not present, (
            f"GATED-REFIT-READOUT.md uses globally forbidden over-claim words {present}. A "
            "non-regression gate verdict and a burned-holdout readout do not supply that standard "
            "of evidence. Use 'demonstrated', 'asserted', 'measured' or 'reproduced' instead. "
            "(Matching is by word boundary, so 'provenance' and 'validation' are unaffected.)"
        )

    def test_no_over_claim_word_appears_in_the_group_verdict_section(self) -> None:
        """A feature group is kept or dropped. It is never deployed.

        A KEEP carries the group into a Stage-2 CANDIDATE feature set; whether anything reached
        production is the per-target gate's answer, in a different section.
        """
        section = _section(
            _read_readout(), _GROUP_VERDICT_SECTION_START, _TOP_LEVEL_HEADING
        )
        present = {
            word: _whole_word_hits(section, word)
            for word in _SECTION_SCOPED_FORBIDDEN_WORDS
            if _whole_word_hits(section, word)
        }
        assert not present, (
            f"the group-verdict section over-claims a Stage-1 verdict: {present}. A KEEP carries "
            "a group into the Stage-2 candidate feature set and nothing more -- and on this run "
            "one KEEP group was measured significantly NEGATIVE on two of three targets."
        )

    def test_no_over_claim_word_appears_in_the_retained_target_section(self) -> None:
        """A target the gate REFUSED is not deployed; its incumbent was retained."""
        section = _section(
            _read_readout(), _RETAINED_TARGET_SECTION_START, _ANY_SECTION_HEADING
        )
        present = {
            word: _whole_word_hits(section, word)
            for word in _SECTION_SCOPED_FORBIDDEN_WORDS
            if _whole_word_hits(section, word)
        }
        assert not present, (
            f"the retained-target section over-claims a refusal: {present}. ATS and O/U were "
            "REFUSED by the frozen gate and keep their incumbents; nothing about them was "
            "deployed by this phase."
        )

    def test_the_word_deployed_is_not_banned_document_wide(self) -> None:
        """The counterpart assertion, so the scoping cannot silently harden into a blanket ban.

        This is the guard on the guard. If a future maintainer "tightens" the vocabulary check
        into the Phase-28 blanket form, this test fails and says why: a deploy phase must be able
        to report a deployment, and a ban it cannot satisfy is a ban that gets deleted.
        """
        content = _read_readout()
        assert _whole_word_hits(content, "deployed") > 0, (
            "the readout never uses the word 'deployed'. One target genuinely swapped, and a "
            "deploy phase that cannot say so has had its vocabulary scoping hardened into the "
            "Phase-28 blanket ban -- which is unsatisfiable here by construction."
        )


class TestThreeValuedVerdictVocabulary:
    """UNDETERMINED and DROP are different findings and stay different in the record.

    Written to pass on EVERY outcome -- a zero-keep run and an all-keep run alike -- because it
    asserts the VOCABULARY rather than a particular verdict. Only a COLLAPSED vocabulary fails.
    """

    def test_all_four_frozen_verdict_words_are_declared(self) -> None:
        """The readout names the whole vocabulary, not only the arms that happened to fire."""
        content = _read_readout()
        missing = [v for v in _FROZEN_VERDICT_VOCABULARY if v not in content]
        assert not missing, (
            f"the readout never names the verdict words {missing}. 'We could not tell' "
            "(UNDETERMINED) and 'we measured it and it did not help' (DROP) are different "
            "findings, and a record that cannot express the distinction cannot report it."
        )

    def test_each_group_carries_its_ratified_verdict_word(self) -> None:
        """Every group's published verdict is one of the frozen four, and is ITS verdict."""
        content = _read_readout()
        for group, verdict in _ratified_verdicts().items():
            assert verdict in _FROZEN_VERDICT_VOCABULARY, (
                f"the ratified verdict for {group!r} is {verdict!r}, which is not in the frozen "
                f"vocabulary {_FROZEN_VERDICT_VOCABULARY}"
            )
            cells = _verdict_cells_for(content, group)
            assert cells, (
                f"the readout has no table row whose first cell names {group!r}"
            )
            assert verdict in cells, (
                f"no row for {group!r} carries its ratified verdict {verdict!r} in the VERDICT "
                f"CELL. Second cells found: {cells}. Checking the cell rather than the whole row "
                "is deliberate -- the reason string beside a verdict repeats the verdict word, so "
                "a row-level check stays green when only the verdict cell is flipped."
            )

    def test_an_undetermined_group_is_never_recorded_as_dropped(self) -> None:
        """UNDETERMINED resolves to DROP for the DEPLOY decision and stays UNDETERMINED here.

        Vacuous on this run -- nothing landed on that arm -- and that is worth stating rather
        than passing over: the machinery exists, is asserted, and simply had no occasion to fire.
        """
        content = _read_readout()
        for group, verdict in _ratified_verdicts().items():
            if verdict != VERDICT_UNDETERMINED:
                continue
            cells = _verdict_cells_for(content, group)
            assert VERDICT_DROP not in cells, (
                f"{group!r} was measured {VERDICT_UNDETERMINED} and the readout records "
                f"{VERDICT_DROP} in a verdict cell: {cells}. UNDETERMINED resolves to DROP for "
                "the DEPLOY decision and stays UNDETERMINED in the record -- 'we could not tell' "
                "and 'we measured it and it did not help' are different findings (SPEC R4/R8)."
            )
            assert VERDICT_UNDETERMINED in cells, (
                f"{group!r} was measured {VERDICT_UNDETERMINED} and no verdict cell says so: "
                f"{cells}"
            )


class TestContaminatedLabelling:
    """Every 2023-2024 monetization figure carries the FIXED contaminated vocabulary."""

    def test_the_fixed_vocabulary_appears_beside_the_hold_numbers(self) -> None:
        """The vocabulary is IMPORTED from the frozen module, never re-typed here."""
        content = _read_readout()
        missing = [phrase for phrase in CONTAMINATED_VOCAB if phrase not in content]
        assert not missing, (
            f"the monetization section is missing the fixed contaminated vocabulary {missing} "
            "(backtest.ou_monetization.CONTAMINATED_VOCAB). The 2023-2024 holdout is "
            "in-sample contaminated, and these are the only words its numbers may carry."
        )

    def test_the_structural_label_and_the_hold_split_are_named(self) -> None:
        """A reader must be able to see WHICH numbers the label applies to."""
        content = _read_readout()
        for marker in ("PROVISIONAL_CONTAMINATED", "2023-2024"):
            assert marker in content, (
                f"the monetization section must name {marker!r} so the contamination label is "
                "attached to an identifiable population rather than floating free"
            )


class TestPreRegistrationOrdering:
    """SPEC R4's ancestry claim, checked from git between two SPECIFIC commits."""

    def test_both_ancestry_markers_are_present_and_differ(self) -> None:
        """Both markers are required: ancestry of HEAD is not the claim SPEC R4 makes."""
        content = _read_readout()
        pre = _PRE_REGISTRATION_MARKER_RE.findall(content)
        measurement = _MEASUREMENT_MARKER_RE.findall(content)
        assert len(pre) == 1 and len(measurement) == 1, (
            "the readout must carry exactly one `pre_registration_commit:` and one "
            f"`measurement_commit:` marker line (found {len(pre)} and {len(measurement)})"
        )
        assert pre[0] != measurement[0], (
            "the two recorded SHAs are identical. A rule and the results it produced landing in "
            "one commit is not a pre-registration."
        )

    def test_the_rule_commit_is_a_strict_ancestor_of_the_measurement_commit(
        self,
    ) -> None:
        """The phase's only mechanical anti-rule-shopping proof."""
        _assert_rule_precedes_measurement()

    def test_the_recorded_pre_registration_marker_is_the_resolved_rule_commit(
        self,
    ) -> None:
        """The marker is not merely SOME ancestor -- it is the frozen module's own commit."""
        if _git_history_is_unavailable():
            pytest.skip(SHALLOW_SKIP_MESSAGE)

        recorded = _PRE_REGISTRATION_MARKER_RE.findall(_read_readout())
        assert recorded, "no `pre_registration_commit:` marker in the readout"
        resolved = _rule_commit()
        assert recorded[0] == resolved, (
            f"the readout records {recorded[0]} as the pre-registration commit, but the last "
            f"commit to touch {FROZEN_RULE_MODULE} is {resolved}. Either the frozen rule was "
            "modified after the measurement -- which destroys the evidence -- or the readout is "
            "citing the wrong commit."
        )

    def test_the_measurement_commit_touches_exactly_the_verdict_document(self) -> None:
        """Its message's claim to be the measurement commit is checkable, not merely asserted."""
        if _git_history_is_unavailable():
            pytest.skip(SHALLOW_SKIP_MESSAGE)

        recorded = _MEASUREMENT_MARKER_RE.findall(_read_readout())
        assert recorded, "no `measurement_commit:` marker in the readout"
        paths = [
            line
            for line in _git(
                "show", "--pretty=format:", "--name-only", recorded[0]
            ).stdout.splitlines()
            if line.strip()
        ]
        assert paths == ["config/group_gate_verdict.toml"], (
            f"the measurement commit {recorded[0]} touches {paths}, not exactly the ratified "
            "verdict document. A single-file measurement commit is what makes the claim "
            "checkable; a commit that also carries code cannot be distinguished from one that "
            "adjusted the rule alongside the result."
        )

    def test_the_shallow_checkout_guard_actually_fires(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Prove the guard fires, rather than trusting the probe was wired correctly.

        In a `--depth=1` clone `git merge-base --is-ancestor` fails for want of history, which is
        indistinguishable by exit code from a genuine ancestry violation. A false red on the
        phase's central honesty assertion would train a maintainer to ignore it, so the probe is
        consulted BEFORE any ancestry call -- and this test monkeypatches the probe to report
        shallow and asserts the documented skip is raised.

        It also fails if a future edit moves the probe AFTER the merge-base call, because then the
        monkeypatched probe would no longer short-circuit and the assertions would run.
        """
        module = __import__(__name__, fromlist=["_git_history_is_unavailable"])
        monkeypatch.setattr(module, "_git_history_is_unavailable", lambda: True)

        with pytest.raises(pytest.skip.Exception) as excinfo:
            _assert_rule_precedes_measurement()

        assert SHALLOW_SKIP_MESSAGE in str(excinfo.value), (
            "the shallow-checkout guard skipped with a different message than the documented "
            f"one. Expected to contain: {SHALLOW_SKIP_MESSAGE!r}; got: {excinfo.value!r}"
        )


class TestStateManifestAgreesWithTrackedState:
    """The readout's section 11 and tests/phase30_state.py publish the same values.

    This is what keeps the second tracked home honest. outputs/, artifacts/, data/ and
    .planning/ are ALL gitignored, so these two records are the only places several of these
    values exist at all -- and two records of the same facts that are never compared are two
    records that will eventually disagree.
    """

    @staticmethod
    def _manifest_section() -> str:
        return _section(_read_readout(), "## 11. The Phase-30 state manifest", r"^## ")

    @staticmethod
    def _expected_rows() -> list[tuple[str, str]]:
        """Return (constant name as published, rendered value) for every scalar slot."""
        import tests.phase30_state as state

        rows: list[tuple[str, str]] = [
            ("N01_PARQUET_ROWS_BEFORE", str(state.N01_PARQUET_ROWS_BEFORE)),
            ("N01_DB_ROWS_BEFORE", str(state.N01_DB_ROWS_BEFORE)),
            ("N01_DIVERGENCE_BEFORE", str(state.N01_DIVERGENCE_BEFORE)),
            ("N01_MISSING_GAME_IDS_SHA256", state.N01_MISSING_GAME_IDS_SHA256),
            ("GOLD_2025_ROWS_BEFORE_RESYNC", str(state.GOLD_2025_ROWS_BEFORE_RESYNC)),
            ("ODDS_TIMELINE_ROWS", str(state.ODDS_TIMELINE_ROWS)),
            ("ODDS_TIMELINE_PAIRS", str(state.ODDS_TIMELINE_PAIRS)),
            ("ODDS_TIMELINE_PAIR_LIST_SHA256", state.ODDS_TIMELINE_PAIR_LIST_SHA256),
            (
                "ACCEPTED_RUNG4_FINGERPRINT_SHA256",
                state.ACCEPTED_RUNG4_FINGERPRINT_SHA256,
            ),
            ("PRE_REGISTRATION_COMMIT", state.PRE_REGISTRATION_COMMIT),
            ("MEASUREMENT_COMMIT", state.MEASUREMENT_COMMIT),
            ("GROUP_VERDICT_FILE_SHA256", state.GROUP_VERDICT_FILE_SHA256),
            ("MANIFEST_SHA256_BEFORE", state.MANIFEST_SHA256_BEFORE),
            ("MANIFEST_SHA256_AFTER", state.MANIFEST_SHA256_AFTER),
            ("DATA_BASELINES_TREE_SHA256", state.DATA_BASELINES_TREE_SHA256),
            ("DEPLOYED_BLEND_VERSION", state.DEPLOYED_BLEND_VERSION),
        ]
        rows += [
            (f"ODDS_TIMELINE_SEASON_COUNTS[{season}]", str(count))
            for season, count in sorted(state.ODDS_TIMELINE_SEASON_COUNTS.items())
        ]
        return rows

    def test_every_tracked_constant_is_published_with_its_tracked_value(self) -> None:
        """Name and value must appear on the SAME LINE, so a stale value cannot hide."""
        section = self._manifest_section()
        for name, value in self._expected_rows():
            rows = [line for line in section.splitlines() if f"`{name}`" in line]
            assert rows, (
                f"section 11 does not publish {name}. The readout is the human-readable half of "
                "the manifest and must carry every slot tests/phase30_state.py carries."
            )
            assert any(value in line for line in rows), (
                f"section 11 publishes {name} with a value that is not the tracked "
                f"{value!r}. Rows found: {rows}. Fix the READOUT: tests/phase30_state.py is "
                "append-once and its slots are not edited after the plan that measured them."
            )

    def test_the_slice_digest_counts_are_published(self) -> None:
        """The 583 per-column digests do not belong in a table; their shape does."""
        import tests.phase30_state as state

        section = self._manifest_section()
        total = sum(len(v) for v in state.SLICE_DIGESTS_2021_2024.values())
        assert str(total) in section, (
            f"section 11 must publish the total 2021-2024 slice-digest count ({total})"
        )
        for matrix, digests in sorted(state.SLICE_DIGESTS_2021_2024.items()):
            assert f"`{matrix}` {len(digests)}" in section, (
                f"section 11 must publish {matrix}'s slice-digest count ({len(digests)})"
            )

    def test_the_before_and_after_manifest_digests_differ(self) -> None:
        """BECAUSE a target swapped. Had zero passed they would be equal, and that would be the record."""
        import tests.phase30_state as state

        assert state.MANIFEST_SHA256_BEFORE != state.MANIFEST_SHA256_AFTER, (
            "the recorded before/after production manifest digests are equal, which records a "
            "ZERO-SWAP phase -- but the readout reports a promotion. One of the two is wrong."
        )
        content = _read_readout()
        assert "wp_20260824_113325" in content, (
            "the manifest digests differ, so a target swapped, and the readout must name it"
        )


# The harness-reproduction class that stood here was DELETED on 2026-09-12 by owner
# instruction. It re-ran the analysis harness against live gold and asserted the
# committed point estimates still reproduced. Two defects made it undefendable:
#   1. NOT DETERMINISTIC. The situational-OU delta measured 0.0074 / 0.4424 / 0.4784
#      at 4 / 1 / 8 BLAS threads and only reproduced the committed value at 12. It
#      cannot detect drift because it drifts on its own; the quantity is a difference
#      between noisy per-season estimates, so floating-point reduction order moves it
#      further than the signal does.
#   2. Plan 33.1-08 had already concluded the same thing and designed the successor:
#      generation-gate the harness half so it SKIPS when gold moves. See that plan and
#      D29-06-02 -- 'pinning a point estimate to moving gold is the mistake this guard
#      class made once'.
# The DOC-level assertions in this file are untouched and still run unconditionally.

# ---------------------------------------------------------------------------
# THE GENERATION SEAM, AND WHY THIS MODULE DOES NOT CALL IT -- Plan 33.1-08
# Task 2, 2026-09-14.
#
# Plan 33.1-08 was written to GENERATION-GATE the harness-reproduction class
# above rather than let a gold rebuild turn it red. By the time the plan ran the
# class was already gone, deleted on 2026-09-12 for a different and better
# reason: it was not deterministic, so it could not detect drift because it
# drifted on its own. A seam that refuses to compare across gold generations
# cannot help a measurement that disagrees with itself at a fixed generation.
#
# The seam was still built -- tests/gold_generation.py -- because two OTHER
# harness reproductions did redden on the rung-3 rebuild and needed it, and
# because Phase 33's Wave 14 Elo rung will hit the same wall again. It is
# recorded here so a reader of this file can find it:
#
#   tests.gold_generation.require_gold_generation(expected_key, *, reading,
#       moved_by, recorded_in) -- skips, never fails, and names all three in the
#       skip message.
#   tests.phase33_state.GOLD_GENERATION_AFTER_WEATHER_RUNG -- the live key
#       measured after the rung.
#   tests.phase33_state.GOLD_DERIVED_READINGS -- every reading and the
#       generation it was measured against, including this module's.
#
# The document-level assertions in this file read no gold and are not gated by
# anything. They run unconditionally, and they are what actually guards the
# document.
# ---------------------------------------------------------------------------
