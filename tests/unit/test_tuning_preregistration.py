"""The Phase-33.2 tuning pre-registration is anchored to a commit and locked to a hash.

WHAT THIS PROVES, AND WHAT IT DELIBERATELY CANNOT
-------------------------------------------------
Everything Plan 33.2-23 publishes about its hyperparameter search rests on one mechanical
fact: the margin, the budget, the search space, the measurement site and the not-cleared
rule were committed BEFORE any search number existed. This module asserts the checkable
half of that -- the rule commit resolves, it equals the recorded anchor, that commit
touched nothing else, the file still hashes to its recorded digest, and the commit is a
STRICT ancestor of HEAD (and therefore of the commit that records the search's output,
which lands later on the same history).

It cannot prove that nobody looked at a number first. No test can prove intent. What it
can prove is that the bytes did not move, which is what turns "the rule was frozen" from a
claim into a check.

WHY THE EXPECTED HASH COMES FROM A WITNESS MODULE
-------------------------------------------------
REVIEW-CIRCULAR, the same reason ``tests/unit/test_preregistration_ancestry.py`` records:
a file that must CONTAIN and exactly REPRODUCE its own whole-file hash is self-referential
and has no fixed point. The witness lives in ``tests/phase33_state.py``, OUTSIDE the file
it witnesses, appended in a STRICTLY LATER commit.

THE STALE-WITNESS CONTROL
-------------------------
An ancestry assertion that has only ever been observed PASSING is indistinguishable from
one wired to the wrong thing. ``TestTheStaleWitnessControl`` drives exactly the code paths
these checks use, with a witness that is deliberately wrong, and asserts each one FAILS.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

from config import tuning_preregistration
from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PREREGISTRATION_PATH = tuning_preregistration.PRE_REGISTRATION_PATH

SHALLOW_SKIP_MESSAGE = (
    "git history is unavailable (shallow clone or not a git checkout), so "
    "`git merge-base --is-ancestor` would fail for want of history rather than for want of "
    "ancestry -- and the two are indistinguishable from the exit code alone. Skipping BEFORE "
    "any ancestry call rather than reporting a false ancestry violation."
)


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
    """Return True when git history cannot support an ancestry question."""
    if _git("rev-parse", "--git-dir").returncode != 0:
        return True
    shallow = _git("rev-parse", "--is-shallow-repository")
    return shallow.returncode != 0 or shallow.stdout.strip() == "true"


def _normalized_sha256(relative_path: str) -> str:
    """sha256 of a tracked text file's NEWLINE-NORMALIZED working-tree bytes.

    This repository has ``core.autocrlf=true`` and no ``.gitattributes``, so a raw-byte
    digest would hold only on the platform that measured it.
    """
    raw = (REPO_ROOT / relative_path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def _resolved_rule_commit() -> str:
    """The LAST commit to touch the pre-registration, resolved from git."""
    return _git("log", "-1", "--format=%H", "--", PREREGISTRATION_PATH).stdout.strip()


# ---------------------------------------------------------------------------
# The parameterised checks. Each takes the witness explicitly, so the stale-witness
# control below can drive the SAME code path with a deliberately wrong value and assert it
# fails -- rather than trusting that a green run means the check is wired up.
# ---------------------------------------------------------------------------


def check_digest(expected: str) -> None:
    """Assert the pre-registration still hashes to *expected*."""
    assert _SHA256_RE.match(expected), (
        f"the recorded digest for {PREREGISTRATION_PATH} is not a 64-character sha256: "
        f"{expected!r}"
    )
    actual = _normalized_sha256(PREREGISTRATION_PATH)
    assert actual == expected, (
        f"{PREREGISTRATION_PATH} has CHANGED since the search was pre-registered.\n"
        f"  recorded (tests/phase33_state.py): {expected}\n"
        f"  recomputed from the working tree:  {actual}\n"
        "Editing the rule after the search has run does not fix a bug -- it destroys the "
        "evidence. If the rule was wrong, that is a FINDING about the rule, and the "
        "finding is what gets reported."
    )


def check_commit_matches_git(expected_commit: str) -> None:
    """Assert git resolves *expected_commit* as the pre-registration's last-modifying commit."""
    resolved = _resolved_rule_commit()
    assert _SHA1_RE.match(resolved), (
        f"git could not resolve the last commit to touch {PREREGISTRATION_PATH} "
        f"(got {resolved!r}). Without it there is no rule commit to assert ancestry from."
    )
    assert resolved == expected_commit, (
        "the resolved pre-registration commit does NOT match the recorded anchor.\n"
        f"  resolved from git: {resolved}\n"
        f"  recorded:          {expected_commit}\n"
        "Either the rule was re-committed and the witness was not re-measured, or the "
        "witness was edited."
    )


def check_commit_touches_only_the_rule(commit: str) -> None:
    """Assert *commit* changed the pre-registration and nothing else."""
    listing = _git("show", "--name-only", "--format=", commit)
    assert listing.returncode == 0, listing.stderr
    touched = sorted(path for path in listing.stdout.split() if path)
    assert touched == [PREREGISTRATION_PATH], (
        f"the pre-registration commit {commit} touches {touched}, expected exactly "
        f"[{PREREGISTRATION_PATH!r}]. Anything else in that commit means the anchor points "
        "at a commit that did more than freeze the rule."
    )


def check_strict_ancestor_of_head(commit: str) -> None:
    """Assert *commit* is an ancestor of HEAD and is not HEAD itself.

    HEAD is the right later endpoint here because every commit that RUNS the search lands
    on this same history after the rule. STRICT matters: a rule that landed together with
    the numbers it produced was not registered in advance.
    """
    head = _git("rev-parse", "HEAD").stdout.strip()
    assert _SHA1_RE.match(head), head
    assert commit != head, (
        f"the pre-registration commit IS HEAD ({commit}). Nothing has been committed after "
        "it, so it cannot yet be shown to have preceded anything."
    )
    ancestry = _git("merge-base", "--is-ancestor", commit, head)
    assert ancestry.returncode == 0, (
        f"the pre-registration commit {commit} is NOT an ancestor of HEAD ({head}). The "
        "margin, the budget and the search space must provably precede the search they "
        "govern."
    )


# ---------------------------------------------------------------------------
# The checks, run against the REAL witness
# ---------------------------------------------------------------------------


class TestThePhase332TuningPreRegistration:
    """D33.2-17 / R13: the noise guard was committed before any search number existed."""

    def test_the_witness_covers_exactly_the_file_the_rule_names(self) -> None:
        """Set equality, so a second rule file could not be silently unwitnessed."""
        assert set(phase33_state.P332_23_TUNING_PREREGISTRATION_FILE_SHA256) == {
            PREREGISTRATION_PATH
        }

    def test_the_file_exists_and_is_tracked(self) -> None:
        assert (REPO_ROOT / PREREGISTRATION_PATH).is_file()
        tracked = _git("ls-files", PREREGISTRATION_PATH).stdout.split()
        assert PREREGISTRATION_PATH in tracked, (
            f"{PREREGISTRATION_PATH} is on disk but NOT tracked by git. An untracked "
            "pre-registration cannot anchor anything: there is no commit to assert "
            "ancestry from."
        )

    def test_it_still_hashes_to_its_recorded_digest(self) -> None:
        check_digest(
            phase33_state.P332_23_TUNING_PREREGISTRATION_FILE_SHA256[
                PREREGISTRATION_PATH
            ]
        )

    def test_it_does_not_contain_its_own_recorded_hash(self) -> None:
        """REVIEW-CIRCULAR, asserted rather than merely explained."""
        expected = phase33_state.P332_23_TUNING_PREREGISTRATION_FILE_SHA256[
            PREREGISTRATION_PATH
        ]
        content = (REPO_ROOT / PREREGISTRATION_PATH).read_text(encoding="utf-8")
        assert expected not in content, (
            f"{PREREGISTRATION_PATH} CONTAINS its own recorded sha256. A file carrying its "
            "own whole-file hash has no fixed point: writing the hash changes the bytes it "
            "was computed over."
        )

    def test_the_recorded_commit_is_what_git_resolves(self) -> None:
        if _git_history_is_unavailable():
            pytest.skip(SHALLOW_SKIP_MESSAGE)
        check_commit_matches_git(phase33_state.P332_23_TUNING_PREREGISTRATION_COMMIT)

    def test_the_commit_contains_only_the_rule_file(self) -> None:
        if _git_history_is_unavailable():
            pytest.skip(SHALLOW_SKIP_MESSAGE)
        check_commit_touches_only_the_rule(
            phase33_state.P332_23_TUNING_PREREGISTRATION_COMMIT
        )

    def test_the_rule_commit_is_a_STRICT_ancestor_of_head(self) -> None:
        if _git_history_is_unavailable():
            pytest.skip(SHALLOW_SKIP_MESSAGE)
        check_strict_ancestor_of_head(
            phase33_state.P332_23_TUNING_PREREGISTRATION_COMMIT
        )

    def test_the_witness_landed_after_the_rule_it_witnesses(self) -> None:
        if _git_history_is_unavailable():
            pytest.skip(SHALLOW_SKIP_MESSAGE)
        witness_commit = _git(
            "log", "-1", "--format=%H", "--", "tests/phase33_state.py"
        ).stdout.strip()
        assert _SHA1_RE.match(witness_commit), witness_commit
        rule = phase33_state.P332_23_TUNING_PREREGISTRATION_COMMIT
        assert witness_commit != rule, (
            "the witness and the rule are the SAME commit. A file cannot record the hash "
            "of a commit it is part of."
        )
        ancestry = _git("merge-base", "--is-ancestor", rule, witness_commit)
        assert ancestry.returncode == 0, (
            f"the rule commit {rule} is NOT an ancestor of the commit that records its "
            f"anchor ({witness_commit})."
        )


class TestTheStaleWitnessControl:
    """FAIL-CLOSED controls: each check FAILS against a deliberately wrong witness.

    Without these, a green run is indistinguishable from a check comparing two constants
    that happen to agree because neither is measured from anything.
    """

    def test_a_stale_digest_fails_the_content_lock(self) -> None:
        stale = hashlib.sha256(b"a witness nobody re-measured").hexdigest()
        assert (
            stale
            != phase33_state.P332_23_TUNING_PREREGISTRATION_FILE_SHA256[
                PREREGISTRATION_PATH
            ]
        )
        with pytest.raises(AssertionError, match="has CHANGED"):
            check_digest(stale)

    def test_a_one_byte_edit_would_move_the_digest(self) -> None:
        """Proved IN MEMORY: mutating the committed rule to test the test is the
        carelessness the lock exists to catch."""
        raw = (REPO_ROOT / PREREGISTRATION_PATH).read_bytes().replace(b"\r\n", b"\n")
        mutated = bytearray(raw)
        mutated[-2] ^= 0x01
        assert (
            hashlib.sha256(bytes(mutated)).hexdigest()
            != phase33_state.P332_23_TUNING_PREREGISTRATION_FILE_SHA256[
                PREREGISTRATION_PATH
            ]
        )

    def test_a_stale_commit_fails_the_anchor_check(self) -> None:
        if _git_history_is_unavailable():
            pytest.skip(SHALLOW_SKIP_MESSAGE)
        stale = "0" * 40
        with pytest.raises(AssertionError, match="does NOT match the recorded anchor"):
            check_commit_matches_git(stale)

    def test_head_itself_fails_the_strict_ancestor_check(self) -> None:
        """The STRICT half: a rule committed AS the latest commit proves nothing yet."""
        if _git_history_is_unavailable():
            pytest.skip(SHALLOW_SKIP_MESSAGE)
        head = _git("rev-parse", "HEAD").stdout.strip()
        with pytest.raises(AssertionError, match="IS HEAD"):
            check_strict_ancestor_of_head(head)

    def test_a_multi_file_commit_fails_the_touches_only_check(self) -> None:
        """A commit that did more than freeze the rule is refused.

        HEAD of this repository is not a single-file pre-registration commit, so it is the
        natural negative case and needs no synthetic history.
        """
        if _git_history_is_unavailable():
            pytest.skip(SHALLOW_SKIP_MESSAGE)
        multi = _git(
            "log", "-1", "--format=%H", "--", "tests/phase33_state.py"
        ).stdout.strip()
        with pytest.raises(AssertionError, match="expected exactly"):
            check_commit_touches_only_the_rule(multi)


class TestThePreRegistrationIsNonVacuous:
    """A pre-registration that pre-registers nothing would satisfy every check above."""

    def test_every_target_carries_a_margin_a_budget_and_a_space(self) -> None:
        for target in ("wp", "ats", "ou"):
            assert tuning_preregistration.BEAT_RANDOM_MARGIN_BY_TARGET[target] > 0.0
            assert tuning_preregistration.TRIAL_BUDGET_BY_TARGET[target] >= 900
            assert len(tuning_preregistration.SEARCH_SPACE_BY_TARGET[target]) > 0
            assert tuning_preregistration.METRIC_BY_TARGET[target]

    def test_the_rule_and_the_comparator_are_stated(self) -> None:
        assert tuning_preregistration.NOT_CLEARED_RULE
        assert tuning_preregistration.RANDOM_COMPARATOR
        assert tuning_preregistration.OUTER_COMPARISON_RULE
        assert tuning_preregistration.PRUNER_CONFIG

    def test_the_owner_rulings_are_recorded_verbatim_with_their_date(self) -> None:
        assert tuning_preregistration.OWNER_RULING_DATE == "2026-09-23"
        assert "margin-from-fold-variation" in tuning_preregistration.OWNER_RULING_Q1
        assert "fall-back-to-defaults" in tuning_preregistration.OWNER_RULING_Q2

    def test_every_bounded_parameter_records_todays_bound_beside_the_new_one(
        self,
    ) -> None:
        """The widening is VISIBLE rather than asserted; the two that could not widen
        say so and say why."""
        for target, space in tuning_preregistration.SEARCH_SPACE_BY_TARGET.items():
            for name, spec in space.items():
                where = f"{target}.{name}"
                if spec.current is None:
                    assert spec.note, (
                        f"{where} records no prior bound and no note; a parameter that "
                        "was not searched before must say so."
                    )
                    continue
                if not spec.widened:
                    assert spec.note, (
                        f"{where} is recorded as NOT widened with no reason. An "
                        "un-widenable parameter and a forgotten one look identical "
                        "without one."
                    )

    def test_the_outer_season_is_derived_and_is_never_the_spent_hold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert tuning_preregistration.outer_comparison_season(range(2002, 2026)) == 2024
        assert tuning_preregistration.EXCLUDED_OUTER_SEASONS == (2025,)

        # The refusal is LATENT on today's corpus: LATEST_COMPLETED_SEASON is 2025 and the
        # holdout is the last TWO completed seasons, so the first holdout season cannot be
        # 2025 without the calendar moving. Driving the branch with the exclusion list
        # pointed at the season the rule DOES derive is what proves the guard is wired up
        # rather than merely written -- and what proves it REFUSES rather than quietly
        # substituting another season.
        monkeypatch.setattr(
            tuning_preregistration, "EXCLUDED_OUTER_SEASONS", (2024, 2025)
        )
        with pytest.raises(ValueError, match="EXCLUDED from the outer comparison"):
            tuning_preregistration.outer_comparison_season(range(2002, 2026))

    def test_the_two_arms_can_never_share_a_study_name(self) -> None:
        tpe = tuning_preregistration.study_name("wp", "tag", "tpe")
        random_arm = tuning_preregistration.study_name("wp", "tag", "random")
        assert tpe != random_arm
        assert tpe.endswith("_tpe")
        assert random_arm.endswith("_random")
        with pytest.raises(ValueError, match="arm must be"):
            tuning_preregistration.study_name("wp", "tag", "third")

    def test_the_search_space_digest_discriminates(self) -> None:
        wp = tuning_preregistration.search_space_digest("wp")
        ats = tuning_preregistration.search_space_digest("ats")
        assert _SHA256_RE.match(wp)
        assert wp != ats, (
            "two genuinely different search spaces hash the same, so the digest recorded "
            "in each artifact proves nothing about which space was searched"
        )

    def test_the_margin_decision_reads_the_pre_registered_bar(self) -> None:
        bar = tuning_preregistration.BEAT_RANDOM_MARGIN_BY_TARGET["ats"]
        cleared, gap = tuning_preregistration.margin_cleared("ats", 10.0, 10.0 + bar)
        assert cleared is True
        assert gap == pytest.approx(bar)
        cleared, _ = tuning_preregistration.margin_cleared(
            "ats", 10.0, 10.0 + bar - 0.01
        )
        assert cleared is False
