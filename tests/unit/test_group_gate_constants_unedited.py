"""The frozen Stage-1 rule is UNEDITED, and the re-measured verdict says what it measured.

Plan 33.2-22 Task 2 (D33.2-15). Two things are asserted here and they are different claims:

1. ``backtest/group_gate_constants.py`` -- the pre-registered rule -- is byte-unchanged. The
   re-measurement re-runs that rule on corrected gold under a new OBJECTIVE and a new INPUT;
   if the rule itself moved, the whole exercise would be rule-shopping wearing a re-measurement
   costume, which is precisely what the pre-registration exists to prevent.

2. ``config/group_gate_verdict.toml`` -- the re-measured verdict -- records the objective, the
   seasons and the gold it was measured on. A verdict with no gold digest is a verdict nobody
   can reproduce, and a verdict with no objective invites being read as a CLV lift.

WHY THE UNEDITED PROOF REUSES THE PHASE-30 WITNESS RATHER THAN PINNING A NEW DIGEST
----------------------------------------------------------------------------------
``tests.phase30_state.PRE_REGISTRATION_COMMIT`` already records the last commit to touch the
frozen module. A second digest of the same file would be a duplicate witness that can drift
from the first; and a RAW-BYTE digest would hold only on the platform it was pinned on, since
this repository has ``core.autocrlf=true`` and no ``.gitattributes`` -- the file is LF in the
git object store and CRLF in a fresh Windows working tree.

So the proof is two comparisons against that ONE witness, and they fail on DIFFERENT things:

  * ``git log -1 --format=%H -- <path>`` must EQUAL ``PRE_REGISTRATION_COMMIT``. This is the
    half that catches a COMMITTED edit: the instant an edit is committed, ``git log -1``
    returns the editing commit, which is not the recorded one. A worktree-versus-last-commit
    comparison misses that case entirely, because the file then matches its own new blob.
  * the working tree's NEWLINE-NORMALIZED bytes must equal the blob at that commit. This is
    the half that catches an UNCOMMITTED edit, and the normalization is what makes it hold on
    a CRLF checkout and an LF one alike.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

from backtest.ev_chain_constants import HOLD_SEASONS_P31
from backtest.group_gate_constants import GRID_GROUPS, GRID_TARGETS
from tests.phase30_state import PRE_REGISTRATION_COMMIT
from tests.phase33_state import P332_20_CLEAN_BUILD_GOLD_GENERATION

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The FROZEN rule, addressed as a PATH: what is asserted about it is a git fact, not a value.
FROZEN_RULE_RELPATH = "backtest/group_gate_constants.py"
FROZEN_RULE_PATH = REPO_ROOT / FROZEN_RULE_RELPATH

#: The re-measured verdict, which Task 2's script writes.
VERDICT_TOML = REPO_ROOT / "config" / "group_gate_verdict.toml"

#: The provenance keys the re-measured document must carry, in its own ``[remeasurement]``
#: table. Listed here as the CONTRACT the generator is written against, so a generator that
#: quietly stopped emitting one fails rather than producing an unreproducible verdict.
REQUIRED_REMEASUREMENT_KEYS = (
    "plan",
    "decision",
    "objective",
    "objective_anchor",
    "measured_seasons",
    "selection_seasons",
    "hp_val_seasons",
    "holdout_seasons",
    "excluded_hold_seasons",
    "gold_generation_digest",
    "frozen_rule_commit",
    "measured_at",
    "supersedes_commit",
    "closing_line_used",
    "thread_limit",
)


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run one git command in the repo root, never raising on a non-zero exit."""
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )


def _git_history_is_unavailable() -> bool:
    """True when git history cannot answer a commit question in this checkout.

    Two cases, neither of which is an ancestry violation: this is not a git checkout at all,
    or it is a shallow one whose history was truncated by ``--depth``.
    """
    if _git("rev-parse", "--git-dir").returncode != 0:
        return True
    shallow = _git("rev-parse", "--is-shallow-repository")
    return shallow.returncode != 0 or shallow.stdout.decode().strip() == "true"


_SHALLOW_SKIP = (
    "git history is unavailable here (non-git or shallow checkout), so a commit-identity "
    "question cannot be answered. Skipping is correct: an unanswerable comparison is not a "
    "passing one, and it is not a failing one either."
)


def normalized(content: bytes) -> bytes:
    """Fold every CRLF to LF.

    The ONE normalization both halves of the unedited proof use. A git blob is LF in the
    object store on every platform; a working-tree file under ``core.autocrlf=true`` is CRLF
    on Windows and LF on Linux. Comparing raw bytes would pin a result to one platform.
    """
    return content.replace(b"\r\n", b"\n")


def content_matches_witness_blob(worktree: bytes, blob: bytes) -> bool:
    """True when *worktree* equals *blob* after newline normalization on BOTH sides.

    Exposed as an importable function so the non-vacuity control below drives THIS code path
    rather than a copy of its logic.
    """
    return normalized(worktree) == normalized(blob)


def _witness_blob() -> bytes:
    """The frozen module's bytes as committed at ``PRE_REGISTRATION_COMMIT``."""
    result = _git(
        "cat-file", "blob", f"{PRE_REGISTRATION_COMMIT}:{FROZEN_RULE_RELPATH}"
    )
    assert result.returncode == 0, (
        f"git could not read {FROZEN_RULE_RELPATH} at {PRE_REGISTRATION_COMMIT}: "
        f"{result.stderr.decode(errors='replace')}"
    )
    return result.stdout


class TestTheFrozenRuleIsUnedited:
    """Two comparisons against ONE witness, failing on two different things."""

    def test_the_last_modifying_commit_is_still_the_phase30_witness(self) -> None:
        """Catches a COMMITTED edit, which the content comparison alone would miss."""
        if _git_history_is_unavailable():
            pytest.skip(_SHALLOW_SKIP)
        result = _git("log", "-1", "--format=%H", "--", FROZEN_RULE_RELPATH)
        last = result.stdout.decode().strip()
        assert last == PRE_REGISTRATION_COMMIT, (
            f"the last commit to touch {FROZEN_RULE_RELPATH} is {last}, not the recorded "
            f"pre-registration {PRE_REGISTRATION_COMMIT}. EDITING THAT FILE AFTER THE "
            "MEASUREMENT COMMIT DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. The remedy is "
            "never to re-pin this constant; it is that the rule must not be edited."
        )

    def test_the_normalized_worktree_bytes_equal_the_witness_blob(self) -> None:
        """Catches an UNCOMMITTED edit, which the commit comparison alone would miss."""
        if _git_history_is_unavailable():
            pytest.skip(_SHALLOW_SKIP)
        blob = _witness_blob()
        assert len(blob) > 0, "the witness blob is empty"
        assert content_matches_witness_blob(FROZEN_RULE_PATH.read_bytes(), blob), (
            f"the working-tree {FROZEN_RULE_RELPATH} differs from its bytes at "
            f"{PRE_REGISTRATION_COMMIT}. The re-measurement re-runs this rule on corrected "
            "gold; a rule that moved makes the whole exercise rule-shopping."
        )

    def test_the_witness_is_a_strict_ancestor_of_head(self) -> None:
        """A rule and the results it produced landing in one commit is not a pre-registration."""
        if _git_history_is_unavailable():
            pytest.skip(_SHALLOW_SKIP)
        ancestry = _git("merge-base", "--is-ancestor", PRE_REGISTRATION_COMMIT, "HEAD")
        head = _git("rev-parse", "HEAD").stdout.decode().strip()
        assert ancestry.returncode == 0, (
            f"{PRE_REGISTRATION_COMMIT} is not an ancestor of HEAD ({head})"
        )
        assert head != PRE_REGISTRATION_COMMIT, (
            "the pre-registration commit IS HEAD, so the rule and the measurement it produced "
            "are the same commit"
        )

    def test_grid_groups_and_targets_are_unchanged(self) -> None:
        """Asserted as SETS: a group joining or leaving the grid changes what is measured."""
        assert set(GRID_GROUPS) == {"injury", "snap", "situational"}, GRID_GROUPS
        assert set(GRID_TARGETS) == {"wp", "ats", "ou"}, GRID_TARGETS


class TestTheUneditedProofIsNotVacuous:
    """The control: it FAILS on an altered copy and PASSES on LF and CRLF copies alike."""

    def test_the_comparison_fails_against_a_deliberately_altered_copy(self) -> None:
        if _git_history_is_unavailable():
            pytest.skip(_SHALLOW_SKIP)
        blob = _witness_blob()
        altered = blob.replace(b"MDE_POWER: float = 0.80", b"MDE_POWER: float = 0.90")
        assert altered != blob, (
            "the planted alteration did not change anything, so this control proves nothing"
        )
        assert not content_matches_witness_blob(altered, blob)

    def test_the_comparison_passes_on_an_lf_copy_and_on_a_crlf_copy(self) -> None:
        if _git_history_is_unavailable():
            pytest.skip(_SHALLOW_SKIP)
        blob = _witness_blob()
        lf_copy = normalized(blob)
        crlf_copy = lf_copy.replace(b"\n", b"\r\n")
        assert crlf_copy != lf_copy, "the CRLF copy is identical to the LF one"
        assert content_matches_witness_blob(lf_copy, blob)
        assert content_matches_witness_blob(crlf_copy, blob), (
            "the proof must hold on a CRLF Windows checkout and on an LF one alike; a "
            "raw-byte digest would have held only on the platform it was pinned on"
        )


class TestTheRemeasuredVerdictDocument:
    """The re-measured verdict records what it measured, and keeps the seam Stage 2 reads."""

    @staticmethod
    def _document() -> dict:
        assert VERDICT_TOML.is_file(), f"missing {VERDICT_TOML}"
        with VERDICT_TOML.open("rb") as handle:
            return tomllib.load(handle)

    def test_excluded_groups_is_still_at_the_document_root(self) -> None:
        """``scripts/promote_models.py`` reads the ROOT key; the shape must not move."""
        document = self._document()
        assert "excluded_groups" in document, (
            "the root excluded_groups key is what scripts/promote_models.py reads verbatim at "
            "every re-fit; a re-measured verdict that moved it would silently stop shaping "
            "anything"
        )
        assert isinstance(document["excluded_groups"], list)
        assert set(document["excluded_groups"]) <= set(GRID_GROUPS)

    def test_it_records_the_objective_the_seasons_and_the_gold_it_measured(
        self,
    ) -> None:
        document = self._document()
        assert "remeasurement" in document, (
            "the re-measured verdict carries no [remeasurement] provenance table. A verdict "
            "with no gold digest cannot be reproduced, and one with no objective invites "
            "being read as a CLV lift."
        )
        block = document["remeasurement"]
        missing = [k for k in REQUIRED_REMEASUREMENT_KEYS if k not in block]
        assert not missing, f"the [remeasurement] table is missing {missing}"
        assert block["objective"] == "outcome_loss"
        assert block["closing_line_used"] is False
        assert block["gold_generation_digest"] == P332_20_CLEAN_BUILD_GOLD_GENERATION
        assert block["frozen_rule_commit"] == PRE_REGISTRATION_COMMIT
        assert "clv" not in str(block["objective_anchor"]).lower()

    def test_the_measured_seasons_exclude_the_spent_hold(self) -> None:
        block = self._document()["remeasurement"]
        measured = set(block["measured_seasons"])
        assert measured, "the re-measurement recorded no measured seasons"
        assert not measured & set(HOLD_SEASONS_P31), (
            f"the re-measurement scored the spent single-use hold "
            f"{sorted(measured & set(HOLD_SEASONS_P31))}; choosing feature groups by their "
            "2025 score would make 2025 a selection criterion"
        )
        assert set(block["excluded_hold_seasons"]) == set(HOLD_SEASONS_P31)
        assert measured == set(block["holdout_seasons"])

    def test_the_stage1_block_still_names_the_frozen_rule_and_its_commit(self) -> None:
        stage1 = self._document()["stage1"]
        assert stage1["frozen_rule_module"] == FROZEN_RULE_RELPATH
        assert stage1["preregistration_commit"] == PRE_REGISTRATION_COMMIT

    def test_the_recorded_thread_limit_is_the_one_the_script_pins(self) -> None:
        """A verdict whose thread count is unrecorded is a verdict nobody can reproduce."""
        from scripts.remeasure_group_verdict import REMEASUREMENT_THREAD_LIMIT

        assert self._document()["remeasurement"]["thread_limit"] == (
            REMEASUREMENT_THREAD_LIMIT
        )


class TestTheMeasurementPinsItsThreadCount:
    """The screen's XGBoost legs answer differently at different thread counts.

    MEASURED 2026-09-22: at 12 threads all three groups came back DROP; under the 8-thread cap
    the repo-root ``conftest.py`` applies to every pytest session, all three came back
    UNDETERMINED, with every ATS and O/U delta different and every WP delta identical. A
    measurement that writes a verdict cannot be left in that state, so ``remeasure`` pins the
    pool. These are STRUCTURAL checks -- they assert the pin is applied where the fits happen,
    without paying for two full screens.
    """

    def test_remeasure_applies_the_thread_pin_around_the_screen(self) -> None:
        import inspect

        from scripts.remeasure_group_verdict import remeasure

        source = inspect.getsource(remeasure)
        assert "threadpool_limits" in source, (
            "remeasure does not apply a thread pin, so its verdict depends on how many cores "
            "the machine running it happens to have"
        )
        assert "REMEASUREMENT_THREAD_LIMIT" in source, (
            "the pin must read the NAMED limit, never a literal"
        )

    def test_the_pin_is_one_so_it_is_reproducible_on_any_machine(self) -> None:
        from scripts.remeasure_group_verdict import REMEASUREMENT_THREAD_LIMIT

        assert REMEASUREMENT_THREAD_LIMIT == 1, (
            "a pin above 1 is honourable only on a machine with that many cores, which makes "
            "the measurement reproducible on some machines and not others -- the same defect "
            "class as a raw-byte digest that holds only on the platform it was pinned on"
        )

    def test_threadpoolctl_is_a_declared_dependency(self) -> None:
        """It was already installed as a scikit-learn dependency; now it is imported directly."""
        import tomllib as _tomllib

        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            declared = _tomllib.load(handle)["project"]["dependencies"]
        assert any(dep.startswith("threadpoolctl") for dep in declared), (
            "scripts/remeasure_group_verdict.py imports threadpoolctl directly, so it must be "
            "a declared dependency rather than an implicitly-relied-on transitive one"
        )
