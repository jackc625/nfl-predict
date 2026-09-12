"""No file this phase touches may claim the suite reaches a failure count of zero.

THE MILESTONE-WIDE INVARIANT THIS ENFORCES
------------------------------------------
Five tests on this checkout fail ON PURPOSE. Each encodes an owner-accepted fact from
Phase 30 or Phase 31, and ``tests/phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS`` names them
individually. The healthy form of this suite's summary line is therefore

    5 failed, <N> passed, 9 skipped, 14 xfailed

with the five named -- not a clean line. An assertion, a comment, a docstring or a readout
anywhere in this phase that says the suite reaches a failure count of zero is not an
optimistic statement; it is a FALSE statement about a record the owner accepted, and it is
the shape a disclosure gets erased in. Somebody writes the claim, somebody else later makes
the suite agree with it.

So the claim is forbidden in source, mechanically, over every file this phase adds or
modifies.

WHAT A SOURCE SCAN IS AND IS NOT
--------------------------------
A source scan is a STRUCTURAL guarantee that a SHAPE is impossible. It is never acceptance
evidence for a behaviour. This module proves that no file in the phase's diff contains one
of the forbidden phrasings; it proves nothing whatever about whether the tripwires are
still red. That is ``tests/integration/test_phase33_expected_failure_set.py``'s job, and it
does it by RUNNING them.

HOW THE FILE LIST IS RESOLVED, AND WHY THERE IS NO SECOND LIST
-------------------------------------------------------------
The list is ``git diff --name-only <PHASE_33_BASE_COMMIT>..HEAD`` plus the working tree's
own uncommitted changes, so a file written moments ago is scanned before it is committed.
Where git history is unavailable the scan SKIPS with a pinned message and a companion test
proves that skip fires.

It deliberately does NOT fall back to a committed file list. A second list of the phase's
files is a list that drifts, and a drifted list would silently SHORTEN the scan -- which is
precisely the failure the non-vacuity control below exists to catch. A control that can be
quietly narrowed is worse than one that openly steps aside.

THE FORBIDDEN PHRASES ARE NEVER SPELLED IN THIS FILE
----------------------------------------------------
Every phrase comes from ``tests/phase33_state.FORBIDDEN_SUITE_CLAIM_PHRASES``, including
the one the planted-violation control writes into its temp file. That keeps this module
honest under its own rule rather than exempt from it, and a test below asserts the
self-consistency.

FOUR CONTROLS, following ``tests/unit/test_p31_constants_isolation.py``'s form:
non-vacuity, the assertion, a PLANTED violation, and a no-false-positive control proving
the expected healthy summary form is not flagged.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]

# Only text this project actually authors. A parquet or duckdb file in the diff has no
# prose to make a claim in, and decoding it would fail for reasons unrelated to the rule.
SCANNED_SUFFIXES = (".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt", ".cfg")

# The two files that RECORD the rule. Excluding them is not an exemption from the rule --
# it is the difference between stating a forbidden phrase and making the forbidden claim.
EXCLUDED_FROM_SCAN = (
    "tests/phase33_state.py",
    "tests/unit/test_phase33_no_zero_failures_claim.py",
)

GIT_UNAVAILABLE_SKIP = (
    "the phase's touched-file list cannot be resolved: git history is unavailable here "
    "(a shallow clone, an export, or no git binary), so "
    "PHASE_33_BASE_COMMIT..HEAD names nothing. This is a control that did NOT run on "
    "this checkout, not a control that passed -- and the scan deliberately has no "
    "committed fallback list, because a second list of the phase's files is a list that "
    "drifts and a drifted list silently shortens the scan."
)


def scan_file_for_claim_phrases(path: Path) -> list[str]:
    """Every line of *path* that contains a forbidden suite-claim phrase.

    Matching is case-insensitive: the claim is just as false in a heading as in a
    docstring.

    Args:
        path: A text file to scan.

    Returns:
        Human-readable hits, each naming the line number and the phrase.
    """
    hits: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        for phrase in phase33_state.FORBIDDEN_SUITE_CLAIM_PHRASES:
            if phrase.lower() in lowered:
                hits.append(f"{path.as_posix()}:{number}: claims {phrase!r}")
    return hits


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _git_history_is_unavailable() -> bool:
    """True when ``PHASE_33_BASE_COMMIT`` cannot be resolved in this checkout."""
    try:
        result = _git(
            "cat-file", "-e", f"{phase33_state.PHASE_33_BASE_COMMIT}^{{commit}}"
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return result.returncode != 0


def _candidate_relative_paths() -> list[str]:
    """Repo-relative paths the phase has touched, committed and uncommitted alike."""
    diffed = _git(
        "diff", "--name-only", f"{phase33_state.PHASE_33_BASE_COMMIT}..HEAD"
    ).stdout.splitlines()
    # `--porcelain` lines are `XY path`; a rename is `XY old -> new` and the NEW name is
    # the file that exists, so the last field is taken.
    working: list[str] = []
    for line in _git("status", "--porcelain").stdout.splitlines():
        if not line.strip():
            continue
        working.append(line[3:].strip().split(" -> ")[-1].strip('"'))
    return [*diffed, *working]


def phase_touched_files() -> list[Path]:
    """Every scannable file the phase has touched, or skip with the pinned message.

    Returns:
        Absolute paths that exist on disk, carry a scanned suffix, and are not one of the
        two files that RECORD the rule.

    Raises:
        Skipped: when git history is unavailable.
    """
    if _git_history_is_unavailable():
        pytest.skip(GIT_UNAVAILABLE_SKIP)

    resolved: list[Path] = []
    seen: set[str] = set()
    for raw in _candidate_relative_paths():
        relative = raw.replace("\\", "/")
        if relative in seen or relative in EXCLUDED_FROM_SCAN:
            continue
        seen.add(relative)
        if not relative.endswith(SCANNED_SUFFIXES):
            continue
        # `.planning/` is gitignored in this repository and never appears in a diff; the
        # guard is here so a future config change cannot quietly widen the scan onto
        # planning prose, which is a different artifact with different rules.
        if relative.startswith(".planning/"):
            continue
        candidate = REPO_ROOT / relative
        if candidate.is_file():
            resolved.append(candidate)
    return resolved


# ---------------------------------------------------------------------------
# Control 1: non-vacuity.
# ---------------------------------------------------------------------------


def test_the_scan_visits_a_non_empty_file_list() -> None:
    """A scan over an empty file list reports nothing and proves nothing.

    Anti-vacuity. Every "no claim found" assertion below is trivially satisfiable by
    scanning nothing -- a base commit that stopped resolving, a suffix filter that
    excluded everything, a path prefix that no longer matches. This is the test that makes
    the others mean something.
    """
    files = phase_touched_files()
    assert files, (
        "the suite-claim scan visited ZERO files. Every no-claim assertion below would "
        "then pass while checking nothing. Phase 33 has at minimum modified "
        "tests/conftest.py and added tests/phase33_state.py, so an empty list is a "
        "broken resolver rather than a quiet phase."
    )
    relative = {path.relative_to(REPO_ROOT).as_posix() for path in files}
    assert "tests/conftest.py" in relative, sorted(relative)


# ---------------------------------------------------------------------------
# Control 2: the assertion itself.
# ---------------------------------------------------------------------------


def test_no_file_this_phase_touches_claims_the_suite_reaches_a_clean_line() -> None:
    """The milestone-wide invariant, over the phase's own diff.

    A claim that the suite reaches a failure count of zero is a false statement about an
    owner-accepted record, and it is how the record stops being defended: the claim is
    written first and the suite is made to agree with it later.
    """
    hits: list[str] = []
    for path in phase_touched_files():
        hits.extend(scan_file_for_claim_phrases(path))

    assert not hits, (
        "forbidden claim(s) about the test suite found in files this phase touches:\n"
        + "\n".join(f"  - {line}" for line in hits)
        + "\n\nThe healthy form of this suite's summary line names FIVE deliberate "
        "failures (tests/phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS). Report the line "
        "as measured, with the five named."
    )


def test_this_module_never_spells_a_forbidden_phrase_of_its_own() -> None:
    """Self-consistency: the rule's enforcer is not exempt, it is simply not a claimant.

    ``tests/phase33_state.py`` and this module are excluded from the scan because RECORDING
    a phrase is not MAKING the claim. That exclusion could still be abused, so this test
    shows it is not needed here: every phrase this module handles comes from the constant,
    and its own source contains none of them literally.
    """
    source = Path(__file__).resolve().read_text(encoding="utf-8").lower()
    offenders = [
        phrase
        for phrase in phase33_state.FORBIDDEN_SUITE_CLAIM_PHRASES
        if phrase.lower() in source
    ]
    assert not offenders, (
        "this module spells forbidden phrase(s) literally rather than reading them from "
        f"tests/phase33_state.FORBIDDEN_SUITE_CLAIM_PHRASES: {offenders!r}. It is "
        "excluded from its own scan, so a literal here would be invisible -- which is "
        "exactly the shape of exemption that stops being noticed."
    )


# ---------------------------------------------------------------------------
# Control 3: the PLANTED violation.
# ---------------------------------------------------------------------------


def test_the_scan_flags_a_planted_claim(tmp_path: Path) -> None:
    """Fail-closed control: a file carrying one forbidden phrase IS reported.

    The phrase is read from the constant rather than written here, so this control cannot
    drift away from the rule it proves. The temp file lives under ``tmp_path`` and never
    under ``data/`` or ``artifacts/``, so the autouse write guard has nothing to say.
    """
    phrase = phase33_state.FORBIDDEN_SUITE_CLAIM_PHRASES[0]
    planted = tmp_path / "planted_claim.md"
    planted.write_text(
        "\n".join(
            [
                "# A planted readout",
                "",
                "After this phase the suite reports " + phrase + ".",
                "",
            ]
        ),
        encoding="utf-8",
    )

    hits = scan_file_for_claim_phrases(planted)
    assert hits, (
        "the suite-claim scan did NOT flag a file containing "
        f"{phrase!r}. The real assertion above is therefore a check that has only ever "
        "been observed passing, which is indistinguishable from one that cannot fail."
    )
    assert "planted_claim.md:3" in hits[0], hits


def test_the_scan_flags_every_phrase_in_the_constant(tmp_path: Path) -> None:
    """Each phrase is individually proven live, so none is dead weight in the tuple.

    A phrase nobody has ever seen the scan match is a phrase that may be misspelled,
    mis-cased, or shadowed by another entry. Planting them one at a time is the only way
    to tell.
    """
    for index, phrase in enumerate(phase33_state.FORBIDDEN_SUITE_CLAIM_PHRASES):
        planted = tmp_path / f"planted_{index}.txt"
        # Upper-cased on purpose: the rule is case-insensitive and a claim shouted in a
        # heading is the same claim.
        planted.write_text(
            "The suite now reports " + phrase.upper() + "\n", encoding="utf-8"
        )
        assert scan_file_for_claim_phrases(planted), (
            f"FORBIDDEN_SUITE_CLAIM_PHRASES[{index}] = {phrase!r} is never matched by the "
            "scan, even when planted verbatim in upper case. It is dead weight in the "
            "tuple and gives false assurance."
        )


# ---------------------------------------------------------------------------
# Control 4: no false positive.
# ---------------------------------------------------------------------------


def test_the_expected_healthy_summary_form_is_not_flagged(tmp_path: Path) -> None:
    """Fail-open control: reporting the line AS MEASURED must stay clean.

    ``5 failed, 4045 passed, 9 skipped, 14 xfailed`` is the honest line, and every plan in
    this phase has to be able to write it. A scan that reddened on it would be one a later
    plan had to weaken, and a weakened guard asserts nothing.
    """
    healthy = tmp_path / "healthy.md"
    healthy.write_text(
        "\n".join(
            [
                "Three tiers, guard armed:",
                "",
                "    5 failed, 4045 passed, 9 skipped, 14 xfailed",
                "",
                "The five failures are EXACTLY DELIBERATE_TRIPWIRE_NODE_IDS and must stay red.",
                "The integration tier reported 5 failed and 705 passed.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert scan_file_for_claim_phrases(healthy) == []


def test_the_git_unavailable_skip_guard_actually_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed control on the resolver: the skip is issued, with the pinned message.

    A guard only ever observed NOT firing is indistinguishable from an unwired one.
    Monkeypatching the probe drives exactly the path a shallow clone would take and
    asserts the skip happens BEFORE any history-dependent git call.
    """
    monkeypatch.setattr(
        "tests.unit.test_phase33_no_zero_failures_claim._git_history_is_unavailable",
        lambda: True,
    )
    with pytest.raises(pytest.skip.Exception) as excinfo:
        phase_touched_files()
    assert str(excinfo.value) == GIT_UNAVAILABLE_SKIP


def test_the_phase_base_commit_resolves_and_is_an_ancestor_of_head() -> None:
    """The base the diff is taken against is a real commit, and it precedes HEAD.

    A base that does not resolve makes the diff empty and the scan vacuous; a base that is
    not an ancestor makes the diff a comparison between unrelated histories. Both produce
    a green test over the wrong file set.
    """
    if _git_history_is_unavailable():
        pytest.skip(GIT_UNAVAILABLE_SKIP)

    ancestry = _git(
        "merge-base", "--is-ancestor", phase33_state.PHASE_33_BASE_COMMIT, "HEAD"
    )
    assert ancestry.returncode == 0, (
        f"PHASE_33_BASE_COMMIT {phase33_state.PHASE_33_BASE_COMMIT} is not an ancestor of "
        "HEAD, so the phase's diff is a comparison between unrelated histories and the "
        "scanned file list is not the phase's."
    )
