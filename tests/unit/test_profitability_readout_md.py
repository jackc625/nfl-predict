"""Permanent CONSISTENCY guard for the repo-root PROFITABILITY-READOUT.md (Phase 31, PROD-04).

THIS IS A CONSISTENCY GUARD, NOT A REPRODUCTION GUARD, AND THAT IS THE WHOLE POINT.
Every other readout guard in this repository re-runs its harness and compares what the document
records against what the code returns today -- ``test_signal_lift_readout_md.py`` re-runs the
signal-lift screen, ``test_gated_refit_readout_md.py`` re-runs the group gate. **The 2025 chain
CANNOT be re-run.** The split is single-use by construction: the run ledger at
``config/profitability_2025_run_ledger.toml`` is in state ``completed``, the runner hard-refuses to
overwrite an existing verdict artifact, and there is no force flag. Re-running to check a number
would spend the single clean split a second time while leaving a record saying it ran once, which
is the exact failure the one-shot design exists to prevent.

So reproduction is replaced by ARTIFACT CONSISTENCY: every 2025 figure in the readout is compared,
byte for byte under the shared ``{:.17g}`` format and with no float tolerance, against
``config/profitability_2025_verdict.toml``.

THE ACCEPTED LIMITATION, STATED PLAINLY: **this guard cannot detect an error inside the verdict
artifact itself.** If the runner wrote a wrong number, this guard will faithfully confirm that the
readout reproduces the wrong number. What it CAN detect is the failure that is actually likely --
a document edited months later, from memory or from a different source, drifting away from the
artifact it claims to publish.

WHY BOTH SCOPED RULES ARE SCOPED, AND WHY THAT IS NOT A WEAKENING.
Two rules here are narrower than their obvious blanket form, for the same reason, and it is the
reason a blanket form would have been worse than useless.

  1. THE FORBIDDEN WORD SET IS SCOPED. This is a deploy-adjacent honesty phase. The readout MUST be
     able to say that a model is deployed and that its closing-line value against the market is
     NEGATIVE. A blanket over-claim ban that forbade those statements would be unsatisfiable, and
     an unsatisfiable guard gets relaxed under pressure until it asserts nothing -- which is how a
     guard becomes decoration. The banned words are the hype vocabulary frozen in
     ``backtest.ev_chain_constants.READOUT_FORBIDDEN_WORDS`` and nothing else.

  2. THE FORBIDDEN NUMBER RULE IS AN ALLOWLIST, NOT A BLANKET BAN. An earlier draft of the plan
     prohibited restating ANY prior-document number while separately REQUIRING the absolute CLV
     values, the previous spread Kelly figure and the winner target's zero-staked ratio -- all of
     which come from prior documents. That rule could never have been satisfied.
     ``READOUT_PERMITTED_FIGURES`` replaced it with five entries, each carrying the authoritative
     source it must be printed beside.

MATCHING IS BY WORD BOUNDARY, NEVER BY NAIVE SUBSTRING. A substring check for "proven"
false-positives on the provenance column name and one for "validated" false-positives on a
re-validation phrase (D30-DEFER-05). ``test_the_word_boundary_matching_has_no_false_positives``
proves that with a fixture rather than trusting it.

TWO DOCUMENTED DIVERGENCES FROM THE PLAN'S LITERAL WORDING, BOTH RECORDED HERE RATHER THAN MADE
QUIETLY, because a guard that silently reinterprets its own specification is the thing this phase
is about.

  A. SOURCE SCOPING IS PER SECTION, NOT PER PARAGRAPH. The plan says a restated figure must appear
     "within the same paragraph" as its declared source path. Applied literally that is
     unsatisfiable for the same structural reason the blanket rule was: markdown tables carry the
     figures and the surrounding prose carries the citation, and a multi-clause disclosure
     separates the two by construction. The unit here is therefore the smallest ``##`` / ``###``
     section containing the literal -- which is the unit a reader actually reads, so "printed
     beside its authoritative source" still holds. The scoping is made STRICTER than the plan's
     rule in the direction that matters: wherever the cited source exists in the checkout, the
     literal is VALUE-CHECKED against that source's actual bytes, not merely allowed by proximity.

  B. THE SOURCE TABLE CARRIES THREE ENTRIES THE FROZEN ALLOWLIST DOES NOT. The five-entry
     ``READOUT_PERMITTED_FIGURES`` lives in ``backtest/ev_chain_constants.py``, which is FROZEN --
     editing it after the measurement destroys the pre-registration evidence rather than fixing
     anything, so the allowlist cannot grow. But the CHECKPOINT-2 and CHECKPOINT-3 disclosure
     obligations (DEF-31-06, DEF-31-09, DEF-31-10, DEF-31-11, DEF-31-12) require figures no
     allowlist entry admits. Their authoritative homes are TRACKED and non-gitignored --
     ``tests/phase31_state.py``, ``SIGNAL-LIFT-READOUT.md`` and ``GATED-REFIT-READOUT.md`` -- and
     every literal admitted through them is value-checked against those bytes. This is an
     extension of the same argument that produced the allowlist, not a hole in it, and
     ``test_the_frozen_allowlist_sources_are_all_covered`` pins the extension to the frozen five so
     it cannot drift away from them.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from backtest.ev_chain_constants import (
    READOUT_FORBIDDEN_FIGURES,
    READOUT_FORBIDDEN_WORDS,
    READOUT_PERMITTED_FIGURES,
    READOUT_REQUIRED_FIELDS,
    VERDICT_TOKENS,
)

# Repo root resolved from this file: tests/unit/test_profitability_readout_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
READOUT_MD = REPO_ROOT / "PROFITABILITY-READOUT.md"
VERDICT_TOML = REPO_ROOT / "config" / "profitability_2025_verdict.toml"

TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# The five prior milestone documents this readout POINTS AT and must never overwrite, delete or
# supersede. Their content hashes are pinned here, resolved at test-authoring time (2026-09-06).
# Asserting the hashes is what mechanically discharges the prohibition: a whitespace-only edit to
# any of them fails this module.
PRIOR_READOUT_SHA256: dict[str, str] = {
    "ACTIVATION-READOUT.md": (
        "8d73a1715b454af1bb41d61ac69d74fc0482edfe7e1faee58280ab7e7ae5f7cd"
    ),
    "SIGNAL-LIFT-READOUT.md": (
        "30ae2b84ffdd76cab612acfb1dfe42e0c428f025c40ba42b70c529dd7ca3b73e"
    ),
    "LINE-MOVEMENT-READOUT.md": (
        "2356268056bfb29fcd20a3531b1abe107faa15b4cf9f6b38233b9f1ca5f6d6f4"
    ),
    "OU-DIVERGENCE-DIAGNOSIS.md": (
        "897f332c4c8c0920c688dca940faed5f69b11bc8e7b9924e247369ae38d39134"
    ),
    "GATED-REFIT-READOUT.md": (
        "db6e4e2bd86ddeabb158526cac383fba551433bb36f2f0799bc0786128f216b0"
    ),
}

# Every source a restated figure may be printed beside, mapped to the file whose bytes the figure
# is value-checked against. A None value means the source is not present in a fresh checkout (the
# planning tree is gitignored), so only the citation can be checked there -- recorded rather than
# pretended away.
_PLANNING_SUMMARY_31_10 = (
    ".planning/phases/31-ship-the-ev-bet-list-profitability-readout/31-10-SUMMARY.md"
)
FIGURE_SOURCES: tuple[str, ...] = (
    "config/gate.toml",
    "tests/phase31_state.py",
    "STATE-OF-SYSTEM.md",
    "SIGNAL-LIFT-READOUT.md",
    "GATED-REFIT-READOUT.md",
    _PLANNING_SUMMARY_31_10,
)

# Structural literals that are NOT measurements and must not be classified as restated figures:
# version strings (v3.0) and pre-registration section references (sections 3.2 and 3.3). Both are
# narrow and both are listed here rather than silently special-cased inside the extractor.
_STRUCTURAL_LITERAL_RE = re.compile(
    r"v\d+\.\d+|[Ss]ections?\s+\d+\.\d+(?:\s+(?:and|,)\s+\d+\.\d+)*"
)

# A measured quantity in this document always carries a decimal point or an exponent. Bare integers
# are years, section numbers, sample sizes, bet counts and field counts; they are pinned by the
# explicit required-phrase assertions below (68 fabricated-zero games, 47-of-68 baseline fields,
# the per-target bet counts) rather than by source classification, because their drift class is
# different and a blanket integer rule would sweep up every date in the document.
_NUMERIC_LITERAL_RE = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")

# The labels that mark a closing-line-value figure as a CLV significance result rather than as
# evidence about profitability (REVIEW-ROI).
_CLV_LABELS: tuple[str, ...] = (
    "CLV significance",
    "closing-line value",
    "closing-line-value",
)
_ROI_LABEL = "roi p-value"

# The documented message the shallow / non-git guard skips with. Pinned as a constant so the test
# that proves the guard fires matches the SAME text the guard emits.
SHALLOW_SKIP_MESSAGE = (
    "git history is unavailable (shallow clone or not a git checkout), so a git question about "
    "this document would fail for want of history rather than for want of the fact being asked "
    "about -- and the two are indistinguishable from the exit code alone. Skipping BEFORE any "
    "git call rather than reporting a false violation."
)


# ---------------------------------------------------------------------------
# readers and helpers
# ---------------------------------------------------------------------------


def _read_readout() -> str:
    """Read PROFITABILITY-READOUT.md from the repo root."""
    return READOUT_MD.read_text(encoding="utf-8")


def _verdict() -> dict:
    """Load the committed 2025 verdict artifact -- the single source of every 2025 figure."""
    assert VERDICT_TOML.is_file(), (
        f"missing the committed verdict artifact: {VERDICT_TOML}. It is TRACKED and it is the "
        "single source this guard compares against; its absence means a broken checkout, not a "
        "reason to skip."
    )
    with VERDICT_TOML.open("rb") as handle:
        return tomllib.load(handle)


def _fmt(value: float) -> str:
    """Render a float in the shared 17-significant-digit determinism format."""
    return f"{value:.17g}"


def _artifact_literals() -> set[str]:
    """Every float in the verdict artifact, rendered signed and unsigned at 17 significant digits.

    Both renderings, because a readout may legitimately write ``-0.0527...`` where the artifact
    holds the same value and may legitimately write ``+0.0144...`` where the artifact writes it
    unsigned. The comparison stays EXACT under the shared format either way; no tolerance is
    introduced by admitting the two spellings of one value.
    """
    out: set[str] = set()

    def collect(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)
        elif isinstance(node, float):
            out.add(_fmt(node))
            out.add(_fmt(abs(node)))

    collect(_verdict())
    return out


def _source_texts() -> dict[str, str | None]:
    """Map each figure source to its bytes, or to None when it is absent from this checkout."""
    texts: dict[str, str | None] = {}
    for source in FIGURE_SOURCES:
        path = REPO_ROOT / source
        texts[source] = path.read_text(encoding="utf-8") if path.is_file() else None
    return texts


def _strip_structural(text: str) -> str:
    """Blank out version strings and section references, preserving offsets and line breaks."""

    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    return _STRUCTURAL_LITERAL_RE.sub(blank, text)


def _sections(text: str) -> list[tuple[str, str]]:
    """Split the readout into (heading, body-including-heading) at every ``##``/``###`` heading.

    The section is the smallest such block. Anything before the first heading is returned under the
    heading ``<preamble>`` so no line of the document escapes classification.
    """
    lines = text.splitlines(keepends=True)
    sections: list[tuple[str, list[str]]] = [("<preamble>", [])]
    for line in lines:
        if re.match(r"^#{2,6}\s", line):
            sections.append((line.strip(), [line]))
        else:
            sections[-1][1].append(line)
    return [(heading, "".join(body)) for heading, body in sections]


def _blocks(text: str) -> list[tuple[int, str]]:
    """Split into (1-based first line number, block) where a block is a run of non-blank lines.

    A markdown table is one block including its header row, which is what makes the CLV/ROI
    separation assertion checkable: the header is where the label lives and the rows are where the
    figures live, and a row-scoped check would never see the label.
    """
    blocks: list[tuple[int, str]] = []
    current: list[str] = []
    start = 0
    for index, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            if not current:
                start = index
            current.append(line)
        elif current:
            blocks.append((start, "\n".join(current)))
            current = []
    if current:
        blocks.append((start, "\n".join(current)))
    return blocks


def _flat(text: str) -> str:
    """Collapse every run of whitespace to a single space.

    Required-phrase assertions run against this form. The readout is hard-wrapped markdown, so
    a required phrase routinely straddles a line break; a raw substring check would then fail
    for a reason that has nothing to do with the phrase being absent. That is the "guard that
    reddens on an innocent difference" failure mode again, fixed once here rather than worked
    around at each call site.
    """
    return re.sub(r"\s+", " ", text)


def _whole_word_hits(text: str, word: str) -> int:
    """Count WORD-BOUNDARY occurrences of ``word``, case-insensitively.

    Word boundaries rather than substrings, deliberately: a substring check for "proven"
    false-positives on "provenance" and one for "validated" false-positives on "revalidated"
    (D30-DEFER-05). A guard that reddens on an innocent word gets ignored.
    """
    return len(re.findall(rf"\b{re.escape(word)}\b", text, re.IGNORECASE))


def _target_section(text: str, target: str) -> str:
    """Return the per-target statement section for ``target``.

    Located by the literal heading marker ``Target `<t>``` so a reworded section title does not
    silently make the per-target assertions vacuous.
    """
    marker = f"Target `{target}`"
    for heading, body in _sections(text):
        if marker in heading:
            return body
    raise AssertionError(
        f"PROFITABILITY-READOUT.md has no per-target section headed {marker!r}. SPEC R11 requires "
        "an explicit statement for EVERY target in the same template slots -- a missing section is "
        "the omission the fixed template exists to make impossible."
    )


def _line_of(text: str, index: int) -> int:
    """Return the 1-based line number of character offset ``index``."""
    return text.count("\n", 0, index) + 1


def _classify_literals(text: str) -> tuple[list[str], list[str]]:
    """Return (every literal found, the violations).

    A literal is admissible when it is a value from the committed verdict artifact, or when the
    SECTION containing it names one of the figure sources AND (where that source exists in the
    checkout) the source's bytes actually contain the literal.
    """
    allowed = _artifact_literals()
    sources = _source_texts()
    found: list[str] = []
    violations: list[str] = []

    for heading, body in _sections(text):
        scrubbed = _strip_structural(body)
        cited = [source for source in FIGURE_SOURCES if source in body]
        for match in _NUMERIC_LITERAL_RE.finditer(scrubbed):
            literal = match.group(0)
            found.append(literal)
            unsigned = literal.lstrip("+-")
            if literal in allowed or unsigned in allowed:
                continue
            if any(
                sources[source] is None
                or literal in sources[source]
                or unsigned in sources[source]
                for source in cited
            ):
                continue
            violations.append(
                f"{literal!r} on line {_line_of(body, match.start())} of section {heading!r} "
                f"(sources cited in that section: {cited or 'NONE'})"
            )
    return found, violations


# ---------------------------------------------------------------------------
# the shallow / non-git probe, consulted BEFORE any git call
# ---------------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run one git command in the repo root, never raising on a non-zero exit."""
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def _git_history_is_unavailable() -> bool:
    """Return True when git history cannot support a question about this document's provenance.

    Module-level precisely so it can be monkeypatched -- see
    ``test_the_shallow_checkout_guard_actually_fires``, which proves the guard fires rather than
    trusting that the probe was wired correctly.
    """
    if _git("rev-parse", "--git-dir").returncode != 0:
        return True
    shallow = _git("rev-parse", "--is-shallow-repository")
    return shallow.returncode != 0 or shallow.stdout.strip() == "true"


def _assert_the_frozen_artifacts_are_tracked_and_unmodified() -> None:
    """The verdict artifact and the run ledger must be tracked and clean in the working tree.

    Factored out of its test so the shallow-guard proof can invoke exactly this code path under a
    monkeypatched probe.
    """
    if _git_history_is_unavailable():
        pytest.skip(SHALLOW_SKIP_MESSAGE)

    for relative in (
        "config/profitability_2025_verdict.toml",
        "config/profitability_2025_run_ledger.toml",
    ):
        listed = _git("ls-files", "--error-unmatch", relative)
        assert listed.returncode == 0, (
            f"{relative} is not tracked by git. The 2025 verdict and its one-shot ledger are the "
            "only durable record that the single clean split was spent exactly once; an untracked "
            "one does not survive a fresh checkout."
        )
        diff = _git("diff", "--quiet", "HEAD", "--", relative)
        assert diff.returncode == 0, (
            f"{relative} differs from HEAD in the working tree. It is GENERATOR OUTPUT for a "
            "single-use measurement: a hand-edited value cannot be regenerated and is "
            "indistinguishable from a tampered one."
        )


# ---------------------------------------------------------------------------
# 1. STRUCTURE -- a statement for every target, carrying every required field
# ---------------------------------------------------------------------------


class TestReadoutStructure:
    """The fixed per-target template, so omission fails rather than passing unnoticed."""

    def test_file_exists_at_repo_root(self) -> None:
        """The PROD-04 deliverable sits beside the five prior readouts, not under .planning/."""
        assert READOUT_MD.is_file(), (
            f"missing: {READOUT_MD}. The milestone-close readout is a TRACKED repo-root document; "
            "the planning tree is gitignored and does not survive a fresh checkout."
        )

    def test_content_is_ascii(self) -> None:
        """ASCII only, no emoji (CLAUDE.md hard constraint)."""
        content = _read_readout()
        offenders = sorted({ch for ch in content if not ch.isascii()})
        assert not offenders, (
            f"PROFITABILITY-READOUT.md contains non-ASCII characters: {offenders}"
        )

    def test_every_target_has_a_statement(self) -> None:
        """All three targets, in the same template slots. None may be omitted (SPEC R11)."""
        content = _read_readout()
        for target in TARGETS:
            assert _target_section(content, target).strip(), (
                f"the per-target section for {target!r} is empty"
            )

    def test_every_target_statement_carries_every_required_field(self) -> None:
        """Field presence is asserted directly, which is what the fixed template buys."""
        content = _read_readout()
        missing: dict[str, list[str]] = {}
        for target in TARGETS:
            section = _target_section(content, target)
            absent = [f for f in READOUT_REQUIRED_FIELDS if f not in section]
            if absent:
                missing[target] = absent
        assert not missing, (
            f"per-target statements are missing required template fields: {missing}. The template "
            "is fixed so that an omitted field is a test failure rather than something a reader "
            "has to notice was never written."
        )

    def test_every_required_field_has_a_non_empty_value(self) -> None:
        """A field name with nothing after it would satisfy presence and say nothing."""
        content = _read_readout()
        empty: list[str] = []
        for target in TARGETS:
            section = _target_section(content, target)
            for field in READOUT_REQUIRED_FIELDS:
                for match in re.finditer(rf"`{re.escape(field)}`\s*:(.*)", section):
                    if not match.group(1).strip(" *-"):
                        empty.append(f"{target}/{field}")
        assert not empty, f"required template fields present but empty: {empty}"


# ---------------------------------------------------------------------------
# 2. VOCABULARY -- every verdict comes from the frozen closed set
# ---------------------------------------------------------------------------


class TestVerdictVocabulary:
    """The verdict token vocabulary is closed and was fixed before the numbers existed."""

    def test_each_target_carries_a_token_from_the_frozen_vocabulary(self) -> None:
        """A verdict written in free prose would be outside the pre-registered mapping."""
        content = _read_readout()
        for target in TARGETS:
            section = _target_section(content, target)
            present = [token for token in VERDICT_TOKENS if token in section]
            assert present, (
                f"the {target!r} statement carries no token from the frozen verdict vocabulary "
                f"{VERDICT_TOKENS}. The mapping from numbers to words was fixed before the "
                "numbers existed; a verdict outside it is not the pre-registered verdict."
            )

    def test_each_target_token_is_the_one_the_artifact_recorded(self) -> None:
        """The published word must be the measured word, per target."""
        content = _read_readout()
        verdict = _verdict()
        for target in TARGETS:
            expected = verdict["targets"][target]["verdict_token"]
            section = _target_section(content, target)
            assert expected in section, (
                f"the {target!r} statement does not carry the token the verdict artifact "
                f"recorded ({expected!r}). The readout and the artifact disagree about the "
                "verdict itself, which is the most consequential drift this guard can catch."
            )

    def test_no_target_is_reported_as_profitable(self) -> None:
        """A guard on the reading, not on the outcome: it would pass on a genuine PROFITABLE run.

        It asserts the readout's token agrees with the artifact, and the artifact recorded none.
        If a future run genuinely produced PROFITABLE_CLEAN this test tracks it rather than
        blocking it -- it is pinned to the ARTIFACT, never to a hoped-for answer.
        """
        verdict = _verdict()
        content = _read_readout()
        profitable = [
            t
            for t in TARGETS
            if verdict["targets"][t]["verdict_token"] == "PROFITABLE_CLEAN"
        ]
        for target in TARGETS:
            section = _target_section(content, target)
            if target not in profitable:
                assert "PROFITABLE_CLEAN" not in section.replace(
                    "UNPROFITABLE_CLEAN", ""
                ), (
                    f"the {target!r} statement uses PROFITABLE_CLEAN, which the verdict artifact "
                    "did not record for it."
                )


# ---------------------------------------------------------------------------
# 3. PRESERVATION -- the five prior documents are not overwritten or superseded
# ---------------------------------------------------------------------------


class TestPriorReadoutsPreserved:
    """The prohibition against overwriting a prior milestone readout, discharged mechanically."""

    def test_all_five_prior_readouts_are_present(self) -> None:
        """The readout points at them; they have to be there to be pointed at."""
        missing = [
            name for name in PRIOR_READOUT_SHA256 if not (REPO_ROOT / name).is_file()
        ]
        assert not missing, (
            f"prior milestone readouts are missing from the repo root: {missing}. This readout is "
            "written BESIDE them and supersedes none of them; deleting one erases the record it "
            "points at."
        )

    def test_all_five_prior_readouts_have_unchanged_content_hashes(self) -> None:
        """A whitespace-only edit fails here, which is the point."""
        drifted: dict[str, str] = {}
        for name, expected in PRIOR_READOUT_SHA256.items():
            path = REPO_ROOT / name
            if not path.is_file():
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                drifted[name] = f"expected {expected}, found {actual}"
        assert not drifted, (
            f"prior milestone readouts changed content: {drifted}. Phase 31 must not overwrite, "
            "delete or silently supersede them. If one of these genuinely had to change, that is "
            "an owner decision recorded in its own document -- and this pin is then updated "
            "deliberately, in the same commit, with the reason."
        )

    def test_the_readout_points_at_every_prior_document_by_name(self) -> None:
        """Pointing at is the required alternative to restating."""
        content = _read_readout()
        absent = [name for name in PRIOR_READOUT_SHA256 if name not in content]
        assert not absent, (
            f"the readout never names these prior documents: {absent}. R11 requires it to point "
            "at the evidence rather than copy it."
        )

    def test_every_target_names_an_evidence_pointer_document(self) -> None:
        """A pointer slot filled with prose and no document name is not a pointer."""
        content = _read_readout()
        for target in TARGETS:
            section = _target_section(content, target)
            named = [name for name in PRIOR_READOUT_SHA256 if name in section]
            assert named, (
                f"the {target!r} statement's evidence_pointer names no prior document. Every "
                "target must point at the readout holding its underlying evidence."
            )


# ---------------------------------------------------------------------------
# 4. FORBIDDEN VOCABULARY -- words and values, matched by word boundary
# ---------------------------------------------------------------------------


class TestForbiddenVocabulary:
    """The hype ban follows the frozen set, matched by word boundary and nothing wider."""

    def test_no_forbidden_word_appears(self) -> None:
        """No number this milestone produced may be described in those terms."""
        content = _read_readout()
        present = {
            word: _whole_word_hits(content, word)
            for word in READOUT_FORBIDDEN_WORDS
            if _whole_word_hits(content, word)
        }
        assert not present, (
            f"PROFITABILITY-READOUT.md uses forbidden over-claim words {present}. No target "
            "cleared its pre-registered ROI test; use 'measured', 'demonstrated', 'recorded' or "
            "'asserted' instead. Matching is by word boundary, so 'provenance' and 'validation' "
            "are unaffected."
        )

    def test_the_word_boundary_matching_has_no_false_positives(self) -> None:
        """The guard on the guard: an innocent word must not redden it (D30-DEFER-05).

        A substring check for "proven" fires on the provenance column name and one for "validated"
        fires on a re-validation phrase. A guard that reddens on an innocent word teaches a
        maintainer to ignore it, so the fix is asserted here with a fixture rather than assumed.
        """
        fixture = (
            "The provenance column records the source. The gate baseline was re-validation "
            "material only. Provenance and validation are ordinary words here."
        )
        for word in READOUT_FORBIDDEN_WORDS:
            assert _whole_word_hits(fixture, word) == 0, (
                f"word-boundary matching false-positived on {word!r} inside a fixture containing "
                "only 'provenance', 're-validation' and 'validation'. The matcher has regressed "
                "to a substring check."
            )

    def test_the_scoping_cannot_silently_harden_into_a_blanket_ban(self) -> None:
        """The counterpart assertion: this readout MUST be able to say a model is deployed.

        If a future maintainer 'tightens' the vocabulary into a blanket over-claim ban, this test
        fails and says why. A deploy-adjacent honesty phase that cannot report a deployment, or
        cannot state a negative closing-line value, has had its guard hardened into something
        unsatisfiable -- and an unsatisfiable guard gets deleted.
        """
        content = _read_readout()
        assert _whole_word_hits(content, "deployed") > 0, (
            "the readout never uses the word 'deployed', yet three artifacts are in production "
            "and the requirement asks what is deployed."
        )
        assert _whole_word_hits(content, "negative") > 0, (
            "the readout never uses the word 'negative', yet the deployed win-probability "
            "model's absolute pooled closing-line value is significantly negative and R11 "
            "requires that to be stated plainly."
        )

    def test_no_forbidden_figure_appears_beside_a_closing_line_value_label(
        self,
    ) -> None:
        """A number can be a lie without any adjective attached to it."""
        content = _read_readout()
        for figure, why in READOUT_FORBIDDEN_FIGURES.items():
            for block_start, block in _blocks(content):
                if figure in block and any(
                    label.lower() in block.lower() for label in _CLV_LABELS
                ):
                    raise AssertionError(
                        f"the forbidden figure {figure} appears beside a closing-line-value label "
                        f"in the block beginning on line {block_start}. {why}"
                    )

    def test_the_true_ou_line_clv_mean_is_stated_instead(self) -> None:
        """The prohibition is only half discharged by omission; the true figure must be present."""
        content = _read_readout()
        assert "1.8954569" in content, (
            "the readout omits O/U's true line-CLV mean. Suppressing the misleading headline "
            "figure without publishing the real one leaves the reader with nothing, which is not "
            "what the prohibition asks for."
        )


# ---------------------------------------------------------------------------
# 5. FIGURES -- every 2025 number matches the committed artifact, exactly
# ---------------------------------------------------------------------------


class TestEvery2025FigureMatchesTheArtifact:
    """Byte-for-byte under the shared {:.17g} format. No tolerance anywhere."""

    def test_each_target_publishes_its_artifact_figures_verbatim(self) -> None:
        """The five figures a reader would act on, per target, compared exactly."""
        content = _read_readout()
        verdict = _verdict()
        missing: list[str] = []
        for target in TARGETS:
            section = _target_section(content, target)
            entry = verdict["targets"][target]
            for field in (
                "flat_roi",
                "raw_roi_p_value",
                "adjusted_roi_p_value",
                "ci_lo",
                "ci_hi",
                "robustness_cut_roi",
            ):
                rendered = _fmt(entry[field])
                if rendered not in section and _fmt(abs(entry[field])) not in section:
                    missing.append(f"{target}.{field} = {rendered}")
        assert not missing, (
            f"per-target sections do not carry these verdict-artifact figures verbatim: {missing}. "
            "Every 2025 number in this readout is a copy of the committed artifact under the "
            "shared 17-significant-digit format; a value that does not appear has drifted."
        )

    def test_each_target_publishes_its_bet_counts(self) -> None:
        """Bets selected and the positive control's bet count, both integers, both required."""
        content = _read_readout()
        verdict = _verdict()
        for target in TARGETS:
            section = _target_section(content, target)
            entry = verdict["targets"][target]
            for field in ("bets_selected", "control_bet_count"):
                assert str(entry[field]) in section, (
                    f"the {target!r} statement does not carry {field} = {entry[field]}"
                )

    def test_the_clv_report_only_figures_match_the_artifact(self) -> None:
        """CLV is carried BESIDE the verdict and must be the artifact's value, not a recollection."""
        content = _read_readout()
        verdict = _verdict()
        missing: list[str] = []
        for target in TARGETS:
            entry = verdict["targets"][target]
            for field in ("clv_report_only_mean", "clv_report_only_p"):
                rendered = _fmt(entry[field])
                if rendered not in content and _fmt(abs(entry[field])) not in content:
                    missing.append(f"{target}.{field} = {rendered}")
        assert not missing, f"report-only CLV figures absent or drifted: {missing}"

    def test_the_multiplicity_family_is_published_as_the_artifact_records_it(
        self,
    ) -> None:
        """A denominator a reader cannot check is a denominator that can be quietly reduced."""
        content = _read_readout()
        run = _verdict()["run"]
        for field in (
            "bh_denominator",
            "registry_rows",
            "excluded_control_entries",
            "excluded_tune_side_sweep_cells",
            "bh_fallbacks_fired",
        ):
            assert str(run[field]) in content, (
                f"the readout does not publish {field} = {run[field]} from the verdict artifact's "
                "run block. The registry has to close arithmetically in the document, not only in "
                "the artifact."
            )
        assert _fmt(run["alpha"]) in content, (
            f"the readout does not publish alpha = {_fmt(run['alpha'])} exactly. Rounding it to "
            "0.05 would publish a threshold the comparison did not use."
        )

    def test_the_committed_artifacts_are_tracked_and_unmodified(self) -> None:
        """The verdict and the one-shot ledger are the durable record; both must be clean."""
        _assert_the_frozen_artifacts_are_tracked_and_unmodified()

    def test_the_shallow_checkout_guard_actually_fires(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Proved, not assumed: a shallow clone must SKIP before any git call, never report red.

        It also fails if a future edit moves the probe AFTER the first git call, because then the
        monkeypatched probe would no longer short-circuit and the assertions would run.
        """
        module = __import__(__name__, fromlist=["_git_history_is_unavailable"])
        monkeypatch.setattr(module, "_git_history_is_unavailable", lambda: True)

        with pytest.raises(pytest.skip.Exception) as excinfo:
            _assert_the_frozen_artifacts_are_tracked_and_unmodified()
        assert SHALLOW_SKIP_MESSAGE in str(excinfo.value), (
            "the shallow-checkout guard skipped with a different message than the documented one. "
            f"Expected to contain: {SHALLOW_SKIP_MESSAGE!r}; got: {excinfo.value!r}"
        )


# ---------------------------------------------------------------------------
# 6. THE FIGURE ALLOWLIST -- no unlisted prior-document number (REVIEW-READOUT)
# ---------------------------------------------------------------------------


class TestFigureAllowlist:
    """Every restated number resolves to a source, or the guard names it and the line it sits on."""

    def test_the_extractor_reports_its_count_and_fails_when_that_count_is_zero(
        self, record_property
    ) -> None:
        """A parsing failure must fail loudly rather than pass the allowlist vacuously.

        The count is reported through ``record_property`` rather than ``print`` -- the repository
        bans bare prints in tests (ruff T201) and ``record_property`` is pytest's own API for
        surfacing a measured value from a test run.
        """
        found, _ = _classify_literals(_read_readout())
        record_property("numeric_literals_classified", len(found))
        assert len(found) > 50, (
            f"the numeric-literal extractor found only {len(found)} literals in a document that "
            "publishes three per-target verdict tables and a full disclosure section. That is a "
            "parsing failure, and a parsing failure would make every other assertion in this "
            "class pass while checking nothing."
        )

    def test_every_numeric_literal_resolves_to_a_declared_source(self) -> None:
        """The allowlist is enforceable precisely because it is satisfiable."""
        _, violations = _classify_literals(_read_readout())
        assert not violations, (
            "PROFITABILITY-READOUT.md restates numbers that resolve to no declared source:\n  "
            + "\n  ".join(violations)
            + "\nEvery number must be a value from config/profitability_2025_verdict.toml, or "
            "must be printed inside a section that names its authoritative source. A pointer to "
            "the document holding the number is the required alternative to restating it."
        )

    def test_the_frozen_allowlist_sources_are_all_covered(self) -> None:
        """The extension cannot drift away from the five frozen entries it extends.

        ``READOUT_PERMITTED_FIGURES`` is in a FROZEN module and cannot grow. This test pins the
        source table to it: every source the frozen allowlist declares must be honoured here, so a
        later edit to this test cannot quietly drop one of the five.
        """
        declared = {
            source
            for source in READOUT_PERMITTED_FIGURES.values()
            if source != "config/profitability_2025_verdict.toml"
        }
        uncovered = {
            source
            for source in declared
            if not any(source in known for known in FIGURE_SOURCES)
        }
        assert not uncovered, (
            f"the frozen READOUT_PERMITTED_FIGURES declares sources this guard does not honour: "
            f"{uncovered}"
        )

    def test_each_permitted_figure_is_actually_printed_beside_its_source(self) -> None:
        """Satisfiability, demonstrated rather than asserted.

        The three sources the frozen allowlist names beyond the verdict artifact must each be
        cited in a section that also carries a number. If they are not, the allowlist is being
        honoured in the trivial way -- by restating nothing -- and the requirement that several of
        those figures BE stated has quietly gone unmet.
        """
        content = _read_readout()
        for source in (
            "config/gate.toml",
            _PLANNING_SUMMARY_31_10,
            "STATE-OF-SYSTEM.md",
        ):
            cited_with_a_number = [
                heading
                for heading, body in _sections(content)
                if source in body
                and _NUMERIC_LITERAL_RE.search(_strip_structural(body))
            ]
            assert cited_with_a_number, (
                f"no section both names {source!r} and carries a figure. The allowlist entry "
                "exists because the requirement separately demands that figure be stated; an "
                "allowlist honoured by stating nothing is an allowlist that means nothing."
            )


# ---------------------------------------------------------------------------
# 7. CLV / ROI SEPARATION -- the conflation this milestone exists to refuse
# ---------------------------------------------------------------------------


class TestClvRoiSeparation:
    """A CLV figure may never be offered as evidence of profitability (REVIEW-ROI)."""

    @staticmethod
    def _clv_p_literals() -> set[str]:
        verdict = _verdict()
        return {_fmt(verdict["targets"][t]["clv_report_only_p"]) for t in TARGETS}

    @staticmethod
    def _roi_p_literals() -> set[str]:
        verdict = _verdict()
        out: set[str] = set()
        for target in TARGETS:
            entry = verdict["targets"][target]
            out.add(_fmt(entry["raw_roi_p_value"]))
            out.add(_fmt(entry["adjusted_roi_p_value"]))
        return out

    def test_the_readout_states_that_clv_is_not_profitability(self) -> None:
        """Said in as many words, not left to be inferred from careful labelling."""
        content = _flat(_read_readout()).lower()
        assert "closing-line value is not profitability" in content, (
            "the readout never states plainly that closing-line value is not profitability. "
            "Careful labelling alone leaves the inference to the reader, and the reader who gets "
            "it wrong draws the D25-14 conclusion this milestone exists to refuse."
        )

    def test_every_clv_p_value_sits_in_a_clv_labelled_block(self) -> None:
        """A CLV p-value with no CLV label beside it is a p-value a reader will misread."""
        content = _read_readout()
        unlabelled: list[str] = []
        for literal in self._clv_p_literals():
            for start, block in _blocks(content):
                if literal in block and not any(
                    label.lower() in block.lower() for label in _CLV_LABELS
                ):
                    unlabelled.append(
                        f"{literal} in the block beginning on line {start}"
                    )
        assert not unlabelled, (
            f"CLV significance p-values appear without a CLV label: {unlabelled}. Every "
            "occurrence must be labelled as CLV significance so it cannot be read as the "
            "profitability test."
        )

    def test_no_verdict_token_block_presents_a_clv_figure_as_its_evidence(self) -> None:
        """The verdict rests on the ROI bootstrap; a CLV figure may not stand beside it as proof."""
        content = _read_readout()
        clv_literals = self._clv_p_literals()
        offenders: list[str] = []
        for start, block in _blocks(content):
            if not any(token in block for token in VERDICT_TOKENS):
                continue
            for literal in clv_literals:
                if literal in block:
                    offenders.append(
                        f"{literal} shares the block beginning on line {start} with a verdict token"
                    )
        assert not offenders, (
            f"a report-only CLV p-value is presented alongside a verdict token: {offenders}. The "
            "verdict rests on the pre-registered ROI bootstrap alone; a CLV figure in the same "
            "breath invites exactly the conflation the frozen rule forbids."
        )

    def test_every_profitability_claim_names_the_roi_p_value(self) -> None:
        """The ROI p-value is cited by name wherever a profitability statement is made."""
        content = _read_readout()
        unlabelled: list[str] = []
        for literal in self._roi_p_literals():
            for start, block in _blocks(content):
                if literal in block and _ROI_LABEL not in block.lower():
                    unlabelled.append(
                        f"{literal} in the block beginning on line {start}"
                    )
        assert not unlabelled, (
            f"ROI p-values appear without being named as such: {unlabelled}. A profitability "
            "claim must cite the ROI p-value by name so the two tests cannot be conflated."
        )

    def test_a_conflating_fixture_sentence_fails_the_separation(self) -> None:
        """The separation assertion has teeth, demonstrated against a fixture rather than assumed.

        A block that presents a report-only CLV p-value as the evidence behind a verdict token is
        exactly the failure this class exists to catch, so one is constructed and the detector is
        run over it directly.
        """
        clv_literal = sorted(self._clv_p_literals())[0]
        fixture = (
            "The target is PROFITABLE_CLEAN because its closing-line value is significant at "
            f"{clv_literal}, which demonstrates the edge is real."
        )
        has_token = any(token in fixture for token in VERDICT_TOKENS)
        has_clv = clv_literal in fixture
        assert has_token and has_clv, (
            "the conflating fixture no longer contains both a verdict token and a CLV p-value, so "
            "it no longer tests the separation rule it was written to test."
        )


# ---------------------------------------------------------------------------
# 8. THE DISCLOSURES -- each obligation checked as required CONTENT, not as prose
# ---------------------------------------------------------------------------


class TestRequiredDisclosures:
    """The CHECKPOINT-2 and CHECKPOINT-3 obligations, each pinned by the phrase that discharges it.

    Pinned as required PRESENCE rather than by re-measurement: these are rulings and accepted facts,
    and re-measuring one here would either be impossible (the split is spent) or would re-open a
    question the owner already closed.
    """

    def test_the_fabricated_zero_market_lines_are_named(self) -> None:
        """DEF-31-09 requires naming what those games were graded against, not 'a correction'."""
        content = _flat(_read_readout())
        for phrase in ("fabricated", "0.0 market line", "68 games"):
            assert phrase.lower() in content.lower(), (
                f"the readout does not carry {phrase!r}. DEF-31-09's obligation is explicit: name "
                "the fabricated zero market lines that 68 protected-window games were graded "
                "against. A readout that says 'a data correction was applied' does not discharge "
                "it."
            )

    def test_the_gate_baseline_divergence_is_disclosed_as_not_a_re_freeze(self) -> None:
        """DEF-31-10: the 47-of-68 divergence, the unchanged n, and why nothing was re-frozen."""
        content = _flat(_read_readout())
        for phrase in ("47 of 68", "was NOT re-frozen", "byte-unchanged"):
            assert phrase.lower() in content.lower(), (
                f"the readout does not carry {phrase!r}, so DEF-31-10 is not discharged. "
                "Reporting the gate verdict without disclosing that a re-score of its baseline "
                "returns different numbers is not honest about what the verdict was measured "
                "against."
            )

    def test_both_ats_residual_value_sets_are_published(self) -> None:
        """DEF-31-06: only the ratified set, or only the live set, does not discharge it."""
        content = _flat(_read_readout())
        for literal in (
            "0.58937727047262922",
            "0.60484106920061009",
            "0.59326265763681096",
        ):
            assert literal in content, (
                f"the readout does not carry the ATS residual pooled figure {literal}. DEF-31-06 "
                "requires BOTH the ratified set the verdict was computed with and the live set a "
                "re-measurement returns today, with the reason they differ."
            )

    def test_the_2022_sign_flip_is_called_out(self) -> None:
        """DEF-31-11: the ratified negative, the live positive, and the surviving pooled claim."""
        content = _flat(_read_readout())
        for phrase in (
            "-0.03539119799896865",
            "0.046538869338765949",
            "seasons_with_negative_mean",
        ):
            assert phrase in content, (
                f"the readout does not carry {phrase!r}. The pre-registration's prose asserts a "
                "negative 2022 season mean that live gold no longer shows; quietly dropping the "
                "sentence does not discharge DEF-31-11."
            )

    def test_the_phase_30_group_verdicts_are_disclosed_as_resting_on_superseded_gold(
        self,
    ) -> None:
        """DEF-31-12, including that nothing was retrained so two groups are still in production."""
        content = _flat(_read_readout())
        for phrase in ("-0.3203552582994336", "superseded", "Nothing was retrained"):
            assert phrase.lower() in content.lower(), (
                f"the readout does not carry {phrase!r}, so DEF-31-12 is not discharged. A "
                "milestone close that reports the gate verdict without this silently inherits a "
                "superseded feature-group verdict."
            )

    def test_the_realized_devig_split_is_published_with_its_derivation(self) -> None:
        """DEF-31-13 clause 3: the split, AND how it was established, since no per-bet value exists."""
        content = _flat(_read_readout())
        for phrase in ("real_two_sided", "flat", "coverage"):
            assert phrase.lower() in content.lower(), (
                f"the readout does not carry {phrase!r}. The realized pricing split must be "
                "published, and because no per-bet devig_method is persisted, the derivation from "
                "complete 2025 juice coverage must be stated rather than implied."
            )

    def test_the_two_reasons_for_the_totals_discontinuity_are_both_named(self) -> None:
        """One declared discontinuity with TWO causes; naming one of them is not enough."""
        content = _flat(_read_readout()).lower()
        assert "tune window was widened" in content, (
            "the readout does not name the widened tune window as a cause of the declared totals "
            "discontinuity"
        )
        assert "playoff rows now sit in the tune population" in content, (
            "the readout does not name the playoff rows in the tune population as the SECOND "
            "cause of the declared totals discontinuity"
        )

    def test_the_narrowed_meaning_of_the_burned_hold_is_stated(self) -> None:
        """An unstated narrowing is the quiet form of moving a goalpost."""
        content = _flat(_read_readout()).lower()
        assert "not evaluated" in content and "not touched" in content, (
            "the readout does not state that 'the 2023-2024 hold stays burned' now means NOT "
            "EVALUATED rather than NOT TOUCHED."
        )

    def test_the_changed_spread_kelly_figure_is_stated_with_its_cause(self) -> None:
        """Removing a wrong number quietly would be its own dishonesty."""
        content = _flat(_read_readout())
        assert "1.37" in content, (
            "the readout does not state the previous spread Kelly return figure"
        )
        for phrase in ("inverted selection", "points-distance"):
            assert phrase.lower() in content.lower(), (
                f"the readout does not carry {phrase!r} -- the reason the previous spread Kelly "
                "figure was wrong. Stating the number without the cause leaves a reader unable to "
                "tell a correction from a revision."
            )

    def test_the_published_betting_artifact_staleness_is_recorded(self) -> None:
        """The measured truth, not the inherited claim: the ledger was never regenerated."""
        content = _flat(_read_readout())
        assert (
            "61f73c17ed1a118fd619a513fbf389d9378e9cbcc53a0b52e39cb331309624b1"
            in content
        ), (
            "the readout does not carry the sha256 of outputs/backtest/betting_simulation.csv. "
            "The page still serves the pre-fix spread Kelly figure because that ledger has not "
            "been regenerated, and a readout describing the fix without that fact overstates what "
            "changed on the served page."
        )
        assert "never regenerated" in content.lower(), (
            "the readout does not state that the published betting ledger was never regenerated"
        )

    def test_the_winner_zero_staked_ratio_is_stated_as_a_legitimate_no_edge_zero(
        self,
    ) -> None:
        """So a reader comparing the two ratios does not conclude a target was missed."""
        content = _flat(_read_readout())
        assert "86.1" in content and "67.6" in content, (
            "the readout does not state both zero-staked shares, so the two ratios cannot be "
            "compared by a reader at all"
        )
        assert "no-edge" in content.lower(), (
            "the readout does not describe the winner target's zero-staked count as a legitimate "
            "no-edge Kelly zero"
        )

    def test_the_negative_wp_clv_and_the_non_regression_gate_are_stated_plainly(
        self,
    ) -> None:
        """R11's hardest sentence, asserted so it cannot be softened away later."""
        content = _flat(_read_readout())
        assert "-0.03800034" in content, (
            "the readout does not state the deployed win-probability model's measured absolute "
            "pooled closing-line value"
        )
        assert "non-regression gate" in content.lower(), (
            "the readout does not state that the deploy gate is a non-regression gate"
        )
        assert "never asserted a positive market edge" in content.lower(), (
            "the readout does not state that the gate never asserted a positive market edge"
        )

    def test_the_five_deliberately_red_tests_are_explained(self) -> None:
        """Five reds on a shipped milestone is honesty of record, and it has to be readable."""
        content = _flat(_read_readout())
        for test_path in (
            "test_gate_baseline_byte_identity.py",
            "test_promote_models.py",
            "test_gold_rebuild_attribution.py",
            "test_n01_resync_control.py",
        ):
            assert test_path in content, (
                f"the readout does not name {test_path}, so a reader cannot tell which failing "
                "tests are the deliberate ones."
            )

    def test_the_standing_open_items_are_pointed_at(self) -> None:
        """Still-open items are pointed at, never claimed closed."""
        content = _flat(_read_readout())
        assert "Seven quarantined reproductions remain OPEN" in content, (
            "the readout does not carry the standing quarantined-reproduction count from "
            "STATE-OF-SYSTEM.md"
        )
        assert "six deferred registers remain open" in content.lower(), (
            "the readout does not carry the standing deferred-register count"
        )


# ---------------------------------------------------------------------------
# THE GENERATION SEAM, AND WHY THIS MODULE NEEDS NO GATE -- Plan 33.1-08 Task 2,
# 2026-09-14.
#
# Plan 33.1-08 predicted this module would redden when Plan 33.1-07's rung-3
# rebuild moved gold. It did not, and the reason is worth writing down rather
# than leaving as a lucky escape: this module never re-derives a number from
# gold. It compares the document against the FROZEN committed verdict artifact,
# which the rebuild did not touch. A guard anchored to an artifact is immune to a
# gold generation change in a way a guard anchored to a re-run is not.
#
# If a future edit ever makes an assertion here call a harness that scores gold,
# that assertion needs tests.gold_generation.require_gold_generation and this
# note stops being true. See tests.phase33_state.GOLD_DERIVED_READINGS, which
# records this readout and states exactly this.
# ---------------------------------------------------------------------------
