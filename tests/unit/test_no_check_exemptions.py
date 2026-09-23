"""There is no way to exempt a source from the information-time check (Plan 33.2-20).

THE STANDING PROHIBITION THIS MODULE GUARDS
-------------------------------------------
SPEC R2's must-not: no source, game or season may be exempted from the information-time
check through an allow-list, a labelled exception or a report-only mode, and there must be
no code path that REPORTS a violation without refusing. That is a property of a SHAPE, not
of a run: a mode added six months from now would pass every behavioural test in the tree
while making the whole gate optional. So it is checked structurally, over the parsed trees
of the three modules the check path lives in -- ``features/provenance.py`` (the gate),
``scripts/build_features.py`` (which calls it) and ``scripts/validate_features.py`` (which
reports on persisted gold, and is the third place a downgrade could live).

WHY THE SCAN READS STRING CONSTANTS, AND WHY NOT ALL OF THEM
------------------------------------------------------------
A scan over identifiers alone goes BLIND to the form such a hatch is most likely to take:

    if os.environ["GSD_REPORT_ONLY"]:
        log.warning(...); return

There is no identifier named ``report_only`` anywhere in that. The name lives in a STRING,
in SUBSCRIPT position. So string constants are in the subject -- but ONLY where the
constant is a ``Subscript`` slice or an ``Assign`` / ``AnnAssign`` value, which is where a
switch can actually be read or bound.

A DOCSTRING IS A STRING CONSTANT TOO, and that is the whole reason for the restriction.
Plan 33.2-20's final registration comment is REQUIRED to state in as many words that there
is no allow-list, no labelled exception and no report-only mode -- an unrestricted scan
would be tripped by the very sentence it exists to make true. A docstring is an
``ast.Expr`` statement whose value is the constant: neither a subscript slice nor an
assignment value, so it is out of reach while ``os.environ["GSD_REPORT_ONLY"]`` is still
CAUGHT. Comments are absent from the AST entirely, which is why that registration text is
written as ``#`` comments.

THE BASELINE IS IMPORTED, NOT RE-DECLARED, AND IT IS NOT AN ALLOW-LIST
----------------------------------------------------------------------
The scan does not start from zero. MEASURED 2026-09-16 against the live tree, with the
node-shape restriction applied, it returns exactly ONE hit:
``discrete_indicators_exempt`` -- a KEYWORD ARGUMENT in a structured-log call at
``scripts/build_features.py:1154`` that counts ``_is_discrete_indicator`` columns for the
z-scoring preserving set. It has nothing whatever to do with information time.

That token is declared ONCE, as ``features.provenance.KNOWN_UNRELATED_SCAN_TOKENS``, and
this module IMPORTS it. It is declared in a SOURCE module rather than here for two
reasons: Plan 33.2-20 Task 1's own verify has to read it at a boundary where this module
does not yet exist (importing it from here would raise ``ModuleNotFoundError`` and fail
that gate on every run whether or not the work was done), and a constant whose NAME
contained ``exempt`` inside a SCANNED file would enter its own hit set through its
``ast.Name`` assignment target and make the signal permanently non-empty.

**THIS IS NOT AN ALLOW-LIST ON THE INFORMATION-TIME CHECK.** The check has none and gains
none: a source is checked, or the build refuses and names it. What is declared here is a
measured, SHRINK-ONLY baseline on a SOURCE SCAN's vocabulary, bounded in BOTH directions --
``NEW_EXEMPTION_TOKENS`` (hits minus baseline) must be empty, and ``STALE_KNOWN`` (baseline
minus hits) must be empty too, so a declared token that has left the tree must be removed
from the declaration and the allowance can never outlive its subject and be reused to cover
a later hit. A reader who confuses the two will read this module as the thing it exists to
forbid, which is why the distinction is stated here and again beside the declaration.

THE TWO PRE-EXISTING SAFETY DOCSTRINGS ARE BYTE-UNCHANGED, BY ASSERTION
-----------------------------------------------------------------------
An UNRESTRICTED form of this scan also hits two docstring sentences in
``scripts/build_features.py`` -- the "...so the level exemption would be an EMPTY tuple..."
z-scoring note and the "The active builder's preserving set. FAIL-CLOSED, per builder. An
exemption that silently does nothing is worse than none..." preserving-set note. Both
document an UNRELATED fail-closed mechanism. Rewording safety documentation to make an
unrelated gate pass is the failure this plan exists to prevent, one level up, so a test
below pins both sentences byte for byte. They leave the subject by NODE SHAPE, not by
being edited.

THE FOUR STRUCTURAL CONTROLS
----------------------------
1. non-vacuity -- three files parsed, a non-empty imported baseline, a non-trivial node
   count;
2. the assertion -- ``NEW_EXEMPTION_TOKENS`` empty, ``STALE_KNOWN`` empty, zero
   ``os.environ`` / ``getenv`` lookups in any of the three;
3. a planted violation -- a synthetic ``if report_only: log.warning(...); return`` branch,
   and the environment-variable form of the same hatch, are both flagged;
4. no false positive -- a module whose DOCSTRING narrates the absent exemption mechanisms
   at length is NOT flagged, and neither is an ordinary logging call.

Run this module:  uv run pytest tests/unit/test_no_check_exemptions.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from features.provenance import KNOWN_UNRELATED_SCAN_TOKENS

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The three modules the information-time check path lives in.
CHECK_PATH_MODULES: tuple[str, ...] = (
    "features/provenance.py",
    "scripts/build_features.py",
    "scripts/validate_features.py",
)

#: The vocabulary a downgrade would be spelled in. ``exempt`` is the live collision with the
#: tree as it stands; the hyphenated ``allow-list`` matches nothing and is deliberately not
#: listed, because the registration comment must be free to use that spelling.
EXEMPTION_PATTERN = re.compile(
    r"report_only|allow_list|allowlist|skip_check|exempt", re.IGNORECASE
)

#: The two FAIL-CLOSED safety texts an UNRESTRICTED scan hits, pinned byte for byte. Both
#: describe an unrelated z-scoring / preserving-set mechanism and must not be reworded to
#: make this gate pass.
#:
#: MEASURED CORRECTION TO PLAN 33.2-20's TEXT. The plan calls both of them DOCSTRINGS. Only
#: the second is: the first is the REFUSAL MESSAGE of the WR-10 preserving-set guard, built
#: as an implicitly-concatenated f-string assigned to ``msg``. That makes it an
#: ``ast.JoinedStr`` rather than an ``ast.Constant``, so it is out of the scan's subject for
#: a second, independent reason -- a formatted string is not a constant at all. The node
#: shape keeps both out; what it keeps them out of is the same.
PROTECTED_SAFETY_TEXTS: tuple[str, ...] = (
    (
        'f"preserving set ({sorted(active_preserved)}), so the level exemption "\n'
        '                "would be an EMPTY tuple and the coverage flag would be '
        'z-scored back to "'
    ),
    (
        '"""The active builder\'s preserving set. FAIL-CLOSED, per builder.\n\n'
        "        An exemption that silently does nothing is worse than none"
    ),
)


def _identifier_names(tree: ast.AST) -> set[str]:
    """Every NAME a switch could be spelled as: identifiers, parameters, keywords, defs."""
    names: set[str] = set()
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    names |= {
        keyword.arg or ""
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        for keyword in n.keywords
    }
    names |= {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    return names


def _switch_position_strings(tree: ast.AST) -> set[str]:
    """String constants in the two positions a switch can be READ or BOUND from.

    A ``Subscript`` slice catches ``os.environ["GSD_REPORT_ONLY"]``; an ``Assign`` /
    ``AnnAssign`` value catches ``MODE = "report_only"``. A bare string STATEMENT -- a
    docstring -- is neither, and can build no switch.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            found.add(node.slice.value)
        if (
            isinstance(node, ast.Assign | ast.AnnAssign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            found.add(node.value.value)
    return found


def exemption_tokens(tree: ast.AST) -> set[str]:
    """Every token in *tree* matching the exemption vocabulary, in a reachable position."""
    subject = _identifier_names(tree) | _switch_position_strings(tree)
    return {token for token in subject if token and EXEMPTION_PATTERN.search(token)}


def environment_lookup_lines(tree: ast.AST) -> list[int]:
    """Lines of every ``os.environ[...]`` / ``os.getenv(...)`` read. There must be none."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "getenv":
                lines.append(node.lineno)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "environ"
        ):
            lines.append(node.lineno)
    return sorted(lines)


def _trees() -> list[tuple[str, ast.Module]]:
    return [
        (path, ast.parse((REPO_ROOT / path).read_text(encoding="utf-8")))
        for path in CHECK_PATH_MODULES
    ]


def _all_hits() -> set[str]:
    return {token for _, tree in _trees() for token in exemption_tokens(tree)}


# ---------------------------------------------------------------------------
# Control 1: non-vacuity
# ---------------------------------------------------------------------------


class TestTheScanIsNotVacuous:
    def test_all_three_check_path_modules_parse(self) -> None:
        trees = _trees()
        assert len(trees) == 3
        for path, tree in trees:
            assert len(list(ast.walk(tree))) > 200, f"{path} parsed to almost nothing"

    def test_the_imported_baseline_is_non_empty(self) -> None:
        """The non-vacuity control on the SUBTRACTION the binding signal performs.

        With an empty baseline, "hits minus baseline is empty" would only be saying
        "there are no hits", and the measured fact that the tree already carries one
        unrelated token would have gone unrecorded.
        """
        assert len(KNOWN_UNRELATED_SCAN_TOKENS) > 0

    def test_the_baseline_is_declared_in_a_source_module_not_here(self) -> None:
        """One declaration, one home -- and Task 1's verify can read it before this exists.

        Checked over this module's own PARSED TREE, never over its text: a substring
        check would match the assertion that performs it, which is the self-reference a
        text scan cannot see its way out of.
        """
        from features import provenance

        assert provenance.KNOWN_UNRELATED_SCAN_TOKENS is KNOWN_UNRELATED_SCAN_TOKENS

        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        bound_here = [
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id == "KNOWN_UNRELATED_SCAN_TOKENS"
        ] + [
            node.target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "KNOWN_UNRELATED_SCAN_TOKENS"
        ]
        assert bound_here == [], (
            "this module BINDS the baseline name instead of importing it. Two "
            "declarations drift, and a name matching the scan's own vocabulary inside a "
            "scanned file enters its own hit set."
        )

    def test_the_baselines_reason_is_recorded_here(self) -> None:
        """Restated in this module's docstring, per Plan 33.2-20 Task 2."""
        assert __doc__ is not None
        assert "discrete_indicators_exempt" in __doc__
        assert "z-scoring" in __doc__
        assert "NOT AN ALLOW-LIST" in __doc__


# ---------------------------------------------------------------------------
# Control 2: the assertion
# ---------------------------------------------------------------------------


class TestNoExemptionMechanismExists:
    def test_no_new_token_beyond_the_declared_baseline(self) -> None:
        new_tokens = sorted(_all_hits() - set(KNOWN_UNRELATED_SCAN_TOKENS))
        assert new_tokens == [], (
            f"the information-time check path gained token(s) {new_tokens}. A source is "
            "checked, or the build refuses and names it -- there is no allow-list, no "
            "labelled exception and no report-only mode, and none may be added. If a "
            "token is genuinely unrelated, MEASURE it and add it to "
            "features.provenance.KNOWN_UNRELATED_SCAN_TOKENS with its reason."
        )

    def test_the_baseline_has_not_outlived_its_subject(self) -> None:
        stale = sorted(set(KNOWN_UNRELATED_SCAN_TOKENS) - _all_hits())
        assert stale == [], (
            f"declared baseline token(s) {stale} are no longer in the tree. The baseline "
            "SHRINKS ONLY: a stale entry is an allowance with no subject, and an "
            "allowance with no subject is available to cover the next real hit."
        )

    @pytest.mark.parametrize("path", CHECK_PATH_MODULES)
    def test_no_module_reads_an_environment_variable(self, path: str) -> None:
        tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
        lines = environment_lookup_lines(tree)
        assert lines == [], (
            f"{path} reads an environment variable at line(s) {lines}. A gate whose "
            "behaviour depends on the environment is a gate that can be turned off from "
            "outside the repository, with nothing in the diff to show for it."
        )


class TestTheTwoSafetyTextsAreByteUnchanged:
    """Rewording safety documentation to satisfy an unrelated gate is forbidden by name."""

    @pytest.mark.parametrize("sentence", PROTECTED_SAFETY_TEXTS)
    def test_the_sentence_is_still_there_verbatim(self, sentence: str) -> None:
        source = (REPO_ROOT / "scripts/build_features.py").read_text(encoding="utf-8")
        assert sentence in source, (
            "a pre-existing FAIL-CLOSED safety text in scripts/build_features.py was "
            "reworded. It describes an unrelated z-scoring / preserving-set mechanism "
            "and it leaves this scan's subject by NODE SHAPE, not by being edited. "
            "Restore it."
        )

    def test_they_are_out_of_the_subject_by_shape_not_by_wording(self) -> None:
        """Proof that the restriction, not an edit, is what keeps them out."""
        tree = ast.parse(
            (REPO_ROOT / "scripts/build_features.py").read_text(encoding="utf-8")
        )
        narrating = {
            node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        assert any(EXEMPTION_PATTERN.search(text) for text in narrating), (
            "no docstring in scripts/build_features.py matches the exemption "
            "vocabulary any more, so this control is asserting nothing. Either the "
            "safety docstrings were reworded (forbidden) or the vocabulary moved."
        )
        assert not (narrating & _all_hits()), (
            "a narrating docstring entered the hit set, so the node-shape restriction "
            "is not holding."
        )


# ---------------------------------------------------------------------------
# Control 3: planted violations
# ---------------------------------------------------------------------------

_PLANTED_FLAG_BRANCH = '''
"""A module that quietly makes the check optional."""
import logging

log = logging.getLogger(__name__)


def check(source, report_only=False):
    if report_only:
        log.warning("information-time violation in %s", source)
        return
    raise RuntimeError(source)
'''

_PLANTED_ENVIRONMENT_HATCH = '''
"""A module whose switch lives in a STRING, which an identifier scan cannot see."""
import logging
import os

log = logging.getLogger(__name__)


def check(source):
    if os.environ["GSD_REPORT_ONLY"]:
        log.warning("information-time violation in %s", source)
        return
    raise RuntimeError(source)
'''

_PLANTED_ASSIGNED_MODE = """
MODE = "report_only"
"""


class TestThePlantedViolationsAreFlagged:
    def test_a_report_only_parameter_and_branch_is_flagged(self) -> None:
        hits = exemption_tokens(ast.parse(_PLANTED_FLAG_BRANCH))
        assert "report_only" in hits

    def test_the_environment_variable_hatch_is_flagged_through_its_string(self) -> None:
        """The form a strip-then-grep scan would go blind to."""
        tree = ast.parse(_PLANTED_ENVIRONMENT_HATCH)
        assert "GSD_REPORT_ONLY" in exemption_tokens(tree)
        assert environment_lookup_lines(tree) != []

    def test_a_getenv_call_is_flagged_as_an_environment_lookup(self) -> None:
        tree = ast.parse("import os\nMODE = os.getenv('X')\n")
        assert environment_lookup_lines(tree) != []

    def test_an_assigned_mode_string_is_flagged(self) -> None:
        assert "report_only" in exemption_tokens(ast.parse(_PLANTED_ASSIGNED_MODE))

    def test_the_planted_branch_would_fail_the_live_assertion(self) -> None:
        """End to end: with the plant in the subject, the binding signal is non-empty."""
        hits = _all_hits() | exemption_tokens(ast.parse(_PLANTED_FLAG_BRANCH))
        assert sorted(hits - set(KNOWN_UNRELATED_SCAN_TOKENS)) == ["report_only"]


# ---------------------------------------------------------------------------
# Control 4: no false positives
# ---------------------------------------------------------------------------

_NARRATING_MODULE = '''
"""A module that NARRATES the absent mechanisms, which the real one is required to do.

There is no allow-list here, no labelled exception and no report_only mode. A source is
checked, or the build refuses and names it. Nothing is exempt, nothing is skip_checked and
nothing is allowlisted.
"""
import logging

log = logging.getLogger(__name__)


def check(source):
    log.info("checked", source=source, rows=1)
    raise RuntimeError(source)
'''


class TestTheNoFalsePositiveControls:
    def test_a_docstring_narrating_the_absent_mechanisms_is_not_flagged(self) -> None:
        """The exact text Plan 33.2-20's registration comment is REQUIRED to contain."""
        assert exemption_tokens(ast.parse(_NARRATING_MODULE)) == set()

    def test_an_ordinary_logging_call_is_not_flagged(self) -> None:
        tree = ast.parse(
            "import logging\n"
            "log = logging.getLogger(__name__)\n"
            "def f(rows):\n"
            "    log.info('Dropped a group', group='market', count=rows)\n"
        )
        assert exemption_tokens(tree) == set()

    def test_the_narrating_module_reads_no_environment_variable(self) -> None:
        assert environment_lookup_lines(ast.parse(_NARRATING_MODULE)) == []

    def test_the_live_registration_comment_says_the_mechanisms_do_not_exist(
        self,
    ) -> None:
        """And it is a COMMENT, so it is free -- absent from the AST entirely."""
        source = (REPO_ROOT / "scripts/build_features.py").read_text(encoding="utf-8")
        assert "THERE IS NO EXEMPTION MECHANISM OF ANY KIND" in source
        assert "No allow-list. No labelled exception. No" in source
        assert "report-only mode." in source
        # And it really is a COMMENT: nothing in the parsed tree carries that sentence.
        tree = ast.parse(source)
        constants = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert not any(
            "THERE IS NO EXEMPTION MECHANISM" in text for text in constants
        ), (
            "the registration text has become a STRING. Comments are absent from the AST "
            "and are therefore free to name the mechanisms that do not exist; a string "
            "would trip the very check the sentence explains."
        )
