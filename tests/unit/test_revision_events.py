"""The committed revision event-class and severity vocabulary (D32-12, Plan 32-03 Task 1).

These tests are the contract `data/revision_events.py` is held to. Phases 34 and 35 import
that module, so a member renamed later touches every consumer AND every already-written
verdict -- which is why the shape (seven classes, three severities, a total severity table)
is asserted here rather than left to review.

Three of the assertions below are the ones that actually earn their keep:

* ``set(DEFAULT_SEVERITY) == set(RevisionEventClass)`` -- adding an event class without
  giving it a severity FAILS, so the table can never go partial.
* the graded escalation is a NUMBER (``severity_rank`` of the graded default strictly
  exceeds the ungraded one), so D32-12's "the graded case escalates" is checkable rather
  than a sentence in a docstring.
* ``UNKNOWN`` outranks ``INFORMATIONAL`` -- PITFALLS F2's trap is that a dead detector and a
  healthy system are the same observable, so "I could not tell" must never be as quiet as
  "I looked and saw nothing wrong".

ASCII only, no emoji (CLAUDE.md hard constraint).

Run:  .venv/Scripts/python.exe -m pytest tests/unit/test_revision_events.py -q
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from data import revision_events
from data.revision_events import (
    CORRECTION_OWED,
    CORRECTION_OWED_SCOPE,
    DEFAULT_SEVERITY,
    SEVERITY_ORDER,
    VERDICT_SCHEMA_VERSION,
    RevisionEventClass,
    RevisionSeverity,
    severity_rank,
)

# The seven event classes D32-12 fixes, and nothing else. Written out as literals rather
# than derived from the enum so that a member SILENTLY renamed in the module fails here.
EXPECTED_EVENT_CLASSES: tuple[str, ...] = (
    "SEALED_REVISION",
    "LIVE_REVISION_GRADED",
    "LIVE_REVISION",
    "NO_PRIOR_CAPTURE",
    "KNOWN_DIVERGENCE_STABLE",
    "CLEAN",
    "UNKNOWN",
)

EXPECTED_SEVERITIES: tuple[str, ...] = ("INFORMATIONAL", "WARNING", "CRITICAL")


def _module_source_tree() -> ast.Module:
    """The module's own source, parsed. Used for the no-imports / no-logic assertions."""
    path = Path(revision_events.__file__)
    return ast.parse(path.read_text(encoding="utf-8"))


def _isort_style_sorted(names: list[str]) -> list[str]:
    """Ruff RUF022's isort-style order: SCREAMING_SNAKE, then CamelCase, then snake_case.

    The repository lints ``__all__`` with RUF022, whose "sorted" is this three-group order
    and NOT ``sorted()``. Asserting against plain ``sorted()`` would put this test in a fight
    with ``ruff --fix``, which the repository runs as a pre-commit hook.
    """

    def group(name: str) -> int:
        if name.isupper():
            return 0
        if name[:1].isupper():
            return 1
        return 2

    return sorted(names, key=lambda name: (group(name), name))


class TestTheVocabularyIsClosed:
    """Seven event classes, three severities, and every value its own lowercase name."""

    def test_the_event_class_has_exactly_the_seven_named_members(self) -> None:
        assert [member.name for member in RevisionEventClass] == list(
            EXPECTED_EVENT_CLASSES
        )

    def test_the_severity_has_exactly_the_three_named_members(self) -> None:
        assert [member.name for member in RevisionSeverity] == list(EXPECTED_SEVERITIES)

    def test_every_event_class_value_is_its_own_lowercase_name(self) -> None:
        for member in RevisionEventClass:
            assert member.value == member.name.lower()

    def test_every_severity_value_is_its_own_lowercase_name(self) -> None:
        for member in RevisionSeverity:
            assert member.value == member.name.lower()

    def test_a_verdict_dict_serialises_without_a_custom_encoder(self) -> None:
        # A StrEnum member IS a str, so json.dumps writes the value with no ``.value`` at
        # the write site -- the reason this module uses StrEnum and not plain Enum.
        payload = json.dumps(
            {
                "event_class": RevisionEventClass.CLEAN,
                "severity": RevisionSeverity.CRITICAL,
            }
        )
        assert json.loads(payload) == {"event_class": "clean", "severity": "critical"}


class TestTheSeverityTableIsTotal:
    """Every event class has a severity, and the severities encode the D32-12 rules."""

    def test_the_table_covers_every_event_class_and_no_more(self) -> None:
        # The load-bearing one: a new event class without a severity FAILS here rather
        # than reaching a KeyError inside a detector at 6 PM on a Friday.
        assert set(DEFAULT_SEVERITY) == set(RevisionEventClass)

    def test_every_mapped_value_is_a_severity_member(self) -> None:
        for severity in DEFAULT_SEVERITY.values():
            assert isinstance(severity, RevisionSeverity)

    def test_a_sealed_revision_is_critical(self) -> None:
        assert DEFAULT_SEVERITY[RevisionEventClass.SEALED_REVISION] is (
            RevisionSeverity.CRITICAL
        )

    def test_a_known_stable_divergence_is_informational(self) -> None:
        # D32-10: an acknowledged, stable divergence must not re-fire at the volume of a
        # new one, or it is wallpaper by week three.
        assert DEFAULT_SEVERITY[RevisionEventClass.KNOWN_DIVERGENCE_STABLE] is (
            RevisionSeverity.INFORMATIONAL
        )

    def test_a_clean_verdict_is_informational(self) -> None:
        assert DEFAULT_SEVERITY[RevisionEventClass.CLEAN] is (
            RevisionSeverity.INFORMATIONAL
        )


class TestTheSeverityOrderIsANumber:
    """severity_rank turns "louder than" into an integer comparison."""

    def test_the_order_is_the_three_severities_ascending(self) -> None:
        assert SEVERITY_ORDER == (
            RevisionSeverity.INFORMATIONAL,
            RevisionSeverity.WARNING,
            RevisionSeverity.CRITICAL,
        )

    def test_critical_outranks_warning_outranks_informational(self) -> None:
        assert severity_rank(RevisionSeverity.CRITICAL) > severity_rank(
            RevisionSeverity.WARNING
        )
        assert severity_rank(RevisionSeverity.WARNING) > severity_rank(
            RevisionSeverity.INFORMATIONAL
        )

    def test_the_graded_case_escalates_above_the_ungraded_one(self) -> None:
        # D32-12, expressed as a number: a live revision touching an ALREADY-GRADED week
        # is strictly louder than an ordinary in-season revision of a week nobody bet.
        assert severity_rank(
            DEFAULT_SEVERITY[RevisionEventClass.LIVE_REVISION_GRADED]
        ) > severity_rank(DEFAULT_SEVERITY[RevisionEventClass.LIVE_REVISION])

    def test_an_ordinary_live_revision_is_not_critical(self) -> None:
        # PITFALLS B3: an in-season revision of the current week is NORMAL. A detector
        # that fires CRITICAL every week is wallpaper by week three.
        assert (
            DEFAULT_SEVERITY[RevisionEventClass.LIVE_REVISION]
            is not RevisionSeverity.CRITICAL
        )

    def test_unknown_is_never_as_quiet_as_informational(self) -> None:
        # PITFALLS F2: a dead detector and a healthy system must not be the same observable.
        assert severity_rank(DEFAULT_SEVERITY[RevisionEventClass.UNKNOWN]) > (
            severity_rank(RevisionSeverity.INFORMATIONAL)
        )

    def test_unknown_is_quieter_than_a_confirmed_sealed_move(self) -> None:
        # "I could not tell" is an honest claim, not a confirmed finding. D32-09 makes
        # neither of them blocking, so overstating UNKNOWN buys nothing and costs trust.
        assert severity_rank(DEFAULT_SEVERITY[RevisionEventClass.UNKNOWN]) < (
            severity_rank(DEFAULT_SEVERITY[RevisionEventClass.SEALED_REVISION])
        )


class TestSeverityRankRefusesAWordOutsideTheVocabulary:
    """The _KNOWN_VERDICTS guard idiom: refuse, and name the frozen vocabulary."""

    @pytest.mark.parametrize("value", ["critical", "CRITICAL", 2, None, object()])
    def test_a_non_member_raises_value_error(self, value: object) -> None:
        # ``"critical"`` matters most: RevisionSeverity is a StrEnum, so a bare string
        # compares EQUAL to a member and would sail through a naive tuple.index lookup.
        with pytest.raises(ValueError, match="SEVERITY_ORDER"):
            severity_rank(value)  # type: ignore[arg-type]

    def test_the_message_names_the_frozen_vocabulary(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            severity_rank("informational")  # type: ignore[arg-type]
        message = str(excinfo.value)
        assert "SEVERITY_ORDER" in message
        assert "informational" in message


class TestTheQueryableCorrectionKeys:
    """D32-12 raises the ledger obligation as DATA; Phase 34 discharges it."""

    def test_the_two_keys_are_the_literal_snake_case_strings(self) -> None:
        assert CORRECTION_OWED == "correction_owed"
        assert CORRECTION_OWED_SCOPE == "correction_owed_scope"

    def test_the_schema_version_is_an_int_stamped_on_every_verdict(self) -> None:
        assert isinstance(VERDICT_SCHEMA_VERSION, int)
        assert VERDICT_SCHEMA_VERSION >= 1


class TestTheModuleIsConstantsOnly:
    """No project import, no I/O, no behaviour beyond the one pure rank lookup."""

    def test_it_imports_only_future_and_enum(self) -> None:
        tree = _module_source_tree()
        modules = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        modules |= {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert sorted(modules) == ["__future__", "enum"]

    def test_it_defines_no_class_beyond_the_two_enums(self) -> None:
        tree = _module_source_tree()
        classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        assert classes == ["RevisionEventClass", "RevisionSeverity"]

    def test_it_defines_no_function_beyond_severity_rank(self) -> None:
        tree = _module_source_tree()
        functions = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        assert functions == ["severity_rank"]

    def test_all_is_explicit_and_in_the_repository_canonical_order(self) -> None:
        names = list(revision_events.__all__)
        assert names == _isort_style_sorted(names)
        assert len(names) == len(set(names))
        for name in names:
            assert hasattr(revision_events, name)

    def test_all_names_every_public_symbol(self) -> None:
        public = {
            name
            for name in vars(revision_events)
            if not name.startswith("_") and name not in {"annotations", "StrEnum"}
        }
        assert public == set(revision_events.__all__)


class TestTheDocstringCarriesTheReversibilityWarning:
    """The file is a contract, not a convenience, and says so where a reader will look."""

    def test_it_names_both_importing_phases(self) -> None:
        doc = revision_events.__doc__ or ""
        assert "Phase 34" in doc
        assert "Phase 35" in doc

    def test_it_states_that_a_rename_touches_every_written_verdict(self) -> None:
        doc = (revision_events.__doc__ or "").lower()
        assert "rename" in doc
        assert "already-written verdict" in doc

    def test_it_is_ascii_only(self) -> None:
        source = Path(revision_events.__file__).read_text(encoding="utf-8")
        assert source.isascii()
