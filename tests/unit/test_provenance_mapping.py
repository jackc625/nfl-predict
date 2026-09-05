"""The two orthogonal D31-22 honesty labels: every case pinned, one stamping site, two vocabularies.

Phase 31, plan 31-13 (SPEC R8, D31-22, PROD-03). Three separable claims live here because they
are one seam -- each can be satisfied alone while the honesty property still fails.

1. **Every case of the mapping is pinned.** ``api.cache.classify_row_provenance`` keys on
   (season, run_mode) and returns TWO independent facts: ``provenance`` says HOW the row was
   produced, ``validation_type`` says what evidentiary weight it carries. Two columns rather than
   one exist for exactly one reason -- a 2025 row is replay-PRODUCED and clean-holdout WEIGHT at
   the same time, and no single label can say both. Every combination is enumerated below,
   including the two refusals, because a defaulted label is precisely the mislabel the function
   exists to prevent.

2. **The mapping is called from ONE place in the write path.**
   ``api.cache.stamp_bet_list_provenance`` is that place, and an AST scan over the production tree
   asserts the count is exactly one and REPORTS what it found. A second stamping site is how the
   labels drift: one caller updated, another left writing the retired constant, and the clean 2025
   holdout rows published as contaminated -- the inverse of the prohibition, and equally false.

3. **The two vocabularies have OPPOSITE directions and must stay disjoint.** LANDMINE-6: the
   inherited ``backtest.ou_monetization.CONTAMINATED_VOCAB`` is a REQUIRED-PRESENCE set -- the
   Phase-30 readout guard asserts those phrases ARE PRESENT wherever a Phase-27/30 figure is
   quoted. The Phase-31 ``backtest.ev_chain_constants.READOUT_FORBIDDEN_WORDS`` is a FORBIDDEN set
   -- those words must be ABSENT from the clean statement. Extending the contaminated vocabulary
   naively to the Phase-31 artifacts would therefore REQUIRE the clean readout to contain the
   contaminated phrases, asserting the opposite of what is meant. The two sets are kept disjoint
   and their directions are stated here rather than inferred.

Neither the Phase-27 provisional constant nor the contaminated vocabulary is deleted by this
phase; the Phase-27 and Phase-30 record depends on them and the shipped readout guard imports
them. Their continued existence is asserted below.

Run this module:  .venv/Scripts/python.exe -m pytest tests/unit/test_provenance_mapping.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from api.cache import (
    PROVENANCE_BACKTEST_REPLAY,
    PROVENANCE_FORWARD,
    RUN_MODE_FORWARD,
    RUN_MODE_REPLAY,
    VALIDATION_TYPE_CLEAN_HOLDOUT,
    VALIDATION_TYPE_CONTAMINATED,
    VALIDATION_TYPE_FORWARD_REALIZED,
    classify_row_provenance,
    stamp_bet_list_provenance,
)
from backtest.ev_chain_constants import READOUT_FORBIDDEN_WORDS
from backtest.ou_monetization import CONTAMINATED_VOCAB, VALIDATION_TYPE_PROVISIONAL

REPO_ROOT = Path(__file__).resolve().parents[2]

# The PRODUCTION tree the AST scan walks. ``tests/`` is deliberately excluded -- a test calling the
# mapping directly to assert what it returns is not a stamping site, and forbidding that would make
# this very module a violation. ``outputs/`` holds archived copies of retired source that nothing
# imports.
_WRITE_PATH_ROOTS: tuple[str, ...] = (
    "api",
    "backtest",
    "pipeline",
    "scripts",
    "models",
    "features",
    "utils",
    "ratings",
)

_MAPPING_NAME = "classify_row_provenance"

# The ONE function permitted to call it, and the module it lives in.
_SOLE_STAMPING_SITE = ("api/cache.py", "stamp_bet_list_provenance")

_REPLAY_CONTAMINATED_SEASONS = (2021, 2022, 2023, 2024)
_CLEAN_HOLDOUT_SEASON = 2025


# ---------------------------------------------------------------------------
# 1. Every case of the two-column mapping
# ---------------------------------------------------------------------------


class TestEveryCaseOfTheMappingIsPinned:
    """Each row of the (season x run_mode) table has its own test asserting BOTH columns."""

    @pytest.mark.parametrize("season", _REPLAY_CONTAMINATED_SEASONS)
    def test_a_replay_row_in_the_burned_window_is_replay_and_contaminated(
        self, season: int
    ) -> None:
        provenance, validation_type = classify_row_provenance(season, RUN_MODE_REPLAY)
        assert provenance == PROVENANCE_BACKTEST_REPLAY
        assert validation_type == VALIDATION_TYPE_CONTAMINATED

    def test_a_2025_replay_row_is_replay_produced_and_clean_holdout_weight(
        self,
    ) -> None:
        """The case that makes TWO columns necessary rather than one.

        2025 is the single unburned split. A row reconstructed from it is replay-PRODUCED (it was
        never recommended in advance) and clean-holdout WEIGHT (the split was not spent on
        tuning). One column would have to choose, and either choice would be a lie.
        """
        provenance, validation_type = classify_row_provenance(
            _CLEAN_HOLDOUT_SEASON, RUN_MODE_REPLAY
        )
        assert provenance == PROVENANCE_BACKTEST_REPLAY
        assert validation_type == VALIDATION_TYPE_CLEAN_HOLDOUT
        assert validation_type != VALIDATION_TYPE_CONTAMINATED, (
            "the clean 2025 holdout must never be stamped contaminated -- that is the inverse of "
            "the prohibition and equally false (D31-22)"
        )

    @pytest.mark.parametrize("season", [2019, 2021, 2024, 2025, 2026, 2031])
    def test_a_forward_row_is_forward_and_forward_realized_for_any_season(
        self, season: int
    ) -> None:
        """Forward provenance is season-independent: a bet written before kickoff is a record."""
        assert classify_row_provenance(season, RUN_MODE_FORWARD) == (
            PROVENANCE_FORWARD,
            VALIDATION_TYPE_FORWARD_REALIZED,
        )

    def test_the_pinned_cases_cover_the_whole_declared_vocabulary(self) -> None:
        """No declared label is unreachable and no reachable label is undeclared."""
        produced = {
            classify_row_provenance(season, RUN_MODE_REPLAY)
            for season in (*_REPLAY_CONTAMINATED_SEASONS, _CLEAN_HOLDOUT_SEASON)
        } | {classify_row_provenance(2026, RUN_MODE_FORWARD)}
        assert {p for p, _ in produced} == {
            PROVENANCE_BACKTEST_REPLAY,
            PROVENANCE_FORWARD,
        }
        assert {v for _, v in produced} == {
            VALIDATION_TYPE_CONTAMINATED,
            VALIDATION_TYPE_CLEAN_HOLDOUT,
            VALIDATION_TYPE_FORWARD_REALIZED,
        }


class TestTheMappingRefusesRatherThanDefaults:
    """Both refusals name the offending value, so the caller can see what it passed."""

    @pytest.mark.parametrize("run_mode", ["backfill", "REPLAY", "", "forwards"])
    def test_an_out_of_vocabulary_run_mode_raises_with_the_value_in_the_message(
        self, run_mode: str
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            classify_row_provenance(2023, run_mode)
        assert repr(run_mode) in str(excinfo.value)

    @pytest.mark.parametrize("season", [2019, 2020, 2026, 1999])
    def test_an_out_of_range_replay_season_raises_with_the_value_in_the_message(
        self, season: int
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            classify_row_provenance(season, RUN_MODE_REPLAY)
        assert repr(season) in str(excinfo.value)

    def test_no_combination_silently_returns_a_default(self) -> None:
        """A refusal is a RAISE. Nothing returns a placeholder pair."""
        for season, run_mode in ((2020, RUN_MODE_REPLAY), (2023, "unknown")):
            with pytest.raises(ValueError):
                classify_row_provenance(season, run_mode)


# ---------------------------------------------------------------------------
# 2. The stamping seam
# ---------------------------------------------------------------------------


def _frame(seasons: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"g{index}" for index, _ in enumerate(seasons)],
            "season": seasons,
        }
    )


class TestTheStampingSeam:
    """``stamp_bet_list_provenance`` writes both columns onto every row through the mapping."""

    def test_a_replay_frame_spanning_the_burned_window_and_2025_gets_both_labels(
        self,
    ) -> None:
        stamped = stamp_bet_list_provenance(_frame([2021, 2024, 2025]), RUN_MODE_REPLAY)
        assert list(stamped["provenance"]) == [PROVENANCE_BACKTEST_REPLAY] * 3
        assert list(stamped["validation_type"]) == [
            VALIDATION_TYPE_CONTAMINATED,
            VALIDATION_TYPE_CONTAMINATED,
            VALIDATION_TYPE_CLEAN_HOLDOUT,
        ]

    def test_a_forward_frame_gets_the_forward_pair(self) -> None:
        stamped = stamp_bet_list_provenance(_frame([2025, 2026]), RUN_MODE_FORWARD)
        assert list(stamped["provenance"]) == [PROVENANCE_FORWARD] * 2
        assert (
            list(stamped["validation_type"]) == [VALIDATION_TYPE_FORWARD_REALIZED] * 2
        )

    def test_the_input_frame_is_not_mutated(self) -> None:
        """The caller's frame is left alone; stamping returns a new frame."""
        original = _frame([2023])
        stamp_bet_list_provenance(original, RUN_MODE_REPLAY)
        assert "provenance" not in original.columns
        assert "validation_type" not in original.columns

    def test_an_empty_frame_returns_both_columns_and_zero_rows(self) -> None:
        stamped = stamp_bet_list_provenance(_frame([]), RUN_MODE_REPLAY)
        assert len(stamped) == 0
        assert "provenance" in stamped.columns
        assert "validation_type" in stamped.columns

    def test_a_frame_without_a_season_column_raises_a_named_key_error(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            stamp_bet_list_provenance(pd.DataFrame({"game_id": ["g"]}), RUN_MODE_REPLAY)
        assert "season" in str(excinfo.value)

    def test_an_out_of_range_season_in_the_frame_refuses_the_whole_stamp(self) -> None:
        """One unstampable row refuses the frame; it never inherits a neighbour's label."""
        with pytest.raises(ValueError) as excinfo:
            stamp_bet_list_provenance(_frame([2023, 2019]), RUN_MODE_REPLAY)
        assert "2019" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2b. The AST scan: EXACTLY one call site in the write path
# ---------------------------------------------------------------------------


def _production_python_files() -> list[Path]:
    files: list[Path] = []
    for root in _WRITE_PATH_ROOTS:
        root_dir = REPO_ROOT / root
        if not root_dir.is_dir():
            continue
        files.extend(
            path for path in root_dir.rglob("*.py") if "__pycache__" not in path.parts
        )
    assert files, (
        "the production walk found no Python files; a scan that walks nothing passes VACUOUSLY"
    )
    return files


def _enclosing_function_names(tree: ast.AST) -> dict[int, str]:
    """Map every node inside a function body to that function's name, innermost winning."""
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                owner[id(child)] = node.name
    return owner


def _is_mapping_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == _MAPPING_NAME
    if isinstance(func, ast.Attribute):
        return func.attr == _MAPPING_NAME
    return False


def _mapping_call_sites() -> list[tuple[str, int, str]]:
    """Return (posix path, lineno, enclosing function) for every call of the mapping."""
    sites: list[tuple[str, int, str]] = []
    for path in _production_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_function_names(tree)
        for node in ast.walk(tree):
            if not _is_mapping_call(node):
                continue
            rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            sites.append((rel, node.lineno, owner.get(id(node), "<module>")))
    return sorted(sites)


class TestExactlyOneStampingSiteExistsInTheWritePath:
    """A second stamping site cannot appear later without turning this red."""

    def test_the_mapping_is_called_exactly_once_and_the_count_is_reported(self) -> None:
        sites = _mapping_call_sites()
        rendered = "\n".join(f"  {p}:{ln} in {fn}()" for p, ln, fn in sites)
        assert len(sites) == 1, (
            f"AST scan found {len(sites)} call site(s) of {_MAPPING_NAME} in the write path "
            f"{_WRITE_PATH_ROOTS}; exactly 1 is permitted so no second stamping site can "
            f"drift:\n{rendered}"
        )

    def test_the_one_call_site_is_the_declared_stamping_function(self) -> None:
        sites = _mapping_call_sites()
        assert len(sites) == 1
        path, _lineno, function = sites[0]
        assert (path, function) == _SOLE_STAMPING_SITE, (
            f"the sole {_MAPPING_NAME} call moved to {path}:{function}(); the declared stamping "
            f"site is {_SOLE_STAMPING_SITE[0]}:{_SOLE_STAMPING_SITE[1]}()"
        )

    def test_the_scan_would_catch_a_second_site(self) -> None:
        """The scan is load-bearing: it counts BOTH shapes in a synthetic two-site module."""
        source = (
            "def a():\n"
            "    return classify_row_provenance(2023, 'replay')\n"
            "def b():\n"
            "    return cache.classify_row_provenance(2025, 'replay')\n"
        )
        tree = ast.parse(source)
        owner = _enclosing_function_names(tree)
        found = [owner.get(id(n)) for n in ast.walk(tree) if _is_mapping_call(n)]
        assert sorted(found) == ["a", "b"], found


# ---------------------------------------------------------------------------
# 3. Two vocabularies, opposite directions, disjoint (LANDMINE-6)
# ---------------------------------------------------------------------------


class TestTheTwoVocabulariesKeepTheirOppositeDirections:
    """A REQUIRED-PRESENCE set and a FORBIDDEN set must never share a member."""

    def test_the_inherited_contaminated_vocabulary_still_exists(self) -> None:
        """Phase 31 does NOT delete it -- the Phase-27/30 record and its readout guard need it."""
        assert CONTAMINATED_VOCAB
        assert all(isinstance(phrase, str) and phrase for phrase in CONTAMINATED_VOCAB)
        assert VALIDATION_TYPE_PROVISIONAL == "PROVISIONAL_CONTAMINATED"

    def test_the_two_vocabularies_are_disjoint(self) -> None:
        required_presence = {phrase.lower() for phrase in CONTAMINATED_VOCAB}
        forbidden = {word.lower() for word in READOUT_FORBIDDEN_WORDS}
        overlap = required_presence & forbidden
        assert not overlap, sorted(overlap)

    def test_no_forbidden_word_is_contained_in_a_required_phrase(self) -> None:
        """Substring containment counts.

        A required phrase carrying a forbidden word would make the two guards mutually
        unsatisfiable on any document that must satisfy both.
        """
        offending = [
            (phrase, word)
            for phrase in CONTAMINATED_VOCAB
            for word in READOUT_FORBIDDEN_WORDS
            if word.lower() in phrase.lower()
        ]
        assert not offending, offending

    def test_the_containment_check_runs_in_both_directions(self) -> None:
        """Neither set contains the other's members, whichever way the containment is read."""
        assert not [
            (word, phrase)
            for word in READOUT_FORBIDDEN_WORDS
            for phrase in CONTAMINATED_VOCAB
            if phrase.lower() in word.lower()
        ]

    def test_the_two_validation_type_vocabularies_do_not_collide(self) -> None:
        """The Phase-31 labels are lower-case snake tokens; the Phase-27 one is a SHOUTED constant.

        A reader (and a grep) can tell a Phase-31 row from a Phase-27 one without context.
        """
        phase31 = {
            VALIDATION_TYPE_CONTAMINATED,
            VALIDATION_TYPE_CLEAN_HOLDOUT,
            VALIDATION_TYPE_FORWARD_REALIZED,
        }
        assert VALIDATION_TYPE_PROVISIONAL not in phase31
        assert all(label == label.lower() for label in phase31)
