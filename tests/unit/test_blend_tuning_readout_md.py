"""Permanent doc-drift guard for the repo-root BLEND-TUNING-READOUT.md (Plan 33.2-24 Task 3).

A PERMANENT committed test, following ``tests/unit/test_signal_lift_readout_md.py``. It guards:

  - the file is missing or not at the repo root, or carries non-ASCII (CLAUDE.md);
  - a required section -- the weights, the exclusions, the stated 2025 gap, the "what is not
    here" section, the not-clean-evidence label -- was silently dropped;
  - the over-claim word ``proven`` appears, or the retired incumbent blend is named by its id
    (the "what is NOT here" section DESCRIBES it instead, so the two cannot collide);
  - THE BINDING CHECK: the exclusion counts the document publishes are EQUAL to the counts the
    state manifest records, class by class. ``documented_excluded_count`` and
    ``state_manifest_excluded_count`` each return a ``{class: games}`` mapping over the two
    exclusion classes, so a summarised exclusion, a dropped class or a readout that drifts from
    the manifest all fail the identity. A phrase check such as ``'2025' in text`` has no reachable
    failing state; this identity does, and a planted-drift control proves it.

Where the blend artifact itself is on disk (``artifacts/`` is gitignored), its recorded counts,
weights, gold digest and source ids are held equal to the same manifest -- so the document, the
manifest and the artifact are one answer.

WHICH SLOT (Plan 33.2-24 step 24b). The readout describes the blend the production swap
installs, which is now the step-24b re-fit (``P332_24B_*``): the same fit with the two 2024
Christmas games, once filed a week early in the owned timeline, back in the corpus. Was: every
assertion read ``P332_24_*``, which stays byte-unchanged as the record of the first fit, and
``TestTheFirstFitStaysOnTheRecord`` holds the document to naming it too.

STEP 25b. The models were re-fitted with the snap coverage flag left out and the blend re-tuned
on them, so the current-blend assertions read ``P332_25B_*`` (was ``P332_24B_*``, byte-unchanged
and still read by the first-fit class and for the corpus's named games, which did not change).

It asserts RULINGS and COUNT RELATIONSHIPS, never a re-derived point estimate.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
READOUT_MD = REPO_ROOT / "BLEND-TUNING-READOUT.md"
ARTIFACTS = REPO_ROOT / "artifacts"

#: The two exclusion classes, in the order the document publishes them.
EXCLUSION_CLASSES: tuple[str, ...] = ("no_prelock_line", "no_prior_fold_converter")

_REQUIRED_SECTION_MARKERS: tuple[str, ...] = (
    "## 1. What the blend does, in plain English",
    "## 2. The three fixed weights",
    "## 3. What the weights mean",
    "## 4. How the weights were fitted",
    "## 5. The exclusions",
    "## 6. The known gap: 2025",
    "## 7. What is NOT here",
)

_STATED_GAP = (
    "No free source has pre-lock lines for 2025, so the blend and the bets cannot be checked "
    "on 2025 at lock time."
)
_NOT_CLEAN_EVIDENCE_LABEL = "built on re-measured past seasons; not clean evidence"
_RETIRED_INCUMBENT_ID = "blend_dynamic_20260606_020635"

_EXCLUSION_ROW = re.compile(
    r"^\|\s*(?P<cls>[a-z_]+)\s*\|\s*(?P<games>[0-9,]+)\s*\|", re.MULTILINE
)
_WEIGHT_ROW = re.compile(
    r"^\|\s*(?P<target>WP|ATS|O/U)\s*\|\s*(?P<weight>[0-9]\.[0-9]{2})\s*\|",
    re.MULTILINE,
)
_TARGET_KEY = {"WP": "wp", "ATS": "ats", "O/U": "ou"}


def _read_readout() -> str:
    return READOUT_MD.read_text(encoding="utf-8")


def documented_excluded_count(text: str | None = None) -> dict[str, int] | None:
    """The ``{class: games}`` mapping the readout's exclusion table publishes.

    Only table rows whose first cell is one of :data:`EXCLUSION_CLASSES` are read, so prose
    mentioning a class name cannot satisfy it. Returns None when the document is absent.
    """
    if text is None:
        if not READOUT_MD.is_file():
            return None
        text = _read_readout()
    counts: dict[str, int] = {}
    for match in _EXCLUSION_ROW.finditer(text):
        if match.group("cls") in EXCLUSION_CLASSES:
            counts[match.group("cls")] = int(match.group("games").replace(",", ""))
    return counts


def state_manifest_excluded_count() -> dict[str, int] | None:
    """The ``{class: games}`` mapping ``tests/phase33_state.P332_25B_EXCLUDED_COUNTS`` records."""
    recorded = getattr(phase33_state, "P332_25B_EXCLUDED_COUNTS", None)
    if recorded is None:
        return None
    return {str(cls): int(games) for cls, games in recorded}


def documented_weights(text: str | None = None) -> dict[str, float]:
    """The ``{target: weight}`` mapping the readout's weights table publishes."""
    text = _read_readout() if text is None else text
    return {
        _TARGET_KEY[m.group("target")]: float(m.group("weight"))
        for m in _WEIGHT_ROW.finditer(text.split("## 3.", 1)[0])
    }


class TestTheReadoutExists:
    def test_it_is_at_the_repo_root(self) -> None:
        assert READOUT_MD.is_file(), f"missing: {READOUT_MD}"

    def test_it_is_ascii(self) -> None:
        assert _read_readout().isascii()

    def test_every_required_section_is_present(self) -> None:
        text = _read_readout()
        missing = [m for m in _REQUIRED_SECTION_MARKERS if m not in text]
        assert not missing, missing

    def test_it_says_proven_nowhere(self) -> None:
        assert "proven" not in _read_readout().lower()

    def test_it_describes_the_retired_blend_without_naming_its_id(self) -> None:
        text = _read_readout()
        assert _RETIRED_INCUMBENT_ID not in text
        assert "retired week-varying blend" in text

    def test_the_stated_gap_is_present(self) -> None:
        """Whitespace-normalized, so a Markdown line wrap inside the sentence is not a drop."""
        assert _STATED_GAP in " ".join(_read_readout().split())

    def test_it_carries_the_not_clean_evidence_label(self) -> None:
        assert _NOT_CLEAN_EVIDENCE_LABEL in _read_readout().lower()


class TestTheExclusionCountIdentity:
    """The readout's one checkable claim: its exclusion counts ARE the manifest's."""

    def test_the_document_publishes_both_classes(self) -> None:
        documented = documented_excluded_count()
        assert documented is not None
        assert set(documented) == set(EXCLUSION_CLASSES), documented

    def test_the_manifest_records_both_classes(self) -> None:
        recorded = state_manifest_excluded_count()
        assert recorded is not None
        assert set(recorded) == set(EXCLUSION_CLASSES), recorded

    def test_the_published_counts_equal_the_manifest(self) -> None:
        assert documented_excluded_count() == state_manifest_excluded_count()

    def test_a_drifted_count_fails_the_identity(self) -> None:
        """Planted control: the identity has a reachable failing state."""
        drifted = _read_readout().replace(
            "| no_prelock_line | 60 |", "| no_prelock_line | 61 |"
        )
        assert drifted != _read_readout(), "the planted edit did not apply"
        assert documented_excluded_count(drifted) != state_manifest_excluded_count()

    def test_a_dropped_class_fails_the_identity(self) -> None:
        dropped = re.sub(
            r"^\| no_prior_fold_converter \|.*$", "", _read_readout(), flags=re.M
        )
        assert documented_excluded_count(dropped) != state_manifest_excluded_count()

    def test_every_excluded_regular_season_game_is_named(self) -> None:
        text = _read_readout()
        for game_id in phase33_state.P332_24B_NO_PRELOCK_REGULAR_SEASON_GAMES:
            assert f"`{game_id}`" in text, game_id

    def test_every_unjoined_timeline_id_is_named(self) -> None:
        text = _read_readout()
        for game_id in phase33_state.P332_24B_UNJOINED_TIMELINE_IDS:
            assert f"`{game_id}`" in text, game_id


class TestThePublishedWeights:
    def test_the_weights_table_equals_the_manifest(self) -> None:
        assert documented_weights() == dict(phase33_state.P332_25B_BLEND_WEIGHTS)

    def test_the_boundary_weights_are_the_manifests(self) -> None:
        boundary = sorted(
            target
            for target, weight in phase33_state.P332_25B_BLEND_WEIGHTS
            if weight in (0.0, 1.0)
        )
        assert boundary == sorted(phase33_state.P332_25B_BOUNDARY_WEIGHT_TARGETS)

    def test_the_readout_names_the_artifact_the_manifest_records(self) -> None:
        assert f"`{phase33_state.P332_25B_BLEND_ARTIFACT_ID}`" in _read_readout()


@pytest.fixture(scope="module")
def blend_payload() -> dict:
    path = ARTIFACTS / phase33_state.P332_25B_BLEND_ARTIFACT_ID / "blend_weights.json"
    if not path.is_file():
        pytest.skip(
            f"evidence-backed skip: {path} is absent from this checkout (artifacts/ is "
            "gitignored), so there is no artifact to hold equal to the manifest"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class TestTheArtifactIsTheSameAnswer:
    def test_its_excluded_counts_equal_the_manifest(self, blend_payload: dict) -> None:
        assert blend_payload["excluded_counts"] == state_manifest_excluded_count()

    def test_its_weights_equal_the_manifest(self, blend_payload: dict) -> None:
        assert blend_payload["weights"] == dict(phase33_state.P332_25B_BLEND_WEIGHTS)

    def test_its_gold_digest_is_the_refit_generation(self, blend_payload: dict) -> None:
        assert (
            blend_payload["gold_generation_digest"]
            == phase33_state.P332_25B_BLEND_GOLD_GENERATION
            == phase33_state.P332_25B_REFIT_GOLD_GENERATION
        )

    def test_its_source_ids_are_the_recorded_refit(self, blend_payload: dict) -> None:
        assert blend_payload["source_artifact_ids"] == dict(
            phase33_state.P332_25B_BLEND_SOURCE_ARTIFACT_IDS
        )

    def test_its_corpus_and_converter_are_the_manifests(
        self, blend_payload: dict
    ) -> None:
        assert (
            blend_payload["tuning_corpus"]["rows"]
            == phase33_state.P332_25B_TUNING_CORPUS_ROWS
        )
        assert blend_payload["n_games"] == dict(phase33_state.P332_25B_TUNING_GAMES)
        assert (
            blend_payload["market_probability_artifact_id"]
            == phase33_state.P332_25B_BLEND_CONVERTER_ARTIFACT_ID
        )
        assert (
            blend_payload["thread_limit"] == phase33_state.P332_25B_BLEND_THREAD_LIMIT
        )

    def test_it_is_one_payload_file(self) -> None:
        directory = ARTIFACTS / phase33_state.P332_25B_BLEND_ARTIFACT_ID
        if not directory.is_dir():
            pytest.skip("the blend artifact is absent from this checkout")
        assert sorted(p.name for p in directory.iterdir()) == ["blend_weights.json"]


class TestTheFirstFitStaysOnTheRecord:
    """The first fit is superseded, never erased: the document names it and what moved."""

    def test_the_first_blend_is_named_as_superseded(self) -> None:
        text = _read_readout()
        assert f"`{phase33_state.P332_24_BLEND_ARTIFACT_ID}`" in text
        assert (
            phase33_state.P332_24_BLEND_ARTIFACT_ID
            != phase33_state.P332_24B_BLEND_ARTIFACT_ID
        )

    def test_the_corpus_grew_by_exactly_the_two_rekeyed_games(self) -> None:
        rekeyed = {
            scheduled for _, scheduled in phase33_state.P332_24B_ODDS_TIMELINE_REKEY_MAP
        }
        assert rekeyed <= set(phase33_state.P332_24_NO_PRELOCK_REGULAR_SEASON_GAMES)
        assert rekeyed.isdisjoint(
            phase33_state.P332_24B_NO_PRELOCK_REGULAR_SEASON_GAMES
        )
        grown = len(rekeyed)
        assert (
            phase33_state.P332_24_TUNING_CORPUS_ROWS + grown
            == phase33_state.P332_24B_TUNING_CORPUS_ROWS
        )
        before = dict(phase33_state.P332_24_EXCLUDED_COUNTS)
        after = dict(phase33_state.P332_24B_EXCLUDED_COUNTS)
        assert after["no_prelock_line"] == before["no_prelock_line"] - grown
        assert after["no_prior_fold_converter"] == before["no_prior_fold_converter"]
        games_before = dict(phase33_state.P332_24_TUNING_GAMES)
        games_after = dict(phase33_state.P332_24B_TUNING_GAMES)
        assert {t: games_after[t] - games_before[t] for t in games_after} == {
            "wp": grown,
            "ats": grown,
            "ou": grown,
        }

    def test_no_weight_moved(self) -> None:
        assert dict(phase33_state.P332_24B_BLEND_WEIGHTS) == dict(
            phase33_state.P332_24_BLEND_WEIGHTS
        )
