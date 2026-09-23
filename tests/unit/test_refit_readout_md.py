"""Permanent doc-drift guard for the repo-root REFIT-READOUT.md (Plan 33.2-25 Task 2, SPEC R13).

A PERMANENT committed test on the ``tests/unit/test_signal_lift_readout_md.py`` pattern. It
guards the published record of the corrected models and blend against these failure modes:

  - the file is missing, not at the repo root, non-ASCII, or has lost a required section;
  - a RESULTS section lost the not-clean-evidence label. Checked PER SECTION, never once per
    file, and results sections are found STRUCTURALLY (any section carrying a table row with a
    decimal number), so a results section added later without the label fails too (D33.2-07);
  - a pre-correction comparator entered the record. Checked STRUCTURALLY, not by the absence
    of the string ``gate.toml``: the document is REQUIRED to say that no ``config/gate.toml``
    baseline appears in it, so a check on that string would make the mandated sentence and a
    passing test mutually exclusive. What binds is (a) no pre-swap pointer value anywhere --
    the values are read from the state manifest, never retyped here -- (b) the timestamped
    artifact ids the document names are EXACTLY the four the swap installs, and (c) no
    ``[baseline.*]`` section key anywhere;
  - a published number from a fold that saw its own season: for EVERY fold row the latest
    season any part of the fit touched is strictly earlier than the season predicted, every
    published results season has a fold, and the fold count is non-zero;
  - the document drifting from the record: the four ids equal the state manifest's, and where
    the artifacts are on disk (``artifacts/`` is gitignored) every published per-season figure,
    the fold windows, the feature counts and the gold generation equal what the artifacts
    recorded.

It asserts RECORDED values and RULINGS, never a re-derived point estimate -- pinning a number
to moving gold is a mistake this repo has already made (D29-06-02).

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
READOUT_MD = REPO_ROOT / "REFIT-READOUT.md"
ARTIFACTS = REPO_ROOT / "artifacts"

#: The label every results section carries, compared whitespace-normalized so a Markdown line
#: wrap is not a drop.
NOT_CLEAN_EVIDENCE_LABEL = (
    "**Not clean evidence.** These are re-measured past seasons, built under the new rule on "
    "corrected inputs. Only the 2026 season, recorded live, counts (D33.2-07). No number here "
    "is evidence of accuracy or profitability."
)

#: The mandated statement that no pre-correction comparator appears (whitespace-normalized).
NO_COMPARATOR_STATEMENT = (
    "No pre-correction artifact id appears anywhere in this document, and no "
    "`config/gate.toml` baseline appears in it either"
)

REQUIRED_SECTION_MARKERS: tuple[str, ...] = (
    "## 1. What was re-fitted, on what, and why nothing is compared with the old models",
    "## 2. The four artifacts",
    "## 3. The folds",
    "## 4. Results: out-of-sample accuracy, per target",
    "## 5. Results: the corrected models against the market's pre-lock opinion",
    "## 6. Results: the random-baseline margin",
    "## 7. What the models lost when the betting lines left",
    "## 8. The feature sets",
    "## 9. What is NOT here, and why",
    "## 10. What happens next",
)

#: The sections that publish results -- each must be found by the structural rule too.
REQUIRED_RESULTS_SECTIONS: tuple[str, ...] = (
    "## 4. Results",
    "## 5. Results",
    "## 6. Results",
    "## 7. What the models lost",
)

_TIMESTAMPED_ID = re.compile(r"\b(?:wp|ats|ou|blend)_\d{8}_\d{6}\b")
_DECIMAL_TABLE_ROW = re.compile(r"^\|.*\d\.\d", re.MULTILINE)
_FOLD_ROW = re.compile(
    r"^\|\s*(?P<kind>holdout|blend)-(?P<season>\d{4})\s*\|(?P<rest>.*)$"
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_SEASON_ROW = re.compile(r"^\|\s*(?P<season>\d{4})\s*\|(?P<rest>.*)$")
_TARGET_LABEL = {"WP": "wp", "ATS": "ats", "O/U": "ou"}


def _read() -> str:
    return READOUT_MD.read_text(encoding="utf-8")


def _normalized(text: str) -> str:
    return " ".join(text.split())


def split_sections(text: str) -> dict[str, str]:
    """``{level-2 heading line: body}`` in document order."""
    sections: dict[str, str] = {}
    heading: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if heading is not None:
                sections[heading] = "\n".join(lines)
            heading, lines = line, []
        elif heading is not None:
            lines.append(line)
    if heading is not None:
        sections[heading] = "\n".join(lines)
    return sections


def results_sections(text: str) -> dict[str, str]:
    """Every section carrying a table row with a decimal number -- a published result."""
    return {
        heading: body
        for heading, body in split_sections(text).items()
        if _DECIMAL_TABLE_ROW.search(body)
    }


def unlabelled_results_sections(text: str) -> list[str]:
    """The results sections that do NOT carry the not-clean-evidence label."""
    label = _normalized(NOT_CLEAN_EVIDENCE_LABEL)
    return [
        heading
        for heading, body in results_sections(text).items()
        if label not in _normalized(body)
    ]


def documented_folds(text: str) -> list[tuple[str, int, list[int]]]:
    """``(fold, predicted season, every season any part of the fit touched)`` per fold row."""
    folds: list[tuple[str, int, list[int]]] = []
    for line in text.splitlines():
        match = _FOLD_ROW.match(line)
        if match is None:
            continue
        cells = [cell.strip() for cell in match.group("rest").split("|")]
        # cells[0] is the test season; cells[1:] are the fit windows.
        fit_years = [int(year) for cell in cells[1:] for year in _span_years(cell)]
        folds.append(
            (f"{match.group('kind')}-{match.group('season')}", int(cells[0]), fit_years)
        )
    return folds


def _span_years(cell: str) -> list[int]:
    """Every season a cell names, expanding ``2002-2023`` into its whole span."""
    found = [int(year) for year in _YEAR.findall(cell)]
    if re.fullmatch(r"\s*\d{4}-\d{4}\s*", cell) and len(found) == 2:
        return list(range(found[0], found[1] + 1))
    return found


def fold_violations(folds: list[tuple[str, int, list[int]]]) -> list[str]:
    """The folds whose fit touched the season it predicts, or a later one."""
    return [
        f"{name}: predicted {season}, but its fit touched {max(years)}"
        for name, season, years in folds
        if years and max(years) >= season
    ]


def documented_season_rows(text: str) -> dict[str, dict[int, list[float]]]:
    """Section 4's per-target tables: ``{target: {season: [numbers after the season]}}``."""
    body = split_sections(text)[_heading_starting(text, "## 4. Results")]
    blocks = {
        "wp": body.split("**Win probability**", 1)[1].split("**Spread**", 1)[0],
        "ats": body.split("**Spread**", 1)[1].split("**Total**", 1)[0],
        "ou": body.split("**Total**", 1)[1],
    }
    parsed: dict[str, dict[int, list[float]]] = {}
    for target, block in blocks.items():
        rows: dict[int, list[float]] = {}
        for line in block.splitlines():
            match = _SEASON_ROW.match(line)
            if match is not None:
                cells = [c.strip() for c in match.group("rest").split("|") if c.strip()]
                rows[int(match.group("season"))] = [float(c) for c in cells]
        parsed[target] = rows
    return parsed


def _heading_starting(text: str, prefix: str) -> str:
    matches = [h for h in split_sections(text) if h.startswith(prefix)]
    assert len(matches) == 1, (prefix, matches)
    return matches[0]


def _target_rows(text: str, section_prefix: str) -> dict[str, list[str]]:
    """``{target: cells after the target}`` for the table rows of one section."""
    body = split_sections(text)[_heading_starting(text, section_prefix)]
    rows: dict[str, list[str]] = {}
    for line in body.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells and cells[0] in _TARGET_LABEL:
            rows[_TARGET_LABEL[cells[0]]] = cells[1:]
    return rows


# ---------------------------------------------------------------------------
# The document exists and says what it must
# ---------------------------------------------------------------------------


class TestTheReadoutExists:
    def test_it_is_at_the_repo_root(self) -> None:
        assert READOUT_MD.is_file(), f"missing: {READOUT_MD}"

    def test_it_is_ascii(self) -> None:
        assert _read().isascii()

    def test_every_required_section_is_present(self) -> None:
        text = _read()
        missing = [m for m in REQUIRED_SECTION_MARKERS if m not in text]
        assert not missing, missing

    def test_it_says_proven_nowhere(self) -> None:
        assert "proven" not in _read().lower()

    def test_it_carries_no_over_claim_word(self) -> None:
        from backtest.ev_chain_constants import READOUT_FORBIDDEN_WORDS

        assert READOUT_FORBIDDEN_WORDS, "the imported over-claim dictionary is empty"
        lowered = _read().lower()
        assert [w for w in READOUT_FORBIDDEN_WORDS if w in lowered] == []

    def test_the_label_opens_the_document(self) -> None:
        """Before the first section, so no reader meets a number unlabelled."""
        preamble = _read().split("\n## ", 1)[0]
        assert _normalized(NOT_CLEAN_EVIDENCE_LABEL) in _normalized(preamble)


# ---------------------------------------------------------------------------
# Every results section carries the label -- per section, found structurally
# ---------------------------------------------------------------------------


class TestEveryResultsSectionIsLabelled:
    def test_the_structural_rule_finds_the_results_sections(self) -> None:
        """Non-vacuity: the rule finds every required results section, and at least four."""
        found = results_sections(_read())
        assert len(found) >= len(REQUIRED_RESULTS_SECTIONS)
        for prefix in REQUIRED_RESULTS_SECTIONS:
            assert any(h.startswith(prefix) for h in found), prefix

    def test_every_results_section_carries_the_label(self) -> None:
        assert unlabelled_results_sections(_read()) == []

    def test_a_results_section_stripped_of_its_label_is_caught(self) -> None:
        """Planted control: removing ONE section's label is a failure, not a file-level pass."""
        text = _read()
        heading = _heading_starting(text, "## 6. Results")
        body = split_sections(text)[heading]
        stripped_body = body.replace("**Not clean evidence.**", "**Label removed.**", 1)
        assert stripped_body != body, "the planted edit did not apply"
        planted = text.replace(body, stripped_body, 1)
        assert unlabelled_results_sections(planted) == [heading]


# ---------------------------------------------------------------------------
# No pre-correction comparator -- structurally
# ---------------------------------------------------------------------------


class TestNoPreCorrectionComparator:
    def test_no_pre_swap_pointer_value_appears(self) -> None:
        text = _read()
        present = [
            artifact_id
            for _key, artifact_id in phase33_state.P332_25_PRE_SWAP_LATEST_JSON
            if artifact_id in text
        ]
        assert present == []

    def test_the_timestamped_ids_are_exactly_the_four_installed(self) -> None:
        documented = set(_TIMESTAMPED_ID.findall(_read()))
        installed = {
            artifact_id for _key, artifact_id in phase33_state.P332_25_SWAP_ARTIFACT_IDS
        }
        assert documented == installed

    def test_no_gate_baseline_section_key_appears(self) -> None:
        assert re.findall(r"\[baseline\.[A-Za-z0-9_]+\]", _read()) == []

    def test_the_mandated_statement_is_present(self) -> None:
        assert _normalized(NO_COMPARATOR_STATEMENT) in _normalized(_read())

    def test_a_planted_old_id_in_a_results_row_is_caught(self) -> None:
        """Planted control: the id scan has a reachable failing state."""
        old_wp = dict(phase33_state.P332_25_PRE_SWAP_LATEST_JSON)["wp"]
        planted = _read().replace("| 2024 | 285 |", f"| 2024 | 285 | {old_wp} |", 1)
        assert planted != _read(), "the planted edit did not apply"
        installed = {v for _k, v in phase33_state.P332_25_SWAP_ARTIFACT_IDS}
        assert set(_TIMESTAMPED_ID.findall(planted)) != installed


# ---------------------------------------------------------------------------
# The four ids are the state manifest's
# ---------------------------------------------------------------------------


class TestTheFourIdsAreTheRecordedOnes:
    def test_the_swap_slot_is_the_refit_plus_the_step_24b_blend(self) -> None:
        assert dict(phase33_state.P332_25_SWAP_ARTIFACT_IDS) == {
            **dict(phase33_state.P332_23_REFIT_ARTIFACT_IDS),
            "blend": phase33_state.P332_24B_BLEND_ARTIFACT_ID,
        }

    def test_the_artifact_table_maps_each_key_to_its_recorded_id(self) -> None:
        text = _read()
        body = split_sections(text)[_heading_starting(text, "## 2. The four artifacts")]
        table: dict[str, str] = {}
        for line in body.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.startswith("|") and cells[0] in ("wp", "ats", "ou", "blend"):
                table[cells[0]] = cells[1].strip("`")
        assert table == dict(phase33_state.P332_25_SWAP_ARTIFACT_IDS)

    def test_it_names_the_recorded_gold_generation(self) -> None:
        assert phase33_state.P332_25_SWAP_GOLD_GENERATION in _read()
        assert (
            phase33_state.P332_25_SWAP_GOLD_GENERATION
            == phase33_state.P332_23_REFIT_GOLD_GENERATION
            == phase33_state.P332_24B_BLEND_GOLD_GENERATION
        )


class TestThePreSwapRecordIsReversible:
    """The recorded pre-swap manifest is internally one answer, so a hand restore is exact."""

    def test_the_text_digests_to_the_recorded_sha256(self) -> None:
        text = phase33_state.P332_25_PRE_SWAP_LATEST_JSON_TEXT
        assert (
            hashlib.sha256(text.encode("ascii")).hexdigest()
            == phase33_state.P332_25_PRE_SWAP_LATEST_JSON_SHA256
        )

    def test_the_text_parses_to_the_recorded_pointers(self) -> None:
        parsed = json.loads(phase33_state.P332_25_PRE_SWAP_LATEST_JSON_TEXT)
        assert list(parsed.items()) == list(phase33_state.P332_25_PRE_SWAP_LATEST_JSON)

    def test_the_swap_moves_every_pointer_and_no_key(self) -> None:
        before = dict(phase33_state.P332_25_PRE_SWAP_LATEST_JSON)
        after = dict(phase33_state.P332_25_SWAP_ARTIFACT_IDS)
        assert list(before) == list(after) == ["wp", "ats", "ou", "blend"]
        assert all(before[key] != after[key] for key in before)


# ---------------------------------------------------------------------------
# Every published number comes from a fold whose training seasons all precede it
# ---------------------------------------------------------------------------


class TestTheFoldOrdering:
    def test_the_fold_count_is_non_zero(self) -> None:
        """Non-vacuity: an ordering check over zero folds proves nothing."""
        folds = documented_folds(_read())
        assert len(folds) >= 2
        assert all(years for _name, _season, years in folds)

    def test_every_fold_fits_only_on_earlier_seasons(self) -> None:
        assert fold_violations(documented_folds(_read())) == []

    def test_a_fold_that_saw_its_own_season_is_caught(self) -> None:
        """Planted control: the ordering check has a reachable failing state."""
        planted = _read().replace(
            "| holdout-2024 | 2024 | 2002-2023 |",
            "| holdout-2024 | 2024 | 2002-2024 |",
            1,
        )
        assert planted != _read(), "the planted edit did not apply"
        assert fold_violations(documented_folds(planted)) != []

    def test_every_published_holdout_season_has_a_holdout_fold(self) -> None:
        text = _read()
        holdout = {
            season
            for name, season, _years in documented_folds(text)
            if name.startswith("holdout-")
        }
        published = {
            season for rows in documented_season_rows(text).values() for season in rows
        }
        assert published, "section 4 publishes no per-season row"
        assert published <= holdout

    def test_every_blend_season_has_a_blend_fold(self) -> None:
        text = _read()
        blend = {
            season
            for name, season, _years in documented_folds(text)
            if name.startswith("blend-")
        }
        for cells in _target_rows(text, "## 5. Results").values():
            for season in _span_years(cells[-1]):
                assert season in blend, season


# ---------------------------------------------------------------------------
# The published rulings equal the state manifest
# ---------------------------------------------------------------------------


class TestThePublishedRulingsAreTheRecord:
    def test_the_blend_weights_are_the_step_24b_record(self) -> None:
        rows = _target_rows(_read(), "## 5. Results")
        assert {t: float(cells[0]) for t, cells in rows.items()} == dict(
            phase33_state.P332_24B_BLEND_WEIGHTS
        )

    def test_the_margin_verdicts_are_the_plan_23_record(self) -> None:
        rows = _target_rows(_read(), "## 6. Results")
        recorded = {
            target: (cleared, bar, arm)
            for target, cleared, _gap, bar, arm in phase33_state.P332_23_MARGIN_VERDICTS
        }
        assert set(rows) == set(recorded)
        # cells: metric, random score, searched score, gap, bar, cleared, settings shipped.
        for target, cells in rows.items():
            cleared, bar, arm = recorded[target]
            assert float(cells[4]) == bar, target
            assert cells[5] == ("yes" if cleared else "no"), target
            assert arm in cells[6], target

    def test_the_feature_counts_are_the_plan_23_record(self) -> None:
        rows = _target_rows(_read(), "## 8. The feature sets")
        assert {t: int(cells[0]) for t, cells in rows.items()} == dict(
            phase33_state.P332_23_REFIT_FEATURE_COUNTS
        )
        assert {t: int(cells[1]) for t, cells in rows.items()} == {
            "wp": 0,
            "ats": 0,
            "ou": 0,
        }


# ---------------------------------------------------------------------------
# The document and the artifacts are one answer (skipped where artifacts/ is absent)
# ---------------------------------------------------------------------------


def _metadata(target: str) -> dict:
    artifact_id = dict(phase33_state.P332_25_SWAP_ARTIFACT_IDS)[target]
    path = ARTIFACTS / artifact_id / "metadata.json"
    if not path.is_file():
        pytest.skip(
            f"evidence-backed skip: {path} is absent from this checkout (artifacts/ is "
            "gitignored), so there is no artifact to hold the document equal to"
        )
    return json.loads(path.read_text(encoding="utf-8"))


#: Section 4's column order per target, as the artifacts record the same quantities.
_SEASON_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "wp": ("n_games", "accuracy", "mae"),
    "ats": ("n_games", "mae", "rmse", "r2"),
    "ou": ("n_games", "mae", "rmse", "r2"),
}


class TestTheArtifactsAreTheSameAnswer:
    @pytest.mark.parametrize("target", ["wp", "ats", "ou"])
    def test_every_published_season_figure_is_the_recorded_one(
        self, target: str
    ) -> None:
        metadata = _metadata(target)
        documented = documented_season_rows(_read())[target]
        recorded = {row["season"]: row for row in metadata["season_results"]}
        assert set(documented) == set(recorded)
        for season, values in documented.items():
            expected = [
                round(float(recorded[season][key]), 6)
                for key in _SEASON_METRIC_KEYS[target]
            ]
            assert values == expected, (target, season)

    @pytest.mark.parametrize("target", ["wp", "ats", "ou"])
    def test_the_holdout_folds_are_the_recorded_windows(self, target: str) -> None:
        config = _metadata(target)["config"]
        holdout_rows: dict[int, list[str]] = {}
        for line in _read().splitlines():
            match = _FOLD_ROW.match(line)
            if match is not None and match.group("kind") == "holdout":
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                holdout_rows[int(match.group("season"))] = cells
        assert sorted(holdout_rows) == config["holdout_seasons"]
        train = config["train_seasons"]
        # cells: fold, test season, model fitted on, features chosen on, calibration fitted on.
        for season, cells in holdout_rows.items():
            # models.temporal.WalkForwardSplitter.generate_splits: season < holdout season.
            assert _span_years(cells[2]) == list(range(train[0], season))
            assert _span_years(cells[3]) == train
            assert _span_years(cells[4]) == config["hp_val_seasons"]

    @pytest.mark.parametrize("target", ["wp", "ats", "ou"])
    def test_each_model_records_the_swap_gold(self, target: str) -> None:
        assert (
            _metadata(target)["gold_generation_digest"]
            == phase33_state.P332_25_SWAP_GOLD_GENERATION
        )

    def test_the_blend_records_the_swap_gold(self) -> None:
        blend_id = dict(phase33_state.P332_25_SWAP_ARTIFACT_IDS)["blend"]
        path = ARTIFACTS / blend_id / "blend_weights.json"
        if not path.is_file():
            pytest.skip(f"evidence-backed skip: {path} is absent from this checkout")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert (
            payload["gold_generation_digest"]
            == phase33_state.P332_25_SWAP_GOLD_GENERATION
        )

    def test_the_wp_calibration_error_is_the_recorded_one(self) -> None:
        ece = round(float(_metadata("wp")["ece"]), 6)
        assert f"is {ece:.6f}." in _normalized(_read())
