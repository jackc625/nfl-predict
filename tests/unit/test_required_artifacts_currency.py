"""An artifact that EXISTS is not an artifact that is CURRENT.

Phase 33, Plan 33-07 Task 2 (COLD-08, D33-30, T-33-33).

WHAT THIS MODULE PINS
---------------------
``pipeline.steps.step_verify_data_artifacts`` used to be six bare ``Path.exists()`` calls.
A gate built out of ``exists()`` cannot tell last season's matrix from this week's, so the
DATA/PREDICTIONS boundary -- the last point at which a stale input can be caught before it
becomes a published prediction -- passed on exactly the input the boundary exists to stop.

``_REQUIRED_ARTIFACTS`` is now ``(path, phase_boundary, coverage_check)`` triples and each
artifact is checked AT ITS OWN BOUNDARY:

* source and silver artifacts before feature building (``step_verify_data_artifacts``);
* the gold matrices after the gold build (``step_verify_gold_currency``);
* the prediction artifacts after prediction (``step_verify_prediction_currency``).

WHY THE SPLIT IS THE POINT AND NOT AN IMPLEMENTATION DETAIL
------------------------------------------------------------
Checking gold currency inside the DATA gate -- which runs BEFORE the step that builds gold
-- converts an ORDERING FACT into a stale-artifact refusal. A gate that cries wolf on a
correct run is worse than no gate, because the first thing an operator does with one is
stop reading it. ``test_the_data_gate_passes_when_no_current_week_gold_exists_at_all`` is
that false refusal, asserted directly rather than argued.

WEEK-AWARENESS IS A SEPARATE CLAIM AND IT IS ASSERTED SEPARATELY
------------------------------------------------------------------
The check asks whether the artifact carries rows for the CURRENT ``(season, week)`` and
asks nothing else. It must not require a full season: a week-2 run against a two-week-old
season is the normal case this phase exists to serve, and a completeness check would
refuse the live cold start on its second Friday.

NO TEST HERE TOUCHES A PRODUCTION STORE. Every case builds its own artifact tree under
``tmp_path`` and drives the boundary step with the process working directory moved onto it;
the steps resolve their paths relatively, which is what makes that redirection total.

Run this module:  uv run pytest tests/unit/test_required_artifacts_currency.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pipeline import steps

# The pinned "current" week every case in this module runs against. Week 2 rather than
# week 1 on purpose: it is the first week at which "the artifact carries SOME rows" and
# "the artifact carries THIS week's rows" can disagree.
_SEASON = 2026
_WEEK = 2

_SILVER_GAMES = "data/silver/games.parquet"
_SILVER_ELO = "data/silver/elo_game_snapshots.parquet"
_SILVER_TEAM_FORM = "data/silver/team_form_features.parquet"
_GOLD_MATRICES = (
    "data/gold/features_wp.parquet",
    "data/gold/features_ats.parquet",
    "data/gold/features_ou.parquet",
)


# ---------------------------------------------------------------------------
# Tree builders -- everything is written under tmp_path, never under data/
# ---------------------------------------------------------------------------


def _season_week_frame(pairs: tuple[tuple[int, int], ...]) -> pd.DataFrame:
    """A frame in the ``season`` / ``week`` shape the silver and gold artifacts use."""
    return pd.DataFrame(
        {
            "game_id": [f"{season}_{week:02d}_AAA_BBB" for season, week in pairs],
            "season": [season for season, _ in pairs],
            "week": [week for _, week in pairs],
        }
    )


def _target_season_week_frame(pairs: tuple[tuple[int, int], ...]) -> pd.DataFrame:
    """``team_form_features`` keys its week on ``target_season`` / ``target_week``.

    It is the reason the coverage check is a PER-ARTIFACT callable rather than one
    hardcoded pair of column names: a single spelling would silently answer False for this
    artifact on every week of every season, and a gate that always refuses is as useless as
    one that never does.
    """
    return pd.DataFrame(
        {
            "team": ["BUF" for _ in pairs],
            "target_season": [season for season, _ in pairs],
            "target_week": [week for _, week in pairs],
        }
    )


def _write(root: Path, relative: str, frame: pd.DataFrame) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _build_tree(
    root: Path,
    *,
    silver_pairs: tuple[tuple[int, int], ...] = ((_SEASON, 1), (_SEASON, _WEEK)),
    gold_pairs: tuple[tuple[int, int], ...] | None = ((_SEASON, _WEEK),),
    omit: tuple[str, ...] = (),
) -> None:
    """Write a complete artifact tree under *root*, minus anything named in *omit*."""
    if _SILVER_GAMES not in omit:
        _write(root, _SILVER_GAMES, _season_week_frame(silver_pairs))
    if _SILVER_ELO not in omit:
        _write(root, _SILVER_ELO, _season_week_frame(silver_pairs))
    if _SILVER_TEAM_FORM not in omit:
        _write(root, _SILVER_TEAM_FORM, _target_season_week_frame(silver_pairs))
    if gold_pairs is not None:
        for matrix in _GOLD_MATRICES:
            if matrix not in omit:
                _write(root, matrix, _season_week_frame(gold_pairs))


def _predictions_csv(root: Path, pairs: tuple[tuple[int, int], ...]) -> Path:
    path = root / "outputs" / "predictions" / f"predictions_{_SEASON}_week{_WEEK}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    _season_week_frame(pairs).to_csv(path, index=False)
    return path


@pytest.fixture
def pinned_week(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the resolver every boundary step reads its ``(season, week)`` from."""
    monkeypatch.setattr(
        "utils.date_utils.get_current_nfl_week", lambda: (_SEASON, _WEEK)
    )


def _capture(step: Any) -> BaseException | None:
    """Run *step* and return whatever it raised, or None when it passed.

    Returning the exception rather than asserting inside a ``pytest.raises`` block is what
    lets a test say "the gate accepted this" as its FIRST assertion. A ``pytest.raises``
    that does not fire reports "DID NOT RAISE <class>", which names the class before it
    names the behaviour.
    """
    try:
        step()
    except BaseException as raised:
        return raised
    return None


# ---------------------------------------------------------------------------
# The DATA boundary: missing, stale, and current
# ---------------------------------------------------------------------------


def test_the_data_gate_passes_when_every_silver_artifact_carries_the_current_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    """The positive control. Without it every refusal below could be a gate that always fires."""
    _build_tree(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert _capture(steps.step_verify_data_artifacts) is None


def test_the_data_gate_refuses_a_silver_artifact_that_exists_but_carries_no_current_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    """T-33-33: an artifact present on disk and a week out of date is the failure in scope."""
    _build_tree(tmp_path, silver_pairs=((_SEASON, 1),))
    monkeypatch.chdir(tmp_path)

    raised = _capture(steps.step_verify_data_artifacts)
    assert raised is not None, (
        "the DATA gate ACCEPTED three silver artifacts that exist but carry no row for "
        f"{_SEASON} week {_WEEK}; a bare exists() check cannot tell a current artifact "
        "from a stale one, which is the whole defect this task closes"
    )
    assert isinstance(raised, steps.StaleDataArtifactError), raised
    message = str(raised)
    assert str(_SEASON) in message and str(_WEEK) in message, message
    assert _SILVER_GAMES in message, message


def test_the_data_gate_reports_a_missing_artifact_with_the_missing_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    """A file that is not there is MISSING, and keeps the message it always had."""
    _build_tree(tmp_path, omit=(_SILVER_ELO,))
    monkeypatch.chdir(tmp_path)

    raised = _capture(steps.step_verify_data_artifacts)
    assert raised is not None, (
        "the DATA gate accepted a tree with no elo snapshot at all"
    )
    assert "Missing data artifacts" in str(raised), raised
    assert _SILVER_ELO in str(raised), raised


def test_missing_and_stale_are_two_distinct_classes_with_two_distinct_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    """A missing artifact must never be reported as stale, nor a stale one as missing.

    They call for different actions -- run the ingest, versus find out why the ingest ran
    and produced nothing for this week -- so an operator who cannot tell them apart from
    the message is being sent to the wrong place.
    """
    missing_root = tmp_path / "missing"
    stale_root = tmp_path / "stale"
    missing_root.mkdir()
    stale_root.mkdir()
    _build_tree(missing_root, omit=(_SILVER_GAMES,))
    _build_tree(stale_root, silver_pairs=((_SEASON, 1),))

    monkeypatch.chdir(missing_root)
    missing = _capture(steps.step_verify_data_artifacts)
    monkeypatch.chdir(stale_root)
    stale = _capture(steps.step_verify_data_artifacts)

    assert missing is not None and stale is not None
    assert not isinstance(missing, steps.StaleDataArtifactError), (
        "an ABSENT artifact was reported as STALE; the two failures need different fixes"
    )
    assert isinstance(stale, steps.StaleDataArtifactError), stale
    assert "Missing data artifacts" not in str(stale), stale
    assert "Stale" not in str(missing), missing


# ---------------------------------------------------------------------------
# The split itself: the DATA gate does not see gold
# ---------------------------------------------------------------------------


def test_the_data_gate_passes_when_no_current_week_gold_exists_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    """THE FALSE REFUSAL THE SPLIT EXISTS TO PREVENT, asserted directly.

    ``build_features`` runs in the PREDICTIONS phase, so at the DATA gate the current
    week's gold has not been built yet and CANNOT have been. A currency check on gold
    inside this gate would refuse every correct full-mode run at step 8.
    """
    _build_tree(tmp_path, gold_pairs=None)
    monkeypatch.chdir(tmp_path)

    assert _capture(steps.step_verify_data_artifacts) is None, (
        "the DATA gate refused a tree whose gold has not been built yet -- but the step "
        "that builds gold runs AFTER this gate, so this is an ordering fact being "
        "reported as a stale artifact"
    )


def test_the_data_gate_reports_only_the_data_boundary_artifacts_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    """Structural: the set of paths the DATA gate can report is the silver three, exactly.

    Driven by deleting the WHOLE tree and reading which paths the refusal names, so the
    claim is about the gate's own scope rather than about a list restated here.
    """
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)

    raised = _capture(steps.step_verify_data_artifacts)
    assert raised is not None, "an empty tree passed the DATA gate"
    reported = {
        path
        for path in (*_GOLD_MATRICES, _SILVER_GAMES, _SILVER_ELO, _SILVER_TEAM_FORM)
        if path in str(raised)
    }
    assert reported == {_SILVER_GAMES, _SILVER_ELO, _SILVER_TEAM_FORM}, sorted(reported)


def test_every_required_artifact_declares_a_boundary_from_the_closed_vocabulary() -> (
    None
):
    """A triple whose boundary is a typo would be checked at NO boundary and never run."""
    unknown = [
        (path, boundary)
        for path, boundary, _check in steps._REQUIRED_ARTIFACTS
        if boundary not in steps.ARTIFACT_PHASE_BOUNDARIES
    ]
    assert not unknown, unknown
    assert len({boundary for _p, boundary, _c in steps._REQUIRED_ARTIFACTS}) > 1, (
        "every artifact still declares the SAME boundary, so the split did not happen "
        "and the DATA gate is still checking downstream artifacts"
    )


# ---------------------------------------------------------------------------
# Week-awareness: coverage of THIS week, never season completeness
# ---------------------------------------------------------------------------


def test_a_two_week_old_season_passes_for_a_week_it_carries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    """The live cold start's second Friday: two weeks of rows, and that is CORRECT."""
    _build_tree(tmp_path, silver_pairs=((_SEASON, 1), (_SEASON, _WEEK)))
    monkeypatch.chdir(tmp_path)

    assert _capture(steps.step_verify_data_artifacts) is None, (
        "a two-week-old season was refused; the check asks for coverage of the CURRENT "
        "week and must never require a completed season"
    )


def test_the_same_two_week_season_fails_only_when_the_current_week_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same artifacts, one week later. The verdict flips, and nothing else changed."""
    _build_tree(tmp_path, silver_pairs=((_SEASON, 1), (_SEASON, _WEEK)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "utils.date_utils.get_current_nfl_week", lambda: (_SEASON, _WEEK + 1)
    )

    raised = _capture(steps.step_verify_data_artifacts)
    assert raised is not None, (
        "week 3 was accepted against artifacts whose newest row is week 2"
    )
    assert isinstance(raised, steps.StaleDataArtifactError), raised


def test_the_coverage_check_imposes_no_season_completeness_requirement(
    tmp_path: Path,
) -> None:
    """Asserted on the checker itself, not only through a step that calls it."""
    path = _write(
        tmp_path, _SILVER_GAMES, _season_week_frame(((_SEASON, 1), (_SEASON, 2)))
    )

    assert steps._covers_season_week(str(path), _SEASON, 2) is True
    assert steps._covers_season_week(str(path), _SEASON, 1) is True
    assert steps._covers_season_week(str(path), _SEASON, 3) is False
    assert steps._covers_season_week(str(path), _SEASON - 1, 1) is False


def test_the_team_form_check_reads_its_own_week_columns(tmp_path: Path) -> None:
    """``team_form_features`` has no ``week`` column at all; it has ``target_week``."""
    path = _write(
        tmp_path, _SILVER_TEAM_FORM, _target_season_week_frame(((_SEASON, _WEEK),))
    )

    assert steps._covers_target_season_week(str(path), _SEASON, _WEEK) is True
    assert steps._covers_target_season_week(str(path), _SEASON, _WEEK + 1) is False
    assert steps._covers_season_week(str(path), _SEASON, _WEEK) is False


# ---------------------------------------------------------------------------
# The GOLD boundary
# ---------------------------------------------------------------------------


def test_the_gold_currency_step_passes_when_gold_carries_the_current_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    _build_tree(tmp_path)
    monkeypatch.chdir(tmp_path)

    gate = getattr(steps, "step_verify_gold_currency", None)
    assert gate is not None, "pipeline.steps defines no step_verify_gold_currency"
    assert _capture(gate) is None


def test_the_gold_currency_step_refuses_gold_that_carries_no_current_week_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    """R9's refusal, one step earlier and louder than the selection tier can manage.

    ``build_weekly_candidates`` raises the same fact at selection time. This one fires at
    the first point in the run where gold exists at all, so the operator learns it before
    the bet list is even attempted -- and the selection-tier error stays in place as the
    backstop.
    """
    _build_tree(tmp_path, gold_pairs=((_SEASON, 1),))
    monkeypatch.chdir(tmp_path)

    gate = getattr(steps, "step_verify_gold_currency", None)
    assert gate is not None, "pipeline.steps defines no step_verify_gold_currency"
    raised = _capture(gate)
    assert raised is not None, (
        "the gold-currency gate accepted three matrices carrying no row for "
        f"{_SEASON} week {_WEEK}"
    )
    assert isinstance(raised, steps.StaleDataArtifactError), raised
    assert "features_wp.parquet" in str(raised), raised


def test_the_gold_currency_step_reports_absent_gold_as_missing_not_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    _build_tree(tmp_path, omit=("data/gold/features_ou.parquet",))
    monkeypatch.chdir(tmp_path)

    gate = getattr(steps, "step_verify_gold_currency", None)
    assert gate is not None, "pipeline.steps defines no step_verify_gold_currency"
    raised = _capture(gate)
    assert raised is not None, "absent gold passed the gold-currency gate"
    assert "Missing data artifacts" in str(raised), raised
    assert "features_ou.parquet" in str(raised), raised


# ---------------------------------------------------------------------------
# The PREDICTIONS boundary
# ---------------------------------------------------------------------------


def test_the_prediction_currency_step_passes_on_a_current_week_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    _predictions_csv(tmp_path, ((_SEASON, _WEEK),))
    monkeypatch.chdir(tmp_path)

    gate = getattr(steps, "step_verify_prediction_currency", None)
    assert gate is not None, "pipeline.steps defines no step_verify_prediction_currency"
    assert _capture(gate) is None


def test_the_prediction_currency_step_refuses_a_file_whose_rows_are_another_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    """The filename says week 2 and the rows say week 1. The rows are the fact."""
    _predictions_csv(tmp_path, ((_SEASON, 1),))
    monkeypatch.chdir(tmp_path)

    gate = getattr(steps, "step_verify_prediction_currency", None)
    assert gate is not None, "pipeline.steps defines no step_verify_prediction_currency"
    raised = _capture(gate)
    assert raised is not None, (
        "a prediction file named for week 2 and holding week 1 rows was accepted"
    )
    assert isinstance(raised, steps.StaleDataArtifactError), raised


def test_the_prediction_currency_step_reports_an_absent_file_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    monkeypatch.chdir(tmp_path)

    gate = getattr(steps, "step_verify_prediction_currency", None)
    assert gate is not None, "pipeline.steps defines no step_verify_prediction_currency"
    raised = _capture(gate)
    assert raised is not None, "an absent prediction file passed the prediction gate"
    assert "Missing data artifacts" in str(raised), raised
