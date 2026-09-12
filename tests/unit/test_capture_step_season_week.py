"""The live-season capture is a step the Friday run executes, on the run's own week.

Phase 33, Plan 33-07 Task 3 (COLD-08, T-33-35, T-33-36; discharges D32-04).

WHAT PHASE 32 LEFT OPEN, DELIBERATELY, AND THIS CLOSES
--------------------------------------------------------
``scripts/capture_live_season.py`` shipped as a standalone, fully tested CLI whose own
module docstring says it is "NOT WIRED INTO THE PIPELINE, DELIBERATELY (D32-04)" and names
Phase 33 as the phase that wires it. An unwired capture is a capture nobody runs, and the
obligation was carried forward explicitly rather than assumed. It is now the FIRST step of
the DATA phase.

IT CALLS ``run_capture``, NEVER ``main`` (Codex HIGH, verified against live source)
-------------------------------------------------------------------------------------
``main(argv=None)`` parses argv and, when ``args.week is None``, prints
"ERROR: --week is required for a capture..." and returns ``EXIT_USAGE``. A zero-argument
pipeline step calling ``main()`` therefore CANNOT capture -- it returns a usage error every
single Friday, and a step that only checked for an exception would read that as success:
the exact "reports success while doing nothing" shape ``step_build_elo`` was repaired for
in Plan 33-03. ``run_capture(datasets, season, week, *, data_root, manifest_dir, ...)`` is
the directly callable seam, and ``test_the_step_never_reaches_main`` asserts by AST that
the name ``main`` appears nowhere in the step's body.

ONE WEEK RESOLUTION, NOT TWO
------------------------------
The capture must be recorded against the SAME ``(season, week)`` the rest of the DATA phase
ingests. Two independent resolutions can disagree across a midnight boundary, and a capture
labelled with a different week than the one ingested is worse than no capture: it looks
like evidence and is not. Both steps read ``pipeline.steps._resolve_current_week``, and the
test below drives them from ONE pinned resolver and compares what each actually received.

NOTHING HERE TOUCHES THE NETWORK OR THE LIVE ZONE. ``run_capture`` is replaced by a
recorder in every case, so no dataset is fetched, no bronze snapshot is written and
``config/upstream_live/2026.json`` is never appended to.

Run this module:  uv run pytest tests/unit/test_capture_step_season_week.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from data.upstream_pin import DATASET_COLUMNS, ZoneWriteRefused, default_data_root
from pipeline import steps
from scripts import capture_live_season

_SEASON = 2026
_WEEK = 2


class _CaptureRecorder:
    """Stands in for ``run_capture`` and records exactly what the step handed it."""

    def __init__(self, code: int = 0) -> None:
        self.code = code
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> int:
        self.calls.append((args, kwargs))
        return self.code

    def value(self, name: str, index: int) -> Any:
        """One argument of the single recorded call, positional or keyword.

        ``run_capture``'s first three parameters are positional-or-keyword, so reading both
        forms keeps these assertions about the VALUES the step passed rather than about the
        calling convention it happened to choose.
        """
        assert len(self.calls) == 1, f"expected one capture call, got {len(self.calls)}"
        args, kwargs = self.calls[0]
        return kwargs[name] if name in kwargs else args[index]

    @property
    def season_and_week(self) -> tuple[int, int]:
        return int(self.value("season", 1)), int(self.value("week", 2))


class _IngesterRecorder:
    """Stands in for ``GameDataIngester`` and records the seasons it was asked for."""

    seasons: list[list[int] | None] = []

    def __init__(self) -> None:
        pass

    def ingest_games(
        self,
        seasons: list[int] | None = None,
        weeks: list[int] | None = None,
        include_results: bool = True,
    ) -> None:
        type(self).seasons.append(seasons)


@pytest.fixture
def pinned_week(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.date_utils.get_current_nfl_week", lambda: (_SEASON, _WEEK)
    )


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _CaptureRecorder:
    stub = _CaptureRecorder()
    monkeypatch.setattr(capture_live_season, "run_capture", stub)
    return stub


def _step() -> Any:
    step = getattr(steps, "step_capture_live_season", None)
    assert step is not None, (
        "pipeline.steps defines no step_capture_live_season; the Phase-32 capture is "
        "still unwired and D32-04 is still open"
    )
    return step


# ---------------------------------------------------------------------------
# It calls the right seam
# ---------------------------------------------------------------------------


def test_a_successful_capture_returns_without_raising(
    pinned_week: None, recorder: _CaptureRecorder
) -> None:
    """The positive control: exit code 0 is a capture that happened, and the step is done."""
    _step()()
    assert len(recorder.calls) == 1


def test_the_step_hands_run_capture_the_resolved_season_and_week(
    pinned_week: None, recorder: _CaptureRecorder
) -> None:
    """A capture with no explicit week cannot capture at all; these two must arrive."""
    _step()()
    assert recorder.season_and_week == (_SEASON, _WEEK)


def test_the_step_computes_datasets_and_data_root_the_way_the_cli_does(
    pinned_week: None, recorder: _CaptureRecorder
) -> None:
    """The two derived arguments are READ off the CLI's own construction, not invented.

    ``main`` computes ``sorted(DATASET_COLUMNS)`` and ``default_data_root()``. A step that
    invented its own defaults would capture a different dataset set into a different root
    and the two paths would diverge silently.
    """
    _step()()
    assert list(recorder.value("datasets", 0)) == sorted(DATASET_COLUMNS)
    assert Path(recorder.value("data_root", 3)) == Path(default_data_root())
    assert Path(recorder.value("manifest_dir", 4)) == Path(
        capture_live_season.LIVE_MANIFEST_DIR
    )


def test_the_step_never_reaches_main(pinned_week: None) -> None:
    """``main`` returns EXIT_USAGE for a zero-argument call, so it can never capture."""
    source = inspect.getsource(_step())
    tree = ast.parse(source.lstrip())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "run_capture" in names, names
    assert "main" not in names, (
        "step_capture_live_season references `main`, which parses argv and returns "
        "EXIT_USAGE when --week is absent -- a step calling it would report a usage error "
        "every Friday while appearing to run"
    )


# ---------------------------------------------------------------------------
# One week resolution, shared with the ingest step
# ---------------------------------------------------------------------------


def test_the_capture_step_receives_the_same_pair_the_ingest_step_does(
    monkeypatch: pytest.MonkeyPatch, pinned_week: None, recorder: _CaptureRecorder
) -> None:
    """Driven from ONE resolved context, and the two recorded pairs are compared.

    Not asserted by reading both call sites: two sites that read the same function today
    can be edited apart tomorrow, and the failure -- a capture filed under a week nobody
    ingested -- would look like a data gap rather than like a bug.
    """
    _IngesterRecorder.seasons = []
    monkeypatch.setattr("scripts.ingest_games.GameDataIngester", _IngesterRecorder)

    _step()()
    steps.step_ingest_games()

    captured_season, captured_week = recorder.season_and_week
    assert _IngesterRecorder.seasons == [[captured_season]], (
        f"the capture recorded season {captured_season} and the ingest step asked for "
        f"{_IngesterRecorder.seasons}; a capture filed under a week nobody ingested is "
        "worse than no capture"
    )
    assert (captured_season, captured_week) == (_SEASON, _WEEK)


# ---------------------------------------------------------------------------
# The two failure classes
# ---------------------------------------------------------------------------


def test_a_capture_failure_code_is_raised_naming_the_season_week_and_code(
    monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    """A non-zero return is never swallowed. T-33-36: silence here means a stale live zone."""
    monkeypatch.setattr(
        capture_live_season,
        "run_capture",
        _CaptureRecorder(capture_live_season.EXIT_CAPTURE_FAILED),
    )

    with pytest.raises(RuntimeError) as raised:
        _step()()

    message = str(raised.value)
    assert str(_SEASON) in message, message
    assert str(_WEEK) in message, message
    assert str(capture_live_season.EXIT_CAPTURE_FAILED) in message, message


def test_a_usage_class_code_produces_a_distinguishable_message(
    monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    """A usage error and a capture failure need different fixes, so they read differently."""
    monkeypatch.setattr(
        capture_live_season,
        "run_capture",
        _CaptureRecorder(capture_live_season.EXIT_USAGE),
    )
    with pytest.raises(RuntimeError) as usage:
        _step()()

    monkeypatch.setattr(
        capture_live_season,
        "run_capture",
        _CaptureRecorder(capture_live_season.EXIT_CAPTURE_FAILED),
    )
    with pytest.raises(RuntimeError) as failure:
        _step()()

    assert type(usage.value) is not type(failure.value), (
        "a usage error and a capture failure raise the SAME class; a caller cannot tell "
        "'this run asked for the wrong thing' from 'the capture broke'"
    )
    assert str(usage.value) != str(failure.value)


def test_a_sealed_season_refusal_is_reported_as_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, pinned_week: None
) -> None:
    """Asking the live tool for a SEALED season is an operator mistake, not a capture failure.

    ``run_capture`` lets ``ZoneWriteRefused`` escape -- the CLI's own ``main`` is what turns
    it into ``EXIT_USAGE`` -- so a step that called ``run_capture`` directly and caught only
    non-zero returns would surface a bare traceback naming a zone rule. It is raised as the
    same usage class the wrong-arguments case uses, because the fix is the same kind: point
    the run at the season this tool owns.
    """

    def refuse(*_args: Any, **_kwargs: Any) -> int:
        raise ZoneWriteRefused("season 2019 belongs to the SEALED zone")

    monkeypatch.setattr(capture_live_season, "run_capture", refuse)

    with pytest.raises(RuntimeError) as raised:
        _step()()

    assert "SEALED" in str(raised.value) or "sealed" in str(raised.value), raised.value
    assert str(_SEASON) in str(raised.value), raised.value
