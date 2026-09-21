"""The live-skip POLICY: catch a per-game refusal, record it, drop the game, re-run the rest.

Phase 33.2, Plan 33.2-03 Task 3 (SPEC R3; D33.2-05 live half; threats T-33.2-03-07..10).

WHAT THIS MODULE PROVES, ON SYNTHETIC EXCEPTIONS
------------------------------------------------
``pipeline/live_skip.py`` is production code, and this module drives it without a pipeline run
or a production store:

* the refusal set is CLOSED -- exactly ``InformationTimeViolation`` and ``LockPassedError``,
  compared by class identity -- so a later plan cannot widen it into a general failure
  suppressor without failing here;
* a refusal is mapped onto the closed reason vocabulary, and an exception the policy cannot map
  (or that names no game) is REFUSED rather than defaulted;
* the records built carry all eight required keys, one per (game, source, reason);
* the process exclusion register excludes, reads and resets, and is empty at import;
* re-applying the policy to an already-excluded game raises ``SkipNotConvergingError`` and
  writes nothing -- a build that ignores the exclusion must be loud, not looped;
* at the orchestrator's step seam (``FridayPipeline._execute_step`` on a synthetic step): a
  refusal is caught, recorded and re-run; an ordinary exception keeps today's ``FAILED`` result
  untouched; history mode never applies the policy; the round cap is DERIVED from the
  feature-source registry;
* the passed-lock refusal names every game whose lock is before the decision instant, treats
  at-lock as admissible, and honours the exclusion before refusing;
* the real feature build drops excluded games from its sources before the gate reads them.

The end-to-end proof through the real entry point is ``tests/integration/test_daily_run_skips.py``.
Skip records are written only under ``tmp_path``.

Run this module:  uv run pytest tests/unit/test_live_skip_policy.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest import weekly_bet_list
from features import provenance
from pipeline import live_skip, skip_log
from pipeline.live_skip import (
    LIVE_SKIP_EXCEPTIONS,
    GamesLockPassedError,
    SkipNotConvergingError,
    UnskippableRefusalError,
    apply_skip_policy,
    exclude_games,
    excluded_games,
    excluded_ids_from,
    refuse_passed_locks,
    reset_excluded_games,
    skip_reason_for,
    skip_records_from,
)
from pipeline.steps import PipelinePhase, StepDefinition, StepStatus

_RUN_ID = "2026-09-19T17:30:00-04:00"
_RUN_DATE = "2026-09-19"
_GAME = "2026_W03_KC@BUF"
_OTHER = "2026_W03_NYJ@MIA"


def _late(*game_ids: str, source: str = "elo") -> provenance.InformationTimeViolation:
    """The payload the real gate raises for post-lock values (features/provenance.py)."""
    return provenance.InformationTimeViolation(
        f"information-time violation in {source!r}",
        {
            "source": source,
            "game_ids": list(game_ids),
            "information_times": ["2026-09-19T22:41:00+00:00"] * len(game_ids),
            "locks": ["2026-09-19T22:00:00+00:00"] * len(game_ids),
            "violation_type": "information_time",
        },
    )


@pytest.fixture(autouse=True)
def clean_register() -> Iterator[None]:
    reset_excluded_games()
    yield
    reset_excluded_games()


@pytest.fixture
def record_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config" / "skip_records.jsonl"
    monkeypatch.setattr(skip_log, "SKIP_RECORD_PATH", path)
    return path


class TestTheClosedRefusalSet:
    def test_it_is_exactly_the_two_per_game_refusals_by_class_identity(self) -> None:
        """A same-named class in another module must not satisfy this."""
        assert len(LIVE_SKIP_EXCEPTIONS) == 2
        assert set(LIVE_SKIP_EXCEPTIONS) == {
            provenance.InformationTimeViolation,
            weekly_bet_list.LockPassedError,
        }

    @pytest.mark.parametrize("cls", [SkipNotConvergingError, UnskippableRefusalError])
    def test_the_policy_refusals_escape_every_quiet_catch_tuple(
        self, cls: type
    ) -> None:
        assert cls.__bases__ == (Exception,)
        assert not issubclass(cls, (RuntimeError, ValueError, ImportError))

    def test_the_named_passed_lock_refusal_is_still_a_lock_passed_error(self) -> None:
        assert issubclass(GamesLockPassedError, weekly_bet_list.LockPassedError)


class TestTheReasonMapping:
    def test_a_late_value_is_post_lock(self) -> None:
        assert skip_reason_for(_late(_GAME)) == "post_lock"

    def test_an_undated_value_is_undated(self) -> None:
        exc = provenance.UndatedSourceError(
            "no time", {"source": "elo", "game_ids": [_GAME]}
        )
        assert skip_reason_for(exc) == "undated"

    def test_a_naive_information_time_is_undated(self) -> None:
        exc = provenance.InformationTimeViolation(
            "naive",
            {
                "source": "elo",
                "game_ids": [_GAME],
                "violation_type": "naive_information_time",
            },
        )
        assert skip_reason_for(exc) == "undated"

    def test_a_coverage_gap_is_no_provenance(self) -> None:
        exc = provenance.ProvenanceCoverageError(
            "gap", {"source": "elo", "game_ids": [_GAME]}
        )
        assert skip_reason_for(exc) == "no_provenance"

    def test_a_passed_lock_is_post_lock(self) -> None:
        exc = GamesLockPassedError(
            "passed", {"source": "decision_instant", "game_ids": [_GAME]}
        )
        assert skip_reason_for(exc) == "post_lock"

    def test_a_gold_column_refusal_is_not_mapped_to_a_default(self) -> None:
        """refuse_provenance_columns raises the base class with no game: not a per-game skip."""
        exc = provenance.InformationTimeViolation(
            "a provenance column reached gold",
            {"columns": ["information_time"], "violation_type": "provenance_column"},
        )
        with pytest.raises(UnskippableRefusalError):
            skip_reason_for(exc)

    def test_an_ordinary_exception_is_refused_not_labelled(self) -> None:
        with pytest.raises(UnskippableRefusalError):
            skip_reason_for(ValueError("unrelated"))


class TestTheRecordsAndTheExcludedIds:
    def test_one_record_per_game_carrying_all_eight_keys(self) -> None:
        records = skip_records_from(
            _late(_GAME, _OTHER), run_id=_RUN_ID, run_date_et=_RUN_DATE
        )
        assert [r["game_id"] for r in records] == [_GAME, _OTHER]
        for record in records:
            assert set(record) == set(skip_log.REQUIRED_ENTRY_KEYS)
            assert record["source"] == "elo"
            assert record["reason"] == "post_lock"
            assert record["run_id"] == _RUN_ID
            assert record["run_date_et"] == _RUN_DATE
            assert record["information_time"] == "2026-09-19T22:41:00+00:00"
            assert record["lock"] == "2026-09-19T22:00:00+00:00"
            assert datetime.fromisoformat(record["recorded_at"]).tzinfo is not None

    def test_an_undated_record_carries_a_null_time_not_an_invented_one(self) -> None:
        exc = provenance.UndatedSourceError(
            "no time", {"source": "elo", "game_ids": [_GAME]}
        )
        (record,) = skip_records_from(exc, run_id=_RUN_ID, run_date_et=_RUN_DATE)
        assert record["information_time"] is None
        assert record["lock"] is None

    def test_the_excluded_ids_are_the_games_the_refusal_names(self) -> None:
        assert excluded_ids_from(_late(_GAME, _OTHER)) == frozenset({_GAME, _OTHER})

    def test_a_refusal_naming_no_game_cannot_be_turned_into_a_skip(self) -> None:
        """A plain LockPassedError carries only prose; parsing it would guess."""
        with pytest.raises(UnskippableRefusalError):
            excluded_ids_from(weekly_bet_list.LockPassedError("games passed"))


class TestTheRegister:
    def test_exclude_read_reset(self) -> None:
        assert excluded_games() == frozenset()
        exclude_games({_GAME})
        exclude_games([_OTHER])
        assert excluded_games() == frozenset({_GAME, _OTHER})
        reset_excluded_games()
        assert excluded_games() == frozenset()

    def test_the_read_is_a_snapshot_not_a_live_handle(self) -> None:
        snapshot = excluded_games()
        exclude_games({_GAME})
        assert snapshot == frozenset()


class TestApplyingThePolicy:
    def test_it_records_excludes_and_returns_the_games(self, record_path: Path) -> None:
        dropped = apply_skip_policy(_late(_GAME), run_id=_RUN_ID, run_date_et=_RUN_DATE)
        assert dropped == frozenset({_GAME})
        assert excluded_games() == frozenset({_GAME})
        (record,) = skip_log.read_skip_records(record_path)
        assert record["game_id"] == _GAME

    def test_an_already_excluded_game_raises_and_writes_nothing(
        self, record_path: Path
    ) -> None:
        apply_skip_policy(_late(_GAME), run_id=_RUN_ID, run_date_et=_RUN_DATE)
        before = record_path.read_bytes()
        with pytest.raises(SkipNotConvergingError, match=_GAME):
            apply_skip_policy(
                _late(_GAME, source="injury"), run_id=_RUN_ID, run_date_et=_RUN_DATE
            )
        assert record_path.read_bytes() == before

    def test_an_exception_outside_the_set_is_refused(self, record_path: Path) -> None:
        with pytest.raises(UnskippableRefusalError):
            apply_skip_policy(
                RuntimeError("boom"), run_id=_RUN_ID, run_date_et=_RUN_DATE
            )
        assert not record_path.exists()
        assert excluded_games() == frozenset()


# ---------------------------------------------------------------------------
# The orchestrator's step seam, on synthetic steps
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_factory(monkeypatch: pytest.MonkeyPatch, record_path: Path) -> Any:
    """A FridayPipeline whose constructor touches nothing but the pinned week."""
    from pipeline import orchestrator

    monkeypatch.setattr(orchestrator, "get_current_nfl_week", lambda: (2026, 3))

    def _make(*, history_mode: bool = False) -> Any:
        return orchestrator.FridayPipeline(force=True, history_mode=history_mode)

    return _make


def _step(body: Any) -> StepDefinition:
    return StepDefinition("synthetic", body, PipelinePhase.PREDICTIONS, critical=True)


class _RaisesOnceFor:
    """A step body that refuses the listed games until they are excluded, then succeeds."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
        named = set(getattr(self.exc, "details", {}).get("game_ids", []))
        if not named <= excluded_games():
            raise self.exc


class TestTheStepSeam:
    def test_a_refusal_is_caught_recorded_and_the_step_re_run(
        self, pipeline_factory: Any, record_path: Path
    ) -> None:
        body = _RaisesOnceFor(_late(_GAME))
        pipeline = pipeline_factory()
        result = pipeline._execute_step(_step(body))

        assert result.status is StepStatus.SUCCESS
        assert body.calls == 2
        assert excluded_games() == frozenset({_GAME})
        assert pipeline.execution_log.skipped_games == [_GAME]
        assert [r["game_id"] for r in skip_log.read_skip_records(record_path)] == [
            _GAME
        ]

    def test_a_lock_passed_refusal_is_treated_identically(
        self, pipeline_factory: Any, record_path: Path
    ) -> None:
        exc = GamesLockPassedError(
            "passed", {"source": "decision_instant", "game_ids": [_GAME]}
        )
        result = pipeline_factory()._execute_step(_step(_RaisesOnceFor(exc)))
        assert result.status is StepStatus.SUCCESS
        (record,) = skip_log.read_skip_records(record_path)
        assert record["reason"] == "post_lock"

    def test_an_ordinary_exception_keeps_todays_failed_result(
        self, pipeline_factory: Any, record_path: Path
    ) -> None:
        def body() -> None:
            raise RuntimeError("ordinary failure")

        result = pipeline_factory()._execute_step(_step(body))
        assert result.status is StepStatus.FAILED
        assert result.error == "ordinary failure"
        assert excluded_games() == frozenset()
        assert not record_path.exists()

    def test_history_mode_never_applies_the_policy(
        self, pipeline_factory: Any, record_path: Path
    ) -> None:
        body = _RaisesOnceFor(_late(_GAME))
        result = pipeline_factory(history_mode=True)._execute_step(_step(body))
        assert result.status is StepStatus.FAILED
        assert body.calls == 1
        assert excluded_games() == frozenset()
        assert not record_path.exists()

    def test_a_build_that_ignores_the_exclusion_fails_loudly(
        self, pipeline_factory: Any, record_path: Path
    ) -> None:
        exc = _late(_GAME)

        def ignores_the_register() -> None:
            raise exc

        result = pipeline_factory()._execute_step(_step(ignores_the_register))
        assert result.status is StepStatus.FAILED
        assert result.error is not None and "did not converge" in result.error
        assert len(skip_log.read_skip_records(record_path)) == 1

    def test_the_round_loop_is_bounded_by_the_derived_cap(
        self, pipeline_factory: Any, monkeypatch: pytest.MonkeyPatch, record_path: Path
    ) -> None:
        """Each round names a NEW game, so only the cap can stop it."""
        monkeypatch.setattr(live_skip, "max_skip_rounds", lambda: 3)
        counter = {"n": 0}

        def always_a_new_game() -> None:
            counter["n"] += 1
            raise _late(f"2026_W03_G{counter['n']:02d}@H{counter['n']:02d}")

        result = pipeline_factory()._execute_step(_step(always_a_new_game))
        assert result.status is StepStatus.FAILED
        assert result.error is not None and "did not converge" in result.error
        assert counter["n"] == 4, "three skip rounds, then the fourth refusal stops it"

    def test_the_cap_is_derived_from_the_feature_source_registry(self) -> None:
        from scripts import build_features

        assert (
            live_skip.max_skip_rounds() == len(build_features.FEATURE_SOURCE_KEYS) + 1
        )
        # The registry the cap reads IS the one the information-time stage walks.
        assert (
            tuple(build_features.SUPPLIER_ATTRIBUTES)
            == build_features.FEATURE_SOURCE_KEYS
        )

    def test_the_orchestrator_resets_the_register_before_the_step_loop(self) -> None:
        import ast

        from pipeline import orchestrator

        source = inspect.getsource(orchestrator.FridayPipeline.run)
        tree = ast.parse(source.lstrip())
        calls = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "reset_excluded_games"
        ]
        loops = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.For)]
        assert calls, "run() never resets the exclusion register"
        assert min(calls) < min(loops), (
            "the register is reset after the step loop starts"
        )


# ---------------------------------------------------------------------------
# The passed-lock refusal
# ---------------------------------------------------------------------------


def _schedule() -> pd.DataFrame:
    """A Thursday game (locks Wednesday 18:00 ET) and a Sunday game (locks Saturday)."""
    return pd.DataFrame(
        {
            "game_id": ["2026_W03_TNF@HOM", "2026_W03_SUN@DAY"],
            "kickoff_et": [
                pd.Timestamp("2026-09-18T00:15:00+00:00"),  # Thu 20:15 ET
                pd.Timestamp("2026-09-20T17:00:00+00:00"),  # Sun 13:00 ET
            ],
        }
    )


_THURSDAY_LOCK = datetime(2026, 9, 16, 22, 0, tzinfo=UTC)  # Wed 18:00 ET


class TestThePassedLockRefusal:
    def test_a_game_whose_lock_is_before_the_decision_is_named(self) -> None:
        with pytest.raises(GamesLockPassedError) as caught:
            refuse_passed_locks(
                _schedule(), decided_at=_THURSDAY_LOCK + timedelta(seconds=1)
            )
        assert caught.value.details["game_ids"] == ["2026_W03_TNF@HOM"]
        assert caught.value.details["source"] == "decision_instant"
        assert isinstance(caught.value, weekly_bet_list.LockPassedError)

    def test_at_lock_is_admissible(self) -> None:
        refuse_passed_locks(_schedule(), decided_at=_THURSDAY_LOCK)

    def test_an_excluded_game_is_dropped_before_the_refusal(self) -> None:
        refuse_passed_locks(
            _schedule(),
            decided_at=_THURSDAY_LOCK + timedelta(days=1),
            excluded_game_ids=frozenset({"2026_W03_TNF@HOM"}),
        )

    def test_a_naive_decision_instant_is_refused_not_relabelled(self) -> None:
        with pytest.raises(ValueError):
            refuse_passed_locks(_schedule(), decided_at=datetime(2026, 9, 17, 12, 0))


# ---------------------------------------------------------------------------
# The two consumers' interfaces
# ---------------------------------------------------------------------------


def test_the_prediction_writer_takes_an_empty_keyword_only_exclusion() -> None:
    from scripts.generate_current_week_predictions import generate_and_write

    parameter = inspect.signature(generate_and_write).parameters["excluded_game_ids"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == frozenset()
    assert isinstance(parameter.default, frozenset)


def test_the_feature_build_takes_an_empty_keyword_only_exclusion() -> None:
    from scripts.build_features import FeatureMatrixBuilder

    parameter = inspect.signature(
        FeatureMatrixBuilder.generate_feature_matrices
    ).parameters["excluded_game_ids"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == frozenset()


class TestTheBuildDropsExcludedGamesFromItsSources:
    def test_frames_with_a_game_id_lose_the_excluded_rows(self) -> None:
        from scripts.build_features import drop_excluded_games

        sources = {
            "games": pd.DataFrame({"game_id": [_GAME, _OTHER], "season": [2026, 2026]}),
            "elo": pd.DataFrame(
                {"game_id": [_GAME, _OTHER], "home_elo": [1500.0, 1510.0]}
            ),
            "team_form": pd.DataFrame({"team": ["KC"], "rolling_epa": [0.1]}),
        }
        kept = drop_excluded_games(sources, frozenset({_GAME}))
        assert list(kept["games"]["game_id"]) == [_OTHER]
        assert list(kept["elo"]["game_id"]) == [_OTHER]
        pd.testing.assert_frame_equal(kept["team_form"], sources["team_form"])

    def test_an_empty_exclusion_returns_the_frames_untouched(self) -> None:
        from scripts.build_features import drop_excluded_games

        sources = {"games": pd.DataFrame({"game_id": [_GAME]})}
        kept = drop_excluded_games(sources, frozenset())
        assert kept["games"] is sources["games"]
