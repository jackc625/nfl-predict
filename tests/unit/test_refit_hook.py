"""The daily run's re-fit side step: one due date attempted, recorded, swapped, closed (Plan 37-13).

WHAT IS PROVEN (D-15, D-17, D-37, SPEC R9, R10, P4, research Pitfalls 1 and 2)
-------------------------------------------------------------------------------
* A due cadence date runs end to end through ``refit.hook.attempt_date``: the real records, the
  real cadence trigger and the REAL swap (``refit.swap.install_generation``) on fixture stores,
  producing ``install_intent`` -> ``attempt installed`` -> ``closed ran`` and a swapped tmp
  ``latest.json``. The heavy steps (the D-37 build, training, re-measure, checks) are injected
  through ``RefitSteps``; their real functions are proven in Plans 37-07..37-11, and the R10 rows
  test runs the REAL ``refit.training.train_generation``.
* Every outcome is a record: a refusal names its check, a crash names its exception class only,
  a busy lock records nothing, a future date has no record, a missed date runs late with its own
  cutoff, the next date closes an older open one ``not run``.
* Catch-up and recovery: every A/A week evaluated in a run is on the outcome, and a date whose
  swap a crash interrupted is closed on the next run without re-training.

Every ledger, artifacts dir, gold and silver here is a pytest temporary fixture. The captured code
commit is pinned to a clean fixture value by one autouse fixture, so no test depends on whether the
developer's checkout is clean (a ``+dirty`` value would make the real install refuse with
``clean_code_commit``, which is the check working, not a defect).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import refit.integrity
import refit.records
import refit.witness
from backtest.refit_preregistration_2026 import REFIT_RECIPE_ID
from backtest.weekly_bet_list import load_bet_rule_record
from forward_ledger.schema import ARM_LIVE
from forward_ledger.store import read_entries, verify_chain
from models.artifacts import ResolvedArtifacts
from refit import swap
from refit.cadence import cadence_dates, date_statuses
from refit.generations import generation_for_ids, installed_generations
from refit.hook import (
    REFIT_STATUSES,
    RefitSteps,
    Remeasured,
    attempt_date,
    run_refit_side_step,
)
from refit.integrity import IntegrityRefusal
from refit.records import (
    REFIT_DIRNAME,
    append_aa_record,
    attempt_lock,
    read_aa_records,
    read_date_events,
)
from refit.training import GenerationTraining, train_generation
from tests.fixtures.refit_gold import build_refit_fixture
from tests.unit.test_forward_ledger_store import make_row
from tests.unit.test_phase37_preregistration_ancestry import first_refit_intent
from tests.unit.test_refit_aa import WEEK8_KICKOFFS, week8_rows, write_twins
from tests.unit.test_refit_decision_path import (
    GENERATION_ONE_IDS,
    build_stores,
    record,
    run_log_path,  # noqa: F401 - a pytest fixture, used by name
    write_generation_bet_rule,
)
from tests.unit.test_refit_swap import (
    CLEAN_COMMIT,
    NEW_IDS,
    OLD_IDS,
    build_artifacts,
    ids_of,
    make_generation_one_installable,
    manifest,
    remeasured_numbers,
    sha256_of,
    write_record,
)

_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# The runs, each five minutes after a 17:00 ET cutoff (22:00 UTC once DST has ended).
NOV3 = datetime(2026, 11, 3, 22, 5, tzinfo=UTC)
NOV4 = datetime(2026, 11, 4, 22, 5, tzinfo=UTC)
NOV10 = datetime(2026, 11, 10, 22, 5, tzinfo=UTC)
DEC1 = datetime(2026, 12, 1, 22, 5, tzinfo=UTC)

FIRST_CUTOFF: str = "2026-11-03T22:00:00+00:00"

# A finished game whose score is missing at the first attempt (the fixture's postponed game).
UNSCORED_GAME: str = "2026_08_PPP_QQQ"

# A crash message the records must never carry (D-10: the class name only).
SECRET_TEXT: str = "secret-ish text"


@pytest.fixture(autouse=True)
def _clean_code_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hook captures ``refit.records.code_commit(...)``; here it is always a clean commit."""
    monkeypatch.setattr(refit.records, "code_commit", lambda _repo_root: CLEAN_COMMIT)


# ---------------------------------------------------------------------------
# The fixture world (tmp only) and the injected steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    """One tmp store set: the re-fit fixture's gold and silver, an artifacts dir, a ledger."""

    ledger: Path
    artifacts: Path
    gold: Path
    silver: Path
    sha: str
    snapshots: pd.DataFrame


def build_world(root: Path) -> World:
    """Gold + silver from the re-fit fixture; artifacts serving OLD_IDS beside NEW_IDS."""
    fixture = build_refit_fixture(root / "fixture")
    artifacts = build_artifacts(root / "stores")
    ledger = root / "ledger"
    return World(
        ledger=ledger,
        artifacts=artifacts,
        gold=fixture.gold_dir,
        silver=fixture.silver_dir,
        sha=write_record(ledger, NEW_IDS),
        snapshots=fixture.snapshots,
    )


def record_clean_week(ledger: Path, week: int = 8) -> None:
    assert append_aa_record(
        ledger, {"season": 2026, "week": week, "clean": True, "reason": "clean"}
    )


def fake_training(no_final_score_ids: tuple[str, ...] = ()) -> GenerationTraining:
    return GenerationTraining(
        artifact_ids={t: NEW_IDS.model_id_for(t) for t in _TARGETS},
        metadata={},
        gold_generation_digest="fixture-gold",
        no_final_score_ids=no_final_score_ids,
    )


@dataclass
class Fakes:
    """The injected heavy steps, recording what they were handed."""

    world: World
    calls: list[str] = field(default_factory=list)
    train_kwargs: list[dict[str, Any]] = field(default_factory=list)
    trainings: list[GenerationTraining] = field(default_factory=list)
    remeasure_games: list[pd.DataFrame] = field(default_factory=list)
    no_score: tuple[str, ...] = ()
    check_error: Exception | None = None
    ids: ResolvedArtifacts = NEW_IDS

    def prepare(self, run_date_et: date, *, slate: Any) -> None:
        self.calls.append("prepare")

    def train(self, cutoff_utc: datetime, **kwargs: Any) -> GenerationTraining:
        self.calls.append("train")
        self.train_kwargs.append({"cutoff_utc": cutoff_utc, **kwargs})
        return fake_training(self.no_score)

    def remeasure(self, training: GenerationTraining, **kwargs: Any) -> Remeasured:
        self.calls.append("remeasure")
        self.trainings.append(training)
        self.remeasure_games.append(kwargs["games"])
        return Remeasured(
            ids=ids_of(self.ids),
            bet_rule_sha256=self.world.sha,
            numbers=remeasured_numbers(),
        )

    def checks(self, **_kwargs: Any) -> None:
        self.calls.append("checks")
        if self.check_error is not None:
            raise self.check_error

    def steps(self, **overrides: Any) -> RefitSteps:
        options: dict[str, Any] = {
            "prepare": self.prepare,
            "train": self.train,
            "remeasure": self.remeasure,
            "checks": self.checks,
        }
        options.update(overrides)
        return RefitSteps(**options)


def side_step(world: World, steps: RefitSteps, now: datetime, **kwargs: Any) -> Any:
    """One daily run's side step on *world*, run on the ET day of *now*."""
    options: dict[str, Any] = {
        "run_date_et": now.date(),
        "now_utc": now,
        "slate": None,
        "schedule_refreshed": True,
        "refresh_schedule": None,
        "ledger_dir": world.ledger,
        "artifacts_dir": world.artifacts,
        "gold_dir": world.gold,
        "silver_dir": world.silver,
        "steps": steps,
    }
    options.update(kwargs)
    return run_refit_side_step(**options)


def kinds(ledger: Path) -> list[tuple[str, str]]:
    """``(event, outcome or status)`` of every cadence-date event, in file order."""
    return [
        (event["event"], event.get("outcome") or event.get("status") or "")
        for event in read_date_events(ledger)
    ]


def attempts(ledger: Path) -> list[dict[str, Any]]:
    return [e for e in read_date_events(ledger) if e["event"] == "attempt"]


def tree(root: Path) -> dict[str, str]:
    """Every file under *root* (relative path -> sha256); empty when *root* is absent."""
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): sha256_of(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def served_pointers(artifacts: Path) -> dict[str, str]:
    written = manifest(artifacts)
    return {key: written[key] for key in ("wp", "ats", "ou", "blend")}


def pointers_of(ids: ResolvedArtifacts) -> dict[str, str]:
    return {key: getattr(ids, key) for key in ("wp", "ats", "ou", "blend")}


# ---------------------------------------------------------------------------
# Task 1 (tracer): one due date, end to end
# ---------------------------------------------------------------------------


def test_a_due_date_runs_and_installs(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    fakes = Fakes(world)

    outcome = side_step(world, fakes.steps(), NOV3)

    assert outcome.status == "ran"
    assert outcome.status in REFIT_STATUSES
    assert (outcome.list_week, outcome.date_index, outcome.date) == (8, 1, "2026-11-03")
    assert (outcome.nfl_week, outcome.next_date) == (9, "2026-12-01")
    assert outcome.wrote_records is True
    assert outcome.after_ids == ids_of(NEW_IDS)
    assert fakes.calls == ["prepare", "train", "remeasure", "checks"]
    assert kinds(world.ledger) == [
        ("install_intent", ""),
        ("attempt", "installed"),
        ("closed", "ran"),
    ]
    intent, attempt, closed = read_date_events(world.ledger)
    assert attempt["cutoff_utc"] == FIRST_CUTOFF
    assert attempt["attempted_at_utc"] == NOV3.isoformat()
    assert attempt["run_date_et"] == "2026-11-03"
    assert attempt["code_commit"] == CLEAN_COMMIT
    assert intent["code_commit"] == attempt["code_commit"]
    assert attempt["before_ids"] == ids_of(OLD_IDS)
    assert attempt["after_ids"] == ids_of(NEW_IDS)
    # The date's eligibility snapshot, read from silver by the hook itself (R10).
    assert attempt["no_final_score_ids"] == [UNSCORED_GAME]
    assert closed["after_ids"] == ids_of(NEW_IDS)
    assert served_pointers(world.artifacts) == pointers_of(NEW_IDS)


def test_not_due_records_nothing(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    fresh = tmp_path / "fresh_ledger"
    fakes = Fakes(world)
    before = datetime(2026, 10, 21, 21, 30, tzinfo=UTC)

    outcome = side_step(world, fakes.steps(), before, ledger_dir=fresh)

    assert outcome.status == "waiting"
    assert outcome.wrote_records is False
    assert not fresh.exists(), "a day with nothing due created a file"
    assert fakes.calls == []


def test_each_refusal_is_recorded_with_its_check(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    fakes = Fakes(world, check_error=IntegrityRefusal("scores_frame", "planted"))
    latest = sha256_of(world.artifacts / "latest.json")

    outcome = side_step(world, fakes.steps(), NOV3)

    assert (outcome.status, outcome.check) == ("refused", "scores_frame")
    assert kinds(world.ledger) == [("attempt", "refused")]
    (attempt,) = attempts(world.ledger)
    assert attempt["check"] == "scores_frame"
    assert attempt["exception_class"] is None
    assert sha256_of(world.artifacts / "latest.json") == latest


def test_a_crash_is_recorded_by_class_only(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    fakes = Fakes(world, check_error=ValueError(SECRET_TEXT))

    outcome = side_step(world, fakes.steps(), NOV3)

    assert (outcome.status, outcome.exception_class) == ("crashed", "ValueError")
    (attempt,) = attempts(world.ledger)
    assert (attempt["outcome"], attempt["exception_class"]) == ("crashed", "ValueError")
    for path in (world.ledger / REFIT_DIRNAME).rglob("*.jsonl"):
        assert SECRET_TEXT not in path.read_text(encoding="utf-8")


def test_fingerprint_refuses_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edited = lambda *_a, **_k: ["ROLLING-REFIT-PREREGISTRATION.md"]  # noqa: E731
    monkeypatch.setattr(refit.witness, "fingerprint_mismatches", edited)
    # refit.integrity.check_recipe_fingerprint reads the name it imported from refit.witness.
    monkeypatch.setattr(refit.integrity, "fingerprint_mismatches", edited)
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    fakes = Fakes(world)

    outcome = side_step(world, fakes.steps(), NOV3)

    assert (outcome.status, outcome.check) == ("refused", "recipe_fingerprint")
    assert fakes.calls == [], "the build or training ran on an edited record"
    assert kinds(world.ledger) == [("attempt", "refused")]
    # A first attempt refused before anything was built still fixes the date's eligibility.
    assert attempts(world.ledger)[0]["no_final_score_ids"] == [UNSCORED_GAME]


def test_a_missed_date_runs_late_with_the_same_cutoff_and_rows(tmp_path: Path) -> None:
    rows: dict[str, dict[str, Any]] = {}
    for name, now in (("punctual", NOV3), ("late", NOV4)):
        world = build_world(tmp_path / name)
        record_clean_week(world.ledger)
        fakes = Fakes(world)
        real_train = partial(train_generation, snapshots=world.snapshots)

        outcome = side_step(world, fakes.steps(train=real_train), now)

        assert outcome.status == ("ran" if name == "punctual" else "ran late")
        (attempt,) = attempts(world.ledger)
        assert attempt["cutoff_utc"] == FIRST_CUTOFF
        assert attempt["attempted_at_utc"] == now.isoformat()
        (training,) = fakes.trainings
        rows[name] = {
            target: (
                training.metadata[target]["training_rows"],
                training.metadata[target]["exclusions"],
            )
            for target in _TARGETS
        }
    assert kinds(tmp_path / "late" / "ledger")[-1] == ("closed", "ran late")
    assert rows["late"] == rows["punctual"]


def test_a_retry_after_refusal_keeps_the_cutoff(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    fakes = Fakes(world, check_error=IntegrityRefusal("scores_frame", "planted"))
    assert side_step(world, fakes.steps(), NOV3).status == "refused"

    fakes.check_error = None
    outcome = side_step(world, fakes.steps(), NOV4)

    assert outcome.status == "ran late"
    first, second = attempts(world.ledger)
    assert first["cutoff_utc"] == second["cutoff_utc"] == FIRST_CUTOFF
    assert (first["run_date_et"], second["run_date_et"]) == ("2026-11-03", "2026-11-04")
    assert kinds(world.ledger)[-1] == ("closed", "ran late")


def test_the_next_date_closes_the_older(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    fakes = Fakes(world, check_error=IntegrityRefusal("scores_frame", "planted"))
    assert side_step(world, fakes.steps(), NOV3).status == "refused"

    fakes.check_error = None
    outcome = side_step(world, fakes.steps(), DEC1)

    assert outcome.closed_not_run == ("2026-11-03",)
    assert (outcome.status, outcome.date_index, outcome.date) == (
        "ran",
        2,
        "2026-12-01",
    )
    events = read_date_events(world.ledger)
    closes = {(e["date_index"], e["status"]) for e in events if e["event"] == "closed"}
    assert closes == {(1, "not run"), (2, "ran")}


def test_a_busy_lock_records_nothing(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    fakes = Fakes(world)

    with attempt_lock(world.ledger):
        outcome = side_step(world, fakes.steps(), NOV3)

    assert outcome.status == "busy"
    assert outcome.wrote_records is False
    assert read_date_events(world.ledger) == []
    assert fakes.calls == []


def test_one_record_per_passed_date(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    fakes = Fakes(world, check_error=IntegrityRefusal("scores_frame", "planted"))
    assert side_step(world, fakes.steps(), NOV3).status == "refused"
    fakes.check_error = None
    assert side_step(world, fakes.steps(), NOV4).status == "ran late"
    assert side_step(world, fakes.steps(), DEC1).status == "ran"

    statuses = date_statuses(cadence_dates(8), read_date_events(world.ledger))
    for status in statuses:
        if status.date.cutoff_utc <= DEC1:
            assert status.status in ("ran", "ran late"), status
            assert status.attempts >= 1
        else:
            assert (status.status, status.attempts) == ("open", 0), status
    future = {s.date.index for s in statuses if s.date.cutoff_utc > DEC1}
    assert future == {3, 4}
    assert not any(e["date_index"] in future for e in read_date_events(world.ledger))


# ---------------------------------------------------------------------------
# The A/A weeks (D-02) and the catch-up run (review round 1)
# ---------------------------------------------------------------------------

WEEK9_KICKOFFS: dict[str, str] = {
    "2026_09_NYJ_MIA": "2026-11-05T20:15:00-05:00",
    "2026_09_GB_DET": "2026-11-08T13:00:00-05:00",
}


def week9_rows(arm: str) -> list[dict[str, Any]]:
    """One row per game and target of week 9 in *arm*, decided at its first decision run."""
    return [
        make_row(
            game_id=game_id,
            week=9,
            target=target,
            freeze_ts="2026-11-04T18:00:00-05:00",
            decided_at_utc="2026-11-04T22:10:00+00:00",
            arm=arm,
        )
        for game_id in WEEK9_KICKOFFS
        for target in _TARGETS
    ]


def write_aa_schedule(silver: Path) -> None:
    """Silver ``games`` holding exactly the A/A fixture weeks 8 and 9."""
    rows = [
        {"game_id": game_id, "season": 2026, "week": week, "kickoff_et": kickoff}
        for week, kickoffs in ((8, WEEK8_KICKOFFS), (9, WEEK9_KICKOFFS))
        for game_id, kickoff in kickoffs.items()
    ]
    frame = pd.DataFrame(rows)
    frame["kickoff_et"] = pd.to_datetime(frame["kickoff_et"], utc=True)
    frame["home_score"] = 24.0
    frame["away_score"] = 17.0
    frame.to_parquet(silver / "games.parquet", index=False)


def write_unclean_week_8(ledger: Path) -> None:
    shadow = week8_rows("shadow")
    shadow[4] = {**shadow[4], "per_bet_ev": 0.0378}
    write_twins(ledger, week8_rows("live"), shadow)


def test_an_unclean_aa_week_waits(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    write_aa_schedule(world.silver)
    write_unclean_week_8(world.ledger)
    fakes = Fakes(world)

    outcome = side_step(world, fakes.steps(), NOV3)

    assert outcome.status == "waiting"
    (result,) = outcome.aa_results
    assert (result.week, result.clean, result.reason) == (8, False, "field_mismatch")
    assert result.first_mismatch is not None
    assert result.first_mismatch["field"] == "per_bet_ev"
    assert [(r["week"], r["clean"]) for r in read_aa_records(world.ledger)] == [
        (8, False)
    ]
    assert read_date_events(world.ledger) == []
    assert fakes.calls == []

    write_twins(world.ledger, week9_rows("live"), week9_rows("shadow"))
    outcome = side_step(world, fakes.steps(), NOV10)

    assert [(r.week, r.clean) for r in outcome.aa_results] == [(9, True)]
    assert (outcome.status, outcome.list_week, outcome.date) == ("ran", 9, "2026-11-10")


def test_a_catch_up_run_keeps_every_aa_result(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    write_aa_schedule(world.silver)
    write_unclean_week_8(world.ledger)
    write_twins(world.ledger, week9_rows("live"), week9_rows("shadow"))
    fakes = Fakes(world)

    outcome = side_step(world, fakes.steps(), NOV10)

    assert [(r.week, r.clean) for r in outcome.aa_results] == [(8, False), (9, True)]
    first_mismatch = outcome.aa_results[0].first_mismatch
    assert first_mismatch is not None
    assert first_mismatch["field"] == "per_bet_ev"
    assert (outcome.status, outcome.list_week, outcome.date) == ("ran", 9, "2026-11-10")
    assert [(r["week"], r["clean"]) for r in read_aa_records(world.ledger)] == [
        (8, False),
        (9, True),
    ]


# ---------------------------------------------------------------------------
# Recovery and retry identity (research Pitfall 2, SPEC R10)
# ---------------------------------------------------------------------------


def test_pending_install_closes_without_retraining(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    swap.install_generation(
        world.ledger,
        world.artifacts,
        date=cadence_dates(8)[0],
        ids=ids_of(NEW_IDS),
        bet_rule_sha256=world.sha,
        code_commit=CLEAN_COMMIT,
    )
    assert kinds(world.ledger) == [("install_intent", "")]
    fakes = Fakes(world)

    outcome = side_step(world, fakes.steps(), NOV4)

    assert outcome.recovered_close == ("2026-11-03", "ran late")
    assert outcome.status == "ran late"
    assert outcome.wrote_records is True
    assert fakes.calls == [], "a swapped date was re-trained"
    assert kinds(world.ledger) == [("install_intent", ""), ("closed", "ran late")]


def land_the_postponed_score(silver: Path) -> None:
    """The postponed game's result lands in silver (between two attempts of one date)."""
    games_path = silver / "games.parquet"
    games = pd.read_parquet(games_path)
    landed = games["game_id"] == UNSCORED_GAME
    games.loc[landed, ["home_score", "away_score"]] = [27.0, 20.0]
    games.to_parquet(games_path, index=False)


def assert_the_retry_held_the_game(fakes: Fakes, retry: int) -> None:
    """Attempt *retry* trained and re-measured with the postponed game's score blanked."""
    assert fakes.train_kwargs[retry]["held_no_score_ids"] == frozenset({UNSCORED_GAME})
    retry_games = fakes.remeasure_games[retry]
    held_row = retry_games.loc[retry_games["game_id"] == UNSCORED_GAME]
    assert len(held_row) == 1
    assert np.isnan(held_row[["home_score", "away_score"]].to_numpy(dtype=float)).all()


def test_a_retry_holds_the_first_attempts_unscored_games(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    fakes = Fakes(
        world,
        no_score=(UNSCORED_GAME,),
        check_error=IntegrityRefusal("scores_frame", "planted"),
    )
    assert side_step(world, fakes.steps(), NOV3).status == "refused"

    land_the_postponed_score(world.silver)
    fakes.check_error = None

    assert side_step(world, fakes.steps(), NOV4).status == "ran late"

    assert fakes.train_kwargs[0]["held_no_score_ids"] == frozenset({UNSCORED_GAME})
    assert_the_retry_held_the_game(fakes, retry=1)
    assert [a["no_final_score_ids"] for a in attempts(world.ledger)] == [
        [UNSCORED_GAME],
        [UNSCORED_GAME],
    ]


def test_a_first_attempt_that_crashes_before_training_still_holds_its_unscored_games(
    tmp_path: Path,
) -> None:
    # CR-C01 / WR-B04: eligibility is read at the date's FIRST attempt (record section 8),
    # however early that attempt fails.
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    fakes = Fakes(world)

    def build_crashes(_run_date_et: date, *, slate: Any) -> None:
        msg = "the nightly build failed"
        raise OSError(msg)

    outcome = side_step(world, fakes.steps(prepare=build_crashes), NOV3)

    assert (outcome.status, outcome.exception_class) == ("crashed", "OSError")
    assert fakes.train_kwargs == [], "the crashed attempt reached training"
    (first,) = attempts(world.ledger)
    assert first["no_final_score_ids"] == [UNSCORED_GAME]

    land_the_postponed_score(world.silver)

    assert side_step(world, fakes.steps(), NOV4).status == "ran late"

    assert_the_retry_held_the_game(fakes, retry=0)
    assert [a["no_final_score_ids"] for a in attempts(world.ledger)] == [
        [UNSCORED_GAME],
        [UNSCORED_GAME],
    ]


def test_a_crash_after_the_swap_closes_on_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    fakes = Fakes(world)
    real_replace: Callable[..., None] = swap.replace_manifest

    def replace_then_die(mapping: Any, artifacts_dir: Path) -> None:
        real_replace(mapping, artifacts_dir)
        msg = "the run died right after the swap wrote latest.json"
        raise RuntimeError(msg)

    with monkeypatch.context() as patch:
        patch.setattr(swap, "replace_manifest", replace_then_die)
        outcome = side_step(world, fakes.steps(), NOV3)

    assert (outcome.status, outcome.exception_class) == ("crashed", "RuntimeError")
    assert kinds(world.ledger) == [("install_intent", ""), ("attempt", "crashed")]
    assert served_pointers(world.artifacts) == pointers_of(NEW_IDS)
    generation = generation_for_ids(NEW_IDS, installed_generations(world.ledger))
    assert generation.number == 1
    calls_after_crash = list(fakes.calls)

    outcome = side_step(world, fakes.steps(), NOV4)

    assert outcome.recovered_close == ("2026-11-03", "ran late")
    assert kinds(world.ledger) == [
        ("install_intent", ""),
        ("attempt", "crashed"),
        ("closed", "ran late"),
    ]
    assert fakes.calls == calls_after_crash, "the recovered date was re-trained"
    intent = first_refit_intent(world.ledger)
    assert intent is not None
    (crashed,) = attempts(world.ledger)
    assert intent["code_commit"] == crashed["code_commit"] == CLEAN_COMMIT


def test_lock_passed_refreshes_before_training(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    record_clean_week(world.ledger)
    fakes = Fakes(world)

    outcome = attempt_date(
        cadence_dates(8)[0],
        run_date_et=NOV3.date(),
        now_utc=NOV3,
        slate=None,
        schedule_refreshed=False,
        refresh_schedule=lambda: fakes.calls.append("refresh"),
        first_refit=True,
        ledger_dir=world.ledger,
        artifacts_dir=world.artifacts,
        gold_dir=world.gold,
        silver_dir=world.silver,
        steps=fakes.steps(),
    )

    assert outcome.status == "ran"
    assert fakes.calls.count("refresh") == 1
    assert fakes.calls[:2] == ["refresh", "prepare"]


def test_a_lock_passed_run_refreshes_once_before_the_aa_week_is_judged(
    tmp_path: Path,
) -> None:
    # WR-C03: a week-8 game moved to Tuesday 19:00 ET after Monday's refresh. The run that
    # starts at 18:10 ET must judge the binding A/A week on the refreshed schedule.
    world = build_world(tmp_path)
    write_aa_schedule(world.silver)
    write_twins(world.ledger, week8_rows("live"), week8_rows("shadow"))
    refreshed: list[str] = []

    def refresh_moves_a_game() -> None:
        refreshed.append("refresh")
        games_path = world.silver / "games.parquet"
        games = pd.read_parquet(games_path)
        moved = games["game_id"] == "2026_08_SF_SEA"
        games.loc[moved, "kickoff_et"] = pd.Timestamp("2026-11-04T00:00:00+00:00")
        games.to_parquet(games_path, index=False)

    late_start = datetime(2026, 11, 3, 23, 10, tzinfo=UTC)
    fakes = Fakes(world)
    outcome = side_step(
        world,
        fakes.steps(),
        late_start,
        schedule_refreshed=False,
        refresh_schedule=refresh_moves_a_game,
    )

    assert refreshed == ["refresh"]
    assert [(r.week, r.clean, r.reason) for r in outcome.aa_results] == [
        (8, False, "week_not_complete")
    ]
    assert outcome.status == "waiting"
    assert fakes.calls == []

    # With the week already recorded clean, a due date is still refreshed exactly once.
    other = build_world(tmp_path / "due")
    record_clean_week(other.ledger)
    counted = Fakes(other)
    outcome = side_step(
        other,
        counted.steps(),
        late_start,
        schedule_refreshed=False,
        refresh_schedule=lambda: counted.calls.append("refresh"),
    )

    assert outcome.status == "ran"
    assert counted.calls.count("refresh") == 1
    assert counted.calls[:2] == ["refresh", "prepare"]


def test_an_installed_generation_decides_on_the_next_run(
    tmp_path: Path,
    run_log_path: Path,  # noqa: F811 - the imported fixture, requested by name
) -> None:
    stores = build_stores(tmp_path / "stores")
    make_generation_one_installable(stores.artifacts)
    ledger = tmp_path / "ledger"
    record_clean_week(ledger)
    written: list[tuple[Path, str]] = []

    def remeasure(_training: GenerationTraining, **_kwargs: Any) -> Remeasured:
        path, sha = write_generation_bet_rule(ledger, GENERATION_ONE_IDS)
        written.append((path, sha))
        return Remeasured(
            ids=ids_of(GENERATION_ONE_IDS),
            bet_rule_sha256=sha,
            numbers=remeasured_numbers(),
        )

    world = World(
        ledger=ledger,
        artifacts=stores.artifacts,
        gold=stores.gold,
        silver=stores.silver,
        sha="",
        snapshots=pd.DataFrame(),
    )
    fakes = Fakes(world, ids=GENERATION_ONE_IDS)

    outcome = side_step(world, fakes.steps(remeasure=remeasure), NOV3)

    assert outcome.status == "ran"
    ((path, sha),) = written
    generations = installed_generations(ledger)
    assert [g.number for g in generations] == [0, 1]
    assert generations[1].ids == GENERATION_ONE_IDS
    assert generations[1].bet_rule_path == path
    fits, thresholds = load_bet_rule_record(generations[1].bet_rule_path)
    assert set(fits) == set(thresholds) == set(_TARGETS)
    intent = next(e for e in read_date_events(ledger) if e["event"] == "install_intent")
    assert intent["bet_rule_sha256"] == sha

    record(stores, ledger)

    assert verify_chain(read_entries(ledger)).ok
    live = [
        entry.immutable
        for entry in read_entries(ledger)
        if entry.kind == "row" and entry.immutable["arm"] == ARM_LIVE
    ]
    assert live, "the fixture week decided nothing, so nothing was tested"
    for row in live:
        assert row["recipe_id"] == REFIT_RECIPE_ID
        assert row["model_artifact_id"] == GENERATION_ONE_IDS.model_id_for(
            row["target"]
        )
        assert row["blend_id"] == GENERATION_ONE_IDS.blend
