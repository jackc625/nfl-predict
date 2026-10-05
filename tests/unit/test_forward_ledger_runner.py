"""The runner: the ONE place the daily run talks to the ledger (Plan 34-15 Task 1).

``forward_ledger.runner`` records a slate (resolve once, build the decision bundle, stamp at the one
stamping site, first pick stands, store the snapshot and the artifact copies, then ONE guarded
commit with the slate's publish deadline), settles the ledger (grading, owed corrections and
closing finalizations in ONE atomic write) and syncs it after the run without ever raising.
``forward_ledger.run_log`` gains ``read_events`` and the per-run id every ledger event carries.

Every store here lives under ``tmp_path``: the decision bundle is injected, the artifacts, gold,
silver and live manifest are synthetic, and the run log is redirected. Nothing reads or writes the
repository's ``ledger/``, ``logs/``, ``data/`` or ``artifacts/`` (COLD-05), and no git runs.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.recipe_registry import RECIPE_REGISTRY
from backtest.weekly_bet_list import DecisionBundle, PublishDeadlinePassedError
from data.upstream_live import AS_OF_ENV, LIVE_MANIFEST_SCHEMA_VERSION
from forward_ledger import run_log, runner, store
from forward_ledger.canonical import IMMUTABLE_COLUMNS_V1
from forward_ledger.snapshots import SNAPSHOTS_DIRNAME
from forward_ledger.stamps import IN_FORCE_RECIPE_ID
from forward_ledger.store import (
    LEDGER_STAMP_COLUMNS,
    append_rows,
    apply_updates,
    ledger_path,
    read_entries,
    verify_chain,
)
from forward_ledger.sync import SyncOutcome
from models.artifacts import ResolvedArtifacts
from pipeline.daily_steps import DailySlate
from tests.unit.test_forward_ledger_store import graded, key_of, make_row
from tests.unit.test_ledger_corrections import capture as owed_capture
from tests.unit.test_ledger_settle import STRATEGIES, ats_row

SEASON = 2026
WEEK = 6
GAMES = ("2026_06_KC_BUF", "2026_06_DAL_PHI")
DECIDED_AT = datetime(2026, 10, 14, 21, 18, 10, tzinfo=UTC)
LOCK = datetime(2026, 10, 14, 22, 0, tzinfo=UTC)
SETTLE_AT = datetime(2026, 10, 21, 21, 0, 5, tzinfo=UTC)
CONVERTER_SLOPE = 0.15

_RECIPE = RECIPE_REGISTRY[IN_FORCE_RECIPE_ID]
RESOLVED = ResolvedArtifacts(
    wp=_RECIPE.model_artifact_ids["wp"],
    ats=_RECIPE.model_artifact_ids["ats"],
    ou=_RECIPE.model_artifact_ids["ou"],
    blend=_RECIPE.blend_id,
    converter=_RECIPE.converter_id,
)

# The first 23 immutable columns: what the decision itself emits. The 11 Phase-34 stamps are set
# only by ``forward_ledger.stamps`` -- the runner must not receive them pre-filled.
_DECISION_COLUMNS: tuple[str, ...] = IMMUTABLE_COLUMNS_V1[:23]


# ---------------------------------------------------------------------------
# Fixture builders (tmp_path only)
# ---------------------------------------------------------------------------


def _decided_row(
    game_id: str, decided_at: datetime, **overrides: Any
) -> dict[str, Any]:
    """One decided ATS row as the decision path emits it: no Phase-34 stamp yet."""
    fields: dict[str, Any] = {
        "game_id": game_id,
        "target": "ats",
        "bet_side": "home_cover",
        "slipped_line": -3.0,
        "decided_at_utc": decided_at.isoformat(),
        **overrides,
    }
    row = make_row(**fields)
    return {name: row[name] for name in _DECISION_COLUMNS}


def _frame(decided_at: datetime, **per_game: dict[str, Any]) -> pd.DataFrame:
    rows = [
        _decided_row(game_id, decided_at, **per_game.get(game_id, {}))
        for game_id in GAMES
    ]
    return pd.DataFrame(rows)


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": list(GAMES),
            "season": [SEASON] * len(GAMES),
            "week": [WEEK] * len(GAMES),
            "gameday": ["2026-10-15"] * len(GAMES),
        }
    )


def _gold() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": list(GAMES),
            "season": [SEASON] * len(GAMES),
            "week": [WEEK] * len(GAMES),
            "elo_diff": [12.5, -3.0],
        }
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": list(GAMES),
            "target": ["ats"] * len(GAMES),
            "model_value": [-3.2, 1.5],
        }
    )


class _Env:
    """The synthetic stores one runner call reads, all under one ``tmp_path``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.ledger = root / "ledger"
        self.artifacts = root / "artifacts"
        self.gold = root / "gold"
        self.silver = root / "silver"
        self.manifests = root / "upstream_live"
        self.chain_fit = root / "chain_fit.json"
        self.built: list[dict[str, Any]] = []
        self.frame_overrides: dict[str, dict[str, Any]] = {}

        for artifact_id in (
            RESOLVED.wp,
            RESOLVED.ats,
            RESOLVED.ou,
            RESOLVED.blend,
            RESOLVED.converter,
        ):
            directory = self.artifacts / str(artifact_id)
            directory.mkdir(parents=True)
            (directory / "metadata.json").write_text(
                json.dumps({"id": artifact_id, "slope_beta": CONVERTER_SLOPE}),
                encoding="utf-8",
            )
        # The blend payload and the manifest ``resolve_production_artifacts`` reads.
        (self.artifacts / RESOLVED.blend / "blend_weights.json").write_text(
            json.dumps(
                {
                    "weights": {"wp": 0.0, "ats": 0.0, "ou": 0.12},
                    "market_probability_artifact_id": RESOLVED.converter,
                    "market_probability_slope_beta": CONVERTER_SLOPE,
                }
            ),
            encoding="utf-8",
        )
        (self.artifacts / "latest.json").write_text(
            json.dumps(
                {
                    "wp": RESOLVED.wp,
                    "ats": RESOLVED.ats,
                    "ou": RESOLVED.ou,
                    "blend": RESOLVED.blend,
                }
            ),
            encoding="utf-8",
        )
        self.gold.mkdir()
        for target in ("wp", "ats", "ou"):
            _gold().to_parquet(self.gold / f"features_{target}.parquet", index=False)
        self.silver.mkdir()
        self.write_odds(pd.DataFrame())
        self.chain_fit.write_bytes(b'{"record_id": "synthetic_chain_fit"}\n')
        self.write_manifest()

    def write_odds(self, extra: pd.DataFrame) -> None:
        base = pd.DataFrame(
            {
                "game_id": [GAMES[0]],
                "sportsbook": ["draftkings"],
                "snapshot_ts": [pd.Timestamp(LOCK)],
                "created_at": [pd.Timestamp("2026-10-14T20:00:00Z")],
                "spread": [-2.5],
                "total": [44.5],
                "ml_home": [-130.0],
                "ml_away": [110.0],
            }
        )
        frame = pd.concat([base, extra], ignore_index=True) if not extra.empty else base
        frame.to_parquet(self.silver / "odds_snapshot.parquet", index=False)

    def write_games(self, *games: tuple[str, float | None, float | None]) -> None:
        pd.DataFrame(
            {
                "game_id": [game[0] for game in games],
                "season": [SEASON] * len(games),
                "week": [WEEK] * len(games),
                "kickoff_et": [pd.Timestamp("2026-10-16T00:15:00Z")] * len(games),
                "home_score": pd.Series([game[1] for game in games], dtype="float64"),
                "away_score": pd.Series([game[2] for game in games], dtype="float64"),
            }
        ).to_parquet(self.silver / "games.parquet", index=False)

    def write_manifest(self, *captures: dict[str, Any]) -> None:
        self.manifests.mkdir(exist_ok=True)
        manifest = {
            "schema_version": LIVE_MANIFEST_SCHEMA_VERSION,
            "season": SEASON,
            "datasets": {
                "schedules": {
                    "captures": [
                        {
                            "week": WEEK,
                            "sequence": 1,
                            "sha256": "s" * 64,
                            "path": "bronze/s",
                            "captured_at_utc": "2026-10-14T20:30:00+00:00",
                        },
                        *captures,
                    ]
                }
            },
        }
        (self.manifests / f"{SEASON}.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def builder(self, season: int, week: int, **kwargs: Any) -> DecisionBundle:
        """The injected decision seam: records its call, returns a synthetic bundle."""
        self.built.append({"season": season, "week": week, **kwargs})
        now = kwargs["now"]
        return DecisionBundle(
            frame=_frame(now, **self.frame_overrides),
            candidates=_candidates(),
            schedule=_schedule(),
            gold_inputs={target: _gold() for target in ("wp", "ats", "ou")},
            fits={},
            resolved=kwargs["resolved"],
            chain_fit_path=self.chain_fit,
            run_instant=now,
        )

    def record(self, decided_at: datetime = DECIDED_AT, **kwargs: Any) -> Any:
        options: dict[str, Any] = {
            "decided_at": decided_at,
            "excluded_game_ids": frozenset({"2026_06_SF_SEA"}),
            "publish_by": None,
            "ledger_dir": self.ledger,
            "artifacts_dir": self.artifacts,
            "gold_dir": self.gold,
            "silver_dir": self.silver,
            "manifest_dir": self.manifests,
            "bundle_builder": self.builder,
        }
        options.update(kwargs)
        return runner.record_forward_slate(_slate(), **options)

    def settle(self, **kwargs: Any) -> Any:
        options: dict[str, Any] = {
            "now": SETTLE_AT,
            "ledger_dir": self.ledger,
            "silver_dir": self.silver,
            "manifest_dir": self.manifests,
            "strategies": STRATEGIES,
        }
        options.update(kwargs)
        return runner.settle_ledger(**options)


def _slate() -> DailySlate:
    return DailySlate(
        run_date_et=date(2026, 10, 14),
        lock=LOCK,
        schedule=_schedule(),
    )


@pytest.fixture
def run_log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the ledger run log into *tmp_path*; production ``logs/`` is never written."""
    path = tmp_path / "logs" / "ledger_runs.jsonl"
    monkeypatch.setattr(run_log, "LEDGER_RUN_LOG", path)
    return path


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_log_path: Path) -> _Env:
    monkeypatch.delenv(AS_OF_ENV, raising=False)
    return _Env(tmp_path)


def _events(path: Path, event: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    return [r for r in records if event is None or r["event"] == event]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_dirs(ledger_dir: Path) -> list[str]:
    root = ledger_dir / SNAPSHOTS_DIRNAME
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


# ---------------------------------------------------------------------------
# record_forward_slate
# ---------------------------------------------------------------------------


def test_record_appends_stamped_rows_end_to_end(env: _Env, run_log_path: Path) -> None:
    outcome = env.record()

    assert outcome.appended == 2
    assert outcome.wrote is True
    (call,) = env.built
    assert (call["season"], call["week"], call["now"]) == (SEASON, WEEK, DECIDED_AT)
    assert call["excluded_game_ids"] == frozenset({"2026_06_SF_SEA"})
    assert call["resolved"] == RESOLVED, "the runner did not resolve the ids once"

    entries = read_entries(env.ledger)
    assert verify_chain(entries).ok
    assert [entry.immutable["game_id"] for entry in entries] == list(GAMES)
    for entry in entries:
        for column in LEDGER_STAMP_COLUMNS:
            assert entry.immutable[column] is not None, column
        assert entry.immutable["recipe_id"] == IN_FORCE_RECIPE_ID
        assert entry.immutable["model_artifact_id"] == RESOLVED.ats
        assert entry.immutable["arm"] == "live"

    (digest,) = _snapshot_dirs(env.ledger)
    assert entries[0].immutable["decision_snapshot_digest"] == digest
    for artifact_id in (RESOLVED.wp, RESOLVED.blend, RESOLVED.converter):
        assert (env.ledger / "artifacts" / str(artifact_id) / "metadata.json").is_file()
    assert (env.ledger / "recipes").is_dir()

    (append,) = _events(run_log_path, "append")
    assert append["appended_rows"] == 2
    assert append["head_hash"] == entries[-1].chain_hash
    assert append["entry_count"] == 2


def test_identical_repeat_writes_nothing(env: _Env) -> None:
    env.record()
    before = _sha(ledger_path(env.ledger))
    snapshots = _snapshot_dirs(env.ledger)

    outcome = env.record(decided_at=DECIDED_AT + timedelta(minutes=12))

    assert outcome.appended == 0
    assert outcome.identical == 2
    assert outcome.wrote is False
    assert _sha(ledger_path(env.ledger)) == before
    assert _snapshot_dirs(env.ledger) == snapshots, "a repeat stored a second snapshot"


def test_conflicting_rows_refused_others_appended(
    env: _Env, run_log_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env.record()
    env.frame_overrides = {GAMES[0]: {"bet_side": "away_cover"}}

    def _with_new_game(season: int, week: int, **kwargs: Any) -> DecisionBundle:
        bundle = env.builder(season, week, **kwargs)
        extra = pd.DataFrame([_decided_row("2026_06_NYJ_MIA", kwargs["now"])])
        return DecisionBundle(
            frame=pd.concat([bundle.frame, extra], ignore_index=True),
            candidates=bundle.candidates,
            schedule=bundle.schedule,
            gold_inputs=bundle.gold_inputs,
            fits=bundle.fits,
            resolved=bundle.resolved,
            chain_fit_path=bundle.chain_fit_path,
            run_instant=bundle.run_instant,
        )

    outcome = env.record(
        decided_at=DECIDED_AT + timedelta(minutes=5), bundle_builder=_with_new_game
    )

    assert outcome.appended == 1
    assert len(outcome.refused) == 1
    entries = read_entries(env.ledger)
    assert len(entries) == 3
    assert entries[0].immutable["bet_side"] == "home_cover", "the stored pick moved"
    assert entries[2].immutable["game_id"] == "2026_06_NYJ_MIA"
    (refusal,) = _events(run_log_path, "first_pick_refusal")
    assert refusal["key"][0] == GAMES[0]
    assert "bet_side" in refusal["fields"]
    assert f"LEDGER_REFUSED= {GAMES[0]}" in capsys.readouterr().out


def test_publish_deadline_passed_raises_and_writes_nothing(env: _Env) -> None:
    with pytest.raises(PublishDeadlinePassedError):
        env.record(publish_by=datetime(2026, 1, 1, tzinfo=UTC))
    assert not ledger_path(env.ledger).exists()


# ---------------------------------------------------------------------------
# settle_ledger
# ---------------------------------------------------------------------------


def _count_replaces(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    replaced: list[Path] = []
    real = store._replace_with_retry

    def _counting(path: Path, payload: bytes) -> None:
        replaced.append(path)
        real(path, payload)

    monkeypatch.setattr(store, "_replace_with_retry", _counting)
    return replaced


def test_settle_one_atomic_write(
    env: _Env, run_log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settled = ats_row(GAMES[0])
    pending = ats_row(GAMES[1])
    append_rows(env.ledger, [settled, pending])
    apply_updates(
        env.ledger, grading_updates={key_of(settled): graded("win", 0.9, 1.1)}
    )
    # GAMES[0] was graded a win; its score is now restated to a home loss by 10 (owed week 6).
    env.write_games((GAMES[0], 14.0, 24.0), (GAMES[1], 24.0, 17.0))
    env.write_manifest(owed_capture(8, 4, "2026-10-21T20:00:00+00:00"))
    replaced = _count_replaces(monkeypatch)

    outcome = env.settle()

    assert outcome.changed is True
    assert (outcome.graded, outcome.corrections, outcome.closing_finalized) == (1, 1, 2)
    assert len(replaced) == 1, "the settle pass did not write in one atomic replace"
    entries = read_entries(env.ledger)
    assert verify_chain(entries).ok
    assert entries[1].grading is not None
    assert entries[1].grading["grading_status"] == "win"
    assert entries[0].grading is not None
    assert entries[0].grading["grading_status"] == "win", (
        "the one-way grader was undone"
    )
    assert entries[2].kind == "correction"
    assert entries[2].immutable["corrected_grading_status"] == "loss"
    for entry in entries[:2]:
        assert entry.closing is not None
        assert entry.closing["closing_null_reason"] == "capture_missed"
    assert len(_events(run_log_path, "settle")) == 1
    assert len(_events(run_log_path, "correction_appended")) == 1
    assert len(_events(run_log_path, "closing_finalized")) == 2


def test_settle_nothing_to_do_writes_nothing(
    env: _Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    append_rows(env.ledger, [ats_row(GAMES[0])])
    env.write_games((GAMES[0], None, None))
    before = _sha(ledger_path(env.ledger))
    replaced = _count_replaces(monkeypatch)

    outcome = env.settle()

    assert outcome.changed is False
    assert replaced == []
    assert _sha(ledger_path(env.ledger)) == before


def test_settle_withdrawn_score_logged(env: _Env, run_log_path: Path) -> None:
    settled = ats_row(GAMES[0])
    append_rows(env.ledger, [settled])
    apply_updates(
        env.ledger, grading_updates={key_of(settled): graded("win", 0.9, 1.1)}
    )
    env.write_games((GAMES[0], None, None))
    env.write_manifest(owed_capture(8, 4, "2026-10-21T20:00:00+00:00"))

    outcome = env.settle()

    assert outcome.observations == 1
    (withdrawn,) = _events(run_log_path, "correction_label_withdrawn")
    assert withdrawn["key"][0] == GAMES[0]
    assert withdrawn["source"]["sequence"] == 4


# ---------------------------------------------------------------------------
# sync_after_run and the run log
# ---------------------------------------------------------------------------


def test_sync_after_run_never_raises(env: _Env, run_log_path: Path) -> None:
    def _broken(**_kwargs: Any) -> SyncOutcome:
        raise RuntimeError("git could not start")

    outcome = runner.sync_after_run(
        ledger_dir=env.ledger, repo_dir=env.root, publisher=_broken
    )

    assert outcome is None
    (refused,) = _events(run_log_path, "sync_refused")
    assert "git could not start" in refused["reason"]


def test_read_events_filters_by_event(tmp_path: Path) -> None:
    path = tmp_path / "ledger_runs.jsonl"
    assert run_log.read_events(path, "closing_capture") == []
    run_log.record_event("closing_capture", path=path, outcome="captured", order=1)
    run_log.record_event("append", path=path, appended_rows=3)
    run_log.record_event("closing_capture", path=path, outcome="skipped", order=2)

    captures = run_log.read_events(path, "closing_capture")

    assert [record["order"] for record in captures] == [1, 2]
    assert len(run_log.read_events(path)) == 3

    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(ValueError, match="line 4"):
        run_log.read_events(path)


def test_events_carry_the_bound_run_id(env: _Env, run_log_path: Path) -> None:
    def _publisher(*, ledger_dir: Path, repo_dir: Path, log: Any) -> SyncOutcome:
        log("anchor_push", ok=True, head_hash="h", entry_count=2)
        return SyncOutcome(False, True, None, False, False, None, None)

    with run_log.bound_run_id("r1"):
        assert run_log.current_run_id() == "r1"
        run_log.record_event("cutover", note="direct")
        env.record()
        env.write_games((GAMES[0], 24.0, 17.0), (GAMES[1], 20.0, 23.0))
        env.settle()
        runner.sync_after_run(
            ledger_dir=env.ledger, repo_dir=env.root, publisher=_publisher
        )
    run_log.record_event("cutover", note="outside")

    assert run_log.current_run_id() is None
    events = _events(run_log_path)
    inside, outside = events[:-1], events[-1]
    names = {record["event"] for record in inside}
    assert {"cutover", "append", "settle", "anchor_push"} <= names
    assert all(record["run_id"] == "r1" for record in inside)
    assert outside["run_id"] is None
    assert run_log.new_run_id() != run_log.new_run_id()
