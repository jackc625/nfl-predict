"""A decision after a re-fit swap decides, stamps and stores with that generation (Plan 37-10).

WHAT IS PROVEN (research Pitfall 1, D-19, D-20)
-----------------------------------------------
A fixture generation 1 is installed the way the re-fit hook installs one: its models are saved
under new ids, its bet-rule record is built by ``refit.remeasure.bet_rule_payload`` and written by
``refit.records.write_bet_rule_record``, and ``ledger/refit/dates.jsonl`` records the install
through ``refit.records.append_date_event`` -- the real producers, so the producer and the
consumer meet on the one stored path convention. A temporary ``latest.json`` then names the
generation-1 ids, and the REAL ``record_forward_slate`` decides the fixture week:

* the live rows are stamped ``recipe-2026-refit-v1`` with the generation-1 model and blend ids,
  and the decision used the generation's own thresholds (a WP threshold planted to admit nothing
  suppresses every WP row; each WP row the row-19 rule bets carries ``edge_below_threshold``);
* the frozen twins keep the pin and ``recipe-2026-row19-v1``, and the store accepts both arms;
* ids no install records are refused by name before anything is written;
* an install whose date never closed (a crash after the swap) still decides as generation 1;
* while ``latest.json`` names the pin, the decision call is exactly today's (D-04);
* a bet-rule file edited after install -- even to valid JSON -- is refused and nothing is written.

Every store lives under pytest's temporary directories: production ``artifacts/``, ``data/``,
``outputs/``, ``logs/`` and ``ledger/`` are never read or written (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.ats_ev_chain import ChainFit
from backtest.neutral_hfa_cold_start_constants import (
    CHAIN_FIT_BIAS_2026,
    EDGE_TIER_THRESHOLDS_BY_TARGET,
)
from backtest.refit_preregistration_2026 import REFIT_RECIPE_ID
from backtest.shadow_pin_2026 import SHADOW_RECIPE
from backtest.weekly_bet_list import (
    REFIT_GENERATION_BET_RULE_KIND,
    build_weekly_decision_bundle,
)
from data.upstream_live import AS_OF_ENV
from forward_ledger import run_log, runner
from forward_ledger.artifacts_copy import ARTIFACT_COPIES_DIRNAME
from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.schema import ARM_LIVE, ARM_SHADOW
from forward_ledger.shadow import build_shadow_bundle
from forward_ledger.snapshots import SNAPSHOTS_DIRNAME
from forward_ledger.stamps import IN_FORCE_RECIPE_ID
from forward_ledger.store import ledger_path, read_entries, verify_chain
from models.artifacts import ResolvedArtifacts
from refit.generations import (
    UnknownGenerationError,
    generation_of_row,
    installed_generations,
)
from refit.records import (
    DATES_FILENAME,
    RefitRecordFormatError,
    append_date_event,
    bet_rule_relpath,
    read_date_events,
    refit_dir,
    write_bet_rule_record,
)
from refit.remeasure import BET_RULE_RECORD_KIND, BetRuleNumbers, bet_rule_payload
from tests.fixtures.decision_frame import chain_fit_record
from tests.unit.test_replay_ledger import (
    BLEND_ID,
    CONVERTER_ID,
    DECIDED_AT,
    MODEL_IDS,
    _write_artifacts,
    _write_gold,
    _write_silver,
)
from tests.unit.test_shadow_arm import WEEK, Stores, _slate, _write_manifest

_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# The generation-1 ids: new model and blend ids, the same bound converter.
GENERATION_ONE_IDS = ResolvedArtifacts(
    wp="wp_20261103_220501",
    ats="ats_20261103_220502",
    ou="ou_20261103_220503",
    blend="blend_20261103_221000",
    converter=CONVERTER_ID,
)
ROW19_IDS = ResolvedArtifacts(
    wp=MODEL_IDS["wp"],
    ats=MODEL_IDS["ats"],
    ou=MODEL_IDS["ou"],
    blend=BLEND_ID,
    converter=CONVERTER_ID,
)

# A WP second-test threshold no edge can clear: a probability edge is below 1.
ADMIT_NO_WP: tuple[float, float] = (0.99, 0.98)

# A fixture code commit (append_date_event requires one on every install intent).
FIXTURE_CODE_COMMIT: str = "c" * 40
FIXTURE_LIST_WEEK: int = 8
_CUTOFF = datetime(2026, 11, 3, 22, 0, tzinfo=UTC)

# Today's builder keywords: the runner's generation-0 call passes exactly these.
TODAY_BUILDER_KEYWORDS: frozenset[str] = frozenset(
    {
        "artifacts_dir",
        "gold_dir",
        "silver_dir",
        "now",
        "excluded_game_ids",
        "resolved",
    }
)


# ---------------------------------------------------------------------------
# Cadence-date events, written only through refit.records.append_date_event
# ---------------------------------------------------------------------------


def _common(event: str, date_index: int, date: str, nfl_week: int) -> dict[str, Any]:
    return {
        "event": event,
        "list_week": FIXTURE_LIST_WEEK,
        "date_index": date_index,
        "date": date,
        "nfl_week": nfl_week,
    }


def ids_mapping(ids: ResolvedArtifacts) -> dict[str, Any]:
    """The ``ids`` an install intent records: wp, ats, ou, blend, converter."""
    return dataclasses.asdict(ids)


def intent_event(
    ids: ResolvedArtifacts,
    sha256: str,
    *,
    date_index: int = 1,
    date: str = "2026-11-03",
    nfl_week: int = 9,
) -> dict[str, Any]:
    """An ``install_intent`` naming *ids* and the bet rule *sha256* (the one stored path)."""
    return {
        **_common("install_intent", date_index, date, nfl_week),
        "ids": ids_mapping(ids),
        "bet_rule_path": bet_rule_relpath(sha256),
        "bet_rule_sha256": sha256,
        "code_commit": FIXTURE_CODE_COMMIT,
    }


def attempt_event(
    outcome: str,
    *,
    check: str | None = None,
    exception_class: str | None = None,
    date_index: int = 1,
    date: str = "2026-11-03",
    nfl_week: int = 9,
) -> dict[str, Any]:
    """An ``attempt`` with *outcome* (installed / refused / crashed)."""
    return {
        **_common("attempt", date_index, date, nfl_week),
        "cutoff_utc": _CUTOFF.isoformat(),
        "attempted_at_utc": _CUTOFF.isoformat(),
        "run_date_et": date,
        "code_commit": FIXTURE_CODE_COMMIT,
        "outcome": outcome,
        "check": check,
        "exception_class": exception_class,
        "before_ids": None,
        "after_ids": None,
    }


def closed_event(
    status: str,
    *,
    date_index: int = 1,
    date: str = "2026-11-03",
    nfl_week: int = 9,
) -> dict[str, Any]:
    """A ``closed`` event with *status* (ran / ran late / not run)."""
    return {
        **_common("closed", date_index, date, nfl_week),
        "status": status,
        "closed_at_utc": _CUTOFF.isoformat(),
        "before_ids": None,
        "after_ids": None,
    }


# ---------------------------------------------------------------------------
# The generation's bet rule, built and written by the real producers
# ---------------------------------------------------------------------------


def fixture_bet_rule_numbers(
    edge_tier_thresholds: Mapping[str, tuple[float, float]],
) -> BetRuleNumbers:
    """Bet-rule numbers matching the fixture's row-19 rule except for *edge_tier_thresholds*.

    The floors (0.0), residual SD (13.0) and 2026 bias (the committed overlay) are the fixture
    row-19 record's own, so the generation decides like the frozen arm everywhere a threshold
    does not differ.
    """
    bias = {target: {2026: float(CHAIN_FIT_BIAS_2026[target])} for target in _TARGETS}
    return BetRuleNumbers(
        chain_fits={
            target: ChainFit(
                target=target,
                frozen_sd=13.0,
                season_bias_by_season={},
                tune_fit_seasons=(),
                threshold_window="fixture",
                bias_pool_by_season={},
            )
            for target in _TARGETS
        },
        calibration_gate_passed=dict.fromkeys(_TARGETS),
        fallback_trigger=dict.fromkeys(_TARGETS),
        lean_by_target={target: bias[target][2026] for target in _TARGETS},
        season_bias_by_season=bias,
        sd_correction_bias_by_season={target: {} for target in _TARGETS},
        edge_tier_thresholds=dict(edge_tier_thresholds),
        ev_floors=dict.fromkeys(_TARGETS, 0.0),
        windows={},
        excluded_counts={},
        rows_by_target={},
        edges=pd.DataFrame(),
    )


def write_generation_bet_rule(
    ledger: Path,
    ids: ResolvedArtifacts,
    edge_tier_thresholds: Mapping[str, tuple[float, float]] | None = None,
    *,
    date_index: int = 1,
) -> tuple[Path, str]:
    """Write a generation's bet-rule record into *ledger*; return (path, sha256)."""
    thresholds = edge_tier_thresholds or {
        target: pair
        for target, pair in EDGE_TIER_THRESHOLDS_BY_TARGET.items()
        if pair is not None
    }
    payload = bet_rule_payload(
        fixture_bet_rule_numbers(thresholds),
        artifact_ids={target: str(value) for target, value in ids_mapping(ids).items()},
        cutoff_utc=_CUTOFF,
        list_week=FIXTURE_LIST_WEEK,
        date_index=date_index,
        blend_weights={"wp": 0.0, "ats": 0.0, "ou": 0.0},
    )
    return write_bet_rule_record(ledger, payload)


# ---------------------------------------------------------------------------
# The fixture stores and the installed generation (tmp only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InstalledFixture:
    """A fixture week's stores, a ledger directory and (when installed) generation 1's record."""

    stores: Stores
    ledger: Path
    bet_rule_path: Path | None
    bet_rule_sha256: str | None


def write_manifest_ids(artifacts: Path, ids: ResolvedArtifacts) -> None:
    """Point the temporary ``latest.json`` at *ids* (tests write it directly, never production)."""
    (artifacts / "latest.json").write_text(
        json.dumps({"wp": ids.wp, "ats": ids.ats, "ou": ids.ou, "blend": ids.blend}),
        encoding="utf-8",
    )


def build_stores(root: Path) -> Stores:
    """The fixture week's stores, with generation 1's artifacts saved beside the row-19 ones.

    Generation 1's models are the fixture models copied under the new ids, so the two arms
    score the same numbers and differ only where the bet rule differs.
    """
    stores = Stores(
        artifacts=root / "artifacts",
        gold=root / "gold",
        silver=root / "silver",
        manifests=root / "upstream_live",
        chain_fit=root / "chain_fit.json",
    )
    stores.artifacts.mkdir(parents=True)
    _write_artifacts(stores.artifacts)
    for target in _TARGETS:
        shutil.copytree(
            stores.artifacts / MODEL_IDS[target],
            stores.artifacts / GENERATION_ONE_IDS.model_id_for(target),
        )
    shutil.copytree(
        stores.artifacts / BLEND_ID, stores.artifacts / GENERATION_ONE_IDS.blend
    )
    _write_gold(stores.gold)
    _write_silver(stores.silver)
    _write_manifest(stores.manifests)
    stores.chain_fit.write_text(json.dumps(chain_fit_record(0.0)), encoding="utf-8")
    return stores


def install_generation_one(
    root: Path,
    *,
    edge_tier_thresholds: Mapping[str, tuple[float, float]] | None = None,
    closed: bool = True,
) -> InstalledFixture:
    """Install a fixture generation 1: its record, its intent (and its close), and latest.json."""
    stores = build_stores(root)
    ledger = root / "ledger"
    path, sha = write_generation_bet_rule(
        ledger, GENERATION_ONE_IDS, edge_tier_thresholds
    )
    assert append_date_event(ledger, intent_event(GENERATION_ONE_IDS, sha))
    if closed:
        assert append_date_event(ledger, attempt_event("installed"))
        assert append_date_event(ledger, closed_event("ran"))
    write_manifest_ids(stores.artifacts, GENERATION_ONE_IDS)
    return InstalledFixture(
        stores=stores, ledger=ledger, bet_rule_path=path, bet_rule_sha256=sha
    )


@pytest.fixture
def run_log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the ledger run log into *tmp_path*; production ``logs/`` is never written."""
    monkeypatch.delenv(AS_OF_ENV, raising=False)
    path = tmp_path / "logs" / "ledger_runs.jsonl"
    monkeypatch.setattr(run_log, "LEDGER_RUN_LOG", path)
    return path


def record(stores: Stores, ledger: Path, **kwargs: Any) -> runner.RecordOutcome:
    """One real ``record_forward_slate`` over the fixture stores, the frozen arm on."""
    options: dict[str, Any] = {
        "decided_at": DECIDED_AT[WEEK],
        "excluded_game_ids": frozenset(),
        "publish_by": None,
        "ledger_dir": ledger,
        "artifacts_dir": stores.artifacts,
        "gold_dir": stores.gold,
        "silver_dir": stores.silver,
        "manifest_dir": stores.manifests,
        "bundle_builder": partial(
            build_weekly_decision_bundle, chain_fit_path=stores.chain_fit
        ),
        "shadow_builder": partial(build_shadow_bundle, pin=stores.pin()),
    }
    options.update(kwargs)
    return runner.record_forward_slate(_slate(), **options)


def ledger_rows(ledger: Path) -> list[dict[str, Any]]:
    return [
        entry.immutable
        for entry in read_entries(ledger)
        if entry.kind == ENTRY_KIND_ROW
    ]


def _arm(rows: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["arm"] == arm]


def _quad(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row["game_id"], row["season"], row["week"], row["target"])


def _ledger_state(ledger: Path) -> tuple[bytes, list[str], list[str]]:
    """The ledger file's bytes and every snapshot and artifact-copy path under *ledger*."""

    def _listing(directory: Path) -> list[str]:
        if not directory.exists():
            return []
        return sorted(
            path.relative_to(ledger).as_posix() for path in directory.rglob("*")
        )

    return (
        ledger_path(ledger).read_bytes(),
        _listing(ledger / SNAPSHOTS_DIRNAME),
        _listing(ledger / ARTIFACT_COPIES_DIRNAME),
    )


# ---------------------------------------------------------------------------
# Task 1 (tracer): swap, then decide, stamp and store with generation 1
# ---------------------------------------------------------------------------


def test_decision_after_a_swap_uses_generation_1(
    tmp_path: Path, run_log_path: Path
) -> None:
    thresholds = {**EDGE_TIER_THRESHOLDS_BY_TARGET, "wp": ADMIT_NO_WP}
    installed = install_generation_one(
        tmp_path,
        edge_tier_thresholds={k: v for k, v in thresholds.items() if v is not None},
    )
    assert REFIT_GENERATION_BET_RULE_KIND == BET_RULE_RECORD_KIND
    generations = installed_generations(installed.ledger)
    assert [g.number for g in generations] == [0, 1]
    assert generations[1].ids == GENERATION_ONE_IDS
    assert generations[1].bet_rule_path == installed.bet_rule_path

    outcome = record(installed.stores, installed.ledger)

    assert verify_chain(read_entries(installed.ledger)).ok
    rows = ledger_rows(installed.ledger)
    live, shadow = _arm(rows, ARM_LIVE), _arm(rows, ARM_SHADOW)
    assert live, "the fixture week decided nothing, so nothing was tested"
    assert outcome.appended == len(live)
    assert sorted(_quad(row) for row in shadow) == sorted(_quad(row) for row in live)
    for row in live:
        assert row["recipe_id"] == REFIT_RECIPE_ID
        assert row["model_artifact_id"] == GENERATION_ONE_IDS.model_id_for(
            row["target"]
        )
        assert row["blend_id"] == GENERATION_ONE_IDS.blend
        assert generation_of_row(row["model_artifact_id"], ARM_LIVE, generations) == 1
    for row in shadow:
        assert row["recipe_id"] == SHADOW_RECIPE
        assert row["model_artifact_id"] == MODEL_IDS[row["target"]]
        assert row["blend_id"] == BLEND_ID
        assert generation_of_row(row["model_artifact_id"], ARM_SHADOW, generations) == 0

    # The generation's own thresholds decided: no WP bet, and every WP game the row-19 rule
    # bets is suppressed by the planted second-test threshold.
    live_by_quad = {_quad(row): row for row in live}
    wp_live = [row for row in live if row["target"] == "wp"]
    assert wp_live
    assert {row["status"] for row in wp_live} == {"suppressed"}
    row19_bets = [
        row for row in shadow if row["target"] == "wp" and row["status"] == "live"
    ]
    assert row19_bets, (
        "the row-19 rule bets no WP game, so the threshold was not tested"
    )
    for twin in row19_bets:
        assert live_by_quad[_quad(twin)]["rejection_reason"] == "edge_below_threshold"


def test_an_unregistered_id_is_refused_by_name(
    tmp_path: Path, run_log_path: Path
) -> None:
    stores = build_stores(tmp_path)
    ledger = tmp_path / "ledger"
    write_manifest_ids(stores.artifacts, GENERATION_ONE_IDS)

    with pytest.raises(UnknownGenerationError) as raised:
        record(stores, ledger)

    message = str(raised.value)
    for value in ids_mapping(GENERATION_ONE_IDS).values():
        assert str(value) in message
    assert not ledger.exists(), "a refused decision wrote into the ledger"


def test_pending_install_after_a_crash_is_accepted(
    tmp_path: Path, run_log_path: Path
) -> None:
    installed = install_generation_one(tmp_path, closed=False)
    assert [g.number for g in installed_generations(installed.ledger)] == [0, 1]

    record(installed.stores, installed.ledger)

    live = _arm(ledger_rows(installed.ledger), ARM_LIVE)
    assert live
    for row in live:
        assert row["recipe_id"] == REFIT_RECIPE_ID
        assert row["model_artifact_id"] == GENERATION_ONE_IDS.model_id_for(
            row["target"]
        )


def test_a_torn_dates_tail_never_blocks_a_generation_1_decision(
    tmp_path: Path, run_log_path: Path
) -> None:
    """WR-A02: a crash mid-append of a re-fit record never stands between a game and its lock."""
    installed = install_generation_one(tmp_path)
    dates = refit_dir(installed.ledger) / DATES_FILENAME
    dates.write_bytes(dates.read_bytes() + b'{"event": "attempt", "list_wee')
    # The re-fit hook's strict read still refuses the torn line.
    with pytest.raises(RefitRecordFormatError):
        read_date_events(installed.ledger)

    outcome = record(installed.stores, installed.ledger)

    live = _arm(ledger_rows(installed.ledger), ARM_LIVE)
    assert live
    assert outcome.appended == len(live)
    for row in live:
        assert row["recipe_id"] == REFIT_RECIPE_ID
        assert row["model_artifact_id"] == GENERATION_ONE_IDS.model_id_for(
            row["target"]
        )


def test_generation_zero_call_is_unchanged(tmp_path: Path, run_log_path: Path) -> None:
    # A generation-1 install is on record, but latest.json names the pin: generation 0.
    installed = install_generation_one(tmp_path)
    write_manifest_ids(installed.stores.artifacts, ROW19_IDS)
    calls: list[dict[str, Any]] = []
    real = partial(
        build_weekly_decision_bundle, chain_fit_path=installed.stores.chain_fit
    )

    def _recording(season: int, week: int, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(season, week, **kwargs)

    record(installed.stores, installed.ledger, bundle_builder=_recording)

    assert len(calls) == 1
    assert set(calls[0]) == TODAY_BUILDER_KEYWORDS
    assert calls[0]["resolved"] == ROW19_IDS
    live = _arm(ledger_rows(installed.ledger), ARM_LIVE)
    assert live
    assert {row["recipe_id"] for row in live} == {IN_FORCE_RECIPE_ID}
    assert {row["model_artifact_id"] for row in live} <= set(MODEL_IDS.values())


def test_a_bet_rule_edited_after_install_is_refused(
    tmp_path: Path, run_log_path: Path
) -> None:
    installed = install_generation_one(tmp_path)
    record(installed.stores, installed.ledger)
    before = _ledger_state(installed.ledger)
    assert installed.bet_rule_path is not None

    edited = json.loads(installed.bet_rule_path.read_text(encoding="ascii"))
    edited["edge_tier_thresholds"]["wp"] = list(ADMIT_NO_WP)
    installed.bet_rule_path.write_text(json.dumps(edited), encoding="ascii")

    with pytest.raises(RefitRecordFormatError) as raised:
        record(installed.stores, installed.ledger)

    assert installed.bet_rule_path.as_posix() in str(raised.value)
    assert _ledger_state(installed.ledger) == before
