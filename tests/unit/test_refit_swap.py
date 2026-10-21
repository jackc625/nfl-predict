"""The gate-free go-live: integrity checks, then the one swap surface, intent first (Plan 37-11).

WHAT IS PROVEN (RFIT-06, SPEC R5, D-25, D-34, research Pitfalls 1 and 2)
-----------------------------------------------------------------------
* A passing fixture generation is installed through ``models.artifacts.replace_manifest`` -- the
  four pointers in one write -- and its ``install_intent`` is on disk BEFORE that write.
* The pre-registered ``clean_code_commit`` check refuses an unknown or dirty code commit before
  anything is read or written, so the permanent install record never names one.
* A missing bet-rule record, a declined intent write and a bundle the serving loaders refuse each
  stop the install with ``latest.json`` byte-unchanged.
* ``pending_install`` finds an open date whose last intent ``latest.json`` serves, whatever attempt
  follows that intent (a crash can come after the swap wrote).
* An installed generation round-trips: its record, its intent and its close lead the REAL decision
  path to decide and stamp ``recipe-2026-refit-v1`` with the generation's ids.

Every artifacts dir, ``latest.json``, ledger and gold here is a pytest temporary fixture; the
production ``artifacts/latest.json`` is never read or written, and every install passes the clean
fixture commit ``"c" * 40`` explicitly, so no test depends on the developer's working tree.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest

from backtest.refit_preregistration_2026 import (
    EV_FLOORS,
    EXCLUDED_FEATURE_GROUPS,
    GROUP_VERDICT_NORMALIZED_SHA256,
    INTEGRITY_CHECKS,
    MODEL_SETTINGS,
    REFIT_RECIPE_ID,
    THREAD_LIMIT,
)
from backtest.weekly_bet_list import load_bet_rule_record
from data.write_sink import RecordingSink, active_sink
from forward_ledger.schema import ARM_LIVE
from forward_ledger.store import read_entries, verify_chain
from models.artifacts import ResolvedArtifacts
from models.blending import BLEND_PAYLOAD_FILENAME, BlendProvenance
from models.train import (
    GOLD_GENERATION_DIGEST_METADATA_KEY,
    GROUP_VERDICT_DIGEST_METADATA_KEY,
    THREAD_LIMIT_METADATA_KEY,
)
from refit import swap
from refit.cadence import CadenceDate, cadence_dates
from refit.generations import installed_generations
from refit.integrity import IntegrityRefusal, check_aa_proof, run_integrity_checks
from refit.records import (
    append_date_event,
    bet_rule_relpath,
    read_date_events,
)
from refit.remeasure import BetRuleNumbers
from refit.swap import (
    InstallIntentNotRecordedError,
    install_generation,
    pending_install,
)
from refit.training import GenerationTraining
from refit.witness import PREREGISTRATION_PATHS, REPO_ROOT
from tests.unit.test_refit_decision_path import (
    GENERATION_ONE_IDS,
    attempt_event,
    build_stores,
    closed_event,
    fixture_bet_rule_numbers,
    intent_event,
    ledger_rows,
    record,
    run_log_path,  # noqa: F401 - a pytest fixture, used by name
    write_generation_bet_rule,
)
from tests.unit.test_shadow_arm import Stores

# The clean fixture code commit every install in this file passes explicitly.
CLEAN_COMMIT: str = "c" * 40

GOLD_DIGEST: str = "gold-generation-digest-refit-swap"
CONVERTER_ID: str = "market_probability_20260923_195443"
CONVERTER_SLOPE: float = 0.15

# Generation 0 (served before the install) and generation 1 (installed), as fixture ids.
OLD_IDS = ResolvedArtifacts(
    wp="wp_20261004_050223",
    ats="ats_20261004_050228",
    ou="ou_20261004_050232",
    blend="blend_20261004_050521",
    converter=CONVERTER_ID,
)
NEW_IDS = ResolvedArtifacts(
    wp="wp_20261103_220501",
    ats="ats_20261103_220502",
    ou="ou_20261103_220503",
    blend="blend_20261103_221000",
    converter=CONVERTER_ID,
)

# A manifest key outside the four pointers: the swap must leave it untouched.
EXTRA_MANIFEST_KEY: str = "note"
EXTRA_MANIFEST_VALUE: str = "kept"

_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")


# ---------------------------------------------------------------------------
# Fixture bundles (tmp only)
# ---------------------------------------------------------------------------


def ids_of(ids: ResolvedArtifacts) -> dict[str, str]:
    """The five ids an install names, as text."""
    return {key: str(value) for key, value in dataclasses.asdict(ids).items()}


def first_date() -> CadenceDate:
    """Date 1 of the week-8 list (Tue 2026-11-03), the decision-path fixture's date."""
    return cadence_dates(8)[0]


def _write_model(root: Path, target: str, version: str) -> None:
    """A model directory with the file set ``load_model_artifact`` reads."""
    directory = root / version
    directory.mkdir(parents=True)
    joblib.dump({"estimator": target}, directory / "model.pkl")
    (directory / "metadata.json").write_text(
        json.dumps({"target": target, "gold_generation_digest": GOLD_DIGEST}),
        encoding="utf-8",
    )
    (directory / "feature_list.json").write_text(
        json.dumps(["f1", "f2"]), encoding="utf-8"
    )


def write_blend(
    root: Path,
    version: str,
    sources: Mapping[str, str],
    *,
    converter: str = CONVERTER_ID,
    digest: str = GOLD_DIGEST,
) -> None:
    """A blend payload bound to *converter*, whose provenance names *sources*."""
    provenance = BlendProvenance(
        gold_generation_digest=digest,
        source_artifact_ids=dict(sources),
        tuning_corpus_rows=100,
        excluded_counts={"no_prelock_line": 0},
        thread_limit=1,
    )
    directory = root / version
    directory.mkdir(parents=True, exist_ok=True)
    (directory / BLEND_PAYLOAD_FILENAME).write_text(
        json.dumps(
            {
                "weights": {"wp": 0.0, "ats": 0.0, "ou": 0.0},
                "market_probability_artifact_id": converter,
                "market_probability_slope_beta": CONVERTER_SLOPE,
                **provenance.to_payload(),
            }
        ),
        encoding="utf-8",
    )


def _write_converter(root: Path, converter: str = CONVERTER_ID) -> None:
    directory = root / converter
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps({"artifact_id": converter, "slope_beta": CONVERTER_SLOPE}),
        encoding="utf-8",
    )


def write_bundle(
    root: Path,
    ids: ResolvedArtifacts,
    *,
    blend_sources: Mapping[str, str] | None = None,
) -> None:
    """The three models and the blend of *ids* (the blend tuned on *blend_sources*)."""
    for target in _TARGETS:
        _write_model(root, target, ids.model_id_for(target))
    sources = blend_sources or {t: ids.model_id_for(t) for t in _TARGETS}
    write_blend(root, ids.blend, sources)


def write_latest(artifacts: Path, ids: ResolvedArtifacts) -> Path:
    """Point the temporary ``latest.json`` at *ids*, beside one key the swap must keep."""
    path = artifacts / "latest.json"
    path.write_text(
        json.dumps(
            {
                "wp": ids.wp,
                "ats": ids.ats,
                "ou": ids.ou,
                "blend": ids.blend,
                EXTRA_MANIFEST_KEY: EXTRA_MANIFEST_VALUE,
            }
        ),
        encoding="utf-8",
    )
    return path


def build_artifacts(
    root: Path, *, new_blend_sources: Mapping[str, str] | None = None
) -> Path:
    """An artifacts dir holding the old and the new bundle, latest.json serving the old one."""
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    _write_converter(artifacts)
    write_bundle(artifacts, OLD_IDS)
    write_bundle(artifacts, NEW_IDS, blend_sources=new_blend_sources)
    write_latest(artifacts, OLD_IDS)
    return artifacts


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(artifacts: Path) -> dict[str, Any]:
    return json.loads((artifacts / "latest.json").read_text(encoding="utf-8"))


def recording_replace_manifest(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ledger: Path | None = None,
    passthrough: bool = True,
) -> list[dict[str, Any]]:
    """Patch ``refit.swap.replace_manifest`` to record each call (and the events at that moment)."""
    calls: list[dict[str, Any]] = []
    real: Callable[..., None] = swap.replace_manifest

    def recording(mapping: Mapping[str, str], artifacts_dir: Path) -> None:
        calls.append(
            {
                "mapping": dict(mapping),
                "events": [] if ledger is None else read_date_events(ledger),
                "latest": json.loads(
                    (Path(artifacts_dir) / "latest.json").read_text(encoding="utf-8")
                ),
            }
        )
        if passthrough:
            real(mapping, artifacts_dir)

    monkeypatch.setattr(swap, "replace_manifest", recording)
    return calls


def write_record(ledger: Path, ids: ResolvedArtifacts) -> str:
    """Write a generation bet-rule record naming *ids* into *ledger*; return its sha256."""
    _path, sha = write_generation_bet_rule(ledger, ids)
    return sha


def install(ledger: Path, artifacts: Path, sha: str, **kwargs: Any) -> Any:
    """``install_generation`` of NEW_IDS on date 1 with the clean fixture commit."""
    options: dict[str, Any] = {
        "date": first_date(),
        "ids": ids_of(NEW_IDS),
        "bet_rule_sha256": sha,
        "code_commit": CLEAN_COMMIT,
    }
    options.update(kwargs)
    return install_generation(ledger, artifacts, **options)


# ---------------------------------------------------------------------------
# Task 1 (tracer): the swap, intent first
# ---------------------------------------------------------------------------


def test_a_passing_generation_swaps_four_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = build_artifacts(tmp_path)
    ledger = tmp_path / "ledger"
    sha = write_record(ledger, NEW_IDS)
    calls = recording_replace_manifest(monkeypatch, ledger=ledger)

    before, after = install(ledger, artifacts, sha)

    written = manifest(artifacts)
    for key in ("wp", "ats", "ou", "blend"):
        assert written[key] == getattr(NEW_IDS, key)
    assert written[EXTRA_MANIFEST_KEY] == EXTRA_MANIFEST_VALUE
    assert before == ids_of(OLD_IDS)
    assert after == ids_of(NEW_IDS)

    # The intent was on disk when the swap was called, and latest.json still named the old ids.
    assert len(calls) == 1
    assert calls[0]["mapping"] == {
        key: getattr(NEW_IDS, key) for key in ("wp", "ats", "ou", "blend")
    }
    assert calls[0]["latest"]["wp"] == OLD_IDS.wp
    intents = [e for e in calls[0]["events"] if e["event"] == "install_intent"]
    assert len(intents) == 1
    intent = intents[0]
    assert intent["ids"] == ids_of(NEW_IDS)
    assert intent["bet_rule_path"] == bet_rule_relpath(sha)
    assert intent["bet_rule_sha256"] == sha
    assert intent["code_commit"] == CLEAN_COMMIT
    date = first_date()
    assert (intent["list_week"], intent["date_index"], intent["nfl_week"]) == (
        date.list_week,
        date.index,
        date.nfl_week,
    )
    assert intent["date"] == date.date.isoformat()


def test_a_missing_bet_rule_record_refuses_before_the_intent(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path)
    ledger = tmp_path / "ledger"
    latest_sha = sha256_of(artifacts / "latest.json")

    with pytest.raises(FileNotFoundError):
        install(ledger, artifacts, "a" * 64)

    assert read_date_events(ledger) == []
    assert sha256_of(artifacts / "latest.json") == latest_sha


def test_a_bundle_the_validators_refuse_is_bundle_validation(tmp_path: Path) -> None:
    # The new blend was tuned on the OLD models, so it would serve beside models it never saw.
    artifacts = build_artifacts(
        tmp_path, new_blend_sources={t: OLD_IDS.model_id_for(t) for t in _TARGETS}
    )
    ledger = tmp_path / "ledger"
    sha = write_record(ledger, NEW_IDS)
    latest_sha = sha256_of(artifacts / "latest.json")

    with pytest.raises(IntegrityRefusal) as raised:
        install(ledger, artifacts, sha)

    assert raised.value.check == "bundle_validation"
    assert "bundle_validation" in str(raised.value)
    assert sha256_of(artifacts / "latest.json") == latest_sha


@pytest.mark.parametrize("follow_up", [None, "installed", "crashed"])
def test_pending_install_is_found(tmp_path: Path, follow_up: str | None) -> None:
    artifacts = build_artifacts(tmp_path)
    ledger = tmp_path / "ledger"
    sha = write_record(ledger, NEW_IDS)
    install(ledger, artifacts, sha)
    if follow_up is not None:
        assert append_date_event(ledger, attempt_event(follow_up))

    assert pending_install(ledger, artifacts) == first_date().index

    # latest.json naming other ids: the intent was never served, so nothing is pending.
    write_latest(artifacts, OLD_IDS)
    assert pending_install(ledger, artifacts) is None


def test_pending_install_is_none_without_an_intent(tmp_path: Path) -> None:
    artifacts = build_artifacts(tmp_path)
    assert pending_install(tmp_path / "ledger", artifacts) is None


def test_a_declined_intent_write_refuses_the_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = build_artifacts(tmp_path)
    ledger = tmp_path / "ledger"
    sha = write_record(ledger, NEW_IDS)
    calls = recording_replace_manifest(monkeypatch)
    latest_sha = sha256_of(artifacts / "latest.json")

    with (
        active_sink(RecordingSink()) as sink,
        pytest.raises(InstallIntentNotRecordedError),
    ):
        install(ledger, artifacts, sha)

    assert any("dates.jsonl" in write.target for write in sink.intended_writes)
    assert calls == []
    assert read_date_events(ledger) == []
    assert sha256_of(artifacts / "latest.json") == latest_sha


@pytest.mark.parametrize("code_commit", [CLEAN_COMMIT + "+dirty", None])
def test_a_dirty_or_unknown_code_commit_refuses_before_the_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code_commit: str | None
) -> None:
    artifacts = build_artifacts(tmp_path)
    ledger = tmp_path / "ledger"
    sha = write_record(ledger, NEW_IDS)
    calls = recording_replace_manifest(monkeypatch)
    events_before = read_date_events(ledger)
    latest_sha = sha256_of(artifacts / "latest.json")

    with pytest.raises(IntegrityRefusal) as raised:
        install(ledger, artifacts, sha, code_commit=code_commit)

    assert raised.value.check == "clean_code_commit"
    assert ("dirty" if code_commit else "unknown") in str(raised.value)
    assert read_date_events(ledger) == events_before
    assert calls == []
    assert sha256_of(artifacts / "latest.json") == latest_sha


# ---------------------------------------------------------------------------
# The round trip: record -> install -> generations -> bet rule -> one live decision
# ---------------------------------------------------------------------------


def make_generation_one_installable(artifacts: Path) -> None:
    """Give the decision-path fixture's generation-1 copies what the swap validates.

    The copies carry the fixture models' metadata, which records no target and no gold, and the
    copied blend carries no provenance; the trainer and the blend tuner write both for a real
    generation. The row-19 copies are left exactly as the decision path reads them.
    """
    for target in _TARGETS:
        path = artifacts / GENERATION_ONE_IDS.model_id_for(target) / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update({"target": target, "gold_generation_digest": GOLD_DIGEST})
        path.write_text(json.dumps(metadata), encoding="utf-8")
    payload_path = artifacts / GENERATION_ONE_IDS.blend / BLEND_PAYLOAD_FILENAME
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    provenance = BlendProvenance(
        gold_generation_digest=GOLD_DIGEST,
        source_artifact_ids={t: GENERATION_ONE_IDS.model_id_for(t) for t in _TARGETS},
        tuning_corpus_rows=100,
        excluded_counts={"no_prelock_line": 0},
        thread_limit=1,
    )
    payload.update(provenance.to_payload())
    payload_path.write_text(json.dumps(payload), encoding="utf-8")


def test_an_installed_generation_round_trips_to_a_decision(
    tmp_path: Path,
    run_log_path: Path,  # noqa: F811 - the imported fixture, requested by name
) -> None:
    stores = build_stores(tmp_path)
    make_generation_one_installable(stores.artifacts)
    ledger = tmp_path / "ledger"
    path, sha = write_generation_bet_rule(ledger, GENERATION_ONE_IDS)

    before, after = install_generation(
        ledger,
        stores.artifacts,
        date=first_date(),
        ids=ids_of(GENERATION_ONE_IDS),
        bet_rule_sha256=sha,
        code_commit=CLEAN_COMMIT,
    )
    assert after == ids_of(GENERATION_ONE_IDS)
    assert before["wp"] != after["wp"]
    assert append_date_event(ledger, attempt_event("installed"))
    assert append_date_event(ledger, closed_event("ran"))

    generations = installed_generations(ledger)
    assert [g.number for g in generations] == [0, 1]
    assert generations[1].ids == GENERATION_ONE_IDS
    assert generations[1].bet_rule_path == path
    fits, thresholds = load_bet_rule_record(generations[1].bet_rule_path)
    assert set(fits) == set(_TARGETS)
    assert set(thresholds) == set(_TARGETS)

    record(stores, ledger)

    assert verify_chain(read_entries(ledger)).ok
    live = [row for row in ledger_rows(ledger) if row["arm"] == ARM_LIVE]
    assert live, "the fixture week decided nothing, so nothing was tested"
    for row in live:
        assert row["recipe_id"] == REFIT_RECIPE_ID
        assert row["model_artifact_id"] == GENERATION_ONE_IDS.model_id_for(
            row["target"]
        )
        assert row["blend_id"] == GENERATION_ONE_IDS.blend


def test_the_intent_helper_and_the_swap_store_one_path_form(tmp_path: Path) -> None:
    """The fixture intent (Plan 37-10) and the swap's intent store the same bet-rule path text."""
    artifacts = build_artifacts(tmp_path)
    ledger = tmp_path / "ledger"
    sha = write_record(ledger, NEW_IDS)
    install(ledger, artifacts, sha)

    written = next(
        e for e in read_date_events(ledger) if e["event"] == "install_intent"
    )
    fixture = intent_event(NEW_IDS, sha)
    assert written["bet_rule_path"] == fixture["bet_rule_path"]
    assert written["ids"] == fixture["ids"]


# ---------------------------------------------------------------------------
# Task 2: the pre-registered checks, each refusing by name with latest.json untouched
# ---------------------------------------------------------------------------

LIST_WEEK: int = 8
CLEAN_WEEK_8: dict[str, Any] = {"season": 2026, "week": 8, "clean": True}
UNCLEAN_WEEK_8: dict[str, Any] = {
    "season": 2026,
    "week": 8,
    "clean": False,
    "reason": "field_mismatch",
}
FIXTURE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "wp": (0.05, 0.02),
    "ats": (1.0, 0.5),
    "ou": (0.03, 0.01),
}
USABLE_ROWS: int = 150
# A helper-season game (2024) planted among the calibration games.
HELPER_SEASON_GAME: str = "2024_01_AAA_BBB"
# An injury-group input, which the frozen pool excludes.
EXCLUDED_INPUT: str = "home_qb_out_flag"


def history_games() -> pd.DataFrame:
    """Two finished games a season, 2002-2025: the training and calibration history."""
    rows = [
        {
            "game_id": f"{season}_{week:02d}_AAA_BBB",
            "season": season,
            "week": week,
            "kickoff_et": pd.Timestamp(
                datetime(season, 9, 7 * week, 17, 0, tzinfo=UTC)
            ),
            "home_score": float(20 + week),
            "away_score": float(17),
        }
        for season in range(2002, 2026)
        for week in (1, 2)
    ]
    # The rest of the 2026 schedule: a game after the cutoff, so 2026 is not a complete season.
    rows.append(
        {
            "game_id": "2026_18_GGG_HHH",
            "season": 2026,
            "week": 18,
            "kickoff_et": pd.Timestamp(datetime(2027, 1, 10, 18, 0, tzinfo=UTC)),
            "home_score": float("nan"),
            "away_score": float("nan"),
        }
    )
    return pd.DataFrame(rows)


def training_record(games: pd.DataFrame, cutoff: datetime) -> GenerationTraining:
    """What train_generation would record for generation 1 over *games* at *cutoff*."""
    kickoff = pd.to_datetime(games["kickoff_et"], utc=True)
    scored = games["home_score"].notna() & games["away_score"].notna()
    before = kickoff < pd.Timestamp(cutoff)
    no_final_score = sorted(games.loc[before & ~scored, "game_id"].astype(str))
    calibration_ids = sorted(games.loc[games["season"] == 2025, "game_id"].astype(str))
    metadata = {
        target: {
            "target": target,
            "feature_names": ["f1", "f2"],
            "refit_recipe": REFIT_RECIPE_ID,
            "cutoff_utc": cutoff.isoformat(),
            "training_rows": int((before & scored).sum()),
            "exclusions": {"no_final_score": no_final_score, "no_gold_row": []},
            "selected_inputs": ["f1", "f2"],
            "calibration": {
                "season": 2025,
                "helper_seasons": list(range(2002, 2025)),
                "helper_inputs": ["f1", "f2"],
                "game_ids": calibration_ids,
                "method": "platt" if target == "wp" else "normal",
            },
            "best_params": dict(MODEL_SETTINGS[target]),
            "exclude_groups": list(EXCLUDED_FEATURE_GROUPS),
            "exclude_groups_provenance": "verdict",
            GROUP_VERDICT_DIGEST_METADATA_KEY: GROUP_VERDICT_NORMALIZED_SHA256,
            THREAD_LIMIT_METADATA_KEY: THREAD_LIMIT,
            GOLD_GENERATION_DIGEST_METADATA_KEY: GOLD_DIGEST,
        }
        for target in _TARGETS
    }
    return GenerationTraining(
        artifact_ids={t: GENERATION_ONE_IDS.model_id_for(t) for t in _TARGETS},
        metadata=metadata,
        gold_generation_digest=GOLD_DIGEST,
        no_final_score_ids=tuple(no_final_score),
    )


def remeasured_numbers() -> BetRuleNumbers:
    """Generation 1's re-measured numbers: 150 usable rows a target, the fixed EV floors."""
    return dataclasses.replace(
        fixture_bet_rule_numbers(FIXTURE_THRESHOLDS),
        ev_floors=dict(EV_FLOORS),
        rows_by_target={
            target: {"usable": USABLE_ROWS, "residual_pool": 180, "edge_rows": 150}
            for target in _TARGETS
        },
    )


@dataclass(frozen=True)
class Baseline:
    """A generation 1 that passes every check: stores, its record, its schedule."""

    stores: Stores
    ledger: Path
    sha: str
    games: pd.DataFrame

    def check_options(self) -> dict[str, Any]:
        """The keywords of run_integrity_checks for this generation, every check passing."""
        cutoff = first_date().cutoff_utc
        return {
            "repo_root": REPO_ROOT,
            "list_week": LIST_WEEK,
            "aa_records": [CLEAN_WEEK_8],
            "first_refit": True,
            "training": training_record(self.games, cutoff),
            "games": self.games,
            "cutoff_utc": cutoff,
            "numbers": remeasured_numbers(),
        }


def build_baseline(root: Path) -> Baseline:
    stores = build_stores(root)
    make_generation_one_installable(stores.artifacts)
    ledger = root / "ledger"
    _path, sha = write_generation_bet_rule(ledger, GENERATION_ONE_IDS)
    silver = pd.read_parquet(stores.silver / "games.parquet")
    games = pd.concat([history_games(), silver], ignore_index=True)
    return Baseline(stores=stores, ledger=ledger, sha=sha, games=games)


def attempt(baseline: Baseline, options: Mapping[str, Any]) -> None:
    """The hook's order: every pre-swap check, then the install."""
    run_integrity_checks(**options)
    install_generation(
        baseline.ledger,
        baseline.stores.artifacts,
        date=first_date(),
        ids=ids_of(GENERATION_ONE_IDS),
        bet_rule_sha256=baseline.sha,
        code_commit=CLEAN_COMMIT,
    )


def _with_metadata(
    training: GenerationTraining, target: str, **changes: Any
) -> GenerationTraining:
    metadata = {name: dict(values) for name, values in training.metadata.items()}
    metadata[target].update(changes)
    return dataclasses.replace(training, metadata=metadata)


# Each plant takes the baseline, its check options and a scratch dir, and returns the options
# with one planted failure (or plants it in the stores and returns the options unchanged).
Plant = Callable[[Baseline, dict[str, Any], Path], dict[str, Any]]


def _plant_edited_record(
    _baseline: Baseline, options: dict[str, Any], scratch: Path
) -> dict[str, Any]:
    repo = scratch / "repo"
    for relative in PREREGISTRATION_PATHS:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    edited = repo / PREREGISTRATION_PATHS[0]
    edited.write_bytes(edited.read_bytes() + b"x")
    return {**options, "repo_root": repo}


def _plant_unclean_aa(
    _baseline: Baseline, options: dict[str, Any], _scratch: Path
) -> dict[str, Any]:
    return {**options, "aa_records": [UNCLEAN_WEEK_8]}


def _plant_row_count_off_by_one(
    _baseline: Baseline, options: dict[str, Any], _scratch: Path
) -> dict[str, Any]:
    training = options["training"]
    rows = training.metadata["ou"]["training_rows"] + 1
    return {**options, "training": _with_metadata(training, "ou", training_rows=rows)}


def _plant_row_at_the_cutoff(
    _baseline: Baseline, options: dict[str, Any], _scratch: Path
) -> dict[str, Any]:
    at_cutoff = pd.DataFrame(
        [
            {
                "game_id": "2026_09_EEE_FFF",
                "season": 2026,
                "week": 9,
                "kickoff_et": pd.Timestamp(first_date().cutoff_utc),
                "home_score": 20.0,
                "away_score": 10.0,
            }
        ]
    )
    games = pd.concat([options["games"], at_cutoff], ignore_index=True)
    training = options["training"]
    for target in _TARGETS:
        counted = training.metadata[target]["training_rows"] + 1
        training = _with_metadata(training, target, training_rows=counted)
    return {**options, "games": games, "training": training}


def _plant_excluded_input(
    _baseline: Baseline, options: dict[str, Any], _scratch: Path
) -> dict[str, Any]:
    names = ["f1", "f2", EXCLUDED_INPUT]
    training = _with_metadata(
        options["training"], "wp", feature_names=names, selected_inputs=names
    )
    return {**options, "training": training}


def _plant_over_the_cap(
    _baseline: Baseline, options: dict[str, Any], _scratch: Path
) -> dict[str, Any]:
    names = [f"f{index}" for index in range(21)]
    training = _with_metadata(
        options["training"], "wp", feature_names=names, selected_inputs=names
    )
    return {**options, "training": training}


def _plant_non_default_setting(
    _baseline: Baseline, options: dict[str, Any], _scratch: Path
) -> dict[str, Any]:
    settings = {**MODEL_SETTINGS["ats"], "n_estimators": 300}
    training = _with_metadata(options["training"], "ats", best_params=settings)
    return {**options, "training": training}


def _plant_two_threads(
    _baseline: Baseline, options: dict[str, Any], _scratch: Path
) -> dict[str, Any]:
    training = _with_metadata(
        options["training"], "wp", **{THREAD_LIMIT_METADATA_KEY: 2}
    )
    return {**options, "training": training}


def _plant_helper_game_in_calibration(
    _baseline: Baseline, options: dict[str, Any], _scratch: Path
) -> dict[str, Any]:
    training = options["training"]
    calibration = dict(training.metadata["wp"]["calibration"])
    calibration["game_ids"] = [*calibration["game_ids"], HELPER_SEASON_GAME]
    return {
        **options,
        "training": _with_metadata(training, "wp", calibration=calibration),
    }


def _plant_99_usable_rows(
    _baseline: Baseline, options: dict[str, Any], _scratch: Path
) -> dict[str, Any]:
    numbers = options["numbers"]
    rows = {target: dict(values) for target, values in numbers.rows_by_target.items()}
    rows["wp"]["usable"] = 99
    return {**options, "numbers": dataclasses.replace(numbers, rows_by_target=rows)}


def _plant_moved_floor(
    _baseline: Baseline, options: dict[str, Any], _scratch: Path
) -> dict[str, Any]:
    numbers = options["numbers"]
    floors = {**numbers.ev_floors, "ou": 0.0}
    return {**options, "numbers": dataclasses.replace(numbers, ev_floors=floors)}


def _plant_blend_on_other_models(
    baseline: Baseline, options: dict[str, Any], _scratch: Path
) -> dict[str, Any]:
    payload_path = (
        baseline.stores.artifacts / GENERATION_ONE_IDS.blend / BLEND_PAYLOAD_FILENAME
    )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["source_artifact_ids"] = {t: OLD_IDS.model_id_for(t) for t in _TARGETS}
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    return options


PLANTS: dict[str, list[Plant]] = {
    "recipe_fingerprint": [_plant_edited_record],
    "aa_proof": [_plant_unclean_aa],
    "training_scope": [_plant_row_count_off_by_one, _plant_row_at_the_cutoff],
    "input_pool_and_caps": [_plant_excluded_input, _plant_over_the_cap],
    "fixed_settings": [_plant_non_default_setting, _plant_two_threads],
    "out_of_sample_calibration": [_plant_helper_game_in_calibration],
    "remeasure_complete": [_plant_99_usable_rows, _plant_moved_floor],
    "bundle_validation": [_plant_blend_on_other_models],
}

PLANTED_CASES: list[Any] = [
    pytest.param(check, plant, id=f"{check}-{plant.__name__.removeprefix('_plant_')}")
    for check, plants in PLANTS.items()
    for plant in plants
]

# The checks the swap itself runs, after the pre-swap checks.
SWAP_CHECKS: frozenset[str] = frozenset({"clean_code_commit", "bundle_validation"})


def _check_of(plant: Plant) -> str:
    return next(check for check, plants in PLANTS.items() if plant in plants)


def test_the_baseline_passes_every_check_and_installs(tmp_path: Path) -> None:
    """Non-vacuity: with nothing planted, every check passes and the four ids go live."""
    baseline = build_baseline(tmp_path)

    attempt(baseline, baseline.check_options())

    written = manifest(baseline.stores.artifacts)
    for key in ("wp", "ats", "ou", "blend"):
        assert written[key] == getattr(GENERATION_ONE_IDS, key)


@pytest.mark.parametrize(("check", "plant"), PLANTED_CASES)
def test_each_check_refuses_by_name(tmp_path: Path, check: str, plant: Plant) -> None:
    baseline = build_baseline(tmp_path)
    options = plant(baseline, baseline.check_options(), tmp_path / "scratch")
    latest = baseline.stores.artifacts / "latest.json"
    latest_sha = sha256_of(latest)

    with pytest.raises(IntegrityRefusal) as raised:
        attempt(baseline, options)

    assert raised.value.check == check
    assert str(raised.value).startswith(f"{check}: ")
    assert sha256_of(latest) == latest_sha
    if check not in SWAP_CHECKS:
        assert read_date_events(baseline.ledger) == []


def test_every_pre_registered_check_has_a_planted_failure() -> None:
    """Every name but clean_code_commit is planted here; clean_code_commit has its own test."""
    assert set(PLANTS) == set(INTEGRITY_CHECKS) - {
        "clean_code_commit",
        "scores_frame",
        "dry_stamp_and_validate",
    }


@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        (_plant_unclean_aa, _plant_99_usable_rows),
        (_plant_row_count_off_by_one, _plant_two_threads),
        (_plant_edited_record, _plant_helper_game_in_calibration),
    ],
)
def test_checks_run_in_the_registered_order(
    tmp_path: Path, earlier: Plant, later: Plant
) -> None:
    baseline = build_baseline(tmp_path)
    scratch = tmp_path / "scratch"
    options = later(baseline, baseline.check_options(), scratch)
    options = earlier(baseline, options, scratch)
    assert INTEGRITY_CHECKS.index(_check_of(earlier)) < INTEGRITY_CHECKS.index(
        _check_of(later)
    )

    with pytest.raises(IntegrityRefusal) as raised:
        run_integrity_checks(**options)

    assert raised.value.check == _check_of(earlier)


def test_aa_proof_only_for_the_first_refit(tmp_path: Path) -> None:
    # A second-date re-fit is never held by the A/A record.
    check_aa_proof(LIST_WEEK, [UNCLEAN_WEEK_8], first_refit=False)
    baseline = build_baseline(tmp_path)
    run_integrity_checks(
        **{
            **baseline.check_options(),
            "aa_records": [UNCLEAN_WEEK_8],
            "first_refit": False,
        }
    )

    # The first re-fit is: an unclean week 8 selects no week-8 list, but lets week 9's stand.
    with pytest.raises(IntegrityRefusal) as raised:
        check_aa_proof(LIST_WEEK, [UNCLEAN_WEEK_8], first_refit=True)
    assert raised.value.check == "aa_proof"
    week_9 = {"season": 2026, "week": 9, "clean": True}
    check_aa_proof(9, [UNCLEAN_WEEK_8, week_9], first_refit=True)
    with pytest.raises(IntegrityRefusal):
        check_aa_proof(9, [CLEAN_WEEK_8, week_9], first_refit=True)
