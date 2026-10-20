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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import joblib
import pytest
from refit.integrity import IntegrityRefusal
from refit.swap import (
    InstallIntentNotRecordedError,
    install_generation,
    pending_install,
)

from backtest.refit_preregistration_2026 import REFIT_RECIPE_ID
from backtest.weekly_bet_list import load_bet_rule_record
from data.write_sink import RecordingSink, active_sink
from forward_ledger.schema import ARM_LIVE
from forward_ledger.store import read_entries, verify_chain
from models.artifacts import ResolvedArtifacts
from models.blending import BLEND_PAYLOAD_FILENAME, BlendProvenance
from refit import swap
from refit.cadence import CadenceDate, cadence_dates
from refit.generations import installed_generations
from refit.records import (
    append_date_event,
    bet_rule_relpath,
    read_date_events,
)
from tests.unit.test_refit_decision_path import (
    GENERATION_ONE_IDS,
    attempt_event,
    build_stores,
    closed_event,
    intent_event,
    ledger_rows,
    record,
    run_log_path,  # noqa: F401 - a pytest fixture, used by name
    write_generation_bet_rule,
)

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
