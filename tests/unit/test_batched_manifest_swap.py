"""The batched four-pointer production swap: prove the whole bundle LOADS, then write once.

WHY THIS MODULE EXISTS (Plan 33.2-25 Task 1, SPEC R13)
------------------------------------------------------
``models.artifacts.update_manifest`` is per-target by design, so installing four corrected
artifacts through it takes four writes -- and a failure between any two of them leaves
``artifacts/latest.json`` naming a MIX of old and new, which R13 forbids.
``models.artifacts.replace_manifest`` is the batched writer that sits BESIDE it: it validates
every named artifact through the same loaders the serving path uses, cross-checks the four
against each other, and only then writes once through the one atomic helper.

WHAT THE FIXTURES DELIBERATELY LOOK LIKE
----------------------------------------
* A model directory holds ``model.pkl``, ``metadata.json`` and ``feature_list.json`` -- the
  file set ``save_model_artifact`` writes and ``load_model_artifact`` reads.
* The blend directory is written by the REAL ``MarketBlender.save_blend_artifacts`` and so
  holds ONLY ``blend_weights.json`` -- no ``metadata.json``. An earlier draft of this plan
  validated "every mapped directory contains a metadata.json", which would have REJECTED the
  very blend Plan 33.2-24 produced (reviews round ``f924749``, Codex HIGH). The complete-bundle
  test therefore fails by name on any regression to that rule.
* The converter directory holds its one ``metadata.json`` payload, which the blend's binding
  cross-check reads.

Every fixture lives under ``tmp_path``; nothing here reads or writes production ``artifacts/``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import joblib
import pytest

from models import artifacts
from models.blending import (
    BLEND_PAYLOAD_FILENAME,
    BlendProvenance,
    BlendWeights,
    MarketBlender,
    TuningResult,
)

GOLD = "gold-generation-digest-A"
OTHER_GOLD = "gold-generation-digest-B"
CONVERTER_ID = "market_probability_20990101_000000"
CONVERTER_SLOPE = 0.15

MODEL_IDS: dict[str, str] = {
    "wp": "wp_20990101_000001",
    "ats": "ats_20990101_000002",
    "ou": "ou_20990101_000003",
}

#: The pre-swap manifest every test starts from: four OLD pointers plus a key the mapping never
#: names, which must survive the swap untouched.
OLD_MANIFEST: dict[str, str] = {
    "wp": "wp_old",
    "ats": "ats_old",
    "ou": "ou_old",
    "blend": "blend_dynamic_old",
    "shadow": "shadow_arm_keep_me",
}


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _write_model(
    root: Path,
    key: str,
    version: str,
    *,
    digest: str = GOLD,
    recorded_target: str | None = None,
    with_model_pickle: bool = True,
) -> Path:
    """A model directory with the file set ``load_model_artifact`` reads."""
    directory = root / version
    directory.mkdir(parents=True)
    if with_model_pickle:
        joblib.dump({"estimator": key}, directory / "model.pkl")
    metadata = {
        "target": recorded_target if recorded_target is not None else key,
        "gold_generation_digest": digest,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (directory / "feature_list.json").write_text(
        json.dumps(["feature_a", "feature_b"]), encoding="utf-8"
    )
    return directory


def _write_converter(root: Path, converter_id: str = CONVERTER_ID) -> Path:
    """A converter directory carrying the one payload the blend's binding check reads."""
    directory = root / converter_id
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps({"slope_beta": CONVERTER_SLOPE}), encoding="utf-8"
    )
    return directory


def _tuning_result() -> TuningResult:
    """A minimal fixed-weight tuning record, enough for the real writer."""
    targets = ("wp", "ats", "ou")
    return TuningResult(
        weights=BlendWeights(
            wp_model_weight=0.0, ats_model_weight=0.0, ou_model_weight=0.12
        ),
        objective_by_target={
            "wp": "log_loss",
            "ats": "mean_absolute_error",
            "ou": "mean_absolute_error",
        },
        loss_by_target=dict.fromkeys(targets, 1.0),
        market_only_loss_by_target=dict.fromkeys(targets, 1.0),
        model_only_loss_by_target=dict.fromkeys(targets, 1.1),
        grid_by_target={t: [(0.0, 1.0), (1.0, 1.1)] for t in targets},
        seasons_by_target={t: [2021, 2022] for t in targets},
        n_games=dict.fromkeys(targets, 100),
        season_best_weight_by_target={t: {2021: 0.0, 2022: 0.1} for t in targets},
    )


def _write_blend(
    root: Path,
    *,
    digest: str = GOLD,
    sources: dict[str, str] | None = None,
) -> str:
    """A blend directory written by the REAL writer -- so it holds ONLY blend_weights.json."""
    blender = MarketBlender(
        market_probability_artifact_id=CONVERTER_ID,
        market_probability_slope_beta=CONVERTER_SLOPE,
    )
    provenance = BlendProvenance(
        gold_generation_digest=digest,
        source_artifact_ids=dict(sources if sources is not None else MODEL_IDS),
        tuning_corpus_rows=100,
        excluded_counts={"no_prelock_line": 1},
        thread_limit=1,
    )
    directory = blender.save_blend_artifacts(
        _tuning_result(), artifacts_dir=root, provenance=provenance
    )
    return directory.name


def _edit_blend_payload(root: Path, version: str, **changes: Any) -> None:
    """Rewrite keys of an already-written blend payload (None deletes the key)."""
    path = root / version / BLEND_PAYLOAD_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(root: Path, manifest: dict[str, str] | None = None) -> Path:
    path = root / "latest.json"
    path.write_text(json.dumps(manifest or OLD_MANIFEST, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def bundle(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A complete, valid four-artifact bundle under tmp_path, plus the old manifest."""
    for key, version in MODEL_IDS.items():
        _write_model(tmp_path, key, version)
    _write_converter(tmp_path)
    blend_version = _write_blend(tmp_path)
    _write_manifest(tmp_path)
    return tmp_path, {**MODEL_IDS, "blend": blend_version}


def _refusal(root: Path, mapping: dict[str, str]) -> Any:
    """Run the swap expecting a refusal; assert the manifest did not move by one byte."""
    latest = root / "latest.json"
    before = latest.read_bytes()
    with pytest.raises(artifacts.ArtifactBundleInvalidError) as excinfo:
        artifacts.replace_manifest(mapping, root)
    assert latest.read_bytes() == before, (
        "a refused bundle moved latest.json -- the swap must write NOTHING unless every "
        "artifact validates"
    )
    return excinfo.value


# ---------------------------------------------------------------------------
# The writer and its registry exist, beside the per-target writer
# ---------------------------------------------------------------------------


def test_the_batched_writer_exists_beside_the_per_target_one() -> None:
    """``replace_manifest`` is ADDED; ``update_manifest`` and the atomic helper stay."""
    assert hasattr(artifacts, "replace_manifest"), (
        "models.artifacts.replace_manifest is missing"
    )
    assert hasattr(artifacts, "update_manifest"), "the per-target writer must be KEPT"
    assert hasattr(artifacts, "_atomic_write_json"), (
        "the one atomic-write helper must stay"
    )


def test_the_registry_is_keyed_by_exactly_the_four_manifest_keys() -> None:
    """ONE per-kind registry: the models load through the model loader, the blend through its own.

    Asserted on the RESOLVED callables' source, so a validator that quietly reverted to a
    file-existence check -- the defect this task removes -- fails by name.
    """
    registry = artifacts.ARTIFACT_VALIDATORS
    assert sorted(registry) == ["ats", "blend", "ou", "wp"]
    for key in ("wp", "ats", "ou"):
        assert "load_model_artifact" in inspect.getsource(registry[key]), key
    assert "from_artifacts" in inspect.getsource(registry["blend"])


# ---------------------------------------------------------------------------
# The complete bundle
# ---------------------------------------------------------------------------


def test_a_blend_holding_only_blend_weights_json_validates(
    bundle: tuple[Path, dict[str, str]],
) -> None:
    """The case the earlier "has a metadata.json" rule would have REJECTED."""
    root, mapping = bundle
    blend_dir = root / mapping["blend"]
    assert sorted(p.name for p in blend_dir.iterdir()) == [BLEND_PAYLOAD_FILENAME], (
        "fixture sanity: the real writer should have produced ONLY blend_weights.json"
    )

    facts = artifacts.ARTIFACT_VALIDATORS["blend"](mapping["blend"], root)

    assert facts.gold_generation_digest == GOLD
    assert facts.source_artifact_ids == MODEL_IDS
    assert facts.market_probability_artifact_id == CONVERTER_ID


def test_a_complete_bundle_writes_all_four_pointers(
    bundle: tuple[Path, dict[str, str]],
) -> None:
    root, mapping = bundle

    artifacts.replace_manifest(mapping, root)

    written = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    for key, version in mapping.items():
        assert written[key] == version, key


def test_a_key_outside_the_mapping_survives_with_its_old_value(
    bundle: tuple[Path, dict[str, str]],
) -> None:
    """Pitfall 4, the failure update_manifest's docstring warns about, must not return here."""
    root, mapping = bundle
    before = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    # Non-vacuity: an empty manifest or an empty mapping would satisfy "survives" trivially.
    assert mapping, "fixture sanity: the mapping must be non-empty"
    assert before, "fixture sanity: the pre-swap manifest must be non-empty"
    untouched = set(before) - set(mapping)
    assert untouched == {"shadow"}, (
        "fixture sanity: one key must sit outside the mapping"
    )

    artifacts.replace_manifest(mapping, root)

    after = json.loads((root / "latest.json").read_text(encoding="utf-8"))
    assert after["shadow"] == before["shadow"]
    assert set(after) == set(before) | set(mapping)


def test_the_write_goes_through_the_one_atomic_helper_exactly_once(
    bundle: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A counting delegate: a second atomic-write implementation would leave the count at 0."""
    root, mapping = bundle
    real = artifacts._atomic_write_json
    calls: list[Path] = []

    def counting(path: Path, data: dict[str, Any]) -> None:
        calls.append(Path(path))
        real(path, data)

    monkeypatch.setattr(artifacts, "_atomic_write_json", counting)

    artifacts.replace_manifest(mapping, root)

    assert calls == [root / "latest.json"]


def test_a_crash_mid_write_leaves_the_old_manifest_whole(
    bundle: tuple[Path, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename is the only step that publishes; a failure before it changes nothing."""
    root, mapping = bundle
    latest = root / "latest.json"
    before = latest.read_bytes()
    files_before = sorted(p.name for p in root.iterdir())

    def crash(self: Path, target: Any) -> Path:
        msg = "simulated crash between the temp write and the rename"
        raise OSError(msg)

    monkeypatch.setattr(Path, "replace", crash)

    with pytest.raises(OSError, match="simulated crash"):
        artifacts.replace_manifest(mapping, root)

    assert latest.read_bytes() == before
    assert sorted(p.name for p in root.iterdir()) == files_before, (
        "a crashed write left a temp file behind"
    )


# ---------------------------------------------------------------------------
# Refusals: each leaves latest.json byte-identical
# ---------------------------------------------------------------------------


def test_a_missing_directory_is_refused(bundle: tuple[Path, dict[str, str]]) -> None:
    root, mapping = bundle
    error = _refusal(root, {**mapping, "ou": "ou_20990101_999999"})
    assert set(error.failures) == {"ou"}
    assert "ou_20990101_999999" in str(error)


def test_a_model_directory_without_model_pkl_is_refused(tmp_path: Path) -> None:
    """Existence is not loadability: metadata.json is present, model.pkl is not."""
    for key, version in MODEL_IDS.items():
        _write_model(tmp_path, key, version, with_model_pickle=(key != "ats"))
    _write_converter(tmp_path)
    blend_version = _write_blend(tmp_path)
    _write_manifest(tmp_path)
    assert (tmp_path / MODEL_IDS["ats"] / "metadata.json").exists(), "fixture sanity"

    error = _refusal(tmp_path, {**MODEL_IDS, "blend": blend_version})

    assert set(error.failures) == {"ats"}


def test_a_model_recording_a_different_target_is_refused(tmp_path: Path) -> None:
    for key, version in MODEL_IDS.items():
        _write_model(
            tmp_path, key, version, recorded_target=("ou" if key == "ats" else None)
        )
    _write_converter(tmp_path)
    blend_version = _write_blend(tmp_path)
    _write_manifest(tmp_path)

    error = _refusal(tmp_path, {**MODEL_IDS, "blend": blend_version})

    assert set(error.failures) == {"ats"}
    assert "'ou'" in error.failures["ats"]


def test_a_model_with_no_recorded_gold_digest_is_refused(tmp_path: Path) -> None:
    for key, version in MODEL_IDS.items():
        _write_model(tmp_path, key, version)
    metadata_path = tmp_path / MODEL_IDS["wp"] / "metadata.json"
    metadata_path.write_text(json.dumps({"target": "wp"}), encoding="utf-8")
    _write_converter(tmp_path)
    blend_version = _write_blend(tmp_path)
    _write_manifest(tmp_path)

    error = _refusal(tmp_path, {**MODEL_IDS, "blend": blend_version})

    assert set(error.failures) == {"wp"}


def test_a_gold_digest_disagreement_is_refused_naming_the_keys(tmp_path: Path) -> None:
    """A bundle trained on two golds is the mixed manifest R13 forbids, one level down."""
    for key, version in MODEL_IDS.items():
        _write_model(
            tmp_path, key, version, digest=(OTHER_GOLD if key == "ats" else GOLD)
        )
    _write_converter(tmp_path)
    blend_version = _write_blend(tmp_path)
    _write_manifest(tmp_path)

    error = _refusal(tmp_path, {**MODEL_IDS, "blend": blend_version})

    message = str(error)
    assert OTHER_GOLD in message and GOLD in message
    for key in ("wp", "ats", "ou", "blend"):
        assert key in message


def test_blend_sources_that_differ_from_the_mapping_are_refused(tmp_path: Path) -> None:
    for key, version in MODEL_IDS.items():
        _write_model(tmp_path, key, version)
    _write_converter(tmp_path)
    blend_version = _write_blend(
        tmp_path, sources={**MODEL_IDS, "ou": "ou_some_other_candidate"}
    )
    _write_manifest(tmp_path)

    error = _refusal(tmp_path, {**MODEL_IDS, "blend": blend_version})

    assert "ou_some_other_candidate" in str(error)


def test_a_missing_converter_directory_is_refused(
    bundle: tuple[Path, dict[str, str]],
) -> None:
    root, mapping = bundle
    (root / CONVERTER_ID / "metadata.json").unlink()
    (root / CONVERTER_ID).rmdir()

    error = _refusal(root, mapping)

    assert "blend" in error.failures
    assert CONVERTER_ID in str(error)


def test_a_blend_with_no_converter_binding_is_refused(
    bundle: tuple[Path, dict[str, str]],
) -> None:
    root, mapping = bundle
    _edit_blend_payload(
        root,
        mapping["blend"],
        market_probability_artifact_id=None,
        market_probability_slope_beta=None,
    )

    error = _refusal(root, mapping)

    assert set(error.failures) == {"blend"}


def test_a_blend_with_no_provenance_is_refused(
    bundle: tuple[Path, dict[str, str]],
) -> None:
    root, mapping = bundle
    _edit_blend_payload(
        root,
        mapping["blend"],
        gold_generation_digest=None,
        source_artifact_ids=None,
        tuning_corpus=None,
        excluded_counts=None,
        thread_limit=None,
    )

    error = _refusal(root, mapping)

    assert set(error.failures) == {"blend"}


def test_a_blend_carrying_a_dynamic_section_is_refused(
    bundle: tuple[Path, dict[str, str]],
) -> None:
    """D33.2-10 retired the week-varying blend; the swap must not let one back in."""
    root, mapping = bundle
    _edit_blend_payload(root, mapping["blend"], dynamic={"wp": {"k": 1.0}})

    error = _refusal(root, mapping)

    assert set(error.failures) == {"blend"}
    assert "dynamic" in error.failures["blend"]


def test_every_failure_is_reported_together(tmp_path: Path) -> None:
    """Two broken artifacts produce ONE error naming both -- never first-only."""
    for key, version in MODEL_IDS.items():
        if key != "wp":
            _write_model(tmp_path, key, version, with_model_pickle=(key != "ats"))
    _write_converter(tmp_path)
    blend_version = _write_blend(tmp_path)
    _write_manifest(tmp_path)

    error = _refusal(tmp_path, {**MODEL_IDS, "blend": blend_version})

    assert set(error.failures) == {"wp", "ats"}
    assert MODEL_IDS["wp"] in str(error) and MODEL_IDS["ats"] in str(error)


def test_a_key_with_no_registered_validator_is_refused(
    bundle: tuple[Path, dict[str, str]],
) -> None:
    root, mapping = bundle
    error = _refusal(root, {**mapping, "shadow": "shadow_new"})
    assert "shadow" in error.failures


def test_an_incomplete_bundle_is_refused(bundle: tuple[Path, dict[str, str]]) -> None:
    """R13: latest.json is written only once ALL FOUR exist -- three is a mixed manifest."""
    root, mapping = bundle
    partial = {key: version for key, version in mapping.items() if key != "blend"}
    error = _refusal(root, partial)
    assert "blend" in error.failures


def test_a_version_that_escapes_the_artifacts_root_is_refused(
    bundle: tuple[Path, dict[str, str]],
) -> None:
    root, mapping = bundle
    error = _refusal(root, {**mapping, "wp": f"../{MODEL_IDS['wp']}"})
    assert set(error.failures) == {"wp"}


# ---------------------------------------------------------------------------
# The per-target writer is kept, and still per-target
# ---------------------------------------------------------------------------


def test_update_manifest_still_writes_one_key_and_leaves_the_rest(
    tmp_path: Path,
) -> None:
    latest = _write_manifest(tmp_path)

    artifacts.update_manifest("ats", "ats_new", tmp_path)

    after = json.loads(latest.read_text(encoding="utf-8"))
    assert after == {**OLD_MANIFEST, "ats": "ats_new"}
