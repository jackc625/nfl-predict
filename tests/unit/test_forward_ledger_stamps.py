"""The artifact and blend stamps come from scoring, never from a re-read (Plan 34-05, LDGR-03).

WHAT THIS MODULE PINS
---------------------
``artifacts/latest.json`` is the ONE production swap surface, and before this plan one weekly
decision read it up to eight times: each of the three scorers and the blend loader resolved it
on their own, and a lock instant spanning two weeks ran that whole set twice. A swap landing
between two of those reads would score a row with one model and stamp it with another.

So the manifest is now read ONCE per decision into a frozen ``ResolvedArtifacts``, and every
scorer and blender call in that decision is handed its explicit version. The tests below prove
the read is read-only and single, that an incomplete manifest is refused by name, that the
scorer honours an explicit version over the manifest, and -- the adjacency edge the SPEC names --
that rewriting ``latest.json`` between resolution and scoring changes nothing the decision
carries.

Every store here lives under ``tmp_path``; production ``artifacts/`` is never touched.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from backtest import diagnose as diagnose_module
from models import artifacts as artifacts_module

CONVERTER_ID = "market_probability_20990101_000000"
CONVERTER_SLOPE = 0.15
BLEND_ID = "blend_20990101_000004"

ORIGINAL_IDS: dict[str, str] = {
    "wp": "wp_20990101_000001",
    "ats": "ats_20990101_000002",
    "ou": "ou_20990101_000003",
    "blend": BLEND_ID,
}

# The ids a mid-run swap installs. They name nothing on disk: a decision that re-read the
# manifest after the swap would ask for one of them, which is exactly what the spies record.
SWAPPED_IDS: dict[str, str] = {
    "wp": "wp_20990202_000001",
    "ats": "ats_20990202_000002",
    "ou": "ou_20990202_000003",
    "blend": "blend_20990202_000004",
}


# ---------------------------------------------------------------------------
# Fixture builders (tmp_path only)
# ---------------------------------------------------------------------------


def _write_manifest(root: Path, manifest: dict[str, str]) -> Path:
    path = root / "latest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _write_converter(root: Path, converter_id: str = CONVERTER_ID) -> None:
    directory = root / converter_id
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps({"slope_beta": CONVERTER_SLOPE}), encoding="utf-8"
    )


def _write_blend(
    root: Path, version: str = BLEND_ID, *, converter_id: str | None = CONVERTER_ID
) -> None:
    """A blend payload holding the fields ``MarketBlender.from_artifacts`` reads."""
    directory = root / version
    directory.mkdir(parents=True)
    payload: dict[str, Any] = {"weights": {"wp": 0.0, "ats": 0.0, "ou": 0.12}}
    if converter_id is not None:
        payload["market_probability_artifact_id"] = converter_id
        payload["market_probability_slope_beta"] = CONVERTER_SLOPE
    (directory / "blend_weights.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_wp_model(root: Path, version: str, home_win_rate: float) -> None:
    """A real, loadable WP artifact whose every prediction is *home_win_rate*."""
    labels = np.array(
        [1] * int(home_win_rate * 10) + [0] * (10 - int(home_win_rate * 10))
    )
    model = DummyClassifier(strategy="prior").fit(np.zeros((10, 1)), labels)
    directory = root / version
    directory.mkdir(parents=True)
    joblib.dump(model, directory / "model.pkl")
    (directory / "metadata.json").write_text(
        json.dumps({"target": "wp"}), encoding="utf-8"
    )
    (directory / "feature_list.json").write_text(
        json.dumps(["feature_a"]), encoding="utf-8"
    )


def _resolve(root: Path) -> Any:
    """The one manifest resolver, or an assertion naming what is missing."""
    resolver = getattr(artifacts_module, "resolve_production_artifacts", None)
    assert resolver is not None, (
        "models.artifacts defines no resolve_production_artifacts; every scorer and blender "
        "call still re-reads latest.json on its own"
    )
    return resolver(root)


def _resolved(**ids: str | None) -> Any:
    resolved_cls = getattr(artifacts_module, "ResolvedArtifacts", None)
    assert resolved_cls is not None, "models.artifacts defines no ResolvedArtifacts"
    return resolved_cls(**ids)


def _score(target: str, **kwargs: Any) -> pd.DataFrame:
    """``score_deployed_artifacts``, asserting first that it accepts an explicit version."""
    scorer = diagnose_module.score_deployed_artifacts
    assert "version" in inspect.signature(scorer).parameters, (
        "score_deployed_artifacts takes no version; it can only ever score whatever "
        "latest.json names at the instant it is called"
    )
    return scorer(target, **kwargs)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def production_like(tmp_path: Path) -> Path:
    """A complete four-pointer manifest whose blend binds a converter, under tmp_path."""
    root = tmp_path / "artifacts"
    root.mkdir()
    _write_converter(root)
    _write_blend(root)
    _write_manifest(root, ORIGINAL_IDS)
    return root


# ---------------------------------------------------------------------------
# Task 1: resolve once, read-only, and an explicit version on the scorer
# ---------------------------------------------------------------------------


def test_resolve_reads_once_and_never_writes(
    production_like: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Five ids out, the converter taken from the blend's own binding, the manifest untouched."""
    manifest_path = production_like / "latest.json"
    before = _sha256(manifest_path)

    reads: list[str] = []
    real_read_text = pathlib.Path.read_text

    def counting_read_text(self: pathlib.Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "latest.json":
            reads.append(str(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", counting_read_text)
    resolved = _resolve(production_like)
    monkeypatch.undo()

    assert resolved == _resolved(
        wp=ORIGINAL_IDS["wp"],
        ats=ORIGINAL_IDS["ats"],
        ou=ORIGINAL_IDS["ou"],
        blend=ORIGINAL_IDS["blend"],
        converter=CONVERTER_ID,
    )
    assert len(reads) == 1, f"latest.json was read {len(reads)} times: {reads}"
    assert _sha256(manifest_path) == before, "resolution rewrote latest.json"


def test_resolve_records_no_converter_when_the_blend_binds_none(
    tmp_path: Path,
) -> None:
    """A blend with no binding yields converter None -- recorded honestly, never invented."""
    root = tmp_path / "artifacts"
    root.mkdir()
    _write_blend(root, converter_id=None)
    _write_manifest(root, ORIGINAL_IDS)

    assert _resolve(root).converter is None


@pytest.mark.parametrize("missing", ["ou", "blend"])
def test_resolve_refuses_incomplete_manifest(
    production_like: Path, missing: str
) -> None:
    """A manifest without one of the four pointers is refused by name; nothing is defaulted."""
    _write_manifest(
        production_like,
        {key: value for key, value in ORIGINAL_IDS.items() if key != missing},
    )

    with pytest.raises(KeyError, match=missing):
        _resolve(production_like)


def test_model_id_for_names_each_target_and_refuses_the_blend(
    production_like: Path,
) -> None:
    resolved = _resolve(production_like)

    assert [resolved.model_id_for(t) for t in ("wp", "ats", "ou")] == [
        ORIGINAL_IDS["wp"],
        ORIGINAL_IDS["ats"],
        ORIGINAL_IDS["ou"],
    ]
    with pytest.raises(KeyError, match="blend"):
        resolved.model_id_for("blend")


def _wp_gold() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2026, 2026],
            "week": [5, 5],
            "feature_a": [0.0, 1.0],
            "home_win": [1, 0],
        }
    )


@pytest.fixture
def two_wp_models(tmp_path: Path) -> Path:
    """``wp_A`` predicts 0.3 everywhere, ``wp_B`` 0.8; the manifest names ``wp_B``."""
    root = tmp_path / "artifacts"
    root.mkdir()
    _write_wp_model(root, "wp_A", 0.3)
    _write_wp_model(root, "wp_B", 0.8)
    _write_manifest(root, {"wp": "wp_B"})
    return root


def test_scorer_uses_explicit_version(two_wp_models: Path) -> None:
    """An explicit version wins over whatever latest.json names."""
    scored = _score(
        "wp", gold_df=_wp_gold(), artifacts_dir=two_wp_models, version="wp_A"
    )

    assert scored["model_prob"].tolist() == pytest.approx([0.3, 0.3])


def test_scorer_default_unchanged(two_wp_models: Path) -> None:
    """``version=None`` keeps today's behaviour: the manifest names the model."""
    scored = _score("wp", gold_df=_wp_gold(), artifacts_dir=two_wp_models)

    assert scored["model_prob"].tolist() == pytest.approx([0.8, 0.8])
