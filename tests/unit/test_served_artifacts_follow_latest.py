"""The models that actually SERVE (``artifacts/latest.json``) carry no market line and one gold.

``test_no_market_in_artifacts.py`` checks a FIXED historical id list (the 2026-09-23 re-fit). A
later fix swapped the served models, so that list no longer names what serves. This module
follows ``latest.json`` -- whatever ids it names -- and asserts on the WRITTEN files:

  * no market-line column in ``feature_list.json`` or in ``metadata.json`` ``feature_names``,
    using the SAME predicate as the historical test (``market_columns_in``);
  * the three served models record one and the same ``gold_generation_digest``, and the served
    blend, which records one, records that same digest.

It deliberately does NOT compare with the live gold digest: the daily run rebuilds gold every
day, so live gold legitimately moves past the models' training gold. Agreement among the served
artifacts is the property.

Controls: non-vacuity (non-empty lists, a digest present) and planted violations.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_no_market_in_artifacts import market_columns_in

ARTIFACTS_ROOT = Path(__file__).resolve().parents[2] / "artifacts"
LATEST = ARTIFACTS_ROOT / "latest.json"
TARGETS = ("wp", "ats", "ou")

pytestmark = pytest.mark.skipif(
    not LATEST.exists(),
    reason="artifacts/latest.json absent (artifacts/ is gitignored)",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _served() -> dict[str, str]:
    return _read(LATEST)


def served_market_columns(directory: Path) -> dict[str, list[str]]:
    """Market columns in each of the two written feature lists of one model directory."""
    return {
        "feature_list.json": market_columns_in(_read(directory / "feature_list.json")),
        "metadata.json:feature_names": market_columns_in(
            _read(directory / "metadata.json")["feature_names"]
        ),
    }


@pytest.fixture(scope="module")
def served() -> dict[str, str]:
    ids = _served()
    for key in (*TARGETS, "blend"):
        assert key in ids, f"latest.json names no '{key}'"
        assert (ARTIFACTS_ROOT / ids[key]).is_dir(), f"{ids[key]} is not on disk"
    return ids


class TestServedModelsCarryNoMarketLine:
    @pytest.mark.parametrize("target", TARGETS)
    def test_the_served_feature_lists_contain_no_market_line(
        self, served: dict[str, str], target: str
    ) -> None:
        directory = ARTIFACTS_ROOT / served[target]
        found = served_market_columns(directory)
        assert not any(found.values()), (
            f"served {target} model {served[target]} carries a market line: {found}"
        )

    @pytest.mark.parametrize("target", TARGETS)
    def test_non_vacuity_the_served_lists_are_non_empty_and_agree(
        self, served: dict[str, str], target: str
    ) -> None:
        directory = ARTIFACTS_ROOT / served[target]
        listed = _read(directory / "feature_list.json")
        assert len(listed) > 0
        assert listed == _read(directory / "metadata.json")["feature_names"]

    def test_a_planted_market_column_in_either_written_list_is_caught(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "feature_list.json").write_text(
            json.dumps(["home_elo", "snapshot_total"]), encoding="utf-8"
        )
        (tmp_path / "metadata.json").write_text(
            json.dumps({"feature_names": ["home_elo", "snapshot_spread"]}),
            encoding="utf-8",
        )
        found = served_market_columns(tmp_path)
        assert found == {
            "feature_list.json": ["snapshot_total"],
            "metadata.json:feature_names": ["snapshot_spread"],
        }


def _gold_digests(ids: dict[str, str]) -> dict[str, str | None]:
    digests = {
        target: _read(ARTIFACTS_ROOT / ids[target] / "metadata.json").get(
            "gold_generation_digest"
        )
        for target in TARGETS
    }
    digests["blend"] = _read(ARTIFACTS_ROOT / ids["blend"] / "blend_weights.json").get(
        "gold_generation_digest"
    )
    return digests


class TestServedArtifactsShareOneGold:
    def test_every_served_artifact_records_one_and_the_same_gold(
        self, served: dict[str, str]
    ) -> None:
        digests = _gold_digests(served)
        assert all(isinstance(d, str) and len(d) == 64 for d in digests.values()), (
            f"a served artifact records no gold digest: {digests}"
        )
        assert len(set(digests.values())) == 1, (
            f"the served artifacts were trained on different gold: {digests}"
        )

    def test_a_planted_disagreeing_digest_is_caught(
        self, served: dict[str, str]
    ) -> None:
        digests = _gold_digests(served)
        digests["ou"] = "0" * 64
        assert len(set(digests.values())) != 1
