"""The controls that let the re-pointed live-manifest tests fail (Phase 37, Plan 37-20).

``tests/live_manifest_record.py`` tells the live-manifest tests what ``artifacts/latest.json`` must
hold: the newest recorded generation, row 19 until the first recorded install. These controls
prove it on temporary ledgers written by the real producers (``refit.records``): no install is
row 19; a recorded install is the end state; a refused install is not; an intent whose swap has not
happened waits, a swapped one counts, and a closed install that is no longer served is still the
expected state, so the comparing tests fail on it. They also exercise the generation >= 1
branches before any real install: the expected bytes against the real writer
(``models.artifacts._atomic_write_json``), and the metadata contract against what the real producer
(``refit.training.train_generation``) saves.

Every store is under ``tmp_path``; production ``ledger/`` and ``artifacts/`` are never touched.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backtest.refit_preregistration_2026 import REFIT_RECIPE_ID
from models.artifacts import _atomic_write_json
from refit.generations import Generation
from refit.records import append_date_event
from refit.training import train_generation
from tests.fixtures.refit_gold import FIXTURE_CUTOFF_UTC, build_refit_fixture
from tests.live_manifest_record import (
    MANIFEST_KEYS,
    expected_manifest_text,
    recorded_live_manifest,
    refit_metadata_problems,
)
from tests.phase33_state import ROW19_SWAP_ARTIFACT_IDS
from tests.unit.test_refit_decision_path import (
    GENERATION_ONE_IDS,
    attempt_event,
    closed_event,
    intent_event,
    write_generation_bet_rule,
)

ROW19 = dict(ROW19_SWAP_ARTIFACT_IDS)
GENERATION_ONE = {key: getattr(GENERATION_ONE_IDS, key) for key in MANIFEST_KEYS}


def _generation(number: int, date_index: int) -> Generation:
    """Generation *number*, installed on date *date_index* of list 8 (its ids are not read)."""
    return Generation(
        number=number,
        ids=GENERATION_ONE_IDS,
        recipe=REFIT_RECIPE_ID,
        bet_rule_path=Path("ledger/refit/bet_rules") / f"{'a' * 64}.json",
        date_index=date_index,
        list_week=8,
    )


def _record_intent(ledger: Path) -> None:
    """A bet-rule record and an install_intent for date 1 naming the generation-1 ids."""
    _path, sha = write_generation_bet_rule(ledger, GENERATION_ONE_IDS)
    assert append_date_event(ledger, intent_event(GENERATION_ONE_IDS, sha))


def _write_latest(path: Path, ids: dict[str, str]) -> None:
    """A temporary latest.json written by the real manifest writer."""
    _atomic_write_json(path, {key: ids[key] for key in MANIFEST_KEYS})


def test_no_install_recorded_is_the_row19_record(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    assert not ledger.exists()
    assert recorded_live_manifest(
        ledger_dir=ledger, latest_path=tmp_path / "latest.json"
    ) == (0, ROW19)


def test_a_recorded_install_is_the_end_state(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    _record_intent(ledger)
    assert append_date_event(ledger, closed_event("ran"))

    assert recorded_live_manifest(
        ledger_dir=ledger, latest_path=tmp_path / "latest.json"
    ) == (1, GENERATION_ONE)


def test_a_refused_install_is_not_the_end_state(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    _record_intent(ledger)
    assert append_date_event(
        ledger, attempt_event("refused", check="bundle_validation")
    )

    assert recorded_live_manifest(
        ledger_dir=ledger, latest_path=tmp_path / "latest.json"
    ) == (0, ROW19)


def test_an_intent_not_yet_swapped_is_not_the_end_state(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    latest = tmp_path / "latest.json"
    _record_intent(ledger)

    # The run died after the intent, before replace_manifest wrote: row 19 still serves.
    _write_latest(latest, ROW19)
    assert recorded_live_manifest(ledger_dir=ledger, latest_path=latest) == (0, ROW19)

    # The swap happened and the recording stopped after it: generation 1 is in force.
    _write_latest(latest, GENERATION_ONE)
    assert recorded_live_manifest(ledger_dir=ledger, latest_path=latest) == (
        1,
        GENERATION_ONE,
    )

    # A completed install that is no longer served (an unrecorded roll-back) stays the
    # expected state, so the comparing tests fail on the manifest latest.json now holds.
    assert append_date_event(ledger, closed_event("ran"))
    _write_latest(latest, ROW19)
    generation, expected = recorded_live_manifest(ledger_dir=ledger, latest_path=latest)
    assert (generation, expected) == (1, GENERATION_ONE)
    assert json.loads(latest.read_text(encoding="utf-8")) != expected


def test_the_expected_text_is_what_the_writer_writes(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    _write_latest(path, GENERATION_ONE)

    written = path.read_bytes().replace(b"\r\n", b"\n")
    assert written == expected_manifest_text(GENERATION_ONE).encode("ascii")


def _without(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in mapping.items() if name != key}


def test_refit_metadata_problems_reads_the_real_producer(tmp_path: Path) -> None:
    store = build_refit_fixture(tmp_path)
    # List 8, date 1: Tue 2026-11-03 17:00 America/New_York.
    assert datetime(2026, 11, 3, 22, 0, tzinfo=UTC) == FIXTURE_CUTOFF_UTC
    training = train_generation(
        FIXTURE_CUTOFF_UTC,
        gold_dir=store.gold_dir,
        silver_dir=store.silver_dir,
        artifacts_dir=store.artifacts_dir,
        snapshots=store.snapshots,
        save=True,
    )
    assert sorted(training.artifact_ids) == ["ats", "ou", "wp"]

    for target, version in training.artifact_ids.items():
        metadata = json.loads(
            (store.artifacts_dir / version / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        generation_one = _generation(1, date_index=1)
        assert refit_metadata_problems(metadata, generation_one) == [], target

        wrong_season = {
            **metadata,
            "calibration": {**metadata["calibration"], "season": 2024},
        }
        problems = refit_metadata_problems(wrong_season, generation_one)
        assert len(problems) == 1 and "calibration season" in problems[0], problems

        problems = refit_metadata_problems(
            _without(metadata, "cutoff_utc"), generation_one
        )
        assert len(problems) == 1 and "cutoff_utc" in problems[0], problems

        wrong_recipe = {**metadata, "refit_recipe": "recipe-2026-row19-v1"}
        problems = refit_metadata_problems(wrong_recipe, generation_one)
        assert len(problems) == 1 and "refit_recipe" in problems[0], problems

        # WR-D07: generation 2 (list 8, date 2: Tue 2026-12-01) serving generation 1's models is a
        # stale re-install; their 2026-11-03 cutoff is inside the season but not generation 2's.
        problems = refit_metadata_problems(metadata, _generation(2, date_index=2))
        assert len(problems) == 1 and "2026-12-01T17:00:00-05:00" in problems[0], (
            problems
        )
