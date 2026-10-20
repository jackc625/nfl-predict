"""Generation 0 by the pin: a row's generation derived from its stamped model id (D-19, R11, RFIT-08).

A row's generation is never stored; it is read from the row's own ``model_artifact_id``. The
pinned row-19 ids are generation 0, a NULL stamp (weeks 3-4) is no generation, an installed
generation's ids map to its number, and any other id is refused by name. A shadow row is
generation 0 ONLY with a pinned id: the arm alone never decides the generation.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from backtest.refit_preregistration_2026 import REFIT_RECIPE_ID
from data.upstream_live import AS_OF_ENV
from forward_ledger import run_log
from forward_ledger.schema import ARM_LIVE, ARM_SHADOW
from forward_ledger.shadow import SHADOW_PIN
from forward_ledger.store import ledger_path
from models.artifacts import ResolvedArtifacts
from refit.generations import (
    GENERATION_ZERO,
    Generation,
    UnknownGenerationError,
    generation_for_ids,
    generation_of_row,
    generation_zero_ids,
    installed_generations,
)
from refit.records import append_date_event, bet_rule_record_path
from tests.unit.test_forward_ledger_store import make_row
from tests.unit.test_refit_decision_path import (
    GENERATION_ONE_IDS,
    attempt_event,
    build_stores,
    closed_event,
    ids_mapping,
    intent_event,
    ledger_rows,
    record,
    write_generation_bet_rule,
    write_manifest_ids,
)

_PINNED_MODEL_IDS: tuple[str, ...] = tuple(SHADOW_PIN.model_ids.values())

_GENERATION_ONE = ResolvedArtifacts(
    wp="wp_20261101_050000",
    ats="ats_20261101_050000",
    ou="ou_20261101_050000",
    blend="blend_20261101_050000",
    converter="market_probability_20260923_195443",
)
_INSTALLS: tuple[Generation, ...] = (
    Generation(
        number=1,
        ids=_GENERATION_ONE,
        recipe=REFIT_RECIPE_ID,
        bet_rule_path=Path("ledger/refit/bet_rules") / f"{'a' * 64}.json",
        date_index=1,
        list_week=8,
    ),
)


def _generation(
    row: dict[str, object], installs: Sequence[Generation] = ()
) -> int | None:
    model_id = row["model_artifact_id"]
    assert model_id is None or isinstance(model_id, str)
    return generation_of_row(model_id, str(row["arm"]), installs)


def test_generation_zero_ids_are_the_pin() -> None:
    assert GENERATION_ZERO == 0
    assert generation_zero_ids() == SHADOW_PIN.resolved()


@pytest.mark.parametrize("model_id", _PINNED_MODEL_IDS)
def test_shadow_rows_are_generation_zero(model_id: str) -> None:
    row = make_row(arm=ARM_SHADOW, model_artifact_id=model_id)
    assert _generation(row) == GENERATION_ZERO


@pytest.mark.parametrize(
    ("model_id", "installs"),
    [
        ("ats_20250101_000000", ()),
        (None, ()),
        (_GENERATION_ONE.ats, _INSTALLS),
    ],
    ids=["unknown-id", "null-id", "installed-generation-id"],
)
def test_a_shadow_row_with_any_other_id_is_refused(
    model_id: str | None, installs: tuple[Generation, ...]
) -> None:
    row = make_row(arm=ARM_SHADOW, model_artifact_id=model_id)
    with pytest.raises(UnknownGenerationError) as raised:
        _generation(row, installs)
    message = str(raised.value)
    assert repr(model_id) in message
    assert ARM_SHADOW in message


@pytest.mark.parametrize("model_id", _PINNED_MODEL_IDS)
def test_pinned_ids_on_a_live_row_are_generation_zero(model_id: str) -> None:
    row = make_row(arm=ARM_LIVE, model_artifact_id=model_id)
    assert _generation(row) == GENERATION_ZERO
    assert _generation(row, _INSTALLS) == GENERATION_ZERO


def test_null_stamp_maps_to_no_generation() -> None:
    row = make_row(arm=ARM_LIVE, week=3, model_artifact_id=None)
    assert _generation(row) is None
    assert _generation(row, _INSTALLS) is None


def test_unregistered_id_is_refused() -> None:
    row = make_row(arm=ARM_LIVE, model_artifact_id="ats_20250101_000000")
    with pytest.raises(UnknownGenerationError) as raised:
        _generation(row, _INSTALLS)
    assert "'ats_20250101_000000'" in str(raised.value)


@pytest.mark.parametrize("target", ["wp", "ats", "ou"])
def test_installed_ids_map_to_their_generation(target: str) -> None:
    row = make_row(arm=ARM_LIVE, model_artifact_id=_GENERATION_ONE.model_id_for(target))
    assert _generation(row, _INSTALLS) == 1


# ---------------------------------------------------------------------------
# Plan 37-10 Task 2: generation identity over a season; every install stop resolvable
# ---------------------------------------------------------------------------

_PIN_IDS = SHADOW_PIN.resolved()
# The ids of an install on a second cadence date, after date 1 closed not run.
_SECOND_DATE_IDS = ResolvedArtifacts(
    wp="wp_20261201_220501",
    ats="ats_20261201_220502",
    ou="ou_20261201_220503",
    blend="blend_20261201_221000",
    converter=_GENERATION_ONE.converter,
)
_SECOND_DATE: dict[str, Any] = {"date_index": 2, "date": "2026-12-01", "nfl_week": 13}
_SEASON_START = datetime(2026, 9, 1, 21, 0, tzinfo=UTC)


def _install(ledger: Path, ids: ResolvedArtifacts, *events: dict[str, Any]) -> None:
    """Record date 1's install of *ids* (its bet rule, its intent), then *events*, in order."""
    _, sha = write_generation_bet_rule(ledger, ids)
    assert append_date_event(ledger, intent_event(ids, sha))
    for event in events:
        assert append_date_event(ledger, event)


def _season_rows() -> list[dict[str, Any]]:
    """NULL live weeks 3-4, generation-0 live weeks 5-8, generation-1 live 9-10, shadow 5-10."""
    rows: list[dict[str, Any]] = []
    for week in range(3, 11):
        decided = (_SEASON_START + timedelta(days=7 * week)).isoformat()
        for target in ("wp", "ats", "ou"):
            common = {"week": week, "target": target, "decided_at_utc": decided}
            if week <= 4:
                live_id = None
            elif week <= 8:
                live_id = _PIN_IDS.model_id_for(target)
            else:
                live_id = _GENERATION_ONE.model_id_for(target)
            rows.append(make_row(arm=ARM_LIVE, model_artifact_id=live_id, **common))
            if week >= 5:
                rows.append(
                    make_row(
                        arm=ARM_SHADOW,
                        model_artifact_id=_PIN_IDS.model_id_for(target),
                        **common,
                    )
                )
    return rows


def _row_generation(
    row: dict[str, Any], generations: Sequence[Generation]
) -> int | None:
    return generation_of_row(row["model_artifact_id"], str(row["arm"]), generations)


def _installed_season(tmp_path: Path) -> tuple[Generation, ...]:
    ledger = tmp_path / "ledger"
    _install(ledger, _GENERATION_ONE, attempt_event("installed"), closed_event("ran"))
    return installed_generations(ledger)


def test_every_row_maps_to_one_generation(tmp_path: Path) -> None:
    generations = _installed_season(tmp_path)
    assert [g.number for g in generations] == [0, 1]

    for row in _season_rows():
        generation = _row_generation(row, generations)
        if row["arm"] == ARM_SHADOW:
            assert generation == GENERATION_ZERO
        elif row["week"] <= 4:
            assert generation is None
        elif row["week"] <= 8:
            assert generation == GENERATION_ZERO
        else:
            assert generation == 1


def test_live_generations_never_decrease(tmp_path: Path) -> None:
    generations = _installed_season(tmp_path)
    live = sorted(
        (row for row in _season_rows() if row["arm"] == ARM_LIVE),
        key=lambda row: row["decided_at_utc"],
    )
    numbers = [
        number
        for number in (_row_generation(row, generations) for row in live)
        if number is not None
    ]
    assert set(numbers) == {0, 1}
    assert numbers == sorted(numbers)


def test_a_row_decided_before_a_midweek_catchup_keeps_its_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A game decided under generation 0 keeps it after a catch-up install (first pick stands).

    The fixture week stands in for the week-9 Thursday game: its keys are decided once under the
    pin, generation 1 is installed, and the same keys are decided again with the new ids.
    """
    monkeypatch.delenv(AS_OF_ENV, raising=False)
    monkeypatch.setattr(
        run_log, "LEDGER_RUN_LOG", tmp_path / "logs" / "ledger_runs.jsonl"
    )
    stores = build_stores(tmp_path)
    ledger = tmp_path / "ledger"
    first = record(stores, ledger)
    assert first.appended
    stored = ledger_path(ledger).read_bytes()

    _install(
        ledger, GENERATION_ONE_IDS, attempt_event("installed"), closed_event("ran")
    )
    write_manifest_ids(stores.artifacts, GENERATION_ONE_IDS)
    second = record(stores, ledger)

    assert second.appended == 0
    assert len(second.refused) == first.appended
    for _key, fields in second.refused:
        assert "model_artifact_id" in fields
    assert ledger_path(ledger).read_bytes() == stored
    generations = installed_generations(ledger)
    assert [g.number for g in generations] == [0, 1]
    live = [row for row in ledger_rows(ledger) if row["arm"] == ARM_LIVE]
    assert live
    assert {_row_generation(row, generations) for row in live} == {GENERATION_ZERO}


def test_a_not_run_date_consumes_no_number(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    _install(ledger, _GENERATION_ONE, closed_event("not run"))
    _, sha = write_generation_bet_rule(ledger, _SECOND_DATE_IDS, date_index=2)
    for event in (
        intent_event(_SECOND_DATE_IDS, sha, **_SECOND_DATE),
        attempt_event("installed", **_SECOND_DATE),
        closed_event("ran", **_SECOND_DATE),
    ):
        assert append_date_event(ledger, event)

    generations = installed_generations(ledger)
    assert [g.number for g in generations] == [0, 1]
    assert generations[1].ids == _SECOND_DATE_IDS
    assert generations[1].date_index == 2
    assert generations[1].bet_rule_path == bet_rule_record_path(ledger, sha)
    with pytest.raises(UnknownGenerationError):
        generation_for_ids(_GENERATION_ONE, generations)


def test_every_stop_in_the_recording_order_keeps_the_served_ids_resolvable(
    tmp_path: Path,
) -> None:
    """Pitfall 2: intent -> (swap) -> attempt installed -> closed ran; a stop after each step."""
    ledger = tmp_path / "ledger"
    _, sha = write_generation_bet_rule(ledger, _GENERATION_ONE)
    for event in (
        intent_event(_GENERATION_ONE, sha),
        attempt_event("installed"),
        closed_event("ran"),
    ):
        assert append_date_event(ledger, event)
        generations = installed_generations(ledger)
        assert generation_for_ids(_GENERATION_ONE, generations).number == 1
        assert generation_for_ids(_PIN_IDS, generations).number == GENERATION_ZERO


def test_a_crashed_attempt_after_the_swap_keeps_the_served_ids_resolvable(
    tmp_path: Path,
) -> None:
    """replace_manifest wrote and then raised: one crashed attempt, no installed one, no close."""
    ledger = tmp_path / "ledger"
    _install(
        ledger, _GENERATION_ONE, attempt_event("crashed", exception_class="OSError")
    )

    generations = installed_generations(ledger)
    assert generation_for_ids(_GENERATION_ONE, generations).number == 1


def test_a_refused_attempt_after_the_intent_does_not_count(tmp_path: Path) -> None:
    """The control: bundle_validation refuses before the swap writes, so the ids never served."""
    ledger = tmp_path / "ledger"
    _install(
        ledger,
        _GENERATION_ONE,
        attempt_event("refused", check="bundle_validation"),
    )

    generations = installed_generations(ledger)
    assert [g.number for g in generations] == [GENERATION_ZERO]
    with pytest.raises(UnknownGenerationError) as raised:
        generation_for_ids(_GENERATION_ONE, generations)
    for value in ids_mapping(_GENERATION_ONE).values():
        assert str(value) in str(raised.value)
    assert generation_for_ids(_PIN_IDS, generations).number == GENERATION_ZERO
