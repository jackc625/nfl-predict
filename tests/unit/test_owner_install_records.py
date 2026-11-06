"""An owner-recorded install becomes a generation (MIDSEASON-SWITCH-RECORD.md section 4).

WHAT IS PROVEN
--------------
The v2 models are installed once by an owner command that records the install, append-only, in
``ledger/refit/owner_installs.jsonl`` (``refit.records``). Every reader of the generations must
accept such an install:

* ``refit.generations.installed_generations`` numbers it after the cadence generations (today's
  generation 1, so the owner install is generation 2), with the record's recipe, no cadence date
  and the record's training cutoff; a refused attempt never counts, and a refusal belongs to its
  own attempt only;
* ``refit.generations.in_force_recipe`` answers the recipe of the generation ``latest.json``
  names, and the cache's edge bands read the owner generation's bet rule;
* the Season panel (``refit.comparison.panel_frames``) starts the owner generation's period on the
  ET calendar day of its install, and ``scripts/compare_refit_arms.py`` exits 0 when that
  generation has no live rows of its own (Review Focus 4);
* the record is validated one way on write and on read: a dirty code commit, another bet-rule
  path form, an unknown event or a missing key is refused.

The cadence history is today's ``ledger/refit/dates.jsonl`` as the fixture helpers of
``tests/unit/test_refit_decision_path.py`` write it: generation 1 installed on cadence date 1
(2026-11-03, list week 8), its attempt ``installed`` and its date closed ``ran``.

Every store lives under pytest's temporary directories; nothing reads or writes production
``ledger/``, ``artifacts/`` or ``data/`` (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from api.cache_schema import REFIT_PANEL_PERIOD
from backtest.midseason_switch_2026 import INSTALLED_RECIPE, RECORD_PATH
from backtest.refit_preregistration_2026 import MARKET_CONVERTER_ID, REFIT_RECIPE_ID
from forward_ledger.shadow import SHADOW_PIN
from forward_ledger.store import read_entries
from models.artifacts import ResolvedArtifacts
from refit.comparison import paired_frame, panel_frames, period_summary
from refit.generations import (
    Generation,
    UnknownGenerationError,
    generation_for_ids,
    in_force_edge_thresholds,
    in_force_recipe,
    installed_generations,
)
from refit.records import (
    OWNER_INSTALLS_FILENAME,
    RefitRecordFormatError,
    append_date_event,
    append_owner_install,
    bet_rule_relpath,
    read_owner_installs,
    refit_dir,
)
from scripts import compare_refit_arms
from tests.unit.test_refit_comparison import (
    ats_pairs,
    games_frame,
    write_pairs,
    write_silver,
)
from tests.unit.test_refit_decision_path import (
    GENERATION_ONE_IDS,
    attempt_event,
    closed_event,
    ids_mapping,
    intent_event,
    write_generation_bet_rule,
    write_manifest_ids,
)

# The v2 generation's ids, as an owner install on the switch Tuesday, Tue 2026-11-10, would name
# them. RE-POINTED 2026-11-06 for MIDSEASON-SWITCH-RECORD.md section 8: the switch moved from week
# 13 to week 10, so the fixture install moves from Tue 2026-12-01 to Tue 2026-11-10.
V2_IDS = ResolvedArtifacts(
    wp="wp_20261111_030101",
    ats="ats_20261111_030102",
    ou="ou_20261111_030103",
    blend="blend_20261111_030500",
    converter=MARKET_CONVERTER_ID,
)
# 22:00 ET on Tue 2026-11-10 is already Wed 2026-11-11 in UTC: the period start must be the ET day.
CUTOFF = datetime(2026, 11, 11, 3, 0, tzinfo=UTC)
INSTALL_DATE = "2026-11-10"
# A bet-rule sha no file is written for: the record validator never reads the bet-rule file.
UNWRITTEN_SHA = "f" * 64
CLEAN_COMMIT = "d" * 40
CONVERTER_SLOPE = 0.15


def intent(**overrides: Any) -> dict[str, Any]:
    """An owner ``install_intent`` naming :data:`V2_IDS`; *overrides* replace fields."""
    sha = overrides.pop("bet_rule_sha256", UNWRITTEN_SHA)
    event: dict[str, Any] = {
        "event": "install_intent",
        "install_date": INSTALL_DATE,
        "cutoff_utc": CUTOFF.isoformat(),
        "ruling": f"{RECORD_PATH} section 4",
        "recipe": INSTALLED_RECIPE,
        "ids": ids_mapping(V2_IDS),
        "bet_rule_path": bet_rule_relpath(sha),
        "bet_rule_sha256": sha,
        "code_commit": CLEAN_COMMIT,
        "gold_generation_digest": "e" * 64,
        "before_ids": ids_mapping(GENERATION_ONE_IDS),
    }
    event.update(overrides)
    return event


def installed_event(ids: ResolvedArtifacts = V2_IDS) -> dict[str, Any]:
    return {
        "event": "installed",
        "install_date": INSTALL_DATE,
        "after_ids": ids_mapping(ids),
    }


def refused_event() -> dict[str, Any]:
    return {
        "event": "refused",
        "install_date": INSTALL_DATE,
        "check": "bundle_validation",
        "exception_class": "ArtifactBundleInvalidError",
    }


def ledger_with_cadence_history(tmp_path: Path) -> Path:
    """Today's cadence history: generation 1 installed on 2026-11-03, its date closed ``ran``."""
    ledger = tmp_path / "ledger"
    _path, sha = write_generation_bet_rule(ledger, GENERATION_ONE_IDS)
    for event in (
        intent_event(GENERATION_ONE_IDS, sha),
        attempt_event("installed"),
        closed_event("ran"),
    ):
        assert append_date_event(ledger, event)
    return ledger


def ledger_with_owner_install(
    tmp_path: Path,
    *,
    recipe: str = INSTALLED_RECIPE,
    installed: bool = False,
    refused: bool = False,
) -> Path:
    """The cadence history plus one owner install of :data:`V2_IDS` (its bet rule written first)."""
    ledger = ledger_with_cadence_history(tmp_path)
    # A stand-in for the v2 bet rule the owner command writes: a real record under its sha.
    _path, sha = write_generation_bet_rule(ledger, V2_IDS)
    assert append_owner_install(ledger, intent(recipe=recipe, bet_rule_sha256=sha))
    if installed:
        assert append_owner_install(ledger, installed_event())
    if refused:
        assert append_owner_install(ledger, refused_event())
    return ledger


def artifacts_naming(ids: ResolvedArtifacts, artifacts: Path) -> Path:
    """A temporary artifacts root whose ``latest.json`` names *ids* (blend and converter only).

    ``resolve_production_artifacts`` loads the blend for its bound converter and nothing else.
    """
    assert ids.converter is not None
    blend = artifacts / ids.blend
    blend.mkdir(parents=True)
    (blend / "blend_weights.json").write_text(
        json.dumps(
            {
                "weights": {"wp": 0.0, "ats": 0.0, "ou": 0.0},
                "market_probability_artifact_id": ids.converter,
                "market_probability_slope_beta": CONVERTER_SLOPE,
            }
        ),
        encoding="utf-8",
    )
    converter = artifacts / ids.converter
    converter.mkdir()
    (converter / "metadata.json").write_text(
        json.dumps({"artifact_id": ids.converter, "slope_beta": CONVERTER_SLOPE}),
        encoding="utf-8",
    )
    write_manifest_ids(artifacts, ids)
    return artifacts


def period_starts_of(ledger: Path) -> dict[int, date]:
    """Generation number -> its Season-panel period start, read through ``panel_frames``."""
    panel, _aa_status = panel_frames(read_entries(ledger), games_frame(), ledger)
    periods = panel[
        (panel["row_kind"] == REFIT_PANEL_PERIOD)
        & (panel["target"] == "ats")
        & panel["period_start"].notna()
    ]
    return {
        int(index): date.fromisoformat(str(start))
        for index, start in zip(
            periods["period_index"], periods["period_start"], strict=True
        )
    }


# ---------------------------------------------------------------------------
# The generations
# ---------------------------------------------------------------------------


def test_an_owner_install_is_the_newest_generation(tmp_path: Path) -> None:
    ledger = ledger_with_owner_install(
        tmp_path, recipe=INSTALLED_RECIPE, installed=True
    )

    generations = installed_generations(ledger)
    newest = generations[-1]

    assert [g.number for g in generations] == [0, 1, 2]
    assert (newest.number, newest.recipe, newest.list_week) == (
        2,
        INSTALLED_RECIPE,
        None,
    )
    assert newest.date_index is None
    assert newest.cutoff_utc == CUTOFF
    assert newest.ids == V2_IDS
    assert generation_for_ids(V2_IDS, generations).number == 2
    # The cadence generation keeps the re-fit recipe and no cutoff of its own.
    assert generations[1].recipe == REFIT_RECIPE_ID
    assert generations[1].cutoff_utc is None
    assert generation_for_ids(GENERATION_ONE_IDS, generations).number == 1


def test_a_refused_owner_install_does_not_count(tmp_path: Path) -> None:
    ledger = ledger_with_owner_install(tmp_path, refused=True)

    generations = installed_generations(ledger)

    assert generations[-1].number == 1
    with pytest.raises(UnknownGenerationError):
        generation_for_ids(V2_IDS, generations)


def test_a_refusal_belongs_to_its_own_attempt(tmp_path: Path) -> None:
    """A refusal proves only its own attempt's ids never served: never an earlier intent's."""
    retried = ledger_with_owner_install(tmp_path / "retried", refused=True)
    assert append_owner_install(retried, intent())
    assert append_owner_install(retried, installed_event())
    retry = installed_generations(retried)
    assert [g.number for g in retry] == [0, 1, 2]
    # Generation 2 is the retry (its own bet rule), never the refused first attempt.
    assert retry[-1].bet_rule_path.name == f"{UNWRITTEN_SHA}.json"

    other = ResolvedArtifacts(
        **{**ids_mapping(V2_IDS), "blend": "blend_20261111_040000"}
    )
    served = ledger_with_owner_install(tmp_path / "served", installed=True)
    assert append_owner_install(served, intent(ids=ids_mapping(other)))
    assert append_owner_install(served, refused_event())
    generations = installed_generations(served)
    assert [g.number for g in generations] == [0, 1, 2]
    assert generation_for_ids(V2_IDS, generations).number == 2
    with pytest.raises(UnknownGenerationError):
        generation_for_ids(other, generations)


# ---------------------------------------------------------------------------
# The record: one validator, on write and on read
# ---------------------------------------------------------------------------


def test_the_owner_record_refuses_a_dirty_commit_and_a_bad_bet_rule_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(RefitRecordFormatError, match="code_commit"):
        append_owner_install(tmp_path, intent(code_commit="a" * 40 + "+dirty"))
    with pytest.raises(RefitRecordFormatError, match="bet_rule_path"):
        append_owner_install(
            tmp_path, intent(bet_rule_path="refit/bet_rules/other.json")
        )
    with pytest.raises(RefitRecordFormatError, match="vocabulary"):
        append_owner_install(tmp_path, {**installed_event(), "event": "rolled_back"})
    no_recipe = intent()
    del no_recipe["recipe"]
    with pytest.raises(RefitRecordFormatError, match="'recipe' is missing"):
        append_owner_install(tmp_path, no_recipe)
    assert not (refit_dir(tmp_path) / OWNER_INSTALLS_FILENAME).exists()


def test_the_owner_record_round_trips_and_refuses_a_bad_stored_line(
    tmp_path: Path,
) -> None:
    assert read_owner_installs(tmp_path) == []
    events = [intent(), installed_event()]
    for event in events:
        assert append_owner_install(tmp_path, event) is True
    stored = read_owner_installs(tmp_path)
    assert [
        {key: row[key] for key in event}
        for event, row in zip(events, stored, strict=True)
    ] == events
    assert all("recorded_at_utc" in row for row in stored)

    path = refit_dir(tmp_path) / OWNER_INSTALLS_FILENAME
    good = path.read_text(encoding="utf-8")
    path.write_text(
        good + json.dumps(intent(code_commit=None)) + "\n", encoding="utf-8"
    )
    with pytest.raises(RefitRecordFormatError) as raised:
        read_owner_installs(tmp_path)
    assert f"{OWNER_INSTALLS_FILENAME} line 3" in str(raised.value)
    assert "code_commit" in str(raised.value)

    path.write_text(good + '{"event": "insta', encoding="utf-8")
    with pytest.raises(RefitRecordFormatError):
        read_owner_installs(tmp_path)
    assert len(read_owner_installs(tmp_path, skip_torn_tail=True)) == 2


# ---------------------------------------------------------------------------
# The readers: the recipe in force, the edge bands, the Season panel, the comparison
# ---------------------------------------------------------------------------


def test_in_force_recipe_follows_latest_json(tmp_path: Path) -> None:
    ledger = ledger_with_owner_install(tmp_path, installed=True)

    assert (
        in_force_recipe(artifacts_naming(V2_IDS, tmp_path / "v2"), ledger)
        == INSTALLED_RECIPE
    )
    assert (
        in_force_recipe(artifacts_naming(GENERATION_ONE_IDS, tmp_path / "gen1"), ledger)
        == REFIT_RECIPE_ID
    )
    assert (
        in_force_recipe(
            artifacts_naming(SHADOW_PIN.resolved(), tmp_path / "pin"), ledger
        )
        == SHADOW_PIN.recipe
    )


def test_the_edge_bands_read_the_owner_generations_bet_rule(tmp_path: Path) -> None:
    ledger = ledger_with_owner_install(tmp_path, installed=True)
    owner = installed_generations(ledger)[-1]
    stored = json.loads(owner.bet_rule_path.read_text(encoding="ascii"))

    thresholds = in_force_edge_thresholds(
        artifacts_naming(V2_IDS, tmp_path / "v2"), ledger
    )

    assert thresholds == {
        target: None if pair is None else tuple(pair)
        for target, pair in stored["edge_tier_thresholds"].items()
    }


def test_the_season_panel_starts_an_owner_generation_on_its_install_date(
    tmp_path: Path,
) -> None:
    """Review Focus 4: the owner generation has no live row, and the panel still builds."""
    ledger = ledger_with_owner_install(tmp_path, installed=True)
    write_pairs(ledger, ats_pairs())

    assert period_starts_of(ledger) == {1: date(2026, 11, 3), 2: date(2026, 11, 10)}
    panel, _aa_status = panel_frames(read_entries(ledger), games_frame(), ledger)
    owner = panel.loc[(panel["period_index"] == 2) & (panel["target"] == "ats")]
    assert owner["period_label"].tolist() == ["Period 2, from Tue Nov 10"]
    assert owner["paired_games"].tolist() == [0]


def test_a_period_start_is_never_guessed() -> None:
    """A cadence generation with no date event, or an owner one with no cutoff, still raises."""
    empty = paired_frame([], pd.DataFrame(games_frame().iloc[0:0]), ())
    cadence = Generation(
        number=1,
        ids=GENERATION_ONE_IDS,
        recipe=REFIT_RECIPE_ID,
        bet_rule_path=Path("unused.json"),
        date_index=1,
        list_week=8,
    )
    no_cutoff = Generation(
        number=1,
        ids=V2_IDS,
        recipe=INSTALLED_RECIPE,
        bet_rule_path=Path("unused.json"),
        date_index=None,
        list_week=None,
    )
    for generation in (cadence, no_cutoff):
        with pytest.raises(ValueError, match="never guessed"):
            period_summary(empty, (generation,), [])


def test_compare_refit_arms_exits_zero_with_an_owner_generation_and_no_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review Focus 4: generation-1 live and frozen rows, and an owner install with none."""
    ledger = ledger_with_owner_install(tmp_path, installed=True)
    write_pairs(ledger, ats_pairs())
    silver = write_silver(tmp_path / "silver")

    code = compare_refit_arms.main(
        ["--ledger-dir", str(ledger), "--silver-dir", str(silver)]
    )

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    cells = [line.split()[1:4] for line in lines if line.startswith("CELL= ")]
    assert ["ats", "gen=1", "n=3"] in cells
    assert [cell for cell in cells if cell[1] == "gen=2"] == [
        ["wp", "gen=2", "n=0"],
        ["ats", "gen=2", "n=0"],
        ["ou", "gen=2", "n=0"],
    ]
    assert any(line.startswith("VERDICT= ats ") for line in lines)
