"""The recorded owner install of the v2 models (MIDSEASON-SWITCH-RECORD.md section 4).

WHAT IS PROVEN
--------------
``refit.owner_install.run_owner_install`` is the one owner command of the switch: it rebuilds gold
as v2, trains the three v2 models once, saves them, tunes their blend, writes their bet-rule record,
records the install in ``ledger/refit/owner_installs.jsonl`` and swaps ``artifacts/latest.json``
through ``models.artifacts.replace_manifest`` -- or refuses and puts the v1 gold back.

* Review Focus 2: while any 2026 game of weeks 1-12 is still before its lock, the install refuses
  and writes nothing; a second v2 install refuses too.
* The v2 metadata names the fixed inputs and no ledger stamp column; the v2 bet rule parses
  through the decision path's own reader with no EV floor and no edge band.
* A successful install swaps ``latest.json``, records ``install_intent`` then ``installed`` (one
  ET install date), and the in-force generation is the v2 one, with no bands.
* Review Focus 1: a nonzero tuned weight refuses before anything is recorded, and the v1 gold is
  rebuilt. So does a failed v2 gold build, a dirty tree (refused before any build), a declined
  intent write, and a refused or crashed swap -- the last two appending a ``refused`` event, so
  the never-served intent never counts as a generation.
* Review fix R1 (PR #6): a failure after ``latest.json`` names the v2 ids -- an error or an
  interrupt -- is an install, never a refusal: ``installed`` is recorded, the v1 gold is not
  rebuilt under the v2 models, and the v2 recipe is in force.

The gold build, the clean-commit check and the blend tune are stubbed; training is real, on a
small gold. Every artifacts dir, ledger, gold and silver store is a pytest temporary directory;
production ``artifacts/``, ``ledger/`` and ``data/`` are never read or written.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

import utils.game_lock as lock_rule
from api.cache import CACHE_SCHEMA, _load_feature_importances
from backtest.midseason_switch_2026 import INSTALLED_RECIPE, RECORD_PATH
from backtest.refit_preregistration_2026 import MARKET_CONVERTER_ID, REFIT_RECIPE_ID
from backtest.weekly_bet_list import load_bet_rule_record
from forward_ledger.canonical import IMMUTABLE_COLUMNS_V1
from models.artifacts import load_model_artifact
from models.v2_recipe import V2_INPUTS, V2_TARGETS, build_v2_estimator
from refit import owner_install
from refit.generations import (
    in_force_edge_thresholds,
    in_force_recipe,
    installed_generations,
)
from refit.owner_install import (
    InstallRefused,
    install_preconditions,
    v2_bet_rule_payload,
    v2_feature_importances,
    v2_model_metadata,
)
from refit.records import read_owner_installs, write_bet_rule_record
from refit.witness import REPO_ROOT
from scripts import install_v2
from scripts.build_features import FeatureMatrixBuilder
from scripts.build_gold_v2 import FeatureMatrixBuilderV2Live
from tests.unit.test_owner_install_records import ledger_with_cadence_history
from tests.unit.test_refit_decision_path import (
    GENERATION_ONE_IDS,
    write_manifest_ids,
)
from tests.unit.test_refit_swap import _write_converter, write_blend, write_bundle

CLEAN_COMMIT: str = "c" * 40
DIRTY_COMMIT: str = f"{'c' * 40}+dirty"

# Tue 2026-12-01 21:30 ET, already Wed 2026-12-02 in UTC: the install date must be the ET day.
NOW = datetime(2026, 12, 2, 2, 30, tzinfo=UTC)
INSTALL_DATE: str = "2026-12-01"

# The Monday-night week-12 game kicks off Mon 2026-11-30, so it locks Sun 2026-11-29 18:00 ET.
WEEK12_LAST_LOCK = datetime(2026, 11, 29, 18, 0, tzinfo=lock_rule.ET)
DEC1_15ET = datetime(2026, 12, 1, 15, 0, tzinfo=lock_rule.ET)
CUT = datetime(2026, 12, 1, 20, 0, tzinfo=UTC)

V2_IDS: dict[str, str] = {
    "wp": "wp_20261201_200001",
    "ats": "ats_20261201_200002",
    "ou": "ou_20261201_200003",
    "blend": "blend_20261201_200500",
    "converter": MARKET_CONVERTER_ID,
}
ZERO: dict[str, float] = {"wp": 0.0, "ats": 0.0, "ou": 0.0}
BLEND_VERSION: str = "blend_20261202_023000"

# The 11 ledger stamp columns: no dict key the install writes may carry one of these names.
STAMP_COLUMNS: frozenset[str] = frozenset(
    IMMUTABLE_COLUMNS_V1[IMMUTABLE_COLUMNS_V1.index("decided_at_utc") + 1 :]
)
SWITCH_RECORD_FILES: tuple[str, str] = (
    RECORD_PATH,
    "backtest/midseason_switch_2026.py",
)


# ---------------------------------------------------------------------------
# Fixture stores (tmp only)
# ---------------------------------------------------------------------------


def schedule() -> pd.DataFrame:
    """A silver ``games`` frame: 2025, 2026 weeks 1-12 played, week 13 still to come."""
    rng = np.random.default_rng(7)
    rows: list[dict[str, Any]] = []

    def add(season: int, week: int, kickoff: datetime, played: bool) -> None:
        home = float(rng.integers(10, 35)) if played else np.nan
        away = float(rng.integers(10, 35)) if played else np.nan
        rows.append(
            {
                "game_id": f"{season}_{week:02d}_G{len(rows):03d}",
                "season": season,
                "week": week,
                "kickoff_et": pd.Timestamp(kickoff),
                "home_score": home,
                "away_score": away,
            }
        )

    first_2025 = datetime(2025, 9, 7, 13, 0, tzinfo=lock_rule.ET)
    for week in range(1, 19):
        for _game in range(2):
            add(2025, week, first_2025 + timedelta(weeks=week - 1), True)
    first_2026 = datetime(2026, 9, 13, 13, 0, tzinfo=lock_rule.ET)
    for week in range(1, 13):
        for _game in range(2):
            add(2026, week, first_2026 + timedelta(weeks=week - 1), True)
    add(2026, 12, datetime(2026, 11, 30, 20, 15, tzinfo=lock_rule.ET), True)
    add(2026, 13, datetime(2026, 12, 3, 20, 15, tzinfo=lock_rule.ET), False)
    add(2026, 13, datetime(2026, 12, 6, 13, 0, tzinfo=lock_rule.ET), False)
    return pd.DataFrame(rows)


def write_gold(gold: Path, games: pd.DataFrame) -> None:
    """One gold row per game per target, carrying the target's fixed v2 inputs."""
    gold.mkdir(parents=True)
    rng = np.random.default_rng(11)
    for target in V2_TARGETS:
        frame = games.loc[:, ["game_id", "season", "week"]].copy()
        for column in V2_INPUTS[target]:
            frame[column] = rng.normal(size=len(frame))
        frame["feature_timestamp"] = pd.Timestamp("2026-12-01T22:00:00Z")
        frame.to_parquet(gold / f"features_{target}.parquet", index=False)


@dataclass(frozen=True)
class InstallStores:
    """The four stores one install reads and writes, all under pytest's tmp_path."""

    artifacts: Path
    ledger: Path
    gold: Path
    silver: Path


def build_stores(tmp_path: Path) -> InstallStores:
    """Generation 1 serving (its record in the ledger), plus v1-shaped gold and silver games."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_converter(artifacts)
    write_bundle(artifacts, GENERATION_ONE_IDS)
    write_manifest_ids(artifacts, GENERATION_ONE_IDS)
    ledger = ledger_with_cadence_history(tmp_path)

    games = schedule()
    silver = tmp_path / "silver"
    silver.mkdir()
    games.to_parquet(silver / "games.parquet", index=False)
    gold = tmp_path / "gold"
    write_gold(gold, games)
    return InstallStores(artifacts=artifacts, ledger=ledger, gold=gold, silver=silver)


@dataclass
class GoldSpy:
    """The stubbed gold build: records each builder class, raising for any in *fail_on*."""

    fail_on: frozenset[type] = frozenset()
    calls: list[type] = field(default_factory=list)

    def __call__(self, builder: type) -> None:
        self.calls.append(builder)
        if builder in self.fail_on:
            msg = f"planted gold build failure: {builder.__name__}"
            raise RuntimeError(msg)


@dataclass
class TuneStub:
    """The stubbed blend tune: writes a real blend bound to the converter, returns its weights.

    Attributes:
        ou_weight: The O/U weight the "tune" answers (Study 1a earned 0).
        blend_digest: The gold digest the blend records; None records the models' digest.
        calls: The keyword arguments of every call.
    """

    ou_weight: float = 0.0
    blend_digest: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        artifacts: Path = kwargs["artifacts_dir"]
        digest = self.blend_digest or kwargs["gold_generation_digest"]
        write_blend(artifacts, BLEND_VERSION, kwargs["model_ids"], digest=digest)
        weights = SimpleNamespace(
            wp_model_weight=0.0, ats_model_weight=0.0, ou_model_weight=self.ou_weight
        )
        return SimpleNamespace(
            artifact_dir=artifacts / BLEND_VERSION,
            converter_artifact_id=MARKET_CONVERTER_ID,
            result=SimpleNamespace(weights=weights),
        )


def run_install(
    stores: InstallStores,
    spy: GoldSpy,
    monkeypatch: pytest.MonkeyPatch,
    tune: TuneStub,
    *,
    commit: str = CLEAN_COMMIT,
) -> owner_install.InstallOutcome:
    """``run_owner_install`` on the fixture stores, the clean-commit check and the tune stubbed."""
    monkeypatch.setattr(owner_install, "code_commit", lambda _root: commit)
    monkeypatch.setattr(owner_install, "tune_v2_blend", tune)
    return owner_install.run_owner_install(
        artifacts_dir=stores.artifacts,
        ledger_dir=stores.ledger,
        gold_dir=stores.gold,
        silver_dir=stores.silver,
        now_utc=NOW,
        build_gold=spy,
    )


def latest_bytes(stores: InstallStores) -> bytes:
    return (stores.artifacts / "latest.json").read_bytes()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ---------------------------------------------------------------------------
# Preconditions (Review Focus 2)
# ---------------------------------------------------------------------------


def test_the_install_refuses_before_week_12s_last_lock() -> None:
    with pytest.raises(InstallRefused, match="week"):
        install_preconditions(
            schedule(),
            now_utc=WEEK12_LAST_LOCK - timedelta(seconds=1),
            in_force=REFIT_RECIPE_ID,
        )


def test_the_install_may_run_from_week_12s_last_lock() -> None:
    """The control: at the last lock nothing of weeks 1-12 is still before its lock."""
    install_preconditions(
        schedule(), now_utc=WEEK12_LAST_LOCK, in_force=REFIT_RECIPE_ID
    )


def test_the_install_refuses_a_second_v2_install() -> None:
    with pytest.raises(InstallRefused, match=INSTALLED_RECIPE):
        install_preconditions(schedule(), now_utc=DEC1_15ET, in_force=INSTALLED_RECIPE)


def test_the_install_refuses_a_schedule_without_weeks_1_to_12() -> None:
    """No week-12 game in the schedule is no proof week 12 has locked: refused, never passed."""
    games = schedule()
    later_only = games.loc[(games["season"] == 2026) & (games["week"] >= 13)]
    with pytest.raises(InstallRefused, match="no 2026 game"):
        install_preconditions(later_only, now_utc=DEC1_15ET, in_force=REFIT_RECIPE_ID)


# ---------------------------------------------------------------------------
# The metadata and the bet rule
# ---------------------------------------------------------------------------


def test_v2_metadata_names_the_fixed_inputs_and_no_stamp_keys() -> None:
    meta = v2_model_metadata(
        "ou",
        cutoff_utc=CUT,
        training_rows=6600,
        gold_generation_digest="g",
        gold_history_digest="h" * 64,
        commit=CLEAN_COMMIT,
        feature_importances=dict.fromkeys(V2_INPUTS["ou"], 0.5),
    )
    assert meta["feature_names"] == list(V2_INPUTS["ou"])
    assert "recipe_id" not in meta and "arm" not in meta
    assert not set(meta) & STAMP_COLUMNS
    assert meta["v2_recipe"] == INSTALLED_RECIPE
    assert meta["target"] == "ou"
    assert meta["cutoff_utc"] == CUT.isoformat()
    assert meta["training_rows"] == 6600
    assert meta["best_params"] == {"alpha": 1.0}
    assert meta["exclude_groups"] == []
    assert meta["gold_generation_digest"] == "g"
    assert meta["code_commit"] == CLEAN_COMMIT
    assert meta["switch_record_sha256"] == {
        path: normalized_sha256(REPO_ROOT / path) for path in SWITCH_RECORD_FILES
    }
    wp = v2_model_metadata(
        "wp",
        cutoff_utc=CUT,
        training_rows=1,
        gold_generation_digest="g",
        gold_history_digest="h" * 64,
        commit=CLEAN_COMMIT,
        feature_importances=dict.fromkeys(V2_INPUTS["wp"], 0.5),
    )
    assert wp["best_params"] == {"C": 1.0}
    assert wp["label"] == "home_win"


def fitted_v2(target: str) -> Any:
    """The target's v2 recipe fitted on a small synthetic frame of its own inputs."""
    rng = np.random.default_rng(11)
    inputs = list(V2_INPUTS[target])
    frame = pd.DataFrame(rng.normal(size=(200, len(inputs))), columns=pd.Index(inputs))
    if target == "wp":
        label = (frame[inputs[0]] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    else:
        label = 3.0 * frame[inputs[0]] - frame[inputs[1]] + rng.normal(size=200)
    return build_v2_estimator(target).fit(frame, label)


def test_v2_importances_are_the_absolute_standardised_coefficients() -> None:
    for target in V2_TARGETS:
        model = fitted_v2(target)
        importances = v2_feature_importances(target, model)
        coefficients = np.ravel(model.named_steps["model"].coef_)
        assert list(importances) == list(V2_INPUTS[target]), target
        assert list(importances.values()) == [abs(float(c)) for c in coefficients]
        assert all(value >= 0.0 for value in importances.values())
    # WP keeps the home-win row of its one-row coefficient matrix.
    wp = fitted_v2("wp")
    assert wp.named_steps["model"].coef_.shape == (1, len(V2_INPUTS["wp"]))


def test_v2_importances_refuse_a_dropped_input() -> None:
    inputs = list(V2_INPUTS["ou"])
    frame = pd.DataFrame(
        np.random.default_rng(3).normal(size=(50, len(inputs))),
        columns=pd.Index(inputs),
    )
    frame["wind_mph"] = np.nan  # the median imputer drops an all-missing column
    model = build_v2_estimator("ou").fit(frame, frame[inputs[0]])
    with pytest.raises(ValueError, match="all-missing input was dropped"):
        v2_feature_importances("ou", model)


def test_v2_metadata_refuses_importances_not_keyed_by_the_inputs() -> None:
    with pytest.raises(ValueError, match="recipe inputs"):
        v2_model_metadata(
            "wp",
            cutoff_utc=CUT,
            training_rows=1,
            gold_generation_digest="g",
            gold_history_digest="h" * 64,
            commit=CLEAN_COMMIT,
            feature_importances={"elo_diff": 1.0},
        )


def test_the_v2_bet_rule_parses_with_no_floors_and_no_bands(tmp_path: Path) -> None:
    path, _sha = write_bet_rule_record(
        tmp_path,
        v2_bet_rule_payload(artifact_ids=V2_IDS, cutoff_utc=CUT, blend_weights=ZERO),
    )
    fits, thresholds = load_bet_rule_record(path)
    assert dict(thresholds) == {"wp": None, "ats": None, "ou": None}
    for target in V2_TARGETS:
        assert fits[target].ev_floor_t is None
        assert fits[target].frozen_sd is None
        assert fits[target].calibration_gate_passed is None
    record = json.loads(path.read_text(encoding="ascii"))
    assert record["refit_recipe"] == INSTALLED_RECIPE
    assert record["install_path"] == "owner"
    assert record["artifact_ids"] == V2_IDS
    assert record["blend_weights"] == ZERO


def test_the_v2_bet_rule_needs_all_five_ids() -> None:
    partial = {key: value for key, value in V2_IDS.items() if key != "converter"}
    with pytest.raises(ValueError, match="converter"):
        v2_bet_rule_payload(artifact_ids=partial, cutoff_utc=CUT, blend_weights=ZERO)


# ---------------------------------------------------------------------------
# The install, end to end
# ---------------------------------------------------------------------------


def test_a_successful_install_swaps_latest_json_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores = build_stores(tmp_path)
    spy = GoldSpy()
    tune = TuneStub()

    outcome = run_install(stores, spy, monkeypatch, tune)

    latest = json.loads((stores.artifacts / "latest.json").read_text())
    assert latest["wp"] == outcome.ids["wp"]
    for key in ("ats", "ou", "blend"):
        assert latest[key] == outcome.ids[key]
    assert outcome.ids["blend"] == BLEND_VERSION
    assert outcome.ids["converter"] == MARKET_CONVERTER_ID
    assert outcome.blend_weights == ZERO

    events = read_owner_installs(stores.ledger)
    assert [e["event"] for e in events] == ["install_intent", "installed"]
    intent, installed = events
    assert {intent["install_date"], installed["install_date"]} == {INSTALL_DATE}
    assert intent["cutoff_utc"] == NOW.isoformat()
    assert intent["recipe"] == INSTALLED_RECIPE
    assert intent["ids"] == outcome.ids
    assert intent["bet_rule_sha256"] == outcome.bet_rule_sha256
    assert intent["code_commit"] == CLEAN_COMMIT
    assert intent["before_ids"]["wp"] == GENERATION_ONE_IDS.wp
    assert installed["after_ids"] == outcome.ids

    assert in_force_edge_thresholds(stores.artifacts, stores.ledger) == {
        "wp": None,
        "ats": None,
        "ou": None,
    }
    assert in_force_recipe(stores.artifacts, stores.ledger) == INSTALLED_RECIPE
    assert [g.number for g in installed_generations(stores.ledger)] == [0, 1, 2]
    assert spy.calls == [FeatureMatrixBuilderV2Live]
    assert tune.calls[0]["cutoff_utc"] == NOW
    assert tune.calls[0]["model_ids"] == {t: outcome.ids[t] for t in V2_TARGETS}


def test_the_saved_models_are_the_fixed_recipe_trained_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores = build_stores(tmp_path)
    outcome = run_install(stores, GoldSpy(), monkeypatch, TuneStub())

    games = schedule()
    finished_before_now = int(
        (
            (pd.to_datetime(games["kickoff_et"], utc=True) < pd.Timestamp(NOW))
            & games["home_score"].notna()
        ).sum()
    )
    digest = read_owner_installs(stores.ledger)[0]["gold_generation_digest"]
    for target in V2_TARGETS:
        loaded = load_model_artifact(
            target, version=outcome.ids[target], artifacts_dir=stores.artifacts
        )
        assert loaded["feature_list"] == list(V2_INPUTS[target])
        metadata = loaded["metadata"]
        assert metadata["training_rows"] == finished_before_now
        assert metadata["gold_generation_digest"] == digest
        assert metadata["cutoff_utc"] == NOW.isoformat()
        assert [name for name, _step in loaded["model"].steps] == [
            "impute",
            "scale",
            "model",
        ]
    wp = load_model_artifact(
        "wp", version=outcome.ids["wp"], artifacts_dir=stores.artifacts
    )
    assert wp["metadata"]["preprocessing_is_model"] is True
    assert wp["preprocessing"] is wp["model"]


def test_the_saved_models_record_the_gold_history_they_were_fitted_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2002-2025 history key is recorded, and a 2026 append or a new build clock keeps it."""
    stores = build_stores(tmp_path)
    expected = owner_install.gold_history_content_sha256(stores.gold)
    outcome = run_install(stores, GoldSpy(), monkeypatch, TuneStub())
    for target in V2_TARGETS:
        loaded = load_model_artifact(
            target, version=outcome.ids[target], artifacts_dir=stores.artifacts
        )
        assert loaded["metadata"]["gold_history_content_sha256"] == expected

    wp_path = stores.gold / "features_wp.parquet"
    wp = pd.read_parquet(wp_path)
    appended = wp.loc[wp["season"] == 2026].head(1).assign(game_id="2026_14_NEW")
    moved_clock = pd.concat([wp, appended], ignore_index=True).assign(
        feature_timestamp=pd.Timestamp("2026-12-02T22:00:00Z")
    )
    moved_clock.to_parquet(wp_path, index=False)
    assert owner_install.gold_history_content_sha256(stores.gold) == expected

    history_edit = moved_clock.copy()
    first_2025 = history_edit.index[history_edit["season"] == 2025][0]
    history_edit.loc[first_2025, "elo_diff"] += 1.0
    history_edit.to_parquet(wp_path, index=False)
    assert owner_install.gold_history_content_sha256(stores.gold) != expected


def test_the_saved_models_carry_the_importances_the_cache_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model-insights page reads ``feature_importances`` from each served metadata.json."""
    stores = build_stores(tmp_path)
    outcome = run_install(stores, GoldSpy(), monkeypatch, TuneStub())

    conn = duckdb.connect()
    conn.execute(CACHE_SCHEMA)
    rows = _load_feature_importances(conn, stores.artifacts, dict(outcome.ids))
    assert rows == sum(len(V2_INPUTS[target]) for target in V2_TARGETS)
    for target in V2_TARGETS:
        loaded = load_model_artifact(
            target, version=outcome.ids[target], artifacts_dir=stores.artifacts
        )
        expected = v2_feature_importances(target, loaded["model"])
        assert loaded["metadata"]["feature_importances"] == expected
        served = dict(
            conn.execute(
                "SELECT feature_name, importance FROM feature_importances WHERE target = ?",
                [target],
            ).fetchall()
        )
        assert served == expected, target


def test_a_nonzero_tuned_weight_refuses_and_restores_the_v1_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores = build_stores(tmp_path)
    before = latest_bytes(stores)
    spy = GoldSpy()

    with pytest.raises(InstallRefused, match="weight"):
        run_install(stores, spy, monkeypatch, TuneStub(ou_weight=0.02))

    assert spy.calls == [FeatureMatrixBuilderV2Live, FeatureMatrixBuilder]
    assert latest_bytes(stores) == before and read_owner_installs(stores.ledger) == []


def test_a_failed_v2_gold_build_restores_the_v1_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores = build_stores(tmp_path)
    before = latest_bytes(stores)
    spy = GoldSpy(fail_on=frozenset({FeatureMatrixBuilderV2Live}))

    with pytest.raises(RuntimeError, match="planted gold build failure"):
        run_install(stores, spy, monkeypatch, TuneStub())

    assert spy.calls == [FeatureMatrixBuilderV2Live, FeatureMatrixBuilder]
    assert latest_bytes(stores) == before and read_owner_installs(stores.ledger) == []


def test_a_failed_v1_restore_names_the_recovery_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores = build_stores(tmp_path)
    spy = GoldSpy(fail_on=frozenset({FeatureMatrixBuilderV2Live, FeatureMatrixBuilder}))

    with pytest.raises(InstallRefused, match=r"scripts\.rebuild_gold"):
        run_install(stores, spy, monkeypatch, TuneStub())

    assert spy.calls == [FeatureMatrixBuilderV2Live, FeatureMatrixBuilder]


def test_a_dirty_tree_refuses_before_any_gold_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stores = build_stores(tmp_path)
    before = latest_bytes(stores)
    spy = GoldSpy()

    with pytest.raises(InstallRefused, match="dirty"):
        run_install(stores, spy, monkeypatch, TuneStub(), commit=DIRTY_COMMIT)

    assert spy.calls == []
    assert latest_bytes(stores) == before and read_owner_installs(stores.ledger) == []


def test_a_declined_intent_write_refuses_the_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Controller ruling 1: a sink that declines the intent means no swap (refit/swap.py:198)."""
    stores = build_stores(tmp_path)
    before = latest_bytes(stores)
    spy = GoldSpy()
    appended: list[Mapping[str, Any]] = []

    def declining(_ledger: Path, event: Mapping[str, Any]) -> bool:
        appended.append(event)
        return False

    monkeypatch.setattr(owner_install, "append_owner_install", declining)

    with pytest.raises(InstallRefused, match="not written"):
        run_install(stores, spy, monkeypatch, TuneStub())

    assert [event["event"] for event in appended] == ["install_intent"]
    assert spy.calls == [FeatureMatrixBuilderV2Live, FeatureMatrixBuilder]
    assert latest_bytes(stores) == before


def test_a_refused_swap_records_refused_and_restores_the_v1_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blend tuned on another gold fails replace_manifest's cross-check after the intent."""
    stores = build_stores(tmp_path)
    before = latest_bytes(stores)
    spy = GoldSpy()

    with pytest.raises(InstallRefused, match="bundle_validation"):
        run_install(stores, spy, monkeypatch, TuneStub(blend_digest="another-gold"))

    events = read_owner_installs(stores.ledger)
    assert [e["event"] for e in events] == ["install_intent", "refused"]
    refused = events[1]
    assert refused["install_date"] == events[0]["install_date"] == INSTALL_DATE
    assert refused["check"] == "bundle_validation"
    assert refused["exception_class"] == "ArtifactBundleInvalidError"
    assert spy.calls == [FeatureMatrixBuilderV2Live, FeatureMatrixBuilder]
    assert latest_bytes(stores) == before
    # The never-served intent is no generation.
    assert [g.number for g in installed_generations(stores.ledger)] == [0, 1]


def test_a_crashed_swap_records_refused_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Controller ruling 2: EVERY failure after the intent and before the swap returns."""
    stores = build_stores(tmp_path)
    before = latest_bytes(stores)
    spy = GoldSpy()

    def crashing(_mapping: Mapping[str, str], _artifacts_dir: Path) -> None:
        msg = "planted disk failure"
        raise OSError(msg)

    monkeypatch.setattr(owner_install, "replace_manifest", crashing)

    with pytest.raises(InstallRefused, match="planted disk failure"):
        run_install(stores, spy, monkeypatch, TuneStub())

    events = read_owner_installs(stores.ledger)
    assert [e["event"] for e in events] == ["install_intent", "refused"]
    assert events[1]["check"] == "swap"
    assert events[1]["exception_class"] == "OSError"
    assert events[1]["install_date"] == INSTALL_DATE
    assert spy.calls == [FeatureMatrixBuilderV2Live, FeatureMatrixBuilder]
    assert latest_bytes(stores) == before
    assert [g.number for g in installed_generations(stores.ledger)] == [0, 1]


def _swap_lands_then_raises(
    failure: BaseException,
) -> Callable[[Mapping[str, str], Path], None]:
    """The real ``replace_manifest``, then *failure*: a crash after latest.json was written."""
    real_replace_manifest = owner_install.replace_manifest

    def landing_then_raising(mapping: Mapping[str, str], artifacts_dir: Path) -> None:
        real_replace_manifest(mapping, artifacts_dir)
        raise failure

    return landing_then_raising


def _assert_the_landed_swap_is_installed(stores: InstallStores, spy: GoldSpy) -> None:
    """Recorded as installed, the v2 ids served, no v1 gold restore, the v2 recipe in force."""
    events = read_owner_installs(stores.ledger)
    assert [e["event"] for e in events] == ["install_intent", "installed"]
    intent, installed = events
    assert installed["install_date"] == intent["install_date"] == INSTALL_DATE
    assert installed["after_ids"] == intent["ids"]
    latest = json.loads((stores.artifacts / "latest.json").read_text())
    assert {key: latest[key] for key in ("wp", "ats", "ou", "blend")} == {
        key: intent["ids"][key] for key in ("wp", "ats", "ou", "blend")
    }
    assert spy.calls == [FeatureMatrixBuilderV2Live]
    assert in_force_recipe(stores.artifacts, stores.ledger) == INSTALLED_RECIPE


def test_an_error_after_the_swap_landed_is_an_install_not_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review fix R1: latest.json names the v2 ids, so the v2 models serve whatever failed after
    the write. Recording ``refused`` would hide a served generation (every later
    ``in_force_recipe`` raises) and rebuilding the v1 gold would feed it to the v2 models."""
    stores = build_stores(tmp_path)
    spy = GoldSpy()
    monkeypatch.setattr(
        owner_install,
        "replace_manifest",
        _swap_lands_then_raises(RuntimeError("planted failure after the write")),
    )

    outcome = run_install(stores, spy, monkeypatch, TuneStub())

    _assert_the_landed_swap_is_installed(stores, spy)
    assert read_owner_installs(stores.ledger)[0]["ids"] == outcome.ids


def test_an_interrupt_after_the_swap_landed_is_an_install_and_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review fix R1: an interrupt after the write still records the install, then propagates."""
    stores = build_stores(tmp_path)
    spy = GoldSpy()
    monkeypatch.setattr(
        owner_install, "replace_manifest", _swap_lands_then_raises(KeyboardInterrupt())
    )

    with pytest.raises(KeyboardInterrupt):
        run_install(stores, spy, monkeypatch, TuneStub())

    _assert_the_landed_swap_is_installed(stores, spy)


def test_the_manifest_serves_only_exactly_the_intended_pointers(tmp_path: Path) -> None:
    """Never raises: an absent or unparseable manifest proves nothing was swapped."""
    pointers = {key: V2_IDS[key] for key in ("wp", "ats", "ou", "blend")}
    latest = tmp_path / "latest.json"
    assert owner_install._manifest_serves(tmp_path, pointers) is False

    latest.write_text("{not json", encoding="ascii")
    assert owner_install._manifest_serves(tmp_path, pointers) is False

    latest.write_text(json.dumps({**pointers, "wp": "wp_older"}), encoding="ascii")
    assert owner_install._manifest_serves(tmp_path, pointers) is False

    latest.write_text(json.dumps({**pointers, "extra": "kept"}), encoding="ascii")
    assert owner_install._manifest_serves(tmp_path, pointers) is True
    assert owner_install._manifest_serves(tmp_path, {}) is False


# ---------------------------------------------------------------------------
# The blend tune's wiring and the owner command
# ---------------------------------------------------------------------------


def test_the_blend_tune_is_the_walk_forward_at_the_install_instant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``backtest/study_1a.py:481-495``'s calls, at the install instant, on the new models."""
    calls: dict[str, Any] = {}
    games = schedule()

    def walk_forward(gold_dir: Path, frame: pd.DataFrame, cutoff: datetime) -> Any:
        calls["walk_forward"] = (gold_dir, frame, cutoff)
        return "predictions"

    def corpus(silver_dir: Path, frame: pd.DataFrame, cutoff: datetime) -> Any:
        calls["corpus"] = (silver_dir, frame, cutoff)
        return "corpus"

    def converter(converter_id: str, artifacts_dir: Path) -> Any:
        calls["converter"] = (converter_id, artifacts_dir)
        return "converter"

    def slopes(loaded: Any) -> Any:
        calls["slopes"] = loaded
        return "slopes"

    def tune(*args: Any, **kwargs: Any) -> Any:
        calls["tune"] = (args, kwargs)
        return "run"

    monkeypatch.setattr(owner_install, "walk_forward_v2", walk_forward)
    monkeypatch.setattr(owner_install, "extended_corpus", corpus)
    monkeypatch.setattr(owner_install, "load_market_probability_artifact", converter)
    monkeypatch.setattr(owner_install, "extended_slopes", slopes)
    monkeypatch.setattr(owner_install, "tune_generation_blend", tune)
    model_ids = {t: V2_IDS[t] for t in V2_TARGETS}

    run = owner_install.tune_v2_blend(
        artifacts_dir=tmp_path / "a",
        silver_dir=tmp_path / "s",
        gold_dir=tmp_path / "g",
        games=games,
        cutoff_utc=NOW,
        model_ids=model_ids,
        gold_generation_digest="digest",
    )

    assert run == "run"
    assert calls["walk_forward"] == (tmp_path / "g", games, NOW)
    assert calls["corpus"] == (tmp_path / "s", games, NOW)
    assert calls["converter"] == (MARKET_CONVERTER_ID, tmp_path / "a")
    assert calls["slopes"] == "converter"
    args, kwargs = calls["tune"]
    assert args == (tmp_path / "a", tmp_path / "s", tmp_path / "g")
    assert kwargs == {
        "artifact_ids": model_ids,
        "predictions": "predictions",
        "corpus": "corpus",
        "slopes": "slopes",
        "gold_generation": "digest",
    }


def test_the_command_prints_installed_and_the_five_ids(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    def installed(**kwargs: Any) -> owner_install.InstallOutcome:
        seen.update(kwargs)
        return owner_install.InstallOutcome(
            ids=dict(V2_IDS), bet_rule_sha256="a" * 64, blend_weights=dict(ZERO)
        )

    monkeypatch.setattr(owner_install, "run_owner_install", installed)

    assert install_v2.main([]) == 0

    out = capsys.readouterr().out
    line = next(row for row in out.splitlines() if row.startswith("INSTALLED="))
    for key, value in V2_IDS.items():
        assert f"{key}={value}" in line
    assert seen["artifacts_dir"] == Path("artifacts")
    assert seen["gold_dir"] == Path("data/gold")
    assert seen["silver_dir"] == Path("data/silver")
    assert seen["ledger_dir"] == Path("ledger")
    assert seen["now_utc"].tzinfo is not None
    assert callable(seen["build_gold"])


def test_the_command_prints_the_refusal_and_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refused(**_kwargs: Any) -> owner_install.InstallOutcome:
        msg = "planted refusal"
        raise InstallRefused(msg)

    monkeypatch.setattr(owner_install, "run_owner_install", refused)

    assert install_v2.main([]) == 2

    assert "INSTALL_REFUSED= planted refusal" in capsys.readouterr().out
