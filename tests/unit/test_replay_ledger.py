"""Replay every ledger row from what was stored when it was decided (Plan 34-17, LDGR-09, D-02).

WHAT IS PROVEN
--------------
A fixture season is written through the REAL path: tiny real models saved by
``models.artifacts.save_model_artifact`` under the in-force recipe's ids, the real decision bundle
(``build_weekly_decision_bundle`` over a synthetic gold, silver schedule and odds store), the one
stamping site, the content-addressed snapshot writer, the artifact and chain-fit copies and the
guarded ``commit_changes``. Replay then re-derives every row from ``ledger/`` alone and must
reproduce it: categoricals exactly, numerics within 1e-9, suppressed rows with their reason.

Every way replay can fail to match is named: a changed snapshot byte names the part, a removed
artifact copy names its id, a perturbed model output names the field. A migrated row with no
snapshot is "not replayable" (never a pass) unless it is a verdict row, which fails. A live and a
shadow row of the same game, week and target are two results (``LEDGER_ROW_KEY``). Closing values
written after the decision cannot move a replayed decision (CLV report-only).

Everything lives under pytest's temporary directories: production ``artifacts/``, ``data/``,
``outputs/`` and ``ledger/`` are never read or written.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, LogisticRegression

from backtest.diagnose import score_deployed_artifacts
from backtest.recipe_registry import RECIPE_REGISTRY
from backtest.weekly_bet_list import build_weekly_decision_bundle
from forward_ledger.artifacts_copy import (
    ARTIFACT_COPIES_DIRNAME,
    ensure_artifact_copies,
    ensure_recipe_record_copy,
)
from forward_ledger.canonical import ENTRY_KIND_ROW, IMMUTABLE_COLUMNS_V1
from forward_ledger.replay import (
    REPLAY_TOLERANCE,
    ReplayResult,
    replay_ledger,
    replay_snapshot_rows,
)
from forward_ledger.repro_key import ReproKey
from forward_ledger.schema import (
    ARM_LIVE,
    ARM_SHADOW,
    LEDGER_ROW_KEY,
    VERDICT_SCOPE_PRE_VERDICT,
    VERDICT_SCOPE_VERDICT,
)
from forward_ledger.snapshots import (
    SNAPSHOTS_DIRNAME,
    compute_snapshot_digest,
    write_decision_snapshot,
)
from forward_ledger.stamps import IN_FORCE_RECIPE_ID, stamp_ledger_rows
from forward_ledger.store import (
    LEDGER_STAMP_COLUMNS,
    build_entry,
    commit_changes,
    ledger_head,
    read_entries,
    write_entries,
)
from models.artifacts import save_model_artifact
from tests.fixtures.decision_frame import chain_fit_record

SEASON = 2026
WEEKS: tuple[int, ...] = (3, 4)
GAMES: dict[int, tuple[str, ...]] = {
    3: ("2026_03_KC_BUF", "2026_03_DAL_PHI", "2026_03_SF_SEA"),
    4: ("2026_04_NYJ_MIA", "2026_04_GB_CHI", "2026_04_LAR_ARI"),
}
# Each week's last game has no odds row, so it is suppressed as ``missing_snapshot``.
UNPRICED: frozenset[str] = frozenset(games[-1] for games in GAMES.values())

# Sunday 13:00 ET kickoffs; each game locks at 18:00 ET the day before (22:00 UTC in EDT).
KICKOFF: dict[int, datetime] = {
    3: datetime(2026, 9, 27, 17, 0, tzinfo=UTC),
    4: datetime(2026, 10, 4, 17, 0, tzinfo=UTC),
}
LOCK: dict[int, datetime] = {
    3: datetime(2026, 9, 26, 22, 0, tzinfo=UTC),
    4: datetime(2026, 10, 3, 22, 0, tzinfo=UTC),
}
DECIDED_AT: dict[int, datetime] = {
    week: lock - timedelta(hours=1) for week, lock in LOCK.items()
}

FEATURES: list[str] = ["f1", "f2"]
GAME_FEATURES: dict[str, tuple[float, float]] = {
    "2026_03_KC_BUF": (1.5, -0.5),
    "2026_03_DAL_PHI": (-1.2, 0.8),
    "2026_03_SF_SEA": (0.3, 0.1),
    "2026_04_NYJ_MIA": (-0.4, 1.6),
    "2026_04_GB_CHI": (2.0, 0.4),
    "2026_04_LAR_ARI": (-0.9, -1.1),
}
# (spread, total, ml_home, ml_away) for the priced games.
GAME_ODDS: dict[str, tuple[float, float, float, float]] = {
    "2026_03_KC_BUF": (-3.0, 47.0, -150.0, 130.0),
    "2026_03_DAL_PHI": (2.5, 41.5, 120.0, -140.0),
    "2026_04_NYJ_MIA": (1.0, 50.5, 105.0, -125.0),
    "2026_04_GB_CHI": (-7.5, 44.0, -320.0, 260.0),
}
CONVERTER_SLOPE = 0.15

_RECIPE = RECIPE_REGISTRY[IN_FORCE_RECIPE_ID]
MODEL_IDS: dict[str, str] = dict(_RECIPE.model_artifact_ids)
BLEND_ID: str = _RECIPE.blend_id
CONVERTER_ID: str = str(_RECIPE.converter_id)
REPRO = ReproKey(
    upstream_capture_key="fixture-upstream-capture",
    gold_generation_key="fixture-gold-generation",
    odds_snapshot_digest="fixture-odds-digest",
)

# The shadow row: one game/week/target that also carries a live row.
SHADOW_GAME = "2026_04_GB_CHI"
SHADOW_TARGET = "ats"


# ---------------------------------------------------------------------------
# The fixture season, written through the real path (tmp only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureSeason:
    """A written fixture ledger and the stores it was decided from."""

    ledger: Path
    artifacts: Path
    digests: dict[int, str]


def _train_models() -> dict[str, tuple[Any, Any]]:
    """Tiny real models over two features: WP logistic + isotonic, ATS/OU linear."""
    rng = np.random.default_rng(34_17)
    features = pd.DataFrame(rng.uniform(-2.0, 2.0, size=(200, 2)), columns=FEATURES)
    margin = 4.0 * features["f1"] - 2.0 * features["f2"] + rng.normal(0.0, 3.0, 200)
    total = 45.0 + 3.0 * features["f2"] + rng.normal(0.0, 4.0, 200)
    wins = (margin > 0).astype(int)

    wp = LogisticRegression().fit(features, wins)
    # Bounded inside (0, 1): the Kelly sizer refuses a probability of exactly 0 or 1.
    calibrator = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip").fit(
        wp.predict_proba(features)[:, 1], wins
    )
    return {
        "wp": (wp, calibrator),
        "ats": (LinearRegression().fit(features, margin), None),
        "ou": (LinearRegression().fit(features, total), None),
    }


def _write_artifacts(artifacts: Path) -> None:
    """Real saved models renamed to the recipe ids, the blend, its converter and latest.json."""
    staging = artifacts / "_staging"
    for target, (model, calibrator) in _train_models().items():
        saved = save_model_artifact(
            model,
            target,
            {"fixture": "test_replay_ledger"},
            FEATURES,
            calibrator=calibrator,
            artifacts_dir=staging,
            update_latest=False,
        )
        saved.rename(artifacts / MODEL_IDS[target])
    staging.rmdir()

    blend = artifacts / BLEND_ID
    blend.mkdir()
    (blend / "blend_weights.json").write_text(
        json.dumps(
            {
                "weights": {"wp": 0.0, "ats": 0.0, "ou": 0.0},
                "market_probability_artifact_id": CONVERTER_ID,
                "market_probability_slope_beta": CONVERTER_SLOPE,
            }
        ),
        encoding="utf-8",
    )
    converter = artifacts / CONVERTER_ID
    converter.mkdir()
    (converter / "metadata.json").write_text(
        json.dumps({"artifact_id": CONVERTER_ID, "slope_beta": CONVERTER_SLOPE}),
        encoding="utf-8",
    )
    (artifacts / "latest.json").write_text(
        json.dumps({**MODEL_IDS, "blend": BLEND_ID}), encoding="utf-8"
    )


def _write_gold(gold: Path) -> None:
    rows = [
        {
            "game_id": game_id,
            "season": SEASON,
            "week": week,
            "f1": GAME_FEATURES[game_id][0],
            "f2": GAME_FEATURES[game_id][1],
            "home_win": np.nan,
            "home_margin": np.nan,
            "total_points": np.nan,
        }
        for week, games in GAMES.items()
        for game_id in games
    ]
    gold.mkdir()
    for target in ("wp", "ats", "ou"):
        pd.DataFrame(rows).to_parquet(gold / f"features_{target}.parquet", index=False)


def _write_silver(silver: Path) -> None:
    silver.mkdir()
    pd.DataFrame(
        {
            "game_id": [game for week in WEEKS for game in GAMES[week]],
            "season": [SEASON] * 6,
            "week": [week for week in WEEKS for _game in GAMES[week]],
            "kickoff_et": [
                pd.Timestamp(KICKOFF[week]) for week in WEEKS for _ in GAMES[week]
            ],
            "home_score": pd.Series([np.nan] * 6, dtype="float64"),
            "away_score": pd.Series([np.nan] * 6, dtype="float64"),
        }
    ).to_parquet(silver / "games.parquet", index=False)

    odds_rows = []
    for week in WEEKS:
        for game_id in GAMES[week]:
            if game_id in UNPRICED:
                continue
            spread, total, ml_home, ml_away = GAME_ODDS[game_id]
            odds_rows.append(
                {
                    "game_id": game_id,
                    "sportsbook": "draftkings",
                    "snapshot_ts": pd.Timestamp(LOCK[week]),
                    "created_at": pd.Timestamp(LOCK[week] - timedelta(hours=2)),
                    "spread": spread,
                    "total": total,
                    "ml_home": ml_home,
                    "ml_away": ml_away,
                    "spread_ju_home": -110.0,
                    "spread_ju_away": -110.0,
                    "total_over_ju": -110.0,
                    "total_under_ju": -110.0,
                    "is_live": False,
                }
            )
    pd.DataFrame(odds_rows).to_parquet(silver / "odds_snapshot.parquet", index=False)


def _immutable_rows(stamped: pd.DataFrame) -> list[dict[str, Any]]:
    return stamped[list(IMMUTABLE_COLUMNS_V1)].to_dict(orient="records")


def build_fixture_season(root: Path) -> FixtureSeason:
    """Decide weeks 3 and 4 through the real path and write them to a ledger under *root*."""
    artifacts = root / "artifacts"
    gold = root / "gold"
    silver = root / "silver"
    ledger = root / "ledger"
    chain_fit = root / "chain_fit.json"
    artifacts.mkdir(parents=True)
    _write_artifacts(artifacts)
    _write_gold(gold)
    _write_silver(silver)
    chain_fit.write_text(json.dumps(chain_fit_record(0.0)), encoding="utf-8")

    digests: dict[int, str] = {}
    for week in WEEKS:
        bundle = build_weekly_decision_bundle(
            SEASON,
            week,
            artifacts_dir=artifacts,
            gold_dir=gold,
            silver_dir=silver,
            chain_fit_path=chain_fit,
            now=DECIDED_AT[week],
        )
        digest = compute_snapshot_digest(bundle)
        stamped = stamp_ledger_rows(
            bundle.frame,
            resolved=bundle.resolved,
            repro=REPRO,
            snapshot_digest=digest,
            scope=None,
            ledger_has_verdict_rows=False,
        )
        assert write_decision_snapshot(ledger, bundle) == digest
        assert bundle.resolved is not None
        ensure_artifact_copies(ledger, bundle.resolved, artifacts)
        ensure_recipe_record_copy(ledger, bundle.chain_fit_path)
        commit_changes(ledger, new_rows=_immutable_rows(stamped))
        digests[week] = digest

        if week == 4:
            # A shadow row over the SAME bundle and snapshot as a live row of one quad.
            mask = (bundle.frame["game_id"] == SHADOW_GAME) & (
                bundle.frame["target"] == SHADOW_TARGET
            )
            shadow = stamp_ledger_rows(
                bundle.frame[mask],
                resolved=bundle.resolved,
                repro=REPRO,
                snapshot_digest=digest,
                scope=None,
                ledger_has_verdict_rows=False,
                arm=ARM_SHADOW,
            )
            commit_changes(ledger, new_rows=_immutable_rows(shadow))
    return FixtureSeason(ledger=ledger, artifacts=artifacts, digests=digests)


@pytest.fixture(scope="module")
def fixture_season(tmp_path_factory: pytest.TempPathFactory) -> FixtureSeason:
    return build_fixture_season(tmp_path_factory.mktemp("replay_fixture_season"))


@pytest.fixture
def ledger(fixture_season: FixtureSeason, tmp_path: Path) -> Path:
    """A private copy of the fixture ledger, so a test may tamper with it."""
    copy = tmp_path / "ledger"
    shutil.copytree(fixture_season.ledger, copy)
    return copy


def _rows(ledger_dir: Path) -> list[dict[str, Any]]:
    return [
        entry.immutable
        for entry in read_entries(ledger_dir)
        if entry.kind == ENTRY_KIND_ROW
    ]


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in LEDGER_ROW_KEY)


def _by_key(results: list[ReplayResult]) -> dict[tuple[Any, ...], ReplayResult]:
    return {result.key: result for result in results}


def _append_migrated_row(
    ledger_dir: Path, *, game_id: str, verdict_scope: str
) -> tuple[Any, ...]:
    """Append a NULL-snapshot row (the migration's shape) onto the chain; return its key."""
    entries = read_entries(ledger_dir)
    template = next(
        entry.immutable for entry in entries if entry.kind == ENTRY_KIND_ROW
    )
    values = {
        **template,
        "game_id": game_id,
        "week": 2,
        "verdict_scope": verdict_scope,
        "regime_label": None,
        **dict.fromkeys(LEDGER_STAMP_COLUMNS),
    }
    head, count = ledger_head(entries)
    entry = build_entry(head, count, ENTRY_KIND_ROW, values)
    write_entries(ledger_dir, [*entries, entry])
    return _key(values)


# ---------------------------------------------------------------------------
# Task 1: the replay engine
# ---------------------------------------------------------------------------


def test_fixture_season_replays_exactly(ledger: Path) -> None:
    rows = _rows(ledger)
    statuses = {row["status"] for row in rows}
    assert statuses == {"live", "suppressed"}, statuses
    assert any(row["rejection_reason"] == "missing_snapshot" for row in rows)

    results = replay_ledger(ledger)

    assert [result.key for result in results] == [_key(row) for row in rows]
    failing = [result for result in results if result.status != "pass"]
    assert failing == []
    assert all(result.mismatches == () for result in results)
    assert REPLAY_TOLERANCE == 1e-9


def test_rows_share_a_snapshot(ledger: Path, fixture_season: FixtureSeason) -> None:
    directories = sorted(path.name for path in (ledger / SNAPSHOTS_DIRNAME).iterdir())
    assert directories == sorted(fixture_season.digests.values())

    rows = _rows(ledger)
    for week, digest in fixture_season.digests.items():
        week_rows = [row for row in rows if row["week"] == week]
        assert {row["decision_snapshot_digest"] for row in week_rows} == {digest}
        assert len({row["game_id"] for row in week_rows}) == len(GAMES[week])

        results = replay_snapshot_rows(ledger, digest, week_rows)
        assert [result.status for result in results] == ["pass"] * len(week_rows)


def test_snapshot_byte_change_named(
    ledger: Path, fixture_season: FixtureSeason
) -> None:
    digest = fixture_season.digests[3]
    part = ledger / SNAPSHOTS_DIRNAME / digest / "gold_ats.parquet"
    data = bytearray(part.read_bytes())
    data[len(data) // 2] ^= 0x01
    part.write_bytes(bytes(data))

    results = _by_key(replay_ledger(ledger))
    week3 = [result for key, result in results.items() if key[2] == 3]
    assert week3
    for result in week3:
        assert result.status == "fail"
        assert result.reason is not None
        assert "digest mismatch" in result.reason
        assert "gold_ats" in result.reason
    assert all(
        result.status == "pass" for key, result in results.items() if key[2] == 4
    )


def test_missing_artifact_named(ledger: Path) -> None:
    shutil.rmtree(ledger / ARTIFACT_COPIES_DIRNAME / MODEL_IDS["ou"])

    results = replay_ledger(ledger)

    assert results
    for result in results:
        assert result.status == "fail"
        assert result.reason is not None
        assert MODEL_IDS["ou"] in result.reason


def test_perturbed_model_output_names_field(
    ledger: Path, fixture_season: FixtureSeason
) -> None:
    def perturbed(target: str, **kwargs: Any) -> pd.DataFrame:
        scored = score_deployed_artifacts(target, **kwargs)
        if target == "wp":
            scored["model_prob"] = scored["model_prob"] + 1e-6
        return scored

    digest = fixture_season.digests[3]
    rows = [row for row in _rows(ledger) if row["decision_snapshot_digest"] == digest]
    results = _by_key(replay_snapshot_rows(ledger, digest, rows, scorer=perturbed))

    wp_rows = [
        row for row in rows if row["target"] == "wp" and row["model_value"] is not None
    ]
    assert wp_rows
    for row in wp_rows:
        result = results[_key(row)]
        assert result.status == "fail"
        assert "model_value" in result.mismatches
    for result in results.values():
        if result.status == "fail":
            assert result.mismatches, result


def test_migrated_row_not_replayable(ledger: Path) -> None:
    key = _append_migrated_row(
        ledger, game_id="2026_02_NE_NYG", verdict_scope=VERDICT_SCOPE_PRE_VERDICT
    )

    results = _by_key(replay_ledger(ledger))

    assert results[key].status == "not_replayable"
    assert results[key].reason is not None
    others = [result for result_key, result in results.items() if result_key != key]
    assert all(result.status == "pass" for result in others)


def test_verdict_row_without_snapshot_fails(ledger: Path) -> None:
    key = _append_migrated_row(
        ledger, game_id="2026_02_NE_NYG", verdict_scope=VERDICT_SCOPE_VERDICT
    )

    result = _by_key(replay_ledger(ledger))[key]

    assert result.status == "fail"
    assert result.reason is not None
    assert "snapshot" in result.reason


def test_closing_columns_do_not_affect_replay(ledger: Path) -> None:
    before = replay_ledger(ledger)
    closing = {
        _key(row): {
            "closing_line": -10.5,
            "closing_odds": 250.0,
            "closing_sportsbook": "fanduel",
            "closing_captured_at": "2026-10-03T21:59:00+00:00",
            "forward_clv": 9.75 + index,
        }
        for index, row in enumerate(_rows(ledger))
    }
    assert commit_changes(ledger, closing_updates=closing).wrote

    after = replay_ledger(ledger)

    assert after == before
    assert all(result.status == "pass" for result in after)


def test_reads_only_ledger_copies(
    ledger: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert not (elsewhere / "artifacts").exists()
    assert not (elsewhere / "outputs").exists()

    results = replay_ledger(ledger)

    assert results
    assert all(result.status == "pass" for result in results)


def test_live_and_shadow_same_quad_replay_separately(ledger: Path) -> None:
    results = replay_ledger(ledger)
    quad = (SHADOW_GAME, SEASON, 4, SHADOW_TARGET)
    same_quad = [result for result in results if result.key[:4] == quad]

    assert sorted(result.key[4] for result in same_quad) == sorted(
        [ARM_LIVE, ARM_SHADOW]
    )
    assert all(result.status == "pass" for result in same_quad)


# ---------------------------------------------------------------------------
# Task 2: the replay CLI
# ---------------------------------------------------------------------------


def _run_cli(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, list[str]]:
    from scripts import replay_ledger as cli

    code = cli.main(list(args))
    return code, capsys.readouterr().out.splitlines()


def _key_text(key: tuple[Any, ...]) -> str:
    return "|".join(str(part) for part in key)


def _replay_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("REPLAY= ")]


def test_cli_all_pass_exit_zero(
    ledger: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    migrated = _append_migrated_row(
        ledger, game_id="2026_02_NE_NYG", verdict_scope=VERDICT_SCOPE_PRE_VERDICT
    )
    rows = _rows(ledger)

    code, lines = _run_cli(capsys, "--ledger-dir", str(ledger))

    replayed = _replay_lines(lines)
    assert len(replayed) == len(rows)
    for line, row in zip(replayed, rows, strict=True):
        status = "not_replayable" if _key(row) == migrated else "pass"
        assert line.startswith(f"REPLAY= {_key_text(_key(row))} {status}"), line
    assert f"REPLAY_PASS= {len(rows) - 1}" in lines
    assert "REPLAY_NOT_REPLAYABLE= 1" in lines
    assert "REPLAY_FAIL= 0" in lines
    assert code == 0


def test_cli_failure_exit_one(
    ledger: Path, fixture_season: FixtureSeason, capsys: pytest.CaptureFixture[str]
) -> None:
    part = ledger / SNAPSHOTS_DIRNAME / fixture_season.digests[3] / "gold_ats.parquet"
    data = bytearray(part.read_bytes())
    data[len(data) // 2] ^= 0x01
    part.write_bytes(bytes(data))
    week3 = [row for row in _rows(ledger) if row["week"] == 3]

    code, lines = _run_cli(capsys, "--ledger-dir", str(ledger))

    for row in week3:
        line = next(
            line
            for line in _replay_lines(lines)
            if line.startswith(f"REPLAY= {_key_text(_key(row))} ")
        )
        assert line.startswith(f"REPLAY= {_key_text(_key(row))} fail "), line
        assert "gold_ats" in line
    assert f"REPLAY_FAIL= {len(week3)}" in lines
    assert code == 1


def test_cli_filters(ledger: Path, capsys: pytest.CaptureFixture[str]) -> None:
    expected = [
        row
        for row in _rows(ledger)
        if row["week"] == 4 and row["game_id"] == SHADOW_GAME
    ]
    assert len(expected) == 4  # wp, ats, ou live and the ats shadow

    code, lines = _run_cli(
        capsys, "--ledger-dir", str(ledger), "--week", "4", "--game-id", SHADOW_GAME
    )

    assert _replay_lines(lines) == [
        f"REPLAY= {_key_text(_key(row))} pass" for row in expected
    ]
    assert f"REPLAY_PASS= {len(expected)}" in lines
    assert code == 0


def test_cli_unreadable_ledger_exit_two(
    ledger: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store_file = ledger / "forward_2026.jsonl"
    original = store_file.read_bytes()

    # A broken chain: one stored value changed, the line still parses.
    lines = original.split(b"\n")
    first = json.loads(lines[0])
    first["immutable"]["model_value"] = 0.123
    lines[0] = json.dumps(first, separators=(",", ":")).encode("ascii")
    store_file.write_bytes(b"\n".join(lines))
    code, output = _run_cli(capsys, "--ledger-dir", str(ledger))
    assert code == 2
    assert _replay_lines(output) == []

    # An unreadable store.
    store_file.write_bytes(b"not a ledger line\n")
    code, output = _run_cli(capsys, "--ledger-dir", str(ledger))
    assert code == 2
    assert any(line.startswith("LEDGER_UNREADABLE= ") for line in output)
    assert _replay_lines(output) == []
