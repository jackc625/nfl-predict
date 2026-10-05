"""The content-addressed decision-input snapshot and the reproduction key (Plan 34-08, LDGR-09).

WHAT THIS MODULE PINS
---------------------
A ledger row is only re-derivable if what the decision SAW was stored when it was decided: the
gold matrix is rebuilt nightly, and a gold row rebuilt after results land can differ from the
decision-time row (NF-08). So each decision run stores ONE snapshot -- the candidates frame the
selector received, the schedule spine, every target's gold rows as scored, and a meta record --
under ``ledger/snapshots/<digest>/``, where the digest is taken over the same kind of canonical
form the chain uses (LDGR-05), never over parquet bytes.

The tests prove the frame digest is independent of row order, column order and a parquet round
trip, that every null form digests alike, that a non-unique sort key is refused (a tie would let
row order move the digest), that the snapshot round-trips, is idempotent and is computable without
writing, that one changed byte names its part, and that a failed write leaves no directory. The
reproduction key's three halves are pinned too: the gold generation key equals the test-tier
formula it re-implements, the odds digest sees only rows admissible at the decision instant, and
the upstream key names every dataset's capture as the build read it.

Every store here lives under ``tmp_path``; production ``data/``, ``artifacts/`` and
``config/upstream_live/`` are never read.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from forward_ledger.snapshots import (
    SNAPSHOT_PART_KEYS,
    SNAPSHOT_PARTS,
    SNAPSHOTS_DIRNAME,
    SnapshotDigestMismatchError,
    compute_snapshot_digest,
    load_decision_snapshot,
    write_decision_snapshot,
)

import tests.gold_generation as test_gold_generation
from backtest.weekly_bet_list import DecisionBundle
from data.upstream_live import AS_OF_ENV, AsOfCapture, as_of_capture
from forward_ledger import repro_key
from forward_ledger.canonical import CanonicalValueError, frame_digest
from models.artifacts import ResolvedArtifacts

SEASON = 2026
WEEK = 6
GAME_IDS = ("2026_06_BUF_NYJ", "2026_06_KC_LV", "2026_06_SF_SEA")
RUN_INSTANT = datetime(2026, 10, 14, 21, 0, 5, tzinfo=UTC)
RESOLVED = ResolvedArtifacts(
    wp="wp_20990101_000001",
    ats="ats_20990101_000002",
    ou="ou_20990101_000003",
    blend="blend_20990101_000004",
    converter="market_probability_20990101_000000",
)


# ---------------------------------------------------------------------------
# Fixture builders (synthetic frames; tmp_path only)
# ---------------------------------------------------------------------------


def _mixed_frame() -> pd.DataFrame:
    """One column of every kind the snapshot parts carry, with nulls where a kind allows them."""
    return pd.DataFrame(
        {
            "game_id": list(GAME_IDS),
            "model_value": [0.5123456789012345, -0.0, 3.25],
            "rest_days": pd.array([7, None, 6], dtype="Int64"),
            "sportsbook": ["draftkings", None, "fanduel"],
            "is_live": [True, False, True],
            "snapshot_ts": pd.to_datetime(
                [
                    "2026-10-14T21:00:00.123456789Z",
                    None,
                    "2026-10-14T20:59:00Z",
                ],
                utc=True,
                format="ISO8601",
            ),
        }
    )


def _gold(target: str) -> pd.DataFrame:
    offset = {"wp": 0.0, "ats": 1.0, "ou": 2.0}[target]
    return pd.DataFrame(
        {
            "game_id": list(GAME_IDS),
            "season": [SEASON] * 3,
            "week": [WEEK] * 3,
            "elo_diff": [12.5 + offset, -3.0 + offset, 0.1 + offset],
            "feature_timestamp": pd.to_datetime(["2026-10-13T12:00:00Z"] * 3, utc=True),
        }
    )


def _candidates() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in ("wp", "ats", "ou"):
        for position, game_id in enumerate(GAME_IDS):
            rows.append(
                {
                    "game_id": game_id,
                    "target": target,
                    "model_value": 0.25 * position
                    + {"wp": 0.1, "ats": 1.5, "ou": 44.5}[target],
                    "sportsbook": "draftkings",
                    "is_live": False,
                    "snapshot_ts": pd.Timestamp("2026-10-14T20:30:00Z"),
                    "gameday": "2026-10-18",
                }
            )
    return pd.DataFrame(rows)


def _schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": list(GAME_IDS),
            "season": [SEASON] * 3,
            "week": [WEEK] * 3,
            "gameday": ["2026-10-18"] * 3,
        }
    )


def _bundle(
    tmp_path: Path, *, candidates: pd.DataFrame | None = None
) -> DecisionBundle:
    chain_fit = tmp_path / "chain_fit.json"
    if not chain_fit.exists():
        chain_fit.write_bytes(b'{"record_id": "synthetic_chain_fit"}\n')
    return DecisionBundle(
        frame=pd.DataFrame(),
        candidates=_candidates() if candidates is None else candidates,
        schedule=_schedule(),
        gold_inputs={target: _gold(target) for target in ("wp", "ats", "ou")},
        fits={},
        resolved=RESOLVED,
        chain_fit_path=chain_fit,
        run_instant=RUN_INSTANT,
    )


def _round_trip(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    frame.to_parquet(path, index=False)
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# frame_digest: the canonical form the snapshot digest is built from
# ---------------------------------------------------------------------------


def test_frame_digest_is_order_and_dtype_stable(tmp_path: Path) -> None:
    frame = _mixed_frame()
    digest = frame_digest(frame, ("game_id",))
    assert len(digest) == 64

    shuffled = frame.sample(frac=1.0, random_state=7)[list(reversed(frame.columns))]
    assert frame_digest(shuffled, ("game_id",)) == digest

    read_back = _round_trip(frame, tmp_path / "frame.parquet")
    assert frame_digest(read_back, ("game_id",)) == digest

    flipped = frame.copy()
    flipped.loc[0, "model_value"] = np.nextafter(flipped.loc[0, "model_value"], 1.0)
    assert frame_digest(flipped, ("game_id",)) != digest


def test_frame_digest_null_forms() -> None:
    digests = {
        repr(null): frame_digest(
            pd.DataFrame({"game_id": ["g1"], "value": pd.Series([null], dtype=object)}),
            ("game_id",),
        )
        for null in (None, float("nan"), pd.NA, pd.NaT)
    }
    assert len(set(digests.values())) == 1, digests


def test_frame_digest_refuses_duplicate_keys(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2"],
            "target": ["wp", "wp", "wp"],
            "v": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(CanonicalValueError, match=r"game_id.*target.*g1"):
        frame_digest(frame, ("game_id", "target"))

    duplicated = pd.concat([_candidates(), _candidates().iloc[[0]]], ignore_index=True)
    with pytest.raises(CanonicalValueError, match="candidates"):
        compute_snapshot_digest(_bundle(tmp_path, candidates=duplicated))


# ---------------------------------------------------------------------------
# The stored snapshot
# ---------------------------------------------------------------------------


def test_snapshot_written_and_reloaded(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    bundle = _bundle(tmp_path)

    digest = write_decision_snapshot(ledger, bundle)

    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)
    directory = ledger / SNAPSHOTS_DIRNAME / digest
    assert directory.is_dir()
    assert sorted(path.name for path in directory.iterdir()) == sorted(
        [f"{part}.parquet" for part in SNAPSHOT_PARTS] + ["meta.json"]
    )

    snapshot = load_decision_snapshot(ledger, digest)
    expected = {
        "candidates": bundle.candidates,
        "schedule": bundle.schedule,
        **{
            f"gold_{target}": bundle.gold_inputs[target]
            for target in ("wp", "ats", "ou")
        },
    }
    assert set(snapshot.frames) == set(SNAPSHOT_PARTS)
    for part, frame in expected.items():
        pd.testing.assert_frame_equal(
            snapshot.frames[part].reset_index(drop=True),
            frame.reset_index(drop=True),
        )
    meta = snapshot.meta
    assert meta["season"] == SEASON
    assert meta["week"] == WEEK
    assert meta["resolved"]["converter"] == RESOLVED.converter
    assert meta["run_instant"] == RUN_INSTANT.isoformat()
    assert set(meta["library_versions"]) >= {
        "python",
        "pandas",
        "pyarrow",
        "numpy",
        "scikit-learn",
        "xgboost",
    }
    assert len(meta["uv_lock_sha256"]) == 64
    assert len(meta["chain_fit_sha256"]) == 64


def test_snapshot_is_idempotent(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    first = write_decision_snapshot(ledger, _bundle(tmp_path))
    second = write_decision_snapshot(ledger, _bundle(tmp_path))
    assert first == second
    assert [path.name for path in (ledger / SNAPSHOTS_DIRNAME).iterdir()] == [first]


def test_compute_digest_equals_written_digest(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    bundle = _bundle(tmp_path)
    extra = {"lock_instant": "2026-10-14T18:00:00-04:00"}

    computed = compute_snapshot_digest(bundle, meta_extra=extra)
    assert not ledger.exists(), "computing the digest must write nothing"
    assert write_decision_snapshot(ledger, bundle, meta_extra=extra) == computed
    assert compute_snapshot_digest(bundle) != computed


def test_one_byte_change_names_the_part(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    digest = write_decision_snapshot(ledger, _bundle(tmp_path))
    part_path = ledger / SNAPSHOTS_DIRNAME / digest / "candidates.parquet"
    data = bytearray(part_path.read_bytes())
    data[len(data) // 2] ^= 0x01
    part_path.write_bytes(bytes(data))

    with pytest.raises(SnapshotDigestMismatchError, match="candidates") as raised:
        load_decision_snapshot(ledger, digest)
    assert raised.value.part == "candidates"


def test_no_partial_snapshot_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger"
    original = pd.DataFrame.to_parquet
    calls: list[int] = []

    def failing_to_parquet(self: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 3:
            msg = "disk full (simulated)"
            raise OSError(msg)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failing_to_parquet)

    with pytest.raises(OSError, match="simulated"):
        write_decision_snapshot(ledger, _bundle(tmp_path))
    snapshots_dir = ledger / SNAPSHOTS_DIRNAME
    leftovers = list(snapshots_dir.iterdir()) if snapshots_dir.exists() else []
    assert leftovers == [], f"a failed write left {leftovers}"


def test_part_keys_cover_exactly_the_parts() -> None:
    assert set(SNAPSHOT_PART_KEYS) == set(SNAPSHOT_PARTS)
    assert SNAPSHOT_PART_KEYS["candidates"] == ("game_id", "target")


# ---------------------------------------------------------------------------
# The reproduction key
# ---------------------------------------------------------------------------


def test_gold_generation_key_matches_test_formula(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold_dir = tmp_path / "data" / "gold"
    gold_dir.mkdir(parents=True)
    for target in ("wp", "ats", "ou"):
        _gold(target).to_parquet(gold_dir / f"features_{target}.parquet", index=False)

    monkeypatch.setattr(test_gold_generation, "REPO_ROOT", tmp_path)

    production = repro_key.gold_generation_key(gold_dir)
    assert production == test_gold_generation.gold_generation_key()

    (gold_dir / "features_ats.parquet").write_bytes(b"not the same bytes")
    assert repro_key.gold_generation_key(gold_dir) != production


def _odds(created_at: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [GAME_IDS[0]] * len(created_at),
            "sportsbook": [f"book{i}" for i in range(len(created_at))],
            "snapshot_ts": pd.to_datetime(
                ["2026-10-17T22:00:00Z"] * len(created_at), utc=True
            ),
            "created_at": pd.to_datetime(created_at, utc=True),
            "spread": [-3.5] * len(created_at),
        }
    )


def test_odds_snapshot_digest_admissible_only() -> None:
    decided_at = datetime(2026, 10, 14, 22, 0, tzinfo=UTC)
    base = _odds(["2026-10-14T20:00:00Z", "2026-10-14T21:00:00Z"])
    digest = repro_key.odds_snapshot_digest(base, GAME_IDS, decided_at)

    later = pd.concat([base, _odds(["2026-10-14T22:00:01Z"]).assign(sportsbook="late")])
    assert repro_key.odds_snapshot_digest(later, GAME_IDS, decided_at) == digest

    other_game = _odds(["2026-10-14T19:00:00Z"]).assign(game_id="2025_06_OTHER")
    with_other = pd.concat([base, other_game])
    assert repro_key.odds_snapshot_digest(with_other, GAME_IDS, decided_at) == digest

    earlier = pd.concat(
        [base, _odds(["2026-10-14T19:00:00Z"]).assign(sportsbook="early")]
    )
    assert repro_key.odds_snapshot_digest(earlier, GAME_IDS, decided_at) != digest

    at_instant = pd.concat(
        [base, _odds([decided_at.isoformat()]).assign(sportsbook="at_lock")]
    )
    assert repro_key.odds_snapshot_digest(at_instant, GAME_IDS, decided_at) != digest


def _manifest() -> dict[str, Any]:
    def capture(week: int, sequence: int, sha: str) -> dict[str, Any]:
        return {
            "week": week,
            "sequence": sequence,
            "sha256": sha,
            "path": f"bronze/{sha}",
        }

    return {
        "schema_version": 1,
        "season": SEASON,
        "datasets": {
            "schedules": {"captures": [capture(1, 1, "s11"), capture(2, 1, "s21")]},
            "pbp": {
                "captures": [
                    capture(1, 1, "p11"),
                    capture(2, 1, "p21"),
                    capture(2, 2, "p22"),
                ]
            },
        },
    }


def test_upstream_capture_key_names_every_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(AS_OF_ENV, raising=False)
    manifest = _manifest()

    newest = repro_key.upstream_capture_key(manifest)
    assert json.loads(newest) == {
        "pbp": [SEASON, 2, 2, "p22"],
        "schedules": [SEASON, 2, 1, "s21"],
    }
    assert newest == json.dumps(
        json.loads(newest), sort_keys=True, separators=(",", ":")
    )

    with as_of_capture(1):
        scoped = repro_key.upstream_capture_key(manifest)
    assert json.loads(scoped) == {
        "pbp": [SEASON, 1, 1, "p11"],
        "schedules": [SEASON, 1, 1, "s11"],
    }

    exact = repro_key.upstream_capture_key(manifest, as_of=AsOfCapture(2, 1))
    assert json.loads(exact) == {
        "pbp": [SEASON, 2, 1, "p21"],
        "schedules": [SEASON, 2, 1, "s21"],
    }


def test_build_repro_key_assembles_the_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(AS_OF_ENV, raising=False)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    for target in ("wp", "ats", "ou"):
        _gold(target).to_parquet(gold_dir / f"features_{target}.parquet", index=False)
    odds = _odds(["2026-10-14T20:00:00Z"])
    decided_at = RUN_INSTANT + timedelta(hours=1)

    key = repro_key.build_repro_key(
        manifest=_manifest(),
        gold_dir=gold_dir,
        odds=odds,
        game_ids=GAME_IDS,
        decided_at=decided_at,
    )

    assert key == repro_key.ReproKey(
        upstream_capture_key=repro_key.upstream_capture_key(_manifest()),
        gold_generation_key=repro_key.gold_generation_key(gold_dir),
        odds_snapshot_digest=repro_key.odds_snapshot_digest(odds, GAME_IDS, decided_at),
    )
