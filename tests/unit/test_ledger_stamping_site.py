"""Self-contained artifact copies and the ONE stamping site (Plan 34-08, LDGR-03, LDGR-11, D-02).

WHAT THIS MODULE PINS
---------------------
Two halves of "a stamped row plus ``ledger/`` alone carries everything replay needs":

1. **The ledger keeps its own copies** (D-02, D-11). Every model, blend AND bound converter
   artifact directory a row references -- the converter too, because the WP second test reads its
   slope (34-RESEARCH Pitfall 16) -- is copied into ``ledger/artifacts/<id>/`` on first reference
   and verified by tree digest; a later call copies nothing, and a copy that has drifted from what
   scored the row is refused by name. The chain-fit record is copied by its sha256.

2. **One site stamps the 11 Phase-34 immutable columns.** ``forward_ledger.stamps`` sets
   ``arm``, the model / blend / recipe / fill ids, the three reproduction-key values, the snapshot
   digest, the verdict-scope label and the regime label -- and nothing else may. It refuses scoring
   ids the in-force recipe does not list, and once any verdict row exists it refuses to label a new
   row without a declaration instead of silently calling it pre-verdict (34-RESEARCH Pitfall 13).
   An AST scan over the production tree holds the single-site rule, with controls proving it can
   fail and does not flag the migration's NULL fill.

Every store here lives under ``tmp_path``; production ``artifacts/`` is never read.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.recipe_registry import RECIPE_REGISTRY
from forward_ledger.artifacts_copy import (
    ARTIFACT_COPIES_DIRNAME,
    RECIPE_COPIES_DIRNAME,
    ArtifactCopyMismatchError,
    ensure_artifact_copies,
    ensure_recipe_record_copy,
    tree_digest,
)
from forward_ledger.canonical import IMMUTABLE_COLUMNS_V1
from forward_ledger.declarations import (
    VerdictScope,
    VerdictScopeUndeclaredError,
    verdict_scope_label,
)
from forward_ledger.repro_key import ReproKey
from forward_ledger.schema import (
    ARM_LIVE,
    BET_LIST_COLUMNS,
    REGIME_LABEL_BOOTSTRAP,
    VERDICT_SCOPE_PRE_VERDICT,
)
from forward_ledger.stamps import (
    IN_FORCE_RECIPE_ID,
    RecipeArtifactMismatchError,
    stamp_ledger_rows,
)
from forward_ledger.store import MissingStampError
from models.artifacts import ResolvedArtifacts

REPO_ROOT = Path(__file__).resolve().parents[2]

# The 11 Phase-34 immutable columns: every v1 immutable column after ``decided_at_utc``.
PHASE34_STAMP_COLUMNS: tuple[str, ...] = IMMUTABLE_COLUMNS_V1[
    IMMUTABLE_COLUMNS_V1.index("decided_at_utc") + 1 :
]

_RECIPE = RECIPE_REGISTRY[IN_FORCE_RECIPE_ID]
IN_FORCE_RESOLVED = ResolvedArtifacts(
    wp=_RECIPE.model_artifact_ids["wp"],
    ats=_RECIPE.model_artifact_ids["ats"],
    ou=_RECIPE.model_artifact_ids["ou"],
    blend=_RECIPE.blend_id,
    converter=_RECIPE.converter_id,
)
REPRO = ReproKey(
    upstream_capture_key='{"pbp":[2026,6,1,"aa"]}',
    gold_generation_key="b" * 64,
    odds_snapshot_digest="c" * 64,
)
SNAPSHOT_DIGEST = "d" * 64
SCOPE = VerdictScope(
    season=2026,
    start_week=6,
    end_week=22,
    includes_playoff_weeks=True,
    includes_neutral_site_games=True,
    counted_arm=ARM_LIVE,
    outcome_rule="synthetic",
    fill_convention_id="fill-v1",
    bootstrap_regime_weeks=(2, 3, 4),
)

SYNTHETIC_RESOLVED = ResolvedArtifacts(
    wp="wp_20990101_000001",
    ats="ats_20990101_000002",
    ou="ou_20990101_000003",
    blend="blend_20990101_000004",
    converter="market_probability_20990101_000000",
)


# ---------------------------------------------------------------------------
# Fixture builders (tmp_path only)
# ---------------------------------------------------------------------------


def _artifacts_dir(root: Path) -> Path:
    artifacts = root / "artifacts"
    for artifact_id in (
        SYNTHETIC_RESOLVED.wp,
        SYNTHETIC_RESOLVED.ats,
        SYNTHETIC_RESOLVED.ou,
        SYNTHETIC_RESOLVED.blend,
        SYNTHETIC_RESOLVED.converter,
    ):
        assert artifact_id is not None
        directory = artifacts / artifact_id
        (directory / "nested").mkdir(parents=True)
        (directory / "payload.json").write_bytes(
            f'{{"id": "{artifact_id}"}}\n'.encode()
        )
        (directory / "nested" / "model.bin").write_bytes(artifact_id.encode() * 3)
    return artifacts


def _frame(weeks: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week in weeks:
        for target in ("wp", "ats", "ou"):
            row: dict[str, Any] = dict.fromkeys(BET_LIST_COLUMNS)
            row.update(
                {
                    "game_id": f"2026_{week:02d}_BUF_NYJ",
                    "season": 2026,
                    "week": week,
                    "target": target,
                    "decided_at_utc": "2026-10-14T21:00:05+00:00",
                    "provenance": "forward",
                    "validation_type": "forward_realized",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=pd.Index(BET_LIST_COLUMNS))


def _stamp(frame: pd.DataFrame, **overrides: Any) -> pd.DataFrame:
    arguments: dict[str, Any] = {
        "resolved": IN_FORCE_RESOLVED,
        "repro": REPRO,
        "snapshot_digest": SNAPSHOT_DIGEST,
        "scope": SCOPE,
        "ledger_has_verdict_rows": False,
    }
    arguments.update(overrides)
    return stamp_ledger_rows(frame, **arguments)


# ---------------------------------------------------------------------------
# Self-contained artifact copies (D-02, D-11, Pitfall 16)
# ---------------------------------------------------------------------------


def test_copies_every_referenced_artifact_including_converter(tmp_path: Path) -> None:
    artifacts = _artifacts_dir(tmp_path)
    ledger = tmp_path / "ledger"

    copied = ensure_artifact_copies(ledger, SYNTHETIC_RESOLVED, artifacts_dir=artifacts)

    expected = [
        SYNTHETIC_RESOLVED.wp,
        SYNTHETIC_RESOLVED.ats,
        SYNTHETIC_RESOLVED.ou,
        SYNTHETIC_RESOLVED.blend,
        SYNTHETIC_RESOLVED.converter,
    ]
    assert sorted(copied) == sorted(expected)
    copies = ledger / ARTIFACT_COPIES_DIRNAME
    assert sorted(path.name for path in copies.iterdir()) == sorted(expected)
    assert (copies / "market_probability_20990101_000000").is_dir()
    for artifact_id in expected:
        assert artifact_id is not None
        assert tree_digest(copies / artifact_id) == tree_digest(artifacts / artifact_id)


def test_copy_is_idempotent_and_detects_drift(tmp_path: Path) -> None:
    artifacts = _artifacts_dir(tmp_path)
    ledger = tmp_path / "ledger"
    ensure_artifact_copies(ledger, SYNTHETIC_RESOLVED, artifacts_dir=artifacts)

    assert (
        ensure_artifact_copies(ledger, SYNTHETIC_RESOLVED, artifacts_dir=artifacts)
        == []
    )

    drifted = (
        ledger / ARTIFACT_COPIES_DIRNAME / SYNTHETIC_RESOLVED.blend / "payload.json"
    )
    data = bytearray(drifted.read_bytes())
    data[0] ^= 0x01
    drifted.write_bytes(bytes(data))
    with pytest.raises(ArtifactCopyMismatchError, match=SYNTHETIC_RESOLVED.blend):
        ensure_artifact_copies(ledger, SYNTHETIC_RESOLVED, artifacts_dir=artifacts)


def test_recipe_record_copied_by_sha(tmp_path: Path) -> None:
    chain_fit = tmp_path / "neutral_hfa_chain_fit.json"
    chain_fit.write_bytes(b'{"record_id": "synthetic"}\n')
    ledger = tmp_path / "ledger"

    sha = ensure_recipe_record_copy(ledger, chain_fit)

    assert sha == hashlib.sha256(chain_fit.read_bytes()).hexdigest()
    copy = ledger / RECIPE_COPIES_DIRNAME / f"{sha}.json"
    assert copy.read_bytes() == chain_fit.read_bytes()
    assert ensure_recipe_record_copy(ledger, chain_fit) == sha
    assert [path.name for path in (ledger / RECIPE_COPIES_DIRNAME).iterdir()] == [
        copy.name
    ]


# ---------------------------------------------------------------------------
# The one stamping site (LDGR-03, LDGR-11)
# ---------------------------------------------------------------------------


def test_stamp_sets_every_stamp_column() -> None:
    frame = _frame((5, 6))
    before = frame.copy()

    stamped = _stamp(frame)

    pd.testing.assert_frame_equal(frame, before)
    assert list(stamped.columns) == list(BET_LIST_COLUMNS)
    for _, row in stamped.iterrows():
        assert row["arm"] == ARM_LIVE
        assert row["model_artifact_id"] == IN_FORCE_RESOLVED.model_id_for(row["target"])
        assert row["blend_id"] == IN_FORCE_RESOLVED.blend
        assert row["recipe_id"] == IN_FORCE_RECIPE_ID
        assert row["fill_convention_id"] == "fill-v1"
        assert row["upstream_capture_key"] == REPRO.upstream_capture_key
        assert row["gold_generation_key"] == REPRO.gold_generation_key
        assert row["odds_snapshot_digest"] == REPRO.odds_snapshot_digest
        assert row["decision_snapshot_digest"] == SNAPSHOT_DIGEST
        assert row["verdict_scope"] == verdict_scope_label(
            int(row["season"]), int(row["week"]), SCOPE
        )
    assert set(stamped.loc[stamped["week"] == 5, "verdict_scope"]) == {"pre_verdict"}
    assert set(stamped.loc[stamped["week"] == 6, "verdict_scope"]) == {"verdict"}
    untouched = ["decided_at_utc", "provenance", "validation_type"]
    pd.testing.assert_frame_equal(stamped[untouched], before[untouched])


def test_stamp_refuses_ids_outside_the_recipe() -> None:
    frame = _frame((6,))
    with pytest.raises(RecipeArtifactMismatchError, match="wp_20990101_000001"):
        _stamp(frame, resolved=replace(IN_FORCE_RESOLVED, wp="wp_20990101_000001"))
    with pytest.raises(RecipeArtifactMismatchError, match="converter"):
        _stamp(frame, resolved=replace(IN_FORCE_RESOLVED, converter=None))


def test_stamp_refuses_an_unresolved_decision() -> None:
    with pytest.raises(MissingStampError, match="resolved"):
        _stamp(_frame((6,)), resolved=None)


def test_bootstrap_label_only_weeks_2_to_4() -> None:
    stamped = _stamp(_frame((1, 2, 3, 4, 5)))
    by_week = stamped.groupby("week")["regime_label"].agg(lambda s: set(s.tolist()))
    assert by_week.to_dict() == {
        1: {None},
        2: {REGIME_LABEL_BOOTSTRAP},
        3: {REGIME_LABEL_BOOTSTRAP},
        4: {REGIME_LABEL_BOOTSTRAP},
        5: {None},
    }


def test_undeclared_scope_refused_once_verdict_rows_exist() -> None:
    frame = _frame((6,))
    with pytest.raises(VerdictScopeUndeclaredError):
        _stamp(frame, scope=None, ledger_has_verdict_rows=True)

    before_any_verdict = _stamp(frame, scope=None, ledger_has_verdict_rows=False)
    assert set(before_any_verdict["verdict_scope"]) == {VERDICT_SCOPE_PRE_VERDICT}


# ---------------------------------------------------------------------------
# The single-site rule, over the production tree (AST)
# ---------------------------------------------------------------------------

# Every production package a stamp could be written from.
_PRODUCTION_ROOTS: tuple[str, ...] = (
    "api",
    "backtest",
    "data",
    "features",
    "forward_ledger",
    "models",
    "pipeline",
    "scripts",
    "utils",
)
_STAMPING_SITE = "forward_ledger/stamps.py"
# The migration (Plan 34-12) writes the pre-ledger rows, whose stamps are NULL forever.
_NULL_ONLY_SITE = "forward_ledger/migration.py"


def _is_null_constant(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _subscript_columns(target: ast.expr) -> list[str]:
    """The stamp columns a subscript assignment target names (``x["c"]``, ``x.loc[:, "c"]``)."""
    if not isinstance(target, ast.Subscript):
        return []
    keys = target.slice.elts if isinstance(target.slice, ast.Tuple) else [target.slice]
    return [
        key.value
        for key in keys
        if isinstance(key, ast.Constant) and key.value in PHASE34_STAMP_COLUMNS
    ]


def _stamp_writes(source: str) -> list[tuple[int, str, bool]]:
    """``(line, column, writes NULL)`` for every assignment of a Phase-34 stamp column."""
    writes: list[tuple[int, str, bool]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                writes += [
                    (node.lineno, column, _is_null_constant(node.value))
                    for column in _subscript_columns(target)
                ]
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            writes += [
                (node.lineno, column, _is_null_constant(node.value))
                for column in _subscript_columns(node.target)
            ]
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "assign"
        ):
            writes += [
                (node.lineno, keyword.arg, _is_null_constant(keyword.value))
                for keyword in node.keywords
                if keyword.arg in PHASE34_STAMP_COLUMNS
            ]
    return writes


def _offences(relative: str, source: str) -> list[str]:
    offences: list[str] = []
    for line, column, writes_null in _stamp_writes(source):
        if relative == _STAMPING_SITE:
            continue
        if relative == _NULL_ONLY_SITE and writes_null:
            continue
        offences.append(f"{relative}:{line} assigns {column!r}")
    return offences


def _production_sources() -> dict[str, str]:
    sources: dict[str, str] = {}
    for root in _PRODUCTION_ROOTS:
        directory = REPO_ROOT / root
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            sources[relative] = path.read_text(encoding="utf-8")
    return sources


def test_single_stamping_site() -> None:
    assert len(PHASE34_STAMP_COLUMNS) == 11
    sources = _production_sources()
    assert _STAMPING_SITE in sources, "the scan never reached the stamping site"

    # Non-vacuity: the scanner finds every stamp column assigned at the one site.
    assigned_at_site = {
        column for _, column, _ in _stamp_writes(sources[_STAMPING_SITE])
    }
    assert assigned_at_site == set(PHASE34_STAMP_COLUMNS)

    offences = [
        offence
        for relative, source in sorted(sources.items())
        for offence in _offences(relative, source)
    ]
    assert offences == [], (
        "Phase-34 stamps assigned outside forward_ledger/stamps.py:\n"
        + ("\n".join(offences))
    )


def test_single_stamping_site_controls() -> None:
    planted = textwrap.dedent(
        """
        def sneak(frame, other):
            frame["blend_id"] = other
            frame.loc[:, "recipe_id"] = "recipe-x"
            return frame.assign(arm="live")
        """
    )
    assert len(_offences("backtest/weekly_bet_list.py", planted)) == 3

    migration_null = 'def migrate(frame):\n    frame["model_artifact_id"] = None\n'
    assert _offences(_NULL_ONLY_SITE, migration_null) == []

    migration_value = 'def migrate(frame):\n    frame["model_artifact_id"] = "wp_x"\n'
    assert len(_offences(_NULL_ONLY_SITE, migration_value)) == 1
