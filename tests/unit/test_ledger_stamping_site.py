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
   An AST scan over the production tree holds the single-site rule. It sees subscript, keyword
   and dict-key writes (``{**row, "arm": ...}``, ``dict.fromkeys(...)``), and the migration's D-14
   labelling is an explicit, reviewed allowance of exact values, with controls proving the scan can
   fail on a planted dict write and on any migration value outside that allowance.

Every store here lives under ``tmp_path``; production ``artifacts/`` is never read.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

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
    "refit",
    "scripts",
    "utils",
)
_STAMPING_SITE = "forward_ledger/stamps.py"
# The migration (Plan 34-12, D-14) writes the pre-ledger rows. It is the one other site allowed to
# write a stamp column, and only as reviewed here: NULL for any stamp (its artifact, recipe, fill and
# reproduction stamps stay NULL forever), plus EXACTLY these non-NULL value expressions -- its
# ``arm`` / ``verdict_scope`` / ``regime_label`` labelling, written as dict keys in ``_migrated_row``
# and repeated in ``_check_payload``'s expected labels. Any other value it writes is an offence, so a
# change to the migration's labelling must change this allowance in the same review.
_MIGRATION_SITE = "forward_ledger/migration.py"
_MIGRATION_ALLOWED_VALUES: dict[str, frozenset[str]] = {
    "arm": frozenset({"ARM_LIVE"}),
    "verdict_scope": frozenset({"VERDICT_SCOPE_PRE_VERDICT"}),
    "regime_label": frozenset(
        {"_regime_label(int(stored['week']))", "_regime_label(values['week'])"}
    ),
}
# Two more dict-key sites the scan sees that write no stamp, each reviewed down to the exact value:
#   * the canonical DDL type maps name each stamp column's SQL type (``"arm": "VARCHAR"``);
#   * a correction entry (D-05) carries the corrected row's OWN ``arm`` -- a copy, never a new stamp.
_REVIEWED_VALUE_SITES: dict[str, dict[str, frozenset[str]]] = {
    _MIGRATION_SITE: _MIGRATION_ALLOWED_VALUES,
    "forward_ledger/canonical.py": dict.fromkeys(
        PHASE34_STAMP_COLUMNS, frozenset({"'VARCHAR'"})
    ),
    "forward_ledger/corrections.py": {"arm": frozenset({"row['arm']"})},
}
# The display tables re-type a row's own ``arm`` (NaN -> None) as the web cache stores it; the
# value is the row's own, never a new stamp (Plan 34-16's graded-outcomes path).
_NULL_NORMALIZING_SITES: dict[str, frozenset[str]] = {
    "api/cache.py": frozenset({"arm"}),
    "backtest/bet_tracker.py": frozenset({"arm"}),
}


def _stamp_bearing_names() -> dict[str, frozenset[str]]:
    """Column-tuple constants a ``dict.fromkeys`` may expand, by bare name -> the stamps they hold.

    Built from the live schema objects (never a hand list), so a renamed or widened column tuple
    cannot fall out of the scan. A ``dict.fromkeys`` over a name outside these modules (a local
    variable, an unrelated constant) is not resolved; the dict-literal and keyword shapes still are.
    """
    import api.cache
    import backtest.weekly_bet_list
    import forward_ledger.canonical
    import forward_ledger.schema
    import forward_ledger.store

    stamps = set(PHASE34_STAMP_COLUMNS)
    names: dict[str, frozenset[str]] = {}
    for module in (
        api.cache,
        backtest.weekly_bet_list,
        forward_ledger.canonical,
        forward_ledger.schema,
        forward_ledger.store,
    ):
        for name, value in vars(module).items():
            if not name.isupper() or not isinstance(
                value, tuple | list | set | frozenset
            ):
                continue
            held = stamps & {item for item in value if isinstance(item, str)}
            if held:
                names[name] = names.get(name, frozenset()) | frozenset(held)
    return names


_STAMP_BEARING_NAMES = _stamp_bearing_names()


class _StampWrite(NamedTuple):
    """One Phase-34 stamp write the scanner found."""

    line: int
    column: str
    writes_null: bool
    rewrites_itself: bool
    value: str  # the written expression as source text; "None" when no value is given


def _is_null_constant(node: ast.expr | None) -> bool:
    return node is None or (isinstance(node, ast.Constant) and node.value is None)


def _write(
    line: int, column: str, value: ast.expr | None, *, rewrites_itself: bool = False
) -> _StampWrite:
    return _StampWrite(
        line,
        column,
        _is_null_constant(value),
        rewrites_itself,
        "None" if value is None else ast.unparse(value),
    )


def _called(node: ast.Call) -> str | None:
    """The called name: ``f`` for ``f(...)``, ``m`` for ``x.m(...)``."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _named_stamps(node: ast.expr) -> frozenset[str]:
    """The stamps a bare or dotted column-tuple name holds (``X`` or ``m.X``)."""
    name = (
        node.id
        if isinstance(node, ast.Name)
        else node.attr
        if isinstance(node, ast.Attribute)
        else None
    )
    return _STAMP_BEARING_NAMES.get(name, frozenset()) if name else frozenset()


def _key_stamps(keys: ast.expr) -> frozenset[str]:
    """The stamp columns a ``dict.fromkeys`` keys argument names (literal, starred or named)."""
    if not isinstance(keys, ast.Tuple | ast.List | ast.Set):
        return _named_stamps(keys)
    found: set[str] = set()
    for element in keys.elts:
        if isinstance(element, ast.Starred):
            found |= _named_stamps(element.value)
        elif isinstance(element, ast.Constant):
            if element.value in PHASE34_STAMP_COLUMNS:
                found.add(element.value)
        else:
            found |= _named_stamps(element)
    return frozenset(found)


def _key_order_only(tree: ast.AST) -> set[int]:
    """``dict.fromkeys`` calls used only for their key ORDER (``tuple(...)`` / ``list(...)``).

    ``tuple(dict.fromkeys([...]))`` is the ordered-dedupe idiom for a column list; its values are
    discarded, so it writes no row value.
    """
    return {
        id(node.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _called(node) in {"tuple", "list"}
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Call)
        and _called(node.args[0]) == "fromkeys"
    }


def _call_writes(node: ast.Call, key_order_only: set[int]) -> list[_StampWrite]:
    """Stamp writes made by a call: keyword, ``fromkeys``, ``insert``, ``setdefault``, setitem."""
    called = _called(node)
    if called in {"assign", "update", "dict"}:
        return [
            _write(node.lineno, keyword.arg, keyword.value)
            for keyword in node.keywords
            if keyword.arg in PHASE34_STAMP_COLUMNS
        ]
    if called == "fromkeys" and node.args and id(node) not in key_order_only:
        value = node.args[1] if len(node.args) > 1 else None
        return [
            _write(node.lineno, column, value)
            for column in sorted(_key_stamps(node.args[0]))
        ]
    # (key position, value position): ``frame.insert(loc, "c", v)``, ``d.setdefault("c", v)``,
    # ``operator.setitem(d, "c", v)``, ``d.__setitem__("c", v)``.
    positions = {
        "insert": (1, 2),
        "setdefault": (0, 1),
        "setitem": (1, 2),
        "__setitem__": (0, 1),
    }
    if called not in positions:
        return []
    key_at, value_at = positions[called]
    if len(node.args) <= key_at:
        return []
    key = node.args[key_at]
    if not (isinstance(key, ast.Constant) and key.value in PHASE34_STAMP_COLUMNS):
        return []
    if called == "insert" and len(node.args) <= value_at:
        return []  # ``list.insert(i, x)`` names no column
    value = node.args[value_at] if len(node.args) > value_at else None
    return [_write(node.lineno, key.value, value)]


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


def _rewrites_itself(target: ast.expr, value: ast.expr | None, column: str) -> bool:
    """True when *value* is computed from the target's own column (``x["c"] = f(x["c"])``)."""
    if value is None or not isinstance(target, ast.Subscript):
        return False
    base = ast.unparse(target.value)
    return any(
        isinstance(node, ast.Subscript)
        and ast.unparse(node.value) == base
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == column
        for node in ast.walk(value)
    )


def _stamp_writes(source: str) -> list[_StampWrite]:
    """Every Phase-34 stamp write in *source*, in each shape a row value can be written.

    Subscript assignment (``x["c"] = v``, ``x.loc[:, "c"] = v``, augmented / annotated), dict-literal
    keys (``{"c": v}``, ``{**row, "c": v}``), keywords (``dict(c=v)``, ``d.update(c=v)``,
    ``frame.assign(c=v)``), ``dict.fromkeys(columns[, v])``, ``frame.insert(loc, "c", v)``,
    ``d.setdefault("c", v)`` and setitem. ``d.update({...})`` is seen through its dict literal.
    """
    tree = ast.parse(source)
    key_order_only = _key_order_only(tree)
    writes: list[_StampWrite] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                writes += [
                    _write(
                        node.lineno,
                        column,
                        node.value,
                        rewrites_itself=_rewrites_itself(target, node.value, column),
                    )
                    for column in _subscript_columns(target)
                ]
        elif isinstance(node, ast.AugAssign | ast.AnnAssign):
            writes += [
                # An annotation alone (``x["c"]: T``) writes nothing; ``None`` here is "no value".
                _write(node.lineno, column, node.value)
                for column in _subscript_columns(node.target)
                if node.value is not None
            ]
        elif isinstance(node, ast.Dict):
            writes += [
                _write(key.lineno, key.value, value)
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and key.value in PHASE34_STAMP_COLUMNS
            ]
        elif isinstance(node, ast.Call):
            writes += _call_writes(node, key_order_only)
    return writes


def _reviewed(relative: str, write: _StampWrite) -> bool:
    """A write the review allows at *relative*: the migration's NULLs, or an exact allowed value."""
    if relative == _MIGRATION_SITE and write.writes_null:
        return True
    allowed = _REVIEWED_VALUE_SITES.get(relative, {})
    return write.value in allowed.get(write.column, frozenset())


def _offences(relative: str, source: str) -> list[str]:
    offences: list[str] = []
    for write in _stamp_writes(source):
        if relative == _STAMPING_SITE:
            continue
        if _reviewed(relative, write):
            continue
        if write.rewrites_itself and write.column in _NULL_NORMALIZING_SITES.get(
            relative, frozenset()
        ):
            continue
        offences.append(
            f"{relative}:{write.line} assigns {write.column!r} = {write.value}"
        )
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
        write.column for write in _stamp_writes(sources[_STAMPING_SITE])
    }
    assert assigned_at_site == set(PHASE34_STAMP_COLUMNS)

    # Non-vacuity at the migration (CR-03): the scanner SEES its D-14 dict-key writes -- the three
    # labels at their allowed values and every other stamp NULL through ``dict.fromkeys`` -- so the
    # allowance below is exercised, not silently unused.
    migration_writes = _stamp_writes(sources[_MIGRATION_SITE])
    labels = {
        (write.column, write.value)
        for write in migration_writes
        if not write.writes_null
    }
    assert labels == {
        (column, value)
        for column, values in _MIGRATION_ALLOWED_VALUES.items()
        for value in values
    }
    nulled = {write.column for write in migration_writes if write.writes_null}
    assert nulled == set(PHASE34_STAMP_COLUMNS) - set(_MIGRATION_ALLOWED_VALUES)

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

    # CR-03: the dict shapes a ledger row actually travels in. Each plant is one non-NULL stamp
    # written outside stamps.py, and each is an offence.
    dict_plants = (
        'def f(r):\n    return {**r, "arm": "shadow"}\n',
        'def f(r):\n    return {"recipe_id": "recipe-x", "game_id": r}\n',
        'def f():\n    return dict(blend_id="blend_x")\n',
        'def f(r):\n    r.update({"model_artifact_id": "wp_x"})\n',
        'def f(r):\n    r.update(verdict_scope="verdict")\n',
        'def f(frame):\n    frame.insert(0, "fill_convention_id", "fill-v9")\n',
        'def f(r):\n    r.setdefault("regime_label", "bootstrap_regime")\n',
        'def f(r):\n    operator.setitem(r, "odds_snapshot_digest", "x")\n',
        'def f():\n    return dict.fromkeys(("decision_snapshot_digest",), "d")\n',
        'def f():\n    return dict.fromkeys(LEDGER_STAMP_COLUMNS, "x")\n',
    )
    for plant in dict_plants:
        assert len(_offences("backtest/x.py", plant)) >= 1, plant
    # A NULL ``dict.fromkeys`` over the stamp tuple outside the migration is still an offence.
    assert _offences(
        "backtest/x.py", "def f():\n    return dict.fromkeys(LEDGER_STAMP_COLUMNS)\n"
    )
    # ... while the key-order idiom writes no value, and a non-stamp key is not a stamp.
    key_order = "COLS = tuple(dict.fromkeys([*LEDGER_STAMP_COLUMNS, 'game_id']))\n"
    assert _offences("backtest/x.py", key_order) == []
    assert _offences("backtest/x.py", 'def f():\n    return {"game_id": 1}\n') == []

    # The migration's allowance is exact: its reviewed labels and NULLs pass, anything else fails.
    migration_ok = textwrap.dedent(
        """
        def m(stored):
            return {
                "arm": ARM_LIVE,
                **dict.fromkeys(LEDGER_STAMP_COLUMNS),
                "verdict_scope": VERDICT_SCOPE_PRE_VERDICT,
                "regime_label": _regime_label(int(stored['week'])),
            }
        """
    )
    assert _offences(_MIGRATION_SITE, migration_ok) == []
    migration_null = 'def migrate(frame):\n    frame["model_artifact_id"] = None\n'
    assert _offences(_MIGRATION_SITE, migration_null) == []

    migration_value = 'def migrate(frame):\n    frame["model_artifact_id"] = "wp_x"\n'
    assert len(_offences(_MIGRATION_SITE, migration_value)) == 1
    migration_dict_values = textwrap.dedent(
        """
        def m(row):
            return {
                **row,
                "arm": "shadow",
                "verdict_scope": "verdict",
                "regime_label": "bootstrap_regime",
                "recipe_id": "recipe-x",
            }
        """
    )
    assert len(_offences(_MIGRATION_SITE, migration_dict_values)) == 4
    migration_fromkeys = (
        "def m():\n    return dict.fromkeys(LEDGER_STAMP_COLUMNS, 'x')\n"
    )
    assert len(_offences(_MIGRATION_SITE, migration_fromkeys)) == 8

    # The other reviewed sites are exact too: the row's own arm / a type name pass, a value fails,
    # and neither site may write NULL stamps (that allowance is the migration's alone).
    correction = 'def c(row):\n    return {"arm": row["arm"]}\n'
    assert _offences("forward_ledger/corrections.py", correction) == []
    assert _offences(
        "forward_ledger/corrections.py", 'def c():\n    return {"arm": "shadow"}\n'
    )
    assert _offences(
        "forward_ledger/corrections.py", 'def c():\n    return {"arm": None}\n'
    )
    assert (
        _offences("forward_ledger/canonical.py", 'T = {"recipe_id": "VARCHAR"}\n') == []
    )
    assert _offences("forward_ledger/canonical.py", 'T = {"recipe_id": "recipe-x"}\n')

    normalized = 'def tidy(rows):\n    rows["arm"] = _null_as_none(rows["arm"])\n'
    assert _offences("api/cache.py", normalized) == []

    display_value = 'def tidy(rows, other):\n    rows["arm"] = "live"\n    rows["arm"] = other["arm"]\n'
    assert len(_offences("api/cache.py", display_value)) == 2
