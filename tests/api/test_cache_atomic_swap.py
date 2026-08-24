"""SPEC R6 all-or-nothing proof for ``api.cache.populate_cache`` (Phase 30, Plan 30-12).

WHAT SPEC R6 ASKS FOR, AND WHAT THE CODE ALREADY DOES
-----------------------------------------------------
SPEC R6 requires that an interrupted cache repopulation never leaves the cache half-populated
against new artifacts while the web pages read old ones. ``populate_cache`` ALREADY satisfies this:
it connects to a temporary database file beside the destination, builds the ENTIRE cache inside it
(schema, feature importances, backtest predictions, metrics, season metrics, simulation results,
betting bets, predictions, game context, pre-rendered charts, cache metadata), closes the
connection, and only then renames the temporary file into place.

No wrapper is added here. Re-implementing a capability the module already has is precisely this
repo's characteristic expensive failure, so this module's job is to PIN the property, not to build
it.

THE GRANULARITY IS WHOLE-CACHE, WHICH IS STRONGER THAN SPEC R6 ASKS FOR
-----------------------------------------------------------------------
SPEC R6's concern is expressed per target. The implemented swap is whole-database: every target's
rows land in the same rename. A whole-cache atomic swap satisfies the per-target requirement for
all three targets simultaneously, so there is no target-ordering window at all.

THE ONE HONEST LIMITATION, RECORDED RATHER THAN PAPERED OVER
-------------------------------------------------------------
The final sequence unlinks the destination immediately before renaming the temporary file over it
(``api/cache.py``: ``if db_path.exists(): db_path.unlink()`` then ``tmp_path.rename(db_path)``). A
crash inside that window leaves NO cache rather than a mixed one. Pages would then fail to open the
cache instead of serving a blend of old and new predictions. That is still not the failure SPEC R6
names -- there is no half-populated state and no mixed read -- but it IS a real availability gap and
it belongs in the record. It is NOT fixed by this plan.

WHY THE PROPERTY IS PINNED TWICE
---------------------------------
Each guard covers the other's blind spot:

* ``test_populate_cache_source_builds_into_a_temp_file_and_renames`` reads the module's own AST. It
  is the guard that fires on the SPECIFIC regression this phase cares about -- a future refactor
  that starts writing the live cache in place. It is brittle in the direction that matters least: a
  behaviour-preserving refactor with different spelling could turn it red.
* ``test_populate_cache_swaps_the_destination_in_one_step`` asserts the OBSERVABLE consequence from
  the outside, so a behaviour-preserving refactor still passes while an in-place-write refactor
  still fails.

Between them, a behaviour-preserving refactor passes and an in-place-write refactor fails, which is
the correct pair of outcomes.

The behavioural test observes the in-flight state through a monkeypatched hook on one of the ordered
load steps -- NO threading and NO sleep-based timing, which would make it flaky for no gain.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import api.cache as cache_module
from api.cache import populate_cache

# The destination suffix chain the implementation produces beside the live cache.
_TMP_SUFFIX = ".tmp.duckdb"


# ---------------------------------------------------------------------------
# Guard 1: source-level -- populate_cache connects to a temp path, not the destination
# ---------------------------------------------------------------------------


def _populate_cache_ast() -> ast.FunctionDef:
    """Return the parsed ``populate_cache`` function definition.

    Parsed from ``inspect.getsource`` rather than from a hard-coded path so the guard follows the
    function if the module is ever split.
    """
    source = inspect.getsource(populate_cache)
    tree = ast.parse(inspect.cleandoc(source))
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef), (
        "expected populate_cache's source to parse to a single function definition"
    )
    return func


def _names_in(node: ast.AST) -> set[str]:
    """Return every bare identifier referenced anywhere inside *node*."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def test_populate_cache_source_builds_into_a_temp_file_and_renames() -> None:
    """SPEC R6 source guard: the cache is built into a temp file and renamed into place.

    Asserts, structurally rather than by substring match:

      1. ``duckdb.connect`` is called with a handle that is NOT the ``db_path`` destination.
      2. That handle was derived from ``db_path`` (so the temp file is a SIBLING of the
         destination, which is what makes the final rename a same-filesystem atomic operation
         rather than a cross-device copy).
      3. The destination is produced by calling ``.rename(db_path)`` on that same handle.

    A future refactor that switches to connecting straight to the live cache turns this red, which
    is the specific regression this phase cares about. See the module docstring for why a second,
    behavioural guard sits beside it.
    """
    func = _populate_cache_ast()
    dest_param = func.args.args[0].arg
    assert dest_param == "db_path", (
        f"populate_cache's destination parameter is named {dest_param!r}; this guard assumes "
        "'db_path' and must be updated deliberately if the signature changes"
    )

    # Local names assigned from an expression that mentions the destination parameter.
    derived_from_dest: dict[str, str] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and dest_param in _names_in(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    derived_from_dest[target.id] = ast.unparse(node.value)

    # 1 + 2: the duckdb.connect handle is a destination-derived temp path, not the destination.
    connect_calls = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
    ]
    assert len(connect_calls) == 1, (
        f"expected exactly one duckdb.connect call in populate_cache, found {len(connect_calls)}"
    )
    connect_arg_names = _names_in(connect_calls[0])
    assert dest_param not in connect_arg_names, (
        "populate_cache connects directly to the live cache destination -- the cache is no longer "
        "built into a temporary file, so an interrupted run can leave a half-populated cache "
        "(SPEC R6 all-or-nothing violated)"
    )
    temp_handles = connect_arg_names & set(derived_from_dest)
    assert temp_handles, (
        "populate_cache's duckdb.connect handle is not derived from db_path; the build target must "
        "be a SIBLING of the destination so the final rename is an atomic same-filesystem move. "
        f"connect referenced {sorted(connect_arg_names)}; destination-derived locals were "
        f"{sorted(derived_from_dest)}"
    )

    # 3: the destination is produced by renaming that same handle.
    renames = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "rename"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in temp_handles
        and dest_param in {name for arg in node.args for name in _names_in(arg)}
    ]
    assert renames, (
        "populate_cache never renames its temporary build file onto db_path -- the destination is "
        "not produced by an atomic rename (SPEC R6 all-or-nothing violated)"
    )


# ---------------------------------------------------------------------------
# Guard 2: behavioural -- observe the in-flight temp sibling and the post-run swap
# ---------------------------------------------------------------------------


def test_populate_cache_swaps_the_destination_in_one_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC R6 behavioural guard: the destination appears whole, in one step, at the end.

    Points ``populate_cache`` at a destination under ``tmp_path`` with EMPTY input directories (every
    loader is missing-input tolerant and warns-then-returns-0), so the run is hermetic and cheap.

    The in-flight observation is taken by wrapping ONE of the ordered load steps
    (``_load_metrics_summary``, which runs after the schema is created and well before the final
    rename). At that moment the assertion is that a ``.tmp.duckdb`` sibling of the destination
    EXISTS while the destination itself does NOT. After the run, the temp path is gone and the
    destination is present and openable.

    That is the observable consequence of the atomic swap, not its spelling: a refactor that
    preserves the behaviour with different internals still passes, and a refactor that starts
    writing the live cache in place fails here as well as in the source guard.

    Deliberately no threading and no sleep-based timing -- a hook on an ordered step observes the
    same window deterministically.
    """
    db_path = tmp_path / "web_cache.duckdb"
    empty_inputs = tmp_path / "empty"
    empty_inputs.mkdir()

    observed: dict[str, object] = {}
    real_loader = cache_module._load_metrics_summary

    def _observing_loader(*args: object, **kwargs: object) -> int:
        """Record the mid-run filesystem state, then defer to the real ordered load step."""
        observed["dest_exists"] = db_path.exists()
        observed["siblings"] = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
        return real_loader(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cache_module, "_load_metrics_summary", _observing_loader)

    populate_cache(
        db_path=db_path,
        artifacts_dir=empty_inputs,
        outputs_dir=empty_inputs,
        gold_dir=empty_inputs,
        silver_dir=empty_inputs,
    )

    # The hook must actually have fired, or the assertions below would be vacuous.
    assert observed, (
        "the observing hook never ran -- populate_cache no longer calls _load_metrics_summary, so "
        "this guard observed nothing and must be re-pointed at another ordered load step"
    )

    # IN FLIGHT: a temp sibling exists and the destination does not.
    assert observed["dest_exists"] is False, (
        "the live cache destination already existed mid-run -- populate_cache is building in place, "
        "so an interrupted run would leave a half-populated cache (SPEC R6 all-or-nothing violated)"
    )
    siblings = observed["siblings"]
    assert isinstance(siblings, list)
    tmp_siblings = [name for name in siblings if name.endswith(_TMP_SUFFIX)]
    assert tmp_siblings == [db_path.stem + _TMP_SUFFIX], (
        "expected exactly one in-flight temporary build file named "
        f"{db_path.stem + _TMP_SUFFIX!r} beside the destination; saw {siblings}"
    )

    # AFTER: the temp path is gone and the destination is present and openable.
    assert not (tmp_path / (db_path.stem + _TMP_SUFFIX)).exists(), (
        "the temporary build file survived the run -- it was copied rather than renamed, so the "
        "destination was not produced by a single atomic step"
    )
    assert db_path.exists(), "populate_cache did not produce the destination cache"

    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert "predictions" in tables, (
        "the swapped-in cache does not carry the CACHE_SCHEMA tables -- the destination is not a "
        f"complete cache; saw {sorted(tables)}"
    )
