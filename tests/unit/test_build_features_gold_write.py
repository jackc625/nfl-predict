"""G-02: the gold write is a single-file REPLACE, and the line-movement seams are gone.

Two contracts live here, both of them about ``save_feature_matrices`` and the
build path that feeds it.

**1. The gold write replaces the table; it does not append to it.**

The original guard (Phase 20, CR-01, D-10, commit 0269a15) removed
``partition_cols=["season"]`` from the gold write, because ``pq.write_to_dataset``
routes partition directories into the SHARED ``data/gold/`` root where all three
matrices (``features_wp``/``ats``/``ou``) collide in the same ``season=YYYY/``
directory and every current-week run appends a NEW hash-named parquet instead of
overwriting. That fix stands and its assertion is unchanged below.

What the original fix got only half right is the mode it fell back to.  Removing
``partition_cols`` left ``save_dataframe``'s DEFAULT ``append_mode=True`` path,
which is correct for a rebuild that only ever ADDS columns -- and this project had
never done anything else.  Phase 30 rung 3 is the first rebuild in the repository's
history that REMOVES columns, and the append path cannot express that: it loads the
existing table, drops the rows whose ``game_id`` appears in the new frame (on a full
rebuild, all of them), and then ``pd.concat``s -- and a concat UNIONS columns even
when one operand has zero rows.  A 194-column in-memory frame would therefore have
been written back as a 209-column file with the fifteen dropped columns present and
entirely null, and ``check_gold_integrity`` would have failed on all-null columns
for a reason that looks nothing like the actual cause.

``replace_mode`` says the passed frame IS the table.  It also makes a repeated
full rebuild idempotent, which is what SPEC R1's byte-identical re-run acceptance
needs.  ``tests/integration/test_storage_replace_mode.py`` is the committed proof
of the storage-layer semantics; the guard here is the call-site half.

It is NOT passed unconditionally, and this guard was originally written as though
it should be.  ``save_feature_matrices`` is reachable with a season filter --
``build_features.py --season 2025`` produces a 285-row, 2025-only matrix -- and
replace mode writes that slice AS the gold table in BOTH stores, destroying
2002-2024 (CR-01).  The write mode therefore follows the BUILD'S SCOPE: a full
rebuild replaces, which is what lets it narrow the schema; a scoped build merges
latest-wins on ``game_id``.  The assertion below pins BOTH halves -- replace mode
must be present (or the removal never lands) and must be the scope flag rather
than a literal (or a per-season build truncates gold).  The behavioural proof of
both is ``tests/integration/test_gold_write_scope.py``.

**2. No line-movement seam remains in this module (SPEC R3 / D29-07-01).**

``combine_features`` has NO generic loop over ``feature_sources``, so the family
needed TWO seams to reach gold -- the ``feature_sources`` registration and an
EXPLICIT merge block -- and leaving either one in place would resurrect it on the
next rebuild.  Both are gone.  The class below is the 28-06 lesson run in reverse:
it proves the module can no longer construct, register, merge or neutral-fill the
family, rather than asserting the behaviour of a guard that no longer exists.

Every guard here asserts on PARSED structure (``ast``), never on source text.  A
textual guard for a removed expression necessarily matches the prose that explains
the removal, which is the self-referential hazard Plans 30-04 and 30-06 both hit.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pandas as pd
import pytest


def _module_source() -> str:
    import scripts.build_features as build_features_mod

    return inspect.getsource(build_features_mod)


def _parsed_method(method) -> ast.AST:
    """Parse a single method's source into an AST (dedented so it parses alone)."""
    return ast.parse(textwrap.dedent(inspect.getsource(method)))


def test_gold_write_call_has_no_partition_cols():
    """The save_dataframe(..., layer="gold") call must NOT pass partition_cols.

    Regression: if partition_cols= reappears in the gold write, pq.write_to_dataset
    writes season=YYYY/ dirs into the shared data/gold/ root, all three feature
    matrices collide there, and every rebuild appends new hash-named files instead of
    overwriting.  (Phase 20, commit 0269a15, CR-01, D-10)
    """
    source = _module_source()

    # Locate the character offset of every gold-layer save_dataframe call, then
    # extract a window around it to check for partition_cols.
    gold_write_marker = 'layer="gold"'

    occurrences = []
    search_start = 0
    while True:
        idx = source.find(gold_write_marker, search_start)
        if idx == -1:
            break
        occurrences.append(idx)
        search_start = idx + 1

    assert len(occurrences) >= 1, (
        'scripts/build_features.py has no save_dataframe call with layer="gold" -- '
        "the gold write was removed or renamed, which itself is a regression"
    )

    regressions_found = []
    for idx in occurrences:
        window_start = max(0, idx - 600)
        window = source[window_start : idx + 200]

        if "save_dataframe(" not in window:
            continue

        last_call_start = window.rfind("save_dataframe(")
        call_fragment = window[last_call_start:]

        if "partition_cols" in call_fragment:
            regressions_found.append(textwrap.shorten(call_fragment, width=200))

    assert regressions_found == [], (
        "REGRESSION DETECTED: the gold-layer save_dataframe call in "
        "scripts/build_features.py now passes partition_cols. "
        "This re-introduces the shared-root partitioned-append antipattern "
        "(season=YYYY/ dirs in gold/, multiplying cardinality on every rebuild). "
        f"Offending call fragment(s): {regressions_found}"
    )


class TestTheGoldWriteReplacesTheTable:
    """``replace_mode`` on a FULL rebuild is what makes a column REMOVAL land on disk."""

    @staticmethod
    def _gold_save_dataframe_call() -> ast.Call:
        from scripts.build_features import FeatureMatrixBuilder

        tree = _parsed_method(FeatureMatrixBuilder.save_feature_matrices)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "save_dataframe"
        ]
        assert len(calls) == 1, (
            "save_feature_matrices no longer contains exactly one save_dataframe "
            f"call ({len(calls)} found) -- the gold write was refactored and this "
            "guard can no longer say what it claims"
        )
        return calls[0]

    def test_the_gold_write_passes_replace_mode(self) -> None:
        """On a FULL rebuild the passed frame must BE the table, so narrower writes narrower.

        Without replace mode, ``save_dataframe``'s default append path concats the
        existing 209-column table onto the new 194-column frame and the concat unions
        the columns back in as all-null.  Every previous rebuild in this project only
        ADDED columns, where that union is a harmless no-op -- which is why nothing
        has ever caught it.
        """
        call = self._gold_save_dataframe_call()
        keywords = {kw.arg: kw.value for kw in call.keywords}

        assert "replace_mode" in keywords, (
            "the gold save_dataframe call does not pass replace_mode. Its default "
            "append path UNIONS columns through a concat, so a rebuild that removes "
            "a column writes the column back as all-NaN and the removal never lands."
        )

    def test_replace_mode_is_the_scope_flag_and_NOT_an_unconditional_literal(
        self,
    ) -> None:
        """CR-01: a literal True here destroys full-history gold on a per-season build.

        ``save_feature_matrices`` is reachable with a season filter, and ``--season``
        is a command ``PIPELINE.md`` and ``RUNBOOK.md`` both published.  Under an
        unconditional ``replace_mode=True`` a 285-row 2025-only matrix became the
        whole gold table in DuckDB AND parquet, destroying 2002-2024.  Before Plan
        30-07 the default append path merged latest-wins on ``game_id`` and the
        identical command was safe, so this was a regression the phase introduced.

        The mode must be a NAME bound from the build's scope -- not a constant in
        either direction.  A literal ``False`` would be just as wrong: it would make
        the rung-3 column drop unexpressible, which is the guard above.
        """
        call = self._gold_save_dataframe_call()
        value = {kw.arg: kw.value for kw in call.keywords}["replace_mode"]

        assert not isinstance(value, ast.Constant), (
            f"replace_mode is passed as the literal {ast.dump(value)}. It must be "
            "derived from the build's scope: a scoped (--season / --week) build "
            "carries only its slice, and writing that AS the table destroys every "
            "season it did not carry."
        )
        assert isinstance(value, ast.Name), (
            f"replace_mode is passed as {ast.dump(value)}; expected a simple name "
            "bound from the build scope so the two modes are legible at the call site"
        )

        # The scope flag must actually be computed from BOTH scope parameters --
        # --week alone is a scoped build too.
        from scripts.build_features import FeatureMatrixBuilder

        tree = _parsed_method(FeatureMatrixBuilder.save_feature_matrices)
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == value.id for t in node.targets)
        ]
        assert assignments, (
            f"'{value.id}' is passed as replace_mode but is never assigned in "
            "save_feature_matrices, so what decides the write mode is not visible here"
        )
        scope_names = {
            node.id
            for node in ast.walk(assignments[0].value)
            if isinstance(node, ast.Name)
        }
        assert {"target_season", "target_week"} <= scope_names, (
            f"'{value.id}' is computed from {sorted(scope_names)}; it must consider "
            "BOTH target_season and target_week -- a --week-only build is scoped too"
        )

    def test_the_gold_write_does_not_re_enable_partitioning(self) -> None:
        """Structural companion to the textual partition guard above."""
        call = self._gold_save_dataframe_call()
        assert "partition_cols" not in {kw.arg for kw in call.keywords}


class TestADropThatRemovesNothingIsRefused:
    """SPEC R3's ``empty`` edge: a drop that silently does nothing is worse than none."""

    @staticmethod
    def _frame_with_the_family() -> pd.DataFrame:
        """A minimal frame carrying members of the line-movement family."""
        return pd.DataFrame(
            {
                "game_id": ["G1", "G2"],
                "season": [2023, 2023],
                "snapshot_total": [44.0, 45.0],
                "opening_total": [44.5, 45.5],
                "total_drift": [0.5, -0.5],
                "line_movement_coverage": [1.0, 0.0],
            }
        )

    def test_a_zero_match_drop_raises_rather_than_proceeding(self) -> None:
        """No member present means the predicate is wrong, not that the work is done.

        A drop expressed as a predicate can be misspelled, and a misspelled
        predicate removes nothing while every downstream width and presence check
        reads exactly as it would after a successful drop.  The refusal is what
        stops that producing gold that only LOOKS dropped.
        """
        from scripts.build_features import drop_feature_group

        frame = pd.DataFrame({"game_id": ["G1"], "snapshot_total": [44.0]})
        with pytest.raises(ValueError, match="only LOOKS dropped"):
            drop_feature_group(frame, "line_movement")

    def test_the_drop_removes_the_family_and_nothing_else(self) -> None:
        from scripts.build_features import drop_feature_group

        out = drop_feature_group(self._frame_with_the_family(), "line_movement")

        assert list(out.columns) == ["game_id", "season", "snapshot_total"], (
            "the drop must remove exactly the family; snapshot_total is the freeze "
            "anchor and a pre-existing baseline feature, not a Phase-29 column"
        )

    def test_the_column_set_comes_from_the_group_registry(self) -> None:
        """D30-02: one registry, never a second list of the fifteen names.

        Proved by MONKEYPATCHING the registry's predicate and observing the drop
        follow it. A module carrying its own copy of the names would be unmoved.
        """
        from backtest import signal_lift
        from scripts.build_features import drop_feature_group

        frame = self._frame_with_the_family()
        original = signal_lift._GROUP_PREDICATE["line_movement"]
        try:
            signal_lift._GROUP_PREDICATE["line_movement"] = lambda col: (
                col == "snapshot_total"
            )
            out = drop_feature_group(frame, "line_movement")
        finally:
            signal_lift._GROUP_PREDICATE["line_movement"] = original

        assert "snapshot_total" not in out.columns
        assert "opening_total" in out.columns, (
            "the drop ignored the patched registry predicate, so it is reading a "
            "second, private list of the family's names (D30-02)"
        )


class TestTheLineMovementSeamsAreGone:
    """The 28-06 lesson in reverse: neither seam can land the family any more.

    ``combine_features`` merges each source through an EXPLICIT per-source block
    and has no generic loop, so the family needed BOTH the ``feature_sources``
    registration AND a merge block.  Removing one and leaving the other would look
    fixed and resurrect the columns on the next rebuild.
    """

    def test_the_module_no_longer_imports_the_builder(self) -> None:
        """AST on the imports, so the guard cannot match the prose that explains it."""
        tree = ast.parse(_module_source())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom | ast.Import):
                imported.update(alias.name for alias in node.names)

        assert "LineMovementBuilder" not in imported, (
            "scripts/build_features.py still imports LineMovementBuilder -- the "
            "builder is the head of the registration seam (SPEC R3)"
        )

    def test_the_builder_is_not_constructed_as_an_attribute(self) -> None:
        from scripts.build_features import FeatureMatrixBuilder

        builder = FeatureMatrixBuilder()
        assert not hasattr(builder, "line_movement_builder"), (
            "FeatureMatrixBuilder still constructs a line-movement builder; the "
            "registration seam is only half removed"
        )

    def test_loading_the_sources_registers_no_line_movement_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The registration seam, exercised rather than inspected."""
        import scripts.build_features as build_features_mod

        builder = build_features_mod.FeatureMatrixBuilder()
        games = pd.DataFrame(
            {
                "game_id": ["2023_W01_A@B"],
                "season": [2023],
                "week": [1],
                "home_team": ["NYJ"],
                "away_team": ["MIA"],
                # Every build now derives each game's lock from its kickoff before any
                # source loads (Plan 33.2-13), so a kickoff-less fixture is refused.
                "kickoff_et": [pd.Timestamp("2023-09-10 17:00", tz="UTC")],
            }
        )
        monkeypatch.setattr(
            build_features_mod, "load_dataframe", lambda *a, **k: games.copy()
        )
        for calc_attr in (
            "elo_calc",
            "contextual_calc",
            "market_calc",
            "qb_tracker",
            "snap_builder",
            "injury_builder",
        ):
            monkeypatch.setattr(
                getattr(builder, calc_attr),
                "build_features",
                lambda *a, **k: pd.DataFrame(),
            )

        sources = builder.load_all_feature_sources()

        assert "line_movement" not in sources, (
            "load_all_feature_sources still registers a line_movement source. "
            "Registration alone does not reach gold, but it routes the frame "
            "through the LeakageGate and is half of the seam pair."
        )

    def test_combine_features_emits_none_of_the_family_even_when_handed_one(
        self,
    ) -> None:
        """The merge seam, run in reverse -- the strongest proof available.

        The source frame is asserted POPULATED first.  Without that pin this test
        would pass just as happily against a builder that produced nothing, which
        is the vacuous-pass shape this phase exists to eliminate.
        """
        from backtest.signal_lift import group_columns
        from scripts.build_features import FeatureMatrixBuilder

        games = pd.DataFrame(
            {
                "game_id": ["G1", "G2"],
                "season": [2023, 2023],
                "week": [1, 2],
                "home_team": ["NYJ", "BUF"],
                "away_team": ["MIA", "NE"],
            }
        )
        line_movement = pd.DataFrame(
            {
                "game_id": ["G1", "G2"],
                "opening_total": [44.0, 45.0],
                "total_drift": [0.5, -0.5],
                "line_movement_coverage": [1.0, 0.0],
            }
        )
        offered = group_columns(line_movement, "line_movement")
        assert len(offered) == 3, (
            "fixture sanity: the source frame must actually CARRY the family, or "
            f"'none arrived' would be trivially true. Offered: {offered}"
        )

        combined = FeatureMatrixBuilder().combine_features(
            {"games": games, "line_movement": line_movement}
        )

        arrived = group_columns(combined, "line_movement")
        assert arrived == [], (
            f"combine_features merged {arrived} from a registered line_movement "
            "source -- the explicit merge block is still present, and the family "
            "will return to gold on the next rebuild (SPEC R3)"
        )

    def test_no_line_movement_neutral_default_path_remains(self) -> None:
        """The WR-10 guard was the family's last consumer in this module.

        It filled the family from ``LineMovementBuilder._neutral_features`` instead
        of a median, because the median of ``line_movement_coverage`` is 1.0 and a
        median fill fabricated coverage.  With the family gone from gold the guard
        is unreachable, and an unreachable guard that still names a removed builder
        is a false statement about what the module does.  Nothing is lost: the SPEC
        R3 drop runs on the COMBINED matrix, before any imputation, so a
        reinstated family never reaches this method at all.
        """
        from scripts.build_features import FeatureMatrixBuilder

        tree = _parsed_method(FeatureMatrixBuilder.handle_missing_data_and_outliers)
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }

        assert "line_movement_builder" not in attributes
        assert "_neutral_features" not in attributes, (
            "handle_missing_data_and_outliers still calls a neutral-defaults "
            "provider; the WR-10 guard survived the family it guarded"
        )

    def test_the_two_games_filters_agree_on_season_only(self) -> None:
        """Source guard: the season filter must not require a week (WR-10).

        Retained after the drop.  The defect it pins is general -- every builder
        filters season and week independently, so a season-only build that left
        ``games_df`` unfiltered would leave most rows NaN after any left merge.
        The line-movement family was where it was FOUND, not where it applies.
        """
        import scripts.build_features as build_features_mod

        source = inspect.getsource(build_features_mod.FeatureMatrixBuilder)
        assert (
            "if target_season and target_week:\n                games_df" not in source
        ), (
            "the games filter still requires BOTH season and week, so a season-only "
            "build leaves games unfiltered while the builders filter independently"
        )


class TestTheDropIsWiredIntoTheBuild:
    """The removal is enforced on the combined matrix, before the gold write."""

    def test_generate_feature_matrices_enforces_the_removal(self) -> None:
        from scripts.build_features import FeatureMatrixBuilder

        tree = _parsed_method(FeatureMatrixBuilder.generate_feature_matrices)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "_enforce_groups_dropped" in called, (
            "generate_feature_matrices no longer enforces the SPEC R3 removal on "
            "the combined matrix, so a reinstated seam would reach gold unnoticed"
        )

    def test_a_reinstated_family_is_removed_before_the_gold_write(self) -> None:
        """Defence in depth, exercised: hand the enforcer a contaminated frame."""
        from scripts.build_features import FeatureMatrixBuilder

        frame = pd.DataFrame(
            {
                "game_id": ["G1", "G2"],
                "season": [2023, 2023],
                "snapshot_total": [44.0, 45.0],
                "opening_total": [44.5, np.nan],
                "line_movement_coverage": [1.0, 0.0],
                "precip_mm": [np.nan, np.nan],
                "raw_precip_mm": [np.nan, np.nan],
                "precip_prob": [0.2, 0.7],
            }
        )
        out = FeatureMatrixBuilder()._enforce_groups_dropped(
            frame, ("line_movement", "weather_unsupplied")
        )

        # The generalised body drops EVERY group it is handed -- line movement AND the
        # weather inputs no forecast supplies (Plan 33.2-12) -- and nothing else.
        assert "opening_total" not in out.columns
        assert "line_movement_coverage" not in out.columns
        assert "precip_mm" not in out.columns
        assert "raw_precip_mm" not in out.columns
        assert "snapshot_total" in out.columns
        assert "precip_prob" in out.columns

    def test_a_clean_frame_passes_through_untouched(self) -> None:
        """The intended path: both seams gone, so there is nothing to drop.

        The enforcer must NOT raise here.  ``drop_feature_group``'s empty-match
        refusal is a statement about a drop that was asked to remove something and
        removed nothing; a build whose seams are gone was never asking.
        """
        from scripts.build_features import FeatureMatrixBuilder

        frame = pd.DataFrame(
            {
                "game_id": ["G1"],
                "season": [2023],
                "snapshot_total": [44.0],
                "total_movement": [0.5],
            }
        )
        out = FeatureMatrixBuilder()._enforce_groups_dropped(
            frame, ("line_movement", "weather_unsupplied")
        )

        assert list(out.columns) == list(frame.columns)


def test_gold_write_call_targets_correct_table_names():
    """The gold write produces features_wp, features_ats, features_ou tables.

    Sanity guard: the table_name= values passed to the gold save_dataframe call
    all follow the features_{target} pattern, confirming the call site has not
    been renamed or pointed at an unexpected table.
    """
    source = _module_source()

    assert 'f"features_{target}"' in source or "features_{target}" in source, (
        "scripts/build_features.py no longer constructs 'features_{target}' as the "
        "gold table name -- the gold write call site has been refactored in an "
        "unexpected way; review for partition_cols regression manually"
    )

    for target in ("wp", "ats", "ou"):
        assert target in source, (
            f"Target '{target}' not found in build_features source -- the gold "
            f"write for features_{target} may have been removed"
        )
