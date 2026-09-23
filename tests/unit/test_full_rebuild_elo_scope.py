"""The full rebuild carries only the games an Elo snapshot can date (Plan 33.2-20).

THE PROBLEM THIS RESOLVES, AND THE ONE IT MUST NOT CREATE
---------------------------------------------------------
Under the owned provenance contract a game in the Elo SOURCE frame with no
``elo_game_snapshots`` row gets no provenance row, and the gate's two-way coverage refuses
it by name. That is correct and is not touched here.

But ``--all-seasons`` loads EVERY silver game, and silver carries the whole unplayed 2026
schedule. MEASURED 2026-09-22 on production silver: 272 games in 2026, snapshots for weeks
1 and 2 only (32 games, 15 of them provisional), and 240 games beyond that with none --
because a pre-game Elo for a game five months out does not exist. A full history rebuild
refuses on all 240, which is the gate working and a build that cannot run.

TWO ROUTES WERE AVAILABLE AND THE SECOND WAS REJECTED ON ITS MERITS. Either scope the build
to games that CAN carry a snapshot, or have the provisional-snapshot path cover them.
Generating a "provisional" pre-game rating for a week-18 game before week 3 is played is a
fabricated number wearing a real column's name -- the exact defect this phase removes. So
the build is SCOPED.

THE DANGER OF A SCOPE, AND THE BOUND THAT CLOSES IT
---------------------------------------------------
A rule that drops any game without a snapshot would ALSO silently drop a 2006 game whose
Elo row went missing -- turning a coverage defect into a quieter, smaller gold table. So the
drop is bounded: a game is removed only when it has NO snapshot AND is UNPLAYED. A PLAYED
game with no snapshot RAISES, naming the games. The two halves are asserted separately
below, and the refusal arm is asserted to be LIVE on the real corpus rather than
theoretical.

FULL REBUILD ONLY. A ``--season`` / ``--week`` build is untouched, exactly as
``scope_games_through_season`` is: the live daily path builds a week whose provisional
snapshots were written moments earlier, and a missing one there must still refuse rather
than quietly build nothing.

Run this module:  uv run pytest tests/unit/test_full_rebuild_elo_scope.py -q

Sandboxed: every check below is in memory or a READ of production silver.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from features.provenance import ProvenanceCoverageError
from scripts.build_features import scope_full_rebuild_to_elo_coverage

REPO_ROOT = Path(__file__).resolve().parents[2]


def _games(rows: list[tuple[str, int, object]]) -> pd.DataFrame:
    """``(game_id, season, home_score)`` -- a null score means the game is unplayed."""
    return pd.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "season": [r[1] for r in rows],
            "home_score": [r[2] for r in rows],
            "away_score": [r[2] for r in rows],
        }
    )


def _snapshots(game_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"game_id": game_ids})


class TestTheUnplayedFutureIsScopedOut:
    def test_an_unplayed_game_with_no_snapshot_is_dropped(self) -> None:
        games = _games([("2025_W01_A@B", 2025, 24), ("2026_W12_C@D", 2026, None)])
        scoped = scope_full_rebuild_to_elo_coverage(games, _snapshots(["2025_W01_A@B"]))
        assert list(scoped["game_id"]) == ["2025_W01_A@B"]

    def test_an_unplayed_game_WITH_a_snapshot_is_kept(self) -> None:
        """The provisional week is the point of the whole exercise: it stays in gold."""
        games = _games([("2026_W02_A@B", 2026, None)])
        scoped = scope_full_rebuild_to_elo_coverage(games, _snapshots(["2026_W02_A@B"]))
        assert list(scoped["game_id"]) == ["2026_W02_A@B"]

    def test_a_frame_with_nothing_to_drop_is_returned_unchanged(self) -> None:
        """The same OBJECT, so a history build is byte-for-byte the build it was."""
        games = _games([("2025_W01_A@B", 2025, 24)])
        assert (
            scope_full_rebuild_to_elo_coverage(games, _snapshots(["2025_W01_A@B"]))
            is games
        )

    def test_an_empty_frame_is_returned_unchanged(self) -> None:
        games = _games([])
        assert scope_full_rebuild_to_elo_coverage(games, _snapshots([])) is games


class TestAPlayedGameWithNoSnapshotStillRefuses:
    """The bound that stops a scope becoming a silent history-dropper."""

    def test_it_raises_naming_the_games(self) -> None:
        games = _games([("2006_W03_A@B", 2006, 17), ("2026_W12_C@D", 2026, None)])
        with pytest.raises(ProvenanceCoverageError) as exc:
            scope_full_rebuild_to_elo_coverage(games, _snapshots([]))
        assert exc.value.details["game_ids"] == ["2006_W03_A@B"]
        assert exc.value.details["violation_type"] == "played_game_without_a_snapshot"

    def test_a_game_played_on_only_one_side_of_the_score_still_counts_as_played(
        self,
    ) -> None:
        games = pd.DataFrame(
            {
                "game_id": ["2006_W03_A@B"],
                "season": [2006],
                "home_score": [17.0],
                "away_score": [None],
            }
        )
        with pytest.raises(ProvenanceCoverageError):
            scope_full_rebuild_to_elo_coverage(games, _snapshots([]))

    def test_the_refusal_names_the_elo_source(self) -> None:
        games = _games([("2006_W03_A@B", 2006, 17)])
        with pytest.raises(ProvenanceCoverageError) as exc:
            scope_full_rebuild_to_elo_coverage(games, _snapshots([]))
        assert exc.value.details["source"] == "elo"
        assert "not relaxed" in str(exc.value)


class TestItAppliesToTheFullRebuildOnly:
    """A scoped build must behave exactly as it did: a missing snapshot there refuses."""

    def test_the_call_site_is_guarded_by_the_full_rebuild_condition(self) -> None:
        """Structural: the one call sits inside `target_season is None and week is None`.

        Asserted over the parsed tree rather than by driving a scoped build, because the
        property is about WHEN the function runs, and a behavioural test could pass with
        the guard removed on any frame that happens to have full coverage.
        """
        tree = ast.parse(
            (REPO_ROOT / "scripts" / "build_features.py").read_text(encoding="utf-8")
        )
        call_lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "scope_full_rebuild_to_elo_coverage"
        ]
        assert len(call_lines) == 1, (
            f"the scope is called from {len(call_lines)} site(s) {call_lines}; it is a "
            "FULL-rebuild bound and one guarded call site is what makes that checkable."
        )
        guarded = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and node.lineno <= call_lines[0] <= (node.end_lineno or node.lineno)
            and "target_season is None" in ast.unparse(node.test)
            and "target_week is None" in ast.unparse(node.test)
        ]
        assert guarded, (
            "the call to scope_full_rebuild_to_elo_coverage is not inside a "
            "`target_season is None and target_week is None` guard, so a SCOPED build "
            "would silently drop its own games instead of refusing on a missing snapshot."
        )


class TestTheRefusalArmIsLiveOnTheRealCorpus:
    """Non-vacuity: the two arms reach real, different populations of production silver."""

    @staticmethod
    def _silver(name: str) -> pd.DataFrame:
        path = REPO_ROOT / "data" / "silver" / f"{name}.parquet"
        if not path.exists():
            pytest.skip(f"silver {name} is not built on this checkout")
        return pd.read_parquet(path)

    def test_every_played_game_in_production_silver_has_a_snapshot(self) -> None:
        """So the refusal arm is a live guard, not a theoretical one."""
        games = self._silver("games")
        snapshots = self._silver("elo_game_snapshots")
        played = games[games["home_score"].notna() | games["away_score"].notna()]
        assert len(played) > 6000
        missing = sorted(
            set(played["game_id"].astype(str)) - set(snapshots["game_id"].astype(str))
        )
        assert missing == [], (
            f"{len(missing)} PLAYED production game(s) have no Elo snapshot: "
            f"{missing[:10]}. The full rebuild refuses on these rather than dropping "
            "them, which is the point of the bound."
        )

    def test_the_scope_drops_only_unplayed_games_from_production_silver(self) -> None:
        games = self._silver("games")
        snapshots = self._silver("elo_game_snapshots")
        scoped = scope_full_rebuild_to_elo_coverage(games, snapshots)
        dropped = games[~games["game_id"].isin(scoped["game_id"])]
        assert len(dropped) > 0, (
            "nothing was dropped, so this test is asserting about a corpus the scope "
            "does not reach. Re-measure: the 2026 schedule beyond the provisional week "
            "is what it exists for."
        )
        assert dropped["home_score"].isna().all()
        assert dropped["away_score"].isna().all()
        assert set(dropped["season"].unique()) == {2026}, (
            f"the scope dropped games from season(s) "
            f"{sorted(dropped['season'].unique())}. Only the in-progress season's future "
            "should be beyond Elo coverage; anything else is a finding."
        )
