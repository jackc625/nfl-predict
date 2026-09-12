"""A provisional Elo row can SERVE a live week and can never TRAIN a model.

WHY THIS MODULE PROVES THE GUARD AT FOUR PLACES AND NOT ONE (Codex HIGH)
-----------------------------------------------------------------------
The obvious place for a "no provisional rows in training" guard is
``features/elo_features.py``, because that is where Elo becomes a feature. Measured
against live source, a guard there protects NOT ONE training path. Every trainer reads
gold DIRECTLY:

* ``models/train_wp.py``, ``models/train_ats.py`` and ``models/train_ou.py`` each call
  ``load_dataframe("features_*", layer="gold")`` inside a ``try:``;
* ``models/train.py`` calls ``pd.read_parquet(features_path)`` and bypasses
  ``load_dataframe`` entirely.

None of the four passes through ``EloFeatureBuilder.build_features``. So the refusal is
wired at each boundary and PROVEN at each boundary -- a guard demonstrated at one entry
point says nothing whatsoever about the other three. ``TRAINER_GOLD_LOAD_SITES`` pins the
set, and the drift test below fails if a fifth trainer appears without the guard.

WHY THE EXCEPTION TYPE IS LOAD-BEARING (Codex HIGH)
---------------------------------------------------
``scripts/build_features.py`` wraps every optional source load -- the Elo build among
them -- in ``except _SOURCE_LOAD_ERRORS``, which converts a failure into an EMPTY FRAME
and logs a warning. A refusal caught there does not refuse anything: it produces a gold
build with no Elo columns and a green exit. ``ProvisionalSnapshotAsTrainingInputError``
therefore derives from ``RuntimeError``, which is deliberately absent from
``_SOURCE_LOAD_ERRORS`` and from all four trainers' handler tuples. That exclusion is
asserted directly, member by member, AND behaviourally with a control showing what the
swallowed path actually looks like.

WHY PRECEDENCE NEEDS ITS OWN RULE (Codex MEDIUM)
------------------------------------------------
"The real result replaces the provisional row" is only half a rule. The other half is
that the reverse must never happen: a stale schedule fetch replayed after the results
land would otherwise overwrite a REAL snapshot with a provisional one, and the table
would silently un-learn a played game. Real beats provisional; provisional replaces only
provisional; and the whole-table invariant -- no completed game carries the flag -- is
asserted over the STORED table rather than at the writer, because a writer-side
assertion cannot catch a replay that arrives through a different path.

NOTHING HERE WRITES ``data/`` OR ``artifacts/``. Every write lands under ``tmp_path``.
"""

from __future__ import annotations

import importlib
import inspect
import sys

import pandas as pd
import pytest

from tests.fixtures.elo_sandbox import (
    make_season_games,
    read_sandbox_table,
    redirect_storage_to_sandbox,
    sandbox_builder,
)
from tests.phase33_state import TRAINER_GOLD_LOAD_SITES

LIVE_SEASON = 2026

# The one game the fixture frames flag as provisional, named so the refusal message can
# be asserted against a literal rather than against "some id".
PROVISIONAL_GAME_ID = f"{LIVE_SEASON}_W02_MIA@BUF"

# The gold-side argv for each trainer whose main() takes --season/--week.
_TRAINER_ARGV = {
    "models.train_wp": ["train_wp", "--season", str(LIVE_SEASON), "--week", "all"],
    "models.train_ats": ["train_ats", "--season", str(LIVE_SEASON), "--week", "all"],
    "models.train_ou": ["train_ou", "--season", str(LIVE_SEASON), "--week", "all"],
}

_TRAINER_MATRIX = {
    "models.train_wp": "features_wp",
    "models.train_ats": "features_ats",
    "models.train_ou": "features_ou",
    "models.train": "features_wp",
}


def _snapshots_with_one_provisional() -> pd.DataFrame:
    """A twelve-column snapshot frame: three real rows and one provisional."""
    from scripts.build_elo import build_snapshot_frame

    rows = [
        {
            "game_id": f"{LIVE_SEASON}_W01_MIA@BUF",
            "season": LIVE_SEASON,
            "week": 1,
            "home_team": "BUF",
            "away_team": "MIA",
            "home_elo_pre": 1600.0,
            "away_elo_pre": 1500.0,
            "home_elo_uncertainty": 200.0,
            "away_elo_uncertainty": 210.0,
            "elo_prob_home": 0.66,
            "hfa_used": 48.0,
            "is_provisional": False,
        },
        {
            "game_id": f"{LIVE_SEASON}_W01_DEN@KC",
            "season": LIVE_SEASON,
            "week": 1,
            "home_team": "KC",
            "away_team": "DEN",
            "home_elo_pre": 1650.0,
            "away_elo_pre": 1480.0,
            "home_elo_uncertainty": 190.0,
            "away_elo_uncertainty": 220.0,
            "elo_prob_home": 0.74,
            "hfa_used": 48.0,
            "is_provisional": False,
        },
        {
            "game_id": PROVISIONAL_GAME_ID,
            "season": LIVE_SEASON,
            "week": 2,
            "home_team": "BUF",
            "away_team": "MIA",
            "home_elo_pre": 1612.0,
            "away_elo_pre": 1494.0,
            "home_elo_uncertainty": 195.0,
            "away_elo_uncertainty": 205.0,
            "elo_prob_home": 0.68,
            "hfa_used": 48.0,
            "is_provisional": True,
        },
    ]
    return build_snapshot_frame(rows)


def _gold_frame() -> pd.DataFrame:
    """A gold-shaped matrix whose game_ids include the provisional one.

    Gold carries NO ``is_provisional`` column -- the join subset excludes it -- which is
    exactly why the guard has to resolve the BACKING snapshot rows rather than read a
    flag off the frame in front of it.
    """
    return pd.DataFrame(
        [
            {
                "game_id": f"{LIVE_SEASON}_W01_MIA@BUF",
                "season": LIVE_SEASON,
                "week": 1,
                "home_elo": 1600.0,
                "away_elo": 1500.0,
                "home_won": 1,
                "ats_result": 1,
                "total_result": 1,
            },
            {
                "game_id": PROVISIONAL_GAME_ID,
                "season": LIVE_SEASON,
                "week": 2,
                "home_elo": 1612.0,
                "away_elo": 1494.0,
                "home_won": 0,
                "ats_result": 0,
                "total_result": 0,
            },
        ]
    )


def _seed_trainer_sandbox(monkeypatch, tmp_path, module_name: str):
    """Seed a sandbox gold matrix plus the backing snapshots, and return the root."""
    from data.storage import save_dataframe

    sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
    save_dataframe(
        _snapshots_with_one_provisional(),
        "elo_game_snapshots",
        layer="silver",
        replace_mode=True,
    )
    save_dataframe(
        _gold_frame(), _TRAINER_MATRIX[module_name], layer="gold", replace_mode=True
    )
    return sandbox


class TestEveryTrainerGoldBoundaryRefusesAProvisionalRow:
    """Four entry points, four tests. One proven says nothing about the other three."""

    @pytest.mark.parametrize(
        "module_name", ["models.train_wp", "models.train_ats", "models.train_ou"]
    )
    def test_a_load_dataframe_trainer_refuses_by_name(
        self, tmp_path, monkeypatch, module_name: str
    ) -> None:
        from features.elo_features import ProvisionalSnapshotAsTrainingInputError

        _seed_trainer_sandbox(monkeypatch, tmp_path, module_name)
        monkeypatch.setattr(sys, "argv", _TRAINER_ARGV[module_name])
        trainer = importlib.import_module(module_name)

        with pytest.raises(ProvisionalSnapshotAsTrainingInputError) as excinfo:
            trainer.main()

        assert PROVISIONAL_GAME_ID in str(excinfo.value), (
            "the refusal must NAME the offending game_id, or the operator cannot tell "
            f"which game to wait on. Message: {excinfo.value}"
        )

    def test_the_read_parquet_trainer_refuses_by_name(
        self, tmp_path, monkeypatch
    ) -> None:
        """``models/train.py`` bypasses ``load_dataframe`` -- a FOURTH boundary.

        It reads ``data/gold/features_{target}.parquet`` off a RELATIVE path, so the
        sandbox here is a working directory as well as a redirected store.
        """
        from features.elo_features import ProvisionalSnapshotAsTrainingInputError

        _seed_trainer_sandbox(monkeypatch, tmp_path, "models.train")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["train", "--target", "wp", "--no-clv"])
        trainer = importlib.import_module("models.train")

        with pytest.raises(ProvisionalSnapshotAsTrainingInputError) as excinfo:
            trainer.main()

        assert PROVISIONAL_GAME_ID in str(excinfo.value)

    def test_the_refusal_is_not_catchable_by_any_trainer_handler_tuple(self) -> None:
        """The trainers catch ``(FileNotFoundError, OSError, ValueError, KeyError)``.

        A refusal that landed in one of those would be logged and the trainer would
        ``return`` -- a silent no-train indistinguishable from a missing matrix. Asserted
        on the TYPE rather than on the handler text, because the handler text is what a
        future edit changes.
        """
        from features.elo_features import ProvisionalSnapshotAsTrainingInputError

        for caught in (FileNotFoundError, OSError, ValueError, KeyError):
            assert not issubclass(ProvisionalSnapshotAsTrainingInputError, caught), (
                f"ProvisionalSnapshotAsTrainingInputError must not be a {caught.__name__}: "
                "every trainer's gold-load handler catches that type and would convert "
                "the refusal into a silent return."
            )


class TestTheGuardIsWiredAtEveryPinnedSite:
    """Drift: a fifth trainer appearing without the guard is a failure."""

    def test_every_pinned_site_resolves_and_calls_the_guard(self) -> None:
        assert len(TRAINER_GOLD_LOAD_SITES) == 4

        unguarded: list[tuple[str, str]] = []
        for module_name, function_name, _line in TRAINER_GOLD_LOAD_SITES:
            function = getattr(
                importlib.import_module(module_name), function_name, None
            )
            if function is None or "assert_no_provisional_training_rows" not in (
                inspect.getsource(function)
            ):
                unguarded.append((module_name, function_name))

        assert unguarded == [], (
            f"these gold-loading entry points do not call the guard: {unguarded}. A "
            "trainer that loads gold without it is an unprotected training path, and "
            "the guard being present in three of four proves nothing about the fourth."
        )

    def test_every_pinned_line_still_names_a_gold_load(self) -> None:
        """Provenance check: the recorded line is where the load WAS measured.

        Asserted loosely on purpose -- the line is recorded as provenance, not as a
        pin (the ruling ``CLV_REPORT_ONLY_SITES`` already records). What is asserted is
        that the recorded position is inside the file and reads like a gold load, so a
        transcription error in the tuple is caught while an edit above it is not
        punished.
        """
        from pathlib import Path

        for module_name, _function_name, line in TRAINER_GOLD_LOAD_SITES:
            path = Path(module_name.replace(".", "/") + ".py")
            assert path.is_file(), f"{path} does not exist"
            lines = path.read_text(encoding="utf-8").splitlines()
            assert 1 <= line <= len(lines), (
                f"{module_name} has {len(lines)} lines; the recorded {line} is outside it"
            )
            text = lines[line - 1]
            assert "features_" in text and (
                "load_dataframe" in text or "read_parquet" in text
            ), f"{module_name}:{line} does not read like a gold load: {text.strip()!r}"


class TestTheRefusalCannotBecomeAnEmptyFrame:
    """The swallowed-into-an-empty-frame path Codex named, with its control."""

    @staticmethod
    def _builder_with_raising_elo(monkeypatch, tmp_path, error: Exception):
        from data.storage import save_dataframe
        from scripts.build_features import FeatureMatrixBuilder

        redirect_storage_to_sandbox(monkeypatch, tmp_path)
        save_dataframe(
            make_season_games(LIVE_SEASON, weeks=1),
            "games",
            layer="silver",
            replace_mode=True,
        )

        builder = FeatureMatrixBuilder()

        def _raise(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(builder.elo_calc, "build_features", _raise)
        return builder

    def test_a_source_load_error_IS_swallowed_into_an_empty_frame(
        self, tmp_path, monkeypatch
    ) -> None:
        """The CONTROL. This is the behaviour the dedicated exception must escape.

        Without this test, "the refusal propagates" could be true simply because the
        handler does not exist. It exists, it fires, and it returns a zero-row Elo
        frame with nothing but a warning in the log.
        """
        builder = self._builder_with_raising_elo(
            monkeypatch, tmp_path, ValueError("an ordinary source-load failure")
        )

        sources = builder.load_all_feature_sources(target_season=LIVE_SEASON)

        assert "elo" in sources
        assert len(sources["elo"]) == 0, (
            "a _SOURCE_LOAD_ERRORS member is converted into an empty Elo frame -- that "
            "is today's graceful-degradation contract and the reason a TRAINING refusal "
            "must not be one of those types."
        )

    def test_the_provisional_refusal_reaches_the_caller_instead(
        self, tmp_path, monkeypatch
    ) -> None:
        from features.elo_features import ProvisionalSnapshotAsTrainingInputError

        builder = self._builder_with_raising_elo(
            monkeypatch,
            tmp_path,
            ProvisionalSnapshotAsTrainingInputError(
                f"provisional rows reached training: ['{PROVISIONAL_GAME_ID}']"
            ),
        )

        with pytest.raises(ProvisionalSnapshotAsTrainingInputError):
            builder.load_all_feature_sources(target_season=LIVE_SEASON)

    def test_build_features_does_not_name_the_refusal_in_any_handler(self) -> None:
        """The exclusion is by TYPE, not by an explicit ``except`` that re-raises.

        Naming it in a handler would be the same mistake in the other direction: the
        handler would have to be kept in step with the exception hierarchy forever.
        Asserted as an AST fact so a future ``except
        ProvisionalSnapshotAsTrainingInputError`` cannot be added quietly.
        """
        import ast
        from pathlib import Path

        tree = ast.parse(Path("scripts/build_features.py").read_text(encoding="utf-8"))
        handlers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.type is not None
        ]
        named = {
            child.id
            for handler in handlers
            for child in ast.walk(handler.type)
            if isinstance(child, ast.Name)
        } | {
            child.attr
            for handler in handlers
            for child in ast.walk(handler.type)
            if isinstance(child, ast.Attribute)
        }

        assert "ProvisionalSnapshotAsTrainingInputError" not in named
        assert not ({"Exception", "BaseException"} & named), (
            "a bare Exception handler in this module would catch the refusal again, "
            f"whatever the type hierarchy says. Caught: {sorted(named)}"
        )

    def test_the_refusal_is_excluded_from_the_source_load_error_tuple(self) -> None:
        from features.elo_features import ProvisionalSnapshotAsTrainingInputError
        from scripts.build_features import _SOURCE_LOAD_ERRORS

        for caught in _SOURCE_LOAD_ERRORS:
            assert not issubclass(ProvisionalSnapshotAsTrainingInputError, caught), (
                f"the refusal is a subclass of {caught.__name__}, which is in "
                "_SOURCE_LOAD_ERRORS -- so the Elo source guard would convert it into "
                "an empty frame and the gold build would exit green with no Elo."
            )


class TestPrecedenceInBothDirections:
    """Real beats provisional. Provisional replaces only provisional."""

    @staticmethod
    def _season_with_week_one_played(monkeypatch, tmp_path):
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(LIVE_SEASON, weeks=2, graded_weeks=(1,))
        builder = sandbox_builder(sandbox, games)
        update = builder.update_current_season(season=LIVE_SEASON)
        builder.save_live_append(
            LIVE_SEASON,
            snapshots=update.snapshots,
            games_with_elo=update.games_with_elo,
            rating_history=update.rating_history,
        )
        return sandbox, games, builder

    def test_a_real_row_replaces_a_provisional_one_for_the_same_game(
        self, tmp_path, monkeypatch
    ) -> None:
        from scripts.build_elo import PROVISIONAL_COLUMN

        sandbox, games, builder = self._season_with_week_one_played(
            monkeypatch, tmp_path
        )
        provisional = builder.snapshot_upcoming_week(LIVE_SEASON, 2)
        builder.save_live_append(
            LIVE_SEASON,
            snapshots=provisional,
            games_with_elo=pd.DataFrame(),
            rating_history=pd.DataFrame(),
        )

        graded = games.copy()
        graded.loc[graded["week"] == 2, ["home_score", "away_score"]] = [27.0, 17.0]
        monday = sandbox_builder(sandbox, graded)
        landed = monday.update_current_season(season=LIVE_SEASON)
        monday.save_live_append(
            LIVE_SEASON,
            snapshots=landed.snapshots,
            games_with_elo=landed.games_with_elo,
            rating_history=landed.rating_history,
        )

        stored = read_sandbox_table(sandbox, "elo_game_snapshots")
        assert not stored[PROVISIONAL_COLUMN].any(), (
            "once every game is played no stored row may still claim to be provisional"
        )

    def test_a_provisional_row_offered_over_a_real_one_is_refused_by_name(
        self, tmp_path, monkeypatch
    ) -> None:
        from scripts.build_elo import PROVISIONAL_COLUMN, EloProvisionalPrecedenceError

        sandbox, _games, builder = self._season_with_week_one_played(
            monkeypatch, tmp_path
        )
        stored = read_sandbox_table(sandbox, "elo_game_snapshots")
        assert len(stored) > 0 and not stored[PROVISIONAL_COLUMN].any()

        # Re-label a week-1 row -- a game that HAS been played -- as provisional.
        stale = stored.copy()
        stale[PROVISIONAL_COLUMN] = True

        with pytest.raises(EloProvisionalPrecedenceError) as excinfo:
            builder.save_live_append(
                LIVE_SEASON,
                snapshots=stale,
                games_with_elo=pd.DataFrame(),
                rating_history=pd.DataFrame(),
            )

        message = str(excinfo.value)
        assert stored.iloc[0]["game_id"] in message, (
            f"the refusal must name the game_ids it protected. Message: {message}"
        )

    def test_a_stale_replay_after_the_results_land_reverts_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        """The failure this rule exists for: a Friday fetch replayed on Monday."""
        from scripts.build_elo import PROVISIONAL_COLUMN, EloProvisionalPrecedenceError
        from tests.fixtures.elo_sandbox import per_season_row_digests

        sandbox, games, builder = self._season_with_week_one_played(
            monkeypatch, tmp_path
        )
        friday = builder.snapshot_upcoming_week(LIVE_SEASON, 2)

        graded = games.copy()
        graded.loc[graded["week"] == 2, ["home_score", "away_score"]] = [27.0, 17.0]
        monday = sandbox_builder(sandbox, graded)
        landed = monday.update_current_season(season=LIVE_SEASON)
        monday.save_live_append(
            LIVE_SEASON,
            snapshots=landed.snapshots,
            games_with_elo=landed.games_with_elo,
            rating_history=landed.rating_history,
        )
        before = per_season_row_digests(
            read_sandbox_table(sandbox, "elo_game_snapshots")
        )

        with pytest.raises(EloProvisionalPrecedenceError):
            monday.save_live_append(
                LIVE_SEASON,
                snapshots=friday,
                games_with_elo=pd.DataFrame(),
                rating_history=pd.DataFrame(),
            )

        after_frame = read_sandbox_table(sandbox, "elo_game_snapshots")
        assert per_season_row_digests(after_frame) == before, (
            "a refused replay must leave the stored rows byte-identical; a refusal that "
            "half-applied would be worse than no refusal at all"
        )
        assert not after_frame[PROVISIONAL_COLUMN].any()

    def test_no_completed_game_carries_the_flag_in_the_stored_table(
        self, tmp_path, monkeypatch
    ) -> None:
        """The WHOLE-TABLE invariant, asserted over the stored table after a mixed write.

        Not a property of the writer: a writer-side assertion cannot catch a stale replay
        that arrives through a different path, and the snapshot table carries no scores
        of its own, so the invariant can only be evaluated by joining it to ``games``.
        """
        from scripts.build_elo import PROVISIONAL_COLUMN

        sandbox, games, builder = self._season_with_week_one_played(
            monkeypatch, tmp_path
        )
        provisional = builder.snapshot_upcoming_week(LIVE_SEASON, 2)
        builder.save_live_append(
            LIVE_SEASON,
            snapshots=provisional,
            games_with_elo=pd.DataFrame(),
            rating_history=pd.DataFrame(),
        )

        stored = read_sandbox_table(sandbox, "elo_game_snapshots")
        joined = stored.merge(
            games[["game_id", "home_score", "away_score"]], on="game_id", how="left"
        )
        completed = joined[joined["home_score"].notna() & joined["away_score"].notna()]
        assert len(completed) > 0, "the mixed write must contain completed games"
        offenders = completed[completed[PROVISIONAL_COLUMN]]["game_id"].tolist()
        assert offenders == [], (
            f"these games have scores and are still flagged provisional: {offenders}"
        )
        # And the control: the unplayed half IS flagged, so the invariant above is not
        # holding merely because nothing is flagged anywhere.
        unplayed = joined[joined["home_score"].isna()]
        assert bool(unplayed[PROVISIONAL_COLUMN].all())


class TestAPreFlagParquetReadsBackAsFalse:
    """Schema back-compat at the ONE read seam (Antigravity LOW). No third state."""

    def test_an_eleven_column_snapshot_parquet_reads_back_with_the_flag_false(
        self, tmp_path, monkeypatch
    ) -> None:
        from data.storage import save_dataframe
        from features.elo_features import load_elo_snapshots
        from scripts.build_elo import ELO_SNAPSHOT_COLUMNS, PROVISIONAL_COLUMN

        redirect_storage_to_sandbox(monkeypatch, tmp_path)
        pre_flag = _snapshots_with_one_provisional()[list(ELO_SNAPSHOT_COLUMNS)]
        assert PROVISIONAL_COLUMN not in pre_flag.columns
        save_dataframe(
            pre_flag, "elo_game_snapshots", layer="silver", replace_mode=True
        )

        read_back = load_elo_snapshots()

        assert PROVISIONAL_COLUMN in read_back.columns, (
            "a parquet written before the flag existed must read back WITH it, filled "
            "at the read seam -- not raise a PyArrow schema-union error and not leave "
            "each call site to guess"
        )
        assert read_back[PROVISIONAL_COLUMN].dtype == bool
        assert not read_back[PROVISIONAL_COLUMN].isna().any(), "no null third state"
        assert not read_back[PROVISIONAL_COLUMN].any(), (
            "every pre-flag row describes a game that was already played"
        )

    def test_a_live_append_onto_a_pre_flag_table_leaves_no_nulls_behind(
        self, tmp_path, monkeypatch
    ) -> None:
        """The production case: the live 2,227-row table has eleven columns TODAY.

        ``upsert_silver`` concatenates the surviving rows with the new ones, so a
        twelve-column append onto an eleven-column table would leave 2,227 NaNs in a
        two-valued column. The writer aligns the stored schema first.
        """
        from data.storage import save_dataframe
        from scripts.build_elo import ELO_SNAPSHOT_COLUMNS, PROVISIONAL_COLUMN

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(LIVE_SEASON, weeks=2, graded_weeks=(1,))
        builder = sandbox_builder(sandbox, games)

        # A pre-flag stored table, exactly as production carries it today.
        seeded = _snapshots_with_one_provisional().head(2)[list(ELO_SNAPSHOT_COLUMNS)]
        save_dataframe(seeded, "elo_game_snapshots", layer="silver", replace_mode=True)

        update = builder.update_current_season(season=LIVE_SEASON)
        builder.save_live_append(
            LIVE_SEASON,
            snapshots=update.snapshots,
            games_with_elo=update.games_with_elo,
            rating_history=update.rating_history,
        )

        stored = read_sandbox_table(sandbox, "elo_game_snapshots")
        assert PROVISIONAL_COLUMN in stored.columns
        assert not stored[PROVISIONAL_COLUMN].isna().any(), (
            "the pre-flag rows must be backfilled to False, not left as NaN in a "
            f"two-valued column. Stored dtype: {stored[PROVISIONAL_COLUMN].dtype}"
        )


class TestProvisionalExposureIsObservableNotOnlyRefusable:
    """``provisional_row_counts`` (Codex MEDIUM): counts by (season, week)."""

    def test_counts_are_reported_per_season_and_week(self) -> None:
        from features.elo_features import provisional_row_counts

        counts = provisional_row_counts(_snapshots_with_one_provisional())

        assert counts == {(LIVE_SEASON, 2): 1}, (
            "one provisional row in week 2 and none in week 1. Reporting the counts is "
            f"what makes accidental exposure visible rather than merely refusable. Got "
            f"{counts}."
        )

    def test_a_frame_with_no_provisional_rows_reports_nothing(self) -> None:
        from features.elo_features import provisional_row_counts
        from scripts.build_elo import PROVISIONAL_COLUMN

        frame = _snapshots_with_one_provisional()
        frame = frame[~frame[PROVISIONAL_COLUMN]]

        assert provisional_row_counts(frame) == {}

    def test_a_pre_flag_frame_reports_nothing_rather_than_raising(self) -> None:
        from features.elo_features import provisional_row_counts
        from scripts.build_elo import ELO_SNAPSHOT_COLUMNS

        pre_flag = _snapshots_with_one_provisional()[list(ELO_SNAPSHOT_COLUMNS)]

        assert provisional_row_counts(pre_flag) == {}
