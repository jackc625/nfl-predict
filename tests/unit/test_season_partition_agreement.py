"""Every site that defines the season partition agrees, and changing one alone fails by name.

WHAT THIS MODULE IS FOR (SPEC R6)
----------------------------------
The season partition used to be declared at ELEVEN places in this repository, each with its
own literal. R6's acceptance is that they "agree on one season partition, asserted by a test
that fails if any one of them is changed alone". This is that test.

THE SITES THAT IMPORT THE RULE ARE NOT THE RISK. Ten of the eleven now call
``conf.season_partition``, and a site that computes its value cannot drift from the rule by
construction -- asserting that it agrees is nearly tautological. The risk is the SEPARATE
LITERAL, and after Plan 33.1-09 there are three places where one could come back:

  1. ``backtest/engine.py``'s ``max_backtest_season`` -- the site CONTEXT's four-site inventory
     never named, and the one that silently dropped every 2025 row from every consumer that
     loaded gold through the engine. A literal here does not raise; it produces a
     full-looking frame with a season missing.
  2. ``scripts/promote_models.py``'s two holdout bounds -- the legacy promotion path, which can
     disagree with the gate without any other check noticing.
  3. ``config/gate.toml``'s ``gate.seasons.holdout`` -- the one site that CANNOT derive,
     because TOML has no import. It is a GENERATED MIRROR (see
     ``scripts/sync_gate_holdout.py``), and this test is the thing that catches it when it is
     stale -- whether or not anybody ever runs that command.

EVERY SITE IS EVALUATED, NOT TEXT-MATCHED. The module is imported and the attribute read; the
TOML is parsed; ``models/train.py``'s parser is BUILT and its defaults read. So a reformat is
not a failure and a value difference is. This is the discipline
``tests/api/test_cache_betting.py``'s three-site ``BET_LIST_COLUMNS`` drift test established,
applied to a different kind of fact.

THE SITE LIST COMES FROM THE RECORD, NOT FROM A SECOND LITERAL HERE. It is driven by
``tests.phase33_state.SEASON_PARTITION_SITES``, and a reader must exist for every row -- set
equality in both directions -- so the test and the record cannot disagree about which sites ARE
the partition. A site added to the record with no reader is a failure here, not a silent gap.

THE MUTATION CONTROL IS NOT OPTIONAL. Sixteen values that all derive from one rule agree
trivially; a green agreement test would prove only that nothing was checked. The mutation tests
below change ONE site and assert the check fails AND names that site.

WHAT THIS ASSERTS OVER, AND WHAT IT DELIBERATELY DOES NOT
----------------------------------------------------------
SOURCE sites only. R6's target names "all three model configs", which read
``holdout_seasons: [2021..2024]`` -- but those live in ``artifacts/<id>/metadata.json``, the
RECORD of a past training run, and D33.1-04 PROHIBITS editing them. So the correct assertion is
the opposite of agreement: they must still read 2021-2024 and must DIFFER from the live
partition, and that difference must be REPORTED rather than erased. That is
``TestTheIncumbentRecordsDifferAndAreNotEdited`` below.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest

from conf.season_partition import (
    backtest_seasons,
    default_season_partition,
)
from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The committed gate config. A module-level name so the mutation control can point it at a
#: temporary copy without touching the real file.
GATE_TOML_PATH = REPO_ROOT / "config" / "gate.toml"

#: The three deployed incumbents, and the window every one of them RECORDS. Not a partition
#: site: a historical fact about three past training runs.
_INCUMBENT_RECORDED_WINDOW = (2021, 2022, 2023, 2024)


def _partition():
    """The live partition, from the one committed rule."""
    return default_season_partition()


def _gate_toml_holdout() -> tuple[int, ...]:
    """Parse ``gate.seasons.holdout`` out of the committed TOML.

    Read through ``tomllib`` rather than by regex on purpose: the mirror's VALUE is what must
    agree, and a value read by parsing survives any reformatting of the file that a future
    editor might do.
    """
    parsed = tomllib.loads(GATE_TOML_PATH.read_text(encoding="utf-8"))
    return tuple(int(season) for season in parsed["gate"]["seasons"]["holdout"])


def _train_argparse_default(flag: str) -> tuple[int, ...]:
    """BUILD ``models/train.py``'s parser and read one ``--config-*-seasons`` default.

    Built and parsed rather than grepped: the defaults are computed from the rule at parser
    construction time, so reading the source text would assert about an expression rather than
    about the value a run would actually get.
    """
    from models.train import build_parser

    args = build_parser().parse_args(["--target", "wp"])
    raw: str = getattr(args, flag)
    return tuple(int(season) for season in raw.split(","))


#: Per-site readers. The KEY is the ``(path, symbol)`` pair recorded in
#: ``SEASON_PARTITION_SITES``; the VALUE is (how to read it, what it must equal).
#: The second element is a callable over the partition so the expectation is DERIVED too --
#: writing the expected tuples out here would make this module a seventeenth site.
_SITE_READERS: dict[
    tuple[str, str], tuple[Callable[[], object], Callable[[object], object]]
] = {
    ("models/deploy_gate.py", "HOLDOUT_SEASONS"): (
        lambda: tuple(__import__("models.deploy_gate", fromlist=["x"]).HOLDOUT_SEASONS),
        lambda p: tuple(p.holdout),
    ),
    ("conf/settings.py", "BacktestConfig().seasons"): (
        lambda: tuple(
            __import__("conf.settings", fromlist=["x"]).BacktestConfig().seasons
        ),
        lambda p: tuple(backtest_seasons(p)),
    ),
    ("models/temporal.py", "TemporalSplitConfig.default().train_seasons"): (
        lambda: tuple(
            __import__("models.temporal", fromlist=["x"])
            .TemporalSplitConfig.default()
            .train_seasons
        ),
        lambda p: tuple(p.selection),
    ),
    ("models/temporal.py", "TemporalSplitConfig.default().hp_val_seasons"): (
        lambda: tuple(
            __import__("models.temporal", fromlist=["x"])
            .TemporalSplitConfig.default()
            .hp_val_seasons
        ),
        lambda p: tuple(p.hp_val),
    ),
    ("models/temporal.py", "TemporalSplitConfig.default().holdout_seasons"): (
        lambda: tuple(
            __import__("models.temporal", fromlist=["x"])
            .TemporalSplitConfig.default()
            .holdout_seasons
        ),
        lambda p: tuple(p.holdout),
    ),
    ("models/train.py", "--config-train-seasons default"): (
        lambda: _train_argparse_default("config_train_seasons"),
        lambda p: tuple(p.selection),
    ),
    ("models/train.py", "--config-hp-val-seasons default"): (
        lambda: _train_argparse_default("config_hp_val_seasons"),
        lambda p: tuple(p.hp_val),
    ),
    ("models/train.py", "--config-holdout-seasons default"): (
        lambda: _train_argparse_default("config_holdout_seasons"),
        lambda p: tuple(p.holdout),
    ),
    ("backtest/engine.py", "BacktestConfig().holdout_seasons"): (
        lambda: tuple(
            __import__("backtest.engine", fromlist=["x"])
            .BacktestConfig()
            .holdout_seasons
        ),
        lambda p: tuple(p.holdout),
    ),
    ("backtest/engine.py", "BacktestConfig().max_backtest_season"): (
        lambda: (
            __import__("backtest.engine", fromlist=["x"])
            .BacktestConfig()
            .max_backtest_season
        ),
        lambda p: p.latest_completed_season,
    ),
    ("backtest/engine.py", "BacktestConfig().first_data_season"): (
        lambda: (
            __import__("backtest.engine", fromlist=["x"])
            .BacktestConfig()
            .first_data_season
        ),
        lambda p: p.selection[0],
    ),
    ("scripts/promote_models.py", "_HOLDOUT_FIRST_SEASON"): (
        lambda: (
            __import__("scripts.promote_models", fromlist=["x"])._HOLDOUT_FIRST_SEASON
        ),
        lambda p: p.holdout[0],
    ),
    ("scripts/promote_models.py", "_HOLDOUT_LAST_SEASON"): (
        lambda: (
            __import__("scripts.promote_models", fromlist=["x"])._HOLDOUT_LAST_SEASON
        ),
        lambda p: p.holdout[-1],
    ),
    ("config/gate.toml", "gate.seasons.holdout"): (
        _gate_toml_holdout,
        lambda p: tuple(p.holdout),
    ),
    ("scripts/retrain_models.py", "_default_backtest_config().holdout_seasons"): (
        lambda: tuple(
            __import__("scripts.retrain_models", fromlist=["x"])
            ._default_backtest_config()
            .holdout_seasons
        ),
        lambda p: tuple(p.holdout),
    ),
    ("features/team_form.py", "TEAM_FORM_PER_GAME_FIRST_SEASON"): (
        lambda: (
            __import__(
                "features.team_form", fromlist=["x"]
            ).TEAM_FORM_PER_GAME_FIRST_SEASON
        ),
        lambda p: p.selection[0],
    ),
}


def disagreeing_sites() -> list[str]:
    """Return ``"path :: symbol"`` for every site whose value is not the rule's.

    Factored out of its tests so the mutation control can drive exactly the code path the
    agreement assertion drives, rather than a re-implementation of it.
    """
    partition = _partition()
    out: list[str] = []
    for (path, symbol), (read, expected_of) in _SITE_READERS.items():
        actual = read()
        expected = expected_of(partition)
        if actual != expected:
            out.append(f"{path} :: {symbol} (is {actual!r}, rule says {expected!r})")
    return out


class TestTheRecordAndTheReadersNameTheSameSites:
    """Neither can drift from the other without a failure here."""

    def test_every_recorded_site_has_a_reader_and_vice_versa(self) -> None:
        recorded = {
            (path, symbol) for path, symbol, _ in phase33_state.SEASON_PARTITION_SITES
        }
        assert recorded == set(_SITE_READERS), (
            "tests.phase33_state.SEASON_PARTITION_SITES and this module's readers disagree "
            "about which sites ARE the partition.\n"
            f"  recorded but unread: {sorted(recorded - set(_SITE_READERS))}\n"
            f"  read but unrecorded: {sorted(set(_SITE_READERS) - recorded)}\n"
            "A site recorded with no reader is an unchecked site; a reader with no record is "
            "a partition site nobody wrote down."
        )

    def test_the_mechanism_vocabulary_is_closed_and_the_toml_is_a_mirror(self) -> None:
        """Ruling S2: the one site that cannot derive is recorded as what it is.

        TOML has no import, so ``config/gate.toml`` mirrors the rule rather than deriving from
        it. Calling all the sites "deriving" would overstate the mechanism for that one, in a
        milestone whose whole point is not overstating mechanisms.
        """
        mechanisms = {
            mechanism for _, _, mechanism in phase33_state.SEASON_PARTITION_SITES
        }
        assert mechanisms <= {"derives", "generated mirror"}, mechanisms

        mirrors = {
            (path, symbol)
            for path, symbol, mechanism in phase33_state.SEASON_PARTITION_SITES
            if mechanism == "generated mirror"
        }
        assert mirrors == {("config/gate.toml", "gate.seasons.holdout")}, (
            f"expected exactly the TOML site to be a generated mirror; got {sorted(mirrors)}."
        )


class TestEverySiteAgreesWithTheRule:
    """The agreement itself, evaluated site by site."""

    def test_no_site_disagrees_with_the_committed_rule(self) -> None:
        disagreements = disagreeing_sites()
        assert disagreements == [], (
            "these partition sites disagree with conf.season_partition:\n  "
            + "\n  ".join(disagreements)
            + "\nEach site must DERIVE from the rule (or, for config/gate.toml, be "
            "regenerated by scripts/sync_gate_holdout.py). A site that disagrees is a "
            "second declaration of the partition."
        )

    @pytest.mark.parametrize(
        ("path", "symbol"),
        sorted(_SITE_READERS),
        ids=lambda value: value.replace("/", "-"),
    )
    def test_each_site_individually(self, path: str, symbol: str) -> None:
        """Per-site, so a failure names ONE site instead of a list."""
        read, expected_of = _SITE_READERS[(path, symbol)]
        assert read() == expected_of(_partition()), f"{path} :: {symbol}"

    def test_the_agreed_partition_is_the_one_recorded_in_phase33_state(self) -> None:
        """An OUTSIDE reference, so sixteen sites deriving from one broken rule would fail.

        Without this, every assertion above would still pass if the rule itself changed: they
        compare the sites to the rule, and the rule to itself.

        Re-pointed by Plan 33.2-18 Task 2: the rule's second amendment (D33.2-14) moved the
        selection window to 2002, so the outside reference is the record appended with that
        amendment's witness. Was: ``phase33_state.SEASON_PARTITION_AFTER`` (selection
        2018-2022), kept unedited as the record of the rule before the amendment.
        """
        partition = _partition()
        after = phase33_state.P332_18_SEASON_PARTITION_RULE_PARTITION_AFTER
        assert tuple(partition.selection) == after["selection"]
        assert tuple(partition.hp_val) == after["hp_val"]
        assert tuple(partition.holdout) == after["holdout"]
        assert partition.final_fit[0] == after["final_fit_first"]
        assert partition.final_fit[-1] == after["final_fit_last"]
        assert len(partition.final_fit) == after["final_fit_count"]
        assert partition.latest_completed_season == after["latest_completed_season"]

    def test_2025_is_in_the_partition(self) -> None:
        """SPEC R6's acceptance, stated by name because it is the point of the requirement.

        2025 was absent from every model config, `conf.settings` stopped at 2024, and
        `backtest/engine.py`'s `max_backtest_season` dropped it from every consumer that loaded
        gold through the engine. It is in gold -- 285 games -- and it is now in the partition.
        """
        partition = _partition()
        assert 2025 in partition.holdout, partition.holdout
        assert 2025 in partition.final_fit, partition.final_fit[-3:]

        from backtest.engine import BacktestConfig

        assert BacktestConfig().max_backtest_season >= 2025, (
            "backtest.engine's max_backtest_season is below 2025, so _load_features would "
            "drop every 2025 row before any consumer saw it -- which is exactly how the "
            "season stayed invisible."
        )

    def test_the_three_evaluated_sets_are_disjoint_and_non_empty(self) -> None:
        """SPEC R6's other two acceptance clauses (the adjacency and empty edge cases)."""
        partition = _partition()
        selection = set(partition.selection)
        hp_val = set(partition.hp_val)
        holdout = set(partition.holdout)

        assert selection and hp_val and holdout
        assert not (selection & hp_val)
        assert not (selection & holdout)
        assert not (hp_val & holdout)

        from models.temporal import TemporalSplitConfig

        TemporalSplitConfig(
            train_seasons=list(partition.selection),
            hp_val_seasons=list(partition.hp_val),
            holdout_seasons=list(partition.holdout),
        ).validate()


class TestChangingOneSiteAloneFails:
    """The mutation control. Without it a green agreement test proves nothing was checked."""

    def test_mutation_of_the_engine_max_backtest_season_alone_is_caught(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 2025-hider. A literal here produces a full-looking frame with a season missing.

        The mutation swaps in a SUBCLASS carrying the old literal rather than setting the
        class attribute. ``BacktestConfig`` is a dataclass, so its field default is baked into
        the generated ``__init__`` at class-creation time and setting the attribute changes
        nothing -- the reader would still see the derived value and this control would pass for
        free. That mistake was made and corrected here rather than left as a green assertion
        proving nothing. The subclass is also the more faithful simulation: it is what
        "somebody wrote a literal back into this one site" actually looks like.
        """
        import dataclasses

        from backtest import engine

        @dataclasses.dataclass
        class _StaleEngineConfig(engine.BacktestConfig):
            max_backtest_season: int = 2024

        monkeypatch.setattr(engine, "BacktestConfig", _StaleEngineConfig)
        disagreements = disagreeing_sites()
        assert any("max_backtest_season" in d for d in disagreements), disagreements
        assert not any("holdout_seasons" in d for d in disagreements), (
            "the mutation leaked into a second engine site, so this no longer proves that "
            f"changing ONE site alone is caught: {disagreements}"
        )

    def test_mutation_of_the_deploy_gate_holdout_alone_is_caught(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The single-source gate constant."""
        import models.deploy_gate as gate

        monkeypatch.setattr(gate, "HOLDOUT_SEASONS", (2021, 2022, 2023, 2024))
        disagreements = disagreeing_sites()
        assert any("HOLDOUT_SEASONS" in d for d in disagreements), disagreements

    def test_mutation_of_the_gate_toml_mirror_alone_is_caught(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The GENERATED MIRROR, and the reason the test evaluates the TOML rather than
        trusting that somebody ran ``scripts/sync_gate_holdout.py``.

        Driven against a COPY in tmp_path: the real ``config/gate.toml`` carries the
        ``[baseline.*]`` block whose bytes must not move (D33-11, D33.1-05), and a test that
        wrote to it would be exactly the kind of edit those guards exist to catch.
        """
        stale = tmp_path / "gate.toml"
        stale.write_text(
            GATE_TOML_PATH.read_text(encoding="utf-8").replace(
                "holdout = [2024, 2025]", "holdout = [2021, 2022, 2023, 2024]"
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "tests.unit.test_season_partition_agreement.GATE_TOML_PATH", stale
        )
        disagreements = disagreeing_sites()
        assert any("gate.seasons.holdout" in d for d in disagreements), disagreements

    def test_mutation_of_the_promote_models_bound_alone_is_caught(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The legacy promotion path, which can disagree with the gate unnoticed."""
        import scripts.promote_models as promote

        monkeypatch.setattr(promote, "_HOLDOUT_LAST_SEASON", 2024)
        disagreements = disagreeing_sites()
        assert any("_HOLDOUT_LAST_SEASON" in d for d in disagreements), disagreements

    def test_the_unmutated_check_is_green(self) -> None:
        """Paired with the four above: the mutations are what fail, not the checker."""
        assert disagreeing_sites() == []


class TestTheIncumbentRecordsDifferAndAreNotEdited:
    """D33.1-04 / RESEARCH 11.4: the artifacts are a RECORD, and the difference is the answer.

    R6's target names "all three model configs". They live in ``artifacts/<id>/metadata.json``
    and record ``holdout_seasons: [2021..2024]``. Editing them to agree with the live partition
    would falsify the record of a past training run to unblock a gate -- the defect class this
    milestone exists to detect. So the assertion is that they DIFFER, and that the difference
    is reported rather than erased. They change only when a future re-fit writes new ones.
    """

    def _incumbent_windows(self) -> dict[str, tuple[int, ...]]:
        manifest_path = REPO_ROOT / "artifacts" / "latest.json"
        if not manifest_path.exists():
            pytest.skip(
                "artifacts/latest.json is absent, so there is no deployed incumbent to read. "
                "Restore it per RUNBOOK's clean-checkout step."
            )
        versions = json.loads(manifest_path.read_text(encoding="utf-8"))
        windows: dict[str, tuple[int, ...]] = {}
        for target in ("wp", "ats", "ou"):
            version = versions.get(target)
            if version is None:
                continue
            metadata_path = REPO_ROOT / "artifacts" / version / "metadata.json"
            if not metadata_path.exists():
                continue
            config = json.loads(metadata_path.read_text(encoding="utf-8")).get("config")
            if not config or "holdout_seasons" not in config:
                continue
            windows[target] = tuple(int(s) for s in config["holdout_seasons"])
        if not windows:
            pytest.skip(
                "no deployed incumbent metadata carries a holdout window to read."
            )
        return windows

    def test_every_incumbent_still_records_the_pre_correction_window(self) -> None:
        for target, window in self._incumbent_windows().items():
            assert window == _INCUMBENT_RECORDED_WINDOW, (
                f"the deployed '{target}' incumbent records holdout {list(window)}, not the "
                f"{list(_INCUMBENT_RECORDED_WINDOW)} it was trained on. A metadata.json is "
                "the RECORD of a past training run; if this changed without a re-fit, a "
                "record was edited (D33.1-04 prohibits exactly that)."
            )

    def test_every_incumbent_window_DIFFERS_from_the_live_partition(self) -> None:
        live = tuple(_partition().holdout)
        for target, window in self._incumbent_windows().items():
            assert window != live, (
                f"the deployed '{target}' incumbent's recorded holdout now EQUALS the live "
                f"partition {list(live)}. Under D33.1-04 the correct outcome is a reported "
                "DIFFERENCE, not agreement -- agreement here means somebody edited the "
                "record instead of re-fitting."
            )

    def test_the_difference_is_REPORTED_rather_than_raised(self) -> None:
        """``_incumbent_window`` used to raise on this. It must now report and continue."""
        from scripts.promote_models import _incumbent_window

        artifacts_dir = REPO_ROOT / "artifacts"
        if not (artifacts_dir / "latest.json").exists():
            pytest.skip(
                "artifacts/latest.json is absent; nothing to derive a window from."
            )

        for target in sorted(self._incumbent_windows()):
            window = _incumbent_window(target, artifacts_dir)
            report = window["window_report"]
            assert report, (
                f"'{target}': the incumbent's recorded window differs from the live "
                "partition, so _incumbent_window must REPORT it. An empty report means the "
                "difference is silent, which is what the raise used to prevent."
            )
            assert "2021" in report and "2024" in report, report
            # Review CR-01: all three windows come from the committed rule now, so the
            # report must name every field that moved. The train window is the one that
            # was silently inherited from a VOID artifact before the fix.
            assert "train:" in report, (
                "the report names only some of the fields that moved; a window difference "
                f"that is not stated is the CR-01 defect. Got: {report}"
            )
            assert "in-sample" in report.lower(), (
                "the report must name the consequence -- the gate's re-score of an artifact "
                f"fitted on those seasons is IN-SAMPLE. Got: {report}"
            )


class TestTheRuleWillNotTreatALiveSeasonAsCompleted:
    """Code review WR-01: ``completed_seasons_from`` had no upper bound.

    ``completed_seasons_from`` dropped anything below ``CORPUS_FIRST_SEASON`` and applied no
    ceiling, while its own docstring told callers to "pass a gold frame's ``season`` column".
    Gold is rebuilt DURING a season and 2026 is underway, so following that instruction
    mid-season returned a partition whose holdout was a PARTIAL season, whose hp-val fold had
    moved, and whose final fit covered games that had not been played -- with no error.

    These are the assertions that would have caught it. They are written against the RULE
    rather than against a caller, because the rule is where the bound belongs: a ceiling
    enforced by each caller is a ceiling that the next caller forgets.
    """

    def test_a_season_beyond_the_latest_completed_one_is_dropped(self) -> None:
        from conf.season_partition import (
            LATEST_COMPLETED_SEASON,
            completed_seasons_from,
        )

        live = LATEST_COMPLETED_SEASON + 1
        kept = completed_seasons_from([*range(2002, LATEST_COMPLETED_SEASON + 1), live])

        assert live not in kept, (
            f"season {live} survived completed_seasons_from. It is not COMPLETE -- "
            "LATEST_COMPLETED_SEASON says so -- and a partition that treats an "
            "in-progress season as completed holds out a partial season and finally "
            "fits on games that have not been played."
        )
        assert kept[-1] == LATEST_COMPLETED_SEASON, kept[-1]

    def test_a_gold_frame_carrying_live_rows_cannot_move_the_holdout(self) -> None:
        """The end-to-end shape, measured the way the defect was measured."""
        from conf.season_partition import (
            LATEST_COMPLETED_SEASON,
            derive_season_partition,
        )

        with_live_rows = derive_season_partition(
            range(2002, LATEST_COMPLETED_SEASON + 2)
        )

        assert with_live_rows.holdout == _partition().holdout, (
            f"a completed-season pool containing the live season produced holdout "
            f"{list(with_live_rows.holdout)}, not {list(_partition().holdout)}. Before the "
            "fix, derive_season_partition(range(2002, 2027)) returned hp_val (2024,) and "
            "holdout (2025, 2026) -- silently."
        )
        assert with_live_rows.hp_val == _partition().hp_val
        assert with_live_rows.final_fit[-1] == LATEST_COMPLETED_SEASON

    def test_the_floor_still_drops_too(self) -> None:
        """The control: a ceiling that swallowed the floor would pass the two above."""
        from conf.season_partition import CORPUS_FIRST_SEASON, completed_seasons_from

        kept = completed_seasons_from([1999, 2001, CORPUS_FIRST_SEASON, 2010])

        assert kept == (CORPUS_FIRST_SEASON, 2010), kept
