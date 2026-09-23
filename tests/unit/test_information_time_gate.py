"""The information-time gate: each source's per-game information time against that game's lock.

Phase 33.2, Plan 33.2-01 Task 2 (SPEC R2, D33.2-01). This module replaces the pinned
``LeakageGate.check_time_fence`` tests. That check compared each row's KICKOFF to the build
clock, which asked the wrong question: it blocked every unplayed game and could never catch a
value built from post-lock information. The intents those tests carried are kept here, each
re-asked as the question the gate now answers:

* "raises on future data"  -> a value whose information time is one second after its game's
  lock raises and NAMES the game;
* "passes clean data"      -> a value timed exactly at the lock passes (``<=``);
* "an inert check announces itself (WR-01)" -> a source that loads ZERO rows is reported BY
  NAME as ``empty_unchecked`` and is never counted as checked;
* "convert, never relabel a timezone" -> a UTC-stored instant is compared as the same
  instant, and a NAIVE one now RAISES instead of being aligned.

THE CONTRACT UNDER TEST (Plan 33.2-01 ``<owned_contract>``)
----------------------------------------------------------
Provenance is a PER-ROW frame of ``(game_id, basis, information_time)``. ``per_row`` rows are
lock-compared; ``no_information`` rows carry a NULL time and are VALUE-checked against the
source's declared ``no_information_signature()``. The two bases coexist inside one source --
Elo's 2002 week 1 is genuinely undatable (both teams still at the synthetic 1500 start state)
while week 2 onward is genuinely dated.

Run this module:  uv run pytest tests/unit/test_information_time_gate.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from features.elo_features import EloFeatureBuilder, ensure_provisional_flag
from features.protocol import FeatureBuilder, InformationTimeProvider
from features.provenance import (
    DECLARED_GAME_DURATION,
    PROVENANCE_COLUMNS,
    CoverageReport,
    InformationBasis,
    InformationTimeGate,
    InformationTimeViolation,
    ProvenanceCoverageError,
    SourceCheckState,
    UndatedSourceError,
    build_lock_frame,
    refuse_provenance_columns,
)
from scripts.build_features import _SOURCE_LOAD_ERRORS
from utils import game_lock

ET = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[2]

# Two real-shaped games. G_SUN locks Saturday 2023-12-16 18:00 ET; G_MON locks Sunday
# 2023-12-17 18:00 ET.
G_SUN = "2023_W15_MIN@CIN"
G_MON = "2023_W15_PHI@SEA"
LOCK_SUN = datetime(2023, 12, 16, 18, 0, tzinfo=ET)
LOCK_MON = datetime(2023, 12, 17, 18, 0, tzinfo=ET)


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [G_SUN, G_MON],
            "kickoff_et": [
                pd.Timestamp("2023-12-17 13:00", tz=ET).tz_convert("UTC"),
                pd.Timestamp("2023-12-18 20:15", tz=ET).tz_convert("UTC"),
            ],
        }
    )


def _source() -> pd.DataFrame:
    return pd.DataFrame({"game_id": [G_SUN, G_MON], "elo_home": [1512.0, 1487.0]})


def _provenance(
    times: list[object], bases: list[str] | None = None, ids: list[str] | None = None
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ids if ids is not None else [G_SUN, G_MON],
            "basis": bases if bases is not None else ["per_row"] * len(times),
            "information_time": pd.Series(times, dtype="object"),
        }
    )


_SIGNATURE = {"elo_home": 1500.0}


def _check(
    provenance: pd.DataFrame,
    source: pd.DataFrame | None = None,
    signature: dict | None = None,
    gate: InformationTimeGate | None = None,
) -> SourceCheckState:
    gate = gate if gate is not None else InformationTimeGate()
    return gate.check(
        "elo",
        source if source is not None else _source(),
        provenance,
        build_lock_frame(_games()),
        no_information_signature=_SIGNATURE if signature is None else signature,
    )


# ===========================================================================
# The vocabulary: two bases, two report states, one frame shape, four report sets
# ===========================================================================


class TestTheVocabulary:
    def test_provenance_columns_are_the_one_frame_shape(self) -> None:
        assert PROVENANCE_COLUMNS == ("game_id", "basis", "information_time")

    def test_there_are_exactly_two_bases(self) -> None:
        assert sorted(m.value for m in InformationBasis) == [
            "no_information",
            "per_row",
        ]

    def test_there_are_exactly_two_report_states(self) -> None:
        assert sorted(m.value for m in SourceCheckState) == [
            "checked",
            "empty_unchecked",
        ]

    def test_the_two_vocabularies_share_no_member(self) -> None:
        """``empty_unchecked`` is what the gate COULD DO, never a basis a source declares."""
        bases = {m.value for m in InformationBasis}
        states = {m.value for m in SourceCheckState}
        assert bases.isdisjoint(states)

    def test_the_coverage_report_has_four_named_sets(self) -> None:
        assert [f.name for f in dataclasses.fields(CoverageReport)] == [
            "checked_sources",
            "empty_unchecked_sources",
            "unregistered_sources",
            "post_stage1_sources",
        ]

    def test_the_declared_game_duration_is_four_hours(self) -> None:
        assert timedelta(hours=4) == DECLARED_GAME_DURATION


class TestTheExceptionHierarchyDodgesTheSourceLoadHandler:
    """``_SOURCE_LOAD_ERRORS`` turns a caught failure into an EMPTY frame. A refusal must escape it."""

    @pytest.mark.parametrize(
        "exc",
        [InformationTimeViolation, UndatedSourceError, ProvenanceCoverageError],
        ids=lambda e: e.__name__,
    )
    def test_no_refusal_is_a_member_of_the_source_load_tuple(self, exc: type) -> None:
        assert not issubclass(exc, _SOURCE_LOAD_ERRORS)
        assert not issubclass(exc, ValueError)
        assert not issubclass(exc, RuntimeError)

    def test_the_two_specific_refusals_are_information_time_violations(self) -> None:
        assert issubclass(UndatedSourceError, InformationTimeViolation)
        assert issubclass(ProvenanceCoverageError, InformationTimeViolation)


# ===========================================================================
# The lock comparison (moved from test_leakage_gate: "raises on future data" /
# "passes clean data")
# ===========================================================================


class TestTheLockComparison:
    def test_at_lock_passes(self) -> None:
        assert _check(_provenance([LOCK_SUN, LOCK_MON])) is SourceCheckState.CHECKED

    def test_one_second_after_the_lock_raises_naming_the_game(self) -> None:
        late = LOCK_MON + timedelta(seconds=1)
        with pytest.raises(InformationTimeViolation) as exc:
            _check(_provenance([LOCK_SUN, late]))
        message = str(exc.value)
        assert G_MON in message
        assert G_SUN not in message
        assert "elo" in message
        details = exc.value.details
        assert details["source"] == "elo"
        assert details["game_ids"] == [G_MON]
        assert details["violation_type"] == "information_time"
        assert len(details["information_times"]) == 1
        assert len(details["locks"]) == 1

    def test_the_violation_names_the_information_time_and_the_lock(self) -> None:
        late = LOCK_MON + timedelta(seconds=1)
        with pytest.raises(InformationTimeViolation) as exc:
            _check(_provenance([LOCK_SUN, late]))
        message = str(exc.value)
        assert "2023-12-17" in message  # both instants fall on the lock's date
        assert "23:00:01" in message or "18:00:01" in message

    def test_a_utc_stored_instant_exactly_at_the_et_lock_is_admitted(self) -> None:
        """Convert, never relabel: 23:00 UTC IS 18:00 EST (moved from as_of_fence)."""
        at_lock_utc = pd.Timestamp("2023-12-16 23:00", tz="UTC")
        at_lock_mon_utc = pd.Timestamp("2023-12-17 23:00", tz="UTC")
        state = _check(_provenance([at_lock_utc, at_lock_mon_utc]))
        assert state is SourceCheckState.CHECKED

    def test_a_utc_stored_instant_one_second_late_is_refused(self) -> None:
        late_utc = pd.Timestamp("2023-12-16 23:00:01", tz="UTC")
        with pytest.raises(InformationTimeViolation, match=G_SUN):
            _check(_provenance([late_utc, LOCK_MON]))

    def test_a_naive_information_time_raises_rather_than_being_aligned(self) -> None:
        """The old fence ALIGNED a naive value to ET. Under D33.2-01 it is refused."""
        naive = datetime(2023, 12, 16, 12, 0)
        with pytest.raises(InformationTimeViolation, match="timezone"):
            _check(_provenance([naive, LOCK_MON]))


# ===========================================================================
# Coverage: two-way, one row per game, the wrong-id-format trap (RESEARCH P8)
# ===========================================================================


class TestCoverage:
    def test_a_zero_match_provenance_frame_raises_its_own_message(self) -> None:
        wrong_format = _provenance(
            [LOCK_SUN, LOCK_MON], ids=["2023_15_MIN_CIN", "2023_15_PHI_SEA"]
        )
        with pytest.raises(ProvenanceCoverageError, match="ZERO"):
            _check(wrong_format)

    def test_a_duplicate_provenance_row_raises(self) -> None:
        dup = _provenance([LOCK_SUN, LOCK_SUN, LOCK_MON], ids=[G_SUN, G_SUN, G_MON])
        with pytest.raises(ProvenanceCoverageError, match=G_SUN):
            _check(dup)

    def test_a_game_missing_from_the_map_raises_naming_it(self) -> None:
        missing = _provenance([LOCK_SUN], ids=[G_SUN])
        with pytest.raises(ProvenanceCoverageError) as exc:
            _check(missing)
        assert G_MON in str(exc.value)
        assert "source -> map" in str(exc.value)

    def test_a_map_row_for_a_game_the_source_lacks_raises(self) -> None:
        extra = _provenance(
            [LOCK_SUN, LOCK_MON, LOCK_MON], ids=[G_SUN, G_MON, "2023_W15_DAL@BUF"]
        )
        with pytest.raises(ProvenanceCoverageError) as exc:
            _check(extra)
        assert "2023_W15_DAL@BUF" in str(exc.value)
        assert "map -> source" in str(exc.value)

    def test_a_game_with_no_lock_raises(self) -> None:
        source = pd.concat(
            [_source(), pd.DataFrame({"game_id": ["NO_LOCK"], "elo_home": [1500.0]})],
            ignore_index=True,
        )
        prov = _provenance(
            [LOCK_SUN, LOCK_MON, LOCK_MON], ids=[G_SUN, G_MON, "NO_LOCK"]
        )
        with pytest.raises(ProvenanceCoverageError, match="NO_LOCK"):
            _check(prov, source=source)

    def test_the_wrong_columns_raise(self) -> None:
        bad = _provenance([LOCK_SUN, LOCK_MON]).rename(
            columns={"information_time": "when"}
        )
        with pytest.raises(ProvenanceCoverageError, match="information_time"):
            _check(bad)

    def test_an_unknown_basis_raises(self) -> None:
        bad = _provenance([LOCK_SUN, LOCK_MON], bases=["per_row", "report_only"])
        with pytest.raises(ProvenanceCoverageError, match="report_only"):
            _check(bad)


# ===========================================================================
# The four basis/null combinations and the anti-exemption guard
# ===========================================================================


class TestTheFourBasisCases:
    def test_per_row_with_a_null_time_is_undated(self) -> None:
        with pytest.raises(UndatedSourceError, match=G_MON):
            _check(_provenance([LOCK_SUN, None]))

    def test_no_information_with_a_non_null_time_is_contradictory(self) -> None:
        prov = _provenance([LOCK_SUN, LOCK_MON], bases=["per_row", "no_information"])
        with pytest.raises(ProvenanceCoverageError, match=G_MON):
            _check(prov)

    def test_a_no_information_row_matching_its_signature_passes(self) -> None:
        source = pd.DataFrame({"game_id": [G_SUN, G_MON], "elo_home": [1512.0, 1500.0]})
        prov = _provenance([LOCK_SUN, None], bases=["per_row", "no_information"])
        assert _check(prov, source=source) is SourceCheckState.CHECKED

    def test_a_false_no_information_claim_is_refused(self) -> None:
        """The claim is CHECKED, never believed (RESEARCH P2)."""
        source = pd.DataFrame({"game_id": [G_SUN, G_MON], "elo_home": [1512.0, 1487.0]})
        prov = _provenance([LOCK_SUN, None], bases=["per_row", "no_information"])
        with pytest.raises(UndatedSourceError) as exc:
            _check(prov, source=source)
        message = str(exc.value)
        assert G_MON in message
        assert "elo_home" in message
        assert "1487" in message

    def test_an_empty_signature_beside_a_no_information_row_is_refused(self) -> None:
        """The anti-exemption guard: 'uncheckable' cannot be self-granted."""
        source = pd.DataFrame({"game_id": [G_SUN, G_MON], "elo_home": [1512.0, 1500.0]})
        prov = _provenance([LOCK_SUN, None], bases=["per_row", "no_information"])
        with pytest.raises(ProvenanceCoverageError, match="signature"):
            _check(prov, source=source, signature={})

    def test_a_none_signature_entry_means_the_value_must_be_null(self) -> None:
        source = pd.DataFrame({"game_id": [G_SUN, G_MON], "elo_home": [1512.0, None]})
        prov = _provenance([LOCK_SUN, None], bases=["per_row", "no_information"])
        assert (
            _check(prov, source=source, signature={"elo_home": None})
            is SourceCheckState.CHECKED
        )
        source.loc[1, "elo_home"] = 3.0
        with pytest.raises(UndatedSourceError, match="elo_home"):
            _check(prov, source=source, signature={"elo_home": None})

    def test_a_signature_naming_an_absent_column_is_refused(self) -> None:
        source = pd.DataFrame({"game_id": [G_SUN, G_MON], "elo_home": [1512.0, 1500.0]})
        prov = _provenance([LOCK_SUN, None], bases=["per_row", "no_information"])
        with pytest.raises(ProvenanceCoverageError, match="not_a_column"):
            _check(prov, source=source, signature={"not_a_column": 0.0})

    def test_mixed_bases_pass_in_one_call(self) -> None:
        source = pd.DataFrame({"game_id": [G_SUN, G_MON], "elo_home": [1500.0, 1487.0]})
        prov = _provenance([None, LOCK_MON], bases=["no_information", "per_row"])
        assert _check(prov, source=source) is SourceCheckState.CHECKED


# ===========================================================================
# The empty source (moved from test_leakage_gate's WR-01 "inspected NOTHING")
# ===========================================================================


class TestTheEmptySource:
    def test_a_zero_row_source_is_empty_unchecked_by_name(self) -> None:
        gate = InformationTimeGate()
        state = gate.check(
            "snaps",
            pd.DataFrame(),
            pd.DataFrame({column: [] for column in PROVENANCE_COLUMNS}),
            build_lock_frame(_games()),
            no_information_signature={},
        )
        assert state is SourceCheckState.EMPTY_UNCHECKED
        assert gate.empty_unchecked_sources == ("snaps",)
        assert gate.checked_sources == ()

    def test_the_checked_tally_counts_only_checked_sources(self) -> None:
        gate = InformationTimeGate()
        _check(_provenance([LOCK_SUN, LOCK_MON]), gate=gate)
        gate.check(
            "injury",
            pd.DataFrame(),
            pd.DataFrame({column: [] for column in PROVENANCE_COLUMNS}),
            build_lock_frame(_games()),
            no_information_signature={},
        )
        assert gate.checked_sources == ("elo",)
        assert gate.empty_unchecked_sources == ("injury",)


# ===========================================================================
# The lock-frame wrapper and the gold dtype guard
# ===========================================================================


class TestBuildLockFrameIsAWrapperNotAnAlias:
    """Plan 33.2-02's identity delegate patches ``utils.game_lock``; an alias would miss it."""

    def test_it_reaches_utils_game_lock_at_call_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        real = game_lock.lock_frame

        def _counting(frame: pd.DataFrame) -> pd.Series:
            calls.append(len(frame))
            return real(frame)

        monkeypatch.setattr(game_lock, "lock_frame", _counting)
        locks = build_lock_frame(_games())
        assert calls == [2]
        assert locks[G_SUN] == LOCK_SUN


class TestTheGoldDtypeGuard:
    def test_a_numeric_frame_passes(self) -> None:
        refuse_provenance_columns(
            pd.DataFrame({"game_id": ["G"], "elo_home": [1.0]}),
            build_clock_columns=("feature_timestamp",),
        )

    def test_a_datetime_column_is_refused(self) -> None:
        frame = pd.DataFrame(
            {"game_id": ["G"], "kickoff_et": [pd.Timestamp("2023-01-01", tz="UTC")]}
        )
        with pytest.raises(InformationTimeViolation, match="kickoff_et"):
            refuse_provenance_columns(frame, build_clock_columns=("feature_timestamp",))

    def test_a_provenance_named_column_is_refused(self) -> None:
        frame = pd.DataFrame({"game_id": ["G"], "elo_information_time": [1.0]})
        with pytest.raises(InformationTimeViolation, match="elo_information_time"):
            refuse_provenance_columns(frame, build_clock_columns=("feature_timestamp",))

    def test_the_registered_build_clock_is_the_only_datetime_admitted(self) -> None:
        frame = pd.DataFrame(
            {
                "game_id": ["G"],
                "feature_timestamp": [pd.Timestamp("2026-01-01", tz="UTC")],
            }
        )
        refuse_provenance_columns(frame, build_clock_columns=("feature_timestamp",))
        with pytest.raises(InformationTimeViolation, match="feature_timestamp"):
            refuse_provenance_columns(frame, build_clock_columns=())


# ===========================================================================
# The Elo supplier, on REAL 2002 silver (read-only)
# ===========================================================================


@pytest.fixture(scope="module")
def silver_games() -> pd.DataFrame:
    return pd.read_parquet(REPO_ROOT / "data" / "silver" / "games.parquet")


@pytest.fixture(scope="module")
def silver_snapshots() -> pd.DataFrame:
    return ensure_provisional_flag(
        pd.read_parquet(REPO_ROOT / "data" / "silver" / "elo_game_snapshots.parquet")
    )


def _elo_builder(games: pd.DataFrame, snapshots: pd.DataFrame) -> EloFeatureBuilder:
    builder = EloFeatureBuilder()
    builder._snapshots_df = snapshots
    builder._history_df = games
    return builder


def _slice_2002(games: pd.DataFrame, weeks: tuple[int, ...]) -> pd.DataFrame:
    return games.loc[(games["season"] == 2002) & (games["week"].isin(weeks))].copy()


_AS_OF = datetime(2026, 9, 21, 12, 0, tzinfo=ET)


class TestTheEloSupplierConforms:
    def test_elo_satisfies_both_protocols_with_no_shared_base(self) -> None:
        builder = EloFeatureBuilder()
        assert isinstance(builder, InformationTimeProvider)
        assert isinstance(builder, FeatureBuilder)

    def test_the_signature_is_read_off_the_start_state(self) -> None:
        signature = EloFeatureBuilder().no_information_signature()
        assert signature == {
            "home_elo": 1500.0,
            "away_elo": 1500.0,
            "home_elo_uncertainty": 350.0,
            "away_elo_uncertainty": 350.0,
        }
        assert "elo_prob_home" not in signature
        assert "hfa_used" not in signature


class TestTheEloSupplierOnReal2002:
    """The three cases the cross-AI review named, each stated as a case."""

    def test_the_first_game_of_2002_is_no_information(
        self, silver_games: pd.DataFrame, silver_snapshots: pd.DataFrame
    ) -> None:
        games = _slice_2002(silver_games, (1,))
        builder = _elo_builder(silver_games, silver_snapshots)
        prov = builder.information_times(games).set_index("game_id")
        row = prov.loc["2002_W01_SF@NYG"]
        assert row["basis"] == "no_information"
        assert pd.isna(row["information_time"])

    def test_every_2002_week_1_row_is_no_information_including_monday_night(
        self, silver_games: pd.DataFrame, silver_snapshots: pd.DataFrame
    ) -> None:
        """Per-TEAM, not global: by the Monday lock other games had finished, not ITS teams'."""
        games = _slice_2002(silver_games, (1,))
        builder = _elo_builder(silver_games, silver_snapshots)
        prov = builder.information_times(games)
        assert len(prov) == len(games) == 16
        assert set(prov["basis"]) == {"no_information"}
        assert prov["information_time"].isna().all()
        monday = games.sort_values("kickoff_et").iloc[-1]["game_id"]
        basis_by_game = dict(zip(prov["game_id"], prov["basis"], strict=True))
        assert basis_by_game[monday] == "no_information"

    def test_the_week_1_values_satisfy_the_declared_signature(
        self, silver_games: pd.DataFrame, silver_snapshots: pd.DataFrame
    ) -> None:
        games = _slice_2002(silver_games, (1,))
        builder = _elo_builder(silver_games, silver_snapshots)
        state = InformationTimeGate().check(
            "elo",
            builder.build_features(games, _AS_OF),
            builder.information_times(games),
            build_lock_frame(games),
            no_information_signature=builder.no_information_signature(),
        )
        assert state is SourceCheckState.CHECKED

    def test_weeks_2_and_3_are_per_row_and_at_or_before_their_locks(
        self, silver_games: pd.DataFrame, silver_snapshots: pd.DataFrame
    ) -> None:
        games = _slice_2002(silver_games, (2, 3))
        builder = _elo_builder(silver_games, silver_snapshots)
        prov = builder.information_times(games).set_index("game_id")
        assert set(prov["basis"]) == {"per_row"}
        locks = build_lock_frame(games)
        for gid, when in prov["information_time"].items():
            assert game_lock.is_admissible(when, locks[gid]), gid

    def test_a_week_2_row_is_timed_by_the_latest_week_1_game_it_reads(
        self, silver_games: pd.DataFrame, silver_snapshots: pd.DataFrame
    ) -> None:
        """The rank columns read every team's week-2 state, so the week-1 finale is a contributor.

        Every team played in 2002 week 1, so every week-2 row's rank inputs include the
        Monday-night game's result. Its end (kickoff + the declared duration) is therefore
        the information time of EVERY week-2 row -- not the build clock, not kickoff_et.
        """
        week1 = _slice_2002(silver_games, (1,))
        expected = week1["kickoff_et"].max() + DECLARED_GAME_DURATION
        games = _slice_2002(silver_games, (2,))
        builder = _elo_builder(silver_games, silver_snapshots)
        prov = builder.information_times(games)
        assert (prov["information_time"] == expected).all()

    def test_a_missing_snapshot_is_a_coverage_defect_not_no_information(
        self, silver_games: pd.DataFrame, silver_snapshots: pd.DataFrame
    ) -> None:
        games = _slice_2002(silver_games, (1, 2))
        dropped = games[games["week"] == 2].iloc[0]["game_id"]
        builder = _elo_builder(
            silver_games, silver_snapshots.loc[silver_snapshots["game_id"] != dropped]
        )
        source = builder.build_features(games, _AS_OF)
        prov = builder.information_times(games)
        assert dropped in set(source["game_id"])
        assert dropped not in set(prov["game_id"])
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate().check(
                "elo",
                source,
                prov,
                build_lock_frame(games),
                no_information_signature=builder.no_information_signature(),
            )
        assert dropped in str(exc.value)
        assert "source -> map" in str(exc.value)

    def test_mixed_rows_pass_in_one_call_and_a_planted_late_row_is_refused(
        self, silver_games: pd.DataFrame, silver_snapshots: pd.DataFrame
    ) -> None:
        games = _slice_2002(silver_games, (1, 2, 3))
        builder = _elo_builder(silver_games, silver_snapshots)
        source = builder.build_features(games, _AS_OF)
        prov = builder.information_times(games)
        locks = build_lock_frame(games)
        signature = builder.no_information_signature()

        assert set(prov["basis"]) == {"per_row", "no_information"}
        assert (
            InformationTimeGate().check(
                "elo", source, prov, locks, no_information_signature=signature
            )
            is SourceCheckState.CHECKED
        )

        planted = prov.copy()
        target = planted.index[planted["basis"] == "per_row"][0]
        target_game = planted.loc[target, "game_id"]
        planted.loc[target, "information_time"] = locks[target_game] + timedelta(
            seconds=1
        )
        with pytest.raises(InformationTimeViolation) as exc:
            InformationTimeGate().check(
                "elo", source, planted, locks, no_information_signature=signature
            )
        assert exc.value.details["game_ids"] == [target_game]


# ===========================================================================
# PLAN 33.2-20 TASK 1: the gate ARMED.
#
# Until this plan the gate checked WHATEVER WAS REGISTERED. From here it checks that
# EVERYTHING IS REGISTERED, that every registered key's merge disposition holds on the
# FINAL matrices, that the opponent-adjusted family merged after Stage 1 was checked at
# its own merge site, and that the `games` key -- which DEFINES the lock and so cannot
# supply a non-circular information time -- carries the second declared basis with its
# values checked rather than silence.
#
# The ledger exists because the registration state used to live in a PROSE COMMENT that
# was rewritten from memory at each plan: `team_form` went missing from it entirely and
# `games` went undisposed. A dict the gate READS is not a comment.
# ===========================================================================


from features.provenance import (
    EXPECTED_CHECKED_SOURCES,
    KNOWN_UNRELATED_SCAN_TOKENS,
    MERGE_DISPOSITION_BY_KEY,
    MERGE_DISPOSITIONS,
    POST_STAGE1_FAMILY_DISPOSITIONS,
    REGISTRY_KEY_DISPOSITIONS,
    derive_post_stage1_gap,
)

_NINE_REGISTRY_KEYS: frozenset[str] = frozenset(
    {
        "games",
        "team_form",
        "elo",
        "contextual",
        "weather",
        "market",
        "qb_tracking",
        "snaps",
        "injury",
    }
)


def _report(
    checked: tuple[str, ...],
    empty: tuple[str, ...] = (),
    unregistered: tuple[str, ...] = (),
    post_stage1: tuple[str, ...] | None = None,
) -> CoverageReport:
    return CoverageReport(
        checked_sources=checked,
        empty_unchecked_sources=empty,
        unregistered_sources=unregistered,
        post_stage1_sources=(
            derive_post_stage1_gap(checked) if post_stage1 is None else post_stage1
        ),
    )


_ALL_TEN: tuple[str, ...] = (
    *sorted(_NINE_REGISTRY_KEYS),
    *sorted(POST_STAGE1_FAMILY_DISPOSITIONS),
)


class TestTheNineKeyLedger:
    """One row per registry key, and the gate raises on a difference in EITHER direction."""

    def test_the_ledger_has_exactly_the_nine_registry_keys(self) -> None:
        assert set(REGISTRY_KEY_DISPOSITIONS) == _NINE_REGISTRY_KEYS
        assert len(REGISTRY_KEY_DISPOSITIONS) == 9

    def test_every_row_says_something(self) -> None:
        empty = sorted(k for k, v in REGISTRY_KEY_DISPOSITIONS.items() if not v.strip())
        assert empty == []

    def test_the_ledger_matches_the_live_registry(self) -> None:
        from scripts.build_features import FEATURE_SOURCE_KEYS

        live = {"games", *FEATURE_SOURCE_KEYS}
        assert set(REGISTRY_KEY_DISPOSITIONS) == live

    def test_a_tenth_registry_key_raises_naming_the_direction(self) -> None:
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.assert_registry_is_fully_disposed(
                [*_NINE_REGISTRY_KEYS, "line_movement"]
            )
        assert "line_movement" in str(exc.value)
        assert exc.value.details["registered_without_a_ledger_row"] == ["line_movement"]
        assert exc.value.details["ledger_rows_without_a_registry_key"] == []

    def test_a_ledger_row_with_no_registry_key_raises_the_other_direction(self) -> None:
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.assert_registry_is_fully_disposed(
                sorted(_NINE_REGISTRY_KEYS - {"team_form"})
            )
        assert "team_form" in str(exc.value)
        assert exc.value.details["ledger_rows_without_a_registry_key"] == ["team_form"]
        assert exc.value.details["registered_without_a_ledger_row"] == []

    def test_the_live_registry_passes(self) -> None:
        from scripts.build_features import FEATURE_SOURCE_KEYS

        InformationTimeGate.assert_registry_is_fully_disposed(
            ["games", *FEATURE_SOURCE_KEYS]
        )

    def test_the_games_row_names_the_basis_and_the_values_check(self) -> None:
        row = REGISTRY_KEY_DISPOSITIONS["games"]
        assert "no_information" in row
        assert "facts_at_lock" in row

    def test_the_weather_row_names_the_plan_that_owns_the_fence(self) -> None:
        assert "33.2-12" in REGISTRY_KEY_DISPOSITIONS["weather"]

    def test_the_team_form_row_exists_and_names_its_supplier(self) -> None:
        """The key that went missing from the prose comment; the whole reason for the dict."""
        assert "team_form" in REGISTRY_KEY_DISPOSITIONS
        assert "33.2-14" in REGISTRY_KEY_DISPOSITIONS["team_form"]


class TestTheMergeDispositionVocabulary:
    def test_there_are_exactly_two_merge_dispositions(self) -> None:
        assert MERGE_DISPOSITIONS == ("merged", "checked_not_merged")

    def test_every_registry_key_has_exactly_one(self) -> None:
        assert set(MERGE_DISPOSITION_BY_KEY) == set(REGISTRY_KEY_DISPOSITIONS)
        bad = sorted(
            k
            for k, v in MERGE_DISPOSITION_BY_KEY.items()
            if v not in MERGE_DISPOSITIONS
        )
        assert bad == []

    def test_market_is_the_only_checked_not_merged_key(self) -> None:
        """Pinned to ONE key, so the second value cannot widen into a way to excuse a source."""
        assert sorted(
            k for k, v in MERGE_DISPOSITION_BY_KEY.items() if v == "checked_not_merged"
        ) == ["market"]

    def test_the_post_stage1_family_is_declared_separately_and_is_merged(self) -> None:
        assert sorted(POST_STAGE1_FAMILY_DISPOSITIONS) == ["opponent_adj"]
        assert "opponent_adj" not in REGISTRY_KEY_DISPOSITIONS

    def test_the_expected_checked_set_is_exactly_ten_names(self) -> None:
        assert frozenset(_ALL_TEN) == EXPECTED_CHECKED_SOURCES
        assert len(EXPECTED_CHECKED_SOURCES) == 10


class TestTheKnownUnrelatedScanTokens:
    """The source-scan baseline, declared HERE so Task 1's own verify can read it."""

    def test_it_is_the_one_measured_token(self) -> None:
        assert KNOWN_UNRELATED_SCAN_TOKENS == ("discrete_indicators_exempt",)

    def test_its_own_name_is_outside_the_scan_vocabulary(self) -> None:
        """features/provenance.py is SCANNED, so a name matching the pattern would self-hit."""
        import re

        pattern = re.compile(
            r"report_only|allow_list|allowlist|skip_check|exempt", re.I
        )
        assert not pattern.search("KNOWN_UNRELATED_SCAN_TOKENS")

    def test_the_baseline_is_non_empty(self) -> None:
        """The non-vacuity control on the subtraction the scan performs."""
        assert len(KNOWN_UNRELATED_SCAN_TOKENS) > 0


# ---------------------------------------------------------------------------
# The DISPOSITION-AWARE merge assertion, on the FINAL matrices
# ---------------------------------------------------------------------------

#: A correct post-rung-9 shape: team_form arrives RENAMED with home_/away_ prefixes,
#: qb_tracking arrives as home_qb_adjustment / away_qb_adjustment, and market arrives
#: nowhere. A source-NAME match would reject all three.
_CORRECT_ARRIVALS: dict[str, tuple[str, ...]] = {
    "games": ("game_id", "season", "week", "home_score", "away_score"),
    "team_form": ("home_off_rolling_epa", "away_def_rolling_epa"),
    "elo": ("home_elo", "away_elo"),
    "contextual": ("rest_differential",),
    "weather": ("temp_f", "wind_mph"),
    "market": (),
    "qb_tracking": ("home_qb_adjustment", "away_qb_adjustment"),
    "snaps": ("home_snap_continuity",),
    "injury": ("home_injury_coverage",),
    "opponent_adj": ("home_off_rolling_opp_adj_epa",),
}


def _correct_matrix() -> pd.DataFrame:
    columns = sorted({c for cols in _CORRECT_ARRIVALS.values() for c in cols})
    return pd.DataFrame({column: [0.0] for column in columns})


class TestTheDispositionAwareMergeAssertion:
    """Registration proves nothing about arrival -- combine_features has no generic loop."""

    def test_correct_post_rung_nine_gold_passes(self) -> None:
        InformationTimeGate.assert_merge_dispositions(
            {"features_wp": _correct_matrix()}, _CORRECT_ARRIVALS
        )

    def test_a_merged_key_with_an_empty_arrival_record_raises(self) -> None:
        arrivals = {**_CORRECT_ARRIVALS, "snaps": ()}
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.assert_merge_dispositions(
                {"features_wp": _correct_matrix()}, arrivals
            )
        assert "snaps" in str(exc.value)
        assert "merged" in str(exc.value)

    def test_a_merged_key_whose_arrivals_left_the_final_matrix_raises(self) -> None:
        matrix = _correct_matrix().drop(columns=["home_elo", "away_elo"])
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.assert_merge_dispositions(
                {"features_ats": matrix}, _CORRECT_ARRIVALS
            )
        assert "elo" in str(exc.value)
        assert "features_ats" in str(exc.value)

    def test_the_failure_message_quotes_the_phase28_lesson(self) -> None:
        arrivals = {**_CORRECT_ARRIVALS, "injury": ()}
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.assert_merge_dispositions(
                {"features_wp": _correct_matrix()}, arrivals
            )
        assert "NO generic loop over feature_sources" in str(exc.value)

    def test_market_with_a_restored_merge_block_raises(self) -> None:
        """A restored merge seam lands columns; the checked_not_merged inverse catches it."""
        arrivals = {**_CORRECT_ARRIVALS, "market": ("snapshot_spread",)}
        matrix = _correct_matrix()
        matrix["snapshot_spread"] = 0.0
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.assert_merge_dispositions(
                {"features_wp": matrix}, arrivals
            )
        assert "market" in str(exc.value)
        assert "checked_not_merged" in str(exc.value)

    def test_a_market_predicate_column_in_a_final_matrix_raises(self) -> None:
        """Even with an EMPTY arrival record: the predicate is the registry's own answer."""
        matrix = _correct_matrix()
        matrix["snapshot_total"] = 0.0
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.assert_merge_dispositions(
                {"features_ou": matrix}, _CORRECT_ARRIVALS
            )
        assert "snapshot_total" in str(exc.value)

    def test_a_key_with_no_arrival_record_at_all_raises(self) -> None:
        arrivals = {k: v for k, v in _CORRECT_ARRIVALS.items() if k != "contextual"}
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.assert_merge_dispositions(
                {"features_wp": _correct_matrix()}, arrivals
            )
        assert "contextual" in str(exc.value)

    def test_the_post_stage1_family_must_arrive_too(self) -> None:
        matrix = _correct_matrix().drop(columns=["home_off_rolling_opp_adj_epa"])
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.assert_merge_dispositions(
                {"features_wp": matrix}, _CORRECT_ARRIVALS
            )
        assert "opponent_adj" in str(exc.value)


# ---------------------------------------------------------------------------
# The ONE final coverage refusal
# ---------------------------------------------------------------------------


class TestTheFinalCoverageRefusal:
    def test_all_ten_checked_passes(self) -> None:
        InformationTimeGate.refuse_incomplete_coverage(_report(_ALL_TEN))

    def test_nine_registry_keys_without_the_family_refuses(self) -> None:
        """The exact gap an exact nine-key registry equality cannot express."""
        nine = tuple(sorted(_NINE_REGISTRY_KEYS))
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.refuse_incomplete_coverage(_report(nine))
        assert "opponent_adj" in str(exc.value)

    def test_a_post_stage1_merge_that_skipped_its_gate_check_refuses_by_name(
        self,
    ) -> None:
        nine = tuple(sorted(_NINE_REGISTRY_KEYS))
        report = _report(nine)
        assert report.post_stage1_sources == ("opponent_adj",)
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.refuse_incomplete_coverage(report)
        assert exc.value.details["post_stage1_unchecked"] == ["opponent_adj"]

    def test_the_post_stage1_set_is_derived_never_a_literal(self) -> None:
        assert derive_post_stage1_gap(_ALL_TEN) == ()
        assert derive_post_stage1_gap(()) == ("opponent_adj",)

    def test_an_empty_unchecked_source_refuses_by_name(self) -> None:
        report = _report(tuple(n for n in _ALL_TEN if n != "snaps"), empty=("snaps",))
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.refuse_incomplete_coverage(report)
        assert exc.value.details["empty_unchecked"] == ["snaps"]

    def test_an_unregistered_source_refuses_by_name(self) -> None:
        report = _report(
            tuple(n for n in _ALL_TEN if n != "market"), unregistered=("market",)
        )
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.refuse_incomplete_coverage(report)
        assert exc.value.details["unregistered"] == ["market"]

    def test_a_source_checked_that_is_not_a_declared_name_refuses(self) -> None:
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.refuse_incomplete_coverage(
                _report((*_ALL_TEN, "line_movement"))
            )
        assert "line_movement" in str(exc.value)

    def test_the_refusal_names_the_ten_it_wanted(self) -> None:
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate.refuse_incomplete_coverage(_report(("elo",)))
        assert exc.value.details["expected_checked"] == sorted(_ALL_TEN)


# ---------------------------------------------------------------------------
# The `games` disposition: the SECOND declared basis, values CHECKED
# ---------------------------------------------------------------------------

_PLANTED_MOVE = """
[[moves]]
game_id = "{game_id}"
move_index = 1
what_moved = "{what_moved}"
from_value = "{from_value}"
to_value = "{to_value}"
announced_at_utc = "{announced}"
source_published_at = "{announced}"
lock_utc = "{lock}"
verdict = "{verdict}"
source_url = "https://example.invalid/planted"
notes = "planted by tests/unit/test_information_time_gate.py"
"""


def _games_with_schedule_facts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [G_SUN, G_MON],
            "season": [2023, 2023],
            "week": [15, 15],
            "home_team": ["CIN", "SEA"],
            "away_team": ["MIN", "PHI"],
            "stadium_id": ["CIN00", "SEA00"],
            "kickoff_et": [
                pd.Timestamp("2023-12-17 13:00", tz=ET).tz_convert("UTC"),
                pd.Timestamp("2023-12-18 20:15", tz=ET).tz_convert("UTC"),
            ],
        }
    )


class TestTheGamesDisposition:
    """`games` DEFINES the lock, so it takes the OTHER basis -- and pays its price."""

    def test_a_frame_whose_facts_match_the_table_is_checked(self, tmp_path) -> None:
        table = tmp_path / "moves.toml"
        table.write_text("", encoding="utf-8")
        gate = InformationTimeGate()
        state = gate.check_games(_games_with_schedule_facts(), table_path=table)
        assert state is SourceCheckState.CHECKED
        assert gate.checked_sources == ("games",)

    def test_an_empty_games_frame_is_empty_unchecked_by_name(self, tmp_path) -> None:
        table = tmp_path / "moves.toml"
        table.write_text("", encoding="utf-8")
        gate = InformationTimeGate()
        state = gate.check_games(pd.DataFrame(), table_path=table)
        assert state is SourceCheckState.EMPTY_UNCHECKED
        assert gate.empty_unchecked_sources == ("games",)

    def test_a_post_lock_date_move_the_frame_carries_raises_naming_the_game(
        self, tmp_path
    ) -> None:
        """A neutralised WEEK or DATE is post-lock schedule information in gold's own week
        column: the games frame hands both forward unresolved."""
        games = _games_with_schedule_facts()
        table = tmp_path / "moves.toml"
        table.write_text(
            _PLANTED_MOVE.format(
                game_id=G_SUN,
                what_moved="date",
                from_value="2023-12-16",
                to_value="2023-12-17",
                announced="2023-12-17T12:00:00+00:00",
                lock="2023-12-16T23:00:00+00:00",
                verdict="post_lock",
            ),
            encoding="utf-8",
        )
        with pytest.raises(ProvenanceCoverageError) as exc:
            InformationTimeGate().check_games(games, table_path=table)
        assert G_SUN in str(exc.value)
        assert exc.value.details["game_ids"] == [G_SUN]
        assert exc.value.details["fields"] == ["kickoff_et"]

    def test_the_comparison_is_load_bearing_not_decorative(self, tmp_path) -> None:
        """The control: WITHOUT the facts_at_lock comparison the planted game PASSES.

        This is what proves the `games` row is a checked disposition rather than an
        exemption wearing one's clothes.
        """
        from features.schedule_moves import facts_at_lock

        games = _games_with_schedule_facts()
        table = tmp_path / "moves.toml"
        table.write_text(
            _PLANTED_MOVE.format(
                game_id=G_SUN,
                what_moved="date",
                from_value="2023-12-16",
                to_value="2023-12-17",
                announced="2023-12-17T12:00:00+00:00",
                lock="2023-12-16T23:00:00+00:00",
                verdict="post_lock",
            ),
            encoding="utf-8",
        )
        # A check that only ever compares a row against ITSELF -- which is what removing
        # the facts_at_lock comparison leaves -- finds nothing, on this very frame.
        without_the_comparison = [
            row["game_id"]
            for _, row in games.iterrows()
            if row["kickoff_et"] != row["kickoff_et"]
        ]
        assert without_the_comparison == []

        # And the comparison itself really does see a difference for that game.
        row = games.loc[games["game_id"] == G_SUN].iloc[0]
        facts = facts_at_lock(row["game_id"], row, table_path=table)
        assert facts.neutralised is True
        assert facts.kickoff_et is not None
        assert facts.kickoff_et != row["kickoff_et"]

    def test_a_post_lock_venue_move_is_allowed_because_the_build_resolves_it(
        self, tmp_path
    ) -> None:
        """The only real post-lock move in the live table is a VENUE move.

        Every builder resolves a game's venue through ``facts_at_lock``, so a neutralised
        stadium is the fact the build USES. A neutralised WEEK or KICKOFF is not: those
        travel into gold unresolved, which is why they refuse above.
        """
        games = _games_with_schedule_facts()
        table = tmp_path / "moves.toml"
        table.write_text(
            _PLANTED_MOVE.format(
                game_id=G_SUN,
                what_moved="venue",
                from_value="SDG00",
                to_value="CIN00",
                announced="2023-12-17T12:00:00+00:00",
                lock="2023-12-16T23:00:00+00:00",
                verdict="post_lock",
            ),
            encoding="utf-8",
        )
        assert (
            InformationTimeGate().check_games(games, table_path=table)
            is SourceCheckState.CHECKED
        )

    def test_the_real_move_table_and_real_silver_games_pass(self) -> None:
        """The production corpus: the one post-lock move is a venue move, and it is fine."""
        games = pd.read_parquet(REPO_ROOT / "data" / "silver" / "games.parquet")
        assert len(games) > 6000
        assert InformationTimeGate().check_games(games) is SourceCheckState.CHECKED
