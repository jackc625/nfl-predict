"""The shared 2026 fixtures match what PRODUCTION ingestion actually produces.

WHY THIS MODULE EXISTS (the Codex MEDIUM this plan folded in)
--------------------------------------------------------------
A captured fixture validated only against its own static contents is self-consistent
fiction. It keeps passing while ``scripts/ingest_games.py``'s transform moves underneath
it, and the drift is discovered by whichever plan first uses the fixture against real
production output -- three plans and two weeks later, as a confusing failure in code that
did nothing wrong.

So the assertions here are made against the TRANSFORMED frame, and the expected column set
is DERIVED FROM THE TRANSFORM'S OWN SOURCE rather than transcribed. A column added to
``transform_schedule_data`` and not to the fixture's expectation is then a failure at the
moment it is added, which is the only moment anybody can act on it cheaply.

IT ALSO PROVES THE FIXTURE WRITES NOTHING
-----------------------------------------
``tests/fixtures/season_2026.py`` reads a parquet INSIDE the gitignored production store.
Plan 33-01's autouse content guard is the runtime backstop; the AST scan below is the
structural one, and it is here rather than in the fixture module because a module that
audits itself can be edited to stop.

ONE PRE-EXISTING FINDING IS PINNED HERE RATHER THAN FIXED
---------------------------------------------------------
The production transform reads ``row.get("neutral_site", False)``, and the nflverse
schedule feed has NO ``neutral_site`` column -- it carries ``location``. So all 272
transformed rows read ``neutral_site == False`` even though eight of them are the 2026
international games. That is a real defect in ingestion, it PREDATES this phase, and
COLD-09 (a later plan) is what owns it. It is pinned as an asserted observation so the
plan that fixes it sees this test go red and has to say so, rather than discovering the
behaviour by accident.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests import phase33_state
from tests.fixtures import season_2026

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_MODULE_PATH = REPO_ROOT / "tests" / "fixtures" / "season_2026.py"
INGEST_MODULE_PATH = REPO_ROOT / "scripts" / "ingest_games.py"

# Call names that WRITE. A fixture module making one of these would be the COLD-05
# violation class, authored by the module whose job is to make COLD-05 checkable.
WRITE_SHAPED_CALLS = frozenset(
    {
        "open",
        "to_parquet",
        "to_csv",
        "to_json",
        "to_sql",
        "write_text",
        "write_bytes",
        "write",
        "writelines",
        "save_dataframe",
        "save_bronze_snapshot",
        "upsert_silver",
        "mkdir",
        "touch",
        "unlink",
    }
)

PLAN_33_06_PENDING_SKIP = (
    "the three identity columns Plan 33-06 adds to the ingestion transform are not "
    "recorded yet: tests/phase33_state.PLAN_33_06_IDENTITY_COLUMNS is absent because "
    "that plan has not landed. This is a control that did NOT run on this checkout, not "
    "a control that passed."
)

CAPTURE_ABSENT_SKIP_PREFIX = "the captured 2026 schedule is not present at"


def _require_capture() -> None:
    """Skip with an evidence-backed reason when the gitignored capture is absent."""
    if not season_2026.CAPTURED_SCHEDULE_PATH.is_file():
        pytest.skip(
            f"{CAPTURE_ABSENT_SKIP_PREFIX} "
            f"{season_2026.CAPTURED_SCHEDULE_PATH.as_posix()}; it lives under the "
            "gitignored production store and does not travel with the repository."
        )


# ---------------------------------------------------------------------------
# The fixture module writes nothing.
# ---------------------------------------------------------------------------


def write_shaped_calls(path: Path) -> list[str]:
    """Every write-shaped call in *path*, by line and name.

    BOTH call shapes are examined -- ``frame.to_parquet(...)`` (an attribute) and
    ``open(...)`` (a bare name). An attribute-only scan is the obvious version and it
    misses the single most direct way to write a file.

    Args:
        path: A Python source file.

    Returns:
        Human-readable hits.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        if name in WRITE_SHAPED_CALLS:
            hits.append(f"{path.as_posix()}:{node.lineno}: calls {name}()")
    return sorted(hits)


def test_the_write_scan_visits_a_module_that_makes_calls() -> None:
    """Anti-vacuity: the no-write assertion is over a module that really does call things."""
    tree = ast.parse(
        FIXTURE_MODULE_PATH.read_text(encoding="utf-8"),
        filename=str(FIXTURE_MODULE_PATH),
    )
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert len(calls) >= 10, (
        f"tests/fixtures/season_2026.py makes only {len(calls)} calls -- too few for the "
        "no-write scan below to mean anything."
    )


def test_the_fixture_module_makes_no_write_shaped_call() -> None:
    """T-33-11: a fixture that wrote the captured bronze schedule is the violation class."""
    hits = write_shaped_calls(FIXTURE_MODULE_PATH)
    assert not hits, (
        "tests/fixtures/season_2026.py makes write-shaped call(s):\n"
        + "\n".join(f"  - {line}" for line in hits)
        + "\n\nThe module READS a parquet inside the gitignored production store. A write "
        "there moves bytes every later digest in this phase is compared against."
    )


def test_the_write_scan_flags_a_planted_write(tmp_path: Path) -> None:
    """Fail-closed control: a module that writes IS reported, in both call shapes."""
    planted = tmp_path / "planted_writer.py"
    planted.write_text(
        "\n".join(
            [
                "def persist(frame, path):",
                "    frame.to_parquet(path)",
                "    handle = open(path, 'w')",
                "    return handle",
                "",
            ]
        ),
        encoding="utf-8",
    )
    joined = "\n".join(write_shaped_calls(planted))
    assert "to_parquet()" in joined, joined
    assert "open()" in joined, joined


# ---------------------------------------------------------------------------
# The fixture matches the PRODUCTION transform, not itself.
# ---------------------------------------------------------------------------


def production_transform_columns() -> list[str]:
    """The column names ``transform_schedule_data`` emits, read from its own source.

    Derived rather than transcribed: a transcribed list is a second list, and a second
    list drifts. This one cannot, because it IS the transform's ``game_record`` literal.

    Returns:
        The keys of the ``game_record`` dict, in source order.

    Raises:
        AssertionError: when the literal cannot be located, which would make every
            assertion built on it vacuous.
    """
    tree = ast.parse(
        INGEST_MODULE_PATH.read_text(encoding="utf-8"), filename=str(INGEST_MODULE_PATH)
    )
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.FunctionDef)
            or node.name != "transform_schedule_data"
        ):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Assign) or not isinstance(
                inner.value, ast.Dict
            ):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "game_record"
                for target in inner.targets
            ):
                continue
            return [
                key.value
                for key in inner.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
    raise AssertionError(
        "could not locate the `game_record` dict literal inside "
        "scripts/ingest_games.py::transform_schedule_data. The column expectation below "
        "would then be derived from nothing and would pass vacuously."
    )


def test_the_transform_column_derivation_is_not_vacuous() -> None:
    """The derived list is real: the transform emits a substantial record."""
    columns = production_transform_columns()
    assert len(columns) >= 10, columns
    assert "game_id" in columns
    assert "kickoff_et" in columns


def test_the_transformed_fixture_carries_every_column_production_emits() -> None:
    """The fixture is validated through the PRODUCTION transformer, not against itself.

    Asserted as SET EQUALITY in both directions: a column the transform emits and the
    fixture loses is a broken fixture, and a column the fixture carries that the transform
    does not emit is a fixture that has grown a schema of its own.
    """
    _require_capture()
    transformed = season_2026.transform_captured_schedule()
    assert set(transformed.columns) == set(production_transform_columns()), (
        "the transformed fixture's columns disagree with what "
        "scripts/ingest_games.py::transform_schedule_data emits.\n"
        f"  fixture:   {sorted(transformed.columns)}\n"
        f"  transform: {sorted(production_transform_columns())}"
    )
    assert len(transformed) == phase33_state.CAPTURED_SCHEDULE_ROWS


def test_the_transformed_fixture_has_the_dtypes_downstream_code_expects() -> None:
    """Dtypes, not just names: a column of the right name and wrong type still breaks.

    ``home_score`` / ``away_score`` / ``result`` are float rather than int because the
    capture is taken MID-WEEK-1: exactly two games are graded and the other 270 carry NA.
    That mixed shape is the cold start this whole phase is about, and asserting it here is
    what stops a later plan assuming integers because a historical season happened to
    provide them.
    """
    _require_capture()
    transformed = season_2026.transform_captured_schedule()
    dtypes = {name: str(dtype) for name, dtype in transformed.dtypes.items()}

    assert dtypes["game_id"] == "object"
    assert dtypes["season"] == "int64"
    assert dtypes["week"] == "int64"
    assert dtypes["kickoff_et"].startswith("datetime64")
    assert dtypes["home_team"] == "object"
    assert dtypes["away_team"] == "object"
    assert dtypes["venue"] == "object"
    assert dtypes["venue_roof"] == "object"
    assert dtypes["neutral_site"] == "bool"
    graded = phase33_state.CAPTURED_SCHEDULE_GRADED_GAMES
    for partial in ("home_score", "away_score", "result"):
        assert dtypes[partial] == "float64", (
            f"{partial} is {dtypes[partial]}, not float64. Most of the 2026 capture is "
            "ungraded, so the column must be nullable; an integer column would mean the "
            "NAs were filled from somewhere."
        )
        assert int(transformed[partial].notna().sum()) == graded, (
            f"{partial} is populated on {int(transformed[partial].notna().sum())} rows, "
            f"not the {graded} games graded at capture time. More means a later "
            "capture was substituted; fewer means scores were lost in transform."
        )
    assert int(transformed["home_score"].isna().sum()) == (
        phase33_state.CAPTURED_SCHEDULE_ROWS - graded
    )


def test_the_production_transform_carries_the_neutral_site_fact() -> None:
    """INVERTED BY PLAN 33-06 (COLD-09) ON 2026-09-12, AND THAT IS THE POINT.

    This test used to be called ``test_the_production_transform_DROPS_the_neutral_site
    _fact`` and it asserted the WRONG behaviour ON PURPOSE. Plan 33-02 pinned the
    pre-existing defect -- ``transform_schedule_data`` read
    ``row.get("neutral_site", False)`` against a feed that has no ``neutral_site``
    column, so all 272 rows read False including the eight the feed marks
    ``location == "Neutral"`` -- and its docstring said, in as many words: "When
    COLD-09 lands, this test goes red and that plan must say what it changed."

    COLD-09 landed. It is INVERTED rather than deleted, because a deleted test proves
    nothing about what changed while an inverted one proves exactly what changed: the
    same feed, the same eight rows, the opposite answer. The plan says so in its
    SUMMARY.

    The feed STILL has no ``neutral_site`` column -- that half of the pin is unchanged
    and still asserted. What changed is that the transform now DERIVES the fact from
    ``location`` instead of defaulting it away.
    """
    _require_capture()
    raw = season_2026.load_captured_schedule()
    transformed = season_2026.transform_captured_schedule()

    assert "neutral_site" not in raw.columns, (
        "the feed now carries a neutral_site column of its own. The COLD-09 "
        "derivation reads `location`; if the feed started supplying the fact "
        "directly, decide deliberately which source wins and record it."
    )
    assert (raw["location"] == "Neutral").sum() == (
        phase33_state.NEUTRAL_SITE_GAME_COUNT_2026
    )
    assert int(transformed["neutral_site"].sum()) == (
        phase33_state.NEUTRAL_SITE_GAME_COUNT_2026
    ), (
        "the transform reports "
        f"{int(transformed['neutral_site'].sum())} neutral-site games, not "
        f"{phase33_state.NEUTRAL_SITE_GAME_COUNT_2026}. Zero here is the ORIGINAL "
        'defect returning: `row.get("neutral_site", False)` against a feed that '
        "has no such column."
    )


def test_the_three_plan_33_06_identity_columns_reach_the_transformed_frame() -> None:
    """Guarded until Plan 33-06 lands, with its skip proven to fire below."""
    if not hasattr(phase33_state, "PLAN_33_06_IDENTITY_COLUMNS"):
        pytest.skip(PLAN_33_06_PENDING_SKIP)
    _require_capture()

    transformed = season_2026.transform_captured_schedule()
    missing = [
        column
        for column in phase33_state.PLAN_33_06_IDENTITY_COLUMNS
        if column not in transformed.columns
    ]
    assert not missing, (
        f"identity column(s) {missing!r} recorded by Plan 33-06 do not reach the "
        "TRANSFORMED frame. A builder that gate-passes while its columns never reach the "
        "output is the 28-06 trap this project has already paid for once."
    )


def test_the_plan_33_06_skip_guard_actually_fires() -> None:
    """Fail-closed control: while the constant is absent, the guard steps aside by name."""
    if hasattr(phase33_state, "PLAN_33_06_IDENTITY_COLUMNS"):
        assert len(phase33_state.PLAN_33_06_IDENTITY_COLUMNS) == 3
        return
    with pytest.raises(pytest.skip.Exception) as excinfo:
        if not hasattr(phase33_state, "PLAN_33_06_IDENTITY_COLUMNS"):
            pytest.skip(PLAN_33_06_PENDING_SKIP)
    assert str(excinfo.value) == PLAN_33_06_PENDING_SKIP


# ---------------------------------------------------------------------------
# The measured 2026 facts the fixtures assert against.
# ---------------------------------------------------------------------------


def test_the_recorded_feed_column_order_matches_the_capture() -> None:
    """The recorded column list is what lets this module import without the lake.

    It is therefore a claim about the capture, and a claim about a file has to be checked
    against the file wherever the file is available.
    """
    _require_capture()
    raw = season_2026.load_captured_schedule()
    assert tuple(raw.columns) == season_2026.CAPTURED_FEED_COLUMNS
    assert len(season_2026.CAPTURED_FEED_COLUMNS) == (
        phase33_state.CAPTURED_SCHEDULE_COLUMNS
    )


def test_the_derived_bye_weeks_reproduce_the_measured_expectation() -> None:
    """The derivation and the measurement agree for all ten weeks, week 12 included."""
    _require_capture()
    derived = season_2026.bye_teams_by_week()

    for week, expected in sorted(season_2026.BYE_TEAMS_BY_WEEK.items()):
        assert derived[week] == expected, (
            f"week {week}: derived {sorted(derived[week])!r}, measured "
            f"{sorted(expected)!r}"
        )

    assert derived[12] == frozenset(), (
        "week 12 has byes in the derivation. It is the natural NEGATIVE CONTROL -- all 32 "
        "teams play -- and it is the reason a test must not parameterise blindly over "
        "weeks 5-14."
    )
    # Every week outside 5-14 has a full slate, which is what makes the ten weeks above
    # the complete bye set rather than a sample of it.
    for week, byes in sorted(derived.items()):
        if week not in season_2026.BYE_TEAMS_BY_WEEK:
            assert byes == frozenset(), (week, sorted(byes))


def test_the_schedule_uses_exactly_the_canonical_thirty_two_teams() -> None:
    """The derivation subtracts from the CANONICAL 32, so the two sets must agree.

    If the feed spelled a team differently -- ``LAR`` for ``LA``, say -- the difference
    would show up as a phantom bye every single week, and the bye expectation above would
    have been measured on that phantom rather than catching it.
    """
    _require_capture()
    from utils.team_data import ALL_TEAMS

    raw = season_2026.load_captured_schedule()
    playing = set(raw["home_team"]) | set(raw["away_team"])
    assert playing == set(ALL_TEAMS), {
        "in_schedule_only": sorted(playing - set(ALL_TEAMS)),
        "canonical_only": sorted(set(ALL_TEAMS) - playing),
    }


def test_the_neutral_site_games_are_the_eight_international_stadiums() -> None:
    """The eight 2026 international games, by stadium id, in week order.

    MUN01 is not GER00 and RIO00 is not SAO00: the 2026 venues are NOT the ones history
    used, which is exactly the kind of fact a cold start gets wrong by reaching for a
    prior season's mapping.
    """
    _require_capture()
    neutral = season_2026.neutral_site_games()

    assert len(neutral) == phase33_state.NEUTRAL_SITE_GAME_COUNT_2026
    assert tuple(neutral.sort_values("week")["stadium_id"]) == (
        phase33_state.INTERNATIONAL_STADIUM_IDS
    )
    assert neutral["stadium_id"].notna().all()
    assert neutral["stadium"].notna().all()
    assert (neutral["stadium"].str.len() > 0).all()
    assert sorted(neutral["week"]) == [1, 3, 4, 6, 7, 9, 10, 11]


# ---------------------------------------------------------------------------
# The two CONSTRUCTED fixtures.
# ---------------------------------------------------------------------------


def test_the_week_19_fixture_is_a_postseason_frame_in_the_feed_shape() -> None:
    """Held out by the SPEC's own edge table: 2026 weeks 19-22 are unseeded."""
    frame = season_2026.WEEK_19_POSTSEASON_FIXTURE

    assert list(frame.columns) == list(season_2026.CAPTURED_FEED_COLUMNS)
    assert set(frame["week"]) == {19}
    assert set(frame["game_type"]) == {"WC"}
    assert len(frame) >= 2
    assert "weeks 19-22 are unseeded" in (
        season_2026._build_week_19_postseason_fixture.__doc__ or ""
    )


def test_the_captured_schedule_really_has_no_postseason_rows() -> None:
    """The reason the week-19 fixture is constructed, asserted rather than asserted-about.

    If the capture ever GAINS postseason rows, the constructed fixture stops being a
    held-out case and becomes a stand-in for data that exists -- which is a different
    thing, and a worse one.
    """
    _require_capture()
    raw = season_2026.load_captured_schedule()
    assert set(raw["game_type"]) == {"REG"}
    assert sorted(raw["week"].unique()) == list(range(1, 19))


def test_the_half_point_ats_frame_covers_the_cases_real_data_cannot_supply() -> None:
    """Half point, pick-em with a non-zero prediction, both tails, and a null spread."""
    frame = season_2026.build_half_point_ats_frame()

    half_point = frame[frame["market_spread"].abs() == 0.5]
    assert len(half_point) >= 1, frame.to_dict("records")

    pick_em = frame[frame["market_spread"] == 0.0]
    assert len(pick_em) >= 1
    assert (pick_em["ats_prediction"] != 0.0).all(), (
        "the pick-em row's ats_prediction is zero, so the D33-31 repair "
        "(ats_edge = ats_prediction - market_spread, including at market_spread == 0) "
        "and the old forced zero give the SAME answer there. A case that cannot "
        "distinguish the two proves nothing at the one point the behaviour changes."
    )

    assert frame["market_spread"].isna().sum() >= 1
    assert (frame["market_spread"] < -5).any()
    assert (frame["market_spread"] > 5).any()

    # The two spellings must never disagree, or which name a plan reaches for decides the
    # answer.
    mirrored = frame["market_spread"].equals(frame["spread"])
    assert mirrored, frame[["market_spread", "spread"]].to_dict("records")


def test_the_session_fixtures_hand_out_defensive_copies(
    captured_2026_schedule,
    week_19_postseason_fixture,
    half_point_ats_rows,
) -> None:
    """Session scope is only safe because each consumer gets its own frame.

    Mutating the object a session fixture returned must not reach the module-level source,
    or one consumer's in-place edit silently becomes every later consumer's input and the
    failure surfaces in whichever test ran last.
    """
    week_19_postseason_fixture.loc[:, "game_type"] = "MUTATED"
    assert set(season_2026.WEEK_19_POSTSEASON_FIXTURE["game_type"]) == {"WC"}

    half_point_ats_rows.loc[:, "market_spread"] = 99.0
    assert season_2026.build_half_point_ats_frame()["market_spread"].abs().max() < 20.0

    assert captured_2026_schedule.shape == (
        phase33_state.CAPTURED_SCHEDULE_ROWS,
        phase33_state.CAPTURED_SCHEDULE_COLUMNS,
    )
    captured_2026_schedule.loc[:, "week"] = 0
    assert sorted(season_2026.load_captured_schedule()["week"].unique()) == list(
        range(1, 19)
    )


def test_a_moved_capture_raises_by_name_rather_than_loosening(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail-closed control: a shape mismatch is a NAMED refusal, never a smaller frame.

    Every expectation in this phase was measured on 272 x 46. A loader that returned
    whatever it found would let all of them keep passing against a frame nobody looked at.
    """
    import pandas as pd

    decoy = tmp_path / "moved_capture.parquet"
    pd.DataFrame({"game_id": ["x"], "week": [1]}).to_parquet(decoy)
    monkeypatch.setattr(season_2026, "CAPTURED_SCHEDULE_PATH", decoy)

    with pytest.raises(season_2026.CapturedScheduleShapeError) as excinfo:
        season_2026.load_captured_schedule()
    assert "MOVED" in str(excinfo.value)
    assert "do NOT loosen" in str(excinfo.value)


def test_an_absent_capture_raises_by_name_rather_than_returning_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail-closed control: absence is a named refusal a fixture turns into a visible skip.

    The message must carry "not present at", which ``tests/conftest._EVIDENCE_SKIP_MARKERS``
    recognises -- that is what makes the skip say out loud that a control did not run here.
    """
    monkeypatch.setattr(
        season_2026, "CAPTURED_SCHEDULE_PATH", tmp_path / "absent.parquet"
    )
    with pytest.raises(season_2026.CapturedScheduleUnavailableError) as excinfo:
        season_2026.load_captured_schedule()

    from tests.conftest import is_evidence_backed_skip

    assert is_evidence_backed_skip(str(excinfo.value)), str(excinfo.value)
