"""The 2025 slice is a SEASON-COLUMN slice, never a calendar-year one (SPEC R2 adjacency).

TEST CLASS: **plain unit test**. Every frame is synthesized in memory and written under
``tmp_path``; the module reads nothing under ``data/`` and passes on a fresh checkout.

WHY THIS EDGE IS PINNED AS A STRUCTURAL FACT RATHER THAN LEFT AS A CONVENTION.
Phase 31 is permitted to rebuild exactly ONE slice of gold: the 2025 season. Everything
else -- the 2021-2024 window the binding gate measures on -- must come back BYTE-IDENTICAL,
and that byte-identity is the whole control (SPEC R2). An NFL season straddles two calendar
years: the 2024 season's playoffs are played in January and February of calendar 2025. So a
CALENDAR-YEAR reading of "the 2025 slice" would silently pull 2024-season playoff rows into
the rebuilt slice, move them, and then report the 2021-2024 window as unmoved -- turning the
R2 guarantee into a false negative that no downstream check could catch, because the moved
rows would have been filed under the season that was allowed to move.

Two assertions carry that:

1. A BEHAVIOURAL one. A 2024-season game with a January-2025 kickoff lands in the 2024
   fingerprint bucket, proven by digest equality against the season-2024 subset and digest
   INEQUALITY against the calendar-2024 subset -- so the assertion discriminates between
   the two readings rather than merely being satisfied by one.
2. A STRUCTURAL one. An AST scan of ``scripts/fingerprint_gold.py`` (the per-season
   grouping) and ``scripts/build_features.py`` (the ``--season`` scoping path) proves that
   every season predicate in either module reads the ``season`` column and nothing else --
   never a kickoff, date or timestamp column. One definition, two consumers.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from scripts.fingerprint_gold import GOLD_MATRICES, fingerprint_gold, fingerprint_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
_THIS_MODULE = Path(__file__)

# The two modules that read a season predicate: the fingerprint's per-season grouping and
# the build's --season scoping. Named here so a failure says WHICH module it scanned.
SCANNED_MODULES = (
    REPO_ROOT / "scripts" / "fingerprint_gold.py",
    REPO_ROOT / "scripts" / "build_features.py",
)

# The ONLY column names a season predicate may read. ``target_season`` is the same fact
# under the name the team-form and Elo tables give it.
SEASON_COLUMNS = frozenset({"season", "target_season"})

# Names a season predicate must NEVER read. Each is a KICKOFF or a CLOCK, and each would
# express the calendar-year reading this module exists to rule out.
FORBIDDEN_SUBSTRINGS = (
    "kickoff",
    "gameday",
    "game_date",
    "gametime",
    "start_time",
    "timestamp",
    "datetime",
)


# ---------------------------------------------------------------------------
# The synthetic straddling frame
# ---------------------------------------------------------------------------


def _straddling_frame() -> pd.DataFrame:
    """Return a gold-shaped frame whose 2024 season straddles two calendar years.

    ``2024_W22_KC@BUF`` is a 2024-SEASON conference championship played in January of
    calendar 2025 -- the exact row a calendar-year reading would misfile into the 2025
    slice.
    """
    return pd.DataFrame(
        {
            "game_id": [
                "2024_W01_KC@BAL",
                "2024_W22_KC@BUF",
                "2025_W01_PHI@DAL",
            ],
            "season": [2024, 2024, 2025],
            "week": [1, 22, 1],
            "kickoff": pd.to_datetime(
                ["2024-09-05T20:20:00", "2025-01-26T15:00:00", "2025-09-04T20:20:00"]
            ),
            "home_rest_days": [7.0, 14.0, 7.0],
        }
    )


def _write_gold(root: Path, frame: pd.DataFrame) -> Path:
    """Write *frame* as all three gold matrices under *root*, and return *root*."""
    gold = root / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    for matrix in GOLD_MATRICES:
        frame.to_parquet(gold / f"{matrix}.parquet", engine="pyarrow", index=False)
    return root


def _rows_where(frame: pd.DataFrame, mask) -> pd.DataFrame:
    """Return the rows of *frame* selected by *mask*, typed as a frame."""
    return cast("pd.DataFrame", frame[mask])


def _digest(frame: pd.DataFrame, column: str, season: str) -> str:
    """Return *column*'s per-season digest for *season* under the module's own rule."""
    return fingerprint_matrix(frame)["columns"][column][season]


# ---------------------------------------------------------------------------
# The behavioural assertion
# ---------------------------------------------------------------------------


class TestTheSeasonBucketIsDefinedByTheSeasonColumn:
    """A January-2025 kickoff belongs to the 2024 slice because its ``season`` is 2024."""

    def test_the_january_kickoff_row_lands_in_the_2024_bucket(self, tmp_path: Path):
        frame = _straddling_frame()
        fingerprint = fingerprint_gold(base_path=_write_gold(tmp_path, frame))

        matrix = fingerprint["features_wp"]
        assert matrix["rows_per_season"] == {"2024": 2, "2025": 1}, (
            "the January-2025 kickoff of a 2024-SEASON game must be counted in 2024"
        )

        season_2024 = _rows_where(frame, frame["season"] == 2024)
        assert matrix["columns"]["home_rest_days"]["2024"] == _digest(
            season_2024, "home_rest_days", "2024"
        ), "the 2024 bucket must hash exactly the two season-2024 rows"

    def test_a_calendar_year_reading_would_produce_a_DIFFERENT_2024_bucket(
        self, tmp_path: Path
    ):
        """The discriminating half: the two readings are not the same set of rows."""
        frame = _straddling_frame()
        fingerprint = fingerprint_gold(base_path=_write_gold(tmp_path, frame))

        calendar_2024 = _rows_where(frame, frame["kickoff"].dt.year == 2024)
        assert len(calendar_2024) == 1, (
            "the fixture must actually straddle, or this assertion proves nothing"
        )

        assert fingerprint["features_wp"]["columns"]["home_rest_days"][
            "2024"
        ] != _digest(calendar_2024, "home_rest_days", "2024"), (
            "the season-column bucket and the calendar-year bucket must differ, "
            "otherwise this fixture cannot tell the two readings apart"
        )

    def test_the_2025_bucket_holds_only_season_2025_rows(self, tmp_path: Path):
        frame = _straddling_frame()
        fingerprint = fingerprint_gold(base_path=_write_gold(tmp_path, frame))

        season_2025 = _rows_where(frame, frame["season"] == 2025)
        assert fingerprint["features_wp"]["columns"]["home_rest_days"][
            "2025"
        ] == _digest(season_2025, "home_rest_days", "2025")
        assert fingerprint["features_wp"]["seasons"] == [2024, 2025]

    def test_the_grouping_ignores_the_kickoff_column_entirely(self, tmp_path: Path):
        """Moving every kickoff into another calendar year changes no season bucket."""
        frame = _straddling_frame()
        shifted = frame.copy()
        shifted["kickoff"] = shifted["kickoff"] + pd.DateOffset(years=3)

        original = fingerprint_gold(base_path=_write_gold(tmp_path / "a", frame))
        moved = fingerprint_gold(base_path=_write_gold(tmp_path / "b", shifted))

        assert (
            original["features_wp"]["rows_per_season"]
            == (moved["features_wp"]["rows_per_season"])
        )
        assert (
            original["features_wp"]["columns"]["home_rest_days"]
            == moved["features_wp"]["columns"]["home_rest_days"]
        )


# ---------------------------------------------------------------------------
# The structural assertion
# ---------------------------------------------------------------------------


def _subscript_keys(node: ast.AST) -> set[str]:
    """Return every constant string subscript key appearing anywhere inside *node*."""
    return {
        child.slice.value
        for child in ast.walk(node)
        if isinstance(child, ast.Subscript)
        and isinstance(child.slice, ast.Constant)
        and isinstance(child.slice.value, str)
    }


def _is_season_operand(node: ast.AST) -> bool:
    """True when *node* is a season scalar or a ``frame["season"]``-shaped subscript."""
    if isinstance(node, ast.Name) and node.id in SEASON_COLUMNS:
        return True
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
        and node.slice.value in SEASON_COLUMNS
    )


def _season_predicates_in_source(source: str) -> list[tuple[int, set[str]]]:
    """Return ``(lineno, subscript keys)`` for every season predicate in *source*.

    A season predicate is a comparison with a season scalar or a season-column
    subscript on one side -- which is every place either module decides what belongs
    to a season.
    """
    tree = ast.parse(source)
    predicates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if any(_is_season_operand(operand) for operand in operands):
            predicates.append((node.lineno, _subscript_keys(node)))
    return sorted(predicates)


def _season_predicates(path: Path) -> list[tuple[int, set[str]]]:
    """Return every season predicate in the module at *path*."""
    return _season_predicates_in_source(path.read_text(encoding="utf-8"))


class TestNoSeasonPredicateReadsAKickoffOrTimestampColumn:
    """One definition, two consumers: the grouping key is the ``season`` column."""

    @pytest.mark.parametrize("path", SCANNED_MODULES, ids=lambda p: p.name)
    def test_the_scan_is_not_vacuous(self, path: Path):
        """A scan that finds nothing would pass every assertion below for free."""
        predicates = _season_predicates(path)

        assert predicates, (
            f"the AST scan of {path.relative_to(REPO_ROOT).as_posix()} found ZERO "
            "season predicates. Either the module was restructured or the scan is "
            "broken; in both cases the assertions below prove nothing."
        )
        assert any(keys & SEASON_COLUMNS for _, keys in predicates), (
            f"{path.relative_to(REPO_ROOT).as_posix()} carries season predicates but "
            "none of them reads a season COLUMN, so the grouping key is unproven"
        )

    @pytest.mark.parametrize("path", SCANNED_MODULES, ids=lambda p: p.name)
    def test_every_season_predicate_reads_only_a_season_column(self, path: Path):
        module = path.relative_to(REPO_ROOT).as_posix()
        offenders = [
            (lineno, sorted(keys - SEASON_COLUMNS))
            for lineno, keys in _season_predicates(path)
            if not keys <= SEASON_COLUMNS
        ]

        assert offenders == [], (
            f"{module} decides season membership from a column outside "
            f"{sorted(SEASON_COLUMNS)}: {offenders}. The 2025 slice must be a "
            "season-column slice, or the R2 byte-identity guarantee on 2021-2024 can "
            "be defeated by a calendar-year reading."
        )

    @pytest.mark.parametrize("path", SCANNED_MODULES, ids=lambda p: p.name)
    def test_no_season_predicate_names_a_kickoff_or_timestamp_column(self, path: Path):
        module = path.relative_to(REPO_ROOT).as_posix()
        offenders = [
            (lineno, sorted(key for key in keys if _is_forbidden(key)))
            for lineno, keys in _season_predicates(path)
            if any(_is_forbidden(key) for key in keys)
        ]

        assert offenders == [], (
            f"{module} reads a kickoff or clock column inside a season predicate: "
            f"{offenders}. A 2024-season game kicks off in calendar 2025, so that is "
            "the calendar-year reading in disguise."
        )


def _is_forbidden(key: str) -> bool:
    """True when a column name names a kickoff or a clock rather than a season."""
    lowered = key.lower()
    return any(substring in lowered for substring in FORBIDDEN_SUBSTRINGS)


class TestTheScannerCanActuallyFail:
    """A guard that has never been shown to fail is a guard nobody has tested.

    Both assertions above pass against the live modules, which is the outcome the
    phase wants -- and is also exactly what a broken scanner would produce. These
    cases run the same scanner over SYNTHETIC sources carrying the calendar-year
    reading, so the passing verdict on the live modules carries information.
    """

    _CALENDAR_YEAR_SOURCE = (
        "def slice_season(df, season):\n"
        "    return df[df['kickoff'].dt.year == season]\n"
    )

    def test_it_flags_a_predicate_that_reads_a_kickoff_column(self):
        predicates = _season_predicates_in_source(self._CALENDAR_YEAR_SOURCE)

        assert predicates, "the synthetic calendar-year predicate was not even found"
        keys = set().union(*(keys for _, keys in predicates))
        assert not keys <= SEASON_COLUMNS, (
            "the scanner accepted a predicate that slices on kickoff.dt.year"
        )
        assert any(_is_forbidden(key) for key in keys)

    def test_it_flags_a_predicate_that_reads_a_feature_timestamp_column(self):
        source = (
            "def slice_season(df, season):\n"
            "    return df[df['feature_timestamp'].dt.year == season]\n"
        )
        keys = set().union(*(keys for _, keys in _season_predicates_in_source(source)))

        assert any(_is_forbidden(key) for key in keys)

    def test_it_accepts_the_season_column_reading(self):
        source = (
            "def slice_season(df, season):\n    return df[df['season'] == season]\n"
        )
        predicates = _season_predicates_in_source(source)

        assert predicates
        keys = set().union(*(keys for _, keys in predicates))
        assert keys <= SEASON_COLUMNS
        assert not any(_is_forbidden(key) for key in keys)


class TestThisModuleReadsNothingUnderData:
    """``data/`` is a hard boundary; a unit test that reads live gold is not a unit test."""

    def test_every_fingerprint_gold_call_supplies_an_explicit_base_path(self):
        """Without ``base_path``, ``fingerprint_gold`` reads the configured data root."""
        tree = ast.parse(_THIS_MODULE.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "fingerprint_gold"
        ]

        assert calls, "the scan found no fingerprint_gold call, so it proves nothing"
        for call in calls:
            assert any(keyword.arg == "base_path" for keyword in call.keywords), (
                f"line {call.lineno}: fingerprint_gold must be called with an explicit "
                "base_path, or it falls back to the configured data/ root"
            )
