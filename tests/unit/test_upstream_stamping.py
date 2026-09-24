"""Every captured row carries its SOURCE'S OWN information time beside our capture instant.

D33.2-18 asks for two independent proofs, either sufficient, corroborating each other:

* odds -- the bookmaker's ``last_update`` and the latest market-level ``last_update``
  (``market_last_update``), beside the capture instant ``created_at``;
* weather -- the pinned model run's ``model_run_available_at`` (``meta.json``), beside the
  capture instant ``forecast_time`` (asserted in ``test_openmeteo_model_pin.py``);
* nflverse -- the release asset's ``updated_at`` (``upstream_captured_at``, Plan 33.2-15).

A missing upstream stamp is NULL or a named refusal, NEVER our own clock. The AST scan at the
bottom of this module carries that no-substitution claim (T-33.2-27-02), with a planted
violation proving it can fire.

PLAN 33.2-02'S DEFERRED ITEM, RULED HERE: ``OddsSchema.validate_timestamps`` used to RELABEL a
naive ``snapshot_ts`` / ``last_update`` / ``created_at`` as UTC -- a 4-5 hour error on an Eastern
wall-clock value, silently. It now REFUSES naive, as ``OddsTimelineSchema`` already did, and both
production writers of ``OddsSchema`` are proved to hand it only aware-or-``None`` values.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data.quality_gates import validate_bronze_to_silver
from data.schemas import OddsSchema

GAME_ID = "2026_W04_DAL@PHI"
LOCK = datetime(2026, 9, 26, 22, 0, tzinfo=UTC)
CAPTURED_AT = datetime(2026, 9, 26, 21, 30, tzinfo=UTC)

TIME_FIELDS: tuple[str, ...] = ("snapshot_ts", "last_update", "created_at")
NAIVE_DATETIME = datetime(2026, 9, 26, 18, 0)
NAIVE_STRING = "2026-09-26 18:00:00"
AWARE_STRING = "2026-09-26T18:00:00-04:00"


def _base_row() -> dict[str, Any]:
    return {
        "game_id": GAME_ID,
        "snapshot_ts": LOCK,
        "sportsbook": "draftkings",
        "last_update": CAPTURED_AT,
        "created_at": CAPTURED_AT,
    }


def _verdict(**override: Any) -> str:
    try:
        OddsSchema(**{**_base_row(), **override})
    except ValueError:
        return "REFUSED"
    return "ACCEPTED"


def probe_odds_schema_naive() -> dict[str, dict[str, str]]:
    """The live ``OddsSchema``'s verdict on naive, aware and ``None`` timestamps.

    Computed against the REAL class, so a verify that reads it reads a ruling, not a restated
    expectation. ``none`` is keyed by ``last_update`` alone, the one nullable time field.
    """
    return {
        "naive_datetime": {f: _verdict(**{f: NAIVE_DATETIME}) for f in TIME_FIELDS},
        "naive_string": {f: _verdict(**{f: NAIVE_STRING}) for f in TIME_FIELDS},
        "aware": {f: _verdict(**{f: AWARE_STRING}) for f in TIME_FIELDS},
        "none": {"last_update": _verdict(last_update=None)},
    }


def _pre_revision_relabel(value: Any) -> Any:
    """The validator body ``OddsSchema.validate_timestamps`` had before Plan 33.2-27, verbatim.

    Copied from ``data/schemas.py`` at ``7039156`` so the CONTROL can show the refusal is new
    behaviour: this body ACCEPTS a naive value by attaching UTC to it.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass
    if isinstance(value, str):
        return pd.to_datetime(value)
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


class TestOddsSchemaRefusesNaive:
    """Naive refused on all three fields; aware accepted; ``None`` stays ``None``."""

    @pytest.mark.parametrize("field", TIME_FIELDS)
    def test_a_naive_datetime_is_refused(self, field: str):
        with pytest.raises(ValueError, match=field):
            OddsSchema(**{**_base_row(), field: NAIVE_DATETIME})

    @pytest.mark.parametrize("field", TIME_FIELDS)
    def test_a_naive_parsing_string_is_refused(self, field: str):
        with pytest.raises(ValueError, match=field):
            OddsSchema(**{**_base_row(), field: NAIVE_STRING})

    @pytest.mark.parametrize("field", TIME_FIELDS)
    def test_an_offset_bearing_string_and_an_aware_datetime_pass(self, field: str):
        OddsSchema(**{**_base_row(), field: AWARE_STRING})
        OddsSchema(**{**_base_row(), field: CAPTURED_AT})

    @pytest.mark.parametrize("missing", [None, float("nan"), pd.NaT])
    def test_a_missing_last_update_stays_none(self, missing: Any):
        assert OddsSchema(**{**_base_row(), "last_update": missing}).last_update is None

    @pytest.mark.parametrize("value", [NAIVE_DATETIME, NAIVE_STRING])
    def test_control_the_pre_revision_relabel_accepted_the_same_input(self, value: Any):
        relabelled = pd.Timestamp(_pre_revision_relabel(value))
        assert relabelled is not None
        if isinstance(value, datetime):
            assert relabelled.tzinfo is not None, (
                "the old body attached UTC to a naive value"
            )
        else:
            assert relabelled.tzinfo is None, (
                "the old body parsed the naive string to a naive Timestamp and Pydantic then "
                "accepted it"
            )

    def test_the_probe_reports_the_ruling(self):
        assert probe_odds_schema_naive() == {
            "naive_datetime": dict.fromkeys(TIME_FIELDS, "REFUSED"),
            "naive_string": dict.fromkeys(TIME_FIELDS, "REFUSED"),
            "aware": dict.fromkeys(TIME_FIELDS, "ACCEPTED"),
            "none": {"last_update": "ACCEPTED"},
        }

    def test_the_timeline_docstring_no_longer_documents_a_divergence(self):
        from data.schemas import OddsTimelineSchema

        doc = OddsTimelineSchema.__doc__ or ""
        assert "silently attaches UTC" not in doc


def _assert_aware_or_none(frame: pd.DataFrame, columns: tuple[str, ...]) -> int:
    checked = 0
    for column in columns:
        for value in frame[column]:
            if value is None or pd.isna(value):
                continue
            assert pd.Timestamp(value).tzinfo is not None, (
                f"{column} {value!r} is naive"
            )
            checked += 1
    return checked


class TestBothProductionWritersHandTheSchemaAwareValues:
    """The live capture and the historical replay each emit only aware-or-None times."""

    def test_the_live_capture_row_construction(self):
        from scripts.ingest_odds import OddsDataIngester

        ingester = OddsDataIngester.__new__(OddsDataIngester)
        schedule = pd.DataFrame(
            [
                {
                    "game_id": GAME_ID,
                    "season": 2026,
                    "week": 4,
                    "home_team": "PHI",
                    "away_team": "DAL",
                    "kickoff_et": pd.Timestamp("2026-09-27T17:00:00Z"),
                }
            ]
        )
        payload = [
            {
                "home_team": "Philadelphia Eagles",
                "away_team": "Dallas Cowboys",
                "commence_time": "2026-09-27T17:00:00Z",
                "bookmakers": [
                    {
                        "key": "draftkings",
                        "last_update": "2026-09-26T20:00:00Z",
                        "markets": [
                            {
                                "key": "totals",
                                "last_update": "2026-09-26T19:55:00Z",
                                "outcomes": [
                                    {"name": "Over", "price": -110, "point": 44.5},
                                    {"name": "Under", "price": -110, "point": 44.5},
                                ],
                            }
                        ],
                    },
                    {"key": "fanduel", "markets": []},
                ],
            }
        ]
        frame = ingester.transform_odds_data(
            payload, schedule=schedule, locks={GAME_ID: LOCK}, captured_at=CAPTURED_AT
        )
        columns = ("snapshot_ts", "last_update", "market_last_update", "created_at")
        assert _assert_aware_or_none(frame, columns) > 0
        validated = ingester.validate_odds_data(frame)
        assert len(validated) == len(frame) == 2, "a row was refused by the schema"

    def test_the_historical_replay_row_construction(self):
        from scripts.ingest_historical_odds import transform_nfl_odds_with_counts

        schedule = pd.DataFrame(
            [
                {
                    "season": 2024,
                    "week": 1,
                    "gameday": "2024-09-08",
                    "gametime": "20:20",
                    "home_team": "KC",
                    "away_team": "BAL",
                    "game_type": "REG",
                    "spread_line": -3.0,
                    "total_line": 46.5,
                    "home_moneyline": -150,
                    "away_moneyline": 130,
                    "home_spread_odds": -110,
                    "away_spread_odds": -110,
                    "over_odds": -110,
                    "under_odds": -110,
                }
            ]
        )
        report = transform_nfl_odds_with_counts(schedule)
        assert _assert_aware_or_none(report.odds, TIME_FIELDS) > 0
        validated = validate_bronze_to_silver(report.odds, OddsSchema)
        assert len(validated) == 1


class TestOddsUpstreamStamps:
    """The bookmaker's own time is kept; an absent one is NULL, never our capture instant."""

    def test_absent_upstream_times_are_null_not_the_capture_instant(self):
        from scripts.ingest_odds import OddsDataIngester

        ingester = OddsDataIngester.__new__(OddsDataIngester)
        ingester._unmatched = []
        ingester._kickoff_disagreements = []
        schedule = pd.DataFrame(
            [
                {
                    "game_id": GAME_ID,
                    "season": 2026,
                    "week": 4,
                    "home_team": "PHI",
                    "away_team": "DAL",
                    "kickoff_et": pd.Timestamp("2026-09-27T17:00:00Z"),
                }
            ]
        )
        rows = ingester._process_game_odds(
            {
                "home_team": "Philadelphia Eagles",
                "away_team": "Dallas Cowboys",
                "bookmakers": [{"key": "draftkings", "markets": []}],
            },
            schedule=schedule,
            locks={GAME_ID: LOCK},
            captured_at=CAPTURED_AT,
        )
        assert rows[0]["last_update"] is None
        assert rows[0]["market_last_update"] is None
        assert rows[0]["created_at"] == CAPTURED_AT


class TestNflverseRowsCarryTheReleaseStamp:
    """The snap ingest stamps every row with the asset's ``updated_at``; a failure refuses."""

    def _raw(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "game_id": "2026_03_DAL_PHI",
                    "pfr_player_id": "AbcdXy00",
                    "player": "A Player",
                    "position": "WR",
                    "team": "PHI",
                    "opponent": "DAL",
                    "season": 2026,
                    "week": 3,
                    "game_type": "REG",
                    "offense_snaps": 50.0,
                    "offense_pct": 0.8,
                    "defense_snaps": 0.0,
                    "defense_pct": 0.0,
                    "st_snaps": 2.0,
                    "st_pct": 0.1,
                }
            ]
        )

    def test_every_row_carries_the_aware_release_stamp(self, tmp_path: Path):
        from scripts.ingest_snaps import ingest_snaps_season

        stamp = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
        validated = ingest_snaps_season(
            2026,
            loader=lambda _season: self._raw(),
            stamp_reader=lambda _asset, fresh=False: stamp,
            base_path=tmp_path,
        )
        assert set(pd.to_datetime(validated["upstream_captured_at"], utc=True)) == {
            pd.Timestamp(stamp)
        }

    def test_an_unreadable_stamp_refuses_and_writes_nothing(self, tmp_path: Path):
        from data.upstream_asset_stamp import UpstreamStampUnavailable
        from scripts.ingest_snaps import ingest_snaps_season

        def _refuse(_asset, fresh=False):
            raise UpstreamStampUnavailable("rate limited")

        with pytest.raises(UpstreamStampUnavailable):
            ingest_snaps_season(
                2026,
                loader=lambda _season: self._raw(),
                stamp_reader=_refuse,
                base_path=tmp_path,
            )
        assert not (tmp_path / "silver").exists()


# ---------------------------------------------------------------------------
# The no-substitution scan (T-33.2-27-02)
# ---------------------------------------------------------------------------

#: Every column that carries a SOURCE'S OWN information time.
UPSTREAM_TIME_FIELDS: frozenset[str] = frozenset(
    {
        "last_update",
        "market_last_update",
        "model_run_available_at",
        "upstream_captured_at",
    }
)

#: The capture modules whose functions write rows.
CAPTURE_MODULES: tuple[str, ...] = (
    "scripts.ingest_odds",
    "scripts.ingest_weather",
    "scripts.ingest_snaps",
    "scripts.ingest_injuries",
)

#: Calls that read OUR clock.
_CLOCK_ATTRS = frozenset({"now", "utcnow", "today"})
_CLOCK_NAMES = frozenset({"_observe_capture_instant"})


def _is_clock_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in _CLOCK_ATTRS:
        return True
    return isinstance(func, ast.Name) and func.id in _CLOCK_NAMES


def _derives_from_clock(expr: ast.AST | None, tainted: set[str]) -> bool:
    if expr is None:
        return False
    return any(
        _is_clock_call(node) or (isinstance(node, ast.Name) and node.id in tainted)
        for node in ast.walk(expr)
    )


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    constants[target.id] = node.value.value
    return constants


def _field_name(node: ast.AST | None, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def substitution_violations(
    function: ast.FunctionDef | ast.AsyncFunctionDef, constants: dict[str, str]
) -> list[str]:
    """Every place *function* puts a clock-derived value into an upstream-time field."""
    tainted: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(function):
            if isinstance(node, ast.Assign | ast.AnnAssign) and _derives_from_clock(
                node.value, tainted
            ):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in tainted:
                        tainted.add(target.id)
                        changed = True

    found: list[str] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if _field_name(
                    key, constants
                ) in UPSTREAM_TIME_FIELDS and _derives_from_clock(value, tainted):
                    found.append(f"{function.name}:{node.lineno} dict key")
        elif isinstance(node, ast.keyword):
            if node.arg in UPSTREAM_TIME_FIELDS and _derives_from_clock(
                node.value, tainted
            ):
                found.append(f"{function.name}:{node.value.lineno} keyword {node.arg}")
        elif isinstance(node, ast.Assign) and _derives_from_clock(node.value, tainted):
            for target in node.targets:
                name = None
                if isinstance(target, ast.Subscript):
                    name = _field_name(target.slice, constants)
                elif isinstance(target, ast.Name):
                    name = target.id
                if name in UPSTREAM_TIME_FIELDS:
                    found.append(f"{function.name}:{node.lineno} assignment {name}")
    return found


def scan_source(source: str) -> tuple[int, list[str]]:
    """``(function nodes walked, violations)`` over *source*, walked as function nodes."""
    tree = ast.parse(source)
    constants = _module_string_constants(tree)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    violations = [v for fn in functions for v in substitution_violations(fn, constants)]
    return len(functions), violations


class TestNoCapturePathSubstitutesOurClock:
    """T-33.2-27-02: no capture function writes a clock value into an upstream-time field."""

    @pytest.mark.parametrize("module_name", CAPTURE_MODULES)
    def test_the_capture_module_has_no_substitution(self, module_name: str):
        import importlib

        source = inspect.getsource(importlib.import_module(module_name))
        walked, violations = scan_source(source)
        assert walked > 0, f"{module_name}: the scan walked no function"
        assert violations == [], f"{module_name}: {violations}"

    @pytest.mark.parametrize(
        "planted",
        [
            "def capture():\n    now = datetime.now(UTC)\n    return {'last_update': now}\n",
            "def capture(df):\n    df['market_last_update'] = datetime.now(UTC)\n",
            "def capture():\n"
            "    t = _observe_capture_instant()\n"
            "    s = t\n"
            "    return Row(model_run_available_at=s)\n",
            "COLUMN = 'upstream_captured_at'\n"
            "def capture(df):\n    df[COLUMN] = pd.Timestamp.now(tz='UTC')\n",
        ],
    )
    def test_control_a_planted_substitution_is_caught(self, planted: str):
        _, violations = scan_source(planted)
        assert violations, f"the scan missed a planted substitution:\n{planted}"

    def test_control_the_capture_instant_itself_is_not_a_violation(self):
        """No false positive: stamping OUR clock into the capture column is correct."""
        clean = (
            "def capture():\n"
            "    captured_at = datetime.now(UTC)\n"
            "    return {'created_at': captured_at, 'forecast_time': captured_at}\n"
        )
        assert scan_source(clean)[1] == []
