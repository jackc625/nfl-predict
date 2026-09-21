"""The emergency-schedule-move evidence table is complete, well-formed and true to silver.

Plan 33.2-10 Task 2 (SPEC R8, D33.2-04, D33.2-21; threats T-33.2-10-01, -02, -03, -04).

What is asserted, and why:

* COVERAGE, both directions: every candidate in ``config/schedule_move_candidates.toml``
  has a row and no row names a game outside it, so neither a dropped candidate nor an
  invented one passes;
* THE VERDICT RULE: a ``pre_lock`` row has a source and an announcement at or before the
  lock (AT the lock counts as known), checked on every real row and on planted rows that
  must be refused or accepted;
* THE LOCK RULE: every ``lock_utc`` equals ``utils.game_lock.game_lock`` of the game's
  silver kickoff, recomputed here;
* TRUTH TO SILVER: for every game, the ``to_value`` of its last move in each fact family is
  what silver actually records (venue -> ``stadium_id``, date -> the ET kickoff date, week
  -> week number and date), consecutive moves in a family chain (``from`` of move k+1 is
  ``to`` of move k), and every scheduled row's facts equal silver's. So a row that
  misdescribes a move fails here, not silently in gold;
* the named rows (the closest call, Ike, Irma) exist, Ike and Irma as ``week`` rows, and the
  table header states the week-move limitation.

The rule-level checker below is the table's contract. ``features.schedule_moves`` (Task 4)
applies the same contract when it loads the table.

The real-data tests read production silver READ-ONLY and skip when it is not built.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import tomllib
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from utils.date_utils import kickoff_wall_clock_et
from utils.game_lock import game_lock

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_PATH = REPO_ROOT / "config" / "schedule_moves.toml"
MANIFEST_PATH = REPO_ROOT / "config" / "schedule_move_candidates.toml"
GAMES_PATH = REPO_ROOT / "data" / "silver" / "games.parquet"
VENUES_PATH = REPO_ROOT / "data" / "venues.json"

WHAT_MOVED_VOCABULARY: frozenset[str] = frozenset(
    {"venue", "date", "week", "scheduled_not_moved"}
)
VERDICTS: frozenset[str] = frozenset({"pre_lock", "post_lock"})
REQUIRED_FIELDS: tuple[str, ...] = (
    "game_id",
    "move_index",
    "what_moved",
    "from_value",
    "to_value",
    "announced_at_utc",
    "source_url",
    "source_title",
    "source_published_at",
    "lock_utc",
    "verdict",
    "notes",
)

CLOSEST_CALL = "2010_W14_NYG@MIN"
IKE = "2008_W10_BAL@HOU"
IRMA = "2017_W11_TB@MIA"


def parse_utc(value: str) -> datetime:
    """A ``...Z`` timestamp as an aware datetime; anything naive is refused."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"naive timestamp {value!r}")
    return parsed


def row_violations(row: dict[str, object]) -> list[str]:
    """Every way *row* breaks the table's contract (empty when it is well-formed)."""
    problems = [f"missing {f}" for f in REQUIRED_FIELDS if f not in row]
    if problems:
        return problems
    if row["what_moved"] not in WHAT_MOVED_VOCABULARY:
        problems.append(f"what_moved {row['what_moved']!r} is outside the vocabulary")
    if row["verdict"] not in VERDICTS:
        problems.append(f"verdict {row['verdict']!r} is not pre_lock or post_lock")
    if (
        row["what_moved"] == "scheduled_not_moved"
        and row["from_value"] != row["to_value"]
    ):
        problems.append("a scheduled_not_moved row must have from_value == to_value")
    if not str(row["notes"]).strip():
        problems.append("notes is empty")
    lock = parse_utc(str(row["lock_utc"]))
    if row["verdict"] == "pre_lock":
        if not str(row["source_url"]).strip():
            problems.append("pre_lock without a source_url")
        announced = str(row["announced_at_utc"]).strip()
        if not announced:
            problems.append("pre_lock without an announced_at_utc")
        elif parse_utc(announced) > lock:
            problems.append("pre_lock announced after the lock")
    return problems


def _load(path: Path, key: str) -> list[dict[str, object]]:
    return list(tomllib.loads(path.read_text(encoding="utf-8"))[key])


@pytest.fixture(scope="module")
def rows() -> list[dict[str, object]]:
    return _load(TABLE_PATH, "moves")


@pytest.fixture(scope="module")
def candidates() -> list[dict[str, object]]:
    return _load(MANIFEST_PATH, "candidates")


@pytest.fixture(scope="module")
def silver() -> pd.DataFrame:
    if not GAMES_PATH.exists():
        pytest.skip("silver games is not built on this checkout")
    return pd.read_parquet(GAMES_PATH).set_index("game_id")


def _by_game(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["game_id"]), []).append(row)
    return {
        g: sorted(rs, key=lambda r: int(str(r["move_index"])))
        for g, rs in grouped.items()
    }


def _silver_facts(silver: pd.DataFrame, game_id: str) -> dict[str, str]:
    row = silver.loc[game_id]
    et_date = kickoff_wall_clock_et(row["kickoff_et"]).date().isoformat()
    return {
        "venue": str(row["stadium_id"]),
        "date": et_date,
        "week": f"W{int(row['week']):02d} {et_date}",
    }


# ------------------------------------------------------------------------- coverage


class TestCoverage:
    def test_the_manifest_is_not_empty(self, candidates) -> None:
        assert len(candidates) > 0

    def test_every_candidate_has_a_row(self, rows, candidates) -> None:
        covered = {str(r["game_id"]) for r in rows}
        assert sorted({str(c["game_id"]) for c in candidates} - covered) == []

    def test_no_row_names_a_game_outside_the_manifest(self, rows, candidates) -> None:
        manifest = {str(c["game_id"]) for c in candidates}
        assert sorted({str(r["game_id"]) for r in rows} - manifest) == []

    def test_the_named_rows_exist(self, rows) -> None:
        by_game = _by_game(rows)
        assert CLOSEST_CALL in by_game
        assert [r["what_moved"] for r in by_game[IKE]].count("week") == 1
        assert [r["what_moved"] for r in by_game[IRMA]] == ["week"]


# ------------------------------------------------------------------- the contract


class TestEveryRowKeepsTheContract:
    def test_no_row_violates_the_contract(self, rows) -> None:
        broken = {
            f"{r.get('game_id')}#{r.get('move_index')}": row_violations(r)
            for r in rows
            if row_violations(r)
        }
        assert broken == {}

    def test_every_pre_lock_row_has_a_source_and_a_timely_announcement(
        self, rows
    ) -> None:
        bad = [
            r["game_id"]
            for r in rows
            if r["verdict"] == "pre_lock"
            and (
                not str(r["source_url"]).strip()
                or parse_utc(str(r["announced_at_utc"])) > parse_utc(str(r["lock_utc"]))
            )
        ]
        assert bad == []

    def test_every_sourced_row_states_its_publication_time(self, rows) -> None:
        missing = [
            r["game_id"]
            for r in rows
            if str(r["source_url"]).strip()
            and not (
                str(r["source_published_at"]).strip()
                and str(r["announced_at_utc"]).strip()
            )
        ]
        assert missing == []

    def test_every_real_post_lock_row_says_why(self, rows) -> None:
        silent = [
            r["game_id"]
            for r in rows
            if r["verdict"] == "post_lock"
            and r["what_moved"] != "scheduled_not_moved"
            and "D33.2-21" not in str(r["notes"])
        ]
        assert silent == []

    def test_move_indexes_are_one_based_and_contiguous(self, rows) -> None:
        bad = {
            g: [int(str(r["move_index"])) for r in rs]
            for g, rs in _by_game(rows).items()
            if [int(str(r["move_index"])) for r in rs] != list(range(1, len(rs) + 1))
        }
        assert bad == {}

    def test_a_game_with_several_moves_has_distinct_move_indexes(self, rows) -> None:
        multi = {g: rs for g, rs in _by_game(rows).items() if len(rs) > 1}
        assert CLOSEST_CALL in multi  # the closest call is a two-move game
        assert all(
            len({r["move_index"] for r in rs}) == len(rs) for rs in multi.values()
        )

    def test_scheduled_games_never_move(self, rows) -> None:
        assert [
            r["game_id"]
            for r in rows
            if r["what_moved"] == "scheduled_not_moved"
            and r["from_value"] != r["to_value"]
        ] == []

    def test_every_venue_value_is_a_known_stadium(self, rows) -> None:
        payload = json.loads(VENUES_PATH.read_text(encoding="utf-8"))
        known = {str(v["stadium_id"]) for v in payload["venues"]}
        unknown = sorted(
            {
                value
                for r in rows
                if r["what_moved"] == "venue"
                for value in (str(r["from_value"]), str(r["to_value"]))
            }
            - known
        )
        assert unknown == []


class TestTheContractRefusesWhatItShould:
    """Non-vacuity: the checker bites on planted rows."""

    @staticmethod
    def _planted(**overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "game_id": "2030_W01_AAA@HHH",
            "move_index": 1,
            "what_moved": "venue",
            "from_value": "HHH00",
            "to_value": "OTH00",
            "announced_at_utc": "2030-09-07T22:00:00Z",
            "source_url": "https://example.invalid/story",
            "source_title": "planted",
            "source_published_at": "2030-09-07T22:00:00Z",
            "lock_utc": "2030-09-07T22:00:00Z",
            "verdict": "pre_lock",
            "notes": "planted",
        }
        row.update(overrides)
        return row

    def test_an_announcement_exactly_at_the_lock_is_pre_lock(self) -> None:
        assert row_violations(self._planted()) == []

    def test_pre_lock_with_an_empty_source_is_refused(self) -> None:
        assert "pre_lock without a source_url" in row_violations(
            self._planted(source_url="")
        )

    def test_pre_lock_one_second_after_the_lock_is_refused(self) -> None:
        late = self._planted(announced_at_utc="2030-09-07T22:00:01Z")
        assert "pre_lock announced after the lock" in row_violations(late)

    def test_post_lock_needs_no_source(self) -> None:
        assert (
            row_violations(
                self._planted(
                    verdict="post_lock",
                    source_url="",
                    announced_at_utc="",
                    notes="none found",
                )
            )
            == []
        )

    def test_an_empty_what_moved_is_refused(self) -> None:
        assert row_violations(self._planted(what_moved=""))

    def test_a_scheduled_row_that_moves_is_refused(self) -> None:
        assert row_violations(self._planted(what_moved="scheduled_not_moved"))

    def test_a_naive_lock_is_refused(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            row_violations(self._planted(lock_utc="2030-09-07T22:00:00"))


# ------------------------------------------------------------ the lock and silver


class TestTheTableIsTrueToTheLockRuleAndToSilver:
    def test_every_lock_is_the_lock_rule(self, rows, silver) -> None:
        drifted = [
            f"{r['game_id']}#{r['move_index']}"
            for r in rows
            if parse_utc(str(r["lock_utc"]))
            != game_lock(
                silver.at[str(r["game_id"]), "kickoff_et"], game_id=str(r["game_id"])
            )
        ]
        assert drifted == []

    def test_the_last_move_in_each_family_lands_on_the_silver_fact(
        self, rows, silver
    ) -> None:
        wrong = []
        for game_id, moves in _by_game(rows).items():
            actual = _silver_facts(silver, game_id)
            last: dict[str, str] = {}
            for move in moves:
                what, to = str(move["what_moved"]), str(move["to_value"])
                if what == "scheduled_not_moved":
                    continue
                last[what] = to
                if what == "week":  # a week move also changes the kickoff date
                    last["date"] = to.rsplit(" ", maxsplit=1)[-1]
                elif what == "date" and "week" in last:
                    last["week"] = f"{last['week'].split(' ')[0]} {to}"
            wrong.extend(
                f"{game_id} {family}: table {value!r} vs silver {actual[family]!r}"
                for family, value in last.items()
                if value != actual[family]
            )
        assert wrong == []

    def test_consecutive_moves_chain(self, rows) -> None:
        broken = []
        for game_id, moves in _by_game(rows).items():
            date_now: str | None = None
            venue_now: str | None = None
            for move in moves:
                what, frm, to = (
                    move["what_moved"],
                    str(move["from_value"]),
                    str(move["to_value"]),
                )
                if what == "venue":
                    if venue_now is not None and frm != venue_now:
                        broken.append(f"{game_id}#{move['move_index']} venue")
                    venue_now = to
                elif what in {"date", "week"}:
                    frm_date = frm.rsplit(" ", maxsplit=1)[-1]
                    if date_now is not None and frm_date != date_now:
                        broken.append(f"{game_id}#{move['move_index']} date")
                    date_now = to.rsplit(" ", maxsplit=1)[-1]
        assert broken == []

    def test_every_scheduled_row_records_the_silver_facts(self, rows, silver) -> None:
        wrong = []
        for r in rows:
            if r["what_moved"] != "scheduled_not_moved":
                continue
            actual = _silver_facts(silver, str(r["game_id"]))
            expected = f"{actual['venue']} {actual['date']}"
            if r["to_value"] != expected:
                wrong.append(f"{r['game_id']}: {r['to_value']!r} vs {expected!r}")
        assert wrong == []


class TestTheLimitationIsStated:
    def test_the_header_states_the_week_move_blind_spot(self) -> None:
        header = TABLE_PATH.read_text(encoding="utf-8").split("[[moves]]", 1)[0]
        for phrase in ("ORIGINAL schedule", "Ike", "Irma", "WEEK", "D33.2-21"):
            assert phrase in header
