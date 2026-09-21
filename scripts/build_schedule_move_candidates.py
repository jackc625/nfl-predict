"""Enumerate every candidate emergency schedule move since 2002 (Plan 33.2-10 Task 1, SPEC R8).

WHY THIS EXISTS
---------------
D33.2-04 treats schedule and venue facts as known at each game's lock (18:00 ET on the
day before kickoff). That holds for the spring schedule and for rule-bound changes. It
does not hold for a hurricane, a roof collapse or a blizzard. Every such move must be
proven announced at or before its game's lock, or the game is rebuilt with the facts as
they stood before the move (D33.2-21).

This script produces the candidate list that proof is researched against, and nothing
else. It is READ-ONLY on ``data/``. With ``--write`` it writes one committed file,
``config/schedule_move_candidates.toml``. That manifest is always this script's output:
a hand-added or hand-dropped row fails ``tests/unit/test_schedule_move_candidates.py``.

THREE MECHANICAL CRITERIA AND TWO COMMITTED INPUTS, UNIONED
-----------------------------------------------------------
1. ``neutral_domestic``: ``neutral_site`` is True and the venue is in the United States.
   Silver stores the POST-MOVE venue and hides a relocation inside ``neutral_site``
   (33.2-RESEARCH.md section 8.2): ``2010_W14_NYG@MIN`` sits at Ford Field flagged
   neutral, so it is not "away from the home team's usual stadium" at all.
2. ``tue_wed``: a Tuesday or Wednesday kickoff, with the weekday taken in
   America/New_York. ``kickoff_et`` is stored tz-aware in UTC, so a weekday read off the
   raw value turns every Monday-night game into a Tuesday one.
3. ``away_unflagged``: a game away from the home team's usual stadium (its modal stadium
   over its non-neutral home games that season) that does NOT carry ``neutral_site``.
4. ``named_event:<key>``: every event D33.2-04 names, through ``NAMED_EVENT_GAME_IDS``,
   unioned UNCONDITIONALLY whether or not a scan also surfaces the game.
5. ``research_discovered``: a game Task 2's citation research surfaced that none of the
   above names, through ``RESEARCH_DISCOVERED_GAME_IDS``. Adding a game here and
   re-running with ``--write`` is the ONLY way the manifest grows.

The 2025 international games corrected by Plan 33.2-09
(``config/international_venue_corrections.toml``) are venue facts, not moves. They are
removed from every mechanical criterion by reading that file, and a committed input that
names one of them is refused.

THE LIMITATION, STATED
----------------------
This repository does not hold the ORIGINAL schedule, only the schedule as played.
Hurricane Ike (2008) and Hurricane Irma (2017) each moved a game to a LATER WEEK, which
changes neither its venue nor, necessarily, its weekday. A week-number move is therefore
invisible to every mechanical scan here. Those games are in the manifest ONLY because
criterion 4 names them, which is what lets Task 2 record a citation for them. Without a
citation they take the D33.2-21 default: treated as announced after the lock and rebuilt
with the pre-move facts. ``WEEK_MOVE_BLINDSPOT=`` counts them, so the limitation is a
number and not only this paragraph.

OUTPUT CONTRACT (printed on every run, one per line)
----------------------------------------------------
``SEASONS=``, ``CANDIDATES=``, ``BY_CRITERION=``, ``WEEK_MOVE_BLINDSPOT=``,
``NAMED_ONLY=``, ``NAMED_EVENTS_UNRESOLVED=`` and ``MANIFEST_IN_SYNC=``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd

import utils.game_lock as lock_rule
from scripts.repair_international_venues import usual_home_stadium
from utils.date_utils import kickoff_wall_clock_et

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
MANIFEST_PATH = REPO_ROOT / "config" / "schedule_move_candidates.toml"
INTERNATIONAL_CORRECTIONS_PATH = (
    REPO_ROOT / "config" / "international_venue_corrections.toml"
)

#: The first season the move proof covers (D33.2-04: "every emergency move since 2002").
#: There is no last season: every season silver holds is enumerated, so a live-season
#: move reaches the manifest the next time it is regenerated.
FIRST_SEASON: int = 2002

#: ``venues.json`` spells the United States one way; this set is the guard's convention.
US_COUNTRY_VALUES: frozenset[str] = frozenset({"US", "USA", "UNITED STATES"})

#: Python ``dayofweek`` values for Tuesday and Wednesday.
TUESDAY_WEDNESDAY: frozenset[int] = frozenset({1, 2})

CRITERION_NEUTRAL_DOMESTIC = "neutral_domestic"
CRITERION_TUE_WED = "tue_wed"
CRITERION_AWAY_UNFLAGGED = "away_unflagged"
CRITERION_NAMED_PREFIX = "named_event:"
CRITERION_RESEARCH = "research_discovered"

MECHANICAL_CRITERIA: tuple[str, ...] = (
    CRITERION_NEUTRAL_DOMESTIC,
    CRITERION_TUE_WED,
    CRITERION_AWAY_UNFLAGGED,
)

#: Every event D33.2-04 names, mapped to the game id(s) it affected. Seeds MEASURED
#: 2026-09-20 in silver ``games``. Unioned UNCONDITIONALLY. A further game belonging to
#: one of these events, found against a dated source, joins its event here: Task 2's
#: research (2026-09-21) added the four Baton Rouge Katrina games, the Bengals game Ike
#: displaced, and eleven more 2020 COVID reschedules. Each added game's source is its row
#: in ``config/schedule_moves.toml``.
NAMED_EVENT_GAME_IDS: dict[str, tuple[str, ...]] = {
    # Hurricane Katrina: all eight of the Saints' 2005 home games, played away from the
    # Superdome. The four at Tiger Stadium (BRG00) are invisible to the scans because
    # BRG00 is the Saints' modal 2005 stadium.
    "katrina_2005": (
        "2005_W02_NYG@NO",
        "2005_W04_BUF@NO",
        "2005_W06_ATL@NO",
        "2005_W08_MIA@NO",
        "2005_W09_CHI@NO",
        "2005_W13_TB@NO",
        "2005_W15_CAR@NO",
        "2005_W16_DET@NO",
    ),
    # Hurricane Ike: Ravens at Texans moved from week 2 to week 10, and the Bengals game
    # it displaced moved from week 10 to week 8 (both WEEK moves).
    "ike_2008": ("2008_W10_BAL@HOU", "2008_W08_CIN@HOU"),
    # Metrodome 2010, first time: roof collapse, Giants game moved to Detroit, Monday.
    "metrodome_2010_roof_collapse": ("2010_W14_NYG@MIN",),
    # Metrodome 2010, second time: Bears game moved to TCF Bank Stadium.
    "metrodome_2010_tcf_bank": ("2010_W15_CHI@MIN",),
    # Buffalo snow 2014: Jets at Bills moved to Detroit, Monday.
    "buffalo_snow_2014": ("2014_W12_NYJ@BUF",),
    # Hurricane Irma: Buccaneers at Dolphins moved from week 1 to week 11 (a WEEK move).
    "irma_2017": ("2017_W11_TB@MIA",),
    # The 49ers in Glendale 2020: Santa Clara County order.
    "glendale_49ers_2020": (
        "2020_W13_BUF@SF",
        "2020_W14_WAS@SF",
        "2020_W17_SEA@SF",
    ),
    # COVID reschedules 2020-2021, including the Sunday-to-Monday moves the weekday
    # alone does not reveal (2020_W04_NE@KC, 2021_W15_LV@CLE) and the October 2020
    # week swaps, which no scan can see without the original schedule.
    "covid_2020_2021": (
        "2020_W04_NE@KC",
        "2020_W05_BUF@TEN",
        "2020_W06_DEN@NE",
        "2020_W06_KC@BUF",
        "2020_W06_NYJ@MIA",
        "2020_W07_JAX@LAC",
        "2020_W07_PIT@TEN",
        "2020_W08_LAC@DEN",
        "2020_W08_PIT@BAL",
        "2020_W10_LAC@MIA",
        "2020_W11_MIA@DEN",
        "2020_W11_NYJ@LAC",
        "2020_W12_BAL@PIT",
        "2020_W13_DAL@BAL",
        "2020_W13_WAS@PIT",
        "2021_W15_LV@CLE",
        "2021_W15_SEA@LA",
        "2021_W15_WAS@PHI",
    ),
    # Hurricane Ida 2021: Packers at Saints moved to Jacksonville.
    "ida_2021": ("2021_W01_GB@NO",),
    # Buffalo snow 2022: Browns at Bills moved to Detroit.
    "buffalo_snow_2022": ("2022_W11_CLE@BUF",),
}

#: Named events whose every game moved to another WEEK. No mechanical criterion can see
#: those without the original schedule, which this repository does not hold. (The
#: October 2020 COVID week swaps have the same shape; they sit inside an event that also
#: holds scan-visible games, so they are counted by ``NAMED_ONLY=`` instead.)
WEEK_MOVE_EVENTS: frozenset[str] = frozenset({"ike_2008", "irma_2017"})

#: A game Task 2's research surfaced that no criterion above names: game id -> the reason
#: and the dated source that surfaced it. Adding an entry and re-running with ``--write``
#: is the documented entry path. Task 2 (2026-09-21) surfaced these five with a
#: read-only weekday scan of silver (September-November Saturdays, Fridays, a second
#: Monday game in a week) cross-checked against Wikipedia's "List of canceled and
#: rescheduled NFL games" and each season's scheduling-changes section.
RESEARCH_DISCOVERED_GAME_IDS: dict[str, dict[str, str]] = {
    "2004_W01_TEN@MIA": {
        "reason": "Hurricane Ivan: moved from Sunday 2004-09-12 to Saturday 2004-09-11",
        "source_url": (
            "http://usatoday30.usatoday.com/sports/football/nfl/"
            "2004-09-09-week1-game-caps_x.htm"
        ),
    },
    "2005_W07_KC@MIA": {
        "reason": "Hurricane Wilma: moved from Sunday 2005-10-23 to Friday 2005-10-21",
        "source_url": (
            "http://usatoday30.usatoday.com/sports/football/nfl/"
            "2005-10-20-chiefs-dolphins-moved_x.htm"
        ),
    },
    "2018_W06_SEA@LV": {
        "reason": (
            "London venue moved from Tottenham Hotspur Stadium to Wembley (stadium "
            "construction delay)"
        ),
        "source_url": (
            "https://www.tottenhamhotspur.com/news/2018/august/urgent-stadium-update/"
        ),
    },
    "2018_W11_KC@LA": {
        "reason": (
            "Moved from Estadio Azteca, Mexico City, to the Los Angeles Memorial Coliseum "
            "(field conditions)"
        ),
        "source_url": (
            "http://www.espn.com/nfl/story/_/id/25268527/"
            "nfl-cancels-mexico-city-trip-moves-chiefs-rams-game-la-monday-night-football"
        ),
    },
    "2023_W19_PIT@BUF": {
        "reason": "Buffalo snow: wild card game moved from Sunday 2024-01-14 to Monday",
        "source_url": (
            "https://www.wivb.com/sports/buffalo-bills/"
            "bills-steelers-playoff-game-postponed-to-monday/"
        ),
    },
}


class CandidateInputError(ValueError):
    """A committed input or a data file does not support the enumeration. Nothing is written."""


@dataclass(frozen=True)
class Candidate:
    """One candidate game with every criterion that surfaced it."""

    game_id: str
    season: int
    week: int
    game_type: str
    stadium_id: str
    neutral_site: bool
    kickoff_utc: str
    kickoff_weekday_et: str
    lock_utc: str
    criteria: tuple[str, ...]

    @property
    def is_mechanical(self) -> bool:
        return any(c in MECHANICAL_CRITERIA for c in self.criteria)

    def as_row(self) -> dict[str, object]:
        return {
            "game_id": self.game_id,
            "season": self.season,
            "week": self.week,
            "game_type": self.game_type,
            "stadium_id": self.stadium_id,
            "neutral_site": self.neutral_site,
            "kickoff_utc": self.kickoff_utc,
            "kickoff_weekday_et": self.kickoff_weekday_et,
            "lock_utc": self.lock_utc,
            "criteria": list(self.criteria),
        }


def utc_iso(value: datetime) -> str:
    """An aware instant as ``YYYY-MM-DDTHH:MM:SSZ``. A naive value is refused."""
    if value.tzinfo is None:
        raise CandidateInputError(f"refusing to format a naive instant: {value!r}")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def lock_utc_for(kickoff_value: object, game_id: str) -> str:
    """The game's lock (``utils.game_lock.game_lock``) as a UTC ISO string."""
    return utc_iso(lock_rule.game_lock(kickoff_value, game_id=game_id))


def load_international_game_ids(
    path: Path = INTERNATIONAL_CORRECTIONS_PATH,
) -> frozenset[str]:
    """The game ids Plan 33.2-09 corrected to a non-US venue (venue facts, not moves)."""
    records = tomllib.loads(path.read_text(encoding="utf-8")).get("correction", [])
    ids = frozenset(str(r["game_id"]) for r in records)
    if not ids:
        raise CandidateInputError(f"{path} holds no [[correction]] entries")
    return ids


def load_venue_countries(data_root: Path) -> dict[str, str]:
    """``stadium_id`` -> ``country`` from ``venues.json`` (a dict with one key, ``venues``)."""
    payload = json.loads((data_root / "venues.json").read_text(encoding="utf-8"))
    return {str(v["stadium_id"]): str(v["country"]) for v in payload["venues"]}


def load_games(data_root: Path) -> pd.DataFrame:
    """Silver ``games`` from *data_root*, read-only."""
    return pd.read_parquet(data_root / "silver" / "games.parquet")


def _et_dayofweek(kickoffs: pd.Series) -> pd.Series:
    """Weekday of each kickoff IN America/New_York, via THE kickoff accessor."""
    return kickoffs.map(lambda k: kickoff_wall_clock_et(k).weekday())


def _mechanical_criteria(
    games: pd.DataFrame,
    venue_countries: Mapping[str, str],
    international_ids: frozenset[str],
) -> dict[str, list[str]]:
    """game_id -> the mechanical criteria that surface it (the three scans)."""
    unknown = sorted(set(games["stadium_id"]) - set(venue_countries))
    if unknown:
        raise CandidateInputError(
            f"stadium_id(s) absent from venues.json: {unknown}. A venue with no record "
            "has no country, so the domestic test cannot be answered."
        )
    domestic = games["stadium_id"].map(
        lambda s: venue_countries[s].strip().upper() in US_COUNTRY_VALUES
    )
    neutral = games["neutral_site"].astype(bool)
    weekday = _et_dayofweek(cast(pd.Series, games["kickoff_et"]))
    usual = usual_home_stadium(games)
    usual_for_row = pd.Series(
        [
            usual.get((s, h))
            for s, h in zip(games["season"], games["home_team"], strict=True)
        ],
        index=games.index,
    )
    masks = {
        CRITERION_NEUTRAL_DOMESTIC: neutral & domestic,
        CRITERION_TUE_WED: weekday.isin(TUESDAY_WEDNESDAY),
        CRITERION_AWAY_UNFLAGGED: ~neutral & (games["stadium_id"] != usual_for_row),
    }
    found: dict[str, list[str]] = {}
    for criterion, mask in masks.items():
        for game_id in games.loc[mask, "game_id"]:
            if game_id in international_ids:
                continue
            found.setdefault(str(game_id), []).append(criterion)
    return found


def enumerate_candidates(
    games: pd.DataFrame,
    venue_countries: Mapping[str, str],
    international_ids: frozenset[str],
    named_events: Mapping[str, Sequence[str]] = NAMED_EVENT_GAME_IDS,
    research_discovered: Mapping[str, Mapping[str, str]] = RESEARCH_DISCOVERED_GAME_IDS,
    first_season: int = FIRST_SEASON,
) -> list[Candidate]:
    """The unioned candidate set, sorted by kickoff then game id.

    Raises:
        CandidateInputError: a named or research game id is absent from *games*, names a
            corrected international game, a research entry has no reason or source, or a
            stadium has no venue record.
    """
    scoped = cast(pd.DataFrame, games.loc[games["season"] >= first_season]).reset_index(
        drop=True
    )
    game_id_column = cast(pd.Series, scoped["game_id"])
    if game_id_column.duplicated().any():
        dupes = sorted(set(game_id_column[game_id_column.duplicated()]))
        raise CandidateInputError(f"game_id(s) appear more than once in games: {dupes}")
    present = set(scoped["game_id"].astype(str))

    criteria = _mechanical_criteria(scoped, venue_countries, international_ids)

    for event_key, game_ids in named_events.items():
        for game_id in game_ids:
            criteria.setdefault(game_id, []).append(
                f"{CRITERION_NAMED_PREFIX}{event_key}"
            )
    for game_id, record in research_discovered.items():
        if (
            not str(record.get("reason", "")).strip()
            or not str(record.get("source_url", "")).strip()
        ):
            raise CandidateInputError(
                f"RESEARCH_DISCOVERED_GAME_IDS[{game_id!r}] needs a non-empty reason and "
                "source_url: research adds a game only with the source that surfaced it"
            )
        criteria.setdefault(game_id, []).append(CRITERION_RESEARCH)

    committed = {
        g for g, cs in criteria.items() if any(c not in MECHANICAL_CRITERIA for c in cs)
    }
    absent = sorted(committed - present)
    if absent:
        raise CandidateInputError(
            f"committed input game id(s) absent from silver games (seasons >= "
            f"{first_season}): {absent}"
        )
    international = sorted(committed & international_ids)
    if international:
        raise CandidateInputError(
            f"committed input names corrected international game(s) {international}; "
            "those are venue facts (Plan 33.2-09), not schedule moves"
        )

    rows = scoped.set_index("game_id").loc[sorted(criteria)]
    candidates = []
    for game_id, row in rows.iterrows():
        kickoff = row["kickoff_et"]
        candidates.append(
            Candidate(
                game_id=str(game_id),
                season=int(row["season"]),
                week=int(row["week"]),
                game_type=str(row["game_type"]),
                stadium_id=str(row["stadium_id"]),
                neutral_site=bool(row["neutral_site"]),
                kickoff_utc=utc_iso(kickoff_wall_clock_et(kickoff)),
                kickoff_weekday_et=kickoff_wall_clock_et(kickoff).strftime("%A"),
                lock_utc=lock_utc_for(kickoff, str(game_id)),
                criteria=tuple(sorted(criteria[str(game_id)])),
            )
        )
    return sorted(candidates, key=lambda c: (c.kickoff_utc, c.game_id))


def unresolved_named_events(
    candidates: Sequence[Candidate],
    named_events: Mapping[str, Sequence[str]] = NAMED_EVENT_GAME_IDS,
) -> list[str]:
    """Event keys none of whose game ids is a candidate. Must be empty."""
    ids = {c.game_id for c in candidates}
    return sorted(k for k, gids in named_events.items() if not set(gids) & ids)


def week_move_blindspot(
    candidates: Sequence[Candidate],
    named_events: Mapping[str, Sequence[str]] = NAMED_EVENT_GAME_IDS,
    week_move_events: frozenset[str] = WEEK_MOVE_EVENTS,
) -> int:
    """Week-move events whose every game is surfaced by NO mechanical criterion."""
    by_id = {c.game_id: c for c in candidates}
    count = 0
    for event_key in sorted(week_move_events):
        games = [by_id[g] for g in named_events.get(event_key, ()) if g in by_id]
        if games and not any(c.is_mechanical for c in games):
            count += 1
    return count


def criterion_counts(candidates: Sequence[Candidate]) -> dict[str, int]:
    """How many candidates each criterion surfaced (a game counts once per criterion)."""
    counts: Counter[str] = Counter()
    for candidate in candidates:
        counts.update(candidate.criteria)
    return dict(sorted(counts.items()))


def _toml_value(value: object) -> str:
    """One TOML value. JSON string escaping is valid TOML basic-string escaping."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise CandidateInputError(f"no TOML rendering for {type(value).__name__}")


MANIFEST_HEADER = """\
# The candidate manifest for Plan 33.2-10's emergency-schedule-move proof (SPEC R8).
#
# GENERATED -- DO NOT HAND-EDIT. This file is the output of
#     uv run python -m scripts.build_schedule_move_candidates --write
# and tests/unit/test_schedule_move_candidates.py asserts it equals a fresh run. A game
# enters ONLY through the script's two committed inputs, NAMED_EVENT_GAME_IDS (every
# D33.2-04 named event, unioned unconditionally) and RESEARCH_DISCOVERED_GAME_IDS (a
# candidate citation research surfaced), followed by a regeneration.
#
# criteria: neutral_domestic (neutral site at a US venue), tue_wed (Tuesday or Wednesday
# kickoff, weekday in America/New_York), away_unflagged (away from the home team's usual
# stadium, not flagged neutral), named_event:<key>, research_discovered.
#
# lock_utc is utils.game_lock.game_lock of the game's kickoff: 18:00 America/New_York on
# the ET calendar day before kickoff, in UTC.
#
# THE LIMITATION: this repository holds no ORIGINAL schedule. Hurricane Ike (2008) and
# Hurricane Irma (2017) moved a game to a later WEEK, which no mechanical scan can see.
# They are rows here ONLY because NAMED_EVENT_GAME_IDS names them. Without a citation
# they take the D33.2-21 default: treated as announced after the lock and rebuilt with
# the pre-move facts.
#
# The 2025 international games corrected by Plan 33.2-09 are venue facts, not moves,
# and are absent by construction (read from config/international_venue_corrections.toml).
#
# ASCII only, no emoji (CLAUDE.md hard constraint).
"""


def render_manifest(candidates: Sequence[Candidate]) -> str:
    """The manifest text. Deterministic: the same candidates render the same bytes."""
    lines = [MANIFEST_HEADER]
    lines.append(f"first_season = {FIRST_SEASON}")
    lines.append(f"candidate_count = {len(candidates)}")
    for candidate in candidates:
        lines.append("")
        lines.append("[[candidates]]")
        for key, value in candidate.as_row().items():
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def read_manifest_rows(path: Path = MANIFEST_PATH) -> list[dict[str, object]]:
    """The committed manifest's ``[[candidates]]`` rows."""
    return list(tomllib.loads(path.read_text(encoding="utf-8")).get("candidates", []))


def build_from_data_root(data_root: Path) -> list[Candidate]:
    """The candidate set from the stores under *data_root* and the committed inputs."""
    return enumerate_candidates(
        load_games(data_root),
        load_venue_countries(data_root),
        load_international_game_ids(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enumerate candidate emergency schedule moves (read-only on data/)."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"write the manifest to {MANIFEST_PATH.relative_to(REPO_ROOT)}",
    )
    args = parser.parse_args(argv)

    candidates = build_from_data_root(args.data_root)
    seasons = sorted({c.season for c in candidates})
    named_only = [c.game_id for c in candidates if not c.is_mechanical]

    print(f"SEASONS= {FIRST_SEASON}..{seasons[-1] if seasons else FIRST_SEASON}")
    print(f"CANDIDATES= {len(candidates)}")
    print(f"BY_CRITERION= {json.dumps(criterion_counts(candidates), sort_keys=True)}")
    print(f"WEEK_MOVE_BLINDSPOT= {week_move_blindspot(candidates)}")
    print(f"NAMED_ONLY= {len(named_only)}")
    print(f"NAMED_ONLY_IDS= {named_only}")
    print(f"NAMED_EVENTS_UNRESOLVED= {unresolved_named_events(candidates)}")

    rendered = render_manifest(candidates)
    if args.write:
        MANIFEST_PATH.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"WROTE= {MANIFEST_PATH.relative_to(REPO_ROOT).as_posix()}")
    in_sync = (
        MANIFEST_PATH.exists() and MANIFEST_PATH.read_text(encoding="utf-8") == rendered
    )
    print(f"MANIFEST_IN_SYNC= {in_sync}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
