"""Compare the decoded day-before bulletins with the on-disk observations (Plan 33.2-11 Task 3).

For every covered 2002-2025 game (a US outdoor or retractable-roof venue), the 12 UTC bulletin of
the ET day before kickoff is decoded from bronze, the forecast hour nearest kickoff is taken, and
its temperature and wind are compared with the ERA5 observation already in
``data/silver/weather.parquet`` (still in place; Plan 33.2-12 replaces it, which is why this runs
now). Six statistics are computed and judged against ``config/mos_tolerance.py``.

WHAT A BREACH MEANS. The bounds are DETECTION BOUNDS ON OUR DECODING, not a claim that the
forecast was accurate. A breach means a unit error, a wrong station, a wrong valid hour or a
mis-joined game. SPEC R6: a breach STOPS the backfill, and the response is to fix the decode or
the station map -- never to widen a bound.

PRE-REGISTRATION IS CHECKED, NOT ASSUMED. Before computing anything this script asserts that
``config/mos_tolerance.py`` has no uncommitted change and that its last-modifying commit is a
STRICT ancestor of ``HEAD``. A bound edited after seeing the numbers is not a pre-registration.
This script READS the tolerance module and never writes it; it writes nothing under ``data/`` or
``artifacts/``. With ``--write-readout`` it renders ``MOS-DECODE-COMPARISON.md`` from the same
numbers it prints, so the document cannot drift from the computation.

THE 68 KICKOFF-HOUR GAMES. 68 Monday/Thursday night games of 2002-2005 carry a 09:00 ET kickoff
(an AM/PM error in the feed; Plan 33.2-12 corrects it). ERA5 was fetched at that same wrong hour
and the forecast hour is taken nearest that same wrong kickoff, so the pairing is internally
consistent and they are compared. So they cannot hide a breach, every bound must pass BOTH over
all compared games and over all compared games excluding them
(``VERDICT_REQUIRES_PASS_WITHOUT_KICKOFF_HOUR_SUBSET``); they are also reported alone.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config.mos_tolerance as tol
from scripts.backfill_mos_forecasts import (
    CORPUS_FIRST_SEASON,
    CORPUS_LAST_SEASON,
    CoveredGame,
    absent_runs_in_bronze,
    load_bronze_run_records,
    load_corpus,
)
from scripts.mos_decode import build_weather_record, run_instant
from utils.date_utils import kickoff_wall_clock_et

REPO_ROOT = Path(__file__).resolve().parent.parent
TOLERANCE_PATH = "config/mos_tolerance.py"
READOUT_NAME = "MOS-DECODE-COMPARISON.md"

#: The measured live-versus-history provider mismatch (RESEARCH 6.6, measured 2026-09-15):
#: 18 stadium stations, one common valid hour, MOS GFS latest run minus Open-Meteo gfs_global.
LIVE_VS_HISTORY_TEMP_MEAN_F = 0.37
LIVE_VS_HISTORY_WIND_MEAN_MPH = 1.56
LIVE_VS_HISTORY_WIND_MAX_MPH = 6.5
WIND_CALM_MODERATE_EDGE_MPH = 5.0


class ToleranceNotPreRegisteredError(RuntimeError):
    """The tolerance module is not committed in a strict ancestor of HEAD."""


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def tolerance_commit() -> str:
    """The last commit to modify ``config/mos_tolerance.py``."""
    return _git("log", "-1", "--format=%H", "--", TOLERANCE_PATH).stdout.strip()


def assert_tolerance_pre_registered() -> str:
    """Refuse unless the tolerance is committed, unmodified, and strictly older than HEAD.

    Returns:
        The tolerance commit sha.

    Raises:
        ToleranceNotPreRegisteredError: naming which of the three conditions failed.
    """
    if _git("diff", "--quiet", "HEAD", "--", TOLERANCE_PATH).returncode != 0:
        raise ToleranceNotPreRegisteredError(
            f"{TOLERANCE_PATH} has uncommitted changes. The bounds are READ, never edited."
        )
    commit = tolerance_commit()
    head = _git("rev-parse", "HEAD").stdout.strip()
    if not commit:
        raise ToleranceNotPreRegisteredError(
            f"{TOLERANCE_PATH} has never been committed"
        )
    if commit == head:
        raise ToleranceNotPreRegisteredError(
            f"{TOLERANCE_PATH} was last modified by HEAD itself ({commit}); it must be a "
            "STRICT ancestor of the commit the comparison runs at"
        )
    if _git("merge-base", "--is-ancestor", commit, head).returncode != 0:
        raise ToleranceNotPreRegisteredError(
            f"{TOLERANCE_PATH}'s commit {commit} is not an ancestor of HEAD {head}"
        )
    return commit


@dataclass(frozen=True)
class Stats:
    """The six statistics over one set of compared games."""

    n: int
    temp_mean_signed_diff: float
    temp_mean_abs_diff: float
    temp_gross_miss_share: float
    wind_mean_signed_diff: float
    wind_mean_abs_diff: float
    temp_p95_abs: float
    temp_max_abs: float
    wind_p95_abs: float
    wind_max_abs: float


def compute_stats(frame: pd.DataFrame) -> Stats:
    """The statistics the bounds are stated over; forecast minus observation."""
    dt = frame["mos_temp_f"] - frame["era5_temp_f"]
    dw = frame["mos_wind_mph"] - frame["era5_wind_mph"]
    return Stats(
        n=len(frame),
        temp_mean_signed_diff=float(dt.mean()),
        temp_mean_abs_diff=float(dt.abs().mean()),
        temp_gross_miss_share=float((dt.abs() > tol.GROSS_TEMP_MISS_F).mean()),
        wind_mean_signed_diff=float(dw.mean()),
        wind_mean_abs_diff=float(dw.abs().mean()),
        temp_p95_abs=float(dt.abs().quantile(0.95)),
        temp_max_abs=float(dt.abs().max()),
        wind_p95_abs=float(dw.abs().quantile(0.95)),
        wind_max_abs=float(dw.abs().max()),
    )


def judge(stats: Stats, resolved_share: float) -> dict[str, bool]:
    """PASS (True) / FAIL (False) for each of the six pre-registered bounds."""
    return {
        "temp_mean_signed_diff": abs(stats.temp_mean_signed_diff)
        <= tol.TEMP_MEAN_SIGNED_DIFF_MAX_ABS_F,
        "temp_mean_abs_diff": stats.temp_mean_abs_diff <= tol.TEMP_MEAN_ABS_DIFF_MAX_F,
        "temp_gross_miss_share": stats.temp_gross_miss_share
        <= tol.TEMP_GROSS_MISS_SHARE_MAX,
        "wind_mean_signed_diff": tol.WIND_MEAN_SIGNED_DIFF_MIN_MPH
        <= stats.wind_mean_signed_diff
        <= tol.WIND_MEAN_SIGNED_DIFF_MAX_MPH,
        "wind_mean_abs_diff": stats.wind_mean_abs_diff
        <= tol.WIND_MEAN_ABS_DIFF_MAX_MPH,
        "resolved_bulletin_share": resolved_share >= tol.RESOLVED_BULLETIN_SHARE_MIN,
    }


def is_kickoff_hour_subset(game: CoveredGame) -> bool:
    """The 2002-2005 night games stored at a 09:00 ET kickoff (the AM/PM feed error)."""
    et = kickoff_wall_clock_et(game.kickoff_utc)
    return game.season <= 2005 and et.hour == 9 and et.minute == 0


@dataclass
class Comparison:
    """Everything the readout needs."""

    tolerance_commit: str
    covered: int
    resolved: int
    unresolved_game_ids: list[str]
    uncoverable_by_venue: dict[str, int]
    frame: pd.DataFrame
    kickoff_hour_subset_n: int
    all_stats: Stats
    excl_stats: Stats
    subset_stats: Stats | None
    all_ruling: dict[str, bool]
    excl_ruling: dict[str, bool]
    absent_runs: tuple[str, ...] = ()

    @property
    def resolved_share(self) -> float:
        return self.resolved / self.covered if self.covered else 0.0

    @property
    def breached(self) -> list[str]:
        names = [n for n in tol.BOUND_NAMES if not self.all_ruling[n]]
        if tol.VERDICT_REQUIRES_PASS_WITHOUT_KICKOFF_HOUR_SUBSET:
            names += [
                f"{n} (excluding the kickoff-hour subset)"
                for n in tol.BOUND_NAMES
                if not self.excl_ruling[n]
            ]
        return names

    @property
    def verdict(self) -> str:
        return "PASS" if not self.breached else "FAIL"


def run_comparison(base_path: Path | str | None = None) -> Comparison:
    """Decode every covered game from bronze and compare with the on-disk ERA5 rows."""
    commit = assert_tolerance_pre_registered()
    root = Path(base_path) if base_path is not None else REPO_ROOT / "data"
    games = pd.read_parquet(root / "silver" / "games.parquet", engine="pyarrow")
    games = games[games["season"].between(CORPUS_FIRST_SEASON, CORPUS_LAST_SEASON)]
    covered, uncoverable = load_corpus(games.reset_index(drop=True))
    runs = load_bronze_run_records(root)

    weather = pd.read_parquet(root / "silver" / "weather.parquet", engine="pyarrow")
    era5 = weather[
        (weather["weather_source"] == "archive") & weather["temp_f"].notna()
    ].set_index("game_id")

    rows, unresolved = [], []
    for game in covered:
        built = build_weather_record(
            game.game_id,
            game.kickoff_utc,
            runs.get((game.station, run_instant(game.lock_date)), []),
        )
        if built is None:
            unresolved.append(game.game_id)
            continue
        if game.game_id not in era5.index:
            continue
        obs = era5.loc[game.game_id]
        rows.append(
            {
                "game_id": game.game_id,
                "season": game.season,
                "station": game.station,
                "stadium_id": game.stadium_id,
                "kickoff_hour_subset": is_kickoff_hour_subset(game),
                "mos_temp_f": built["temp_f"],
                "mos_wind_mph": built["wind_mph"],
                "era5_temp_f": float(obs["temp_f"]),
                "era5_wind_mph": float(obs["wind_mph"]),
            }
        )
    frame = pd.DataFrame(rows)
    subset = frame[frame["kickoff_hour_subset"]]
    excl = frame[~frame["kickoff_hour_subset"]]
    resolved = len(covered) - len(unresolved)
    share = resolved / len(covered) if covered else 0.0
    all_stats, excl_stats = compute_stats(frame), compute_stats(excl)
    by_venue: dict[str, int] = {}
    for game in uncoverable:
        by_venue[game.stadium_id] = by_venue.get(game.stadium_id, 0) + 1
    return Comparison(
        tolerance_commit=commit,
        covered=len(covered),
        resolved=resolved,
        unresolved_game_ids=sorted(unresolved),
        uncoverable_by_venue=dict(sorted(by_venue.items())),
        frame=frame,
        kickoff_hour_subset_n=sum(is_kickoff_hour_subset(g) for g in covered),
        all_stats=all_stats,
        excl_stats=excl_stats,
        subset_stats=compute_stats(subset) if len(subset) else None,
        all_ruling=judge(all_stats, share),
        excl_ruling=judge(excl_stats, share),
        absent_runs=absent_runs_in_bronze(root),
    )


def _value(stats: Stats, name: str, share: float) -> float:
    return share if name == "resolved_bulletin_share" else float(getattr(stats, name))


def _bound_text(name: str) -> str:
    return {
        "temp_mean_signed_diff": f"within +/- {tol.TEMP_MEAN_SIGNED_DIFF_MAX_ABS_F} F",
        "temp_mean_abs_diff": f"<= {tol.TEMP_MEAN_ABS_DIFF_MAX_F} F",
        "temp_gross_miss_share": (
            f"<= {tol.TEMP_GROSS_MISS_SHARE_MAX} (share off by more than "
            f"{tol.GROSS_TEMP_MISS_F} F)"
        ),
        "wind_mean_signed_diff": (
            f"between +{tol.WIND_MEAN_SIGNED_DIFF_MIN_MPH} and "
            f"+{tol.WIND_MEAN_SIGNED_DIFF_MAX_MPH} mph"
        ),
        "wind_mean_abs_diff": f"<= {tol.WIND_MEAN_ABS_DIFF_MAX_MPH} mph",
        "resolved_bulletin_share": f">= {tol.RESOLVED_BULLETIN_SHARE_MIN}",
    }[name]


def print_report(result: Comparison) -> None:
    print(f"TOLERANCE_COMMIT= {result.tolerance_commit}")
    print(f"COVERED_GAMES= {result.covered}")
    print(f"RESOLVED_GAMES= {result.resolved}")
    print(f"UNRESOLVED_GAMES= {len(result.unresolved_game_ids)}")
    print(f"ABSENT_RUNS_IN_BRONZE= {list(result.absent_runs)}")
    print(f"COMPARED_GAMES= {result.all_stats.n}")
    print(f"UNCOVERABLE_GAMES= {sum(result.uncoverable_by_venue.values())}")
    print(f"KICKOFF_HOUR_SUBSET_GAMES= {result.kickoff_hour_subset_n}")
    for label, stats, ruling in (
        ("ALL", result.all_stats, result.all_ruling),
        ("EXCL68", result.excl_stats, result.excl_ruling),
    ):
        for name in tol.BOUND_NAMES:
            value = _value(stats, name, result.resolved_share)
            verdict = "PASS" if ruling[name] else "FAIL"
            print(f"BOUND[{label}] {name}= {value:.4f} {verdict}")
    print(f"BOUND_BREACHED= {result.breached}")
    print(f"VERDICT= {result.verdict}")


def _stats_row(label: str, stats: Stats) -> str:
    return (
        f"| {label} | {stats.n} | {stats.temp_mean_signed_diff:+.2f} | "
        f"{stats.temp_mean_abs_diff:.2f} | {stats.temp_gross_miss_share:.4f} | "
        f"{stats.temp_p95_abs:.1f} | {stats.temp_max_abs:.1f} | "
        f"{stats.wind_mean_signed_diff:+.2f} | {stats.wind_mean_abs_diff:.2f} | "
        f"{stats.wind_p95_abs:.1f} | {stats.wind_max_abs:.1f} |"
    )


def render_readout(result: Comparison) -> str:
    """The committed readout, rendered from the same numbers the script prints."""
    share = result.resolved_share
    lines = [
        "# MOS Decode Comparison (Phase 33.2, Plan 11)",
        "",
        "## What this is",
        "",
        "Past games will use the weather forecast as it stood at each game's lock: the 12 UTC NWS",
        "airport forecast bulletin (MOS: AVN before 2003-12-16, GFS after) of the ET calendar day",
        "before kickoff, fetched free from the Iowa Environmental Mesonet archive. Before that",
        "history replaces anything, this document checks that we DECODE the bulletins correctly,",
        "by comparing each decoded forecast with the ERA5 observation already on disk for the same",
        "game.",
        "",
        "**These are DETECTION BOUNDS ON OUR DECODING, not a claim that the forecast was "
        "accurate.**",
        "A day-before forecast legitimately misses the weather that happened. A breach would mean",
        "a unit error, a wrong station, a wrong forecast hour or a mis-joined game -- not surprising",
        "weather. A breach stops the backfill, and the fix is the decoder or the station map, never",
        "a wider bound.",
        "",
        "## Pre-registration",
        "",
        "The six bounds, the precipitation anchoring and the transmission argument were ruled by",
        f"the owner on {tol.OWNER_RULINGS_DATE}, before any comparison number existed, and",
        f"committed alone in `{result.tolerance_commit}` (`{TOLERANCE_PATH}`). The comparison",
        "script refuses to run unless that commit is a strict ancestor of the commit it runs at,",
        "and `tests/unit/test_mos_tolerance_ancestry.py` asserts it is a strict ancestor of the",
        "commit that adds this document.",
        "",
        "Owner rulings, verbatim:",
        "",
    ]
    lines += [f'- `{key}`: "{answer}"' for key, answer in tol.OWNER_RULINGS.items()]
    lines += [
        "",
        "The transmission figure: the archive stores each bulletin's model cycle time, not when",
        "it went out. The 12 UTC bulletin is public long before the 18:00 ET lock (22:00 UTC in",
        "daylight time, 23:00 UTC in standard time) and the 18 UTC bulletin is never used, so",
        "exactly one run is admissible per game. The often-quoted delay of about 4h15m is",
        "UNVERIFIED: no official document confirms it, and nothing relies on it.",
        "",
        "## Method",
        "",
        "- Population: every 2002-2025 game at a US outdoor or retractable-roof venue, with the",
        "  venue in force at the lock (a game moved after its lock uses its pre-move venue).",
        "- Station: the nearest primary airport to the venue (`config/mos_stations.py`).",
        "- Forecast: the 12 UTC run of the lock date; the forecast hour nearest kickoff (at most",
        "  90 minutes away). Knots are converted to mph once; wind direction is already degrees.",
        "- Observation: the ERA5 row in `data/silver/weather.parquet` for the same game.",
        "- Differences are forecast minus observation. A typical miss is the mean absolute",
        "  difference.",
        "- Every bound must pass over all compared games AND over all compared games excluding",
        "  the kickoff-hour subset (below).",
        "",
        "## Coverage",
        "",
        f"- Covered games: {result.covered}",
        f"- Games with a resolved bulletin: {result.resolved} (share {share:.4f})",
        f"- Covered games with no resolved bulletin: {len(result.unresolved_game_ids)}",
        f"- Games compared with an ERA5 observation: {result.all_stats.n} (a retractable roof",
        "  recorded closed has no ERA5 observation, so it is covered but not compared)",
        f"- Uncoverable games (played outside the USA, no US forecast station exists): "
        f"{sum(result.uncoverable_by_venue.values())}. They are outside the coverage share's",
        "  denominator and take the honest no-forecast path; no stand-in station is ever used.",
        "  By venue: "
        + ", ".join(f"{k} {v}" for k, v in result.uncoverable_by_venue.items())
        + ".",
    ]
    if result.unresolved_game_ids:
        lines.append(
            "- Unresolved game ids: " + ", ".join(result.unresolved_game_ids[:50]) + "."
        )
    if result.absent_runs:
        lines += [
            "- Runs ABSENT FROM THE ARCHIVE (asked twice: the date-range endpoint omitted them",
            "  and the single-run endpoint answered that it holds no results). Their games take",
            "  the no-forecast path; no other run is substituted: "
            + ", ".join(result.absent_runs)
            + ".",
        ]
    lines += [
        "",
        "## Bounds and rulings",
        "",
        "| Bound | Limit | All compared | Ruling | Excluding the kickoff-hour subset | Ruling |",
        "|---|---|---|---|---|---|",
    ]
    for name in tol.BOUND_NAMES:
        a = _value(result.all_stats, name, share)
        e = _value(result.excl_stats, name, share)
        lines.append(
            f"| {name} | {_bound_text(name)} | {a:.4f} | "
            f"{'PASS' if result.all_ruling[name] else 'FAIL'} | {e:.4f} | "
            f"{'PASS' if result.excl_ruling[name] else 'FAIL'} |"
        )
    lines += [
        "",
        f"**VERDICT: {result.verdict}**"
        + (
            "" if not result.breached else f" -- breached: {', '.join(result.breached)}"
        ),
        "",
        "## The six statistics",
        "",
        "Degrees F and mph; forecast minus observation.",
        "",
        "| Set | n | Temp mean | Temp typical miss | Share > 25 F | Temp p95 | Temp max | "
        "Wind mean | Wind typical miss | Wind p95 | Wind max |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        _stats_row("All compared", result.all_stats),
        _stats_row("Excluding the kickoff-hour subset", result.excl_stats),
    ]
    if result.subset_stats is not None:
        lines.append(_stats_row("Kickoff-hour subset alone", result.subset_stats))
    lines += [
        "",
        "## The kickoff-hour subset",
        "",
        "68 Monday and Thursday night games of 2002-2005 are stored in silver with a 09:00 ET",
        "kickoff, an AM/PM error in the feed that Plan 33.2-12 corrects;",
        f"{result.kickoff_hour_subset_n} of them are at covered venues (the rest are indoor). ERA5 was",
        "fetched at that same wrong hour and the forecast hour is taken nearest the same wrong",
        "kickoff, so the pair describes one instant and is compared. They are listed separately",
        "so they cannot widen or hide anything: every bound was judged with and without them.",
        "",
        "## By station",
        "",
        "A bad station choice would show here as an outlier (RESEARCH assumption A2).",
        "",
        "| Station | n | Temp mean | Temp typical miss | Wind mean | Wind typical miss |",
        "|---|---|---|---|---|---|",
    ]
    frame = result.frame.assign(
        dt=result.frame["mos_temp_f"] - result.frame["era5_temp_f"],
        dw=result.frame["mos_wind_mph"] - result.frame["era5_wind_mph"],
    )
    for station, group in frame.groupby("station"):
        lines.append(
            f"| {station} | {len(group)} | {group['dt'].mean():+.2f} | "
            f"{group['dt'].abs().mean():.2f} | {group['dw'].mean():+.2f} | "
            f"{group['dw'].abs().mean():.2f} |"
        )
    lines += [
        "",
        "## Live versus history: a measured, accepted difference",
        "",
        "Live 2026 games keep the Open-Meteo forecast service; history uses these bulletins. The",
        "difference between the two was measured on 2026-09-15 at one common hour across 18",
        f"stadium stations: temperature mean {LIVE_VS_HISTORY_TEMP_MEAN_F:+.2f} F (effectively",
        f"exact); wind mean {LIVE_VS_HISTORY_WIND_MEAN_MPH:+.2f} mph, bulletins higher, maximum gap",
        f"{LIVE_VS_HISTORY_WIND_MAX_MPH} mph. The calm/moderate wind line is "
        f"{WIND_CALM_MODERATE_EDGE_MPH} mph, so on a calm",
        "day a game can switch wind band between training and live serving, and the spread model",
        "uses that band today. This is a known, measured, accepted train/serve difference, handed",
        "to the re-fit plans (33.2-23).",
        "",
        "## What this does not say",
        "",
        "It does not say the forecasts were good. It says the decoded values sit where a correctly",
        "decoded day-before forecast should sit relative to the observed weather, and nowhere a",
        "unit, station, hour or join error would put them.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-readout",
        action="store_true",
        help=f"render {READOUT_NAME} at the repo root from these numbers",
    )
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args(argv)
    result = run_comparison(args.data_root)
    print_report(result)
    if args.write_readout:
        text = render_readout(result)
        if not text.isascii():  # pragma: no cover - a rendering defect, not a data one
            raise RuntimeError("the readout is not ASCII")
        (REPO_ROOT / READOUT_NAME).write_text(text, encoding="utf-8", newline="\n")
        print(f"READOUT_WRITTEN= {READOUT_NAME}")
    return 0 if result.verdict == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
