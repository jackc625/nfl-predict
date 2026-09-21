# MOS Decode Comparison (Phase 33.2, Plan 11)

## What this is

Past games will use the weather forecast as it stood at each game's lock: the 12 UTC NWS
airport forecast bulletin (MOS: AVN before 2003-12-16, GFS after) of the ET calendar day
before kickoff, fetched free from the Iowa Environmental Mesonet archive. Before that
history replaces anything, this document checks that we DECODE the bulletins correctly,
by comparing each decoded forecast with the ERA5 observation already on disk for the same
game.

**These are DETECTION BOUNDS ON OUR DECODING, not a claim that the forecast was accurate.**
A day-before forecast legitimately misses the weather that happened. A breach would mean
a unit error, a wrong station, a wrong forecast hour or a mis-joined game -- not surprising
weather. A breach stops the backfill, and the fix is the decoder or the station map, never
a wider bound.

## Pre-registration

The six bounds, the precipitation anchoring and the transmission argument were ruled by
the owner on 2026-09-21, before any comparison number existed, and
committed alone in `3e19d95d476d9e424df0c0460b84daec6cf2e3b7` (`config/mos_tolerance.py`). The comparison
script refuses to run unless that commit is a strict ancestor of the commit it runs at,
and `tests/unit/test_mos_tolerance_ancestry.py` asserts it is a strict ancestor of the
commit that adds this document.

Owner rulings, verbatim:

- `bounds-as-proposed`: "Proposed limits"
- `qpf-midpoint`: "Middle of each range"
- `commit-argument`: "Label figure unconfirmed"

The transmission figure: the archive stores each bulletin's model cycle time, not when
it went out. The 12 UTC bulletin is public long before the 18:00 ET lock (22:00 UTC in
daylight time, 23:00 UTC in standard time) and the 18 UTC bulletin is never used, so
exactly one run is admissible per game. The often-quoted delay of about 4h15m is
UNVERIFIED: no official document confirms it, and nothing relies on it.

## Method

- Population: every 2002-2025 game at a US outdoor or retractable-roof venue, with the
  venue in force at the lock (a game moved after its lock uses its pre-move venue).
- Station: the nearest primary airport to the venue (`config/mos_stations.py`).
- Forecast: the 12 UTC run of the lock date; the forecast hour nearest kickoff (at most
  90 minutes away). Knots are converted to mph once; wind direction is already degrees.
- Observation: the ERA5 row in `data/silver/weather.parquet` for the same game.
- Differences are forecast minus observation. A typical miss is the mean absolute
  difference.
- Every bound must pass over all compared games AND over all compared games excluding
  the kickoff-hour subset (below).

## Coverage

- Covered games: 5413
- Games with a resolved bulletin: 5412 (share 0.9998)
- Covered games with no resolved bulletin: 1
- Games compared with an ERA5 observation: 4793 (a retractable roof
  recorded closed has no ERA5 observation, so it is covered but not compared)
- Uncoverable games (played outside the USA, no US forecast station exists): 56. They are outside the coverage share's
  denominator and take the honest no-forecast path; no stand-in station is ever used.
  By venue: BER00 1, DUB00 1, FRA00 2, GER00 2, LON00 27, LON01 3, LON02 12, MAD01 1, MEX00 5, SAO00 2.
- Unresolved game ids: 2019_W18_BUF@HOU.
- Runs ABSENT FROM THE ARCHIVE (asked twice: the date-range endpoint omitted them
  and the single-run endpoint answered that it holds no results). Their games take
  the no-forecast path; no other run is substituted: KHOU GFS 2020-01-03T12:00:00.000.

## Bounds and rulings

| Bound | Limit | All compared | Ruling | Excluding the kickoff-hour subset | Ruling |
|---|---|---|---|---|---|
| temp_mean_signed_diff | within +/- 3.0 F | 0.9720 | PASS | 0.9626 | PASS |
| temp_mean_abs_diff | <= 5.0 F | 2.8216 | PASS | 2.8080 | PASS |
| temp_gross_miss_share | <= 0.01 (share off by more than 25.0 F) | 0.0000 | PASS | 0.0000 | PASS |
| wind_mean_signed_diff | between +1.0 and +4.0 mph | 2.2805 | PASS | 2.2864 | PASS |
| wind_mean_abs_diff | <= 6.0 mph | 3.0859 | PASS | 3.0845 | PASS |
| resolved_bulletin_share | >= 0.99 | 0.9998 | PASS | 0.9998 | PASS |

**VERDICT: PASS**

## The six statistics

Degrees F and mph; forecast minus observation.

| Set | n | Temp mean | Temp typical miss | Share > 25 F | Temp p95 | Temp max | Wind mean | Wind typical miss | Wind p95 | Wind max |
|---|---|---|---|---|---|---|---|---|---|---|
| All compared | 4793 | +0.97 | 2.82 | 0.0000 | 7.3 | 16.8 | +2.28 | 3.09 | 7.6 | 21.7 |
| Excluding the kickoff-hour subset | 4736 | +0.96 | 2.81 | 0.0000 | 7.3 | 16.8 | +2.29 | 3.08 | 7.6 | 21.7 |
| Kickoff-hour subset alone | 57 | +1.75 | 3.95 | 0.0000 | 11.7 | 16.4 | +1.79 | 3.20 | 7.9 | 15.2 |

## The kickoff-hour subset

68 Monday and Thursday night games of 2002-2005 are stored in silver with a 09:00 ET
kickoff, an AM/PM error in the feed that Plan 33.2-12 corrects;
57 of them are at covered venues (the rest are indoor). ERA5 was
fetched at that same wrong hour and the forecast hour is taken nearest the same wrong
kickoff, so the pair describes one instant and is compared. They are listed separately
so they cannot widen or hide anything: every bound was judged with and without them.

## By station

A bad station choice would show here as an outlier (RESEARCH assumption A2).

| Station | n | Temp mean | Temp typical miss | Wind mean | Wind typical miss |
|---|---|---|---|---|---|
| KATL | 18 | +3.27 | 3.46 | +0.67 | 2.19 |
| KBNA | 198 | +1.73 | 2.89 | +1.99 | 2.37 |
| KBOS | 219 | +1.12 | 2.71 | +4.54 | 4.64 |
| KBTR | 4 | -0.65 | 2.35 | -1.23 | 2.38 |
| KBUF | 195 | +0.96 | 2.38 | +2.45 | 3.11 |
| KBWI | 204 | +1.40 | 2.79 | +0.35 | 2.52 |
| KCLE | 193 | +1.38 | 2.83 | +0.71 | 2.25 |
| KCLT | 200 | +1.72 | 3.10 | +0.56 | 1.84 |
| KCMI | 8 | -1.51 | 4.19 | +2.98 | 3.09 |
| KCVG | 199 | +0.30 | 2.44 | +2.57 | 2.91 |
| KDAL | 57 | -0.20 | 2.79 | +1.71 | 2.34 |
| KDCA | 197 | +1.88 | 2.94 | +2.35 | 2.82 |
| KDEN | 206 | +1.01 | 3.73 | +4.61 | 5.07 |
| KDFW | 9 | +3.27 | 3.76 | +3.07 | 3.07 |
| KEWR | 394 | +2.24 | 2.90 | +3.23 | 3.68 |
| KFLL | 195 | +1.86 | 2.31 | +3.71 | 4.00 |
| KGRB | 207 | +0.08 | 2.29 | +1.61 | 2.21 |
| KHOU | 46 | +1.96 | 2.54 | +1.62 | 2.38 |
| KIND | 34 | +1.79 | 2.59 | +2.47 | 2.53 |
| KJAX | 188 | +0.52 | 2.64 | +2.08 | 2.82 |
| KLAX | 53 | -3.92 | 4.80 | +4.16 | 4.89 |
| KMCI | 211 | +0.73 | 2.84 | +3.35 | 3.57 |
| KMDW | 193 | +2.74 | 3.55 | +0.99 | 2.31 |
| KMSP | 18 | +1.60 | 2.88 | +3.05 | 3.06 |
| KOAK | 141 | -1.23 | 3.28 | +3.06 | 3.62 |
| KPHL | 211 | +0.95 | 2.40 | +2.62 | 3.00 |
| KPHX | 52 | +2.00 | 3.14 | +0.54 | 2.19 |
| KPIT | 207 | -0.47 | 2.81 | +1.50 | 2.33 |
| KSAN | 126 | -1.85 | 3.65 | +2.78 | 3.14 |
| KSEA | 208 | +0.65 | 2.29 | +0.84 | 2.92 |
| KSFO | 99 | -0.21 | 2.41 | +3.19 | 4.61 |
| KSJC | 103 | -0.03 | 2.65 | +2.51 | 3.34 |
| KTPA | 200 | +1.79 | 2.53 | +0.96 | 2.50 |

## Live versus history: a measured, accepted difference

Live 2026 games keep the Open-Meteo forecast service; history uses these bulletins. The
difference between the two was measured on 2026-09-15 at one common hour across 18
stadium stations: temperature mean +0.37 F (effectively
exact); wind mean +1.56 mph, bulletins higher, maximum gap
6.5 mph. The calm/moderate wind line is 5.0 mph, so on a calm
day a game can switch wind band between training and live serving, and the spread model
uses that band today. This is a known, measured, accepted train/serve difference, handed
to the re-fit plans (33.2-23).

## What this does not say

It does not say the forecasts were good. It says the decoded values sit where a correctly
decoded day-before forecast should sit relative to the observed weather, and nowhere a
unit, station, hour or join error would put them.
