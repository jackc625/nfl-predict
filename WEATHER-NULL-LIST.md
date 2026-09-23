# Games With No Day-Before Forecast (2002-2025)

Since Plan 33.2-12 (the p332_ gold ladder's rung 4), the weather every past game is trained on is the forecast as it stood at the game's lock: the 12 UTC National Weather Service MOS bulletin of the day before kickoff. Where weather applies to a game but no such forecast exists, the game carries NULL weather and a coverage flag (`weather_coverage` false) -- nothing observed afterwards is put in its place. This document lists those games and why each has no forecast. Games played in a fixed-roof dome are not on it: weather does not apply to them, so nothing is missing.

A retractable roof is treated as unknown at the lock (owner ruling 2026-09-21, the gold ladder's extra step 4b, Plan 33.2-14). Whether a retractable roof is open or closed is decided close to kickoff, often because of the weather, so the state it ended up in is not known at the lock -- a "closed" roof partly records that the weather turned bad. Every game at a retractable-roof stadium therefore carries the day-before forecast exactly as an outdoor game does, and the model is told only that the stadium's roof may close (the `venue_retractable` flag, read from the stadium, never from the game). Before step 4b, 620 games played with a retractable roof closed were treated as domes with no weather; they now carry their forecast, or NULL plus the flag where the forecast is missing.

## Current count

NULL-plus-flag games, 2002-2025: 57

For scale: 5412 games carry a forecast and 1030 were played in a fixed-roof dome.

**This document counts the 2002-2025 history only, and gold no longer does.** Plan 33.2-20's clean production build added the 17 played 2026 games, all of which reach gold with no forecast, so gold's uncovered-weather count is 74 where this list says 57. Both numbers are right and they describe different corpora: this list is the history the models train on, tied to silver over 2002-2025 by the guard named below. The 17 are not listed here because they are not the same fact -- 16 of them (2026 week 1) have no silver weather record at all, because the live ingest captured no week-1 forecast, and the seventeenth (`2026_W02_DET@BUF`) has a record whose forecast could not be dated, since it was captured two days after the game was played. The live capture cadence that produced both is Plans 33.2-27/28's; until it records a forecast issue time and stops capturing after kickoff, a 2026 game's weather is an honest unknown rather than a missing bulletin.

## By reason

| Reason | Games |
|---|---|
| Venue outside the United States | 56 |
| US venue, no resolved bulletin | 1 |
| Any other reason | 0 |

**Venue outside the United States.** The MOS bulletins are issued only for stations in the United States, Puerto Rico and the US Virgin Islands, so a game played abroad has no bulletin at all. No stand-in station is ever used. The venue is the one in force at the lock, so the seven 2025 games abroad are here at their real venues (Plan 33.2-09).

**US venue, no resolved bulletin.** One game. The one 12 UTC run confirmed absent from the archive, KHOU on 2020-01-03 (checked twice by Plan 33.2-11: the date-range endpoint omits it and the single-run endpoint answers "no results"), served `2019_W18_BUF@HOU` at NRG Stadium, a retractable-roof stadium. Until step 4b it was recorded as a closed-roof game because its roof was closed on the day; that state was not known at the lock, so weather applies to it and its forecast is honestly missing. No other model run is substituted.

## By season

| Season | Games |
|---|---|
| 2005 | 1 |
| 2007 | 1 |
| 2008 | 1 |
| 2009 | 1 |
| 2010 | 1 |
| 2011 | 1 |
| 2012 | 1 |
| 2013 | 2 |
| 2014 | 3 |
| 2015 | 3 |
| 2016 | 4 |
| 2017 | 5 |
| 2018 | 3 |
| 2019 | 6 |
| 2021 | 2 |
| 2022 | 5 |
| 2023 | 5 |
| 2024 | 5 |
| 2025 | 7 |

## Games outside the United States

| Game | Venue | Country |
|---|---|---|
| 2005_W04_SF@ARI | Estadio Banorte | Mexico |
| 2007_W08_NYG@MIA | Wembley Stadium | United Kingdom |
| 2008_W08_LAC@NO | Wembley Stadium | United Kingdom |
| 2009_W07_NE@TB | Wembley Stadium | United Kingdom |
| 2010_W08_DEN@SF | Wembley Stadium | United Kingdom |
| 2011_W07_CHI@TB | Wembley Stadium | United Kingdom |
| 2012_W08_NE@LA | Wembley Stadium | United Kingdom |
| 2013_W04_PIT@MIN | Wembley Stadium | United Kingdom |
| 2013_W08_SF@JAX | Wembley Stadium | United Kingdom |
| 2014_W04_MIA@LV | Wembley Stadium | United Kingdom |
| 2014_W08_DET@ATL | Wembley Stadium | United Kingdom |
| 2014_W10_DAL@JAX | Wembley Stadium | United Kingdom |
| 2015_W04_NYJ@MIA | Wembley Stadium | United Kingdom |
| 2015_W07_BUF@JAX | Wembley Stadium | United Kingdom |
| 2015_W08_DET@KC | Wembley Stadium | United Kingdom |
| 2016_W04_IND@JAX | Wembley Stadium | United Kingdom |
| 2016_W07_NYG@LA | Twickenham Stadium | United Kingdom |
| 2016_W08_WAS@CIN | Wembley Stadium | United Kingdom |
| 2016_W11_HOU@LV | Estadio Banorte | Mexico |
| 2017_W03_BAL@JAX | Wembley Stadium | United Kingdom |
| 2017_W04_NO@MIA | Wembley Stadium | United Kingdom |
| 2017_W07_ARI@LA | Twickenham Stadium | United Kingdom |
| 2017_W08_MIN@CLE | Twickenham Stadium | United Kingdom |
| 2017_W11_NE@LV | Estadio Banorte | Mexico |
| 2018_W06_SEA@LV | Wembley Stadium | United Kingdom |
| 2018_W07_TEN@LAC | Wembley Stadium | United Kingdom |
| 2018_W08_PHI@JAX | Wembley Stadium | United Kingdom |
| 2019_W05_CHI@LV | Tottenham Hotspur Stadium | United Kingdom |
| 2019_W06_CAR@TB | Tottenham Hotspur Stadium | United Kingdom |
| 2019_W08_CIN@LA | Wembley Stadium | United Kingdom |
| 2019_W09_HOU@JAX | Wembley Stadium | United Kingdom |
| 2019_W11_KC@LAC | Estadio Banorte | Mexico |
| 2021_W05_NYJ@ATL | Tottenham Hotspur Stadium | United Kingdom |
| 2021_W06_MIA@JAX | Tottenham Hotspur Stadium | United Kingdom |
| 2022_W04_MIN@NO | Tottenham Hotspur Stadium | United Kingdom |
| 2022_W05_NYG@GB | Tottenham Hotspur Stadium | United Kingdom |
| 2022_W08_DEN@JAX | Wembley Stadium | United Kingdom |
| 2022_W10_SEA@TB | Allianz Arena | Germany |
| 2022_W11_SF@ARI | Estadio Banorte | Mexico |
| 2023_W04_ATL@JAX | Wembley Stadium | United Kingdom |
| 2023_W05_JAX@BUF | Tottenham Hotspur Stadium | United Kingdom |
| 2023_W06_BAL@TEN | Tottenham Hotspur Stadium | United Kingdom |
| 2023_W09_MIA@KC | Deutsche Bank Park | Germany |
| 2023_W10_IND@NE | Deutsche Bank Park | Germany |
| 2024_W01_GB@PHI | Arena Corinthians | Brazil |
| 2024_W05_NYJ@MIN | Tottenham Hotspur Stadium | United Kingdom |
| 2024_W06_JAX@CHI | Tottenham Hotspur Stadium | United Kingdom |
| 2024_W07_NE@JAX | Wembley Stadium | United Kingdom |
| 2024_W10_NYG@CAR | Allianz Arena | Germany |
| 2025_W01_KC@LAC | Arena Corinthians | Brazil |
| 2025_W04_MIN@PIT | Croke Park | Ireland |
| 2025_W05_MIN@CLE | Tottenham Hotspur Stadium | United Kingdom |
| 2025_W06_DEN@NYJ | Tottenham Hotspur Stadium | United Kingdom |
| 2025_W07_LA@JAX | Wembley Stadium | United Kingdom |
| 2025_W10_ATL@IND | Olympiastadion | Germany |
| 2025_W11_WAS@MIA | Bernabeu | Spain |

## US games with no resolved bulletin

| Game | Venue | Why |
|---|---|---|
| 2019_W18_BUF@HOU | NRG Stadium | The KHOU 12 UTC run of 2020-01-03 is absent from the archive |

## How this list is kept honest

`tests/unit/test_weather_null_list_md.py` recomputes the count, the per-reason counts and the game list from the live silver `weather` table and fails if this document disagrees. The document records a ruling (NULL plus a flag, never a stand-in) and a count tied to the table, not a number to be kept by hand.
