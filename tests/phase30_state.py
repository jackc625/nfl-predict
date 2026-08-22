"""The git-TRACKED Phase-30 state manifest -- measured facts the phase's controls assert against.

WHY THIS MODULE EXISTS
----------------------
Phase 30's single most load-bearing proof is SPEC R2's fail-closed positive control for the
WR-06 leakage fix: after Plan 30-08's N-01 re-sync, every 2021-2024 gold feature value must be
byte-identical, and the re-sync must actually have re-synced something. That control is only a
control if it compares against a divergence measured BEFORE the re-sync -- afterwards the
number is unrecoverable -- and if it still asserts that number on a checkout that never ran the
phase.

The natural home for such a measurement is the run record, and Plan 30-04 Task 2 does write
one: ``outputs/n01/divergence_before.json`` and ``outputs/n01/odds_timeline_baseline.json``.
But ``outputs/`` is gitignored at ``.gitignore:26`` with only ``!outputs/.gitkeep`` at ``:27``
(``git check-ignore -v outputs/n01/divergence_before.json`` resolves to
``.gitignore:26:outputs/``), and the repository-wide ``*.json`` rule at ``.gitignore:250``
excludes them a second time. Those documents do NOT survive a fresh checkout.

For most generated state that is fine, and this repository has a deliberate precedent for
reading it behind a clean skip guard --
``tests/integration/test_activation_parity.py::_require_cache_and_backtest``, whose docstring
says it skips "so the suite is offseason / fresh-checkout safe". Those reads stay exactly as
they are.

The N-01 control is the one case where that pattern breaks, because the gitignored file is not
an INPUT to the proof -- it IS the proof's EXPECTED VALUE. Skip-guarding it would make Phase
30's most load-bearing assertion silently stop asserting anything on every checkout that did
not just run the phase, which is precisely the anti-pattern D30-06 rejects in Plan 30-03's own
words: a skipped test silently stops proving anything. Skip-guarding the control is not an
option; making its expected values DURABLE is.

``.planning/`` is gitignored too (``.gitignore:229``, and ``commit_docs`` is false), so a plan
SUMMARY is not a durable home either. The two homes that actually work in this repository are a
tracked module under ``tests/`` -- this file -- and the repo-root ``GATED-REFIT-READOUT.md``
that Plan 30-13 publishes. This module is the machine-readable half; the readout is the
human-readable half, and Plan 30-13 publishes the whole manifest into it.

The ``outputs/`` scratch documents are still written exactly as before. They are the run record
and carry the full per-season, per-week and per-id detail that does not belong in a constants
module. What changed is only that the committed controls no longer DEPEND on them: a control
reads the tracked constant here, and cross-checks the scratch document against it when the
scratch document happens to be present. A disagreement between the two is itself a failure --
a regenerated scratch file must never silently supersede the committed expectation (T-30-52).

CONSTRAINTS ON THIS MODULE
--------------------------
Constants only. NO project imports, NO I/O and NO logic, so any test at any tier can import it
without cost or side effects. Every value below was MEASURED, never transcribed, and carries
the plan, task and date that measured it.

APPEND PROTOCOL
---------------
Each slot is APPENDED ONCE by its owning plan and is NOT edited afterwards. A later plan that
needs a new durable value appends a new slot here rather than inventing a second home:

* Plan 30-04 (this file's author) -- ``N01_PARQUET_ROWS_BEFORE``, ``N01_DB_ROWS_BEFORE``,
  ``N01_DIVERGENCE_BEFORE``, ``N01_MISSING_GAME_IDS_SHA256``, ``ODDS_TIMELINE_ROWS``,
  ``ODDS_TIMELINE_PAIRS``, ``ODDS_TIMELINE_SEASON_COUNTS``,
  ``ODDS_TIMELINE_PAIR_LIST_SHA256``, ``GOLD_2025_ROWS_BEFORE_RESYNC``.
* Plan 30-08 -- ``SLICE_DIGESTS_2021_2024`` (the per-matrix, per-column 2021-2024 slice digests
  measured BEFORE the re-sync) and ``ACCEPTED_RUNG4_FINGERPRINT_SHA256``.
* Plan 30-10 -- ``PRE_REGISTRATION_COMMIT``, ``MEASUREMENT_COMMIT`` and
  ``GROUP_VERDICT_FILE_SHA256``.
* Plan 30-11 -- ``MANIFEST_SHA256_BEFORE`` and ``MANIFEST_SHA256_AFTER``.

Plan 30-13 publishes the whole manifest into ``GATED-REFIT-READOUT.md``, so every value has a
second, human-readable tracked home.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# N-01: the DuckDB mirror of silver `games` that fell behind its parquet.
#
# MEASURED by Plan 30-04 Task 2 on 2026-08-22, BEFORE Plan 30-08's re-sync, via
#   load_dataframe("games", layer="silver", source="parquet")   -> N01_PARQUET_ROWS_BEFORE
#   load_dataframe("games", layer="silver", source="db")        -> N01_DB_ROWS_BEFORE
# (note the source literal is "db"; "duckdb" raises ValueError at data/storage.py:970).
#
# The divergence is STRICTLY POSITIVE and the capture asserted so at capture time. A zero
# divergence would mean someone had already re-synced, leaving the Plan 30-08 positive control
# with nothing to prove -- a control that can be satisfied by doing nothing is not a control.
# ---------------------------------------------------------------------------
N01_PARQUET_ROWS_BEFORE: Final[int] = 6499
N01_DB_ROWS_BEFORE: Final[int] = 6292
N01_DIVERGENCE_BEFORE: Final[int] = 207

# sha256 over the newline-joined, sorted list of the game_ids present in the parquet copy and
# absent from the DuckDB copy. All 207 are season 2025. The full list lives in
# outputs/n01/divergence_before.json; this digest is what makes that list checkable after the
# scratch document is gone.
N01_MISSING_GAME_IDS_SHA256: Final[str] = (
    "40d871d75da81616744e95cb5fd82da77273f1e08c17f3cfa5e6f4c9a08bd9bd"
)

# Gold's 2025 slice is BEHIND BOTH copies of silver `games` -- gold carries weeks 1-4 while the
# DuckDB table has weeks 1-5 and the parquet has weeks 1-22. This is a FINDING, not a problem,
# and it is why SPEC R2's two clauses are about two DIFFERENT objects: gold's 2025 coverage is
# bounded by the other silver sources in the join, not by `games` alone. Plan 30-08's control
# must therefore NOT be written as "gold grew by 207" -- empirically it will not.
GOLD_2025_ROWS_BEFORE_RESYNC: Final[int] = 49
GOLD_2025_WEEKS_BEFORE_RESYNC: Final[tuple[int, ...]] = (1, 2, 3, 4)
SILVER_PARQUET_2025_WEEKS_BEFORE_RESYNC: Final[tuple[int, ...]] = tuple(range(1, 23))
SILVER_DB_2025_WEEKS_BEFORE_RESYNC: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)

# ---------------------------------------------------------------------------
# The paid odds_timeline archive (T-30-12, SPEC R3).
#
# MEASURED by Plan 30-04 Task 2 on 2026-08-22 through
# load_dataframe("odds_timeline", layer="silver") -- deliberately NOT by reading the parquet
# path, because load_dataframe(source="auto") resolves DuckDB first and only falls back to
# parquet, so this is a statement about what the PIPELINE sees.
#
# 9,957 rows bought with real Odds API credits (Plan 29-05) and re-keyed in place by quick task
# 260816-u0e. Not re-derivable from anything in this repository.
#
# NOTE the integrity gate is asserted from CONTENT, never from `git status --porcelain data/`,
# which is vacuous: .gitignore:22 is `data/`, so porcelain returns empty whether the archive is
# intact or destroyed (N-03).
# ---------------------------------------------------------------------------
ODDS_TIMELINE_ROWS: Final[int] = 9957
ODDS_TIMELINE_PAIRS: Final[int] = 9957
ODDS_TIMELINE_SEASON_COUNTS: Final[dict[int, int]] = {
    2020: 1780,
    2021: 1219,
    2022: 1452,
    2023: 2759,
    2024: 2747,
}

# sha256 over the newline-joined, sorted "game_id\x1fsnapshot_ts" pair list. The counts can all
# match while the CONTENT has moved; this digest is what catches that.
ODDS_TIMELINE_PAIR_LIST_SHA256: Final[str] = (
    "c4e313a788799b8bfbae5b85169c190101684b155b44e3ea28c9b2c4a0e82a2f"
)

# -------------------------------------------------------------------------
# The 2021-2024 gold slice digests, measured BEFORE Plan 30-08's N-01 re-sync.
#
# APPENDED by Plan 30-08 Task 1 on 2026-08-22, from rung-3 gold (194/195/194, 6,263
# rows), through scripts.resync_games_duckdb.slice_digests -- which reuses
# scripts.fingerprint_gold._column_bytes, the ONE canonical byte encoding every
# fingerprint document in this phase was written with. Keyed matrix, then column;
# each value is a full sha256 over that column's 2021-2024 rows ordered by game_id.
#
# THESE ARE THE PRE-RE-SYNC VALUES SPEC R2's BYTE-IDENTITY CONTROL ASSERTS AGAINST.
# After the re-sync they are unrecoverable, and outputs/n01/slice_hash_before.json --
# the run record carrying the same numbers -- is gitignored at .gitignore:26. Without
# this tracked pin the control would have to be skip-guarded on a file that is absent
# on every fresh checkout, which is the one way the phase's most load-bearing proof
# could silently stop proving anything.
#
# A moved digest means a 2025 re-sync moved a 2021-2024 value: the WR-06 fix is
# INCOMPLETE, a whole-frame statistic is still reaching backwards, and the phase is
# BLOCKED. The comparison is EXACT equality -- there is no tolerance, and no float is
# ever re-formatted on either side.
#
# game_id is itself one of the hashed columns, so a slice whose membership or ordering
# moved shows up as a moved game_id digest rather than as an unexplained value move.
# -------------------------------------------------------------------------
SLICE_DIGESTS_2021_2024: Final[dict[str, dict[str, str]]] = {
    "features_ats": {
        "apparent_temp_f": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_abs_timezone_diff_hours": "9298b91383149bea4a84a815b7839eada3bc3da5cd5a53f0aaf35246ce3f0d52",
        "away_availability_coverage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_availability_fraction": "edf3766125d94a222895b9e220b9b230de6b905c2adae633bd4a72adc42d32b5",
        "away_backup_quality_delta": "e8c35db24be24d75ee99e6b2bfb89ca7998b5c21bfd5d9cb4947ac69af54d32d",
        "away_cross_country_travel": "121c7d1584f7134449865599de6c736f7a91bede760cfe29a8d9d3851701fa28",
        "away_date_modified_coverage": "f0439f5325346ae1aba66085628dc1d48591e346ce7facdb46c8d3694535c5c8",
        "away_def_rolling_avg_drive_start_yardline": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_def_rolling_cpoe": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_def_rolling_neutral_pace": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_def_rolling_neutral_pass_rate": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_def_rolling_opp_adj_epa_per_play": "41ec849ae547aa41fde7cfd5e5c341542c997716aa214c99e7a24257ceef7c99",
        "away_def_rolling_opp_adj_pass_epa": "189034e3670d6ddc3c6b9da7a9f1b430c863434105cb6aea2b16012b75e69205",
        "away_def_rolling_opp_adj_rush_epa": "3b7c7c33f3a656582b26db960b372c9d5f9a6947d1a82691afdbad0812d8cb0b",
        "away_def_rolling_pass_success_rate": "ec0485d6930f6dbef4a3cd674de05445b029c07a78a5487d64d8c29649ecd18e",
        "away_def_rolling_red_zone_td_rate": "c2ff5847f5b79ac8dd8959ac598883c754d4758bb542d8984f86ed2f9a7644a4",
        "away_def_rolling_rush_success_rate": "b5e3cf1ae3cefad975f746bbb9dd4f3c5799ee6d825eff8dd94654eeecc9550b",
        "away_def_rolling_success_rate": "b3e965909f429c7cbb6c4539fa0f2136758e2e13c3e2a1b351b6cbbc543b1edd",
        "away_def_rolling_third_down_conversion_rate": "454ae14ea0a7094c9cd91b2aa000ae460ed6509ae7f59e34df6c4153b4098d23",
        "away_eastward_travel": "f2209d292977d4be78e62ed18ebb8565e9862dd92c2f8f96b9404e01288a4379",
        "away_elo": "c2ef4def8dcc974fe2f245ccd44ff206b092111eed46d40b42fedc728d08a64b",
        "away_elo_momentum": "e97f0c24eac7bdc055b281c4873b3e6d359c3227347e54e276cf7d6de9552e5c",
        "away_elo_percentile": "73d1b8663ac23009bd6c655faa3970142f869f3935de9a4822e306e99f1f7982",
        "away_elo_rank": "b260160d6b73b777b448d7454c86ba9fbf77de2b31eb778880e2956d89805832",
        "away_elo_uncertainty": "e865720516218cd9e34167196104913fc43c96aac7cb2deb70cfbc5f62757813",
        "away_injury_coverage": "f0439f5325346ae1aba66085628dc1d48591e346ce7facdb46c8d3694535c5c8",
        "away_letdown_spot": "4aef6b41f49642a9c0cf39eda2a5a9c897cd1d0bb5b5a4f132bfe401a1559313",
        "away_look_ahead_spot": "0c545e2bcdaf14adf75b4917011b11fbed73b766842163256f9d7bfb333f48ec",
        "away_off_bye": "12637ef4fc146aa9057993c944968591c94fd4372cf6e8d22595d3fd651fdff6",
        "away_off_rolling_avg_drive_start_yardline": "eda62e6afb22f92c32d25badf4584752b2ca6c184f843f45c203536d840ad6a4",
        "away_off_rolling_cpoe": "bbb9dd6a4f75d45dab642db5a825966ecc56988e44087b46643dcfdacb43da54",
        "away_off_rolling_neutral_pace": "ecd05ca440650eb2b66d0932a1bd2904568929b478c4e04a7d76ba48e8a99375",
        "away_off_rolling_neutral_pass_rate": "906e5ad9fb67018432e1d6cf31d0cd0902158a4c731743b086d0388ed432a522",
        "away_off_rolling_opp_adj_epa_per_play": "6a46be42ea089003c530c0a783e63d3fed1193f7105ed996db7663d638055a8c",
        "away_off_rolling_opp_adj_pass_epa": "0595c811202d3e3330c3047d96b943d27f6394d095e91ab3f845df14b130e8d5",
        "away_off_rolling_opp_adj_rush_epa": "45125fa970cce5449575fba71d36b3ffbebc3c751aad618fee47e347d2726b46",
        "away_off_rolling_pass_success_rate": "7453474ed6c14aca08839c675da51b8807bfcca319a7485bd892e94d651be7bf",
        "away_off_rolling_red_zone_td_rate": "ecdb2bc318b591ff23feb6451ff0a8ea519ffe9b9de2943a02b0f752deae256d",
        "away_off_rolling_rush_success_rate": "1a8123c62f565a4ef0b0e4c276125059c60e18954dbbbdf7a718cf88654eeb1f",
        "away_off_rolling_success_rate": "5a7b05cdcfc368929f909b65288bf1c6c8c229af51ad04950a4bb3153e03c930",
        "away_off_rolling_third_down_conversion_rate": "40dbe65189decab6ae3fd12fe29490745cc2857cb419e2055aef67b0702ee763",
        "away_qb_adjustment": "7d7efedb80737d36c78e650309faded4958438e8ea21fb20d6454d5f03277072",
        "away_qb_out_flag": "aea6c71f09c93cc4ba188238803b353028ecc721af6d95aabd95348c1e178dcf",
        "away_rest_days": "1288338247489667d2dc2ba0f78f6134855e4dd137edb023999a408b50e8651e",
        "away_rolling_snap_share_db": "1139cac27a078bf4332061f9c0cabf0f36f0e3ffc99b5e7064e7dbff893e47b4",
        "away_rolling_snap_share_dl": "b3e6cffef405808fd00a21d4ca21cb0aa819fd1c5850e4b37c452bff6cbd0f68",
        "away_rolling_snap_share_lb": "e0f9a76d91baf6cc65dcbffd003a5777bde2e1b385b04a459658bbbb05273959",
        "away_rolling_snap_share_ol": "aeea128a406edcdaf37404d285e9d1e0389406942416c6b9baffeea4e34ba3ff",
        "away_rolling_snap_share_qb": "c67c669e1a7052af541668b6620122a558a8263a7a470d2046588b2ea0d5374d",
        "away_rolling_snap_share_rb": "6178b4392fc731106aaaa1c23048f6dfc7bfb91fb302c9826921500d5d2286e6",
        "away_rolling_snap_share_te": "50d803f60b28753d6f1f46ff1ef4bcadbc1c71438428b0317378241cbe8b9731",
        "away_rolling_snap_share_wr": "cc52630a4cc913eb88e97881355ed6a99de587684518466467bf0b5f55bab959",
        "away_score": "db9d7b93547fba37f72a43b6be88458b8e221ade5c5caabadd935717d4fe446f",
        "away_short_rest": "710996bb06a85fd99bf78994e4a87cf0fa80e38523024464ab9a2d63ad41c583",
        "away_snap_concentration": "63bb7f4de6782ced96ae58148dc4dd0372af526084448bb97278ef674b852cc2",
        "away_snap_continuity": "24313595ec16acfff2c55d69e4315121a5550c9dd9389751466c8491ab4ef86c",
        "away_timezone_diff_hours": "ca92dfa6cbbbe6fca4cfada23e07ab6e61b7cf2580d55841c70dba751ead4d33",
        "away_travel_distance_miles": "e5c07a502657ba7207782745b71cadf88f6dc1a2a58ce5a0cb2835cb4b57a856",
        "away_travel_fatigue_score": "99173759915b7f22c063be8078bddc23164112ce15a16d94a2d806589bd097ec",
        "away_westward_travel": "e0c28faf0cc69ce86e11009258e46c7e9e176055ec182876937113bcbdbb5a8d",
        "ball_handling_difficulty": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "both_short_rest": "ecb74c37f08eaeec1816689565fba64a52771ee526150296a5a88884edd022e2",
        "cold_impact_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "defensive_advantage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "elo_diff": "8f51cc89fd8b6142bc3616291946998512cdc0ed282591d9f4803086674ff81b",
        "elo_prob_away": "9080107254cc718b3971de293773b14340a1bc2a2d8ce64e07411f7b1c6cf631",
        "elo_prob_home": "cff1d616a91eb41635b686e18ef78aba2138fd6b32f109ea95a137821699b19e",
        "extreme_weather": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "feature_timestamp": "941e7978e1888b6249cfd13450561668115cb78fd2e2e0bc76893e11d6a4a998",
        "game_day_of_week": "4d3269d59f59072c168c45c75d7a860225fa5234d751f6e376056ada98eef148",
        "game_id": "4e551a4356a9ba78ec9691ac189fa384537915b33c38b3aef9b4c4aa95580149",
        "heat_impact_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "hfa_used": "3e4d5f64c7edb1d99ee6157561214d5bdf872a93d752cf206937f0dc31d4808c",
        "home_availability_coverage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_availability_fraction": "bf72c59985d74e6afd2ad4327965fdf85aad00ec9ec463be2b9d669ae7f06237",
        "home_backup_quality_delta": "fc17b9e5c3fb311752db4948654ab5f4242b400c297b908f99ca5599a3b05e66",
        "home_date_modified_coverage": "f0439f5325346ae1aba66085628dc1d48591e346ce7facdb46c8d3694535c5c8",
        "home_def_rolling_avg_drive_start_yardline": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_def_rolling_cpoe": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_def_rolling_neutral_pace": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_def_rolling_neutral_pass_rate": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_def_rolling_opp_adj_epa_per_play": "3054082acc51cbd36722b6a477950b735862dbc12f14ed0ef20253152c7d1511",
        "home_def_rolling_opp_adj_pass_epa": "5fcf33055c431e8dcf74e64a361b1687417b196b151d35c3614cf362301de0eb",
        "home_def_rolling_opp_adj_rush_epa": "e3c07100fc7db376a4d3097e853bdae82806b66e6582a20b2a1075faa06e58fb",
        "home_def_rolling_pass_success_rate": "b069136cb59a888411a1f82172f2249d0cb9670cd60d11a01b0a330452db522e",
        "home_def_rolling_red_zone_td_rate": "939e328d2403e77b7b2cbee0ef56d51add0119fb101c8f4de7b1d1c0c23fd923",
        "home_def_rolling_rush_success_rate": "08d3906e347628a51c2e420bcaa549860547523b786c85e5ef3f76b5a3c7a1fd",
        "home_def_rolling_success_rate": "2669ecf0e8ebfcff0f58193e2153a553c40da42d971f04b030bc4655de75ff10",
        "home_def_rolling_third_down_conversion_rate": "1795b9bfa30b3a9cfb43e9595e77547f27cc9a72deee4491c9dab52d3f585c32",
        "home_elo": "0e8f5ae04da40c3625d1be68370c2b25070303f46b6f23c4c42320391ba24422",
        "home_elo_momentum": "6a9e33c190925fa175b37e5798be9a7fe1f524ebd4d569f40f7e6e4306554c3f",
        "home_elo_percentile": "e1e3f734372ddf214bc3cea0b2e4483b4e6f3ad3fa997a9a58440f1849232ea9",
        "home_elo_rank": "ed7f0a7f3331423d7fe597db4820654603874d8e4b1caaeddb5ca1177f890b31",
        "home_elo_uncertainty": "c2e8273da7b3035871cd84faf5288fd2b3de1b3a6ee128148d82a9da073f2180",
        "home_injury_coverage": "f0439f5325346ae1aba66085628dc1d48591e346ce7facdb46c8d3694535c5c8",
        "home_letdown_spot": "2ae9a7589cfb85f9d77e6604d85bdb2557e0a09d61bf3b053038d5b40c5e9335",
        "home_look_ahead_spot": "0c42d74e49029bd78bbe6d734b4a9261cfd8b14eed7039bd5cf03c7f64cdf520",
        "home_margin": "96597c79c3f413acf39ba84b36aa7f7fd0518fb593ae34b8a2c79da52b2e6e3b",
        "home_off_bye": "e3224454b8c1fe3b8b16914da962b3c7ddfcfc5d60ace961dbd3da43a4fdbe4b",
        "home_off_rolling_avg_drive_start_yardline": "3f0434258e447905ab38b82ed9e4b17b3abe1d879631adc1564d9de18c892aaa",
        "home_off_rolling_cpoe": "ff23212357e79992ecd488961432a1ea75153baca67c1a417125789ec051ea36",
        "home_off_rolling_neutral_pace": "e518076cec70b6b395b98f7415b098fe58d643ccfde0321f57ca570bf99ff5b1",
        "home_off_rolling_neutral_pass_rate": "f38289f59c44f8c9e31f04502db58c95f54886085b3380b5b9aad220b29d7a7e",
        "home_off_rolling_opp_adj_epa_per_play": "1300409c1b4e9d9db81759ca4a7a16dcbe9536f4867e03940b827b6f1f31d0c9",
        "home_off_rolling_opp_adj_pass_epa": "e689650446c0805595f2201c2294be2f8e136e999ffce302759f88787e76a573",
        "home_off_rolling_opp_adj_rush_epa": "4991f47f31ee5a403478e0d3cdd18abc483d9fbee6d641300e4cbf73b360fa62",
        "home_off_rolling_pass_success_rate": "e76130f8988a61896d7a5171c63f81bc65e98261751704cc23a2769d1260755e",
        "home_off_rolling_red_zone_td_rate": "a14f5331ccc359dbb9897eff1d18f336a480a1b9352416edbb0e79b744ff6005",
        "home_off_rolling_rush_success_rate": "890cc125f8921767b2bde73203f03946eb46da944adf82e179e19d5be974bf4f",
        "home_off_rolling_success_rate": "d5ccc94f40a13f67dfab77fbab35fd5d0b667cc473042b605da052d55d492808",
        "home_off_rolling_third_down_conversion_rate": "a2ce87e20e4435f6a472ca57903f75ead60861a07731ef727216e4430bd6728a",
        "home_qb_adjustment": "1cd5b4f4b05d6cc49244cf50aabbb1706431e6da0b6ccfeacd0bcba4a331c83e",
        "home_qb_out_flag": "446990838a8f7bbe097216cec01bcf1b49769c33f6692c09f2cc480d7357bde4",
        "home_rest_days": "dd97a5f969b49360d3c3bc95227a7d3efccbbdc99890265b38f3fb280d29d004",
        "home_rolling_snap_share_db": "15c72682b0d76f99fdf5454b591da38b69a5d09bdc1365ebe9252fce0bb1f6da",
        "home_rolling_snap_share_dl": "c458ed422beed5be3318877652eea5bc6cc458a780dfd526b6a29027aa967007",
        "home_rolling_snap_share_lb": "55d5d5973edb9251fe1db359e42c8e3176d71e3aa9e67257e646ffc3d2c57b55",
        "home_rolling_snap_share_ol": "510b1bc599c52b253a8952a218be6901a5620015815a77457eba0c404f6c08b7",
        "home_rolling_snap_share_qb": "c854a75f1d1f8394c3c05c1f4f41989fd15f92e49f371baf2dc51faeb9aac077",
        "home_rolling_snap_share_rb": "842a4d25c110a06e504c25afe75e0943ed4a4adcefecf74425efc67ee8521724",
        "home_rolling_snap_share_te": "353756a5ae9a37dc36b2d73295d93e860e2645b5cf5ca5a45c7830566dd39db0",
        "home_rolling_snap_share_wr": "6c422d6e3ac719400fa904daf8c90299e7ebc0dd5b0e2b7ad081be02ef76ff94",
        "home_score": "d85fcb7788736fb928a6a8c9204ff1f20d4926ae0be324ecbef010ca09fa3e08",
        "home_short_rest": "a3a306d1b284b1106ad333d8ac1fe147a2c3faa9aaec0e0fc4f5867444c32f7d",
        "home_snap_concentration": "de5d791dde3879f0e23b1b7d9c1a0340d3654524bb6a785115d5ce9c5b7f3052",
        "home_snap_continuity": "f3467d4b14fe5c91ab35be5ce1b181d1288f9a8fbccd7382b6cacd9deb133e4a",
        "home_weather_advantage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_away_game": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_divisional": "62249d2f8dae907262bd1636fc9ac491d6dc8218c781ca740205b87a0c455e25",
        "is_dry": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_home_game": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_rain": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_snow": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "kicking_difficulty": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "late_season": "2966195be17179438ec44309c617463e4170534e67d8a4d868dce7d53d9f963a",
        "monday_game": "a5c20f31e7c86f8b210cb7eaa2d8d911fbef4b75df33d2f7d85616a179797b2f",
        "passing_difficulty": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "passing_efficiency": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "point_differential": "96597c79c3f413acf39ba84b36aa7f7fd0518fb593ae34b8a2c79da52b2e6e3b",
        "precip_heavy": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_impact_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_light": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_mm": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_moderate": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_none": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_prob": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "raw_humidity_pct": "56242ca08930588005aebed3ba8211b3ba6e93c22befce6a6c869415f2e7a63e",
        "raw_precip_mm": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "raw_precip_prob": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "raw_temp_f": "e33153e4acb315c8d6d0a060799ce0729ce9f490142e97ba5fd522bbbbef6050",
        "raw_weather_severity": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "raw_wind_mph": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "rest_advantage": "33620dd9d93f7d0b58e4943ca2c034ca5d1ec1b758d3710c8bdbf96dec40be82",
        "rushing_advantage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "saturday_game": "c2e3d73d37feba2904b69b0bf9a929969e0e3b39ef8ec9ccb229ddb0f09a22a3",
        "scoring_multiplier": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "scoring_reduction": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "season": "72c1be438e384477c751a772ad4fb6698b19b7d2132f7f0c225295aee78801cb",
        "season_progress": "51db062043eea7773b17197809316cb93fb267a803d357477b855e5f41a5462b",
        "short_week": "f4581839b429eacf1171524b7f8d0450240b0a0dd49d454413e71abcb2b729c4",
        "snapshot_ml_prob_home_fair": "97e544dd4f08d3afc239e74d5748e78898417f117dfc8356b3d13524ba90ec5d",
        "snapshot_spread": "4e68d2f9dae46e7eb19472deeb672d0fa20b81c25e20c55d96dab4e8ef216ff7",
        "snapshot_total": "14cba98c6419162fbe995c20cdadd9b27b11d2357bfb87275b4a0234a5aa3c91",
        "spread_movement": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "surface_mismatch": "78fc0d1e15505b75aadcf39332a625cc27b62c8d252ef62ebe5a012023dffa5b",
        "target_ats": "ecbfc99fa44e9dfe63bae84deff1228022634b27933706f8af1ce2692980ca73",
        "temp_cold": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_cool": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_f": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_hot": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_mild": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_very_cold": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_warm": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "thursday_game": "e537eda90e6d7c85a8d005bafe1f8f1bf779e6e49ecb655997b0f49782bc96e3",
        "total_movement": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "turnover_multiplier": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "venue_capacity": "05e21984e1ed7bb4293d400454fa77322211dbf8614c034acda3bca3f9004fef",
        "venue_cold_climate": "7716537887125989977b412df650356a8fb35fd0030d955141ce8961bf6a81ed",
        "venue_elevation_ft": "326f54ae32997eb22998d81392718df7dd890e05b63281602f5b970166a1951a",
        "venue_high_altitude": "0608a0758e7b242e61bfdbc6033ea745d9694a8310937cdee218d4504f9d13a4",
        "venue_indoor": "d4e9d557ec3df78e69859869277948d204c24fbb4e613abd2f08698806489069",
        "venue_large_stadium": "f4d13a21a9fac9b646b57a01b0c5fecd01db662b3cd590a3c882b0d5f09b69f7",
        "venue_outdoor": "155f9881faf43d5c8a19e8d3ee99d14ae76b1e458951e68472c944a1a7bffa8c",
        "venue_retractable": "ab85732c061ef76c2c9b8af559f97128452c3ee236336ff1595a5cb688827a43",
        "venue_warm_climate": "dcddff60e74133ca18756f27a7d60a89b1c2ec3c25a23e3473ded59cc1d55717",
        "weather_affects_game": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "weather_game": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "weather_severity_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "week": "4690c9f30f7346ea80e8f6efa70a53d0416710f9f10bc0ed998615d6cc969823",
        "wind_calm": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_high": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_impact_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_moderate": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_mph": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_severe": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
    },
    "features_ou": {
        "apparent_temp_f": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_abs_timezone_diff_hours": "9298b91383149bea4a84a815b7839eada3bc3da5cd5a53f0aaf35246ce3f0d52",
        "away_availability_coverage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_availability_fraction": "edf3766125d94a222895b9e220b9b230de6b905c2adae633bd4a72adc42d32b5",
        "away_backup_quality_delta": "e8c35db24be24d75ee99e6b2bfb89ca7998b5c21bfd5d9cb4947ac69af54d32d",
        "away_cross_country_travel": "121c7d1584f7134449865599de6c736f7a91bede760cfe29a8d9d3851701fa28",
        "away_date_modified_coverage": "f0439f5325346ae1aba66085628dc1d48591e346ce7facdb46c8d3694535c5c8",
        "away_def_rolling_avg_drive_start_yardline": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_def_rolling_cpoe": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_def_rolling_neutral_pace": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_def_rolling_neutral_pass_rate": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_def_rolling_opp_adj_epa_per_play": "41ec849ae547aa41fde7cfd5e5c341542c997716aa214c99e7a24257ceef7c99",
        "away_def_rolling_opp_adj_pass_epa": "189034e3670d6ddc3c6b9da7a9f1b430c863434105cb6aea2b16012b75e69205",
        "away_def_rolling_opp_adj_rush_epa": "3b7c7c33f3a656582b26db960b372c9d5f9a6947d1a82691afdbad0812d8cb0b",
        "away_def_rolling_pass_success_rate": "ec0485d6930f6dbef4a3cd674de05445b029c07a78a5487d64d8c29649ecd18e",
        "away_def_rolling_red_zone_td_rate": "c2ff5847f5b79ac8dd8959ac598883c754d4758bb542d8984f86ed2f9a7644a4",
        "away_def_rolling_rush_success_rate": "b5e3cf1ae3cefad975f746bbb9dd4f3c5799ee6d825eff8dd94654eeecc9550b",
        "away_def_rolling_success_rate": "b3e965909f429c7cbb6c4539fa0f2136758e2e13c3e2a1b351b6cbbc543b1edd",
        "away_def_rolling_third_down_conversion_rate": "454ae14ea0a7094c9cd91b2aa000ae460ed6509ae7f59e34df6c4153b4098d23",
        "away_eastward_travel": "f2209d292977d4be78e62ed18ebb8565e9862dd92c2f8f96b9404e01288a4379",
        "away_elo": "c2ef4def8dcc974fe2f245ccd44ff206b092111eed46d40b42fedc728d08a64b",
        "away_elo_momentum": "e97f0c24eac7bdc055b281c4873b3e6d359c3227347e54e276cf7d6de9552e5c",
        "away_elo_percentile": "73d1b8663ac23009bd6c655faa3970142f869f3935de9a4822e306e99f1f7982",
        "away_elo_rank": "b260160d6b73b777b448d7454c86ba9fbf77de2b31eb778880e2956d89805832",
        "away_elo_uncertainty": "e865720516218cd9e34167196104913fc43c96aac7cb2deb70cfbc5f62757813",
        "away_injury_coverage": "f0439f5325346ae1aba66085628dc1d48591e346ce7facdb46c8d3694535c5c8",
        "away_letdown_spot": "4aef6b41f49642a9c0cf39eda2a5a9c897cd1d0bb5b5a4f132bfe401a1559313",
        "away_look_ahead_spot": "0c545e2bcdaf14adf75b4917011b11fbed73b766842163256f9d7bfb333f48ec",
        "away_off_bye": "12637ef4fc146aa9057993c944968591c94fd4372cf6e8d22595d3fd651fdff6",
        "away_off_rolling_avg_drive_start_yardline": "eda62e6afb22f92c32d25badf4584752b2ca6c184f843f45c203536d840ad6a4",
        "away_off_rolling_cpoe": "bbb9dd6a4f75d45dab642db5a825966ecc56988e44087b46643dcfdacb43da54",
        "away_off_rolling_neutral_pace": "ecd05ca440650eb2b66d0932a1bd2904568929b478c4e04a7d76ba48e8a99375",
        "away_off_rolling_neutral_pass_rate": "906e5ad9fb67018432e1d6cf31d0cd0902158a4c731743b086d0388ed432a522",
        "away_off_rolling_opp_adj_epa_per_play": "6a46be42ea089003c530c0a783e63d3fed1193f7105ed996db7663d638055a8c",
        "away_off_rolling_opp_adj_pass_epa": "0595c811202d3e3330c3047d96b943d27f6394d095e91ab3f845df14b130e8d5",
        "away_off_rolling_opp_adj_rush_epa": "45125fa970cce5449575fba71d36b3ffbebc3c751aad618fee47e347d2726b46",
        "away_off_rolling_pass_success_rate": "7453474ed6c14aca08839c675da51b8807bfcca319a7485bd892e94d651be7bf",
        "away_off_rolling_red_zone_td_rate": "ecdb2bc318b591ff23feb6451ff0a8ea519ffe9b9de2943a02b0f752deae256d",
        "away_off_rolling_rush_success_rate": "1a8123c62f565a4ef0b0e4c276125059c60e18954dbbbdf7a718cf88654eeb1f",
        "away_off_rolling_success_rate": "5a7b05cdcfc368929f909b65288bf1c6c8c229af51ad04950a4bb3153e03c930",
        "away_off_rolling_third_down_conversion_rate": "40dbe65189decab6ae3fd12fe29490745cc2857cb419e2055aef67b0702ee763",
        "away_qb_adjustment": "7d7efedb80737d36c78e650309faded4958438e8ea21fb20d6454d5f03277072",
        "away_qb_out_flag": "aea6c71f09c93cc4ba188238803b353028ecc721af6d95aabd95348c1e178dcf",
        "away_rest_days": "1288338247489667d2dc2ba0f78f6134855e4dd137edb023999a408b50e8651e",
        "away_rolling_snap_share_db": "1139cac27a078bf4332061f9c0cabf0f36f0e3ffc99b5e7064e7dbff893e47b4",
        "away_rolling_snap_share_dl": "b3e6cffef405808fd00a21d4ca21cb0aa819fd1c5850e4b37c452bff6cbd0f68",
        "away_rolling_snap_share_lb": "e0f9a76d91baf6cc65dcbffd003a5777bde2e1b385b04a459658bbbb05273959",
        "away_rolling_snap_share_ol": "aeea128a406edcdaf37404d285e9d1e0389406942416c6b9baffeea4e34ba3ff",
        "away_rolling_snap_share_qb": "c67c669e1a7052af541668b6620122a558a8263a7a470d2046588b2ea0d5374d",
        "away_rolling_snap_share_rb": "6178b4392fc731106aaaa1c23048f6dfc7bfb91fb302c9826921500d5d2286e6",
        "away_rolling_snap_share_te": "50d803f60b28753d6f1f46ff1ef4bcadbc1c71438428b0317378241cbe8b9731",
        "away_rolling_snap_share_wr": "cc52630a4cc913eb88e97881355ed6a99de587684518466467bf0b5f55bab959",
        "away_score": "db9d7b93547fba37f72a43b6be88458b8e221ade5c5caabadd935717d4fe446f",
        "away_short_rest": "710996bb06a85fd99bf78994e4a87cf0fa80e38523024464ab9a2d63ad41c583",
        "away_snap_concentration": "63bb7f4de6782ced96ae58148dc4dd0372af526084448bb97278ef674b852cc2",
        "away_snap_continuity": "24313595ec16acfff2c55d69e4315121a5550c9dd9389751466c8491ab4ef86c",
        "away_timezone_diff_hours": "ca92dfa6cbbbe6fca4cfada23e07ab6e61b7cf2580d55841c70dba751ead4d33",
        "away_travel_distance_miles": "e5c07a502657ba7207782745b71cadf88f6dc1a2a58ce5a0cb2835cb4b57a856",
        "away_travel_fatigue_score": "99173759915b7f22c063be8078bddc23164112ce15a16d94a2d806589bd097ec",
        "away_westward_travel": "e0c28faf0cc69ce86e11009258e46c7e9e176055ec182876937113bcbdbb5a8d",
        "ball_handling_difficulty": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "both_short_rest": "ecb74c37f08eaeec1816689565fba64a52771ee526150296a5a88884edd022e2",
        "cold_impact_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "defensive_advantage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "elo_diff": "8f51cc89fd8b6142bc3616291946998512cdc0ed282591d9f4803086674ff81b",
        "elo_prob_away": "9080107254cc718b3971de293773b14340a1bc2a2d8ce64e07411f7b1c6cf631",
        "elo_prob_home": "cff1d616a91eb41635b686e18ef78aba2138fd6b32f109ea95a137821699b19e",
        "extreme_weather": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "feature_timestamp": "941e7978e1888b6249cfd13450561668115cb78fd2e2e0bc76893e11d6a4a998",
        "game_day_of_week": "4d3269d59f59072c168c45c75d7a860225fa5234d751f6e376056ada98eef148",
        "game_id": "4e551a4356a9ba78ec9691ac189fa384537915b33c38b3aef9b4c4aa95580149",
        "heat_impact_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "hfa_used": "3e4d5f64c7edb1d99ee6157561214d5bdf872a93d752cf206937f0dc31d4808c",
        "home_availability_coverage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_availability_fraction": "bf72c59985d74e6afd2ad4327965fdf85aad00ec9ec463be2b9d669ae7f06237",
        "home_backup_quality_delta": "fc17b9e5c3fb311752db4948654ab5f4242b400c297b908f99ca5599a3b05e66",
        "home_date_modified_coverage": "f0439f5325346ae1aba66085628dc1d48591e346ce7facdb46c8d3694535c5c8",
        "home_def_rolling_avg_drive_start_yardline": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_def_rolling_cpoe": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_def_rolling_neutral_pace": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_def_rolling_neutral_pass_rate": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_def_rolling_opp_adj_epa_per_play": "3054082acc51cbd36722b6a477950b735862dbc12f14ed0ef20253152c7d1511",
        "home_def_rolling_opp_adj_pass_epa": "5fcf33055c431e8dcf74e64a361b1687417b196b151d35c3614cf362301de0eb",
        "home_def_rolling_opp_adj_rush_epa": "e3c07100fc7db376a4d3097e853bdae82806b66e6582a20b2a1075faa06e58fb",
        "home_def_rolling_pass_success_rate": "b069136cb59a888411a1f82172f2249d0cb9670cd60d11a01b0a330452db522e",
        "home_def_rolling_red_zone_td_rate": "939e328d2403e77b7b2cbee0ef56d51add0119fb101c8f4de7b1d1c0c23fd923",
        "home_def_rolling_rush_success_rate": "08d3906e347628a51c2e420bcaa549860547523b786c85e5ef3f76b5a3c7a1fd",
        "home_def_rolling_success_rate": "2669ecf0e8ebfcff0f58193e2153a553c40da42d971f04b030bc4655de75ff10",
        "home_def_rolling_third_down_conversion_rate": "1795b9bfa30b3a9cfb43e9595e77547f27cc9a72deee4491c9dab52d3f585c32",
        "home_elo": "0e8f5ae04da40c3625d1be68370c2b25070303f46b6f23c4c42320391ba24422",
        "home_elo_momentum": "6a9e33c190925fa175b37e5798be9a7fe1f524ebd4d569f40f7e6e4306554c3f",
        "home_elo_percentile": "e1e3f734372ddf214bc3cea0b2e4483b4e6f3ad3fa997a9a58440f1849232ea9",
        "home_elo_rank": "ed7f0a7f3331423d7fe597db4820654603874d8e4b1caaeddb5ca1177f890b31",
        "home_elo_uncertainty": "c2e8273da7b3035871cd84faf5288fd2b3de1b3a6ee128148d82a9da073f2180",
        "home_injury_coverage": "f0439f5325346ae1aba66085628dc1d48591e346ce7facdb46c8d3694535c5c8",
        "home_letdown_spot": "2ae9a7589cfb85f9d77e6604d85bdb2557e0a09d61bf3b053038d5b40c5e9335",
        "home_look_ahead_spot": "0c42d74e49029bd78bbe6d734b4a9261cfd8b14eed7039bd5cf03c7f64cdf520",
        "home_off_bye": "e3224454b8c1fe3b8b16914da962b3c7ddfcfc5d60ace961dbd3da43a4fdbe4b",
        "home_off_rolling_avg_drive_start_yardline": "3f0434258e447905ab38b82ed9e4b17b3abe1d879631adc1564d9de18c892aaa",
        "home_off_rolling_cpoe": "ff23212357e79992ecd488961432a1ea75153baca67c1a417125789ec051ea36",
        "home_off_rolling_neutral_pace": "e518076cec70b6b395b98f7415b098fe58d643ccfde0321f57ca570bf99ff5b1",
        "home_off_rolling_neutral_pass_rate": "f38289f59c44f8c9e31f04502db58c95f54886085b3380b5b9aad220b29d7a7e",
        "home_off_rolling_opp_adj_epa_per_play": "1300409c1b4e9d9db81759ca4a7a16dcbe9536f4867e03940b827b6f1f31d0c9",
        "home_off_rolling_opp_adj_pass_epa": "e689650446c0805595f2201c2294be2f8e136e999ffce302759f88787e76a573",
        "home_off_rolling_opp_adj_rush_epa": "4991f47f31ee5a403478e0d3cdd18abc483d9fbee6d641300e4cbf73b360fa62",
        "home_off_rolling_pass_success_rate": "e76130f8988a61896d7a5171c63f81bc65e98261751704cc23a2769d1260755e",
        "home_off_rolling_red_zone_td_rate": "a14f5331ccc359dbb9897eff1d18f336a480a1b9352416edbb0e79b744ff6005",
        "home_off_rolling_rush_success_rate": "890cc125f8921767b2bde73203f03946eb46da944adf82e179e19d5be974bf4f",
        "home_off_rolling_success_rate": "d5ccc94f40a13f67dfab77fbab35fd5d0b667cc473042b605da052d55d492808",
        "home_off_rolling_third_down_conversion_rate": "a2ce87e20e4435f6a472ca57903f75ead60861a07731ef727216e4430bd6728a",
        "home_qb_adjustment": "1cd5b4f4b05d6cc49244cf50aabbb1706431e6da0b6ccfeacd0bcba4a331c83e",
        "home_qb_out_flag": "446990838a8f7bbe097216cec01bcf1b49769c33f6692c09f2cc480d7357bde4",
        "home_rest_days": "dd97a5f969b49360d3c3bc95227a7d3efccbbdc99890265b38f3fb280d29d004",
        "home_rolling_snap_share_db": "15c72682b0d76f99fdf5454b591da38b69a5d09bdc1365ebe9252fce0bb1f6da",
        "home_rolling_snap_share_dl": "c458ed422beed5be3318877652eea5bc6cc458a780dfd526b6a29027aa967007",
        "home_rolling_snap_share_lb": "55d5d5973edb9251fe1db359e42c8e3176d71e3aa9e67257e646ffc3d2c57b55",
        "home_rolling_snap_share_ol": "510b1bc599c52b253a8952a218be6901a5620015815a77457eba0c404f6c08b7",
        "home_rolling_snap_share_qb": "c854a75f1d1f8394c3c05c1f4f41989fd15f92e49f371baf2dc51faeb9aac077",
        "home_rolling_snap_share_rb": "842a4d25c110a06e504c25afe75e0943ed4a4adcefecf74425efc67ee8521724",
        "home_rolling_snap_share_te": "353756a5ae9a37dc36b2d73295d93e860e2645b5cf5ca5a45c7830566dd39db0",
        "home_rolling_snap_share_wr": "6c422d6e3ac719400fa904daf8c90299e7ebc0dd5b0e2b7ad081be02ef76ff94",
        "home_score": "d85fcb7788736fb928a6a8c9204ff1f20d4926ae0be324ecbef010ca09fa3e08",
        "home_short_rest": "a3a306d1b284b1106ad333d8ac1fe147a2c3faa9aaec0e0fc4f5867444c32f7d",
        "home_snap_concentration": "de5d791dde3879f0e23b1b7d9c1a0340d3654524bb6a785115d5ce9c5b7f3052",
        "home_snap_continuity": "f3467d4b14fe5c91ab35be5ce1b181d1288f9a8fbccd7382b6cacd9deb133e4a",
        "home_weather_advantage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_away_game": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_divisional": "62249d2f8dae907262bd1636fc9ac491d6dc8218c781ca740205b87a0c455e25",
        "is_dry": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_home_game": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_rain": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_snow": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "kicking_difficulty": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "late_season": "2966195be17179438ec44309c617463e4170534e67d8a4d868dce7d53d9f963a",
        "monday_game": "a5c20f31e7c86f8b210cb7eaa2d8d911fbef4b75df33d2f7d85616a179797b2f",
        "passing_difficulty": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "passing_efficiency": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_heavy": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_impact_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_light": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_mm": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_moderate": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_none": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_prob": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "raw_humidity_pct": "56242ca08930588005aebed3ba8211b3ba6e93c22befce6a6c869415f2e7a63e",
        "raw_precip_mm": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "raw_precip_prob": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "raw_temp_f": "e33153e4acb315c8d6d0a060799ce0729ce9f490142e97ba5fd522bbbbef6050",
        "raw_weather_severity": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "raw_wind_mph": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "rest_advantage": "33620dd9d93f7d0b58e4943ca2c034ca5d1ec1b758d3710c8bdbf96dec40be82",
        "rushing_advantage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "saturday_game": "c2e3d73d37feba2904b69b0bf9a929969e0e3b39ef8ec9ccb229ddb0f09a22a3",
        "scoring_multiplier": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "scoring_reduction": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "season": "72c1be438e384477c751a772ad4fb6698b19b7d2132f7f0c225295aee78801cb",
        "season_progress": "51db062043eea7773b17197809316cb93fb267a803d357477b855e5f41a5462b",
        "short_week": "f4581839b429eacf1171524b7f8d0450240b0a0dd49d454413e71abcb2b729c4",
        "snapshot_ml_prob_home_fair": "97e544dd4f08d3afc239e74d5748e78898417f117dfc8356b3d13524ba90ec5d",
        "snapshot_spread": "4e68d2f9dae46e7eb19472deeb672d0fa20b81c25e20c55d96dab4e8ef216ff7",
        "snapshot_total": "14cba98c6419162fbe995c20cdadd9b27b11d2357bfb87275b4a0234a5aa3c91",
        "spread_movement": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "surface_mismatch": "78fc0d1e15505b75aadcf39332a625cc27b62c8d252ef62ebe5a012023dffa5b",
        "target_ou": "cac8b9d6c4facb50ba04b89955c88a3f40796503c19605ed3cfe7bf4d3976c44",
        "temp_cold": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_cool": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_f": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_hot": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_mild": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_very_cold": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_warm": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "thursday_game": "e537eda90e6d7c85a8d005bafe1f8f1bf779e6e49ecb655997b0f49782bc96e3",
        "total_movement": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "total_points": "966d668746fc04f0a73f7cebe6656fbcb317ddf28ba9217cfdfa33f9147450b8",
        "turnover_multiplier": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "venue_capacity": "05e21984e1ed7bb4293d400454fa77322211dbf8614c034acda3bca3f9004fef",
        "venue_cold_climate": "7716537887125989977b412df650356a8fb35fd0030d955141ce8961bf6a81ed",
        "venue_elevation_ft": "326f54ae32997eb22998d81392718df7dd890e05b63281602f5b970166a1951a",
        "venue_high_altitude": "0608a0758e7b242e61bfdbc6033ea745d9694a8310937cdee218d4504f9d13a4",
        "venue_indoor": "d4e9d557ec3df78e69859869277948d204c24fbb4e613abd2f08698806489069",
        "venue_large_stadium": "f4d13a21a9fac9b646b57a01b0c5fecd01db662b3cd590a3c882b0d5f09b69f7",
        "venue_outdoor": "155f9881faf43d5c8a19e8d3ee99d14ae76b1e458951e68472c944a1a7bffa8c",
        "venue_retractable": "ab85732c061ef76c2c9b8af559f97128452c3ee236336ff1595a5cb688827a43",
        "venue_warm_climate": "dcddff60e74133ca18756f27a7d60a89b1c2ec3c25a23e3473ded59cc1d55717",
        "weather_affects_game": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "weather_game": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "weather_severity_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "week": "4690c9f30f7346ea80e8f6efa70a53d0416710f9f10bc0ed998615d6cc969823",
        "wind_calm": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_high": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_impact_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_moderate": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_mph": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_severe": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
    },
    "features_wp": {
        "apparent_temp_f": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_abs_timezone_diff_hours": "9298b91383149bea4a84a815b7839eada3bc3da5cd5a53f0aaf35246ce3f0d52",
        "away_availability_coverage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_availability_fraction": "edf3766125d94a222895b9e220b9b230de6b905c2adae633bd4a72adc42d32b5",
        "away_backup_quality_delta": "e8c35db24be24d75ee99e6b2bfb89ca7998b5c21bfd5d9cb4947ac69af54d32d",
        "away_cross_country_travel": "121c7d1584f7134449865599de6c736f7a91bede760cfe29a8d9d3851701fa28",
        "away_date_modified_coverage": "f0439f5325346ae1aba66085628dc1d48591e346ce7facdb46c8d3694535c5c8",
        "away_def_rolling_avg_drive_start_yardline": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_def_rolling_cpoe": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_def_rolling_neutral_pace": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_def_rolling_neutral_pass_rate": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "away_def_rolling_opp_adj_epa_per_play": "41ec849ae547aa41fde7cfd5e5c341542c997716aa214c99e7a24257ceef7c99",
        "away_def_rolling_opp_adj_pass_epa": "189034e3670d6ddc3c6b9da7a9f1b430c863434105cb6aea2b16012b75e69205",
        "away_def_rolling_opp_adj_rush_epa": "3b7c7c33f3a656582b26db960b372c9d5f9a6947d1a82691afdbad0812d8cb0b",
        "away_def_rolling_pass_success_rate": "ec0485d6930f6dbef4a3cd674de05445b029c07a78a5487d64d8c29649ecd18e",
        "away_def_rolling_red_zone_td_rate": "c2ff5847f5b79ac8dd8959ac598883c754d4758bb542d8984f86ed2f9a7644a4",
        "away_def_rolling_rush_success_rate": "b5e3cf1ae3cefad975f746bbb9dd4f3c5799ee6d825eff8dd94654eeecc9550b",
        "away_def_rolling_success_rate": "b3e965909f429c7cbb6c4539fa0f2136758e2e13c3e2a1b351b6cbbc543b1edd",
        "away_def_rolling_third_down_conversion_rate": "454ae14ea0a7094c9cd91b2aa000ae460ed6509ae7f59e34df6c4153b4098d23",
        "away_eastward_travel": "f2209d292977d4be78e62ed18ebb8565e9862dd92c2f8f96b9404e01288a4379",
        "away_elo": "c2ef4def8dcc974fe2f245ccd44ff206b092111eed46d40b42fedc728d08a64b",
        "away_elo_momentum": "e97f0c24eac7bdc055b281c4873b3e6d359c3227347e54e276cf7d6de9552e5c",
        "away_elo_percentile": "73d1b8663ac23009bd6c655faa3970142f869f3935de9a4822e306e99f1f7982",
        "away_elo_rank": "b260160d6b73b777b448d7454c86ba9fbf77de2b31eb778880e2956d89805832",
        "away_elo_uncertainty": "e865720516218cd9e34167196104913fc43c96aac7cb2deb70cfbc5f62757813",
        "away_injury_coverage": "f0439f5325346ae1aba66085628dc1d48591e346ce7facdb46c8d3694535c5c8",
        "away_letdown_spot": "4aef6b41f49642a9c0cf39eda2a5a9c897cd1d0bb5b5a4f132bfe401a1559313",
        "away_look_ahead_spot": "0c545e2bcdaf14adf75b4917011b11fbed73b766842163256f9d7bfb333f48ec",
        "away_off_bye": "12637ef4fc146aa9057993c944968591c94fd4372cf6e8d22595d3fd651fdff6",
        "away_off_rolling_avg_drive_start_yardline": "eda62e6afb22f92c32d25badf4584752b2ca6c184f843f45c203536d840ad6a4",
        "away_off_rolling_cpoe": "bbb9dd6a4f75d45dab642db5a825966ecc56988e44087b46643dcfdacb43da54",
        "away_off_rolling_neutral_pace": "ecd05ca440650eb2b66d0932a1bd2904568929b478c4e04a7d76ba48e8a99375",
        "away_off_rolling_neutral_pass_rate": "906e5ad9fb67018432e1d6cf31d0cd0902158a4c731743b086d0388ed432a522",
        "away_off_rolling_opp_adj_epa_per_play": "6a46be42ea089003c530c0a783e63d3fed1193f7105ed996db7663d638055a8c",
        "away_off_rolling_opp_adj_pass_epa": "0595c811202d3e3330c3047d96b943d27f6394d095e91ab3f845df14b130e8d5",
        "away_off_rolling_opp_adj_rush_epa": "45125fa970cce5449575fba71d36b3ffbebc3c751aad618fee47e347d2726b46",
        "away_off_rolling_pass_success_rate": "7453474ed6c14aca08839c675da51b8807bfcca319a7485bd892e94d651be7bf",
        "away_off_rolling_red_zone_td_rate": "ecdb2bc318b591ff23feb6451ff0a8ea519ffe9b9de2943a02b0f752deae256d",
        "away_off_rolling_rush_success_rate": "1a8123c62f565a4ef0b0e4c276125059c60e18954dbbbdf7a718cf88654eeb1f",
        "away_off_rolling_success_rate": "5a7b05cdcfc368929f909b65288bf1c6c8c229af51ad04950a4bb3153e03c930",
        "away_off_rolling_third_down_conversion_rate": "40dbe65189decab6ae3fd12fe29490745cc2857cb419e2055aef67b0702ee763",
        "away_qb_adjustment": "7d7efedb80737d36c78e650309faded4958438e8ea21fb20d6454d5f03277072",
        "away_qb_out_flag": "aea6c71f09c93cc4ba188238803b353028ecc721af6d95aabd95348c1e178dcf",
        "away_rest_days": "1288338247489667d2dc2ba0f78f6134855e4dd137edb023999a408b50e8651e",
        "away_rolling_snap_share_db": "1139cac27a078bf4332061f9c0cabf0f36f0e3ffc99b5e7064e7dbff893e47b4",
        "away_rolling_snap_share_dl": "b3e6cffef405808fd00a21d4ca21cb0aa819fd1c5850e4b37c452bff6cbd0f68",
        "away_rolling_snap_share_lb": "e0f9a76d91baf6cc65dcbffd003a5777bde2e1b385b04a459658bbbb05273959",
        "away_rolling_snap_share_ol": "aeea128a406edcdaf37404d285e9d1e0389406942416c6b9baffeea4e34ba3ff",
        "away_rolling_snap_share_qb": "c67c669e1a7052af541668b6620122a558a8263a7a470d2046588b2ea0d5374d",
        "away_rolling_snap_share_rb": "6178b4392fc731106aaaa1c23048f6dfc7bfb91fb302c9826921500d5d2286e6",
        "away_rolling_snap_share_te": "50d803f60b28753d6f1f46ff1ef4bcadbc1c71438428b0317378241cbe8b9731",
        "away_rolling_snap_share_wr": "cc52630a4cc913eb88e97881355ed6a99de587684518466467bf0b5f55bab959",
        "away_score": "db9d7b93547fba37f72a43b6be88458b8e221ade5c5caabadd935717d4fe446f",
        "away_short_rest": "710996bb06a85fd99bf78994e4a87cf0fa80e38523024464ab9a2d63ad41c583",
        "away_snap_concentration": "63bb7f4de6782ced96ae58148dc4dd0372af526084448bb97278ef674b852cc2",
        "away_snap_continuity": "24313595ec16acfff2c55d69e4315121a5550c9dd9389751466c8491ab4ef86c",
        "away_timezone_diff_hours": "ca92dfa6cbbbe6fca4cfada23e07ab6e61b7cf2580d55841c70dba751ead4d33",
        "away_travel_distance_miles": "e5c07a502657ba7207782745b71cadf88f6dc1a2a58ce5a0cb2835cb4b57a856",
        "away_travel_fatigue_score": "99173759915b7f22c063be8078bddc23164112ce15a16d94a2d806589bd097ec",
        "away_westward_travel": "e0c28faf0cc69ce86e11009258e46c7e9e176055ec182876937113bcbdbb5a8d",
        "ball_handling_difficulty": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "both_short_rest": "ecb74c37f08eaeec1816689565fba64a52771ee526150296a5a88884edd022e2",
        "cold_impact_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "defensive_advantage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "elo_diff": "8f51cc89fd8b6142bc3616291946998512cdc0ed282591d9f4803086674ff81b",
        "elo_prob_away": "9080107254cc718b3971de293773b14340a1bc2a2d8ce64e07411f7b1c6cf631",
        "elo_prob_home": "cff1d616a91eb41635b686e18ef78aba2138fd6b32f109ea95a137821699b19e",
        "extreme_weather": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "feature_timestamp": "941e7978e1888b6249cfd13450561668115cb78fd2e2e0bc76893e11d6a4a998",
        "game_day_of_week": "4d3269d59f59072c168c45c75d7a860225fa5234d751f6e376056ada98eef148",
        "game_id": "4e551a4356a9ba78ec9691ac189fa384537915b33c38b3aef9b4c4aa95580149",
        "heat_impact_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "hfa_used": "3e4d5f64c7edb1d99ee6157561214d5bdf872a93d752cf206937f0dc31d4808c",
        "home_availability_coverage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_availability_fraction": "bf72c59985d74e6afd2ad4327965fdf85aad00ec9ec463be2b9d669ae7f06237",
        "home_backup_quality_delta": "fc17b9e5c3fb311752db4948654ab5f4242b400c297b908f99ca5599a3b05e66",
        "home_date_modified_coverage": "f0439f5325346ae1aba66085628dc1d48591e346ce7facdb46c8d3694535c5c8",
        "home_def_rolling_avg_drive_start_yardline": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_def_rolling_cpoe": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_def_rolling_neutral_pace": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_def_rolling_neutral_pass_rate": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_def_rolling_opp_adj_epa_per_play": "3054082acc51cbd36722b6a477950b735862dbc12f14ed0ef20253152c7d1511",
        "home_def_rolling_opp_adj_pass_epa": "5fcf33055c431e8dcf74e64a361b1687417b196b151d35c3614cf362301de0eb",
        "home_def_rolling_opp_adj_rush_epa": "e3c07100fc7db376a4d3097e853bdae82806b66e6582a20b2a1075faa06e58fb",
        "home_def_rolling_pass_success_rate": "b069136cb59a888411a1f82172f2249d0cb9670cd60d11a01b0a330452db522e",
        "home_def_rolling_red_zone_td_rate": "939e328d2403e77b7b2cbee0ef56d51add0119fb101c8f4de7b1d1c0c23fd923",
        "home_def_rolling_rush_success_rate": "08d3906e347628a51c2e420bcaa549860547523b786c85e5ef3f76b5a3c7a1fd",
        "home_def_rolling_success_rate": "2669ecf0e8ebfcff0f58193e2153a553c40da42d971f04b030bc4655de75ff10",
        "home_def_rolling_third_down_conversion_rate": "1795b9bfa30b3a9cfb43e9595e77547f27cc9a72deee4491c9dab52d3f585c32",
        "home_elo": "0e8f5ae04da40c3625d1be68370c2b25070303f46b6f23c4c42320391ba24422",
        "home_elo_momentum": "6a9e33c190925fa175b37e5798be9a7fe1f524ebd4d569f40f7e6e4306554c3f",
        "home_elo_percentile": "e1e3f734372ddf214bc3cea0b2e4483b4e6f3ad3fa997a9a58440f1849232ea9",
        "home_elo_rank": "ed7f0a7f3331423d7fe597db4820654603874d8e4b1caaeddb5ca1177f890b31",
        "home_elo_uncertainty": "c2e8273da7b3035871cd84faf5288fd2b3de1b3a6ee128148d82a9da073f2180",
        "home_injury_coverage": "f0439f5325346ae1aba66085628dc1d48591e346ce7facdb46c8d3694535c5c8",
        "home_letdown_spot": "2ae9a7589cfb85f9d77e6604d85bdb2557e0a09d61bf3b053038d5b40c5e9335",
        "home_look_ahead_spot": "0c42d74e49029bd78bbe6d734b4a9261cfd8b14eed7039bd5cf03c7f64cdf520",
        "home_off_bye": "e3224454b8c1fe3b8b16914da962b3c7ddfcfc5d60ace961dbd3da43a4fdbe4b",
        "home_off_rolling_avg_drive_start_yardline": "3f0434258e447905ab38b82ed9e4b17b3abe1d879631adc1564d9de18c892aaa",
        "home_off_rolling_cpoe": "ff23212357e79992ecd488961432a1ea75153baca67c1a417125789ec051ea36",
        "home_off_rolling_neutral_pace": "e518076cec70b6b395b98f7415b098fe58d643ccfde0321f57ca570bf99ff5b1",
        "home_off_rolling_neutral_pass_rate": "f38289f59c44f8c9e31f04502db58c95f54886085b3380b5b9aad220b29d7a7e",
        "home_off_rolling_opp_adj_epa_per_play": "1300409c1b4e9d9db81759ca4a7a16dcbe9536f4867e03940b827b6f1f31d0c9",
        "home_off_rolling_opp_adj_pass_epa": "e689650446c0805595f2201c2294be2f8e136e999ffce302759f88787e76a573",
        "home_off_rolling_opp_adj_rush_epa": "4991f47f31ee5a403478e0d3cdd18abc483d9fbee6d641300e4cbf73b360fa62",
        "home_off_rolling_pass_success_rate": "e76130f8988a61896d7a5171c63f81bc65e98261751704cc23a2769d1260755e",
        "home_off_rolling_red_zone_td_rate": "a14f5331ccc359dbb9897eff1d18f336a480a1b9352416edbb0e79b744ff6005",
        "home_off_rolling_rush_success_rate": "890cc125f8921767b2bde73203f03946eb46da944adf82e179e19d5be974bf4f",
        "home_off_rolling_success_rate": "d5ccc94f40a13f67dfab77fbab35fd5d0b667cc473042b605da052d55d492808",
        "home_off_rolling_third_down_conversion_rate": "a2ce87e20e4435f6a472ca57903f75ead60861a07731ef727216e4430bd6728a",
        "home_qb_adjustment": "1cd5b4f4b05d6cc49244cf50aabbb1706431e6da0b6ccfeacd0bcba4a331c83e",
        "home_qb_out_flag": "446990838a8f7bbe097216cec01bcf1b49769c33f6692c09f2cc480d7357bde4",
        "home_rest_days": "dd97a5f969b49360d3c3bc95227a7d3efccbbdc99890265b38f3fb280d29d004",
        "home_rolling_snap_share_db": "15c72682b0d76f99fdf5454b591da38b69a5d09bdc1365ebe9252fce0bb1f6da",
        "home_rolling_snap_share_dl": "c458ed422beed5be3318877652eea5bc6cc458a780dfd526b6a29027aa967007",
        "home_rolling_snap_share_lb": "55d5d5973edb9251fe1db359e42c8e3176d71e3aa9e67257e646ffc3d2c57b55",
        "home_rolling_snap_share_ol": "510b1bc599c52b253a8952a218be6901a5620015815a77457eba0c404f6c08b7",
        "home_rolling_snap_share_qb": "c854a75f1d1f8394c3c05c1f4f41989fd15f92e49f371baf2dc51faeb9aac077",
        "home_rolling_snap_share_rb": "842a4d25c110a06e504c25afe75e0943ed4a4adcefecf74425efc67ee8521724",
        "home_rolling_snap_share_te": "353756a5ae9a37dc36b2d73295d93e860e2645b5cf5ca5a45c7830566dd39db0",
        "home_rolling_snap_share_wr": "6c422d6e3ac719400fa904daf8c90299e7ebc0dd5b0e2b7ad081be02ef76ff94",
        "home_score": "d85fcb7788736fb928a6a8c9204ff1f20d4926ae0be324ecbef010ca09fa3e08",
        "home_short_rest": "a3a306d1b284b1106ad333d8ac1fe147a2c3faa9aaec0e0fc4f5867444c32f7d",
        "home_snap_concentration": "de5d791dde3879f0e23b1b7d9c1a0340d3654524bb6a785115d5ce9c5b7f3052",
        "home_snap_continuity": "f3467d4b14fe5c91ab35be5ce1b181d1288f9a8fbccd7382b6cacd9deb133e4a",
        "home_weather_advantage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "home_win": "0b25895340a8a31d5817ed295472f7a04c5da511c5aa307df1dbd9954c231bfd",
        "is_away_game": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_divisional": "62249d2f8dae907262bd1636fc9ac491d6dc8218c781ca740205b87a0c455e25",
        "is_dry": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_home_game": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_rain": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "is_snow": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "kicking_difficulty": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "late_season": "2966195be17179438ec44309c617463e4170534e67d8a4d868dce7d53d9f963a",
        "monday_game": "a5c20f31e7c86f8b210cb7eaa2d8d911fbef4b75df33d2f7d85616a179797b2f",
        "passing_difficulty": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "passing_efficiency": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_heavy": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_impact_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_light": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_mm": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_moderate": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_none": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "precip_prob": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "raw_humidity_pct": "56242ca08930588005aebed3ba8211b3ba6e93c22befce6a6c869415f2e7a63e",
        "raw_precip_mm": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "raw_precip_prob": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "raw_temp_f": "e33153e4acb315c8d6d0a060799ce0729ce9f490142e97ba5fd522bbbbef6050",
        "raw_weather_severity": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "raw_wind_mph": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "rest_advantage": "33620dd9d93f7d0b58e4943ca2c034ca5d1ec1b758d3710c8bdbf96dec40be82",
        "rushing_advantage": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "saturday_game": "c2e3d73d37feba2904b69b0bf9a929969e0e3b39ef8ec9ccb229ddb0f09a22a3",
        "scoring_multiplier": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "scoring_reduction": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "season": "72c1be438e384477c751a772ad4fb6698b19b7d2132f7f0c225295aee78801cb",
        "season_progress": "51db062043eea7773b17197809316cb93fb267a803d357477b855e5f41a5462b",
        "short_week": "f4581839b429eacf1171524b7f8d0450240b0a0dd49d454413e71abcb2b729c4",
        "snapshot_ml_prob_home_fair": "97e544dd4f08d3afc239e74d5748e78898417f117dfc8356b3d13524ba90ec5d",
        "snapshot_spread": "4e68d2f9dae46e7eb19472deeb672d0fa20b81c25e20c55d96dab4e8ef216ff7",
        "snapshot_total": "14cba98c6419162fbe995c20cdadd9b27b11d2357bfb87275b4a0234a5aa3c91",
        "spread_movement": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "surface_mismatch": "78fc0d1e15505b75aadcf39332a625cc27b62c8d252ef62ebe5a012023dffa5b",
        "target_wp": "673ce364361c08c2c7cc5f7b987fd8451d9918326dcd5c77aad5920f88123e09",
        "temp_cold": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_cool": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_f": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_hot": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_mild": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_very_cold": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "temp_warm": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "thursday_game": "e537eda90e6d7c85a8d005bafe1f8f1bf779e6e49ecb655997b0f49782bc96e3",
        "total_movement": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "turnover_multiplier": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "venue_capacity": "05e21984e1ed7bb4293d400454fa77322211dbf8614c034acda3bca3f9004fef",
        "venue_cold_climate": "7716537887125989977b412df650356a8fb35fd0030d955141ce8961bf6a81ed",
        "venue_elevation_ft": "326f54ae32997eb22998d81392718df7dd890e05b63281602f5b970166a1951a",
        "venue_high_altitude": "0608a0758e7b242e61bfdbc6033ea745d9694a8310937cdee218d4504f9d13a4",
        "venue_indoor": "d4e9d557ec3df78e69859869277948d204c24fbb4e613abd2f08698806489069",
        "venue_large_stadium": "f4d13a21a9fac9b646b57a01b0c5fecd01db662b3cd590a3c882b0d5f09b69f7",
        "venue_outdoor": "155f9881faf43d5c8a19e8d3ee99d14ae76b1e458951e68472c944a1a7bffa8c",
        "venue_retractable": "ab85732c061ef76c2c9b8af559f97128452c3ee236336ff1595a5cb688827a43",
        "venue_warm_climate": "dcddff60e74133ca18756f27a7d60a89b1c2ec3c25a23e3473ded59cc1d55717",
        "weather_affects_game": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "weather_game": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "weather_severity_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "week": "4690c9f30f7346ea80e8f6efa70a53d0416710f9f10bc0ed998615d6cc969823",
        "wind_calm": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_high": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_impact_score": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_moderate": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_mph": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
        "wind_severe": "175c912e97c90825e0565ed098b986099622dffed9ac8465e3b730b35a0c372b",
    },
}
