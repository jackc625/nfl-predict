# PRECOVERAGE-SCAN -- pre-coverage constants in the three gold matrices

Plan 33.2-17 Task 3 (SPEC R10, D33.2-08 item 2). Before each feature family's first covered
season, gold carried a value nobody measured -- a flat 0.0 with no flag -- and a model reads a
centred 0.0 as "exactly average", a confident claim about a season nobody measured. This
document records the re-scan of all three gold matrices for any such block, and where each
one is closed.

The scan is `scripts/scan_precoverage_constants.py`. It reads gold and writes nothing.

## Method

- Every MODEL INPUT column of `features_wp`, `features_ats` and `features_ou` is scanned:
  numeric, not an identifier / target / display column (the exclusion
  `models.temporal.WalkForwardSplitter` applies). The `*_coverage` flags are the statements
  the scan checks for and are not classified as blocks.
- For each column the scan finds the FIRST season in which it takes more than one distinct
  value, and classifies the column:
  - `varies_from_corpus_start` -- it varies in the matrix's first season (2002);
  - `flagged_precoverage` -- it varies only from a later season, and its family carries a
    coverage flag in the matrix, so an unmeasured row of the block before it says so;
  - `unflagged_precoverage` -- it varies only from a later season and nothing flags the
    block before it: the defect;
  - `never_varies` -- it never takes two distinct values in any season. A separate matter
    (a dead or held-constant column), not counted as a pre-coverage block and not this
    document's subject.
- Every `unflagged_precoverage` column is ROUTED, by family, to the planned fix that closes
  it at rung 8. A block no planned fix closes is `UNROUTED`; the count must be zero here, so
  nothing surfaces for the first time after rung 8's rebuild.
- `CONSTANT_2002_2017` counts the model inputs holding ONE value (NaN counted as a value)
  across every row of 2002-2017 in any matrix -- the span the `RULE_EVIDENCE` figure names.

## Before rung 8

Scanned 2026-09-22 against the gold that exists at Plan 33.2-17's wave: rung 7 plus extra
step 7b (`p332_rung7b.json`), BEFORE rung 8. This section is frozen once written; Plan
33.2-18 appends its After-rung-8 section after its rebuild, and the zero ruling is made there,
against the gold it judges.

Machine-readable lines (read by Plan 33.2-18 and by
`tests/integration/test_no_precoverage_constants.py`):

CLASSIFIED_BEFORE_RUNG8: 519
UNFLAGGED_BEFORE_RUNG8: 144
UNROUTED_BEFORE_RUNG8: 0
CONSTANT_2002_2017_BEFORE_RUNG8: 55

### Counts by class

| Matrix | `varies_from_corpus_start` | `flagged_precoverage` | `unflagged_precoverage` | `never_varies` |
|---|---|---|---|---|
| `features_wp` | 92 | 18 | 48 | 15 |
| `features_ats` | 92 | 18 | 48 | 15 |
| `features_ou` | 92 | 18 | 48 | 15 |

The UNFLAGGED count (144) is NON-zero, as it must be at this wave: gold is
still pre-rung-8, so its placeholder blocks are present, and a scanner reporting zero here
would be a scanner that cannot see them. It counts column-matrix pairs
(48 distinct columns, each in every matrix that carries it).

### Every unflagged pre-coverage column, with its route

**snap** -> Plan 33.2-17 Task 2 -- snap_coverage added; NaN with the flag false before the first covered season; reaches gold at rung 8

- `away_rolling_snap_share_db` -- first varies 2013; in ats, ou, wp
- `away_rolling_snap_share_dl` -- first varies 2013; in ats, ou, wp
- `away_rolling_snap_share_lb` -- first varies 2013; in ats, ou, wp
- `away_rolling_snap_share_ol` -- first varies 2013; in ats, ou, wp
- `away_rolling_snap_share_qb` -- first varies 2013; in ats, ou, wp
- `away_rolling_snap_share_rb` -- first varies 2013; in ats, ou, wp
- `away_rolling_snap_share_te` -- first varies 2013; in ats, ou, wp
- `away_rolling_snap_share_wr` -- first varies 2013; in ats, ou, wp
- `away_snap_concentration` -- first varies 2013; in ats, ou, wp
- `away_snap_continuity` -- first varies 2013; in ats, ou, wp
- `home_rolling_snap_share_db` -- first varies 2013; in ats, ou, wp
- `home_rolling_snap_share_dl` -- first varies 2013; in ats, ou, wp
- `home_rolling_snap_share_lb` -- first varies 2013; in ats, ou, wp
- `home_rolling_snap_share_ol` -- first varies 2013; in ats, ou, wp
- `home_rolling_snap_share_qb` -- first varies 2013; in ats, ou, wp
- `home_rolling_snap_share_rb` -- first varies 2013; in ats, ou, wp
- `home_rolling_snap_share_te` -- first varies 2013; in ats, ou, wp
- `home_rolling_snap_share_wr` -- first varies 2013; in ats, ou, wp
- `home_snap_concentration` -- first varies 2013; in ats, ou, wp
- `home_snap_continuity` -- first varies 2013; in ats, ou, wp

**team_form** -> Plan 33.2-17 Task 1 -- silver team form computed for 2002-2026 from the pinned play-by-play; reaches gold at Plan 33.2-18's rung 8

- `away_def_rolling_pass_success_rate` -- first varies 2020; in ats, ou, wp
- `away_def_rolling_red_zone_td_rate` -- first varies 2020; in ats, ou, wp
- `away_def_rolling_rush_success_rate` -- first varies 2020; in ats, ou, wp
- `away_def_rolling_success_rate` -- first varies 2020; in ats, ou, wp
- `away_def_rolling_third_down_conversion_rate` -- first varies 2020; in ats, ou, wp
- `away_off_rolling_avg_drive_start_yardline` -- first varies 2020; in ats, ou, wp
- `away_off_rolling_neutral_pace` -- first varies 2020; in ats, ou, wp
- `away_off_rolling_neutral_pass_rate` -- first varies 2020; in ats, ou, wp
- `away_off_rolling_pass_success_rate` -- first varies 2020; in ats, ou, wp
- `away_off_rolling_red_zone_td_rate` -- first varies 2020; in ats, ou, wp
- `away_off_rolling_rush_success_rate` -- first varies 2020; in ats, ou, wp
- `away_off_rolling_success_rate` -- first varies 2020; in ats, ou, wp
- `away_off_rolling_third_down_conversion_rate` -- first varies 2020; in ats, ou, wp
- `home_def_rolling_pass_success_rate` -- first varies 2020; in ats, ou, wp
- `home_def_rolling_red_zone_td_rate` -- first varies 2020; in ats, ou, wp
- `home_def_rolling_rush_success_rate` -- first varies 2020; in ats, ou, wp
- `home_def_rolling_success_rate` -- first varies 2020; in ats, ou, wp
- `home_def_rolling_third_down_conversion_rate` -- first varies 2020; in ats, ou, wp
- `home_off_rolling_avg_drive_start_yardline` -- first varies 2020; in ats, ou, wp
- `home_off_rolling_neutral_pace` -- first varies 2020; in ats, ou, wp
- `home_off_rolling_neutral_pass_rate` -- first varies 2020; in ats, ou, wp
- `home_off_rolling_pass_success_rate` -- first varies 2020; in ats, ou, wp
- `home_off_rolling_red_zone_td_rate` -- first varies 2020; in ats, ou, wp
- `home_off_rolling_rush_success_rate` -- first varies 2020; in ats, ou, wp
- `home_off_rolling_success_rate` -- first varies 2020; in ats, ou, wp
- `home_off_rolling_third_down_conversion_rate` -- first varies 2020; in ats, ou, wp

**team_form_source_limited** -> Plan 33.2-17 Task 1 (values from 2006, where the pinned play-by-play carries cpoe) AND Task 2 (NaN beside rolling_cpoe_coverage 0.0 for 2002-2005, where it does not); reaches gold at rung 8

- `away_off_rolling_cpoe` -- first varies 2020; in ats, ou, wp
- `home_off_rolling_cpoe` -- first varies 2020; in ats, ou, wp

### Flagged pre-coverage columns (handled: the family's flag states the absence)

- `away_availability_fraction` -- first varies 2013; flag `away_availability_coverage`; in ats, ou, wp
- `away_backup_quality_delta` -- first varies 2010; flag `away_injury_coverage`; in ats, ou, wp
- `away_def_rolling_opp_adj_epa_per_play` -- first varies 2018; flag `away_def_rolling_opp_adj_coverage`; in ats, ou, wp
- `away_def_rolling_opp_adj_pass_epa` -- first varies 2018; flag `away_def_rolling_opp_adj_coverage`; in ats, ou, wp
- `away_def_rolling_opp_adj_rush_epa` -- first varies 2018; flag `away_def_rolling_opp_adj_coverage`; in ats, ou, wp
- `away_off_rolling_opp_adj_epa_per_play` -- first varies 2018; flag `away_off_rolling_opp_adj_coverage`; in ats, ou, wp
- `away_off_rolling_opp_adj_pass_epa` -- first varies 2018; flag `away_off_rolling_opp_adj_coverage`; in ats, ou, wp
- `away_off_rolling_opp_adj_rush_epa` -- first varies 2018; flag `away_off_rolling_opp_adj_coverage`; in ats, ou, wp
- `away_qb_out_flag` -- first varies 2010; flag `away_injury_coverage`; in ats, ou, wp
- `home_availability_fraction` -- first varies 2013; flag `home_availability_coverage`; in ats, ou, wp
- `home_backup_quality_delta` -- first varies 2010; flag `home_injury_coverage`; in ats, ou, wp
- `home_def_rolling_opp_adj_epa_per_play` -- first varies 2018; flag `home_def_rolling_opp_adj_coverage`; in ats, ou, wp
- `home_def_rolling_opp_adj_pass_epa` -- first varies 2018; flag `home_def_rolling_opp_adj_coverage`; in ats, ou, wp
- `home_def_rolling_opp_adj_rush_epa` -- first varies 2018; flag `home_def_rolling_opp_adj_coverage`; in ats, ou, wp
- `home_off_rolling_opp_adj_epa_per_play` -- first varies 2018; flag `home_off_rolling_opp_adj_coverage`; in ats, ou, wp
- `home_off_rolling_opp_adj_pass_epa` -- first varies 2018; flag `home_off_rolling_opp_adj_coverage`; in ats, ou, wp
- `home_off_rolling_opp_adj_rush_epa` -- first varies 2018; flag `home_off_rolling_opp_adj_coverage`; in ats, ou, wp
- `home_qb_out_flag` -- first varies 2010; flag `home_injury_coverage`; in ats, ou, wp

### Columns that never vary (a separate matter, not this document's subject)

- `away_def_rolling_avg_drive_start_yardline` -- in ats, ou, wp
- `away_def_rolling_cpoe` -- in ats, ou, wp
- `away_def_rolling_neutral_pace` -- in ats, ou, wp
- `away_def_rolling_neutral_pass_rate` -- in ats, ou, wp
- `home_def_rolling_avg_drive_start_yardline` -- in ats, ou, wp
- `home_def_rolling_cpoe` -- in ats, ou, wp
- `home_def_rolling_neutral_pace` -- in ats, ou, wp
- `home_def_rolling_neutral_pass_rate` -- in ats, ou, wp
- `is_away_game` -- in ats, ou, wp
- `is_home_game` -- in ats, ou, wp
- `snapshot_ml_prob_home_fair` -- in ats, ou, wp
- `snapshot_spread` -- in ats, ou, wp
- `snapshot_total` -- in ats, ou, wp
- `spread_movement` -- in ats, ou, wp
- `total_movement` -- in ats, ou, wp

The five market columns among them (`snapshot_*`, `spread_movement`, `total_movement`) are
the constant 0.0 Plan 33.2-14 left until rung 9 removes the market columns from every model
input (Plan 33.2-19); they are listed here and not routed, because they are not a
pre-coverage block.

### The RULE_EVIDENCE figure does not reproduce

`conf/season_partition.py`'s `RULE_EVIDENCE` records "90 gold columns are constant over
2002-2017". That figure does not reproduce. The 2026-09-15 census measured 59 (43 of them
varying from 2018); rungs 4-7 and step 7b have rebuilt gold since, and THIS scan measures
55 model inputs constant over 2002-2017 on the gold that exists now
(`CONSTANT_2002_2017_BEFORE_RUNG8` above). Plan 33.2-18's rewrite of `RULE_EVIDENCE` cites
that line, not a literal.

### What this section does NOT claim

- It makes no zero-unflagged ruling. That ruling is Plan 33.2-18 Task 3's, run with this
  scanner against rung 8's rebuilt gold, the only gold on which it can be true.
- The opponent-adjusted family is flagged here (NaN beside `rolling_opp_adj_coverage` 0.0
  before the per-game pool's floor since rung 7); its widening to 2002 rides Plan 33.2-18's
  floor move.

## After rung 8

Scanned 2026-09-22 against the gold rung 8 rebuilt (Plan 33.2-18 Task 3, `p332_rung8.json`):
the coverage floors retired -- the selection window moved to 2002, team form computed back to
2002, the opponent-adjusted pool widened with the window, and snaps, injury and completion
probability an honest unknown beside a flag before their coverage. This is the section the
zero ruling is made in, against the gold it judges; the section above is left as it was.

Machine-readable lines (read by `tests/integration/test_no_precoverage_constants.py`):

CLASSIFIED_AFTER_RUNG8: 519
UNFLAGGED_AFTER_RUNG8: 0
UNROUTED_AFTER_RUNG8: 0
CONSTANT_2002_2017_AFTER_RUNG8: 15

### Counts by class

| Matrix | `varies_from_corpus_start` | `flagged_precoverage` | `unflagged_precoverage` | `never_varies` |
|---|---|---|---|---|
| `features_wp` | 130 | 28 | 0 | 15 |
| `features_ats` | 130 | 28 | 0 | 15 |
| `features_ou` | 130 | 28 | 0 | 15 |

The UNFLAGGED count is zero in all three matrices. No constant stands in for a family's values
before its first covered season: every block before a family's coverage is now either a
computed value or NaN beside a coverage flag that says so. This is where Plan 33.2-17's
prohibition -- no constant substituted for a family's values before its first covered season --
is shown resolved.

### What moved between the two sections

- **Team form (26 columns) and the opponent-adjusted family (12 columns)** now vary from the
  corpus start (2002): team form from Plan 33.2-17's silver corpus, the opponent-adjusted values
  from the per-game pool the window move widened. Both left the unflagged and flagged lists.
- **The 20 snap columns** moved from unflagged to flagged: NaN beside `{side}_snap_coverage`
  0.0 for 2002-2012, first varying in 2013.
- **`{side}_off_rolling_cpoe`** moved from unflagged to flagged: NaN beside
  `{side}_off_rolling_cpoe_coverage` 0.0 for 2002-2005 (the pinned play-by-play carries no
  completion probability before 2006), first varying in 2006.
- **Injury** (`qb_out_flag`, `backup_quality_delta`) now first varies in 2010, flagged by
  `{side}_injury_coverage`: almost no 2009 report carries a datable time (p332_ step 6b), so
  2002-2008 and nearly all of 2009 are NaN beside a false flag. The one admitted 2009 report is
  an away team's (2009_W17_NO@CAR), so the away side's coverage begins in 2009 and the home
  side's in 2010. `availability_fraction` first varies in 2013, flagged by
  `{side}_availability_coverage`.

### Flagged pre-coverage columns (the family's flag states the absence)

- `away_availability_fraction` -- first varies 2013; flag `away_availability_coverage`; in ats, ou, wp
- `away_backup_quality_delta` -- first varies 2010; flag `away_injury_coverage`; in ats, ou, wp
- `away_off_rolling_cpoe` -- first varies 2006; flag `away_off_rolling_cpoe_coverage`; in ats, ou, wp
- `away_qb_out_flag` -- first varies 2010; flag `away_injury_coverage`; in ats, ou, wp
- `away_rolling_snap_share_db` -- first varies 2013; flag `away_snap_coverage`; in ats, ou, wp
- `away_rolling_snap_share_dl` -- first varies 2013; flag `away_snap_coverage`; in ats, ou, wp
- `away_rolling_snap_share_lb` -- first varies 2013; flag `away_snap_coverage`; in ats, ou, wp
- `away_rolling_snap_share_ol` -- first varies 2013; flag `away_snap_coverage`; in ats, ou, wp
- `away_rolling_snap_share_qb` -- first varies 2013; flag `away_snap_coverage`; in ats, ou, wp
- `away_rolling_snap_share_rb` -- first varies 2013; flag `away_snap_coverage`; in ats, ou, wp
- `away_rolling_snap_share_te` -- first varies 2013; flag `away_snap_coverage`; in ats, ou, wp
- `away_rolling_snap_share_wr` -- first varies 2013; flag `away_snap_coverage`; in ats, ou, wp
- `away_snap_concentration` -- first varies 2013; flag `away_snap_coverage`; in ats, ou, wp
- `away_snap_continuity` -- first varies 2013; flag `away_snap_coverage`; in ats, ou, wp
- `home_availability_fraction` -- first varies 2013; flag `home_availability_coverage`; in ats, ou, wp
- `home_backup_quality_delta` -- first varies 2010; flag `home_injury_coverage`; in ats, ou, wp
- `home_off_rolling_cpoe` -- first varies 2006; flag `home_off_rolling_cpoe_coverage`; in ats, ou, wp
- `home_qb_out_flag` -- first varies 2010; flag `home_injury_coverage`; in ats, ou, wp
- `home_rolling_snap_share_db` -- first varies 2013; flag `home_snap_coverage`; in ats, ou, wp
- `home_rolling_snap_share_dl` -- first varies 2013; flag `home_snap_coverage`; in ats, ou, wp
- `home_rolling_snap_share_lb` -- first varies 2013; flag `home_snap_coverage`; in ats, ou, wp
- `home_rolling_snap_share_ol` -- first varies 2013; flag `home_snap_coverage`; in ats, ou, wp
- `home_rolling_snap_share_qb` -- first varies 2013; flag `home_snap_coverage`; in ats, ou, wp
- `home_rolling_snap_share_rb` -- first varies 2013; flag `home_snap_coverage`; in ats, ou, wp
- `home_rolling_snap_share_te` -- first varies 2013; flag `home_snap_coverage`; in ats, ou, wp
- `home_rolling_snap_share_wr` -- first varies 2013; flag `home_snap_coverage`; in ats, ou, wp
- `home_snap_concentration` -- first varies 2013; flag `home_snap_coverage`; in ats, ou, wp
- `home_snap_continuity` -- first varies 2013; flag `home_snap_coverage`; in ats, ou, wp

### Columns that never vary (not a pre-coverage block, and not closed here)

- `away_def_rolling_avg_drive_start_yardline`, `away_def_rolling_cpoe`,
  `away_def_rolling_neutral_pace`, `away_def_rolling_neutral_pass_rate`,
  `home_def_rolling_avg_drive_start_yardline`, `home_def_rolling_cpoe`,
  `home_def_rolling_neutral_pace`, `home_def_rolling_neutral_pass_rate` -- in ats, ou, wp
- `is_away_game`, `is_home_game` -- in ats, ou, wp
- `snapshot_ml_prob_home_fair`, `snapshot_spread`, `snapshot_total`, `spread_movement`,
  `total_movement` -- in ats, ou, wp

The 15 columns constant over 2002-2017 (`CONSTANT_2002_2017_AFTER_RUNG8`) are exactly these,
constant in EVERY season, not only before 2018. Two of the groups are known and owned: the five
market columns are the constant 0.0 Plan 33.2-14 left until rung 9 removes them from every model
input (Plan 33.2-19). The eight defensive team-form copies are a DEFECT this rung does not own:
the four metrics are offense-only, so their defensive copies are never populated and gold carries
them as a flat 0.0 that reads as "exactly average" in every season. They are not a coverage floor
(no season ever covers them), so closing them is a separate cause and is routed as its own step
(`deferred-items.md`, recorded by Plan 33.2-18); `test_gold_team_form_and_opp_adj_not_constant_within_season`
fails on exactly these eight until then.
