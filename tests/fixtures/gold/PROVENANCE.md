# PROVENANCE -- the frozen pre-Phase-30 gold fixture

**Captured:** 2026-08-22 (UTC), Phase 30 Plan 30-03, Task 1.
**Repository git SHA at capture time:** `dc4d1c0c09ed3b4f5835c801e991aa945f23b479`
(`dc4d1c0`, `feat(30-02): freeze the Stage-1 pre-registration (SPEC R4 ancestry anchor)`).
**Captured BEFORE:** any rung of the D30-17 four-rung rebuild ladder (Plans 30-06 / 30-07 /
30-08). At capture time `data/gold/features_ats.parquet` was still the artifact that produced
every published Phase-29 reading in `LINE-MOVEMENT-READOUT.md`.

ASCII only, no emoji (CLAUDE.md).

---

## 1. What is here

| Fixture | Source path | Bytes | Shape |
|---|---|---|---|
| `features_ats_pre_phase30.parquet` | `data/gold/features_ats.parquet` | 3,696,112 | 6,263 rows x 210 columns |
| `odds_snapshot_pre_phase30.parquet` | `data/silver/odds_snapshot.parquet` | 49,887 | 1,856 rows x 10 columns |

- ATS gold season span: **2002 - 2025** (all 6,263 rows, nothing trimmed).
- `line_movement` columns present in the ATS matrix: **15**
  (`backtest.signal_lift.group_columns(df, "line_movement")`).
- Odds slice: 1,856 distinct `game_id` values spanning the 2018 - 2025 seasons. This is the
  closing-odds table every CLV leg of the reproductions reads.

## 2. Digests, and the match against the source they were copied from

Both digests are `hashlib.sha256` over the RAW BYTES of the file
(`hashlib.sha256(Path(p).read_bytes()).hexdigest()`), never over a pandas round-trip. The files
were copied with `shutil.copyfile`, byte for byte -- re-encoding through pandas would produce a
different parquet with the same logical contents and a different digest, and would destroy the
byte-for-byte-copy claim below.

### `features_ats_pre_phase30.parquet`

```
fixture sha256   733a1273fde4067607480540fd1357e50b81d31e5793b386d4f916e3057a5728
source  sha256   733a1273fde4067607480540fd1357e50b81d31e5793b386d4f916e3057a5728
source  path     data/gold/features_ats.parquet
```

**MATCH LINE:** the two digests above were compared at capture time on 2026-08-22 and were
**EQUAL**. The fixture is a byte-for-byte copy of `data/gold/features_ats.parquet` as it stood at
repository SHA `dc4d1c0`.

### `odds_snapshot_pre_phase30.parquet`

```
fixture sha256   405133f16de9167ee37d1ff81b85456572029d8a936fc0a6b922d564f662a790
source  sha256   405133f16de9167ee37d1ff81b85456572029d8a936fc0a6b922d564f662a790
source  path     data/silver/odds_snapshot.parquet
```

**MATCH LINE:** the two digests above were compared at capture time on 2026-08-22 and were
**EQUAL**. The fixture is a byte-for-byte copy of `data/silver/odds_snapshot.parquet` as it stood
at repository SHA `dc4d1c0`.

**Why the match lines are recorded rather than asserted in prose.** The comparison is only
checkable NOW. Plan 30-06 rung 1 rebuilds `data/gold/features_ats.parquet`; after that the source
file no longer exists in the state it was copied from, and an unrecorded match is unrecoverable.
Printing the source digest beside the fixture digest lets a later reader see the equality instead
of taking it on trust.

## 3. Why this fixture exists

The four harness reproductions in
`tests/unit/test_line_movement_readout_md.py::TestReadoutMatchesHarness` exist to prove one thing:
that the numbers published in `LINE-MOVEMENT-READOUT.md` came out of the committed harness rather
than out of somebody's notes. They did that by re-running `backtest.signal_lift` against LIVE gold
and comparing to the published deltas within `5e-3`.

Phase 30 rebuilds gold **four times, on purpose** (the D30-17 rung ladder: WR-06 bounds, the CR-02
flag repair, the `line_movement` drop, the N-01 re-sync). Any one of those moves the ATS deltas far
beyond `5e-3`, and the `line_movement` drop additionally turns two of the four assertions into
VACUOUS PASSES -- `n_group_columns_selected == 0` and `"line_movement_coverage" not in
group_columns_selected` are both trivially true over an empty selected-column list. A green test
that proves nothing is worse than a red one, because nobody revisits it.

Freezing the pre-rebuild artifact here makes those reproductions permanently immune to every
future rebuild. That is not a workaround for the drop; it is the correct long-term shape for a
guard whose whole point is "the published number came from the committed harness ON THE GOLD THAT
PRODUCED IT".

**Nothing is trimmed, in either dimension, and the reason is not conservatism:**

- **All 210 columns.** The reproductions run a full two-leg walk-forward through
  `ATSTrainer.train_and_evaluate`, whose feature set is derived from the numeric non-ID columns of
  whatever frame it is handed. Dropping columns changes what `SelectFromModel` sees on the train
  window, which changes the locked feature set, which changes every delta -- destroying the only
  thing the fixture exists to prove.
- **All 6,263 rows, seasons 2002-2025.** The three windows the reproductions exercise are the
  canonical one, `COVERAGE_WINDOW_CONFIG` (`backtest/signal_lift.py`) and
  `COVERED_SELECTION_WINDOW_CONFIG`, and `WalkForwardSplitter` is EXPANDING -- for each holdout
  season Y it trains on ALL seasons < Y (`models/temporal.py`). Trimming early seasons would
  change the holdout fits even though those rows sit outside every named window.

## 4. The regeneration rule

**This file and the two parquets are regenerated together or not at all.**

`tests/unit/test_line_movement_readout_md.py::test_fixture_is_the_pre_drop_gold` pins the ATS
matrix's shape `(6263, 210)`, its 15-column `line_movement` count, and BOTH sha256 digests above
against module constants transcribed from this file. Regenerating either fixture to make a failing
assertion pass is **the exact failure that test exists to catch**. If an assertion in this module
goes red, the finding is about the harness or about the record -- it is never a licence to
re-capture the fixture.

The fixture is tracked in git only because `.gitignore` carries a narrow file-level negation,
`!tests/fixtures/gold/*.parquet`, placed after the repository-wide `*.parquet` rule. The
directory-scoped `!tests/fixtures/` entry above it matches the DIRECTORY and does NOT re-include
files inside it, so without the negation `git add` on these paths silently no-ops and the
reproductions pass only on the author's machine.
