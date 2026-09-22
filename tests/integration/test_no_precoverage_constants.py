"""No gold value before a family's first covered season is a constant nobody measured (SPEC R10).

D33.2-08 item 2 / Plan 33.2-17 Task 3. Before each family's first covered season, gold carried a
flat 0.0 with no flag -- and a model reads a centred 0.0 as "exactly average", a confident claim
about a season nobody measured. ``scripts/scan_precoverage_constants.py`` classifies every model
input of every gold matrix; this module pins what it reports.

ONE CAUSE, TWO WAVES. Every gold-dependent node below is a property of rung 8's SINGLE declared
cause, "the coverage floors are retired" (Plan 33.2-18): team form computed back to 2002 (Plan
33.2-17 Task 1's silver corpus), the opponent-adjusted pool widened by the floor move (Plan
33.2-18), and snap / injury values an honest unknown with a flag (Plan 33.2-17 Task 2). None is a
second cause. The nodes are split so the split is visible:

* EVALUATED at Plan 33.2-17 (pre-rung-8 gold): the non-vacuity control, the BEFORE-state control
  (the live unflagged count is non-zero -- the scanner sees the defect rung 8 exists to close),
  the routing check (``UNROUTED == 0``), and the doc-drift guard over ``PRECOVERAGE-SCAN.md``'s
  LATEST section.
* AUTHORED at Plan 33.2-17 and EVALUATED by Plan 33.2-18 Task 3 after the rung-8 rebuild --
  deselected by name at Plan 33.2-17, exactly as Plan 33.2-12 authored
  ``test_forecastless_columns_absent_from_gold``: ``test_zero_unflagged_after_rung8``,
  ``test_gold_team_form_and_opp_adj_not_constant_within_season``,
  ``test_gold_snap_boundary_2012_2013``, ``test_gold_injury_boundary_2008_2009`` and
  ``test_gold_coverage_flag_columns_present``. Each says in its docstring which rung makes it
  true.

The doc-drift guard compares ruling to ruling -- the classes and counts the LATEST section of the
document records against what the live scan returns -- and never pins a point estimate to moving
gold (``tests/unit/test_signal_lift_readout_md.py`` :57-72). Reading the LATEST section is what
keeps one guard correct at both waves: at Plan 33.2-17 that is ``## Before rung 8``; after Plan
33.2-18 appends, it is ``## After rung 8``. Its BEFORE-state control (``UNFLAGGED`` non-zero)
therefore only runs while ``## Before rung 8`` is the latest section.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pandas as pd
import pytest

import scripts.fingerprint_gold as fg
from scripts import scan_precoverage_constants as scan

GOLD = Path("data/gold")
DOCUMENT = Path("PRECOVERAGE-SCAN.md")
BEFORE_SECTION = "## Before rung 8"
AFTER_SECTION = "## After rung 8"

needs_gold = pytest.mark.skipif(
    not all((GOLD / f"{m}.parquet").is_file() for m in scan.GOLD_MATRICES),
    reason="the gold matrices are absent on this checkout (data/ is gitignored)",
)


@cache
def _live() -> dict[str, object]:
    return scan.scan_gold(GOLD)


@cache
def _gold(matrix: str) -> pd.DataFrame:
    return pd.read_parquet(GOLD / f"{matrix}.parquet")


#: A scan section's heading. GENERALISED from the two literal headings this guard was
#: written for (p332_ extra step 8e): the document now carries a third section, because
#: the ladder kept moving gold after rung 8 and a record that is edited whenever the
#: present changes is not a record. The guard's own design already said reading the
#: LATEST section is what keeps ONE guard correct at every wave; this makes that true
#: for any number of them.
_SECTION_HEADING = re.compile(r"^## (?:Before|After) [^\n]+$", flags=re.MULTILINE)

#: The machine-readable lines a section publishes. The SUFFIX names the section
#: (``BEFORE_RUNG8``, ``AFTER_RUNG8``, ``AFTER_STEP8E``) and is read off the line rather
#: than derived from the heading's spelling, so a section cannot half-rename itself.
_MACHINE_LINE = re.compile(
    r"^(?P<key>CLASSIFIED|UNFLAGGED|UNROUTED|CONSTANT_2002_2017)"
    r"_(?P<suffix>[A-Z0-9_]+):\s*(?P<value>\d+)\s*$",
    flags=re.MULTILINE,
)


def _latest_section() -> tuple[str, str]:
    """(heading, body) of the LAST scan-section HEADING in the document.

    Matched as a heading at the start of a line, so prose that merely names a section cannot
    be mistaken for it.
    """
    assert DOCUMENT.is_file(), (
        f"{DOCUMENT} is missing: the before-state was never published"
    )
    text = DOCUMENT.read_text(encoding="utf-8")
    headings = list(_SECTION_HEADING.finditer(text))
    assert headings, "the document carries no Before/After scan section"
    last = headings[-1]
    return last.group(0).strip(), text[last.start() :]


def _recorded(body: str, key: str) -> int:
    """The value of *key*'s machine-readable line in *body*, whatever its section suffix.

    Every machine line in a section must carry the SAME suffix: a section publishing two
    suffixes is half-copied from an older one, and reading either would be reading a
    number that describes different gold.
    """
    lines = list(_MACHINE_LINE.finditer(body))
    assert lines, "the latest section records no machine-readable count lines"
    suffixes = {match.group("suffix") for match in lines}
    assert len(suffixes) == 1, (
        f"the latest section mixes machine-line suffixes {sorted(suffixes)}; its counts "
        "do not all describe the same gold"
    )
    for match in lines:
        if match.group("key") == key:
            return int(match.group("value"))
    msg = f"the latest section records no machine-readable '{key}_*:' line"
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# EVALUATED at Plan 33.2-17.
# ---------------------------------------------------------------------------


@needs_gold
class TestTheScanSeesTheGold:
    def test_non_vacuity_the_scan_classifies_columns(self) -> None:
        assert _live()["classified"] > 0

    def test_every_classified_column_carries_one_of_the_four_classes(self) -> None:
        assert {s.klass for s in _live()["scans"]} <= set(scan.CLASSES)

    def test_every_unflagged_block_is_routed_to_a_planned_fix(self) -> None:
        unrouted = [f"{s.matrix}:{s.column}" for s in _live()["unrouted"]]
        assert unrouted == [], (
            "an unflagged pre-coverage block no planned fix closes would first surface "
            "after rung 8's rebuild, where the only response is to halt"
        )

    def test_before_rung8_the_scan_reports_unflagged_blocks(self) -> None:
        heading, _body = _latest_section()
        if heading != BEFORE_SECTION:
            pytest.skip("rung 8 has run: the before-state control has done its job")
        assert len(_live()["unflagged"]) > 0, (
            "pre-rung-8 gold still carries its placeholder blocks, so a scan reporting "
            "none is a scan that cannot see them"
        )


@needs_gold
class TestTheDocumentRecordsTheLiveRuling:
    def test_the_latest_sections_counts_equal_the_live_scan(self) -> None:
        _heading, body = _latest_section()
        assert _recorded(body, "UNFLAGGED") == len(_live()["unflagged"])
        assert _recorded(body, "CLASSIFIED") == _live()["classified"]

    def test_the_latest_section_lists_every_unflagged_column_with_its_route(
        self,
    ) -> None:
        _heading, body = _latest_section()
        for block in _live()["unflagged"]:
            assert f"`{block.column}`" in body, block.column

    def test_the_document_is_ascii_and_names_the_four_classes(self) -> None:
        assert DOCUMENT.is_file(), f"{DOCUMENT} is missing"
        text = DOCUMENT.read_text(encoding="utf-8")
        assert text.isascii()
        assert "proven" not in text.lower()
        for klass in scan.CLASSES:
            assert klass in text, klass


# ---------------------------------------------------------------------------
# AUTHORED at Plan 33.2-17, EVALUATED by Plan 33.2-18 Task 3 after the rung-8 rebuild.
# Deselected by name at Plan 33.2-17: pre-rung-8 gold cannot make them true.
# ---------------------------------------------------------------------------


@needs_gold
def test_zero_unflagged_after_rung8() -> None:
    """Zero unflagged pre-coverage blocks in all three matrices.

    TRUE ONLY AFTER RUNG 8 (Plan 33.2-18): team form reaches 2002 through Plan 33.2-17's silver
    corpus, the opponent-adjusted pool through the floor move, and snaps / injury carry NaN with
    their flags. The prohibition Plan 33.2-17 owns -- no constant substituted for a family's
    values before its first covered season -- rests on this node's run against rung 8's gold.
    """
    report = _live()
    assert report["classified"] > 0, "non-vacuity"
    assert [f"{s.matrix}:{s.column}" for s in report["unflagged"]] == []


@needs_gold
def test_gold_team_form_and_opp_adj_not_constant_within_season() -> None:
    """No team-form or opponent-adjusted column holds ONE value within any season 2002-2025.

    TRUE ONLY AFTER RUNG 8: before it, team form is a flat block before 2020 and the
    opponent-adjusted family NaN before the per-game pool's floor. A season that is honestly
    blank for a whole column (every row NaN) is not a constant and is not counted here.
    """
    flat = []
    for matrix in scan.GOLD_MATRICES:
        frame = _gold(matrix)
        columns = [
            c
            for c in scan.model_input_columns(frame)
            if scan.column_family(c) in ("team_form", "opp_adj")
        ]
        assert columns, "non-vacuity"
        for season, rows in frame[frame["season"].between(2002, 2025)].groupby(
            "season"
        ):
            for column in columns:
                if (
                    rows[column].nunique(dropna=True) == 1
                    and rows[column].notna().all()
                ):
                    flat.append((matrix, int(season), column))
    assert flat == []


def _snap_value_columns(frame: pd.DataFrame) -> list[str]:
    return [
        c for c in scan.model_input_columns(frame) if scan.column_family(c) == "snap"
    ]


def _undeclared_blanks(
    frame: pd.DataFrame, rows: pd.Series, columns: list[str]
) -> list[tuple[str, int, str]]:
    """Blank cells among *rows* that p332_ extra step 8d did NOT declare.

    OWNER RULING 2026-09-22 ("LEAVE THE CELL BLANK"). A covered row's value is blank
    when its NORMALIZATION STATISTIC could not be formed -- a family's first covered
    season has no prior season to bootstrap from, and its earliest lock group can hold
    fewer than ``min_periods`` admitted rows. That cell was measured and nothing could
    place it; before step 8d it read the neutral 0.0, which claimed it was exactly
    average.

    The allowance is PINNED to the step's own declaration rather than waved: a blank
    is accepted only in a season that column declared, and only up to the cell count
    it declared. A blank anywhere else is still the defect these nodes exist to catch.
    """
    violations: list[tuple[str, int, str]] = []
    for column in columns:
        blank = rows & frame[column].isna()
        if not blank.any():
            continue
        seasons = {str(int(s)) for s in frame.loc[blank, "season"]}
        allowed_seasons = set(
            fg.PHASE332_UNSCORABLE_BLANK_STEP_SEASONS_BY_COLUMN.get(column, ())
        )
        allowed_cells = fg.PHASE332_UNSCORABLE_BLANK_STEP_CELLS_BY_COLUMN.get(column, 0)
        if not seasons <= allowed_seasons or int(blank.sum()) > allowed_cells:
            violations.append((column, int(blank.sum()), ",".join(sorted(seasons))))
    return violations


@needs_gold
def test_gold_snap_boundary_2012_2013() -> None:
    """Snaps: NaN with the flag false for 2002-2012; populated, flag true, from 2013.

    TRUE ONLY AFTER RUNG 8, where Plan 33.2-17 Task 2's builder (NaN before coverage, and
    ``snap_coverage`` in ``SnapCountBuilder._feature_columns()``) first reaches gold. A 2013 row
    with no admitted snap game yet (its team's first game) is NaN with the flag false too: the
    flag says whether THIS value was measured.

    CORRECTED BY PLAN 33.2-18 against rung 8's rebuilt gold. ``snap_continuity`` compares a
    team's two most recent admitted snap games, so on the team's FIRST flagged game -- its window
    holds a single snap game -- it is undefined: NaN beside a set flag, exactly p332_ step 7b's
    declared blank cells (32 per continuity column: the 2013 week-1 and week-2 games). Every
    other snap value is populated wherever the flag is set, continuity is undefined only in that
    one week, and from 2014 on every flagged row carries every value. Was: ``first.loc[populated,
    side_values].notna().all().all()`` over every snap value, continuity included -- a premise
    step 7b's measured blanks contradict.

    CORRECTED AGAIN BY p332_ EXTRA STEP 8d (owner ruling 2026-09-22), for one more
    measured blank: 2013 is the snap family's FIRST covered season, so it has no prior
    season to bootstrap the expanding normalization from, and the season's earliest lock
    group holds fewer than ``min_periods`` admitted rows. Those cells were measured -- the
    flag says so -- and nothing could score them, so they are blank instead of the neutral
    0.0 that used to claim they were exactly average. The allowance is PINNED to step 8d's
    own per-column seasons and cell counts (``_undeclared_blanks``), so a blank anywhere
    else still fails. Was: every flag-true 2013 row asserted populated in every non-
    continuity column.
    """
    for matrix in scan.GOLD_MATRICES:
        frame = _gold(matrix)
        values = _snap_value_columns(frame)
        assert values, "non-vacuity"
        for side in ("home", "away"):
            flag = f"{side}_snap_coverage"
            continuity = f"{side}_snap_continuity"
            side_values = [c for c in values if c.startswith(f"{side}_")]
            others = [c for c in side_values if c != continuity]
            assert continuity in side_values, (matrix, continuity)
            before = frame[frame["season"].between(2002, 2012)]
            assert before[side_values].isna().all().all(), (matrix, side)
            assert (before[flag] == 0.0).all(), (matrix, flag)
            first = frame[frame["season"] == 2013]
            populated = first[flag] == 1.0
            assert populated.any(), (matrix, flag)
            assert _undeclared_blanks(first, populated, others) == [], (matrix, side)
            assert first.loc[~populated, side_values].isna().all().all()
            # Continuity is undefined in the team's FIRST flagged week (step 7b: its
            # window holds a single snap game). Any OTHER undefined week must be one
            # step 8d declared -- the 2013 lock group that could not be scored -- and
            # is checked against that declaration rather than admitted by widening the
            # week set. Was: ``undefined_weeks <= {first flagged week}``, which step
            # 8d's declared 2013 blank in week 3 contradicts.
            first_week = int(first.loc[populated, "week"].min())
            undefined = populated & first[continuity].isna()
            assert undefined.any(), (matrix, continuity)
            beyond = undefined & (first["week"].astype(int) != first_week)
            assert _undeclared_blanks(first, beyond, [continuity]) == [], (
                matrix,
                continuity,
                sorted(set(first.loc[beyond, "week"].astype(int))),
            )
            later = frame[frame["season"] > 2013]
            flagged_later = later[later[flag] == 1.0]
            assert len(flagged_later) > 0, (matrix, flag)
            assert flagged_later[side_values].notna().all().all(), (matrix, side)


@needs_gold
def test_gold_injury_boundary_2008_2009() -> None:
    """Injury: NaN with the flag false for 2002-2008; populated from 2009.

    TRUE ONLY AFTER RUNG 8, where Plan 33.2-17 Task 2's builder first reaches gold: a game that
    admitted no report carries NaN beside ``injury_coverage`` 0.0 rather than "no QB out" and
    "full availability".

    CORRECTED BY PLAN 33.2-18, measured on rung 8's rebuilt gold. 2002-2008 are NaN beside a
    false flag on both sides, as stated. The first COVERED season is per side, not 2009 for both:
    almost no 2009 report carries a ``date_modified`` (p332_ step 6b, Plan 33.2-15), and the one
    2009 game that admitted a report does so for its AWAY team (2009_W17_NO@CAR), so the away
    side's coverage begins in 2009 and the home side's in 2010. Before a side's first covered
    season every value is NaN beside a false flag; every covered row, in every season, is
    populated. A row with the flag false INSIDE a covered season is a within-season gap, which
    p332_ step 7b's point-in-time rule may fill from reports that ended by its lock (owner ruling
    2026-09-22) -- e.g. the 2009 postseason away rows after that week-17 report -- so its value
    is not asserted NaN. The node's name keeps its original spelling so its id is stable. Was:
    both sides ``covered.any()`` in 2009 and every uncovered 2009 row NaN -- premises the
    measured per-side boundary and the step-7b fill contradict.

    CORRECTED AGAIN BY p332_ EXTRA STEP 8d (owner ruling 2026-09-22): in each side's FIRST
    covered season there is no prior season to bootstrap the expanding normalization from,
    so a handful of covered rows have no statistic to be scored against. Those values were
    measured -- the flag says so -- and are now blank rather than the neutral 0.0 that used
    to claim they were exactly average. The allowance is PINNED to step 8d's own per-column
    seasons and cell counts (``_undeclared_blanks``), so a blank in any other season, or one
    cell too many, still fails. Was: every covered row asserted populated.
    """
    for matrix in scan.GOLD_MATRICES:
        frame = _gold(matrix)
        for side in ("home", "away"):
            flag = f"{side}_injury_coverage"
            values = [f"{side}_qb_out_flag", f"{side}_backup_quality_delta"]
            before = frame[frame["season"].between(2002, 2008)]
            assert before[values].isna().all().all(), (matrix, side)
            assert (before[flag] == 0.0).all(), (matrix, flag)
            covered = frame[flag] == 1.0
            assert covered.any(), (matrix, flag)
            first_covered = int(frame.loc[covered, "season"].min())
            assert first_covered in (2009, 2010), (matrix, flag, first_covered)
            ahead = frame[frame["season"] < first_covered]
            assert ahead[values].isna().all().all(), (matrix, side, first_covered)
            assert (ahead[flag] == 0.0).all(), (matrix, flag, first_covered)
            assert _undeclared_blanks(frame, covered, values) == [], (matrix, side)


@needs_gold
def test_gold_coverage_flag_columns_present() -> None:
    """The eight flags this ruling depends on are present in all three matrices.

    ``home_snap_coverage`` / ``away_snap_coverage`` first reach gold at RUNG 8 (Plan 33.2-17
    Task 2 lists the flag in ``SnapCountBuilder._feature_columns()``), and so do
    ``home_off_rolling_cpoe_coverage`` / ``away_off_rolling_cpoe_coverage`` (Plan 33.2-17 Task
    2: the pinned play-by-play carries no cpoe before 2006). The four
    ``*_rolling_opp_adj_coverage`` flags have been present since rung 7 (Plan 33.2-16).
    """
    flags = [
        "home_snap_coverage",
        "away_snap_coverage",
        "home_off_rolling_cpoe_coverage",
        "away_off_rolling_cpoe_coverage",
        *(
            f"{side}_{unit}_rolling_opp_adj_coverage"
            for side in ("home", "away")
            for unit in ("off", "def")
        ),
    ]
    for matrix in scan.GOLD_MATRICES:
        missing = sorted(set(flags) - set(_gold(matrix).columns))
        assert missing == [], (matrix, missing)
