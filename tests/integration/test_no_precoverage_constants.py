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


def _latest_section() -> tuple[str, str]:
    """(heading, body) of the LAST ``## Before/After rung 8`` section in the document."""
    assert DOCUMENT.is_file(), (
        f"{DOCUMENT} is missing: the before-state was never published"
    )
    text = DOCUMENT.read_text(encoding="utf-8")
    starts = [
        (text.rfind(heading), heading)
        for heading in (BEFORE_SECTION, AFTER_SECTION)
        if heading in text
    ]
    position, heading = max(starts)
    return heading, text[position:]


def _recorded(body: str, key: str) -> int:
    match = re.search(rf"^{key}:\s*(\d+)\s*$", body, flags=re.MULTILINE)
    assert match, f"the latest section records no machine-readable '{key}:' line"
    return int(match.group(1))


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
        heading, body = _latest_section()
        suffix = "BEFORE_RUNG8" if heading == BEFORE_SECTION else "AFTER_RUNG8"
        assert _recorded(body, f"UNFLAGGED_{suffix}") == len(_live()["unflagged"])
        assert _recorded(body, f"CLASSIFIED_{suffix}") == _live()["classified"]

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


@needs_gold
def test_gold_snap_boundary_2012_2013() -> None:
    """Snaps: NaN with the flag false for 2002-2012; populated, flag true, from 2013.

    TRUE ONLY AFTER RUNG 8, where Plan 33.2-17 Task 2's builder (NaN before coverage, and
    ``snap_coverage`` in ``SnapCountBuilder._feature_columns()``) first reaches gold. A 2013 row
    with no admitted snap game yet (its team's first game) is NaN with the flag false too: the
    flag says whether THIS value was measured.
    """
    for matrix in scan.GOLD_MATRICES:
        frame = _gold(matrix)
        values = _snap_value_columns(frame)
        assert values, "non-vacuity"
        for side in ("home", "away"):
            flag = f"{side}_snap_coverage"
            side_values = [c for c in values if c.startswith(f"{side}_")]
            before = frame[frame["season"].between(2002, 2012)]
            assert before[side_values].isna().all().all(), (matrix, side)
            assert (before[flag] == 0.0).all(), (matrix, flag)
            first = frame[frame["season"] == 2013]
            populated = first[flag] == 1.0
            assert populated.any(), (matrix, flag)
            assert first.loc[populated, side_values].notna().all().all()
            assert first.loc[~populated, side_values].isna().all().all()


@needs_gold
def test_gold_injury_boundary_2008_2009() -> None:
    """Injury: NaN with the flag false for 2002-2008; populated from 2009.

    TRUE ONLY AFTER RUNG 8, where Plan 33.2-17 Task 2's builder first reaches gold: a game that
    admitted no report carries NaN beside ``injury_coverage`` 0.0 rather than "no QB out" and
    "full availability". 2009 is populated where a report was admitted (a handful of dated
    rows); every 2009 row without one is NaN with the flag false.
    """
    for matrix in scan.GOLD_MATRICES:
        frame = _gold(matrix)
        for side in ("home", "away"):
            flag = f"{side}_injury_coverage"
            values = [f"{side}_qb_out_flag", f"{side}_backup_quality_delta"]
            before = frame[frame["season"].between(2002, 2008)]
            assert before[values].isna().all().all(), (matrix, side)
            assert (before[flag] == 0.0).all(), (matrix, flag)
            first = frame[frame["season"] == 2009]
            covered = first[flag] == 1.0
            assert covered.any(), (matrix, flag)
            assert first.loc[covered, values].notna().all().all()
            assert first.loc[~covered, values].isna().all().all()


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
