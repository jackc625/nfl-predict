"""Codified AUDIT-02 integrity checks + diagnostic emitter (Phase 20 / D-12).

This is the re-runnable verification harness for AUDIT-02: it exercises the
``DataQualityMonitor`` gold-integrity and abbreviation-mismatch methods added in
plan 20-03 (Task 1) against on-disk data and asserts the breadth-integrity
invariants -- schema width, row counts, 32-team completeness, canonical team
mapping, no abbreviation mismatch -- that previously lived only in
``tests/integration/test_data_completeness.py`` (against synthetic fixtures).
Here they run against the LIVE Bronze/Silver/Gold data, codifying the audit so
it is reproducible (the milestone's namesake, D-12 deliverable a).

Scope guards (honored here):
  - D-05: the 2025 trailing partial-season coverage is an EXPECTED documented
    gap, asserted to be classified ``expected_gap`` -- NOT a hard failure.
  - D-10: this wave catalogs; it does NOT fix. The stale ``16 if week <= 18``
    completeness assumption in ``check_data_completeness`` is recorded as a
    finding in the emitted diagnostic, not patched here.
  - D-13: extends the canonical tooling (``data_qa.py`` + ``tests/``); no
    parallel ``scripts/audit_*.py`` verification story.

A session-scoped fixture also writes ``outputs/diagnostics/audit_integrity.md``
capturing the AUDIT-02 findings with ASCII ``[PASS]``/``[FAIL]`` tags (no emoji,
per CLAUDE.md). That diagnostic is the input plan 20-06 folds into
AUDIT-REPORT.md.

Mirrors the assertion style of ``tests/integration/test_data_completeness.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.data_qa import (
    GOLD_FEATURE_MATRICES,
    GOLD_LAST_COMPLETE_SEASON,
    DataQualityMonitor,
)
from utils.team_data import get_all_teams

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_PATH = PROJECT_ROOT / "outputs" / "diagnostics" / "audit_integrity.md"

EXPECTED_TEAM_COUNT = 32


@pytest.fixture(scope="module")
def monitor() -> DataQualityMonitor:
    """A single DataQualityMonitor reused across the integrity assertions."""
    return DataQualityMonitor()


@pytest.fixture(scope="module")
def gold_result(monitor: DataQualityMonitor) -> dict:
    """The gold-layer integrity result computed once against on-disk data."""
    return monitor.check_gold_integrity()


@pytest.fixture(scope="module")
def abbrev_result(monitor: DataQualityMonitor) -> dict:
    """The 32-team / canonical-abbreviation result against on-disk data."""
    return monitor.check_team_abbreviations()


# ---------------------------------------------------------------------------
# (a) Canonical team set + abbreviation mismatch
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCanonicalAbbreviations:
    """AUDIT-02: every on-disk abbreviation is canonical and normalizes cleanly."""

    def test_canonical_set_is_32_teams(self, abbrev_result: dict):
        """utils.team_data.get_all_teams() defines exactly 32 canonical teams."""
        assert len(set(get_all_teams())) == EXPECTED_TEAM_COUNT
        check = abbrev_result["checks"]["canonical_team_count"]
        assert check["status"] == "pass", check
        assert check["count"] == EXPECTED_TEAM_COUNT

    def test_silver_games_teams_are_canonical(self, abbrev_result: dict):
        """Every home_team / away_team in silver games is in get_all_teams()."""
        check = abbrev_result["checks"]["games_canonical_teams"]
        assert check["status"] == "pass", (
            f"Non-canonical team abbreviations in silver games: "
            f"{check['non_canonical']}"
        )
        assert check["non_canonical"] == []

    def test_gold_game_id_teams_are_canonical(self, abbrev_result: dict):
        """Teams encoded in every gold matrix game_id are all canonical."""
        for table in GOLD_FEATURE_MATRICES:
            check = abbrev_result["checks"][f"{table}_game_id_teams"]
            assert check["status"] == "pass", (
                f"{table}: non_canonical={check['non_canonical']} "
                f"unparseable={check['unparseable_sample']}"
            )
            assert check["non_canonical"] == []
            assert check["unparseable_sample"] == []

    def test_normalize_does_not_hard_fail_on_any_abbreviation(
        self, abbrev_result: dict
    ):
        """normalize_team_abbreviation raises on none of the on-disk abbreviations.

        This is the abbreviation-mismatch check: normalize_team_abbreviation
        hard-fails on unknown abbreviations, so a clean run proves no mismatch.
        """
        check = abbrev_result["checks"]["abbreviation_mismatch"]
        assert check["status"] == "pass", (
            f"normalize_team_abbreviation hard-failed on: {check['normalize_failures']}"
        )
        assert check["normalize_failures"] == []
        assert check["abbreviations_checked"] >= EXPECTED_TEAM_COUNT


# ---------------------------------------------------------------------------
# (b) Gold-layer matrix integrity
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGoldIntegrity:
    """AUDIT-02: gold matrices load with expected schema width and non-empty rows."""

    def test_all_three_matrices_have_expected_columns_and_rows(self, gold_result: dict):
        """features_wp/ats/ou load with the GOLD_FEATURE_MATRICES widths and >0 rows.

        Widths are 194/195/194 after the Phase 28 snap/injury/spot widening (the
        expected counts are read from GOLD_FEATURE_MATRICES, not hardcoded here).
        """
        for table, expected_columns in GOLD_FEATURE_MATRICES.items():
            check = gold_result["checks"][table]
            assert check["status"] == "pass", check
            assert check["column_count"] == expected_columns, (
                f"{table}: expected {expected_columns} columns, "
                f"got {check['column_count']}"
            )
            assert check["row_count"] > 0, f"{table} is empty"

    def test_no_all_null_feature_columns(self, gold_result: dict):
        """No gold matrix has an entirely-null column (a broken-builder signal)."""
        for table in GOLD_FEATURE_MATRICES:
            check = gold_result["checks"][table]
            assert check["all_null_columns"] == [], (
                f"{table} has all-null columns: {check['all_null_columns']}"
            )

    def test_season_span_present(self, gold_result: dict):
        """Each matrix reports a season span (min/max present)."""
        for table in GOLD_FEATURE_MATRICES:
            check = gold_result["checks"][table]
            assert check["season_min"] is not None
            assert check["season_max"] is not None
            assert check["season_min"] <= check["season_max"]


# ---------------------------------------------------------------------------
# (c) 2025 trailing gap is an EXPECTED documented gap, not a failure (D-05)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTrailingSeasonGap:
    """D-05: the 2025 partial-season coverage is classified expected, not a fail."""

    def test_trailing_gap_classified_expected(self, gold_result: dict):
        """The trailing season beyond the last-complete season is expected_gap."""
        for table in GOLD_FEATURE_MATRICES:
            check = gold_result["checks"][table]
            # If gold has advanced past the last fully-ingested season, the
            # trailing coverage must be flagged informational, never a fail.
            if check["season_max"] > GOLD_LAST_COMPLETE_SEASON:
                gap = check.get("trailing_season_gap")
                assert gap is not None, (
                    f"{table} has season {check['season_max']} > "
                    f"{GOLD_LAST_COMPLETE_SEASON} but no trailing_season_gap entry"
                )
                assert gap["status"] == "expected_gap", (
                    f"{table} trailing gap must be expected_gap, got {gap['status']}"
                )
                assert gap["row_count"] > 0
                assert gap["weeks_present"]
                # A trailing partial season must NOT flip the matrix to fail.
                assert check["status"] == "pass"


# ---------------------------------------------------------------------------
# Diagnostic emitter -- writes outputs/diagnostics/audit_integrity.md (D-12)
# ---------------------------------------------------------------------------


def _tag(status: str) -> str:
    """Map an internal status to an ASCII PASS/FAIL tag (no emoji, CLAUDE.md)."""
    return "[PASS]" if status in ("pass", "expected_gap") else "[FAIL]"


@pytest.mark.integration
def test_emit_audit_integrity_diagnostic(
    gold_result: dict, abbrev_result: dict
) -> None:
    """Write the AUDIT-02 findings to outputs/diagnostics/audit_integrity.md.

    This is the diagnostic plan 20-06 folds into AUDIT-REPORT.md. It captures
    pass/fail per check, the team-set verification result, the confirmed 2025
    coverage, and the stale ``16 if week <= 18`` completeness assumption noted
    as a finding (D-10 capture, not fixed). ASCII tags only, no emoji.
    """
    lines: list[str] = []
    lines.append("# AUDIT-02 Integrity Diagnostic")
    lines.append("")
    lines.append(
        "AUDIT-02 breadth integrity (schema, row counts, 32-team completeness, "
        "canonical mapping, no abbreviation mismatch) re-verified against on-disk "
        "Bronze/Silver/Gold data by `tests/integration/test_audit_integrity.py` "
        "exercising the `DataQualityMonitor` gold-integrity + abbreviation-mismatch "
        "methods (plan 20-03). Status tags are ASCII `[PASS]` / `[FAIL]` (no emoji)."
    )
    lines.append("")
    lines.append("This file is the input plan 20-06 folds into AUDIT-REPORT.md.")
    lines.append("")

    # -- Gold-layer integrity --------------------------------------------------
    lines.append("## Gold-layer feature-matrix integrity")
    lines.append("")
    lines.append(
        "| Matrix | Status | Rows | Columns (got/expected) | Season span | "
        "All-null columns |"
    )
    lines.append(
        "|--------|--------|------|------------------------|-------------|"
        "------------------|"
    )
    for table, expected_columns in GOLD_FEATURE_MATRICES.items():
        check = gold_result["checks"][table]
        lines.append(
            f"| `{table}` | {_tag(check['status'])} | {check['row_count']} | "
            f"{check['column_count']}/{expected_columns} | "
            f"{check['season_min']}-{check['season_max']} | "
            f"{check['all_null_columns'] or 'none'} |"
        )
    lines.append("")

    # -- 2025 trailing currency gap (D-05) -------------------------------------
    lines.append("## 2025 trailing currency gap (D-05) -- documented, expected")
    lines.append("")
    for table in GOLD_FEATURE_MATRICES:
        gap = gold_result["checks"][table].get("trailing_season_gap")
        if gap:
            lines.append(
                f"- `{table}`: {_tag(gap['status'])} season {gap['season']} present "
                f"for weeks {gap['weeks_present']} ({gap['row_count']} rows). "
                f"Classified `expected_gap` -- NOT a failure; not backfilled this "
                f"phase (D-05)."
            )
    lines.append("")

    # -- 32-team / canonical abbreviations -------------------------------------
    lines.append("## 32-team completeness + canonical abbreviation mismatch")
    lines.append("")
    lines.append("| Check | Status | Detail |")
    lines.append("|-------|--------|--------|")
    for name, check in abbrev_result["checks"].items():
        if not isinstance(check, dict) or "status" not in check:
            continue
        detail_bits = []
        if "count" in check:
            detail_bits.append(f"count={check['count']}")
        if "distinct_team_count" in check:
            detail_bits.append(f"distinct_teams={check['distinct_team_count']}")
        if "non_canonical" in check:
            detail_bits.append(f"non_canonical={check['non_canonical'] or 'none'}")
        if "abbreviations_checked" in check:
            detail_bits.append(f"checked={check['abbreviations_checked']}")
        if "normalize_failures" in check:
            detail_bits.append(
                f"normalize_failures={check['normalize_failures'] or 'none'}"
            )
        lines.append(
            f"| `{name}` | {_tag(check['status'])} | {', '.join(detail_bits)} |"
        )
    lines.append("")

    # -- Findings captured (not fixed, D-10) -----------------------------------
    lines.append("## Findings captured (not fixed this wave, D-10)")
    lines.append("")
    lines.append(
        "- **F-INTEG-01 (stale completeness assumption):** "
        "`scripts/data_qa.py` `check_data_completeness` hard-codes "
        "`expected_count = 16 if week <= 18 else None` (the per-week games count). "
        "This is wrong for the 17-game era (2021+, ~16 games/week but 18 weeks) "
        "and for partial/offseason weeks; the AUDIT-02 gold-integrity check "
        "deliberately does NOT assert against it. Captured for AUDIT-REPORT.md "
        "as a robustness finding (correctness-scoped only if it produces a wrong "
        "gold value -- it does not; it only mis-labels Silver completeness "
        "percentages). [FAIL-as-finding]"
    )
    lines.append("")
    lines.append(
        "- **F-INTEG-02 (silver odds/weather carry no team columns):** "
        "`odds_snapshot` and `weather_forecast` silver tables key by `game_id` only "
        "(no `home_team`/`away_team` columns), so the abbreviation-mismatch check "
        "derives their teams transitively via the gold `game_id` parse and the "
        "silver `games` table. Informational -- not a defect. [PASS]"
    )
    lines.append("")

    # -- Overall verdict -------------------------------------------------------
    gold_statuses = [
        c["status"]
        for c in gold_result["checks"].values()
        if isinstance(c, dict) and "status" in c
    ]
    abbrev_statuses = [
        c["status"]
        for c in abbrev_result["checks"].values()
        if isinstance(c, dict) and "status" in c
    ]
    any_fail = any(s == "fail" for s in gold_statuses + abbrev_statuses)
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"AUDIT-02 integrity as-found: {'[FAIL]' if any_fail else '[PASS]'} "
        "(the 2025 trailing gap is classified expected and does not count as a "
        "failure, per D-05)."
    )
    lines.append("")

    content = "\n".join(lines) + "\n"

    # Guard: no emoji / non-ASCII (CLAUDE.md).
    assert content.isascii(), "Diagnostic must be pure ASCII (no emoji)"

    DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_PATH.write_text(content, encoding="utf-8")

    assert DIAGNOSTIC_PATH.exists()
    written = DIAGNOSTIC_PATH.read_text(encoding="utf-8")
    assert "[PASS]" in written
    assert "AUDIT-02" in written
