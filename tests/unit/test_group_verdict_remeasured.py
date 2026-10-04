"""Permanent doc-drift guard for the repo-root GROUP-VERDICT-READOUT.md (Plan 33.2-22).

Modelled on ``tests/unit/test_signal_lift_readout_md.py``, the committed doc-drift-guard
pattern in this repository, with ONE deliberate strengthening: the before/after structure is
checked over PARSED TABLE ROWS rather than over phrases.

WHY THE STRUCTURE IS PARSED AND NOT GREPPED. The property this readout exists to hold is that
every screened group has a row for every target, that every row carries BOTH verdicts, and that
a row whose two verdicts differ is MARKED as flipped. A phrase-level check for the words
"before" and "after" has no reachable failing state -- both words occur in essentially any
English document -- so it would have reported success while the table silently lost a row.
:func:`parse_verdict_rows` is exposed at module level so the plan's own verification command
drives the SAME parser these tests drive, rather than a second copy of its logic.

THE ANTI-ROT GUARD ASSERTS THE RULING, NEVER THE POINT ESTIMATE. It re-runs the screen exactly
as ``scripts/remeasure_group_verdict.py`` does -- ``objective="outcome_loss"``, the derived
2025-excluding config, no closing-odds frame -- and asserts the VERDICT it returns is the
verdict the document records as current. Pinning a point estimate to moving gold is the mistake
this repository already made once (D29-06-02), and the correction it adopted was to pin the
RULING. The harness half is additionally generation-gated: a reading measured on other gold is
not a wrong reading, so it SKIPS rather than fails.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import subprocess
import tomllib
from pathlib import Path

import pytest

from backtest.ev_chain_constants import HOLD_SEASONS_P31
from backtest.group_gate_constants import GRID_GROUPS, GRID_TARGETS
from tests.gold_generation import require_gold_generation
from tests.phase30_state import GROUP_VERDICT_FILE_SHA256, MEASUREMENT_COMMIT
from tests.phase33_state import P332_20_CLEAN_BUILD_GOLD_GENERATION

REPO_ROOT = Path(__file__).resolve().parents[2]
READOUT_MD = REPO_ROOT / "docs" / "records" / "GROUP-VERDICT-READOUT.md"
VERDICT_TOML = REPO_ROOT / "config" / "group_gate_verdict.toml"

_GOLD_PATHS = {
    target: REPO_ROOT / "data" / "gold" / f"features_{target}.parquet"
    for target in GRID_TARGETS
}

#: Required section markers. Each is a FRAGMENT of a section rather than a whole heading, so a
#: benign reword passes and a dropped section fails.
_REQUIRED_SECTION_MARKERS = (
    "What a feature-group verdict is",
    "The before and after",
    "The two objectives",
    "The gold each verdict was measured on",
    "The seasons each verdict was measured over",
    "Where the Phase-30 verdict came from",
    "What this readout does and does not say",
)

#: The not-clean-evidence label (D33.2-07), in the same words the other readouts use.
_NOT_CLEAN_EVIDENCE_LABEL = "not clean evidence"
_D33207_SENTENCE = "Only the 2026 season, recorded live\nunder the day-before 6 PM ET lock, counts (D33.2-07)"

#: The over-claim word this document may never contain.
_FORBIDDEN_WORDS = ("proven",)

#: The two objectives, which must BOTH be named so the table cannot be read like-for-like.
_CLOSING_OBJECTIVE_PHRASE = "closing-line clv"
_OUTCOME_OBJECTIVE_PHRASE = "outcome-loss"

_FLIP_MARKER = "FLIPPED"


def _read_readout() -> str:
    """Read GROUP-VERDICT-READOUT.md from the repo root."""
    return READOUT_MD.read_text(encoding="utf-8")


def parse_verdict_rows(text: str) -> list[dict[str, object]]:
    """Parse the before/after table into per-(group, target) rows.

    A row is any markdown table line whose FIRST cell names a ``GRID_GROUPS`` member and whose
    SECOND cell names a ``GRID_TARGETS`` member. Header, separator and summary rows therefore
    drop out without needing to be recognised.

    Args:
        text: The readout's full text.

    Returns:
        One dict per row: ``group``, ``target``, ``phase30_statistic``, ``phase30_p``,
        ``phase30_verdict``, ``remeasured_statistic``, ``remeasured_p``,
        ``remeasured_verdict``, ``flipped``.
    """
    groups = set(GRID_GROUPS)
    targets = set(GRID_TARGETS)
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [
            cell.strip().strip("*").strip() for cell in stripped.strip("|").split("|")
        ]
        if len(cells) < 9:
            continue
        if cells[0] not in groups or cells[1] not in targets:
            continue
        rows.append(
            {
                "group": cells[0],
                "target": cells[1],
                "phase30_statistic": cells[2],
                "phase30_p": cells[3],
                "phase30_verdict": cells[4],
                "remeasured_statistic": cells[5],
                "remeasured_p": cells[6],
                "remeasured_verdict": cells[7],
                "flipped": _FLIP_MARKER in cells[8].upper(),
            }
        )
    return rows


def _live_verdicts() -> dict[str, str]:
    """{group -> verdict} from the LIVE, re-measured config/group_gate_verdict.toml."""
    with VERDICT_TOML.open("rb") as handle:
        document = tomllib.load(handle)
    return {
        group: entry["verdict"]
        for group, entry in document["stage1"]["verdicts"].items()
    }


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, check=False
    )


def _git_history_is_unavailable() -> bool:
    if _git("rev-parse", "--git-dir").returncode != 0:
        return True
    shallow = _git("rev-parse", "--is-shallow-repository")
    return shallow.returncode != 0 or shallow.stdout.decode().strip() == "true"


def _phase30_verdicts() -> dict[str, str]:
    """{group -> verdict} from the RATIFIED Phase-30 document's immutable git blob.

    Read from ``MEASUREMENT_COMMIT`` and digest-checked against ``GROUP_VERDICT_FILE_SHA256``
    over newline-normalized bytes -- never from a copy taken during this plan, and never from
    the working-tree file, which is now the RE-MEASURED verdict.
    """
    blob = _git(
        "cat-file", "blob", f"{MEASUREMENT_COMMIT}:config/group_gate_verdict.toml"
    )
    assert blob.returncode == 0, (
        f"git could not read the ratified Phase-30 verdict at {MEASUREMENT_COMMIT}: "
        f"{blob.stderr.decode(errors='replace')}"
    )
    normalized = blob.stdout.replace(b"\r\n", b"\n")
    assert hashlib.sha256(normalized).hexdigest() == GROUP_VERDICT_FILE_SHA256, (
        "the ratified Phase-30 verdict no longer matches its anchor; that document is history "
        "and cannot change"
    )
    document = tomllib.loads(normalized.decode("utf-8"))
    return {
        group: entry["verdict"]
        for group, entry in document["stage1"]["verdicts"].items()
    }


class TestReadoutExists:
    """The deliverable is at the repo root, ASCII, and does not over-claim."""

    def test_file_exists_at_repo_root(self) -> None:
        assert READOUT_MD.is_file(), f"missing: {READOUT_MD}"

    def test_content_is_ascii(self) -> None:
        assert _read_readout().isascii(), (
            "GROUP-VERDICT-READOUT.md contains non-ASCII characters"
        )

    def test_no_over_claim_words(self) -> None:
        lowered = _read_readout().lower()
        present = [word for word in _FORBIDDEN_WORDS if word in lowered]
        assert not present, (
            f"GROUP-VERDICT-READOUT.md over-claims: {present}. A feature-group verdict decides "
            "which inputs a fit may draw on; it establishes nothing."
        )

    def test_all_required_sections_are_present(self) -> None:
        content = _read_readout()
        missing = [m for m in _REQUIRED_SECTION_MARKERS if m not in content]
        assert not missing, f"missing required sections: {missing}"


class TestTheNotCleanEvidenceLabel:
    """D33.2-07: a re-measured past season is published, and labelled."""

    def test_the_label_is_present(self) -> None:
        assert _NOT_CLEAN_EVIDENCE_LABEL in _read_readout().lower()

    def test_the_d33207_sentence_is_present_in_the_standing_words(self) -> None:
        assert _D33207_SENTENCE in _read_readout(), (
            "the readout must carry the D33.2-07 sentence in the same words every other "
            "readout in this repository uses, so the label cannot be read as this document's "
            "own hedge"
        )


class TestTheBeforeAfterTable:
    """The structure is checked as PARSED ROWS and VALUES, never as two common words."""

    def test_the_table_is_not_empty(self) -> None:
        """NON-VACUITY: an empty parse would make every assertion below pass."""
        rows = parse_verdict_rows(_read_readout())
        assert rows, "parse_verdict_rows found no rows at all"
        assert GRID_GROUPS, "GRID_GROUPS is empty"
        assert len(rows) == len(GRID_GROUPS) * len(GRID_TARGETS)

    def test_every_group_has_a_row_for_every_target(self) -> None:
        rows = parse_verdict_rows(_read_readout())
        expected = {(g, t) for g in GRID_GROUPS for t in GRID_TARGETS}
        got = {(r["group"], r["target"]) for r in rows}
        assert not expected - got, f"missing rows: {sorted(expected - got)}"

    def test_every_row_carries_both_verdict_columns(self) -> None:
        rows = parse_verdict_rows(_read_readout())
        empty = [
            (r["group"], r["target"])
            for r in rows
            if not r["phase30_verdict"] or not r["remeasured_verdict"]
        ]
        assert not empty, f"rows missing a verdict column: {empty}"

    def test_every_row_whose_verdicts_differ_is_marked_flipped(self) -> None:
        """A silently-replaced verdict is exactly what this readout exists to prevent."""
        rows = parse_verdict_rows(_read_readout())
        unmarked = [
            (r["group"], r["target"])
            for r in rows
            if r["phase30_verdict"] != r["remeasured_verdict"] and not r["flipped"]
        ]
        assert not unmarked, f"flipped rows not marked as flipped: {unmarked}"

    def test_no_unchanged_row_is_marked_flipped(self) -> None:
        """NO FALSE POSITIVE: the marker means what it says in both directions."""
        rows = parse_verdict_rows(_read_readout())
        wrong = [
            (r["group"], r["target"])
            for r in rows
            if r["phase30_verdict"] == r["remeasured_verdict"] and r["flipped"]
        ]
        assert not wrong, f"unchanged rows marked as flipped: {wrong}"

    def test_the_parser_rejects_a_planted_malformed_row(self) -> None:
        """PLANTED VIOLATION: the parser must not invent a row out of prose."""
        planted = "| injury | wp | +1 | 0.1 | DROP |\n| not_a_group | wp | a | b | c | d | e | f |\n"
        assert parse_verdict_rows(planted) == []


class TestTheReadoutMatchesTheCommittedVerdicts:
    """The two columns are the two committed documents, not a retelling of them."""

    def test_the_remeasured_column_matches_the_live_verdict_file(self) -> None:
        rows = parse_verdict_rows(_read_readout())
        live = _live_verdicts()
        for row in rows:
            assert row["remeasured_verdict"] == live[row["group"]], (
                f"the readout records {row['remeasured_verdict']!r} for "
                f"{row['group']}/{row['target']} while config/group_gate_verdict.toml says "
                f"{live[row['group']]!r}"
            )

    def test_the_phase30_column_matches_the_ratified_blob(self) -> None:
        if _git_history_is_unavailable():
            pytest.skip(
                "git history is unavailable, so the ratified blob cannot be read"
            )
        rows = parse_verdict_rows(_read_readout())
        ratified = _phase30_verdicts()
        for row in rows:
            assert row["phase30_verdict"] == ratified[row["group"]], (
                f"the readout records {row['phase30_verdict']!r} for {row['group']} while the "
                f"ratified Phase-30 document at {MEASUREMENT_COMMIT} says "
                f"{ratified[row['group']]!r}"
            )

    def test_the_excluded_groups_list_agrees_with_the_published_verdicts(self) -> None:
        with VERDICT_TOML.open("rb") as handle:
            document = tomllib.load(handle)
        rows = parse_verdict_rows(_read_readout())
        published_non_keep = {
            r["group"] for r in rows if r["remeasured_verdict"] != "KEEP"
        }
        assert set(document["excluded_groups"]) == published_non_keep


class TestTheObjectivesAreBothNamed:
    """Two different measurements in one table, labelled as such."""

    def test_both_objectives_are_named(self) -> None:
        lowered = _read_readout().lower()
        assert _CLOSING_OBJECTIVE_PHRASE in lowered
        assert _OUTCOME_OBJECTIVE_PHRASE in lowered

    def test_the_different_units_statement_is_present(self) -> None:
        lowered = _read_readout().lower()
        assert "different things in different units" in lowered
        assert "compared only through their verdicts" in lowered, (
            "the readout must state that the two statistics are compared only through their "
            "VERDICTS, so the table cannot be read as a like-for-like change in one number"
        )

    def test_both_gold_digests_are_recorded(self) -> None:
        content = _read_readout()
        assert P332_20_CLEAN_BUILD_GOLD_GENERATION in content
        from tests.phase30_state import ACCEPTED_RUNG4_FINGERPRINT_SHA256

        assert ACCEPTED_RUNG4_FINGERPRINT_SHA256 in content

    def test_the_spent_hold_exclusion_is_stated(self) -> None:
        content = _read_readout()
        for season in HOLD_SEASONS_P31:
            assert str(season) in content
        assert "HOLD_SEASONS_P31" in content


class TestTheAntiRotGuard:
    """Re-run the screen exactly as the script does; assert the RULING, never the estimate."""

    def test_the_harness_returns_the_ruling_the_document_records(self) -> None:
        for target, path in _GOLD_PATHS.items():
            if not path.is_file():
                pytest.skip(f"gold matrix for {target} is absent: {path}")
        require_gold_generation(
            P332_20_CLEAN_BUILD_GOLD_GENERATION,
            reading="GROUP-VERDICT-READOUT.md's re-measured feature-group verdicts",
            moved_by="a gold rebuild after Plan 33.2-20's clean build",
            recorded_in="config/group_gate_verdict.toml's [remeasurement] table",
        )
        from scripts.remeasure_group_verdict import remeasure

        result = remeasure()
        published = {
            r["group"]: r["remeasured_verdict"]
            for r in parse_verdict_rows(_read_readout())
        }
        assert published, "the readout published no verdicts to check"
        for group, entry in result["verdicts"].items():
            assert entry["verdict"] == published[group], (
                f"the committed harness now rules {entry['verdict']!r} for {group!r} while "
                f"GROUP-VERDICT-READOUT.md records {published[group]!r}. Reconcile the "
                "document onto the current ruling; never re-anchor this guard to make it pass."
            )
