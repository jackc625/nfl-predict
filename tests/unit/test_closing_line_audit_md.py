"""The committed closing-line fit audit cannot rot or soften (Plan 33.2-21 Task 3, SPEC R7).

Written on the ``tests/unit/test_signal_lift_readout_md.py`` doc-drift pattern. What it
guards, and why each guard is shaped the way it is:

**Every cell-level assertion resolves its column BY NAME from the header row and reads that
ONE CELL -- never the row line.** This is not fastidiousness. The
``config/gate.toml [baseline.*]`` row's argument NAMES ``GRADING_ONLY`` and ``DELETED`` in
order to reject both, while its own verdict is ``OPEN_ACCEPTED``. Under a whole-line scan
that row reads as three-dispositioned, and the document could not be written correctly and
pass at the same time. Reasoning prose is not a verdict; the ``Disposition`` cell is.

**``PROSE_ONLY_VOCAB`` must be NON-EMPTY.** It is the no-false-positive control that PINS
the match mode to the cell. If anyone re-widens the scan back to the row line, the same row
trips the two-disposition assertion, so the control and the guard fail together rather than
the guard quietly forbidding the prose the document requires.

**One site per row.** A bundled ``Site`` cell contributes one entry to the listed-sites set
while covering two sites in the tree, which is exactly what would let one of them slip past
the anti-rot comparison -- and the two-disposition check structurally cannot catch it,
because a bundle's members usually share a verdict.

**Reader completeness, not reader presence.** D33.2-26 as corrected requires EVERY live
reader of a retained dead comparator to be named. A guard satisfied by SOME reader is a
softer version of the false claim the decision exists to forbid, so the gate-baseline row is
checked against both a pinned list and a STRUCTURAL scan of the tree.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: The audit lives at the repository root, where a reader will find it without being told.
AUDIT_PATH = Path("CLOSING-LINE-AUDIT.md")

#: The CLOSED five-value vocabulary. ``OPEN_ACCEPTED`` exists because D33.2-26 measured that
#: the gate-baseline row could not be written truthfully with four: it has a live reader, so
#: DELETED is false; it grades nothing, so GRADING_ONLY is false; nothing moved it; and it
#: was derived from closing-line CLV. A vocabulary with no true value for a row forces
#: either a false cell or a silent omission, which is what this document exists to stop.
VOCABULARY = (
    "MOVED_TO_PRELOCK",
    "GRADING_ONLY",
    "DELETED",
    "OPEN_ACCEPTED",
    "NOT_A_CLOSING_LINE_FIT",
)

#: The exact header, in order. Column indices are resolved from it, so a renamed or
#: reordered header fails loudly rather than silently reading the wrong cell.
HEADER = ("Site", "Used for", "Disposition", "Discharged by", "Reasoning")

#: The sections a reader needs in order to use this document rather than merely read it.
REQUIRED_SECTIONS = (
    "## What this document is",
    "## What history the fit actually has, stated plainly",
    "## The vocabulary, and what each word commits to",
    "## The table",
    "## Anti-rot",
)

#: The three live readers of ``config/gate.toml [baseline.*]``, plus the line where the
#: flattening function is DEFINED. MEASURED 2026-09-16, re-measured 2026-09-22.
REQUIRED_GATE_BASELINE_READERS = (
    "promote_models.py:915",
    "promote_models.py:1589",
    "retrain_models.py:380",
    "test_frozen_baseline_matches_rescore",
)

#: The symbol through which ``config/gate.toml [baseline.*]`` reaches the deploy gate. Held
#: as a STRING and never imported or called here, so this module does not make itself one of
#: the readers it is auditing.
BASELINE_BUNDLE_SYMBOL = "_baseline_bundle"

#: Every symbol through which a closing line reaches a FIT in this repository. The anti-rot
#: check below fails when one of these exists in the tree and appears nowhere in the table.
CLOSING_LINE_FIT_SYMBOLS = (
    BASELINE_BUNDLE_SYMBOL,
    "extract_noise_profile",
    "_walkforward_clv_series",
    "fit_frozen_residual_sd",
    "EV_FLOOR_GRID",
    "run_comparison",
    "_gate_per_target",
    "TUNING_SEASONS",
)

#: A symbol that certainly EXISTS in the tree and certainly does NOT belong in the audit.
#: It is the planted-violation control: the anti-rot helper must report it.
CONTROL_SYMBOL_NOT_IN_THE_AUDIT = "moneyline_to_probability"

_SCAN_ROOTS = (
    "scripts",
    "backtest",
    "models",
    "api",
    "pipeline",
    "features",
    "conf",
    "utils",
    "data",
    "ratings",
    "tests",
)


# ---------------------------------------------------------------------------
# Parsing the table, by column NAME
# ---------------------------------------------------------------------------


def _cells(line: str) -> list[str]:
    return [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]


def _table_lines(text: str) -> list[str]:
    """Every markdown table line that carries content (the ``|---|`` rule is not one)."""
    return [
        line
        for line in text.splitlines()
        if line.strip().startswith("|") and set(line.strip()) - set("|-: ")
    ]


@pytest.fixture(scope="module")
def audit_text() -> str:
    assert AUDIT_PATH.exists(), f"{AUDIT_PATH} is missing from the repository root"
    return AUDIT_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def rows(audit_text: str) -> list[dict[str, str]]:
    """The data rows, each as a ``{column name: cell}`` mapping."""
    lines = _table_lines(audit_text)
    assert lines, "the audit carries no table"
    header = _cells(lines[0])
    assert header == list(HEADER), (
        f"the table header is {header}, not {list(HEADER)}. Column indices are resolved "
        "by name, so a renamed or reordered header must fail here rather than silently "
        "shift every later cell."
    )
    return [dict(zip(header, _cells(line), strict=False)) for line in lines[1:]]


def _disposition_tokens(row: dict[str, str]) -> list[str]:
    """The vocabulary tokens in a row's DISPOSITION CELL. Never the row line."""
    return [token for token in VOCABULARY if token in row["Disposition"]]


def _row_named(rows: list[dict[str, str]], needle: str) -> dict[str, str]:
    matches = [row for row in rows if needle in row["Site"]]
    assert len(matches) == 1, (
        f"expected exactly one row whose Site names {needle!r}, found {len(matches)}"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# The document itself
# ---------------------------------------------------------------------------


class TestTheDocument:
    def test_it_exists_at_the_repository_root(self) -> None:
        assert AUDIT_PATH.exists()

    def test_it_is_ascii(self, audit_text: str) -> None:
        assert audit_text.isascii()

    def test_it_does_not_claim_anything_is_proven(self, audit_text: str) -> None:
        assert "proven" not in audit_text.lower()

    @pytest.mark.parametrize("section", REQUIRED_SECTIONS)
    def test_it_carries_its_required_sections(
        self, audit_text: str, section: str
    ) -> None:
        assert section in audit_text

    def test_the_preamble_defines_all_three_live_words(self, audit_text: str) -> None:
        preamble = audit_text.split("## The table", maxsplit=1)[0]
        for token in ("GRADING_ONLY", "DELETED", "OPEN_ACCEPTED"):
            assert token in preamble, (
                f"{token} is a live verdict in this document and must be defined before "
                "the table, so a later reader cannot re-derive a softer reading of it"
            )

    def test_it_states_what_history_the_prelock_fit_actually_has(
        self, audit_text: str
    ) -> None:
        """The owned line history is five seasons. The document must not imply more.

        No 2018-2025 stored closing line carries a genuine capture time, so none of them
        can stand in for the seasons the purchase does not cover (D33.2-22 ruling of Plan
        33.2-14).
        """
        coverage = audit_text.split("## What history the fit actually has")[1].split(
            "## The vocabulary", maxsplit=1
        )[0]
        from tests.phase33_state import P332_24B_CONVERTER_N_GAMES

        assert "2020-2024" in coverage
        # Was: "1,342". Plan 33.2-24 step 24b re-fitted the converter on the repaired
        # corpus, so the document states the count the live converter was fitted on.
        assert f"{P332_24B_CONVERTER_N_GAMES:,}" in coverage
        assert "no genuine capture time" in coverage
        for absent_season in ("2018-2019", "2025"):
            assert absent_season in coverage

    def test_it_keeps_the_blend_input_boundary_explicit(self, audit_text: str) -> None:
        """A blend input is not a model input, and the document must say which is which."""
        assert "blend input" in audit_text.lower()
        assert "not a model input" in audit_text.lower()
        assert "D33.2-03" in audit_text


# ---------------------------------------------------------------------------
# The table's shape
# ---------------------------------------------------------------------------


class TestTheTable:
    def test_it_lists_at_least_the_thirteen_known_sites(
        self, rows: list[dict[str, str]]
    ) -> None:
        assert len(rows) >= 13

    def test_no_row_is_ragged(self, rows: list[dict[str, str]]) -> None:
        ragged = [row["Site"][:40] for row in rows if len(row) != len(HEADER)]
        assert ragged == [], (
            "a row with a missing cell shifts every later cell left, so a disposition "
            "check would silently read the wrong column"
        )

    def test_every_disposition_cell_holds_exactly_one_token(
        self, rows: list[dict[str, str]]
    ) -> None:
        zero = [row["Site"][:40] for row in rows if not _disposition_tokens(row)]
        many = [row["Site"][:40] for row in rows if len(_disposition_tokens(row)) > 1]
        assert zero == [], "a row with no disposition is not written"
        assert many == [], (
            "a row with two alternatives is not written either -- 'MOVED or retired' is "
            "an unresolved question wearing a disposition's clothes"
        )

    def test_every_disposition_is_from_the_closed_vocabulary(
        self, rows: list[dict[str, str]]
    ) -> None:
        for row in rows:
            assert row["Disposition"] in VOCABULARY, (
                f"{row['Disposition']!r} is not one of {VOCABULARY}; the Disposition cell "
                "holds the bare token and nothing else"
            )

    def test_no_row_bundles_two_sites(self, rows: list[dict[str, str]]) -> None:
        bundled = [row["Site"][:40] for row in rows if ";" in row["Site"]]
        assert bundled == [], (
            "a bundled row contributes ONE entry to the listed-sites set while covering "
            "two sites in the tree, and the two-disposition check cannot catch it because "
            "a bundle's members usually agree"
        )

    def test_the_prose_only_control_is_non_empty(
        self, rows: list[dict[str, str]]
    ) -> None:
        """The no-false-positive control that pins the match mode to the CELL.

        At least one row's Reasoning must name a vocabulary token that is not its own
        disposition, with the suite still passing. The ``config/gate.toml`` row supplies it
        by construction: its argument says GRADING_ONLY would be false and DELETED would be
        false too, while its verdict is OPEN_ACCEPTED.
        """
        prose_only = [
            row["Site"][:40]
            for row in rows
            if any(
                token in row["Reasoning"] and token not in _disposition_tokens(row)
                for token in VOCABULARY
            )
        ]
        assert prose_only, (
            "PROSE_ONLY_VOCAB is EMPTY. Either the gate.toml row stopped naming the "
            "dispositions it rejects, or someone re-widened a scan to the row line. The "
            "empty list is the failure here, not the pass."
        )

    def test_no_reasoning_cell_carries_an_instruction_to_its_own_author(
        self, rows: list[dict[str, str]]
    ) -> None:
        leftovers = [
            row["Site"][:40]
            for row in rows
            if "name the concrete site" in row["Reasoning"]
            or "nothing reads them" in row["Reasoning"]
        ]
        assert leftovers == [], (
            "an unresolved authoring instruction, or the retired claim that nothing reads "
            "the gate baselines, was about to be committed to the repository root"
        )


# ---------------------------------------------------------------------------
# Individual verdicts
# ---------------------------------------------------------------------------


class TestTheVerdicts:
    def test_the_ou_high_total_boundary_is_deleted(
        self, rows: list[dict[str, str]]
    ) -> None:
        """D33.2-24: deleted, not moved and not an exception."""
        matches = [row for row in rows if "high-total" in row["Site"].lower()]
        assert len(matches) == 1
        assert matches[0]["Disposition"] == "DELETED"

    def test_the_feature_group_screen_is_moved_and_discharged_by_33_2_22(
        self, rows: list[dict[str, str]]
    ) -> None:
        row = _row_named(rows, "_walkforward_clv_series")
        assert row["Disposition"] == "MOVED_TO_PRELOCK"
        assert row["Discharged by"] == "33.2-22"

    def test_the_ev_floor_and_the_residual_sd_name_plan_33_2_29(
        self, rows: list[dict[str, str]]
    ) -> None:
        """D33.2-25 re-derives both. D33.2-24, which an earlier draft cited, re-derives
        neither -- it is the O/U eligibility-gate deletion."""
        ev_floor = _row_named(rows, "EV floor")
        residual_sd = _row_named(rows, "fit_frozen_residual_sd")
        assert ev_floor["Discharged by"] == "33.2-29"
        assert residual_sd["Discharged by"] == "33.2-29"
        assert "EV_FLOOR_GRID" in ev_floor["Site"]
        assert "ou_ev_chain" in residual_sd["Site"]

    def test_the_noise_profile_and_the_mode_gate_are_separate_deleted_rows(
        self, rows: list[dict[str, str]]
    ) -> None:
        noise = _row_named(rows, "extract_noise_profile")
        comparison = _row_named(rows, "run_comparison")
        per_target = _row_named(rows, "_gate_per_target")
        assert noise["Disposition"] == "DELETED"
        assert comparison["Disposition"] == "DELETED"
        assert per_target["Disposition"] == "DELETED"

    def test_the_residual_converters_and_chain_bias_are_not_closing_line_fits(
        self, rows: list[dict[str, str]]
    ) -> None:
        converters = _row_named(rows, "residual converters")
        chain_bias = _row_named(rows, "Chain-fit bias")
        assert converters["Disposition"] == "NOT_A_CLOSING_LINE_FIT"
        assert chain_bias["Disposition"] == "NOT_A_CLOSING_LINE_FIT"

    def test_every_open_accepted_row_names_its_live_reader(
        self, rows: list[dict[str, str]]
    ) -> None:
        unsupported = [
            row["Site"][:40]
            for row in rows
            if row["Disposition"] == "OPEN_ACCEPTED" and ".py:" not in row["Reasoning"]
        ]
        assert unsupported == [], (
            "a disposition meaning 'still read' that does not say BY WHAT is the same "
            "shape of unsupported claim as the assertion it replaced"
        )


class TestTheGateBaselineRow:
    def test_it_is_open_accepted_and_not_deleted(
        self, rows: list[dict[str, str]]
    ) -> None:
        row = _row_named(rows, "gate.toml")
        assert row["Disposition"] == "OPEN_ACCEPTED"

    @pytest.mark.parametrize("reader", REQUIRED_GATE_BASELINE_READERS)
    def test_it_names_every_measured_reader(
        self, rows: list[dict[str, str]], reader: str
    ) -> None:
        """D33.2-26 as corrected: EVERY live reader, not some.

        A row naming one of three satisfies the weak "does it cite any site" check while
        still under-stating what reads a retained dead comparator.
        """
        assert reader in _row_named(rows, "gate.toml")["Reasoning"]

    def test_the_retired_enforcement_claim_appears_nowhere(
        self, audit_text: str
    ) -> None:
        assert "nothing reads them" not in audit_text
        assert "enforcement is that nothing reads" not in audit_text


# ---------------------------------------------------------------------------
# Anti-rot
# ---------------------------------------------------------------------------


def _modules_defining_or_using(symbol: str) -> list[str]:
    """Modules that DEFINE, import or call *symbol* -- parsed, never grepped.

    Node-shaped rather than textual on purpose, the same discipline the disposition guard
    applies by reading a cell rather than a line: a module that names a symbol only in a
    docstring is not a reader of it.
    """
    found: list[str] = []
    for root in _SCAN_ROOTS:
        for path in Path(root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            }
            names |= {
                alias.asname or alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import | ast.ImportFrom)
                for alias in node.names
            }
            names |= {
                node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name | ast.Attribute)
            }
            names |= {
                target.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign | ast.AnnAssign)
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if isinstance(target, ast.Name)
            }
            if symbol in names:
                found.append(path.as_posix())
    return sorted(found)


def _symbols_absent_from_the_table(
    rows: list[dict[str, str]], symbols: tuple[str, ...]
) -> list[str]:
    """Symbols that EXIST in the tree but appear in no cell of the audit table."""
    table_text = " ".join(
        cell
        for row in rows
        for cell in row.values()  # every cell, never the preamble
    )
    return sorted(
        symbol
        for symbol in symbols
        if _modules_defining_or_using(symbol) and symbol not in table_text
    )


class TestAntiRot:
    def test_no_closing_line_fit_site_is_missing_from_the_document(
        self, rows: list[dict[str, str]]
    ) -> None:
        assert _symbols_absent_from_the_table(rows, CLOSING_LINE_FIT_SYMBOLS) == [], (
            "a closing-line fit site exists in the tree and is not listed in the audit, "
            "so the document has gone stale"
        )

    def test_the_anti_rot_guard_fails_on_a_planted_violation(
        self, rows: list[dict[str, str]]
    ) -> None:
        """The control. A guard that cannot fail has not passed."""
        planted = (CONTROL_SYMBOL_NOT_IN_THE_AUDIT,)
        assert _modules_defining_or_using(CONTROL_SYMBOL_NOT_IN_THE_AUDIT), (
            "the control symbol is not in the tree, so this control proves nothing"
        )
        assert _symbols_absent_from_the_table(rows, planted) == [
            CONTROL_SYMBOL_NOT_IN_THE_AUDIT
        ]

    def test_the_gate_baseline_readers_are_derived_from_the_tree_not_declared(
        self, rows: list[dict[str, str]]
    ) -> None:
        """A reader added later is caught here without anyone remembering a list.

        The derivation returns production modules AND the tests that read the sections;
        every one of their file names must appear in the row's Reasoning.
        """
        live = _modules_defining_or_using(BASELINE_BUNDLE_SYMBOL)
        production = [path for path in live if path.startswith("scripts/")]
        assert "scripts/promote_models.py" in production
        assert "scripts/retrain_models.py" in production, (
            "the re-fit path reader is the one this phase actually exercises"
        )

        reasoning = _row_named(rows, "gate.toml")["Reasoning"]
        unnamed = sorted(
            {path.rsplit("/", 1)[-1] for path in production}
            - {
                name
                for name in {path.rsplit("/", 1)[-1] for path in production}
                if name in reasoning
            }
        )
        assert unnamed == [], (
            f"live production readers {unnamed} are not named in the gate-baseline row; "
            "under-naming the readers of a retained dead comparator is a softer version "
            "of the false claim D33.2-26 exists to forbid"
        )
