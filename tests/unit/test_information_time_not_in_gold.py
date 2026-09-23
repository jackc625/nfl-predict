"""No time reaches gold: the sidecar is never a column (SPEC R2, Plan 33.2-20 Task 2).

WHY THIS IS NOT DECORATION
--------------------------
``scripts.build_features.combine_features`` merges every column a source emits, bar a
short per-source drop list. A source that put its information time on its FEATURE frame
rather than on its provenance sidecar would therefore land a datetime column in gold, and
a model keyed on it would be reading the calendar rather than the game. Plan 33.2-01 built
``features.provenance.refuse_provenance_columns`` to make that a refusal at build time;
this module is the standing check on what is ACTUALLY PERSISTED, which is the artefact the
re-fit reads.

THE ONE ADMITTED DATETIME, AND WHY IT IS ADMITTED -- A DECISION, NOT AN OVERSIGHT
---------------------------------------------------------------------------------
``feature_timestamp`` is a datetime column in all three gold matrices. SPEC R2's acceptance
line reads "no gold matrix contains a datetime-typed column", and it is recorded as OPEN in
the phase's ``deferred-items.md`` with Plan 33.2-20 Task 2 as its receiver. Task 2 is this
module, so the decision is made HERE and recorded rather than left to drift:

**KEEP IT, admitted by the ONE registry that names it, and retire it in its own ladder
step.** Three reasons, in the order they bind:

1. **It cannot leak.** ``feature_timestamp`` records WHEN A BUILD RAN. It is one identical
   instant for every row of every matrix, measures nothing about any game, and no model
   reads it -- ``models/temporal.py`` excludes it and ``scripts/build_features.py`` lists
   it in ``exclude_cols``. R2 is about information a model could not have had at a game's
   lock; a constant stamped after every row is not that.
2. **Retiring it MOVES A GOLD WIDTH.** It is a real column in all three matrices, so
   removing it takes the widths from 188/188/187 to 187/187/186. Plan 33.2-20 ASSERTS the
   gold-width pin rung 9 wrote and re-pins nothing (Plan 33.2-12's
   ``<owned_protocol_gold_width_pin>`` P6): its clean build must REPRODUCE rung 9, and a
   width this plan moved would make that assertion meaningless. A width change is a
   declared ladder step with its own cause, rebuild and attribution.
3. **It is woven through the evidence chain.** ``scripts/fingerprint_gold.py`` excludes it
   from every rung comparison by name (``BUILD_CLOCK_COLUMNS``), the idempotency and
   n01-resync controls read it, and the phase state witnesses record it. Removing it is a
   change to how the ladder MEASURES itself, which must not ride along inside a plan whose
   job is to assert that the ladder reproduced.

So the guard admits EXACTLY the registered clock and refuses every other datetime column --
and this module pins both halves: that the registry names exactly one column, and that no
matrix carries a second datetime. The deferred item stays OPEN with its receiver moved to
the re-fit's own gold step; what changes here is that the decision is now recorded with its
reasoning instead of being an unexplained exception.

THE FOUR STRUCTURAL CONTROLS
----------------------------
1. non-vacuity -- the scan reads real matrices and a large, asserted-non-zero column count;
2. the assertion -- no datetime-typed and no provenance-named column in any matrix;
3. a planted violation -- a synthetic frame carrying a datetime column IS flagged;
4. no false positive -- a frame of numeric features is NOT flagged.

Run this module:  uv run pytest tests/unit/test_information_time_not_in_gold.py -q

Sandboxed: every check is a READ. Nothing here writes to any store.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from features.provenance import (
    PROVENANCE_NAME_MARKERS,
    InformationTimeViolation,
    refuse_provenance_columns,
)
from scripts.fingerprint_gold import BUILD_CLOCK_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_ROOT = REPO_ROOT / "data" / "gold"

#: The three canonical matrices the re-fit trains on and the Friday pipeline checks.
GOLD_MATRICES: tuple[str, ...] = ("features_wp", "features_ats", "features_ou")

#: The floor the non-vacuity control asserts. Production gold is 188/188/187 columns over
#: 6,499 rows after p332_ rung 9; a scan that read a handful of columns would "pass"
#: without having looked at gold at all.
MIN_COLUMNS_PER_MATRIX = 100
MIN_ROWS_PER_MATRIX = 6000


def _load(matrix: str) -> pd.DataFrame:
    path = GOLD_ROOT / f"{matrix}.parquet"
    if not path.exists():
        pytest.skip(f"gold {matrix} is not built on this checkout")
    return pd.read_parquet(path)


def _datetime_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[column])
    ]


def _provenance_named_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if str(column).lower() == "basis"
        or any(marker in str(column).lower() for marker in PROVENANCE_NAME_MARKERS)
    ]


class TestTheScanIsNotVacuous:
    """Control 1: it reads the REAL matrices, and enough of them to mean something."""

    @pytest.mark.parametrize("matrix", GOLD_MATRICES)
    def test_each_matrix_is_wide_and_long(self, matrix: str) -> None:
        frame = _load(matrix)
        assert len(frame.columns) >= MIN_COLUMNS_PER_MATRIX, (
            f"{matrix} has {len(frame.columns)} columns. A scan over a handful of columns "
            "reports PASS without having looked at gold."
        )
        assert len(frame) >= MIN_ROWS_PER_MATRIX, (
            f"{matrix} has {len(frame)} rows, which is not the production history."
        )

    def test_all_three_matrices_are_present(self) -> None:
        missing = [
            m for m in GOLD_MATRICES if not (GOLD_ROOT / f"{m}.parquet").exists()
        ]
        if missing:
            pytest.skip(f"gold is not built on this checkout: {missing}")
        assert len(GOLD_MATRICES) == 3


class TestNoTimeColumnReachesGold:
    """Control 2: the assertion itself, on what is PERSISTED."""

    @pytest.mark.parametrize("matrix", GOLD_MATRICES)
    def test_the_only_datetime_column_is_the_registered_build_clock(
        self, matrix: str
    ) -> None:
        frame = _load(matrix)
        found = _datetime_columns(frame)
        unregistered = [c for c in found if c not in BUILD_CLOCK_COLUMNS]
        assert unregistered == [], (
            f"{matrix} carries datetime column(s) {unregistered}, which are not the "
            f"registered build clock {list(BUILD_CLOCK_COLUMNS)}. The sidecar is never a "
            "column: an information time on a FEATURE frame reaches gold through "
            "combine_features, and a schedule instant in a matrix is a key a model can "
            "read the calendar from."
        )

    @pytest.mark.parametrize("matrix", GOLD_MATRICES)
    def test_no_column_carries_a_provenance_name(self, matrix: str) -> None:
        frame = _load(matrix)
        found = _provenance_named_columns(frame)
        assert found == [], (
            f"{matrix} carries provenance-named column(s) {found}. Provenance travels "
            "BESIDE a source's feature frame, never on it."
        )

    @pytest.mark.parametrize("matrix", GOLD_MATRICES)
    def test_the_production_guard_itself_accepts_the_persisted_matrix(
        self, matrix: str
    ) -> None:
        """The same function the build calls, run on what the build wrote."""
        refuse_provenance_columns(
            _load(matrix), build_clock_columns=BUILD_CLOCK_COLUMNS
        )


class TestTheOneAdmittedDatetimeIsDecidedAndPinned:
    """The ``feature_timestamp`` decision, recorded as an assertion rather than as prose."""

    def test_the_registry_admits_exactly_one_column(self) -> None:
        assert BUILD_CLOCK_COLUMNS == ("feature_timestamp",), (
            f"the build-clock registry now admits {list(BUILD_CLOCK_COLUMNS)}. It is the "
            "ONE exception to 'no datetime reaches gold' and it is one column wide; a "
            "second entry is a second exception and needs its own recorded reasoning."
        )

    @pytest.mark.parametrize("matrix", GOLD_MATRICES)
    def test_the_build_clock_is_present_and_constant_within_the_matrix(
        self, matrix: str
    ) -> None:
        """It records WHEN A BUILD RAN, so it must be one instant for the whole matrix.

        This is the property that makes it unable to leak: a column that varies per game
        would be describing the games, and would have to go.
        """
        frame = _load(matrix)
        clock = BUILD_CLOCK_COLUMNS[0]
        assert clock in frame.columns
        assert frame[clock].nunique(dropna=False) == 1, (
            f"{matrix}.{clock} takes {frame[clock].nunique(dropna=False)} distinct "
            "values. A build clock is ONE instant for the whole build; a per-row value "
            "is a fact about the games and must not be in a feature matrix."
        )

    def test_no_model_reads_the_build_clock(self) -> None:
        """The second half of "it cannot leak": nothing selects it as a feature."""
        from models.temporal import _DEFAULT_ID_COLS

        assert BUILD_CLOCK_COLUMNS[0] in set(_DEFAULT_ID_COLS), (
            "the build clock is not in models.temporal._DEFAULT_ID_COLS, so it is no "
            "longer excluded from every model's feature set."
        )

    def test_retiring_it_would_move_the_gold_width(self) -> None:
        """Why it is not retired HERE: this plan asserts the width pin and re-pins nothing.

        Recorded as an assertion so the reasoning cannot rot into a comment nobody checks.
        """
        from scripts.data_qa import GOLD_FEATURE_MATRICES

        clock = BUILD_CLOCK_COLUMNS[0]
        for matrix, pinned in GOLD_FEATURE_MATRICES.items():
            frame = _load(matrix)
            assert clock in frame.columns
            assert len(frame.columns) == pinned
            assert len(frame.columns) - 1 != pinned, "the arithmetic below is wrong"


class TestThePlantedViolationIsFlagged:
    """Control 3: the scan fails on the exact mistake it exists for."""

    def test_a_datetime_column_is_flagged(self) -> None:
        planted = pd.DataFrame(
            {
                "game_id": ["2023_W01_A@B"],
                "elo_diff": [12.5],
                "information_time": [pd.Timestamp("2023-09-09 18:00", tz="UTC")],
            }
        )
        assert _datetime_columns(planted) == ["information_time"]
        with pytest.raises(InformationTimeViolation, match="information_time"):
            refuse_provenance_columns(planted, build_clock_columns=BUILD_CLOCK_COLUMNS)

    def test_a_kickoff_column_is_flagged_even_though_its_name_is_innocent(self) -> None:
        """A schedule instant is refused on its DTYPE, not on its name."""
        planted = pd.DataFrame(
            {
                "game_id": ["2023_W01_A@B"],
                "kickoff_et": [pd.Timestamp("2023-09-10 13:00", tz="UTC")],
            }
        )
        with pytest.raises(InformationTimeViolation, match="kickoff_et"):
            refuse_provenance_columns(planted, build_clock_columns=BUILD_CLOCK_COLUMNS)

    def test_a_provenance_named_numeric_column_is_flagged(self) -> None:
        """And a provenance NAME is refused on the name, whatever its dtype."""
        planted = pd.DataFrame({"game_id": ["G"], "elo_provenance": [1.0]})
        assert _provenance_named_columns(planted) == ["elo_provenance"]
        with pytest.raises(InformationTimeViolation, match="elo_provenance"):
            refuse_provenance_columns(planted, build_clock_columns=BUILD_CLOCK_COLUMNS)


class TestTheNoFalsePositiveControl:
    """Control 4: a frame of ordinary numeric features is NOT flagged."""

    def test_a_numeric_feature_frame_passes(self) -> None:
        clean = pd.DataFrame(
            {
                "game_id": ["2023_W01_A@B", "2023_W01_C@D"],
                "elo_diff": [12.5, -3.0],
                "home_off_rolling_epa_per_play": [0.05, -0.01],
                "wind_mph": [7.0, 12.0],
                "home_injury_coverage": [1.0, 0.0],
            }
        )
        assert _datetime_columns(clean) == []
        assert _provenance_named_columns(clean) == []
        refuse_provenance_columns(clean, build_clock_columns=BUILD_CLOCK_COLUMNS)

    def test_the_registered_build_clock_alone_passes(self) -> None:
        """The admitted exception is admitted, and only under its registered name."""
        frame = pd.DataFrame(
            {
                "game_id": ["G"],
                BUILD_CLOCK_COLUMNS[0]: [pd.Timestamp("2026-09-22", tz="UTC")],
            }
        )
        refuse_provenance_columns(frame, build_clock_columns=BUILD_CLOCK_COLUMNS)
        with pytest.raises(InformationTimeViolation):
            refuse_provenance_columns(frame, build_clock_columns=())


# ---------------------------------------------------------------------------
# THE VALIDATOR REPORTS THE REAL VERDICT (Plan 33.2-20 Task 2)
# ---------------------------------------------------------------------------


class TestTheValidatorHasNoUnconditionalPass:
    """``scripts/validate_features.py`` printed ``[PASS]`` whatever the outcome.

    WHAT WAS THERE. A ``# -- check_time_fence note --`` block that announced
    ``[TIME FENCE] check_time_fence on persisted gold`` and then printed ``[PASS]``
    unconditionally, with a sentence citing ``build_features.py:809``.
    ``LeakageGate.check_time_fence`` was DELETED by Plan 33.2-01, so the line described a
    check that no longer existed and reported a verdict it had not reached. A line that
    says PASS regardless of the outcome is worse than no line, because it is read as
    evidence -- which is the whole class of defect this phase exists to remove.

    Plan 33.2-01 deliberately did NOT edit that block, naming Plan 33.2-20 Task 2 as the
    receiver because this task already owns the line range. This is that replacement.
    """

    @staticmethod
    def _source() -> str:
        return (REPO_ROOT / "scripts" / "validate_features.py").read_text(
            encoding="utf-8"
        )

    def test_the_deleted_check_is_never_REPORTED_only_remembered(self) -> None:
        """The name may appear in a COMMENT; it may not appear in anything printed.

        The subject is the REPORT, not the file. A historical note saying what stood here
        and why it was wrong is exactly what this project asks a removal to leave behind --
        forbidding the string outright would forbid the explanation. What must not survive
        is a verdict: ``check_time_fence`` in a string constant or an f-string is text that
        reaches the diagnostic file or stdout, and that is the false report.

        Comments are absent from the AST, so the distinction is a node-shape one and needs
        no judgement call.
        """
        import ast

        tree = ast.parse(self._source())
        reported: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if (
                    "check_time_fence" in node.value
                    or "build_features.py:809" in node.value
                ):
                    reported.append(node.lineno)
            if isinstance(node, ast.JoinedStr):
                for part in node.values:
                    if (
                        isinstance(part, ast.Constant)
                        and isinstance(part.value, str)
                        and (
                            "check_time_fence" in part.value
                            or "build_features.py:809" in part.value
                        )
                    ):
                        reported.append(node.lineno)
        assert reported == [], (
            f"scripts/validate_features.py REPORTS check_time_fence at line(s) "
            f"{sorted(set(reported))}. Plan 33.2-01 DELETED that method; narrating a "
            "check that does not exist is a false report. Say it in a comment if the "
            "history is worth keeping -- comments are not printed."
        )

    def test_it_names_the_information_time_gate_instead(self) -> None:
        source = self._source()
        assert "INFORMATION TIME" in source
        assert "refuse_provenance_columns" in source, (
            "the block must name the check it actually runs, so a reader can find it."
        )

    def test_the_verdict_is_computed_not_written(self) -> None:
        """No ``[PASS]`` may be emitted from a line that cannot also emit ``[FAIL]``.

        Structural, over the parsed tree: every literal containing ``[PASS]`` in this
        module must sit inside a conditional or a formatted expression. A bare
        ``lines.append("  [PASS] ...")`` at statement level, with no branch above it, is
        the shape that was there.
        """
        import ast

        tree = ast.parse(self._source())
        offenders: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for argument in node.args:
                if (
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, str)
                    and "[PASS]" in argument.value
                    and "[FAIL]" not in argument.value
                ):
                    offenders.append(node.lineno)
        unconditional = [
            line for line in offenders if not _is_inside_a_branch(tree, line)
        ]
        assert unconditional == [], (
            f"scripts/validate_features.py emits an UNCONDITIONAL [PASS] at line(s) "
            f"{unconditional}. A verdict that cannot be FAIL is not a verdict."
        )

    def test_a_violation_is_recorded_as_unexplained_and_exits_non_zero(self) -> None:
        """The verdict has to reach the exit code, not only the report file."""
        source = self._source()
        block = source[source.index("INFORMATION TIME") :]
        assert "unexplained_leakage.append" in block, (
            "the information-time block records no finding, so a violation could not "
            "change the run's verdict and `run_gold_audit` would still return True."
        )

    def test_it_states_that_run_coverage_is_a_build_fact(self) -> None:
        """The counts the gate reports per RUN cannot be read off persisted gold.

        Printing a checked/empty-unchecked count here would be inventing one. The block
        says so and names the command that does print them, which is the honest report.
        """
        source = self._source()
        assert "scripts.build_features" in source
        assert "CHECKED_SOURCES=" in source


def _is_inside_a_branch(tree, lineno: int) -> bool:
    """True when *lineno* sits inside an ``if`` / ``try`` / loop body somewhere in *tree*."""
    import ast

    for node in ast.walk(tree):
        if not isinstance(node, ast.If | ast.Try | ast.For | ast.While):
            continue
        end = getattr(node, "end_lineno", None)
        if end is not None and node.lineno <= lineno <= end:
            return True
    return False
