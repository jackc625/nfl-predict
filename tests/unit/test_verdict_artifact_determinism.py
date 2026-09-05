"""Two renders of one verdict are byte-identical, and no float escaped the specifier.

WHY BYTE-IDENTITY IS THE BAR AND NOT "AGREES TO TWELVE PLACES". The R11 readout drift guard
asserts that every 2025 figure it prints matches the committed verdict artifact
BYTE-FOR-BYTE -- made exact by the ``.17g`` determinism rule rather than by a float tolerance,
because the 2025 chain CANNOT be re-run and a consistency check is all that is available. Under
that arrangement a formatting difference reads as tampering with a measurement artifact rather
than as the accident it would be, so the rendering has to be contractual.

``.17g`` is 17 SIGNIFICANT digits, the precision at which no two distinct IEEE-754 doubles can
render identically. What must NOT happen is a bare interpolation falling through to Python's
default float formatting, which is shortest-round-trip and is not a stable contract across
platforms or patch releases. Only a SCAN catches a field someone forgot to route through the
specifier -- a byte-identity check alone would happily reproduce the same wrong rendering
twice.

WHAT THE SCAN CANNOT SEE, stated rather than left to be discovered: a value whose
shortest-round-trip text is ALREADY its 17-significant-digit text (``2.0``, ``0.5``) renders
identically either way, so a bare interpolation of one would pass. That is not a gap in the
guard so much as a value with no drift to detect; every measured return, p-value and interval
bound in the artifact has a shortest form that differs.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re

import pytest

from backtest.profitability_2025 import (
    FLOAT_FORMAT,
    render_float,
    render_verdict_fields,
    render_verdict_toml,
    verdict_run_record,
)

_ASSIGNMENT = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.+)$")
_NUMERIC = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$")


def _is_toml_float_literal(text: str) -> bool:
    """True when TOML would parse ``text`` as a FLOAT rather than an integer.

    An integer literal is skipped deliberately: bet counts, block counts and the BH
    denominator are counts, and rendering a count through a float specifier would be the
    opposite error.
    """
    if text in ("nan", "inf", "-inf", "+inf"):
        return True
    if not _NUMERIC.match(text):
        return False
    return "." in text or "e" in text or "E" in text


def scan_for_unspecified_floats(text: str) -> tuple[list[str], int]:
    """Report every float literal in ``text`` that did not come from the specifier.

    Returns:
        ``(violations, examined_count)``. The count is returned so the caller can fail when
        it is zero: a scan that examined no numeric field is a green test that asserts
        nothing about the rendering.
    """
    violations: list[str] = []
    examined = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("["):
            continue
        match = _ASSIGNMENT.match(stripped)
        if match is None:
            continue
        value = match.group("value").strip()
        if not _is_toml_float_literal(value):
            continue
        examined += 1
        expected = render_float(float(value))
        if value != expected:
            violations.append(
                f"line {lineno}: {match.group('key')} = {value} -- the "
                f"{FLOAT_FORMAT} specifier renders that value as {expected}, so this field "
                "was interpolated bare and falls through to Python's default float "
                "formatting."
            )
    return violations, examined


class TestTheArtifactRendersDeterministically:
    """Byte-identity across two renders of the SAME result, with no normalisation."""

    def test_two_renders_of_the_same_result_are_byte_identical(
        self, p31_rehearsal_run
    ) -> None:
        result = p31_rehearsal_run["result"]
        first = render_verdict_toml(result)
        second = render_verdict_toml(result)
        assert first == second, (
            "two renders of one result differ. Compared as raw strings with NO normalisation "
            "step on purpose: a normalisation would be the thing hiding the difference."
        )
        assert first.encode("utf-8") == second.encode("utf-8")

    def test_the_written_file_matches_a_fresh_render(self, p31_rehearsal_run) -> None:
        """The bytes on disk are the generator's output, not a variant of it."""
        written = p31_rehearsal_run["verdict_toml_path"].read_text(encoding="utf-8")
        assert written == render_verdict_toml(p31_rehearsal_run["result"])

    def test_the_rendered_fields_are_the_same_object_shape_twice(
        self, p31_rehearsal_run
    ) -> None:
        result = p31_rehearsal_run["result"]
        assert render_verdict_fields(result) == render_verdict_fields(result)

    def test_the_run_record_carries_the_renderers_output_verbatim(
        self, p31_rehearsal_run
    ) -> None:
        """ONE renderer: the machine-readable form embeds exactly what the TOML was built from."""
        result = p31_rehearsal_run["result"]
        assert verdict_run_record(result)["verdict_fields"] == render_verdict_fields(
            result
        )


class TestNoFloatEscapedTheSpecifier:
    """The scan, its non-vacuity, and the proof that it can fire."""

    def test_the_committed_form_has_no_bare_interpolation(
        self, p31_rehearsal_run
    ) -> None:
        text = p31_rehearsal_run["verdict_toml_path"].read_text(encoding="utf-8")
        violations, examined = scan_for_unspecified_floats(text)
        assert examined > 0, (
            "the bare-interpolation scan examined ZERO numeric fields, so its clean result "
            "proves nothing about the artifact's rendering."
        )
        assert not violations, (
            f"fields escaped the {FLOAT_FORMAT} specifier ({examined} examined):\n"
            + "\n".join(f"  - {line}" for line in violations)
        )

    def test_the_scan_examines_a_realistic_number_of_fields(
        self, p31_rehearsal_run
    ) -> None:
        """Anti-vacuity with a floor, so a renderer that emitted two floats cannot pass."""
        text = p31_rehearsal_run["verdict_toml_path"].read_text(encoding="utf-8")
        _violations, examined = scan_for_unspecified_floats(text)
        assert examined >= 20, examined

    def test_the_scan_names_a_planted_bare_interpolation(self) -> None:
        """Fail-closed control: a scan only ever seen passing may be unable to fail.

        ``0.1`` is what a bare ``f"{value}"`` produces; the specifier renders the SAME double
        as ``0.10000000000000001``, so the two are distinguishable and the scan says which
        field is which.
        """
        planted = "[targets.wp]\nflat_roi = 0.1\nbets_selected = 42\n"
        violations, examined = scan_for_unspecified_floats(planted)
        assert examined == 1, (
            "the integer field was counted as a float, or the float field was missed."
        )
        assert len(violations) == 1
        assert "flat_roi" in violations[0]
        assert "0.10000000000000001" in violations[0]

    def test_the_scan_accepts_a_correctly_specified_value(self) -> None:
        """Fail-open control: a guard that reddens on correct output gets weakened."""
        correct = f"[targets.wp]\nflat_roi = {render_float(0.1)}\nn_blocks = 18\n"
        violations, examined = scan_for_unspecified_floats(correct)
        assert (violations, examined) == ([], 1)

    @pytest.mark.parametrize("value", [None, 0.0, -0.0, 1.0, 2.5, 1e-30, -1e18])
    def test_every_rendered_value_survives_the_scan(self, value: float | None) -> None:
        text = f"x = {render_float(value)}\n"
        assert scan_for_unspecified_floats(text) == ([], 1)

    def test_nan_renders_as_a_toml_float_and_is_examined(self) -> None:
        """An ABSENT return renders as ``nan`` -- a float, never a silently omitted key."""
        violations, examined = scan_for_unspecified_floats("flat_roi = nan\n")
        assert (violations, examined) == ([], 1)

    def test_a_whole_number_keeps_its_float_suffix(self) -> None:
        """``.17g`` gives ``2`` for 2.0, which TOML would parse as an INTEGER."""
        assert render_float(2.0) == "2.0"
        assert render_float(-3.0) == "-3.0"
        assert _is_toml_float_literal(render_float(2.0))
