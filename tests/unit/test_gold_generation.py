"""Controls for the gold GENERATION seam (Plan 33.1-08 Task 2).

WHY THIS MODULE EXISTS
----------------------
``tests/gold_generation.require_gold_generation`` makes other tests SKIP. A skip
is silent by design, and a seam that skips for a reason nobody checked is
indistinguishable from a seam that skips unconditionally -- which would turn
every guard hanging off it into a no-op while the terminal still prints green.
That is the exact failure mode ``tests/unit/test_weather_archive_quarantined.py``
keeps a non-vacuity control for.

So this module proves the gate in BOTH directions:

  * a MATCHING key does NOT skip -- without this, the seam could be skipping
    everything and nobody would know,
  * a DIFFERING key DOES skip, and the message names the reading, so the skip a
    human reads in ``-rs`` output is actionable rather than a bare "skipped".

It also proves the two refusals, because a key that silently degrades is worse
than no key: a missing matrix raises rather than producing a stable key over the
two that remain, and a stat-signature digest is refused by name (D33-32).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import pytest

from tests.gold_generation import (
    GOLD_MATRIX_PATHS,
    GoldGenerationUnprovableError,
    GoldMatrixMissingError,
    gold_generation_key,
    require_gold_generation,
)

# A key that is well-formed but cannot be today's gold. 64 hex characters so the
# test is exercising the COMPARISON rather than accidentally exercising a length
# or format check.
_A_DIFFERENT_GENERATION = "0" * 64

# The four arguments every real call site passes, kept in one place so the tests
# below assert against the same strings a caller would see.
_READING = "OU-DIVERGENCE-DIAGNOSIS.md's pooled over-share anchor of 0.727"
_MOVED_BY = "Phase 33.1's weather rung"
_RECORDED_IN = "the Phase 33.1 weather readout"


def _gold_is_present() -> bool:
    """True when all three gold matrices can be digested in this checkout."""
    try:
        gold_generation_key()
    except (GoldMatrixMissingError, GoldGenerationUnprovableError):
        return False
    return True


class TestTheKeyIsWellFormed:
    """gold_generation_key returns a real sha256 over real bytes."""

    def test_the_key_is_64_hex_characters(self) -> None:
        """A sha256 hexdigest, so a caller can pin it as a constant."""
        if not _gold_is_present():
            pytest.skip("gold is not present in this checkout")
        key = gold_generation_key()
        assert len(key) == 64, (
            f"expected a 64-character sha256 hexdigest, got {len(key)}"
        )
        assert all(c in "0123456789abcdef" for c in key), f"not hex: {key!r}"

    def test_the_key_is_stable_across_calls(self) -> None:
        """Two calls with nothing in between agree; the key is of the DATA, not the clock."""
        if not _gold_is_present():
            pytest.skip("gold is not present in this checkout")
        assert gold_generation_key() == gold_generation_key()

    def test_all_three_matrices_are_named(self) -> None:
        """The generation is a property of all three matrices together, not of one."""
        assert GOLD_MATRIX_PATHS == (
            "data/gold/features_wp.parquet",
            "data/gold/features_ats.parquet",
            "data/gold/features_ou.parquet",
        )


class TestTheGateIsReal:
    """The load-bearing pair: a matching key does NOT skip, a differing key DOES."""

    def test_a_matching_key_does_not_skip(self) -> None:
        """THE NON-VACUITY CONTROL.

        Without this, every generation-gated class in the repository could be
        skipping unconditionally and the suite would look identical. This is the
        assertion that makes a skip elsewhere mean something.
        """
        if not _gold_is_present():
            pytest.skip("gold is not present in this checkout")
        live = gold_generation_key()
        # If this skips, the exception propagates and the test does not pass --
        # which is the whole point. No try/except: a swallowed skip would be the
        # vacuity this control exists to detect.
        require_gold_generation(
            live,
            reading=_READING,
            moved_by=_MOVED_BY,
            recorded_in=_RECORDED_IN,
        )

    def test_a_differing_key_skips(self) -> None:
        """A reading measured on other gold is not compared against this gold."""
        if not _gold_is_present():
            pytest.skip("gold is not present in this checkout")
        with pytest.raises(pytest.skip.Exception):
            require_gold_generation(
                _A_DIFFERENT_GENERATION,
                reading=_READING,
                moved_by=_MOVED_BY,
                recorded_in=_RECORDED_IN,
            )

    def test_the_skip_message_names_the_reading_the_mover_and_the_record(self) -> None:
        """A skip a human cannot act on is a skip nobody reads.

        The message must carry all three arguments plus BOTH keys, so a reader of
        ``pytest -rs`` output can tell WHICH reading was set aside, WHAT moved
        gold out from under it, and WHERE that is written down.
        """
        if not _gold_is_present():
            pytest.skip("gold is not present in this checkout")
        live = gold_generation_key()
        with pytest.raises(pytest.skip.Exception) as excinfo:
            require_gold_generation(
                _A_DIFFERENT_GENERATION,
                reading=_READING,
                moved_by=_MOVED_BY,
                recorded_in=_RECORDED_IN,
            )
        message = str(excinfo.value)
        for fragment in (
            _READING,
            _MOVED_BY,
            _RECORDED_IN,
            live,
            _A_DIFFERENT_GENERATION,
        ):
            assert fragment in message, (
                f"the skip message does not carry {fragment!r}; it reads: {message!r}"
            )


class TestTheRefusals:
    """A key that degrades silently is worse than no key at all."""

    def test_a_missing_matrix_raises_rather_than_keying_over_the_rest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two matrices cannot answer a question asked about three."""
        monkeypatch.setattr(
            "tests.gold_generation.GOLD_MATRIX_PATHS",
            (*GOLD_MATRIX_PATHS, "data/gold/features_no_such_target.parquet"),
        )
        with pytest.raises(GoldMatrixMissingError) as excinfo:
            gold_generation_key()
        assert "features_no_such_target.parquet" in str(excinfo.value)

    def test_a_stat_signature_digest_is_refused_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D33-32: size and mtime cannot show that content is unchanged."""
        if not _gold_is_present():
            pytest.skip("gold is not present in this checkout")
        monkeypatch.setattr(
            "tests.gold_generation.digest_file",
            lambda _path: "stat-size-mtime:12345:1699999999.0",
        )
        with pytest.raises(GoldGenerationUnprovableError) as excinfo:
            gold_generation_key()
        assert "stat-size-mtime:" in str(excinfo.value)

    def test_an_absent_gold_tree_skips_rather_than_erroring(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A clean checkout without gold cannot compare generations, and says so."""
        monkeypatch.setattr(
            "tests.gold_generation.GOLD_MATRIX_PATHS",
            ("data/gold/features_no_such_target.parquet",),
        )
        with pytest.raises(pytest.skip.Exception) as excinfo:
            require_gold_generation(
                _A_DIFFERENT_GENERATION,
                reading=_READING,
                moved_by=_MOVED_BY,
                recorded_in=_RECORDED_IN,
            )
        assert _READING in str(excinfo.value)
