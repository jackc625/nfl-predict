"""Permanent doc-drift guard for the repo-root LIVE-COLD-START-READOUT.md (Plan 33-18 Task 9).

A PERMANENT committed test on the ``tests/unit/test_refit_readout_md.py`` pattern. It guards
Phase 33's closing record against these failure modes:

  - the file is missing, not at the repo root, non-ASCII, or has lost one of its ten sections;
  - a required phrase is gone -- above all "re-derivation" for the Elo work (F-03: no known-good
    copy of the burn-in existed anywhere, so the work was derived again, not recovered) and the
    sentence that a zero exit code is not the evidence (SPEC R14);
  - a forbidden word entered it: the Elo wording rule's two spellings
    (``LIVE_READOUT_FORBIDDEN_WORDS``), the repo's over-claim dictionary
    (``backtest.ev_chain_constants.READOUT_FORBIDDEN_WORDS``) and the phase's suite-claim
    phrasings (``FORBIDDEN_SUITE_CLAIM_PHRASES``). All three are READ, never spelled here;
  - the document drifting from the manifest: the checked gold build clock, the two week-3 games
    with no prediction, the observed bracket date, the two anchor commits and the recorded
    tripwire line are read from ``tests/phase33_state.py`` and must appear verbatim;
  - a superseded record presented as live: every Phase-33 model and blend id that Phase 33.2
    replaced, and the superseded ``11761c7`` freeze, must appear only on lines that also name
    what replaced it (D33-38), and no statistic of the Plan 33-15 gate verdicts
    (``config/phase33_gate_verdict.toml``) may appear at all;
  - the readout going unregistered in ``tests/unit/test_old_rule_labels.py``'s repo-root
    partition (an unregistered tracked readout makes that module raise at import).

Two planted controls prove the word check and the replacement check each have a reachable
failing state. It asserts RECORDED values and rulings, never a re-derived point estimate.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import math
import re
import tomllib
from pathlib import Path

from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
READOUT_MD = REPO_ROOT / "docs" / "records" / "LIVE-COLD-START-READOUT.md"
GATE_VERDICT_TOML = REPO_ROOT / "config" / "phase33_gate_verdict.toml"

_DECIMAL_IN_TEXT = re.compile(r"[+-]?\d+\.\d{3,}")


def _read() -> str:
    return READOUT_MD.read_text(encoding="utf-8")


def superseded_pairs() -> list[tuple[str, str]]:
    """``(superseded, replacement)`` for every Phase-33 record Phase 33.2 replaced (D33-38)."""
    replacements = dict(phase33_state.P332_25B_SWAP_ARTIFACT_IDS)
    pairs = [
        (old, replacements[key])
        for key, old in phase33_state.P332_25_PRE_SWAP_LATEST_JSON
    ]
    pairs.append(
        (
            phase33_state.PRE_REGISTRATION_COMMIT[:7],
            phase33_state.COLD_START_CORRECTIVE_COMMIT_SHA[:7],
        )
    )
    return pairs


def superseded_without_replacement(text: str) -> list[str]:
    """Superseded records that are absent, or appear on a line not naming the replacement."""
    problems: list[str] = []
    lines = text.splitlines()
    for old, new in superseded_pairs():
        carrying = [line for line in lines if old in line]
        if not carrying:
            problems.append(f"{old}: never named")
        problems.extend(
            f"{old}: named without {new} on line {line.strip()[:80]!r}"
            for line in carrying
            if new not in line
        )
    return problems


def forbidden_hits(text: str) -> list[str]:
    """Every forbidden word or suite-claim phrase present (case-insensitive substring)."""
    from backtest.ev_chain_constants import READOUT_FORBIDDEN_WORDS

    vocabulary = (
        tuple(phase33_state.LIVE_READOUT_FORBIDDEN_WORDS)
        + tuple(READOUT_FORBIDDEN_WORDS)
        + tuple(phase33_state.FORBIDDEN_SUITE_CLAIM_PHRASES)
    )
    lowered = text.lower()
    return [word for word in vocabulary if word.lower() in lowered]


def gate_verdict_statistics() -> set[str]:
    """Every textual form of a Plan 33-15 verdict statistic the readout must not restate."""
    record = tomllib.loads(GATE_VERDICT_TOML.read_text(encoding="utf-8"))
    forms: set[str] = set()
    for verdict in record["verdicts"].values():
        values = [
            verdict.get(key) for key in ("paired_statistic", "p_value", "paired_mean")
        ]
        for table in ("secondary_scalars", "comparator_secondary_scalars"):
            values.extend(verdict.get(table, {}).values())
        for value in values:
            if not isinstance(value, float) or math.isnan(value):
                continue
            forms.add(repr(value))
            if abs(value) >= 1e-3:
                forms.update({f"{value:.4f}", f"{value:.6f}"})
        for reason in verdict.get("reasons", []):
            forms.update(_DECIMAL_IN_TEXT.findall(reason))
    return forms


def state_derived_phrases() -> list[str]:
    """Values the readout reports, read from the manifest so the two cannot disagree."""
    return [
        phase33_state.LIVE_ACCEPTANCE_GOLD_BUILD_CLOCK_UTC,
        *(
            game
            for game, _lock, _reason in phase33_state.LIVE_ACCEPTANCE_WEEK3_NOT_PREDICTED
        ),
        phase33_state.DAILY_BRACKET_OBSERVED_RUN_DATE_ET,
        phase33_state.COLD_START_CORRECTIVE_COMMIT_SHA[:7],
        phase33_state.PRE_REGISTRATION_COMMIT[:7],
        phase33_state.CLOSE_TRIPWIRE_RUN_LINE,
    ]


class TestTheReadoutExists:
    def test_it_is_at_the_repo_root(self) -> None:
        assert READOUT_MD.is_file(), f"missing: {READOUT_MD}"

    def test_it_is_ascii(self) -> None:
        assert _read().isascii()

    def test_every_section_is_present_in_order(self) -> None:
        text = _read()
        positions = [
            text.find(marker) for marker in phase33_state.LIVE_READOUT_SECTION_MARKERS
        ]
        missing = [
            marker
            for marker, position in zip(
                phase33_state.LIVE_READOUT_SECTION_MARKERS, positions, strict=True
            )
            if position < 0
        ]
        assert missing == []
        assert positions == sorted(positions)

    def test_every_required_phrase_is_present(self) -> None:
        lowered = _read().lower()
        missing = [
            phrase
            for phrase in phase33_state.LIVE_READOUT_REQUIRED_PHRASES
            if phrase.lower() not in lowered
        ]
        assert missing == []

    def test_every_state_derived_value_is_present(self) -> None:
        text = _read()
        assert [value for value in state_derived_phrases() if value not in text] == []


class TestTheWordingRules:
    def test_no_forbidden_word_or_suite_claim_appears(self) -> None:
        assert forbidden_hits(_read()) == []

    def test_a_planted_forbidden_word_is_caught(self) -> None:
        """Planted control: the word check has a reachable failing state."""
        word = phase33_state.LIVE_READOUT_FORBIDDEN_WORDS[0]
        planted = _read() + f"\nThe Elo history was {word}d from a copy.\n"
        assert word in forbidden_hits(planted)


class TestSupersededRecordsAreNamedOnlyBesideTheirReplacements:
    def test_the_pairs_cover_the_four_swapped_records_and_the_freeze(self) -> None:
        """Non-vacuity: the check runs over five pairs, not over nothing."""
        assert len(superseded_pairs()) == 5

    def test_every_superseded_record_sits_beside_its_replacement(self) -> None:
        assert superseded_without_replacement(_read()) == []

    def test_a_planted_bare_superseded_id_is_caught(self) -> None:
        """Planted control: a superseded id on a line without its replacement fails."""
        old = dict(phase33_state.P332_25_PRE_SWAP_LATEST_JSON)["wp"]
        planted = _read() + f"\nThe live win model is {old}.\n"
        assert any(
            old in problem for problem in superseded_without_replacement(planted)
        )

    def test_no_plan_33_15_verdict_statistic_is_restated(self) -> None:
        statistics = gate_verdict_statistics()
        assert statistics, (
            "non-vacuity: the verdict record yielded no statistic to check"
        )
        text = _read()
        assert sorted(form for form in statistics if form in text) == []


class TestTheReadoutIsRegistered:
    def test_it_is_in_the_old_rule_label_partition_with_a_reason(self) -> None:
        from tests.unit.test_old_rule_labels import NO_PREFIX_NUMBERS_REASONS

        reason = NO_PREFIX_NUMBERS_REASONS.get(READOUT_MD.name, "")
        assert reason.strip(), f"{READOUT_MD.name} carries no partition reason"
