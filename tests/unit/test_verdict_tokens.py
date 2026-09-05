"""Five tokens, one per outcome, and a CLV p-value that can never reach the decision.

THE TRAP THIS MODULE EXISTS TO CLOSE. The repository's only significance primitive tests CLV,
not ROI. Substituting a CLV p-value for a profitability p-value would silently convert
"significant CLV" into "profitable" -- exactly the D25-14 and D26-09 trap this project has
already been caught by once, and exactly what Phase 31 exists to prevent. It would produce a
superficially valid but dishonest verdict, which is worse than a failed run.

The firewall is proven three ways, because one would be a convention:

1. STRUCTURALLY: :class:`TargetMeasurement` -- the only thing the verdict rule sees -- carries
   no closing-line-value field at all, so there is no CLV value in scope to read even
   dynamically.
2. BY AST SCAN: every function reachable from the verdict-assignment roots is visited and
   asserted to reference nothing CLV-shaped. The scan REPORTS how many functions it visited
   and fails when that number is zero, so it cannot pass vacuously.
3. BEHAVIOURALLY: a target carrying a wildly significant CLV p-value and a non-significant ROI
   p-value receives INCONCLUSIVE, and the SAME target with a significant ROI p-value receives
   PROFITABLE. The ROI p-value is therefore provably the driver.

The CLV figure is still CARRIED into the artifact as disclosed report-only context. It is
reported BESIDE the verdict and never inside it -- disclosed rather than suppressed.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import tomllib
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

from backtest.ev_chain_constants import (
    ALPHA,
    ROI_MIN_ATTAINABLE_P,
    VERDICT_TOKEN_MEANINGS,
    VERDICT_TOKENS,
)
from backtest.ou_monetization import CONTAMINATED_VOCAB
from backtest.profitability_2025 import (
    PROFITABLE,
    UNDISCHARGEABLE_NO_BETS,
    UNDISCHARGEABLE_NO_CHAIN,
    UNPROFITABLE,
    VERDICT_INCONCLUSIVE,
    TargetMeasurement,
    assign_verdict_token,
    build_verdict_records,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = (REPO_ROOT / "backtest" / "profitability_2025.py").read_text(
    encoding="utf-8"
)

# The two roots of the verdict-assignment path. Everything reachable from them is scanned.
VERDICT_ROOTS: tuple[str, ...] = ("build_verdict_records", "assign_verdict_token")


def _measurement(**overrides) -> TargetMeasurement:
    """A hand-built measurement, defaulting to a clean profitable shape."""
    base = {
        "target": "ats",
        "chain_resolved": True,
        "control_passed": True,
        "control_bet_count": 40,
        "bet_count": 55,
        "flat_roi": 0.08,
        "adjusted_p": 0.001,
        "raw_p": 0.0004997501249375312,
        "p_absent_reason": None,
        "n_blocks": 18,
        "n_push": 1,
        "n_ungraded": 0,
        "fallback_fired": False,
        "fallback_trigger": None,
    }
    base.update(overrides)
    return TargetMeasurement(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The reachability scan
# ---------------------------------------------------------------------------

# Anything whose NAME marks it as a closing-line-value quantity. Deliberately broad: the
# failure being prevented is a CLV number reaching the decision, and there is no legitimate
# reason for the verdict path to name one at all.
_CLV_SHAPED = "clv"


def scan_verdict_path_for_clv(
    source: str, roots: tuple[str, ...] = VERDICT_ROOTS
) -> tuple[list[str], list[str]]:
    """Walk the call graph from ``roots`` and report any CLV-shaped reference.

    Docstrings are EXCLUDED (a docstring that says "no CLV value is in scope here" must not
    trip a guard about code), and only module-level function definitions are followed --
    which is what the verdict path is made of.

    Returns:
        ``(violations, visited_function_names)``. The visited list is returned so the caller
        can fail when it is empty; a scan over nothing is a green test that proves nothing.
    """
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    visited: list[str] = []
    queue = [name for name in roots if name in functions]
    seen: set[str] = set(queue)
    violations: list[str] = []

    while queue:
        name = queue.pop()
        visited.append(name)
        node = functions[name]

        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]

        for statement in body:
            for inner in ast.walk(statement):
                if (
                    isinstance(inner, ast.Attribute)
                    and _CLV_SHAPED in inner.attr.lower()
                ):
                    violations.append(
                        f"{name}:{inner.lineno}: reads the attribute {inner.attr!r}"
                    )
                elif isinstance(inner, ast.Name) and _CLV_SHAPED in inner.id.lower():
                    violations.append(
                        f"{name}:{inner.lineno}: references the name {inner.id!r}"
                    )
                elif (
                    isinstance(inner, ast.Constant)
                    and isinstance(inner.value, str)
                    and _CLV_SHAPED in inner.value.lower()
                ):
                    violations.append(
                        f"{name}:{inner.lineno}: names {inner.value!r} as a key or literal"
                    )
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id in functions
                    and inner.func.id not in seen
                ):
                    seen.add(inner.func.id)
                    queue.append(inner.func.id)

    return violations, visited


class TestTheClvFirewallIsStructuralAndScanned:
    """Three independent proofs that a CLV p-value cannot drive a token."""

    def test_the_measurement_the_verdict_sees_has_no_clv_field(self) -> None:
        names = {field.name for field in dataclass_fields(TargetMeasurement)}
        offending = sorted(name for name in names if _CLV_SHAPED in name.lower())
        assert not offending, (
            f"TargetMeasurement carries {offending}. The firewall is that class: the verdict "
            "path must have no CLV value in scope at all, not merely refrain from reading one."
        )

    def test_the_scan_finds_no_clv_reference_and_visited_a_non_empty_set(self) -> None:
        violations, visited = scan_verdict_path_for_clv(RUNNER_SOURCE)
        assert visited, (
            "the CLV firewall scan visited ZERO functions, so its no-violation result proves "
            "nothing. Check that the verdict-assignment roots still exist under these names: "
            f"{VERDICT_ROOTS}."
        )
        assert set(VERDICT_ROOTS) <= set(visited), visited
        assert len(visited) >= 2, visited
        assert not violations, (
            "the verdict-assignment path reads a closing-line-value quantity:\n"
            + "\n".join(f"  - {line}" for line in violations)
            + f"\n\nFunctions visited ({len(visited)}): {sorted(visited)}"
        )

    def test_the_scan_reports_a_planted_clv_read(self) -> None:
        """Fail-closed control: a scan only ever observed passing may be unable to fail."""
        planted = (
            "def assign_verdict_token(measurement):\n"
            "    return _leak(measurement)\n"
            "\n"
            "def _leak(measurement):\n"
            "    return measurement.clv_p_value < 0.05\n"
        )
        violations, visited = scan_verdict_path_for_clv(
            planted, ("assign_verdict_token",)
        )
        assert "_leak" in visited, (
            "the scan did not follow the call into the helper, so a CLV read one level down "
            "would go unreported."
        )
        assert any("clv_p_value" in line for line in violations), violations

    def test_a_significant_clv_cannot_turn_an_insignificant_roi_into_profitable(
        self,
    ) -> None:
        """The behavioural counterpart. The CLV figure below is 1e-8 and changes nothing."""
        clv_p_value = 1e-8
        assert clv_p_value < ALPHA, "the fixture's CLV must be wildly significant"

        insignificant_roi = _measurement(target="wp", adjusted_p=0.4)
        token, reason = assign_verdict_token(insignificant_roi)
        assert token == VERDICT_INCONCLUSIVE, token
        assert "does NOT clear alpha" in reason

        # The SAME target, changing ONLY the ROI p-value, flips the token. The ROI p-value is
        # therefore provably the driver and the CLV p-value provably is not.
        significant_roi = _measurement(target="wp", adjusted_p=0.001)
        assert assign_verdict_token(significant_roi)[0] == PROFITABLE

    def test_the_clv_figure_is_still_disclosed_in_the_artifact(
        self, p31_rehearsal_run
    ) -> None:
        """Report-only means reported BESIDE the verdict, not suppressed."""
        text = p31_rehearsal_run["verdict_toml_path"].read_text(encoding="utf-8")
        parsed = tomllib.loads(text)
        assert "REPORT-ONLY" in text
        assert parsed["run"]["clv_p_value_is_report_only"] is True
        for target, record in parsed["targets"].items():
            for field in (
                "clv_report_only_mean",
                "clv_report_only_p",
                "clv_report_only_n",
            ):
                assert field in record, f"{target} does not disclose {field}"


class TestEveryOutcomeHasExactlyOneToken:
    """Five tokens, mutually exclusive and jointly exhaustive, one test each."""

    def test_a_profitable_target(self) -> None:
        token, reason = assign_verdict_token(
            _measurement(flat_roi=0.08, adjusted_p=0.001)
        )
        assert token == PROFITABLE
        assert "strictly below alpha" in reason

    def test_an_unprofitable_target(self) -> None:
        token, reason = assign_verdict_token(
            _measurement(flat_roi=-0.11, adjusted_p=0.9)
        )
        assert token == UNPROFITABLE
        assert "MEASURED result" in reason

    def test_a_break_even_target_is_unprofitable_not_inconclusive(self) -> None:
        """Exactly zero is NOT strictly positive, and a measured break-even is a result."""
        assert (
            assign_verdict_token(_measurement(flat_roi=0.0, adjusted_p=0.001))[0]
            == UNPROFITABLE
        )

    def test_an_inconclusive_target(self) -> None:
        token, reason = assign_verdict_token(
            _measurement(flat_roi=0.08, adjusted_p=0.4)
        )
        assert token == VERDICT_INCONCLUSIVE
        assert "never called PROFITABLE" in reason

    def test_a_zero_bet_target_whose_control_passed(self) -> None:
        token, reason = assign_verdict_token(
            _measurement(
                bet_count=0,
                flat_roi=None,
                adjusted_p=None,
                raw_p=None,
                control_passed=True,
                control_bet_count=31,
                p_absent_reason="no bets were selected",
            )
        )
        assert token == UNDISCHARGEABLE_NO_BETS
        assert "NO EDGE and not a broken chain" in reason
        assert "31 bets" in reason

    def test_a_target_whose_chain_did_not_resolve(self) -> None:
        token, reason = assign_verdict_token(
            _measurement(chain_resolved=False, bet_count=0, flat_roi=None)
        )
        assert token == UNDISCHARGEABLE_NO_CHAIN
        assert "could not run" in reason

    def test_a_zero_bet_target_whose_control_also_failed_is_no_chain_not_no_bets(
        self,
    ) -> None:
        """What the counterfactual control buys: a zero that means broken, named as such."""
        token, reason = assign_verdict_token(
            _measurement(
                bet_count=0,
                flat_roi=None,
                adjusted_p=None,
                control_passed=False,
                control_bet_count=0,
            )
        )
        assert token == UNDISCHARGEABLE_NO_CHAIN
        assert "unexplained zero" in reason

    def test_every_token_in_the_frozen_vocabulary_is_reachable(self) -> None:
        """Anti-vacuity for the set: no token exists that nothing can produce."""
        produced = {
            assign_verdict_token(_measurement(flat_roi=0.08, adjusted_p=0.001))[0],
            assign_verdict_token(_measurement(flat_roi=-0.1, adjusted_p=0.9))[0],
            assign_verdict_token(_measurement(flat_roi=0.08, adjusted_p=0.4))[0],
            assign_verdict_token(
                _measurement(bet_count=0, flat_roi=None, control_passed=True)
            )[0],
            assign_verdict_token(_measurement(chain_resolved=False))[0],
        }
        assert produced == set(VERDICT_TOKENS)


class TestTheSignificanceBoundaryAndTheFiniteSampleRule:
    """The two adjacency edges the pre-registration fixed in advance."""

    def test_an_adjusted_p_exactly_equal_to_alpha_is_not_significant(self) -> None:
        token, reason = assign_verdict_token(
            _measurement(flat_roi=0.08, adjusted_p=ALPHA)
        )
        assert token == VERDICT_INCONCLUSIVE, (
            "an adjusted p EXACTLY at alpha was called significant. The comparison is "
            "STRICTLY LESS THAN, the canonical convention, fixed before the numbers existed."
        )
        assert "strictly less than" in reason

    def test_a_hair_below_alpha_is_significant(self) -> None:
        """The boundary binds in one direction only; the other side must still pass."""
        assert (
            assign_verdict_token(
                _measurement(flat_roi=0.08, adjusted_p=ALPHA * (1 - 1e-12))
            )[0]
            == PROFITABLE
        )

    def test_a_floor_above_alpha_reports_inconclusive_with_the_stated_reason(
        self,
    ) -> None:
        """The registered consequence, exercised by driving alpha BELOW the floor.

        On the frozen configuration the minimum attainable p is 1/2001, which is below alpha,
        so this branch does not bind today. The pre-registration registered it anyway, because
        a rule that only exists once it binds is a rule chosen after the fact -- so it is
        asserted directly rather than left unexercised.
        """
        tiny_alpha = ROI_MIN_ATTAINABLE_P / 10.0
        token, reason = assign_verdict_token(
            _measurement(flat_roi=0.25, adjusted_p=ROI_MIN_ATTAINABLE_P),
            alpha=tiny_alpha,
        )
        assert token == VERDICT_INCONCLUSIVE
        assert "MINIMUM ATTAINABLE p" in reason
        assert "NEVER called PROFITABLE" in reason

    def test_an_untestable_positive_return_is_inconclusive_not_profitable(self) -> None:
        token, reason = assign_verdict_token(
            _measurement(
                flat_roi=0.4,
                adjusted_p=None,
                raw_p=None,
                n_blocks=1,
                p_absent_reason="the hold bets span fewer than 2 (season, week) blocks",
            )
        )
        assert token == VERDICT_INCONCLUSIVE
        assert "fewer than 2" in reason


class TestZeroBetsIsAFirstClassResult:
    """The record shape does not change with the token; only the RETURN goes absent."""

    def test_every_record_carries_the_same_field_names(self) -> None:
        records = build_verdict_records(
            [
                _measurement(target="wp", flat_roi=0.08, adjusted_p=0.001),
                _measurement(
                    target="ats",
                    bet_count=0,
                    flat_roi=None,
                    adjusted_p=None,
                    raw_p=None,
                ),
                _measurement(
                    target="ou", chain_resolved=False, bet_count=0, flat_roi=None
                ),
            ]
        )
        shapes = {target: set(record) for target, record in records.items()}
        assert shapes["wp"] == shapes["ats"] == shapes["ou"], (
            "an UNDISCHARGEABLE target's record has a different field set from a betting "
            "target's, so the readout template's slots change shape with the data and an "
            "absent verdict reads as a gap rather than as a result."
        )

    def test_a_zero_bet_targets_return_is_absent_and_never_zero(self) -> None:
        records = build_verdict_records(
            [
                _measurement(
                    target="ats",
                    bet_count=0,
                    flat_roi=None,
                    adjusted_p=None,
                    raw_p=None,
                )
            ]
        )
        record = records["ats"]
        assert record["verdict_token"] == UNDISCHARGEABLE_NO_BETS
        assert record["has_return"] is False
        assert record["flat_roi"] is None, (
            "a zero-bet target reported a RETURN. A return of zero is a measured break-even "
            "and no bets is not; the two must never render as the same number."
        )
        assert record["flat_roi"] != 0.0

    def test_the_records_are_in_the_canonical_target_order(self) -> None:
        records = build_verdict_records(
            [
                _measurement(target="ou"),
                _measurement(target="wp"),
                _measurement(target="ats"),
            ]
        )
        assert list(records) == ["wp", "ats", "ou"]

    def test_every_record_carries_the_frozen_meaning_of_its_token(self) -> None:
        records = build_verdict_records([_measurement(target="wp")])
        record = records["wp"]
        assert (
            record["token_meaning"] == VERDICT_TOKEN_MEANINGS[record["verdict_token"]]
        )


class TestTheVerdictVocabularyCannotBeMistakenForAContaminatedOne:
    """A 2025 token must never read as a Phase-27 or Phase-30 provisional figure."""

    def test_the_two_vocabularies_are_disjoint(self) -> None:
        clean = {token.lower() for token in VERDICT_TOKENS}
        contaminated = {phrase.lower() for phrase in CONTAMINATED_VOCAB}
        assert not (clean & contaminated), sorted(clean & contaminated)

    def test_no_verdict_token_contains_a_contaminated_word(self) -> None:
        offending = [
            (token, phrase)
            for token in VERDICT_TOKENS
            for phrase in CONTAMINATED_VOCAB
            if phrase.lower() in token.lower() or token.lower() in phrase.lower()
        ]
        assert not offending, offending

    def test_every_clean_token_says_clean_and_every_undischargeable_says_so(
        self,
    ) -> None:
        for token in VERDICT_TOKENS:
            assert token.endswith("_CLEAN") or token.startswith("UNDISCHARGEABLE_"), (
                f"{token} carries neither the clean marker nor the undischargeable one, so a "
                "reader cannot tell at a glance which split it was measured on."
            )


@pytest.mark.parametrize("token", VERDICT_TOKENS)
def test_every_frozen_token_has_a_stated_meaning(token: str) -> None:
    assert VERDICT_TOKEN_MEANINGS[token].strip()
