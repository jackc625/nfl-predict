"""Permanent doc-drift guard for the repo-root LINE-MOVEMENT-READOUT.md (Phase 29, SIG-04 / D-16).

Mirrors ``tests/unit/test_signal_lift_readout_md.py`` (the committed doc-drift-guard pattern): a
PERMANENT committed test, NOT a throwaway ``scripts/check_*.py``. It guards the SIG-04 / D-16
deliverable ``LINE-MOVEMENT-READOUT.md`` -- the budget-gate go/no-go record -- against the
failure modes that would silently corrupt the gate record:

  - the file is missing or not at the repo root,
  - the content carries non-ASCII (emoji / cp1252-hostile) characters (CLAUDE.md hard constraint),
  - a required section (the tiered cost, the plausibility argument, the decision, the lift
    placeholder, the scope-down/slip note) was silently dropped,
  - the SCREEN-NOT-DEPLOY invariant was violated -- the doc must say "carried to Phase 30" and
    must NEVER say "deployed" / "proven" (a budget-gate spike is not a deploy decision, D-16),
  - the machine-readable ``selected_branch:`` marker is missing, duplicated, or carries an
    unknown token (downstream branch gating reads this literal token, not prose -- review 29-01
    LOW: assert EXACTLY ONE valid branch token).

Plan 29-07 (the backfill path) extends it with the NUMERIC lift-reproduction check
(``TestReadoutMatchesHarness``), exactly as the Phase-28 readout guard does: the Section-4 numbers
must reproduce from a live ``backtest.signal_lift`` run, and -- the load-bearing one for this
phase -- the claim that NO line-movement column reached any model under the canonical window must
reproduce too. If a future gold rebuild or config change makes the family selectable, that
assertion fails and Section 4 has to be rewritten rather than silently going stale.

PHASE 30 RE-SCOPE (Plan 30-03, D30-06)
--------------------------------------
``TestReadoutMatchesHarness`` now runs against a COMMITTED FIXTURE of the pre-Phase-30 gold
(``tests/fixtures/gold/``), not against live ``data/gold``. Two reasons, and the second is the
one that matters:

  1. Phase 30 rebuilds gold four times on purpose (WR-06 bounds, the CR-02 flag repair, the
     line_movement drop, the N-01 re-sync). Any one of those moves the ATS deltas far past the
     ``5e-3`` tolerance these tests compare within, so pinned-to-live-gold reproductions were
     always going to go stale -- with or without the drop.
  2. The drop specifically turns two of the four reproductions into VACUOUS PASSES. The old
     skip-guard does NOT fire after the drop (gold still exists, it is merely 15 columns
     narrower), so the assertions run, and both ``n_group_columns_selected == 0`` and
     ``"line_movement_coverage" not in group_columns_selected`` become trivially true over an
     EMPTY selected-column list. A green test that proves nothing is worse than a red one,
     because nobody revisits it.

So the scope of these four tests is now explicit and narrower: they prove the published Phase-29
numbers were produced by the COMMITTED HARNESS ON THE GOLD THAT PRODUCED THEM. They say nothing
whatsoever about live gold. What they used to assert about the present is replaced by
``test_line_movement_family_is_absent_from_live_gold``, and the fixture's own identity is pinned
by ``test_fixture_is_the_pre_drop_gold`` so it cannot be quietly regenerated to rescue a failing
assertion.

NOT AN OVERSIGHT: what Phase 30 deliberately LEAVES IN PLACE
-----------------------------------------------------------
``features/line_movement.py`` and the ``features/validation.py`` leakage-keyword entry are
deliberately retained by this phase, and ``"line_movement"`` STAYS registered in
``backtest.signal_lift._GROUP_PREDICATE``. With the registration retained,
``group_columns(post_drop_gold, "line_movement")`` returns an empty list, ``excluded_columns``
adds nothing, and the committed registry-membership assertion in
``tests/unit/test_signal_lift_readout_md.py`` (``test_phase28_baseline_is_pinned_against_later_widening``)
stays green. Removing the registration would turn that test red for no benefit AND would re-arm
the 29-06 trap -- a family registered by no predicate falls straight into the BASELINE leg of
every screen -- for the next phase that widens gold.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

# Repo root resolved from this file: tests/unit/test_line_movement_readout_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
READOUT_MD = REPO_ROOT / "LINE-MOVEMENT-READOUT.md"

# ---------------------------------------------------------------------------
# The Phase-30 frozen fixture (Plan 30-03, D30-06).
#
# COMMITTED artifacts, captured at repository SHA dc4d1c0 before any rung of the D30-17 rebuild
# ladder ran, byte-for-byte copies of data/gold/features_ats.parquet and
# data/silver/odds_snapshot.parquet as they stood when every published Phase-29 reading was
# measured. They are tracked only because .gitignore carries the narrow file-level negation
# `!tests/fixtures/gold/*.parquet` after the repository-wide `*.parquet` rule.
#
# Digests, shape and column count are transcribed from tests/fixtures/gold/PROVENANCE.md, which
# also prints each fixture digest beside the digest of the SOURCE it was copied from.
# ---------------------------------------------------------------------------
_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "gold"
_FIXTURE_GOLD = _FIXTURE_DIR / "features_ats_pre_phase30.parquet"
_FIXTURE_ODDS = _FIXTURE_DIR / "odds_snapshot_pre_phase30.parquet"
_FIXTURE_PROVENANCE = _FIXTURE_DIR / "PROVENANCE.md"

_FIXTURE_GOLD_SHA256 = (
    "733a1273fde4067607480540fd1357e50b81d31e5793b386d4f916e3057a5728"
)
_FIXTURE_ODDS_SHA256 = (
    "405133f16de9167ee37d1ff81b85456572029d8a936fc0a6b922d564f662a790"
)
_FIXTURE_GOLD_SHAPE = (6263, 210)
_FIXTURE_GOLD_LINE_MOVEMENT_COLUMNS = 15

# Live gold -- read ONLY by test_line_movement_family_is_absent_from_live_gold, which is the
# Phase-30 DROP proof (SPEC R3). `data/` is gitignored, so live gold can legitimately be absent
# on a fresh checkout and that test skip-guards on presence. The four harness reproductions do
# NOT read these paths any more.
_LIVE_GOLD_PATHS = {
    target: REPO_ROOT / "data" / "gold" / f"features_{target}.parquet"
    for target in ("wp", "ats", "ou")
}

# Required section markers (the D-16 five-section contract). A substring of each section header so
# a benign reword does not trip the guard but a dropped section does.
_REQUIRED_SECTION_MARKERS = (
    "Tiered cost",  # 1 -- the (a)/(b)/(c) x window credit+dollar menu
    "Plausibility (D-02)",  # 2 -- the non-redundancy argument
    "Decision:",  # 3 -- the owner branch decision
    "Lift results",  # 4 -- the backfill-path lift placeholder
    "Honest scope-down",  # 5 -- the scope-down / slip note
    "Phase-30 annotation",  # 6 -- the CR-03 inert-tell finding + the DROP-as-decision record
)

# Section 6 (Plan 30-03) content contract: the literal tokens the CR-03 annotation must carry.
#
# Two claims are load-bearing and are pinned as phrases rather than as section presence, because
# a section can survive while its meaning is edited out from under it:
#
#   1. the tell is STRUCTURALLY INERT -- so the published "did not fire" readings are annotated
#      rather than silently trusted;
#   2. the DROP is a DECISION taken on the structural 2018-2019 selection-window fact, and
#      explicitly NOT on the tell -- so no reader can mistake one for the other (T-30-21).
_REQUIRED_SECTION6_MARKERS = (
    "CR-03",
    "structurally inert",  # (1) the finding, in the words the SPEC asks for
    "the DROP is a DECISION",  # (2) the grounds, stated as a decision not a calendar accident
    "does NOT rest on the confound tell",  # (2) the separation, stated where it is checkable
    "SPEC R3",
    "D29-07-01",
    "2020-06-06",  # the archive floor that makes the family unselectable
    "0 of 15 can EVER be selected",  # the structural fact the DROP actually rests on
    "canonical 2018-2019 selection window",  # retained unchanged
    "retained on disk untouched",  # the paid odds_timeline archive is not destroyed
    "NOT-MEASURED run by construction",  # what --phase 29 becomes after the drop
    "honestly reports the family as absent",  # ... rather than having been removed
)

# Literal phrases the doc must carry. The D-04 pricing freshness stamp used to live
# here too and no longer does -- see _FRESHNESS_RE below for why a literal is the
# wrong tool for a recency claim.
_REQUIRED_PHRASES = ("carried to Phase 30",)  # the screen-not-deploy carry phrase

# WR-13: the D-04 pricing-freshness stamp is checked for RECENCY, not for a fixed
# literal.
#
# _REQUIRED_PHRASES used to assert the literal "checked 2026-06-29" as "the
# 7-day-freshness re-confirmation stamp". That guard can never fail on staleness
# -- it asserts the presence of a frozen string, so it passes just as happily when
# the stamp is a year old. A freshness check that cannot detect staleness is worse
# than none: it reads as assurance. Parse the dates and assert an age bound.
_FRESHNESS_RE = re.compile(r"checked (\d{4}-\d{2}-\d{2})")

# Generous on purpose. The point is not to nag; it is that a credit/dollar figure
# quoted from a vendor's price list must not be presented as current indefinitely.
_MAX_STAMP_AGE_DAYS = 400

# Section 4 (Plan 29-07) content contract: the lift grid must carry the per-target deltas, the
# covered-span annotation, the D-02 priced-in caveat, the METHOD line, and -- the honesty clause
# this phase turns on -- the disclosure that the canonical grid's columns never reached a model.
_REQUIRED_SECTION4_MARKERS = (
    "METHOD",
    "baseline_exclude_groups=('line_movement',)",  # the baseline KEEPS Phase 28 (review 29-07 HIGH)
    "train_and_evaluate(tune=False)",  # the out-of-sample walk-forward anchor
    "priced-in",  # the D-02 redundancy caveat (Section 2 phrasing, carried into 4)
    "2020-06-06",  # the archive floor / covered span
    "D29-06-01",  # the orphaned-2020 disclosure (now marked RESOLVED)
    "selection churn",  # what the canonical deltas actually are
    "7,210 credits",  # the real cost, recorded against what was learned
    # Quick task 260816-u0e: the re-key, the pre-registration audit trail, and the
    # headline grid. Each marker names a piece of Section 4 that would otherwise be
    # free to go stale unnoticed.
    "SUPERSEDED",  # 4c is retained verbatim and labelled, never deleted
    "4c-bis",  # the superseding pre-registration
    "pre_registration_commit:",  # the machine-readable ordering marker
    "RE-MEASURED",  # 4b's figures are current, not preserved-and-stale
    "HEADLINE",  # 4d exists
    "764 paired games",  # the headline's measured sample
    "confound tell",  # the pre-registered line_movement_coverage season-proxy tell
    "NON-DEFAULT",  # the D-Q2 consequence for Phase 30
    # Quick task 260817-dyp: the leak fix, the corrected grid, and the audit trail that keeps
    # every superseded reading in place beside it.
    "4d-bis",  # the corrected headline subsection exists
    "PRE-LEAK-FIX",  # 4d is retained and relabelled, never overwritten
    "4e-bis",  # the corrected ruling
    "THE RULE WAS NOT RE-OPENED",  # the D-R3 claim, stated where it can be checked
    "paired_sufficient",  # the WR-02 sufficiency field, recorded per cell
    "MIN_CLV_SAMPLE",  # the threshold the refusal arm was inert against
    "zero games admit a snapshot at or after kickoff",  # the leak-purge verification
)

# The three canonical-window per-target deltas, RE-ANCHORED 2026-08-17 to the post-leak-fix
# reading (quick task 260817-dyp).
#
# 4a IS NO LONGER AN UPSTREAM-DRIFT CONTROL, and this constant is where that shows up. Its
# 2018-2019 selection window cannot see the line-movement family (still 0/15 selected), but the
# WR-06 timezone normalization moved home_rest_days / away_rest_days / rest_advantage in
# seasons 2018-2025 -- BASELINE features inside that very window -- so both legs' models changed
# and the deltas moved with them. The pre-leak-fix values (wp +0.000000, ats +0.147814,
# ou -0.221188) are retained in the doc as history and asserted by the token list below.
_CANONICAL_DELTAS = {"wp": 0.000000, "ats": -0.136848, "ou": -0.294672}

# The coverage-window diagnostic (4b), third reading -- RE-MEASURED 2026-08-17 on post-leak-fix
# gold. Its walk-forward trains on every season below 2024, so it sees both the corrected
# line-movement values and the corrected rest-days values. The sequence of readings is
# pre-re-key, then post-re-key pre-leak-fix, then post-leak-fix; all three are retained in the
# doc and asserted by the token list below.
_COVERAGE_WINDOW_ATS_DELTA = 0.750368

# The pre-registered headline grid, CORRECTED (4d-bis): train 2018-2020 / hp-val 2021 / measure
# 2022-2024, re-run once on post-leak-fix gold under the rule already committed in 5660ee3.
# The pre-leak-fix reading (+0.151237 on 5/15 columns) is retained in 4d and asserted below.
_HEADLINE_ATS_DELTA = -0.208582
_HEADLINE_ATS_GROUP_COLS_SELECTED = 4
_HEADLINE_MEASURE_WINDOW = "2022-2024"

# Delta tokens that must NOT appear in either pre-registration commit's diff: if a headline
# number is present in the commit that registered the rule, the rule was not written first.
_HEADLINE_DELTA_TOKENS = (
    "0.151237",
    "0.001429",
    "0.001592",
    "1.012106",
    # The corrected readings (quick task 260817-dyp). Neither registration commit may contain
    # these either -- the rule was written before ANY of these numbers existed.
    "0.208582",
    "0.003519",
    "0.001594",
    "0.750368",
    "0.136848",
    "0.294672",
)

# The machine-readable pre-registration ordering markers (4d).
_PRE_REGISTRATION_MARKER_RE = re.compile(
    r"^(?:superseding_)?pre_registration_commit:\s*([0-9a-f]{40})\s*$", re.MULTILINE
)

# The screen-not-deploy invariant: these over-claim words must NEVER appear (mirrors the negative
# grep in the plan's acceptance: ``grep -ci 'deployed\\|proven'`` must return 0).
_FORBIDDEN_WORDS = ("deployed", "proven")

# The machine-readable branch marker contract (review 29-01 LOW): exactly one line of the form
# ``selected_branch: <token>`` with a token from this set.
_VALID_BRANCH_TOKENS = frozenset(
    {"PENDING", "full-backfill", "forward-collect-only", "slip"}
)
_BRANCH_MARKER_RE = re.compile(r"^selected_branch:\s*(\S+)\s*$", re.MULTILINE)


def _read_readout() -> str:
    """Read LINE-MOVEMENT-READOUT.md from the repo root."""
    return READOUT_MD.read_text(encoding="utf-8")


class TestReadoutExists:
    """LINE-MOVEMENT-READOUT.md must exist at the repo root and be ASCII-only."""

    def test_file_exists_at_repo_root(self) -> None:
        """The SIG-04 / D-16 deliverable is present at the repo root (NOT under .planning/)."""
        assert READOUT_MD.is_file(), f"missing: {READOUT_MD}"

    def test_content_is_ascii(self) -> None:
        """Content is ASCII-only (no emoji / non-ASCII, per CLAUDE.md)."""
        content = _read_readout()
        assert content.isascii(), (
            "LINE-MOVEMENT-READOUT.md contains non-ASCII characters"
        )


class TestReadoutSections:
    """All five required sections (D-16) must be present."""

    def test_all_required_section_markers_present(self) -> None:
        """Every required section marker is present; names any missing so a drop is actionable."""
        content = _read_readout()
        missing = [m for m in _REQUIRED_SECTION_MARKERS if m not in content]
        assert not missing, (
            f"LINE-MOVEMENT-READOUT.md missing required sections: {missing}"
        )

    def test_tiered_cost_table_has_all_three_rungs(self) -> None:
        """Section 1 prices the full (a)/(b)/(c) tiered menu (D-03)."""
        content = _read_readout()
        for rung in ("(a) full trajectory", "(b) full trajectory", "(c) two-anchor"):
            assert rung in content, f"the tiered cost menu is missing rung {rung!r}"

    def test_cost_is_an_estimate_not_a_call(self) -> None:
        """Section 1 is headed as an ESTIMATE from published pricing (D-04 -- endpoint not called)."""
        content = _read_readout()
        assert "estimated from" in content.lower(), (
            "the cost section must state it is ESTIMATED from published pricing (D-04)"
        )
        assert "not called" in content.lower(), (
            "the cost section must state the historical endpoint was NOT called (D-04)"
        )


class TestScreenNotDeployInvariant:
    """The doc must say 'carried to Phase 30' and must NEVER say 'deployed' / 'proven' (D-16)."""

    def test_required_phrases_present(self) -> None:
        """The carry phrase + the freshness re-confirmation stamp are present."""
        content = _read_readout()
        missing = [p for p in _REQUIRED_PHRASES if p not in content]
        assert not missing, (
            f"LINE-MOVEMENT-READOUT.md missing required phrases: {missing}"
        )

    def test_pricing_stamp_is_present_and_not_ancient(self) -> None:
        """WR-13: the D-04 freshness stamp is checked for RECENCY, not presence.

        The old guard asserted the literal "checked 2026-06-29", so it could never
        go red as that stamp aged -- a freshness check incapable of failing on
        staleness is worse than none, because it reads as assurance. The readout
        quotes Odds API credit and dollar figures; those must not be presented as
        current indefinitely.
        """
        stamps = [date.fromisoformat(s) for s in _FRESHNESS_RE.findall(_read_readout())]

        assert stamps, (
            "the D-04 pricing freshness stamp is missing -- the tiered-cost "
            "section must record when the pricing was last confirmed"
        )

        age = (datetime.now(UTC).date() - max(stamps)).days
        assert age <= _MAX_STAMP_AGE_DAYS, (
            f"the newest pricing stamp is {age} days old (max "
            f"{_MAX_STAMP_AGE_DAYS}); re-confirm Odds API pricing before quoting "
            f"these credit figures again, then update the stamp"
        )

    def test_no_over_claim_words(self) -> None:
        """The over-claim words 'deployed' / 'proven' must NOT appear (a spike is not a deploy)."""
        lowered = _read_readout().lower()
        present = [w for w in _FORBIDDEN_WORDS if w in lowered]
        assert not present, (
            f"LINE-MOVEMENT-READOUT.md over-claims a budget-gate as a deploy: {present} "
            "(D-16 -- use 'carried to Phase 30', never 'deployed' / 'proven')"
        )

    def test_no_literal_api_key(self) -> None:
        """The readout records dollars/credits only; the ODDS_API_KEY is never echoed."""
        lowered = _read_readout().lower()
        assert "apikey=" not in lowered, (
            "the readout must not echo a literal apiKey= value"
        )


class TestSelectedBranchMarker:
    """Exactly one machine-readable ``selected_branch:`` token with a valid value (review 29-01 LOW)."""

    def test_exactly_one_marker_line(self) -> None:
        """There is EXACTLY ONE ``selected_branch:`` marker line (not zero, not multiple)."""
        matches = _BRANCH_MARKER_RE.findall(_read_readout())
        assert len(matches) == 1, (
            f"expected exactly one 'selected_branch:' marker line, found {len(matches)}: {matches}"
        )

    def test_marker_token_is_valid(self) -> None:
        """The marker's token is one of {PENDING, full-backfill, forward-collect-only, slip}."""
        matches = _BRANCH_MARKER_RE.findall(_read_readout())
        assert len(matches) == 1, (
            f"expected exactly one 'selected_branch:' marker line, found {len(matches)}: {matches}"
        )
        token = matches[0]
        assert token in _VALID_BRANCH_TOKENS, (
            f"selected_branch token {token!r} is not one of {sorted(_VALID_BRANCH_TOKENS)}"
        )


class TestLiftSectionContent:
    """Section 4 (Plan 29-07) carries the grid, the caveats, and the not-measured disclosure."""

    def test_required_section4_markers_present(self) -> None:
        content = _read_readout()
        missing = [m for m in _REQUIRED_SECTION4_MARKERS if m not in content]
        assert not missing, f"LINE-MOVEMENT-READOUT.md Section 4 missing: {missing}"

    def test_per_target_grid_covers_all_three_targets(self) -> None:
        """Both grids report WP / ATS / OU (the D-13 rule is per-target)."""
        content = _read_readout()
        for label in ("| WP ", "| ATS ", "| OU "):
            assert content.count(label) >= 2, (
                f"expected {label!r} in both the canonical and the coverage-window grid"
            )

    def test_canonical_deltas_are_recorded(self) -> None:
        """The published numbers are in the doc (the doc-drift anchor).

        Includes the pre-re-key 4b readings (+0.816061 / 0.00044), which the doc
        retains in prose as history rather than deleting, and the re-measured and
        headline numbers that superseded them.
        """
        content = _read_readout()
        for token in (
            # Pre-leak-fix history -- every one of these must STAY in the doc.
            "+0.147814",
            "-0.221188",
            "+0.816061",
            "0.00044",
            "+1.012106",
            "+0.151237",
            "+0.001429",
            # Post-leak-fix readings (quick task 260817-dyp).
            "-0.136848",
            "-0.294672",
            "+0.750368",
            "-0.208582",
            "-0.003519",
            "-0.001594",
        ):
            assert token in content, (
                f"the readout is missing the published number {token}"
            )

    def test_not_measured_disclosure_is_present(self) -> None:
        """The canonical grid MUST be labelled as not evidence about the group.

        Publishing the 4a deltas as a lift -- without saying no model ever used the columns --
        is the exact honesty-of-record failure this phase's threat register calls out.
        """
        content = _read_readout()
        assert "0 / 15" in content, (
            "Section 4a must state that 0 of 15 group columns reached the model"
        )
        assert "not evidence about line movement" in content.lower()

    def test_screen_not_deploy_ruling_language(self) -> None:
        """A KEEP is a CARRY to Phase 30, never a ship."""
        content = _read_readout()
        assert "CARRY the line-movement family to Phase 30" in content
        assert "not shipped" in content.lower()


class TestCR03Annotation:
    """Section 6 (Plan 30-03): the inert-tell finding and the DROP-as-decision record.

    D25-10 Pitfall 5 -- the doc and its guard move in the SAME commit, or the suite is left red
    in a way that reads as unrelated to whoever runs it next.
    """

    def test_cr03_annotation_is_present(self) -> None:
        """Every load-bearing phrase of the annotation is present, and named if missing."""
        content = _read_readout()
        missing = [m for m in _REQUIRED_SECTION6_MARKERS if m not in content]
        assert not missing, (
            f"LINE-MOVEMENT-READOUT.md Section 6 (the CR-03 annotation) is missing: {missing}. "
            "These are not decorative: the annotation exists so a reader cannot mistake the "
            "structurally inert confound tell for the grounds of the Phase-30 DROP, which rest "
            "on the 2018-2019 selection window instead (T-30-21)."
        )

    def test_annotation_is_appended_beside_the_published_record(self) -> None:
        """Section 6 sits AFTER every published Phase-29 reading, never among or over them.

        The append discipline is what makes a superseded number recoverable (D29-06-02). If the
        annotation ever migrates above Section 5, it has started editing the record rather than
        annotating it.
        """
        content = _read_readout()
        annotation_at = content.index("## 6. Phase-30 annotation")
        for earlier in ("### 4e-bis.", "## 5. Honest scope-down"):
            assert content.index(earlier) < annotation_at, (
                f"the Phase-30 annotation must come AFTER {earlier} -- it is appended beside "
                "the published record, not inserted into it"
            )

    def test_the_published_tell_readings_are_retained_not_rewritten(self) -> None:
        """4d / 4d-bis still say the tell did not fire; the annotation sits beside that claim.

        The failure this pins is the tempting one: having found the tell inert, rewrite the
        sections that cite it. That would destroy the record of what was believed when the
        headline was published.
        """
        content = _read_readout()
        assert "did not fire" in content, (
            "the published 'the confound tell did not fire' reading must be RETAINED -- the "
            "CR-03 annotation goes beside it, never over it"
        )
        assert "rests on a guard that, on this" in content, (
            "the annotation must state plainly that the retained claim rests on a guard which "
            "could not have fired on this gold"
        )


def _require_fixture() -> None:
    """Assert the committed pre-Phase-30 fixture is present.

    This is deliberately a FAILURE, not a ``pytest.skip``. The old guard skipped on missing
    ``data/gold`` because ``data/`` is gitignored and may genuinely be absent. The fixture is
    the opposite: it is a COMMITTED artifact of this repository, so its absence means the
    checkout is broken or somebody deleted it -- neither of which should pass silently. A skip
    here would let the whole reproduction class vanish without a sound, which is the exact
    failure mode the Phase-30 re-scope exists to eliminate.
    """
    missing = [p for p in (_FIXTURE_GOLD, _FIXTURE_ODDS) if not p.exists()]
    assert not missing, (
        f"the committed pre-Phase-30 gold fixture is missing: {[str(p) for p in missing]}. "
        "These files are tracked (see .gitignore's !tests/fixtures/gold/*.parquet negation) "
        f"and described in {_FIXTURE_PROVENANCE}. Restore them from git rather than "
        "regenerating them -- a regenerated fixture is not the gold that produced the "
        "published Phase-29 numbers."
    )


class TestFixtureIdentity:
    """The frozen fixture cannot be quietly regenerated to rescue a failing assertion."""

    def test_fixture_is_the_pre_drop_gold(self) -> None:
        """Shape, the 15-column line_movement count, and BOTH sha256 digests are pinned.

        Without this, the four reproductions below could be "fixed" after any future rebuild by
        re-capturing the fixture from whatever gold happens to be on disk -- which would make
        them reproduce a number they were never measured against and quietly destroy the only
        evidence that the published Phase-29 readings came from the committed harness.
        """
        _require_fixture()

        import pandas as pd

        from backtest.signal_lift import group_columns

        gold = pd.read_parquet(_FIXTURE_GOLD)

        regenerated = (
            "If this failed because the fixture was RE-CAPTURED from current gold, that is "
            "the failure this test exists to catch, not a reason to update the constant. The "
            f"fixture and {_FIXTURE_PROVENANCE.name} are regenerated together or not at all; "
            "a red assertion in this module is a finding about the harness or the record, "
            "never a licence to re-capture."
        )

        assert gold.shape == _FIXTURE_GOLD_SHAPE, (
            f"the fixture is {gold.shape}, not the pre-Phase-30 ATS gold "
            f"{_FIXTURE_GOLD_SHAPE}. {regenerated}"
        )
        assert (
            len(group_columns(gold, "line_movement"))
            == _FIXTURE_GOLD_LINE_MOVEMENT_COLUMNS
        ), (
            "the fixture must still carry the full 15-column line_movement family -- it is the "
            f"PRE-drop gold by definition. {regenerated}"
        )

        for path, expected in (
            (_FIXTURE_GOLD, _FIXTURE_GOLD_SHA256),
            (_FIXTURE_ODDS, _FIXTURE_ODDS_SHA256),
        ):
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == expected, (
                f"{path.name} sha256 is {actual}, not the digest recorded in "
                f"{_FIXTURE_PROVENANCE} ({expected}). {regenerated}"
            )

    def test_provenance_records_both_digests(self) -> None:
        """The module constants are transcribed FROM PROVENANCE.md, so they must agree with it.

        A digest pinned in code but absent from the provenance record is unauditable: a reader
        has no way to check what the fixture was copied from, which is the whole point of the
        record.
        """
        assert _FIXTURE_PROVENANCE.is_file(), f"missing: {_FIXTURE_PROVENANCE}"
        content = _FIXTURE_PROVENANCE.read_text(encoding="utf-8")
        assert content.isascii(), "PROVENANCE.md must stay pure ASCII (CLAUDE.md)"
        for digest in (_FIXTURE_GOLD_SHA256, _FIXTURE_ODDS_SHA256):
            assert digest in content, (
                f"digest {digest} is pinned in this module but is not recorded in "
                f"{_FIXTURE_PROVENANCE}"
            )
        assert "MATCH LINE" in content, (
            "PROVENANCE.md must print each fixture digest beside the digest of the SOURCE file "
            "it was copied from, with an explicit statement that the two were equal at capture "
            "time -- that comparison is only checkable before the rebuild destroys the source"
        )


class TestLineMovementDropInLiveGold:
    """The Phase-30 DROP proof (SPEC R3) -- what the reproductions used to say about the present.

    The four reproductions below are now scoped to a frozen artifact and deliberately say NOTHING
    about live gold. This is the test that does.
    """

    def test_line_movement_family_is_absent_from_live_gold(self) -> None:
        """Zero line_movement columns in all three live matrices, once Plan 30-07 has landed.

        Honest in BOTH states, and deliberately not a tautology:

          - while the family is still present in ALL THREE matrices, this SKIPS with a message
            naming Plan 30-07 -- that is the expected pre-drop state, not a pass;
          - once the drop has landed it ASSERTS, and a partial drop (gone from some matrices but
            not others) falls through to the assertion and goes RED rather than being swallowed
            by the skip.
        """
        import pandas as pd

        from backtest.signal_lift import group_columns

        absent = [t for t, p in _LIVE_GOLD_PATHS.items() if not p.exists()]
        if absent:
            pytest.skip(
                f"live gold not present for {absent} (data/ is gitignored); "
                "the DROP proof needs a built data lake"
            )

        per_target = {
            target: group_columns(pd.read_parquet(path), "line_movement")
            for target, path in _LIVE_GOLD_PATHS.items()
        }

        if all(cols for cols in per_target.values()):
            counts = {t: len(c) for t, c in per_target.items()}
            pytest.skip(
                f"the line_movement family is still in every live matrix ({counts}) -- this is "
                "the expected PRE-drop state. Plan 30-07 performs the SPEC R3 drop; this "
                "assertion turns live at that point and must not be softened before then."
            )

        for target, cols in per_target.items():
            assert cols == [], (
                f"features_{target}.parquet still carries {len(cols)} line_movement columns "
                f"{cols} while at least one other matrix has none. The SPEC R3 drop is "
                "PARTIAL, which is worse than not having run: the three matrices no longer "
                "agree on the candidate feature set."
            )


# The harness-reproduction class that stood here was DELETED on 2026-09-12 by owner
# instruction. It re-ran the analysis harness against live gold and asserted the
# committed point estimates still reproduced. Two defects made it undefendable:
#   1. NOT DETERMINISTIC. The situational-OU delta measured 0.0074 / 0.4424 / 0.4784
#      at 4 / 1 / 8 BLAS threads and only reproduced the committed value at 12. It
#      cannot detect drift because it drifts on its own; the quantity is a difference
#      between noisy per-season estimates, so floating-point reduction order moves it
#      further than the signal does.
#   2. Plan 33.1-08 had already concluded the same thing and designed the successor:
#      generation-gate the harness half so it SKIPS when gold moves. See that plan and
#      D29-06-02 -- 'pinning a point estimate to moving gold is the mistake this guard
#      class made once'.
# The DOC-level assertions in this file are untouched and still run unconditionally.


class TestPreRegistrationOrdering:
    """The D-Q4 claim is checked from git, not asserted in prose.

    The whole value of a pre-registration is that the rule existed before the
    numbers did. That is only worth something if it is verifiable, so the readout
    carries the registering commits as machine-readable SHAs and this test
    resolves them: each must be an ancestor of HEAD, and neither commit's own diff
    may contain a headline number.
    """

    @staticmethod
    def _git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def _marker_shas() -> list[str]:
        return _PRE_REGISTRATION_MARKER_RE.findall(_read_readout())

    def test_both_registration_markers_are_present(self) -> None:
        assert len(self._marker_shas()) == 2, (
            "Section 4d must carry both the original and the superseding "
            "pre_registration_commit markers"
        )

    def test_registration_commits_precede_head(self) -> None:
        """`git merge-base --is-ancestor` -- the ordering claim, checked."""
        if self._git("rev-parse", "--git-dir").returncode != 0:
            pytest.skip("not a git checkout")

        for sha in self._marker_shas():
            if self._git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
                pytest.skip(f"commit {sha} not available in this checkout")
            result = self._git("merge-base", "--is-ancestor", sha, "HEAD")
            assert result.returncode == 0, (
                f"pre-registration commit {sha} is NOT an ancestor of HEAD -- the "
                "rule cannot be shown to have been written before the results"
            )

    def test_registration_commits_carry_no_headline_numbers(self) -> None:
        """A registration that already knew the answer is not a registration."""
        if self._git("rev-parse", "--git-dir").returncode != 0:
            pytest.skip("not a git checkout")

        for sha in self._marker_shas():
            if self._git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
                pytest.skip(f"commit {sha} not available in this checkout")
            diff = self._git("show", sha, "--", "LINE-MOVEMENT-READOUT.md").stdout
            added = "\n".join(
                line for line in diff.splitlines() if line.startswith("+")
            )
            leaked = [token for token in _HEADLINE_DELTA_TOKENS if token in added]
            assert not leaked, (
                f"pre-registration commit {sha} already contains headline numbers "
                f"{leaked} -- it was not written before the results existed"
            )


# ---------------------------------------------------------------------------
# THE GENERATION SEAM, AND WHY THIS MODULE DOES NOT CALL IT -- Plan 33.1-08
# Task 2, 2026-09-14.
#
# Plan 33.1-08 was written to GENERATION-GATE the harness-reproduction class
# above rather than let a gold rebuild turn it red. By the time the plan ran the
# class was already gone, deleted on 2026-09-12 for a different and better
# reason: it was not deterministic, so it could not detect drift because it
# drifted on its own. A seam that refuses to compare across gold generations
# cannot help a measurement that disagrees with itself at a fixed generation.
#
# The seam was still built -- tests/gold_generation.py -- because two OTHER
# harness reproductions did redden on the rung-3 rebuild and needed it, and
# because Phase 33's Wave 14 Elo rung will hit the same wall again. It is
# recorded here so a reader of this file can find it:
#
#   tests.gold_generation.require_gold_generation(expected_key, *, reading,
#       moved_by, recorded_in) -- skips, never fails, and names all three in the
#       skip message.
#   tests.phase33_state.GOLD_GENERATION_AFTER_WEATHER_RUNG -- the live key
#       measured after the rung.
#   tests.phase33_state.GOLD_DERIVED_READINGS -- every reading and the
#       generation it was measured against, including this module's.
#
# The document-level assertions in this file read no gold and are not gated by
# anything. They run unconditionally, and they are what actually guards the
# document.
# ---------------------------------------------------------------------------
