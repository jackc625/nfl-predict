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

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import re
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

# Repo root resolved from this file: tests/unit/test_line_movement_readout_md.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
READOUT_MD = REPO_ROOT / "LINE-MOVEMENT-READOUT.md"

# Gold/odds presence skip-guard: the harness-run validation needs the Plan 29-06 rebuilt gold.
_GOLD_ATS_PATH = REPO_ROOT / "data" / "gold" / "features_ats.parquet"
_ODDS_PATH = REPO_ROOT / "data" / "silver" / "odds_snapshot.parquet"

# Required section markers (the D-16 five-section contract). A substring of each section header so
# a benign reword does not trip the guard but a dropped section does.
_REQUIRED_SECTION_MARKERS = (
    "Tiered cost",  # 1 -- the (a)/(b)/(c) x window credit+dollar menu
    "Plausibility (D-02)",  # 2 -- the non-redundancy argument
    "Decision:",  # 3 -- the owner branch decision
    "Lift results",  # 4 -- the backfill-path lift placeholder
    "Honest scope-down",  # 5 -- the scope-down / slip note
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


@pytest.mark.integration
class TestReadoutMatchesHarness:
    """The doc's numbers ARE the harness numbers -- re-run, not re-typed.

    Runs the ATS cell of BOTH published grids (the load-bearing cell in each) and asserts the
    reproduced delta matches the doc. Also pins the structural claim the whole ruling rests on:
    under the canonical window NO line-movement column is selected, and under the coverage window
    some are. If a future rebuild changes either fact, this fails and Section 4 must be rewritten.
    """

    @staticmethod
    def _run(
        coverage_window: bool = False, covered_selection_window: bool = False
    ) -> dict:
        """Run the ATS cell under one of the three published windows.

        Takes the window rather than duplicating the invocation per test, and
        goes through ``screen_kwargs_for_phase`` so the guard runs exactly what
        the CLI runs.
        """
        import warnings

        import pandas as pd

        from backtest.signal_lift import run_signal_lift_screen, screen_kwargs_for_phase

        gold = pd.read_parquet(_GOLD_ATS_PATH)
        odds = pd.read_parquet(_ODDS_PATH)
        kwargs = screen_kwargs_for_phase(
            29,
            coverage_window=coverage_window,
            covered_selection_window=covered_selection_window,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_signal_lift_screen(
                gold_by_target={"ats": gold},
                closing_odds_df=odds,
                targets=["ats"],
                **kwargs,
            )

    def test_canonical_ats_cell_reproduces_and_used_no_group_columns(self) -> None:
        if not (_GOLD_ATS_PATH.exists() and _ODDS_PATH.exists()):
            pytest.skip(
                f"Canonical gold/odds not present at {_GOLD_ATS_PATH} / {_ODDS_PATH}"
            )

        cell = self._run(coverage_window=False)["groups"]["line_movement"][
            "per_target"
        ]["ats"]

        assert cell["delta_mean"] is not None
        # WR-13: the message INTERPOLATES the module constant. It used to compare
        # against _CANONICAL_DELTAS["ats"] (-0.136848) while printing "+0.147814"
        # -- a superseded number -- so the one artefact a maintainer reads when
        # the guard fires named a value the guard was not checking.
        assert abs(cell["delta_mean"] - _CANONICAL_DELTAS["ats"]) < 5e-3, (
            f"canonical ATS delta {cell['delta_mean']} drifted from the readout's "
            f"{_CANONICAL_DELTAS['ats']:+f}"
        )
        # The load-bearing claim: the canonical window cannot select the family at all.
        assert cell["n_group_columns"] == 15
        assert cell["n_group_columns_selected"] == 0, (
            "a line-movement column is now selectable under the canonical window -- Section 4a's "
            "'not evidence' framing no longer holds and must be rewritten"
        )
        assert cell["measurable"] is False
        assert "+0.147814" in _read_readout()

    def test_coverage_window_ats_cell_reproduces_and_used_group_columns(self) -> None:
        if not (_GOLD_ATS_PATH.exists() and _ODDS_PATH.exists()):
            pytest.skip(
                f"Canonical gold/odds not present at {_GOLD_ATS_PATH} / {_ODDS_PATH}"
            )

        result = self._run(coverage_window=True)
        cell = result["groups"]["line_movement"]["per_target"]["ats"]

        assert result["measure_window"] == "2024", (
            "the diagnostic measures one season and must be labelled as such"
        )
        assert cell["delta_mean"] is not None
        assert abs(cell["delta_mean"] - _COVERAGE_WINDOW_ATS_DELTA) < 5e-3, (
            f"coverage-window ATS delta {cell['delta_mean']} drifted from the "
            f"readout's re-measured +{_COVERAGE_WINDOW_ATS_DELTA}"
        )
        assert cell["n_group_columns_selected"] > 0, (
            "the diagnostic window is only meaningful if the family is actually selected"
        )
        assert cell["measurable"] is True
        assert "+1.012106" in _read_readout()
        # The pre-re-key reading is retained as history and must not be dropped.
        assert "+0.816061" in _read_readout()

    def test_headline_ats_cell_reproduces_from_the_covered_selection_window(
        self,
    ) -> None:
        """The 4d headline is the harness's number, re-run rather than re-typed."""
        if not (_GOLD_ATS_PATH.exists() and _ODDS_PATH.exists()):
            pytest.skip(
                f"Canonical gold/odds not present at {_GOLD_ATS_PATH} / {_ODDS_PATH}"
            )

        result = self._run(covered_selection_window=True)
        cell = result["groups"]["line_movement"]["per_target"]["ats"]

        assert result["measure_window"] == _HEADLINE_MEASURE_WINDOW, (
            "the headline measures three seasons and must be labelled as such"
        )
        assert cell["delta_mean"] is not None
        # WR-13: `+{constant}` printed "+-0.208582" for the negative headline.
        # The corrected result being NEGATIVE is the phase's finding, so a failure
        # message that mangles its sign is the wrong place to be sloppy.
        assert abs(cell["delta_mean"] - _HEADLINE_ATS_DELTA) < 5e-3, (
            f"headline ATS delta {cell['delta_mean']} drifted from the readout's "
            f"{_HEADLINE_ATS_DELTA:+f}"
        )
        # The measurability precondition the whole grid's meaning rests on: under
        # this window the family IS selected, so the cell is evidence about line
        # movement rather than selection churn.
        assert cell["n_group_columns"] == 15
        assert cell["n_group_columns_selected"] == _HEADLINE_ATS_GROUP_COLS_SELECTED, (
            f"the readout publishes {_HEADLINE_ATS_GROUP_COLS_SELECTED}/15 group "
            f"columns used for ATS; the harness now reports "
            f"{cell['n_group_columns_selected']}"
        )
        assert cell["measurable"] is True
        content = _read_readout()
        assert "+0.151237" in content
        assert f"{_HEADLINE_ATS_GROUP_COLS_SELECTED} / 15" in content

    def test_headline_confound_tell_is_applied_as_pre_registered(self) -> None:
        """``line_movement_coverage`` selected => the doc must call the cell confounded.

        4c-bis item 6 registered this tell BEFORE the run. The doc currently
        reports that it did not fire; if a rebuild makes it fire, that claim
        becomes false and Section 4d has to be rewritten rather than silently
        going stale.
        """
        if not (_GOLD_ATS_PATH.exists() and _ODDS_PATH.exists()):
            pytest.skip(
                f"Canonical gold/odds not present at {_GOLD_ATS_PATH} / {_ODDS_PATH}"
            )

        cell = self._run(covered_selection_window=True)["groups"]["line_movement"][
            "per_target"
        ]["ats"]
        fired = "line_movement_coverage" in cell["group_columns_selected"]
        assert not fired, (
            "line_movement_coverage is now selected under the covered window -- per "
            "the pre-registered tell that cell is CONFOUNDED (a season proxy, not "
            "market information) and Section 4d must say so"
        )
        assert "did not fire" in _read_readout()


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
