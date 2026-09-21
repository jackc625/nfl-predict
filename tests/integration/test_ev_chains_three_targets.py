"""All three EV chains run end to end through ONE registry, on a burned season.

WHY THIS RUNS ON THE REGISTERED PROXY SPLIT AND NOT ON A HOLD-ONLY OVERRIDE
---------------------------------------------------------------------------
SPEC R1 asks for an end-to-end run of all three chains. It cannot be exercised on 2025: that
split is single-use and Plan 31-14 spends it. So the run uses ``REHEARSAL_PROXY_SPLIT`` --
tune 2021-2023, hold 2024 -- which is DISJOINT on both sides.

Overriding only the hold to 2024 while leaving the frozen tune window at ``TUNE_SEASONS_P31``
(2021-2024) would make the hold season a TUNE season. The Phase-31 fence rejects a hold season
consumed by the SD fit or by threshold tuning, so such a run would either FAIL, or would only
pass against a fence weakened enough to permit real leakage -- in the module whose whole job is
to refuse leakage. The proxy split narrows the TUNE side too, so the fence runs at FULL
strength and passing it is EVIDENCE rather than silence.

That the fence was live is asserted directly, not assumed: the run's fence report must name
2024 as the hold and 2021-2023 as the tune, 2025 must appear nowhere in it, and a deliberate
variant that puts 2024 back into the tune set must raise ``LeakageError``.

THIS MODULE IS NOT A UNIT TEST. It reads real gold, the real silver odds lake and the deployed
model artifacts, and scores three deployed artifacts over seven seasons. It is skipped -- with
a reason registered in ``tests/conftest.py``'s evidence-backed skip markers -- on a checkout
that does not carry those, so a green suite elsewhere cannot silently exclude it without
saying so.

NOTHING HERE READS 2025. The rehearsal window does not name it, and its result is DISCARDED:
never reported, never published, never compared.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from backtest.ats_ev_chain import (
    FENCE_WINDOW_P31,
    FENCE_WINDOW_REHEARSAL,
    ChainFit,
    LeakageError,
    assert_fit_window_p31,
)
from backtest.ev_chain_constants import (
    REHEARSAL_PROXY_SPLIT,
    TRIAL_ENTRY_KIND_CONTROL,
    TRIAL_ENTRY_KIND_INFERENCE,
    VERDICT_TOKENS,
)
from backtest.profitability_2025 import (
    CANONICAL_TARGETS,
    UNDISCHARGEABLE_NO_BETS,
    ProxySplitRefusedError,
    run_armed_2025_verdict,
    run_profitability_2025,
)
from tests.p31_synthetic_candidates import build_synthetic_candidates

_GOLD_PATHS = tuple(
    Path("data") / "gold" / f"features_{target}.parquet" for target in CANONICAL_TARGETS
)
_ODDS_PATH = Path("data") / "silver" / "odds_snapshot.parquet"
_ARTIFACT_MANIFEST = Path("artifacts") / "latest.json"


def _require_inputs() -> None:
    """Skip with a REGISTERED reason when the real inputs are absent.

    Deliberately narrow: it guards on the ABSENCE OF INPUTS only. Widening it to swallow an
    exception from the run itself would turn a genuine leakage or wiring finding into a green
    skip, which is the one outcome this module exists to prevent.
    """
    missing = [
        path
        for path in (*_GOLD_PATHS, _ODDS_PATH, _ARTIFACT_MANIFEST)
        if not path.exists()
    ]
    if missing:
        pytest.skip(
            "the real gold, silver odds and deployed artifacts are not present at "
            f"{[str(path) for path in missing]}; the three-target end-to-end run is not "
            "derivable on this checkout (see PIPELINE.md)."
        )


@pytest.fixture(scope="module")
def real_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """ONE full run over the REGISTERED proxy split, through the REAL loaders.

    Module-scoped: it scores three deployed artifacts over 2018-2024 gold, sweeps a
    fifteen-cell EV-floor grid, selects a pooled three-target hold week by week and runs three
    counterfactual passes. Every assertion below reads this one run.
    """
    _require_inputs()
    tmp_path = tmp_path_factory.mktemp("p31_three_target_run")
    return {
        "result": run_profitability_2025(
            FENCE_WINDOW_REHEARSAL,
            verdict_toml_path=tmp_path / "rehearsal_verdict.toml",
            verdict_json_path=tmp_path / "rehearsal_verdict.json",
            ledger_path=tmp_path / "rehearsal_ledger.toml",
        ),
        "verdict_toml_path": tmp_path / "rehearsal_verdict.toml",
    }


@pytest.mark.integration
class TestAllThreeChainsRunEndToEndThroughOneRegistry:
    """SPEC R1's end-to-end acceptance, exercised on a burned season."""

    def test_each_target_emits_a_decision_set(self, real_run) -> None:
        targets = real_run["result"]["targets"]
        assert set(targets) == set(CANONICAL_TARGETS)
        assert list(targets) == list(CANONICAL_TARGETS), (
            "the per-target records are not in the repository's canonical order."
        )
        for target, record in targets.items():
            assert record["verdict_token"] in VERDICT_TOKENS, (target, record)
            assert record["chain_resolved"] is True, (
                f"the {target} chain did not resolve on real 2024 data at all; a required "
                "input was absent for every candidate."
            )
            assert record["control_passed"] is True, (
                f"the {target} shifted-edge counterfactual pass over the ACTUAL hold frame "
                "selected ZERO bets. A zero that a manufactured positive edge cannot move is "
                "an unexplained zero, not evidence of no edge (D31-08)."
            )

    def test_there_is_exactly_one_registry_spanning_all_three_targets(
        self, real_run
    ) -> None:
        registry = real_run["result"]["registry"]
        assert {entry["target"] for entry in registry} == set(CANONICAL_TARGETS)
        assert len({entry["entry_id"] for entry in registry}) == len(registry), (
            "two registry entries share an id, so the correction cannot be reproduced from it."
        )
        kinds = {entry["entry_kind"] for entry in registry}
        assert kinds == {TRIAL_ENTRY_KIND_INFERENCE, TRIAL_ENTRY_KIND_CONTROL}

    def test_the_denominator_is_the_pre_registered_enumeration(self, real_run) -> None:
        run = real_run["result"]["run"]
        assert run["bh_denominator"] == 6 + run["bh_fallbacks_fired"]
        assert (
            run["registry_rows"]
            == run["bh_denominator"]
            + run["excluded_control_entries"]
            + run["excluded_tune_side_sweep_cells"]
        ), run

    def test_the_run_leaves_the_production_stores_untouched(
        self, real_run, data_boundary_guard
    ) -> None:
        """The HARD BOUNDARY, claimed by CONTENT DIGEST and never by ``git status data/``.

        ``.gitignore`` blankets ``data/``, so the git check is structurally incapable of
        failing; it returns empty whether the archive is intact or destroyed. The fixture
        hashes file contents before and after instead.
        """
        assert real_run["result"]["run"]["bh_denominator"] >= 6


@pytest.mark.integration
class TestTheFenceWasLiveAndNamedTheRightWindow:
    """Passing a fence is evidence only if the fence could have failed."""

    def test_the_fence_report_names_2024_as_the_hold_and_2021_2023_as_the_tune(
        self, real_run
    ) -> None:
        reports = real_run["result"]["fence_reports"]
        assert set(reports) == set(CANONICAL_TARGETS)
        for target, report in reports.items():
            assert report["hold_seasons"] == [2024], target
            assert report["tune_seasons"] == [2021, 2022, 2023], target
            assert report["prior_residual_seed_seasons"] == [2018, 2019, 2020], target
            assert report["fence_held"] is True, target
            assert report["window_is_the_preregistered_rule"] is False, target

    def test_2025_appears_nowhere_in_the_fence_report(self, real_run) -> None:
        for target, report in real_run["result"]["fence_reports"].items():
            seasons = (
                list(report["hold_seasons"])
                + list(report["tune_seasons"])
                + list(report["prior_residual_seed_seasons"])
                + list(report["bias_seasons"])
                + list(report["tune_fit_seasons"])
            )
            assert 2025 not in seasons, (target, seasons)
            assert "2025" not in report["threshold_window"], target

    def test_the_rehearsal_split_is_the_registered_disjoint_one(self) -> None:
        assert tuple(REHEARSAL_PROXY_SPLIT["tune_seasons"]) == (2021, 2022, 2023)
        assert tuple(REHEARSAL_PROXY_SPLIT["hold_seasons"]) == (2024,)
        assert FENCE_WINDOW_REHEARSAL.tune_seasons == (2021, 2022, 2023)
        assert FENCE_WINDOW_REHEARSAL.hold_seasons == (2024,)
        assert not set(FENCE_WINDOW_REHEARSAL.tune_seasons) & set(
            FENCE_WINDOW_REHEARSAL.hold_seasons
        )

    def test_putting_2024_back_into_the_tune_set_raises_leakage_error(self) -> None:
        """The proof the fence is LIVE rather than merely silent."""
        leaky = ChainFit(
            target="ats",
            frozen_sd=13.0,
            season_bias_by_season={2024: 0.5},
            # 2024 is the rehearsal HOLD. A fit that consumed it is the exact leak.
            tune_fit_seasons=(2021, 2022, 2023, 2024),
            threshold_window=FENCE_WINDOW_REHEARSAL.threshold_window,
            bias_pool_by_season={2024: (2021, 2022, 2023)},
        )
        with pytest.raises(LeakageError, match=r"HOLD season\(s\) \[2024\]"):
            assert_fit_window_p31(leaky, FENCE_WINDOW_REHEARSAL)

        # THE SAME LEAK, judged by the FROZEN window, is SILENT -- which is precisely why a
        # hold-only override would have proved nothing. 2024 is a frozen TUNE season, so a fit
        # that consumed it raises nothing there, and the report names a hold the rehearsal
        # never ran on. Only the threshold-window label differs below, because check (b)
        # compares it against the window being fenced.
        same_leak_frozen_label = ChainFit(
            target=leaky.target,
            frozen_sd=leaky.frozen_sd,
            season_bias_by_season=leaky.season_bias_by_season,
            tune_fit_seasons=leaky.tune_fit_seasons,
            threshold_window=FENCE_WINDOW_P31.threshold_window,
            bias_pool_by_season=leaky.bias_pool_by_season,
        )
        report = assert_fit_window_p31(same_leak_frozen_label, FENCE_WINDOW_P31)
        assert report["fence_held"] is True
        assert report["hold_seasons"] == [2025], (
            "the frozen window accepted a fit that consumed 2024 and named 2025 as its hold. "
            "That is the silence the disjoint proxy split exists to avoid: the fence passes "
            "while reporting a window the rehearsal never ran on."
        )
        assert 2024 in report["tune_fit_seasons"]

    def test_a_threshold_window_from_the_wrong_split_raises(self) -> None:
        mismatched = ChainFit(
            target="ou",
            frozen_sd=11.0,
            season_bias_by_season={2024: -0.4},
            tune_fit_seasons=(2021, 2022, 2023),
            threshold_window=FENCE_WINDOW_P31.threshold_window,
            bias_pool_by_season={2024: (2021, 2022, 2023)},
        )
        with pytest.raises(LeakageError, match="threshold"):
            assert_fit_window_p31(mismatched, FENCE_WINDOW_REHEARSAL)


@pytest.mark.integration
class TestTheArmedPathCannotConsumeTheProxySplit:
    """A rehearsal configuration must never leak into a binding run."""

    def test_the_armed_entry_point_takes_no_split(self) -> None:
        with pytest.raises(ProxySplitRefusedError, match="accepts NO arguments"):
            run_armed_2025_verdict(split=REHEARSAL_PROXY_SPLIT)
        with pytest.raises(ProxySplitRefusedError):
            run_armed_2025_verdict(window=FENCE_WINDOW_REHEARSAL)

    def test_the_frozen_split_declares_the_proxy_is_not_consumable(self) -> None:
        assert REHEARSAL_PROXY_SPLIT["is_the_preregistered_rule"] is False
        assert REHEARSAL_PROXY_SPLIT["consumable_by_the_armed_run"] is False
        assert "DISCARDED" in str(REHEARSAL_PROXY_SPLIT["result_disposition"])

    def test_a_rehearsal_may_not_write_to_a_production_path(self, tmp_path) -> None:
        with pytest.raises(ProxySplitRefusedError, match="PRODUCTION path"):
            run_profitability_2025(
                FENCE_WINDOW_REHEARSAL,
                verdict_toml_path="config/profitability_2025_verdict.toml",
                verdict_json_path=tmp_path / "v.json",
                ledger_path=tmp_path / "l.toml",
            )


@pytest.mark.integration
class TestAZeroBetTargetIsStillOnTheRecord:
    """A chain that emits zero bets does not stop the run, and is not an absence.

    THE ZERO IS CONSTRUCTED, and that is stated rather than hidden. Real 2024 data happens to
    produce bets for all three targets, and a case that never occurs cannot be asserted about.
    So one target's model output is set EXACTLY equal to its market number, which puts every
    candidate inside the LOCKED no-bet band and yields ``no_bet_side`` for all of them -- a
    genuine zero-bet chain driven through the same runner, the same registry and the same
    correction as any other.
    """

    @pytest.fixture(scope="class")
    def zero_bet_run(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        _require_inputs()
        frames = build_synthetic_candidates()
        silent = frames["ou"].copy()
        silent["model_total"] = silent["closing_total"].astype(float)
        frames = {**frames, "ou": silent}

        tmp_path = tmp_path_factory.mktemp("p31_zero_bet_run")
        return run_profitability_2025(
            FENCE_WINDOW_REHEARSAL,
            verdict_toml_path=tmp_path / "verdict.toml",
            verdict_json_path=tmp_path / "verdict.json",
            ledger_path=tmp_path / "ledger.toml",
            candidates_by_target=frames,
        )

    def test_the_silent_target_selected_zero_bets(self, zero_bet_run) -> None:
        assert zero_bet_run["targets"]["ou"]["bets_selected"] == 0

    def test_the_other_two_targets_still_produced_verdicts(self, zero_bet_run) -> None:
        for target in ("wp", "ats"):
            record = zero_bet_run["targets"][target]
            assert record["verdict_token"] in VERDICT_TOKENS
            assert record["bets_selected"] > 0, (
                f"{target} produced no bets either, so this fixture does not demonstrate that "
                "ONE silent chain leaves the others running."
            )

    def test_the_zero_bet_target_still_carries_a_token_and_a_registry_row(
        self, zero_bet_run
    ) -> None:
        record = zero_bet_run["targets"]["ou"]
        assert record["verdict_token"] == UNDISCHARGEABLE_NO_BETS
        assert record["control_passed"] is True
        assert record["has_return"] is False
        assert record["flat_roi"] is None, (
            "a zero-bet target reported a return. A return of zero is a measured break-even "
            "and no bets is not."
        )
        rows = [entry for entry in zero_bet_run["registry"] if entry["target"] == "ou"]
        assert rows, "the zero-bet target vanished from the registry entirely."
        assert any(entry["entry_id"] == "ou/primary" for entry in rows)
        assert any(entry["entry_kind"] == TRIAL_ENTRY_KIND_CONTROL for entry in rows), (
            "the zero-bet target's counterfactual control is not on the record."
        )

    def test_the_denominator_still_counts_the_silent_target(self, zero_bet_run) -> None:
        """The family is ENUMERATED in advance; a silent target does not shrink it."""
        run = zero_bet_run["run"]
        assert run["bh_denominator"] == 6 + run["bh_fallbacks_fired"], (
            "the silent target's entries left the family, which would quietly make the "
            "surviving targets' tests easier because another target had no bets."
        )
