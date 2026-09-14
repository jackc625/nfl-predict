"""The drift tripwire is fed the FROZEN frame, proven by driving the real promotion path.

WHY THIS IS AN INTEGRATION TEST AND NOT A UNIT TEST
----------------------------------------------------
Because the defect is in WHICH FRAME the production path supplies, and a unit test cannot see
that.

A unit test would hand ``_drift_tripwire`` a hand-built 2021-2024 frame and assert it behaves.
It would pass. Production would still hard-abort -- deterministically, on every promotion run --
because ``scripts/promote_models`` built its frame with ``_load_gold_holdout``, which slices
gold to the LIVE holdout bounds. Once Plan 33.1-09 moved those bounds to 2024-2025, seasons
2021, 2022 and 2023 arrived at the tripwire with ZERO rows, against a non-None frozen per-season
``n`` compared as an EXACT integer. The check raises, every time, and the green unit test says
nothing about it.

So this module calls the REAL functions, in the REAL order the runner calls them --
``_load_gold_holdout``, ``_load_drift_reproduction_frame``, ``_score_baseline_clv``,
``_drift_tripwire`` -- on the REAL production gold and the REAL deployed artifacts, and asserts
what reaches the tripwire. It carries the NEGATIVE CONTROL too: passing the LIVE frame must
still abort, so the module proves the original defect was real rather than only that the new
path works.

WHAT THIS MODULE DELIBERATELY DOES NOT ASSERT
----------------------------------------------
That every target's frozen baseline still reproduces. It does not, and that is a PRESERVED
DISCLOSURE rather than a defect for this plan to fix: ``config/gate.toml``'s ``[baseline.*]``
block diverges from a re-score in 47 of 68 fields, D33-11 and D33.1-05 both refuse to re-freeze
it, and ``tests/phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS`` carries two tripwires that are RED
because of it.

MEASURED 2026-09-14 on the post-weather-rung gold, driving the real path:

  * WP  -- the frozen record STILL reproduces on its own seasons. Per-season sample sizes match
    the committed ``n`` exactly (272 / 271 / 272 / 272) and every per-season mean is inside the
    5e-3 recomputation band. WP is therefore the target on which "the tripwire does not raise"
    is a statement about the FRAME, which is what this plan changed.
  * ATS -- aborts on a POOLED MEAN drift (+0.0414 re-scored against -0.0015 frozen) on BOTH
    frames. Nothing to do with the frame.
  * O/U -- the same shape, larger (+3.136 against +1.099 frozen).

Under the owner's standing ruling of 2026-09-14 the pre-correction inputs were wrong, so an
ATS or O/U re-score on CORRECTED gold failing to reproduce a baseline computed on the corrupted
inputs is the expected outcome, not a new finding. It is recorded here so a future reader does
not mistake this module's target-specific assertions for an oversight.

The frame-shape assertions -- the load-bearing ones -- run on ALL THREE targets.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models import deploy_gate

REPO_ROOT = Path(__file__).resolve().parents[2]

_TARGETS = ("wp", "ats", "ou")

#: The one target whose frozen record still reproduces on its own seasons, so the "does not
#: raise" assertion is about the frame rather than about the preserved 47-of-68 divergence.
#: See the module docstring for the measurement.
_REPRODUCING_TARGET = "wp"

_GOLD_PATHS = {
    target: REPO_ROOT / "data" / "gold" / f"features_{target}.parquet"
    for target in _TARGETS
}
_ODDS_PATH = REPO_ROOT / "data" / "silver" / "odds_snapshot.parquet"
_MANIFEST_PATH = REPO_ROOT / "artifacts" / "latest.json"

_MISSING_GOLD_REASON = (
    "Canonical gold is not present. Build it with: "
    "uv run python scripts/build_features.py --all-seasons "
    "(see PIPELINE.md). This test drives the REAL promotion path and has nothing "
    "to assert without the real frames."
)
_MISSING_ARTIFACTS_REASON = (
    "The production artifacts manifest is not present at artifacts/latest.json, so the "
    "deployed incumbent cannot be re-scored. Restore it per RUNBOOK's clean-checkout step."
)

pytestmark = pytest.mark.integration


def _fixtures_present() -> bool:
    """True when the real gold, odds and deployed artifacts this module needs all exist."""
    return (
        all(path.exists() for path in _GOLD_PATHS.values())
        and _ODDS_PATH.exists()
        and _MANIFEST_PATH.exists()
    )


@pytest.fixture(scope="module")
def real_path_frames() -> dict[str, dict[str, object]]:
    """Drive the REAL legacy-runner call sequence once, and hand back what it produced.

    Module-scoped because each target costs one gold read plus two artifact re-scores, and
    the point of the module is that these are the production functions rather than fixtures.

    Returns:
        ``{target: {"live": frame, "frozen": frame, "live_valid": frame,
        "frozen_valid": frame}}`` -- the two gold frames and their two re-scored per-game CLV
        frames, exactly as ``scripts/promote_models.main`` builds them.
    """
    if not _fixtures_present():
        pytest.skip(
            _MISSING_GOLD_REASON
            if not _GOLD_PATHS["wp"].exists()
            else _MISSING_ARTIFACTS_REASON
        )

    from backtest.engine import BacktestEngine
    from scripts import promote_models

    engine = BacktestEngine()
    odds_df = engine._load_closing_odds()
    artifacts_dir = REPO_ROOT / "artifacts"

    built: dict[str, dict[str, object]] = {}
    for target in _TARGETS:
        live = promote_models._load_gold_holdout(target, engine)
        frozen = promote_models._load_drift_reproduction_frame(target, engine)
        built[target] = {
            "live": live,
            "frozen": frozen,
            "live_valid": promote_models._score_baseline_clv(
                target, live, odds_df, artifacts_dir
            ),
            "frozen_valid": promote_models._score_baseline_clv(
                target, frozen, odds_df, artifacts_dir
            ),
        }
    return built


class TestTheTwoFramesAreDistinctAndEachCarriesItsOwnSeasons:
    """The load-bearing assertions, on all three targets."""

    @pytest.mark.parametrize("target", _TARGETS)
    def test_the_drift_frame_covers_exactly_the_frozen_baseline_seasons(
        self, target: str, real_path_frames: dict[str, dict[str, object]]
    ) -> None:
        """``_load_drift_reproduction_frame``'s season set EQUALS FROZEN_BASELINE_SEASONS."""
        frozen = real_path_frames[target]["frozen"]
        seasons = tuple(sorted(int(s) for s in frozen["season"].unique()))
        assert seasons == tuple(deploy_gate.FROZEN_BASELINE_SEASONS), (
            f"the drift-reproduction frame for '{target}' covers {list(seasons)}, not the "
            f"frozen baseline seasons {list(deploy_gate.FROZEN_BASELINE_SEASONS)}. The "
            "tripwire asks whether the frozen record reproduces on the seasons it was "
            "frozen over; it can only ask that on those seasons' rows."
        )

    @pytest.mark.parametrize("target", _TARGETS)
    def test_the_live_frame_carries_the_live_partition_including_2025(
        self, target: str, real_path_frames: dict[str, dict[str, object]]
    ) -> None:
        """``_load_gold_holdout``'s season set EQUALS the live holdout, 2025 included.

        2025 in this frame is the whole of SPEC R6's point: ``backtest/engine.py``'s
        ``max_backtest_season`` used to drop it from every consumer that loaded gold through
        the engine, so this frame could not have carried it however the holdout was written.
        """
        live = real_path_frames[target]["live"]
        seasons = tuple(sorted(int(s) for s in live["season"].unique()))
        assert seasons == tuple(deploy_gate.HOLDOUT_SEASONS), (
            f"the live holdout frame for '{target}' covers {list(seasons)}, not the live "
            f"partition {list(deploy_gate.HOLDOUT_SEASONS)}."
        )
        assert 2024 in seasons and 2025 in seasons, (
            f"the candidate-side frame for '{target}' must carry 2024 AND 2025; it carries "
            f"{list(seasons)}."
        )

    @pytest.mark.parametrize("target", _TARGETS)
    def test_the_two_frames_are_not_the_same_population(
        self, target: str, real_path_frames: dict[str, dict[str, object]]
    ) -> None:
        """Asserted rather than assumed: reusing one frame for both is the defect."""
        live = real_path_frames[target]["live"]
        frozen = real_path_frames[target]["frozen"]
        live_seasons = {int(s) for s in live["season"].unique()}
        frozen_seasons = {int(s) for s in frozen["season"].unique()}
        assert live_seasons != frozen_seasons

    @pytest.mark.parametrize("target", _TARGETS)
    def test_every_frozen_season_reaches_the_tripwire_with_a_NON_ZERO_row_count(
        self, target: str, real_path_frames: dict[str, dict[str, object]]
    ) -> None:
        """The exact fact the abort turned on, measured on the frame the runner supplies."""
        frozen_valid = real_path_frames[target]["frozen_valid"]
        counts = frozen_valid.groupby("season").size().to_dict()
        for season in deploy_gate.FROZEN_BASELINE_SEASONS:
            assert int(counts.get(season, 0)) > 0, (
                f"season {season} reaches _drift_tripwire for '{target}' with ZERO rows "
                f"(counts: {counts}). The frozen block carries a per-season n compared as an "
                "EXACT integer, so a zero-row season is a guaranteed abort."
            )


class TestTheLiveFrameWouldAbort:
    """The negative control: the pre-fix path really was broken, not merely suspect."""

    @pytest.mark.parametrize("target", _TARGETS)
    def test_the_live_frame_carries_ZERO_rows_for_every_frozen_season(
        self, target: str, real_path_frames: dict[str, dict[str, object]]
    ) -> None:
        """The mechanical cause of the abort, target-independent and measured.

        The live partition (2024-2025) and the frozen baseline seasons (2021-2024) overlap in
        2024 only, so 2021, 2022 and 2023 each arrive empty. Asserted on every target because
        this fact does not depend on which target's baseline still reproduces.
        """
        live_valid = real_path_frames[target]["live_valid"]
        counts = live_valid.groupby("season").size().to_dict()
        empty = [
            int(season)
            for season in deploy_gate.FROZEN_BASELINE_SEASONS
            if int(counts.get(season, 0)) == 0
        ]
        assert empty, (
            f"the live frame for '{target}' carries rows for every frozen season "
            f"({counts}), so the abort this plan fixed would not reproduce. That can only "
            "happen if the live and frozen windows have been collapsed back together."
        )

    def test_passing_the_live_frame_RAISES_naming_a_zero_sample_size(
        self, real_path_frames: dict[str, dict[str, object]]
    ) -> None:
        """The real ``_drift_tripwire``, the real frozen config, the WRONG frame.

        Asserted on WP specifically, and the reason is recorded rather than left implicit:
        WP is the one target whose frozen record still reproduces within tolerance on its own
        seasons (see the module docstring), so WP is the only target where the tripwire gets
        as far as the per-season sample-size check. ATS and O/U abort earlier, on the POOLED
        MEAN, on BOTH frames -- that is the preserved 47-of-68-field divergence, not this
        defect, and asserting a zero-sample-size message for them would be asserting something
        that is not true.
        """
        from models import deploy_gate as gate
        from scripts import promote_models

        cfg = gate.load_gate_config()
        live_valid = real_path_frames[_REPRODUCING_TARGET]["live_valid"]
        with pytest.raises(ValueError, match=r"sample size 0"):
            promote_models._drift_tripwire(_REPRODUCING_TARGET, live_valid, cfg)

    def test_passing_the_frozen_frame_does_NOT_raise(
        self, real_path_frames: dict[str, dict[str, object]]
    ) -> None:
        """The same call, the same config, the RIGHT frame: it passes.

        Paired with the test above, this is what isolates the frame as the cause. Same target,
        same tripwire, same frozen config -- only the frame differs, and only one of the two
        raises.
        """
        from models import deploy_gate as gate
        from scripts import promote_models

        cfg = gate.load_gate_config()
        frozen_valid = real_path_frames[_REPRODUCING_TARGET]["frozen_valid"]
        promote_models._drift_tripwire(_REPRODUCING_TARGET, frozen_valid, cfg)


class TestThePromotionPathWiresTheFramesToTheRightConsumers:
    """A source-level check, because the frames above prove only what THEY are.

    The frames could be correct and the runner could still pass the wrong one. This reads the
    module source and asserts the tripwire call site names the drift frame, mirroring the plan's
    ``TRIPWIRE_ARGS`` scan so the guarantee survives a future refactor that moves the frames
    around.
    """

    def test_the_tripwire_call_site_does_not_receive_a_gold_holdout_variable(
        self,
    ) -> None:
        import inspect

        from scripts import promote_models

        source = [
            line
            for line in inspect.getsource(promote_models).splitlines()
            if not line.lstrip().startswith("#")
        ]
        call_sites = [
            line.strip()
            for line in source
            if "_drift_tripwire(" in line and "def " not in line
        ]
        assert call_sites, (
            "no _drift_tripwire call site found in scripts/promote_models"
        )
        for site in call_sites:
            assert "gold_holdout" not in site, (
                f"a _drift_tripwire call site passes a gold_holdout-named frame: {site!r}. "
                "That is the LIVE frame, and handing it to the tripwire is the deterministic "
                "abort this module exists to prevent."
            )

    def test_the_two_loaders_are_separate_functions(self) -> None:
        from scripts import promote_models

        assert callable(promote_models._load_gold_holdout)
        assert callable(promote_models._load_drift_reproduction_frame)
        assert (
            promote_models._load_gold_holdout
            is not promote_models._load_drift_reproduction_frame
        )
