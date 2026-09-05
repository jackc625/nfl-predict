"""Integration test for Phase 5 feature lift validation.

Validates that new features (FEAT-12 through FEAT-21) do not degrade CLV
compared to the Phase 4 baseline. Per D-06, features are validated in
4 incremental logical groups, each building on the previous:

  Cycle 1 (baseline -> +QB):
    Add: home_qb_adjustment, away_qb_adjustment
  Cycle 2 (baseline+QB -> +PBP-derived):
    Add: rolling_cpoe, rolling_avg_drive_start_yardline, rolling_neutral_pace
  Cycle 3 (baseline+QB+PBP -> +opponent-adjusted):
    Add: rolling_opp_adj_epa_per_play, etc. (replacing raw EPA)
  Cycle 4 (baseline+QB+PBP+opp_adj -> +contextual):
    Add: season_progress, late_season, surface_mismatch, is_divisional

The final test_all_features_clv_not_degraded serves as the combined gate.

Run with: uv run pytest tests/integration/test_lift_validation.py -x -v --timeout=1200
Note: Each group retrain takes ~2-5 minutes. Full suite ~15-25 minutes.
"""

import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

logger = logging.getLogger(__name__)

# Mark as slow/integration
pytestmark = [pytest.mark.integration, pytest.mark.slow]

ARTIFACTS_DIR = Path("artifacts")
GOLD_DIR = Path("data/gold")
DEGRADATION_THRESHOLD = 0.005  # 0.5% allowed noise


# ---------------------------------------------------------------------------
# Plan 31-11: this module must never retrain into the PRODUCTION artifacts tree
# ---------------------------------------------------------------------------
#
# WHAT IT USED TO DO. ``TestLiftValidation``'s autouse class fixture called the snapshot
# helper, and each of the four group tests called the training helper -- both with no
# artifacts directory at all, so both defaulted to ``ARTIFACTS_DIR``, i.e. the real
# ``artifacts/``. A successful run therefore wrote new model directories
# into production and REWROTE ``artifacts/latest.json``, which is the deployed-model
# manifest read by the API, the backtest engine and the deploy gate. A retrain started
# by ``pytest tests/integration`` would have silently swapped the models this milestone
# is measuring a profitability verdict against.
#
# IT RAN. ``artifacts/_phase4_baseline_backup/`` carries mtime 2026-09-04 22:06, inside
# Plan 31-10's post-merge suite. ``artifacts/latest.json`` survived only because the
# training subprocess exited non-zero before it could be rewritten. That is luck, not a
# control.
#
# THE FIX IS FAIL-CLOSED, not a skip. The helpers now REFUSE the production tree, in the
# shape this repository already uses for the same class of mistake
# (``backtest.group_gate._reject_data_path``, T-30-14): the refusal fires BEFORE any
# work, so it cannot arrive after the damage. The class runs against a sandbox copy of
# the production artifacts, so every assertion it makes is unchanged -- the baseline CLV
# it reads is byte-identical to production's, because it is a copy of production's.


class ProductionArtifactsWriteRefused(ValueError):
    """A helper that WRITES was pointed at the production artifacts tree."""


def reject_production_artifacts_dir(artifacts_dir: Path) -> Path:
    """Return *artifacts_dir* resolved, or refuse if it is the production tree.

    Called by every helper in this module that writes. Read-only helpers are free to
    look at production; it is writing to it that swaps deployed models.
    """
    resolved = Path(artifacts_dir).resolve()
    if resolved == ARTIFACTS_DIR.resolve():
        raise ProductionArtifactsWriteRefused(
            f"refusing to write the PRODUCTION artifacts tree at {resolved}. "
            "artifacts/latest.json is the deployed-model manifest -- rewriting it from "
            "a test swaps the models the API, the backtest engine and the deploy gate "
            "all read. Pass a sandbox directory (see the sandbox_artifacts fixture) "
            "instead."
        )
    return resolved


def load_baseline_clv(
    artifacts_dir: Path = ARTIFACTS_DIR,
) -> dict[str, float]:
    """Load Phase 4 baseline CLV from the current latest artifacts.

    Reads artifacts/latest.json to find each target's current artifact
    directory, then reads metadata.json -> clv_summary -> mean_clv.

    Must be called BEFORE any retraining overwrites the artifacts.
    """
    latest_path = artifacts_dir / "latest.json"
    if not latest_path.exists():
        pytest.skip("No artifacts/latest.json found -- no Phase 4 baseline to compare")

    manifest = json.loads(latest_path.read_text())
    baseline = {}
    for target in ["wp", "ats", "ou"]:
        if target not in manifest:
            pytest.skip(f"Target '{target}' not in latest.json manifest")
        artifact_dir = artifacts_dir / manifest[target]
        metadata_path = artifact_dir / "metadata.json"
        if not metadata_path.exists():
            pytest.skip(f"metadata.json not found at {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        clv_summary = metadata.get("clv_summary", {})
        mean_clv = clv_summary.get("mean_clv")
        if mean_clv is None:
            pytest.skip(f"mean_clv not found in {metadata_path}")
        baseline[target] = float(mean_clv)

    logger.info("Loaded Phase 4 baseline CLV: %s", baseline)
    return baseline


def snapshot_baseline_artifacts(
    artifacts_dir: Path,
) -> Path:
    """Copy current artifacts to a backup directory before retraining.

    Returns the path to the backup directory. This preserves the Phase 4
    baseline so it can be referenced even after retraining overwrites latest.json.

    *artifacts_dir* is REQUIRED and must not be the production tree: this function
    creates a directory inside it, and it used to default to production.
    """
    artifacts_dir = reject_production_artifacts_dir(artifacts_dir)
    backup_dir = artifacts_dir / "_phase4_baseline_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    latest_path = artifacts_dir / "latest.json"
    if not latest_path.exists():
        pytest.skip("No artifacts/latest.json to snapshot")

    manifest = json.loads(latest_path.read_text())
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Copy latest.json
    shutil.copy2(latest_path, backup_dir / "latest.json")

    # Copy each target's artifact directory
    for target in ["wp", "ats", "ou"]:
        if target in manifest:
            src = artifacts_dir / manifest[target]
            if src.exists():
                shutil.copytree(src, backup_dir / manifest[target])

    logger.info("Snapshotted Phase 4 baseline to %s", backup_dir)
    return backup_dir


def run_training_and_get_clv(
    target: str,
    artifacts_dir: Path,
) -> dict[str, float]:
    """Run model training via subprocess and return CLV results.

    Invokes ``uv run python -m models.train --target {target} --artifacts-dir {artifacts_dir}``.
    After training completes, reads the newly written metadata.json for each
    target to extract clv_summary.mean_clv.

    Args:
        target: "all", "wp", "ats", or "ou"
        artifacts_dir: Where to save artifacts. REQUIRED, and refused if it is the
            production tree -- training rewrites ``latest.json``, which is the
            deployed-model manifest. This parameter used to default to ``artifacts/``.

    Returns:
        Dict mapping target name to mean CLV float.
    """
    artifacts_dir = reject_production_artifacts_dir(artifacts_dir)
    cmd = [
        sys.executable,
        "-m",
        "models.train",
        "--target",
        target,
        "--artifacts-dir",
        str(artifacts_dir),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(Path.cwd()),
    )
    if result.returncode != 0:
        pytest.fail(
            f"Training failed (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

    # Read CLV from newly produced artifacts
    targets = ["wp", "ats", "ou"] if target == "all" else [target]
    latest_path = artifacts_dir / "latest.json"
    manifest = json.loads(latest_path.read_text())

    clv_results = {}
    for t in targets:
        artifact_dir = artifacts_dir / manifest[t]
        metadata = json.loads((artifact_dir / "metadata.json").read_text())
        clv_summary = metadata.get("clv_summary", {})
        mean_clv = clv_summary.get("mean_clv", 0.0)
        clv_results[t] = float(mean_clv)

    logger.info("Training CLV results: %s", clv_results)
    return clv_results


def _check_clv_not_degraded(
    baseline: dict[str, float],
    current: dict[str, float],
    group_name: str,
) -> list[str]:
    """Compare current CLV against baseline, return list of degradation warnings.

    Per D-07: log diagnostics rather than hard-fail on individual groups.
    Returns empty list if all targets are within threshold.
    """
    warnings_list = []
    for target in ["wp", "ats", "ou"]:
        if target not in baseline or target not in current:
            continue
        degradation = baseline[target] - current[target]
        if degradation >= DEGRADATION_THRESHOLD:
            msg = (
                f"[{group_name}] {target} CLV degraded by {degradation:.4f} "
                f"(baseline={baseline[target]:.4f}, current={current[target]:.4f}). "
                f"Per D-07: investigate alternative formulations."
            )
            warnings_list.append(msg)
            logger.warning(msg)
        else:
            logger.info(
                "[%s] %s CLV OK: baseline=%.4f, current=%.4f, delta=%.4f",
                group_name,
                target,
                baseline[target],
                current[target],
                -degradation,
            )
    return warnings_list


@pytest.fixture(scope="class")
def sandbox_artifacts(tmp_path_factory) -> Iterator[Path]:
    """A writable COPY of the production artifacts tree, plus a boundary assertion.

    The copy keeps every assertion in this module honest: the baseline CLV read here is
    byte-identical to production's because it IS production's, and the retrain writes
    land beside it instead of on top of it. Roughly 600 KB, so copying is free.

    On teardown the PRODUCTION tree is digested and compared against the snapshot taken
    at setup. `git status` cannot make this check -- `artifacts/` is gitignored, exactly
    as `data/` is -- so it is done by content.
    """
    from tests.data_boundary import (
        PRODUCTION_ARTIFACTS_ROOT,
        assert_tree_unchanged,
        digest_tree,
    )

    production_before = digest_tree(PRODUCTION_ARTIFACTS_ROOT)

    latest_path = ARTIFACTS_DIR / "latest.json"
    if not latest_path.exists():
        pytest.skip(
            "the deployed artifact manifest is not present at artifacts/latest.json, "
            "so there is no baseline to copy or compare against."
        )

    sandbox = tmp_path_factory.mktemp("artifacts_sandbox")
    shutil.copy2(latest_path, sandbox / "latest.json")
    manifest = json.loads(latest_path.read_text())
    for target in ["wp", "ats", "ou"]:
        source = ARTIFACTS_DIR / manifest.get(target, "")
        if target in manifest and source.exists():
            shutil.copytree(source, sandbox / manifest[target])

    yield sandbox

    assert_tree_unchanged(
        production_before,
        digest_tree(PRODUCTION_ARTIFACTS_ROOT),
        PRODUCTION_ARTIFACTS_ROOT,
    )


class TestLiftValidation:
    """Validate that Phase 5 features improve or maintain CLV.

    Per D-06, features are validated in 4 incremental logical groups.
    Each group test retrains with cumulative features and checks CLV.

    Every write this class makes lands in ``sandbox_artifacts``. It used to land in
    production ``artifacts/`` -- see the module header and the refusal above.
    """

    @pytest.fixture(autouse=True, scope="class")
    def baseline_clv(self, sandbox_artifacts):
        """Snapshot and load Phase 4 baseline CLV once for all tests."""
        # Snapshot before any retraining -- inside the sandbox, never production.
        snapshot_baseline_artifacts(sandbox_artifacts)
        baseline = load_baseline_clv(sandbox_artifacts)
        TestLiftValidation._baseline = baseline
        TestLiftValidation._artifacts_dir = sandbox_artifacts
        return baseline

    def test_group_1_qb_features(self):
        """Group 1: Baseline + QB features (FEAT-12, FEAT-13).

        Adds: home_qb_adjustment, away_qb_adjustment.
        Retrains all targets and compares CLV to Phase 4 baseline.
        """
        current = run_training_and_get_clv(
            target="all", artifacts_dir=self._artifacts_dir
        )
        warnings_list = _check_clv_not_degraded(
            self._baseline, current, "Group 1: QB features"
        )
        # Store for incremental comparison
        TestLiftValidation._group1_clv = current
        if warnings_list:
            logger.warning(
                "Group 1 (QB) showed degradation in %d target(s). "
                "Per D-07: investigate before removing.",
                len(warnings_list),
            )

    def test_group_2_pbp_derived(self):
        """Group 2: Baseline + QB + PBP-derived (FEAT-15, FEAT-20, FEAT-21).

        Adds: rolling_cpoe, rolling_avg_drive_start_yardline, rolling_neutral_pace.
        Retrains and compares CLV to Phase 4 baseline.

        Note: In the full pipeline, all features are computed together.
        This test validates that the cumulative feature set at this point
        does not degrade CLV vs the Phase 4 baseline.
        """
        current = run_training_and_get_clv(
            target="all", artifacts_dir=self._artifacts_dir
        )
        warnings_list = _check_clv_not_degraded(
            self._baseline, current, "Group 2: PBP-derived"
        )
        TestLiftValidation._group2_clv = current
        if warnings_list:
            logger.warning(
                "Group 2 (PBP) showed degradation in %d target(s). "
                "Per D-07: investigate before removing.",
                len(warnings_list),
            )

    def test_group_3_opponent_adjusted(self):
        """Group 3: Baseline + QB + PBP + Opponent-adjusted EPA (FEAT-14).

        Adds: rolling_opp_adj_epa_per_play, rolling_opp_adj_pass_epa,
        rolling_opp_adj_rush_epa (replacing raw EPA columns).
        Retrains and compares CLV to Phase 4 baseline.
        """
        current = run_training_and_get_clv(
            target="all", artifacts_dir=self._artifacts_dir
        )
        warnings_list = _check_clv_not_degraded(
            self._baseline, current, "Group 3: Opponent-adjusted EPA"
        )
        TestLiftValidation._group3_clv = current
        if warnings_list:
            logger.warning(
                "Group 3 (Opp Adj) showed degradation in %d target(s). "
                "Per D-07: investigate before removing.",
                len(warnings_list),
            )

    def test_group_4_contextual(self):
        """Group 4: All Phase 5 features including contextual (FEAT-17, FEAT-18, FEAT-19).

        Adds: season_progress, late_season, surface_mismatch, is_divisional.
        Retrains and compares CLV to Phase 4 baseline.
        """
        current = run_training_and_get_clv(
            target="all", artifacts_dir=self._artifacts_dir
        )
        warnings_list = _check_clv_not_degraded(
            self._baseline, current, "Group 4: Contextual"
        )
        TestLiftValidation._group4_clv = current
        if warnings_list:
            logger.warning(
                "Group 4 (Contextual) showed degradation in %d target(s). "
                "Per D-07: investigate before removing.",
                len(warnings_list),
            )

    def test_all_features_clv_not_degraded(self):
        """Final gate: full Phase 5 feature set CLV must not degrade vs Phase 4 baseline.

        This is the hard gate. Individual group tests above log warnings per D-07,
        but this combined test asserts. Uses the latest training results from
        the group 4 run (which has all features).
        A small degradation threshold (0.5%) is allowed for noise.
        """
        baseline = self._baseline
        # Use group 4 results (all features) or retrain if not available
        current = getattr(self, "_group4_clv", None)
        if current is None:
            current = run_training_and_get_clv(
                target="all", artifacts_dir=self._artifacts_dir
            )

        for target in ["wp", "ats", "ou"]:
            degradation = baseline[target] - current[target]
            assert degradation < DEGRADATION_THRESHOLD, (
                f"{target} CLV degraded by {degradation:.4f} "
                f"(baseline={baseline[target]:.4f}, current={current[target]:.4f}). "
                f"Per D-07: investigate alternative formulations before removing features."
            )

    def test_feature_count_within_budget(self):
        """Feature selection should keep counts within soft caps.

        After feature selection with max_features=20, verify that the
        selected feature lists are reasonable. Phase 4 targets:
        WP ~16, ATS ~22, O/U ~21 (soft caps).
        """
        latest_path = ARTIFACTS_DIR / "latest.json"
        if not latest_path.exists():
            pytest.skip("No artifacts to check feature counts")
        manifest = json.loads(latest_path.read_text())

        for target in ["wp", "ats", "ou"]:
            artifact_dir = ARTIFACTS_DIR / manifest[target]
            feature_list = json.loads((artifact_dir / "feature_list.json").read_text())
            n_features = len(feature_list)
            # Hard max with buffer -- feature selection should handle this
            assert n_features <= 25, (
                f"{target} has {n_features} features (max 25). Features: {feature_list}"
            )
            logger.info(
                "%s feature count: %d (features: %s)",
                target,
                n_features,
                feature_list,
            )

    def test_new_features_present_in_gold(self):
        """Verify new features appear in Gold layer after pipeline run."""
        expected_features = {
            "home_qb_adjustment",
            "away_qb_adjustment",
            "season_progress",
            "late_season",
            "surface_mismatch",
            "is_divisional",
        }
        # These appear with home/away and off/def prefixes
        expected_rolling_patterns = [
            "rolling_opp_adj_epa_per_play",
            "rolling_cpoe",
            "rolling_avg_drive_start_yardline",
            "rolling_neutral_pace",
        ]
        # These raw EPA columns must NOT be present (replaced by opp_adj)
        forbidden_raw_epa = [
            "home_off_rolling_epa_per_play",
            "away_off_rolling_epa_per_play",
            "home_def_rolling_epa_per_play",
            "away_def_rolling_epa_per_play",
            "home_off_rolling_pass_epa",
            "away_off_rolling_pass_epa",
            "home_def_rolling_pass_epa",
            "away_def_rolling_pass_epa",
            "home_off_rolling_rush_epa",
            "away_off_rolling_rush_epa",
            "home_def_rolling_rush_epa",
            "away_def_rolling_rush_epa",
        ]

        for target in ["wp", "ats", "ou"]:
            gold_path = GOLD_DIR / f"features_{target}.parquet"
            if not gold_path.exists():
                pytest.skip(f"Gold matrix not found at {gold_path}")

            df = pd.read_parquet(gold_path)
            cols = set(df.columns)

            # Check direct feature names
            for feat in expected_features:
                assert feat in cols, (
                    f"Feature '{feat}' missing from {target} Gold matrix. "
                    f"Available columns: {sorted(cols)}"
                )

            # Check rolling patterns (appear with prefixes)
            for pattern in expected_rolling_patterns:
                matching = [c for c in cols if pattern in c]
                assert len(matching) > 0, (
                    f"No columns matching '{pattern}' in {target} Gold matrix. "
                    f"Available rolling columns: "
                    f"{sorted(c for c in cols if 'rolling' in c)}"
                )

            # Check raw EPA columns are gone (replaced by opp_adj)
            for raw_col in forbidden_raw_epa:
                assert raw_col not in cols, (
                    f"Raw EPA column '{raw_col}' still present in {target} "
                    f"Gold matrix. Should have been replaced by "
                    f"opponent-adjusted version."
                )


class TestTheProductionArtifactsRefusal:
    """The fix must be verifiable WITHOUT a 15-25 minute retrain.

    The behavioural proof of the sandbox redirect is a full four-cycle retrain, which
    nobody runs casually -- so a fix proven only that way is a fix nobody checks. These
    tests cost milliseconds and pin the refusal itself, which is the load-bearing part:
    if it fires, no write can reach production regardless of what the class does.
    """

    def test_the_production_tree_is_refused_by_the_training_helper(self) -> None:
        with pytest.raises(ProductionArtifactsWriteRefused) as excinfo:
            run_training_and_get_clv(target="all", artifacts_dir=ARTIFACTS_DIR)
        assert "latest.json" in str(excinfo.value), (
            "the refusal must name the deployed-model manifest, so the reader knows "
            "what was nearly overwritten."
        )

    def test_the_production_tree_is_refused_by_the_snapshot_helper(self) -> None:
        with pytest.raises(ProductionArtifactsWriteRefused):
            snapshot_baseline_artifacts(ARTIFACTS_DIR)

    def test_an_absolute_path_to_the_production_tree_is_also_refused(self) -> None:
        """Resolution happens before comparison, so a longer spelling cannot slip past."""
        with pytest.raises(ProductionArtifactsWriteRefused):
            snapshot_baseline_artifacts(ARTIFACTS_DIR.resolve())
        with pytest.raises(ProductionArtifactsWriteRefused):
            snapshot_baseline_artifacts(Path.cwd() / "artifacts")

    def test_a_sandbox_directory_is_accepted(self, tmp_path: Path) -> None:
        assert reject_production_artifacts_dir(tmp_path) == tmp_path.resolve()

    def test_neither_writing_helper_still_defaults_to_production(self) -> None:
        """A default argument is how this defect existed for six milestones."""
        import inspect

        for helper in (run_training_and_get_clv, snapshot_baseline_artifacts):
            signature = inspect.signature(helper)
            parameter = signature.parameters["artifacts_dir"]
            assert parameter.default is inspect.Parameter.empty, (
                f"{helper.__name__} has a default artifacts_dir of "
                f"{parameter.default!r}. A caller that forgets the argument must fail "
                "loudly, not silently write the deployed-model tree."
            )

    def test_no_call_site_in_this_module_omits_the_artifacts_directory(self) -> None:
        """A source scan, so a newly added call site cannot reintroduce the default."""
        import ast

        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        writing_helpers = {"run_training_and_get_clv", "snapshot_baseline_artifacts"}
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name not in writing_helpers:
                continue
            supplied = {kw.arg for kw in node.keywords}
            if "artifacts_dir" in supplied or len(node.args) >= (
                2 if name == "run_training_and_get_clv" else 1
            ):
                continue
            offenders.append(f"{name} at line {node.lineno}")

        assert not offenders, (
            "these call sites do not pass artifacts_dir, so they would write wherever "
            f"the helper defaults: {offenders}"
        )
