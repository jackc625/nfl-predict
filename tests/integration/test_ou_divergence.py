"""Wave 0 smoke + number-anchor suite for the O/U divergence harness (Phase 26, plan 26-01).

Mirrors ``tests/integration/test_diag_diagnosis.py`` structure (the canonical diagnosis-harness
smoke pattern, D26-15). Covers the OUM-01 measurement correctness for the FIRST HALF of
``backtest/ou_divergence.py`` -- the integrity preamble (D26-04), the bias-vs-anticipation
decomposition (D26-05, the LEAD section), and the prior-season bias-adjusted re-score (D26-18).
The extended bucket sweep (D26-06), the trial registry + BH-FDR correction (D26-10), the
structural bar (D26-11), the edge-magnitude monotonicity sweep (D26-03), and the throwaway EV
preview (D26-07/16) are added by Plan 26-03 (the ``trial_registry`` / ``count_parity`` /
``ev_preview_not_imported`` selectors live here too).

Load-bearing number anchors (reproduced this session against the DEPLOYED v1.0 OU artifact
``ou_20260326_163930`` over 2021-2024 canonical gold):
  - pooled line_clv +1.1095 on n=1087 with-line games (52 excluded)
  - model-picks-over share pooled 0.727 with the 2021 sign-flip (18.4% over, line_clv -1.21)
  - pooled residual SD 12.946

Selectors (``-k``): deployed_population, bias_over_share, coverage_counts,
provenance_label_split, no_train_no_write, production_files_untouched, determinism,
trial_registry, count_parity, edge_magnitude, survivable_subpopulation, ev_preview,
ev_preview_not_imported.

Self-judging boundary (T-26-02): a real harness run must leave models/clv.py, config/gate.toml,
and backtest/simulation.py byte-identical (the diagnosis must not edit its own judge). The
no-train/no-write-gold guard mirrors the Phase-22 HARD BOUNDARY (LOAD + predict only).

Provenance terminology split (Codex HIGH, D26-04): the integrity preamble carries TWO DISTINCT
flags -- ``mock_free_odds`` (no mock/synthetic ODDS: sportsbook in {consensus, draftkings},
is_live all False) and ``synthetic_snapshot_ts`` (fabricated single-stamp TIMESTAMP reality:
8 distinct snapshot_ts, 1 line per game). The overloaded word "synthetic" must not conflate the
odds-contamination concept with the timestamp-fabrication concept.

Reproducibility convention: the shared fixture loads gold + normalized closing odds via the
engine loaders (``_load_features`` / ``_load_closing_odds``) rather than re-reading parquet so the
LAR->LA canonical team-abbreviation normalization matches the backtest exactly (CLAUDE.md).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
from pathlib import Path

import pandas as pd
import pytest

from backtest.engine import BacktestEngine

# Gold presence skip-guard: the integration tests need the Phase-20 rebuilt canonical gold.
_GOLD_WP_PATH = Path("data/gold/features_wp.parquet")

# Number anchors (reproduced this session; see module docstring).
ANCHOR_OU_POOLED_LINE_CLV = 1.1095
ANCHOR_N_WITH_LINE = 1087
ANCHOR_MODEL_OVER_SHARE = 0.727
ANCHOR_RESID_SD_POOLED = 12.946

# Tolerances: tight for line_clv/over_share, exact for the game count, looser for resid SD.
_TOL_CLV = 5e-3
_TOL_OVER_SHARE = 5e-3
_TOL_RESID_SD = 5e-2

# The expected resolved deployed OU artifact identity (D25-14 retained v1.0).
DEPLOYED_OU_ARTIFACT = "ou_20260326_163930"

# Production files the diagnosis must NOT edit (the self-judging boundary, T-26-02).
_PRODUCTION_JUDGE_FILES = (
    Path("models/clv.py"),
    Path("config/gate.toml"),
    Path("backtest/simulation.py"),
)


@pytest.fixture(scope="module")
def gold_and_odds_2021_2024() -> dict[str, object]:
    """Shared fixture: 2021-2024 gold per target + normalized closing odds.

    Reuses ``BacktestEngine._load_features`` (gold, season-filtered) and
    ``BacktestEngine._load_closing_odds`` (LAR->LA normalized game_ids) so the harness fixture
    matches the backtest's canonical team-mapping exactly. Does NOT read the parquet raw.

    Returns a dict with ``gold`` (dict[target -> 2021-2024 frame]) and ``odds`` (normalized
    closing-odds frame). Skips the whole module if canonical gold is absent.
    """
    if not _GOLD_WP_PATH.exists():
        pytest.skip(f"Canonical gold not present at {_GOLD_WP_PATH}")

    engine = BacktestEngine()
    gold: dict[str, pd.DataFrame] = {}
    for target in ("wp", "ats", "ou"):
        df = engine._load_features(target)
        gold[target] = df[(df["season"] >= 2021) & (df["season"] <= 2024)].copy()
    odds = engine._load_closing_odds()
    return {"gold": gold, "odds": odds}


def _sha256(path: Path) -> str:
    """Return the stdlib-hashlib sha256 hex digest of a file (byte-identity anchor)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.integration
class TestOuDivergence:
    """Wave-0 anchors + provenance/self-judging/boundary guards for the divergence harness."""

    # -- deployed_population: the harness scores the DEPLOYED OU artifact ------------------

    def test_deployed_population(self, gold_and_odds_2021_2024) -> None:
        """Preamble reports n=1087 with-line, pooled line_clv ~+1.1095 from the deployed artifact.

        The resolved artifact identity is asserted via a STABLE provenance source (the loaded
        artifact's directory name / artifacts/latest.json), NOT a metadata.json version key --
        the OU metadata.json has NO version key (Codex LOW, verification-confirmed).
        """
        from backtest.ou_divergence import bias_vs_anticipation, integrity_preamble

        preamble = integrity_preamble(odds_df=gold_and_odds_2021_2024["odds"])
        assert preamble["n_with_line"] == ANCHOR_N_WITH_LINE, (
            f"with-line population {preamble['n_with_line']} != anchor {ANCHOR_N_WITH_LINE}"
        )
        # Stable provenance: the deployed artifact identity, never a metadata version key.
        assert preamble["deployed_artifact"] == DEPLOYED_OU_ARTIFACT, (
            f"resolved deployed artifact {preamble['deployed_artifact']!r} != "
            f"{DEPLOYED_OU_ARTIFACT!r}"
        )

        bias = bias_vs_anticipation(preds=None, odds=gold_and_odds_2021_2024["odds"])
        assert bias["n"] == ANCHOR_N_WITH_LINE
        assert abs(bias["pooled_line_clv"] - ANCHOR_OU_POOLED_LINE_CLV) < _TOL_CLV, (
            f"pooled line_clv {bias['pooled_line_clv']} drifted from anchor "
            f"{ANCHOR_OU_POOLED_LINE_CLV}"
        )

    # -- bias_over_share: the reproduced 72.7%-over finding + the 2021 sign-flip ----------

    def test_bias_over_share(self, gold_and_odds_2021_2024) -> None:
        """Pooled model-picks-over share ~0.727; 2021 is sign-flipped (over<0.30, line_clv<0)."""
        from backtest.ou_divergence import bias_vs_anticipation

        bias = bias_vs_anticipation(preds=None, odds=gold_and_odds_2021_2024["odds"])
        assert (
            abs(bias["pooled_over_share"] - ANCHOR_MODEL_OVER_SHARE) < _TOL_OVER_SHARE
        ), (
            f"pooled over-share {bias['pooled_over_share']} drifted from anchor "
            f"{ANCHOR_MODEL_OVER_SHARE}"
        )

        per_season = bias["per_season"]
        assert 2021 in per_season, "per-season table must report 2021"
        s2021 = per_season[2021]
        assert s2021["over_share"] < 0.30, (
            f"2021 model-over share {s2021['over_share']} should be sign-flipped (<0.30)"
        )
        assert s2021["line_clv"] < 0.0, (
            f"2021 line_clv {s2021['line_clv']} should be negative (the sign-flip canary)"
        )

    # -- coverage_counts: no orphan metric (every table carries a coverage count) ----------

    def test_coverage_counts(self, gold_and_odds_2021_2024) -> None:
        """Every emitted table carries both an n and an exclusion/coverage count (D26-04(iv))."""
        from backtest.ou_divergence import (
            bias_vs_anticipation,
            debiased_rescore,
            integrity_preamble,
        )

        odds = gold_and_odds_2021_2024["odds"]
        preamble = integrity_preamble(odds_df=odds)
        assert "n_total" in preamble and "n_with_line" in preamble
        assert "n_excluded" in preamble, "preamble must carry an exclusion count"
        assert preamble["n_total"] == preamble["n_with_line"] + preamble["n_excluded"]
        assert "excluded_characterization" in preamble, (
            "preamble must characterize the excluded games (selection-bias check)"
        )

        bias = bias_vs_anticipation(preds=None, odds=odds)
        assert "n" in bias and "n_excluded" in bias, (
            "bias section must carry both n and n_excluded (no orphan metric)"
        )
        for season, row in bias["per_season"].items():
            assert "n" in row, f"per-season row {season} must carry an n"

        debiased = debiased_rescore(preds=None, odds=odds)
        for season, row in debiased["per_season_debiased_line_clv"].items():
            assert "n" in row, f"debiased per-season row {season} must carry an n"

    # -- provenance_label_split: two DISTINCT provenance flags (Codex HIGH) ----------------

    def test_provenance_label_split(self, gold_and_odds_2021_2024) -> None:
        """mock_free_odds and synthetic_snapshot_ts are SEPARATE keys, both True on real data.

        Codex HIGH: the overloaded word "synthetic" must split into two concepts -- mock/synthetic
        ODDS contamination (mock_free_odds) is distinct from synthetic TIMESTAMP fabrication
        (synthetic_snapshot_ts). There must be no single key conflating the two.
        """
        from backtest.ou_divergence import integrity_preamble

        preamble = integrity_preamble(odds_df=gold_and_odds_2021_2024["odds"])
        assert "mock_free_odds" in preamble, (
            "preamble must carry the mock_free_odds flag"
        )
        assert "synthetic_snapshot_ts" in preamble, (
            "preamble must carry the synthetic_snapshot_ts flag"
        )
        assert preamble["mock_free_odds"] is True, (
            "real silver odds are mock-free (sportsbook consensus/draftkings, is_live False)"
        )
        assert preamble["synthetic_snapshot_ts"] is True, (
            "the single-stamp-per-game timestamp reality is synthetic (8 distinct stamps)"
        )
        # The two concepts must not be conflated under a single shared key.
        assert "synthetic_odds" not in preamble, (
            "must not conflate mock-odds contamination with timestamp fabrication"
        )
        assert preamble["snapshot_ts_distinct"] == 8
        assert preamble["snapshots_per_game_max"] == 1

    # -- no_train_no_write: source-grep + runtime to_parquet spy ---------------------------

    def test_no_train_no_write(self, gold_and_odds_2021_2024, monkeypatch) -> None:
        """The harness imports no train_* module and writes no data/ parquet (HARD BOUNDARY)."""
        import backtest.ou_divergence as ou_mod

        source = inspect.getsource(ou_mod)
        forbidden_imports = (
            "import train_",
            "from models.train_",
            "from models.trainers",
            "import models.train",
            "scripts.train_models",
            "scripts.retrain_models",
        )
        for token in forbidden_imports:
            assert token not in source, (
                f"ou_divergence.py must not import a trainer ({token!r})"
            )
        assert ".to_parquet(" not in source, (
            "ou_divergence.py must not write any parquet"
        )

        # Runtime behavioral guard: a real preamble+bias run writes no data/ parquet.
        calls: list[str] = []
        orig_to_parquet = pd.DataFrame.to_parquet

        def spy_to_parquet(self, path, *args, **kwargs):
            calls.append(str(path))
            return orig_to_parquet(self, path, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_parquet", spy_to_parquet)

        odds = gold_and_odds_2021_2024["odds"]
        ou_mod.integrity_preamble(odds_df=odds)
        ou_mod.bias_vs_anticipation(preds=None, odds=odds)

        data_writes = [c for c in calls if "data/" in c.replace("\\", "/")]
        assert not data_writes, (
            "the harness must never write data/ (LOAD + predict only); "
            f"observed: {data_writes}"
        )

    # -- production_files_untouched: sha256 byte-identity across a real run ----------------

    def test_production_files_untouched(self, gold_and_odds_2021_2024) -> None:
        """models/clv.py, config/gate.toml, simulation.py are byte-identical across a real run.

        The behavioral proof of the self-judging boundary (T-26-02): the diagnosis must not edit
        its own judge. Mirrors the Phase-25 recorded-seed-sha256 byte-identity pattern.
        """
        from backtest.ou_divergence import (
            bias_vs_anticipation,
            debiased_rescore,
            integrity_preamble,
        )

        before = {f: _sha256(f) for f in _PRODUCTION_JUDGE_FILES}

        odds = gold_and_odds_2021_2024["odds"]
        integrity_preamble(odds_df=odds)
        bias_vs_anticipation(preds=None, odds=odds)
        debiased_rescore(preds=None, odds=odds)

        after = {f: _sha256(f) for f in _PRODUCTION_JUDGE_FILES}
        for f in _PRODUCTION_JUDGE_FILES:
            assert before[f] == after[f], (
                f"{f} was modified by the harness run (self-judging boundary violated): "
                f"{before[f]} -> {after[f]}"
            )

    # -- determinism: two runs are value-identical -----------------------------------------

    def test_determinism(self, gold_and_odds_2021_2024) -> None:
        """Two harness runs are value-identical on the sorted per-game frame (anti-rot)."""
        from backtest.ou_divergence import bias_vs_anticipation

        odds = gold_and_odds_2021_2024["odds"]
        run_a = bias_vs_anticipation(preds=None, odds=odds)
        run_b = bias_vs_anticipation(preds=None, odds=odds)

        frame_a = run_a["per_game"].sort_values("game_id").reset_index(drop=True)
        frame_b = run_b["per_game"].sort_values("game_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(frame_a, frame_b)

    # -- debiased re-score: prior-seasons-only estimation (D26-18) --------------------------

    def test_debiased_prior_season_only(self, gold_and_odds_2021_2024) -> None:
        """The de-biased read covers 2022-2024 (NOT 2021) with prior-seasons-only bias estimation.

        Acceptance (D26-18): season 2021 is absent; the 2022 bias-subtracted equals the 2021-only
        mean(model_total - actual); the output carries an interpretation, a pooled value, and the
        no-prior-seasons caveat.
        """
        from backtest.diagnose import score_deployed_artifacts
        from backtest.ou_divergence import debiased_rescore
        from models.clv import compute_clv_for_predictions

        odds = gold_and_odds_2021_2024["odds"]
        result = debiased_rescore(preds=None, odds=odds)

        per_season = result["per_season_debiased_line_clv"]
        assert set(per_season) == {2022, 2023, 2024}, (
            f"de-biased read must cover 2022-2024 only, got {sorted(per_season)}"
        )
        assert 2021 not in per_season, (
            "2021 must be absent (no prior seasons to de-bias from)"
        )
        assert result["interpretation"] in (
            "nothing underneath",
            "residual anticipation",
        )
        assert result["pooled_debiased_line_clv"] is not None
        assert "no prior seasons" in result["caveat"].lower()

        # The 2022 bias must equal the 2021-ONLY mean(model_total - actual) (walk-forward).
        preds = score_deployed_artifacts("ou")
        clv = compute_clv_for_predictions(preds, odds, "ou")
        valid = clv[clv["has_closing_odds"]]
        s2021 = valid[valid["season"] == 2021]
        expected_bias_2022 = float((s2021["model_total"] - s2021["actual"]).mean())
        assert (
            abs(result["per_season_bias_subtracted"][2022] - expected_bias_2022) < 1e-9
        ), (
            "2022 bias must be estimated on 2021 ONLY (no season-2022 row in its own estimate)"
        )

    def test_debiased_uses_production_clv_no_hand_roll(self) -> None:
        """The de-biased re-score recomputes line_clv via the production CLV function, not by hand.

        Source-grep: ``debiased_rescore`` must call ``compute_clv_for_predictions`` and must NOT
        contain a hand-rolled ``model_total - closing_total`` (or ``- total``) subtraction outside
        the production CLV call (Codex MEDIUM: CLV reuse, not hand-coding).
        """
        import inspect

        from backtest.ou_divergence import debiased_rescore

        source = inspect.getsource(debiased_rescore)
        assert "compute_clv_for_predictions" in source, (
            "debiased_rescore must recompute line_clv via the production CLV function"
        )
        for hand_rolled in (
            "model_total - closing_total",
            "- closing_total",
            "model_total - total",
        ):
            assert hand_rolled not in source, (
                f"debiased_rescore must not hand-roll CLV ({hand_rolled!r})"
            )

    # -- trial_registry: BH-FDR denominator with the strengthened invariants (D26-10) -----

    def test_trial_registry(self, gold_and_odds_2021_2024) -> None:
        """The trial registry is the full BH-FDR denominator with the strengthened invariants.

        Asserts (Codex HIGH, verification-refuted-but-kept): the registry is non-empty; every
        entry carries an n + n_excluded; for the non-None-p subset the BH-adjusted p is
        element-wise >= the raw p (holds by construction for scipy BH), every adjusted_p is in
        [0,1], adjusted_p is monotone non-decreasing when sorted by raw_p, and each adjusted_p is
        aligned to the same entry as its raw_p (alignment by index over the non-None subset).
        """
        from backtest.ou_divergence import extended_bucket_sweep

        odds = gold_and_odds_2021_2024["odds"]
        result = extended_bucket_sweep(preds=None, odds=odds)

        assert "trial_registry" in result and "n_trials" in result
        registry = result["trial_registry"]
        assert len(registry) > 0, "the trial registry must log at least one bucket/test"

        for entry in registry:
            assert "n" in entry, f"registry entry missing n: {entry}"
            assert "n_excluded" in entry, f"registry entry missing n_excluded: {entry}"
            assert "cut_name" in entry and "bucket_label" in entry and "stream" in entry
            assert "raw_p" in entry and "adjusted_p" in entry

        # n_trials == the count of non-None-p entries fed to BH-FDR.
        tested = [e for e in registry if e["raw_p"] is not None]
        assert result["n_trials"] == len(tested), (
            f"n_trials {result['n_trials']} != non-None-p count {len(tested)}"
        )
        assert len(tested) > 0, "at least one bucket must have a testable p-value"

        # Every tested entry: adjusted_p present, in [0,1], and >= raw_p (scipy BH invariant).
        for e in tested:
            adj = e["adjusted_p"]
            assert adj is not None, f"tested entry has no adjusted_p: {e}"
            assert 0.0 <= adj <= 1.0, f"adjusted_p {adj} out of [0,1]"
            assert adj >= e["raw_p"] - 1e-9, (
                f"adjusted_p {adj} < raw_p {e['raw_p']} (BH must not shrink below raw)"
            )

        # Sorted-by-raw-p monotonicity of adjusted_p (the safer invariant Codex named).
        by_raw = sorted(tested, key=lambda e: e["raw_p"])
        adj_sorted = [e["adjusted_p"] for e in by_raw]
        for prev, curr in itertools.pairwise(adj_sorted):
            assert curr >= prev - 1e-9, (
                f"adjusted_p not monotone in raw-p order: {prev} -> {curr}"
            )

        # Entries with no testable sample are flagged insufficient_sample, not dropped.
        insufficient = [e for e in registry if e["raw_p"] is None]
        for e in insufficient:
            assert (
                e.get("insufficient_sample") is True or e.get("unavailable") is True
            ), f"a None-p entry must be flagged insufficient_sample/unavailable: {e}"

    # -- count_parity: simulator inner-merge preserves bucket membership (Codex HIGH) ------

    def test_count_parity(self, gold_and_odds_2021_2024) -> None:
        """For one representative bucket, graded bet count equals the with-line game count.

        The BettingSimulator inner-merges on game_id at min_edge_threshold=0.0 and grades every
        with-line game in the slice (no duplication). The count-parity assert is cheap insurance
        (Codex HIGH, verification-refuted as harmless hygiene): input -> with-line -> graded are
        consistent for the slice.
        """
        from backtest.ou_divergence import bucket_count_parity

        odds = gold_and_odds_2021_2024["odds"]
        parity = bucket_count_parity(preds=None, odds=odds)

        assert parity["n_with_line"] > 0, "representative bucket must be non-empty"
        assert parity["n_graded"] == parity["n_with_line"], (
            f"graded bets {parity['n_graded']} != with-line games {parity['n_with_line']} "
            "(simulator inner-merge must preserve membership at min_edge=0.0)"
        )
        assert parity["n_input"] >= parity["n_with_line"], (
            "input games must be >= with-line games (some lack a closing line)"
        )

    # -- coverage_counts (extended to the sweep buckets) -----------------------------------

    def test_coverage_counts_sweep(self, gold_and_odds_2021_2024) -> None:
        """Every sweep bucket carries both an n and a coverage/exclusion count (no orphan)."""
        from backtest.ou_divergence import extended_bucket_sweep

        odds = gold_and_odds_2021_2024["odds"]
        result = extended_bucket_sweep(preds=None, odds=odds)

        assert "sweep" in result, "extended_bucket_sweep must return per-cut tables"
        for cut_name, buckets in result["sweep"].items():
            assert isinstance(buckets, dict), (
                f"cut {cut_name} must map to a bucket dict"
            )
            for label, bucket in buckets.items():
                if bucket.get("unavailable"):
                    # An unavailable cut still carries coverage metadata, never a silent skip.
                    assert "coverage_note" in bucket, (
                        f"unavailable bucket {cut_name}/{label} must carry a coverage_note"
                    )
                    continue
                assert "n" in bucket, f"bucket {cut_name}/{label} must carry an n"
                assert "n_excluded" in bucket, (
                    f"bucket {cut_name}/{label} must carry an n_excluded (no orphan metric)"
                )

        # The weather cut must reference the CORRECT gold columns (NOT is_outdoor).
        source_path = Path("backtest/ou_divergence.py")
        src = source_path.read_text(encoding="utf-8")
        assert "venue_outdoor" in src, "weather cut must reference venue_outdoor"
        assert "weather_severity_score" in src, (
            "weather cut must reference weather_severity_score"
        )
        assert "is_outdoor" not in src, (
            "the non-existent gold column is_outdoor must not be referenced (Codex MED)"
        )

    # -- edge_magnitude: monotonicity sweep over min_edge_threshold (D26-03) ----------------

    def test_edge_magnitude(self, gold_and_odds_2021_2024) -> None:
        """The edge-magnitude sweep reports hit-rate AND bet count at each grid threshold."""
        from backtest.ou_divergence import EDGE_MAGNITUDE_GRID, edge_magnitude_sweep

        odds = gold_and_odds_2021_2024["odds"]
        result = edge_magnitude_sweep(preds=None, odds=odds)

        assert "grid" in result
        for threshold in EDGE_MAGNITUDE_GRID:
            assert threshold in result["grid"], f"missing grid point {threshold}"
            row = result["grid"][threshold]
            assert "n_bets" in row, f"grid {threshold} must carry n_bets"
            assert "hit_rate" in row, f"grid {threshold} must carry hit_rate"
        # The base (0.0) point is the D26-03 all-games straight-pick.
        assert 0.0 in result["grid"]
        assert result["grid"][0.0]["n_bets"] > 0

    # -- survivable_subpopulation: the D26-11 structural bar (graded-edge direction) --------

    def test_survivable_subpopulation_classification(
        self, gold_and_odds_2021_2024
    ) -> None:
        """name_survivable_subpopulation classifies on graded-edge direction, not line_clv sign.

        Acceptance (Codex MED): each candidate carries raw_p, adjusted_p, n, a length-4 per-season
        graded-edge direction vector, and a classification in {survivable,
        suggestive_not_survivable}. A bucket failing N>=175 or <3-of-4-seasons graded-edge
        agreement is suggestive_not_survivable even with adjusted_p < 0.05.
        """
        from backtest.ou_divergence import (
            extended_bucket_sweep,
            name_survivable_subpopulation,
        )

        odds = gold_and_odds_2021_2024["odds"]
        sweep = extended_bucket_sweep(preds=None, odds=odds)
        result = name_survivable_subpopulation(sweep)

        assert "candidates" in result and "any_survivable" in result
        for cand in result["candidates"]:
            assert "raw_p" in cand and "adjusted_p" in cand and "n" in cand
            assert "direction" in cand, (
                "candidate must carry a per-season direction vector"
            )
            assert len(cand["direction"]) == 4, (
                f"direction vector must be length 4, got {len(cand['direction'])}"
            )
            assert cand["classification"] in (
                "survivable",
                "suggestive_not_survivable",
            ), f"bad classification {cand['classification']!r}"

        # The structural-bar direction consumes graded_edge_direction_by_season, not line_clv sign.
        source = Path("backtest/ou_divergence.py").read_text(encoding="utf-8")
        assert "graded_edge_direction_by_season" in source, (
            "the structural bar must consume graded_edge_direction_by_season (Codex MED)"
        )

    # -- ev_preview: side-specific slippage-adjusted cover probability (D26-07/16) -----------

    def test_ev_preview_side_specific_slippage(self, gold_and_odds_2021_2024) -> None:
        """The EV preview applies the half-point AGAINST the bet side (over: +0.5, under: -0.5).

        Numeric check (Codex MED + consensus): flipping the bet side flips which half-point offset
        is used. For a fixed model_total/closing_total/sd, the over-side p_side uses
        (closing_total + 0.5) and the under-side p_side uses (closing_total - 0.5).
        """
        from backtest.ou_divergence import (
            SD_SENSITIVITY_BAND,
            throwaway_ev_preview,
        )

        # A small synthetic frame: one over-pick game, one under-pick game, same line.
        frame = pd.DataFrame(
            {
                "game_id": ["g_over", "g_under"],
                "season": [2023, 2023],
                "model_total": [48.0, 40.0],  # over pick / under pick vs a 44.0 line
                "total": [44.0, 44.0],
                "actual": [47.0, 41.0],
            }
        )
        result = throwaway_ev_preview(frame, sd=13.0)

        assert "disclaimer" in result and "EXPLORATORY" in result["disclaimer"]
        assert "devig_method" in result
        assert result["devig_method"] in ("real_nflreadpy", "flat_-110")
        assert abs(result["breakeven"] - 0.5238) < 1e-3, (
            "breakeven must be 0.5238 at -110"
        )

        # EV reported across the SD band plus the in-harness fit.
        assert "by_sd" in result
        lo, hi = SD_SENSITIVITY_BAND
        for sd_point in (lo, hi):
            assert sd_point in result["by_sd"], f"missing SD band point {sd_point}"
            assert "ev" in result["by_sd"][sd_point]

        # Side-specific slippage numeric check: the over and under p_side use opposite offsets.
        from scipy.stats import norm

        sd = 13.0
        # over pick: model 48 vs line 44 -> p_side = P(actual > 44.5)
        expected_over = 1.0 - norm.cdf((44.0 + 0.5 - 48.0) / sd)
        # under pick: model 40 vs line 44 -> p_side = P(actual < 43.5)
        expected_under = norm.cdf((44.0 - 0.5 - 40.0) / sd)
        per_bet = {b["game_id"]: b for b in result["per_bet"]}
        assert abs(per_bet["g_over"]["p_side"] - expected_over) < 1e-9, (
            "over-side p_side must apply +0.5 against the bet"
        )
        assert abs(per_bet["g_under"]["p_side"] - expected_under) < 1e-9, (
            "under-side p_side must apply -0.5 against the bet"
        )
        assert per_bet["g_over"]["bet_side"] == "over"
        assert per_bet["g_under"]["bet_side"] == "under"


def test_ev_preview_not_imported_by_production() -> None:
    """No production module DEFINES or CALLS throwaway_ev_preview (no-leak guard, T-26-08).

    The ``throwaway_ev_preview(`` call/def token must appear only in backtest/ou_divergence.py and
    the test files -- never in scripts/, app/, api/, or models/ (Phase 27 builds the real chain).
    The bare symbol may appear in a comment/docstring of the harness/test (Codex LOW: grep the
    call/def form, tolerant of prose mentions).
    """
    from pathlib import Path

    call_token = "throwaway_ev_preview" + "("
    search_roots = ["scripts", "app", "api", "models"]
    offenders: list[str] = []
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for py in root_path.rglob("*.py"):
            text = py.read_text(encoding="utf-8")
            if call_token in text or f"def {call_token}" in text:
                offenders.append(str(py))
    assert not offenders, (
        f"throwaway_ev_preview must not be defined/called in production modules: {offenders}"
    )

    # Positive confirmation: the call/def token IS present in the harness.
    harness = Path("backtest/ou_divergence.py").read_text(encoding="utf-8")
    assert call_token in harness or "def throwaway_ev_preview" in harness, (
        "the harness must define throwaway_ev_preview"
    )


def test_ev_preview_does_not_import_total_converter() -> None:
    """throwaway_ev_preview reuses the converter MATH pattern but does NOT import the class."""
    import inspect

    from backtest.ou_divergence import throwaway_ev_preview

    source = inspect.getsource(throwaway_ev_preview)
    assert "TotalDistributionConverter" not in source, (
        "the throwaway preview must not couple to the production converter class (T-26-08)"
    )


def test_sweep_uses_simulator_not_hand_rolled_grader() -> None:
    """The sweep grades via both_population_hit_rates / BettingSimulator, never a hand-roll."""
    from pathlib import Path

    src = Path("backtest/ou_divergence.py").read_text(encoding="utf-8")
    assert "both_population_hit_rates" in src or "BettingSimulator" in src, (
        "the sweep must consume the LOCKED simulator hit-rate"
    )
    # The forbidden hand-rolled graders (MODEL-DIAGNOSIS.md explicitly warns against these).
    assert "cover_accuracy" not in src, (
        "must not use the non-line-graded cover_accuracy"
    )
    assert "over_accuracy" not in src, "must not use the non-line-graded over_accuracy"
    # The no-slippage survival path uses the sanctioned knob, not hand-rolled half-point math.
    assert "slippage_points=0.0" in src, (
        "the slippage-survival cut must use SimulationConfig(slippage_points=0.0)"
    )


def test_fixture_uses_normalized_odds_loader() -> None:
    """The fixture reuses the normalized-odds loader, not the raw odds parquet (LAR->LA mapping)."""
    test_source = Path(__file__).read_text(encoding="utf-8")
    assert "_load_closing_odds" in test_source
    # The genuine anti-pattern: a read_parquet call on the raw odds snapshot. The token is joined
    # at runtime so this assertion does not match its own literal in the file source.
    raw_odds_antipattern = "read_parquet(" + chr(34) + "data/silver/odds_snapshot"
    assert raw_odds_antipattern not in test_source
