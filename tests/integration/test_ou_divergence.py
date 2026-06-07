"""Wave 0 smoke + number-anchor suite for the O/U divergence harness (Phase 26, plan 26-01).

Mirrors ``tests/integration/test_diag_diagnosis.py`` structure (the canonical diagnosis-harness
smoke pattern, D26-15). Covers the OUM-01 measurement correctness for the FIRST HALF of
``backtest/ou_divergence.py`` -- the integrity preamble (D26-04), the bias-vs-anticipation
decomposition (D26-05, the LEAD section), and the prior-season bias-adjusted re-score (D26-18).
The extended bucket sweep, trial registry, and EV preview are Plan 26-03 (the ``trial_registry``
selector is added there and may be absent here).

Load-bearing number anchors (reproduced this session against the DEPLOYED v1.0 OU artifact
``ou_20260326_163930`` over 2021-2024 canonical gold):
  - pooled line_clv +1.1095 on n=1087 with-line games (52 excluded)
  - model-picks-over share pooled 0.727 with the 2021 sign-flip (18.4% over, line_clv -1.21)
  - pooled residual SD 12.946

Selectors (``-k``): deployed_population, bias_over_share, coverage_counts,
provenance_label_split, no_train_no_write, production_files_untouched, determinism.

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


def test_fixture_uses_normalized_odds_loader() -> None:
    """The fixture reuses the normalized-odds loader, not the raw odds parquet (LAR->LA mapping)."""
    test_source = Path(__file__).read_text(encoding="utf-8")
    assert "_load_closing_odds" in test_source
    # The genuine anti-pattern: a read_parquet call on the raw odds snapshot. The token is joined
    # at runtime so this assertion does not match its own literal in the file source.
    raw_odds_antipattern = "read_parquet(" + chr(34) + "data/silver/odds_snapshot"
    assert raw_odds_antipattern not in test_source
