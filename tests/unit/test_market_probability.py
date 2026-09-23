"""The fitted spread-to-win-probability converter (Plan 33.2-21 Task 1, D33.2-09).

What these tests pin, and why each one exists:

* the FUNCTIONAL FORM -- exactly 0.5 at a zero margin for any slope, monotone, bounded --
  because the whole point of the no-intercept logistic is that a pick'em game is the
  market's own coin flip;
* the SIGN CONVENTION -- a deliberately double-flipped input must RAISE, not ship. The
  owned ``odds_timeline`` stores the OPPOSITE sign from ``odds_snapshot`` (D33.2-23,
  measured corr -0.9867), so the one flip at the reader is the single place that can be
  wrong, and a second flip produces a NEGATIVE slope;
* the CLOSING-LINE PROHIBITION (SPEC R7) -- the fit refuses a frame carrying closing-line
  columns and refuses a row whose information time is after its own game's lock;
* the TWO DISTINCT OUTPUTS -- ``slope_beta`` (serving, fitted over every owned season) and
  ``walk_forward_slopes`` (per season, fitted on strictly earlier seasons). A historical
  game converted with ``slope_beta`` would be converted by a slope fitted partly on its own
  outcome, which is what ``oof_market_probability`` exists to make impossible;
* the ARTIFACT'S IMMUTABILITY -- a converter id names ONE payload forever, so a blend can
  bind an id rather than re-resolve one (reviews round ``f924749``, Codex HIGH);
* the DELETION of ``utils.probability_utils.convert_spread_to_moneyline``, by a PARSED
  identifier scan with a PLANTED-VIOLATION control, so the scan is proved to be capable of
  failing rather than merely observed to pass.

Every write in this module goes to ``tmp_path``. Nothing here touches ``artifacts/`` -- the
production fit is run once, by the plan's own ``<verify>``, inside a declared digest bracket.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.market_probability import (
    PLAUSIBLE_SLOPE_RANGE,
    REQUIRED_ARTIFACT_FIELDS,
    ClosingLineInFitError,
    ImplausibleMarketSlopeError,
    MarketProbabilityArtifactError,
    NoPriorFoldError,
    fit_market_probability,
    fit_slope,
    load_market_probability_artifact,
    load_owned_prelock_lines,
    market_home_win_probability,
    oof_market_probability,
    save_market_probability_artifact,
)

# The identifier this plan DELETES. Named here as a string rather than imported, for the
# obvious reason.
_DEAD_CONVERTER = "convert_spread_to_moneyline"

#: The repo roots the deletion scan walks. ``tests`` is deliberately excluded: this module
#: names the dead identifier in prose, and the planted-violation control below writes a
#: file that defines it.
_SCAN_ROOTS = (
    "scripts",
    "features",
    "backtest",
    "models",
    "utils",
    "conf",
    "api",
    "pipeline",
    "data",
    "ratings",
)

#: A margin grid with a zero point, so the synthetic corpus below carries the pick'em case.
_MARGIN_GRID = (-14.0, -10.0, -7.0, -3.0, 0.0, 3.0, 7.0, 10.0, 14.0)

#: Games per margin per season in the synthetic corpus. Large enough that the rounded
#: win counts reproduce the generating slope to ~1e-3, which is what lets these tests
#: assert a slope rather than merely a sign.
_GAMES_PER_CELL = 200


def _synthetic_season(season: int, beta: float) -> pd.DataFrame:
    """One season of owned pre-lock lines whose MLE slope is (essentially) *beta*.

    Grouped-binomial construction rather than a random draw: at each margin the win count
    is ``round(k * sigmoid(beta * m))``, so the score equation ``sum x (y - p) = 0`` is
    satisfied at *beta* up to the rounding. No seed, no luck, and the same numbers on every
    machine.
    """
    rows: list[dict[str, object]] = []
    lock = pd.Timestamp(f"{season}-09-10 22:00:00", tz="UTC")
    index = 0
    for margin in _MARGIN_GRID:
        wins = round(_GAMES_PER_CELL / (1.0 + math.exp(-beta * margin)))
        for i in range(_GAMES_PER_CELL):
            rows.append(
                {
                    "game_id": f"{season}_W{(index % 17) + 1:02d}_AAA{index}@BBB{index}",
                    "season": season,
                    "home_fav_margin": margin,
                    "home_win": 1 if i < wins else 0,
                    "snapshot_ts": lock - pd.Timedelta(hours=6),
                    "lock": lock,
                }
            )
            index += 1
    return pd.DataFrame(rows)


def _synthetic_corpus(betas: dict[int, float]) -> pd.DataFrame:
    """The owned pre-lock corpus, one season per entry of *betas*."""
    return pd.concat(
        [_synthetic_season(season, beta) for season, beta in sorted(betas.items())],
        ignore_index=True,
    )


#: Five seasons, because ``conf.season_partition.derive_season_partition`` needs at least
#: four completed seasons to produce a partition at all. 2020 carries a DIFFERENT slope
#: from the rest, which is what makes "fitted on strictly earlier seasons" observable.
_CORPUS_BETAS = {2020: 0.170, 2021: 0.135, 2022: 0.150, 2023: 0.150, 2024: 0.150}


@pytest.fixture()
def corpus() -> pd.DataFrame:
    return _synthetic_corpus(_CORPUS_BETAS)


# ---------------------------------------------------------------------------
# The functional form
# ---------------------------------------------------------------------------


class TestFunctionalForm:
    """P_home = sigmoid(beta * home_fav_margin), with no intercept."""

    @pytest.mark.parametrize("beta", [0.01, 0.13, 0.1512, 0.18, 5.0])
    def test_zero_margin_is_exactly_one_half_for_any_beta(self, beta: float) -> None:
        assert market_home_win_probability(0.0, beta) == 0.5

    def test_scalar_input_returns_a_plain_float(self) -> None:
        value = market_home_win_probability(3.0, 0.1512)
        assert isinstance(value, float)

    def test_monotone_and_bounded(self) -> None:
        margins = np.linspace(-30.0, 30.0, 121)
        probs = market_home_win_probability(margins, 0.1512)
        assert np.all(np.diff(probs) > 0.0)
        assert np.all(probs > 0.0)
        assert np.all(probs < 1.0)

    def test_symmetric_about_the_pick_em(self) -> None:
        assert market_home_win_probability(7.0, 0.1512) + market_home_win_probability(
            -7.0, 0.1512
        ) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# The walk-forward fit
# ---------------------------------------------------------------------------


class TestFit:
    def test_pooled_slope_lands_in_the_plausible_band(
        self, corpus: pd.DataFrame
    ) -> None:
        fit = fit_market_probability(corpus)
        low, high = PLAUSIBLE_SLOPE_RANGE
        assert low <= fit.slope_beta <= high

    def test_slope_recovers_the_generating_value(self) -> None:
        frame = _synthetic_corpus(dict.fromkeys(range(2020, 2025), 0.1512))
        fit = fit_market_probability(frame)
        assert fit.slope_beta == pytest.approx(0.1512, abs=2e-3)

    def test_training_seasons_and_n_games_describe_the_fitted_rows(
        self, corpus: pd.DataFrame
    ) -> None:
        fit = fit_market_probability(corpus)
        assert fit.training_seasons == (2020, 2021, 2022, 2023, 2024)
        assert fit.n_games == len(corpus)

    def test_a_double_flipped_input_raises_rather_than_shipping(
        self, corpus: pd.DataFrame
    ) -> None:
        flipped = corpus.copy()
        flipped["home_fav_margin"] = -flipped["home_fav_margin"]
        with pytest.raises(ImplausibleMarketSlopeError) as excinfo:
            fit_market_probability(flipped)
        # The refusal NAMES the measured value, so the reader is told what was wrong
        # rather than merely that something was.
        assert "-0.1" in str(excinfo.value)

    def test_walk_forward_slope_is_fitted_on_strictly_earlier_seasons(
        self, corpus: pd.DataFrame
    ) -> None:
        fit = fit_market_probability(corpus)
        prior_only = corpus[corpus["season"] < 2021]
        expected = fit_slope(
            prior_only["home_fav_margin"].to_numpy(dtype=float),
            prior_only["home_win"].to_numpy(dtype=float),
        )
        assert fit.walk_forward_slopes[2021] == pytest.approx(expected, abs=1e-12)
        # ... and it is NOT the serving slope, which is the whole point of carrying both.
        assert fit.walk_forward_slopes[2021] != pytest.approx(fit.slope_beta, abs=1e-6)

    def test_the_first_owned_season_has_no_walk_forward_slope(
        self, corpus: pd.DataFrame
    ) -> None:
        fit = fit_market_probability(corpus)
        assert 2020 not in fit.walk_forward_slopes
        assert sorted(fit.walk_forward_slopes) == [2021, 2022, 2023, 2024]

    def test_seasons_outside_the_partition_rule_are_dropped(
        self, corpus: pd.DataFrame
    ) -> None:
        # A live, INCOMPLETE season must never enter a fit. The rule's own
        # completed-season clamp is what drops it -- there is no second season
        # arithmetic in this module.
        live = _synthetic_season(2026, 0.150)
        fit = fit_market_probability(pd.concat([corpus, live], ignore_index=True))
        assert 2026 not in fit.training_seasons
        assert fit.n_games == len(corpus)


class TestTheFitNeverSeesAClosingLine:
    def test_a_closing_moneyline_column_is_refused(self, corpus: pd.DataFrame) -> None:
        contaminated = corpus.copy()
        contaminated["ml_home"] = -150
        contaminated["ml_away"] = 130
        with pytest.raises(ClosingLineInFitError):
            fit_market_probability(contaminated)

    def test_a_post_lock_snapshot_is_refused_and_names_the_game(
        self, corpus: pd.DataFrame
    ) -> None:
        contaminated = corpus.copy()
        offender = str(contaminated.loc[0, "game_id"])
        contaminated.loc[0, "snapshot_ts"] = contaminated.loc[0, "lock"] + pd.Timedelta(
            seconds=1
        )
        with pytest.raises(ClosingLineInFitError) as excinfo:
            fit_market_probability(contaminated)
        assert offender in str(excinfo.value)

    def test_an_at_lock_snapshot_is_admissible(self, corpus: pd.DataFrame) -> None:
        at_lock = corpus.copy()
        at_lock["snapshot_ts"] = at_lock["lock"]
        fit = fit_market_probability(at_lock)
        assert fit.n_games == len(corpus)


# ---------------------------------------------------------------------------
# Out-of-fold conversion
# ---------------------------------------------------------------------------


class TestOofMarketProbability:
    def test_each_season_is_converted_with_its_own_prior_only_slope(
        self, corpus: pd.DataFrame
    ) -> None:
        fit = fit_market_probability(corpus)
        graded = corpus[corpus["season"] > 2020].reset_index(drop=True)
        probs = oof_market_probability(graded, fit.walk_forward_slopes)
        expected = np.array(
            [
                market_home_win_probability(
                    float(row.home_fav_margin),
                    fit.walk_forward_slopes[int(row.season)],
                )
                for row in graded.itertuples()
            ]
        )
        np.testing.assert_allclose(probs, expected, rtol=0, atol=0)

    def test_it_never_converts_with_the_serving_slope(
        self, corpus: pd.DataFrame
    ) -> None:
        fit = fit_market_probability(corpus)
        season_2021 = corpus[corpus["season"] == 2021].reset_index(drop=True)
        probs = oof_market_probability(season_2021, fit.walk_forward_slopes)
        with_serving = market_home_win_probability(
            season_2021["home_fav_margin"].to_numpy(dtype=float), fit.slope_beta
        )
        assert not np.allclose(probs, with_serving)

    def test_a_season_with_no_prior_fold_raises(self, corpus: pd.DataFrame) -> None:
        fit = fit_market_probability(corpus)
        first_season = corpus[corpus["season"] == 2020].reset_index(drop=True)
        with pytest.raises(NoPriorFoldError) as excinfo:
            oof_market_probability(first_season, fit.walk_forward_slopes)
        assert "2020" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------


class TestArtifact:
    def test_payload_carries_every_required_field(
        self, corpus: pd.DataFrame, tmp_path: Path
    ) -> None:
        fit = fit_market_probability(corpus)
        directory = save_market_probability_artifact(fit, artifacts_dir=tmp_path)
        payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        assert set(REQUIRED_ARTIFACT_FIELDS) - set(payload) == set()
        assert directory.name.startswith("market_probability")
        assert payload["slope_beta"] == pytest.approx(fit.slope_beta, abs=0.0)
        assert sorted(int(s) for s in payload["walk_forward_slopes"]) == [
            2021,
            2022,
            2023,
            2024,
        ]

    def test_latest_json_is_not_touched(
        self, corpus: pd.DataFrame, tmp_path: Path
    ) -> None:
        # The converter is deliberately NOT a fifth latest.json pointer (R13). A blend
        # binds its id; nothing resolves "the newest converter directory".
        fit = fit_market_probability(corpus)
        save_market_probability_artifact(fit, artifacts_dir=tmp_path)
        assert not (tmp_path / "latest.json").exists()

    def test_a_second_fit_refuses_to_write_into_an_existing_directory(
        self, corpus: pd.DataFrame, tmp_path: Path
    ) -> None:
        fit = fit_market_probability(corpus)
        directory = save_market_probability_artifact(fit, artifacts_dir=tmp_path)
        before = (directory / "metadata.json").read_bytes()

        with pytest.raises(MarketProbabilityArtifactError):
            save_market_probability_artifact(
                fit, artifacts_dir=tmp_path, artifact_id=directory.name
            )

        assert (directory / "metadata.json").read_bytes() == before

    def test_the_loader_returns_exactly_the_named_payload(
        self, corpus: pd.DataFrame, tmp_path: Path
    ) -> None:
        first = save_market_probability_artifact(
            fit_market_probability(corpus), artifacts_dir=tmp_path
        )
        other_fit = fit_market_probability(
            _synthetic_corpus(dict.fromkeys(range(2020, 2025), 0.170))
        )
        second = save_market_probability_artifact(
            other_fit, artifacts_dir=tmp_path, artifact_id="market_probability_planted"
        )

        assert load_market_probability_artifact(first.name, tmp_path)[
            "slope_beta"
        ] != pytest.approx(
            load_market_probability_artifact(second.name, tmp_path)["slope_beta"]
        )

    def test_the_loader_raises_on_a_missing_id(self, tmp_path: Path) -> None:
        with pytest.raises(MarketProbabilityArtifactError) as excinfo:
            load_market_probability_artifact("market_probability_absent", tmp_path)
        assert "market_probability_absent" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The owned corpus, read-only
# ---------------------------------------------------------------------------


class TestTheOwnedCorpus:
    """Read-only against production silver. Writes nothing."""

    def test_the_reader_reproduces_the_measured_owned_corpus(self) -> None:
        if not Path("data/silver/odds_timeline.parquet").exists():
            pytest.skip("owned odds_timeline is not present in this checkout")
        frame = load_owned_prelock_lines()
        # MEASURED 2026-09-15 and re-measured 2026-09-22: 1,346 of 1,408 scheduled
        # 2020-2024 games carry a snapshot at or before their lock; 1,342 after ties.
        assert len(frame) == 1342
        assert set(frame["season"]) == {2020, 2021, 2022, 2023, 2024}
        correlation = float(
            np.corrcoef(frame["home_fav_margin"], frame["home_win"])[0, 1]
        )
        assert correlation == pytest.approx(0.3922, abs=5e-4)

    def test_the_fitted_slope_lands_in_the_plausible_band(self) -> None:
        if not Path("data/silver/odds_timeline.parquet").exists():
            pytest.skip("owned odds_timeline is not present in this checkout")
        fit = fit_market_probability(load_owned_prelock_lines())
        low, high = PLAUSIBLE_SLOPE_RANGE
        assert low <= fit.slope_beta <= high
        assert fit.slope_beta == pytest.approx(0.1512, abs=1e-3)
        assert all(low <= slope <= high for slope in fit.walk_forward_slopes.values())


# ---------------------------------------------------------------------------
# The dead 0.25-slope converter is gone
# ---------------------------------------------------------------------------


def _identifiers(tree: ast.AST) -> set[str]:
    """Every NAME the module binds or reads -- parsed, never grepped.

    A docstring that explains why the 0.25-slope converter was deleted has to name it.
    Matching on source text would make that explanation impossible to write; matching on
    the parsed identifier set does not.
    """
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {
        alias.asname or alias.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for alias in n.names
    }
    names |= {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    return names


def _files_naming(identifier: str, files: list[Path]) -> list[str]:
    hits: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if identifier in _identifiers(tree):
            hits.append(path.as_posix())
    return sorted(hits)


class TestTheWronglyCalibratedConverterIsGone:
    def test_the_attribute_no_longer_exists(self) -> None:
        from utils import probability_utils

        assert not hasattr(probability_utils, _DEAD_CONVERTER)
        # Its equally dead sibling goes with it: both had zero call sites.
        assert not hasattr(probability_utils, "convert_total_to_over_under_ml")

    def test_no_module_in_the_tree_names_it(self) -> None:
        files = [path for root in _SCAN_ROOTS for path in Path(root).rglob("*.py")]
        assert files, "the scan found no modules -- a vacuous pass is not a pass"
        assert _files_naming(_DEAD_CONVERTER, files) == []

    def test_the_scan_fails_on_a_planted_violation(self, tmp_path: Path) -> None:
        """The control. A scan that cannot fail has not passed."""
        planted = tmp_path / "planted.py"
        planted.write_text(
            "def convert_spread_to_moneyline(spread):\n"
            "    return 1 / (1 + 2.718 ** (-spread * 0.25))\n",
            encoding="utf-8",
        )
        assert _files_naming(_DEAD_CONVERTER, [planted]) == [planted.as_posix()]
