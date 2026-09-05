"""Synthetic candidate frames for the Phase-31 one-shot runner's guard tests.

ONE builder, shared by four test modules, because three copies of a fixture is exactly the
second-list failure this project keeps paying for: a guard tuned against one copy silently
stops describing the others.

WHY SYNTHETIC AND NOT REAL GOLD. Every test that drives ``run_profitability_2025`` here is a
GUARD test -- it asserts a refusal, an ordering, a token rule or a rendering, none of which is a
property of the data. Reading real gold would make those guards depend on artifacts that are
gitignored, would cost a deployed-artifact scoring pass per run, and would tempt a future author
to point one at the 2025 hold. The frames below span 2018-2024 ONLY: 2025 appears nowhere in
this module, and nothing here can read it.

The integration run in ``tests/integration/test_ev_chains_three_targets.py`` is the one that
exercises the REAL loaders, on the disjoint ``REHEARSAL_PROXY_SPLIT`` (tune 2021-2023, hold
2024) and never on 2025.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 2018-2020 is the strictly-prior bias seed, 2021-2023 the rehearsal tune window and 2024 its
# hold. 2025 IS DELIBERATELY ABSENT: the single unburned season is not reachable from any test
# fixture, which is a stronger guarantee than remembering not to filter it in.
SYNTHETIC_SEASONS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
SYNTHETIC_WEEKS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
SYNTHETIC_GAMES_PER_WEEK: int = 16

# The frozen WP calibration gate needs N_BINS * MIN_BIN_OBS = 100 tune observations, so the
# per-season count is chosen to clear that on the three-season tune window with margin. A frame
# that under-fills the bins makes the gate RAISE rather than fail, which is correct behaviour
# but would turn every token test into a fixture-shape test.
SYNTHETIC_SEED: int = 3112


def build_synthetic_candidates(
    seed: int = SYNTHETIC_SEED,
    *,
    seasons: tuple[int, ...] = SYNTHETIC_SEASONS,
) -> dict[str, pd.DataFrame]:
    """Build one candidate frame per target, in the runner's own column contract.

    Each frame carries what the target's registered selection strategy requires, plus the
    realized value the selector grades against and the provenance columns the OUM-06 hard-fail
    reads. Prices are generated from the same latent probability the outcome is drawn from and
    then vigged, so the market is approximately fair and any measured edge is noise -- which is
    what a guard fixture should be. A fixture with a built-in edge would make a PROFITABLE
    token look like a property of the runner rather than of the data.

    Args:
        seed: The numpy generator seed. Fixed by default so every guard test sees the same
            frames and a failure is reproducible.
        seasons: The seasons to generate. NEVER includes 2025.

    Returns:
        ``{"wp": frame, "ats": frame, "ou": frame}``.
    """
    if 2025 in seasons:
        msg = (
            "the synthetic fixture refuses to generate 2025. The 2025 split is single-use and "
            "unburned; a test fixture that could produce it would make an accidental read "
            "indistinguishable from a deliberate one."
        )
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    rows = [
        {"game_id": f"{season}_W{week:02d}_G{game:02d}", "season": season, "week": week}
        for season in seasons
        for week in SYNTHETIC_WEEKS
        for game in range(SYNTHETIC_GAMES_PER_WEEK)
    ]
    base = pd.DataFrame(rows)
    n = len(base)

    # WP: a latent home-win probability, a slightly noisy model estimate of it, a realized
    # outcome drawn from the latent, and a two-sided moneyline vigged around the latent.
    latent = rng.uniform(0.25, 0.75, n)
    wp = base.copy()
    wp["model_prob"] = np.clip(latent + rng.normal(0.0, 0.03, n), 0.05, 0.95)
    wp["actual"] = (rng.uniform(size=n) < latent).astype(int)
    wp["ml_home"] = [_american(p, favoured=p >= 0.5) for p in latent]
    wp["ml_away"] = [_american(1.0 - p, favoured=p < 0.5) for p in latent]
    wp["target"] = "wp"

    # ATS: a half-point market spread on the HOME-MARGIN scale (positive = home favoured), a
    # noisy model margin, and a realized margin.
    spread = np.round(rng.uniform(-10.0, 10.0, n) * 2.0) / 2.0
    ats = base.copy()
    ats["closing_spread"] = spread
    ats["model_spread"] = spread + rng.normal(0.4, 2.5, n)
    ats["actual"] = spread + rng.normal(0.3, 13.0, n)
    ats["target"] = "ats"

    # O/U: a market total straddling the pre-hold high-total boundary (~48.0) so BOTH arms of
    # the frozen under-OR-high UNION are exercised, a noisy model total, and a realized total.
    total = np.round(rng.uniform(38.0, 56.0, n) * 2.0) / 2.0
    ou = base.copy()
    ou["closing_total"] = total
    ou["model_total"] = total + rng.normal(-0.5, 3.0, n)
    ou["actual"] = total + rng.normal(-0.4, 13.0, n)
    ou["target"] = "ou"

    frames: dict[str, pd.DataFrame] = {}
    for target, frame in (("wp", wp), ("ats", ats), ("ou", ou)):
        frame["sportsbook"] = "consensus"
        frame["is_live"] = False
        frames[target] = frame
    return frames


def _american(probability: float, *, favoured: bool) -> float:
    """A vigged American price for ``probability``, quoted as a favourite or an underdog."""
    probability = float(min(max(probability, 0.05), 0.95))
    if favoured:
        return -100.0 * probability / (1.0 - probability) * 1.02
    return 100.0 * (1.0 - probability) / probability * 0.98
