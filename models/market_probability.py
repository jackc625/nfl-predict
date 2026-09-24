"""The market's win-probability opinion, fitted on lines we actually owned (D33.2-09).

WHY THIS MODULE EXISTS
----------------------
The WP blend used to take its market half from a devigged two-sided CLOSING moneyline --
a price that did not exist at the game's lock -- and to return the unblended model
probability, silently, whenever that moneyline was absent. The owned ``odds_timeline``
(the Phase-29 purchase) carries spreads and totals with real capture times and NO
moneyline of any kind, so neither half of that arrangement can survive D33.2-03. This
module supplies the replacement: a spread-to-win-probability conversion fitted ONLY on
spreads we owned at or before each game's lock, used for blend tuning, for the 2026
edge-threshold derivation and for the live comparison alike.

A BETTING LINE IS STILL NOT A MODEL INPUT
-----------------------------------------
Say this plainly, because the two things look alike from a distance and are not alike at
all. D33.2-03 removed every betting line from the three models' FEATURE SETS, and rung 9
(Plan 33.2-19) removed those columns from gold. Nothing here puts one back. What this
module produces is a BLEND input: the market's own opinion, standing beside the model's,
which the blend then weighs. The model never sees it; it is applied AFTER the model has
predicted. A later reader who finds a spread in this file should read this paragraph
before concluding the fence was breached.

THE FUNCTIONAL FORM: A ONE-PARAMETER LOGISTIC WITH NO INTERCEPT
---------------------------------------------------------------
``P_home = sigmoid(beta * home_fav_margin)``.

The justification is measured, not aesthetic (33.2-RESEARCH.md section 7.3, reproduced
read-only on 2026-09-22 against the same stores):

* the fitted INTERCEPT is small, unstable, and shrinking toward zero as the training set
  grows -- -0.2177 (train 2020) -> -0.1657 -> -0.0913 -> -0.0612 (train 2020-2023);
* a spread of zero being a coin flip is the MARKET'S OWN claim, and the pooled fit agrees
  with it to within a twentieth of a probability point (implied 0.4837 at spread 0);
* n = 1,342 graded games leaves no room for a second parameter, let alone a spline.

This is a CHOICE, not a derivation. RESEARCH assumption A9 is recorded at LOW confidence
and says so in as many words: the walk-forward table supports the no-intercept form, and a
planner could reasonably have kept the intercept. Recorded here so a later reader knows
which it was.

THE SIGN FLIP HAPPENS ONCE, IN ONE NAMED FUNCTION
-------------------------------------------------
``odds_timeline`` stores the OPPOSITE sign from ``odds_snapshot`` and the live ingest
(D33.2-23; measured corr between the two stores -0.9867). MEASURED on 1,342 graded
2020-2024 games: corr(home_win, -timeline_spread) = +0.3922, corr(home_win,
timeline_spread) = -0.3922. So ``home_fav_margin = -spread``, and the flip lives in
:func:`timeline_spread_to_home_fav_margin` alone -- never at a call site, never twice. The
owned timeline has exactly two readers, :func:`load_owned_prelock_lines` (this fit) and
``models.blending_data.select_prelock_lines`` (the blend's tuning corpus, Plan 33.2-24), and
both call that one function, so the convention is known in ONE place rather than restated at
each reader. A SECOND flip is not a silent error here: it drives the fitted slope negative,
and :class:`ImplausibleMarketSlopeError` refuses it by name with the measured value in the
message.

TWO OUTPUTS, NOT ONE
--------------------
A walk-forward loop that returns a single slope invites the wrong one being used
(reviews round ``f924749``, Codex MEDIUM). So the artifact carries BOTH:

* ``slope_beta`` -- the FINAL SERVING slope, fitted over every owned pre-lock season. This
  is what a blend binds and what 2026 serving uses.
* ``walk_forward_slopes`` -- ``{season: slope}``, each fitted on seasons STRICTLY EARLIER
  than its key. The first owned season (2020 on today's corpus) has no entry, because it
  has no prior fold.

Any HISTORICAL consumer must convert through :func:`oof_market_probability`, which uses a
game's own season's prior-only slope. The two consumers this matters for are named here so
neither has to be found: **the blend tuning** (Plan 33.2-24) and **the 2026 edge-threshold
derivation** (Plan 33.2-26). Converting a 2022 game with ``slope_beta`` would convert it
with a slope fitted partly on that game's own outcome.

THE ARTIFACT IS IMMUTABLE AND HAS EXACTLY ONE READER
-----------------------------------------------------
This is a FIT, so it carries a versioned artifact directory rather than living in
``utils/`` as a free function. That distinction is not bookkeeping: the tree already held
an UNVERSIONED spread-to-probability sigmoid with a hardcoded 0.25 slope
(``utils.probability_utils.convert_spread_to_moneyline``), which looked like arithmetic
and was roughly 65% steeper than the owned pre-lock spreads support -- at a +7 spread it
claimed 0.85 where the data says 0.73, which would manufacture an apparent edge at EVERY
spread. It is DELETED in the same commit that adds this module. One answer on disk (D30-02).

:func:`save_market_probability_artifact` always writes a NEW timestamped directory and
REFUSES to write into an existing one, so an artifact id names exactly one payload
forever. That is what lets ``models.blending.MarketBlender`` bind an id rather than
re-resolve one at serving time. :func:`load_market_probability_artifact` is the one reader.

``artifacts/latest.json`` stays FOUR pointers (SPEC R13). The converter is deliberately not
a fifth: it is bound on the blend side, in ``blend_weights.json``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from conf.season_partition import derive_season_partition

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping, Sequence

__all__ = [
    "ARTIFACT_PREFIX",
    "ARTIFACT_VERSION",
    "OWNED_LINE_SEASONS",
    "PAYLOAD_FILENAME",
    "PLAUSIBLE_SLOPE_RANGE",
    "REQUIRED_ARTIFACT_FIELDS",
    "ClosingLineInFitError",
    "ImplausibleMarketSlopeError",
    "MarketProbabilityArtifactError",
    "MarketProbabilityError",
    "MarketProbabilityFit",
    "NoPriorFoldError",
    "SlopeDidNotConvergeError",
    "fit_market_probability",
    "fit_slope",
    "load_market_probability_artifact",
    "load_owned_prelock_lines",
    "market_home_win_probability",
    "oof_market_probability",
    "run_production_fit",
    "save_market_probability_artifact",
    "timeline_spread_to_home_fav_margin",
]

#: Every artifact directory this module writes starts with this. The blend's binding
#: cross-check and the plan's digest bracket both read the prefix, so it is named once.
ARTIFACT_PREFIX: str = "market_probability"

#: The payload file inside an artifact directory. ``metadata.json`` matches the name every
#: other artifact directory under ``artifacts/`` already uses.
PAYLOAD_FILENAME: str = "metadata.json"

#: The artifact schema version. Bumped only when a field's MEANING changes.
ARTIFACT_VERSION: str = "1.0"

#: The fields a payload must carry to be an auditable record of a fit rather than a number
#: on disk. A reader a year from now must be able to say what was fitted, on what, when.
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "version",
    "slope_beta",
    "walk_forward_slopes",
    "training_seasons",
    "n_games",
    "input_digest",
    "fitted_at",
)

#: The band a fitted slope must land in, or the fit REFUSES.
#:
#: MEASURED on the owned pre-lock corpus (33.2-RESEARCH.md 7.3, reproduced 2026-09-22):
#: pooled no-intercept slope 0.15124; prior-only walk-forward slopes 0.15855 (2021),
#: 0.13566 (2022), 0.14295 (2023), 0.14273 (2024). The band brackets all five with margin.
#:
#: What it is FOR: a slope outside it means the join or the sign flip is wrong, not that
#: the market changed. A double flip yields a NEGATIVE slope and is caught here. The
#: acceptance band is acknowledged (reviews round ``f924749``, Codex, deferred by owner
#: scope decision 2026-09-20) to be derived from the same corpus it judges; the sign check
#: it subsumes is independent of the numbers, and a fit outside the band HALTS LOUDLY with
#: the measured value named rather than shipping a silent wrong number.
PLAUSIBLE_SLOPE_RANGE: tuple[float, float] = (0.13, 0.18)

#: The seasons the owned ``odds_timeline`` covers. The Odds API's historical data begins
#: 2020-06, so 2018-2019 are unbuyable at any price; the forward-capture path has never
#: run, so 2025 is empty. This tuple is a COVERAGE FACT about the purchase, not a window
#: rule -- the fold boundaries come from ``conf.season_partition``.
OWNED_LINE_SEASONS: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)

#: Columns whose mere presence means a CLOSING line reached the frame. ``ml_home`` /
#: ``ml_away`` are the two-sided closing moneyline the WP blend used to devig;
#: ``spread_line`` is the nflverse closing spread; the rest are self-describing.
_CLOSING_LINE_COLUMNS: tuple[str, ...] = (
    "ml_home",
    "ml_away",
    "spread_line",
    "closing_spread",
    "closing_total",
    "is_closing",
)

#: The columns a fit frame must carry. ``snapshot_ts`` and ``lock`` are REQUIRED rather
#: than optional: a line with no information time cannot be shown to be pre-lock, and the
#: D33.2-22 ruling of Plan 33.2-14 makes such a line inadmissible.
_REQUIRED_FIT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "home_fav_margin",
    "home_win",
    "snapshot_ts",
    "lock",
)

_NEWTON_MAX_STEPS: int = 200
_NEWTON_TOLERANCE: float = 1e-13


class MarketProbabilityError(Exception):
    """Base class for every refusal this module raises.

    Inherits ``Exception`` and NOT ``ValueError`` / ``KeyError`` / ``RuntimeError`` /
    ``ImportError``, for the reason ``data.sealed_probe_log.SealedProbeLogCorrupt`` records:
    several call sites in this repository catch that tuple and degrade quietly, and a
    converter refusal degraded into "carry on with no market opinion" is precisely the
    silent no-blend this plan exists to abolish.
    """


class ImplausibleMarketSlopeError(MarketProbabilityError):
    """A fitted slope fell outside :data:`PLAUSIBLE_SLOPE_RANGE`.

    The likeliest cause by far is the sign convention: ``odds_timeline`` stores the
    opposite sign from ``odds_snapshot``, so a frame flipped twice (or not at all) fits a
    NEGATIVE slope. The second likeliest is a broken join. Neither is a market fact, and
    neither may be shipped.
    """


class ClosingLineInFitError(MarketProbabilityError):
    """A closing line, or a line timed after its own game's lock, reached the fit.

    SPEC R7 prohibits it outright. Closing lines remain valid for GRADING bets and
    measuring CLV; they may not fit anything a prediction depends on.
    """


class MarketProbabilityArtifactError(MarketProbabilityError):
    """An artifact directory is absent, unreadable, or already exists.

    "Already exists" is in that list deliberately. An artifact id must name exactly one
    payload forever, because a blend binds the id; a fit that could overwrite a directory
    would make the binding meaningless.
    """


class NoPriorFoldError(MarketProbabilityError):
    """A historical game's season has no prior-only slope, so it cannot be converted.

    Raised rather than silently falling back to ``slope_beta``. Falling back would convert
    the game with a slope fitted partly on that game's own outcome, which is the exact
    leak ``walk_forward_slopes`` exists to prevent.
    """


class SlopeDidNotConvergeError(MarketProbabilityError):
    """Newton-Raphson ran out of steps, or produced a non-finite slope (A33.2-review IN-02).

    A separable or degenerate corpus can drive the likelihood's maximum to infinity. The last
    iterate of such a run is not a fitted slope; returning it and relying on the plausibility
    band to catch it would be catching it by accident.
    """


@dataclass(frozen=True)
class MarketProbabilityFit:
    """The two slopes and the provenance that makes them auditable.

    Attributes:
        slope_beta: The SERVING slope, fitted over every owned pre-lock season.
        walk_forward_slopes: ``{season: slope}``, each fitted on strictly earlier seasons.
            The first owned season is absent: it has no prior fold.
        training_seasons: The seasons that actually entered the fit, after the season
            partition rule's completed-season clamp.
        n_games: How many graded games the serving slope was fitted on.
        input_digest: sha256 over the canonical fit rows -- see :func:`_input_digest`.
    """

    slope_beta: float
    walk_forward_slopes: dict[int, float]
    training_seasons: tuple[int, ...]
    n_games: int
    input_digest: str

    def to_payload(self) -> dict[str, Any]:
        """The JSON payload written into the artifact directory."""
        return {
            "version": ARTIFACT_VERSION,
            "slope_beta": self.slope_beta,
            "walk_forward_slopes": {
                str(season): slope
                for season, slope in sorted(self.walk_forward_slopes.items())
            },
            "training_seasons": list(self.training_seasons),
            "n_games": self.n_games,
            "input_digest": self.input_digest,
            "fitted_at": datetime.now(tz=UTC).isoformat(),
        }


# ---------------------------------------------------------------------------
# The functional form
# ---------------------------------------------------------------------------


def market_home_win_probability(
    home_fav_margin: float | Sequence[float] | np.ndarray,
    beta: float,
) -> float | np.ndarray:
    """The market's implied home win probability: ``sigmoid(beta * home_fav_margin)``.

    Args:
        home_fav_margin: The market's margin, POSITIVE when the home team is favoured.
            That is the ``odds_snapshot`` / live-ingest convention;
            :func:`load_owned_prelock_lines` flips ``odds_timeline`` onto it once.
        beta: The fitted slope. A blend passes its BOUND slope and nothing else.

    Returns:
        A ``float`` for scalar input, an ``ndarray`` otherwise. At a margin of exactly
        zero the result is exactly 0.5 for every beta -- a pick'em is the market's own
        coin flip, which is the claim the no-intercept form encodes.
    """
    margin = np.asarray(home_fav_margin, dtype=float)
    probability = 1.0 / (1.0 + np.exp(-float(beta) * margin))
    if margin.ndim == 0:
        return float(probability)
    return probability


def fit_slope(margins: np.ndarray, wins: np.ndarray) -> float:
    """Maximum-likelihood slope of the no-intercept logistic, by Newton-Raphson.

    Hand-rolled rather than pulled from a library because the model has ONE parameter and
    no intercept: ``sklearn.LogisticRegression`` fits an intercept unless told not to and
    regularizes by default (both would silently change the answer), and ``statsmodels`` is
    not a dependency of this project.

    Args:
        margins: Home-favoured margins.
        wins: 1.0 where the home team won, 0.0 where it lost. Ties must already be gone --
            a tie is not an outcome this two-valued model has a place for.

    Returns:
        The fitted slope.

    Raises:
        ClosingLineInFitError: never. This function judges nothing; the admissibility
            checks live in :func:`fit_market_probability`.
        ValueError: when the inputs are empty or of different lengths.
        SlopeDidNotConvergeError: when the iteration does not converge within
            ``_NEWTON_MAX_STEPS`` steps or the slope is not finite.
    """
    x = np.asarray(margins, dtype=float)
    y = np.asarray(wins, dtype=float)
    if x.size == 0:
        msg = "cannot fit a slope on zero games"
        raise ValueError(msg)
    if x.shape != y.shape:
        msg = f"margins {x.shape} and wins {y.shape} must be the same length"
        raise ValueError(msg)

    beta = 0.1
    step = float("nan")
    for _ in range(_NEWTON_MAX_STEPS):
        probability = 1.0 / (1.0 + np.exp(-beta * x))
        gradient = float(np.sum(x * (y - probability)))
        hessian = -float(np.sum(x * x * probability * (1.0 - probability)))
        if hessian == 0.0:
            msg = (
                "the log-likelihood is flat in beta: every margin is zero, so the data "
                "carries no information about the slope"
            )
            raise ValueError(msg)
        step = gradient / hessian
        beta -= step
        if not np.isfinite(beta):
            msg = (
                f"the slope diverged to {beta!r}: the corpus is separable or degenerate, "
                "so the likelihood has no finite maximum"
            )
            raise SlopeDidNotConvergeError(msg)
        if abs(step) < _NEWTON_TOLERANCE:
            return float(beta)
    msg = (
        f"Newton-Raphson did not converge in {_NEWTON_MAX_STEPS} steps (last step "
        f"{step!r}, beta {beta!r}); the last iterate is not a fitted slope"
    )
    raise SlopeDidNotConvergeError(msg)


# ---------------------------------------------------------------------------
# Admissibility
# ---------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [
        column for column in _REQUIRED_FIT_COLUMNS if column not in frame.columns
    ]
    if missing:
        msg = (
            f"the fit frame is missing {missing}; it must carry "
            f"{list(_REQUIRED_FIT_COLUMNS)}. snapshot_ts and lock are required because a "
            "line with no information time cannot be shown to be pre-lock, and an "
            "undatable line is inadmissible (D33.2-22 ruling of Plan 33.2-14)."
        )
        raise ClosingLineInFitError(msg)


def assert_no_closing_line(frame: pd.DataFrame) -> None:
    """Refuse a frame carrying a closing line, by column or by timing.

    Two independent checks, because a closing line can arrive either way:

    1. a CLOSING-LINE COLUMN is present at all (``ml_home`` / ``ml_away`` are the devigged
       moneyline pair the WP blend used to consume; the owned timeline has no moneyline of
       any kind, so their presence means a different store was joined in);
    2. a row's ``snapshot_ts`` is AFTER its own game's ``lock``. At-lock is admissible
       (``<=``), matching ``utils.game_lock.is_admissible``; one second later is not.

    Raises:
        ClosingLineInFitError: naming the offending columns or the offending games.
    """
    _require_columns(frame)

    present = [column for column in _CLOSING_LINE_COLUMNS if column in frame.columns]
    if present:
        msg = (
            f"a closing line reached the fit: the frame carries {present}. SPEC R7 "
            "forbids any closing line in this fit; the owned odds_timeline carries no "
            "moneyline of any kind, so these columns can only have come from a closing "
            "store. Closing lines stay valid for grading bets and measuring CLV."
        )
        raise ClosingLineInFitError(msg)

    snapshot = pd.to_datetime(frame["snapshot_ts"], utc=True)
    lock = pd.to_datetime(frame["lock"], utc=True)
    undated = snapshot.isna() | lock.isna()
    if bool(undated.any()):
        offenders = sorted(map(str, frame.loc[undated, "game_id"]))
        msg = (
            f"{len(offenders)} row(s) carry no information time or no lock and are "
            f"therefore INADMISSIBLE, e.g. {offenders[:10]}. A line with a fabricated or "
            "missing capture time is not a pre-lock line."
        )
        raise ClosingLineInFitError(msg)

    post_lock = snapshot > lock
    if bool(post_lock.any()):
        offenders = sorted(map(str, frame.loc[post_lock, "game_id"]))
        msg = (
            f"{len(offenders)} row(s) are timed AFTER their own game's lock and are "
            f"therefore post-lock lines, e.g. {offenders[:10]}. At-lock information is "
            "admissible; one second later is not (D33.2-01)."
        )
        raise ClosingLineInFitError(msg)


def _require_plausible(slope: float, label: str) -> None:
    low, high = PLAUSIBLE_SLOPE_RANGE
    if not (low <= slope <= high):
        msg = (
            f"{label} fitted to {slope:.6f}, outside the plausible band [{low}, {high}]. "
            "The likeliest cause is the sign convention: odds_timeline stores the "
            "OPPOSITE sign from odds_snapshot (D33.2-23, corr -0.9867), so a frame "
            "flipped twice or not at all fits a negative slope. The second likeliest is "
            "a broken join. MEASURED on the owned corpus: pooled 0.15124, prior-only "
            "walk-forward 0.13566 to 0.15855. Refusing rather than shipping."
        )
        raise ImplausibleMarketSlopeError(msg)


def _input_digest(frame: pd.DataFrame) -> str:
    """sha256 over the canonical fit rows.

    Not ``digest_artifact_dir``: the input here is not a directory but a DERIVED corpus,
    and digesting the silver parquet would record the bytes of a file rather than the rows
    that were actually fitted (the lock join and the tie drop both sit in between). The
    rows are serialized sorted by ``game_id`` with fixed precision, so the digest is a
    function of the fit's CONTENT alone and reproduces on any machine.
    """
    ordered = frame.sort_values("game_id")
    lines = [
        f"{row.game_id}\t{int(row.season)}\t{float(row.home_fav_margin):.6f}\t"
        f"{int(row.home_win)}"
        for row in ordered.itertuples()
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def fit_market_probability(frame: pd.DataFrame) -> MarketProbabilityFit:
    """Fit the converter on owned pre-lock spreads, expanding-window walk-forward.

    The fold boundaries come from ``conf.season_partition.derive_season_partition`` --
    that module IS the window rule, and a second season arithmetic here would be exactly
    the duplicated-rule failure D30-02 names. Two consequences fall out of using it rather
    than a literal: seasons outside ``[CORPUS_FIRST_SEASON, LATEST_COMPLETED_SEASON]`` are
    dropped, so a LIVE, incomplete season can never enter a fit; and the widened selection
    window (D33.2-14) is inherited automatically.

    Args:
        frame: Owned pre-lock lines. Must carry ``game_id``, ``season``,
            ``home_fav_margin``, ``home_win``, ``snapshot_ts`` and ``lock``.

    Returns:
        The :class:`MarketProbabilityFit`.

    Raises:
        ClosingLineInFitError: when a closing line, an undated line or a post-lock line
            reached the frame.
        ImplausibleMarketSlopeError: when the serving slope or ANY walk-forward slope
            falls outside :data:`PLAUSIBLE_SLOPE_RANGE`.
    """
    assert_no_closing_line(frame)

    partition = derive_season_partition(frame["season"])
    usable = frame[frame["season"].isin(partition.final_fit)].copy()
    if usable.empty:
        msg = (
            "no row survives the season partition rule's completed-season clamp: the "
            f"frame carries {sorted(set(frame['season']))} and the rule admits "
            f"{list(partition.final_fit)}"
        )
        raise ValueError(msg)

    margins = usable["home_fav_margin"].to_numpy(dtype=float)
    wins = usable["home_win"].to_numpy(dtype=float)

    slope_beta = fit_slope(margins, wins)
    _require_plausible(slope_beta, "the serving slope (slope_beta)")

    training_seasons = tuple(sorted({int(season) for season in usable["season"]}))
    walk_forward_slopes: dict[int, float] = {}
    for season in training_seasons:
        prior = usable[usable["season"] < season]
        if prior.empty:
            # The first owned season has no prior fold and therefore no out-of-fold
            # slope. Absent from the mapping, never filled with slope_beta.
            continue
        slope = fit_slope(
            prior["home_fav_margin"].to_numpy(dtype=float),
            prior["home_win"].to_numpy(dtype=float),
        )
        _require_plausible(slope, f"the {season} walk-forward slope")
        walk_forward_slopes[season] = slope

    return MarketProbabilityFit(
        slope_beta=slope_beta,
        walk_forward_slopes=walk_forward_slopes,
        training_seasons=training_seasons,
        n_games=len(usable),
        input_digest=_input_digest(usable),
    )


def oof_market_probability(
    frame: pd.DataFrame,
    walk_forward_slopes: Mapping[int | str, float],
) -> np.ndarray:
    """Convert HISTORICAL games, each with its own season's prior-only slope.

    Every historical consumer -- the blend tuning (Plan 33.2-24) and the 2026
    edge-threshold derivation (Plan 33.2-26) -- goes through here rather than through
    ``slope_beta``. Converting a game with the serving slope would convert it with a slope
    fitted partly on that game's own outcome.

    *walk_forward_slopes* is a REQUIRED argument rather than something this function looks
    up, so passing ``slope_beta`` by accident is not a shape the call site can take.

    Args:
        frame: Rows carrying ``season`` and ``home_fav_margin``.
        walk_forward_slopes: The artifact's ``walk_forward_slopes``. String keys (as JSON
            stores them) and integer keys are both accepted.

    Returns:
        One market probability per row, in the frame's own order.

    Raises:
        NoPriorFoldError: when any row's season has no prior-only slope.
    """
    slopes = {
        int(season): float(slope) for season, slope in walk_forward_slopes.items()
    }
    seasons = [int(season) for season in frame["season"]]

    unknown = sorted({season for season in seasons if season not in slopes})
    if unknown:
        msg = (
            f"season(s) {unknown} have no prior-only slope, so their games cannot be "
            f"converted out of fold (available: {sorted(slopes)}). The first owned season "
            "has no earlier season to fit on. Refusing rather than falling back to the "
            "serving slope, which was fitted partly on these games' own outcomes."
        )
        raise NoPriorFoldError(msg)

    margins = frame["home_fav_margin"].to_numpy(dtype=float)
    betas = np.array([slopes[season] for season in seasons], dtype=float)
    return 1.0 / (1.0 + np.exp(-betas * margins))


# ---------------------------------------------------------------------------
# The owned corpus
# ---------------------------------------------------------------------------


def timeline_spread_to_home_fav_margin(
    spread: pd.Series | np.ndarray,
) -> np.ndarray:
    """The owned ``odds_timeline`` spread, flipped onto the home-margin scale. THE ONE FLIP.

    ``odds_timeline`` stores the OPPOSITE sign from ``odds_snapshot`` and the live ingest
    (D33.2-23). The returned values are POSITIVE when the home team is favoured -- the
    ``home_fav_margin`` scale :func:`market_home_win_probability` documents and the scale the
    ATS blend's market spread is on. MEASURED 2026-09-22 on the 1,342 graded games the
    converter fits on: corr(home_fav_margin, home_win) = +0.3922, and -0.3922 unflipped.

    Every reader of the owned timeline calls this rather than negating inline, so a reader
    cannot silently disagree with another about the convention.
    """
    return -np.asarray(spread, dtype=float)


def load_owned_prelock_lines(
    silver_dir: Path | str = Path("data/silver"),
) -> pd.DataFrame:
    """The owned pre-lock spreads, joined to outcomes, with the sign flipped ONCE.

    What this reader does, in order:

    1. loads silver ``odds_timeline`` (the owned Phase-29 purchase -- the ONLY store in
       this project with real capture times) and silver ``games``;
    2. derives each game's lock through ``utils.game_lock.lock_frame``, which IS the rule;
    3. keeps snapshots at or before that game's own lock (``<=``, at-lock admissible);
    4. takes the LAST such snapshot per game -- the most recent honest opinion;
    5. FLIPS THE SIGN through :func:`timeline_spread_to_home_fav_margin`, the one flip.
       MEASURED 2026-09-22 on the 1,342 graded games this returns: corr(home_fav_margin,
       home_win) = +0.3922, and -0.3922 as stored (D33.2-23);
    6. drops ties, which the two-valued model has no place for.

    Returns:
        A frame carrying ``game_id``, ``season``, ``home_fav_margin``, ``home_win``,
        ``snapshot_ts`` and ``lock``. No moneyline column and no closing line: the store
        it reads has neither.
    """
    from utils.game_lock import lock_frame

    root = Path(silver_dir)
    timeline = pd.read_parquet(root / "odds_timeline.parquet")
    games = pd.read_parquet(root / "games.parquet")

    scheduled = games[games["season"].isin(OWNED_LINE_SEASONS)].copy()
    locks = lock_frame(scheduled)

    merged = timeline.merge(
        scheduled[["game_id", "season", "home_score", "away_score"]],
        on="game_id",
        how="inner",
    )
    merged["lock"] = pd.to_datetime(merged["game_id"].map(locks), utc=True)
    merged["snapshot_ts"] = pd.to_datetime(merged["snapshot_ts"], utc=True)

    pre_lock = merged[merged["snapshot_ts"] <= merged["lock"]]
    latest = (
        pre_lock.sort_values("snapshot_ts").groupby("game_id", as_index=False).tail(1)
    )

    graded = latest.dropna(subset=["spread", "home_score", "away_score"])
    graded = graded[graded["home_score"] != graded["away_score"]]

    return pd.DataFrame(
        {
            "game_id": graded["game_id"].astype(str).to_numpy(),
            "season": graded["season"].astype(int).to_numpy(),
            # THE ONE FLIP, through the one named function. See step 5 above.
            "home_fav_margin": timeline_spread_to_home_fav_margin(graded["spread"]),
            "home_win": (graded["home_score"] > graded["away_score"])
            .astype(int)
            .to_numpy(),
            "snapshot_ts": graded["snapshot_ts"].to_numpy(),
            "lock": graded["lock"].to_numpy(),
        }
    ).sort_values("game_id", ignore_index=True)


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------


def save_market_probability_artifact(
    fit: MarketProbabilityFit,
    artifacts_dir: Path | str = Path("artifacts"),
    artifact_id: str | None = None,
) -> Path:
    """Write *fit* into a NEW ``market_probability_*`` directory. Never into an old one.

    Args:
        fit: The fit to record.
        artifacts_dir: The artifacts root.
        artifact_id: The directory name. Defaults to ``market_probability_{timestamp}``.
            Supplying one that already exists RAISES -- that is the property that makes an
            id name one payload forever, which is what lets a blend bind it.

    Returns:
        The created directory.

    Raises:
        MarketProbabilityArtifactError: when the target directory already exists.
    """
    root = Path(artifacts_dir)
    name = artifact_id or (
        f"{ARTIFACT_PREFIX}_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    directory = root / name
    if directory.exists():
        msg = (
            f"refusing to write into the existing artifact directory {directory}: an "
            "artifact id names exactly ONE payload forever, because a blend binds the id "
            "rather than re-resolving it. Write a new directory instead."
        )
        raise MarketProbabilityArtifactError(msg)

    directory.mkdir(parents=True)

    # The SAME atomic-write helper ``update_manifest`` and ``save_blend_artifacts`` use,
    # so there is one atomic-write implementation in this repository rather than three.
    from models.artifacts import _atomic_write_json

    _atomic_write_json(directory / PAYLOAD_FILENAME, fit.to_payload())
    return directory


def load_market_probability_artifact(
    artifact_id: str,
    artifacts_dir: Path | str = Path("artifacts"),
) -> dict[str, Any]:
    """The payload of EXACTLY the named directory. The one reader of a converter artifact.

    No "latest converter directory" lookup exists anywhere in this module, deliberately:
    resolving a converter at serving time is how training and serving come to use
    different ones (reviews round ``f924749``, Codex HIGH).

    Raises:
        MarketProbabilityArtifactError: when the directory or its payload is absent.
    """
    payload_path = Path(artifacts_dir) / artifact_id / PAYLOAD_FILENAME
    if not payload_path.exists():
        msg = (
            f"market-probability artifact {artifact_id!r} not found: {payload_path} does "
            "not exist. An absent artifact is not an empty one."
        )
        raise MarketProbabilityArtifactError(msg)
    return json.loads(payload_path.read_text(encoding="utf-8"))


def run_production_fit(
    artifacts_dir: Path | str = Path("artifacts"),
    silver_dir: Path | str = Path("data/silver"),
) -> Path:
    """Fit on the owned corpus and write ONE new artifact directory.

    This is the only function in this module that writes under production ``artifacts/``,
    and the plan that introduces it brackets that write with a
    ``tests/data_boundary.py`` digest snapshot. ``artifacts/latest.json`` is NOT touched:
    the converter is bound on the blend side, not through the manifest (SPEC R13).

    THE THREAD PIN (Plan 33.2-24 step 24b, the owner's rule for every fit that produces a
    published number). The Newton steps reduce over the corpus through numpy's dot, which
    a multi-threaded BLAS may sum in a thread-count-dependent order; the pool is pinned to
    ``config.tuning_preregistration.PINNED_THREAD_COUNT`` -- the SAME value the blend fit
    and the model re-fits use -- so the slope's last bits are a function of the data alone.
    """
    from threadpoolctl import threadpool_limits

    from config.tuning_preregistration import PINNED_THREAD_COUNT

    with threadpool_limits(limits=PINNED_THREAD_COUNT):
        fit = fit_market_probability(load_owned_prelock_lines(silver_dir))
    return save_market_probability_artifact(fit, artifacts_dir=artifacts_dir)


def main(argv: list[str] | None = None) -> int:
    """``python -m models.market_probability --fit``."""
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument(
        "--fit",
        action="store_true",
        help="fit on the owned pre-lock corpus and write a new artifact directory",
    )
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--silver-dir", default="data/silver")
    args = parser.parse_args(argv)

    if not args.fit:
        parser.print_help()
        return 2

    directory = run_production_fit(
        artifacts_dir=args.artifacts_dir, silver_dir=args.silver_dir
    )
    payload = json.loads((directory / PAYLOAD_FILENAME).read_text(encoding="utf-8"))
    print(f"MARKET_PROB_ARTIFACT= {directory.name}")  # noqa: T201
    print(f"SLOPE_BETA= {payload['slope_beta']}")  # noqa: T201
    print(f"N_GAMES= {payload['n_games']}")  # noqa: T201
    print(f"TRAINING_SEASONS= {payload['training_seasons']}")  # noqa: T201
    print(f"WALK_FORWARD_SLOPES= {payload['walk_forward_slopes']}")  # noqa: T201
    print(f"INPUT_DIGEST= {payload['input_digest']}")  # noqa: T201
    from config.tuning_preregistration import PINNED_THREAD_COUNT

    print(f"THREAD_LIMIT= {PINNED_THREAD_COUNT}")  # noqa: T201
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main in tests
    sys.exit(main())
