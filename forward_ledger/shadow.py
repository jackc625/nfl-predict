"""The frozen shadow arm: the live decision's own inputs, decided again by the pinned row-19 rule.

WHY THE SHADOW ARM RE-SCORES THE LIVE BUNDLE (D-04, RFIT-03)
------------------------------------------------------------
Phase 37 compares a re-fitted live arm with a frozen arm that keeps today's row-19 models and bet
rule all season. The comparison is only fair when both arms decided the SAME games from the SAME
inputs at the SAME instant. So the shadow arm never builds a second decision from disk: it takes
the live :class:`~backtest.weekly_bet_list.DecisionBundle` -- its candidates, schedule, gold rows
and run instant -- re-scores the candidates with the pinned model ids through the one re-score
routine (``forward_ledger.rescore``), and runs the one selection path with the row-19 bet rule.

While both arms use the row-19 models (the A/A weeks before the first re-fit), the shadow bundle is
content-identical to the live one, so its decision-snapshot digest is identical too. That needs the
same ``decided_at``, the same chain-fit path text and the same resolved ids (research Pitfall 5),
which is why the run instant and the schedule are carried over and the chain-fit path is the pin's
own repo-relative text.

THE PIN, NEVER THE PRODUCTION MANIFEST (RFIT-04, P2)
----------------------------------------------------
The ids come from ``backtest.shadow_pin_2026``, a committed constants module nothing writes. Every
scorer call and the blend-slope read pass an explicit ``version=``, so this module never resolves an
artifact through ``latest.json`` and works in an artifacts directory that holds none. Before use it
refuses a chain-fit record that does not hash to the pinned sha256 and a pinned blend bound to any
converter but the pinned one (:class:`ShadowPinMismatchError`).

This module writes nothing; the runner stamps and stores the rows it returns.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from api.cache import RUN_MODE_FORWARD
from backtest.diagnose import score_deployed_artifacts
from backtest.neutral_hfa_cold_start_constants import EDGE_TIER_THRESHOLDS_BY_TARGET
from backtest.shadow_pin_2026 import (
    SHADOW_BLEND,
    SHADOW_CHAIN_FIT_PATH,
    SHADOW_CHAIN_FIT_SHA256,
    SHADOW_CONVERTER,
    SHADOW_MODEL_IDS,
    SHADOW_RECIPE,
)
from backtest.weekly_bet_list import (
    DecisionBundle,
    load_frozen_chain_fit,
    records_to_bet_list_frame,
    select_weekly_bets,
)
from forward_ledger.rescore import (
    Scorer,
    reattach_spread_probability,
    rescore_candidates,
)
from models.artifacts import ResolvedArtifacts

__all__ = [
    "SHADOW_COMMIT_HEADROOM",
    "SHADOW_PIN",
    "ShadowDeadlineHeadroomError",
    "ShadowPin",
    "ShadowPinMismatchError",
    "build_shadow_bundle",
]

# The least time that must remain before the slate's publish deadline once the shadow rows are
# built: less, and the shadow rows are dropped so the live write cannot be refused for lateness.
SHADOW_COMMIT_HEADROOM: timedelta = timedelta(seconds=120)


class ShadowPinMismatchError(Exception):
    """The files on disk are not the ones the pin names (chain-fit sha256 or bound converter).

    Inherits bare ``Exception`` (the ``data.graded_weeks`` rule): a frozen arm deciding with
    anything but its pinned inputs must never be degraded into a quiet pass.
    """


class ShadowDeadlineHeadroomError(Exception):
    """Too little time is left before the slate's publish deadline to write the shadow rows.

    Raised inside the runner's shadow block once the shadow rows are built and before any shadow
    side file is written, so the shadow rows are dropped as holes and the live write goes ahead.
    The ledger store re-checks the deadline immediately before it writes
    (``forward_ledger.store.commit_changes``), so slow shadow work must never reach it.
    """


@dataclass(frozen=True)
class ShadowPin:
    """The frozen arm's inputs: three model ids, the blend, its converter and the bet rule.

    Attributes:
        model_ids: Model target (``wp``, ``ats``, ``ou``) -> the artifact id to score with.
        blend: The blend artifact id (its bound slope feeds the WP second test).
        converter: The market-probability converter the blend must be bound to.
        recipe: The registered recipe id every shadow row is stamped with.
        chain_fit_path: The chain-fit run record the bet rule reads (repo-relative text).
        chain_fit_sha256: The sha256 that record must hash to.
    """

    model_ids: Mapping[str, str]
    blend: str
    converter: str
    recipe: str
    chain_fit_path: str
    chain_fit_sha256: str

    def resolved(self) -> ResolvedArtifacts:
        """The pin as the resolved ids a decision is stamped with."""
        return ResolvedArtifacts(
            wp=self.model_ids["wp"],
            ats=self.model_ids["ats"],
            ou=self.model_ids["ou"],
            blend=self.blend,
            converter=self.converter,
        )


# The committed pin. ``model_ids`` is the read-only proxy itself, never a copy.
SHADOW_PIN: ShadowPin = ShadowPin(
    model_ids=SHADOW_MODEL_IDS,
    blend=SHADOW_BLEND,
    converter=SHADOW_CONVERTER,
    recipe=SHADOW_RECIPE,
    chain_fit_path=SHADOW_CHAIN_FIT_PATH,
    chain_fit_sha256=SHADOW_CHAIN_FIT_SHA256,
)


def _require_pinned_files(pin: ShadowPin, artifacts_dir: Path) -> None:
    """Refuse unless the chain-fit record and the blend's bound converter are the pinned ones."""
    from models.blending import MarketBlender

    sha = hashlib.sha256(Path(pin.chain_fit_path).read_bytes()).hexdigest()
    if sha != pin.chain_fit_sha256:
        msg = (
            f"the chain-fit record {pin.chain_fit_path} hashes to {sha}, not to the pinned "
            f"{pin.chain_fit_sha256}"
        )
        raise ShadowPinMismatchError(msg)
    bound = MarketBlender.from_artifacts(
        artifacts_dir, version=pin.blend
    ).market_probability_artifact_id
    if bound != pin.converter:
        msg = (
            f"the pinned blend {pin.blend} is bound to converter {bound!r}, not to the pinned "
            f"{pin.converter!r}"
        )
        raise ShadowPinMismatchError(msg)


def build_shadow_bundle(
    live: DecisionBundle,
    *,
    artifacts_dir: Path,
    pin: ShadowPin = SHADOW_PIN,
    scorer: Scorer = score_deployed_artifacts,
) -> DecisionBundle:
    """The frozen arm's decision over *live*'s own inputs. Writes nothing.

    Args:
        live: The live decision bundle (its candidates, schedule, gold rows and run instant).
        artifacts_dir: Where the pinned artifacts live; it need not hold a ``latest.json``.
        pin: The frozen arm's inputs (the committed pin by default).
        scorer: The model scorer (``backtest.diagnose.score_deployed_artifacts``).

    Returns:
        A :class:`DecisionBundle` with the shadow frame and re-scored candidates, the live
        schedule, gold rows and run instant, the row-19 fits, the pin's resolved ids and its
        chain-fit path.

    Raises:
        ShadowPinMismatchError: the chain-fit record or the blend's converter is not the pin's.
        Everything the scorer, the blend loader and the selection path raise, by name.
    """
    _require_pinned_files(pin, artifacts_dir)
    candidates = live.candidates.copy()
    if not candidates.empty:
        rescore_candidates(
            candidates, live.gold_inputs, artifacts_dir, pin.model_ids, scorer
        )
        candidates = reattach_spread_probability(candidates, artifacts_dir, pin.blend)
    fits = load_frozen_chain_fit(pin.chain_fit_path)
    result = select_weekly_bets(
        candidates,
        live.schedule,
        fits,
        edge_thresholds=EDGE_TIER_THRESHOLDS_BY_TARGET,
    )
    frame = records_to_bet_list_frame(
        result, fits, run_mode=RUN_MODE_FORWARD, decided_at=live.run_instant
    )
    return DecisionBundle(
        frame=frame,
        candidates=candidates,
        schedule=live.schedule,
        gold_inputs=live.gold_inputs,
        fits=fits,
        resolved=pin.resolved(),
        chain_fit_path=Path(pin.chain_fit_path),
        run_instant=live.run_instant,
    )
