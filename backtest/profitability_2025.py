"""The ONE-SHOT 2025 profitability runner and JUDGE (Phase 31, plan 31-12; PROD-04, SPEC R3).

THIS MODULE DECIDES; IT MEASURES NOTHING. ``backtest/wp_ev_chain.py``,
``backtest/ats_ev_chain.py`` and ``backtest/ou_ev_chain.py`` price candidates,
``backtest/bet_selector.py`` is the single bet-decision source (LOCKED-2, BET-01),
``backtest/roi_significance.py`` owns the pre-registered ROI hypothesis test, and
``backtest/diagnose.py`` owns alpha and the CLV significance primitive. This module adds ONLY
the multiplicity correction and the verdict -- exactly the relationship
``backtest/group_gate.py`` has to ``backtest/signal_lift.py`` (D30-04), and
``models/deploy_gate.py`` has to ``backtest/diagnose.py`` (D24-13).

THE FROZEN RULE LIVES ELSEWHERE, AND THAT SPLIT IS THE WHOLE POINT. The windows, the game-type
scope, the eligibility rules, the multiplicity family, the ROI test, the run-ledger states and
the verdict vocabulary are all in ``backtest/ev_chain_constants.py`` and in
``PROFITABILITY-PREREGISTRATION.md``, whose COMBINED last-modifying commit is the git-ancestry
anchor SPEC R3 asserts against. The MECHANICS here stay editable by design: a bug in the
orchestration, the rendering or the CLI is fixed HERE without disturbing that proof.

WHAT MAKES "EXACTLY ONCE" STRUCTURAL RATHER THAN PROCEDURAL
-----------------------------------------------------------
Two guards, and the second exists because the first is not enough.

1. **The runner hard-refuses to overwrite an existing verdict artifact, BEFORE the run.** A
   refusal arriving after a full walk-forward pass would be a refusal nobody could afford to
   trust, which is the same reason ``backtest/group_gate.main`` validates its output path
   before the gate runs.
2. **A durable exclusive run ledger is acquired BEFORE the first hold-season read.** The
   artifact check alone is NOT crash-safe: the runner READS AND MEASURES the hold before it
   writes the artifact, so a crash, a kill, a power loss or an unhandled exception anywhere in
   that window leaves NO artifact on disk -- and the next invocation sees a clean refusal state
   and cheerfully spends the single-use split a second time. For an asset that cannot be
   recovered, "we would have noticed" is not a control.

THERE IS NO FORCE FLAG, no overwrite flag and no environment escape. A ledger in state
``started`` or ``failed`` is NOT automatically rerunnable: a new attempt requires an OWNER
RULING recorded in the ledger, which is a deliberate human edit to a git-tracked file and
therefore leaves a diff.

THE VERDICT IS DRIVEN BY THE ROI p-VALUE AND NEVER BY A CLV p-VALUE
-------------------------------------------------------------------
``CLV_P_VALUE_IS_REPORT_ONLY`` is True in the frozen rule. Reusing a CLV p-value as a
profitability p-value would silently convert "significant CLV" into "profitable" -- the D25-14
and D26-09 trap this milestone exists to end. The firewall here is STRUCTURAL, not a
convention: :func:`assign_verdict_token` and :func:`build_verdict_records` consume a
:class:`TargetMeasurement`, a frozen dataclass that HAS NO CLV FIELD AT ALL, so there is no
CLV value in scope for the verdict path to read even dynamically.
``tests/unit/test_verdict_tokens.py`` additionally AST-scans every function reachable from
those two roots and fails on any CLV reference. The CLV figure is still CARRIED into the
artifact as disclosed report-only context, beside the verdict and never inside it.

HARD BOUNDARY. This module imports no ``train_*`` module, re-fits nothing, swaps nothing, and
NEVER writes under ``data/`` (``utils.paths.reject_data_path`` refuses such a path outright).
It LOADS deployed artifacts through ``score_deployed_artifacts`` and runs inference only.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tomllib
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.ats_ev_chain import (
    ATS_JUICE_FIELDS,
    FENCE_WINDOW_P31,
    FENCE_WINDOW_REHEARSAL,
    REGISTERED_FENCE_WINDOWS,
    ChainFit,
    FenceWindow,
    assert_fit_window_p31,
)
from backtest.bet_selector import BetSelector
from backtest.diagnose import clv_significance, score_deployed_artifacts
from backtest.ev_chain_constants import (
    ALPHA,
    BH_FAMILY_SPEC,
    CLV_P_VALUE_IS_REPORT_ONLY,
    PREREGISTRATION_PATHS,
    ROBUSTNESS_CUT_BH_COUNT,
    ROBUSTNESS_CUTS_P31,
    ROI_MIN_ATTAINABLE_P,
    ROI_P_VALUE_METHOD,
    RUN_LEDGER_PATH,
    RUN_LEDGER_STATES,
    RUN_LEDGER_TRANSITION_RULE,
    TRIAL_ENTRY_KIND_CONTROL,
    TRIAL_ENTRY_KIND_INFERENCE,
    TRIAL_REGISTRY_FIELDS_P31,
    VERDICT_TOKEN_MEANINGS,
    VERDICT_TOKENS,
)
from backtest.ou_divergence import dedupe_odds_by_book_preference
from backtest.ou_ev_chain import (
    EV_FLOOR_GRID,
    american_to_payout,
    devig,
    estimate_prior_season_bias,
    fit_frozen_residual_sd,
)
from backtest.roi_significance import (
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    flat_roi,
    roi_ci_and_p,
)
from backtest.selector_strategies import (
    OU_JUICE_FIELDS,
    ATSStrategy,
    OUStrategy,
    WPStrategy,
)
from backtest.wp_ev_chain import WPGateResult, run_wp_calibration_gate
from utils import get_logger
from utils.paths import reject_data_path

logger = get_logger(__name__)

__all__ = [
    "ARTIFACT_ABSENT_RETURN",
    "CANONICAL_TARGETS",
    "DEFAULT_VERDICT_JSON_PATH",
    "DEFAULT_VERDICT_TOML_PATH",
    "FLOAT_FORMAT",
    "PROFITABLE",
    "UNDISCHARGEABLE_NO_BETS",
    "UNDISCHARGEABLE_NO_CHAIN",
    "UNPROFITABLE",
    "VERDICT_INCONCLUSIVE",
    "ProxySplitRefusedError",
    "RunLedgerError",
    "TargetMeasurement",
    "VerdictArtifactExistsError",
    "acquire_run_ledger",
    "assign_verdict_token",
    "benjamini_hochberg_adjusted",
    "bh_family_members",
    "build_parser",
    "build_verdict_records",
    "main",
    "mark_run_ledger",
    "preregistration_commit",
    "read_run_ledger",
    "registry_exclusion_counts",
    "render_float",
    "render_verdict_fields",
    "render_verdict_toml",
    "run_armed_2025_verdict",
    "run_profitability_2025",
    "verdict_run_record",
    "write_armed_ledger",
    "write_verdict_artifact",
]


_REPO_ROOT = Path(__file__).resolve().parent.parent

# The repository's canonical target order, so a registry listing, a per-target table and every
# other per-target artifact in this project read in the same order.
CANONICAL_TARGETS: tuple[str, str, str] = ("wp", "ats", "ou")

# The verdict vocabulary, bound BY IDENTITY to the frozen tuple's members rather than restated
# as literals. A second spelling of a token is a second vocabulary for one fact.
PROFITABLE: str = VERDICT_TOKENS[0]
UNPROFITABLE: str = VERDICT_TOKENS[1]
VERDICT_INCONCLUSIVE: str = VERDICT_TOKENS[2]
UNDISCHARGEABLE_NO_BETS: str = VERDICT_TOKENS[3]
UNDISCHARGEABLE_NO_CHAIN: str = VERDICT_TOKENS[4]

# THE FLOAT PRECISION, chosen once and applied to every float this module emits. Copied
# VERBATIM from backtest/group_gate.py:370-384 together with its rationale, because the same
# fact has to hold in both places and a paraphrase is where two renderings drift apart.
#
# ``.17g`` is 17 SIGNIFICANT digits, which is the precision at which no two distinct IEEE-754
# doubles can render identically -- so no two distinct measured values can collide. A fixed
# number of DECIMAL places was rejected: a p-value of 1e-30 and one of 1e-20 would both flatten
# to 0.000000000000 at twelve places, erasing a real difference in the very column the verdict
# turns on.
#
# What must NOT be used is a bare interpolation of the float, which falls through to Python's
# default float formatting. That is shortest-round-trip and is not a contractual, stable
# rendering across platforms or patch releases. The readout drift guard asserts that every 2025
# figure it prints matches the committed verdict artifact byte-for-byte, so a formatting
# difference there would read as tampering with a measurement artifact rather than as the
# accident it would be.
FLOAT_FORMAT: str = ".17g"

# The committed generator-output form (the Phase-30 precedent: a ratified verdict is a
# git-TRACKED config file, never a gitignored JSON blob -- the D24-06/D24-07 anti-landmine
# control). Plan 31-14 commits it; this module only writes it.
DEFAULT_VERDICT_TOML_PATH: Path = Path("config") / "profitability_2025_verdict.toml"

# The machine-readable run record, under the gitignored outputs tree. NEVER under data/.
DEFAULT_VERDICT_JSON_PATH: Path = (
    Path("outputs") / "p31" / "profitability_2025_verdict.json"
)

_PRODUCTION_PATHS: tuple[Path, ...] = (
    DEFAULT_VERDICT_TOML_PATH,
    DEFAULT_VERDICT_JSON_PATH,
    Path(RUN_LEDGER_PATH),
)

# What a per-target record says instead of a return when NO bets were selected. The field NAME
# is present on every per-target record regardless of token -- the readout template's slots do
# not change shape with the data -- but a zero-bet target's RETURN is ABSENT: rendered as the
# TOML float ``nan`` and flanked by ``has_return = false``. It is never 0.0, because a return
# of zero is a measured break-even and "no bets" is not.
ARTIFACT_ABSENT_RETURN: str = "nan"

# The shifted-edge counterfactual control (D31-08). The shift is large enough that the
# manufactured edge clears the widest EV floor on the frozen grid by a wide margin; these are
# CONTROL magnitudes, not tunable parameters, and nothing they produce is ever reported as a
# result.
CONTROL_SHIFT_POINTS: float = 10.0
CONTROL_PROB_EDGE: float = 0.15

# The per-target residual/label contracts. Stated as data so the three chains cannot silently
# diverge on what "the model output" and "the realized value" mean.
_MODEL_COLUMN: Mapping[str, str] = {
    "wp": "model_prob",
    "ats": "model_spread",
    "ou": "model_total",
}
_MARKET_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "wp": ("ml_home", "ml_away"),
    "ats": ("closing_spread",),
    "ou": ("closing_total",),
}
# The silver odds column each target's market column is renamed FROM.
_ODDS_SOURCE_COLUMN: Mapping[str, Mapping[str, str]] = {
    "wp": {},
    "ats": {"spread": "closing_spread"},
    "ou": {"total": "closing_total"},
}

_UNPRICEABLE_REASONS: frozenset[str] = frozenset(
    {"missing_snapshot", "missing_prediction"}
)

# The OPTIONAL two-sided juice columns each target is priced on (DEF-31-13, ruled 2026-09-05).
# TAKEN FROM THE STRATEGIES' OWN FIELD CONSTANTS, never restated, so the columns this loader
# CARRIES are by construction the columns the selection path READS -- a loader that dropped one
# would leave the chain devigging nothing while every docstring said it did. WP needs no entry: its
# price is ``ml_home`` / ``ml_away``, which are already REQUIRED market fields for that target.
_JUICE_COLUMNS_BY_TARGET: Mapping[str, tuple[str, ...]] = {
    "wp": (),
    "ats": ATS_JUICE_FIELDS,
    "ou": OU_JUICE_FIELDS,
}


def _juice_columns_for(target: str, available: Iterable[str]) -> list[str]:
    """The juice columns to carry for ``target``, restricted to those the odds table HAS.

    The juice is OPTIONAL by design: the selection path falls back to the flat reference juice for
    a row without it (D27-13), and the stored silver table carries the four columns for some
    seasons and not others. Intersecting with what is present is therefore a coverage fact, not a
    softened requirement -- a target whose columns are entirely absent prices at the fallback and
    says so on every record through ``devig_method``.
    """
    present = set(available)
    return [column for column in _JUICE_COLUMNS_BY_TARGET[target] if column in present]


class RunLedgerError(RuntimeError):
    """Raised when the durable one-shot run ledger forbids this invocation.

    Carries the offending attempt's id, state and timestamp, and states the re-entry rule: a
    ``started`` or ``failed`` attempt is NOT automatically rerunnable, because ``started`` is
    exactly the case where the hold may already have been read. Silently re-arming a crashed
    attempt would spend the single clean split twice while leaving a record that says it ran
    once.
    """


class VerdictArtifactExistsError(FileExistsError):
    """Raised when a verdict artifact already exists at a requested output path.

    Validated BEFORE the run, at the same seam ``backtest/group_gate.main`` validates its
    output path: a refusal that arrives after a full walk-forward pass is a refusal nobody
    could afford to trust.
    """


class ProxySplitRefusedError(ValueError):
    """Raised when a caller tries to steer the ARMED production run.

    The armed entry point takes NO split, window or path argument. Its window is the frozen
    Phase-31 rule and its ledger and artifact paths are the production ones, both by
    construction rather than by default -- so a rehearsal configuration can never leak into a
    binding run, and a binding run can never be redirected onto throwaway paths.
    """


# ---------------------------------------------------------------------------
# Rendering: ONE float specifier, applied everywhere
# ---------------------------------------------------------------------------


def render_float(value: float | None) -> str:
    """Render one float as a deterministic TOML float literal at :data:`FLOAT_FORMAT`.

    A None value renders as the TOML float ``nan``, so a key is always present and the
    artifact's shape does not change with the data -- an absent return and a rendering that
    forgot the field would otherwise be indistinguishable.

    A ``.0`` suffix is appended when ``.17g`` produced an exponent-free, dot-free string for a
    whole number, which TOML would otherwise parse as an INTEGER.
    """
    if value is None:
        return "nan"
    text = format(float(value), FLOAT_FORMAT)
    if text in ("nan", "inf", "-inf"):
        return text
    if "." not in text and "e" not in text and "E" not in text:
        text = f"{text}.0"
    return text


def render_str(value: object) -> str:
    """Render one value as a TOML basic string, escaped deterministically."""
    return json.dumps("" if value is None else str(value), ensure_ascii=True)


def render_bool(value: object) -> str:
    """Render one value as a TOML boolean."""
    return "true" if bool(value) else "false"


def render_int(value: object) -> str:
    """Render one value as a TOML integer."""
    return str(int(value))


def render_int_list(values: Iterable[object]) -> str:
    """Render a sequence of integers as a TOML inline array."""
    return "[" + ", ".join(render_int(value) for value in values) + "]"


def render_str_list(values: Iterable[object]) -> str:
    """Render a sequence of strings as a TOML inline array."""
    return "[" + ", ".join(render_str(value) for value in values) + "]"


# ---------------------------------------------------------------------------
# The pre-registration anchor, RESOLVED and never transcribed
# ---------------------------------------------------------------------------


def preregistration_commit(repo_root: Path | None = None) -> str:
    """Resolve the FROZEN pre-registration's last-modifying commit SHA from git.

    RESOLVED, never transcribed, following ``backtest.group_gate.preregistration_commit``. A
    SHA typed into this source would be a second, silently divergable copy of the one fact
    SPEC R3's ancestry assertion rests on.

    Both halves of the pre-registration are asked about together --
    ``PROFITABILITY-PREREGISTRATION.md`` and ``backtest/ev_chain_constants.py``, named by
    ``PREREGISTRATION_PATHS`` so every guard resolves them from ONE place -- because it is
    their COMBINED last-modifying commit that is the anchor.

    Args:
        repo_root: Repository root to ask git about. Defaults to this file's repository.

    Returns:
        The full 40-character SHA, or ``"UNRESOLVED"`` when git cannot answer (a source
        checkout without git history, for instance). The sentinel is deliberately not an empty
        string: an empty value in the artifact would read as "no anchor" rather than as "the
        anchor could not be resolved here".
    """
    root = repo_root or _REPO_ROOT
    try:
        completed = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", *PREREGISTRATION_PATHS],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNRESOLVED"
    return completed.stdout.strip() or "UNRESOLVED"


# ---------------------------------------------------------------------------
# The durable exclusive one-shot run ledger (REVIEW-ONESHOT)
# ---------------------------------------------------------------------------

_LEDGER_ARMED, _LEDGER_STARTED, _LEDGER_COMPLETED, _LEDGER_FAILED = RUN_LEDGER_STATES


def _render_ledger(fields: Mapping[str, Any]) -> str:
    """Render the ledger as TOML. Hand-rendered: the standard library has no TOML writer."""
    lines = [
        "# =============================================================================",
        "# The DURABLE EXCLUSIVE one-shot run ledger for the 2025 clean-split verdict.",
        "#",
        "# This file is the record that the single-use 2025 split was spent. It is created by",
        "# an EXCLUSIVE file creation BEFORE the first hold-season read, so a crash between",
        "# reading the split and writing the verdict artifact leaves a 'failed' attempt on",
        "# disk rather than a silently re-spendable split.",
        "#",
        "# A 'started' or 'failed' attempt is NOT automatically rerunnable. There is NO force",
        "# flag: a new attempt requires an OWNER RULING recorded here, which is a deliberate",
        "# human edit to a git-tracked file and therefore leaves a diff.",
        "# =============================================================================",
        "",
    ]
    lines.extend(_render_ledger_field(key, value) for key, value in fields.items())
    return "\n".join(lines) + "\n"


def _render_ledger_field(key: str, value: Any) -> str:
    """Render one ledger field, choosing the TOML type from the Python type."""
    if isinstance(value, bool):
        return f"{key} = {render_bool(value)}"
    if isinstance(value, int):
        return f"{key} = {render_int(value)}"
    if isinstance(value, (list, tuple)):
        return f"{key} = {render_str_list(value)}"
    return f"{key} = {render_str(value)}"


def read_run_ledger(path: Path | str) -> dict[str, Any]:
    """Read and parse an existing run ledger.

    Raises:
        RunLedgerError: when the file exists but cannot be parsed. An UNREADABLE ledger
            BLOCKS. The alternative -- treating an unparseable ledger as absent -- would make
            a truncated write (a crash between the exclusive create and the content write)
            look exactly like a clean slate, which is the very window the ledger closes.
    """
    ledger_path = Path(path)
    try:
        with ledger_path.open("rb") as handle:
            return dict(tomllib.load(handle))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        msg = (
            f"the run ledger at {ledger_path} exists but could not be read ({exc}). An "
            "unreadable ledger BLOCKS: it is indistinguishable from a crashed attempt whose "
            "content write never landed, and the single-use split must not be spent on that "
            "ambiguity. Resolve it by hand and record an owner ruling."
        )
        raise RunLedgerError(msg) from exc


def write_armed_ledger(path: Path | str, *, owner_ruling: str) -> dict[str, Any]:
    """ARM the ledger for a binding run (CHECKPOINT 3, D31-16). EXCLUSIVE create.

    Arming is the OWNER's deliberate act, not something the runner may do for itself: the
    armed production entry point REQUIRES a pre-existing ``armed`` ledger and transitions it to
    ``started``. Without that requirement, running the module by accident would spend the
    split, and no refusal downstream could give it back.

    This function refuses to overwrite an existing ledger, so a crashed attempt cannot be
    re-armed by re-running a command; clearing it is a deliberate human edit to a tracked file.

    Args:
        path: The ledger path.
        owner_ruling: The owner's recorded ruling -- what they are authorising and why. An
            empty ruling is refused: an unexplained arming is indistinguishable from an
            accident.

    Returns:
        The written ledger fields.

    Raises:
        RunLedgerError: when a ledger already exists, or when ``owner_ruling`` is empty.
    """
    if not owner_ruling.strip():
        msg = (
            "arming the one-shot run ledger requires a non-empty owner ruling. The ruling IS "
            "the authorisation, and an unexplained arming is indistinguishable from an "
            "accident."
        )
        raise RunLedgerError(msg)

    ledger_path = Path(path)
    fields: dict[str, Any] = {
        "state": _LEDGER_ARMED,
        "attempt_id": uuid.uuid4().hex,
        "armed_at_utc": datetime.now(UTC).isoformat(),
        "owner_ruling": owner_ruling,
        "transition_rule": RUN_LEDGER_TRANSITION_RULE,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ledger_path.open("x", encoding="utf-8") as handle:
            handle.write(_render_ledger(fields))
    except FileExistsError as exc:
        existing = read_run_ledger(ledger_path)
        msg = (
            f"refusing to arm {ledger_path}: a ledger already exists in state "
            f"{existing.get('state')!r} (attempt {existing.get('attempt_id')!r}). "
            + RUN_LEDGER_TRANSITION_RULE
        )
        raise RunLedgerError(msg) from exc
    return fields


def acquire_run_ledger(
    path: Path | str,
    *,
    window: FenceWindow,
    preregistration_sha: str,
) -> dict[str, Any]:
    """Acquire the durable exclusive run ledger BEFORE the first hold-season read.

    Two acquisition modes, both of which end with the ledger in state ``started`` on disk
    before any hold row has been touched:

    * **The FROZEN Phase-31 window requires a pre-existing ``armed`` ledger** and performs the
      pre-registered ``armed -> started`` transition. Arming is CHECKPOINT 3, the owner's act
      (:func:`write_armed_ledger`). This is what stops an accidental invocation from spending
      the single-use split: with no armed ledger the binding run simply refuses.
    * **A rehearsal window creates the ledger with an EXCLUSIVE ``open(path, "x")``**, which
      raises ``FileExistsError`` atomically rather than TRUNCATING. A rehearsal needs no owner
      ceremony -- its split is already burned -- but it still leaves a durable record, so the
      crash-window guard is exercised by the same code path the binding run uses.

    Any other pre-existing state (``started``, ``failed``, ``completed``) BLOCKS. In
    particular a ``started`` or ``failed`` attempt is NOT automatically rerunnable: ``started``
    is exactly the case where the hold may already have been read, and re-running would spend
    the split twice while leaving a record that says it ran once. There is NO force flag and no
    environment escape; clearing the ledger is a deliberate human edit to a tracked file.

    Args:
        path: The ledger path.
        window: The registered fence window this attempt runs under.
        preregistration_sha: The resolved pre-registration commit, recorded on the attempt so
            the ledger names the rule the attempt was bound by.

    Returns:
        The ledger fields as written, including the ``attempt_id``.

    Raises:
        RunLedgerError: when the ledger forbids this invocation.
    """
    ledger_path = Path(path)
    now = datetime.now(UTC).isoformat()
    started = {
        "state": _LEDGER_STARTED,
        "attempt_id": uuid.uuid4().hex,
        "started_at_utc": now,
        "preregistration_commit": preregistration_sha,
        "window_label": window.label,
        "window_is_the_preregistered_rule": window.is_the_preregistered_rule,
        "hold_seasons": [str(season) for season in window.hold_seasons],
        "transition_rule": RUN_LEDGER_TRANSITION_RULE,
    }

    if window.is_the_preregistered_rule:
        if not ledger_path.exists():
            msg = (
                f"refusing to run the BINDING 2025 verdict: no run ledger exists at "
                f"{ledger_path}, so this attempt was never armed. CHECKPOINT 3 (D31-16) arms "
                "it by recording an owner ruling; the runner does not arm itself, because an "
                "accidental invocation would otherwise spend the single-use split and no "
                "refusal downstream could give it back."
            )
            raise RunLedgerError(msg)
        existing = read_run_ledger(ledger_path)
        state = str(existing.get("state", ""))
        if state != _LEDGER_ARMED:
            raise RunLedgerError(_blocked_message(ledger_path, existing))
        started["armed_at_utc"] = existing.get("armed_at_utc", "")
        started["owner_ruling"] = existing.get("owner_ruling", "")
        started["prior_attempt_id"] = existing.get("attempt_id", "")
        ledger_path.write_text(_render_ledger(started), encoding="utf-8")
        return started

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # THE EXCLUSIVE CREATE. "x" raises FileExistsError atomically; "w" would TRUNCATE, and
        # truncating a ledger is exactly how a spent split becomes re-spendable.
        with ledger_path.open("x", encoding="utf-8") as handle:
            handle.write(_render_ledger(started))
    except FileExistsError as exc:
        existing = read_run_ledger(ledger_path)
        if str(existing.get("state", "")) == _LEDGER_ARMED:
            started["armed_at_utc"] = existing.get("armed_at_utc", "")
            started["owner_ruling"] = existing.get("owner_ruling", "")
            started["prior_attempt_id"] = existing.get("attempt_id", "")
            ledger_path.write_text(_render_ledger(started), encoding="utf-8")
            return started
        raise RunLedgerError(_blocked_message(ledger_path, existing)) from exc
    return started


def _blocked_message(ledger_path: Path, existing: Mapping[str, Any]) -> str:
    """The refusal text naming the offending attempt, its state and its timestamp."""
    timestamp = (
        existing.get("started_at_utc")
        or existing.get("armed_at_utc")
        or existing.get("finished_at_utc")
        or "unknown"
    )
    return (
        f"the one-shot run ledger at {ledger_path} BLOCKS this invocation: attempt "
        f"{existing.get('attempt_id')!r} is in state {existing.get('state')!r} "
        f"(timestamp {timestamp!r}). "
        + RUN_LEDGER_TRANSITION_RULE
        + " An OWNER RULING must be recorded in the ledger before a new attempt may be armed."
    )


def mark_run_ledger(path: Path | str, state: str, **extra: Any) -> dict[str, Any]:
    """Write a terminal state onto the ledger this invocation already holds.

    Only ``completed`` and ``failed`` are writable here. ``armed`` is the owner's act and
    ``started`` is :func:`acquire_run_ledger`'s; allowing this function to write either would
    make it the force flag the design forbids.

    Raises:
        RunLedgerError: for a state outside ``{completed, failed}``.
    """
    if state not in (_LEDGER_COMPLETED, _LEDGER_FAILED):
        msg = (
            f"mark_run_ledger refuses to write state {state!r}. Only "
            f"{_LEDGER_COMPLETED!r} and {_LEDGER_FAILED!r} are terminal transitions this "
            "runner may make; re-arming is an OWNER RULING recorded by hand."
        )
        raise RunLedgerError(msg)
    ledger_path = Path(path)
    fields = read_run_ledger(ledger_path)
    fields["state"] = state
    fields["finished_at_utc"] = datetime.now(UTC).isoformat()
    fields.update(extra)
    ledger_path.write_text(_render_ledger(fields), encoding="utf-8")
    return fields


# ---------------------------------------------------------------------------
# Output-path validation, BEFORE the run
# ---------------------------------------------------------------------------


def _validate_output_path(path: Path | str, what: str) -> Path:
    """Resolve an output path, refusing ``data/`` and refusing to overwrite.

    Raises:
        ValueError: when the path lands under a ``data/`` tree.
        VerdictArtifactExistsError: when a verdict artifact is already there.
    """
    resolved = reject_data_path(
        path,
        what=what,
        suggestion=DEFAULT_VERDICT_JSON_PATH.as_posix(),
    )
    if resolved.exists():
        msg = (
            f"refusing to overwrite the existing {what} at {resolved}. The 2025 split is "
            "single-use: an artifact already there means the verdict was already produced, "
            "and overwriting it would replace a binding measurement with a second one while "
            "leaving a record that says it ran once. There is NO force flag."
        )
        raise VerdictArtifactExistsError(msg)
    return resolved


# ---------------------------------------------------------------------------
# The multiplicity correction, with the PRE-REGISTERED denominator
# ---------------------------------------------------------------------------


def bh_family_members(
    registry: Sequence[Mapping[str, Any]], hold_window_label: str
) -> list[Mapping[str, Any]]:
    """The registry entries that ENTER the Benjamini-Hochberg family.

    The membership rule is READ OFF the frozen registry schema rather than carried as a
    separate flag, so an entry cannot be in the registry under one description and in the
    family under another:

    * ``entry_kind`` must be ``inference``. Every ``control`` entry is excluded -- the
      shifted-edge counterfactual pass (D31-08) has no p-value to correct, so counting it would
      statistically penalise the verdict for running a code-liveness check.
    * ``sample_window`` must be the HOLD window. The tune-side ``EV_FLOOR_GRID`` sweep is
      RECORDED for transparency and excluded, because its grid cells ran entirely on the tune
      split and never touched the hold.

    Registry row count and family size are therefore allowed to differ, visibly and by design
    (:func:`registry_exclusion_counts` states the difference arithmetically).
    """
    return [
        entry
        for entry in registry
        if entry.get("entry_kind") == TRIAL_ENTRY_KIND_INFERENCE
        and entry.get("sample_window") == hold_window_label
    ]


def registry_exclusion_counts(
    registry: Sequence[Mapping[str, Any]], hold_window_label: str
) -> dict[str, int]:
    """Account for every registry row, so the row-count/denominator gap is arithmetic.

    Returns the row count, the family size, and the two exclusion counts -- control entries and
    tune-side sweep cells. ``rows == family + control + tune_side`` by construction; asserting
    that identity is what stops the gap being read as a bug.
    """
    control = sum(
        1 for entry in registry if entry.get("entry_kind") == TRIAL_ENTRY_KIND_CONTROL
    )
    tune_side = sum(
        1
        for entry in registry
        if entry.get("entry_kind") == TRIAL_ENTRY_KIND_INFERENCE
        and entry.get("sample_window") != hold_window_label
    )
    family = len(bh_family_members(registry, hold_window_label))
    return {
        "registry_rows": len(registry),
        "bh_family_size": family,
        "excluded_control_entries": control,
        "excluded_tune_side_sweep_cells": tune_side,
    }


def benjamini_hochberg_adjusted(pvalues: Sequence[float], m: int) -> list[float]:
    """BH step-up adjusted q-values at an EXPLICIT family size ``m``.

    ``scipy.stats.false_discovery_control`` cannot express this: it infers ``m`` from the
    length of its input, and the Phase-31 family size is PRE-REGISTERED (6 with no fallback
    fired, 7-9 with one, two or three) rather than derived from how many entries happened to
    carry a testable p-value. A target that selected zero bets contributes a family entry with
    no p; correcting the remaining entries at a smaller ``m`` would quietly make the surviving
    tests easier because another target had no bets, which is a denominator chosen after the
    fact.

    The untestable entries are treated as ``p = 1``, which is what "no evidence against the
    null" means; they sort last and therefore never affect the observed entries' ranks. The
    adjusted value is the standard monotone step-up::

        q_(i) = min(1, min over j >= i of  m * p_(j) / j)

    When ``m == len(pvalues)`` this is exactly ``false_discovery_control(..., method="bh")``,
    which ``tests/unit/test_bh_family_denominator.py`` asserts against scipy directly.

    Args:
        pvalues: The observed raw p-values, in caller order.
        m: The PRE-REGISTERED family size. Must be at least ``len(pvalues)``.

    Returns:
        The adjusted q-values, in the caller's input order.

    Raises:
        ValueError: when ``m`` is smaller than the number of supplied p-values, which would
            mean the family is smaller than the set of tests actually run.
    """
    if not pvalues:
        return []
    if m < len(pvalues):
        msg = (
            f"the pre-registered BH family size m={m} is smaller than the "
            f"{len(pvalues)} p-values supplied. The family cannot be smaller than the set of "
            "tests actually performed on the hold."
        )
        raise ValueError(msg)

    order = sorted(range(len(pvalues)), key=lambda i: float(pvalues[i]))
    adjusted = [0.0] * len(pvalues)
    running = 1.0
    for rank in range(len(order), 0, -1):
        index = order[rank - 1]
        candidate = float(pvalues[index]) * m / rank
        running = min(running, candidate, 1.0)
        adjusted[index] = running
    return adjusted


# ---------------------------------------------------------------------------
# The VERDICT. No CLV value is in scope anywhere below this line.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetMeasurement:
    """Everything the verdict rule is allowed to see about one target.

    THE FIREWALL IS THIS CLASS. It carries NO closing-line-value field of any kind, so the
    verdict path has no CLV value in scope to read -- not by attribute, not by key, not
    dynamically. ``CLV_P_VALUE_IS_REPORT_ONLY`` is therefore enforced structurally rather than
    by a convention someone has to remember, and the AST scan in
    ``tests/unit/test_verdict_tokens.py`` is a second, independent proof of the same fact.

    Attributes:
        target: The target code.
        chain_resolved: Whether the chain ran at all on the hold frame. False means a required
            input was absent for every candidate, so nothing could be priced.
        control_passed: Whether the shifted-edge counterfactual pass over the ACTUAL hold frame
            selected at least one bet (D31-08). This is what makes a ZERO result defensible:
            a synthetic-only control passes even when a wiring fault produced a false zero.
        control_bet_count: How many bets the counterfactual selected.
        bet_count: How many bets the REAL selection made on the hold.
        flat_roi: The observed flat-stake ROI, or None when no bet was made.
        adjusted_p: The BH-adjusted pre-registered ROI p-value, or None when untestable.
        raw_p: The unadjusted pre-registered ROI p-value, or None when untestable.
        p_absent_reason: Why the ROI p-value is absent, when it is.
        n_blocks: The number of (season, week) resampling blocks the hold bets spanned.
        n_push: Bets carried as a push (outcome None with a realized value present).
        n_ungraded: Bets with no realized value at all.
        fallback_fired: Whether the target's REGISTERED calibration fallback fired.
        fallback_trigger: What made it fire, or None.
    """

    target: str
    chain_resolved: bool
    control_passed: bool
    control_bet_count: int
    bet_count: int
    flat_roi: float | None
    adjusted_p: float | None
    raw_p: float | None
    p_absent_reason: str | None
    n_blocks: int
    n_push: int
    n_ungraded: int
    fallback_fired: bool
    fallback_trigger: str | None


def assign_verdict_token(
    measurement: TargetMeasurement,
    *,
    alpha: float = ALPHA,
    min_attainable_p: float = ROI_MIN_ATTAINABLE_P,
) -> tuple[str, str]:
    """Assign ONE of the five frozen verdict tokens, with the reason it was assigned.

    The arms are applied in the pre-registered order, and they are mutually exclusive and
    jointly exhaustive over the outcome space:

    1. The chain could not run at all -> ``UNDISCHARGEABLE_NO_CHAIN``.
    2. Zero bets AND the counterfactual control PASSED -> ``UNDISCHARGEABLE_NO_BETS``. A
       first-class RESULT filling the same template slots as any other verdict, and an
       explicitly defined PASS under SPEC R1.
    3. Zero bets and the control FAILED -> ``UNDISCHARGEABLE_NO_CHAIN``. Splitting these two is
       what the control buys: collapsing them would make zero ambiguous between "no edge" and
       "broken", which is the whole reason the control runs on the ACTUAL hold frame.
    4. A non-positive return -> ``UNPROFITABLE_CLEAN``. A measured result, not a failure to
       measure.
    5. A strictly positive return, and the BH-adjusted pre-registered ROI p-value STRICTLY
       BELOW alpha -> ``PROFITABLE_CLEAN``. Otherwise ``INCONCLUSIVE_CLEAN``, including the
       registered case where the MINIMUM ATTAINABLE p exceeds alpha, which is reported with
       that stated reason. A positive return that cannot be distinguished from zero is
       INCONCLUSIVE and is NEVER called PROFITABLE.

    The comparison against alpha is STRICTLY LESS THAN, so an adjusted p exactly EQUAL to alpha
    is NOT significant -- the canonical convention, fixed here rather than left to a reader.

    Args:
        measurement: The target's measured outcome. Carries no CLV field; see
            :class:`TargetMeasurement`.
        alpha: The significance level. IMPORTED by default from the one alpha this repository
            judges at; it is a parameter only so the registered finite-sample consequence can
            be exercised by a test that drives alpha below the attainable floor.
        min_attainable_p: The smallest p the pre-registered bootstrap can return.

    Returns:
        ``(token, reason)``. The token is always a member of the frozen ``VERDICT_TOKENS``.
    """
    target = measurement.target

    if not measurement.chain_resolved:
        return UNDISCHARGEABLE_NO_CHAIN, (
            f"[{target}] the EV chain did not resolve on the hold frame at all -- a required "
            "input was absent, so no candidate could be priced. Distinct from NO_BETS: there "
            "the chain ran and declined; here it could not run."
        )

    if measurement.bet_count == 0:
        if measurement.control_passed:
            return UNDISCHARGEABLE_NO_BETS, (
                f"[{target}] the chain ran end to end on the hold frame and selected ZERO "
                "bets. The shifted-edge counterfactual pass over the SAME frame selected "
                f"{measurement.control_bet_count} bets, so the zero means NO EDGE and not a "
                "broken chain. This is a RESULT and an explicitly defined PASS (SPEC R1)."
            )
        return UNDISCHARGEABLE_NO_CHAIN, (
            f"[{target}] the chain selected zero bets AND its shifted-edge counterfactual "
            "pass over the same hold frame ALSO selected zero. A zero that a manufactured "
            "positive edge cannot move is not evidence of no edge -- it is an unexplained "
            "zero, and it is reported as NO_CHAIN rather than as NO_BETS."
        )

    if measurement.flat_roi is None:
        return UNDISCHARGEABLE_NO_CHAIN, (
            f"[{target}] {measurement.bet_count} bets were selected but no flat-stake return "
            "could be computed (no positive total stake). That is a wiring fault, not a "
            "measured return, and it is never reported as a return of 0."
        )

    roi = float(measurement.flat_roi)
    if roi <= 0.0:
        return UNPROFITABLE, (
            f"[{target}] {measurement.bet_count} bets were selected and the observed "
            f"flat-stake return is {render_float(roi)}, which is not strictly positive. This "
            "is a MEASURED result, not a failure to measure."
        )

    if min_attainable_p > alpha:
        return VERDICT_INCONCLUSIVE, (
            f"[{target}] the observed flat-stake return is {render_float(roi)}, but the "
            f"MINIMUM ATTAINABLE p of the pre-registered bootstrap is "
            f"{render_float(min_attainable_p)}, which EXCEEDS alpha "
            f"{render_float(alpha)}. The test cannot reach significance at any observed "
            "return on this configuration, so a positive return reports INCONCLUSIVE with "
            "that stated reason and is NEVER called PROFITABLE."
        )

    if measurement.adjusted_p is None:
        return VERDICT_INCONCLUSIVE, (
            f"[{target}] the observed flat-stake return is {render_float(roi)}, but the "
            "pre-registered ROI p-value could not be computed: "
            f"{measurement.p_absent_reason or 'no reference distribution was available'}. A "
            "positive return that cannot be distinguished from zero is INCONCLUSIVE."
        )

    adjusted = float(measurement.adjusted_p)
    if adjusted < alpha:
        return PROFITABLE, (
            f"[{target}] {measurement.bet_count} bets, observed flat-stake return "
            f"{render_float(roi)} (strictly positive), and the pre-registered one-sided ROI "
            f"p-value is {render_float(adjusted)} after the Benjamini-Hochberg correction "
            f"over the enumerated family, which is strictly below alpha {render_float(alpha)}."
        )
    return VERDICT_INCONCLUSIVE, (
        f"[{target}] {measurement.bet_count} bets and an observed flat-stake return of "
        f"{render_float(roi)}, but the pre-registered one-sided ROI p-value is "
        f"{render_float(adjusted)} after correction, which does NOT clear alpha "
        f"{render_float(alpha)} (the comparison is strictly less than, so a value exactly at "
        "alpha is not significant). A positive return that cannot be distinguished from zero "
        "is INCONCLUSIVE and is never called PROFITABLE."
    )


def build_verdict_records(
    measurements: Sequence[TargetMeasurement],
    *,
    alpha: float = ALPHA,
    min_attainable_p: float = ROI_MIN_ATTAINABLE_P,
) -> dict[str, dict[str, Any]]:
    """Assign a verdict to every target, in the canonical order.

    Every record carries the SAME field set regardless of token, so the readout template's
    slots do not change shape with the data and an UNDISCHARGEABLE target reads as a result
    rather than as a gap. The one thing that changes is that a zero-bet target's RETURN is
    ABSENT (``has_return = false``, the value rendered as the TOML float ``nan``) rather than
    reported as 0.0 -- a return of zero is a measured break-even and no bets is not.

    No CLV value is in scope in this function or in anything it calls; see
    :class:`TargetMeasurement`.
    """
    records: dict[str, dict[str, Any]] = {}
    by_target = {measurement.target: measurement for measurement in measurements}
    for target in [t for t in CANONICAL_TARGETS if t in by_target] + sorted(
        t for t in by_target if t not in CANONICAL_TARGETS
    ):
        measurement = by_target[target]
        token, reason = assign_verdict_token(
            measurement, alpha=alpha, min_attainable_p=min_attainable_p
        )
        has_return = measurement.bet_count > 0 and measurement.flat_roi is not None
        records[target] = {
            "target": target,
            "verdict_token": token,
            "verdict_reason": reason,
            "token_meaning": VERDICT_TOKEN_MEANINGS[token],
            "chain_resolved": measurement.chain_resolved,
            "control_passed": measurement.control_passed,
            "control_bet_count": measurement.control_bet_count,
            "bets_selected": measurement.bet_count,
            "has_return": has_return,
            "flat_roi": measurement.flat_roi if has_return else None,
            "raw_roi_p_value": measurement.raw_p,
            "adjusted_roi_p_value": measurement.adjusted_p,
            "roi_p_value_absent_reason": measurement.p_absent_reason,
            "n_blocks": measurement.n_blocks,
            "n_push": measurement.n_push,
            "n_ungraded": measurement.n_ungraded,
            "fallback_fired": measurement.fallback_fired,
            "fallback_trigger": measurement.fallback_trigger,
        }
    return records


# ---------------------------------------------------------------------------
# Candidate loading (the HOLD read is its own named seam)
# ---------------------------------------------------------------------------


def _load_candidate_frames(
    seasons: Sequence[int],
    *,
    candidates_by_target: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Score the deployed artifacts over ``seasons`` and join the closing market. READ ONLY.

    Mirrors ``backtest.ou_monetization._build_scored_candidates`` one target at a time: LOAD
    the deployed artifact through ``score_deployed_artifacts`` and run inference, join the raw
    closing line from SILVER (never the gold z-scored ``snapshot_total``), and carry the
    provenance columns so the selector's OUM-06 hard-fail can judge them.

    Args:
        seasons: The seasons to load. Nothing outside this list is read.
        candidates_by_target: Pre-built frames, used INSTEAD of reading gold. The injection
            seam the guard tests drive, mirroring ``run_ou_monetization``'s ``gold_df``.

    Returns:
        ``{target -> candidate frame}``, each carrying ``game_id``, ``season``, ``week``,
        ``target``, the target's model column, its market columns, ``actual`` and provenance.
    """
    wanted = sorted({int(season) for season in seasons})
    frames: dict[str, pd.DataFrame] = {}

    if candidates_by_target is not None:
        for target in CANONICAL_TARGETS:
            frame = candidates_by_target.get(target)
            if frame is None:
                frames[target] = pd.DataFrame()
                continue
            frames[target] = frame[frame["season"].isin(wanted)].copy()
        return frames

    from backtest.engine import BacktestEngine

    # ONE row per game before the join (WR-15). ``_load_closing_odds`` returns the whole silver
    # table with no dedupe AND normalizes ``game_id`` (LAR -> LA) on the way, which can itself
    # create two rows sharing one key. A left join against a duplicated key FANS OUT the candidate
    # frame: one game becomes two candidate rows, two priced bets, two entries in
    # ``_per_bet_frame`` and double weight in both the ROI numerator and the block bootstrap.
    # ``BetSelector._build_universe`` refuses a duplicate (game_id, target) pair by name, but this
    # path passes no ``scheduled_games``, so nothing downstream would catch the fan-out. The
    # sibling weekly path guards this explicitly; this one did not.
    odds = dedupe_odds_by_book_preference(BacktestEngine()._load_closing_odds())
    for target in CANONICAL_TARGETS:
        gold = pd.read_parquet(f"data/gold/features_{target}.parquet")
        gold = gold[gold["season"].isin(wanted)].copy()
        if gold.empty:
            frames[target] = pd.DataFrame()
            continue
        preds = score_deployed_artifacts(target, gold_df=gold)
        keep = ["game_id", "ml_home", "ml_away", "spread", "total"]
        keep += [col for col in ("sportsbook", "is_live") if col in odds.columns]
        # The stored two-sided juice the chain devigs (DEF-31-13, ruled 2026-09-05). Carried under
        # its STORED name, un-renamed, because that is the name the frozen pre-registration
        # sections 3.2/3.3 step 4 give it and the name the strategies read. Without this the ruled
        # devig would be inert on exactly the run it was ruled for: every hold bet would price at
        # the flat fallback while the pre-registration said otherwise.
        keep += _juice_columns_for(target, odds.columns)
        merged = preds.merge(odds[keep], on="game_id", how="left")
        if len(merged) != len(preds):
            msg = (
                f"the closing-odds join fanned out target {target!r}: {len(preds)} scored games "
                f"became {len(merged)} candidate rows. A duplicated game_id in the odds table "
                "would double-count those games in the ROI and in the block bootstrap."
            )
            raise ValueError(msg)
        merged = merged.rename(columns=dict(_ODDS_SOURCE_COLUMN[target]))
        merged["target"] = target
        required = _MARKET_COLUMNS[target]
        for column in required:
            merged = merged[merged[column].notna()]
        frames[target] = merged.reset_index(drop=True)
    return frames


def _load_tune_frames(
    seasons: Sequence[int],
    *,
    candidates_by_target: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the TUNE-side candidates (the tune window plus the strictly-prior bias seed)."""
    return _load_candidate_frames(seasons, candidates_by_target=candidates_by_target)


def _load_hold_frames(
    seasons: Sequence[int],
    *,
    candidates_by_target: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the HOLD-side candidates. THE FIRST READ OF THE HOLD HAPPENS HERE.

    A separately named function rather than a flag on the loader above, so "the ledger is
    acquired before the first hold read" is a fact a spy can assert about the call ORDER rather
    than a comment about the intended one.
    """
    return _load_candidate_frames(seasons, candidates_by_target=candidates_by_target)


# ---------------------------------------------------------------------------
# Tune-only fitting
# ---------------------------------------------------------------------------


def _residuals_by_season(frame: pd.DataFrame, target: str) -> dict[int, np.ndarray]:
    """Per-season residual arrays under the LOCKED contract ``actual - prediction``.

    One contract for all three targets, on each target's own scale: the O/U total, the ATS
    home margin, and -- for WP's REGISTERED fallback only -- the probability scale.
    """
    column = _MODEL_COLUMN[target]
    out: dict[int, np.ndarray] = {}
    for season in sorted(int(s) for s in frame["season"].unique()):
        sub = frame[frame["season"] == season]
        out[season] = sub["actual"].to_numpy(dtype=float) - sub[column].to_numpy(
            dtype=float
        )
    return out


@dataclass(frozen=True)
class _TargetFit:
    """One target's tune-only fit, its fence report, and the EV floor it froze."""

    target: str
    chain_fit: ChainFit
    gate: WPGateResult | None
    fence_report: dict[str, Any]
    ev_floor_t: float
    sweep_rows: list[dict[str, Any]]
    tune_bet_counts: dict[float, int]


def _fit_target_on_tune(
    target: str,
    tune_frame: pd.DataFrame,
    window: FenceWindow,
) -> tuple[ChainFit, WPGateResult | None]:
    """Fit every nuisance parameter for one target on the TUNE window ONLY.

    The prior-season walk-forward bias reads STRICTLY-PRIOR seasons through the existing
    ``estimate_prior_season_bias``; the frozen residual SD comes from the existing
    ``fit_frozen_residual_sd`` over BIAS-CORRECTED tune residuals. Neither estimator is
    re-derived here. WP fits no residual SD BY DESIGN (D31-07) and instead runs the frozen
    reliability and Brier gate on the tune split.

    Returns:
        ``(ChainFit, WPGateResult | None)``. The ChainFit names the seasons every fit consumed,
        which is what makes the leakage fence possible at all.
    """
    tune_seasons = tuple(
        season for season in window.tune_seasons if season in set(tune_frame["season"])
    )
    resid_by_season = _residuals_by_season(tune_frame, target)

    bias_seasons = tuple(sorted({*window.tune_seasons, *window.hold_seasons}))
    bias_by_season: dict[int, float] = {}
    bias_pool_by_season: dict[int, tuple[int, ...]] = {}
    for season in bias_seasons:
        pool = tuple(sorted(s for s in resid_by_season if s < season))
        if not pool:
            continue
        bias_by_season[season] = float(
            estimate_prior_season_bias(resid_by_season, season)
        )
        bias_pool_by_season[season] = pool

    frozen_sd: float | None = None
    gate: WPGateResult | None = None
    tune_fit_seasons: tuple[int, ...] = ()

    if target == "wp":
        tune_rows = tune_frame[tune_frame["season"].isin(window.tune_seasons)]
        gate = run_wp_calibration_gate(
            tune_rows["model_prob"].to_numpy(dtype=float),
            tune_rows["actual"].to_numpy(dtype=float),
        )
        tune_fit_seasons = tune_seasons
    else:
        column = _MODEL_COLUMN[target]
        corrected: list[float] = []
        consumed: set[int] = set()
        for season in tune_seasons:
            sub = tune_frame[tune_frame["season"] == season]
            if sub.empty or season not in bias_by_season:
                continue
            bias = bias_by_season[season]
            corrected.extend(
                (
                    sub["actual"].to_numpy(dtype=float)
                    - (sub[column].to_numpy(dtype=float) + bias)
                ).tolist()
            )
            consumed.add(season)
        frozen_sd = fit_frozen_residual_sd(np.asarray(corrected, dtype=float))
        tune_fit_seasons = tuple(sorted(consumed))

    # No per-target eligibility boundary: the O/U one was deleted with its gate (D33.2-24), so
    # every target's fit is built the same way.
    chain_fit = ChainFit(
        target=target,
        frozen_sd=frozen_sd,
        season_bias_by_season=bias_by_season,
        tune_fit_seasons=tune_fit_seasons,
        threshold_window=window.threshold_window,
        bias_pool_by_season=bias_pool_by_season,
    )
    return chain_fit, gate


def _require_frozen_sd(target: str, fit: ChainFit) -> float:
    """The frozen residual SD for a LINE target, refusing an unusable one (WR-05).

    ``float(fit.frozen_sd or 0.0)`` was silent in exactly the case that matters. A zero SD reaches
    ``calibrated_p_cover`` / ``calibrated_p_over``, where ``z = (line - corrected) / sd`` divides
    by zero, so ``norm.cdf`` returns 0.0 or 1.0 and every candidate clips to the probability
    bound; every 0.999 candidate then clears any EV floor and Kelly stakes it at the per-bet cap.
    The PRICING path refuses this input by name (``ats_ev_chain.py``), so the selection path was
    strictly weaker than the path it mirrors. ``or 0.0`` also swallowed a legitimately-0.0 value.

    Raises:
        ValueError: naming the target and the offending value.
    """
    value = fit.frozen_sd
    if value is None or not math.isfinite(value) or value <= 0:
        msg = (
            f"target {target!r} has no usable frozen residual SD ({value!r}); a zero, absent or "
            "non-finite SD divides by zero in the calibrated-probability converter and clips "
            "every candidate to the probability bound, which Kelly then stakes at the per-bet "
            "cap. The chain path already refuses this input by name."
        )
        raise ValueError(msg)
    return float(value)


def _strategy_for_target(target: str, fit: ChainFit, gate: WPGateResult | None) -> Any:
    """Build the registered selection strategy for one target from its fit."""
    if target == "wp":
        return WPStrategy(season_bias_by_season=fit.season_bias_by_season, gate=gate)
    if target == "ats":
        return ATSStrategy(
            frozen_sd=_require_frozen_sd(target, fit),
            season_bias_by_season=fit.season_bias_by_season,
        )
    return OUStrategy(
        frozen_sd=_require_frozen_sd(target, fit),
        season_bias_by_season=dict(fit.season_bias_by_season),
    )


def _select(
    frame: pd.DataFrame,
    strategies: Sequence[Any],
    ev_floor_t: float | Mapping[str, float],
    *,
    frozen_sd: float,
) -> Any:
    """Route a candidate frame through the SINGLE bet-decision source (LOCKED-2, BET-01).

    The pricing, the admission, the sizing and the grading all happen inside
    ``BetSelector.select``. This runner supplies the frozen inputs and reads the decision set;
    it prices nothing of its own, which is what keeps exactly one bet-decision path in the
    repository.
    """
    selector = BetSelector(
        frozen_sd=frozen_sd,
        season_bias_by_season={},
        ev_floor_t=ev_floor_t,
        strategies=list(strategies),
    )
    return selector.select(frame)


def _per_bet_frame(selected: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """The flat-stake per-bet frame the ROI, the bootstrap and the cuts all read.

    Generalises the Phase-27 ``_records_to_per_bet`` convention from a hardcoded -110 payout to
    the price the bet was ACTUALLY made at (``selected_odds``, D31-04, DEF-31-13). Every target's
    win therefore pays what its own market quoted: the game's moneyline for WP, and the devigged
    stored two-sided price for the spread and total. A bet whose row carried no two-sided price
    pays ``american_to_payout(-110) == 100/110``, which is what the Phase-27 O/U rows all do, so
    that reproduction is exact. A flat -110 payout on a -320 favourite would turn a losing bet into
    a winning one on paper, and the same distortion applies in miniature to an asymmetric -125 /
    +105 spread price.

    A push and an UNGRADED bet both carry a payout of 0.0 against a stake of 1.0, exactly as
    the frozen helper does. They are counted separately on the per-target record so the
    disclosure is on the artifact rather than buried in an average.
    """
    rows: list[dict[str, Any]] = []
    for record in selected:
        outcome = record.get("outcome")
        if outcome is True:
            payout = american_to_payout(int(record["selected_odds"]))
        elif outcome is False:
            payout = -1.0
        else:
            payout = 0.0
        rows.append(
            {
                "game_id": record["game_id"],
                "season": int(record["season"]),
                "week": int(record["week"]),
                "target": record.get("target"),
                "outcome": outcome,
                "flat_stake": 1.0,
                "payout_flat": payout,
                "kelly_stake": float(record.get("kelly_stake") or 0.0),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "season",
            "week",
            "target",
            "outcome",
            "flat_stake",
            "payout_flat",
            "kelly_stake",
        ],
    )


def _robustness_cut_frames(per_bet: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The two named robustness cuts, each as its own frame.

    ``backtest/ou_monetization.py:690-710`` assigns BOTH cut names the value of
    ``_flat_roi_from_records(reg_season)`` from the SAME ``week <= 18`` filter, so the two cuts
    compute the BYTE-IDENTICAL frame. Both names are kept for git-ancestry continuity with the
    Phase-27 record; the identity is asserted by frame comparison rather than argued, and the
    pair contributes ONE entry to the BH family (``ROBUSTNESS_CUT_BH_COUNT``).
    """
    if per_bet.empty:
        return dict.fromkeys(ROBUSTNESS_CUTS_P31, per_bet)
    regular_season = per_bet[per_bet["week"] <= 18]
    return dict.fromkeys(ROBUSTNESS_CUTS_P31, regular_season)


# ---------------------------------------------------------------------------
# The shifted-edge counterfactual control (D31-08)
# ---------------------------------------------------------------------------


def _shift_edge_for_control(target: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Manufacture a positive edge on the ACTUAL hold frame (D31-08).

    A synthetic-only control passes even when a wiring fault -- a missing column, a NaN, a
    silently empty merge -- produced a false zero, so the counterfactual runs on the REAL
    frame the verdict was measured on. Only the model output moves; every market value, key,
    season and week is the frame's own.

    The manufactured edge is deliberately large. This is a LIVENESS check on the selection
    path, not a sensitivity study, and nothing it produces is ever reported as a result: its
    registry entry carries ``entry_kind = control`` and never enters the BH family.
    """
    shifted = frame.copy()
    if shifted.empty:
        return shifted
    if target == "ou":
        shifted["model_total"] = (
            shifted["closing_total"].astype(float) - CONTROL_SHIFT_POINTS
        )
        return shifted
    if target == "ats":
        shifted["model_spread"] = (
            shifted["closing_spread"].astype(float) + CONTROL_SHIFT_POINTS
        )
        return shifted

    # WP: price the side the market already favours at the devigged fair probability PLUS the
    # control edge, so the manufactured edge is positive on the side the chain will bet.
    probabilities: list[float] = []
    for ml_home, ml_away in zip(
        shifted["ml_home"].astype(float), shifted["ml_away"].astype(float)
    ):
        fair_home = float(devig(over_odds=ml_home, under_odds=ml_away)["fair_over"])
        if fair_home >= 0.5:
            probabilities.append(min(fair_home + CONTROL_PROB_EDGE, 0.99))
        else:
            probabilities.append(max(fair_home - CONTROL_PROB_EDGE, 0.01))
    shifted["model_prob"] = probabilities
    return shifted


# ---------------------------------------------------------------------------
# The trial registry
# ---------------------------------------------------------------------------


def _registry_entry(
    *,
    entry_id: str,
    target: str,
    threshold: float,
    subpopulation_rule: str,
    calibration_method: str,
    sd_source: str,
    devig_method: str,
    sizing_policy: str,
    sample_window: str,
    robustness_cut: str,
    raw_p: float | None,
    roi: float | None,
    ci: tuple[float | None, float | None] | None,
    bet_count: int,
    entry_kind: str,
    fallback_trigger: str | None,
) -> dict[str, Any]:
    """One registry row, in the FROZEN pre-registered field order.

    Built from ``dict.fromkeys(TRIAL_REGISTRY_FIELDS_P31)`` so the key order IS the frozen
    tuple's order and cannot drift with the code that fills it. ``entry_id`` and ``target``
    lead the row as identity: the pre-registration requires ONE registry spanning all three
    targets, and the frozen schema -- a per-entry field tuple -- has no field that names which
    target an entry belongs to.
    """
    entry: dict[str, Any] = {"entry_id": entry_id, "target": target}
    entry.update(dict.fromkeys(TRIAL_REGISTRY_FIELDS_P31))
    entry.update(
        {
            "threshold": float(threshold),
            "subpopulation_rule": subpopulation_rule,
            "calibration_method": calibration_method,
            "sd_source": sd_source,
            "devig_method": devig_method,
            "sizing_policy": sizing_policy,
            "sample_window": sample_window,
            "robustness_cut": robustness_cut,
            "raw_p": raw_p,
            "adjusted_p": None,
            "roi": roi,
            "ci": ci,
            "bet_count": int(bet_count),
            "entry_kind": entry_kind,
            "fallback_trigger": fallback_trigger,
        }
    )
    return entry


_SUBPOP_RULE: Mapping[str, str] = {
    "wp": "none (D31-05: no eligibility gate; the EV floor alone decides)",
    "ats": "none (D31-05: no eligibility gate; the EV floor alone decides)",
    # The frozen 2025 run's registry named the Phase-26/27 UNION here (D31-06). That UNION was
    # deleted by D33.2-24, so a run of this code applies no O/U gate and its registry says so; the
    # published 2025 O/U verdict stays in the record labelled old-rule (R16).
    "ou": "none (D33.2-24: no eligibility gate; the EV floor alone decides)",
}
_CALIBRATION_METHOD: Mapping[str, str] = {
    "wp": "deployed_isotonic_used_unchanged (D31-07 default)",
    "ats": "prior_season_mean_bias_correction",
    "ou": "prior_season_mean_bias_subtraction",
}
_SD_SOURCE: Mapping[str, str] = {
    "wp": "none (D31-07: a calibrated classifier has no residual SD)",
    "ats": "frozen_tune_corrected_sd (home-margin scale)",
    "ou": "frozen_tune_corrected_sd (total scale)",
}
# The devig RULE each target is priced under. Per-bet, which rule actually applied is on the
# decision record's own ``devig_method``; these strings name the rule, not a measured split.
_DEVIG_METHOD: Mapping[str, str] = {
    "wp": "real_two_sided_moneyline (per game and per side, D31-04)",
    "ats": (
        "real_two_sided_devig(spread_ju_home, spread_ju_away); flat -110 when the stored juice "
        "is absent (pre-registration 3.2 step 4; DEF-31-13 ruled 2026-09-05, superseding "
        "D31-04's flat-quoted characterisation)"
    ),
    "ou": (
        "real_two_sided_devig(total_over_ju, total_under_ju); flat -110 when the stored juice "
        "is absent (pre-registration 3.3 step 4; DEF-31-13 ruled 2026-09-05, superseding "
        "D31-04's flat-quoted characterisation)"
    ),
}
_SIZING_POLICY: str = (
    "quarter_kelly -> 5pct per-bet cap -> same-game-and-side de-weight -> "
    "10pct POOLED weekly cap (D31-02/D31-03, LOCKED order)"
)


# ---------------------------------------------------------------------------
# The renderer. BOTH artifact forms come from here and nowhere else.
# ---------------------------------------------------------------------------

# The per-target field order in the rendered artifact. IDENTICAL for every target regardless
# of its token (D31-36): omission is structurally impossible rather than something to remember.
_TARGET_FIELD_ORDER: tuple[str, ...] = (
    "verdict_token",
    "verdict_reason",
    "token_meaning",
    "chain_resolved",
    "control_passed",
    "control_bet_count",
    "bets_selected",
    "has_return",
    "flat_roi",
    "raw_roi_p_value",
    "adjusted_roi_p_value",
    "roi_p_value_absent_reason",
    "n_blocks",
    "n_push",
    "n_ungraded",
    "fallback_fired",
    "fallback_trigger",
    "ci_lo",
    "ci_hi",
    "robustness_cut_roi",
    "robustness_cuts_identical",
    "clv_report_only_mean",
    "clv_report_only_p",
    "clv_report_only_n",
)

_TARGET_FLOAT_FIELDS: frozenset[str] = frozenset(
    {
        "flat_roi",
        "raw_roi_p_value",
        "adjusted_roi_p_value",
        "ci_lo",
        "ci_hi",
        "robustness_cut_roi",
        "clv_report_only_mean",
        "clv_report_only_p",
    }
)
_TARGET_INT_FIELDS: frozenset[str] = frozenset(
    {
        "control_bet_count",
        "bets_selected",
        "n_blocks",
        "n_push",
        "n_ungraded",
        "clv_report_only_n",
    }
)
_TARGET_BOOL_FIELDS: frozenset[str] = frozenset(
    {
        "chain_resolved",
        "control_passed",
        "has_return",
        "fallback_fired",
        "robustness_cuts_identical",
    }
)

_REGISTRY_FLOAT_FIELDS: frozenset[str] = frozenset(
    {"threshold", "raw_p", "adjusted_p", "roi", "ci_lo", "ci_hi"}
)


def _render_target_field(name: str, value: Any) -> str:
    """Render one per-target field with the type its column contract fixes."""
    if name in _TARGET_FLOAT_FIELDS:
        return render_float(None if value is None else float(value))
    if name in _TARGET_INT_FIELDS:
        return render_int(0 if value is None else value)
    if name in _TARGET_BOOL_FIELDS:
        return render_bool(value)
    return render_str(value)


def render_verdict_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    """THE ONE RENDERER. Every numeric field in either artifact form comes from here.

    Both artifact forms -- the committed TOML under ``config/`` and the machine-readable run
    record under the gitignored ``outputs/`` tree -- are built from this single mapping of
    rendered STRINGS, so the two cannot disagree about a number. Nothing downstream formats a
    float; a field that skipped this function is exactly what
    ``tests/unit/test_verdict_artifact_determinism.py``'s bare-interpolation scan hunts for.

    Returns:
        ``{"run": {field -> text}, "targets": {target -> {field -> text}},
        "registry": [ {field -> text}, ... ]}``. Every value is already a TOML literal.
    """
    run = result["run"]
    fields_run: dict[str, str] = {
        "requirement": render_str("PROD-04"),
        "preregistration_commit": render_str(run["preregistration_commit"]),
        "preregistration_paths": render_str_list(PREREGISTRATION_PATHS),
        "attempt_id": render_str(run["attempt_id"]),
        "completed_at_utc": render_str(run["completed_at_utc"]),
        "window_label": render_str(run["window_label"]),
        "window_is_the_preregistered_rule": render_bool(
            run["window_is_the_preregistered_rule"]
        ),
        "tune_seasons": render_int_list(run["tune_seasons"]),
        "hold_seasons": render_int_list(run["hold_seasons"]),
        "alpha": render_float(run["alpha"]),
        "correction_method": render_str(run["correction_method"]),
        "bh_denominator": render_int(run["bh_denominator"]),
        "bh_denominator_rule": render_str(run["bh_denominator_rule"]),
        "bh_fallbacks_fired": render_int(run["bh_fallbacks_fired"]),
        "registry_rows": render_int(run["registry_rows"]),
        "excluded_control_entries": render_int(run["excluded_control_entries"]),
        "excluded_tune_side_sweep_cells": render_int(
            run["excluded_tune_side_sweep_cells"]
        ),
        "robustness_cuts": render_str_list(ROBUSTNESS_CUTS_P31),
        "robustness_cut_bh_count": render_int(ROBUSTNESS_CUT_BH_COUNT),
        "roi_p_value_method": render_str(ROI_P_VALUE_METHOD),
        "roi_min_attainable_p": render_float(ROI_MIN_ATTAINABLE_P),
        "clv_p_value_is_report_only": render_bool(CLV_P_VALUE_IS_REPORT_ONLY),
        "bootstrap_b": render_int(run["bootstrap_b"]),
        "bootstrap_seed": render_int(run["bootstrap_seed"]),
    }

    fields_targets: dict[str, dict[str, str]] = {}
    for target, record in result["targets"].items():
        fields_targets[target] = {
            name: _render_target_field(name, record.get(name))
            for name in _TARGET_FIELD_ORDER
        }

    fields_registry: list[dict[str, str]] = []
    for entry in result["registry"]:
        rendered: dict[str, str] = {
            "entry_id": render_str(entry["entry_id"]),
            "target": render_str(entry["target"]),
        }
        for name in TRIAL_REGISTRY_FIELDS_P31:
            value = entry.get(name)
            if name == "ci":
                bounds = value or (None, None)
                rendered["ci_lo"] = render_float(bounds[0])
                rendered["ci_hi"] = render_float(bounds[1])
                continue
            if name == "bet_count":
                rendered[name] = render_int(value or 0)
            elif name in _REGISTRY_FLOAT_FIELDS:
                rendered[name] = render_float(None if value is None else float(value))
            else:
                rendered[name] = render_str(value)
        fields_registry.append(rendered)

    return {"run": fields_run, "targets": fields_targets, "registry": fields_registry}


def render_verdict_toml(result: Mapping[str, Any]) -> str:
    """Render the COMPLETE 2025 verdict as a deterministic, ready-to-commit TOML document.

    The document is GENERATOR OUTPUT: it is written by the runner and committed as-is, never
    hand-edited, following the Phase-30 precedent where the ratified verdict is a git-tracked
    configuration file. Every value comes from :func:`render_verdict_fields`, so two renders of
    the same result are byte-identical.
    """
    fields = render_verdict_fields(result)
    run = fields["run"]
    lines: list[str] = [
        "# =============================================================================",
        f"# {DEFAULT_VERDICT_TOML_PATH.as_posix()} -- the BINDING 2025 clean-split verdict",
        "# Phase 31 (ship the +EV bet list and profitability readout) -- PROD-04, SPEC R3.",
        "#",
        "# GENERATOR OUTPUT. Produced by `python -m backtest.profitability_2025`. Do NOT",
        "# hand-edit any value below: the 2025 split is SINGLE-USE, so a hand-edited value",
        "# cannot be regenerated and is indistinguishable from a tampered one.",
        "#",
        "# The rule that produced this verdict was FROZEN before any 2025 number existed:",
        f"#   pre-registration : {run['preregistration_paths']}",
        f"#   anchor commit    : {run['preregistration_commit']}",
        "# SPEC R3 requires that commit to be a STRICT git ancestor of -- and not equal to --",
        "# the commit recording this artifact.",
        "#",
        "# The CLV p-value below is REPORT-ONLY. It is reported BESIDE the verdict and never",
        "# entered the multiplicity family or drove any token; the verdict rests on the",
        "# pre-registered one-sided ROI bootstrap p-value alone.",
        "#",
        "# ASCII only, no emoji (CLAUDE.md hard constraint).",
        "# =============================================================================",
        "",
        "[run]",
    ]
    lines.extend(f"{name} = {text}" for name, text in run.items())

    for target, target_fields in fields["targets"].items():
        lines.extend(["", f"[targets.{target}]"])
        lines.extend(f"{name} = {text}" for name, text in target_fields.items())

    lines.extend(
        [
            "",
            "# The COMPLETE trial registry, spanning all three targets in ONE artifact. Row",
            "# count and BH denominator differ visibly and BY DESIGN: every read of the hold is",
            "# on the record, and the denominator counts only actual hold-side inferences.",
        ]
    )
    for entry in fields["registry"]:
        lines.append("")
        lines.append("[[registry]]")
        lines.extend(f"{name} = {text}" for name, text in entry.items())

    return "\n".join(lines) + "\n"


def verdict_run_record(result: Mapping[str, Any]) -> dict[str, Any]:
    """The machine-readable run record, built from the SAME rendered fields as the TOML.

    ``verdict_fields`` is the renderer's output VERBATIM, so the two artifact forms cannot
    carry different numbers for one measurement; the determinism test compares the TOML's
    parsed values against these strings directly. Everything else on the record is context the
    committed configuration file does not need to carry -- the fence reports, the report-only
    CLV summaries, the ledger, and the tune-side fit.
    """
    return {
        "verdict_fields": render_verdict_fields(result),
        "run": dict(result["run"]),
        "targets": {
            target: dict(record) for target, record in result["targets"].items()
        },
        "registry": [dict(entry) for entry in result["registry"]],
        "fence_reports": result.get("fence_reports", {}),
        "clv_report_only": result.get("clv_report_only", {}),
        "tune_fit": result.get("tune_fit", {}),
        "ledger": result.get("ledger", {}),
        "bh_family_spec": dict(BH_FAMILY_SPEC),
    }


def write_verdict_artifact(
    result: Mapping[str, Any], toml_path: Path, json_path: Path
) -> None:
    """Write BOTH artifact forms from the one renderer.

    The committed generator-output form lands under ``config/`` (git-tracked, the D24-06/D24-07
    anti-gitignore-landmine control) and the machine-readable run record under the gitignored
    ``outputs/`` tree. Neither may live under ``data/``; the path validation that enforces that
    ran before the measurement.
    """
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(render_verdict_toml(result), encoding="utf-8")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(verdict_run_record(result), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# The one-shot run
# ---------------------------------------------------------------------------


def run_profitability_2025(
    window: FenceWindow = FENCE_WINDOW_REHEARSAL,
    *,
    verdict_toml_path: Path | str | None = None,
    verdict_json_path: Path | str | None = None,
    ledger_path: Path | str | None = None,
    candidates_by_target: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Run the pre-registered chain over a REGISTERED window and judge it.

    THE FROZEN WINDOW'S PATHS ARE NOT OVERRIDABLE. When ``window`` is the pre-registered rule
    the ledger and both artifact paths are the production ones by construction, so a binding
    run cannot be redirected onto throwaway paths and thereby escape the one-shot ledger.
    Conversely a REHEARSAL must NAME its own three paths explicitly -- there is no default --
    so it can never land on a production path by omission, and each one is checked against the
    production set before anything runs.

    The order below is the contract, not a convenience:

    1. Validate both output paths and REFUSE an existing artifact -- before any expensive work.
    2. Resolve the pre-registration anchor from git.
    3. ACQUIRE THE RUN LEDGER -- before the first hold-season read.
    4. Load the TUNE side, fit every nuisance parameter on it, sweep the EV floor on it, and
       assert the Phase-31 fence over every fit input. No hold row has been read yet.
    5. FIRST HOLD READ. Select through the single bet-decision source, grade, bootstrap.
    6. Run the shifted-edge counterfactual control on the SAME hold frame.
    7. Assemble ONE registry, apply the pre-registered correction, assign one token per target.
    8. Write both artifact forms, then mark the ledger ``completed``.

    Any exception after step 3 marks the ledger ``failed`` and re-raises, so the crash window
    between reading the hold and writing the artifact leaves a durable record rather than a
    silently re-spendable split.

    Args:
        window: A REGISTERED fence window. Defaults to the rehearsal proxy, so the frozen
            window is never reached by omission.
        verdict_toml_path: Where the committed form goes. Required for a rehearsal, refused
            for the frozen window.
        verdict_json_path: Where the run record goes. Same rule.
        ledger_path: Where the run ledger goes. Same rule.
        candidates_by_target: Pre-built candidate frames, used instead of reading gold.

    Returns:
        The structured result: ``run``, ``targets``, ``registry``, ``fence_reports``,
        ``clv_report_only``, ``tune_fit`` and ``ledger``.

    Raises:
        ProxySplitRefusedError: when a path override is supplied for the frozen window, when a
            rehearsal omits a path, or when a rehearsal names a production path.
        LeakageError: when any fitted parameter saw a hold season.
        RunLedgerError: when the ledger forbids this invocation.
        VerdictArtifactExistsError: when an artifact already exists at an output path.
    """
    if not any(window == registered for registered in REGISTERED_FENCE_WINDOWS):
        msg = (
            f"refusing to run against the UNREGISTERED window {window!r}. Exactly two windows "
            "are admissible: the frozen Phase-31 rule and the disjoint rehearsal proxy, both "
            "declared in the frozen pre-registration."
        )
        raise ProxySplitRefusedError(msg)

    toml_path, json_path, ledger = _resolve_run_paths(
        window, verdict_toml_path, verdict_json_path, ledger_path
    )

    # (1) The refusal, BEFORE any work. A refusal arriving after a full walk-forward pass
    # would be a refusal nobody could afford to trust.
    resolved_toml = _validate_output_path(
        toml_path, "the committed 2025 verdict artifact"
    )
    resolved_json = _validate_output_path(json_path, "the 2025 verdict run record")

    # (2) The anchor, RESOLVED from git and never transcribed.
    anchor = preregistration_commit()

    # (3) THE LEDGER, before the first hold read.
    ledger_fields = acquire_run_ledger(
        ledger, window=window, preregistration_sha=anchor
    )

    try:
        result = _measure_and_judge(
            window=window,
            anchor=anchor,
            ledger_fields=ledger_fields,
            candidates_by_target=candidates_by_target,
        )
        write_verdict_artifact(result, resolved_toml, resolved_json)
    except BaseException as exc:
        mark_run_ledger(
            ledger,
            _LEDGER_FAILED,
            failure_type=type(exc).__name__,
            failure_message=str(exc)[:500],
        )
        raise

    result["ledger"] = mark_run_ledger(
        ledger,
        _LEDGER_COMPLETED,
        verdict_artifact=str(resolved_toml),
        verdict_run_record=str(resolved_json),
    )
    logger.info(
        "profitability_2025 run complete",
        window=window.label,
        attempt_id=ledger_fields["attempt_id"],
        tokens={
            target: record["verdict_token"]
            for target, record in result["targets"].items()
        },
    )
    return result


def _resolve_run_paths(
    window: FenceWindow,
    verdict_toml_path: Path | str | None,
    verdict_json_path: Path | str | None,
    ledger_path: Path | str | None,
) -> tuple[Path, Path, Path]:
    """Resolve the three run paths under the window's own rule. See the runner's docstring."""
    supplied = (verdict_toml_path, verdict_json_path, ledger_path)
    if window.is_the_preregistered_rule:
        if any(path is not None for path in supplied):
            msg = (
                "the FROZEN Phase-31 window's ledger and artifact paths are not overridable. "
                "Redirecting a binding run onto throwaway paths would let it escape the "
                "one-shot ledger entirely, which is the single control standing between the "
                "unburned 2025 split and being spent twice."
            )
            raise ProxySplitRefusedError(msg)
        return (
            DEFAULT_VERDICT_TOML_PATH,
            DEFAULT_VERDICT_JSON_PATH,
            Path(RUN_LEDGER_PATH),
        )

    if any(path is None for path in supplied):
        msg = (
            "a REHEARSAL run must NAME its verdict artifact, run record and ledger paths "
            "explicitly. There is no default, so a rehearsal cannot land on a production path "
            "by omission -- and a rehearsal result is DISCARDED, never reported, never "
            "published and never compared."
        )
        raise ProxySplitRefusedError(msg)

    resolved = tuple(Path(path) for path in supplied)  # type: ignore[arg-type]
    production = {path.resolve() for path in _PRODUCTION_PATHS}
    for path in resolved:
        if path.resolve() in production:
            msg = (
                f"a rehearsal run named the PRODUCTION path {path}. The rehearsal split is "
                "not the pre-registered rule and its result is discarded; writing it to a "
                "production path would put a discarded number where the binding one belongs."
            )
            raise ProxySplitRefusedError(msg)
    return resolved  # type: ignore[return-value]


def _measure_and_judge(
    *,
    window: FenceWindow,
    anchor: str,
    ledger_fields: Mapping[str, Any],
    candidates_by_target: Mapping[str, pd.DataFrame] | None,
) -> dict[str, Any]:
    """The measurement and the judgement, between ledger acquisition and the artifact write."""
    tune_side_seasons = sorted({*window.prior_residual_seasons, *window.tune_seasons})
    hold_window_label = "hold_" + "_".join(
        str(season) for season in window.hold_seasons
    )

    # (4) TUNE SIDE ONLY. No hold row is read anywhere in this block.
    tune_frames = _load_tune_frames(
        tune_side_seasons, candidates_by_target=candidates_by_target
    )

    fits: dict[str, _TargetFit] = {}
    fence_reports: dict[str, Any] = {}
    registry: list[dict[str, Any]] = []

    for target in CANONICAL_TARGETS:
        tune_frame = tune_frames.get(target, pd.DataFrame())
        if tune_frame.empty:
            msg = (
                f"[{target}] the tune window {tune_side_seasons} produced no candidates, so "
                "no nuisance parameter can be fit. That is a wiring fault, not a measured "
                "zero, and it is refused rather than carried into a verdict."
            )
            raise ValueError(msg)

        chain_fit, gate = _fit_target_on_tune(target, tune_frame, window)

        # The FENCE, over every fit input, BEFORE anything reads the hold. The Phase-31 fence
        # is used and never the Phase-27 one, which reads the Phase-27 window and would
        # therefore PASS while naming the wrong hold seasons (D31-14, T-31-19).
        fence_reports[target] = assert_fit_window_p31(chain_fit, window)

        strategy = _strategy_for_target(target, chain_fit, gate)
        tune_only = tune_frame[tune_frame["season"].isin(window.tune_seasons)].copy()

        sweep_rows: list[dict[str, Any]] = []
        tune_bet_counts: dict[float, int] = {}
        best_t: float | None = None
        best_roi: float | None = None
        for floor in EV_FLOOR_GRID:
            selection = _select(
                tune_only,
                [strategy],
                float(floor),
                frozen_sd=float(chain_fit.frozen_sd or 1.0),
            )
            per_bet = _per_bet_frame(selection.selected)
            # The POINT ESTIMATE only. A tune-side cell carries no interval and no p-value in
            # the registry -- it is excluded from the BH family by construction -- so running
            # the 2000-replicate bootstrap on it would compute a number nothing reads.
            roi = None if per_bet.empty else flat_roi(per_bet)
            tune_bet_counts[float(floor)] = len(selection.selected)
            sweep_rows.append(
                _registry_entry(
                    entry_id=f"{target}/sweep/t={render_float(float(floor))}",
                    target=target,
                    threshold=float(floor),
                    subpopulation_rule=_SUBPOP_RULE[target],
                    calibration_method=_CALIBRATION_METHOD[target],
                    sd_source=_SD_SOURCE[target],
                    devig_method=_DEVIG_METHOD[target],
                    sizing_policy=_SIZING_POLICY,
                    sample_window=window.threshold_window,
                    robustness_cut="none",
                    # A tune-side cell carries NO p-value. The only p-value in reach on the
                    # tune split is a CLV one, and a CLV p-value has no business in this
                    # registry's raw_p column even on a row that is excluded from the family.
                    raw_p=None,
                    roi=roi,
                    ci=None,
                    bet_count=len(selection.selected),
                    entry_kind=TRIAL_ENTRY_KIND_INFERENCE,
                    fallback_trigger=None,
                )
            )
            if (
                len(selection.selected)
                and roi is not None
                and (best_roi is None or roi > best_roi)
            ):
                best_roi = roi
                best_t = float(floor)

        # ROI -- not significance -- chooses the frozen t, exactly as the Phase-27 sweep does.
        # When no floor admits a bet on the tune split, freeze the GRID FLOOR and let the hold
        # report an honest zero-bet result. Never invent a t outside the pre-registered grid.
        chosen_t = best_t if best_t is not None else float(EV_FLOOR_GRID[0])

        registry.extend(sweep_rows)
        fits[target] = _TargetFit(
            target=target,
            chain_fit=chain_fit,
            gate=gate,
            fence_report=fence_reports[target],
            ev_floor_t=chosen_t,
            sweep_rows=sweep_rows,
            tune_bet_counts=tune_bet_counts,
        )

    # (5) THE FIRST HOLD READ.
    hold_frames = _load_hold_frames(
        list(window.hold_seasons), candidates_by_target=candidates_by_target
    )

    strategies = [
        _strategy_for_target(target, fits[target].chain_fit, fits[target].gate)
        for target in CANONICAL_TARGETS
    ]
    ev_floor_by_target = {
        target: fits[target].ev_floor_t for target in CANONICAL_TARGETS
    }
    combined = pd.concat(
        [hold_frames.get(target, pd.DataFrame()) for target in CANONICAL_TARGETS],
        ignore_index=True,
    )
    selection = _select(
        combined,
        strategies,
        ev_floor_by_target,
        frozen_sd=float(fits["ou"].chain_fit.frozen_sd or 1.0),
    )

    measurements: list[TargetMeasurement] = []
    clv_report_only: dict[str, Any] = {}
    per_target_extra: dict[str, dict[str, Any]] = {}
    raw_by_entry: dict[str, float] = {}
    # ``{fallback_entry_id: primary_entry_id}`` for every target whose calibration fallback
    # fired (WR-11). The fallback row is a BOOKKEEPING row, not a second hypothesis -- see the
    # comment where it is appended -- so it is mapped to its primary's adjusted q rather than
    # entering the ranking with a duplicate of its primary's raw p.
    fallback_mirrors: dict[str, str] = {}

    for target in CANONICAL_TARGETS:
        hold_frame = hold_frames.get(target, pd.DataFrame())
        selected = [r for r in selection.selected if r.get("target") == target]
        rejected = [r for r in selection.rejected if r.get("target") == target]
        unpriceable = sum(
            1 for r in rejected if r.get("rejection_reason") in _UNPRICEABLE_REASONS
        )
        chain_resolved = bool(len(hold_frame)) and unpriceable < len(hold_frame)

        per_bet = _per_bet_frame(selected)
        stats = roi_ci_and_p(per_bet)
        cuts = _robustness_cut_frames(per_bet)
        cut_frames = list(cuts.values())
        cuts_identical = (
            all(frame.equals(cut_frames[0]) for frame in cut_frames[1:])
            if cut_frames
            else True
        )
        cut_stats = roi_ci_and_p(cut_frames[0]) if cut_frames else roi_ci_and_p(per_bet)

        n_push = sum(
            1
            for r in selected
            if r.get("outcome") is None and r.get("_actual_total") is not None
        )
        n_ungraded = sum(
            1
            for r in selected
            if r.get("outcome") is None and r.get("_actual_total") is None
        )

        # The shifted-edge counterfactual control, on the ACTUAL hold frame (D31-08).
        control_selection = _select(
            _shift_edge_for_control(target, hold_frame),
            [_strategy_for_target(target, fits[target].chain_fit, fits[target].gate)],
            fits[target].ev_floor_t,
            frozen_sd=float(fits[target].chain_fit.frozen_sd or 1.0),
        )
        control_per_bet = _per_bet_frame(control_selection.selected)
        # The control's return is a LIVENESS observation on manufactured input and is never a
        # result, so it too carries a point estimate and no interval or p-value.
        control_roi = None if control_per_bet.empty else flat_roi(control_per_bet)

        gate = fits[target].gate
        fallback_fired = bool(gate is not None and gate.fallback_fired)
        fallback_trigger = gate.fallback_trigger if gate is not None else None

        primary_id = f"{target}/primary"
        registry.append(
            _registry_entry(
                entry_id=primary_id,
                target=target,
                threshold=fits[target].ev_floor_t,
                subpopulation_rule=_SUBPOP_RULE[target],
                calibration_method=_CALIBRATION_METHOD[target],
                sd_source=_SD_SOURCE[target],
                devig_method=_DEVIG_METHOD[target],
                sizing_policy=_SIZING_POLICY,
                sample_window=hold_window_label,
                robustness_cut="none",
                raw_p=stats["p_value"],
                roi=stats["point_estimate"],
                ci=(stats["ci_lo"], stats["ci_hi"]),
                bet_count=len(selected),
                entry_kind=TRIAL_ENTRY_KIND_INFERENCE,
                fallback_trigger=None,
            )
        )
        if stats["p_value"] is not None:
            raw_by_entry[primary_id] = float(stats["p_value"])

        cut_id = f"{target}/robustness"
        registry.append(
            _registry_entry(
                entry_id=cut_id,
                target=target,
                threshold=fits[target].ev_floor_t,
                subpopulation_rule=_SUBPOP_RULE[target],
                calibration_method=_CALIBRATION_METHOD[target],
                sd_source=_SD_SOURCE[target],
                devig_method=_DEVIG_METHOD[target],
                sizing_policy=_SIZING_POLICY,
                sample_window=hold_window_label,
                # BOTH cut names on ONE row. They compute the byte-identical frame
                # (ou_monetization.py:690-710), so one statistic is reported twice and the
                # pair contributes ONE entry to the family (ROBUSTNESS_CUT_BH_COUNT).
                robustness_cut=" + ".join(ROBUSTNESS_CUTS_P31)
                + " (byte-identical frames; counted once)",
                raw_p=cut_stats["p_value"],
                roi=cut_stats["point_estimate"],
                ci=(cut_stats["ci_lo"], cut_stats["ci_hi"]),
                bet_count=len(cut_frames[0]) if cut_frames else 0,
                entry_kind=TRIAL_ENTRY_KIND_INFERENCE,
                fallback_trigger=None,
            )
        )
        if cut_stats["p_value"] is not None:
            raw_by_entry[cut_id] = float(cut_stats["p_value"])

        if fallback_fired:
            fallback_id = f"{target}/calibration_fallback"
            registry.append(
                _registry_entry(
                    entry_id=fallback_id,
                    target=target,
                    threshold=fits[target].ev_floor_t,
                    subpopulation_rule=_SUBPOP_RULE[target],
                    calibration_method=_CALIBRATION_METHOD[target],
                    sd_source=_SD_SOURCE[target],
                    devig_method=_DEVIG_METHOD[target],
                    sizing_policy=_SIZING_POLICY,
                    sample_window=hold_window_label,
                    robustness_cut="none",
                    raw_p=stats["p_value"],
                    roi=stats["point_estimate"],
                    ci=(stats["ci_lo"], stats["ci_hi"]),
                    bet_count=len(selected),
                    entry_kind=TRIAL_ENTRY_KIND_INFERENCE,
                    fallback_trigger=fallback_trigger,
                )
            )
            # IT DOES NOT ENTER THE RANKING (WR-11), THOUGH IT DOES COUNT TOWARD m.
            #
            # When the fallback fires, the strategy is built WITH the gate, so there is exactly
            # ONE selection and ONE statistic. This row records that same statistic a second time
            # -- ``raw_p=stats["p_value"]`` and ``roi=stats["point_estimate"]`` are the identical
            # numbers as this target's ``/primary`` row. It is a disclosure that the fallback
            # fired and what it produced, NOT an independent hypothesis.
            #
            # Adding it to ``raw_by_entry`` put the SAME p-value into the BH input twice. Because
            # a tied p occupies two adjacent ranks and the step-up takes ``min_{j>=i} m*p_(j)/j``,
            # the duplicate at rank i+1 gives ``m*p/(i+1) < m*p/i`` and pulls the PRIMARY's
            # adjusted q DOWN. Measured on the shipped run's p-value shape, with the family
            # growing 6 -> 7: WP's q went 0.600 -> 0.560 WITH the duplicate, against 0.700
            # without it. So firing a fallback made the verdict EASIER to call significant --
            # the opposite of what a multiplicity correction is for.
            #
            # The FROZEN ``BH_FAMILY_SPEC`` pre-registers the MEMBERSHIP ("0 to 3 entries") and
            # the denominator ("7, 8 or 9 -- one additional entry per target whose calibration
            # fallback fired"). Both are preserved: ``bh_family_members`` selects on
            # ``entry_kind``/``sample_window``, not on ``raw_p``, so this row still grows ``m``.
            # What was never pre-registered is that it would carry the identical statistic INTO
            # the ranking.
            #
            # Its ``adjusted_p`` is mirrored from the primary's after the correction, so the row
            # is complete and visibly states the same q as the inference it duplicates.
            #
            # NOTE: no fallback fired in the shipped 2025 run (bh_denominator = 6), so nothing
            # published is affected. The correction only ever becomes STRICTER here.
            fallback_mirrors[fallback_id] = primary_id

        registry.append(
            _registry_entry(
                entry_id=f"{target}/control/shifted_edge",
                target=target,
                threshold=fits[target].ev_floor_t,
                subpopulation_rule=_SUBPOP_RULE[target],
                calibration_method=_CALIBRATION_METHOD[target],
                sd_source=_SD_SOURCE[target],
                devig_method=_DEVIG_METHOD[target],
                sizing_policy=_SIZING_POLICY,
                sample_window=hold_window_label,
                robustness_cut="none",
                # A control has NO p-value to correct, which is why it is excluded from the
                # family: counting it would statistically penalise the verdict for running a
                # code-liveness check.
                raw_p=None,
                roi=control_roi,
                ci=None,
                bet_count=len(control_selection.selected),
                entry_kind=TRIAL_ENTRY_KIND_CONTROL,
                fallback_trigger=None,
            )
        )

        # REPORT-ONLY CLV, computed and carried OUTSIDE the verdict path. It is disclosed
        # rather than suppressed, and it never reaches TargetMeasurement.
        clv_values = [float(r["clv"]) for r in selected if r.get("clv") is not None]
        clv_report_only[target] = (
            dict(clv_significance(clv_values)) if clv_values else None
        )

        per_target_extra[target] = {
            "ci_lo": stats["ci_lo"],
            "ci_hi": stats["ci_hi"],
            "robustness_cut_roi": cut_stats["point_estimate"],
            "robustness_cuts_identical": cuts_identical,
        }
        measurements.append(
            TargetMeasurement(
                target=target,
                chain_resolved=chain_resolved,
                control_passed=bool(control_selection.selected),
                control_bet_count=len(control_selection.selected),
                bet_count=len(selected),
                flat_roi=stats["point_estimate"],
                adjusted_p=None,
                raw_p=stats["p_value"],
                p_absent_reason=stats["p_value_absent_reason"],
                n_blocks=int(stats["n_blocks"]),
                n_push=n_push,
                n_ungraded=n_ungraded,
                fallback_fired=fallback_fired,
                fallback_trigger=fallback_trigger,
            )
        )

    # (7) The PRE-REGISTERED correction over the enumerated family.
    family = bh_family_members(registry, hold_window_label)
    denominator = len(family)
    tested_ids = [
        entry["entry_id"] for entry in family if entry["entry_id"] in raw_by_entry
    ]
    adjusted = benjamini_hochberg_adjusted(
        [raw_by_entry[entry_id] for entry_id in tested_ids], denominator
    )
    adjusted_by_entry = dict(zip(tested_ids, adjusted, strict=True))
    # The fallback rows mirror their primary's q -- they ARE that inference, recorded a second
    # time (WR-11). Mirroring rather than re-deriving is what keeps them out of the ranking.
    for fallback_id, primary_id in fallback_mirrors.items():
        if primary_id in adjusted_by_entry:
            adjusted_by_entry[fallback_id] = adjusted_by_entry[primary_id]
    for entry in registry:
        if entry["entry_id"] in adjusted_by_entry:
            entry["adjusted_p"] = adjusted_by_entry[entry["entry_id"]]

    measurements = [
        replace(
            measurement,
            adjusted_p=adjusted_by_entry.get(f"{measurement.target}/primary"),
        )
        for measurement in measurements
    ]

    verdicts = build_verdict_records(measurements)
    for target, record in verdicts.items():
        record.update(per_target_extra[target])
        summary = clv_report_only.get(target) or {}
        record["clv_report_only_mean"] = summary.get("mean")
        record["clv_report_only_p"] = summary.get("p")
        record["clv_report_only_n"] = summary.get("n", 0)

    counts = registry_exclusion_counts(registry, hold_window_label)
    fallbacks_fired = sum(
        1 for measurement in measurements if measurement.fallback_fired
    )

    return {
        "run": {
            "preregistration_commit": anchor,
            "attempt_id": ledger_fields["attempt_id"],
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "window_label": window.label,
            "window_is_the_preregistered_rule": window.is_the_preregistered_rule,
            "tune_seasons": list(window.tune_seasons),
            "hold_seasons": list(window.hold_seasons),
            "hold_window_label": hold_window_label,
            "alpha": ALPHA,
            "correction_method": str(BH_FAMILY_SPEC["correction_method"]),
            "bh_denominator": denominator,
            "bh_denominator_rule": str(BH_FAMILY_SPEC["scope"]),
            "bh_fallbacks_fired": fallbacks_fired,
            "bootstrap_b": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            **counts,
        },
        "targets": verdicts,
        "registry": registry,
        "fence_reports": fence_reports,
        "clv_report_only": clv_report_only,
        "tune_fit": {
            target: {
                "ev_floor_t": fits[target].ev_floor_t,
                "frozen_sd": fits[target].chain_fit.frozen_sd,
                "tune_bet_counts": fits[target].tune_bet_counts,
                "season_bias_by_season": dict(
                    fits[target].chain_fit.season_bias_by_season
                ),
                "calibration_gate_passed": (
                    None if fits[target].gate is None else fits[target].gate.passed
                ),
            }
            for target in CANONICAL_TARGETS
        },
        "ledger": dict(ledger_fields),
    }


def run_armed_2025_verdict(**forbidden: Any) -> dict[str, Any]:
    """The ARMED production entry point. Takes NO split, window or path argument.

    Its window is the frozen Phase-31 rule and its ledger and artifact paths are the production
    ones, both by construction. A rehearsal configuration therefore cannot leak into a binding
    run: there is no parameter to pass one through, and the catch-all below turns an attempt to
    do so into a NAMED refusal rather than a bare ``TypeError``.

    It additionally requires a pre-existing ``armed`` ledger (CHECKPOINT 3, D31-16). Running
    this module by accident refuses; arming is the owner's deliberate act.

    Raises:
        ProxySplitRefusedError: for ANY keyword argument.
    """
    if forbidden:
        msg = (
            f"run_armed_2025_verdict accepts NO arguments; got {sorted(forbidden)}. The "
            "armed run's window is the frozen Phase-31 rule and its paths are the production "
            "ones, both by construction. The rehearsal proxy split is explicitly NOT "
            "consumable by the armed run, and there is no parameter that would let it be."
        )
        raise ProxySplitRefusedError(msg)
    return run_profitability_2025(FENCE_WINDOW_P31)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    It exposes NO flag that overwrites, forces, skips a refusal or clears the run ledger. That
    absence is the design: the 2025 split is single-use, and a flag that could spend it twice
    would be found and used eventually.
    """
    return argparse.ArgumentParser(
        description=(
            "Run the BINDING one-shot 2025 clean-split profitability verdict across all "
            "three targets and write the verdict artifact. Requires an ARMED run ledger "
            "(CHECKPOINT 3). There is no force flag, no overwrite flag and no way to "
            "re-spend the split. READ-ONLY with respect to data/."
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run the armed verdict and report where the artifacts landed."""
    build_parser().parse_args(argv)
    result = run_armed_2025_verdict()
    for target, record in result["targets"].items():
        logger.info(
            "2025 verdict",
            target=target,
            token=record["verdict_token"],
            bets_selected=record["bets_selected"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
