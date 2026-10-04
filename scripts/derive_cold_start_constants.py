"""The DETERMINISTIC derivation that EMITS the Phase-33 cold-start pre-registration.

Phase 33, Plan 33-16 Task 2 (COLD-07, CLEAN-01, D33-19/D33-20/D33-03/D33-21).

WHY THIS PROGRAM EXISTS AT ALL
-------------------------------
The two quantities this phase pre-registers -- the 2026 chain-fit bias and the six per-target
edge thresholds -- could have been measured in an interactive session and the resulting numbers
typed into a frozen module. A future reviewer would then be able to check the arithmetic and
nothing else: not which artifacts were scored, not which rows were eligible, not which quantile
convention produced a threshold. This program is what makes those three things reproducible.
It takes the END-STATE artifact IDS AND THEIR CONTENT DIGESTS as explicit arguments, refuses
when any input has moved, and EMITS both halves of the pre-registration.

WHAT "DETERMINISTIC" MEANS HERE, PRECISELY
-------------------------------------------
Running this twice with the same arguments against the same inputs produces BYTE-IDENTICAL
output. There is no timestamp in the emitted files, no dict is rendered in insertion order, and
every float is rendered through an explicit format rather than through whatever ``str`` happens
to do. That is why the emitted module can be committed as a frozen rule: a reviewer can re-run
this and diff.

THE DIGEST RULE, STATED ONCE
-----------------------------
Two instruments, chosen by WHERE the input lives and never mixed silently:

  * GIT-TRACKED TEXT inputs -- here, only ``config/phase33_gate_verdict.toml`` -- are digested
    over NEWLINE-NORMALIZED bytes. This repository has ``core.autocrlf=true`` and no
    ``.gitattributes``, so a tracked text file is LF in the git blob and CRLF in a fresh
    Windows working tree. A raw-byte digest would pin a value that holds only on the machine
    that measured it.
  * PRODUCTION-STORE inputs -- everything under ``artifacts/`` and ``data/``, which are BOTH
    gitignored -- are digested over RAW bytes. Normalization would buy nothing (there is no git
    blob to agree with) and would cost agreement with ``tests.data_boundary.digest_file``, the
    instrument this repository already uses for those stores. The raw rule is what makes this
    derivation's recorded digest for ``artifacts/latest.json`` literally equal to the committed
    ``tests.phase33_state.POST_GATE_MANIFEST_DIGEST``, so the two records are cross-checkable
    rather than merely consistent-looking.

An ARTIFACT DIRECTORY is digested as a MANIFEST: the sha256 over ``<relpath>\\0<file sha256>\\n``
lines for every file under it, sorted by POSIX relative path. So a digest identifies the whole
directory -- model, calibrator, feature list, preprocessing and metadata together -- rather
than one file that happens to be the one somebody thought to check.

THE EMITTED MODULE IS RUFF-FORMATTED BEFORE IT IS WRITTEN, ON PURPOSE
----------------------------------------------------------------------
This repository runs ``ruff-format`` as a pre-commit hook. An emitted file that the hook would
then rewrite is a file whose COMMITTED bytes differ from what this program produces -- so the
committed rule could not be reproduced by re-running the program, which is the entire claim
this derivation exists to support. Worse, the hook would be silently reformatting a FROZEN
pre-registration. So the formatter runs HERE, as the last step of emission, and the formatter's
version is recorded in the emitted module: a ruff upgrade that changed the formatting would
then fail the reproduce-check loudly and for a stated reason, rather than quietly.

WHAT THIS PROGRAM DOES NOT DO
------------------------------
It writes NOTHING under ``data/``, ``outputs/`` or ``artifacts/``. Those are guarded production
stores and this is a read-only measurement of them. Its only writes are the two rule files it
is told to emit.

It also declares NO second estimator. The 2026 bias comes from the EXISTING
``backtest.ou_ev_chain.estimate_prior_season_bias``, which is imported rather than reimplemented
(D33-03/D33-21: "no second estimator is written").

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from backtest.ou_ev_chain import estimate_prior_season_bias
from models.market_probability import oof_market_probability

REPO_ROOT = Path(__file__).resolve().parents[1]

# The three canonical targets, in the one order every emitted mapping is rendered in.
TARGETS: tuple[str, str, str] = ("ats", "ou", "wp")

# The two pre-registration files this program emits. Repo-root-relative POSIX paths, and the
# same two names the emitted module's own ``PREREGISTRATION_PATHS`` carries.
MODULE_PATH = "backtest/cold_start_constants.py"
DOCUMENT_PATH = "COLD-START-PREREGISTRATION.md"

# The population the thresholds are derived on: the D31-04 pinned 2021-2024 backtest
# population, which is ALSO the population the label movement is measured against. That
# circularity is stated rather than discovered -- see the emitted module.
# The frozen window is stated ONCE, as a window (Plan 33.2-17, SPEC R10): a `*_FIRST_SEASON`
# name assigned a year literal reads as a COVERAGE FLOOR, and every coverage floor lives in
# conf/season_partition.py alone. This pair is NOT a floor -- it is a pinned measurement span
# -- so its literal lives under a name that says so and the two bounds derive from it. The
# values are unchanged.
THRESHOLD_WINDOW_SEASONS: tuple[int, int] = (2021, 2024)
THRESHOLD_FIRST_SEASON = THRESHOLD_WINDOW_SEASONS[0]
THRESHOLD_LAST_SEASON = THRESHOLD_WINDOW_SEASONS[1]

# The pool the 2026 chain-fit bias is estimated from: strictly-prior completed seasons.
BIAS_WINDOW_SEASONS: tuple[int, int] = (2021, 2025)
BIAS_FIRST_SEASON = BIAS_WINDOW_SEASONS[0]
BIAS_LAST_SEASON = BIAS_WINDOW_SEASONS[1]
BIAS_TARGET_SEASON = 2026

# The quantile convention, NAMED rather than left to a library default that could change on an
# upgrade. ``numpy.quantile(..., method="linear")`` is numpy's own documented default today;
# pinning it here means a future numpy that changed its default would not silently move a
# frozen threshold.
QUANTILE_METHOD = "linear"

# WP's thresholds do NOT move (D33-20). Stated as literals here because they are the ANCHOR the
# other two targets are derived against, not an output of this program.
WP_HIGH_THRESHOLD = 0.05
WP_MEDIUM_THRESHOLD = 0.02

# The O/U edge's denominator floor, mirroring ``api.cache`` exactly.
OU_TOTAL_FLOOR = 30.0

# Plan 33-08's declared allowance, restated in the emitted module as part of the record.
FIX_CYCLE_ALLOWANCE = 0

# The per-target residual/label contract, in the shape
# ``backtest.profitability_2025._MODEL_COLUMN`` already uses. One contract for all three
# targets, each on its own scale.
MODEL_COLUMN: Mapping[str, str] = {
    "wp": "model_prob",
    "ats": "model_spread",
    "ou": "model_total",
}

# The gitignored production stores. An input under one of these is digested over RAW bytes; see
# the module docstring's digest rule. ``outputs/`` joined with the superseding correction (Plan
# 33.2-26), whose chain-fit record lives there; the 11761c7 derivation reads nothing under it.
_PRODUCTION_STORE_PREFIXES = ("artifacts/", "data/", "outputs/")

_TEXT_SUFFIXES = (".json", ".toml")


class DigestMismatchError(RuntimeError):
    """A declared input digest does not match the file on disk, so the run REFUSES.

    Raised by name rather than warned about. An artifact that moved between the gate and this
    derivation is a DIFFERENT measurement, and a threshold frozen against it would be frozen
    against something other than what the record says was scored.
    """


class EmptyResidualPoolError(ValueError):
    """No strictly-prior season is available, so NO bias is invented (R10 edge).

    There is no fallback to the target season's own data. The underlying refusal comes from
    ``estimate_prior_season_bias``; this type exists so the refusal keeps a Phase-33 name as it
    crosses this boundary.
    """


# ---------------------------------------------------------------------------
# The digest instruments
# ---------------------------------------------------------------------------


def digest_bytes(raw: bytes) -> str:
    """sha256 over RAW bytes."""
    return hashlib.sha256(raw).hexdigest()


def digest_text_file(path: Path) -> str:
    """sha256 over a text file's NEWLINE-NORMALIZED bytes (see the module docstring)."""
    return digest_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def digest_binary_file(path: Path) -> str:
    """sha256 over a binary file's RAW bytes."""
    return digest_bytes(path.read_bytes())


def digest_input_file(path: Path, key: str | None = None) -> str:
    """Digest ONE input file with the instrument its LOCATION requires.

    Args:
        path: The file to digest.
        key: The repo-relative POSIX key this input is recorded under. When given it is what
            decides the instrument; when omitted the decision falls back to the suffix, which
            is the right answer for a file inside an artifact directory (always raw).
    """
    if key is not None and key.startswith(_PRODUCTION_STORE_PREFIXES):
        return digest_binary_file(path)
    if key is not None and path.suffix.lower() in _TEXT_SUFFIXES:
        return digest_text_file(path)
    return digest_binary_file(path)


def digest_artifact_dir(root: Path) -> str:
    """Digest a whole artifact DIRECTORY as a sorted manifest of its files.

    Args:
        root: The artifact directory, e.g. ``artifacts/wp_20260914_221745``.

    Returns:
        The sha256 over ``<relpath>\\0<file sha256>\\n`` lines, sorted by relative POSIX path.

    Raises:
        FileNotFoundError: when the directory is absent. An absent artifact is not an empty
            one, and returning a digest for nothing would let a missing input pass as a match.
    """
    if not root.is_dir():
        msg = f"artifact directory not found: {root.as_posix()}"
        raise FileNotFoundError(msg)
    lines: list[bytes] = []
    for candidate in sorted(
        root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()
    ):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        lines.append(
            relative.encode("utf-8")
            + b"\0"
            + digest_input_file(candidate).encode("ascii")
            + b"\n"
        )
    return digest_bytes(b"".join(lines))


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Derivation:
    """Everything the two emitted files state, measured once and rendered twice."""

    artifacts: Mapping[str, str]
    residual_sources: Mapping[str, Mapping[str, Any]]
    thresholds: Mapping[str, tuple[float, float]]
    raw_thresholds: Mapping[str, tuple[float, float]]
    wp_reference_shares: Mapping[str, float]
    band_shares_after: Mapping[str, Mapping[str, float]]
    band_counts_after: Mapping[str, Mapping[str, int]]
    band_shares_under_current: Mapping[str, Mapping[str, float]]
    band_counts_under_current: Mapping[str, Mapping[str, int]]
    games_changing_band: Mapping[str, int]
    bias: Mapping[str, float]
    bias_by_season: Mapping[str, Mapping[int, float]]
    eligible_counts: Mapping[str, Mapping[str, int]]
    input_digests: Mapping[str, str]
    formatter: str


def _verify_digests(declared: Mapping[str, str], repo_root: Path) -> dict[str, str]:
    """Recompute every declared input digest and REFUSE on the first mismatch.

    Runs BEFORE any scoring, deliberately: a run that is going to refuse should refuse in a
    second rather than after three model loads.

    Args:
        declared: ``{input key -> declared digest}``. Keys are either repo-relative file paths
            or ``artifact:<id>`` for a whole artifact directory.
        repo_root: The repository root every relative key resolves against.

    Returns:
        The recomputed digests, in the same key order the caller declared them.

    Raises:
        DigestMismatchError: naming the input and BOTH digests.
    """
    measured: dict[str, str] = {}
    for key in sorted(declared):
        if key.startswith("artifact:"):
            actual = digest_artifact_dir(repo_root / "artifacts" / key.split(":", 1)[1])
        else:
            actual = digest_input_file(repo_root / key, key)
        expected = declared[key]
        if actual != expected:
            msg = (
                f"declared digest does not match the input on disk for {key!r}: "
                f"declared {expected}, measured {actual}. An input that moved between the "
                "gate and this derivation is a DIFFERENT measurement; the derivation refuses "
                "rather than freezing a threshold against something other than what the "
                "record says was scored."
            )
            raise DigestMismatchError(msg)
        measured[key] = actual
    return measured


def _residual_source(
    target: str,
    manifest: Mapping[str, Any],
    verdicts: Mapping[str, Any],
) -> dict[str, Any]:
    """Decide, from the COMMITTED record, where this target's residuals come from.

    Reads ``config/phase33_gate_verdict.toml`` for the verdict and the candidate/incumbent IDs,
    and ``artifacts/latest.json`` for what is actually deployed. The deployed artifact is the
    residual source either way; what this records is WHICH it is and under what verdict, so a
    FAIL verdict shipped under an owner override is visible as exactly that rather than being
    presented as a pass.
    """
    row = verdicts["verdicts"][target]
    deployed = str(manifest[target])
    candidate = str(row["candidate_artifact"])
    incumbent = str(row["incumbent_artifact"])
    verdict = str(row["verdict"])
    if deployed == candidate:
        kind = "promoted_refit"
    elif deployed == incumbent:
        kind = "retained_incumbent"
    else:
        msg = (
            f"the deployed {target} artifact {deployed!r} is neither the gate's candidate "
            f"({candidate!r}) nor its incumbent ({incumbent!r}); the residual source cannot "
            "be decided from the committed record."
        )
        raise RuntimeError(msg)
    return {
        "artifact": deployed,
        "kind": kind,
        "verdict": verdict,
        "promoted_against_verdict": bool(
            kind == "promoted_refit" and verdict != "PASS"
        ),
    }


def _load_gold(
    target: str, first_season: int, last_season: int, repo_root: Path
) -> pd.DataFrame:
    """Read one target's gold matrix, sliced to an inclusive season range. READ-ONLY."""
    gold = pd.read_parquet(repo_root / "data" / "gold" / f"features_{target}.parquet")
    sliced = gold[(gold["season"] >= first_season) & (gold["season"] <= last_season)]
    return sliced.reset_index(drop=True)


def _edge_frame(scored: Mapping[str, pd.DataFrame], odds: pd.DataFrame) -> pd.DataFrame:
    """Build the three per-game edges EXACTLY as ``api.cache._load_predictions`` does.

    The expressions are mirrored rather than imported because the cache computes them inline on
    a frame it assembles from ``outputs/backtest/predictions_all.csv``. Mirroring them here is
    what makes the thresholds thresholds ON THE PUBLISHED EDGE rather than on a near-relative
    of it; the shapes are pinned by ``tests/unit/test_cold_start_derivation_cli.py``.
    """
    from utils.probability_utils import moneyline_to_probability

    frame = (
        scored["wp"][["game_id", "season", "model_prob"]]
        .merge(scored["ats"][["game_id", "model_spread"]], on="game_id", how="inner")
        .merge(scored["ou"][["game_id", "model_total"]], on="game_id", how="inner")
        .merge(
            odds[["game_id", "ml_home", "ml_away", "spread", "total"]],
            on="game_id",
            how="left",
        )
    )

    # WP: the probability CLV the cache reads, recomputed through the SAME proportional devig
    # ``models.clv.compute_probability_clv`` uses. NULL where the game has no moneyline.
    has_ml = frame["ml_home"].notna() & frame["ml_away"].notna()
    frame["wp_edge"] = np.nan
    if bool(has_ml.any()):
        home_raw = frame.loc[has_ml, "ml_home"].apply(
            lambda ml: moneyline_to_probability(int(ml))
        )
        away_raw = frame.loc[has_ml, "ml_away"].apply(
            lambda ml: moneyline_to_probability(int(ml))
        )
        fair_home = home_raw / (home_raw + away_raw)
        frame.loc[has_ml, "wp_edge"] = frame.loc[has_ml, "model_prob"] - fair_home

    # ATS: model home margin MINUS market home margin, in POINTS, with NO denominator
    # (R13 / D33-05, Plan 33-10). A game with NO STORED SPREAD has NO edge; a pick-em is a REAL
    # line and keeps its nonzero disagreement (D33-31).
    frame["ats_edge"] = (frame["model_spread"] - frame["spread"]).where(
        frame["spread"].notna()
    )

    # O/U: model total vs market total, normalized by the market total floored at 30.
    frame["ou_edge"] = (frame["model_total"] - frame["total"]) / frame["total"].clip(
        lower=OU_TOTAL_FLOOR
    )
    return frame


def _band(magnitudes: np.ndarray, high: float, medium: float) -> np.ndarray:
    """The three-band rule, with ``utils.edge_tier``'s STRICT ``>`` comparisons."""
    return np.where(
        magnitudes > high, "high", np.where(magnitudes > medium, "medium", "low")
    )


def _shares_and_counts(
    magnitudes: np.ndarray, high: float, medium: float
) -> tuple[dict[str, float], dict[str, int]]:
    """Band shares (rounded to 4 dp) and raw game counts over ONE target's eligible rows."""
    banded = _band(magnitudes, high, medium)
    total = len(banded)
    counts = {
        label: int((banded == label).sum()) for label in ("low", "medium", "high")
    }
    shares = {label: round(counts[label] / total, 4) for label in counts}
    return shares, counts


def _anchored_pair(
    magnitudes: np.ndarray, cumulative_low: float, cumulative_low_medium: float
) -> tuple[float, float]:
    """The UNROUNDED ``(high, medium)`` pair reproducing WP's band shares on *magnitudes*.

    The medium threshold is the quantile at WP's cumulative "low" share and the high threshold
    the quantile at its cumulative "low + medium" share, under :data:`QUANTILE_METHOD`. ONE
    implementation, read by the 11761c7 derivation and by its superseding correction alike.
    """
    raw_medium = float(np.quantile(magnitudes, cumulative_low, method=QUANTILE_METHOD))
    raw_high = float(
        np.quantile(magnitudes, cumulative_low_medium, method=QUANTILE_METHOD)
    )
    return raw_high, raw_medium


def measure(
    artifacts: Mapping[str, str],
    input_digests: Mapping[str, str],
    repo_root: Path = REPO_ROOT,
) -> Derivation:
    """Derive BOTH pre-registered quantities from the END-STATE artifacts. READ-ONLY.

    Args:
        artifacts: ``{target -> deployed artifact id}``, as declared on the command line and
            cross-checked against ``artifacts/latest.json``.
        input_digests: The verified input digests, recorded into the emitted module.
        repo_root: The repository root.

    Returns:
        The full :class:`Derivation`.

    Raises:
        EmptyResidualPoolError: when a target has no strictly-prior season to estimate a bias
            from. NO bias is invented and there is no fallback to the target season's own data.
    """
    import json

    from backtest.diagnose import score_deployed_artifacts
    from backtest.engine import BacktestEngine

    manifest = json.loads((repo_root / "artifacts" / "latest.json").read_text())
    for target, declared in artifacts.items():
        deployed = str(manifest[target])
        if deployed != declared:
            msg = (
                f"the declared {target} artifact {declared!r} is not what "
                f"artifacts/latest.json deploys ({deployed!r}); the derivation refuses rather "
                "than scoring an artifact the manifest does not name."
            )
            raise DigestMismatchError(msg)

    with (repo_root / "config" / "phase33_gate_verdict.toml").open("rb") as handle:
        verdicts = tomllib.load(handle)
    residual_sources = {
        target: _residual_source(target, manifest, verdicts) for target in TARGETS
    }

    odds = BacktestEngine()._load_closing_odds()

    # (1) THE THRESHOLDS, over the D31-04 pinned 2021-2024 population.
    threshold_gold = {
        target: _load_gold(
            target, THRESHOLD_FIRST_SEASON, THRESHOLD_LAST_SEASON, repo_root
        )
        for target in TARGETS
    }
    threshold_scored = {
        target: score_deployed_artifacts(
            target,
            gold_df=threshold_gold[target],
            artifacts_dir=repo_root / "artifacts",
        )
        for target in TARGETS
    }
    edges = _edge_frame(threshold_scored, odds)

    wp_magnitudes = edges.loc[edges["wp_edge"].notna(), "wp_edge"].abs().to_numpy(float)
    if wp_magnitudes.size == 0:
        msg = "no WP row carries a computable edge, so there is no anchor to derive against."
        raise EmptyResidualPoolError(msg)

    # WP's OWN band shares under its unchanged 0.05 / 0.02 pair. These are the ANCHOR: ATS's
    # and O/U's thresholds are the values reproducing these three shares on their own
    # distributions.
    wp_shares, wp_counts = _shares_and_counts(
        wp_magnitudes, WP_HIGH_THRESHOLD, WP_MEDIUM_THRESHOLD
    )
    cumulative_low = float((wp_magnitudes <= WP_MEDIUM_THRESHOLD).mean())
    cumulative_low_medium = float((wp_magnitudes <= WP_HIGH_THRESHOLD).mean())

    thresholds: dict[str, tuple[float, float]] = {
        "wp": (WP_HIGH_THRESHOLD, WP_MEDIUM_THRESHOLD)
    }
    raw_thresholds: dict[str, tuple[float, float]] = {
        "wp": (WP_HIGH_THRESHOLD, WP_MEDIUM_THRESHOLD)
    }
    eligible_counts: dict[str, dict[str, int]] = {}
    shares_after: dict[str, dict[str, float]] = {}
    counts_after: dict[str, dict[str, int]] = {}
    shares_current: dict[str, dict[str, float]] = {}
    counts_current: dict[str, dict[str, int]] = {}
    changing: dict[str, int] = {}

    for target in TARGETS:
        column = f"{target}_edge"
        magnitudes = edges.loc[edges[column].notna(), column].abs().to_numpy(float)
        if target != "wp":
            raw_high, raw_medium = _anchored_pair(
                magnitudes, cumulative_low, cumulative_low_medium
            )
            raw_thresholds[target] = (raw_high, raw_medium)
            thresholds[target] = (round(raw_high, 4), round(raw_medium, 4))
        high, medium = thresholds[target]
        shares_after[target], counts_after[target] = _shares_and_counts(
            magnitudes, high, medium
        )
        shares_current[target], counts_current[target] = _shares_and_counts(
            magnitudes, WP_HIGH_THRESHOLD, WP_MEDIUM_THRESHOLD
        )
        changing[target] = int(
            (
                _band(magnitudes, WP_HIGH_THRESHOLD, WP_MEDIUM_THRESHOLD)
                != _band(magnitudes, high, medium)
            ).sum()
        )
        eligible_counts.setdefault(target, {})["threshold_rows"] = int(magnitudes.size)
        eligible_counts[target]["threshold_population_rows"] = len(edges)

    # (2) THE 2026 CHAIN-FIT BIAS, over strictly-prior seasons 2021-2025, through the EXISTING
    # estimator. No second estimator is written (D33-03 / D33-21).
    bias: dict[str, float] = {}
    bias_by_season: dict[str, dict[int, float]] = {}
    for target in TARGETS:
        gold = _load_gold(target, BIAS_FIRST_SEASON, BIAS_LAST_SEASON, repo_root)
        scored = score_deployed_artifacts(
            target, gold_df=gold, artifacts_dir=repo_root / "artifacts"
        )
        column = MODEL_COLUMN[target]
        residuals: dict[int, np.ndarray] = {}
        per_season: dict[int, float] = {}
        for season in sorted(int(value) for value in scored["season"].unique()):
            rows = scored[scored["season"] == season]
            residual = rows["actual"].to_numpy(float) - rows[column].to_numpy(float)
            residuals[season] = residual
            per_season[season] = float(residual.mean())
        try:
            bias[target] = float(
                estimate_prior_season_bias(residuals, BIAS_TARGET_SEASON)
            )
        except ValueError as error:  # pragma: no cover - guarded by the 2021-2025 slice
            msg = (
                f"the strictly-prior residual pool for {target} is EMPTY, so no 2026 bias "
                "can be estimated. No bias is invented and there is no fallback to the "
                f"target season's own data: {error}"
            )
            raise EmptyResidualPoolError(msg) from error
        bias_by_season[target] = per_season
        eligible_counts[target]["bias_rows"] = int(
            sum(len(v) for v in residuals.values())
        )
        eligible_counts[target]["bias_seasons"] = len(residuals)

    return Derivation(
        artifacts=dict(artifacts),
        residual_sources=residual_sources,
        thresholds=thresholds,
        raw_thresholds=raw_thresholds,
        wp_reference_shares=wp_shares,
        band_shares_after=shares_after,
        band_counts_after=counts_after,
        band_shares_under_current=shares_current,
        band_counts_under_current=counts_current,
        games_changing_band=changing,
        bias=bias,
        bias_by_season=bias_by_season,
        eligible_counts=eligible_counts,
        input_digests=dict(input_digests),
        formatter=formatter_version(),
    )


# ---------------------------------------------------------------------------
# The renderers -- pure functions of a Derivation, so determinism is checkable cheaply
# ---------------------------------------------------------------------------


def formatter_version() -> str:
    """The ruff version that will format the emitted module, e.g. ``ruff 0.15.6``."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def ruff_format(text: str, filename: str) -> str:
    """Return *text* as ``ruff format`` would write it for a file named *filename*.

    Run as the LAST step of emission so the committed bytes are what this program produces and
    the pre-commit hook has nothing left to rewrite. See the module docstring.
    """
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--stdin-filename", filename, "-"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _render_mapping(mapping: Mapping[str, Any], indent: str = "    ") -> str:
    """Render a str-keyed mapping in SORTED key order, one entry per line."""
    lines = []
    for key in sorted(mapping):
        lines.append(f"{indent}{key!r}: {mapping[key]!r},")
    return "\n".join(lines)


def _render_share_table(table: Mapping[str, Mapping[str, float]]) -> str:
    lines = []
    for target in sorted(table):
        row = table[target]
        rendered = ", ".join(
            f"{label!r}: {row[label]!r}" for label in ("low", "medium", "high")
        )
        lines.append(f"    {target!r}: {{{rendered}}},")
    return "\n".join(lines)


def _render_count_table(table: Mapping[str, Mapping[str, int]]) -> str:
    lines = []
    for target in sorted(table):
        row = table[target]
        rendered = ", ".join(
            f"{label!r}: {row[label]!r}" for label in ("low", "medium", "high")
        )
        lines.append(f"    {target!r}: {{{rendered}}},")
    return "\n".join(lines)


def render_module(derivation: Derivation) -> str:
    """Emit ``backtest/cold_start_constants.py``. Pure; deterministic; ASCII."""
    thresholds = derivation.thresholds
    threshold_lines = "\n".join(
        f"    {target!r}: ({thresholds[target][0]:.4f}, {thresholds[target][1]:.4f}),"
        for target in sorted(thresholds)
    )
    raw_lines = "\n".join(
        f"    {target!r}: ({derivation.raw_thresholds[target][0]!r}, "
        f"{derivation.raw_thresholds[target][1]!r}),"
        for target in sorted(derivation.raw_thresholds)
    )
    source_lines = "\n".join(
        f"    {target!r}: {{"
        + ", ".join(
            f"{key!r}: {derivation.residual_sources[target][key]!r}"
            for key in sorted(derivation.residual_sources[target])
        )
        + "},"
        for target in sorted(derivation.residual_sources)
    )
    bias_season_lines = "\n".join(
        f"    {target!r}: {{"
        + ", ".join(
            f"{season}: {derivation.bias_by_season[target][season]!r}"
            for season in sorted(derivation.bias_by_season[target])
        )
        + "},"
        for target in sorted(derivation.bias_by_season)
    )
    eligible_lines = "\n".join(
        f"    {target!r}: {{"
        + ", ".join(
            f"{key!r}: {derivation.eligible_counts[target][key]!r}"
            for key in sorted(derivation.eligible_counts[target])
        )
        + "},"
        for target in sorted(derivation.eligible_counts)
    )
    digest_lines = _render_mapping(derivation.input_digests)
    bias_lines = "\n".join(
        f"    {target!r}: {derivation.bias[target]!r},"
        for target in sorted(derivation.bias)
    )
    wp_shares = derivation.wp_reference_shares

    return f'''"""The FROZEN Phase-33 cold-start pre-registration (COLD-07, CLEAN-01, D33-19/D33-20).

GENERATOR OUTPUT. Emitted by ``python -m scripts.derive_cold_start_constants`` from the
END-STATE artifacts, in ONE operation (D24-07). Do NOT hand-edit any value below.

ONE-WAY BY CONSTRUCTION, in the shape of ``backtest/ev_chain_constants.py`` (D31-05). This
module IS the Phase-33 cold-start pre-registration, not a description of one. Its
LAST-MODIFYING COMMIT is the git-ancestry anchor: the ancestry guard requires that commit to be
a strict git ANCESTOR of the commit recording the Phase-33 readout.

Stated plainly because it is easy to forget two plans later: EDITING THIS FILE AFTER THE ANCHOR
COMMIT DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. There is no honest repair path. A value
that is wrong here is wrong for the remainder of the phase, and the only legitimate response is
a NEW, VISIBLY-LATER CORRECTIVE COMMIT that explicitly INVALIDATES this pre-registration by
naming its commit sha. Never an edit in place, and never a quiet supersession.

IT IS ONE PRE-REGISTRATION IN TWO FILES. ``COLD-START-PREREGISTRATION.md`` at the repo root is
the human-readable half the owner ratifies. Their COMBINED last-modifying commit is the anchor;
``PREREGISTRATION_PATHS`` below names both so every guard resolves them from one place.

NEITHER FILE RECORDS ITS OWN CONTENT HASH. A document that must CONTAIN and exactly REPRODUCE
its own whole-file hash is self-referential: writing the hash changes the bytes the hash was
computed over, so no fixed point exists without a canonical exclusion rule nobody has defined.
The witness lives OUTSIDE the witnessed files: ``tests/phase33_state.py`` records
``PRE_REGISTRATION_COMMIT``, ``PRE_REGISTRATION_FILE_SHA256`` and
``PRE_REGISTRATION_AUTHOR_DATE`` in a LATER commit under its APPEND PROTOCOL, and
``tests/unit/test_phase33_preregistration_ancestry.py`` recomputes and compares.

THE ANCHORS, AND WHAT EACH ONE CAN ACTUALLY CARRY
--------------------------------------------------
(i) GIT ANCESTRY against the committed Phase-33 readout commit. This proves ordering INSIDE the
repository and nothing about wall-clock time.

(ii) AN EXTERNAL TIME ANCHOR -- a signed tag pushed to the remote, a remote push receipt, or a
CI attestation. Only something produced by a system OTHER than this working tree can establish
that the rule existed before kickoff, which is the property a pre-registration exists to have.
Which kind was obtained, or that none was, is recorded in
``tests.phase33_state.PRE_REGISTRATION_EXTERNAL_ANCHOR_KIND``.

(iii) THE COMMIT'S AUTHOR DATE, against 2026-09-17T20:15:00-04:00. This is CORROBORATION ONLY.
A git author date is LOCALLY SETTABLE -- ``GIT_AUTHOR_DATE`` and ``git commit --date`` both set
it -- so it cannot prove pre-kickoff existence on its own. It is asserted because a recorded
date that disagrees with the record is still a finding, not because it carries the claim.

THE TWO QUANTITIES, AND THE ONE PROPERTY THEY SHARE
-----------------------------------------------------
They live in ONE module because they share the single property that defines them: FROZEN BEFORE
WEEK 2, NEVER RECOMPUTED IN-SEASON. Splitting them would give the phase two anchors to keep
consistent instead of one.

THE DERIVATION IS MILDLY CIRCULAR, AND IT IS STATED RATHER THAN DISCOVERED (D33-20)
------------------------------------------------------------------------------------
The thresholds are derived on the pinned {THRESHOLD_FIRST_SEASON}-{THRESHOLD_LAST_SEASON}
backtest population, and that is ALSO the population the label movement is measured against.
Measuring the movement on a population the thresholds were not derived from would trade a
stated caveat for an unstated mismatch, so the circularity is kept and named here.

THE BIAS POOL IS PARTLY IN-SAMPLE, AND THAT IS ALSO STATED
------------------------------------------------------------
The deployed artifacts were fitted through the final-fit entry point over every completed
season 2002-2025, which INCLUDES the {BIAS_FIRST_SEASON}-{BIAS_LAST_SEASON} residual pool below.
So these residuals are in-sample and the bias they produce is ATTENUATED -- the real
out-of-sample bias is likely larger in magnitude. That direction is the conservative one to
know about and it is recorded rather than left to be inferred.

THE EMPTY-POOL REFUSAL (R10 edge)
-----------------------------------
An EMPTY strictly-prior residual pool REFUSES BY NAME. No bias is invented and there is NO
fallback to the target season's own data. The refusal is
``scripts.derive_cold_start_constants.EmptyResidualPoolError``, wrapping the existing
``backtest.ou_ev_chain.estimate_prior_season_bias`` refusal.

JSON ROUND-TRIPS SEASON KEYS AS STRINGS WHILE THE LOOKUP IS BY INT
-------------------------------------------------------------------
Every season-keyed mapping here uses INT keys in Python. A JSON-facing representation of the
same mapping carries STRING keys, because JSON has no integer keys at all. A consumer that
serializes and reloads one of these mappings must therefore look up ``"2026"`` and not ``2026``
-- stated here so the mismatch is a known conversion rather than a silent miss.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

# THE TWO FILES THAT ARE THE PRE-REGISTRATION. Named here so every guard resolves them from ONE
# place: the ancestry test's `git log -1 --format=%H -- <paths>`, the content-hash comparison
# against tests/phase33_state.PRE_REGISTRATION_FILE_SHA256, and the Plan 33-18 assertion that
# this commit strictly precedes the readout commit. Repo-root-relative POSIX paths.
PREREGISTRATION_PATHS: tuple[str, ...] = (
    "{DOCUMENT_PATH}",
    "{MODULE_PATH}",
)

# THE SIX EDGE THRESHOLDS, ONE PAIR PER TARGET, as (HIGH, MEDIUM) and on EACH TARGET'S OWN
# UNIT. Stated to FOUR DECIMAL PLACES in the committed text, which is the pre-registered
# precision and not merely an upper bound on it.
#
#   ats  POINTS          -- signed model-minus-market home margin (R13 / D33-05, Plan 33-10)
#   ou   RATIO           -- (model total - market total) / max(market total, 30)
#   wp   PROBABILITY     -- model probability minus the devigged fair closing probability
#
# WP's pair is UNCHANGED at 0.05 / 0.02 by design (D33-20): WP labels do not move at all, so
# the 23-point `_EDGE_TIER_SNAPSHOT` recorded before the Phase-31 collapse survives unchanged
# and keeps its evidentiary value instead of being rewritten with a new expectation.
EDGE_TIER_THRESHOLDS_BY_TARGET: dict[str, tuple[float, float]] = {{
{threshold_lines}
}}

# The UNROUNDED quantiles the four-decimal values above were rounded from. Recorded so the
# rounding is visible rather than implied, and so a reviewer re-running the derivation can
# compare against the full-precision figure rather than against an already-rounded one.
EDGE_TIER_THRESHOLDS_UNROUNDED: dict[str, tuple[float, float]] = {{
{raw_lines}
}}

# The unit each pair belongs to. A threshold without a stated unit is a number nobody can
# check, and applying ONE pair to three incompatible units is the defect (DEF-31-17) this
# pre-registration exists to retire.
EDGE_TIER_THRESHOLD_UNITS: dict[str, str] = {{
    'ats': 'points (signed model-minus-market home margin)',
    'ou': 'ratio of the market total, floored at 30',
    'wp': 'probability (model minus devigged fair closing probability)',
}}

# The EXACT quantile convention, NAMED rather than left to a library default that could change
# on an upgrade. ATS's and O/U's thresholds are the values reproducing WP's OWN band shares on
# their own |edge| distributions: the medium threshold is the quantile at WP's cumulative "low"
# share and the high threshold is the quantile at WP's cumulative "low + medium" share, under
# `utils.edge_tier`'s STRICT `>` comparisons.
THRESHOLD_QUANTILE_CONVENTION: str = (
    'numpy.quantile(magnitudes, q, method="linear"); q taken from WP band shares under its '
    'unchanged 0.05 / 0.02 pair; bands assigned with STRICT > so a value exactly at a '
    'threshold falls in the LOWER band'
)

# WP's measured band shares under its unchanged pair -- the ANCHOR the other two reproduce.
# RE-DERIVED on the end-state artifacts rather than copied. The pre-rebuild reference recorded
# in tests.phase33_state.ATS_BAND_SHARES_BEFORE was wp low 0.2677 / medium 0.2999 / high
# 0.4324; the divergence is REAL and is the re-fit's doing, not a measurement error.
WP_ANCHOR_BAND_SHARES: dict[str, float] = {{'low': {wp_shares["low"]!r}, 'medium': {wp_shares["medium"]!r}, 'high': {wp_shares["high"]!r}}}

# The population the thresholds were derived on -- ALSO the population the label movement is
# measured against (the circularity named in the docstring).
THRESHOLD_DERIVATION_POPULATION: str = (
    'gold seasons {THRESHOLD_FIRST_SEASON}-{THRESHOLD_LAST_SEASON} inclusive, scored with '
    'the END-STATE deployed artifacts and joined to data/silver/odds_snapshot.parquet '
    'closing lines; shares taken over the rows carrying a COMPUTABLE edge, which is also '
    'the D31-04 pinned population the label movement is measured against'
)

# The 2026 CHAIN-FIT BIAS, per target, pooled mean residual (actual - predicted) over the
# STRICTLY-PRIOR seasons below. Computed by the EXISTING
# backtest.ou_ev_chain.estimate_prior_season_bias; no second estimator was written.
CHAIN_FIT_BIAS_2026: dict[str, float] = {{
{bias_lines}
}}

CHAIN_FIT_BIAS_SEASONS: tuple[int, ...] = ({BIAS_FIRST_SEASON}, {BIAS_FIRST_SEASON + 1}, {BIAS_FIRST_SEASON + 2}, {BIAS_FIRST_SEASON + 3}, {BIAS_LAST_SEASON})

# The per-season mean residual each pooled bias was estimated from. INT season keys; see the
# docstring's JSON note.
CHAIN_FIT_BIAS_BY_SEASON: dict[str, dict[int, float]] = {{
{bias_season_lines}
}}

# WHERE EACH TARGET'S RESIDUALS CAME FROM, read from config/phase33_gate_verdict.toml and
# artifacts/latest.json rather than assumed. `kind` is the gate disposition; `verdict` is the
# verdict as MEASURED and left standing. A `promoted_refit` carrying a non-PASS `verdict` was
# shipped under a recorded owner override -- the verdict was never softened to match the
# ruling, and this field is where that is visible.
CHAIN_FIT_BIAS_SOURCE_BY_TARGET: dict[str, dict[str, object]] = {{
{source_lines}
}}

# The END-STATE artifact ids every number above was derived from.
DERIVATION_ARTIFACTS: dict[str, str] = {{
{_render_mapping(derivation.artifacts)}
}}

# EVERY input this derivation read, with its digest. GIT-TRACKED TEXT inputs -- here only
# config/phase33_gate_verdict.toml -- are digested over NEWLINE-NORMALIZED bytes, because this
# repository has core.autocrlf=true and no .gitattributes. Everything under artifacts/ and
# data/ is a GITIGNORED PRODUCTION STORE and is digested over RAW bytes, which is the same
# instrument tests.data_boundary.digest_file uses: the artifacts/latest.json digest below is
# therefore literally equal to tests.phase33_state.POST_GATE_MANIFEST_DIGEST. An `artifact:<id>`
# key is the sha256 over a sorted `<relpath>\\0<file sha256>` manifest of the whole directory.
DERIVATION_INPUT_DIGESTS: dict[str, str] = {{
{digest_lines}
}}

# The FORMATTER this module's bytes were produced under. Recorded because this file is
# ruff-formatted as the last step of emission, so that the pre-commit hook has nothing left to
# rewrite in a FROZEN rule. A ruff upgrade that changed the formatting would fail the
# reproduce-check in tests/unit/test_cold_start_derivation_cli.py loudly, and this constant is
# what tells the reader why.
DERIVATION_FORMATTER: str = {derivation.formatter!r}

# The row count behind each derived number, per target. `threshold_rows` is the rows with a
# COMPUTABLE edge; `threshold_population_rows` is the whole population including rows with no
# stored market line; `bias_rows` is the residual pool.
DERIVATION_ELIGIBLE_COUNTS: dict[str, dict[str, int]] = {{
{eligible_lines}
}}

# THE LABEL MOVEMENT, held at ONE model and ONE edge definition so the thresholds are the only
# thing that varies. `_UNDER_CURRENT` bands the SAME end-state edges with the 0.05 / 0.02 pair
# in force today; `_AFTER` bands them with the frozen pairs above. WP's two rows are identical
# by construction and zero WP games change band.
BAND_SHARES_UNDER_CURRENT_THRESHOLDS: dict[str, dict[str, float]] = {{
{_render_share_table(derivation.band_shares_under_current)}
}}

BAND_COUNTS_UNDER_CURRENT_THRESHOLDS: dict[str, dict[str, int]] = {{
{_render_count_table(derivation.band_counts_under_current)}
}}

ATS_BAND_SHARES_AFTER: dict[str, dict[str, float]] = {{
{_render_share_table(derivation.band_shares_after)}
}}

ATS_BAND_COUNTS_AFTER: dict[str, dict[str, int]] = {{
{_render_count_table(derivation.band_counts_after)}
}}

GAMES_CHANGING_BAND: dict[str, int] = {{
{_render_mapping(derivation.games_changing_band)}
}}

# Plan 33-08's declared allowance, RESTATED here as part of the pre-registration record. It was
# declared in scripts/run_phase33_gate.py BEFORE any verdict existed; this is not a second
# declaration but the same one, carried into the frozen rule so a reader of the rule meets it.
FIX_CYCLE_ALLOWANCE: int = {FIX_CYCLE_ALLOWANCE}
'''


def render_document(derivation: Derivation) -> str:
    """Emit ``COLD-START-PREREGISTRATION.md``. Pure; deterministic; ASCII."""
    thresholds = derivation.thresholds
    rows = []
    units = {
        "wp": "probability (model minus devigged fair closing probability)",
        "ats": "points (signed model-minus-market home margin)",
        "ou": "ratio of the market total, floored at 30",
    }
    for target in ("wp", "ats", "ou"):
        high, medium = thresholds[target]
        rows.append(f"| {target} | `{medium:.4f}` | `{high:.4f}` | {units[target]} |")
    threshold_table = "\n".join(rows)

    movement = []
    for target in ("wp", "ats", "ou"):
        before = derivation.band_counts_under_current[target]
        after = derivation.band_counts_after[target]
        movement.append(
            f"| {target} | {before['low']} / {before['medium']} / {before['high']} "
            f"| {after['low']} / {after['medium']} / {after['high']} "
            f"| {derivation.games_changing_band[target]} |"
        )
    movement_table = "\n".join(movement)

    bias_rows = []
    for target in ("wp", "ats", "ou"):
        source = derivation.residual_sources[target]
        bias_rows.append(
            f"| {target} | `{derivation.bias[target]!r}` | `{source['artifact']}` "
            f"| {source['kind']} | {source['verdict']} |"
        )
    bias_table = "\n".join(bias_rows)

    return f"""# COLD-START PRE-REGISTRATION -- the 2026 chain-fit bias and the six edge thresholds

**Status:** FROZEN. This document and `{MODULE_PATH}` are ONE pre-registration in two files,
landed in ONE commit containing nothing else. This is the human-readable half; the constants
module is the half the code reads.

**What this document is for.** Phase 33 ships a live cold start. Two quantities have to exist
BEFORE the numbers they govern arrive, or they are not rules at all -- they are descriptions
written after the fact. The first is the 2026 chain-fit bias: without a frozen value, the
cold-start refusal prescribes re-running a spent, unrepeatable one-shot measurement. The second
is the six per-target edge thresholds: one threshold pair is currently applied to three
incompatible units, which put ATS in the "high" band on the overwhelming majority of games.
Both are frozen once here and NEVER recomputed in-season.

**Editing after the anchor commit does not fix a bug -- it destroys the evidence.** A value
that is wrong here is wrong for the remainder of the phase. The only legitimate response is a
NEW, VISIBLY-LATER CORRECTIVE COMMIT that explicitly INVALIDATES this pre-registration by
naming its commit sha. Never an edit in place, and never a quiet supersession.

**The anchor is recorded outside the files it witnesses.** Neither this document nor
`{MODULE_PATH}` records its own content hash. A file that must CONTAIN and exactly REPRODUCE
its own whole-file hash is self-referential: writing the hash changes the bytes the hash was
computed over, so no fixed point exists without a canonical exclusion rule nobody has defined.

```
preregistration_anchor_recorded_in: tests/phase33_state.py
preregistration_anchor_slots: PRE_REGISTRATION_COMMIT, PRE_REGISTRATION_FILE_SHA256, PRE_REGISTRATION_AUTHOR_DATE, PRE_REGISTRATION_EXTERNAL_ANCHOR, PRE_REGISTRATION_EXTERNAL_ANCHOR_KIND
preregistration_paths: {DOCUMENT_PATH}, {MODULE_PATH}
preregistration_deadline: 2026-09-17T20:15:00-04:00
```

`tests/unit/test_phase33_preregistration_ancestry.py` resolves the commit from git against
those two paths, asserts it equals the recorded slot, recomputes each file's sha256 and
compares, and asserts the commit's AUTHOR date precedes the deadline above.

---

## 1. What anchors this, and what each anchor can actually carry

Three anchors, listed in descending order of what they prove.

1. **Git ancestry** against the committed Phase-33 readout commit. This proves ordering INSIDE
   the repository. It cannot reach wall-clock time, because `data/`, `outputs/` and
   `artifacts/` are ALL gitignored -- so ancestry alone anchors the rule to a readout we write
   ourselves.
2. **An external time anchor** -- a signed tag pushed to the remote, a remote push receipt, or
   a CI attestation. Only something produced by a system OTHER than this working tree can
   establish that the rule existed before kickoff, which is the property a pre-registration
   exists to have. Which kind was obtained is recorded in
   `tests.phase33_state.PRE_REGISTRATION_EXTERNAL_ANCHOR_KIND`.
3. **The commit's author date**, against `2026-09-17T20:15:00-04:00`. This is **CORROBORATION
   ONLY**. A git author date is LOCALLY SETTABLE -- `GIT_AUTHOR_DATE` and `git commit --date`
   both set it -- so it CANNOT prove pre-kickoff existence on its own.

**If no external anchor was obtainable in this environment, this pre-registration is anchored
by git ancestry and a corroborating author date ONLY, and this document says so rather than
implying a strength the evidence does not have.** The authoritative record of which case holds
is `PRE_REGISTRATION_EXTERNAL_ANCHOR_KIND`: the literal `NONE_AVAILABLE` means no external
anchor could be produced, and the reason is stated in the plan's summary. That slot is written
in a LATER commit than this one, because a file cannot record a fact about the commit it is
part of.

**What no anchor can prove.** None of the three can prove that nobody tuned a threshold after
seeing the numbers. No test can prove intent. That clause is judgment-tier and is discharged by
the owner's ratification, recorded with its date.

---

## 2. The six edge thresholds

| target | medium | high | unit |
|---|---|---|---|
{threshold_table}

**WP's pair does not move.** It stays exactly `0.0500` / `0.0200`, so WP labels do not change
at all. That is deliberate: the 23-point `_EDGE_TIER_SNAPSHOT` recorded BEFORE the Phase-31
helper collapse survives unchanged and keeps its evidentiary value, instead of being rewritten
with a new expectation.

**ATS's and O/U's pairs are WP-ANCHORED QUANTILES.** Each is set to the value that reproduces
WP's own band shares on that target's own `|edge|` distribution. WP's measured shares on the
end-state artifacts are low `{derivation.wp_reference_shares["low"]:.4f}` / medium
`{derivation.wp_reference_shares["medium"]:.4f}` / high
`{derivation.wp_reference_shares["high"]:.4f}`. The convention is
`numpy.quantile(..., method="linear")`, named explicitly so a future library default cannot
silently move a frozen threshold, and the bands are assigned with STRICT `>` so a value exactly
at a threshold falls in the LOWER band -- the behaviour `utils.edge_tier` already has.

**The ATS population includes REPAIRED PICK-EM ROWS.** Under the D33-31 owner ruling a pick-em
is a REAL line, so a game with `market_spread == 0` now carries its genuine nonzero
disagreement instead of a forced `0.0`. Those rows shift the ATS distribution's mass, which is
why this derivation had to run AFTER Plan 33-10's repair. A threshold derived from the
pre-repair distribution would have been frozen against a defect.

**The derivation is mildly circular, and it is stated here rather than discovered in review.**
The thresholds are derived on gold seasons {THRESHOLD_FIRST_SEASON}-{THRESHOLD_LAST_SEASON},
and that is ALSO the D31-04 pinned population the label movement below is measured against.
Removing the circularity would mean deriving on a population the movement is not measured
against, which trades a stated caveat for an unstated mismatch.

---

## 3. The label movement, in games

Held at ONE model and ONE edge definition, so the thresholds are the only thing that varies.
Counts are `low / medium / high` over the rows with a computable edge.

| target | under the CURRENT 0.05 / 0.02 pair | under the FROZEN pairs | games changing band |
|---|---|---|---|
{movement_table}

This is a **PUBLISHED-LABEL CHANGE**: the band is rendered and sorted on by `/` and `/betting`.
It is disclosed rather than slipped in, and the owner's acceptance of it is recorded with its
date.

Two "before" figures exist and they answer different questions. The one in the table is the
end-state model's edges under today's thresholds -- it isolates the THRESHOLD change. The
figure recorded in `tests.phase33_state.ATS_BAND_SHARES_BEFORE` (ATS high at 92.64%) was
measured on the PRE-rebuild artifacts under the PRE-repair ratio-scale ATS edge, so the
difference between it and anything here mixes three changes: the unit repair, the re-fit and
the thresholds. Both are recorded; neither overwrites the other.

---

## 4. The 2026 chain-fit bias

The pooled mean residual (`actual - predicted`) over the STRICTLY-PRIOR seasons
{BIAS_FIRST_SEASON}-{BIAS_LAST_SEASON}, on each target's own scale, computed by the EXISTING
`backtest.ou_ev_chain.estimate_prior_season_bias`. **No second estimator was written.**

| target | 2026 bias | residual source artifact | disposition | gate verdict |
|---|---|---|---|---|
{bias_table}

**A `promoted_refit` carrying a non-PASS verdict was shipped under a recorded owner override.**
The verdict was never softened to match the ruling; it stands as measured in
`config/phase33_gate_verdict.toml`, and the override sits BESIDE it. This table is where that
is visible rather than smoothed away.

**The pool is partly in-sample, and the direction of that bias is stated.** The deployed
artifacts were fitted through the final-fit entry point over every completed season 2002-2025,
which INCLUDES this residual pool. So these residuals are in-sample and the bias they produce
is ATTENUATED -- the true out-of-sample bias is likely LARGER in magnitude. That is the
conservative direction to know about.

**An EMPTY strictly-prior residual pool REFUSES BY NAME.** No bias is invented and there is NO
fallback to the target season's own data. The refusal is
`scripts.derive_cold_start_constants.EmptyResidualPoolError`.

**JSON round-trips season keys as strings while the lookup is by int.** Every season-keyed
mapping uses INT keys in Python; a JSON-facing representation of the same mapping carries
STRING keys, because JSON has no integer keys at all. A consumer that serializes and reloads
one must look up `"2026"` and not `2026`.

---

## 5. The fix-cycle allowance is ZERO, and it was declared before any verdict

`FIX_CYCLE_ALLOWANCE = {FIX_CYCLE_ALLOWANCE}`, declared in `scripts/run_phase33_gate.py` BEFORE
any Phase-33 verdict existed and restated in `{MODULE_PATH}` as part of this record. It is not
a second declaration; it is the same one, carried into the frozen rule so a reader of the rule
meets it.

---

## 6. How to reproduce every number above

```
uv run python -m scripts.derive_cold_start_constants --check
```

The derivation is a COMMITTED, DETERMINISTIC program
(`scripts/derive_cold_start_constants.py`). It takes the end-state artifact ids and their
content digests as explicit arguments, REFUSES on any digest mismatch -- naming the input and
both digests -- and emits these two files. Running it twice against the same inputs produces
byte-identical output. Every input it read, with its digest, is recorded in
`DERIVATION_INPUT_DIGESTS`; the row count behind every derived number is in
`DERIVATION_ELIGIBLE_COUNTS`; the quantile convention is in
`THRESHOLD_QUANTILE_CONVENTION`.

---

## 7. What this pre-registration does NOT claim

It does not claim the thresholds are BETTER. No measurement here shows that a game in the new
"high" band is a better bet than a game in the old one. What it claims is narrower and
checkable: that the bands are now defined per target on that target's own unit, that the
values were fixed before the season they govern, and that the movement they cause was measured
and disclosed before anyone saw a 2026 result.

It does not claim the 2026 bias is correct. It claims the bias is the walk-forward estimate the
existing estimator produces from strictly-prior completed seasons, that its pool is partly
in-sample and therefore attenuated, and that it was frozen rather than chosen later.

---

*Phase: 33-live-cold-start-forward-temporal-integrity*
*Frozen by Plan 33-16, before 2026-09-17T20:15:00-04:00*
"""


# ---------------------------------------------------------------------------
# THE SUPERSEDING CORRECTION (Plan 33.2-26, SPEC R14, D33.2-01/09/25)
#
# Commit 11761c7 froze the 2026 edge thresholds and the 2026 chain-fit bias from models and
# closing lines this phase replaced. The frozen pair of files is NEVER edited; this section
# derives the corrected values into a NEW pair (``backtest/corrected_cold_start_constants.py`` +
# ``COLD-START-CORRECTION.md``) that names 11761c7 by sha.
#
# WHAT IS SWAPPED: the models (the three corrected artifacts production serves, through their
# own recorded recipes' walk-forward predictions -- the same predictions Plan 33.2-29's corrected
# chain fit was swept on) and the market (the owned ``odds_timeline`` line at or before each
# game's lock, never a closing line; WP's market side converted OUT OF FOLD). WHAT IS HELD: WP's
# 0.05 / 0.02 anchor pair, the WP-anchored band-share quantiles, ``QUANTILE_METHOD``, the STRICT
# ``>`` bands, the digest refusals and the walk-forward bias estimator.
# ---------------------------------------------------------------------------

CORRECTED_MODULE_PATH = "backtest/corrected_cold_start_constants.py"
CORRECTION_DOCUMENT_PATH = "COLD-START-CORRECTION.md"

#: The record the correction supersedes, by sha. Resolved to the full sha at render time.
SUPERSEDED_PREREGISTRATION_SHORT = "11761c7"

#: The corrected chain fit (Plan 33.2-29), read by EXPLICIT path and never through
#: ``backtest.weekly_bet_list.DEFAULT_CHAIN_FIT_PATH``: this derivation must read the corrected
#: record whatever the live default names, and until the repoint that default is the Phase-31
#: record.
CHAIN_FIT_SOURCE: Path = Path("outputs") / "p332" / "corrected_chain_fit.json"

#: The threshold window: every season with honest pre-lock prices.
CORRECTED_THRESHOLD_SEASONS: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)
CORRECTED_WINDOW_REASON = (
    "2020-2024 is the whole honest corpus, not a preference: 2018-2019 are unbuyable at any "
    "price and 2025 has no free pre-lock source. The single-use 2025 hold is spent and no row "
    "of it enters this derivation."
)

#: The season the corrected chain-fit bias is FOR. Named rather than derived as "the pool's last
#: season plus one": the corrected pool ends at 2024, because no 2025 row is read.
CORRECTED_BIAS_TARGET_SEASON = 2026

#: Below this many rows with a computable edge a target gets NO threshold. The calibration gate's
#: n >= 100 is the nearest existing floor (33.2-CONTEXT.md, Claude's Discretion); a threshold is a
#: quantile, and a quantile over fewer rows than that describes the sample, not the target.
MIN_HONEST_THRESHOLD_ROWS = 100

EDGE_UNITS_CORRECTED: Mapping[str, str] = {
    "ats": "points (signed model-minus-market home margin)",
    "ou": "ratio of the market total, floored at 30",
    "wp": "probability (model minus the spread-derived out-of-fold market probability)",
}


class InsufficientHonestDataForThresholdError(Exception):
    """A target's honest pool is too small to derive a threshold, so it gets NONE (R14).

    Its threshold is recorded as ``None`` and the selection path places no bets for it -- never a
    default, a zero or the superseded 11761c7 value, which would look like a threshold and be a
    relic. Like ``data.sealed_probe_log.SealedProbeLogCorrupt`` it inherits ``Exception``
    directly, NOT ``ValueError`` / ``KeyError`` / ``RuntimeError`` / ``LookupError``: the
    ``except ValueError`` around the bias estimator in :func:`measure` and the selection path's
    absent-input catch tuples would otherwise turn "no threshold" back into "carry on".
    """


class ChainFitReproductionError(Exception):
    """The derivation's own residuals do not reproduce the corrected chain fit's season biases.

    The 2026 bias continues that record's walk-forward series, so it is only the same series if
    the earlier seasons come out identical. A mismatch means the two were computed on different
    predictions, and extending one with the other would be a rule nobody chose.
    """


@dataclass(frozen=True)
class PrelockEdges:
    """The per-game edges over the owned pre-lock corpus, and the WP rows with no converter.

    Attributes:
        frame: One row per corpus game: ``game_id``, ``season``, the market line, the three
            model outputs, ``wp_market_prob`` (out of fold, NaN where no prior fold exists) and
            the three edges.
        wp_excluded: The ``no_prior_fold_converter`` rows (``game_id``, ``season``, ``reason``).
    """

    frame: pd.DataFrame
    wp_excluded: pd.DataFrame


def build_prelock_edge_frame(
    corpus_frame: pd.DataFrame,
    predictions: Mapping[str, pd.DataFrame],
    walk_forward_slopes: Mapping[int | str, float],
) -> PrelockEdges:
    """The three edges of every owned pre-lock game, on each target's own unit. Pure.

    ATS and O/U read the owned pre-lock line directly. WP's market side is converted OUT OF
    FOLD through ``oof_market_probability`` with the bound converter's prior-only slopes: a
    historical row converted with the serving slope would be priced by a slope fitted partly on
    its own outcome. A season with no prior fold (2020 on today's corpus) is recorded as
    ``no_prior_fold_converter`` and never filled; it stays in the ATS and O/U edges.

    Args:
        corpus_frame: ``models.blending_data.PrelockTuningCorpus.frame`` (``market_spread`` on
            the home-margin scale).
        predictions: ``{wp, ats, ou}`` -> ``game_id`` and ``prediction`` per game.
        walk_forward_slopes: The bound converter's ``walk_forward_slopes``.

    Raises:
        SpentHoldSeasonError / DerivationWindowError: a corpus row outside 2020-2024 (a 2025 row
            is named as the spent hold).
        ValueError: a corpus game with no prediction for some target.
    """
    from models.blending_data import EXCLUSION_NO_PRIOR_FOLD_CONVERTER
    from scripts.derive_corrected_ev_chain import refuse_seasons_outside

    refuse_seasons_outside(
        corpus_frame, CORRECTED_THRESHOLD_SEASONS, "the owned pre-lock corpus"
    )
    frame = corpus_frame.loc[
        :, ["game_id", "season", "market_spread", "market_total"]
    ].copy()
    frame["game_id"] = frame["game_id"].astype(str)
    frame["season"] = frame["season"].astype(int)

    for target in TARGETS:
        preds = predictions[target].loc[:, ["game_id", "prediction"]].copy()
        preds["game_id"] = preds["game_id"].astype(str)
        missing = sorted(set(frame["game_id"]) - set(preds["game_id"]))
        if missing:
            msg = (
                f"{len(missing)} owned pre-lock game(s) have no {target} prediction, e.g. "
                f"{missing[:10]}; refusing rather than dropping them uncounted."
            )
            raise ValueError(msg)
        frame = frame.merge(
            preds.rename(columns={"prediction": MODEL_COLUMN[target]}),
            on="game_id",
            how="left",
            validate="one_to_one",
        )

    frame["ats_edge"] = frame["model_spread"] - frame["market_spread"]
    frame["ou_edge"] = (frame["model_total"] - frame["market_total"]) / frame[
        "market_total"
    ].clip(lower=OU_TOTAL_FLOOR)

    covered = frame["season"].isin({int(season) for season in walk_forward_slopes})
    frame["wp_market_prob"] = np.nan
    if bool(covered.any()):
        converted = frame.loc[covered]
        frame.loc[covered, "wp_market_prob"] = oof_market_probability(
            converted.assign(home_fav_margin=converted["market_spread"]),
            walk_forward_slopes,
        )
    frame["wp_edge"] = frame["model_prob"] - frame["wp_market_prob"]

    excluded = frame.loc[~covered, ["game_id", "season"]].assign(
        reason=EXCLUSION_NO_PRIOR_FOLD_CONVERTER
    )
    return PrelockEdges(
        frame=frame.sort_values("game_id", ignore_index=True),
        wp_excluded=excluded.sort_values("game_id", ignore_index=True),
    )


def require_honest_pool(target: str, magnitudes: np.ndarray) -> None:
    """Refuse a threshold over fewer than :data:`MIN_HONEST_THRESHOLD_ROWS` honest rows.

    Raises:
        InsufficientHonestDataForThresholdError: naming the target and both counts.
    """
    if magnitudes.size < MIN_HONEST_THRESHOLD_ROWS:
        msg = (
            f"[{target!r}] only {magnitudes.size} honest pre-lock row(s) carry a computable "
            f"edge, below MIN_HONEST_THRESHOLD_ROWS = {MIN_HONEST_THRESHOLD_ROWS}. This target "
            "gets NO threshold and therefore places no bets; the superseded 11761c7 value is "
            "never borrowed in its place."
        )
        raise InsufficientHonestDataForThresholdError(msg)


@dataclass(frozen=True)
class ThresholdDerivation:
    """The corrected pairs, their refusals, and the label movement they cause."""

    thresholds: dict[str, tuple[float, float] | None]
    raw_thresholds: dict[str, tuple[float, float] | None]
    refusals: dict[str, str]
    wp_reference_shares: dict[str, float] | None
    band_shares_after: dict[str, dict[str, float]]
    band_counts_after: dict[str, dict[str, int]]
    band_shares_under_current: dict[str, dict[str, float]]
    band_counts_under_current: dict[str, dict[str, int]]
    games_changing_band: dict[str, int]
    eligible_rows: dict[str, int]
    population_rows: int


def derive_threshold_pairs(
    edges: pd.DataFrame,
    current_pairs: Mapping[str, tuple[float, float]] | None = None,
) -> ThresholdDerivation:
    """The WP-anchored per-target pairs over *edges*, or a named refusal per target. Pure.

    WP's pair stays 0.05 / 0.02 and its measured band shares are the anchor; ATS's and O/U's
    pairs are the quantiles reproducing those shares on their own ``|edge|`` distributions -- the
    11761c7 recipe, unchanged. A target below :data:`MIN_HONEST_THRESHOLD_ROWS` is refused (its
    pair is ``None``), and a refused WP anchor leaves ATS and O/U nothing to reproduce.

    Args:
        edges: :attr:`PrelockEdges.frame` (``season`` and the three ``*_edge`` columns).
        current_pairs: The pairs in force before this correction, for the label movement.
            Defaults to the superseded 11761c7 pairs, READ from the frozen module.
    """
    if current_pairs is None:
        from backtest.cold_start_constants import EDGE_TIER_THRESHOLDS_BY_TARGET

        current_pairs = EDGE_TIER_THRESHOLDS_BY_TARGET

    magnitudes = {
        target: edges[f"{target}_edge"].dropna().abs().to_numpy(float)
        for target in TARGETS
    }
    thresholds: dict[str, tuple[float, float] | None] = dict.fromkeys(TARGETS)
    raw: dict[str, tuple[float, float] | None] = dict.fromkeys(TARGETS)
    refusals: dict[str, str] = {}
    wp_shares: dict[str, float] | None = None

    try:
        require_honest_pool("wp", magnitudes["wp"])
    except InsufficientHonestDataForThresholdError as refusal:
        refusals["wp"] = str(refusal)
        for target in ("ats", "ou"):
            refusals[target] = (
                f"[{target!r}] the WP anchor is refused, so there are no band shares to "
                "reproduce and this target gets NO threshold. " + str(refusal)
            )
    else:
        wp_shares, _ = _shares_and_counts(
            magnitudes["wp"], WP_HIGH_THRESHOLD, WP_MEDIUM_THRESHOLD
        )
        cumulative_low = float((magnitudes["wp"] <= WP_MEDIUM_THRESHOLD).mean())
        cumulative_low_medium = float((magnitudes["wp"] <= WP_HIGH_THRESHOLD).mean())
        thresholds["wp"] = raw["wp"] = (WP_HIGH_THRESHOLD, WP_MEDIUM_THRESHOLD)
        for target in ("ats", "ou"):
            try:
                require_honest_pool(target, magnitudes[target])
            except InsufficientHonestDataForThresholdError as refusal:
                refusals[target] = str(refusal)
                continue
            raw_high, raw_medium = _anchored_pair(
                magnitudes[target], cumulative_low, cumulative_low_medium
            )
            raw[target] = (raw_high, raw_medium)
            thresholds[target] = (round(raw_high, 4), round(raw_medium, 4))

    shares_after: dict[str, dict[str, float]] = {}
    counts_after: dict[str, dict[str, int]] = {}
    shares_current: dict[str, dict[str, float]] = {}
    counts_current: dict[str, dict[str, int]] = {}
    changing: dict[str, int] = {}
    for target in TARGETS:
        pair = thresholds[target]
        if pair is None:
            continue
        values = magnitudes[target]
        old_high, old_medium = current_pairs[target]
        shares_after[target], counts_after[target] = _shares_and_counts(values, *pair)
        shares_current[target], counts_current[target] = _shares_and_counts(
            values, old_high, old_medium
        )
        changing[target] = int(
            (_band(values, old_high, old_medium) != _band(values, *pair)).sum()
        )

    return ThresholdDerivation(
        thresholds=thresholds,
        raw_thresholds=raw,
        refusals=refusals,
        wp_reference_shares=wp_shares,
        band_shares_after=shares_after,
        band_counts_after=counts_after,
        band_shares_under_current=shares_current,
        band_counts_under_current=counts_current,
        games_changing_band=changing,
        eligible_rows={target: int(magnitudes[target].size) for target in TARGETS},
        population_rows=len(edges),
    )


def extend_prior_season_bias(
    residuals_by_season: Mapping[int, np.ndarray],
    record_biases: Mapping[int, float],
    target_season: int,
) -> float:
    """The walk-forward bias for *target_season*, after REPRODUCING the record's seasons.

    Every season the corrected chain fit prices is re-estimated from *residuals_by_season*
    through the same ``estimate_prior_season_bias`` and must come out EQUAL. Only then is the
    series extended to *target_season*, over every strictly-prior residual the record's own rule
    pools.

    Raises:
        ChainFitReproductionError: a recorded season does not reproduce.
        EmptyResidualPoolError: no strictly-prior season exists.
    """
    pools = {int(season): values for season, values in residuals_by_season.items()}
    for season, recorded in sorted(record_biases.items()):
        reproduced = float(estimate_prior_season_bias(pools, int(season)))
        if reproduced != float(recorded):
            msg = (
                f"season {season}: the corrected chain fit records bias {recorded!r} but the "
                f"same walk-forward residuals give {reproduced!r}. The two were computed on "
                "different predictions; the 2026 bias is not extended from a series it does "
                "not reproduce."
            )
            raise ChainFitReproductionError(msg)
    try:
        return float(estimate_prior_season_bias(pools, target_season))
    except ValueError as error:
        msg = (
            f"no strictly-prior residual exists for {target_season}; no bias is invented: "
            f"{error}"
        )
        raise EmptyResidualPoolError(msg) from error


def load_corrected_chain_fit(repo_root: Path = REPO_ROOT) -> Any:
    """The corrected chain fit, through the live loader, by EXPLICIT path.

    Called through the module attribute so the path it is handed is observable, and handed
    ``repo_root / CHAIN_FIT_SOURCE`` -- never ``DEFAULT_CHAIN_FIT_PATH``.
    """
    from backtest import weekly_bet_list

    return weekly_bet_list.load_frozen_chain_fit(path=repo_root / CHAIN_FIT_SOURCE)


def resolve_corrected_inputs(repo_root: Path = REPO_ROOT) -> tuple[dict[str, str], str]:
    """The four served artifact ids and the converter the LIVE blend binds.

    Precondition of the whole correction (Plan 33.2-25 Task 3 ruled "swap"): the live manifest
    names exactly ``tests.phase33_state.P332_25B_SWAP_ARTIFACT_IDS``. The converter is reached
    through the live blend's own provenance (``blend_weights.json``), never as a fifth manifest
    pointer, and must be the one the blend was recorded binding.

    Raises:
        DigestMismatchError: the live manifest or the blend's converter is not the recorded one.
    """
    import json

    from tests.phase33_state import (
        P332_25B_BLEND_CONVERTER_ARTIFACT_ID,
        P332_25B_SWAP_ARTIFACT_IDS,
    )

    swap = dict(P332_25B_SWAP_ARTIFACT_IDS)
    manifest = json.loads(
        (repo_root / "artifacts" / "latest.json").read_text(encoding="utf-8")
    )
    moved = {key: manifest.get(key) for key in swap if manifest.get(key) != swap[key]}
    if moved:
        msg = (
            f"artifacts/latest.json does not serve the recorded post-swap ids {swap}: {moved}. "
            "The correction is derived from the corrected models only."
        )
        raise DigestMismatchError(msg)
    blend = json.loads(
        (repo_root / "artifacts" / swap["blend"] / "blend_weights.json").read_text(
            encoding="utf-8"
        )
    )
    converter_id = str(blend.get("market_probability_artifact_id"))
    if converter_id != P332_25B_BLEND_CONVERTER_ARTIFACT_ID:
        msg = (
            f"the live blend binds converter {converter_id!r}, not the recorded "
            f"{P332_25B_BLEND_CONVERTER_ARTIFACT_ID!r}; there is no yardstick to derive against."
        )
        raise DigestMismatchError(msg)
    return swap, converter_id


def corrected_input_keys(swap: Mapping[str, str], converter_id: str) -> list[str]:
    """Every input the correction reads, in a stable order."""
    keys = [
        "artifacts/latest.json",
        "data/silver/games.parquet",
        "data/silver/odds_timeline.parquet",
        CHAIN_FIT_SOURCE.as_posix(),
    ]
    keys.extend(f"data/gold/features_{target}.parquet" for target in TARGETS)
    keys.extend(f"artifact:{swap[key]}" for key in (*TARGETS, "blend"))
    keys.append(f"artifact:{converter_id}")
    return keys


def measure_input_digests_for(keys: Sequence[str], repo_root: Path) -> dict[str, str]:
    """Measure each key's digest with the instrument its location requires."""
    measured: dict[str, str] = {}
    for key in keys:
        if key.startswith("artifact:"):
            measured[key] = digest_artifact_dir(
                repo_root / "artifacts" / key.split(":", 1)[1]
            )
        else:
            measured[key] = digest_input_file(repo_root / key, key)
    return measured


@dataclass(frozen=True)
class CorrectedDerivation:
    """Everything the corrected module and the correction document state."""

    superseded_commit: str
    swap: Mapping[str, str]
    converter_id: str
    gold_generation: str
    chain_fit_record_id: str
    chain_fit_floors: Mapping[str, float | None]
    chain_fit_sds: Mapping[str, float | None]
    thresholds: ThresholdDerivation
    superseded_pairs: Mapping[str, tuple[float, float]]
    wp_excluded_no_prior_fold: int
    excluded_no_prelock_line: int
    bias: Mapping[str, float]
    superseded_bias: Mapping[str, float]
    bias_by_season: Mapping[str, Mapping[int, float]]
    bias_seasons: tuple[int, ...]
    bias_rows: Mapping[str, int]
    input_digests: Mapping[str, str]
    formatter: str
    thread_limit: int


def measure_corrected(
    input_digests: Mapping[str, str], repo_root: Path = REPO_ROOT
) -> CorrectedDerivation:
    """Derive the corrected thresholds and 2026 bias. READ-ONLY over data/, artifacts/, outputs/.

    Raises:
        DigestMismatchError: the manifest, the converter or the gold is not the recorded one.
        ChainFitReproductionError: the residuals do not reproduce the corrected chain fit.
    """
    import json

    from threadpoolctl import threadpool_limits

    import backtest.cold_start_constants as superseded
    from backtest.profitability_2025 import _residuals_by_season
    from backtest.tune import (
        _gold_predictions_fn,
        common_gold_generation,
        read_source_recipes,
    )
    from backtest.weekly_bet_list import frozen_overlay_season
    from config.tuning_preregistration import PINNED_THREAD_COUNT
    from models.blending_data import load_tuning_period_data
    from models.market_probability import load_market_probability_artifact
    from scripts.derive_corrected_ev_chain import (
        BIAS_SEED_SEASONS,
        build_candidate_frames,
        fit_frame_for,
    )
    from tests.gold_generation import gold_generation_key
    from tests.phase33_state import P332_25B_REFIT_GOLD_GENERATION

    swap, converter_id = resolve_corrected_inputs(repo_root)
    artifacts_dir = repo_root / "artifacts"
    model_ids = {target: swap[target] for target in TARGETS}
    recipes = read_source_recipes(model_ids, artifacts_dir)
    gold_digest = common_gold_generation(recipes, P332_25B_REFIT_GOLD_GENERATION)
    if gold_generation_key() != gold_digest:
        msg = (
            "the gold on disk is not the generation the corrected models were fitted on "
            f"({gold_digest}); their walk-forward predictions cannot be reproduced."
        )
        raise DigestMismatchError(msg)

    record = json.loads((repo_root / CHAIN_FIT_SOURCE).read_text(encoding="utf-8"))
    if dict(record["artifact_ids"]) != swap or (
        record["market_probability_artifact_id"] != converter_id
    ):
        msg = (
            f"{CHAIN_FIT_SOURCE.as_posix()} was fitted on {record['artifact_ids']} / "
            f"{record['market_probability_artifact_id']}, not the served {swap} / "
            f"{converter_id}."
        )
        raise DigestMismatchError(msg)
    chain_fit = load_corrected_chain_fit(repo_root)

    converter = load_market_probability_artifact(converter_id, artifacts_dir)
    slopes = converter["walk_forward_slopes"]
    corpus = load_tuning_period_data(repo_root / "data" / "silver")
    predict = _gold_predictions_fn(repo_root / "data" / "gold")
    seasons = [*BIAS_SEED_SEASONS, *CORRECTED_THRESHOLD_SEASONS]
    # THE THREAD PIN (Plan 33.2-22): XGBoost answers differently at different OpenMP thread
    # counts; every published fit is pinned and the value recorded.
    with threadpool_limits(limits=PINNED_THREAD_COUNT):
        predictions = {
            target: predict(target, recipes[target], seasons) for target in TARGETS
        }

    in_window: dict[str, pd.DataFrame] = {
        target: cast(
            "pd.DataFrame", frame[frame["season"].isin(CORRECTED_THRESHOLD_SEASONS)]
        )
        for target, frame in predictions.items()
    }
    edges = build_prelock_edge_frame(corpus.frame, in_window, slopes)
    thresholds = derive_threshold_pairs(
        edges.frame, superseded.EDGE_TIER_THRESHOLDS_BY_TARGET
    )

    # THE 2026 BIAS continues the corrected chain fit's own walk-forward series: the SAME rows
    # (the bias seed plus every corpus game), the SAME residual helper and the SAME estimator.
    frames = build_candidate_frames(corpus.frame, predictions, slopes)
    overlay = frozen_overlay_season()
    bias: dict[str, float] = {}
    bias_by_season: dict[str, dict[int, float]] = {}
    bias_rows: dict[str, int] = {}
    for target in TARGETS:
        residuals = _residuals_by_season(
            fit_frame_for(target, predictions, frames), target
        )
        recorded = {
            season: value
            for season, value in chain_fit[target].season_bias_by_season.items()
            if season != overlay
        }
        bias[target] = extend_prior_season_bias(
            residuals, recorded, CORRECTED_BIAS_TARGET_SEASON
        )
        bias_by_season[target] = {
            season: float(values.mean()) for season, values in sorted(residuals.items())
        }
        bias_rows[target] = int(sum(len(values) for values in residuals.values()))

    return CorrectedDerivation(
        superseded_commit=_resolve_commit(SUPERSEDED_PREREGISTRATION_SHORT, repo_root),
        swap=swap,
        converter_id=converter_id,
        gold_generation=gold_digest,
        chain_fit_record_id=str(record["record_id"]),
        chain_fit_floors={t: chain_fit[t].ev_floor_t for t in TARGETS},
        chain_fit_sds={t: chain_fit[t].frozen_sd for t in TARGETS},
        thresholds=thresholds,
        superseded_pairs=dict(superseded.EDGE_TIER_THRESHOLDS_BY_TARGET),
        wp_excluded_no_prior_fold=len(edges.wp_excluded),
        excluded_no_prelock_line=len(corpus.excluded),
        bias=bias,
        superseded_bias=dict(superseded.CHAIN_FIT_BIAS_2026),
        bias_by_season=bias_by_season,
        bias_seasons=tuple(sorted(bias_by_season["ats"])),
        bias_rows=bias_rows,
        input_digests=dict(input_digests),
        formatter=formatter_version(),
        thread_limit=int(PINNED_THREAD_COUNT),
    )


def _resolve_commit(short: str, repo_root: Path) -> str:
    """The full sha of *short*, refusing a sha git cannot resolve."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{short}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _pair_text(pair: tuple[float, float] | None) -> str:
    return "None" if pair is None else f"({pair[0]:.4f}, {pair[1]:.4f})"


def _raw_pair_text(pair: tuple[float, float] | None) -> str:
    return "None" if pair is None else f"({pair[0]!r}, {pair[1]!r})"


def _target_lines(values: Mapping[str, Any], render: Any = repr) -> str:
    return "\n".join(
        f"    {target!r}: {render(values[target])}," for target in sorted(values)
    )


def render_corrected_module(derivation: CorrectedDerivation) -> str:
    """Emit ``backtest/corrected_cold_start_constants.py``. Pure; deterministic; ASCII."""
    d = derivation
    t = d.thresholds
    wp_seasons = tuple(
        season for season in CORRECTED_THRESHOLD_SEASONS if season != 2020
    )
    sources = {
        target: {
            "artifact": d.swap[target],
            "kind": "owner_ruled_swap",
            "residuals": "walk-forward refits of this artifact's own recorded recipe",
            "verdict": "NOT_GATED (SPEC R13: owner readiness ruling, no pass/fail gate)",
        }
        for target in TARGETS
    }
    eligible = {
        target: {
            "bias_rows": d.bias_rows[target],
            "bias_seasons": len(d.bias_by_season[target]),
            "threshold_population_rows": t.population_rows,
            "threshold_rows": t.eligible_rows[target],
        }
        for target in TARGETS
    }
    bias_season_lines = "\n".join(
        f"    {target!r}: {{"
        + ", ".join(
            f"{season}: {value!r}" for season, value in d.bias_by_season[target].items()
        )
        + "},"
        for target in sorted(d.bias_by_season)
    )
    wp_shares = t.wp_reference_shares
    wp_share_text = (
        "None"
        if wp_shares is None
        else "{"
        + ", ".join(f"{k!r}: {wp_shares[k]!r}" for k in ("low", "medium", "high"))
        + "}"
    )

    return f'''"""SUPERSEDING CORRECTION of the frozen 2026 cold-start rule (Plan 33.2-26, SPEC R14).

GENERATOR OUTPUT. Emitted by ``python -m scripts.derive_cold_start_constants --corrected``.
Do NOT hand-edit any value below; re-run the derivation.

WHAT IT SUPERSEDES. Commit ``{SUPERSEDED_PREREGISTRATION_SHORT}`` ({d.superseded_commit}) froze the 2026 edge-band
thresholds and the 2026 chain-fit bias in ``backtest/cold_start_constants.py`` +
``COLD-START-PREREGISTRATION.md``. Those values were derived from models fitted on inputs later
found defective and from CLOSING lines, and those models are gone. This module SUPERSEDES them.
The originals are byte-unchanged and remain the record of what was frozen and when; nothing here
edits them. ``COLD-START-CORRECTION.md`` is the human-readable half of this record.

WHAT MOVED, symbol by symbol:
  * ``EDGE_TIER_THRESHOLDS_BY_TARGET`` / ``_UNROUNDED`` -- re-derived on the corrected models'
    walk-forward predictions over the owned PRE-LOCK lines ({CORRECTED_THRESHOLD_SEASONS[0]}-{CORRECTED_THRESHOLD_SEASONS[-1]}), never a closing
    line. WP's market side is the spread-derived probability converted OUT OF FOLD
    (``models.market_probability.oof_market_probability``); its first season has no prior fold
    and leaves the WP part as ``no_prior_fold_converter``. A target below
    ``MIN_HONEST_THRESHOLD_ROWS`` gets ``None`` -- NO threshold, therefore NO bets and NO edge
    band -- never a default, a zero or the old value (``EDGE_TIER_THRESHOLD_REFUSALS`` says why);
  * ``WP_ANCHOR_BAND_SHARES`` -- re-measured on that population;
  * ``CHAIN_FIT_BIAS_2026`` / ``_SEASONS`` / ``_BY_SEASON`` -- the corrected chain fit's own
    walk-forward bias series (Plan 33.2-29, ``outputs/p332/corrected_chain_fit.json``) extended
    to ``CHAIN_FIT_BIAS_TARGET_SEASON``: the derivation reproduces every season that record prices
    before extending it. The pool ends at 2024 because no 2025 row is read, which is why the target
    season is NAMED here rather than derived as the pool's last season plus one;
  * ``THRESHOLD_DERIVATION_POPULATION``, ``EDGE_TIER_THRESHOLD_UNITS`` (WP's market side),
    ``CHAIN_FIT_BIAS_SOURCE_BY_TARGET``, ``DERIVATION_*`` and the band tables -- restated for the
    corrected population. "UNDER_CURRENT" now means under the superseded 11761c7 pairs.

WHAT DID NOT MOVE: WP's 0.05 / 0.02 pair (the anchor), the WP-anchored band-share quantile rule
(``THRESHOLD_QUANTILE_CONVENTION``, ``numpy.quantile`` method "linear"), the STRICT ``>`` bands,
the digest refusals, the walk-forward bias estimator
(``backtest.ou_ev_chain.estimate_prior_season_bias``) and ``FIX_CYCLE_ALLOWANCE``.

THE OTHER TWO MOVED PARTS OF THE 2026 BET RULE -- the EV floor and the frozen residual SD -- are
superseded separately, naming ``ee20773``: ``backtest.corrected_ev_chain_constants`` and
``EV-CHAIN-CORRECTION.md`` (Plan 33.2-29).

NOT CLEAN EVIDENCE (D33.2-07): every number here is re-measured on past seasons. It sets a
threshold; it does not show one is profitable. Only the 2026 season, recorded live, counts.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

# THE TWO FILES THAT ARE THIS CORRECTION, and the two it supersedes. Repo-root-relative POSIX.
PREREGISTRATION_PATHS: tuple[str, ...] = (
    "{CORRECTION_DOCUMENT_PATH}",
    "{CORRECTED_MODULE_PATH}",
)
SUPERSEDED_PREREGISTRATION_PATHS: tuple[str, ...] = (
    "{DOCUMENT_PATH}",
    "{MODULE_PATH}",
)
SUPERSEDED_PREREGISTRATION_COMMIT: str = {d.superseded_commit!r}

# THE CORRECTED EDGE THRESHOLDS, (HIGH, MEDIUM) per target on its OWN unit, four decimal places.
# ``None`` = NO honest threshold: no bets and no edge band for that target.
EDGE_TIER_THRESHOLDS_BY_TARGET: dict[str, tuple[float, float] | None] = {{
{_target_lines(t.thresholds, _pair_text)}
}}

EDGE_TIER_THRESHOLDS_UNROUNDED: dict[str, tuple[float, float] | None] = {{
{_target_lines(t.raw_thresholds, _raw_pair_text)}
}}

# Why a target has no threshold, when one has none. Empty when every target has one.
EDGE_TIER_THRESHOLD_REFUSALS: dict[str, str] = {{
{_target_lines(t.refusals)}
}}

# The pairs this correction supersedes, READ from the frozen 11761c7 module and restated so the
# old value sits beside the new one.
SUPERSEDED_EDGE_TIER_THRESHOLDS_BY_TARGET: dict[str, tuple[float, float]] = {{
{_target_lines(d.superseded_pairs, _pair_text)}
}}

EDGE_TIER_THRESHOLD_UNITS: dict[str, str] = {{
{_target_lines(EDGE_UNITS_CORRECTED)}
}}

THRESHOLD_QUANTILE_CONVENTION: str = (
    'numpy.quantile(magnitudes, q, method="linear"); q taken from WP band shares under its '
    'unchanged 0.05 / 0.02 pair; bands assigned with STRICT > so a value exactly at a '
    'threshold falls in the LOWER band'
)

# Below this many rows with a computable edge a target gets NO threshold (the calibration gate's
# n >= 100 is the nearest existing floor).
MIN_HONEST_THRESHOLD_ROWS: int = {MIN_HONEST_THRESHOLD_ROWS}

# WP's measured band shares under its unchanged pair -- the ANCHOR the other two reproduce.
WP_ANCHOR_BAND_SHARES: dict[str, float] | None = {wp_share_text}

THRESHOLD_DERIVATION_SEASONS: tuple[int, ...] = {CORRECTED_THRESHOLD_SEASONS!r}
THRESHOLD_WP_DERIVATION_SEASONS: tuple[int, ...] = {wp_seasons!r}
THRESHOLD_WINDOW_REASON: str = {CORRECTED_WINDOW_REASON!r}

# The WP rows with a pre-lock line whose season has no prior-fold converter slope. Counted,
# never filled with the serving slope or a neighbouring season's.
WP_EXCLUDED_NO_PRIOR_FOLD_CONVERTER: int = {d.wp_excluded_no_prior_fold}

# Scheduled games in the window with no owned line at or before their lock.
EXCLUDED_NO_PRELOCK_LINE: int = {d.excluded_no_prelock_line}

THRESHOLD_DERIVATION_POPULATION: str = (
    'the owned pre-lock corpus (silver odds_timeline, the last line at or before each '
    "game's own lock) over {CORRECTED_THRESHOLD_SEASONS[0]}-{CORRECTED_THRESHOLD_SEASONS[-1]}, scored with the corrected models' walk-forward "
    'predictions (each season predicted by a fit on strictly earlier seasons); WP over '
    '{wp_seasons[0]}-{wp_seasons[-1]} only, its market side converted out of fold; shares taken over the rows '
    'carrying a COMPUTABLE edge'
)

# THE 2026 CHAIN-FIT BIAS, per target: the pooled mean residual (actual - predicted) over every
# strictly-prior season the corrected chain fit's walk-forward rule pools.
CHAIN_FIT_BIAS_2026: dict[str, float] = {{
{_target_lines(d.bias)}
}}

CHAIN_FIT_BIAS_TARGET_SEASON: int = {CORRECTED_BIAS_TARGET_SEASON}

CHAIN_FIT_BIAS_SEASONS: tuple[int, ...] = {d.bias_seasons!r}

# The per-season mean residual each pooled bias was estimated from. INT season keys; a JSON
# representation of the same mapping carries STRING keys.
CHAIN_FIT_BIAS_BY_SEASON: dict[str, dict[int, float]] = {{
{bias_season_lines}
}}

SUPERSEDED_CHAIN_FIT_BIAS_2026: dict[str, float] = {{
{_target_lines(d.superseded_bias)}
}}

CHAIN_FIT_BIAS_SOURCE_BY_TARGET: dict[str, dict[str, object]] = {{
{_target_lines(sources)}
}}

# The corrected chain fit this bias series continues (Plan 33.2-29), read by explicit path.
CHAIN_FIT_SOURCE_RECORD_PATH: str = {CHAIN_FIT_SOURCE.as_posix()!r}
CHAIN_FIT_SOURCE_RECORD_ID: str = {d.chain_fit_record_id!r}

# The corrected model artifacts every number above was derived from, the live blend, and the
# converter that blend binds.
DERIVATION_ARTIFACTS: dict[str, str] = {{
{_target_lines({target: d.swap[target] for target in TARGETS})}
}}
LIVE_BLEND_ARTIFACT_ID: str = {d.swap["blend"]!r}
MARKET_PROBABILITY_ARTIFACT_ID: str = {d.converter_id!r}
DERIVATION_GOLD_GENERATION: str = {d.gold_generation!r}

# The OpenMP / BLAS thread count every walk-forward fit was pinned to.
DERIVATION_THREAD_LIMIT: int = {d.thread_limit}

# EVERY input this derivation read, with its digest (gitignored stores over RAW bytes; an
# ``artifact:<id>`` key is the sha256 over the directory's sorted file manifest).
DERIVATION_INPUT_DIGESTS: dict[str, str] = {{
{_render_mapping(d.input_digests)}
}}

DERIVATION_FORMATTER: str = {d.formatter!r}

DERIVATION_ELIGIBLE_COUNTS: dict[str, dict[str, int]] = {{
{_target_lines(eligible)}
}}

# THE LABEL MOVEMENT on the corrected edges: under the superseded 11761c7 pairs, and under the
# corrected pairs. A target with no threshold has no row.
BAND_SHARES_UNDER_CURRENT_THRESHOLDS: dict[str, dict[str, float]] = {{
{_render_share_table(t.band_shares_under_current)}
}}

BAND_COUNTS_UNDER_CURRENT_THRESHOLDS: dict[str, dict[str, int]] = {{
{_render_count_table(t.band_counts_under_current)}
}}

ATS_BAND_SHARES_AFTER: dict[str, dict[str, float]] = {{
{_render_share_table(t.band_shares_after)}
}}

ATS_BAND_COUNTS_AFTER: dict[str, dict[str, int]] = {{
{_render_count_table(t.band_counts_after)}
}}

GAMES_CHANGING_BAND: dict[str, int] = {{
{_render_mapping(t.games_changing_band)}
}}

# Plan 33-08's declared allowance, carried unchanged.
FIX_CYCLE_ALLOWANCE: int = {FIX_CYCLE_ALLOWANCE}

# Where the other two moved parts of the 2026 bet rule live (Plan 33.2-29, superseding ee20773).
EV_CHAIN_CORRECTION_MODULE: str = "backtest.corrected_ev_chain_constants"
'''


def render_correction_document(derivation: CorrectedDerivation) -> str:
    """Emit ``COLD-START-CORRECTION.md``. Pure; deterministic; ASCII."""
    d = derivation
    t = d.thresholds
    rows = []
    for target in ("wp", "ats", "ou"):
        old = d.superseded_pairs[target]
        new = t.thresholds[target]
        new_text = (
            "NONE (no bets)" if new is None else f"`{new[1]:.4f}` / `{new[0]:.4f}`"
        )
        rows.append(
            f"| {target} | `{old[1]:.4f}` / `{old[0]:.4f}` | {new_text} "
            f"| {t.eligible_rows[target]} | {EDGE_UNITS_CORRECTED[target]} |"
        )
    threshold_table = "\n".join(rows)

    refusal_lines = (
        "\n".join(
            f"- `{target}`: {text}" for target, text in sorted(t.refusals.items())
        )
        if t.refusals
        else "None. Every target cleared the honest-data floor, so every target has a threshold."
    )

    bias_rows = "\n".join(
        f"| {target} | `{d.superseded_bias[target]!r}` | `{d.bias[target]!r}` |"
        for target in ("wp", "ats", "ou")
    )

    movement = []
    for target in ("wp", "ats", "ou"):
        if target not in t.band_counts_after:
            movement.append(f"| {target} | no threshold | no threshold | n/a |")
            continue
        before = t.band_counts_under_current[target]
        after = t.band_counts_after[target]
        movement.append(
            f"| {target} | {before['low']} / {before['medium']} / {before['high']} "
            f"| {after['low']} / {after['medium']} / {after['high']} "
            f"| {t.games_changing_band[target]} |"
        )
    movement_table = "\n".join(movement)

    ev_rows = "\n".join(
        f"| {target} | `{d.chain_fit_floors[target]!r}` | `{d.chain_fit_sds[target]!r}` |"
        for target in ("wp", "ats", "ou")
    )
    wp_shares = t.wp_reference_shares
    share_text = (
        "not measured (the WP anchor was refused)"
        if wp_shares is None
        else (
            f"low `{wp_shares['low']:.4f}` / medium `{wp_shares['medium']:.4f}` / high "
            f"`{wp_shares['high']:.4f}`"
        )
    )
    seasons = f"{CORRECTED_THRESHOLD_SEASONS[0]}-{CORRECTED_THRESHOLD_SEASONS[-1]}"
    wp_window = f"{CORRECTED_THRESHOLD_SEASONS[1]}-{CORRECTED_THRESHOLD_SEASONS[-1]}"
    wp_pair = t.thresholds["wp"]
    wp_admission = "no threshold" if wp_pair is None else f"`{wp_pair[1]:.4f}`"

    return f"""# COLD-START CORRECTION -- superseding the 2026 rule frozen at `{SUPERSEDED_PREREGISTRATION_SHORT}`

**Status:** a SUPERSEDING CORRECTION. This document and `{CORRECTED_MODULE_PATH}` are one
record in two files. The record they supersede -- `{DOCUMENT_PATH}` and `{MODULE_PATH}`,
frozen at commit `{SUPERSEDED_PREREGISTRATION_SHORT}` (`{d.superseded_commit}`) -- is
**byte-unchanged** and stays the record of what was frozen and when. Nothing here edits it.

**Why it is superseded.** `{SUPERSEDED_PREREGISTRATION_SHORT}` froze the 2026 edge thresholds and the 2026 chain-fit
bias from models fitted on inputs later found defective (Phase 33.2) and from CLOSING lines, which
did not exist at a game's lock. Those models are gone. A value derived from them sits inside the
live 2026 bet rule, so it is replaced visibly rather than edited quietly.

**Not clean evidence (D33.2-07).** Every number below is re-measured on past seasons. It sets a
threshold; it does not show that a bet in any band is profitable. Only the 2026 season, recorded
live under the new lock rule, counts as evidence.

---

## 1. What was swapped, and what was held

**Swapped:** the models -- the three corrected artifacts production serves
(`{d.swap["wp"]}`, `{d.swap["ats"]}`, `{d.swap["ou"]}`), through their own recorded recipes'
walk-forward predictions, each season predicted by a fit on strictly earlier seasons -- and the
market: the owned `odds_timeline` line at or before each game's lock, never a closing line.
WP's market side is the spread-derived probability from converter `{d.converter_id}` (the one
the live blend `{d.swap["blend"]}` binds), converted OUT OF FOLD for every historical game.

**Held:** WP's `0.0500` / `0.0200` anchor pair, the WP-anchored band-share quantile rule
(`numpy.quantile`, method "linear"), the STRICT `>` bands, the digest refusals and the
walk-forward bias estimator.

**The window is {seasons}.** {CORRECTED_WINDOW_REASON}
WP uses {wp_window} only: the window's first season has no prior-fold converter slope, so
**{d.wp_excluded_no_prior_fold} games leave the WP derivation as `no_prior_fold_converter`**
(counted, never filled with the serving slope). They stay in the ATS and O/U derivations.
{d.excluded_no_prelock_line} scheduled games in the window had no owned line at or before their
lock and are not in any part of it.

---

## 2. The edge thresholds, old beside new

`medium / high`, each on its target's own unit, over the rows with a computable edge.

| target | superseded (`{SUPERSEDED_PREREGISTRATION_SHORT}`) | corrected | rows | unit |
|---|---|---|---|---|
{threshold_table}

WP's measured band shares under its unchanged pair, the anchor ATS and O/U reproduce: {share_text}.

**A target with too little honest data gets NO threshold.** Below
`MIN_HONEST_THRESHOLD_ROWS = {MIN_HONEST_THRESHOLD_ROWS}` rows the derivation raises
`InsufficientHonestDataForThresholdError`, records `None`, and that target places no bets and
carries no edge band. The superseded value is never borrowed. Refusals in this derivation:

{refusal_lines}

---

## 3. The label movement, in games

Counts are `low / medium / high` over the corrected edges, under the superseded pairs and under
the corrected ones.

| target | under the `{SUPERSEDED_PREREGISTRATION_SHORT}` pairs | under the corrected pairs | games changing band |
|---|---|---|---|
{movement_table}

---

## 4. The 2026 chain-fit bias, old beside new

| target | superseded (`{SUPERSEDED_PREREGISTRATION_SHORT}`) | corrected |
|---|---|---|
{bias_rows}

The corrected value continues the walk-forward bias series of the corrected chain fit
(`{CHAIN_FIT_SOURCE.as_posix()}`, record `{d.chain_fit_record_id}`, Plan 33.2-29): the same rows,
the same residual helper and the same estimator. The derivation first reproduces every season
that record prices, exactly, and only then extends the series to 2026, pooling every
strictly-prior season it covers ({d.bias_seasons[0]}-{d.bias_seasons[-1]}). No 2025 row is read, so the pool ends at
2024 and the target season is named (`CHAIN_FIT_BIAS_TARGET_SEASON`) rather than inferred from
it. The superseded value was pooled IN-SAMPLE over the retired models' own training seasons; this
one is out of sample.

---

## 5. The other two moved parts of the 2026 bet rule: the EV floor and the frozen residual SD

The 2026 bet rule is these thresholds and this bias PLUS the per-target **EV floor** (the number
that decides whether a bet is placed at all) and the **frozen residual SD** (the scale that turns
a model's miss into a bet's expected value). D33.2-25 rules that all of them move together. The
EV floor and the residual SD are superseded separately -- naming `ee20773`, the Phase-31
pre-registration they came from -- by Plan 33.2-29: `backtest/corrected_ev_chain_constants.py`
and `EV-CHAIN-CORRECTION.md`. Their values, as the corrected chain fit records them:

| target | EV floor | frozen residual SD |
|---|---|---|
{ev_rows}

WP fits no residual SD by design (D31-07). A `None` EV floor would mean no honest floor and no
bets for that target; none is `None` here.

---

## 6. How the live rule changes

The live rule changes in ONE commit (Plan 33.2-26 Task 3), which moves every live reader
together: the chain-fit record path, this bias, and the edge bands read by the web cache and the
current-week predictions. Before that commit the live bet list read the Phase-31 chain-fit record
and the `{SUPERSEDED_PREREGISTRATION_SHORT}` bias; no run can judge corrected floors against an uncorrected bias,
or the reverse. From that commit on, a 2026 WIN bet must pass two tests:
its edge over the spread-derived market probability, on the side bet, is above WP's corrected
MEDIUM threshold ({wp_admission}: the edge band is at least "medium"), and the moneyline captured at
that game's lock still leaves positive value after the book's cut (D33.2-11). A target whose
threshold is `None` places no bets at all.

---

## 7. How to reproduce every number above

```
OMP_NUM_THREADS=1 uv run python -m scripts.derive_cold_start_constants --corrected --trust-inputs
```

The derivation reads the four served artifact ids from `tests/phase33_state.py`, refuses unless
`artifacts/latest.json` serves them and the live blend binds the recorded converter, pins every
fit to {d.thread_limit} thread, and records every input's digest in `DERIVATION_INPUT_DIGESTS`.
Running it twice against the same inputs produces byte-identical files.

---

*Phase: 33.2-information-time-integrity-day-before-kickoff-lock-and-hones*
*Plan 33.2-26, superseding `{SUPERSEDED_PREREGISTRATION_SHORT}`*
"""


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def default_input_keys(artifacts: Mapping[str, str]) -> list[str]:
    """The canonical input-key list this derivation reads, in a stable order."""
    keys = [
        "artifacts/latest.json",
        "config/phase33_gate_verdict.toml",
        "data/silver/odds_snapshot.parquet",
    ]
    keys.extend(f"data/gold/features_{target}.parquet" for target in TARGETS)
    keys.extend(f"artifact:{artifacts[target]}" for target in TARGETS)
    return keys


def measure_input_digests(
    artifacts: Mapping[str, str], repo_root: Path = REPO_ROOT
) -> dict[str, str]:
    """Measure every canonical input digest. The ``--print-digests`` helper uses this."""
    measured: dict[str, str] = {}
    for key in default_input_keys(artifacts):
        if key.startswith("artifact:"):
            measured[key] = digest_artifact_dir(
                repo_root / "artifacts" / key.split(":", 1)[1]
            )
        else:
            measured[key] = digest_input_file(repo_root / key, key)
    return measured


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the derivation's arguments."""
    parser = argparse.ArgumentParser(
        prog="derive_cold_start_constants",
        description=(
            "Derive the Phase-33 cold-start pre-registration from the END-STATE artifacts and "
            "emit its two rule files. Read-only over data/, outputs/ and artifacts/."
        ),
    )
    parser.add_argument("--wp-artifact")
    parser.add_argument("--ats-artifact")
    parser.add_argument("--ou-artifact")
    parser.add_argument(
        "--corrected",
        action="store_true",
        help=(
            "Emit the SUPERSEDING CORRECTION of 11761c7 (Plan 33.2-26) instead: the served "
            "artifact ids come from tests/phase33_state.py, so no --*-artifact is given."
        ),
    )
    parser.add_argument(
        "--digest",
        action="append",
        default=[],
        metavar="KEY=SHA256",
        help=(
            "A declared input digest. KEY is a repo-relative path or 'artifact:<id>'. "
            "Repeatable. Every canonical input must be declared unless --trust-inputs is given."
        ),
    )
    parser.add_argument(
        "--trust-inputs",
        action="store_true",
        help=(
            "Measure the input digests instead of verifying declared ones. For the FIRST run, "
            "which is what produces the digests a later run declares. Never for a check run."
        ),
    )
    # No default here: the destination depends on the mode. Without --corrected it is the
    # original pre-registration (MODULE_PATH / DOCUMENT_PATH); with --corrected it is the
    # correction (CORRECTED_MODULE_PATH / CORRECTION_DOCUMENT_PATH). Both modes honour an
    # explicit path (review WR-16: --corrected used to drop it silently).
    parser.add_argument("--out-module")
    parser.add_argument("--out-doc")
    parser.add_argument(
        "--print-digests",
        action="store_true",
        help="Print the measured canonical input digests and exit without emitting anything.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Reproduce the frozen pre-registration WITHOUT writing anything: read the artifact "
            "ids and input digests it records, verify them, render both files in memory and "
            "compare them with the committed ones. Exits 1 on a difference."
        ),
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.check and args.corrected:
        parser.error(
            "--check reproduces the original pre-registration; drop --corrected"
        )
    missing = [
        flag
        for flag, value in (
            ("--wp-artifact", args.wp_artifact),
            ("--ats-artifact", args.ats_artifact),
            ("--ou-artifact", args.ou_artifact),
        )
        if value is None
    ]
    if missing and not (args.corrected or args.check):
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def _declared_digests(entries: Sequence[str]) -> dict[str, str]:
    """Parse repeated ``--digest KEY=SHA256`` arguments."""
    declared: dict[str, str] = {}
    for entry in entries:
        key, _, value = entry.strip().partition("=")
        key, value = key.strip(), value.strip()
        if not key or not value:
            msg = f"--digest expects KEY=SHA256, got {entry!r}"
            raise SystemExit(msg)
        declared[key] = value
    return declared


def _write_ascii(repo_root: Path, outputs: Sequence[tuple[str, str]]) -> None:
    """Refuse non-ASCII text, then write each ``(text, relative path)`` with LF endings."""
    for text, name in outputs:
        if not text.isascii():
            msg = f"{name} is not pure ASCII; refusing to emit (CLAUDE.md hard constraint)."
            raise SystemExit(msg)
    for text, name in outputs:
        (repo_root / name).write_text(text, encoding="utf-8", newline="\n")
        sys.stdout.write(f"emitted {name}\n")


def main_corrected(args: argparse.Namespace, repo_root: Path) -> int:
    """Verify the inputs, derive the superseding correction, and emit its two files.

    ``--print-digests`` prints the measured input digests and writes nothing, and
    ``--out-module`` / ``--out-doc`` redirect the two files, exactly as without
    ``--corrected`` (review WR-16). Before, both were dropped silently, so a digest-printing
    run with ``--trust-inputs`` rewrote the live 2026 bet rule's files.
    """
    swap, converter_id = resolve_corrected_inputs(repo_root)
    keys = corrected_input_keys(swap, converter_id)
    if args.print_digests:
        for key, value in measure_input_digests_for(keys, repo_root).items():
            sys.stdout.write(f"{key}={value}\n")
        return 0
    if args.trust_inputs:
        input_digests = measure_input_digests_for(keys, repo_root)
    else:
        declared = _declared_digests(args.digest)
        missing = sorted(set(keys) - set(declared))
        if missing:
            msg = (
                "every input must carry a declared digest; missing: "
                f"{missing}. Pass --trust-inputs only on the first run."
            )
            raise SystemExit(msg)
        input_digests = _verify_digests(declared, repo_root)

    derivation = measure_corrected(input_digests, repo_root)
    module_text = ruff_format(
        render_corrected_module(derivation), CORRECTED_MODULE_PATH
    )
    _write_ascii(
        repo_root,
        (
            (module_text, args.out_module or CORRECTED_MODULE_PATH),
            (
                render_correction_document(derivation),
                args.out_doc or CORRECTION_DOCUMENT_PATH,
            ),
        ),
    )
    t = derivation.thresholds
    sys.stdout.write(
        f"THRESHOLDS= {t.thresholds}\nREFUSALS= {sorted(t.refusals)}\n"
        f"ELIGIBLE_ROWS= {t.eligible_rows}\n"
        f"WP_EXCLUDED_NO_PRIOR_FOLD= {derivation.wp_excluded_no_prior_fold}\n"
        f"WP_ANCHOR_SHARES= {t.wp_reference_shares}\n"
        f"BIAS_2026= {dict(derivation.bias)}\nBIAS_SEASONS= {derivation.bias_seasons}\n"
        f"THREAD_LIMIT= {derivation.thread_limit}\n"
    )
    return 0


def _frozen_literal(module_path: Path, name: str) -> dict[str, str]:
    """The literal value the committed module assigns to *name*, read without importing it."""
    for node in ast.parse(module_path.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return dict(ast.literal_eval(node.value))
    msg = f"{module_path} records no {name}, so there is nothing to reproduce against."
    raise SystemExit(msg)


def main_check(repo_root: Path) -> int:
    """Reproduce the frozen pre-registration in memory and compare it; write NOTHING.

    THE COMMAND THE FROZEN DOCUMENT NAMES (review WR-15). ``COLD-START-PREREGISTRATION.md``
    section 6 tells a reviewer to run ``--check``, and until this existed the only way to
    reproduce the numbers was the emitting path, whose default destinations ARE the frozen
    files. This path takes the artifact ids and input digests from the committed module itself,
    refuses on any moved input exactly as an emitting run does (naming the input and both
    digests), and compares the rendered texts with the committed ones. A CRLF checkout is
    normalised to the LF this program writes before the comparison.

    Returns:
        0 when both files reproduce, 1 when either differs.
    """
    module_path = repo_root / MODULE_PATH
    artifacts = _frozen_literal(module_path, "DERIVATION_ARTIFACTS")
    declared = _frozen_literal(module_path, "DERIVATION_INPUT_DIGESTS")
    missing = sorted(set(default_input_keys(artifacts)) - set(declared))
    if missing:
        msg = (
            f"the committed record declares no digest for {missing}; cannot reproduce."
        )
        raise SystemExit(msg)
    derivation = measure(artifacts, _verify_digests(declared, repo_root), repo_root)
    rendered = (
        (ruff_format(render_module(derivation), MODULE_PATH), MODULE_PATH),
        (render_document(derivation), DOCUMENT_PATH),
    )
    differing = [
        name
        for text, name in rendered
        if (repo_root / name).read_bytes().replace(b"\r\n", b"\n")
        != text.encode("utf-8")
    ]
    if differing:
        sys.stdout.write(f"DIFFERS from the committed file(s): {differing}\n")
        return 1
    sys.stdout.write(f"reproduced {MODULE_PATH} and {DOCUMENT_PATH}; nothing written\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Verify the inputs, derive both quantities, and emit the two rule files."""
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    if args.corrected:
        return main_corrected(args, repo_root)
    if args.check:
        return main_check(repo_root)
    # parse_args refused a missing id outside --corrected, so all three are present here.
    artifacts = {
        "wp": str(args.wp_artifact),
        "ats": str(args.ats_artifact),
        "ou": str(args.ou_artifact),
    }

    if args.print_digests:
        for key, value in measure_input_digests(artifacts, repo_root).items():
            sys.stdout.write(f"{key}={value}\n")
        return 0

    if args.trust_inputs:
        input_digests = measure_input_digests(artifacts, repo_root)
    else:
        declared = _declared_digests(args.digest)
        required = set(default_input_keys(artifacts))
        missing = sorted(required - set(declared))
        if missing:
            msg = (
                "every canonical input must carry a declared digest; missing: "
                f"{missing}. Pass --trust-inputs only on the first run."
            )
            raise SystemExit(msg)
        input_digests = _verify_digests(declared, repo_root)

    derivation = measure(artifacts, input_digests, repo_root)
    args.out_module = args.out_module or MODULE_PATH
    args.out_doc = args.out_doc or DOCUMENT_PATH

    module_text = ruff_format(render_module(derivation), MODULE_PATH)
    document_text = render_document(derivation)
    for text, name in ((module_text, args.out_module), (document_text, args.out_doc)):
        if not text.isascii():
            msg = f"{name} is not pure ASCII; refusing to emit (CLAUDE.md hard constraint)."
            raise SystemExit(msg)

    (repo_root / args.out_module).write_text(
        module_text, encoding="utf-8", newline="\n"
    )
    (repo_root / args.out_doc).write_text(document_text, encoding="utf-8", newline="\n")
    sys.stdout.write(f"emitted {args.out_module}\nemitted {args.out_doc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
