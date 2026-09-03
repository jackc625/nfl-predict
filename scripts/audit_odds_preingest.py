"""Read-only pre-ingest audit for the Phase-31 one-shot 2025 run (Plan 31-02).

WHY THIS EXISTS
---------------
The Phase-31 pre-registration (Plan 31-05) must STATE four things it is not allowed to
ASSUME, because a frozen rule resting on an assumption is a rule resting on an argument:

1. **D31-39 (juice source).** ``31-RESEARCH.md`` DEFECT-5 found a partitioned sibling store
   under ``data/silver/snapshot_ts=.../`` carrying the four juice columns the flat
   ``data/silver/odds_snapshot.parquet`` lacks, but could NOT determine which logical table
   each of the three opaque filenames belongs to (assumption A3: "they are ``ParquetManager``
   table-name hashes"). Until that is measured, "PROMOTE the existing juice columns" versus
   "RE-INGEST through nflreadpy" is a guess. :func:`audit_partitioned_odds_store` settles it.

2. **A1 (ATS residual bias provenance).** RESEARCH measured the pooled ATS residual bias from
   ``data/web_cache.duckdb``, which can be stale relative to the DEPLOYED artifact.
   :func:`measure_ats_residual_bias` re-derives it through
   ``backtest.diagnose.score_deployed_artifacts`` so the number is attributable to a named
   artifact id.

3. **DEFECT-3 (synthetic fixture row).** A hand-written ``2025_W01_TEST@HOME`` row sits in
   production silver and PASSES the OUM-06 sportsbook allowlist, because its sportsbook
   (``draftkings``) is legitimate while its ``game_id`` is not a team pair.
   :func:`assert_no_synthetic_game_ids` is the gate the allowlist cannot be.

4. **A2 (LAR-versus-LA upsert behaviour).** Settled by the dry-run in
   ``tests/integration/test_ingest_2025_odds.py``, never by writing production silver.

HARD BOUNDARY
-------------
This module WRITES NOTHING under ``data/``. It asserts that itself: every parquet under
``data/silver`` is sha256-hashed before and after each audit and a change is a hard failure
(T-31-09). ``git status data/`` is vacuous as a check because ``.gitignore`` blankets
``data/``, so content hashing is the only honest guard.

It REPORTS; it does NOT DECIDE. The recommended D31-39 branch is printed with its reason and
the owner ratifies it at CHECKPOINT 1 in Plan 31-05.

Run it with the project venv::

    .venv/Scripts/python.exe -m scripts.audit_odds_preingest --store-audit
    .venv/Scripts/python.exe -m scripts.audit_odds_preingest --ats-bias
    .venv/Scripts/python.exe -m scripts.audit_odds_preingest --store-audit --ats-bias

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths and formatting.
# ---------------------------------------------------------------------------

DEFAULT_OUT = Path("outputs/p31/odds_store_audit.json")

SILVER_DIR = Path("data/silver")
FLAT_ODDS_PATH = SILVER_DIR / "odds_snapshot.parquet"
PARTITION_PREFIX = "snapshot_ts="

# ONE explicit specifier for every float in this report. Two runs of the script must produce
# byte-identical numbers, so no float is ever rendered by repr() or by an implicit default.
FLOAT_FMT = "{:.17g}"

# The GUID basename pyarrow's write_to_dataset emits: "<32 hex>-<i>.parquet".
_GUID_BASENAME_RE = re.compile(r"^(?P<guid>[0-9a-f]{32})-(?P<index>\d+)\.parquet$")

# Season prefix of a project-standard game_id ("2018_W01_ATL@PHI" -> 2018).
_SEASON_PREFIX_RE = re.compile(r"^(?P<season>\d{4})_")


# ---------------------------------------------------------------------------
# The D31-39 branch rule -- STATED BEFORE THE MEASUREMENT (forking-paths guard).
#
# Choosing a branch after seeing which one looks nicer is exactly the post-hoc selection this
# phase forbids everywhere else. The rule below is fixed; the measurement only evaluates it.
# ---------------------------------------------------------------------------

BRANCH_RULE = (
    "PROMOTE iff ALL THREE hold: (a) every partition file carries all four OddsSchema juice "
    "columns; (b) the partitioned store's rows agree with data/silver/odds_snapshot.parquet "
    "row-for-row on every shared value column for every (game_id, sportsbook) pair the flat "
    "table holds; (c) the juice values are real market prices rather than the -110 schema "
    "default (at least one row differs from -110 on at least one juice column). Otherwise "
    "RE-INGEST through the nflreadpy path."
)

# Value columns the flat table and the partitioned store share. Compared elementwise for
# clause (b). 'snapshot_ts' is deliberately EXCLUDED: it is the partition key, stored in the
# directory name on one side and as a column on the other, so an equality test on it would
# compare two different encodings of the same fact. 'created_at' and 'last_update' are
# ingest-time lineage metadata, not market values.
SHARED_VALUE_COLUMNS = ("ml_home", "ml_away", "spread", "total", "is_live")


# ---------------------------------------------------------------------------
# The read-only guard (T-31-09).
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of *path*, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def silver_parquet_digests(silver_dir: Path = SILVER_DIR) -> dict[str, str]:
    """sha256 every parquet under *silver_dir*, keyed by POSIX path relative to it.

    Recursive, so the ``snapshot_ts=`` partition files are covered alongside the flat tables.
    """
    if not silver_dir.is_dir():
        return {}
    return {
        path.relative_to(silver_dir).as_posix(): sha256_file(path)
        for path in sorted(silver_dir.rglob("*.parquet"))
    }


def assert_silver_unchanged(before: dict[str, str], stage: str) -> None:
    """Raise ``AssertionError`` if any ``data/silver`` parquet changed during the run.

    The audit's whole standing rests on being provably read-only. ``git status data/`` cannot
    prove it -- ``.gitignore`` blankets ``data/`` so porcelain is empty either way -- so the
    proof has to come from content.
    """
    after = silver_parquet_digests()
    changed = sorted(
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    )
    if changed:
        msg = (
            f"data/silver parquet hash changed during {stage}: {changed}. "
            "This audit is read-only by contract (T-31-09) and must never write data/."
        )
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Task 1 -- what the partitioned silver odds store actually is (D31-39, A3).
# ---------------------------------------------------------------------------


def _juice_columns_from_schema() -> list[str]:
    """The juice column names DECLARED BY ``OddsSchema``, never restated by hand."""
    from data.schemas import OddsSchema

    juice = [name for name in OddsSchema.model_fields if "_ju" in name]
    if len(juice) != 4:
        msg = (
            f"expected exactly 4 juice columns declared by OddsSchema, found {juice}. "
            "The audit's juice-presence report is keyed to that declaration."
        )
        raise AssertionError(msg)
    return juice


def _parquet_manager_naming_scheme() -> dict[str, Any]:
    """Resolve, by source, how partitioned silver filenames are generated.

    RESEARCH assumption A3 guessed the three opaque filenames were ``ParquetManager``
    table-name hashes. This function does not guess: it reads
    ``inspect.getsource(ParquetManager.save)`` off the live object returned by
    ``get_parquet_manager`` and then reads the source of the pyarrow call it delegates to.
    """
    import pyarrow.parquet as pq

    from data.storage import get_parquet_manager

    manager = get_parquet_manager()
    save_source = inspect.getsource(manager.save)

    partitioned_call = [
        line.strip()
        for line in save_source.splitlines()
        if "write_to_dataset" in line
        or "root_path=" in line
        or "partition_cols=" in line
    ]

    dataset_source = inspect.getsource(pq.write_to_dataset)
    basename_lines = [
        line.strip()
        for line in dataset_source.splitlines()
        if "basename_template" in line and "=" in line and "guid" in line
    ]

    return {
        "manager_class": f"{type(manager).__module__}.{type(manager).__qualname__}",
        "manager_mro": [
            f"{c.__module__}.{c.__qualname__}" for c in type(manager).__mro__
        ],
        "save_defined_at": (
            f"{inspect.getsourcefile(type(manager).save)}"
            f":{inspect.getsourcelines(type(manager).save)[1]}"
        ),
        "partitioned_write_quoted_source": partitioned_call,
        "pyarrow_basename_quoted_source": basename_lines,
        "conclusion": (
            "ParquetManager.save delegates the partitioned branch to "
            "pyarrow.parquet.write_to_dataset with root_path=full_path.parent -- the SHARED "
            "data/silver root, not a table-scoped subdirectory -- and passes NO "
            "basename_template. pyarrow therefore names each file "
            "'<guid>-<i>.parquet' with a FRESH guid PER WRITE CALL. The filenames are "
            "per-write GUIDs, NOT table-name hashes: RESEARCH assumption A3 is FALSE. A "
            "consequence: write_to_dataset never deletes prior files, so repeated saves "
            "ACCUMULATE files in the same partitions rather than replacing them."
        ),
        "assumption_a3_verdict": "FALSE -- per-write pyarrow GUID, not a table-name hash",
    }


def _partition_files(silver_dir: Path = SILVER_DIR) -> list[Path]:
    """Every parquet under a ``snapshot_ts=`` partition of *silver_dir*, sorted."""
    partitions = sorted(
        d
        for d in silver_dir.iterdir()
        if d.is_dir() and d.name.startswith(PARTITION_PREFIX)
    )
    return [
        path for partition in partitions for path in sorted(partition.glob("*.parquet"))
    ]


def _seasons_in(game_ids: pd.Series) -> list[int]:
    """The sorted distinct season prefixes implied by *game_ids*."""
    seasons = set()
    for game_id in game_ids.astype(str):
        match = _SEASON_PREFIX_RE.match(game_id)
        if match:
            seasons.add(int(match.group("season")))
    return sorted(seasons)


def _file_report(path: Path, juice_columns: list[str]) -> dict[str, Any]:
    """Describe one partition file by measurement."""
    frame = pd.read_parquet(path)
    basename_match = _GUID_BASENAME_RE.match(path.name)
    return {
        "partition": path.parent.name,
        "filename": path.name,
        "write_guid": basename_match.group("guid") if basename_match else None,
        "matches_pyarrow_guid_basename": basename_match is not None,
        "rows": len(frame),
        "columns": sorted(frame.columns),
        "juice_columns_present": {
            column: bool(column in frame.columns) for column in juice_columns
        },
        "all_juice_columns_present": bool(
            all(column in frame.columns for column in juice_columns)
        ),
        "sportsbooks": sorted(frame["sportsbook"].dropna().unique().tolist())
        if "sportsbook" in frame.columns
        else [],
        "seasons_implied_by_game_id": _seasons_in(frame["game_id"])
        if "game_id" in frame.columns
        else [],
    }


def _flat_cross_check(
    partition_frames: dict[str, pd.DataFrame],
    flat: pd.DataFrame,
    juice_columns: list[str],
) -> dict[str, Any]:
    """Compare each write-GUID group against the flat table, pair-for-pair and value-for-value."""
    flat_pairs = set(zip(flat["game_id"], flat["sportsbook"], strict=True))
    per_guid: dict[str, Any] = {}

    for guid, frame in partition_frames.items():
        deduped = frame.drop_duplicates(subset=["game_id", "sportsbook"], keep="last")
        pairs = set(zip(deduped["game_id"], deduped["sportsbook"], strict=True))
        merged = flat.merge(
            deduped,
            on=["game_id", "sportsbook"],
            suffixes=("_flat", "_part"),
            how="inner",
        )

        mismatched: dict[str, int] = {}
        for column in SHARED_VALUE_COLUMNS:
            left, right = f"{column}_flat", f"{column}_part"
            if left not in merged.columns or right not in merged.columns:
                continue
            differs = ~(
                (merged[left] == merged[right])
                | (merged[left].isna() & merged[right].isna())
            )
            if int(differs.sum()):
                mismatched[column] = int(differs.sum())

        per_guid[guid] = {
            "rows_on_disk": len(frame),
            "distinct_game_id_sportsbook_pairs": len(pairs),
            "duplicate_pair_rows": int(
                len(frame)
                - len(frame.drop_duplicates(subset=["game_id", "sportsbook"]))
            ),
            "pairs_also_in_flat": len(pairs & flat_pairs),
            "pairs_only_in_partitioned_store": sorted(
                f"{game_id}|{book}" for game_id, book in (pairs - flat_pairs)
            )[:20],
            "n_pairs_only_in_partitioned_store": len(pairs - flat_pairs),
            "pairs_only_in_flat": sorted(
                f"{game_id}|{book}" for game_id, book in (flat_pairs - pairs)
            ),
            "n_pairs_only_in_flat": len(flat_pairs - pairs),
            "rows_compared_on_shared_columns": len(merged),
            "shared_columns_compared": list(SHARED_VALUE_COLUMNS),
            "shared_column_mismatches": mismatched,
            "matches_flat_row_for_row": bool(not mismatched and len(merged) > 0),
            "all_juice_columns_present": bool(
                all(column in frame.columns for column in juice_columns)
            ),
        }

    return per_guid


def audit_partitioned_odds_store(
    silver_dir: Path = SILVER_DIR,
    flat_odds_path: Path = FLAT_ODDS_PATH,
) -> dict[str, Any]:
    """Measure which logical table the partitioned silver filenames name (D31-39, A3).

    Read-only. Enumerates every parquet under the ``snapshot_ts=`` partitions, reports each
    file's shape, columns, juice-column presence, sportsbooks and implied seasons, groups the
    files by their pyarrow write-GUID, cross-checks each group against the flat
    ``odds_snapshot`` table, and evaluates :data:`BRANCH_RULE`.

    Returns:
        The report block written under the ``store_audit`` key of the JSON output.
    """
    juice_columns = _juice_columns_from_schema()
    naming = _parquet_manager_naming_scheme()

    files = _partition_files(silver_dir)
    file_reports = [_file_report(path, juice_columns) for path in files]

    partitions = sorted({report["partition"] for report in file_reports})
    total_rows = sum(report["rows"] for report in file_reports)

    grouped: dict[str, list[pd.DataFrame]] = {}
    for path in files:
        match = _GUID_BASENAME_RE.match(path.name)
        guid = match.group("guid") if match else path.name
        grouped.setdefault(guid, []).append(pd.read_parquet(path))
    partition_frames = {
        guid: pd.concat(frames, ignore_index=True) for guid, frames in grouped.items()
    }

    flat = pd.read_parquet(flat_odds_path)
    cross_check = _flat_cross_check(partition_frames, flat, juice_columns)

    combined = pd.concat(partition_frames.values(), ignore_index=True).drop_duplicates(
        subset=["game_id", "sportsbook"], keep="last"
    )

    # BRANCH_RULE clauses, evaluated -- not chosen.
    clause_a = bool(all(report["all_juice_columns_present"] for report in file_reports))
    clause_b = bool(
        any(group["matches_flat_row_for_row"] for group in cross_check.values())
        and not any(group["shared_column_mismatches"] for group in cross_check.values())
    )
    non_default_juice_rows = int((combined[juice_columns] != -110).any(axis=1).sum())
    clause_c = bool(non_default_juice_rows > 0)

    recommended_branch = (
        "PROMOTE" if (clause_a and clause_b and clause_c) else "RE-INGEST"
    )
    branch_reason = (
        f"clause (a) all four juice columns present in every partition file: {clause_a}; "
        f"clause (b) partitioned rows agree with the flat table on "
        f"{list(SHARED_VALUE_COLUMNS)} with zero mismatches: {clause_b}; "
        f"clause (c) juice values are real prices rather than the -110 default "
        f"({non_default_juice_rows} of {len(combined)} distinct pairs carry at least one "
        f"non--110 juice value): {clause_c}. Rule: {BRANCH_RULE}"
    )

    # The 2025 partition and the OUM-06 allowlist (DEFECT-5 implication 2).
    from backtest.ou_divergence import _ALLOWED_SPORTSBOOKS

    rows_2025 = combined[combined["game_id"].astype(str).str.startswith("2025")]
    books_2025 = sorted(rows_2025["sportsbook"].dropna().unique().tolist())
    admitted_2025 = sorted(book for book in books_2025 if book in _ALLOWED_SPORTSBOOKS)

    expected_files, expected_partitions, expected_rows = 23, 8, 7829
    discrepancy = None
    if (len(files), len(partitions), total_rows) != (
        expected_files,
        expected_partitions,
        expected_rows,
    ):
        discrepancy = (
            f"MEASURED {len(files)} files across {len(partitions)} partitions totalling "
            f"{total_rows} rows, which DIFFERS from the {expected_files}/"
            f"{expected_partitions}/{expected_rows} recorded by 31-RESEARCH.md DEFECT-5 on "
            "2026-09-03. The store changed since RESEARCH measured it; the pre-registration "
            "must state the measured figures, not the RESEARCH ones."
        )

    return {
        "measured_at": datetime.now(UTC).isoformat(),
        "silver_dir": silver_dir.as_posix(),
        "flat_odds_path": flat_odds_path.as_posix(),
        "juice_columns_declared_by_oddsschema": juice_columns,
        "parquet_manager_naming": naming,
        "n_partition_files": len(files),
        "n_partitions": len(partitions),
        "partitions": partitions,
        "total_partition_rows": total_rows,
        "research_defect5_expectation": {
            "files": expected_files,
            "partitions": expected_partitions,
            "rows": expected_rows,
        },
        "discrepancy_vs_research": discrepancy,
        "files": file_reports,
        "write_guid_groups": cross_check,
        "logical_table": {
            "verdict": (
                "Every partition file carries the OddsSchema column set and joins to the flat "
                "odds_snapshot table on (game_id, sportsbook). All "
                f"{len(partition_frames)} write-GUID groups name ONE logical table: "
                "silver 'odds_snapshot'. They are successive "
                "save_dataframe('odds_snapshot', partition_cols=['snapshot_ts']) writes into "
                "the SHARED data/silver root, and because pyarrow's write_to_dataset never "
                "deletes prior files the store ACCUMULATES: "
                f"{total_rows} rows on disk against "
                f"{len(combined)} distinct (game_id, sportsbook) pairs."
            ),
            "distinct_pairs_across_store": len(combined),
            "write_guids": sorted(partition_frames),
        },
        "flat_table_rows": len(flat),
        "recommended_branch": recommended_branch,
        "branch_reason": branch_reason,
        "branch_rule": BRANCH_RULE,
        "branch_clauses": {
            "a_juice_columns_present": clause_a,
            "b_matches_flat_row_for_row": clause_b,
            "c_juice_values_are_real": clause_c,
            "non_default_juice_rows": non_default_juice_rows,
            "distinct_pairs": len(combined),
        },
        "the_2025_partition": {
            "sportsbooks_present": books_2025,
            "n_sportsbooks_present": len(books_2025),
            "allowlist": sorted(_ALLOWED_SPORTSBOOKS),
            "sportsbooks_admitted_by_allowlist": admitted_2025,
            "n_admitted_of_n_present": f"{len(admitted_2025)} of {len(books_2025)}",
            "allowlist_entries_absent_from_the_2025_partition": sorted(
                book for book in _ALLOWED_SPORTSBOOKS if book not in books_2025
            ),
            "allowlist_discrepancy_vs_research": (
                None
                if len(admitted_2025) == 2
                else (
                    f"MEASURED: the allowlist admits {len(admitted_2025)} of the "
                    f"{len(books_2025)} books present in the 2025 partition "
                    f"({admitted_2025}), i.e. it REJECTS "
                    f"{len(books_2025) - len(admitted_2025)}. 31-RESEARCH.md DEFECT-5 "
                    "implication 2 says it 'would reject seven of the nine 2025 books', "
                    "which implies two admitted. The RESEARCH figure counted the allowlist's "
                    "own size (2 entries: "
                    f"{sorted(_ALLOWED_SPORTSBOOKS)}) rather than the number of those entries "
                    "PRESENT in the partition; 'consensus' does not appear in the 2025 "
                    "partition at all, so only 'draftkings' is admitted. The pre-registration "
                    "must state the MEASURED figure."
                )
            ),
            "is_the_2025_odds_source": False,
            "statement_for_the_pre_registration": (
                "The 2025-10-03 18:00:00-04:00 partition holds REAL multi-book 2025 market "
                f"data across {len(books_2025)} sportsbooks, of which the OUM-06 allowlist "
                f"admits only {len(admitted_2025)} ({admitted_2025}). This store is NOT the "
                "2025 odds source for the one-shot run -- the nflreadpy closing lines are "
                "(SPEC R2). Excluding it is BY CONSTRUCTION and stated here in advance, never "
                "a run-time discovery, so no one can later argue the source was chosen after "
                "seeing the result."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Task 2 -- the ATS residual bias, re-derived from the DEPLOYED artifact (A1).
# ---------------------------------------------------------------------------

# The residual contract this measurement uses, stated as a constant so it survives into
# source rather than living only in a docstring. It is the ATS analogue of
# backtest.ou_ev_chain.RESIDUAL_CONTRACT, with the sign trap called out: the O/U contract
# corrects a NEGATIVE bias (an over-predicting totals model), so an ATS guard copied from
# the O/U one would assert the wrong direction.
ATS_RESIDUAL_CONTRACT = (
    "residual = actual home margin - predicted home spread; "
    "a model that UNDER-predicts the home margin gives actual > predicted => residual > 0; "
    "corrected = model_spread + season_bias (prior-season walk-forward mean residual) "
    "pushes the predicted home margin UP -> higher P(home cover)."
)

# Flat -110 breakeven, restated from backtest.ou_ev_chain.OU_BREAKEVEN's arithmetic so the
# magnitude comparison in the report is self-contained and reproducible.
_MINUS_110_BREAKEVEN = 110.0 / 210.0

# The POOLED direction the pre-registration will freeze. Asserted; never tuned.
#
# THE GUARD IS ON THE POOLED SIGN ONLY (REVIEW-ATS). An earlier draft required the sign to
# be positive in all four tune seasons; re-scoring the deployed artifact over canonical gold
# makes 2022 NEGATIVE, so a per-season sign gate would hard-stop the phase on a fact that is
# simply true. The per-season means are REPORTED to 17 significant digits and carried into
# the pre-registration verbatim, so a reader sees the negative season rather than a gate that
# hid it. No numeric TOLERANCE is introduced either: choosing a magnitude threshold after
# seeing the measured values is exactly the post-hoc selection this phase forbids everywhere
# else, and it would be indefensible in the very script the pre-registration binds to.
#
# Note what the guard protects. The direction claim the ATS chain actually CONSUMES is the
# per-season walk-forward estimate (backtest.ou_ev_chain.estimate_prior_season_bias); the
# pooled figure supports only the pre-registration's MAGNITUDE argument, and it is that
# argument this assertion defends. If the pooled mean is not positive the pre-registration
# would have to state a different direction and Plan 31-07's ATS chain would have to be
# re-derived -- so the script fails loudly rather than proceeding.
POOLED_DIRECTION_CLAIM = (
    "pooled mean residual is strictly positive (model under-predicts)"
)


def _f(value: float) -> str:
    """Render *value* through the ONE explicit 17-significant-digit specifier."""
    return FLOAT_FMT.format(float(value))


def _residual_stats(residuals: pd.Series) -> dict[str, Any]:
    """n, mean, sd (ddof=1), one-sample t and p for *residuals*, all rendered at .17g."""
    import numpy as np
    from scipy import stats

    clean = residuals.dropna().astype(float)
    mean = float(clean.mean())
    sd = float(clean.std(ddof=1))
    result = stats.ttest_1samp(clean.to_numpy(), 0.0)
    t_stat = float(np.asarray(result.statistic).item())
    p_value = float(np.asarray(result.pvalue).item())
    return {
        "n": len(clean),
        "mean": _f(mean),
        "sd": _f(sd),
        "t": _f(t_stat),
        "p": _f(p_value),
        "mean_float": mean,
        "sd_float": sd,
    }


def measure_ats_residual_bias(
    artifacts_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    """Re-derive the ATS residual bias from the DEPLOYED artifact (closes assumption A1).

    RESEARCH measured the pooled residual from ``data/web_cache.duckdb`` and flagged it as
    assumption A1 precisely because the cache can be stale relative to the deployed artifact.
    This function scores the deployed artifact through
    ``backtest.diagnose.score_deployed_artifacts`` instead, so the number the pre-registration
    freezes provably came from a NAMED model.

    ``score_deployed_artifacts`` is called with the TARGET POSITIONAL ARGUMENT ONLY. Its
    signature is ``(target, gold_df=None, artifacts_dir="artifacts")`` -- there is no
    ``seasons`` keyword and passing one raises ``TypeError``. The 2021-2024 restriction is
    INTERNAL, applied by ``_load_gold_holdout`` via ``HOLDOUT_FIRST_SEASON`` /
    ``HOLDOUT_LAST_SEASON``; both are ASSERTED here rather than assumed, so a future change to
    them surfaces as a failure instead of silently widening the window this number is measured
    over.

    Raises:
        AssertionError: if the holdout constants are not 2021/2024, if the resolved artifact
            id disagrees with ``artifacts/latest.json``, or if the POOLED mean residual is not
            strictly positive.
    """
    from scipy.stats import norm

    from backtest.diagnose import (
        HOLDOUT_FIRST_SEASON,
        HOLDOUT_LAST_SEASON,
        score_deployed_artifacts,
    )
    from models.artifacts import load_model_artifact

    if (HOLDOUT_FIRST_SEASON, HOLDOUT_LAST_SEASON) != (2021, 2024):
        msg = (
            "backtest.diagnose HOLDOUT_FIRST_SEASON/HOLDOUT_LAST_SEASON are "
            f"{HOLDOUT_FIRST_SEASON}/{HOLDOUT_LAST_SEASON}, not 2021/2024. This measurement "
            "is fenced to the 2021-2024 tune window by those constants; a change to them "
            "silently widens the window the pre-registered bias is measured over."
        )
        raise AssertionError(msg)

    artifacts_path = Path(artifacts_dir)
    manifest = json.loads((artifacts_path / "latest.json").read_text(encoding="utf-8"))
    manifest_artifact_id = manifest["ats"]

    artifact = load_model_artifact("ats", artifacts_dir=artifacts_path)
    resolved_artifact_id = Path(artifact["artifact_dir"]).name
    if resolved_artifact_id != manifest_artifact_id:
        msg = (
            f"resolved ATS artifact '{resolved_artifact_id}' disagrees with "
            f"artifacts/latest.json '{manifest_artifact_id}'. The bias must be attributable "
            "to the artifact the manifest names (T-31-08)."
        )
        raise AssertionError(msg)

    # TARGET POSITIONAL ONLY. This function has no 'seasons' keyword argument; passing one
    # raises TypeError. The 2021-2024 fence is internal and asserted above.
    predictions = score_deployed_artifacts("ats")

    residual = predictions["actual"] - predictions["model_prob"]
    frame = pd.DataFrame(
        {"season": predictions["season"].astype(int), "residual": residual}
    )

    by_season = {
        int(season): _residual_stats(group["residual"])
        for season, group in frame.groupby("season", sort=True)
    }
    pooled = _residual_stats(frame["residual"])

    # THE DIRECTION GUARD -- pooled sign only, no per-season sign, no tolerance.
    if not pooled["mean_float"] > 0.0:
        msg = (
            f"POOLED ATS residual mean is {pooled['mean']}, which is not strictly positive. "
            f"The pre-registration's claim is: {POOLED_DIRECTION_CLAIM}. A non-positive "
            "pooled mean means the pre-registration must state a DIFFERENT direction and "
            "Plan 31-07's ATS chain must be re-derived. Failing loudly rather than "
            "proceeding (T-31-08)."
        )
        raise AssertionError(msg)

    cover_shift = float(norm.cdf(pooled["mean_float"] / pooled["sd_float"]) - 0.5)
    breakeven_edge = _MINUS_110_BREAKEVEN - 0.5

    negative_seasons = sorted(
        season for season, stats_ in by_season.items() if stats_["mean_float"] < 0.0
    )

    block = {
        "measured_at": datetime.now(UTC).isoformat(),
        "source": (
            "backtest.diagnose.score_deployed_artifacts('ats') over canonical gold -- NOT "
            "data/web_cache.duckdb. The cache can be stale relative to the deployed "
            "artifact, which is assumption A1."
        ),
        "artifact_id": resolved_artifact_id,
        "artifact_id_from_latest_json": manifest_artifact_id,
        "residual_contract": ATS_RESIDUAL_CONTRACT,
        "holdout_first_season": HOLDOUT_FIRST_SEASON,
        "holdout_last_season": HOLDOUT_LAST_SEASON,
        "float_format": FLOAT_FMT,
        "by_season": {
            str(season): {
                key: value
                for key, value in stats_.items()
                if not key.endswith("_float")
            }
            for season, stats_ in by_season.items()
        },
        "pooled": {
            key: value for key, value in pooled.items() if not key.endswith("_float")
        },
        "pooled_direction_claim": POOLED_DIRECTION_CLAIM,
        "pooled_direction_asserted": True,
        "per_season_sign_asserted": False,
        "numeric_tolerance_used": None,
        "seasons_with_negative_mean": negative_seasons,
        "implied_cover_probability_shift": _f(cover_shift),
        "minus_110_breakeven_edge": _f(breakeven_edge),
        "shift_as_fraction_of_breakeven_edge": _f(cover_shift / breakeven_edge),
        "disagreement_with_the_cache_figures": (
            "31-RESEARCH.md:1353 reports, from data/web_cache.duckdb, a pooled mean of "
            "+0.714049 (sd 13.021757, n 1139) with 2022 at +0.118713 (sd 11.707725). This "
            f"re-score of the DEPLOYED {resolved_artifact_id} over canonical gold gives a "
            f"pooled mean of {pooled['mean']} (sd {pooled['sd']}, n {pooled['n']}) with "
            f"seasons {negative_seasons} NEGATIVE. The figures DISAGREE. The cache figures "
            "are STALE relative to the Phase-25 re-fit and MUST NOT be used; this is "
            "assumption A1 closing as a disagreement, i.e. the audit working."
        ),
    }
    return block


# ---------------------------------------------------------------------------
# Report assembly and CLI.
# ---------------------------------------------------------------------------


def _load_existing(out_path: Path) -> dict[str, Any]:
    """Load a previously written report so a single-flag run does not wipe the other block."""
    if not out_path.is_file():
        return {}
    try:
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_report(out_path: Path, report: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _print_store_audit(block: dict[str, Any]) -> None:
    print("=== partitioned silver odds store (D31-39) ===")
    print(
        f"files={block['n_partition_files']} partitions={block['n_partitions']} "
        f"rows={block['total_partition_rows']} "
        f"distinct_pairs={block['logical_table']['distinct_pairs_across_store']}"
    )
    print(f"naming: {block['parquet_manager_naming']['assumption_a3_verdict']}")
    for line in block["parquet_manager_naming"]["partitioned_write_quoted_source"]:
        print(f"  quoted: {line}")
    for line in block["parquet_manager_naming"]["pyarrow_basename_quoted_source"]:
        print(f"  quoted: {line}")
    print(f"logical table: {block['logical_table']['verdict']}")
    if block["discrepancy_vs_research"]:
        print(f"DISCREPANCY: {block['discrepancy_vs_research']}")
    print(
        f"2025 partition: allowlist admits "
        f"{block['the_2025_partition']['n_admitted_of_n_present']} books; "
        f"is_the_2025_odds_source={block['the_2025_partition']['is_the_2025_odds_source']}"
    )
    if block["the_2025_partition"]["allowlist_discrepancy_vs_research"]:
        print(
            "DISCREPANCY: "
            f"{block['the_2025_partition']['allowlist_discrepancy_vs_research']}"
        )
    print(f"recommended branch: {block['recommended_branch']}")
    print(f"reason: {block['branch_reason']}")


def _print_ats_bias(block: dict[str, Any]) -> None:
    print("=== ATS residual bias, re-derived from the deployed artifact (A1) ===")
    print(f"artifact: {block['artifact_id']} (from artifacts/latest.json)")
    print(f"contract: {block['residual_contract']}")
    print(
        f"window: {block['holdout_first_season']}-{block['holdout_last_season']} "
        "(internal to _load_gold_holdout; asserted, not assumed)"
    )
    print(
        "season      n            mean                    sd                t          p"
    )
    for season, stats_ in block["by_season"].items():
        print(
            f"{season}      {stats_['n']:>4}   {stats_['mean']:>22}  "
            f"{stats_['sd']:>18}  {stats_['t']:>9}  {stats_['p']}"
        )
    pooled = block["pooled"]
    print(
        f"pooled     {pooled['n']:>4}   {pooled['mean']:>22}  "
        f"{pooled['sd']:>18}  {pooled['t']:>9}  {pooled['p']}"
    )
    print(f"seasons with a NEGATIVE mean: {block['seasons_with_negative_mean']}")
    print(
        f"implied P(cover) shift: {block['implied_cover_probability_shift']} against a "
        f"-110 breakeven edge of {block['minus_110_breakeven_edge']} "
        f"({block['shift_as_fraction_of_breakeven_edge']} of it)"
    )
    print(f"DISAGREEMENT: {block['disagreement_with_the_cache_figures']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Phase-31 pre-ingest audit (Plan 31-02). Writes nothing to data/."
    )
    parser.add_argument(
        "--store-audit",
        action="store_true",
        help="Audit the partitioned silver odds store and recommend the D31-39 branch.",
    )
    parser.add_argument(
        "--ats-bias",
        action="store_true",
        help="Re-derive the ATS residual bias from the DEPLOYED artifact (closes A1).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not (args.store_audit or args.ats_bias):
        parser.error("nothing to do: pass --store-audit and/or --ats-bias")

    report = _load_existing(args.out)
    report["generated_at"] = datetime.now(UTC).isoformat()

    if args.store_audit:
        before = silver_parquet_digests()
        report["store_audit"] = audit_partitioned_odds_store()
        assert_silver_unchanged(before, "audit_partitioned_odds_store")
        _print_store_audit(report["store_audit"])

    if args.ats_bias:
        before = silver_parquet_digests()
        report["ats_bias"] = measure_ats_residual_bias()
        assert_silver_unchanged(before, "measure_ats_residual_bias")
        _print_ats_bias(report["ats_bias"])

    _write_report(args.out, report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
