"""Proof the Stage-2 tuned path really tuned, and the windows were derived not typed.

Plan 30-01 (PROD-01). Two RESEARCH Pitfall-2 failure modes and one D30-12 asymmetry are
pinned here, because each of them fails SILENTLY -- the run completes, an artifact appears,
the gate prints a 2x2, and the number is wrong:

  * A VACUOUS RE-TUNE (T-30-02). ``OptunaTuner.optimize`` computes remaining trials as
    ``max(0, n_trials - len(study.trials))`` under ``load_if_exists=True``, so resuming a
    study that already holds the full budget runs ZERO new trials while still reporting a
    full trial count. A Phase-30 "tuned" candidate would then be carrying v2.0 parameters.
    Closed by a per-phase ``TUNING_STUDY_TAG`` plus a hard RuntimeError on zero new trials.
    The BACKTEST hit the same failure on the same mechanism and is closed separately, by a
    PER-RUN identity with its own proofs in tests/unit/test_backtest_tuning_identity.py
    (Plan 30-16, D30-OWNER-02). This module owns the Stage-2 identity and the invariant that
    neither non-legacy identity is ever the default.
  * A WRITE UNDER ``data/`` (T-30-14). ``OptunaTuner`` defaults ``storage_dir`` to
    ``data/optuna``; this phase's prohibition forbids writing under ``data/`` outside the one
    sanctioned fingerprinted rebuild, and the prohibition's own hash manifest reads through
    ``load_dataframe`` and would NOT have caught a ``data/optuna/`` write. Asserted directly.
  * THE D30-12 WINDOW ASYMMETRY. The deployed ATS incumbent is the D25-05 fix-cycle artifact
    and records a FIVE-season train window, while ``promote_models`` trained every candidate
    on the two-season default -- so the ATS candidate faced its own incumbent from a strictly
    worse configuration, a handicap nobody chose. ``_incumbent_window`` derives each target's
    window from that target's own metadata; a silent fallback to
    ``TemporalSplitConfig.default()`` would reintroduce exactly the bug it exists to fix.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest

from models.trainers import base as base_trainer
from models.trainers.base import (
    TUNING_STORAGE_DIR,
    TUNING_STUDY_TAG,
    BaseTrainer,
    _existing_trial_count,
)
from models.tuning import OptunaTuner
from scripts import promote_models as promote

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_ARTIFACTS = REPO_ROOT / "artifacts"

# The window the D25-05 ATS fix-cycle artifact records. Asserted BY NAME because this exact
# asymmetry is the finding D30-12 exists to correct: a regression here silently hands the ATS
# candidate a worse configuration than its own incumbent, and the gate would still print a 2x2.
_ATS_FIX_CYCLE_TRAIN_WINDOW = "2015,2016,2017,2018,2019"
_DEFAULT_TRAIN_WINDOW = "2018,2019"


# ---------------------------------------------------------------------------
# tmp_path artifacts-tree helpers (the failure branches are driven from these,
# never by damaging the live artifacts/ tree -- it is the rollback target, D25-17)
# ---------------------------------------------------------------------------


def _write_artifacts_tree(
    root: Path,
    manifest: dict[str, str] | None,
    metadata: dict[str, dict[str, Any] | None],
) -> Path:
    """Materialize a throwaway artifacts tree under ``root``.

    Args:
        root: The tmp artifacts dir to create.
        manifest: The ``latest.json`` payload, or None to omit the manifest entirely.
        metadata: ``{version_dir: metadata_payload_or_None}``. A None payload creates the
            version dir WITHOUT a metadata.json.

    Returns:
        The created artifacts root.
    """
    root.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (root / "latest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for version, payload in metadata.items():
        version_dir = root / version
        version_dir.mkdir(parents=True, exist_ok=True)
        if payload is not None:
            (version_dir / "metadata.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
    return root


def _sha256(path: Path) -> str:
    """Return the sha256 of a file (byte-identity, not just size+mtime)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _good_metadata(train: list[int]) -> dict[str, Any]:
    """A metadata payload carrying a well-formed config block."""
    return {
        "target": "ats",
        "config": {
            "train_seasons": train,
            "hp_val_seasons": [2020],
            "holdout_seasons": [2021, 2022, 2023, 2024],
        },
    }


# ---------------------------------------------------------------------------
# D30-12: the window is DERIVED from each incumbent's own metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected_train"),
    [
        ("ats", _ATS_FIX_CYCLE_TRAIN_WINDOW),
        ("wp", _DEFAULT_TRAIN_WINDOW),
        ("ou", _DEFAULT_TRAIN_WINDOW),
    ],
)
def test_incumbent_window_derived_from_live_metadata(
    target: str, expected_train: str
) -> None:
    """Each target's selection window comes from ITS OWN deployed incumbent's metadata.

    Asserted against the LIVE ``artifacts/`` tree, not a copy of it, so the test proves
    D30-12's actual claim. Remediation if this goes red: do NOT edit the expected constant --
    inspect ``artifacts/{version}/metadata.json`` for that target and confirm whether the
    deployed incumbent genuinely changed. If it did, the recorded gate 2x2s were measured on
    the old window and must be re-measured.
    """
    if not (LIVE_ARTIFACTS / "latest.json").exists():
        pytest.skip(
            f"{LIVE_ARTIFACTS / 'latest.json'} not present -- bootstrap the production "
            "manifest per the clean-checkout step (RUNBOOK, Plan 25-05)"
        )

    window = promote._incumbent_window(target, LIVE_ARTIFACTS)

    assert window["train"] == expected_train, (
        f"_incumbent_window('{target}')['train'] is {window['train']!r}, expected "
        f"{expected_train!r}. Either the deployed incumbent changed (re-measure every "
        "recorded gate 2x2) or the derivation regressed to a hardcoded/default window."
    )
    assert window["hp_val"] == "2020", (
        f"_incumbent_window('{target}')['hp_val'] is {window['hp_val']!r}, expected "
        "'2020'. The hp-val fold must be derived from the incumbent, not typed."
    )
    assert window["holdout"] == "2021,2022,2023,2024", (
        f"_incumbent_window('{target}')['holdout'] is {window['holdout']!r}, expected "
        "'2021,2022,2023,2024' (the frozen gate.toml holdout window)."
    )


def test_ats_incumbent_window_is_not_the_default_window() -> None:
    """The ATS window must DIFFER from wp/ou -- that difference IS the D30-12 finding.

    Remediation if this goes red: a single global window has been reintroduced on the
    promote path. The per-target window derived from each incumbent's own metadata is
    primary; ``TemporalSplitConfig.default()`` is a fallback for callers that specify none,
    and is never reached here.
    """
    if not (LIVE_ARTIFACTS / "latest.json").exists():
        pytest.skip(f"{LIVE_ARTIFACTS / 'latest.json'} not present")

    ats = promote._incumbent_window("ats", LIVE_ARTIFACTS)["train"]
    wp = promote._incumbent_window("wp", LIVE_ARTIFACTS)["train"]

    assert ats != wp, (
        f"ATS and WP both resolve to train window {ats!r}. The D25-05 fix-cycle ATS "
        "artifact records a five-season window; identical windows mean the derivation "
        "collapsed to one global default (the exact D30-12 bug)."
    )


# ---------------------------------------------------------------------------
# The four failure branches -- informative, typed, and never a silent default
# ---------------------------------------------------------------------------


def test_missing_manifest_raises_filenotfound_naming_the_path(tmp_path: Path) -> None:
    """An absent ``latest.json`` names the exact missing path, never a bare traceback.

    Remediation if this goes red: restore the explicit existence check in
    ``_incumbent_window``; this function silently underwrites every gate comparison the
    phase records, so its failures must be self-explaining.
    """
    root = _write_artifacts_tree(tmp_path / "artifacts", manifest=None, metadata={})

    with pytest.raises(FileNotFoundError) as excinfo:
        promote._incumbent_window("ats", root)

    assert "latest.json" in str(excinfo.value), (
        f"The FileNotFoundError does not name latest.json: {excinfo.value!r}. The operator "
        "needs the offending path in the message."
    )


def test_absent_target_key_raises_keyerror_naming_the_target(tmp_path: Path) -> None:
    """A manifest with no pointer for the target raises KeyError naming that target.

    Remediation if this goes red: restore the ``version is None`` guard; falling through
    would resolve ``artifacts/None/metadata.json`` and produce a confusing path error.
    """
    root = _write_artifacts_tree(
        tmp_path / "artifacts",
        manifest={"wp": "wp_x", "blend": "blend_x"},
        metadata={"wp_x": _good_metadata([2018, 2019])},
    )

    with pytest.raises(KeyError) as excinfo:
        promote._incumbent_window("ats", root)

    assert "ats" in str(excinfo.value), (
        f"The KeyError does not name the missing target 'ats': {excinfo.value!r}."
    )


def test_missing_metadata_file_raises_filenotfound_naming_the_path(
    tmp_path: Path,
) -> None:
    """A version dir without ``metadata.json`` names that exact path.

    Remediation if this goes red: restore the metadata existence check; an opaque JSON
    decode error here is indistinguishable from a corrupt artifact.
    """
    root = _write_artifacts_tree(
        tmp_path / "artifacts",
        manifest={"ats": "ats_x"},
        metadata={"ats_x": None},
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        promote._incumbent_window("ats", root)

    assert "metadata.json" in str(excinfo.value), (
        f"The FileNotFoundError does not name metadata.json: {excinfo.value!r}."
    )
    assert "ats_x" in str(excinfo.value), (
        f"The FileNotFoundError does not name the version dir 'ats_x': {excinfo.value!r}."
    )


def test_metadata_without_config_block_raises_keyerror(tmp_path: Path) -> None:
    """Metadata lacking a ``config`` block raises KeyError naming the missing key.

    Remediation if this goes red: restore the config-block guard. A metadata file without a
    config block cannot supply a window, and inventing one is the silent-default bug.
    """
    root = _write_artifacts_tree(
        tmp_path / "artifacts",
        manifest={"ats": "ats_x"},
        metadata={"ats_x": {"target": "ats", "feature_names": []}},
    )

    with pytest.raises(KeyError) as excinfo:
        promote._incumbent_window("ats", root)

    assert "config" in str(excinfo.value), (
        f"The KeyError does not name the missing 'config' key: {excinfo.value!r}."
    )


def test_metadata_with_empty_season_list_raises_keyerror(tmp_path: Path) -> None:
    """A config block missing a season list raises rather than emitting an empty window.

    Remediation if this goes red: an empty window would reach models.train's
    ``[int(s) for s in "".split(",")]`` and raise deep inside a subprocess instead of here.
    """
    payload = _good_metadata([2018, 2019])
    del payload["config"]["hp_val_seasons"]
    root = _write_artifacts_tree(
        tmp_path / "artifacts",
        manifest={"ats": "ats_x"},
        metadata={"ats_x": payload},
    )

    with pytest.raises(KeyError) as excinfo:
        promote._incumbent_window("ats", root)

    assert "hp_val_seasons" in str(excinfo.value), (
        f"The KeyError does not name the missing season list: {excinfo.value!r}."
    )


@pytest.mark.parametrize(
    ("manifest", "metadata"),
    [
        (None, {}),
        ({"wp": "wp_x"}, {"wp_x": _good_metadata([2018, 2019])}),
        ({"ats": "ats_x"}, {"ats_x": None}),
        ({"ats": "ats_x"}, {"ats_x": {"target": "ats"}}),
    ],
    ids=["no-manifest", "no-target-key", "no-metadata-file", "no-config-block"],
)
def test_no_failure_branch_returns_a_default_window(
    tmp_path: Path,
    manifest: dict[str, str] | None,
    metadata: dict[str, dict[str, Any] | None],
) -> None:
    """NO branch of ``_incumbent_window`` may return a ``TemporalSplitConfig.default()`` window.

    A silent default is exactly the D30-12 asymmetry this function exists to eliminate: it
    would hand the ATS candidate a two-season window while its incumbent trained on five,
    and the gate would still print a confident 2x2. Remediation if this goes red: delete the
    fallback and raise -- an unresolvable window is a stop, not a guess.
    """
    root = _write_artifacts_tree(tmp_path / "artifacts", manifest, metadata)

    with pytest.raises((FileNotFoundError, KeyError)):
        result = promote._incumbent_window("ats", root)
        pytest.fail(
            f"_incumbent_window returned {result!r} instead of raising. A fallback window "
            "reintroduces the D30-12 asymmetry; remove it."
        )


# ---------------------------------------------------------------------------
# STEP 1 argv: per target, derived window, exclusion pass-through, tuning ON
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["wp", "ats", "ou"])
def test_step1_argv_carries_window_and_exclusion_and_enables_tuning(
    target: str, tmp_path: Path
) -> None:
    """The argv built for each target carries the derived window, the exclusion, and no untuned flag.

    SPEC R5 requires the Stage-2 candidate to be trained WITH Optuna tuning, so the
    ``--no-tune`` flag that STEP 1 carried through Phases 24-25 must be ABSENT.
    Remediation if this goes red: re-inspect ``_build_train_argv`` -- re-adding --no-tune
    silently downgrades every Stage-2 candidate to a straight re-fit.
    """
    window = {
        "train": _ATS_FIX_CYCLE_TRAIN_WINDOW,
        "hp_val": "2020",
        "holdout": "2021,2022,2023,2024",
    }
    argv = promote._build_train_argv(
        target, tmp_path / "staging", window, ["line_movement"]
    )

    assert "--no-tune" not in argv, (
        f"STEP 1 argv for '{target}' still carries --no-tune: {argv}. SPEC R5 requires a "
        "genuinely tuned Stage-2 candidate."
    )
    for flag, value in (
        ("--target", target),
        ("--config-train-seasons", window["train"]),
        ("--config-hp-val-seasons", window["hp_val"]),
        ("--config-holdout-seasons", window["holdout"]),
        ("--exclude-groups", "line_movement"),
    ):
        assert flag in argv, f"STEP 1 argv for '{target}' is missing {flag}: {argv}"
        assert argv[argv.index(flag) + 1] == value, (
            f"STEP 1 argv for '{target}' has {flag}="
            f"{argv[argv.index(flag) + 1]!r}, expected {value!r}. The window must be the "
            "DERIVED one, not a default."
        )
    assert "models.train" in argv, (
        f"STEP 1 argv for '{target}' does not invoke models.train: {argv}"
    )


def test_step1_argv_targets_the_staging_dir_never_production(tmp_path: Path) -> None:
    """The candidate trains into the STAGING dir; production is never the train target.

    Remediation if this goes red: --artifacts-dir must be the staging dir. Training into
    production would write a candidate beside the deployed incumbents (D24-08).
    """
    staging = tmp_path / "artifacts_staging"
    window = {"train": "2018,2019", "hp_val": "2020", "holdout": "2021,2022,2023,2024"}
    argv = promote._build_train_argv("wp", staging, window, [])

    assert argv[argv.index("--artifacts-dir") + 1] == str(staging), (
        f"STEP 1 argv does not point --artifacts-dir at the staging dir: {argv}"
    )


def test_step1_loop_covers_every_gated_target() -> None:
    """STEP 1 issues one subprocess PER TARGET -- the source no longer passes '--target all'.

    A single ``--target all`` invocation cannot carry three different per-target windows,
    which is why D30-12 requires the loop. Remediation if this goes red: restore the
    ``for target in _TARGETS`` loop around the subprocess call in main().
    """
    source = inspect.getsource(promote.main)

    assert '"all"' not in source, (
        "scripts/promote_models.py main() still passes '--target all' to models.train. "
        "One invocation cannot carry three different per-target selection windows (D30-12)."
    )
    assert "_build_train_argv" in source, (
        "main() does not build its train argv through _build_train_argv; the argv this "
        "module tests would then not be the argv that actually runs."
    )
    assert promote._TARGETS == ("wp", "ats", "ou"), (
        f"_TARGETS is {promote._TARGETS}; the STEP 1 loop must cover exactly the gated "
        "targets."
    )


# ---------------------------------------------------------------------------
# _resolve_exclude_groups: override -> ratified verdict -> empty + warning
# ---------------------------------------------------------------------------


def test_explicit_cli_exclusion_wins_and_is_flagged_as_an_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit --exclude-groups beats a ratified verdict file and reports 'override'.

    Remediation if this goes red: precedence must be CLI > verdict > empty. An override that
    silently reported 'verdict' provenance would let a hand-typed list masquerade as the
    ratified Stage-1 result in the checkpoint-4 banner.
    """
    verdict = tmp_path / "group_gate_verdict.toml"
    verdict.write_text('excluded_groups = ["injury"]\n', encoding="utf-8")
    monkeypatch.setattr(promote, "_GROUP_GATE_VERDICT_PATH", verdict)

    args = promote.parse_args(["--exclude-groups", "line_movement"])
    groups, provenance = promote._resolve_exclude_groups(args)

    assert groups == ["line_movement"], (
        f"The CLI override did not win: got {groups!r}. Precedence is CLI > verdict > empty."
    )
    assert provenance == "override", (
        f"Provenance is {provenance!r}, expected 'override'. The banner must say the list "
        "came from the command line and NOT from the ratified Stage-1 verdict."
    )


def test_ratified_verdict_file_is_used_when_no_cli_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no CLI value, the exclusion is DERIVED from the ratified verdict file (D24-07).

    Remediation if this goes red: the generator-output-over-transcription discipline is
    broken -- the list must be read from config/group_gate_verdict.toml, never re-typed.
    """
    verdict = tmp_path / "group_gate_verdict.toml"
    verdict.write_text(
        'excluded_groups = ["line_movement", "snap"]\n', encoding="utf-8"
    )
    monkeypatch.setattr(promote, "_GROUP_GATE_VERDICT_PATH", verdict)

    args = promote.parse_args([])
    groups, provenance = promote._resolve_exclude_groups(args)

    assert groups == ["line_movement", "snap"], (
        f"The verdict file was not read: got {groups!r}."
    )
    assert provenance == "verdict", f"Provenance is {provenance!r}, expected 'verdict'."


def test_no_cli_and_no_verdict_returns_empty_with_a_none_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With neither source present the list is EMPTY and the provenance says so loudly.

    Remediation if this goes red: never invent an exclusion list. An unratified phase must
    train on everything and say plainly that no Stage-1 verdict exists.
    """
    monkeypatch.setattr(promote, "_GROUP_GATE_VERDICT_PATH", tmp_path / "absent.toml")

    args = promote.parse_args([])
    groups, provenance = promote._resolve_exclude_groups(args)

    assert groups == [], f"Expected an empty exclusion list, got {groups!r}."
    assert provenance == "none", (
        f"Provenance is {provenance!r}, expected 'none' so the banner can warn that no "
        "Stage-1 verdict has been ratified."
    )


def test_empty_cli_value_is_an_override_to_exclude_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--exclude-groups ""`` is an explicit override meaning 'exclude nothing'.

    It must NOT silently fall through to the verdict file: the operator asked for an empty
    exclusion. Remediation if this goes red: branch on ``args.exclude_groups is not None``,
    not on truthiness.
    """
    verdict = tmp_path / "group_gate_verdict.toml"
    verdict.write_text('excluded_groups = ["injury"]\n', encoding="utf-8")
    monkeypatch.setattr(promote, "_GROUP_GATE_VERDICT_PATH", verdict)

    args = promote.parse_args(["--exclude-groups", ""])
    groups, provenance = promote._resolve_exclude_groups(args)

    assert groups == [], (
        f"An explicit empty --exclude-groups fell through to the verdict file: {groups!r}. "
        "Branch on 'is not None', not on truthiness."
    )
    assert provenance == "override", (
        f"Provenance is {provenance!r}, expected 'override'."
    )


# ---------------------------------------------------------------------------
# T-30-14: Optuna study storage lives OUTSIDE data/
# ---------------------------------------------------------------------------


def test_tuning_storage_dir_resolves_outside_data() -> None:
    """``TUNING_STORAGE_DIR`` is not inside the repository ``data/`` directory.

    ``OptunaTuner`` defaults to ``data/optuna``; this phase forbids writing under ``data/``
    outside the one sanctioned fingerprinted rebuild, and the prohibition's own before/after
    hash manifest reads through ``load_dataframe`` and would NOT have caught a
    ``data/optuna/`` write. Remediation if this goes red: point TUNING_STORAGE_DIR back
    outside data/ (outputs/ is gitignored and is the right home).
    """
    storage = Path(TUNING_STORAGE_DIR).resolve()
    data_root = (REPO_ROOT / "data").resolve()

    assert not storage.is_relative_to(data_root), (
        f"TUNING_STORAGE_DIR resolves to {storage}, which is inside {data_root}. A Stage-2 "
        "tuned train would write under data/, violating the Plan 30-01 prohibition."
    )


def test_tune_hyperparameters_passes_storage_dir_to_optuna_tuner() -> None:
    """The ``OptunaTuner(`` construction inside ``tune_hyperparameters`` passes ``storage_dir``.

    Source-level call-site guard (the tests/unit/test_build_features_gold_write.py shape):
    omitting the kwarg silently reverts to the ``data/optuna`` default, which no behavioural
    assertion on the constant alone would catch. Remediation if this goes red: pass
    ``storage_dir=TUNING_STORAGE_DIR`` into the OptunaTuner construction.
    """
    source = inspect.getsource(BaseTrainer.tune_hyperparameters)
    marker = "OptunaTuner("
    idx = source.find(marker)

    assert idx != -1, (
        "No OptunaTuner( construction found in BaseTrainer.tune_hyperparameters; the "
        "tuning seam moved and this guard no longer guards anything."
    )
    call_block = source[idx : source.find(")", idx) + 1]
    assert "storage_dir" in call_block, (
        f"The OptunaTuner construction does not pass storage_dir:\n{call_block}\n"
        "Without it the tuner defaults to data/optuna and writes under data/."
    )


def test_study_identity_differs_from_the_v2_studies() -> None:
    """The Phase-30 study name is not the v2.0 ``{target}_tuning_v1`` identity.

    Reusing the v2.0 identity is what makes a re-tune vacuous: the resumed study already
    holds the full trial budget, so ``max(0, n_trials - len(study.trials))`` is zero.
    Remediation if this goes red: bump TUNING_STUDY_TAG -- never delete the v2.0 study files,
    they are the historical record.
    """
    assert TUNING_STUDY_TAG != "v1", (
        "TUNING_STUDY_TAG is still 'v1'; a Phase-30 tuned train would resume the v2.0 study "
        "and run zero new trials while reporting a full trial count."
    )
    for target in ("wp", "ats", "ou"):
        name = f"{target}_tuning_{TUNING_STUDY_TAG}"
        assert name != f"{target}_tuning_v1", (
            f"Resolved study name {name!r} collides with the v2.0 identity."
        )


# ---------------------------------------------------------------------------
# T-30-02: a study already at budget must RAISE, not return stored parameters
# ---------------------------------------------------------------------------


class _DummyModel:
    """A trivial fitted-model stand-in; the objective is never invoked in these tests."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _DummyModel:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class _DummyTrainer(BaseTrainer):
    """Minimal concrete BaseTrainer so the tuning guard can be exercised hermetically."""

    def __init__(self) -> None:
        super().__init__(target="wp")

    def _create_model(self, params: dict) -> _DummyModel:
        return _DummyModel()

    def _get_target_column(self) -> str:
        return "home_win"

    def _predict_raw(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))

    def _get_default_params(self) -> dict:
        return {}

    def _define_search_space(self, trial: optuna.Trial) -> dict:
        return {"x": trial.suggest_float("x", 0.0, 1.0)}


def _prefill_study(storage_dir: Path, study_name: str, n_trials: int) -> None:
    """Create a study under ``storage_dir`` already holding ``n_trials`` completed trials."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    tuner = OptunaTuner(
        study_name=study_name, storage_dir=storage_dir, n_trials=n_trials
    )
    study = optuna.create_study(
        study_name=study_name, storage=tuner.storage_url, direction="minimize"
    )
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=n_trials)


def _training_frame(n_rows: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    """A frame large enough for make_temporal_cv_splits; the objective is never invoked."""
    return (
        pd.DataFrame({"f1": np.arange(n_rows, dtype=float)}),
        pd.Series(np.arange(n_rows) % 2),
    )


def test_zero_new_trials_raises_with_the_remediation_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For an OPTED-IN trainer, a study already at budget makes tune_hyperparameters RAISE.

    Without this guard a resumed full study returns the STORED (v2.0) best parameters while
    ``TuningResult.n_trials`` reports the full count -- a "tuned" Phase-30 candidate that was
    never tuned (T-30-02). Remediation if this goes red: restore the
    ``require_fresh_search and trials_added <= 0`` RuntimeError in tune_hyperparameters.
    """
    n_trials = 3
    monkeypatch.setattr(base_trainer, "TUNING_STORAGE_DIR", tmp_path / "optuna")
    study_name = f"wp_tuning_{TUNING_STUDY_TAG}"
    _prefill_study(tmp_path / "optuna", study_name, n_trials)

    trainer = _DummyTrainer()
    trainer.use_phase30_tuning()
    X_train, y_train = _training_frame()

    with pytest.raises(RuntimeError) as excinfo:
        trainer.tune_hyperparameters(X_train, y_train, n_trials=n_trials)

    message = str(excinfo.value)
    assert study_name in message, (
        f"The zero-new-trials RuntimeError does not name the study: {message!r}."
    )
    assert "TUNING_STUDY_TAG" in message, (
        f"The zero-new-trials RuntimeError does not name the remediation "
        f"(bump TUNING_STUDY_TAG): {message!r}."
    )


def test_use_phase30_tuning_switches_identity_storage_and_guard() -> None:
    """The opt-in flips all three: study tag, storage dir, and the freshness requirement.

    Remediation if this goes red: a partial opt-in is the worst outcome -- e.g. the fresh
    identity WITHOUT the guard silently changes what is searched while still allowing a
    vacuous resume later.
    """
    trainer = _DummyTrainer()
    assert trainer.tuning_study_tag == base_trainer.LEGACY_TUNING_STUDY_TAG
    assert trainer.tuning_storage_dir == base_trainer.LEGACY_TUNING_STORAGE_DIR
    assert trainer.require_fresh_search is False

    trainer.use_phase30_tuning()

    assert trainer.tuning_study_tag == TUNING_STUDY_TAG, (
        "use_phase30_tuning did not switch the study tag."
    )
    assert trainer.tuning_storage_dir == TUNING_STORAGE_DIR, (
        "use_phase30_tuning did not switch the storage dir; the Stage-2 search would write "
        "under data/optuna."
    )
    assert trainer.require_fresh_search is True, (
        "use_phase30_tuning did not arm the freshness guard; a vacuous resume would pass."
    )


def test_non_opted_in_trainer_keeps_the_legacy_identity_and_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trainer that did NOT opt in resumes its legacy study and does NOT raise.

    This is the SCOPING INVARIANT: no tuning identity is ever made the DEFAULT. The
    assertions below are unchanged since Plan 30-01; the REASON was reconciled by Plan 30-16
    (D30-OWNER-02) because the reason it originally gave has been superseded.

    IT USED TO SAY, and this is no longer the reason: that ``backtest.engine`` must keep the
    legacy identity, because a fresh study would make it search real trials and drift the
    frozen v2.1 AUDIT-REPORT anchors. The owner ruled the opposite -- D30-DEFER-01 Option 2 --
    so the backtest now opts into its own PER-RUN identity via
    ``BaseTrainer.use_backtest_tuning(run_id)``, and those anchors were re-ratified
    deliberately with the drift recorded (Plan 30-16 Task 3,
    tests/integration/test_diag_diagnosis.py).

    WHAT THE INVARIANT STILL PROTECTS, and why it is still worth a test. There are now THREE
    identities with three lifetimes: legacy (the default), per-phase (Stage 2), per-run (the
    backtest). Making ANY non-legacy identity the default would change what every caller that
    never opted in trains with, in one move -- ``scripts.retrain_models`` today, and whatever
    is added tomorrow. Remediation if this goes red: keep the identities as explicit opt-ins
    (``models.train.train_target`` and ``BacktestEngine._create_trainer``); do not move either
    into ``BaseTrainer.__init__``.
    """
    legacy_dir = tmp_path / "legacy_optuna"
    monkeypatch.setattr(base_trainer, "LEGACY_TUNING_STORAGE_DIR", legacy_dir)

    trainer = _DummyTrainer()
    # Re-point the instance the way __init__ would have on a fresh construction.
    trainer.tuning_storage_dir = legacy_dir
    n_trials = 3
    _prefill_study(legacy_dir, f"wp_tuning_{trainer.tuning_study_tag}", n_trials)

    X_train, y_train = _training_frame()
    # Must NOT raise: this is the backtest/retrain contract, unchanged from before Phase 30.
    best_params = trainer.tune_hyperparameters(X_train, y_train, n_trials=n_trials)

    assert "x" in best_params, (
        f"The legacy resume path returned {best_params!r}; it must still return the stored "
        "best parameters rather than raising (pre-Phase-30 behaviour, byte-for-byte)."
    )
    assert trainer.tuning_study_tag == base_trainer.LEGACY_TUNING_STUDY_TAG, (
        "A non-opted-in trainer must keep the legacy study identity."
    )


def test_train_target_opts_the_tuned_path_in() -> None:
    """``models.train.train_target`` arms the Phase-30 tuning on its tuned path.

    That call site IS the Stage-2 candidate train (promote STEP 1 shells out to
    ``python -m models.train``), so if the opt-in is dropped the Stage-2 search silently
    reverts to resuming the v2.0 study -- the exact T-30-02 failure. Remediation if this goes
    red: restore ``trainer.use_phase30_tuning()`` under ``if tune:`` in train_target.
    """
    import models.train as train_mod

    source = inspect.getsource(train_mod.train_target)
    assert "use_phase30_tuning()" in source, (
        "models.train.train_target no longer opts into the Phase-30 tuning identity; the "
        "Stage-2 candidate would resume the v2.0 study and never actually search."
    )


def test_existing_trial_count_is_zero_for_an_absent_study(tmp_path: Path) -> None:
    """``_existing_trial_count`` reports 0 when the study (or its file) does not exist.

    Remediation if this goes red: a spurious non-zero count would make a genuinely fresh
    search look resumed and could suppress the guard.
    """
    tuner = OptunaTuner(
        study_name="does_not_exist", storage_dir=tmp_path / "optuna", n_trials=5
    )
    assert _existing_trial_count(tuner) == 0, (
        "_existing_trial_count returned non-zero for an absent study."
    )


def test_existing_trial_count_reads_a_prefilled_study(tmp_path: Path) -> None:
    """``_existing_trial_count`` reports the stored trial count for an existing study.

    Remediation if this goes red: the guard's 'before' reading is wrong, so the new-trial
    delta it computes is wrong too.
    """
    _prefill_study(tmp_path / "optuna", "prefilled", 2)
    tuner = OptunaTuner(
        study_name="prefilled", storage_dir=tmp_path / "optuna", n_trials=2
    )
    assert _existing_trial_count(tuner) == 2, (
        f"_existing_trial_count returned {_existing_trial_count(tuner)}, expected 2."
    )


def test_v2_study_files_are_byte_untouched_across_a_stage2_tuned_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real opted-in tuned search leaves ``data/optuna/*_tuning_v1.db`` byte-unchanged.

    The Plan 30-01 prohibition is 'MUST NOT write under data/ outside the one sanctioned
    fingerprinted rebuild', and the prohibition's own before/after hash manifest reads through
    ``load_dataframe`` -- it would NOT have noticed a ``data/optuna/`` write. This runs an
    ACTUAL search (small budget) with the Phase-30 storage redirected into tmp_path and
    compares size + mtime + sha256 of each v2.0 study file. Remediation if this goes red: the
    tuner is falling back to its ``data/optuna`` default; pass storage_dir explicitly.
    """
    v2_dir = REPO_ROOT / "data" / "optuna"
    v2_dbs = sorted(v2_dir.glob("*_tuning_v1.db"))
    if not v2_dbs:
        pytest.skip(f"No v2.0 study files under {v2_dir} to guard")

    before = {
        db: (db.stat().st_size, db.stat().st_mtime_ns, _sha256(db)) for db in v2_dbs
    }

    monkeypatch.setattr(base_trainer, "TUNING_STORAGE_DIR", tmp_path / "optuna")
    trainer = _DummyTrainer()
    trainer.use_phase30_tuning()
    X_train, y_train = _training_frame()
    trainer.tune_hyperparameters(X_train, y_train, n_trials=2)

    for db, (size, mtime, digest) in before.items():
        now = db.stat()
        assert (now.st_size, now.st_mtime_ns, _sha256(db)) == (size, mtime, digest), (
            f"{db} changed across a Stage-2 tuned search. The v2.0 studies are the "
            "historical record; the Phase-30 search must write only under "
            "TUNING_STORAGE_DIR (outside data/)."
        )

    assert (tmp_path / "optuna").exists(), (
        "The Phase-30 search wrote nothing under the redirected storage dir, so this test "
        "did not actually exercise a search."
    )


def test_v2_study_files_are_left_untouched_by_the_phase_30_storage_dir() -> None:
    """The Phase-30 storage dir is not the v2.0 ``data/optuna`` dir holding the v1 studies.

    The three v2.0 studies are the historical record; nothing in this phase may delete or
    resume them. Remediation if this goes red: repoint TUNING_STORAGE_DIR away from
    data/optuna -- do NOT delete the v1 db files to "make room".
    """
    storage = Path(TUNING_STORAGE_DIR).resolve()
    v2_dir = (REPO_ROOT / "data" / "optuna").resolve()
    assert storage != v2_dir, (
        f"TUNING_STORAGE_DIR is the v2.0 study dir {v2_dir}; a Phase-30 train would resume "
        "the v1 studies and run zero new trials."
    )


# ---------------------------------------------------------------------------
# F8: _clear_staging_dir must not propagate a bare PermissionError mid-promotion
# ---------------------------------------------------------------------------


def test_clear_staging_dir_names_the_locked_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locked/read-only staging subdir raises with the offending path in the message.

    This host is Windows 11, where a held DuckDB/SQLite handle or an open file browser makes
    ``shutil.rmtree`` raise ``PermissionError``. An unhandled traceback here is the worst
    available outcome: it lands BETWEEN clearing one staging dir and training the next.
    Remediation if this goes red: restore the OSError catch-and-re-raise in
    ``_clear_staging_dir`` naming the path and the two likely holders.
    """
    staging = tmp_path / "artifacts_staging"
    locked = staging / "wp_20260101_000000"
    locked.mkdir(parents=True)
    (locked / "model.joblib").write_text("held", encoding="utf-8")

    def _raise_permission_error(path: Any, *args: Any, **kwargs: Any) -> None:
        msg = "[WinError 32] The process cannot access the file"
        raise PermissionError(msg)

    monkeypatch.setattr(shutil, "rmtree", _raise_permission_error)

    with pytest.raises(RuntimeError) as excinfo:
        promote._clear_staging_dir(staging)

    message = str(excinfo.value)
    assert str(locked) in message, (
        f"The re-raised error does not name the locked path {locked}: {message!r}."
    )
    assert "DuckDB" in message or "SQLite" in message, (
        f"The re-raised error does not name a likely holder: {message!r}. The operator "
        "needs to know what to close."
    )


def test_clear_staging_dir_succeeds_on_an_unlocked_tree(tmp_path: Path) -> None:
    """The happy path still clears stale candidate dirs and the stale staging manifest.

    Remediation if this goes red: the hardening broke the normal path; stale dirs left
    behind can be selected as a candidate by newest-by-name resolution (T-24-17).
    """
    staging = tmp_path / "artifacts_staging"
    stale = staging / "wp_20260101_000000"
    stale.mkdir(parents=True)
    (stale / "model.joblib").write_text("stale", encoding="utf-8")
    (staging / "latest.json").write_text(
        '{"wp": "wp_20260101_000000"}', encoding="utf-8"
    )

    promote._clear_staging_dir(staging)

    assert not stale.exists(), f"{stale} survived _clear_staging_dir."
    assert not (staging / "latest.json").exists(), (
        "The stale staging manifest survived _clear_staging_dir."
    )
