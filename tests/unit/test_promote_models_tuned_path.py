"""Proof the Stage-2 tuned path really tuned, and the windows were derived not typed.

Plan 30-01 (PROD-01). Two RESEARCH Pitfall-2 failure modes and one D30-12 asymmetry are
pinned here, because each of them fails SILENTLY -- the run completes, an artifact appears,
the gate prints a 2x2, and the number is wrong:

  * A VACUOUS RE-TUNE (T-30-02). ``OptunaTuner.optimize`` computes remaining trials as
    ``max(0, n_trials - len(study.trials))`` under ``load_if_exists=True``, so resuming a
    study that already holds the full budget runs ZERO new trials while still reporting a
    full trial count. A Phase-30 "tuned" candidate would then be carrying v2.0 parameters.
    Closed by a per-phase ``TUNING_STUDY_TAG`` plus a hard RuntimeError on zero new trials.
    The BACKTEST hit the same failure on the same mechanism and is closed separately, by
    per-RUN study STORAGE under a CONSTANT study name, with its own proofs in
    tests/unit/test_backtest_tuning_identity.py (Plan 30-16, D30-OWNER-02). This module owns
    the Stage-2 identity and the invariant that neither non-legacy identity is ever the
    default.
  * A WRITE UNDER ``data/`` (T-30-14). ``OptunaTuner`` defaults ``storage_dir`` to
    ``data/optuna``; this phase's prohibition forbids writing under ``data/`` outside the one
    sanctioned fingerprinted rebuild, and the prohibition's own hash manifest reads through
    ``load_dataframe`` and would NOT have caught a ``data/optuna/`` write. Asserted directly.
  * THE D30-12 WINDOW ASYMMETRY, AND HOW IT WAS FINALLY CLOSED. The deployed ATS incumbent is
    the D25-05 fix-cycle artifact and records a FIVE-season train window, while
    ``promote_models`` trained every candidate on the two-season default -- so the ATS
    candidate faced its own incumbent from a strictly worse configuration, a handicap nobody
    chose. D30-12's answer was to derive each target's window from that target's own metadata.
    Review finding CR-01 replaced that with the stronger one: ALL THREE windows now come from
    the committed rule in ``conf/season_partition.py``, so both sides of every comparison are
    stated by one reviewable commit rather than by three artifacts that were fitted on data
    since found to be wrong. A window read back out of an artifact's metadata -- or a silent
    fallback to ``TemporalSplitConfig.default()`` -- is the regression these tests catch.

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

from conf.season_partition import default_season_partition
from models import deploy_gate
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


@pytest.mark.parametrize("target", ["wp", "ats", "ou"])
def test_all_three_windows_come_from_the_committed_partition_rule(target: str) -> None:
    """ALL THREE windows -- train, hp_val, holdout -- come from conf/season_partition.py.

    THIS IS THE PROOF FOR REVIEW FINDING CR-01, and the assertion it replaces is the defect
    itself: this test used to pin ``train`` to the incumbent artifact's recorded window
    ("2015,2016,2017,2018,2019" for ATS, "2018,2019" for WP and O/U) and ``hp_val`` to
    "2020", with only the holdout taken from the rule. That is what ``promote_models``
    actually did, so the test was an accurate pin of a wrong behaviour.

    WHY THE OLD SHAPE WAS WRONG. A Wave-15 WP candidate would have been trained as
    ``--config-train-seasons 2018,2019``: feature selection on the 534-row window
    ``conf/season_partition.py``'s own evidence block records as admitting six of ATS's 25
    and nine of O/U's 25 features from pure synthetic noise -- with 45 new live weather
    candidates now entering the same selector -- while the hp-val fold of 2020 against a
    2024-2025 holdout put the calibrator in-sample against the shipped model over five
    seasons instead of one. It also derived the next model's training window from the
    metadata of an artifact the owner has declared VOID.

    Asserted against the LIVE ``artifacts/`` tree so it proves the real promotion path.
    Remediation if this goes red: do NOT re-pin the expectation to whatever the incumbent
    records. Either the committed rule changed -- in which case that commit is the record and
    every window moves with it -- or the promotion path has regressed to reading a window out
    of an artifact's metadata again.
    """
    if not (LIVE_ARTIFACTS / "latest.json").exists():
        pytest.skip(
            f"{LIVE_ARTIFACTS / 'latest.json'} not present -- bootstrap the production "
            "manifest per the clean-checkout step (RUNBOOK, Plan 25-05)"
        )

    partition = default_season_partition()
    window = promote._incumbent_window(target, LIVE_ARTIFACTS)

    expected = {
        "train": ",".join(str(s) for s in partition.selection),
        "hp_val": ",".join(str(s) for s in partition.hp_val),
        "holdout": ",".join(str(s) for s in partition.holdout),
    }
    for flag_stem, want in expected.items():
        assert window[flag_stem] == want, (
            f"_incumbent_window('{target}')[{flag_stem!r}] is {window[flag_stem]!r}, "
            f"expected {want!r} from conf/season_partition.py. Every window on the "
            "promotion path comes from the committed rule (review CR-01); a value that "
            "matches an artifact's metadata instead means the incumbent-derived window is "
            "back."
        )


@pytest.mark.parametrize("target", ["wp", "ats", "ou"])
def test_no_window_field_is_the_incumbents_recorded_window(target: str) -> None:
    """The negative half: no field may equal what the VOID incumbent records.

    The old ``test_ats_incumbent_window_is_not_the_default_window`` lived here and asserted
    the OPPOSITE of this -- that ATS's window differed from WP's, because each came from its
    own artifact. That per-target difference was D30-12's fix for a real asymmetry (the ATS
    candidate faced its own incumbent from a strictly worse configuration). CR-01 closes the
    same asymmetry one level up: both sides are now stated by one committed rule, so the
    windows are identical across targets BY CONSTRUCTION and the per-target difference is
    reported in ``window_report`` instead of silently honoured.

    Measured live, because the three deployed incumbents genuinely record pre-correction
    windows and this must fail if one of them ever leaks back into the argv.
    """
    if not (LIVE_ARTIFACTS / "latest.json").exists():
        pytest.skip(f"{LIVE_ARTIFACTS / 'latest.json'} not present")

    window = promote._incumbent_window(target, LIVE_ARTIFACTS)

    assert window["train"] not in {
        _ATS_FIX_CYCLE_TRAIN_WINDOW,
        _DEFAULT_TRAIN_WINDOW,
    }, (
        f"'{target}' resolves train={window['train']!r}, which is a window recorded by a "
        "deployed pre-correction artifact. Those artifacts were fitted on data since found "
        "to be wrong; deriving the next model's training window from one is the CR-01 "
        "defect."
    )
    assert window["hp_val"] != "2020", (
        f"'{target}' resolves hp_val='2020' -- the incumbents' recorded fold. Against a "
        "2024-2025 holdout that puts the calibrator in-sample against the shipped model "
        "over five seasons."
    )


def test_every_target_gets_the_same_window() -> None:
    """One rule, read once: the three targets cannot disagree about their windows.

    Not a restatement of the test above. That one pins each target against the rule; this
    one would still catch a per-target branch that read the rule correctly for two targets
    and something else for the third.
    """
    if not (LIVE_ARTIFACTS / "latest.json").exists():
        pytest.skip(f"{LIVE_ARTIFACTS / 'latest.json'} not present")

    resolved = {
        target: tuple(
            promote._incumbent_window(target, LIVE_ARTIFACTS)[stem]
            for stem in ("train", "hp_val", "holdout")
        )
        for target in ("wp", "ats", "ou")
    }

    assert len(set(resolved.values())) == 1, (
        f"the three targets resolve DIFFERENT windows: {resolved}. Under the committed rule "
        "they are identical by construction, so a difference means at least one target is "
        "reading its window from somewhere else."
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
# WR-01: the ratified verdict is repo-anchored, and an ARMED run fails CLOSED
# ---------------------------------------------------------------------------


def test_the_verdict_path_is_anchored_to_the_repo_not_the_cwd() -> None:
    """A CWD-relative verdict path made WHAT gets trained depend on where you stood.

    ``_GROUP_GATE_VERDICT_PATH`` was ``Path("config/group_gate_verdict.toml")``. Combined
    with a not-found branch that returns ``([], "none")`` and carries on, running the
    promotion from any directory other than the repo root silently trained every feature
    group -- including the group the frozen rule DROPPED.
    """
    path = promote._GROUP_GATE_VERDICT_PATH

    assert path.is_absolute(), (
        f"_GROUP_GATE_VERDICT_PATH is {path!r}, which is CWD-relative. Anchor it to the "
        "repository via Path(__file__).resolve().parent.parent."
    )
    assert path.name == "group_gate_verdict.toml"
    assert path.parent.name == "config"
    assert path.exists(), (
        f"The ratified Stage-1 verdict is not at the repo-anchored path {path}. It is a "
        "git-tracked file; a moved or deleted verdict is a broken checkout."
    )


def test_an_armed_run_with_no_verdict_and_no_override_REFUSES(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WR-01: ``--promote`` must not train an un-Stage-1-selected candidate and swap it.

    The old behaviour printed a ``[WARNING]``, returned an empty list, trained on EVERY
    group and then swapped whatever passed -- silently reversing a ratified owner decision
    with ``provenance: none`` in the banner as the only trace. The sibling
    ``_incumbent_window`` in this same module already insists that "an unresolvable window
    must be a stop, never a guess"; the exclusion list decides what gets trained at all and
    now gets the same rule on the armed path.
    """
    monkeypatch.setattr(promote, "_GROUP_GATE_VERDICT_PATH", tmp_path / "absent.toml")

    args = promote.parse_args(["--promote"])
    with pytest.raises(FileNotFoundError, match="ARMED promotion"):
        promote._resolve_exclude_groups(args)


def test_an_armed_run_can_still_state_exclude_nothing_deliberately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal is about GUESSING, not about excluding nothing.

    ``--exclude-groups ''`` is an explicit statement and stays available on the armed path.
    """
    monkeypatch.setattr(promote, "_GROUP_GATE_VERDICT_PATH", tmp_path / "absent.toml")

    args = promote.parse_args(["--promote", "--exclude-groups", ""])
    groups, provenance = promote._resolve_exclude_groups(args)

    assert groups == []
    assert provenance == "override"


def test_an_unarmed_dry_run_with_no_verdict_still_warns_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dry run swaps nothing, so it must stay runnable on an unratified checkout."""
    monkeypatch.setattr(promote, "_GROUP_GATE_VERDICT_PATH", tmp_path / "absent.toml")

    groups, provenance = promote._resolve_exclude_groups(promote.parse_args([]))

    assert (groups, provenance) == ([], "none")
    printed = capsys.readouterr().out
    assert "[WARNING]" in printed
    assert "--promote would REFUSE" in printed, (
        "The dry-run warning must say what the armed path would do, or the operator "
        "learns about the refusal only when it fires."
    )


def test_an_armed_run_with_a_ratified_verdict_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal fires on ABSENCE only; a ratified verdict arms normally."""
    verdict = tmp_path / "group_gate_verdict.toml"
    verdict.write_text('excluded_groups = ["injury"]\n', encoding="utf-8")
    monkeypatch.setattr(promote, "_GROUP_GATE_VERDICT_PATH", verdict)

    groups, provenance = promote._resolve_exclude_groups(
        promote.parse_args(["--promote"])
    )

    assert (groups, provenance) == (["injury"], "verdict")


# ---------------------------------------------------------------------------
# WR-02 / review CR-01: the WHOLE window is reconciled against the incumbent's
# record, and every field that moved is named
# ---------------------------------------------------------------------------


def test_a_narrower_incumbent_holdout_is_reported_against_the_live_partition(
    tmp_path: Path,
) -> None:
    """UPDATED by Plan 33.1-09 Task 3: the refusal became a REPORT (D33.1-04).

    WHAT THIS USED TO ASSERT, and why it was right at the time: a derived holdout that was
    not the frozen one would train ON the seasons the gate then scores, so the check RAISED.
    The hazard it named has not gone away and is quoted from the function's own comment:
    ``BaseTrainer.train_and_evaluate`` keeps the model from the LAST split, so the saved
    artifact would have SEEN those seasons, and an in-sample candidate would be compared
    against an out-of-sample baseline -- "a confident 2x2 would print with no warning at all".

    WHY IT IS NOW A REPORT. Under D33.1-01 the deployed model is deliberately fitted on every
    completed season, so that condition is no longer an anomaly to refuse -- it is the chosen
    design, and raising on it would refuse every run. The owner accepted the cost after it was
    stated twice. What must NOT happen is the second half of the old comment: the difference
    printing with no warning. So the assertion moves from "it raises" to "it reports, and the
    report names both windows and the in-sample consequence", which is the part that was
    load-bearing.

    Editing an artifact's metadata.json to make the old check pass is PROHIBITED (D33.1-04)
    and is not what happened here: this drives a synthetic tree in tmp_path.
    """
    from conf.season_partition import default_season_partition

    metadata = _good_metadata([2015, 2016, 2017, 2018, 2019])
    metadata["config"]["holdout_seasons"] = [2021, 2022]
    root = _write_artifacts_tree(
        tmp_path / "artifacts",
        {"ats": "ats_20260101_000000"},
        {"ats_20260101_000000": metadata},
    )

    window = promote._incumbent_window("ats", root)

    report = window["window_report"]
    assert report, "a differing incumbent holdout must be REPORTED, never silent."
    assert "[2021, 2022]" in report, report
    assert "in-sample" in report.lower(), (
        f"the report must name the consequence, not merely the difference: {report}"
    )
    assert window["holdout"] == ",".join(
        str(s) for s in default_season_partition().holdout
    )
    # WIDENED for review CR-01: the train window moved too (the incumbent records
    # 2015-2019, the rule says 2018-2022), and a report that named only the holdout is
    # how a silently-inherited selection window stayed invisible for a whole phase.
    assert "train:" in report, (
        f"the report must name EVERY field that moved, not the holdout alone: {report}"
    )


def test_a_wider_incumbent_holdout_is_reported_against_the_live_partition(
    tmp_path: Path,
) -> None:
    """The report fires on ANY difference, in either direction. See the test above."""
    metadata = _good_metadata([2018, 2019])
    metadata["config"]["holdout_seasons"] = [2020, 2021, 2022, 2023, 2024]
    root = _write_artifacts_tree(
        tmp_path / "artifacts",
        {"ats": "ats_20260101_000000"},
        {"ats_20260101_000000": metadata},
    )

    window = promote._incumbent_window("ats", root)
    assert window["window_report"], (
        "widening is not a safe direction either -- the old check was an EQUALITY and the "
        "report must fire on the same set of differences the raise used to."
    )


def test_an_incumbent_recording_the_live_partition_reports_NOTHING(
    tmp_path: Path,
) -> None:
    """The control that keeps the report from being unconditional.

    A report that fires on every input is not a report. Once a future re-fit writes the live
    partition into a new metadata.json, there is no difference to state and the field must be
    empty -- which is also how a reader will know the difference above was real.

    WIDENED for review CR-01: the control now writes all THREE windows from the rule, because
    the report now compares all three. Writing only the holdout would leave train and hp_val
    differing and the report non-empty, and the control would pass for the wrong reason.
    """
    from conf.season_partition import default_season_partition

    partition = default_season_partition()
    metadata = _good_metadata(list(partition.selection))
    metadata["config"]["hp_val_seasons"] = list(partition.hp_val)
    metadata["config"]["holdout_seasons"] = list(partition.holdout)
    root = _write_artifacts_tree(
        tmp_path / "artifacts",
        {"ats": "ats_20260101_000000"},
        {"ats_20260101_000000": metadata},
    )

    window = promote._incumbent_window("ats", root)
    assert window["window_report"] == "", window["window_report"]


@pytest.mark.parametrize("target", ["wp", "ats", "ou"])
def test_every_live_incumbent_DIFFERS_from_the_live_partition_and_says_so(
    target: str,
) -> None:
    """UPDATED by Plan 33.1-09 Task 3. The latent condition became live.

    It asserted all three incumbents AGREED with the gate holdout, which was true while the
    two were the same four seasons. They are not: all three record [2021..2024] and the live
    partition is the two most recent completed seasons. Under D33.1-04 the correct outcome is
    a reported DIFFERENCE -- agreement here would mean somebody EDITED a record of a past
    training run rather than re-fitting, which is the defect class this milestone exists to
    detect.

    Measured against the LIVE artifacts tree, not a copy, so it asserts the real state.
    """
    if not (LIVE_ARTIFACTS / "latest.json").exists():
        pytest.skip(
            f"{LIVE_ARTIFACTS / 'latest.json'} not present -- artifacts/ is gitignored, so "
            "this evidence-backed control did not run on this checkout"
        )

    window = promote._incumbent_window(target, LIVE_ARTIFACTS)
    live = ",".join(str(int(s)) for s in deploy_gate.HOLDOUT_SEASONS)

    assert window["holdout"] == live, (
        f"'{target}' window carries holdout {window['holdout']!r}, not the live partition "
        f"{live!r}."
    )
    assert window["window_report"], (
        f"'{target}' incumbent's recorded window matches the live partition, so nothing was "
        "reported. All three incumbents record [2021..2024] and no re-fit has run; "
        "agreement means a metadata.json was edited (D33.1-04 prohibits that)."
    )


# ---------------------------------------------------------------------------
# WR-04: the promoted artifact records WHAT was excluded, and on whose authority
# ---------------------------------------------------------------------------


def test_step1_argv_carries_the_exclusion_provenance(tmp_path: Path) -> None:
    """``_build_train_argv`` already knew the provenance and dropped it at the boundary.

    CLAUDE.md requires that "any prediction must be reproducible given the same input data
    snapshot". Re-running ``models.train`` from a promoted artifact's own metadata
    reproduced a DIFFERENT feature set, because the exclusion was recoverable only from the
    git-tracked verdict file plus knowledge of which commit was current.
    """
    window = {"train": "2018,2019", "hp_val": "2020", "holdout": "2021,2022,2023,2024"}
    argv = promote._build_train_argv(
        "ats", tmp_path / "staging", window, ["injury"], "verdict"
    )

    assert "--exclude-groups-provenance" in argv
    assert argv[argv.index("--exclude-groups-provenance") + 1] == "verdict"
    assert argv[argv.index("--exclude-groups") + 1] == "injury"


def test_train_target_records_the_exclusion_in_the_artifact_metadata() -> None:
    """The recorded exclusion must reach ``trainer.metadata``, not just the run log."""
    from models.train import train_target

    source = inspect.getsource(train_target)
    assert 'trainer.metadata["exclude_groups"]' in source
    assert 'trainer.metadata["exclude_groups_provenance"]' in source

    signature = inspect.signature(train_target)
    assert "exclude_groups" in signature.parameters
    assert "exclude_groups_provenance" in signature.parameters


def test_models_train_main_passes_both_through_to_train_target() -> None:
    """The wiring, not just the capability: main() must hand both values over."""
    from models import train as train_mod

    source = inspect.getsource(train_mod.main)
    assert "exclude_groups=exclude_groups" in source
    assert "exclude_groups_provenance=args.exclude_groups_provenance" in source, (
        "models.train.main() does not thread the provenance into train_target, so the "
        "artifact would record the exclusion without saying whether it was ratified."
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
    so the backtest now opts into its own per-RUN study STORAGE via
    ``BaseTrainer.use_backtest_tuning(run_id)`` -- the run id names the storage directory and
    the study NAME stays constant, because optuna's HyperbandPruner brackets trials by a crc32
    of that name -- and those anchors were re-ratified deliberately with the drift recorded
    (Plan 30-16 Task 3, tests/integration/test_diag_diagnosis.py).

    WHAT THE INVARIANT STILL PROTECTS, and why it is still worth a test. There are now THREE
    identities with three lifetimes: legacy (the default), per-phase (Stage 2), per-run
    storage (the backtest). Making ANY non-legacy identity the default would change what
    every caller that never opted in trains with, in one move -- ``scripts.retrain_models``
    today, and whatever is added tomorrow. Remediation if this goes red: keep the identities
    as explicit opt-ins (``models.train.train_target`` and
    ``BacktestEngine._create_trainer``); do not move either into ``BaseTrainer.__init__``.
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
