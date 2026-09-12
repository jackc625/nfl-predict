"""Tests for the relocated in-package ModelValidator (pipeline.model_validation).

These cover the two methods the Friday orchestrator actually calls:
validate_model_availability and validate_model_loadability. They assert the
return-key shape ({name}_available / {name}_loadable booleans) and the
availability/loadability semantics against the REAL artifact convention --
versioned ``artifacts/{target}_{ts}/`` directories resolved via
``artifacts/latest.json`` with keys ``model``/``feature_list`` (CR-01 fix,
FIX-01 / D-03). The former fixtures fabricated the wrong
``{name}_model.joblib`` schema with ``model/scaler/feature_names`` keys and so
masked the defect; they have been rebuilt to mirror
``models.artifacts.save_model_artifact``'s on-disk layout.

Also covers the WR-02 gold-matrix gate, which Phase 33 Plan 33-07 MOVED off
``pipeline.steps.step_verify_data_artifacts`` and onto its own boundary step
``pipeline.steps.step_verify_gold_currency``. The gate itself is unchanged in
kind -- the three concrete matrices are still required by name, and the two
historical dead names still may not reappear -- but it now runs AFTER the step
that builds gold rather than before it, because a currency check that runs
before its producer reports an ordering fact as a stale artifact.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest


def _write_real_artifact(artifacts_dir, target, ts="20260101_000000"):
    """Create a versioned artifact dir mirroring save_model_artifact's outputs.

    Writes ``artifacts_dir/{target}_{ts}/`` with model.pkl + metadata.json +
    feature_list.json, then updates the latest.json manifest pointer to it --
    exactly the layout the production loader (load_model_artifact) expects.

    Returns the created artifact directory Path.
    """
    artifacts_dir = Path(artifacts_dir)
    artifact_dir = artifacts_dir / f"{target}_{ts}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(object(), artifact_dir / "model.pkl")
    (artifact_dir / "metadata.json").write_text(json.dumps({"target": target}))
    (artifact_dir / "feature_list.json").write_text(json.dumps(["a", "b", "c"]))

    latest_path = artifacts_dir / "latest.json"
    manifest = json.loads(latest_path.read_text()) if latest_path.exists() else {}
    manifest[target] = artifact_dir.name
    latest_path.write_text(json.dumps(manifest, indent=2))

    return artifact_dir


# The two artifact names that caused the v2.1 milestone-audit BLOCKER: the DATA-phase
# gate required them and NO build script writes them, so a full-mode Friday run aborted.
# They are named ONCE, here, and both the live assertion and its planted-violation
# control read this tuple.
_DEAD_ARTIFACT_NAMES = (
    "data/silver/elo_ratings.parquet",
    "data/silver/team_form.parquet",
)


def _dead_names_in(triples):
    """Every historical dead artifact name present among *triples*' PATH components.

    Asks the question of the PATH rather than of the whole triple. A membership test
    against the triples themselves compares a string to a tuple, is False for every
    input, and would therefore report a clean gate no matter what the gate required --
    the vacuous form the caller's docstring describes at length.
    """
    paths = {path for path, _boundary, _coverage_check in triples}
    return sorted(name for name in _DEAD_ARTIFACT_NAMES if name in paths)


class TestModelValidatorImport:
    """The class must be importable from the pipeline package, not scripts.*."""

    def test_importable_from_pipeline_package(self):
        from pipeline.model_validation import ModelValidator

        validator = ModelValidator()
        assert hasattr(validator, "validate_model_availability")
        assert hasattr(validator, "validate_model_loadability")

    def test_default_artifacts_dir_is_artifacts(self):
        """The default resolves the real artifacts/ tree, NOT artifacts/models."""
        from pipeline.model_validation import ModelValidator

        validator = ModelValidator()
        assert str(validator.artifacts_dir) == str(Path("artifacts"))


class TestValidateModelAvailability:
    """validate_model_availability resolves each target via latest.json."""

    def test_all_present_returns_all_true(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        for name in ("wp", "ats", "ou"):
            _write_real_artifact(tmp_path, name)

        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_availability(["wp", "ats", "ou"])

        assert result == {
            "wp_available": True,
            "ats_available": True,
            "ou_available": True,
        }

    def test_blend_manifest_key_is_ignored(self, tmp_path):
        """A non-target key (blend) in latest.json must not break resolution."""
        from pipeline.model_validation import ModelValidator

        for name in ("wp", "ats", "ou"):
            _write_real_artifact(tmp_path, name)
        # Inject a blend pointer the validator must ignore (only wp/ats/ou iterated).
        latest_path = tmp_path / "latest.json"
        manifest = json.loads(latest_path.read_text())
        manifest["blend"] = "blend_dynamic_20260101_000000"
        latest_path.write_text(json.dumps(manifest))

        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_availability(["wp", "ats", "ou"])

        assert result == {
            "wp_available": True,
            "ats_available": True,
            "ou_available": True,
        }

    def test_negative_missing_target_key_is_not_available(self, tmp_path):
        """Negative 1: latest.json present but target key missing -> not available."""
        from pipeline.model_validation import ModelValidator

        # Only wp is written to the manifest; ats/ou keys are absent.
        _write_real_artifact(tmp_path, "wp")

        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_availability(["wp", "ats", "ou"])

        assert result["wp_available"] is True
        assert result["ats_available"] is False
        assert result["ou_available"] is False


class TestValidateModelLoadability:
    """validate_model_loadability loads via load_model_artifact (model + feature_list)."""

    def test_valid_artifact_is_loadable(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        _write_real_artifact(tmp_path, "wp")

        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_loadability(["wp"])

        assert result == {"wp_loadable": True}

    def test_missing_target_is_not_loadable(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        # Empty artifacts dir with no latest.json at all.
        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_loadability(["wp"])

        assert result == {"wp_loadable": False}

    def test_negative_model_pkl_absent_is_not_loadable(self, tmp_path):
        """Negative 2: manifest points at a version dir whose model.pkl is absent."""
        from pipeline.model_validation import ModelValidator

        # Point latest.json at a version dir, then remove its model.pkl so the
        # load attempt fails and degrades to not-loadable.
        artifact_dir = _write_real_artifact(tmp_path, "wp")
        (artifact_dir / "model.pkl").unlink()

        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_loadability(["wp"])

        assert result == {"wp_loadable": False}

    def test_corrupt_artifact_is_not_loadable(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        # Valid manifest + dir, but model.pkl is not a real joblib payload so the
        # load raises and is caught (graceful degradation).
        artifact_dir = _write_real_artifact(tmp_path, "wp")
        (artifact_dir / "model.pkl").write_text("not a joblib artifact")

        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_loadability(["wp"])

        assert result == {"wp_loadable": False}


_GATE_SEASON = 2026
_GATE_WEEK = 2


class TestVerifyDataArtifactsGoldGate:
    """WR-02: the three concrete gold matrices are still required BY NAME.

    Phase 33 Plan 33-07 moved WHERE that requirement is checked -- onto
    ``step_verify_gold_currency``, which runs after the gold build -- without
    weakening WHAT is required. Every case below drives the gate that owns the
    matrices now, so the WR-02 recurrence guard survives the move rather than
    quietly becoming a check of an empty subset.

    The fixtures write REAL parquet rows rather than zero-byte stubs, because the
    gates no longer ask only whether a path exists: they ask whether the artifact
    carries a row for the current ``(season, week)``.
    """

    def _rows(self, week):
        return pd.DataFrame(
            {
                "game_id": [f"{_GATE_SEASON}_{week:02d}_AAA_BBB"],
                "season": [_GATE_SEASON],
                "week": [week],
            }
        )

    def _team_form_rows(self, week):
        """``team_form_features`` keys its week on ``target_season`` / ``target_week``."""
        return pd.DataFrame(
            {
                "team": ["BUF"],
                "target_season": [_GATE_SEASON],
                "target_week": [week],
            }
        )

    def _write_silver(self, root, week=_GATE_WEEK):
        """Create the three concrete silver files the DATA-boundary gate requires."""
        silver = root / "data" / "silver"
        silver.mkdir(parents=True, exist_ok=True)
        self._rows(week).to_parquet(silver / "games.parquet", index=False)
        self._rows(week).to_parquet(silver / "elo_game_snapshots.parquet", index=False)
        self._team_form_rows(week).to_parquet(
            silver / "team_form_features.parquet", index=False
        )

    def _write_gold(self, root, matrices, week=_GATE_WEEK):
        gold = root / "data" / "gold"
        gold.mkdir(parents=True, exist_ok=True)
        for name in matrices:
            self._rows(week).to_parquet(gold / name, index=False)

    def _pin_week(self, monkeypatch):
        monkeypatch.setattr(
            "utils.date_utils.get_current_nfl_week",
            lambda: (_GATE_SEASON, _GATE_WEEK),
        )

    def test_raises_when_gold_empty(self, tmp_path, monkeypatch):
        """An empty data/gold/ (no matrices) must raise, not pass as ready."""
        from pipeline.steps import step_verify_gold_currency

        self._write_silver(tmp_path)
        (tmp_path / "data" / "gold").mkdir(parents=True, exist_ok=True)
        self._pin_week(monkeypatch)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(RuntimeError) as exc_info:
            step_verify_gold_currency()

        msg = str(exc_info.value)
        assert "Missing data artifacts" in msg
        assert "features_wp.parquet" in msg

    def test_passes_when_all_three_matrices_present(self, tmp_path, monkeypatch):
        """Ready when all three concrete gold matrices carry the current week."""
        from pipeline.steps import step_verify_gold_currency

        self._write_silver(tmp_path)
        self._write_gold(
            tmp_path,
            ["features_wp.parquet", "features_ats.parquet", "features_ou.parquet"],
        )
        self._pin_week(monkeypatch)
        monkeypatch.chdir(tmp_path)

        # Should not raise.
        step_verify_gold_currency()

    def test_raises_when_one_matrix_missing(self, tmp_path, monkeypatch):
        """A single missing matrix (ou) is reported as missing."""
        from pipeline.steps import step_verify_gold_currency

        self._write_silver(tmp_path)
        self._write_gold(
            tmp_path,
            ["features_wp.parquet", "features_ats.parquet"],
        )
        self._pin_week(monkeypatch)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(RuntimeError) as exc_info:
            step_verify_gold_currency()

        assert "features_ou.parquet" in str(exc_info.value)

    def test_the_data_gate_no_longer_reports_gold_at_all(self, tmp_path, monkeypatch):
        """The move is asserted, not assumed: gold is absent and the DATA gate passes.

        This is the false stale-artifact refusal the boundary split exists to
        prevent. ``build_features`` runs in the PREDICTIONS phase, so at the DATA
        gate the current week's gold cannot exist yet, and a gate that refused it
        would fail every correct full-mode Friday run at DATA step 9.
        """
        from pipeline.steps import step_verify_data_artifacts

        self._write_silver(tmp_path)
        self._pin_week(monkeypatch)
        monkeypatch.chdir(tmp_path)

        # Should not raise: no data/gold/ tree exists at all.
        step_verify_data_artifacts()

    def test_required_artifact_list_matches_real_build_no_drift(self):
        """The gate's required set is a subset of a curated known-producer allowlist (D-02, B2).

        This is the recurrence-axis keystone for the v2.1 milestone-audit BLOCKER:
        ``step_verify_data_artifacts`` once required ``elo_ratings.parquet`` /
        ``team_form.parquet`` -- names NO build script writes -- so a full-mode
        Friday run aborted at the DATA-phase gate. A stub that matched those wrong
        names hid the defect from every unit test.

        To avoid the second-matching-stub trap (RESEARCH Pitfall 1 -- the very
        thing that hid the original defect), this test imports
        ``_REQUIRED_ARTIFACTS`` from the gate as the SINGLE source of truth rather
        than re-declaring a parallel EXPECTED copy. It then makes two assertions:

        1. **Subset check.** Every gate entry is a member of ``allowed``, a curated
           known-producer allowlist hand-maintained below (each entry annotated
           with the build script that writes it). This is a static subset
           assertion between the gate constant and a literal set -- NOT an
           introspection of the build scripts. It catches a gate that requires a
           name absent from the curated allowlist, but does NOT, on its own, catch
           a future change that adds the same bogus name to BOTH the gate and this
           allowlist (the same two-places-edited mistake that caused the original
           BLOCKER).

        2. **Dead-name guards.** The two specific historical dead names that caused
           the BLOCKER (``elo_ratings.parquet`` / ``team_form.parquet``) are
           negative-asserted to never reappear. These guard those two names only;
           they do not generalize to arbitrary future drift.

        When the curated allowlist drifts from the real producers, update both the
        allowlist and its per-entry producer annotations together. The check is
        purely structural (no data-layer dependency), so it always runs --
        including on a clean checkout with no ``data/`` tree.

        THE VACUITY TRAP THIS TEST NOW STEPS AROUND (Phase 33, Plan 33-07 Task 2).
        ``_REQUIRED_ARTIFACTS`` used to be a list of PATH STRINGS and is now a list
        of ``(path, phase_boundary, coverage_check)`` TRIPLES. A bare membership
        test for a dead PATH against a list of TRIPLES is always False -- a string
        is never equal to a tuple -- so both dead-name guards would have kept
        passing while guarding NOTHING, and the exact BLOCKER they were written for
        could have returned unnoticed. The membership question is therefore asked of
        the PATH COMPONENT, through ``_dead_names_in``, and ``_dead_names_in`` is
        itself shown to fire by the planted-violation control below. A guard nobody
        has seen fail is a guard whose shape nobody knows -- which is how the
        original defect survived a unit suite in the first place.
        """
        from pipeline.steps import _REQUIRED_ARTIFACTS, ARTIFACT_PHASE_BOUNDARIES

        # Names a real build actually writes (each producer verified in-repo):
        #   data/silver/games.parquet              <- scripts/ingest_games.py (canonical games table)
        #   data/silver/elo_game_snapshots.parquet <- scripts/build_elo.py save_dataframe(..., "elo_game_snapshots", layer="silver")
        #   data/silver/team_form_features.parquet <- scripts/build_team_form.py save_dataframe(..., "team_form_features", layer="silver")
        #   data/gold/features_wp.parquet          <- scripts/build_features.py (gold matrices)
        #   data/gold/features_ats.parquet         <- scripts/build_features.py (gold matrices)
        #   data/gold/features_ou.parquet          <- scripts/build_features.py (gold matrices)
        produced_silver = {
            "data/silver/games.parquet",
            "data/silver/elo_game_snapshots.parquet",
            "data/silver/team_form_features.parquet",
        }
        produced_gold = {
            "data/gold/features_wp.parquet",
            "data/gold/features_ats.parquet",
            "data/gold/features_ou.parquet",
        }
        allowed = produced_silver | produced_gold

        for path, boundary, coverage_check in _REQUIRED_ARTIFACTS:
            assert path in allowed, f"gate requires {path} but no build produces it"
            assert boundary in ARTIFACT_PHASE_BOUNDARIES, (
                f"{path} declares boundary {boundary!r}, which is outside the closed "
                f"vocabulary {ARTIFACT_PHASE_BOUNDARIES}; an artifact whose boundary "
                "matches no step is checked at NO boundary and its gate never runs"
            )
            assert callable(coverage_check), (
                f"{path} carries no coverage check, so the gate can only ask whether "
                "it exists -- the defect the triple conversion closes"
            )

        # The dead names that caused the BLOCKER must never reappear in the gate.
        assert _dead_names_in(_REQUIRED_ARTIFACTS) == []

        # PLANTED-VIOLATION CONTROL. Re-add one dead name to a LOCAL copy and assert
        # the guard reports it. Without this the two negatives above are unfalsifiable
        # claims about a helper nobody has watched work.
        planted = [
            *_REQUIRED_ARTIFACTS,
            ("data/silver/elo_ratings.parquet", "data", lambda *_a: True),
        ]
        assert _dead_names_in(planted) == ["data/silver/elo_ratings.parquet"], (
            "the dead-name guard did NOT report a re-added dead artifact; it is "
            "passing vacuously and would not catch the BLOCKER's recurrence"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
