"""The committed recipe registry (Phase 34, LDGR-03).

Every 2026 forward ledger row carries a ``recipe_id`` that must resolve to an entry here. An entry
names the training recipe AND the bet rule in force when the row was decided, so a row can always
be traced to how its models were fitted and how its bet was chosen.

ONE-WAY BY CONSTRUCTION, in the shape of ``backtest/ev_chain_constants.py``. This module IS the
pre-registration of the recipes the ledger stamps, not a description of them. Phase 37 ADDS
entries for its frozen-recipe re-fits; it never redefines an existing one. Stated plainly because it
is easy to forget later in the season: EDITING AN ENTRY AFTER WEEK W'S FIRST LOCK DOES NOT FIX A
BUG -- IT DESTROYS THE EVIDENCE. A changed recipe is a NEW recipe id appended here; an entry that
is wrong stays wrong, and the only honest response is to say so in the readout.

THIS MODULE DOES NOT RECORD ITS OWN CONTENT HASH. A file that must contain its own whole-file hash
has no fixed point. The witness lives OUTSIDE it: ``tests/phase34_state.py`` records this file's
commit and normalized sha256 in a LATER commit under its APPEND PROTOCOL, and the Phase-34 ancestry
test recomputes and compares (Plan 34-22).

Standard library only: NO project imports, NO I/O and NO logic beyond the record's own dataclass,
so an entry cannot change meaning when the code around it moves. Every value below was MEASURED on
2026-10-05 from the files themselves, read-only: each model's facts from its own ``metadata.json``
through ``models.artifacts.load_model_artifact``; the blend's provenance and its bound converter
from its ``blend_weights.json`` through ``models.artifacts.ARTIFACT_VALIDATORS["blend"]`` (a blend
directory holds no ``metadata.json``); the bet-rule modules' last commits from git; and the two
sha256 digests from the bytes on disk. ``tests/unit/test_recipe_registry.py`` pins the entry to the
live bet-rule constants and to the artifacts it names.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "IN_FORCE_RECIPE_ID",
    "RECIPE_REGISTRY",
    "RecipeEntry",
]


@dataclass(frozen=True)
class RecipeEntry:
    """One registered recipe: how a row's models were trained and how its bet was chosen.

    Attributes:
        recipe_id: The id a ledger row stamps in its ``recipe_id`` column.
        registered_on: The ISO date the entry was measured and written.
        description: What the recipe is, in plain words.
        model_artifact_ids: ``wp`` / ``ats`` / ``ou`` -> the model artifact directory it produced.
        blend_id: The blend artifact directory tuned on those three models.
        converter_id: The market-probability converter the blend is bound to.
        blend_provenance: The blend's own record of its gold generation and source models.
        training: Per model target, the training facts copied from that model's metadata.json.
        bet_rule_modules: Repo-relative module path -> the commit that last modified it.
        edge_tier_thresholds: The bet rule's per-target edge thresholds, copied as literals.
        chain_fit_record_path: The (gitignored) tune-only chain-fit run record the rule reads.
        chain_fit_record_sha256: The sha256 of that record's bytes.
        fill_convention_id: The pricing and sizing convention the rule was measured under.
        uv_lock_sha256: The sha256 of ``uv.lock``: the library versions replay depends on.
        notes: Anything a reader needs that the fields above do not say.
    """

    recipe_id: str
    registered_on: str
    description: str
    model_artifact_ids: Mapping[str, str]
    blend_id: str
    converter_id: str
    blend_provenance: Mapping[str, Any]
    training: Mapping[str, Mapping[str, Any]]
    bet_rule_modules: Mapping[str, str]
    edge_tier_thresholds: Mapping[str, tuple[float, float]]
    chain_fit_record_path: str
    chain_fit_record_sha256: str
    fill_convention_id: str
    uv_lock_sha256: str
    notes: str


# The row-19 training recipe, identical for all three targets except the feature count and the
# hyperparameters, which are each model's own. Copied from each model's metadata.json.
_TRAIN_SEASONS: tuple[int, ...] = tuple(range(2002, 2023))  # 2002-2022
_FINAL_FIT_SEASONS: tuple[int, ...] = tuple(range(2002, 2026))  # 2002-2025
_GOLD_GENERATION_DIGEST = (
    "9ba3a56885ab3b26524d2255e73b46bf674c9b18043cd7ca2bcc74a71cab9228"
)
_GROUP_VERDICT_DIGEST = (
    "06e147083ad2d11026a45d2587b75851a95ff2306465b997e13ebee8dc949112"
)
_XGB_BEST_PARAMS: Mapping[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
    "n_jobs": -1,
}


def _training_facts(
    *, feature_count: int, best_params: Mapping[str, Any]
) -> Mapping[str, Any]:
    """One model target's training facts, as its metadata.json records them."""
    return {
        "train_seasons": _TRAIN_SEASONS,
        "hp_val_seasons": (2023,),
        "holdout_seasons": (2024, 2025),
        "final_fit_seasons": _FINAL_FIT_SEASONS,
        "final_fit_rows": 6499,
        "gold_generation_digest": _GOLD_GENERATION_DIGEST,
        "feature_selection": {
            "exclude_groups": ("injury", "situational", "snap"),
            "exclude_groups_provenance": "verdict",
            "group_verdict_digest": _GROUP_VERDICT_DIGEST,
            "feature_count": feature_count,
            # Verbatim from final_fit_components.feature_names.
            "rule": (
                "carried over unchanged from the walk-forward stage's selection "
                f"({feature_count} features)"
            ),
        },
        "hyperparameters": {
            "source": "trainer.metadata['best_params']",
            "best_params": best_params,
        },
    }


IN_FORCE_RECIPE_ID: str = "recipe-2026-row19-v1"

RECIPE_REGISTRY: dict[str, RecipeEntry] = {
    IN_FORCE_RECIPE_ID: RecipeEntry(
        recipe_id=IN_FORCE_RECIPE_ID,
        registered_on="2026-10-05",
        description=(
            "The recipe in force when the Phase-34 ledger went live: the row-19 models (Elo with "
            "zero home-field advantage at neutral sites, quick task 261003-vke), trained on the "
            "rebuilt gold, the blend re-tuned on them, and the 2026 bet rule re-measured with the "
            "unchanged recipe (both halves) in commit 55de53a."
        ),
        model_artifact_ids={
            "wp": "wp_20261004_050223",
            "ats": "ats_20261004_050228",
            "ou": "ou_20261004_050232",
        },
        blend_id="blend_20261004_050521",
        converter_id="market_probability_20260923_195443",
        blend_provenance={
            "gold_generation_digest": _GOLD_GENERATION_DIGEST,
            "source_artifact_ids": {
                "wp": "wp_20261004_050223",
                "ats": "ats_20261004_050228",
                "ou": "ou_20261004_050232",
            },
        },
        training={
            "wp": _training_facts(
                feature_count=20,
                best_params={
                    "max_iter": 1000,
                    "solver": "lbfgs",
                    "C": 1.0,
                    "random_state": 42,
                },
            ),
            "ats": _training_facts(feature_count=25, best_params=_XGB_BEST_PARAMS),
            "ou": _training_facts(feature_count=25, best_params=_XGB_BEST_PARAMS),
        },
        bet_rule_modules={
            "backtest/neutral_hfa_cold_start_constants.py": (
                "55de53a27189df52b791a5cc1b16ead470d1302d"
            ),
            "backtest/neutral_hfa_ev_chain_constants.py": (
                "55de53a27189df52b791a5cc1b16ead470d1302d"
            ),
        },
        edge_tier_thresholds={
            "ats": (1.6229, 0.6211),
            "ou": (0.0452, 0.0162),
            "wp": (0.05, 0.02),
        },
        chain_fit_record_path="outputs/row19/neutral_hfa_chain_fit.json",
        chain_fit_record_sha256=(
            "fab1118b8093bf55fd1cd4d7d8ba80eff3f95b7bd9537c4cf4f3a6b2a6ce2906"
        ),
        fill_convention_id="fill-v1",
        uv_lock_sha256=(
            "751d7d13b685a5fecee18ecb981e2cdf7be02afdfceb1dd23c013bd98bc05da0"
        ),
        notes=(
            "The chain-fit record is gitignored generator output (record id "
            "neutral_hfa_chain_fit_20261004_052326), so it is identified here by sha256; the "
            "ledger directory holds its own copy under Phase 34 D-02. The bet rule's human-readable "
            "half is NEUTRAL-HFA-BET-RULE-CORRECTION.md at the repo root, committed with the "
            "two modules named above. The models were promoted by owner ruling, not through the "
            "deploy gate (CHAIN_FIT_BIAS_SOURCE_BY_TARGET: NOT_GATED). Both sha256 digests were "
            "taken over the bytes on disk, which hold LF line endings only."
        ),
    ),
}
