"""The pre-registered comparison of the live arm against the frozen arm (Plan 37-16; SPEC R12).

ROLLING-REFIT-PREREGISTRATION.md section 11 is the rule: each arm's own stored ``model_value`` is
scored (D-18), per game both arms predicted, gap = live loss minus frozen loss; WP is the clipped
log loss of the home-win probability (a tie is y = 0, D-31), ATS / O/U the absolute error in points
against the final margin / total. Every expected number below is computed BY HAND from the fixture
scores, never by the module under test.

Fixture rows are written through the real store (``append_rows`` / ``apply_updates`` /
``commit_changes``) into a tmp ledger and read back with ``read_entries``, and silver scores live in
a tmp directory. Nothing here reads or writes a production store or the network (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.refit_preregistration_2026 import LOG_LOSS_CLIP, REFIT_RECIPE_ID
from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.shadow import SHADOW_PIN
from forward_ledger.store import (
    LedgerEntry,
    append_rows,
    apply_updates,
    build_entry,
    commit_changes,
    ledger_path,
    read_entries,
)
from refit.comparison import line_loss, paired_frame, wp_loss
from refit.generations import Generation
from refit.records import append_date_event
from scripts import compare_refit_arms
from tests.unit.test_forward_ledger_store import graded, key_of, make_row
from tests.unit.test_refit_decision_path import (
    GENERATION_ONE_IDS,
    closed_event,
    intent_event,
)

SEASON = 2026

# Generation 1 as an install records it; the bet-rule path is never read by the comparison.
GENERATIONS: tuple[Generation, ...] = (
    Generation(
        number=1,
        ids=GENERATION_ONE_IDS,
        recipe=REFIT_RECIPE_ID,
        bet_rule_path=Path("ledger/refit/bet_rules") / f"{'a' * 64}.json",
        date_index=1,
        list_week=8,
    ),
)

# The fixture games and their final scores (home, away).
SCORES: dict[str, tuple[int | None, int | None]] = {
    "2026_09_KC_BUF": (24, 22),  # margin +2, total 46
    "2026_09_DAL_PHI": (10, 17),  # margin -7
    "2026_09_SF_SEA": (20, 17),  # margin +3
    "2026_09_NYJ_MIA": (27, 13),  # home win
    "2026_09_LAR_ARI": (20, 20),  # a tie
    "2026_09_DEN_LV": (None, None),  # not played yet
}


def games_frame(
    scores: dict[str, tuple[int | None, int | None]] = SCORES,
) -> pd.DataFrame:
    """A silver-``games``-shaped frame holding *scores*."""
    return pd.DataFrame(
        {
            "game_id": list(scores),
            "season": [SEASON] * len(scores),
            "week": [int(game_id[5:7]) for game_id in scores],
            "home_score": [home for home, _away in scores.values()],
            "away_score": [away for _home, away in scores.values()],
        }
    )


def write_silver(
    directory: Path, scores: dict[str, tuple[int | None, int | None]] = SCORES
) -> Path:
    """Write *scores* as ``games.parquet`` under *directory* and return the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    games_frame(scores).to_parquet(directory / "games.parquet", index=False)
    return directory


def live_row(
    game_id: str, target: str, value: float, *, generation: int = 1, **overrides: Any
) -> dict[str, Any]:
    """A live row of *game_id* scored by generation *generation* (0 = the pin, 1 = re-fit)."""
    if generation == 0:
        stamps = {
            "model_artifact_id": SHADOW_PIN.model_ids[target],
            "blend_id": SHADOW_PIN.resolved().blend,
            "recipe_id": SHADOW_PIN.recipe,
        }
    else:
        stamps = {
            "model_artifact_id": GENERATION_ONE_IDS.model_id_for(target),
            "blend_id": GENERATION_ONE_IDS.blend,
            "recipe_id": REFIT_RECIPE_ID,
        }
    return make_row(
        game_id=game_id,
        week=int(game_id[5:7]),
        target=target,
        model_value=value,
        arm="live",
        **{**stamps, **overrides},
    )


def shadow_row(
    game_id: str, target: str, value: float, **overrides: Any
) -> dict[str, Any]:
    """The frozen twin of *game_id*: always the pin."""
    return make_row(
        game_id=game_id,
        week=int(game_id[5:7]),
        target=target,
        model_value=value,
        arm="shadow",
        model_artifact_id=SHADOW_PIN.model_ids[target],
        blend_id=SHADOW_PIN.resolved().blend,
        recipe_id=SHADOW_PIN.recipe,
        **overrides,
    )


def write_pairs(
    ledger: Path, pairs: list[tuple[dict[str, Any], dict[str, Any] | None]]
) -> None:
    """Append every live row, then every shadow twin, in one commit (as production does)."""
    live = [pair[0] for pair in pairs]
    shadow = [pair[1] for pair in pairs if pair[1] is not None]
    append_rows(ledger, [*live, *shadow])


# The three hand-checked ATS pairs (live, frozen) and their losses against the final margin.
#   KC_BUF  margin +2: live 3.0 -> 1.0, frozen 1.0 -> 1.0, gap  0.0
#   DAL_PHI margin -7: live -3.5 -> 3.5, frozen 2.5 -> 9.5, gap -6.0
#   SF_SEA  margin +3: live -1.0 -> 4.0, frozen 2.0 -> 1.0, gap +3.0
# Mean gap = (0.0 - 6.0 + 3.0) / 3 = -1.0.
ATS_PAIRS: tuple[tuple[str, float, float, float, float], ...] = (
    ("2026_09_KC_BUF", 3.0, 1.0, 1.0, 1.0),
    ("2026_09_DAL_PHI", -3.5, 2.5, 3.5, 9.5),
    ("2026_09_SF_SEA", -1.0, 2.0, 4.0, 1.0),
)


def ats_pairs() -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    return [
        (live_row(game_id, "ats", live), shadow_row(game_id, "ats", frozen))
        for game_id, live, frozen, _live_loss, _frozen_loss in ATS_PAIRS
    ]


def null_stamp_entry(seq: int, week: int) -> LedgerEntry:
    """A weeks 3-4 live row as the migration wrote it: NULL stamps, outside any generation."""
    row = make_row(
        game_id=f"2026_{week:02d}_KC_BUF",
        week=week,
        target="ats",
        model_artifact_id=None,
        blend_id=None,
        recipe_id=None,
        fill_convention_id=None,
        upstream_capture_key=None,
        gold_generation_key=None,
        odds_snapshot_digest=None,
        decision_snapshot_digest=None,
        verdict_scope="pre_verdict",
    )
    return build_entry("0" * 64, seq, ENTRY_KIND_ROW, row)


# ---------------------------------------------------------------------------
# Task 1: the paired per-game losses
# ---------------------------------------------------------------------------


def test_hand_computed_gaps(tmp_path: Path) -> None:
    pairs = ats_pairs()
    pairs.append(
        (
            live_row("2026_09_NYJ_MIA", "wp", 0.6),
            shadow_row("2026_09_NYJ_MIA", "wp", 0.5),
        )
    )
    # O/U on KC_BUF, total 46: live 44.5 -> 1.5, frozen 49.0 -> 3.0, gap -1.5.
    pairs.append(
        (
            live_row("2026_09_KC_BUF", "ou", 44.5),
            shadow_row("2026_09_KC_BUF", "ou", 49.0),
        )
    )
    write_pairs(tmp_path, pairs)

    paired = paired_frame(read_entries(tmp_path), games_frame(), GENERATIONS)
    frame = paired.frame.set_index(["game_id", "target"])

    for game_id, _live, _frozen, live_loss, frozen_loss in ATS_PAIRS:
        row = frame.loc[(game_id, "ats")]
        assert row["live_loss"] == live_loss
        assert row["frozen_loss"] == frozen_loss
        assert row["gap"] == live_loss - frozen_loss
        assert row["generation"] == 1
    assert frame.xs("ats", level="target")["gap"].mean() == -1.0

    wp = frame.loc[("2026_09_NYJ_MIA", "wp")]
    assert wp["gap"] == -math.log(0.6) - (-math.log(0.5))
    assert (wp["home_score"], wp["away_score"]) == (27, 13)

    ou = frame.loc[("2026_09_KC_BUF", "ou")]
    assert (ou["live_loss"], ou["frozen_loss"], ou["gap"]) == (1.5, 3.0, -1.5)


def test_tie_is_home_did_not_win(tmp_path: Path) -> None:
    write_pairs(
        tmp_path,
        [
            (
                live_row("2026_09_LAR_ARI", "wp", 0.7),
                shadow_row("2026_09_LAR_ARI", "wp", 0.4),
            )
        ],
    )

    paired = paired_frame(read_entries(tmp_path), games_frame(), GENERATIONS)
    row = paired.frame.iloc[0]

    # y = 0: the loss is -ln(1 - p) on both arms.
    assert row["live_loss"] == -math.log(1.0 - 0.7)
    assert row["frozen_loss"] == -math.log(1.0 - 0.4)
    assert wp_loss(0.7, 0) == -math.log(1.0 - 0.7)


def test_clip_is_applied() -> None:
    sure_home_loss = wp_loss(1.0, 0)
    sure_home_win = wp_loss(0.0, 1)

    assert LOG_LOSS_CLIP == 1e-15
    for loss in (sure_home_loss, sure_home_win):
        assert math.isfinite(loss)
        assert loss == pytest.approx(-math.log(1e-15), rel=1e-3)
    assert line_loss(-3.5, -7.0) == 3.5


def test_aa_weeks_have_zero_gap(tmp_path: Path) -> None:
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for game_id, value_by_target in (
        ("2026_09_KC_BUF", {"wp": 0.61, "ats": 2.75, "ou": 44.5}),
        ("2026_09_DAL_PHI", {"wp": 0.38, "ats": -4.25, "ou": 41.0}),
    ):
        for target, value in value_by_target.items():
            pairs.append(
                (
                    live_row(game_id, target, value, generation=0),
                    shadow_row(game_id, target, value),
                )
            )
    write_pairs(tmp_path, pairs)

    frame = paired_frame(read_entries(tmp_path), games_frame(), GENERATIONS).frame

    assert len(frame) == 6
    assert set(frame["generation"]) == {0}
    assert all(gap == 0.0 for gap in frame["gap"])


def test_paired_only_and_unpaired_weeks(tmp_path: Path) -> None:
    write_pairs(
        tmp_path,
        [
            (live_row("2026_05_KC_BUF", "ats", 1.5, generation=0), None),
            (live_row("2026_06_KC_BUF", "ats", 2.5, generation=0), None),
            (
                live_row("2026_09_KC_BUF", "ats", 3.0),
                shadow_row("2026_09_KC_BUF", "ats", 1.0),
            ),
            (live_row("2026_09_DAL_PHI", "ats", -3.5), None),
        ],
    )
    entries = [null_stamp_entry(0, 3), null_stamp_entry(1, 4), *read_entries(tmp_path)]

    paired = paired_frame(entries, games_frame(), GENERATIONS)

    assert list(paired.frame["game_id"]) == ["2026_09_KC_BUF"]
    assert paired.holes == (("2026_09_DAL_PHI", SEASON, 9, "ats"),)
    assert paired.unpaired_weeks == (5, 6)
    assert paired.excluded["no_generation"] == 2
    assert not set(paired.frame["week"]) & {3, 4, 5, 6}


def test_an_invariant_failure_is_excluded_and_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "ledger"
    write_pairs(
        ledger,
        [
            (
                live_row("2026_09_KC_BUF", "ats", 3.0, generation=0),
                shadow_row("2026_09_KC_BUF", "ats", 3.0),
            ),
            (
                live_row("2026_09_DAL_PHI", "ats", -3.5, generation=0),
                shadow_row(
                    "2026_09_DAL_PHI",
                    "ats",
                    -3.5,
                    decided_at_utc="2026-10-14T21:20:00+00:00",
                ),
            ),
        ],
    )

    paired = paired_frame(read_entries(ledger), games_frame(), GENERATIONS)

    assert list(paired.frame["game_id"]) == ["2026_09_KC_BUF"]
    assert paired.excluded["invariant_failure"] == 1

    code = compare_refit_arms.main(
        [
            "--ledger-dir",
            str(ledger),
            "--silver-dir",
            str(write_silver(tmp_path / "silver")),
        ]
    )
    assert code == 0
    assert "INVARIANT_FAILURES= 1" in capsys.readouterr().out.splitlines()


def test_the_frame_carries_status_and_in_force_payout(tmp_path: Path) -> None:
    live = live_row("2026_09_KC_BUF", "ats", 3.0)
    frozen = shadow_row(
        "2026_09_KC_BUF",
        "ats",
        1.0,
        status="suppressed",
        rejection_reason="edge_below_threshold",
    )
    write_pairs(tmp_path, [(live, frozen)])
    apply_updates(
        tmp_path,
        grading_updates={
            key_of(live): graded("win", 0.9090909090909091, 1.1363636363636365)
        },
    )
    commit_changes(
        tmp_path,
        new_corrections=[
            {
                "game_id": live["game_id"],
                "season": SEASON,
                "week": 9,
                "target": "ats",
                "arm": "live",
                "original_grading_status": "win",
                "original_payout_flat": 0.9090909090909091,
                "prior_grading_status": "win",
                "prior_payout_flat": 0.9090909090909091,
                "prior_realized_units": 1.1363636363636365,
                "prior_correction_seq": None,
                "corrected_grading_status": "loss",
                "corrected_outcome": False,
                "corrected_payout_flat": -1.0,
                "corrected_realized_units": -1.25,
                "realized_value": -4.0,
                "source_dataset": "schedules",
                "source_season": SEASON,
                "source_week": 9,
                "source_sequence": 4,
                "detected_at_utc": "2026-11-10T21:00:00+00:00",
                "corrected_at_utc": "2026-11-10T21:00:05+00:00",
            }
        ],
    )

    row = paired_frame(read_entries(tmp_path), games_frame(), GENERATIONS).frame.iloc[0]

    assert row["live_status"] == "live"
    assert row["live_grading_status"] == "loss"
    assert row["live_payout_flat"] == -1.0
    assert row["frozen_status"] == "suppressed"
    assert row["frozen_grading_status"] == "pending"
    assert pd.isna(row["frozen_payout_flat"])


def test_cli_prints_one_cell(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "ledger"
    append_date_event(ledger, intent_event(GENERATION_ONE_IDS, "a" * 64))
    append_date_event(ledger, closed_event("ran"))
    write_pairs(ledger, ats_pairs())
    silver = write_silver(tmp_path / "silver")

    code = compare_refit_arms.main(
        ["--ledger-dir", str(ledger), "--silver-dir", str(silver)]
    )

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert "COMPARISON_SEASON= 2026" in lines
    assert "HOLES= 0" in lines
    cells = [line for line in lines if line.startswith("CELL= ")]
    assert len(cells) == 1
    assert cells[0].startswith("CELL= ats gen=1 n=3 ")
    assert "mean_gap=-1.000000" in cells[0]
    assert "too few to test" in cells[0]

    ledger_path(ledger).write_bytes(b"not json\n")
    code = compare_refit_arms.main(
        ["--ledger-dir", str(ledger), "--silver-dir", str(silver)]
    )
    out = capsys.readouterr().out
    assert code == 2
    assert out.startswith("LEDGER_UNREADABLE= ")
