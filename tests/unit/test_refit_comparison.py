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

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.profitability_2025 import benjamini_hochberg_adjusted
from backtest.refit_preregistration_2026 import (
    AA_FIRST_WEEK,
    ALPHA,
    LOG_LOSS_CLIP,
    NO_CLEAN_WEEK_RULE,
    REFIT_RECIPE_ID,
    ROLLUP_WORDS,
)
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
from models.artifacts import ResolvedArtifacts
from refit import comparison
from refit.comparison import (
    EXCLUDED_REASONS,
    FRAME_COLUMNS,
    ComparisonResult,
    PairedFrame,
    decide,
    line_loss,
    paired_frame,
    rollup_word,
    wp_loss,
)
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
    assert "EXCLUDED_INVARIANT_FAILURE= 1" in capsys.readouterr().out.splitlines()


def test_cli_prints_every_exclusion_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WR-D01: a pair left out for want of a final score is counted in the report, not dropped."""
    ledger = tmp_path / "ledger"
    write_pairs(
        ledger,
        [
            (
                live_row("2026_09_KC_BUF", "ats", 3.0, generation=0),
                shadow_row("2026_09_KC_BUF", "ats", 3.0),
            ),
            (
                live_row("2026_09_DEN_LV", "ats", 1.5, generation=0),
                shadow_row("2026_09_DEN_LV", "ats", 1.5),
            ),
        ],
    )

    code = compare_refit_arms.main(
        [
            "--ledger-dir",
            str(ledger),
            "--silver-dir",
            str(write_silver(tmp_path / "silver")),
        ]
    )

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    excluded = [line for line in lines if line.startswith("EXCLUDED_")]
    assert excluded == [
        "EXCLUDED_NO_GENERATION= 0",
        "EXCLUDED_DRESS_REHEARSAL= 0",
        "EXCLUDED_INVARIANT_FAILURE= 0",
        "EXCLUDED_NO_MODEL_VALUE= 0",
        "EXCLUDED_NO_FINAL_SCORE= 1",
    ]
    first_cell = next(
        index for index, line in enumerate(lines) if line.startswith("CELL= ")
    )
    assert lines.index(excluded[-1]) < first_cell


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
    (ats,) = [line for line in cells if line.startswith("CELL= ats ")]
    assert ats.startswith("CELL= ats gen=1 n=3 ")
    assert "mean_gap=-1.000000" in ats
    assert "too few to test" in ats
    # The installed generation is reported for every target, even with no paired game yet.
    assert [line.split()[1:4] for line in cells] == [
        ["wp", "gen=1", "n=0"],
        ["ats", "gen=1", "n=3"],
        ["ou", "gen=1", "n=0"],
    ]

    ledger_path(ledger).write_bytes(b"not json\n")
    code = compare_refit_arms.main(
        ["--ledger-dir", str(ledger), "--silver-dir", str(silver)]
    )
    out = capsys.readouterr().out
    assert code == 2
    assert out.startswith("LEDGER_UNREADABLE= ")


def test_cli_refuses_a_broken_chain_and_a_missing_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WR-D02: no verdict over a ledger whose chain does not verify; a missing store is exit 2."""
    ledger = tmp_path / "ledger"
    write_pairs(ledger, ats_pairs())
    silver = write_silver(tmp_path / "silver")

    code = compare_refit_arms.main(
        ["--ledger-dir", str(ledger), "--silver-dir", str(tmp_path / "no_silver")]
    )
    out = capsys.readouterr().out
    assert code == 2
    assert out.startswith("STORE_UNREADABLE= FileNotFoundError: ")

    text = ledger_path(ledger).read_text(encoding="utf-8")
    tampered = text.replace('"model_value":3.0', '"model_value":9.0', 1)
    assert tampered != text
    ledger_path(ledger).write_text(tampered, encoding="utf-8")

    code = compare_refit_arms.main(
        ["--ledger-dir", str(ledger), "--silver-dir", str(silver)]
    )
    out = capsys.readouterr().out
    assert code == 2
    assert out.startswith("LEDGER_UNREADABLE= the chain does not verify")
    assert "VERDICT=" not in out


# ---------------------------------------------------------------------------
# Task 2: the pre-registered decision
# ---------------------------------------------------------------------------

HELPED, HURT, MIXED, NO_DETECTABLE_DIFFERENCE = ROLLUP_WORDS


def installed(count: int) -> tuple[Generation, ...]:
    """Generations 1..*count* with distinct ids (decide reads only their numbers)."""
    return tuple(
        Generation(
            number=number,
            ids=ResolvedArtifacts(
                wp=f"wp_gen{number}",
                ats=f"ats_gen{number}",
                ou=f"ou_gen{number}",
                blend=f"blend_gen{number}",
                converter="market_probability_20260923_195443",
            ),
            recipe=REFIT_RECIPE_ID,
            bet_rule_path=Path("ledger/refit/bet_rules") / f"{number:064d}.json",
            date_index=number,
            list_week=8,
        )
        for number in range(1, count + 1)
    )


def spread_gaps(n: int, shift: float = 0.0) -> list[float]:
    """*n* gaps with a non-zero spread around *shift*."""
    return [0.2 * ((index % 5) - 2) + shift for index in range(n)]


def alternating_gaps(n: int, mean: float) -> list[float]:
    """*n* (even) gaps of exactly *mean* +/- 1: mean *mean*, sample sd sqrt(n / (n - 1))."""
    return [mean + (1.0 if index % 2 else -1.0) for index in range(n)]


def synthetic_paired(cells: dict[tuple[str, int], Sequence[float]]) -> PairedFrame:
    """A paired frame holding one row per gap of each (target, generation) cell.

    Every live row is a live bet graded a win at -110; every frozen twin is suppressed.
    """
    records = [
        {
            "game_id": f"2026_{9 + generation:02d}_G{index:03d}",
            "season": SEASON,
            "week": 9 + generation,
            "target": target,
            "generation": generation,
            "live_value": 0.0,
            "frozen_value": 0.0,
            "home_score": 20,
            "away_score": 17,
            "live_loss": 1.0 + gap,
            "frozen_loss": 1.0,
            "gap": gap,
            "live_status": "live",
            "live_grading_status": "win",
            "live_payout_flat": 0.9090909090909091,
            "frozen_status": "suppressed",
            "frozen_grading_status": "pending",
            "frozen_payout_flat": None,
        }
        for (target, generation), gaps in cells.items()
        for index, gap in enumerate(gaps)
    ]
    return PairedFrame(
        frame=pd.DataFrame.from_records(records, columns=list(FRAME_COLUMNS)),
        holes=(),
        unpaired_weeks=(),
        excluded=dict.fromkeys(EXCLUDED_REASONS, 0),
    )


@pytest.fixture
def bh_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[float], int]]:
    """Every (p-values, m) the decision hands the real BH implementation."""
    calls: list[tuple[list[float], int]] = []

    def spy(pvalues: Sequence[float], m: int) -> list[float]:
        calls.append((list(pvalues), m))
        return benjamini_hochberg_adjusted(pvalues, m)

    monkeypatch.setattr(comparison, "benjamini_hochberg_adjusted", spy)
    return calls


def cell_of(result: ComparisonResult, target: str, generation: int) -> Any:
    (cell,) = [
        cell
        for cell in result.cells
        if (cell.target, cell.generation) == (target, generation)
    ]
    return cell


def test_bh_family_is_the_tested_cells(
    bh_calls: list[tuple[list[float], int]],
) -> None:
    paired = synthetic_paired(
        {
            ("ats", 1): spread_gaps(58, -0.05),
            ("ats", 2): spread_gaps(61, 0.03),
            ("ats", 3): spread_gaps(42, -0.02),
            ("ats", 4): spread_gaps(3, -0.5),
        }
    )

    result = decide(paired, installed(4))

    assert result.family_size == 3
    assert len(bh_calls) == 1
    pvalues, m = bh_calls[0]
    assert m == 3
    assert len(pvalues) == 3
    small = cell_of(result, "ats", 4)
    assert (small.n, small.p, small.adjusted_p) == (3, None, None)
    assert small.status == "too few to test"
    for generation in (1, 2, 3):
        assert cell_of(result, "ats", generation).adjusted_p is not None


def test_an_all_zero_eligible_cell_counts_in_the_family(
    bh_calls: list[tuple[list[float], int]],
) -> None:
    paired = synthetic_paired(
        {("ats", 1): [0.0] * 40, ("ats", 2): spread_gaps(58, -0.04)}
    )

    result = decide(paired, installed(2))

    zero = cell_of(result, "ats", 1)
    assert zero.status == "no difference (all paired gaps zero)"
    assert (zero.n, zero.p, zero.adjusted_p) == (40, None, None)
    assert bh_calls == [([cell_of(result, "ats", 2).p], 2)]
    tested = cell_of(result, "ats", 2)
    assert tested.adjusted_p == min(1.0, 2 * tested.p)


def test_rollup_words() -> None:
    assert rollup_word(["better", "no difference"]) == HELPED
    assert rollup_word(["worse", "too few to test"]) == HURT
    assert rollup_word(["better", "worse"]) == MIXED
    assert rollup_word(["no difference", "too few to test"]) == NO_DETECTABLE_DIFFERENCE
    assert rollup_word([]) == NO_DETECTABLE_DIFFERENCE

    # Wired through decide: a clearly lower live loss is "better" and the target HELPED.
    result = decide(
        synthetic_paired(
            {("wp", 1): spread_gaps(40, -1.0), ("ats", 1): spread_gaps(40, 1.0)}
        ),
        installed(1),
    )
    assert cell_of(result, "wp", 1).result == "better"
    assert cell_of(result, "ats", 1).result == "worse"
    assert result.verdicts == {
        "wp": HELPED,
        "ats": HURT,
        "ou": NO_DETECTABLE_DIFFERENCE,
    }


def test_generation_zero_is_reported_never_tested(
    bh_calls: list[tuple[list[float], int]],
) -> None:
    paired = synthetic_paired({("wp", 0): [0.0] * 40})

    result = decide(paired, ())

    cell = cell_of(result, "wp", 0)
    assert (cell.n, cell.mean_gap, cell.p, cell.adjusted_p) == (40, 0.0, None, None)
    assert result.family_size == 0
    assert bh_calls == []
    (line,) = [
        line
        for line in compare_refit_arms.report_lines(paired, result)
        if line.startswith("CELL= ")
    ]
    assert line.startswith("CELL= wp gen=0 n=40 ")
    assert "mean_gap=0.000000" in line
    assert "p=none" in line


def test_no_refit_prints_the_end_rule_not_a_verdict_word() -> None:
    """WR-D03: with no generation >= 1 the command states the D-33 end rule, never a roll-up word."""
    paired = synthetic_paired({("ats", 0): [0.0] * 40})

    lines = compare_refit_arms.report_lines(paired, decide(paired, ()))

    verdicts = [line for line in lines if line.startswith("VERDICT= ")]
    assert verdicts == [
        "VERDICT= none (no re-fit generation is installed; the end rule: "
        f"{NO_CLEAN_WEEK_RULE})"
    ]
    assert not any(word in line for line in lines for word in ROLLUP_WORDS)


# One generation-0 pair in the week-7 dress rehearsal and one in the first A/A week (D-38, D-14).
DRESS_GAME: str = f"2026_{AA_FIRST_WEEK - 1:02d}_KC_BUF"
AA_GAME: str = f"2026_{AA_FIRST_WEEK:02d}_KC_BUF"
DRESS_SCORES: dict[str, tuple[int | None, int | None]] = {
    DRESS_GAME: (27, 24),  # margin +3
    AA_GAME: (17, 20),  # margin -3
}


def write_dress_and_aa_pairs(ledger: Path) -> None:
    """The two generation-0 ATS pairs, both arms graded: dress rehearsal win, A/A week loss."""
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = [
        (
            live_row(game_id, "ats", 1.0, generation=0),
            shadow_row(game_id, "ats", 1.0),
        )
        for game_id in (DRESS_GAME, AA_GAME)
    ]
    write_pairs(ledger, pairs)
    outcome = {
        DRESS_GAME: graded("win", 0.9090909090909091, 1.1363636363636365),
        AA_GAME: graded("loss", -1.0, -1.25),
    }
    apply_updates(
        ledger,
        grading_updates={
            key_of(row): outcome[row["game_id"]]
            for pair in pairs
            for row in pair
            if row is not None
        },
    )


def test_the_dress_rehearsal_is_excluded_and_counted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """WR-D04 (D-38): a week-7 pair reaches no cell and is counted under its own reason."""
    ledger = tmp_path / "ledger"
    write_dress_and_aa_pairs(ledger)

    paired = paired_frame(read_entries(ledger), games_frame(DRESS_SCORES), ())

    assert list(paired.frame["game_id"]) == [AA_GAME]
    assert paired.excluded["dress_rehearsal"] == 1
    code = compare_refit_arms.main(
        [
            "--ledger-dir",
            str(ledger),
            "--silver-dir",
            str(write_silver(tmp_path / "silver", DRESS_SCORES)),
        ]
    )
    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert "EXCLUDED_DRESS_REHEARSAL= 1" in lines
    (cell,) = [line for line in lines if line.startswith("CELL= ")]
    assert cell.startswith("CELL= ats gen=0 n=1 ")


def test_the_frozen_arm_has_a_generation_zero_roi_without_the_dress_rehearsal(
    tmp_path: Path,
) -> None:
    """WR-D04 (D-14): both arms report a generation-0 ROI line over the A/A weeks only."""
    write_dress_and_aa_pairs(tmp_path)

    paired = paired_frame(read_entries(tmp_path), games_frame(DRESS_SCORES), ())
    result = decide(paired, ())

    # Only the A/A-week loss counts; the dress-rehearsal win would have made it one win, one loss.
    assert [
        (roi.generation, roi.arm, roi.bets, roi.flat_roi) for roi in result.roi
    ] == [
        (0, "live", 1, -1.0),
        (0, "frozen", 1, -1.0),
    ]


def test_pooled_is_after_and_labelled() -> None:
    # Each period alone misses alpha (t about 1.88 on 30 games); pooled over 60 it does not.
    paired = synthetic_paired(
        {
            ("ou", 0): [0.0] * 6,
            ("ou", 2): alternating_gaps(30, -0.35),
            ("ou", 1): alternating_gaps(30, -0.35),
        }
    )

    result = decide(paired, installed(2))
    lines = compare_refit_arms.report_lines(paired, result)

    kinds = [line.split("= ", 1)[0] for line in lines]
    order = ["CELL", "VERDICT", "POOLED_SECONDARY", "ROI_SECONDARY"]
    positions = [order.index(kind) for kind in kinds if kind in order]
    assert positions == sorted(positions)
    assert set(order) <= set(kinds)
    generations = [
        int(line.split("gen=")[1].split()[0])
        for line in lines
        if line.startswith("CELL= ")
    ]
    assert generations == sorted(generations)
    assert generations[0] == 0

    pooled = {item.target: item for item in result.pooled}["ou"]
    assert pooled.n == 60
    assert pooled.p is not None and pooled.p < ALPHA
    for generation in (1, 2):
        cell = cell_of(result, "ou", generation)
        assert cell.adjusted_p is not None and cell.adjusted_p > ALPHA
        assert cell.result == "no difference"
    assert result.verdicts["ou"] == NO_DETECTABLE_DIFFERENCE
    verdict_line = next(line for line in lines if line.startswith("VERDICT= ou "))
    assert verdict_line.endswith(NO_DETECTABLE_DIFFERENCE)

    pooled_line = next(
        line for line in lines if line.startswith("POOLED_SECONDARY= ou ")
    )
    assert "secondary" in pooled_line
    assert "p=" in pooled_line and "p=none" not in pooled_line
    roi_lines = [line for line in lines if line.startswith("ROI_SECONDARY= ")]
    assert roi_lines and all("report-only" in line for line in roi_lines)
    assert (
        "ROI_SECONDARY= gen=1 arm=live bets=30 flat_roi=0.909091 (report-only, no test)"
        in lines
    )
    assert (
        "ROI_SECONDARY= gen=1 arm=frozen bets=0 flat_roi=none (report-only, no test)"
        in lines
    )


def test_no_clv_in_the_deciding_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "ledger"
    append_date_event(ledger, intent_event(GENERATION_ONE_IDS, "a" * 64))
    append_date_event(ledger, closed_event("ran"))
    pairs = ats_pairs()
    write_pairs(ledger, pairs)
    rows = [row for pair in pairs for row in pair if row is not None]
    apply_updates(
        ledger,
        grading_updates={
            key_of(row): {
                **graded("win", 0.9090909090909091, 1.1363636363636365),
                "clv": 0.031,
            }
            for row in rows
        },
        closing_updates={
            key_of(row): {
                "closing_line": -3.0,
                "closing_odds": -110.0,
                "closing_sportsbook": "draftkings",
                "closing_captured_at": "2026-11-08T17:55:00+00:00",
                "forward_clv": 0.042,
            }
            for row in rows
        },
    )
    assert any(
        (entry.closing or {}).get("forward_clv") == 0.042
        for entry in read_entries(ledger)
    )

    code = compare_refit_arms.main(
        [
            "--ledger-dir",
            str(ledger),
            "--silver-dir",
            str(write_silver(tmp_path / "silver")),
        ]
    )
    lines = capsys.readouterr().out.splitlines()

    assert code == 0
    assert any(line.startswith("CELL= ats gen=1 ") for line in lines)
    assert all("clv" not in line.lower() for line in lines)
    paired = paired_frame(read_entries(ledger), games_frame(), GENERATIONS)
    assert all("clv" not in column.lower() for column in paired.frame.columns)
    serialized = json.dumps(decide(paired, GENERATIONS).as_dict())
    assert "clv" not in serialized.lower()
