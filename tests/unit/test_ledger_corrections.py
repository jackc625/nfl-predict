"""Automatic correction entries from owed ``live_revision_graded`` verdicts (Plan 34-09 Task 2; D-05).

When Phase 32's live detector records that a capture moved an already-graded week
(``correction_owed: true``), every graded live-status ledger row in that week is re-graded under its
own target strategy against the corrected score. Only where the outcome differs from the outcome IN
FORCE is a correction payload produced; applied through ``commit_changes`` it becomes a chained
correction entry, and the row's own grading half is never touched.

Synthetic manifests, games and ledgers under ``tmp_path`` only (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forward_ledger.corrections import (
    CorrectionResult,
    OwedEvent,
    correction_payloads,
    in_force_outcomes,
    owed_correction_events,
)

from data.live_revision import CORRECTION_DISCHARGED_BY
from data.revision_events import RevisionEventClass
from forward_ledger.canonical import CORRECTION_COLUMNS_V1
from forward_ledger.settle import grading_updates, realized_values_from_scores
from forward_ledger.store import (
    append_rows,
    apply_updates,
    commit_changes,
    ledger_head,
    ledger_path,
    read_entries,
    verify_chain,
)
from tests.unit.test_forward_ledger_store import key_of
from tests.unit.test_ledger_settle import STRATEGIES, ats_row, games_frame

GRADED_AT = datetime(2026, 10, 19, 21, 0, tzinfo=UTC)
NOW = datetime(2026, 10, 21, 21, 0, 5, tzinfo=UTC)
NOW_TEXT = "2026-10-21T21:00:05+00:00"

GAME = "2026_W06_KC@BUF"
WIN_PAYOUT = 100 / 110
LOSS_PAYOUT = -1.0


def capture(
    week: int,
    sequence: int,
    captured_at: str,
    *,
    event_class: str = RevisionEventClass.LIVE_REVISION_GRADED,
    owed: bool = True,
    weeks: list[int] | None = None,
    dataset: str = "schedules",
) -> dict[str, Any]:
    """One capture entry of a live manifest, carrying its Phase-32 revision verdict."""
    scope = (
        {
            "season": 2026,
            "weeks": weeks if weeks is not None else [6],
            "dataset": dataset,
            "reason": "upstream restated a graded week",
            "discharged_by": CORRECTION_DISCHARGED_BY,
        }
        if owed
        else None
    )
    return {
        "week": week,
        "sequence": sequence,
        "captured_at_utc": captured_at,
        "sha256": "c" * 64,
        "revision": {
            "dataset": dataset,
            "season": 2026,
            "week": week,
            "sequence": sequence,
            "event_class": str(event_class),
            "correction_owed": owed,
            "correction_owed_scope": scope,
        },
    }


def manifest(*captures: dict[str, Any], dataset: str = "schedules") -> dict[str, Any]:
    return {"season": 2026, "datasets": {dataset: {"captures": list(captures)}}}


EVENT_1 = capture(8, 4, "2026-10-21T21:00:00+00:00")
EVENT_2 = capture(9, 2, "2026-10-28T21:00:00+00:00")


def settled_ledger(ledger_dir: Path, rows: list[dict[str, Any]], margin: float) -> None:
    """Append *rows* and grade every one of them from a home margin of *margin* (24 - x)."""
    append_rows(ledger_dir, rows)
    games = games_frame(
        *((row["game_id"], row["week"], 24.0, 24.0 - margin) for row in rows)
    )
    apply_updates(
        ledger_dir,
        grading_updates=grading_updates(
            read_entries(ledger_dir),
            STRATEGIES,
            realized_values_from_scores(games),
            GRADED_AT,
        ),
    )


def corrected(margin: float, *game_ids: str) -> dict[str, dict[str, float]]:
    """Realized values after a score correction: every named week-6 game ends at home margin *margin*."""
    return realized_values_from_scores(
        games_frame(*((game_id, 6, 24.0, 24.0 - margin) for game_id in game_ids))
    )


def run(
    ledger_dir: Path,
    events_manifest: dict[str, Any],
    realized: dict[str, dict[str, float]],
) -> CorrectionResult:
    return correction_payloads(
        read_entries(ledger_dir),
        owed_correction_events(events_manifest),
        realized,
        STRATEGIES,
        NOW,
    )


def test_owed_event_parsed() -> None:
    informational = capture(
        8,
        3,
        "2026-10-20T21:00:00+00:00",
        event_class=RevisionEventClass.LIVE_REVISION,
        owed=False,
    )
    events = owed_correction_events(manifest(informational, EVENT_1))

    assert events == [
        OwedEvent(
            dataset="schedules",
            season=2026,
            week=8,
            sequence=4,
            captured_at_utc="2026-10-21T21:00:00+00:00",
            scope_weeks=(6,),
        )
    ]


def test_changed_outcome_appends_one_chained_correction(tmp_path: Path) -> None:
    settled_ledger(tmp_path, [ats_row(GAME)], margin=7.0)
    head_before = ledger_head(read_entries(tmp_path))

    result = run(tmp_path, manifest(EVENT_1), corrected(-10.0, GAME))

    assert len(result.payloads) == 1
    payload = result.payloads[0]
    assert tuple(payload) == CORRECTION_COLUMNS_V1
    assert len(payload) == 22
    assert payload["corrected_grading_status"] == "loss"
    assert payload["realized_value"] == -10.0
    assert (
        payload["source_dataset"],
        payload["source_week"],
        payload["source_sequence"],
    ) == (
        "schedules",
        8,
        4,
    )
    assert payload["detected_at_utc"] == "2026-10-21T21:00:00+00:00"
    assert payload["corrected_at_utc"] == NOW_TEXT
    assert result.observations == ()

    commit = commit_changes(tmp_path, new_corrections=result.payloads)

    entries = read_entries(tmp_path)
    assert commit.appended_corrections == 1
    assert [entry.kind for entry in entries] == ["row", "correction"]
    assert ledger_head(entries) != head_before
    assert verify_chain(entries).ok


def test_unchanged_outcome_appends_nothing_and_rerun_too(tmp_path: Path) -> None:
    settled_ledger(tmp_path, [ats_row(GAME)], margin=7.0)

    # The event fired, but the corrected score still covers: nothing is owed to the ledger.
    assert run(tmp_path, manifest(EVENT_1), corrected(10.0, GAME)).payloads == ()
    assert run(tmp_path, manifest(EVENT_1), corrected(10.0, GAME)).payloads == ()

    # Once a changed outcome is recorded, re-running the same event finds it already in force.
    commit_changes(
        tmp_path,
        new_corrections=run(
            tmp_path, manifest(EVENT_1), corrected(-10.0, GAME)
        ).payloads,
    )
    assert run(tmp_path, manifest(EVENT_1), corrected(-10.0, GAME)).payloads == ()


def test_second_correction_supersedes(tmp_path: Path) -> None:
    settled_ledger(tmp_path, [ats_row(GAME)], margin=7.0)
    commit_changes(
        tmp_path,
        new_corrections=run(
            tmp_path, manifest(EVENT_1), corrected(-10.0, GAME)
        ).payloads,
    )

    second = run(tmp_path, manifest(EVENT_1, EVENT_2), corrected(7.0, GAME))
    assert len(second.payloads) == 1
    commit_changes(tmp_path, new_corrections=second.payloads)

    entries = read_entries(tmp_path)
    in_force = in_force_outcomes(entries)[key_of(ats_row(GAME))]
    assert in_force.grading_status == "win"
    assert in_force.outcome is True
    assert in_force.corrected is True
    assert in_force.correction_seq == entries[-1].seq == 2


def test_win_loss_win_history_names_the_prior_state(tmp_path: Path) -> None:
    settled_ledger(tmp_path, [ats_row(GAME)], margin=7.0)

    first = run(tmp_path, manifest(EVENT_1), corrected(-10.0, GAME)).payloads
    commit_changes(tmp_path, new_corrections=first)
    second = run(tmp_path, manifest(EVENT_1, EVENT_2), corrected(7.0, GAME)).payloads
    commit_changes(tmp_path, new_corrections=second)

    entries = read_entries(tmp_path)
    entry_a, entry_b = entries[1].immutable, entries[2].immutable
    assert entry_a["original_grading_status"] == "win"
    assert entry_a["prior_grading_status"] == "win"
    assert entry_a["prior_payout_flat"] == WIN_PAYOUT
    assert entry_a["prior_correction_seq"] is None
    assert entry_a["corrected_grading_status"] == "loss"
    assert entry_a["corrected_payout_flat"] == LOSS_PAYOUT

    # The second entry replaces entry A's LOSS, so it reads loss -> win, never win -> win.
    assert entry_b["original_grading_status"] == "win"
    assert entry_b["original_payout_flat"] == WIN_PAYOUT
    assert entry_b["prior_grading_status"] == "loss"
    assert entry_b["prior_payout_flat"] == entry_a["corrected_payout_flat"]
    assert entry_b["prior_realized_units"] == entry_a["corrected_realized_units"]
    assert entry_b["prior_correction_seq"] == entries[1].seq == 1
    assert entry_b["corrected_grading_status"] == "win"
    assert entry_b["source_week"] == 9
    assert entry_b["source_sequence"] == 2


def test_original_grading_half_untouched(tmp_path: Path) -> None:
    settled_ledger(tmp_path, [ats_row(GAME)], margin=7.0)
    row_line_before = ledger_path(tmp_path).read_bytes().split(b"\n")[0]
    grading_before = read_entries(tmp_path)[0].grading

    commit_changes(
        tmp_path,
        new_corrections=run(
            tmp_path, manifest(EVENT_1), corrected(-10.0, GAME)
        ).payloads,
    )

    assert ledger_path(tmp_path).read_bytes().split(b"\n")[0] == row_line_before
    assert read_entries(tmp_path)[0].grading == grading_before


def test_withdrawn_score_appends_nothing(tmp_path: Path) -> None:
    settled_ledger(tmp_path, [ats_row(GAME)], margin=7.0)
    withdrawn = realized_values_from_scores(games_frame((GAME, 6, None, None)))

    result = run(tmp_path, manifest(EVENT_1), withdrawn)

    assert result.payloads == ()
    assert result.observations == (
        {
            "key": key_of(ats_row(GAME)),
            "week": 6,
            "source": {
                "dataset": "schedules",
                "season": 2026,
                "week": 8,
                "sequence": 4,
            },
        },
    )


def test_pending_and_suppressed_rows_ignored(tmp_path: Path) -> None:
    in_scope = ats_row(GAME)
    suppressed = ats_row(
        "2026_W06_DAL@PHI", status="suppressed", rejection_reason="below_ev_floor"
    )
    other_week = ats_row("2026_W05_SF@SEA", week=5)
    settled_ledger(tmp_path, [in_scope, suppressed, other_week], margin=7.0)
    pending = ats_row("2026_W06_NYJ@MIA")
    append_rows(tmp_path, [pending])
    flipped = realized_values_from_scores(
        games_frame(
            *(
                (row["game_id"], row["week"], 24.0, 34.0)
                for row in (in_scope, suppressed, other_week, pending)
            )
        )
    )

    result = run(tmp_path, manifest(EVENT_1), flipped)

    assert [
        tuple(payload[name] for name in ("game_id", "season", "week", "target", "arm"))
        for payload in result.payloads
    ] == [key_of(in_scope)]
    assert all(
        payload["corrected_grading_status"] == "loss" for payload in result.payloads
    )


def test_correction_keys_include_arm(tmp_path: Path) -> None:
    live = ats_row(GAME)
    shadow = ats_row(GAME, arm="shadow")
    settled_ledger(tmp_path, [live, shadow], margin=7.0)

    result = run(tmp_path, manifest(EVENT_1), corrected(-10.0, GAME))

    assert sorted(payload["arm"] for payload in result.payloads) == ["live", "shadow"]
    commit_changes(tmp_path, new_corrections=result.payloads)
    in_force = in_force_outcomes(read_entries(tmp_path))
    assert in_force[key_of(live)].grading_status == "loss"
    assert in_force[key_of(shadow)].grading_status == "loss"
    assert (
        in_force[key_of(live)].correction_seq != in_force[key_of(shadow)].correction_seq
    )
