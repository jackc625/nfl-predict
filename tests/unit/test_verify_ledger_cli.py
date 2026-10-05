"""The ledger verify CLI end to end (Phase 34, Plan 34-10; LDGR-05, LDGR-06, LDGR-10, LDGR-11, D-19).

The verify CLI is the owner's one check that the forward ledger is what it claims to be: the chain
recomputes from its genesis constant, the head matches the locally committed anchor (authoritative,
so a truncated tail is caught), the GitHub anchor agrees (its lag is a warning, a disagreement a
failure), and the backup's lag is reported.

Every repository here is built under ``tmp_path`` -- the "public" repository carrying the
``ledger-anchor`` branch, a local BARE repository standing in for GitHub, and the ledger directory
itself. No test reads the real repository's refs, contacts GitHub, or touches the production
``ledger/``, ``data/`` or ``outputs/`` (COLD-05).

Most tests drive ``scripts.verify_ledger.main`` in-process (the same argv the owner types); the
empty-ledger test runs it as a REAL subprocess so the module entry point itself is exercised.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import subprocess
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from forward_ledger.anchor import write_anchor_commit
from forward_ledger.backup import commit_backup, ensure_backup_repo
from forward_ledger.canonical import (
    CORRECTION_COLUMNS_V1,
    ENTRY_KIND_CORRECTION,
    ENTRY_KIND_ROW,
    GENESIS_HASH,
)
from forward_ledger.remote_config import BACKUP_PUSHED_REF
from forward_ledger.store import (
    LEDGER_STAMP_COLUMNS,
    LedgerEntry,
    build_entry,
    ledger_path,
    write_entries,
)
from forward_ledger.transport import run_git
from forward_ledger.verify import verify_ledger
from scripts.verify_ledger import main, report_lines
from tests.unit.test_forward_ledger_store import key_of, make_row
from tests.unit.test_ledger_anchor import git_out, make_bare, make_repo

REPO_ROOT = Path(__file__).resolve().parents[2]

GRADED_AT = "2026-10-19T21:00:00+00:00"

# The grading halves an ATS home-cover bet at a slipped -3.0, priced -110, staked 1.25 units,
# settles to. Typed out, not computed by the grader under test: 100/110 per unit on a win.
WIN: dict[str, Any] = {
    "grading_status": "win",
    "outcome": True,
    "clv": 0.0123,
    "payout_flat": 0.9090909090909091,
    "realized_units": 1.1363636363636365,
    "graded_at": GRADED_AT,
}
LOSS: dict[str, Any] = {
    "grading_status": "loss",
    "outcome": False,
    "clv": 0.0123,
    "payout_flat": -1.0,
    "realized_units": -1.25,
    "graded_at": GRADED_AT,
}
PUSH: dict[str, Any] = {
    "grading_status": "push",
    "outcome": None,
    "clv": 0.0123,
    "payout_flat": 0.0,
    "realized_units": 0.0,
    "graded_at": GRADED_AT,
}
PENDING: dict[str, Any] = {
    "grading_status": "pending",
    "outcome": None,
    "clv": None,
    "payout_flat": None,
    "realized_units": None,
    "graded_at": None,
}

# The verdict-scope declaration is injected as an in-memory module (the
# tests/unit/test_ledger_declarations.py idiom); the absent name is never importable.
ABSENT_DECLARATION = "tests_absent_verdict_scope_for_verify"
FAKE_DECLARATION = "tests_fake_verdict_scope_for_verify"

# (game_id, week, home_score, away_score): week 6 settles a win (margin 7 > -3), a loss
# (margin -10) and a push (margin exactly -3); week 7 is not played yet.
WEEK6_WIN = "2026_06_KC_BUF"
WEEK6_LOSS = "2026_06_DAL_PHI"
WEEK6_PUSH = "2026_06_SF_SEA"
WEEK7_A = "2026_07_NYJ_MIA"
WEEK7_B = "2026_07_ARI_ATL"
SCORES: tuple[tuple[str, int, float | None, float | None], ...] = (
    (WEEK6_WIN, 6, 24.0, 17.0),
    (WEEK6_LOSS, 6, 10.0, 20.0),
    (WEEK6_PUSH, 6, 17.0, 20.0),
    (WEEK7_A, 7, None, None),
    (WEEK7_B, 7, None, None),
)


# ---------------------------------------------------------------------------
# Fixture builders, all confined to tmp_path
# ---------------------------------------------------------------------------


def ats_row(game_id: str, week: int = 6, **overrides: Any) -> dict[str, Any]:
    """A live ATS ledger row betting the home cover at a slipped -3.0, priced -110."""
    return make_row(
        game_id=game_id,
        week=week,
        target="ats",
        bet_side="home_cover",
        slipped_line=-3.0,
        **overrides,
    )


def write_ledger(
    ledger_dir: Path, items: Sequence[tuple[str, Mapping[str, Any], Any]]
) -> list[LedgerEntry]:
    """Chain *items* from genesis and write them; each is ``(kind, immutable, grading)``."""
    entries: list[LedgerEntry] = []
    previous = GENESIS_HASH
    for seq, (kind, immutable, grading) in enumerate(items):
        if kind == ENTRY_KIND_ROW:
            entry = build_entry(
                previous, seq, kind, immutable, grading=dict(grading or PENDING)
            )
        else:
            entry = build_entry(previous, seq, ENTRY_KIND_CORRECTION, immutable)
        entries.append(entry)
        previous = entry.chain_hash
    write_entries(ledger_dir, entries)
    return entries


def two_week_items() -> list[tuple[str, Mapping[str, Any], Any]]:
    """Week 6 settled (win, loss, push) and week 7 pending: five row entries."""
    return [
        (ENTRY_KIND_ROW, ats_row(WEEK6_WIN), WIN),
        (ENTRY_KIND_ROW, ats_row(WEEK6_LOSS), LOSS),
        (ENTRY_KIND_ROW, ats_row(WEEK6_PUSH), PUSH),
        (ENTRY_KIND_ROW, ats_row(WEEK7_A, week=7), PENDING),
        (ENTRY_KIND_ROW, ats_row(WEEK7_B, week=7), PENDING),
    ]


def head_at(entries: Sequence[LedgerEntry], count: int) -> str:
    """The chain head after the first *count* entries."""
    return GENESIS_HASH if count == 0 else entries[count - 1].chain_hash


def anchored_repo(
    path: Path, entries: Sequence[LedgerEntry], count: int | None = None
) -> Path:
    """A public repository whose ``ledger-anchor`` commits the head after *count* entries."""
    repo = make_repo(path)
    rows = len(entries) if count is None else count
    write_anchor_commit(repo, head_at(entries, rows), rows)
    return repo


def bare_remote(
    path: Path, head_hash: str | None = None, count: int | None = None
) -> Path:
    """A local bare repository standing in for GitHub, optionally carrying an anchor."""
    remote = make_bare(path)
    if head_hash is not None and count is not None:
        write_anchor_commit(remote, head_hash, count)
    return remote


def cli_args(ledger_dir: Path, repo: Path, *, remote: Path | None = None) -> list[str]:
    """The argv the owner would type; a missing *remote* means ``--skip-remote``."""
    args = ["--ledger-dir", str(ledger_dir), "--repo-dir", str(repo)]
    if remote is None:
        args.append("--skip-remote")
    else:
        args += ["--remote-url", remote.as_posix()]
    return args


def run_cli(
    capsys: pytest.CaptureFixture[str],
    args: Sequence[str],
    *,
    declaration: str = ABSENT_DECLARATION,
) -> tuple[int, str]:
    """``main(args)`` in-process: its exit code and everything it printed.

    *declaration* names the verdict-scope module the run loads -- absent by default, so a test
    never depends on whether the real declaration has been committed yet. An argparse refusal
    exits through ``SystemExit``; its code is the CLI's exit code.
    """
    try:
        code = main(list(args), verdict_scope_module=declaration)
    except SystemExit as exit_:
        code = exit_.code if isinstance(exit_.code, int) else 2
    return code, capsys.readouterr().out


def two_week_setup(
    tmp_path: Path, *, anchored: int | None = None
) -> tuple[Path, Path, list[LedgerEntry]]:
    """The two-week ledger and a public repo anchored at *anchored* entries (the head by default)."""
    ledger = tmp_path / "ledger"
    entries = write_ledger(ledger, two_week_items())
    repo = anchored_repo(tmp_path / "public", entries, anchored)
    return ledger, repo, entries


def key_text(row: Mapping[str, Any]) -> str:
    return "|".join(str(part) for part in key_of(dict(row)))


# ---------------------------------------------------------------------------
# Task 1: chain, local anchor (authoritative), remote anchor lag, backup lag
# ---------------------------------------------------------------------------


def test_clean_ledger_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger, repo, entries = two_week_setup(tmp_path)
    remote = bare_remote(tmp_path / "remote.git", entries[-1].chain_hash, len(entries))

    code, out = run_cli(capsys, cli_args(ledger, repo, remote=remote))

    assert code == 0, out
    assert "LEDGER_ENTRIES= 5" in out
    assert f"HEAD_HASH= {entries[-1].chain_hash}" in out
    assert "CHAIN_OK= True" in out
    assert "LOCAL_ANCHOR_ROWS= 5" in out
    assert "LOCAL_ANCHOR_OK= True" in out
    assert "UNANCHORED_ENTRIES= 0" in out
    assert "REMOTE_ANCHOR_BEHIND_BY= 0" in out
    assert "LOCAL_RESULT= PASS" in out
    assert "EXTERNAL_RESULT= VERIFIED" in out
    assert out.rstrip().splitlines()[-1] == "VERIFY_RESULT= PASS"


def test_empty_ledger_passes_without_anchor(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    repo = make_repo(tmp_path / "public")

    completed = subprocess.run(
        [sys.executable, "-m", "scripts.verify_ledger", *cli_args(ledger, repo)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "LEDGER_ENTRIES= 0" in completed.stdout
    assert f"HEAD_HASH= {GENESIS_HASH}" in completed.stdout
    assert "LOCAL_ANCHOR_OK= True" in completed.stdout
    assert "EXTERNAL_RESULT= SKIPPED" in completed.stdout
    assert "VERIFY_RESULT= PASS" in completed.stdout


def test_one_row_ledger_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "ledger"
    entries = write_ledger(ledger, [(ENTRY_KIND_ROW, ats_row(WEEK7_A, week=7), None)])
    repo = anchored_repo(tmp_path / "public", entries)

    code, out = run_cli(capsys, cli_args(ledger, repo))

    assert code == 0, out
    assert "LEDGER_ENTRIES= 1" in out
    assert "LOCAL_ANCHOR_ROWS= 1" in out
    assert "VERIFY_RESULT= PASS" in out


def test_edited_byte_names_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger, repo, _ = two_week_setup(tmp_path)
    path = ledger_path(ledger)
    lines = path.read_bytes().split(b"\n")
    # One character inside an immutable value of the SETTLED seq-0 row: -3.2 -> -3.3.
    assert lines[0].count(b'"model_value":-3.2,') == 1
    lines[0] = lines[0].replace(b'"model_value":-3.2,', b'"model_value":-3.3,')
    path.write_bytes(b"\n".join(lines))

    code, out = run_cli(capsys, cli_args(ledger, repo))

    assert code == 1, out
    assert "CHAIN_OK= False" in out
    assert "FIRST_BROKEN_SEQ= 0" in out
    assert f"FIRST_BROKEN_KEY= {key_text(ats_row(WEEK6_WIN))}" in out
    assert "VERIFY_RESULT= FAIL" in out


def test_deleted_row_names_first_broken(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger, repo, _ = two_week_setup(tmp_path)
    path = ledger_path(ledger)
    lines = path.read_bytes().split(b"\n")
    del lines[2]  # the week-6 push row, seq 2
    path.write_bytes(b"\n".join(lines))

    code, out = run_cli(capsys, cli_args(ledger, repo))

    assert code == 1, out
    assert "CHAIN_OK= False" in out
    # Position 2 now holds the entry stored as seq 3: the first one that no longer chains.
    assert "FIRST_BROKEN_SEQ= 2" in out
    assert f"FIRST_BROKEN_KEY= {key_text(ats_row(WEEK7_A, week=7))}" in out


def test_truncated_tail_names_head_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger, repo, entries = two_week_setup(tmp_path)
    write_entries(ledger, entries[:-2])

    code, out = run_cli(capsys, cli_args(ledger, repo))

    assert code == 1, out
    # The truncated chain still verifies; only the anchor can tell.
    assert "CHAIN_OK= True" in out
    assert "LEDGER_ENTRIES= 3" in out
    assert "LOCAL_ANCHOR_ROWS= 5" in out
    assert "LOCAL_ANCHOR_OK= False" in out
    assert "ledger shorter than the anchored row count" in out
    assert "LOCAL_RESULT= FAIL" in out
    assert "VERIFY_RESULT= FAIL" in out


def test_unanchored_rows_warn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger, repo, _ = two_week_setup(tmp_path, anchored=3)

    code, out = run_cli(capsys, cli_args(ledger, repo))

    assert code == 0, out
    assert "LOCAL_ANCHOR_ROWS= 3" in out
    assert "LOCAL_ANCHOR_OK= True" in out
    assert "UNANCHORED_ENTRIES= 2" in out
    assert "WARNING=" in out
    assert "VERIFY_RESULT= PASS" in out


def test_remote_one_run_behind_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger, repo, entries = two_week_setup(tmp_path)
    remote = bare_remote(tmp_path / "remote.git", head_at(entries, 2), 2)

    code, out = run_cli(capsys, cli_args(ledger, repo, remote=remote))

    assert code == 0, out
    assert "REMOTE_ANCHOR_BEHIND_BY= 3" in out
    assert "LOCAL_RESULT= PASS" in out
    assert "EXTERNAL_RESULT= BEHIND" in out
    assert "VERIFY_RESULT= PASS" in out


@pytest.mark.parametrize(
    ("remote_hash", "remote_count"),
    [
        pytest.param("1" * 64, 2, id="hash_not_the_ledgers_at_that_count"),
        pytest.param(None, 9, id="count_beyond_the_ledger"),
    ],
)
def test_remote_disagreeing_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    remote_hash: str | None,
    remote_count: int,
) -> None:
    ledger, repo, entries = two_week_setup(tmp_path)
    head = remote_hash if remote_hash is not None else entries[-1].chain_hash
    remote = bare_remote(tmp_path / "remote.git", head, remote_count)

    code, out = run_cli(capsys, cli_args(ledger, repo, remote=remote))

    assert code == 1, out
    # A disagreeing remote is not lag: it fails verification even with every local check clean.
    assert "LOCAL_RESULT= PASS" in out
    assert "EXTERNAL_RESULT= BEHIND" not in out
    assert "EXTERNAL_RESULT= VERIFIED" not in out
    assert "VERIFY_RESULT= FAIL" in out


def test_remote_unreachable_warns(tmp_path: Path) -> None:
    ledger, repo, entries = two_week_setup(tmp_path)
    remote = bare_remote(tmp_path / "remote.git", entries[-1].chain_hash, len(entries))

    def fetch_fails(
        args: Sequence[str], *, cwd: Path, **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        if "fetch" in args:
            return subprocess.CompletedProcess(list(args), 128, b"", b"network down")
        return run_git(args, cwd=cwd, **kwargs)

    report = verify_ledger(
        ledger,
        repo,
        runner=fetch_fails,
        remote_url=remote.as_posix(),
        module_name=ABSENT_DECLARATION,
    )
    lines = report_lines(report)

    assert report.ok, lines
    assert "REMOTE_ANCHOR= unreachable" in lines
    # Local validity and external evidence are never folded into one word.
    assert "LOCAL_RESULT= PASS" in lines
    assert "EXTERNAL_RESULT= UNREACHABLE" in lines
    assert lines[-1] == "VERIFY_RESULT= PASS"
    assert any(line.startswith("WARNING=") for line in lines)


def test_missing_local_anchor_with_rows_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "ledger"
    write_ledger(ledger, two_week_items())
    repo = make_repo(tmp_path / "public")

    code, out = run_cli(capsys, cli_args(ledger, repo))

    assert code == 1, out
    assert "CHAIN_OK= True" in out
    assert "LOCAL_ANCHOR_OK= False" in out
    assert "LOCAL_RESULT= FAIL" in out
    assert "VERIFY_RESULT= FAIL" in out


def test_backup_lag_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger = tmp_path / "ledger"
    ensure_backup_repo(ledger)
    # The setup commit was pushed; the ledger's own commit has not been yet.
    git_out(ledger, "update-ref", BACKUP_PUSHED_REF, "HEAD")
    entries = write_ledger(ledger, two_week_items())
    assert commit_backup(ledger, message="ledger: 5 entries") is not None
    repo = anchored_repo(tmp_path / "public", entries)

    code, out = run_cli(capsys, cli_args(ledger, repo))

    assert code == 0, out
    assert "BACKUP_BEHIND_BY= 1" in out
    assert "BACKUP_UNCOMMITTED= 0" in out
    assert "WARNING=" in out
    assert "VERIFY_RESULT= PASS" in out


# ---------------------------------------------------------------------------
# Task 2: verdict-week stamp and label enforcement (LDGR-06, LDGR-10, LDGR-11)
# ---------------------------------------------------------------------------


def inject_declaration(
    monkeypatch: pytest.MonkeyPatch, start_week: int = 6, **overrides: Any
) -> str:
    """Register an in-memory verdict-scope declaration with week W = *start_week*."""
    constants: dict[str, Any] = {
        "VERDICT_SEASON": 2026,
        "VERDICT_START_WEEK": start_week,
        "VERDICT_END_WEEK": 22,
        "INCLUDES_PLAYOFF_WEEKS": True,
        "INCLUDES_NEUTRAL_SITE_GAMES": True,
        "COUNTED_ARM": "live",
        "OUTCOME_RULE": "corrected outcome in force (D-06)",
        "FILL_CONVENTION_ID": "fill-v1",
        "BOOTSTRAP_REGIME_WEEKS": (2, 3, 4),
    }
    constants.update(overrides)
    module = types.ModuleType(FAKE_DECLARATION)
    for name, value in constants.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, FAKE_DECLARATION, module)
    return FAKE_DECLARATION


def migrated_row(game_id: str, week: int) -> dict[str, Any]:
    """A pre-verdict row migrated from the old writer: every stamp NULL (Plan 34-12)."""
    return ats_row(
        game_id,
        week=week,
        verdict_scope="pre_verdict",
        regime_label="bootstrap_regime",
        **dict.fromkeys(LEDGER_STAMP_COLUMNS),
    )


def correction_for(row: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """A correction entry moving *row* from win to loss: all 22 correction columns."""
    payload: dict[str, Any] = {
        **{name: row[name] for name in ("game_id", "season", "week", "target", "arm")},
        "original_grading_status": "win",
        "original_payout_flat": WIN["payout_flat"],
        "prior_grading_status": "win",
        "prior_payout_flat": WIN["payout_flat"],
        "prior_realized_units": WIN["realized_units"],
        "prior_correction_seq": None,
        "corrected_grading_status": "loss",
        "corrected_outcome": False,
        "corrected_payout_flat": LOSS["payout_flat"],
        "corrected_realized_units": LOSS["realized_units"],
        "realized_value": -10.0,
        "source_dataset": "schedules",
        "source_season": 2026,
        "source_week": row["week"],
        "source_sequence": 4,
        "detected_at_utc": "2026-10-21T21:00:00+00:00",
        "corrected_at_utc": "2026-10-21T21:00:05+00:00",
    }
    payload.update(overrides)
    assert tuple(payload) == CORRECTION_COLUMNS_V1
    return payload


def verdict_setup(
    tmp_path: Path, items: Sequence[tuple[str, Mapping[str, Any], Any]]
) -> list[str]:
    """Write *items*, anchor them, and return the argv (remote skipped)."""
    ledger = tmp_path / "ledger"
    entries = write_ledger(ledger, items)
    repo = anchored_repo(tmp_path / "public", entries)
    return cli_args(ledger, repo)


def failure_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("FAILURE=")]


def test_no_declaration_is_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = verdict_setup(tmp_path, [(ENTRY_KIND_ROW, ats_row(WEEK7_A, week=7), None)])

    code, out = run_cli(capsys, args)

    assert code == 0, out
    assert "VERDICT_DECLARED= False" in out
    assert "WARNING=" in out
    assert "VERIFY_RESULT= PASS" in out


def test_null_stamp_in_verdict_week_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unstamped = ats_row(WEEK6_WIN, recipe_id=None)
    args = verdict_setup(
        tmp_path,
        [
            (ENTRY_KIND_ROW, ats_row(WEEK6_LOSS), None),
            (ENTRY_KIND_ROW, unstamped, None),
        ],
    )

    code, out = run_cli(capsys, args, declaration=inject_declaration(monkeypatch))

    assert code == 1, out
    assert "VERDICT_DECLARED= True" in out
    assert "VERDICT_START_WEEK= 6" in out
    assert "VERDICT_STAMPS_OK= False" in out
    assert "VERDICT_LABELS_OK= True" in out
    failures = failure_lines(out)
    assert any(
        "seq 1" in line and key_text(unstamped) in line and "recipe_id" in line
        for line in failures
    ), failures
    assert "LOCAL_RESULT= FAIL" in out


def test_null_stamps_before_w_pass(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = verdict_setup(
        tmp_path,
        [
            (ENTRY_KIND_ROW, migrated_row("2026_03_NE_NYJ", 3), None),
            (ENTRY_KIND_ROW, migrated_row("2026_04_GB_CHI", 4), None),
            (ENTRY_KIND_ROW, ats_row(WEEK6_WIN), None),
        ],
    )

    code, out = run_cli(capsys, args, declaration=inject_declaration(monkeypatch))

    assert code == 0, out
    assert "VERDICT_DECLARED= True" in out
    assert "VERDICT_STAMPS_OK= True" in out
    assert "VERDICT_LABELS_OK= True" in out
    assert "VERIFY_RESULT= PASS" in out


@pytest.mark.parametrize(
    ("week", "label"),
    [
        pytest.param(6, "pre_verdict", id="week_w_labelled_pre_verdict"),
        pytest.param(5, "verdict", id="week_before_w_labelled_verdict"),
    ],
)
def test_label_disagreeing_with_declaration_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    week: int,
    label: str,
) -> None:
    mislabelled = ats_row(f"2026_{week:02d}_KC_BUF", week=week, verdict_scope=label)
    args = verdict_setup(tmp_path, [(ENTRY_KIND_ROW, mislabelled, None)])

    code, out = run_cli(capsys, args, declaration=inject_declaration(monkeypatch))

    assert code == 1, out
    assert "VERDICT_LABELS_OK= False" in out
    assert any(key_text(mislabelled) in line for line in failure_lines(out)), out
    assert "VERIFY_RESULT= FAIL" in out


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"recipe_id": "recipe-not-registered"}, id="recipe_unresolved"),
        pytest.param({"fill_convention_id": "fill-v0"}, id="fill_not_declared"),
    ],
)
def test_recipe_or_fill_not_declared_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, Any],
) -> None:
    row = ats_row(WEEK6_WIN, **override)
    args = verdict_setup(tmp_path, [(ENTRY_KIND_ROW, row, None)])

    code, out = run_cli(capsys, args, declaration=inject_declaration(monkeypatch))

    assert code == 1, out
    assert "VERDICT_STAMPS_OK= False" in out
    (column,) = override
    assert any(
        key_text(row) in line and column in line for line in failure_lines(out)
    ), out


def test_correction_must_reference_a_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    row = ats_row(WEEK6_WIN)
    orphan = correction_for(ats_row("2026_06_NO_SUCH"))
    args = verdict_setup(
        tmp_path,
        [(ENTRY_KIND_ROW, row, None), (ENTRY_KIND_CORRECTION, orphan, None)],
    )

    code, out = run_cli(capsys, args)

    assert code == 1, out
    assert "CORRECTIONS_OK= False" in out
    assert any(
        "seq 1" in line and key_text(orphan) in line for line in failure_lines(out)
    ), out
    assert "LOCAL_RESULT= FAIL" in out
