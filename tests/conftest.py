"""Pytest configuration and fixtures for NFL Prediction System tests."""

import os
import re
import shutil

# Add project root to path
import sys
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ---------------------------------------------------------------------------
# WR-10: evidence-backed controls that did not run must be VISIBLE, not silent
# ---------------------------------------------------------------------------

# Several of this project's load-bearing controls read run records that are
# deliberately gitignored -- fingerprint documents under ``outputs/``, the gold
# matrices under ``data/``, the deployed artifacts under ``artifacts/`` -- or need
# git history a shallow clone does not have. Each of those guards is individually
# well-reasoned: a control that cannot run must not report a false verdict.
#
# The problem is the AGGREGATE. A suite that skips them still reports zero failures,
# and a published "N passed / 7 skipped / 0 failed" reconciliation holds only on a
# machine that has just run the phase. On any other checkout the skip count is
# materially higher and several controls simply did not execute, with nothing in the
# output saying so. This summary line says so.
#
# Matching is on the skip REASON text. These markers are the vocabulary the guards
# already use; a new guard that skips for absent evidence should use one of them (or
# add one here) rather than inventing a silent phrasing.
_EVIDENCE_SKIP_MARKERS = (
    "gitignored",
    "fingerprint document",
    "git history is unavailable",
    "not present at",
    "absent",
    "not built on this checkout",
    "not populated",
    # Plan 31-02: the live nflreadpy 2025 schedule is an EXTERNAL run record, absent on an
    # offline checkout exactly as a gitignored one is. The Wave-0 completeness gate that
    # reads it is a control, so its non-run must be named rather than counted as an ordinary
    # environment skip.
    "could not be loaded on this checkout",
    # Plan 31-12: the pre-hold high-total eligibility boundary is DERIVED AT IMPORT from the
    # gitignored silver odds lake and reads NaN without it, and the three-target integration
    # run reads gold and the deployed artifacts. Both are controls on the one-shot 2025
    # runner -- the integration run is the ONLY place its real loaders are exercised -- so a
    # checkout that cannot run them must SAY so rather than report a green suite that
    # silently excluded them.
    #
    # Plan 31-14 registers under this SAME marker rather than adding a second spelling of one
    # fact. tests/integration/test_profitability_2025_controls.py has exactly one skip path --
    # the shared `p31_rehearsal_run` fixture, which skips with this reason when the boundary is
    # not derivable -- and it is the SYNTHETIC UNIT half of the 2025 positive control. The other
    # half reads the git-TRACKED verdict artifact and deliberately FAILS rather than skips if it
    # is missing: that artifact travels with the repository, so its absence is a broken checkout
    # and never an environment fact.
    "not derivable on this checkout",
    # Plan 31-13: the real-schedule form of the bet-list completeness invariant reads the
    # gitignored silver schedule. The hand-built form always runs and proves the rule; this
    # one proves the rule meets real data, so a checkout without the lake must NAME it rather
    # than report a green suite that quietly excluded it.
    "not readable on this checkout",
)

_SKIP_REASON_RE = re.compile(r"^Skipped: (.*)$", re.DOTALL)


def _skip_reason(report) -> str:
    """Return a skip report's reason text, however pytest chose to encode it."""
    longrepr = getattr(report, "longrepr", None)
    # The common shape is a (path, lineno, reason) tuple.
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        reason = str(longrepr[2])
        match = _SKIP_REASON_RE.match(reason)
        return match.group(1) if match else reason
    return str(longrepr) if longrepr is not None else ""


def is_evidence_backed_skip(reason: str) -> bool:
    """True when *reason* says a control was skipped for want of absent EVIDENCE.

    Separated from the hook so the discrimination itself is testable without running
    a nested pytest session -- an ordinary conditional skip (a platform guard, an
    optional dependency) must NOT be counted, or the note becomes noise and stops
    being read.
    """
    lowered = reason.lower()
    return any(marker in lowered for marker in _EVIDENCE_SKIP_MARKERS)


def pytest_terminal_summary(terminalreporter) -> None:
    """Everything this session needs to say about itself that a count cannot."""
    _report_evidence_backed_skips(terminalreporter)
    _report_closing_full_sweep(terminalreporter)
    _report_guard_observations(terminalreporter)


def _report_evidence_backed_skips(terminalreporter) -> None:
    """Report how many evidence-backed controls did NOT run on this checkout.

    A green suite is not the same claim on a fresh clone as it is on the machine that
    produced the evidence, and nothing previously distinguished the two.
    """
    skipped = terminalreporter.stats.get("skipped", [])
    evidence_skips = [
        report for report in skipped if is_evidence_backed_skip(_skip_reason(report))
    ]
    if not evidence_skips:
        return

    terminalreporter.write_sep("-", "evidence-backed controls")
    terminalreporter.write_line(
        f"NOTE: {len(evidence_skips)} of {len(skipped)} skipped test(s) are "
        "evidence-backed controls that did NOT run on this checkout -- their "
        "gitignored run records (outputs/, data/, artifacts/) or git history are "
        "absent. A green suite HERE does not include them."
    )
    for report in evidence_skips:
        terminalreporter.write_line(f"  did not run: {report.nodeid}")


# ---------------------------------------------------------------------------
# Plan 31-11: the production stores are guarded by CONTENT, never by git status
# ---------------------------------------------------------------------------

# `git status --porcelain data/` was the boundary check this project ran for two
# phases. `.gitignore:22` blankets `data/`, so that check is structurally incapable
# of failing -- it returns empty whether the archive is intact or destroyed. These
# fixtures hash file contents instead, so a write a test did not intend is a failure
# it can actually report. See tests/data_boundary.py for the two writes this would
# have caught four waves earlier.
#
# OPT-IN, not autouse. Some tests legitimately write under `data/` (the API's
# `data/web_cache.duckdb`, for one), and a guard that fires on legitimate writes is a
# guard that gets deleted. A module that must not touch a production store requests
# the fixture explicitly.


@pytest.fixture
def data_boundary_guard():
    """Fail the test if anything under `data/` was added, removed or rewritten."""
    from tests.data_boundary import (
        PRODUCTION_DATA_ROOT,
        assert_tree_unchanged,
        digest_tree,
    )

    before = digest_tree(PRODUCTION_DATA_ROOT)
    yield before
    assert_tree_unchanged(
        before, digest_tree(PRODUCTION_DATA_ROOT), PRODUCTION_DATA_ROOT
    )


@pytest.fixture
def artifacts_boundary_guard():
    """Fail the test if anything under `artifacts/` was added, removed or rewritten.

    `artifacts/latest.json` is the DEPLOYED-MODEL manifest. A test that rewrites it
    swaps production models, which is why this root is guarded alongside `data/`.
    """
    from tests.data_boundary import (
        PRODUCTION_ARTIFACTS_ROOT,
        assert_tree_unchanged,
        digest_tree,
    )

    before = digest_tree(PRODUCTION_ARTIFACTS_ROOT)
    yield before
    assert_tree_unchanged(
        before, digest_tree(PRODUCTION_ARTIFACTS_ROOT), PRODUCTION_ARTIFACTS_ROOT
    )


# ---------------------------------------------------------------------------
# Plan 33-01 / COLD-05: the production stores are guarded BY DEFAULT
# ---------------------------------------------------------------------------
#
# The two fixtures above are OPT-IN, and that is the defect this section closes. A
# module that never thought about the boundary is UNGUARDED, which is how four
# production overwrites went unreported during Phase 31 -- `test_elo_integration.py`
# rebuilding Elo over `data/silver/elo_game_snapshots.parquet`, and
# `test_lift_validation.py` retraining into production `artifacts/`. Neither module
# had asked to be guarded, so neither was.
#
# `production_store_write_guard` below is AUTOUSE: every test in every tier is
# judged, and a test that legitimately writes a production store says so with a
# PATH-SCOPED marker naming exactly what it writes.
#
# THE INSTRUMENT IS `tests/data_boundary.py`, REUSED AND NEVER REIMPLEMENTED.
# `digest_tree`, `diff_digests` and `format_digest_diff` do the hashing and the
# reporting here exactly as they do for the opt-in fixtures; the only thing this
# section adds is WHEN they run and WHO is exempt.
#
# D33-23 -- ONE full content digest per session, a STAT-ONLY sweep per test.
# Content-hashing 130 MB of production stores after each of four thousand tests is
# not a guard anybody would keep. So the per-test pass reads `(st_size,
# st_mtime_ns)` only, and content-hashes ONLY the keys that moved. The stat map
# decides WHERE TO LOOK; it never renders the verdict.
#
# D33-23, review-hardened -- and a FULL content sweep at session END. A write that
# restores its own size and mtime, or that lands inside the filesystem's timestamp
# resolution, is invisible to the prefilter. The closing sweep is what makes the
# SESSION's verdict content-based rather than metadata-based, and it reports through
# `pytest_terminal_summary` so the finding is attached to the session rather than to
# whichever test happened to run last.

# D33-32 -- and no verdict here rests on metadata. Every digest the guard compares
# is taken with `require_content_digest`, which closes handles, retries, and then
# RAISES BY NAME rather than accepting the weaker stat signature. A legitimately
# locked production store therefore fails a normal run, naming the file and the
# likely holder. That is the intended behaviour: the guard's whole claim is
# content-based evidence, and an unprovable comparison is a refusal, not a pass.

GUARD_DATA_ROOT_ENV = "NFL_GUARD_DATA_ROOT"
GUARD_ARTIFACTS_ROOT_ENV = "NFL_GUARD_ARTIFACTS_ROOT"

# Set to any non-empty value to have the session print what the guard OBSERVED --
# how often the locked-file read path fired, and the largest number of `.duckdb.wal`
# siblings seen beside a tracked store. Off by default so an ordinary run's output
# is unchanged; Plan 33-01 Task 3(d) turns it on for one measured tier.
GUARD_OBSERVATIONS_ENV = "NFL_GUARD_OBSERVATIONS"

CLOSING_SWEEP_HEADER = "SESSION-END FULL CONTENT SWEEP"

# Session-lived guard state. A module global rather than a fixture because
# `pytest_sessionfinish` and `pytest_terminal_summary` are hooks, not fixtures, and
# the closing sweep has to reach the same rebased baseline the per-test pass left.
_GUARD_STATE: dict = {
    "baselines": None,
    "closing_report": None,
    # The MAXIMUM number of `.duckdb.wal` siblings seen at any single sweep. A
    # write-ahead log is transient -- it exists only while a transaction is open --
    # so a count taken once at session start would almost certainly read zero and
    # prove nothing. `.wal` is deliberately NOT in TRACKED_SUFFIXES; widening the
    # tracked set would change what every digest document taken under the narrower
    # set means, so the hypothesis is MEASURED here first.
    "wal_siblings": 0,
}


class _StoreBaseline:
    """One guarded root, its content digests, and its parallel stat map."""

    __slots__ = ("digests", "label", "root", "stats")

    def __init__(self, label: str, root: Path, digests: dict, stats: dict) -> None:
        self.label = label
        self.root = root
        self.digests = digests
        self.stats = stats


def _guarded_roots() -> tuple[tuple[str, Path], ...]:
    """The roots this session guards, as (label, path).

    The LABEL is the repo-relative name a marker declares against ("data" /
    "artifacts"), and it stays stable even when the PATH is redirected. The
    redirection exists for exactly one caller: the nested session in
    `tests/integration/test_data_boundary_guard_arming.py`, which has to watch the
    guard fire without any real production store being written to do it.
    """
    from tests.data_boundary import (
        PRODUCTION_ARTIFACTS_ROOT,
        PRODUCTION_DATA_ROOT,
    )

    data = os.environ.get(GUARD_DATA_ROOT_ENV)
    artifacts = os.environ.get(GUARD_ARTIFACTS_ROOT_ENV)
    return (
        ("data", Path(data) if data else PRODUCTION_DATA_ROOT),
        ("artifacts", Path(artifacts) if artifacts else PRODUCTION_ARTIFACTS_ROOT),
    )


def _stat_sweep(root: Path) -> dict[str, tuple[int, int]]:
    """Map every tracked file under *root* to `(st_size, st_mtime_ns)`.

    The cheap half of D33-23. Keys are POSIX-relative to *root*, identical to
    `digest_tree`'s, so the two maps line up key for key.
    """
    from tests.data_boundary import TRACKED_SUFFIXES

    root_path = Path(root)
    if not root_path.exists():
        return {}

    lowered = tuple(suffix.lower() for suffix in TRACKED_SUFFIXES)
    stats: dict[str, tuple[int, int]] = {}
    wal_siblings = 0
    for candidate in root_path.rglob("*"):
        if candidate.name.endswith(".duckdb.wal"):
            wal_siblings += 1
        if candidate.suffix.lower() not in lowered:
            continue
        try:
            info = candidate.stat()
        except OSError:
            continue
        if not candidate.is_file():
            continue
        stats[candidate.relative_to(root_path).as_posix()] = (
            info.st_size,
            info.st_mtime_ns,
        )
    _GUARD_STATE["wal_siblings"] = max(_GUARD_STATE["wal_siblings"], wal_siblings)
    return stats


def _suspect_paths(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> list[str]:
    """Keys whose stat signature appeared, vanished or moved -- where to look.

    NOT a verdict. A file can be touched, copied over itself or re-saved with
    identical bytes and land here; the caller content-hashes these keys and only
    then decides anything.
    """
    appeared_or_vanished = set(before) ^ set(after)
    moved = {key for key in set(before) & set(after) if before[key] != after[key]}
    return sorted(appeared_or_vanished | moved)


def _marker_paths_for_item(item) -> dict[str, frozenset[str]]:
    """The store paths *item* DECLARES it writes, split by guarded root label.

    A marker declares repo-relative paths -- "data/nfl_predictions.duckdb" -- which
    is what a reader of the test sees on disk. The guard works in root-relative
    keys, so the leading label is stripped here and used to route the exemption to
    the right root. A declared path that names no guarded root is an ERROR rather
    than an ignored line: silently dropping it would grant an exemption the author
    believes they have and the guard does not honour.
    """
    from tests.phase33_state import WRITES_PRODUCTION_STORE_MARKER

    labels = [label for label, _ in _guarded_roots()]
    scoped: dict[str, set[str]] = {label: set() for label in labels}

    marker = item.get_closest_marker(WRITES_PRODUCTION_STORE_MARKER)
    if marker is None:
        return {label: frozenset(keys) for label, keys in scoped.items()}

    declared: list[str] = []
    for value in marker.args:
        declared.extend([value] if isinstance(value, str) else list(value))
    paths = marker.kwargs.get("paths") or ()
    declared.extend([paths] if isinstance(paths, str) else list(paths))

    for raw in declared:
        parts = PurePosixPath(str(raw).replace("\\", "/")).parts
        if len(parts) < 2 or parts[0] not in scoped:
            raise ValueError(
                f"{item.nodeid} declares writes_production_store(paths=[{raw!r}]). A "
                f"declared path must be repo-relative and start with one of {labels} "
                "-- for example 'data/nfl_predictions.duckdb'. A path the guard cannot "
                "route to a root would grant an exemption that silently does nothing."
            )
        scoped[parts[0]].add("/".join(parts[1:]))

    return {label: frozenset(keys) for label, keys in scoped.items()}


def _guard_verdict(
    baseline: _StoreBaseline,
    declared: frozenset[str],
) -> str | None:
    """Judge one root, REBASE it, and return a violation message or None.

    The rebase happens for every suspect key, permitted or not, and that is
    deliberate: each test is judged against the state it actually inherited. Without
    it, one undeclared write would be re-reported by every later test in the session
    and the second report would be about the first test's fault.
    """
    from tests.data_boundary import (
        diff_digests,
        format_digest_diff,
        require_content_digest,
    )

    after_stats = _stat_sweep(baseline.root)
    suspects = _suspect_paths(baseline.stats, after_stats)
    if not suspects:
        return None

    before_subset = {
        key: baseline.digests[key] for key in suspects if key in baseline.digests
    }
    # `require_content_digest`, never `digest_file` (D33-32): a stat signature is an
    # inability to prove content integrity and must not settle a verdict. If the
    # bytes cannot be read even after handles are closed, this RAISES by name.
    after_subset = {
        key: require_content_digest(baseline.root / key)
        for key in suspects
        if key in after_stats
    }
    diff = diff_digests(before_subset, after_subset)

    for key in suspects:
        baseline.stats.pop(key, None)
        baseline.digests.pop(key, None)
        if key in after_stats:
            baseline.stats[key] = after_stats[key]
            baseline.digests[key] = after_subset[key]

    undeclared = {
        category: [key for key in keys if key not in declared]
        for category, keys in diff.items()
    }
    undeclared.setdefault("mixed", [])
    if not any(undeclared.values()):
        return None

    return (
        format_digest_diff(undeclared, before_subset, after_subset, baseline.root)
        + "\n"
        + "\n".join(
            [
                "",
                "DECLARE THE WRITE, OR REDIRECT IT. If this write is legitimate, mark the "
                "test with the paths it writes and nothing else:",
                *[
                    f'    @pytest.mark.writes_production_store(paths=["{baseline.label}/{key}"])'
                    for category in ("added", "removed", "changed", "mixed")
                    for key in undeclared.get(category, [])
                ],
                "The marker exempts the marked test alone, for the paths it names alone; "
                "record it in tests/phase33_state.MARKED_PRODUCTION_WRITERS.",
            ]
        )
    )


def _assert_single_worker(config) -> None:
    """Refuse to arm under multiple workers. A named error, NEVER a skip.

    The guard holds a session baseline and rebases it in test order. Under
    `pytest-xdist` each worker holds its own baseline over a SHARED filesystem, so
    one worker's legitimate declared write is another worker's unexplained data
    move -- the guard would report violations that are artefacts of its own
    concurrency, and the first fix anybody reached for would be to turn it off.

    `pytest.skip` is the wrong answer and is not used: a guard that stands down
    under a configuration it does not understand is worse than no guard, because the
    session still reports green.
    """
    workerinput = getattr(config, "workerinput", None)
    numprocesses = None
    getoption = getattr(config, "getoption", None)
    if callable(getoption):
        numprocesses = getoption("numprocesses", None)

    if workerinput is None and numprocesses in (None, 0):
        return

    raise RuntimeError(
        "The production-store write guard CANNOT ARM under a multi-worker pytest "
        "session (xdist workerinput="
        f"{workerinput!r}, numprocesses={numprocesses!r}). It holds one session "
        "baseline and rebases it in test order; with several workers sharing one "
        "filesystem, one worker's declared write is another worker's unexplained "
        "data move. This suite has a single-worker requirement, and pytest-xdist is "
        "deliberately not installed -- its absence is asserted in "
        "tests/unit/test_write_guard_marker_scope.py. Run the suite single-worker, "
        "in the three tiers PIPELINE.md describes. The guard refuses rather than "
        "skipping on purpose: a guard that stands down still reports green."
    )


def _take_baselines() -> tuple[_StoreBaseline, ...]:
    """ONE full content digest of each guarded root, plus its parallel stat map.

    `content_digest_tree`, not `digest_tree`: the baseline is one half of every
    comparison this session will make, so a stat signature here would poison the
    other side into a permanently UNDECIDED verdict (D33-32).
    """
    from tests.data_boundary import content_digest_tree

    return tuple(
        _StoreBaseline(label, root, content_digest_tree(root), _stat_sweep(root))
        for label, root in _guarded_roots()
    )


def _closing_full_sweep(baselines) -> str | None:
    """The D33-23 review hardening: a FULL content sweep at session end.

    Compared against the baseline AS REBASED by the per-test pass, so anything this
    reports is something no per-test stat sweep saw -- which is itself the
    diagnostic. A write whose size and mtime came back unchanged is exactly the
    shape the prefilter cannot see, and exactly the shape somebody covering their
    tracks would produce.
    """
    from tests.data_boundary import (
        content_digest_tree,
        diff_digests,
        format_digest_diff,
        is_clean,
    )

    sections = []
    for baseline in baselines:
        after = content_digest_tree(baseline.root)
        diff = diff_digests(baseline.digests, after)
        if is_clean(diff):
            continue
        sections.append(
            format_digest_diff(diff, baseline.digests, after, baseline.root)
        )

    if not sections:
        return None

    return "\n\n".join(
        [
            CLOSING_SWEEP_HEADER
            + " -- a production store moved and NO per-test stat sweep saw it.",
            "The per-test prefilter reads (st_size, st_mtime_ns) and content-hashes "
            "only what moved. A write that RESTORES both, or that lands inside this "
            "filesystem's timestamp resolution, passes it unseen. This sweep re-hashes "
            "every tracked file against the session baseline as rebased by the "
            "permitted writes, so the session's verdict is content-based even where "
            "the prefilter's was not. That the finding appears HERE and not against a "
            "test is the diagnostic: the metadata did not move with the bytes.",
            *sections,
        ]
    )


def _report_closing_full_sweep(terminalreporter) -> None:
    report = _GUARD_STATE.get("closing_report")
    if not report:
        return
    terminalreporter.write_sep("-", "production store boundary")
    for line in report.splitlines():
        terminalreporter.write_line(line)


def _report_guard_observations(terminalreporter) -> None:
    """Print what the guard OBSERVED about its own instrument, when asked to.

    Two numbers Plan 33-01 Task 3(d) records rather than assumes. Neither changes
    the session's verdict; both decide what a LATER phase is allowed to conclude.

    STAT_SIGNATURE_OBSERVATIONS -- how often the locked-file read path fired at
    all. "The fallback silently degrades the guard" and "the fallback never fires
    on this machine" are different worlds, and nobody had counted.

    WAL_SIBLING_OBSERVATIONS -- the largest number of `.duckdb.wal` siblings seen
    beside a tracked store at any single sweep. `.wal` is NOT tracked, and widening
    the tracked set on an unmeasured hypothesis would change what every digest
    document taken under the narrower set means. A non-zero count here is the
    evidence a later phase would need before widening it.
    """
    if not os.environ.get(GUARD_OBSERVATIONS_ENV):
        return
    if _GUARD_STATE.get("baselines") is None:
        return

    from tests.data_boundary import locked_read_observations

    terminalreporter.write_sep("-", "production store guard observations")
    terminalreporter.write_line(
        f"STAT_SIGNATURE_OBSERVATIONS={locked_read_observations()}  "
        f"WAL_SIBLING_OBSERVATIONS={_GUARD_STATE['wal_siblings']}"
    )


def pytest_sessionfinish(session, exitstatus) -> None:
    """Run the closing full sweep and FAIL the session on anything it finds.

    Attached to the session rather than to a test on purpose -- nothing the closing
    sweep finds belongs to any particular test, and pinning it to whichever test ran
    last would name an innocent one.
    """
    baselines = _GUARD_STATE.get("baselines")
    if not baselines:
        return
    report = _closing_full_sweep(baselines)
    _GUARD_STATE["closing_report"] = report
    if report and session.exitstatus == 0:
        session.exitstatus = 1


@pytest.fixture(scope="session")
def _production_store_baseline(request):
    """ONE content digest of every guarded root, taken before the first test runs."""
    _assert_single_worker(request.config)
    baselines = _take_baselines()
    _GUARD_STATE["baselines"] = baselines
    _GUARD_STATE["closing_report"] = None
    return baselines


@pytest.fixture(autouse=True)
def production_store_write_guard(request, _production_store_baseline):
    """COLD-05: fail any test that writes a production store it did not declare."""
    yield

    declared = _marker_paths_for_item(request.node)
    violations = [
        message
        for baseline in _production_store_baseline
        if (
            message := _guard_verdict(
                baseline, declared.get(baseline.label, frozenset())
            )
        )
    ]
    if not violations:
        return

    from tests.data_boundary import DataBoundaryViolation

    raise DataBoundaryViolation(
        "\n\n".join(
            [
                "PRODUCTION STORE WRITE GUARD -- "
                f"{request.node.nodeid} wrote a production store it did not declare.",
                *violations,
            ]
        )
    )


@pytest.fixture
def sealed_probe_offline(monkeypatch, tmp_path):
    """Keep Plan 32-08's in-capture detectors OFFLINE and off every committed file.

    Wave 4 wired both detectors into `scripts.capture_live_season`, so a capture now
    reaches `api.github.com` and appends one line to the COMMITTED
    `config/upstream_probe_log.jsonl`. Neither belongs in a test run: the GitHub ceiling
    is 60 requests per hour per IP, and a suite that appends to the committed liveness
    record would forge exactly the evidence that record exists to provide.

    Three seams are redirected, and the choice of stub matters. The release fetch and the
    pin-basis content digest are stubbed to raise `SealedProbeUnavailable`, which the
    detector RECORDS as an explicit UNKNOWN -- so these tests exercise the guard instead
    of bypassing it. The probe log is pointed at `tmp_path`. The graded-week record is
    stubbed to an explicitly-recorded empty set rather than left to read the real
    `outputs/` store, so a machine that happens to have a bet list and one that does not
    run the same test.

    Requested per module with `pytestmark = pytest.mark.usefixtures(...)`, not autouse:
    a module that means to exercise the probe (`tests/integration/test_detect_only.py`)
    must set its own seams up deliberately.
    """
    from scripts import capture_live_season

    def _offline(*_args, **_kwargs):
        msg = "offline test: the sealed probe makes no network call in this suite"
        raise capture_live_season.SealedProbeUnavailable(msg)

    def _no_graded_weeks(season, *, output_dir=None):
        return {
            "season": int(season),
            "weeks": [],
            "source": "tests.conftest.sealed_probe_offline stub",
            "resolved": True,
            "reason": "stubbed in the test suite; the real store is not read here",
        }

    monkeypatch.setattr(capture_live_season, "fetch_release_assets", _offline)
    monkeypatch.setattr(capture_live_season, "_pinned_basis_digest", _offline)
    monkeypatch.setattr(
        capture_live_season,
        "SEALED_PROBE_LOG_PATH",
        tmp_path / "upstream_probe_log.jsonl",
    )
    monkeypatch.setattr(
        "data.graded_weeks.graded_weeks_record", _no_graded_weeks, raising=True
    )


@pytest.fixture(scope="session")
def p31_rehearsal_run(tmp_path_factory):
    """ONE rehearsal run of the Phase-31 one-shot runner, shared across test modules.

    Session-scoped and shared because the run costs a fifteen-cell tune sweep, a pooled
    three-target hold selection and three counterfactual passes, and four modules assert
    different properties of the SAME output. Three private copies would also be three
    fixtures that can drift, which is the second-list failure this project keeps paying for.

    It uses the DISJOINT rehearsal window (tune 2021-2023, hold 2024) over the synthetic
    fixture in ``tests/p31_synthetic_candidates.py``. NOTHING here reads 2025: the fixture
    refuses to generate it, and the rehearsal window does not name it.

    Yields:
        ``{"result", "verdict_toml_path", "verdict_json_path", "ledger_path"}``.
    """
    from backtest.ats_ev_chain import FENCE_WINDOW_REHEARSAL
    from backtest.ou_divergence import HIGH_TOTAL_BOUNDARY_PREHOLD
    from backtest.profitability_2025 import run_profitability_2025
    from tests.p31_synthetic_candidates import build_synthetic_candidates

    if not np.isfinite(HIGH_TOTAL_BOUNDARY_PREHOLD):
        pytest.skip(
            "the leakage-clean pre-hold high-total boundary is not derivable on this "
            "checkout (it is derived at import from the gitignored silver odds lake), so "
            "the O/U eligibility gate cannot be constructed and the runner cannot run."
        )

    tmp_path = tmp_path_factory.mktemp("p31_rehearsal_run")
    paths = {
        "verdict_toml_path": tmp_path / "verdict.toml",
        "verdict_json_path": tmp_path / "verdict.json",
        "ledger_path": tmp_path / "ledger.toml",
    }
    result = run_profitability_2025(
        FENCE_WINDOW_REHEARSAL,
        candidates_by_target=build_synthetic_candidates(),
        **paths,
    )
    return {"result": result, **paths}


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary data directory for tests."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "data"

    # Create data structure
    (data_dir / "bronze").mkdir(parents=True)
    (data_dir / "silver").mkdir(parents=True)
    (data_dir / "gold").mkdir(parents=True)

    yield data_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_games_data():
    """Create mock games data for testing."""
    et_tz = ZoneInfo("America/New_York")

    mock_games = []
    for week in range(1, 4):  # 3 weeks
        week_games = [
            {
                "game_id": f"MOCK_2024_W{week:02d}_BUF@MIA",
                "season": 2024,
                "week": week,
                "home_team": "MIA",
                "away_team": "BUF",
                "kickoff_et": datetime(
                    2024, 9, 7 + (week - 1) * 7, 13, 0, tzinfo=et_tz
                ),
                "home_score": 24 if week <= 2 else None,  # First 2 weeks have results
                "away_score": 21 if week <= 2 else None,
            },
            {
                "game_id": f"MOCK_2024_W{week:02d}_KC@DEN",
                "season": 2024,
                "week": week,
                "home_team": "DEN",
                "away_team": "KC",
                "kickoff_et": datetime(
                    2024, 9, 7 + (week - 1) * 7, 16, 25, tzinfo=et_tz
                ),
                "home_score": 17 if week <= 2 else None,
                "away_score": 28 if week <= 2 else None,
            },
        ]
        mock_games.extend(week_games)

    return pd.DataFrame(mock_games)


@pytest.fixture
def mock_team_form_data():
    """Create mock team form data."""
    teams = ["BUF", "MIA", "KC", "DEN"]
    weeks = [1, 2, 3]
    sides = ["offense", "defense"]

    mock_form = []
    for team in teams:
        for week in weeks:
            for side in sides:
                base_epa = np.random.normal(0.05 if side == "offense" else -0.05, 0.15)
                mock_form.append(
                    {
                        "team": team,
                        "target_season": 2024,
                        "target_week": week,
                        "side": side,
                        "rolling_epa_per_play": base_epa,
                        "rolling_pass_epa_per_play": base_epa
                        + np.random.normal(0, 0.05),
                        "rolling_rush_epa_per_play": base_epa
                        + np.random.normal(0, 0.05),
                        "rolling_success_rate": np.random.uniform(0.35, 0.55),
                        "rolling_neutral_pass_rate": np.random.uniform(0.55, 0.75)
                        if side == "offense"
                        else np.nan,
                    }
                )

    return pd.DataFrame(mock_form)


@pytest.fixture
def mock_elo_data():
    """Create mock Elo ratings data."""
    teams = ["BUF", "MIA", "KC", "DEN"]
    weeks = [1, 2, 3]

    mock_elo = []
    for team in teams:
        base_elo = 1500 + np.random.normal(0, 100)
        for week in weeks:
            mock_elo.append(
                {
                    "team": team,
                    "season": 2024,
                    "week": week,
                    "elo_rating": base_elo + np.random.normal(0, 20),
                    "elo_uncertainty": np.random.uniform(50, 150),
                    "elo_games_played": week + 15,
                    "elo_form_rating": base_elo + np.random.normal(0, 30),
                }
            )

    return pd.DataFrame(mock_elo)


@pytest.fixture
def mock_odds_data():
    """Create mock odds data."""
    games = [
        "MOCK_2024_W01_BUF@MIA",
        "MOCK_2024_W01_KC@DEN",
        "MOCK_2024_W02_BUF@MIA",
        "MOCK_2024_W02_KC@DEN",
        "MOCK_2024_W03_BUF@MIA",
        "MOCK_2024_W03_KC@DEN",
    ]

    mock_odds = []
    for game_id in games:
        mock_odds.append(
            {
                "game_id": game_id,
                "snapshot_spread": np.random.uniform(-14, 14),
                "snapshot_total": np.random.uniform(35, 65),
                "snapshot_ml_home": np.random.randint(-300, 300),
                "snapshot_ml_away": np.random.randint(-300, 300),
                "has_snapshot_lines": 1.0,
            }
        )

    return pd.DataFrame(mock_odds)


@pytest.fixture
def mock_weather_data():
    """Create mock weather data."""
    games = [
        "MOCK_2024_W01_BUF@MIA",
        "MOCK_2024_W01_KC@DEN",
        "MOCK_2024_W02_BUF@MIA",
        "MOCK_2024_W02_KC@DEN",
        "MOCK_2024_W03_BUF@MIA",
        "MOCK_2024_W03_KC@DEN",
    ]

    mock_weather = []
    for game_id in games:
        outdoor = np.random.choice([True, False])
        mock_weather.append(
            {
                "game_id": game_id,
                "weather_affects_game": 1.0 if outdoor else 0.0,
                "temp_f": np.random.uniform(20, 90) if outdoor else 72.0,
                "wind_mph": np.random.uniform(0, 25) if outdoor else 0.0,
                "precip_prob": np.random.uniform(0, 0.8) if outdoor else 0.0,
                "weather_severity_score": np.random.uniform(0, 0.8) if outdoor else 0.0,
            }
        )

    return pd.DataFrame(mock_weather)


@pytest.fixture
def sample_feature_matrix():
    """Create a sample feature matrix with realistic data."""
    np.random.seed(42)  # For reproducible tests

    # Create base games
    games = [
        {
            "game_id": "TEST_2024_W01_BUF@MIA",
            "season": 2024,
            "week": 1,
            "home_team": "MIA",
            "away_team": "BUF",
        },
        {
            "game_id": "TEST_2024_W01_KC@DEN",
            "season": 2024,
            "week": 1,
            "home_team": "DEN",
            "away_team": "KC",
        },
        {
            "game_id": "TEST_2024_W02_DAL@NYG",
            "season": 2024,
            "week": 2,
            "home_team": "NYG",
            "away_team": "DAL",
        },
    ]

    features = []
    for game in games:
        feature_row = game.copy()

        # Add Elo features
        feature_row.update(
            {
                "home_elo_rating": np.random.uniform(1400, 1600),
                "away_elo_rating": np.random.uniform(1400, 1600),
                "elo_diff": np.random.uniform(-200, 200),
            }
        )

        # Add form features
        for side in ["home_off", "away_off", "home_def", "away_def"]:
            feature_row.update(
                {
                    f"{side}_rolling_epa_per_play": np.random.normal(0, 0.2),
                    f"{side}_rolling_success_rate": np.random.uniform(0.3, 0.6),
                }
            )

        # Add contextual features
        feature_row.update(
            {
                "venue_outdoor": np.random.choice([0.0, 1.0]),
                "away_travel_distance_miles": np.random.uniform(200, 2500),
                "home_rest_days": np.random.choice([6, 7, 8, 9, 10]),
                "away_rest_days": np.random.choice([6, 7, 8, 9, 10]),
            }
        )

        # Add weather features
        feature_row.update(
            {
                "temp_f": np.random.uniform(20, 90),
                "wind_mph": np.random.uniform(0, 25),
                "weather_severity_score": np.random.uniform(0, 1),
            }
        )

        # Add market features
        feature_row.update(
            {
                "snapshot_spread": np.random.uniform(-14, 14),
                "snapshot_total": np.random.uniform(35, 65),
                "has_snapshot_lines": 1.0,
            }
        )

        # Add targets
        feature_row.update(
            {
                "target_wp": np.random.choice([0, 1]),
                "target_ats": np.random.uniform(-30, 30),
                "target_ou": np.random.uniform(-15, 15),
            }
        )

        features.append(feature_row)

    return pd.DataFrame(features)


@pytest.fixture
def api_client():
    """Create a test client for the FastAPI application."""
    from api.main import app

    return TestClient(app)


@pytest.fixture
def mock_predictions_data():
    """Create mock predictions data for API tests."""
    return {
        "current_week": 3,
        "current_season": 2024,
        "predictions_available": True,
        "last_updated": "2024-09-15T18:00:00-04:00",
        "games": [
            {
                "game_id": "TEST_2024_W03_BUF@MIA",
                "season": 2024,
                "week": 3,
                "home_team": "MIA",
                "away_team": "BUF",
                "kickoff_et": "2024-09-15T13:00:00-04:00",
                "wp_prob_home": 0.58,
                "wp_prob_away": 0.42,
                "wp_confidence": 0.73,
                "ats_prob_home": 0.52,
                "ats_prob_away": 0.48,
                "ats_confidence": 0.64,
                "ou_prob_over": 0.49,
                "ou_prob_under": 0.51,
                "ou_confidence": 0.56,
                "spread_line": -3.5,
                "total_line": 45.5,
                "wp_edge": 0.08,
                "ats_edge": 0.02,
                "ou_edge": 0.01,
            }
        ],
    }


@pytest.fixture
def mock_backtest_data():
    """Create mock backtest results for API tests."""
    return {
        "summary_metrics": {
            "wp_log_loss": 0.642,
            "wp_brier_score": 0.241,
            "wp_accuracy": 0.652,
            "ats_accuracy": 0.534,
            "ou_accuracy": 0.518,
            "betting_roi": 0.067,
            "total_games": 1024,
            "seasons_tested": ["2020", "2021", "2022", "2023", "2024"],
        },
        "season_breakdown": [
            {
                "season": 2024,
                "wp_accuracy": 0.658,
                "ats_accuracy": 0.541,
                "ou_accuracy": 0.523,
                "betting_roi": 0.071,
                "games_count": 256,
            }
        ],
    }


@pytest.fixture
def mock_calibration_data():
    """Create mock calibration data for API tests."""
    return {
        "wp_calibration": {
            "bin_centers": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "observed_frequencies": [
                0.09,
                0.18,
                0.31,
                0.39,
                0.52,
                0.61,
                0.72,
                0.81,
                0.91,
            ],
            "bin_counts": [45, 78, 92, 123, 156, 134, 89, 67, 32],
        },
        "ats_calibration": {
            "bin_centers": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "observed_frequencies": [
                0.11,
                0.19,
                0.28,
                0.42,
                0.48,
                0.59,
                0.69,
                0.78,
                0.88,
            ],
            "bin_counts": [67, 89, 134, 156, 178, 123, 92, 78, 45],
        },
    }


@pytest.fixture(autouse=True)
def setup_test_environment(temp_data_dir, monkeypatch):
    """Set up test environment with temporary directories."""
    # Patch data directory paths
    monkeypatch.setenv("NFL_DATA_DIR", str(temp_data_dir))

    # Set up logging for tests
    import logging

    logging.basicConfig(level=logging.WARNING)  # Reduce log noise in tests


# Utility functions for tests
def assert_dataframe_structure(df, expected_columns=None, min_rows=0):
    """Assert DataFrame has expected structure."""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"

    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        assert not missing_cols, f"Missing columns: {missing_cols}"


def assert_probability_range(values, tolerance=1e-6):
    """Assert values are valid probabilities (0-1 range)."""
    values = pd.Series(values).dropna()
    assert values.min() >= -tolerance, f"Found probability < 0: {values.min()}"
    assert values.max() <= 1 + tolerance, f"Found probability > 1: {values.max()}"


def assert_no_data_leakage(df, prediction_date=None):
    """Assert no future data is used in features."""
    if prediction_date and "feature_timestamp" in df.columns:
        future_data = df[df["feature_timestamp"] > prediction_date]
        assert len(future_data) == 0, f"Found {len(future_data)} rows with future data"
