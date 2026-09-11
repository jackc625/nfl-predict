"""The two in-capture detectors: recorded, unable to fail a capture, unable to go quiet.

Every test here uses ``tmp_path`` as its data root, its live manifest directory, its
sealed lock and its sealed probe log, so NO real data lake is touched and NO committed
file is written -- not ``config/upstream_pin.json``, not ``config/upstream_pin.sealed.lock``
and not ``config/upstream_probe_log.jsonl``. The GitHub Releases payloads come from
``tests/fixtures/github_releases/``, captured once from the live API and committed (see
that directory's README), so nothing here reaches the network. The module-level
``data_boundary_guard`` is belt and braces on top of that: it content-hashes the
PRODUCTION ``data/`` tree around EVERY test, because ``git status --porcelain data/``
cannot fail -- ``.gitignore:22`` blankets the tree and returns empty whether the archive
is intact or destroyed.

WHAT THIS MODULE IS ABOUT. D32-07 puts a detector INSIDE the thing it is watching, and
the two requirements that creates pull against each other:

* It must never be able to break its host. A capture reads and writes PINNED bytes and is
  provably unaffected by anything upstream did, so failing it on a probe error would make
  the detector a liability -- and a detector that stops the Friday run earns an override
  flag within a month, after which it is off.
* It must never be able to go quiet. A detector that silently stopped and a healthy
  system are the same observable (PITFALLS F2), and the quiet failure is the one nobody
  finds.

A bare ``except: pass`` satisfies the first and destroys the second. The only thing that
reconciles them is recording an explicit UNKNOWN carrying the failure's own text, on a
verdict shaped exactly like a ruled one -- which is what every guard below is asserted to
do.

ON "FETCHES NOTHING". ``--detect-only`` fetches no LIVE dataset content: it writes no
snapshot, appends no capture and writes no manifest, and ``fetch_live_guarded`` is never
reached. The SEALED content probe does deliberately read the sealed upstream asset,
because reading the asset is what a content probe IS; that read never writes anything,
and its scratch parquet lives in a temporary directory that is deleted before the digest
is returned.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from data import sealed_probe, upstream_live
from data.revision_events import RevisionEventClass, RevisionSeverity, severity_rank
from data.sealed_probe import (
    SealedProbeUnavailable,
    parse_release_payload,
    probe_sealed,
)
from data.sealed_probe_log import read_probe_log
from data.storage import save_bronze_snapshot
from data.upstream_pin import (
    DATASET_TABLE_NAMES,
    LIVE_ZONE_FIRST_SEASON,
    SEALED_LOCK_SCHEMA_VERSION,
    UpstreamLiveCorrupt,
    UpstreamPinError,
    digest_file,
)
from scripts import capture_live_season, pin_upstream_snapshot
from tests.data_boundary import diff_digests, digest_tree, is_stat_signature

# EVERY test in this module guards the production data root by CONTENT.
pytestmark = pytest.mark.usefixtures("data_boundary_guard")

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "github_releases"

LIVE_SEASON = LIVE_ZONE_FIRST_SEASON

# The sealed pair every lock below is built around. 2010 is deep inside the sealed zone
# and is present in BOTH committed release payloads, so the metadata baseline can be read
# from the fixture rather than invented.
SEALED_SEASON = 2010

RELEASE_TAGS = ("pbp", "schedules", "depth_charts")


# ---------------------------------------------------------------------------
# Frames, payloads and locks
# ---------------------------------------------------------------------------


def _pbp_frame(season: int, weeks: tuple[int, ...] = (1,), epa: float = 0.25):
    """A play-by-play-shaped frame carrying pinned column names."""
    rows = []
    for week in weeks:
        rows.append(
            {
                "game_id": f"{season}_{week:02d}_HOME_AWAY",
                "season": season,
                "week": week,
                "posteam": "HOME",
                "defteam": "AWAY",
                "epa": epa,
            }
        )
    return pd.DataFrame(rows)


def _schedules_frame(seasons: tuple[int, ...], result: float = 3.0):
    """A schedules-shaped frame, optionally spanning SEVERAL seasons.

    The multi-season form is the real upstream shape: ``nflreadpy.load_schedules``
    downloads ONE monolithic ``games.parquet`` covering 1999-2026 and filters it in
    memory. That filter is half of why the lock's ``sha256`` and the raw published asset
    are different bytes by construction.
    """
    return pd.DataFrame(
        [
            {
                "game_id": f"{season}_01_HOME_AWAY",
                "season": season,
                "week": 1,
                "home_team": "HOME",
                "away_team": "AWAY",
                "result": result,
            }
            for season in seasons
        ]
    )


def _fixture_assets(tag: str) -> dict[str, dict]:
    """The validated ``{asset name: signature}`` map from a committed payload."""
    payload = json.loads((FIXTURE_DIR / f"{tag}.json").read_text(encoding="utf-8"))
    return parse_release_payload(payload)


def _pin_basis_digest(
    frame: pd.DataFrame, dataset: str, season: int, root: Path
) -> str:
    """The digest the SEALED LOCK records, recomputed INDEPENDENTLY of the probe.

    This is ``scripts.pin_upstream_snapshot.capture_season``'s own arithmetic -- narrow,
    write through ``save_bronze_snapshot``, hash the file -- spelled out here rather than
    obtained from the code under test, so "the probe digests on the lock's basis" is a
    comparison against an independently built value and not a tautology.
    """
    path = save_bronze_snapshot(
        pin_upstream_snapshot.narrow(dataset, frame),
        table_name=DATASET_TABLE_NAMES[dataset],
        season=season,
        week=0,
        base_path=root,
    )
    return digest_file(path)


def _lock(
    path: Path,
    *,
    pbp_updated_at: str | None,
    pbp_size: int | None,
    schedules_sha256: str,
) -> Path:
    """Write a two-pair sealed lock: one metadata pair and one content pair."""
    document = {
        "schema_version": SEALED_LOCK_SCHEMA_VERSION,
        "format_note": "test fixture lock",
        "sealed_through_season": 2025,
        "generated_from": "tests/integration/test_detect_only.py",
        "generated_at_utc": "2026-09-11T00:00:00+00:00",
        "datasets": {
            "pbp": {
                str(SEALED_SEASON): {
                    "sha256": "a" * 64,
                    "rows": 1,
                    "bytes": 1,
                    "upstream_updated_at": pbp_updated_at,
                    "upstream_size": pbp_size,
                    "acknowledgement": None,
                }
            },
            "schedules": {
                str(SEALED_SEASON): {
                    "sha256": schedules_sha256,
                    "rows": 1,
                    "bytes": 1,
                    "upstream_updated_at": None,
                    "upstream_size": None,
                    "acknowledgement": None,
                }
            },
        },
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def _json_block(captured: str) -> dict:
    """Parse the ``--json`` payload out of stdout.

    This project logs through structlog, which writes to STDOUT, so the JSON document
    shares the stream with any log line the run emitted. The payload is printed with
    ``indent=2``, so it begins at a line that is exactly ``{`` and ends at the last line
    that is exactly ``}``.
    """
    lines = captured.splitlines()
    starts = [index for index, line in enumerate(lines) if line == "{"]
    ends = [index for index, line in enumerate(lines) if line == "}"]
    assert starts and ends, f"no JSON document was printed:\n{captured}"
    return json.loads("\n".join(lines[starts[0] : ends[-1] + 1]))


# ---------------------------------------------------------------------------
# The controllable stand-in for nflverse and for the GitHub API
# ---------------------------------------------------------------------------


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> dict:
    """One place every outbound call is answered from, and recorded.

    ``frames`` maps ``(dataset, season)`` to the frame nflverse would return -- the SAME
    seam the capture and the pin-basis content digest both go through, because in
    production they are the same call. ``assets`` maps a release tag to the committed
    fixture payload, or to ``None`` to make that tag unfetchable.
    """
    state: dict = {
        "frames": {},
        "assets": {tag: _fixture_assets(tag) for tag in RELEASE_TAGS},
        "fetched": [],
        "tags_fetched": [],
        "graded": {
            "season": LIVE_SEASON,
            "weeks": [],
            "source": "test stub",
            "resolved": True,
            "reason": "stubbed: the real bet-list store is never read here",
        },
    }

    def _fetch_live(dataset: str, season: int) -> pd.DataFrame:
        state["fetched"].append((dataset, season))
        frame = state["frames"].get((dataset, season))
        if frame is None:
            msg = f"no stubbed upstream frame for {dataset} {season}"
            raise ValueError(msg)
        return frame.copy()

    def _fetch_release_assets(tag, *, session=None, timeout=None, rate_limits=None):
        state["tags_fetched"].append(tag)
        if rate_limits is not None:
            rate_limits[tag] = 57
        payload = state["assets"].get(tag)
        if payload is None:
            msg = f"stubbed: tag {tag!r} is unreachable in this test"
            raise SealedProbeUnavailable(msg)
        return payload

    def _graded_weeks_record(season, *, output_dir=None):
        graded = state["graded"]
        if isinstance(graded, BaseException):
            raise graded
        return graded

    monkeypatch.setattr(pin_upstream_snapshot, "fetch_live", _fetch_live)
    monkeypatch.setattr(
        capture_live_season, "fetch_release_assets", _fetch_release_assets
    )
    monkeypatch.setattr("data.graded_weeks.graded_weeks_record", _graded_weeks_record)
    return state


@pytest.fixture
def env(tmp_path: Path, upstream: dict) -> dict:
    """A whole sandboxed world: roots, a matching sealed lock and a clean probe log.

    The lock is built so that a probe of an UNCHANGED upstream reads CLEAN: the metadata
    baseline is the fixture payload's own signature for ``play_by_play_2010.parquet``,
    and the content baseline is the pin-basis digest of the 2010 schedules frame,
    recomputed here through the pin's own capture arithmetic.
    """
    upstream["frames"][("pbp", LIVE_SEASON)] = _pbp_frame(LIVE_SEASON, weeks=(1,))
    upstream["frames"][("schedules", LIVE_SEASON)] = _schedules_frame((LIVE_SEASON,))
    upstream["frames"][("schedules", SEALED_SEASON)] = _schedules_frame(
        (SEALED_SEASON,)
    )

    signature = upstream["assets"]["pbp"][f"play_by_play_{SEALED_SEASON}.parquet"]
    schedules_sha = _pin_basis_digest(
        upstream["frames"][("schedules", SEALED_SEASON)],
        "schedules",
        SEALED_SEASON,
        tmp_path / "pin-basis",
    )
    lock_path = _lock(
        tmp_path / "upstream_pin.sealed.lock",
        pbp_updated_at=signature["updated_at"],
        pbp_size=signature["size"],
        schedules_sha256=schedules_sha,
    )
    return {
        "data_root": tmp_path / "data",
        "manifest_dir": tmp_path / "upstream_live",
        "lock_path": lock_path,
        "log_path": tmp_path / "upstream_probe_log.jsonl",
        "schedules_sha": schedules_sha,
        "upstream": upstream,
        "tmp_path": tmp_path,
    }


def _capture_argv(env: dict, week: int, dataset: str = "pbp") -> list[str]:
    return [
        "--season",
        str(LIVE_SEASON),
        "--week",
        str(week),
        "--dataset",
        dataset,
        "--data-root",
        str(env["data_root"]),
        "--manifest-dir",
        str(env["manifest_dir"]),
        "--sealed-lock",
        str(env["lock_path"]),
        "--sealed-log",
        str(env["log_path"]),
    ]


def _detect_argv(env: dict, dataset: str = "pbp", *, as_json: bool = True) -> list[str]:
    argv = [
        "--season",
        str(LIVE_SEASON),
        "--detect-only",
        "--dataset",
        dataset,
        "--manifest-dir",
        str(env["manifest_dir"]),
        "--sealed-lock",
        str(env["lock_path"]),
        "--sealed-log",
        str(env["log_path"]),
    ]
    if as_json:
        argv.append("--json")
    return argv


def _recorded_entry(env: dict, dataset: str = "pbp") -> dict:
    manifest = upstream_live.load_live_manifest(
        LIVE_SEASON, manifest_dir=env["manifest_dir"]
    )
    assert manifest is not None, "the capture wrote no live manifest at all"
    return manifest["datasets"][dataset]["captures"][-1]


class TestDetectOnlyFetchesNothingAndWritesNoCapture:
    """T-32-39: an ad-hoc question must not be able to write a record."""

    def test_no_dataset_content_is_fetched(
        self, env: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert capture_live_season.main(_capture_argv(env, 1)) == (
            capture_live_season.EXIT_OK
        )

        reached: list[tuple[str, int]] = []

        def _record(dataset: str, season: int):
            reached.append((dataset, season))
            msg = "--detect-only reached the LIVE dataset fetch"
            raise AssertionError(msg)

        monkeypatch.setattr(capture_live_season, "fetch_live_guarded", _record)
        code = capture_live_season.main(_detect_argv(env))

        assert reached == [], f"--detect-only fetched live dataset content: {reached}"
        assert code != capture_live_season.EXIT_CAPTURE_FAILED, (
            "--detect-only reported a capture failure, which means it tried to capture"
        )

    def test_the_live_manifest_bytes_are_unchanged(self, env: dict) -> None:
        capture_live_season.main(_capture_argv(env, 1))
        manifest_path = upstream_live.live_manifest_path(
            LIVE_SEASON, manifest_dir=env["manifest_dir"]
        )
        before = manifest_path.read_bytes()

        capture_live_season.main(_detect_argv(env))

        assert manifest_path.read_bytes() == before, (
            "--detect-only rewrote the committed live manifest; it must write no capture "
            "and no manifest at all"
        )

    def test_the_fixture_data_root_is_unchanged(self, env: dict) -> None:
        capture_live_season.main(_capture_argv(env, 1))
        before = digest_tree(env["data_root"])
        assert before, "the capture wrote nothing, so this comparison proves nothing"
        assert not any(is_stat_signature(value) for value in before.values()), (
            "the before snapshot fell back to a stat signature, so this comparison would "
            "mix instruments"
        )

        capture_live_season.main(_detect_argv(env))

        after = digest_tree(env["data_root"])
        assert not any(is_stat_signature(value) for value in after.values()), (
            "the after snapshot fell back to a stat signature"
        )
        assert diff_digests(before, after) == {
            "added": [],
            "removed": [],
            "changed": [],
        }

    def test_exactly_one_line_is_appended_to_the_sealed_log_recording_detect_only_mode(
        self, env: dict
    ) -> None:
        capture_live_season.main(_capture_argv(env, 1))
        before_bytes = env["log_path"].read_bytes()
        before_lines = len(read_probe_log(env["log_path"]))

        capture_live_season.main(_detect_argv(env))

        entries = read_probe_log(env["log_path"])
        assert len(entries) == before_lines + 1, (
            "--detect-only did not append EXACTLY one line; a probe that ran and left no "
            "trace reintroduces the dead-detector ambiguity the log closes"
        )
        after_bytes = env["log_path"].read_bytes()
        assert after_bytes.startswith(before_bytes), (
            "the detect-only append REWROTE an earlier line; the log is append-only"
        )
        assert entries[-1]["mode"] == capture_live_season.PROBE_LOG_MODE_DETECT_ONLY
        assert entries[-2]["mode"] == capture_live_season.PROBE_LOG_MODE_CAPTURE


class TestDetectOnlyIsTheSameCodePath:
    """D32-07: the ad-hoc check and the capture must not be two rulings."""

    def test_it_reproduces_the_verdict_the_capture_recorded(
        self, env: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env["upstream"]["frames"][("pbp", LIVE_SEASON)] = _pbp_frame(
            LIVE_SEASON, weeks=(1,), epa=0.25
        )
        capture_live_season.main(_capture_argv(env, 1))
        env["upstream"]["frames"][("pbp", LIVE_SEASON)] = _pbp_frame(
            LIVE_SEASON, weeks=(1,), epa=0.75
        )
        capture_live_season.main(_capture_argv(env, 2))

        recorded = dict(_recorded_entry(env)[upstream_live.CAPTURE_VERDICT_KEY])
        assert recorded["event_class"] == str(RevisionEventClass.LIVE_REVISION), (
            "the second capture did not see the restated week, so the comparison below "
            "would hold for the wrong reason"
        )
        capsys.readouterr()

        capture_live_season.main(_detect_argv(env))
        replayed = dict(_json_block(capsys.readouterr().out)["live:pbp"])

        recorded.pop("probed_at_utc", None)
        replayed.pop("probed_at_utc", None)
        assert replayed == recorded, (
            "--detect-only produced a DIFFERENT verdict from the one the capture "
            "recorded for the same two entries, so there are two rulings in this "
            "codebase and only one of them is in the committed record"
        )

    def test_with_fewer_than_two_captures_the_verdict_is_no_prior_capture(
        self, env: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        capture_live_season.main(_capture_argv(env, 1))
        capsys.readouterr()

        capture_live_season.main(_detect_argv(env))
        verdict = _json_block(capsys.readouterr().out)["live:pbp"]

        assert verdict["event_class"] == str(RevisionEventClass.NO_PRIOR_CAPTURE), (
            "one capture was ruled CLEAN rather than 'nothing to compare'; those are "
            "different claims and a season of entries has to tell them apart"
        )

    def test_with_no_captures_at_all_it_is_still_no_prior_capture(
        self, env: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        capture_live_season.main(_detect_argv(env))
        verdict = _json_block(capsys.readouterr().out)["live:pbp"]
        assert verdict["event_class"] == str(RevisionEventClass.NO_PRIOR_CAPTURE)


class TestTheExitCodeContractIsPinned:
    """D32-09: the code is the machine-readable half of the verdict (T-32-41)."""

    def test_the_exit_code_constants_have_their_pinned_values(self) -> None:
        assert capture_live_season.EXIT_OK == 0
        assert capture_live_season.EXIT_CAPTURE_FAILED == 1
        assert capture_live_season.EXIT_USAGE == 2
        assert capture_live_season.EXIT_CRITICAL == 3
        assert capture_live_season.EXIT_UNKNOWN == 4

    def test_a_clean_run_returns_zero(self, env: dict) -> None:
        assert capture_live_season.main(_capture_argv(env, 1)) == (
            capture_live_season.EXIT_OK
        )
        code = capture_live_season.main(_detect_argv(env))
        entries = read_probe_log(env["log_path"])
        assert entries[-1]["event_class"] == str(RevisionEventClass.CLEAN), (
            "the sealed probe did not read CLEAN against an unchanged upstream, so the "
            "exit code below would be right for the wrong reason: "
            f"{entries[-1]['reason']}"
        )
        assert code == capture_live_season.EXIT_OK

    def test_a_sealed_critical_returns_three_from_detect_only_and_zero_from_a_capture(
        self, env: dict
    ) -> None:
        """D32-09 in one assertion: loud where a check reads it, silent where a
        scheduler does."""
        moved = dict(env["upstream"]["assets"]["pbp"])
        name = f"play_by_play_{SEALED_SEASON}.parquet"
        moved[name] = {**moved[name], "updated_at": "2026-09-01T00:00:00Z"}
        env["upstream"]["assets"]["pbp"] = moved

        capture_code = capture_live_season.main(_capture_argv(env, 1))
        detect_code = capture_live_season.main(_detect_argv(env))

        entries = read_probe_log(env["log_path"])
        assert entries[-1]["event_class"] == str(RevisionEventClass.SEALED_REVISION)
        assert capture_code == capture_live_season.EXIT_OK, (
            "a sealed finding made the CAPTURE report failure; the capture recorded the "
            "bytes it was given and is provably unaffected"
        )
        assert detect_code == capture_live_season.EXIT_CRITICAL

    def test_an_unknown_returns_four(self, env: dict) -> None:
        env["upstream"]["assets"]["pbp"] = None

        code = capture_live_season.main(_detect_argv(env))

        entries = read_probe_log(env["log_path"])
        assert entries[-1]["event_class"] == str(RevisionEventClass.UNKNOWN)
        assert code == capture_live_season.EXIT_UNKNOWN

    def test_a_critical_and_an_unknown_together_return_three(self, env: dict) -> None:
        """The adjacency the rule is least obvious about, and BOTH stay recorded."""
        moved = dict(env["upstream"]["assets"]["pbp"])
        name = f"play_by_play_{SEALED_SEASON}.parquet"
        moved[name] = {**moved[name], "updated_at": "2026-09-01T00:00:00Z"}
        env["upstream"]["assets"]["pbp"] = moved

        from data.graded_weeks import GradedWeeksUnavailable

        env["upstream"]["graded"] = GradedWeeksUnavailable("stubbed unreadable store")

        verdicts = capture_live_season.run_detectors(
            dataset="pbp",
            season=LIVE_SEASON,
            current_entry={"week": 1, "sequence": 1, "week_digests": {}},
            prior_entry={"week": 1, "sequence": 1, "week_digests": {}},
            sealed=capture_live_season.SealedProbeRun(lock_path=env["lock_path"]),
        )

        assert verdicts["sealed"]["severity"] == str(RevisionSeverity.CRITICAL)
        assert verdicts["live"]["event_class"] == str(RevisionEventClass.UNKNOWN)
        assert capture_live_season.exit_code_for(verdicts) == (
            capture_live_season.EXIT_CRITICAL
        )
        # Nothing is lost to the precedence: both verdicts survive in full.
        assert verdicts["live"]["reason"]
        assert verdicts["sealed"]["findings"]

    def test_a_usage_error_returns_two(self, env: dict) -> None:
        code = capture_live_season.main(
            [
                "--season",
                str(LIVE_SEASON),
                "--manifest-dir",
                str(env["manifest_dir"]),
                "--sealed-lock",
                str(env["lock_path"]),
                "--sealed-log",
                str(env["log_path"]),
            ]
        )
        assert code == capture_live_season.EXIT_USAGE

    def test_argparse_usage_also_exits_two(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            capture_live_season.main(["--week", "1"])
        assert exit_info.value.code == capture_live_season.EXIT_USAGE

    def test_the_mapping_is_derived_through_the_frozen_severity_ladder(self) -> None:
        assert {
            severity_rank(RevisionSeverity.INFORMATIONAL): capture_live_season.EXIT_OK,
            severity_rank(RevisionSeverity.WARNING): capture_live_season.EXIT_UNKNOWN,
            severity_rank(RevisionSeverity.CRITICAL): capture_live_season.EXIT_CRITICAL,
        } == capture_live_season._EXIT_BY_SEVERITY_RANK
        assert capture_live_season.exit_code_for({}) == capture_live_season.EXIT_OK


class TestADetectorFailureIsRecordedAndNeverSwallowed:
    """T-32-02 and T-32-38: UNKNOWN carrying its reason, and the capture survives."""

    def test_an_unreachable_release_api_records_unknown_and_the_capture_succeeds(
        self, env: dict
    ) -> None:
        for tag in RELEASE_TAGS:
            env["upstream"]["assets"][tag] = None

        code = capture_live_season.main(_capture_argv(env, 1))

        assert code == capture_live_season.EXIT_OK
        entry = _recorded_entry(env)
        assert entry["rows"] == 1, "the capture entry was not written"
        recorded = read_probe_log(env["log_path"])[-1]
        assert recorded["event_class"] == str(RevisionEventClass.UNKNOWN)
        assert recorded["reason"].strip(), "an UNKNOWN was recorded with no reason"
        assert recorded["event_class"] != str(RevisionEventClass.CLEAN)

    def test_an_unresolvable_graded_set_records_unknown_with_nothing_claimed(
        self, env: dict
    ) -> None:
        from data.graded_weeks import GradedWeeksUnavailable

        # A FIRST capture is ruled ``no_prior_capture`` before the graded set is ever
        # consulted, so the store has to break on a capture that HAS something to diff
        # against -- otherwise this test would pass without reaching the branch.
        capture_live_season.main(_capture_argv(env, 1))
        env["upstream"]["graded"] = GradedWeeksUnavailable("stubbed unreadable store")

        code = capture_live_season.main(_capture_argv(env, 2))

        assert code == capture_live_season.EXIT_OK
        verdict = _recorded_entry(env)[upstream_live.CAPTURE_VERDICT_KEY]
        assert verdict["event_class"] == str(RevisionEventClass.UNKNOWN)
        assert verdict["correction_owed"] is None, (
            "a ruling made without knowing what was graded claimed nothing was owed; "
            "False is a claim this branch cannot make"
        )
        assert "stubbed unreadable store" in verdict["reason"]

    def test_an_arbitrary_failure_in_either_half_still_returns_a_verdict_mapping(
        self, env: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom_live(**_kwargs):
            msg = "arbitrary live-half failure"
            raise RuntimeError(msg)

        def _boom_sealed(*_args, **_kwargs):
            msg = "arbitrary sealed-half failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(capture_live_season, "detect_live_revision", _boom_live)
        monkeypatch.setattr(capture_live_season, "probe_sealed", _boom_sealed)

        verdicts = capture_live_season.run_detectors(
            dataset="pbp",
            season=LIVE_SEASON,
            current_entry={"week": 4, "sequence": 2, "week_digests": {}},
            sealed=capture_live_season.SealedProbeRun(lock_path=env["lock_path"]),
        )

        assert set(verdicts) == {"live", "sealed"}
        assert verdicts["live"]["event_class"] == str(RevisionEventClass.UNKNOWN)
        assert "arbitrary live-half failure" in verdicts["live"]["reason"]
        assert verdicts["sealed"]["event_class"] == str(RevisionEventClass.UNKNOWN)
        assert "arbitrary sealed-half failure" in verdicts["sealed"]["reason"]
        assert verdicts["sealed"]["expected"] == 2, (
            "a failed probe recorded no coverage, so the line cannot be audited"
        )
        assert verdicts["sealed"]["checked"] == 0

    def test_one_half_going_blind_does_not_blind_the_other(
        self, env: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom_sealed(*_args, **_kwargs):
            msg = "arbitrary sealed-half failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(capture_live_season, "probe_sealed", _boom_sealed)

        verdicts = capture_live_season.run_detectors(
            dataset="pbp",
            season=LIVE_SEASON,
            current_entry={"week": 4, "sequence": 2, "week_digests": {}},
            sealed=capture_live_season.SealedProbeRun(lock_path=env["lock_path"]),
        )

        assert verdicts["live"]["event_class"] == str(
            RevisionEventClass.NO_PRIOR_CAPTURE
        ), "the sealed half's failure also took out the live ruling"

    def test_a_pin_refusal_is_re_raised_rather_than_recorded(
        self, env: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WR-10, as behaviour and not only as source order.

        ``scripts/ingest_games.py:308-320`` records this exact defect reaching
        production: a broad handler caught the pin's refusal, one warning was logged and
        silver was written with no scores merged. A detector is the worst possible place
        to repeat it.
        """

        def _refuse(**_kwargs):
            msg = "a pin refusal raised inside the detector"
            raise UpstreamPinError(msg)

        monkeypatch.setattr(capture_live_season, "detect_live_revision", _refuse)

        with pytest.raises(UpstreamPinError, match="a pin refusal raised"):
            capture_live_season.run_detectors(
                dataset="pbp",
                season=LIVE_SEASON,
                current_entry={"week": 1, "sequence": 1, "week_digests": {}},
                sealed=capture_live_season.SealedProbeRun(lock_path=env["lock_path"]),
            )

    def test_the_pin_refusal_clause_precedes_every_broad_handler(self) -> None:
        import inspect

        source = inspect.getsource(capture_live_season)
        first_refusal = source.find("except UpstreamPinError")
        first_broad = source.find("except Exception")
        assert first_refusal != -1, "the WR-10 construct is absent"
        assert first_broad != -1
        assert first_refusal < first_broad, (
            "a broad handler is reached before the pin-refusal re-raise, which swallows "
            "the one failure the exception hierarchy exists to protect"
        )

    def test_a_guarded_verdict_has_the_same_key_set_as_a_ruled_one(
        self, env: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*_args, **_kwargs):
            msg = "failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(capture_live_season, "detect_live_revision", _boom)
        monkeypatch.setattr(capture_live_season, "probe_sealed", _boom)

        verdicts = capture_live_season.run_detectors(
            dataset="pbp",
            season=LIVE_SEASON,
            current_entry={"week": 1, "sequence": 1, "week_digests": {}},
            sealed=capture_live_season.SealedProbeRun(lock_path=env["lock_path"]),
        )

        from data.live_revision import VERDICT_KEYS as LIVE_KEYS

        assert set(verdicts["live"]) == set(LIVE_KEYS)
        assert set(verdicts["sealed"]) == set(sealed_probe.VERDICT_KEYS)


class TestASealedCriticalIsLoudAndBlocksNothing:
    """D32-09: recorded and audible, and still not a reason to stop anything."""

    def test_it_names_the_dataset_and_the_seasons_and_leaves_the_pin_untouched(
        self, env: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        moved = dict(env["upstream"]["assets"]["pbp"])
        name = f"play_by_play_{SEALED_SEASON}.parquet"
        moved[name] = {**moved[name], "updated_at": "2026-09-01T00:00:00Z"}
        env["upstream"]["assets"]["pbp"] = moved
        pin_before = Path("config/upstream_pin.json").read_bytes()
        lock_before = env["lock_path"].read_bytes()

        code = capture_live_season.main(_capture_argv(env, 1))

        captured = capsys.readouterr()
        printed = captured.out + captured.err
        assert code == capture_live_season.EXIT_OK
        assert "warning" in captured.out.lower(), (
            f"no WARNING-level record was emitted for a sealed CRITICAL:\n{printed}"
        )
        assert "pbp" in printed and str(SEALED_SEASON) in printed, (
            f"the warning names neither the dataset nor the season:\n{printed}"
        )
        assert Path("config/upstream_pin.json").read_bytes() == pin_before, (
            "a sealed finding RE-FROZE the pin, erasing the evidence the probe exists "
            "to produce"
        )
        assert env["lock_path"].read_bytes() == lock_before, (
            "the probe rewrote its own baseline, which would report clean forever"
        )

    def test_the_finding_is_recorded_in_full_in_the_committed_log(
        self, env: dict
    ) -> None:
        moved = dict(env["upstream"]["assets"]["pbp"])
        name = f"play_by_play_{SEALED_SEASON}.parquet"
        moved[name] = {**moved[name], "updated_at": "2026-09-01T00:00:00Z"}
        env["upstream"]["assets"]["pbp"] = moved

        capture_live_season.main(_capture_argv(env, 1))

        finding = read_probe_log(env["log_path"])[-1]["findings"][0]
        assert finding["dataset"] == "pbp"
        assert finding["season"] == SEALED_SEASON
        assert finding["observed_updated_at"] == "2026-09-01T00:00:00Z"
        assert finding["acknowledged"] is False


class TestTheVerdictRidesInsideItsOwnCaptureEntry:
    """D32-08 and T-32-40: the finding and the bytes cannot drift apart."""

    def test_the_entry_carries_the_verdict_and_the_digest_it_was_ruled_from(
        self, env: dict
    ) -> None:
        capture_live_season.main(_capture_argv(env, 1))
        entry = _recorded_entry(env)

        assert upstream_live.CAPTURE_VERDICT_KEY in entry
        verdict = entry[upstream_live.CAPTURE_VERDICT_KEY]
        assert verdict["week"] == entry["week"]
        assert verdict["sequence"] == entry["sequence"]
        assert entry["week_digests"], (
            "the entry carries no digest for the verdict to be about"
        )
        assert json.loads(
            upstream_live.live_manifest_path(
                LIVE_SEASON, manifest_dir=env["manifest_dir"]
            ).read_text(encoding="utf-8")
        )["datasets"]["pbp"]["captures"][-1][upstream_live.CAPTURE_VERDICT_KEY], (
            "the verdict is in memory but not in the committed write"
        )

    def test_attaching_a_second_verdict_to_an_entry_is_refused(self, env: dict) -> None:
        capture_live_season.main(_capture_argv(env, 1))
        entry = _recorded_entry(env)

        with pytest.raises(UpstreamLiveCorrupt, match="already carries"):
            upstream_live.attach_verdict(entry, {"event_class": "clean"})


class TestTheCommittedProbeLogLine:
    """D32-08: one line per RUN, clean and UNKNOWN alike, from an explicit key set."""

    def test_one_line_per_run_even_for_two_datasets(self, env: dict) -> None:
        code = capture_live_season.main(
            [
                "--season",
                str(LIVE_SEASON),
                "--week",
                "1",
                "--dataset",
                "pbp",
                "--dataset",
                "schedules",
                "--data-root",
                str(env["data_root"]),
                "--manifest-dir",
                str(env["manifest_dir"]),
                "--sealed-lock",
                str(env["lock_path"]),
                "--sealed-log",
                str(env["log_path"]),
            ]
        )

        assert code == capture_live_season.EXIT_OK
        assert len(read_probe_log(env["log_path"])) == 1, (
            "the probe log gained one line PER DATASET; D32-08 makes it one per RUN"
        )
        assert env["upstream"]["tags_fetched"] == sorted(RELEASE_TAGS), (
            "the release API was polled more than once per run, which spends the 60/hr "
            f"ceiling to ask the same question twice: {env['upstream']['tags_fetched']}"
        )
        assert read_probe_log(env["log_path"])[0]["datasets"] == ["pbp", "schedules"]

    def test_a_clean_run_appends_its_line_too(self, env: dict) -> None:
        capture_live_season.main(_capture_argv(env, 1))
        entry = read_probe_log(env["log_path"])[-1]
        assert entry["event_class"] == str(RevisionEventClass.CLEAN), entry["reason"]

    def test_the_line_carries_exactly_the_frozen_key_set(self, env: dict) -> None:
        capture_live_season.main(_capture_argv(env, 1))
        entry = read_probe_log(env["log_path"])[-1]

        assert set(entry) == set(capture_live_season.PROBE_LOG_ENTRY_KEYS)
        assert len(capture_live_season.PROBE_LOG_ENTRY_KEYS) == len(
            set(capture_live_season.PROBE_LOG_ENTRY_KEYS)
        )
        # The thirteen Plan 32-08 named, each by name, plus the schema stamp every other
        # committed record in this phase carries.
        for key in (
            "probed_at_utc",
            "event_class",
            "severity",
            "checked",
            "expected",
            "unresolved",
            "strategy",
            "findings",
            "reason",
            "rate_limit_remaining",
            "mode",
            "season",
            "datasets",
        ):
            assert key in entry
        assert entry["verdict_schema_version"] == 1

    def test_the_line_is_built_from_the_verdict_and_not_from_the_payload(
        self, env: dict
    ) -> None:
        """T-32-06: no upstream-supplied field rides into a committed file."""
        polluted = dict(env["upstream"]["assets"]["pbp"])
        name = f"play_by_play_{SEALED_SEASON}.parquet"
        polluted[name] = {**polluted[name], "author_login": "attacker"}
        env["upstream"]["assets"]["pbp"] = polluted

        capture_live_season.main(_capture_argv(env, 1))

        raw = env["log_path"].read_text(encoding="utf-8")
        assert "author_login" not in raw
        assert "attacker" not in raw

    def test_a_stale_log_is_named_before_the_new_line_is_appended(
        self, env: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        old = datetime.now(UTC) - timedelta(
            days=capture_live_season.SEALED_PROBE_CADENCE_DAYS + 5
        )
        env["log_path"].write_text(
            json.dumps(
                {
                    "event_class": "clean",
                    "severity": "informational",
                    "probed_at_utc": old.isoformat(),
                    "checked": 2,
                    "expected": 2,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        capture_live_season.main(_capture_argv(env, 1))

        printed = capsys.readouterr().out
        assert "NOT to have run" in printed or "not to have run" in printed, (
            f"a gap in the liveness record went unremarked:\n{printed}"
        )
        assert len(read_probe_log(env["log_path"])) == 2

    def test_an_unreadable_log_warns_but_never_fails_the_capture(
        self, env: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        env["log_path"].write_text("{not json at all\n", encoding="utf-8")

        code = capture_live_season.main(_capture_argv(env, 1))

        assert code == capture_live_season.EXIT_OK, (
            "the detector's own bookkeeping failure stopped a capture that had already "
            "written and verified its bytes"
        )
        assert "could not be read" in capsys.readouterr().out


class TestTheContentBasisIsThePinsOwnAndNotTheRawStream:
    """The load-bearing contract 32-05 could only STATE, because probe_sealed is pure.

    ``pbp`` is column-narrowed and ``schedules`` is filtered out of a monolithic
    1999-2026 asset before this project writes its parquet, so the lock's ``sha256`` and
    the raw published bytes are different by construction. Handing ``probe_sealed`` a
    raw-stream digest would report a move on EVERY content pair, forever -- a detector
    that cries wolf on all of them is exactly as useless as one that stays silent.
    """

    def test_the_probe_digests_on_the_locks_own_basis(self, env: dict) -> None:
        computed = capture_live_season._pinned_basis_digest("schedules", SEALED_SEASON)
        assert computed == env["schedules_sha"], (
            "the probe's content digest is NOT the value the pin's own capture "
            "arithmetic produces for the same frame, so it can never agree with a lock "
            "entry and every content pair would report a move forever"
        )

    def test_the_raw_published_bytes_digest_to_something_else_entirely(
        self, env: dict, tmp_path: Path
    ) -> None:
        """The two bases are different bytes, demonstrated rather than asserted."""
        monolith = _schedules_frame((1999, SEALED_SEASON, LIVE_SEASON))
        published = tmp_path / "games.parquet"
        monolith.to_parquet(published, index=False)
        raw_stream_digest = hashlib.sha256(published.read_bytes()).hexdigest()

        assert raw_stream_digest != env["schedules_sha"]

    def test_a_raw_stream_digest_would_report_a_move_on_an_unchanged_pair(
        self, env: dict, tmp_path: Path
    ) -> None:
        """The consequence of getting the basis wrong, made concrete."""
        lock = json.loads(env["lock_path"].read_text(encoding="utf-8"))
        published = tmp_path / "games-raw.parquet"
        _schedules_frame((1999, SEALED_SEASON)).to_parquet(published, index=False)
        raw_stream_digest = hashlib.sha256(published.read_bytes()).hexdigest()

        on_the_wrong_basis = probe_sealed(
            lock,
            assets_by_tag={"pbp": env["upstream"]["assets"]["pbp"]},
            content_digests={("schedules", SEALED_SEASON): raw_stream_digest},
        )
        on_the_right_basis = probe_sealed(
            lock,
            assets_by_tag={"pbp": env["upstream"]["assets"]["pbp"]},
            content_digests={("schedules", SEALED_SEASON): env["schedules_sha"]},
        )

        assert on_the_wrong_basis["event_class"] == str(
            RevisionEventClass.SEALED_REVISION
        ), "the wrong basis did NOT report a move, so this test proves nothing"
        assert on_the_right_basis["event_class"] == str(RevisionEventClass.CLEAN)

    def test_the_probe_never_reaches_for_the_raw_stream_helper(self) -> None:
        import inspect

        source = inspect.getsource(capture_live_season.SealedProbeRun)
        assert "content_digest_for" not in source, (
            "the sealed probe calls data.sealed_probe.content_digest_for, which digests "
            "the RAW published stream -- the wrong basis for the lock's sha256"
        )

    def test_the_scratch_write_never_lands_in_a_data_lake(
        self, env: dict, tmp_path: Path
    ) -> None:
        before = digest_tree(tmp_path)
        capture_live_season._pinned_basis_digest("schedules", SEALED_SEASON)
        after = digest_tree(tmp_path)
        assert diff_digests(before, after) == {
            "added": [],
            "removed": [],
            "changed": [],
        }, "the pin-basis digest left its scratch parquet behind"


class TestSeedingTheSealedBaselineIsDeliberateAndAttributed:
    """Plan 32-09's one-time baseline writer, exercised here against a tmp lock."""

    def _seed_argv(self, env: dict, *, ruled_by: str | None, reseed: bool = False):
        argv = [
            "--season",
            str(LIVE_SEASON),
            "--seed-sealed-signatures",
            "--sealed-lock",
            str(env["lock_path"]),
        ]
        if ruled_by is not None:
            argv += ["--ruled-by", ruled_by]
        if reseed:
            argv.append("--allow-reseed")
        return argv

    def test_it_writes_an_attributed_baseline_and_captures_nothing(
        self, env: dict
    ) -> None:
        lock = json.loads(env["lock_path"].read_text(encoding="utf-8"))
        lock["datasets"]["pbp"][str(SEALED_SEASON)]["upstream_updated_at"] = None
        lock["datasets"]["pbp"][str(SEALED_SEASON)]["upstream_size"] = None
        env["lock_path"].write_text(json.dumps(lock, indent=2), encoding="utf-8")

        code = capture_live_season.main(self._seed_argv(env, ruled_by="tester"))

        assert code == capture_live_season.EXIT_OK
        seeded = json.loads(env["lock_path"].read_text(encoding="utf-8"))
        entry = seeded["datasets"]["pbp"][str(SEALED_SEASON)]
        signature = env["upstream"]["assets"]["pbp"][
            f"play_by_play_{SEALED_SEASON}.parquet"
        ]
        assert entry["upstream_updated_at"] == signature["updated_at"]
        assert entry["upstream_size"] == signature["size"]
        assert seeded["signatures_seeded_by"] == "tester"
        assert not env["log_path"].exists(), (
            "a baseline write appended a probe-log line; a line in that log means the "
            "detector RAN, and a seeding pass saw a baseline it was creating"
        )
        assert env["upstream"]["fetched"] == [("schedules", SEALED_SEASON)], (
            "seeding fetched a LIVE dataset; it captures nothing"
        )

    def test_without_an_attributed_ruled_by_it_refuses(self, env: dict) -> None:
        assert capture_live_season.main(self._seed_argv(env, ruled_by=None)) == (
            capture_live_season.EXIT_USAGE
        )
        assert capture_live_season.main(self._seed_argv(env, ruled_by="   ")) == (
            capture_live_season.EXIT_USAGE
        )

    def test_it_refuses_to_re_seed_an_existing_baseline_without_permission(
        self, env: dict
    ) -> None:
        before = env["lock_path"].read_bytes()

        refused = capture_live_season.main(self._seed_argv(env, ruled_by="tester"))

        assert refused == capture_live_season.EXIT_USAGE
        assert env["lock_path"].read_bytes() == before, (
            "a refused re-seed still rewrote the lock"
        )
        allowed = capture_live_season.main(
            self._seed_argv(env, ruled_by="tester", reseed=True)
        )
        assert allowed == capture_live_season.EXIT_OK
