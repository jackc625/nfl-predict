"""The ONE website push, shared by the daily run and the manual command (Phase 36, D-05, SPEC R5).

THE ORDER, and every step is load-bearing
-----------------------------------------
1. The setup switch (D-23). Unless :data:`PUSH_ENABLED_MARKER` says ``"enabled": true``, every
   attempt writes one ``not_pushed`` / ``push_not_enabled`` record and stops -- before a secret is
   read, the lock is taken, anything is rebuilt or stamped, or the CLI runs.
2. The secrets (D-01, D-15). A missing token or bypass value is a failed push, before the lock.
3. The publish lock (D-18, ``pipeline.cache_lock``), held for the rest of the attempt, so a push
   never overlaps a cache rebuild.
4. The refusals (SPEC R6, amended by D-03): an unfinished rebuild or a missing cache; a stylesheet
   that is out of step or cannot be checked; a data-format mismatch that a rebuild under this lock
   cannot fix.
5. A fresh publish id (uuid4) and push time (UTC), stamped on a COPY of the cache and swapped in
   (D-14, ``api.cache.stamp_publish``). They are minted here, per attempt, and never copied from
   the cache build or a previous push (SPEC must-NOT).
6. The SHA-256 of the stamped file, then ONE production deploy of code and cache together. The
   deploy uploads the WORKING TREE, so the record names the code it shipped: HEAD's SHA, marked
   ``+dirty`` when an uploaded path has an uncommitted change (:func:`_code_commit`).
7. The read-back: a fixed number of GETs of the production ``/health``. The push SUCCEEDED only when
   it returns this attempt's publish id AND the served file's SHA-256 equals the local digest. The
   CLI's exit code is never the evidence -- a deploy can exit 0 and serve something else. A
   read-back that never matches is a failed push: no rollback and no second deploy.
   Then ONE GET of each :data:`SMOKE_PATHS` route through the same client and header, each judged
   by the live check's own ``check_route``. ``/health`` reads four ``cache_meta`` keys and hashes a
   file, so a deployment can answer it while every page answers 500 -- the first real push did
   exactly that (Plan 36-10). Any page that fails makes the push failed (``pages_failed``).
8. Exactly one push-log record (``logs/site_pushes.jsonl``) and one ``SITE_PUSH=`` line.

SECRETS
-------
The token reaches the CLI only as ``VERCEL_TOKEN`` in the child's environment -- never on the
command line, where it would sit in the process list and the task history -- and the CLI runs with
``--global-config`` pointing at a gitignored directory, so the owner's interactive login is never
used. The bypass value is sent only as the ``x-vercel-protection-bypass`` request header, never in a
query string, through a client that does not follow redirects. Neither is ever printed or recorded;
a failure is recorded as a named reason or an exception CLASS name only (Phase 35 D-10).

SEAMS
-----
:data:`RUN_CLI`, :data:`CLIENT_FACTORY`, :data:`SLEEP` and :data:`LOAD_PUSH_SECRETS` are module names
read at call time, so tests inject a fake CLI, an ``httpx.MockTransport`` host, a recording sleeper
and dummy secrets. ``tests/conftest.py`` makes the default CLI refuse and quarantines the log, the
lock, the switch and the CLI config directory.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import SecretStr

from data.write_sink import append_jsonl
from pipeline.cache_lock import cache_publish_lock
from utils.file_lock import FileLockHeldError

if TYPE_CHECKING:
    from api.cache_schema import DataFormatState
    from conf.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_PATH = REPO_ROOT / "data" / "web_cache.duckdb"

# The D-23 switch. Created and committed only after the owner's one-way-door decision (Plan 36-10).
# The push runs only while it is a JSON object whose "enabled" is exactly true; set it to false (or
# `git rm` the file) to pause every push. A missing or unreadable file is switched off.
PUSH_ENABLED_MARKER = REPO_ROOT / "config" / "site_push_enabled.json"

#: The append-only push log: one record per attempt, daily or manual.
SITE_PUSH_LOG = REPO_ROOT / "logs" / "site_pushes.jsonl"

#: Every push-log record has exactly these fields, in this order.
SITE_PUSH_FIELDS: tuple[str, ...] = (
    "time_utc",
    "publish_id",
    "run_id",
    "source",
    "outcome",
    "reason",
    "deployment_id",
    "deployment_url",
    "local_sha256",
    "readback",
    "data_format_found",
    "data_format_expected",
    "code_commit",
)

# The code paths the deploy uploads (the .vercelignore allowlist, the cache aside): an uncommitted
# change under any of them marks the record's code_commit dirty.
UPLOADED_CODE_PATHS: tuple[str, ...] = (
    "api",
    "web",
    "utils",
    "conf",
    "index.py",
    "requirements.txt",
    ".python-version",
    "vercel.json",
    ".vercelignore",
)
DIRTY_SUFFIX = "+dirty"
GIT_TIMEOUT_SECONDS = 30

# The production alias Vercel assigned to the project at its first deployment (Plan 36-09, recorded
# 2026-10-19): the CLI's reported productionUrl. Vercel added the suffix to the bare project name.
PRODUCTION_URL = "https://nfl-predict-psi.vercel.app"

# The ONE CLI, by absolute path (D-16): the version-specific nvm install survives `nvm use`, which
# re-points C:\nvm4w\nodejs, and the S4U task has no user PATH to find a bare `vercel` on.
NODE_EXE = Path(r"C:\Users\jackc\AppData\Local\nvm\v24.18.0\node.exe")
VERCEL_CLI_JS = Path(
    r"C:\Users\jackc\AppData\Local\nvm\v24.18.0\node_modules\vercel\dist\vc.js"
)
# The CLI's own config directory (gitignored under tools/), created on use.
VERCEL_GLOBAL_CONFIG = REPO_ROOT / "tools" / "vercel-global-config"

DEPLOY_TIMEOUT_SECONDS = 900
# The read-back polls a 120 s window every 10 s, plus the first attempt: a FIXED attempt count, so
# a fake sleep ends it as surely as a real one.
READBACK_WINDOW_SECONDS = 120
READBACK_INTERVAL_SECONDS = 10
READBACK_ATTEMPTS = READBACK_WINDOW_SECONDS // READBACK_INTERVAL_SECONDS + 1
READBACK_REQUEST_TIMEOUT_SECONDS = 15
# The routes fetched once each after /health matches: every nav page, one fragment and one export.
# Their expected answers come from scripts.fetch_hosted_routes.routes, the live check's own list.
SMOKE_PATHS: tuple[str, ...] = (
    "/",
    "/bets",
    "/season",
    "/track-record",
    "/how-it-works",
    "/fragments/games",
    "/api/export/json",
)
# The daily push waits for a rebuild to finish; the manual push refuses at once (SPEC R11).
DAILY_LOCK_WAIT_SECONDS = 600
MANUAL_LOCK_WAIT_SECONDS = 0
BYPASS_HEADER = "x-vercel-protection-bypass"
#: Hosts the test suite refuses (``tests/conftest.py``), a subdomain included.
VERCEL_HOSTS = frozenset({"vercel.com", "vercel.app"})

SOURCE_DAILY = "daily"
SOURCE_MANUAL = "manual"

OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_FAILED = "failed"
OUTCOME_REFUSED = "refused"
OUTCOME_NOT_PUSHED = "not_pushed"

# The reason vocabulary: bare identifiers, never text that could carry a value.
REASON_PUSH_NOT_ENABLED = "push_not_enabled"
REASON_LOCK_HELD = "lock_held"
REASON_REBUILD_INCOMPLETE = "rebuild_incomplete"
REASON_CACHE_MISSING = "cache_missing"
REASON_FORMAT_MISMATCH_REBUILD_FAILED = "format_mismatch_rebuild_failed"
REASON_FORMAT_MISMATCH_AFTER_REBUILD = "format_mismatch_after_rebuild"
REASON_MISSING_TOKEN = "missing_token"
REASON_MISSING_BYPASS_SECRET = "missing_bypass_secret"
REASON_CLI_MISSING = "cli_missing"
REASON_DEPLOY_TIMEOUT = "deploy_timeout"
REASON_DEPLOY_OUTPUT_UNPARSEABLE = "deploy_output_unparseable"
REASON_READBACK_FAILED = "readback_failed"
REASON_PAGES_FAILED = "pages_failed"

# What one read-back GET saw (also ``http_<status>`` and ``transport_<ExceptionClass>``). A failed
# page check records ``page_<route>_http_<status>``, ``page_<route>_no_marker`` or
# ``page_<route>_transport_<ExceptionClass>`` instead, ``<route>`` being the path with its slashes
# turned into underscores (``root`` for ``/``).
READBACK_MATCHED = "matched"
READBACK_OLD_ID = "old_id"
READBACK_DIGEST_MISMATCH = "digest_mismatch"
READBACK_NO_ID = "no_id"


@dataclass(frozen=True)
class PushSecrets:
    """The two push secrets from ``.env``; None when a value is absent or blank."""

    vercel_token: SecretStr | None
    bypass_secret: SecretStr | None


@dataclass(frozen=True)
class PushOutcome:
    """What one push attempt did -- exactly what its push-log record says.

    Attributes:
        outcome: ``succeeded``, ``failed``, ``refused`` or ``not_pushed``.
        reason: A named reason or an exception class name; None on success.
        publish_id: This attempt's id, once minted.
        deployment_id: The deployment the CLI reported, once deployed.
        deployment_url: Its per-deployment URL.
        local_sha256: The digest of the stamped cache, as uploaded.
        readback: The last read-back observation.
        data_format_found: The cache's data-format label, once read.
        data_format_expected: The data-format version this code expects, once read.
        code_commit: The commit the deployed code came from, once about to deploy.
    """

    outcome: str
    reason: str | None = None
    publish_id: str | None = None
    deployment_id: str | None = None
    deployment_url: str | None = None
    local_sha256: str | None = None
    readback: str | None = None
    data_format_found: str | None = None
    data_format_expected: int | None = None
    code_commit: str | None = None


def load_push_secrets(settings: Settings | None = None) -> PushSecrets:
    """The push secrets from *settings* (default: the project's ``.env`` settings)."""
    if settings is None:
        from conf.settings import get_settings

        settings = get_settings()
    return PushSecrets(
        vercel_token=_present(settings.vercel_token),
        bypass_secret=_present(settings.vercel_bypass_secret),
    )


def _present(value: SecretStr | None) -> SecretStr | None:
    """*value* when it holds something; None for an absent or blank secret."""
    if value is None or not value.get_secret_value().strip():
        return None
    return value


def _push_enabled() -> bool:
    """Whether the switch file is a JSON object whose ``enabled`` is exactly ``true``."""
    try:
        switch = json.loads(PUSH_ENABLED_MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(switch, dict) and switch.get("enabled") is True


def _default_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """The read-back client: a 15 s per-request timeout, and redirects are NEVER followed.

    A protected deployment answers an unauthenticated request with a redirect to Vercel's sign-in;
    following it would read a page that is not the site. *transport* exists for tests only.
    """
    return httpx.Client(
        timeout=READBACK_REQUEST_TIMEOUT_SECONDS,
        follow_redirects=False,
        transport=transport,
    )


# Seams, read at call time: tests replace them so nothing deploys, reaches Vercel or sleeps.
RUN_CLI: Callable[..., subprocess.CompletedProcess] = subprocess.run
CLIENT_FACTORY: Callable[[], httpx.Client] = _default_client
SLEEP: Callable[[float], None] = time.sleep
LOAD_PUSH_SECRETS: Callable[[], PushSecrets] = load_push_secrets


def run_vercel(
    args: Sequence[str], *, token: str, cwd: Path, timeout: float
) -> subprocess.CompletedProcess:
    """Run the one Vercel CLI with *args*; the token travels only in the child's environment.

    Raises:
        subprocess.TimeoutExpired: the CLI ran longer than *timeout*.
        FileNotFoundError: the node binary is not where :data:`NODE_EXE` says.
    """
    VERCEL_GLOBAL_CONFIG.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["VERCEL_TOKEN"] = token
    env["VERCEL_TELEMETRY_DISABLED"] = "1"
    argv = [
        str(NODE_EXE),
        str(VERCEL_CLI_JS),
        "--global-config",
        str(VERCEL_GLOBAL_CONFIG),
        *args,
    ]
    # UTF-8 with replacement: the CLI writes UTF-8, and an undecodable byte must not crash the
    # push before its output is judged.
    return RUN_CLI(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def read_site_pushes(path: Path | None = None) -> list[dict[str, Any]]:
    """Every push-log record, oldest first. An absent log is no records.

    Raises:
        ValueError: a line is not one JSON object. A malformed log is never read as no records.
    """
    log_path = SITE_PUSH_LOG if path is None else path
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(
        log_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as error:
            msg = f"{log_path.as_posix()} line {number} is not valid JSON"
            raise ValueError(msg) from error
        if not isinstance(record, dict):
            msg = f"{log_path.as_posix()} line {number} is not a JSON object"
            raise ValueError(msg)
        records.append(record)
    return records


def _append(record: dict[str, Any]) -> None:
    """Append one push-log record carrying exactly :data:`SITE_PUSH_FIELDS`, through the sink."""
    append_jsonl(
        SITE_PUSH_LOG,
        {name: record.get(name) for name in SITE_PUSH_FIELDS},
        kind="site_push",
    )


def _finish(
    outcome: PushOutcome, *, source: str, run_id: str | None, detail: str | None
) -> None:
    """Write the attempt's ONE record and print its ONE ``SITE_PUSH=`` line.

    *detail* (the rebuild's exception class, D-03) is printed only, never recorded.
    """
    _append(
        {
            "time_utc": datetime.now(tz=UTC).isoformat(),
            "run_id": run_id,
            "source": source,
            "outcome": outcome.outcome,
            "reason": outcome.reason,
            "publish_id": outcome.publish_id,
            "deployment_id": outcome.deployment_id,
            "deployment_url": outcome.deployment_url,
            "local_sha256": outcome.local_sha256,
            "readback": outcome.readback,
            "data_format_found": outcome.data_format_found,
            "data_format_expected": outcome.data_format_expected,
            "code_commit": outcome.code_commit,
        }
    )
    said = (
        outcome.publish_id if outcome.outcome == OUTCOME_SUCCEEDED else outcome.reason
    )
    words = ["SITE_PUSH=", outcome.outcome, said or "unknown"]
    if detail:
        words.append(detail)
    sys.stdout.write(" ".join(words) + "\n")


def record_not_pushed(
    *, source: str, reason: str, run_id: str | None = None
) -> PushOutcome:
    """Record an attempt that did not push, and touch nothing else (no lock, no CLI, no cache)."""
    outcome = PushOutcome(outcome=OUTCOME_NOT_PUSHED, reason=reason)
    _finish(outcome, source=source, run_id=run_id, detail=None)
    return outcome


@dataclass
class _Attempt:
    """What one attempt has established so far; every exit builds its outcome from this."""

    publish_id: str | None = None
    local_sha256: str | None = None
    deployment_id: str | None = None
    deployment_url: str | None = None
    readback: str | None = None
    data_format_found: str | None = None
    data_format_expected: int | None = None
    code_commit: str | None = None
    rebuild_error: str | None = None

    def outcome(self, outcome: str, reason: str | None = None) -> PushOutcome:
        return PushOutcome(
            outcome=outcome,
            reason=reason,
            publish_id=self.publish_id,
            deployment_id=self.deployment_id,
            deployment_url=self.deployment_url,
            local_sha256=self.local_sha256,
            readback=self.readback,
            data_format_found=self.data_format_found,
            data_format_expected=self.data_format_expected,
            code_commit=self.code_commit,
        )

    def note_format(self, state: DataFormatState) -> None:
        self.data_format_found = state.found_label
        self.data_format_expected = state.expected


def _default_rebuild() -> None:
    """The D-03 self-heal: the populate step's rebuild WITHOUT the lock (the push holds it)."""
    from pipeline import steps

    steps.populate_web_cache_unlocked()


def _default_stylesheet_check() -> str | None:
    """The D-17 stylesheet check."""
    from pipeline import stylesheet_check

    return stylesheet_check.check_stylesheet()


def _read_format(cache_path: Path) -> DataFormatState:
    """The cache's data-format verdict; the short-lived connection is closed before returning."""
    from api.cache_reader import open_cache_readonly, read_data_format
    from api.cache_schema import data_format_state

    if not cache_path.exists():
        return data_format_state(None)
    conn = open_cache_readonly(cache_path)
    try:
        return read_data_format(conn)
    finally:
        conn.close()


def _code_commit(repo_root: Path = REPO_ROOT) -> str | None:
    """HEAD's SHA, plus :data:`DIRTY_SUFFIX` when an uploaded path differs from it; None if unknown.

    The deploy uploads what is on disk, so an edit or an untracked file under
    :data:`UPLOADED_CODE_PATHS` ships too: the suffix says the deployed code is in no commit.
    Gitignored files (``__pycache__``) are not changes.
    """
    git = ("git", "-C", str(repo_root))
    try:
        head = subprocess.run(
            [*git, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
        status = subprocess.run(
            [*git, "status", "--porcelain", "--", *UPLOADED_CODE_PATHS],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = head.stdout.strip()
    if head.returncode != 0 or status.returncode != 0 or not sha:
        return None
    return f"{sha}{DIRTY_SUFFIX}" if status.stdout.strip() else sha


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_deployment(stdout: str) -> tuple[str, str] | None:
    """The deployment's id and https URL from the CLI's JSON output; None when it has neither.

    The CLI prints either ``{"status": "ok", "deployment": {...}}`` (non-interactive) or the bare
    deployment object.
    """
    try:
        payload = json.loads(stdout)
    except ValueError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("deployment"), dict):
        payload = payload["deployment"]
    if not isinstance(payload, dict):
        return None
    deployment_id, url = payload.get("id"), payload.get("url")
    if not (isinstance(deployment_id, str) and deployment_id):
        return None
    if not (isinstance(url, str) and url):
        return None
    if "://" not in url:
        url = f"https://{url}"
    return deployment_id, url


def _observe(
    client: httpx.Client, *, bypass: str, publish_id: str, local_sha256: str
) -> str:
    """One read-back GET of the production ``/health``, classified. Never the response text."""
    try:
        response = client.get(
            f"{PRODUCTION_URL}/health", headers={BYPASS_HEADER: bypass}
        )
    except httpx.TransportError as error:
        return f"transport_{type(error).__name__}"
    if response.status_code != 200:
        return f"http_{response.status_code}"
    try:
        body = response.json()
    except ValueError:
        return READBACK_NO_ID
    if not isinstance(body, dict) or not body.get("publish_id"):
        return READBACK_NO_ID
    if body["publish_id"] != publish_id:
        return READBACK_OLD_ID
    if body.get("cache_sha256") != local_sha256:
        return READBACK_DIGEST_MISMATCH
    return READBACK_MATCHED


def _read_back(*, bypass: str, publish_id: str, local_sha256: str) -> str:
    """Up to :data:`READBACK_ATTEMPTS` GETs, stopping at the first match; the last observation."""
    observed = READBACK_NO_ID
    with CLIENT_FACTORY() as client:
        for attempt in range(1, READBACK_ATTEMPTS + 1):
            observed = _observe(
                client, bypass=bypass, publish_id=publish_id, local_sha256=local_sha256
            )
            if observed == READBACK_MATCHED:
                break
            if attempt < READBACK_ATTEMPTS:
                SLEEP(READBACK_INTERVAL_SECONDS)
    return observed


def _check_pages(*, bypass: str) -> str | None:
    """One GET of each :data:`SMOKE_PATHS` route; the first failure's observation, None if all pass.

    Called only after ``/health`` matched, so the new deployment is already serving: no retry.
    """
    from scripts.fetch_hosted_routes import check_route, routes

    smoke = [route for route in routes(game_id="") if route.path in SMOKE_PATHS]
    with CLIENT_FACTORY() as client:
        for route in smoke:
            name = route.path.strip("/").replace("/", "_") or "root"
            try:
                response = client.get(
                    f"{PRODUCTION_URL}{route.path}", headers={BYPASS_HEADER: bypass}
                )
            except httpx.TransportError as error:
                return f"page_{name}_transport_{type(error).__name__}"
            if response.status_code != route.status:
                return f"page_{name}_http_{response.status_code}"
            if not check_route(route, response):
                return f"page_{name}_no_marker"
    return None


def _run_attempt(
    attempt: _Attempt,
    *,
    cache_path: Path,
    lock_wait_seconds: float,
    rebuild_attempted: bool,
    rebuild_completed: bool,
    rebuild: Callable[[], None],
    check_stylesheet: Callable[[], str | None],
    now: datetime | None,
) -> PushOutcome:
    """Everything after the switch check. Computes the outcome; never writes the record."""
    secrets = LOAD_PUSH_SECRETS()
    if secrets.vercel_token is None:
        return attempt.outcome(OUTCOME_FAILED, REASON_MISSING_TOKEN)
    if secrets.bypass_secret is None:
        return attempt.outcome(OUTCOME_FAILED, REASON_MISSING_BYPASS_SECRET)

    with contextlib.ExitStack() as held:
        try:
            held.enter_context(cache_publish_lock(wait_seconds=lock_wait_seconds))
        except FileLockHeldError:
            return attempt.outcome(OUTCOME_REFUSED, REASON_LOCK_HELD)

        # (a) an unfinished rebuild, or no cache at all: never a deploy of pages with no data.
        if rebuild_attempted and not rebuild_completed:
            return attempt.outcome(OUTCOME_REFUSED, REASON_REBUILD_INCOMPLETE)
        if not cache_path.exists():
            return attempt.outcome(OUTCOME_REFUSED, REASON_CACHE_MISSING)

        # (c) the stylesheet.
        stylesheet_problem = check_stylesheet()
        if stylesheet_problem is not None:
            return attempt.outcome(OUTCOME_REFUSED, stylesheet_problem)

        # (b) the data format, healed by one rebuild under this lock (D-03).
        state = _read_format(cache_path)
        attempt.note_format(state)
        if not state.matches:
            try:
                rebuild()
            except Exception as error:  # noqa: BLE001 - a failed rebuild is a named refusal
                attempt.rebuild_error = type(error).__name__
                return attempt.outcome(
                    OUTCOME_REFUSED, REASON_FORMAT_MISMATCH_REBUILD_FAILED
                )
            state = _read_format(cache_path)
            attempt.note_format(state)
            if not state.matches:
                return attempt.outcome(
                    OUTCOME_REFUSED, REASON_FORMAT_MISMATCH_AFTER_REBUILD
                )

        # A fresh publish id and push time, stamped on a copy, then the digest of what uploads.
        from api.cache import stamp_publish

        attempt.publish_id = uuid.uuid4().hex
        stamp_publish(
            cache_path,
            publish_id=attempt.publish_id,
            push_time=now if now is not None else datetime.now(tz=UTC),
        )
        attempt.local_sha256 = _file_sha256(cache_path)
        attempt.code_commit = _code_commit()

        try:
            completed = run_vercel(
                ["deploy", "--prod", "--yes", "--json"],
                token=secrets.vercel_token.get_secret_value(),
                cwd=REPO_ROOT,
                timeout=DEPLOY_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return attempt.outcome(OUTCOME_FAILED, REASON_DEPLOY_TIMEOUT)
        except FileNotFoundError:
            return attempt.outcome(OUTCOME_FAILED, REASON_CLI_MISSING)
        if completed.returncode != 0:
            return attempt.outcome(
                OUTCOME_FAILED, f"deploy_exit_{completed.returncode}"
            )
        deployment = _parse_deployment(completed.stdout or "")
        if deployment is None:
            return attempt.outcome(OUTCOME_FAILED, REASON_DEPLOY_OUTPUT_UNPARSEABLE)
        attempt.deployment_id, attempt.deployment_url = deployment

        attempt.readback = _read_back(
            bypass=secrets.bypass_secret.get_secret_value(),
            publish_id=attempt.publish_id,
            local_sha256=attempt.local_sha256,
        )
        if attempt.readback != READBACK_MATCHED:
            return attempt.outcome(OUTCOME_FAILED, REASON_READBACK_FAILED)
        page_problem = _check_pages(bypass=secrets.bypass_secret.get_secret_value())
        if page_problem is not None:
            attempt.readback = page_problem
            return attempt.outcome(OUTCOME_FAILED, REASON_PAGES_FAILED)
        return attempt.outcome(OUTCOME_SUCCEEDED)


def push_site(
    *,
    source: str,
    run_id: str | None = None,
    rebuild_attempted: bool = False,
    rebuild_completed: bool = False,
    cache_path: Path | None = None,
    lock_wait_seconds: float | None = None,
    rebuild: Callable[[], None] | None = None,
    check_stylesheet: Callable[[], str | None] | None = None,
    now: datetime | None = None,
) -> PushOutcome:
    """Push the site once: refuse, or stamp, deploy and prove it by read-back (see the module doc).

    Args:
        source: ``daily`` or ``manual``.
        run_id: The daily run's id, when there is one.
        rebuild_attempted: This run tried to rebuild the cache.
        rebuild_completed: That rebuild finished.
        cache_path: The cache to push (default :data:`DEFAULT_CACHE_PATH`).
        lock_wait_seconds: How long to wait for the publish lock (default: the source's wait).
        rebuild: The D-03 self-heal rebuild (default ``pipeline.steps.populate_web_cache_unlocked``).
        check_stylesheet: The D-17 check (default ``pipeline.stylesheet_check.check_stylesheet``).
        now: The push time to stamp (default: the current UTC time).

    Returns:
        The attempt's outcome, exactly as recorded in the push log.
    """
    if not _push_enabled():
        return record_not_pushed(
            source=source, reason=REASON_PUSH_NOT_ENABLED, run_id=run_id
        )

    if lock_wait_seconds is None:
        lock_wait_seconds = (
            MANUAL_LOCK_WAIT_SECONDS
            if source == SOURCE_MANUAL
            else DAILY_LOCK_WAIT_SECONDS
        )
    attempt = _Attempt()
    # ONE CATCH-ALL, ONE WRITE POINT (SPEC R5: exactly one record per attempt). Anything after the
    # switch check that raises -- the secrets load, the lock's own OSError, the stylesheet check,
    # the format read, the stamp, the digest, the deploy-output handling, the read-back -- becomes
    # one failed outcome named by its exception CLASS, never its message. The try sits outside the
    # lock's with block (inside _run_attempt), so the lock is already released here. Only
    # Exception is caught: Ctrl+C and the test guard's pytest.fail still propagate. The record
    # write stays OUTSIDE the try, so a write that raises escapes once and can never be retried
    # into a second record; the caller notes that case (Plan 36-07).
    try:
        outcome = _run_attempt(
            attempt,
            cache_path=DEFAULT_CACHE_PATH if cache_path is None else cache_path,
            lock_wait_seconds=lock_wait_seconds,
            rebuild_attempted=rebuild_attempted,
            rebuild_completed=rebuild_completed,
            rebuild=_default_rebuild if rebuild is None else rebuild,
            check_stylesheet=(
                _default_stylesheet_check
                if check_stylesheet is None
                else check_stylesheet
            ),
            now=now,
        )
    except Exception as error:  # noqa: BLE001 - one failed record, never a missing one
        outcome = attempt.outcome(OUTCOME_FAILED, type(error).__name__)
    _finish(outcome, source=source, run_id=run_id, detail=attempt.rebuild_error)
    return outcome
