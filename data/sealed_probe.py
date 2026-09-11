"""The sealed-zone upstream re-release probe (D32-05, Plan 32-05).

WHAT THIS IS
------------
The sealed zone (seasons at or before ``data.upstream_pin.SEALED_THROUGH_SEASON``) is
pinned: the pipeline reads recorded bytes, never upstream. That makes the pipeline
reproducible, and it also makes it BLIND -- if nflverse re-releases a completed season,
nothing in the read path notices, because the read path never looks. This module is the
thing that looks. It compares the upstream release signature of every pinned
``(dataset, season)`` pair against the baseline recorded in
``config/upstream_pin.sealed.lock`` and returns a verdict stamped with the frozen
``data.revision_events`` vocabulary.

It is NEVER blocking (D32-09). A sealed re-release does not make this project's numbers
wrong -- the pinned bytes are still the pinned bytes -- so the verdict is recorded, not
raised. Blocking a run that is demonstrably correct is how a detector earns an override
flag and then gets ignored.

WHY IT LIVES IN ``data/`` BUT IS CALLED FROM ``scripts/``
----------------------------------------------------------
This module is a LIBRARY. ``scripts/capture_live_season.py`` calls it (Plan 32-08);
``data/upstream_pin.py`` never does, and must never. The pin module's entire contract is
that a READ never touches the network, and the moment a probe sits behind a load the
contract is gone. See ``32-RESEARCH.md``'s Architectural Responsibility Map. Every
network-capable import here is DEFERRED into the function that needs it, in the shape of
``scripts/pin_upstream_snapshot.py::fetch_live``'s deferred ``import nflreadpy``, so
importing this module costs nothing and can never reach out on its own.

THE DATASET SPLIT, AND WHY IT IS NOT A JUDGEMENT CALL
------------------------------------------------------
``32-RESEARCH.md`` Finding 1 established that ``nflreadpy`` 0.1.5 downloads
``play_by_play_{YYYY}.parquet`` and ``depth_charts_{YYYY}.parquet`` as PER-SEASON assets
but downloads schedules as ONE monolithic ``games.parquet`` covering 1999-2026. Release
metadata can therefore attribute a change to a season for two datasets and, structurally
and permanently, cannot for the third. :data:`SEALED_PROBE_STRATEGY` encodes exactly that,
with the measurement recorded beside the row.

THE FAILURE MODE THIS MODULE IS BUILT AROUND
---------------------------------------------
A metadata probe can go silently blind, and blind looks exactly like clean. An asset key
that does not resolve -- because nflverse changed the layout, because the rate limit
returned 403, because a season was deleted -- makes ``assets.get(name)`` return ``None``,
and a naive ``if asset and asset.updated_at != baseline`` reads CLEAN for all three.
PITFALLS F2: a dead detector and a healthy system must not be the same observable. So
:func:`probe_sealed` counts ``checked`` against ``expected`` STRUCTURALLY SEPARATELY from
the finding ruling, and ``checked < expected`` is an UNKNOWN naming what it missed --
never a clean.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from typing import Any

from data.revision_events import (
    DEFAULT_SEVERITY,
    VERDICT_SCHEMA_VERSION,
    RevisionEventClass,
    severity_rank,
)

# The digest chunk size ``data.upstream_pin.digest_file`` uses, imported rather than
# re-typed. 32-RESEARCH.md's "Don't Hand-Roll" row is explicit: two hashers means two
# answers, and a content comparison whose two sides were chunked differently would still
# agree -- but only by luck of the implementation, not by contract.
from data.upstream_pin import _CHUNK_BYTES as PIN_DIGEST_CHUNK_BYTES

__all__ = [
    "ASSET_DOWNLOAD_URL",
    "ASSET_NAME_TEMPLATES",
    "DATASET_RELEASE_TAGS",
    "FINDING_KEYS",
    "GITHUB_RELEASES_TAG_URL",
    "MAX_RELEASE_PAYLOAD_BYTES",
    "PROBE_STRATEGY_CONTENT",
    "PROBE_STRATEGY_METADATA",
    "PROBE_TIMEOUT_SECONDS",
    "SEALED_PROBE_STRATEGY",
    "VERDICT_KEYS",
    "SealedProbeUnavailable",
    "acknowledge_divergence",
    "asset_download_url",
    "asset_name_for",
    "content_digest_for",
    "fetch_release_assets",
    "parse_release_payload",
    "probe_sealed",
    "seed_signatures",
]


class SealedProbeUnavailable(Exception):
    """The probe could not see upstream, so it has nothing honest to say.

    Inherits ``Exception`` and NOT ``RuntimeError`` / ``ValueError`` / ``ImportError``, for
    exactly the reason ``data.upstream_pin.UpstreamPinError`` does. Several wired call
    sites in this repository catch that tuple and degrade to an empty frame or a quiet
    skip; a probe failure caught by one of those handlers would be converted from "I could
    not see upstream" into "upstream is fine", which is the single worst outcome available
    to a detector. Raised loudly, caught deliberately, recorded as UNKNOWN.
    """


# ---------------------------------------------------------------------------
# Where the assets live, and what they are called.
# ---------------------------------------------------------------------------

# The nflverse release tag each dataset's assets are published under. Fixed by
# ``nflreadpy``'s own loader paths (32-RESEARCH.md Finding 1), which is also why
# COVERAGE.md marks ``releases.list`` OPT-OUT: enumerating releases would spend rate limit
# to rediscover a constant.
DATASET_RELEASE_TAGS: dict[str, str] = {
    "pbp": "pbp",
    "schedules": "schedules",
    "depth_charts": "depth_charts",
}

# The asset file name within that tag. Keyed exactly like
# ``data.upstream_pin.DATASET_COLUMNS`` so the three dataset names stay aligned across the
# codebase.
#
# NOTE THE ASYMMETRY: ``schedules`` takes NO season. ``nflreadpy.load_schedules``
# (load_schedules.py:27) downloads ``schedules/games`` -- one file for 1999-2026 -- and
# filters in memory. ``load_pbp`` (load_pbp.py:45) and ``load_depth_charts``
# (load_depth_charts.py:47) download one asset per season.
ASSET_NAME_TEMPLATES: dict[str, str] = {
    "pbp": "play_by_play_{season}.parquet",
    "depth_charts": "depth_charts_{season}.parquet",
    "schedules": "games.parquet",
}

PROBE_STRATEGY_METADATA = "metadata"
PROBE_STRATEGY_CONTENT = "content"

# WHICH PROBE EACH DATASET GETS. This is the whole of D32-05, and it is forced by the
# asset layout above rather than chosen.
SEALED_PROBE_STRATEGY: dict[str, str] = {
    # PER-SEASON asset, so a moved ``updated_at`` attributes to exactly one season.
    # MEASURED 2026-09-10 (32-RESEARCH.md Finding 5): of 25 pinned pbp seasons, ZERO had
    # an asset ``updated_at`` postdating the pin's capture. The metadata probe is quiet on
    # a quiet record, which is the property a detector has to have before anyone will read
    # its output.
    "pbp": PROBE_STRATEGY_METADATA,
    # PER-SEASON asset, same shape. MEASURED 2026-09-10: 0 of 24 pinned seasons moved.
    "depth_charts": PROBE_STRATEGY_METADATA,
    # CONTENT, ALWAYS. NOT metadata, and this is not a preference.
    #
    # THE MEASUREMENT (32-RESEARCH.md Finding 5, 2026-09-10, reproducible on demand):
    # nflverse publishes schedules as ONE monolithic ``games.parquet`` covering
    # 1999-2026. A metadata probe over it cannot attribute a change to a season, and
    # because 2026 results land in that same file it moves continuously during the live
    # season. Its ``updated_at`` was 2026-09-11T01:06:14Z against a pin captured
    # 2026-09-05T04:44:22Z, so a metadata probe reported ALL 27 pinned schedules seasons
    # as moved -- while a content check found all 27 seasons' row counts and the frame
    # width IDENTICAL to the pin. 27 false positives, zero true positives. A detector that
    # opens the season by firing 27 CRITICALs on a provably byte-stable record is the
    # alert-fatigue failure delivered on day one.
    #
    # TWO TEMPTING ALTERNATIVES, NAMED AND REJECTED so nobody reaches for them later:
    #
    # 1. The tag-level ``timestamp.json`` (43 bytes, published in every release tag). It
    #    looks like a perfect zero-cost probe. It is not usable: it tracks the most
    #    recently touched asset in the WHOLE tag, which during 2026 is
    #    ``play_by_play_2026.parquet`` -- verified, the pbp tag timestamp
    #    2026-09-10 09:20:49 EDT matches that asset's updated_at 2026-09-10T13:20:51Z. It
    #    would fire a sealed CRITICAL every single week of the live season. Same
    #    pathology as schedules, at tag granularity. COVERAGE.md marks it OPT-OUT.
    # 2. A parquet-footer ranged GET (row counts and schema for 64 KB instead of 20 MB).
    #    Finding 7 proved it works with absolute offsets. Rejected anyway: it buys nothing
    #    against a measured 20 MB / ~0.4 s full fetch, and it adds a thrift footer parser
    #    plus an "is 64 KB enough footer for a 372-column file" edge case to the one
    #    component whose entire value is being trustworthy.
    "schedules": PROBE_STRATEGY_CONTENT,
}

# The GitHub Releases API endpoint the metadata probe reads. One request per tag returns
# EVERY asset with ``name``, ``size`` and ``updated_at`` -- 3 requests, ~412 KB, ~0.8 s
# covers every pinned season of every pinned dataset (32-RESEARCH.md Finding 4).
GITHUB_RELEASES_TAG_URL = (
    "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}"
)

# The asset download URL, which MUST be the URL ``nflreadpy`` itself fetches. Its base is
# ``NflverseDownloader.BASE_URLS["nflverse-data"]`` and its path is the loader's own
# template; ``tests/unit/test_revision_detector.py`` asserts the two agree by building the
# comparison URL through ``nflreadpy``'s own ``_build_url`` rather than re-typing it. A
# probe watching a file nobody reads is worse than no probe.
ASSET_DOWNLOAD_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/{tag}/{asset_name}"
)

# Every outbound call carries this. A probe with no timeout can hang the weekly run it is
# supposed to be a cheap side effect of (T-32-04).
PROBE_TIMEOUT_SECONDS: float = 30.0

# An explicit ceiling on the metadata response, checked BEFORE the JSON is parsed. The
# largest real payload measured is 236 KB (the pbp tag, 164 assets), so 8 MB is ~35x
# headroom and still refuses an unbounded or hostile body rather than handing it to a
# parser.
MAX_RELEASE_PAYLOAD_BYTES: int = 8 * 1024 * 1024

# ---------------------------------------------------------------------------
# The FROZEN verdict shape.
# ---------------------------------------------------------------------------

# D32-08 makes a verdict a COMMITTED record: one line is appended to
# ``config/upstream_probe_log.jsonl`` per run, for a whole season. So the key set is frozen
# here the way ``data/revision_events.py`` freezes the vocabulary, and for the same reason
# -- a season of entries must stay comparable end to end. Adding a key means bumping
# ``data.revision_events.VERDICT_SCHEMA_VERSION``; renaming one makes the already-written
# entries stop meaning what they said.
VERDICT_KEYS: tuple[str, ...] = (
    "verdict_schema_version",
    "probed_at_utc",
    "event_class",
    "severity",
    "expected",
    "checked",
    "unresolved",
    "strategy",
    "findings",
    "reason",
    "rate_limit_remaining",
)

# Every finding carries the SAME keys, with ``None`` where a strategy does not populate
# one. A shape-stable line is a line a reader can diff a season later; a line whose keys
# depend on which branch produced it is not.
FINDING_KEYS: tuple[str, ...] = (
    "dataset",
    "season",
    "strategy",
    "event_class",
    "acknowledged",
    "superseded_acknowledgement",
    "baseline_updated_at",
    "baseline_size",
    "observed_updated_at",
    "observed_size",
    "baseline_sha256",
    "observed_sha256",
    "ruled_by",
    "ruled_at_utc",
)


# ---------------------------------------------------------------------------
# URL construction.
# ---------------------------------------------------------------------------


def _known_datasets_message(dataset: str) -> str:
    return (
        f"Unknown upstream dataset {dataset!r}. Known: {sorted(ASSET_NAME_TEMPLATES)}."
    )


def asset_name_for(dataset: str, season: int) -> str:
    """Return the upstream asset file name for *dataset* / *season*.

    ``schedules`` ignores *season* -- there is one ``games.parquet`` for every year -- and
    that asymmetry is deliberate rather than an oversight; see
    :data:`ASSET_NAME_TEMPLATES`.
    """
    template = ASSET_NAME_TEMPLATES.get(dataset)
    if template is None:
        raise SealedProbeUnavailable(_known_datasets_message(dataset))
    return template.format(season=season)


def asset_download_url(dataset: str, season: int) -> str:
    """Return the URL ``nflreadpy`` downloads for *dataset* / *season*.

    Asserted equal to ``nflreadpy``'s own constructed URL by
    ``tests/unit/test_revision_detector.py``. If upstream ever changes its layout the test
    fails rather than this module silently watching a file the loader stopped reading.
    """
    tag = DATASET_RELEASE_TAGS.get(dataset)
    if tag is None:
        raise SealedProbeUnavailable(_known_datasets_message(dataset))
    return ASSET_DOWNLOAD_URL.format(
        tag=tag, asset_name=asset_name_for(dataset, season)
    )


# ---------------------------------------------------------------------------
# The untrusted-JSON validator.
# ---------------------------------------------------------------------------


def _require_tz_aware_instant(value: Any, asset_name: str) -> str:
    """Return *value* unchanged once it parses as a timezone-aware ISO-8601 instant."""
    if not isinstance(value, str) or not value.strip():
        msg = (
            f"release asset {asset_name!r}: field 'updated_at' is {value!r}, which is not "
            "a non-empty string. The GitHub releases payload is untrusted third-party "
            "JSON and every field is validated before any of it steers a sealed decision."
        )
        raise SealedProbeUnavailable(msg)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        msg = (
            f"release asset {asset_name!r}: field 'updated_at' is {value!r}, which does "
            f"not parse as ISO-8601 ({exc})."
        )
        raise SealedProbeUnavailable(msg) from exc
    if parsed.tzinfo is None:
        msg = (
            f"release asset {asset_name!r}: field 'updated_at' is {value!r}, which parses "
            "but carries NO TIMEZONE. A naive instant compared against a tz-aware "
            "baseline is a comparison whose answer depends on the machine that ran it."
        )
        raise SealedProbeUnavailable(msg)
    return value


def parse_release_payload(payload: Any) -> dict[str, dict]:
    """Validate a GitHub releases payload and return ``{asset name: signature}``.

    THIS IS THE TRUST BOUNDARY. It is the point at which untrusted third-party JSON
    crosses into a sealed-zone decision, and it is the only place in this phase where an
    upstream-supplied string reaches this process at all.

    THE EXPLICIT CONTROL ON THAT STRING: an asset ``name`` is used ONLY as a dictionary
    key for comparison against a name this module CONSTRUCTED in :func:`asset_name_for`.
    It is never joined into a filesystem path, never passed to ``open``, and never used to
    derive a local file name. Every local path in Phase 32 is built from the
    ``(dataset, season, week, sequence)`` key the project owns (T-32-03).

    WHAT IS DELIBERATELY NOT CARRIED THROUGH:

    * ``created_at`` -- within 0-2 s of ``updated_at`` on 28 of 28 pbp assets, because
      GitHub DELETES and RE-UPLOADS release assets rather than updating them in place. It
      is not a publication anchor, so carrying it would imply a fact it cannot support.
    * Never store the ETag. It is a re-encoded ``Last-Modified``: decoding five sampled
      values reproduced their own ``Last-Modified`` header to the exact second
      (32-RESEARCH.md Finding 3). It carries zero information the timestamp does not, a
      byte-identical re-upload mints a new one, and its equality is therefore NOT a
      content claim. COVERAGE.md marks it OPT-OUT.

    Args:
        payload: The decoded JSON body of ``GET /repos/.../releases/tags/{tag}``.

    Returns:
        ``{asset_name: {"size": int, "updated_at": str}}`` and nothing else.

    Raises:
        SealedProbeUnavailable: If the payload is not a mapping, if ``assets`` is absent
            or is not a list, or if any entry is not a mapping or lacks a usable ``name``,
            integer ``size`` or timezone-aware ISO-8601 ``updated_at``. Every one of these
            makes the run record UNKNOWN -- never clean.
    """
    if not isinstance(payload, dict):
        msg = (
            f"the release payload is a {type(payload).__name__}, not a mapping. A "
            "malformed or truncated response must produce UNKNOWN, never clean."
        )
        raise SealedProbeUnavailable(msg)

    if "assets" not in payload:
        msg = (
            "the release payload has no 'assets' key. Either the API shape changed or the "
            "response was truncated; both are UNKNOWN, never clean."
        )
        raise SealedProbeUnavailable(msg)

    assets = payload["assets"]
    if not isinstance(assets, list):
        msg = (
            f"the release payload's 'assets' field is a {type(assets).__name__}, not a "
            "list."
        )
        raise SealedProbeUnavailable(msg)

    signatures: dict[str, dict] = {}
    for index, entry in enumerate(assets):
        if not isinstance(entry, dict):
            msg = (
                f"release asset at index {index} is a {type(entry).__name__}, not a "
                "mapping."
            )
            raise SealedProbeUnavailable(msg)

        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            msg = (
                f"release asset at index {index}: field 'name' is {name!r}, which is not "
                "a non-empty string. The name is the comparison key, so an unusable one "
                "makes the whole payload unusable."
            )
            raise SealedProbeUnavailable(msg)

        size = entry.get("size")
        # ``bool`` is an ``int`` subclass in Python; a boolean size is a malformed payload,
        # not a small file.
        if not isinstance(size, int) or isinstance(size, bool):
            msg = (
                f"release asset {name!r}: field 'size' is {size!r} "
                f"({type(size).__name__}), which is not an int."
            )
            raise SealedProbeUnavailable(msg)

        updated_at = _require_tz_aware_instant(entry.get("updated_at"), name)
        signatures[name] = {"size": size, "updated_at": updated_at}

    return signatures


# ---------------------------------------------------------------------------
# The two network calls. Both defer their imports.
# ---------------------------------------------------------------------------


def _rate_limit_remaining(headers: Any) -> int | None:
    """Read ``x-ratelimit-remaining`` off a response, or ``None`` when it is unreadable."""
    try:
        raw = headers.get("x-ratelimit-remaining")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def fetch_release_assets(
    tag: str,
    *,
    session: Any = None,
    timeout: float = PROBE_TIMEOUT_SECONDS,
    rate_limits: dict[str, int | None] | None = None,
) -> dict[str, dict]:
    """Fetch one release tag's asset signatures from the GitHub API.

    ``import requests`` is DEFERRED into this function, in the shape of
    ``scripts/pin_upstream_snapshot.py::fetch_live``'s deferred ``import nflreadpy``, so
    importing :mod:`data.sealed_probe` never costs a network-capable import.

    The request is UNAUTHENTICATED by decision (COVERAGE.md ->
    ``auth.authenticated-requests`` OPT-OUT): three requests per run against a 60/hr
    ceiling is ample headroom, and a credential in a committed probe log is a leak with no
    capability gain. No credential header is ever constructed. TLS verification is left at
    the ``requests`` default and is never disabled (T-32-07).

    Args:
        tag: The release tag, e.g. ``"pbp"``. One of :data:`DATASET_RELEASE_TAGS`' values.
        session: An optional ``requests.Session``. Supplied by the caller so three tags
            share one connection; ``None`` uses the module-level ``requests.get``.
        timeout: Seconds, applied to the whole request.
        rate_limits: An optional mutable mapping. When given, ``rate_limits[tag]`` is set
            to the observed ``x-ratelimit-remaining`` (or ``None`` when unreadable). This
            is how the rate limit reaches :func:`probe_sealed`'s verdict without this
            function returning a second value or keeping module state -- a 403 from the
            unauthenticated ceiling must be legible as a rate limit rather than a mystery.

    Returns:
        The validated ``{asset_name: signature}`` mapping from
        :func:`parse_release_payload`.

    Raises:
        SealedProbeUnavailable: On any transport failure, any non-200 status, a response
            body above :data:`MAX_RELEASE_PAYLOAD_BYTES`, a body that is not JSON, or any
            payload :func:`parse_release_payload` refuses.
    """
    import requests

    url = GITHUB_RELEASES_TAG_URL.format(tag=tag)
    getter = session.get if session is not None else requests.get
    try:
        response = getter(url, timeout=timeout)
    except Exception as exc:
        msg = (
            f"could not reach the GitHub releases API for tag {tag!r} at {url}: "
            f"{type(exc).__name__}: {exc}. The probe could not see upstream, so the run "
            "records UNKNOWN rather than clean."
        )
        raise SealedProbeUnavailable(msg) from exc

    remaining = _rate_limit_remaining(getattr(response, "headers", None))
    if rate_limits is not None:
        rate_limits[tag] = remaining

    status = getattr(response, "status_code", None)
    if status != 200:
        limit_text = (
            f" x-ratelimit-remaining={remaining}"
            if remaining is not None
            else " (no x-ratelimit-remaining header)"
        )
        hint = ""
        if status == 403 and remaining == 0:
            hint = (
                " This is the unauthenticated 60/hr per-IP ceiling, not a permissions "
                "problem."
            )
        msg = (
            f"the GitHub releases API returned status {status!r} for tag {tag!r} at "
            f"{url}.{limit_text}{hint}"
        )
        raise SealedProbeUnavailable(msg)

    body = getattr(response, "content", b"")
    if not isinstance(body, bytes | bytearray):
        msg = (
            f"the GitHub releases API response for tag {tag!r} carried a "
            f"{type(body).__name__} body rather than bytes."
        )
        raise SealedProbeUnavailable(msg)
    if len(body) > MAX_RELEASE_PAYLOAD_BYTES:
        msg = (
            f"the GitHub releases API response for tag {tag!r} is {len(body)} bytes, "
            f"above the {MAX_RELEASE_PAYLOAD_BYTES}-byte ceiling. The ceiling is checked "
            "BEFORE parsing so an unbounded response cannot exhaust the weekly run."
        )
        raise SealedProbeUnavailable(msg)

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = (
            f"the GitHub releases API response for tag {tag!r} is not decodable JSON "
            f"({exc})."
        )
        raise SealedProbeUnavailable(msg) from exc

    return parse_release_payload(payload)


def content_digest_for(
    dataset: str,
    season: int,
    *,
    session: Any = None,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> str:
    """Stream one upstream asset and return the sha256 of its bytes AS PUBLISHED.

    This is the CONTENT half. It is what the metadata strategy escalates to on a hit, and
    it is what the ``schedules`` strategy uses instead of metadata.

    WHAT IT DIGESTS, STATED PRECISELY BECAUSE IT MATTERS: the raw upstream asset stream,
    exactly as nflverse published it. That is NOT the same artifact the sealed lock's
    ``sha256`` was computed over -- this project narrows play-by-play to
    ``PBP_PINNED_COLUMNS`` and filters schedules out of the monolithic ``games.parquet``
    before writing its own parquet. The two digests are different bytes by construction.
    See :func:`probe_sealed`'s ``content_digests`` argument for which basis the ruling
    needs; mixing them would report a move on every pair forever.

    The redirect is followed by ``requests`` but the redirect TARGET is never inspected:
    ``release-assets.githubusercontent.com`` issues an EXPIRING signed query string that
    differs per request (32-RESEARCH.md Finding 2, Pitfall 2), so comparing
    ``response.url`` would report a change every single time.

    Raises:
        SealedProbeUnavailable: On any transport failure or any non-200 status.
    """
    import requests

    url = asset_download_url(dataset, season)
    getter = session.get if session is not None else requests.get
    try:
        response = getter(url, stream=True, timeout=timeout)
    except Exception as exc:
        msg = (
            f"could not stream the {dataset} asset for season {season} at {url}: "
            f"{type(exc).__name__}: {exc}."
        )
        raise SealedProbeUnavailable(msg) from exc

    status = getattr(response, "status_code", None)
    if status != 200:
        msg = (
            f"streaming the {dataset} asset for season {season} at {url} returned status "
            f"{status!r}."
        )
        raise SealedProbeUnavailable(msg)

    digest = hashlib.sha256()
    try:
        for chunk in response.iter_content(PIN_DIGEST_CHUNK_BYTES):
            if chunk:
                digest.update(chunk)
    except Exception as exc:
        msg = (
            f"the {dataset} asset stream for season {season} at {url} failed mid-read: "
            f"{type(exc).__name__}: {exc}."
        )
        raise SealedProbeUnavailable(msg) from exc
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The ruling. Pure: it takes already-fetched inputs so every test runs offline.
# ---------------------------------------------------------------------------


def _empty_finding() -> dict[str, Any]:
    return dict.fromkeys(FINDING_KEYS)


def _lock_pairs(lock: dict | None) -> list[tuple[str, int, dict]]:
    """Every ``(dataset, season, entry)`` the lock records, in a stable order."""
    pairs: list[tuple[str, int, dict]] = []
    for dataset, seasons in sorted((lock or {}).get("datasets", {}).items()):
        for season, entry in sorted(seasons.items(), key=lambda item: int(item[0])):
            pairs.append((dataset, int(season), entry))
    return pairs


def _probed_at(now: datetime | None) -> str:
    if now is None:
        from datetime import UTC

        return datetime.now(UTC).isoformat()
    if not isinstance(now, datetime):
        msg = f"probe_sealed's `now` must be a datetime, got {type(now).__name__}."
        raise ValueError(msg)
    if now.tzinfo is None:
        msg = (
            "probe_sealed's `now` must be timezone-aware. A naive stamp on a committed "
            "record means a different instant on every machine that reads it."
        )
        raise ValueError(msg)
    return now.isoformat()


def probe_sealed(
    lock: dict | None,
    *,
    assets_by_tag: dict[str, dict[str, dict] | None],
    content_digests: dict[tuple[str, int], str] | None = None,
    now: datetime | None = None,
    rate_limit_remaining: int | None = None,
) -> dict[str, Any]:
    """Rule on every pinned ``(dataset, season)`` pair and return one frozen verdict.

    PURE. Every input is already fetched, so the whole ruling is exercised offline by
    ``tests/unit/test_revision_detector.py`` against stored payloads.

    THE COVERAGE ASSERTION IS STRUCTURALLY SEPARATE FROM THE FINDING RULING, and that
    separation is the point. ``checked`` counts pairs for which a signature was ACTUALLY
    resolved. ``checked < expected`` is UNKNOWN naming the unresolved pairs -- never
    clean, no matter what the resolved pairs said. An absent asset key, a layout change
    from per-season to monolithic (nflverse has already done exactly that for schedules),
    a rate-limit 403 and a deleted season all present as a missing key, and a naive
    ``if asset and asset.updated_at != baseline`` reads clean for all four.

    A lock entry whose ``upstream_updated_at`` is still ``null`` -- not yet seeded by
    :func:`seed_signatures` -- counts as UNRESOLVED, not as clean. A baseline nobody
    recorded cannot be matched, and calling that agreement would make an unseeded lock the
    quietest possible detector.

    ``size`` is recorded beside ``updated_at`` in every finding but is NOT ruled on
    separately: a moved ``updated_at`` with an unchanged ``size`` is weak evidence of a
    cosmetic rebuild, useful for calibration, not a verdict. The ruling comes from the
    escalated content re-fetch.

    Args:
        lock: The sealed lock document (``data.upstream_pin.load_sealed_lock``). ``None``
            or empty yields ``expected == 0``, which :func:`_rule_run` rules UNKNOWN
            before it rules anything else (CR-01) -- never clean.
        assets_by_tag: ``{release tag: {asset name: signature}}`` as returned by
            :func:`fetch_release_assets`. A tag mapped to ``None`` means that tag could
            not be fetched, and every pair under it is UNRESOLVED with that reason.
        content_digests: ``{(dataset, season): sha256}`` for content-strategy pairs.
            THE BASIS MATTERS: these must be digests of the SAME artifact the lock's
            ``sha256`` was computed over -- the pinned per-season frame as this project
            writes it -- and NOT the raw upstream asset stream
            (:func:`content_digest_for`), which is different bytes for every dataset here.
            This function is pure and cannot check which basis it was handed, so the
            contract is stated here and honoured by the caller (Plan 32-08).
        now: An injected timezone-aware stamp. ``None`` uses the current UTC instant.
        rate_limit_remaining: The observed ``x-ratelimit-remaining``, recorded verbatim on
            the verdict so a run that went blind on the rate limit says so.

    Returns:
        A mapping whose keys are exactly :data:`VERDICT_KEYS`.
    """
    content_digests = content_digests or {}
    probed_at_utc = _probed_at(now)

    pairs = _lock_pairs(lock)
    expected = len(pairs)
    checked = 0
    unresolved: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    strategy: dict[str, str] = {}

    for dataset, season, entry in pairs:
        dataset_strategy = SEALED_PROBE_STRATEGY.get(dataset)
        if dataset_strategy is None:
            unresolved.append(
                {
                    "dataset": dataset,
                    "season": season,
                    "why": _known_datasets_message(dataset),
                }
            )
            continue
        strategy[dataset] = dataset_strategy
        acknowledgement = entry.get("acknowledgement") or None

        if dataset_strategy == PROBE_STRATEGY_METADATA:
            resolved = _rule_metadata_pair(
                dataset=dataset,
                season=season,
                entry=entry,
                assets_by_tag=assets_by_tag,
                acknowledgement=acknowledgement,
            )
        else:
            resolved = _rule_content_pair(
                dataset=dataset,
                season=season,
                entry=entry,
                content_digests=content_digests,
                acknowledgement=acknowledgement,
            )

        why, finding = resolved
        if why is not None:
            unresolved.append({"dataset": dataset, "season": season, "why": why})
            continue
        checked += 1
        if finding is not None:
            findings.append(finding)

    event_class, reason = _rule_run(
        expected=expected,
        checked=checked,
        unresolved=unresolved,
        findings=findings,
    )

    return {
        "verdict_schema_version": VERDICT_SCHEMA_VERSION,
        "probed_at_utc": probed_at_utc,
        "event_class": str(event_class),
        "severity": str(DEFAULT_SEVERITY[event_class]),
        "expected": expected,
        "checked": checked,
        "unresolved": unresolved,
        "strategy": strategy,
        "findings": findings,
        "reason": reason,
        "rate_limit_remaining": rate_limit_remaining,
    }


def _rule_metadata_pair(
    *,
    dataset: str,
    season: int,
    entry: dict,
    assets_by_tag: dict[str, dict[str, dict] | None],
    acknowledgement: dict | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve and rule ONE metadata-strategy pair. Returns ``(why_unresolved, finding)``."""
    baseline_updated_at = entry.get("upstream_updated_at")
    baseline_size = entry.get("upstream_size")
    if baseline_updated_at is None:
        why = (
            "no upstream baseline recorded: the sealed lock's 'upstream_updated_at' for "
            f"{dataset} {season} is null. A baseline nobody recorded cannot be matched, "
            "so this pair is UNRESOLVED rather than agreeing by default. Seed it with "
            "seed_signatures (Plan 32-09)."
        )
        return why, None

    tag = DATASET_RELEASE_TAGS[dataset]
    assets = assets_by_tag.get(tag)
    if assets is None:
        why = (
            f"no release payload for tag {tag!r}, so the {dataset} asset for season "
            f"{season} could not be resolved at all."
        )
        return why, None

    name = asset_name_for(dataset, season)
    observed = assets.get(name)
    if observed is None:
        why = (
            f"asset {name!r} is absent from the {tag!r} release payload. That is a layout "
            "change, a deletion or a truncated response -- all three are things this "
            "probe did NOT look at, never a clean result."
        )
        return why, None

    observed_updated_at = observed.get("updated_at")
    observed_size = observed.get("size")
    if observed_updated_at == baseline_updated_at and observed_size == baseline_size:
        return None, None

    finding = _empty_finding()
    finding.update(
        {
            "dataset": dataset,
            "season": season,
            "strategy": PROBE_STRATEGY_METADATA,
            "baseline_updated_at": baseline_updated_at,
            "baseline_size": baseline_size,
            "observed_updated_at": observed_updated_at,
            "observed_size": observed_size,
        }
    )
    _apply_acknowledgement(
        finding,
        acknowledgement=acknowledgement,
        matches=(
            acknowledgement is not None
            and acknowledgement.get("observed_updated_at") == observed_updated_at
            and acknowledgement.get("observed_size") == observed_size
        ),
    )
    return None, finding


def _rule_content_pair(
    *,
    dataset: str,
    season: int,
    entry: dict,
    content_digests: dict[tuple[str, int], str],
    acknowledgement: dict | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve and rule ONE content-strategy pair. Returns ``(why_unresolved, finding)``."""
    baseline_sha256 = entry.get("sha256")
    if not baseline_sha256:
        why = (
            f"no content baseline recorded: the sealed lock's 'sha256' for {dataset} "
            f"{season} is empty."
        )
        return why, None

    observed_sha256 = content_digests.get((dataset, season))
    if observed_sha256 is None:
        why = (
            f"no content digest was supplied for {dataset} {season}. The content strategy "
            "rules on bytes, so an absent digest means the probe looked at nothing -- "
            "UNRESOLVED, never clean."
        )
        return why, None

    if observed_sha256 == baseline_sha256:
        return None, None

    finding = _empty_finding()
    finding.update(
        {
            "dataset": dataset,
            "season": season,
            "strategy": PROBE_STRATEGY_CONTENT,
            "baseline_sha256": baseline_sha256,
            "observed_sha256": observed_sha256,
        }
    )
    _apply_acknowledgement(
        finding,
        acknowledgement=acknowledgement,
        matches=(
            acknowledgement is not None
            and acknowledgement.get("observed_sha256") == observed_sha256
        ),
    )
    return None, finding


def _apply_acknowledgement(
    finding: dict[str, Any],
    *,
    acknowledgement: dict | None,
    matches: bool,
) -> None:
    """Stamp *finding* with its acknowledgement state. See :func:`acknowledge_divergence`."""
    if matches and acknowledgement is not None:
        finding["event_class"] = str(RevisionEventClass.KNOWN_DIVERGENCE_STABLE)
        finding["acknowledged"] = True
        finding["superseded_acknowledgement"] = False
        finding["ruled_by"] = acknowledgement.get("ruled_by")
        finding["ruled_at_utc"] = acknowledgement.get("ruled_at_utc")
        return
    finding["event_class"] = str(RevisionEventClass.SEALED_REVISION)
    finding["acknowledged"] = False
    finding["superseded_acknowledgement"] = acknowledgement is not None
    if acknowledgement is not None:
        finding["ruled_by"] = acknowledgement.get("ruled_by")
        finding["ruled_at_utc"] = acknowledgement.get("ruled_at_utc")


def _rule_run(
    *,
    expected: int,
    checked: int,
    unresolved: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> tuple[RevisionEventClass, str]:
    """Choose the run's event class and write its reason.

    The coverage assertion is evaluated FIRST and unconditionally. Everything after it is
    a statement about pairs the probe genuinely looked at.

    ZERO COVERAGE IS RULED BEFORE ANYTHING ELSE (CR-01). ``expected == 0`` is the LIMIT
    CASE of the argument :func:`_rule_metadata_pair` already makes one level down -- "a
    baseline nobody recorded cannot be matched" -- and it was the one case left un-ruled.
    An absent lock at the configured path makes ``data.upstream_pin.load_sealed_lock``
    return ``None`` BY DESIGN, so a mistyped ``--sealed-lock`` used to yield
    ``checked 0 of 0`` with ``checked < expected`` False, no findings, and a CLEAN ruling
    appended to the committed ``config/upstream_probe_log.jsonl``. A season of those lines
    is indistinguishable from a season of real clean ones, which destroys the single
    property D32-08's log exists to provide. It is PITFALLS F2 exactly: a detector that
    could not look must never read the same as a healthy system.
    """
    if expected == 0:
        reason = (
            "the sealed lock records ZERO (dataset, season) pairs, so the probe had no "
            "baseline to compare anything against and checked nothing. Recorded as "
            "UNKNOWN and NEVER clean: an absent or empty lock at the configured path is "
            "a detector that could not look, and a dead detector must not read the same "
            "as a healthy system. Either --sealed-lock names a path that does not exist, "
            "or the lock carries no 'datasets'. Generate one with `python -m "
            "scripts.pin_upstream_snapshot --refresh-sealed-lock "
            '--sealed-rewrite-reason "<why>"`.'
        )
        return RevisionEventClass.UNKNOWN, reason

    if checked < expected:
        named = ", ".join(
            f"{item['dataset']} {item['season']}" for item in unresolved[:10]
        )
        more = "" if len(unresolved) <= 10 else f" (and {len(unresolved) - 10} more)"
        first_why = unresolved[0]["why"] if unresolved else "unknown"
        reason = (
            f"checked {checked} of {expected} pinned pairs. UNRESOLVED: {named}{more}. "
            f"First reason: {first_why} A probe that could not resolve every pair it "
            "claims to have checked reports UNKNOWN, because a dead detector and a "
            "healthy system must not be the same observable."
        )
        return RevisionEventClass.UNKNOWN, reason

    if not findings:
        reason = (
            f"checked {checked} of {expected} pinned pairs; every one resolved and "
            "matched its recorded baseline."
        )
        return RevisionEventClass.CLEAN, reason

    # The run's class is the HIGHEST-severity contribution present, ranked through
    # ``data.revision_events.severity_rank`` rather than by a hand-written comparison. The
    # ladder is the vocabulary's to own; a local ``if critical else warning`` here would
    # be a second, silently divergent copy of it.
    event_class = RevisionEventClass.CLEAN
    best_rank = -1
    for finding in findings:
        contribution = RevisionEventClass(finding["event_class"])
        rank = severity_rank(DEFAULT_SEVERITY[contribution])
        if rank > best_rank:
            event_class, best_rank = contribution, rank

    named = ", ".join(
        f"{finding['dataset']} {finding['season']}" for finding in findings[:10]
    )
    more = "" if len(findings) <= 10 else f" (and {len(findings) - 10} more)"
    acknowledged = sum(1 for finding in findings if finding["acknowledged"])
    reason = (
        f"checked {checked} of {expected} pinned pairs; {len(findings)} moved "
        f"({acknowledged} already acknowledged). Affected: {named}{more}."
    )
    return event_class, reason


# ---------------------------------------------------------------------------
# Writing onto the lock. NEITHER of these ever touches config/upstream_pin.json.
# ---------------------------------------------------------------------------


def _require_attributed(**fields: str) -> None:
    """Refuse any blank field, naming every field in the group and which one is blank."""
    blank = [name for name, value in fields.items() if not str(value or "").strip()]
    if not blank:
        return
    msg = (
        f"blank {', '.join(blank)}. This call requires all of "
        f"{', '.join(sorted(fields))} to be non-empty after .strip(): an unattributed, "
        "unexplained write onto the sealed lock is indistinguishable from suppression, "
        "and six months later nobody can tell which it was."
    )
    raise ValueError(msg)


def _lock_entry(lock: dict, dataset: str, season: int) -> dict:
    seasons = lock.get("datasets", {}).get(dataset)
    if not seasons or str(season) not in seasons:
        msg = (
            f"the sealed lock has no entry for {dataset} {season}. An acknowledgement "
            "records a ruling about a pair the lock already tracks; it never creates one."
        )
        raise ValueError(msg)
    return seasons[str(season)]


def acknowledge_divergence(
    lock: dict,
    *,
    dataset: str,
    season: int,
    observed_updated_at: str | None,
    observed_size: int | None,
    observed_sha256: str | None,
    ruled_by: str,
    ruled_at_utc: str,
    reason: str,
) -> dict:
    """Record an owner ruling on an observed upstream divergence (D32-10).

    THIS IS EXPLICITLY NOT A RE-FREEZE. ``config/upstream_pin.json`` is never touched, and
    the pin is never re-captured to make a probe pass. Re-freezing would erase the very
    evidence the probe exists to produce -- the record that upstream MOVED and this
    project chose not to follow. It is the same shape this project already used for the
    gate-baseline divergence in Phase 31: record the finding, refuse to re-freeze, keep the
    tripwires RED. A reviewer should recognise the shape rather than meet a new invention.

    THE ACKNOWLEDGEMENT CARRIES NO EXPIRY. It stands until a new move supersedes it.
    This is the planner's ruling on CONTEXT's open discretion item, which asked
    whether an acknowledgement carries an expiry
    or stands until a new move supersedes it (CONTEXT's own expectation: stands until
    superseded). The reason is mechanical: an expiry would re-fire a CRITICAL on a
    CALENDAR DATE rather than on an OBSERVATION. Alert-fatigue-by-schedule is precisely
    the failure this mechanism exists to avoid -- 32-RESEARCH.md Finding 6 measured four
    distinct re-release events touching completed seasons in the last nine months,
    including the binding 2021-2023 window, so a detector that also fired on a timer would
    be wallpaper by week three.

    A SECOND ACKNOWLEDGEMENT ON THE SAME PAIR MOVES THE FIRST INTO ``supersedes``, so the
    history is append-shaped rather than overwritten. The ruling that was in force when an
    old verdict was written stays readable.

    Args:
        lock: The sealed lock document. It is NOT mutated; a deep copy is returned.
        dataset: The dataset name. Must already be in the lock.
        season: The season. Must already be in the lock.
        observed_updated_at: The upstream ``updated_at`` observed at the moment of the
            ruling (metadata strategy), or ``None``.
        observed_size: The upstream ``size`` observed at the moment of the ruling, or
            ``None``.
        observed_sha256: The content digest observed at the moment of the ruling (content
            strategy), or ``None``.
        ruled_by: WHO ruled. Must be non-empty after ``.strip()``.
        ruled_at_utc: WHEN they ruled, as an ISO-8601 instant.
        reason: WHY. Must be non-empty after ``.strip()``.

    Returns:
        A NEW lock document carrying the acknowledgement on that one entry.

    Raises:
        ValueError: If ``ruled_by`` or ``reason`` is blank, or the pair is not in the lock.
    """
    _require_attributed(ruled_by=ruled_by, reason=reason)

    updated = copy.deepcopy(lock)
    entry = _lock_entry(updated, dataset, season)
    entry["acknowledgement"] = {
        "observed_updated_at": observed_updated_at,
        "observed_size": observed_size,
        "observed_sha256": observed_sha256,
        "ruled_by": ruled_by,
        "ruled_at_utc": ruled_at_utc,
        "reason": reason,
        "supersedes": entry.get("acknowledgement") or None,
    }
    return updated


def seed_signatures(
    lock: dict,
    *,
    assets_by_tag: dict[str, dict[str, dict] | None],
    content_digests: dict[tuple[str, int], str] | None = None,
    seeded_at_utc: str,
    seeded_by: str,
    overwrite: bool = False,
) -> dict:
    """Write the one-time upstream metadata baseline onto the sealed lock.

    PURE in the same sense :func:`probe_sealed` is: it takes already-fetched inputs, so
    every test runs offline. It lives beside :func:`acknowledge_divergence` because both
    write onto the LOCK and neither ever touches ``config/upstream_pin.json``.

    It fills ``upstream_updated_at`` and ``upstream_size`` for every METADATA-strategy
    pair. It leaves ``sha256`` untouched for every CONTENT-strategy pair: that field is
    already the pinned per-season digest and is the content strategy's baseline as it
    stands. When ``content_digests`` supplies a digest for a content-strategy pair it is
    used as a CHECK -- a mismatch refuses the whole seed, because stamping "seeded" on a
    document that already contains an unruled divergence would date-stamp a lie.

    It REFUSES to overwrite a pair whose ``upstream_updated_at`` is already non-null unless
    ``overwrite=True`` is passed explicitly. Re-seeding a baseline silently would erase the
    very divergence a probe exists to find: the next run would compare today's upstream
    against today's upstream and report clean forever.

    It leaves every existing ``acknowledgement`` block untouched. An acknowledgement is a
    ruling about a divergence FROM a baseline, so a seeding pass has no business editing
    one.

    Args:
        lock: The sealed lock document. It is NOT mutated; a deep copy is returned.
        assets_by_tag: ``{release tag: {asset name: signature}}``.
        content_digests: Optional ``{(dataset, season): sha256}`` on the same basis as the
            lock's ``sha256`` (see :func:`probe_sealed`). Pairs absent from the mapping are
            simply not checked.
        seeded_at_utc: The ISO-8601 instant of this seeding run.
        seeded_by: WHO ran it. Must be non-empty after ``.strip()``.
        overwrite: Explicit permission to replace an already-recorded baseline.

    Returns:
        A NEW lock document with ``signatures_seeded_at_utc`` and ``signatures_seeded_by``
        stamped at the top level, plus ``signatures_unseeded`` naming any pair whose asset
        did not resolve -- present only when that list is non-empty, so a clean run carries
        exactly the two keys and a messy one cannot hide behind them.

    Raises:
        ValueError: If ``seeded_by`` or ``seeded_at_utc`` is blank, or a non-null baseline
            would be replaced without ``overwrite=True``.
        SealedProbeUnavailable: If a supplied content digest disagrees with the lock's
            recorded ``sha256`` for that pair.
    """
    _require_attributed(seeded_by=seeded_by, seeded_at_utc=seeded_at_utc)
    content_digests = content_digests or {}

    updated = copy.deepcopy(lock)
    unseeded: list[dict[str, Any]] = []

    for dataset, season, entry in _lock_pairs(updated):
        dataset_strategy = SEALED_PROBE_STRATEGY.get(dataset)
        if dataset_strategy is None:
            unseeded.append(
                {
                    "dataset": dataset,
                    "season": season,
                    "why": _known_datasets_message(dataset),
                }
            )
            continue

        if dataset_strategy == PROBE_STRATEGY_CONTENT:
            supplied = content_digests.get((dataset, season))
            if supplied is not None and supplied != entry.get("sha256"):
                msg = (
                    f"refusing to seed: the content digest supplied for {dataset} "
                    f"{season} ({supplied}) does not match the sealed lock's recorded "
                    f"sha256 ({entry.get('sha256')}). Seeding stamps a baseline as "
                    "agreed; doing that over an unruled divergence would date-stamp a "
                    "lie. Rule on the divergence with acknowledge_divergence first."
                )
                raise SealedProbeUnavailable(msg)
            continue

        if entry.get("upstream_updated_at") is not None and not overwrite:
            msg = (
                f"refusing to re-seed {dataset} {season}: its upstream baseline is "
                f"already recorded as {entry.get('upstream_updated_at')!r}. Re-seeding "
                "silently would compare today's upstream against today's upstream and "
                "report clean forever, erasing the divergence the probe exists to find. "
                "Pass overwrite=True with an attributed seeded_by if that is genuinely "
                "what you mean."
            )
            raise ValueError(msg)

        tag = DATASET_RELEASE_TAGS[dataset]
        assets = assets_by_tag.get(tag)
        observed = (
            None if assets is None else assets.get(asset_name_for(dataset, season))
        )
        if observed is None:
            unseeded.append(
                {
                    "dataset": dataset,
                    "season": season,
                    "why": (
                        f"asset {asset_name_for(dataset, season)!r} did not resolve in "
                        f"the {tag!r} payload, so no baseline was recorded for it. It "
                        "stays null, which probe_sealed reports as UNRESOLVED."
                    ),
                }
            )
            continue

        entry["upstream_updated_at"] = observed.get("updated_at")
        entry["upstream_size"] = observed.get("size")

    updated["signatures_seeded_at_utc"] = seeded_at_utc
    updated["signatures_seeded_by"] = seeded_by
    if unseeded:
        updated["signatures_unseeded"] = unseeded
    return updated
