"""The nflverse release-asset publication stamp -- ONE source, loud on failure (D33.2-16).

Every nflverse data file this repository captures (snap counts, injury reports, and the depth
charts the daily run reads in Plan 33.2-27) is a GitHub release asset of
``nflverse/nflverse-data``, and the releases API reports each asset's ``updated_at``: the
instant upstream last published that file. That is the only PROVABLE time attached to a
captured file, and this module is the only place the repository reads it.

WHAT THE STAMP IS, AND WHAT IT IS NOT (Plan 33.2-15, reviews round ``f924749``)
-----------------------------------------------------------------------------
It is CAPTURE PROVENANCE: a true fact about the file fetched NOW. It is not the time any row
in a season-wide file first became known. So an ingest writes it onto every row it captures as
``upstream_captured_at``, and a builder may treat it as a row's information time only where it
is AT OR BEFORE that row's game's lock -- the forward daily-capture case. A season fetched
after its games carries a stamp after every one of their locks and therefore admits nothing.

THERE IS NO FALLBACK, AND THAT IS THE POINT
-------------------------------------------
When the stamp cannot be read -- a rate limit, a server error, a network failure, an asset
missing from its release, an ``updated_at`` with no time zone -- :func:`asset_published_at`
RAISES :class:`UpstreamStampUnavailable`. It never returns the fetch instant. A fetch instant
is a MANUFACTURED timestamp (RESEARCH pitfall P1): it is always "now", so every row stamped
with it looks freshly captured, and a lock fence that reads it passes by construction. A
refusal costs one run; a plausible fabricated number costs every build that trusts it.

The unauthenticated GitHub API allows 60 requests per hour per IP (RESEARCH Security Domain).
A short in-process TTL cache keeps a daily run to one request per asset. The TTL clock lives
ONLY in :func:`_cache_get` / :func:`_cache_store`, outside the body that produces the returned
stamp, and :func:`_cache_store` refuses anything that is not a tz-aware ``datetime`` -- so the
cache's own clock reading can never be stored, or returned, AS a stamp.
"""

from __future__ import annotations

import re
import time
from datetime import UTC, date, datetime
from typing import Any, Protocol

import requests

#: The release-tag endpoint. ``{tag}`` is the release name, which nflverse uses as the dataset
#: name (``snap_counts``, ``injuries``, ``depth_charts``).
NFLVERSE_RELEASES_API: str = (
    "https://api.github.com/repos/nflverse/nflverse-data/releases/tags/{tag}"
)

#: An nflverse per-season release asset: ``<dataset>_<season>.<extension>``.
_ASSET_NAME = re.compile(r"^(?P<tag>[a-z][a-z_]*?)_(?P<season>\d{4})\.[a-z.]+$")

#: How long a stamp read from the API is reused in-process. Short on purpose: long enough that
#: one daily run reads each asset once, short enough that a later run never reuses a stamp from
#: before an upstream refresh.
STAMP_CACHE_TTL_SECONDS: float = 600.0

_REQUEST_TIMEOUT_SECONDS: float = 30.0

# ---------------------------------------------------------------------------
# MEASURED REFRESH TIMES (RESEARCH section 1, queried 2026-09-15 against the releases API).
#
# The CONTEXT's "the release refreshes once daily ~07:07 UTC" (D33.2-16) is STALE. Measured
# 2026-09-15: the injuries asset refreshed at 12:38:18Z, snap_counts at 11:25:15Z and
# depth_charts at 12:39:20Z, and snap counts refresh FOUR times a day in season (00/06/12/18
# UTC, nflverse data schedule). So the gap from a Wednesday-morning injury refresh to a
# Wednesday 18:00 ET lock is about 9.5 hours, not about 15.
# ---------------------------------------------------------------------------

REFRESH_MEASURED_ON: date = date(2026, 9, 15)

MEASURED_ASSET_REFRESH_UTC: dict[str, str] = {
    "injuries": "12:38:18",
    "snap_counts": "11:25:15",
    "depth_charts": "12:39:20",
}

#: The in-season snap-count refresh hours (UTC), four a day.
SNAP_COUNTS_IN_SEASON_REFRESH_HOURS_UTC: tuple[int, ...] = (0, 6, 12, 18)

#: The CONTEXT figure superseded by the measurement above, kept so the correction is visible.
STALE_CONTEXT_REFRESH_UTC: str = "07:07"

#: THE INJURY-FRESHNESS LIMITATION, stated rather than closed (D33.2-16, Plan 33.2-15's ESPN
#: ruling). The ESPN injuries endpoint is declined as a model input and as an archive: it carries
#: game status and prose, not the official practice-participation report the model is trained
#: on; it is current-season only, so 2009-2024 cannot be rebuilt from it; and it is an
#: undocumented endpoint with no published contract.
INJURY_FRESHNESS_LIMITATION: str = (
    "A Wednesday 18:00 ET lock for a Thursday game sees the nflverse injury file as it stood "
    "at roughly 08:38 ET that morning (its 12:38 UTC refresh, measured 2026-09-15), so it "
    "misses that Wednesday's practice report. No free source with a checkable publication "
    "time carries the same practice-participation fields, so the gap is named, not closed."
)


class UpstreamStampUnavailable(Exception):
    """The upstream publication stamp of an nflverse asset could not be read honestly.

    Inherits ``Exception`` and NOT ``RuntimeError`` / ``ValueError`` / ``ImportError``, the
    same reasoning as ``data.sealed_probe_log.SealedProbeLogCorrupt`` and
    ``data.upstream_pin.UpstreamPinError``: several call sites in this repository catch that
    tuple (``scripts/build_features._SOURCE_LOAD_ERRORS`` among them) and degrade quietly, and
    "the upstream stamp is unavailable" degraded into "use now" is the precise failure this
    module exists to make impossible.
    """


class _HttpSession(Protocol):
    """The slice of ``requests.Session`` the stamp reader uses (a test seam)."""

    def get(self, url: str, **kwargs: Any) -> Any: ...


#: asset name -> (stamp, expires_at on the monotonic clock). Only ``stamp`` is ever returned.
_STAMP_CACHE: dict[str, tuple[datetime, float]] = {}


def clear_stamp_cache() -> None:
    """Forget every cached stamp (tests; a long-lived process that wants a fresh read)."""
    _STAMP_CACHE.clear()


def _cache_get(asset_name: str) -> datetime | None:
    """The cached stamp for *asset_name*, or ``None`` when absent or expired.

    The TTL clock is read HERE, and only here and in :func:`_cache_store`. It decides whether a
    stored stamp is still reused; it never becomes one.
    """
    entry = _STAMP_CACHE.get(asset_name)
    if entry is None:
        return None
    stamp, expires_at = entry
    if time.monotonic() >= expires_at:
        _STAMP_CACHE.pop(asset_name, None)
        return None
    return stamp


def _cache_store(asset_name: str, value: object) -> None:
    """Cache *value* as *asset_name*'s stamp -- ONLY a tz-aware ``datetime`` is accepted.

    Refusing a float, a string, ``None`` or a naive datetime is what keeps a later edit from
    caching the TTL clock's own reading (a float) or any other manufactured value AS a stamp.

    Raises:
        TypeError: *value* is not a tz-aware ``datetime``.
    """
    if not isinstance(value, datetime) or value.tzinfo is None:
        msg = (
            f"refusing to cache {value!r} as the upstream stamp of {asset_name}: only a "
            "tz-aware datetime parsed from the releases API's updated_at is a stamp"
        )
        raise TypeError(msg)
    _STAMP_CACHE[asset_name] = (value, time.monotonic() + STAMP_CACHE_TTL_SECONDS)


def release_tag_for(asset_name: str) -> str:
    """The release tag an nflverse per-season asset lives under (``injuries_2026.parquet`` ->
    ``injuries``).

    Raises:
        UpstreamStampUnavailable: *asset_name* is not an nflverse per-season asset name.
    """
    match = _ASSET_NAME.match(asset_name)
    if match is None:
        msg = (
            f"{asset_name!r} is not an nflverse per-season release asset name "
            "(<dataset>_<season>.<extension>), so no release tag can be derived for it"
        )
        raise UpstreamStampUnavailable(msg)
    return match.group("tag")


def season_asset_name(dataset: str, season: int) -> str:
    """The parquet release asset nflreadpy downloads for *dataset* in *season*."""
    return f"{dataset}_{int(season)}.parquet"


def _rate_limit_reset(headers: Any) -> str:
    """The rate-limit reset instant from the response headers, rendered for a message."""
    raw = headers.get("x-ratelimit-reset") if headers is not None else None
    try:
        return datetime.fromtimestamp(int(raw), tz=UTC).isoformat()
    except (TypeError, ValueError):
        return "unknown (no x-ratelimit-reset header)"


def _parse_updated_at(asset_name: str, raw: object) -> datetime:
    """``updated_at`` parsed as a tz-aware UTC instant; anything else is refused by name."""
    if not isinstance(raw, str) or not raw:
        msg = f"the release entry for {asset_name} carries no updated_at ({raw!r})"
        raise UpstreamStampUnavailable(msg)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        msg = f"the updated_at of {asset_name} is not an ISO-8601 instant: {raw!r}"
        raise UpstreamStampUnavailable(msg) from error
    if parsed.tzinfo is None:
        msg = (
            f"the updated_at of {asset_name} ({raw!r}) carries no time zone; it is refused, "
            "never relabelled as UTC"
        )
        raise UpstreamStampUnavailable(msg)
    return parsed.astimezone(UTC)


def asset_published_at(
    asset_name: str,
    *,
    session: _HttpSession | None = None,
    fresh: bool = False,
) -> datetime:
    """The tz-aware UTC ``updated_at`` of the named nflverse release asset.

    Args:
        asset_name: A per-season asset, e.g. ``injuries_2026.parquet``. Its release tag is
            the dataset prefix.
        session: An object with ``requests.Session.get``'s shape (the test seam). Defaults to
            the ``requests`` module.
        fresh: Bypass the TTL cache and read the API now (the cache is then refreshed with
            the value read). The ingests use this for the read that follows their download.

    Returns:
        The asset's ``updated_at`` -- upstream's own publication instant, never a clock
        reading taken here.

    Raises:
        UpstreamStampUnavailable: on a rate limit (naming the reset instant), any non-200
            response, a network error, an asset missing from its release, or an
            ``updated_at`` that is absent, unparseable or naive. There is deliberately NO
            fallback value: a fetch instant is a manufactured timestamp (RESEARCH P1).
    """
    if not fresh:
        cached = _cache_get(asset_name)
        if cached is not None:
            return cached

    url = NFLVERSE_RELEASES_API.format(tag=release_tag_for(asset_name))
    http = session if session is not None else requests
    try:
        response = http.get(
            url,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            headers={"Accept": "application/vnd.github+json"},
        )
    except requests.RequestException as error:
        msg = f"the releases API could not be reached for {asset_name}: {error}"
        raise UpstreamStampUnavailable(msg) from error

    status = int(response.status_code)
    headers = getattr(response, "headers", None)
    remaining = headers.get("x-ratelimit-remaining") if headers is not None else None
    if status in (403, 429) and remaining == "0":
        msg = (
            f"the GitHub releases API rate limit is exhausted, so the upstream stamp of "
            f"{asset_name} is unavailable; the limit resets at {_rate_limit_reset(headers)}. "
            "Refusing rather than stamping the fetch instant."
        )
        raise UpstreamStampUnavailable(msg)
    if status != 200:
        msg = (
            f"the releases API answered HTTP {status} for {asset_name} ({url}); no stamp "
            "was read and none is manufactured"
        )
        raise UpstreamStampUnavailable(msg)

    try:
        assets = response.json()["assets"]
    except (ValueError, KeyError, TypeError) as error:
        msg = f"the release payload for {asset_name} carries no asset list"
        raise UpstreamStampUnavailable(msg) from error
    entry = next(
        (a for a in assets if isinstance(a, dict) and a.get("name") == asset_name), None
    )
    if entry is None:
        msg = f"{asset_name} is not an asset of its release ({url})"
        raise UpstreamStampUnavailable(msg)

    stamp = _parse_updated_at(asset_name, entry.get("updated_at"))
    _cache_store(asset_name, stamp)
    return stamp
