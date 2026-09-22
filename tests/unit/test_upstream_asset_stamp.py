"""The nflverse release-asset publication stamp: one source, loud on failure (Plan 33.2-15 Task 1).

``data.upstream_asset_stamp.asset_published_at`` returns the ``updated_at`` of a named nflverse
release asset, read from the GitHub releases API. It is the ONLY place this repository learns
when an upstream file was published (D33.2-16).

What is asserted, and why each matters:

* a 200 response returns the asset's own ``updated_at``, parsed tz-aware;
* a rate-limited response (403 with the rate-limit headers exhausted) RAISES
  ``UpstreamStampUnavailable`` naming the asset and the reset time -- it never returns a value;
* any other failure (a 500, a network error, a missing asset, a naive ``updated_at``) raises the
  same exception. There is no path that returns the fetch instant, because a fetch instant is a
  manufactured timestamp and a fence reading one passes by construction (RESEARCH P1);
* the exception inherits ``Exception`` and none of the catch-tuple bases the build degrades on;
* a second call inside the TTL reuses the cached stamp without a second request, a ``fresh``
  read bypasses the cache, and the cache refuses to store anything that is not a tz-aware
  ``datetime`` -- so the TTL clock can never be cached AS a stamp;
* ``asset_published_at``'s own parsed body contains no clock call at all.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from datetime import UTC, datetime

import data.upstream_asset_stamp as stamp_module
import pytest
import requests
from data.upstream_asset_stamp import (
    NFLVERSE_RELEASES_API,
    UpstreamStampUnavailable,
    asset_published_at,
)

ASSET = "injuries_2026.parquet"
UPDATED_AT = "2026-09-21T13:55:51Z"


class _FakeResponse:
    """The slice of ``requests.Response`` the stamp reader touches."""

    def __init__(
        self, status_code: int, payload: object = None, headers: dict | None = None
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.ok = 200 <= status_code < 300

    def json(self) -> object:
        return self._payload


class _FakeSession:
    """Counts requests and answers each with the next canned response."""

    def __init__(self, *responses: _FakeResponse | Exception) -> None:
        self._responses = list(responses)
        self.urls: list[str] = []

    def get(self, url: str, **_kwargs) -> _FakeResponse:
        self.urls.append(url)
        answer = self._responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _release(updated_at: str = UPDATED_AT, name: str = ASSET) -> dict:
    return {
        "tag_name": "injuries",
        "assets": [
            {"name": "injuries_2025.parquet", "updated_at": "2026-09-07T12:23:41Z"},
            {"name": name, "updated_at": updated_at},
        ],
    }


@pytest.fixture(autouse=True)
def _empty_cache():
    """Every test starts and ends with an empty in-process stamp cache."""
    stamp_module.clear_stamp_cache()
    yield
    stamp_module.clear_stamp_cache()


class TestAStampIsReturned:
    def test_a_200_returns_the_assets_own_updated_at_tz_aware(self) -> None:
        session = _FakeSession(_FakeResponse(200, _release()))
        stamp = asset_published_at(ASSET, session=session)
        assert stamp == datetime(2026, 9, 21, 13, 55, 51, tzinfo=UTC)
        assert stamp.tzinfo is not None
        assert stamp.utcoffset().total_seconds() == 0

    def test_the_request_goes_to_the_release_tag_named_by_the_asset(self) -> None:
        session = _FakeSession(_FakeResponse(200, _release()))
        asset_published_at(ASSET, session=session)
        assert session.urls == [NFLVERSE_RELEASES_API.format(tag="injuries")]

    def test_a_snap_counts_asset_reads_the_snap_counts_tag(self) -> None:
        release = _release(name="snap_counts_2025.parquet")
        session = _FakeSession(_FakeResponse(200, release))
        asset_published_at("snap_counts_2025.parquet", session=session)
        assert session.urls == [NFLVERSE_RELEASES_API.format(tag="snap_counts")]


class TestAFailureIsARefusalNeverANumber:
    def test_a_rate_limited_403_raises_naming_the_asset_and_the_reset(self) -> None:
        headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1790058843"}
        session = _FakeSession(_FakeResponse(403, {"message": "rate limit"}, headers))
        with pytest.raises(UpstreamStampUnavailable) as caught:
            asset_published_at(ASSET, session=session)
        message = str(caught.value)
        assert ASSET in message
        assert "rate limit" in message.lower()
        assert "2026-09-22" in message  # the reset instant, rendered

    def test_a_500_raises(self) -> None:
        session = _FakeSession(_FakeResponse(500, {"message": "boom"}))
        with pytest.raises(UpstreamStampUnavailable, match=ASSET):
            asset_published_at(ASSET, session=session)

    def test_a_network_error_raises(self) -> None:
        session = _FakeSession(requests.ConnectionError("unreachable"))
        with pytest.raises(UpstreamStampUnavailable, match=ASSET):
            asset_published_at(ASSET, session=session)

    def test_an_asset_missing_from_the_release_raises(self) -> None:
        session = _FakeSession(
            _FakeResponse(200, _release(name="injuries_2031.parquet"))
        )
        with pytest.raises(UpstreamStampUnavailable, match=ASSET):
            asset_published_at(ASSET, session=session)

    def test_a_naive_updated_at_raises(self) -> None:
        session = _FakeSession(_FakeResponse(200, _release("2026-09-21T13:55:51")))
        with pytest.raises(UpstreamStampUnavailable, match="time zone"):
            asset_published_at(ASSET, session=session)

    def test_an_unreadable_asset_name_raises(self) -> None:
        with pytest.raises(UpstreamStampUnavailable, match="not an nflverse"):
            asset_published_at("injuries.parquet", session=_FakeSession())

    def test_a_failed_read_is_not_cached(self) -> None:
        session = _FakeSession(
            _FakeResponse(500, {"message": "boom"}), _FakeResponse(200, _release())
        )
        with pytest.raises(UpstreamStampUnavailable):
            asset_published_at(ASSET, session=session)
        assert asset_published_at(ASSET, session=session) == datetime(
            2026, 9, 21, 13, 55, 51, tzinfo=UTC
        )


class TestTheExceptionDodgesTheDegradingCatchTuples:
    def test_it_is_a_direct_subclass_of_exception(self) -> None:
        assert UpstreamStampUnavailable.__bases__ == (Exception,)

    def test_it_is_none_of_the_bases_the_build_swallows(self) -> None:
        for base in (RuntimeError, ValueError, ImportError, OSError, LookupError):
            assert not issubclass(UpstreamStampUnavailable, base), base


class TestTheTtlCache:
    def test_a_second_call_inside_the_ttl_makes_no_second_request(self) -> None:
        session = _FakeSession(_FakeResponse(200, _release()))
        first = asset_published_at(ASSET, session=session)
        second = asset_published_at(ASSET, session=session)
        assert first == second
        assert len(session.urls) == 1

    def test_a_fresh_read_bypasses_the_cache_and_refreshes_it(self) -> None:
        later = "2026-09-22T12:38:18Z"
        session = _FakeSession(
            _FakeResponse(200, _release()), _FakeResponse(200, _release(later))
        )
        asset_published_at(ASSET, session=session)
        refreshed = asset_published_at(ASSET, session=session, fresh=True)
        assert refreshed == datetime(2026, 9, 22, 12, 38, 18, tzinfo=UTC)
        assert len(session.urls) == 2
        # ...and the refreshed value is what the cache now serves.
        assert asset_published_at(ASSET, session=session) == refreshed
        assert len(session.urls) == 2

    def test_an_expired_entry_is_fetched_again(self, monkeypatch) -> None:
        session = _FakeSession(
            _FakeResponse(200, _release()), _FakeResponse(200, _release())
        )
        asset_published_at(ASSET, session=session)
        monkeypatch.setattr(stamp_module, "STAMP_CACHE_TTL_SECONDS", -1.0)
        asset_published_at(ASSET, session=session)
        assert len(session.urls) == 2

    @pytest.mark.parametrize(
        "value",
        [
            datetime(2026, 9, 21, 13, 55, 51),  # naive
            1790058843.0,  # a float -- the TTL clock's own shape
            "2026-09-21T13:55:51Z",  # a string
            None,
        ],
    )
    def test_the_cache_refuses_anything_but_an_aware_datetime(self, value) -> None:
        with pytest.raises(TypeError):
            stamp_module._cache_store(ASSET, value)
        assert stamp_module._cache_get(ASSET) is None


class TestTheReturnPathReadsNoClock:
    CLOCKS = ("now", "utcnow", "today", "time", "monotonic", "perf_counter")

    @classmethod
    def _is_clock(cls, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (isinstance(func, ast.Attribute) and func.attr in cls.CLOCKS) or (
            isinstance(func, ast.Name) and func.id in cls.CLOCKS
        )

    def _clock_lines(self, source: str) -> list[int]:
        tree = ast.parse(textwrap.dedent(source))
        return sorted(node.lineno for node in ast.walk(tree) if self._is_clock(node))

    def test_asset_published_at_calls_no_clock(self) -> None:
        source = inspect.getsource(stamp_module.asset_published_at)
        tree = ast.parse(textwrap.dedent(source))
        assert sum(1 for _ in ast.walk(tree)) > 20  # non-vacuity: a real body was read
        assert self._clock_lines(source) == []

    def test_the_scan_catches_a_planted_clock_in_either_call_shape(self) -> None:
        planted_attribute = "def f():\n    return datetime.now(UTC)\n"
        planted_bare = "def f():\n    return monotonic()\n"
        assert self._clock_lines(planted_attribute) == [2]
        assert self._clock_lines(planted_bare) == [2]

    def test_the_ttl_clock_lives_in_the_cache_helpers(self) -> None:
        helpers = inspect.getsource(stamp_module._cache_get) + inspect.getsource(
            stamp_module._cache_store
        )
        assert self._clock_lines(helpers) != []


# ---------------------------------------------------------------------------
# The two silver schemas that CARRY the stamp (Plan 33.2-15 Task 1, adjudication #2).
# ---------------------------------------------------------------------------

#: Every column scripts/ingest_snaps.py keeps (SNAP_KEY_COLUMNS + RAW_SNAP_COLUMNS), plus the stamp.
SNAP_KEPT = (
    "game_id", "pfr_player_id", "player", "position", "team", "opponent", "season",
    "week", "offense_snaps", "offense_pct", "defense_snaps", "defense_pct", "st_snaps",
    "st_pct", "upstream_captured_at",
)  # fmt: skip

#: Every column scripts/ingest_injuries.py keeps (INJURY_COLUMNS + game_id), plus the stamp.
INJURY_KEPT = (
    "gsis_id", "team", "position", "full_name", "season", "week", "game_type",
    "report_status", "report_primary_injury", "report_secondary_injury", "practice_status",
    "practice_primary_injury", "practice_secondary_injury", "date_modified", "game_id",
    "upstream_captured_at",
)  # fmt: skip

CAPTURED = datetime(2026, 9, 21, 11, 3, 43, tzinfo=UTC)


def _snap_row(**overrides) -> dict:
    row = {
        "game_id": "2026_01_ARI_LAC",
        "pfr_player_id": "StraCo01",
        "player": "Cole Strange",
        "position": "G",
        "team": "LAC",
        "opponent": "ARI",
        "season": 2026,
        "week": 1,
        "offense_snaps": 55.0,
        "offense_pct": 1.0,
        "defense_snaps": 0.0,
        "defense_pct": 0.0,
        "st_snaps": 2.0,
        "st_pct": 0.08,
        "upstream_captured_at": CAPTURED,
    }
    row.update(overrides)
    return row


def _injury_row(**overrides) -> dict:
    row = {
        "gsis_id": "00-0036355",
        "team": "LAC",
        "position": "QB",
        "full_name": "Justin Herbert",
        "season": 2026,
        "week": 2,
        "game_type": "REG",
        "report_status": "Questionable",
        "report_primary_injury": "Ankle",
        "report_secondary_injury": None,
        "practice_status": "Limited Participation in Practice",
        "practice_primary_injury": "Ankle",
        "practice_secondary_injury": None,
        "date_modified": None,
        "game_id": "2026_W02_LAC@LV",
        "upstream_captured_at": CAPTURED,
    }
    row.update(overrides)
    return row


class TestTheSilverSchemasCarryTheStamp:
    def test_the_snap_schema_declares_every_kept_column(self) -> None:
        from data.schemas import SnapCountSchema

        assert set(SNAP_KEPT) <= set(SnapCountSchema.model_fields)

    def test_the_injury_schema_declares_every_kept_column(self) -> None:
        from data.schemas import InjurySchema

        assert set(INJURY_KEPT) <= set(InjurySchema.model_fields)

    def test_the_stamp_is_required_on_both(self) -> None:
        from data.schemas import InjurySchema, SnapCountSchema

        assert SnapCountSchema.model_fields["upstream_captured_at"].is_required()
        assert InjurySchema.model_fields["upstream_captured_at"].is_required()

    def test_a_row_without_the_stamp_is_refused(self) -> None:
        from pydantic import ValidationError

        from data.schemas import InjurySchema, SnapCountSchema

        snap = _snap_row()
        del snap["upstream_captured_at"]
        injury = _injury_row()
        del injury["upstream_captured_at"]
        with pytest.raises(ValidationError):
            SnapCountSchema(**snap)
        with pytest.raises(ValidationError):
            InjurySchema(**injury)

    @pytest.mark.parametrize("missing", [None, float("nan")])
    def test_a_null_stamp_is_refused(self, missing) -> None:
        from pydantic import ValidationError

        from data.schemas import InjurySchema, SnapCountSchema

        with pytest.raises(ValidationError):
            SnapCountSchema(**_snap_row(upstream_captured_at=missing))
        with pytest.raises(ValidationError):
            InjurySchema(**_injury_row(upstream_captured_at=missing))

    def test_a_naive_stamp_is_refused_never_relabelled(self) -> None:
        from pydantic import ValidationError

        from data.schemas import InjurySchema, SnapCountSchema

        naive = datetime(2026, 9, 21, 11, 3, 43)
        with pytest.raises(ValidationError, match="time zone"):
            SnapCountSchema(**_snap_row(upstream_captured_at=naive))
        with pytest.raises(ValidationError, match="time zone"):
            InjurySchema(**_injury_row(upstream_captured_at=naive))

    def test_the_injury_date_modified_is_nullable_and_nat_reads_as_none(self) -> None:
        import pandas as pd

        from data.schemas import InjurySchema

        assert InjurySchema(**_injury_row(date_modified=pd.NaT)).date_modified is None
        assert InjurySchema(**_injury_row(date_modified=None)).date_modified is None

    def test_a_naive_date_modified_is_refused(self) -> None:
        from pydantic import ValidationError

        from data.schemas import InjurySchema

        with pytest.raises(ValidationError, match="time zone"):
            InjurySchema(**_injury_row(date_modified=datetime(2024, 9, 4, 12, 55)))

    def test_a_validated_row_keeps_every_value(self) -> None:
        from data.schemas import InjurySchema, SnapCountSchema

        snap = SnapCountSchema(**_snap_row()).model_dump()
        injury = InjurySchema(**_injury_row()).model_dump()
        assert {k: snap[k] for k in SNAP_KEPT} == _snap_row()
        assert {k: injury[k] for k in INJURY_KEPT} == _injury_row()

    def test_nan_optional_values_read_as_none(self) -> None:
        from data.schemas import InjurySchema, SnapCountSchema

        snap = SnapCountSchema(**_snap_row(offense_pct=float("nan"), position=None))
        injury = InjurySchema(**_injury_row(report_status=float("nan")))
        assert snap.offense_pct is None
        assert snap.position is None
        assert injury.report_status is None
