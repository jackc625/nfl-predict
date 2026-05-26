"""Tests for DataService TTL caching (OPS-06).

Covers:
- Repeated calls return identical data (cache hit behavior)
- Different parameters get different cache keys
- Deep-copy prevents cache poisoning from caller mutation
- ``clear_cache()`` forces a fresh query
- Non-cached methods (``get_game_detail``) are NOT cached
- ``_annotate_wp_correct`` does NOT mutate the cached source list
  (end-to-end regression test for review item #4)
- Phase 17 betting accessors (``get_betting_kpis`` / ``get_betting_roi_table``)
  decode the chart_cache JSON blob and fall back to ``{}`` / ``[]`` on a
  missing, malformed, or wrong-type entry.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
from fastapi.testclient import TestClient

from api.cache import CACHE_SCHEMA
from api.services import DataService, _cache, clear_cache


def test_cache_returns_same_result_within_ttl(test_db: Path) -> None:
    """Two calls with the same parameters hit the cache on the second call."""
    clear_cache()
    conn = duckdb.connect(str(test_db), read_only=True)
    try:
        svc = DataService(conn=conn)
        first = svc.get_predictions(season=2024)
        second = svc.get_predictions(season=2024)
        assert first == second
        # Cache should now contain the predictions key
        assert ("predictions", 2024, None, "time") in _cache
    finally:
        conn.close()


def test_cache_key_separates_params(test_db: Path) -> None:
    """Different season parameters produce different cache entries."""
    clear_cache()
    conn = duckdb.connect(str(test_db), read_only=True)
    try:
        svc = DataService(conn=conn)
        svc.get_predictions(season=2024)
        svc.get_predictions(season=2023)
        assert ("predictions", 2024, None, "time") in _cache
        assert ("predictions", 2023, None, "time") in _cache
    finally:
        conn.close()


def test_cache_deep_copy_prevents_caller_mutation(test_db: Path) -> None:
    """Caller mutations on the returned list do not corrupt cached data."""
    clear_cache()
    conn = duckdb.connect(str(test_db), read_only=True)
    try:
        svc = DataService(conn=conn)
        first = svc.get_predictions(season=2024)
        assert first, "fixture should have 2024 games"

        # Mutate the returned data in ways a buggy route handler might.
        first.append({"evil": True})
        first[0]["wp_correct"] = "injected"

        # Second call must return pristine data.
        second = svc.get_predictions(season=2024)
        assert {"evil": True} not in second
        # Pristine rows from the fixture never have 'wp_correct' set.
        assert second[0].get("wp_correct") != "injected"
    finally:
        conn.close()


def test_clear_cache_forces_fresh_query(test_db: Path) -> None:
    """clear_cache() evicts cached entries so the next call re-queries."""
    clear_cache()
    conn = duckdb.connect(str(test_db), read_only=True)
    try:
        svc = DataService(conn=conn)
        svc.get_predictions(season=2024)
        assert len(_cache) >= 1
        clear_cache()
        assert len(_cache) == 0
        svc.get_predictions(season=2024)
        assert len(_cache) >= 1
    finally:
        conn.close()


def test_game_detail_is_not_cached(test_db: Path) -> None:
    """get_game_detail intentionally bypasses the cache.

    Unique-per-game_id lookups have ~0 cache hit rate, so caching them
    would just waste memory.
    """
    clear_cache()
    conn = duckdb.connect(str(test_db), read_only=True)
    try:
        svc = DataService(conn=conn)
        svc.get_game_detail("2024_W01_BUF@KC")
        # No cache entry for game_detail
        for key in _cache:
            assert key[0] != "game_detail"
    finally:
        conn.close()


def test_annotate_wp_correct_does_not_mutate_cached_source(
    test_client: TestClient,
) -> None:
    """Hitting a page that calls _annotate_wp_correct must not mutate cached data.

    Regression test for review item #4: performance_page calls
    _annotate_wp_correct on service.get_predictions(...). If the annotation
    mutates in place, subsequent cache lookups would see the injected
    wp_correct values -- making the cache behave nondeterministically.
    """
    clear_cache()
    response1 = test_client.get("/")
    assert response1.status_code == 200

    # Snapshot whatever predictions are cached after the first request.
    cached_before = None
    for key, value in list(_cache.items()):
        if key[0] == "predictions":
            cached_before = [dict(g) for g in value]  # copy for comparison
            break
    assert cached_before is not None, "expected predictions to be cached"
    # The pristine cached source must NOT contain wp_correct -- that field
    # is only added by the route-layer annotation, which must happen on a
    # copy, not on the cached list.
    for game in cached_before:
        assert "wp_correct" not in game, (
            "cached source contains wp_correct -- _annotate_wp_correct "
            "is mutating the cached list"
        )

    # Second request -- the route must not have mutated the cached copy.
    response2 = test_client.get("/")
    assert response2.status_code == 200

    # Re-read cached predictions; they must equal the snapshot.
    cached_after = None
    for key, value in list(_cache.items()):
        if key[0] == "predictions":
            cached_after = [dict(g) for g in value]
            break
    assert cached_after == cached_before, (
        "cached predictions mutated between requests -- _annotate_wp_correct "
        "leaked into the cached source list"
    )


# ---------------------------------------------------------------------------
# Phase 17 betting accessor fallback (get_betting_kpis / get_betting_roi_table)
# ---------------------------------------------------------------------------
# These accessors read a JSON blob from chart_cache via
# get_chart_html(f"betting_kpis_{scope}") / f"betting_roi_table_{scope}" and
# must return {} / [] when the entry is missing, undecodable, or decodes to the
# wrong container type. The `test_db` fixture seeds decodable dict/list payloads
# for the "all"/"recommended" scopes (conftest BETTING_CHART_IDS markers).


def _writable_db_with_chart_rows(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    """Build a fresh writable cache DB seeding chart_cache (chart_id, html_div).

    Used to inject malformed / wrong-type JSON blobs the read-only `test_db`
    fixture cannot express. Only the schema + the supplied chart rows are
    created; the betting accessors only read chart_cache.
    """
    db_path = tmp_path / "betting_accessor_cache.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        for statement in CACHE_SCHEMA.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)
        now = datetime.now(tz=UTC)
        conn.executemany(
            "INSERT INTO chart_cache VALUES (?, ?, ?)",
            [(chart_id, html, now) for chart_id, html in rows],
        )
    finally:
        conn.close()
    return db_path


def test_betting_accessor_happy_path_decodes_dict_and_list(test_db: Path) -> None:
    """Against the seeded fixture, kpis -> non-empty dict, roi_table -> list.

    The conftest markers store a decodable dict under ``betting_kpis_all`` and a
    decodable list-of-dicts under ``betting_roi_table_all``; the accessors must
    json.loads them into the right container types.
    """
    clear_cache()
    conn = duckdb.connect(str(test_db), read_only=True)
    try:
        svc = DataService(conn=conn)

        kpis = svc.get_betting_kpis("all")
        assert isinstance(kpis, dict)
        assert kpis, "fixture betting_kpis_all should decode to a non-empty dict"
        # The fixture KPI payload carries the 7-card scoreboard keys.
        assert "total_bets" in kpis
        assert "win_rate" in kpis

        roi_table = svc.get_betting_roi_table("all")
        assert isinstance(roi_table, list)
        assert roi_table, "fixture betting_roi_table_all should decode to a list"
        assert isinstance(roi_table[0], dict)
        # Production rows carry the WR-04 contract keys (label, not bare slice).
        assert "label" in roi_table[0]

        # The "recommended" scope is seeded too and must also decode.
        assert isinstance(svc.get_betting_kpis("recommended"), dict)
        assert svc.get_betting_kpis("recommended")
        assert isinstance(svc.get_betting_roi_table("recommended"), list)
        assert svc.get_betting_roi_table("recommended")
    finally:
        conn.close()


def test_betting_accessor_missing_scope_falls_back_to_empty(test_db: Path) -> None:
    """An unknown scope has no chart_cache entry -> {} for kpis, [] for roi."""
    clear_cache()
    conn = duckdb.connect(str(test_db), read_only=True)
    try:
        svc = DataService(conn=conn)

        assert svc.get_betting_kpis("bogus_scope") == {}
        assert svc.get_betting_roi_table("bogus_scope") == []
    finally:
        conn.close()


def test_betting_accessor_malformed_json_falls_back_to_empty(
    tmp_path: Path,
) -> None:
    """Undecodable JSON in the chart_cache blob -> {} / [] (JSONDecodeError)."""
    clear_cache()
    db_path = _writable_db_with_chart_rows(
        tmp_path,
        [
            ("betting_kpis_recommended", "{not json"),
            ("betting_roi_table_recommended", "NOT JSON ["),
        ],
    )
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        svc = DataService(conn=conn)

        # get_chart_html returns the raw (non-empty) string, so this exercises
        # the defensive json.loads branch, not the empty-string early return.
        assert svc.get_betting_kpis("recommended") == {}
        assert svc.get_betting_roi_table("recommended") == []
    finally:
        conn.close()


def test_betting_accessor_wrong_type_json_falls_back_to_empty(
    tmp_path: Path,
) -> None:
    """Valid JSON of the wrong container type -> {} / [] (isinstance guard).

    A JSON *list* stored under the kpis id must not leak through as a list (the
    accessor's ``isinstance(decoded, dict)`` guard returns {}); symmetrically a
    JSON *dict* under the roi_table id must return [].
    """
    clear_cache()
    db_path = _writable_db_with_chart_rows(
        tmp_path,
        [
            ("betting_kpis_recommended", json.dumps([1, 2, 3])),  # list, not dict
            ("betting_roi_table_recommended", json.dumps({"a": 1})),  # dict, not list
        ],
    )
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        svc = DataService(conn=conn)

        assert svc.get_betting_kpis("recommended") == {}
        assert svc.get_betting_roi_table("recommended") == []
    finally:
        conn.close()
