"""History-preservation + idempotency coverage for the team-form upsert (WR-06).

`scripts/build_team_form.py::TeamFormBuilder._upsert_current_week_form` is the NEW
persisted silver write introduced on the current-week path (F-01). It must behave as a
target-keyed latest-wins upsert against silver ``team_form_features``:

  (a) the ``(target_season, target_week)`` rows are REPLACED with the freshly-built ones,
  (b) all OTHER seasons / weeks survive (the WR-01 data-loss invariant the code comment
      is most worried about -- a full-history ``replace_mode`` write would shrink the
      on-disk season span), and
  (c) a repeat run does NOT grow the row count (idempotent; team-form has no ``game_id``
      key so the default append-merge would duplicate).

The contract is verified against ``data/storage.save_dataframe(replace_mode=True)``
(`data/storage.py:821-879`): replace_mode makes the passed DataFrame the whole table.
Storage is redirected to an in-memory fake here so no real ``data/silver`` is touched.
"""

from __future__ import annotations

import pandas as pd

from scripts.build_team_form import TeamFormBuilder


class _FakeSilverStore:
    """In-memory stand-in for the silver ``team_form_features`` table.

    Models exactly the two storage calls the upsert makes:
    - ``load_dataframe(table, layer)`` returns the current table (or raises
      FileNotFoundError when nothing has been written, like a missing parquet).
    - ``save_dataframe(df, table, layer, replace_mode=True)`` overwrites the table
      with ``df`` (replace_mode contract: the passed frame IS the whole table).
    """

    def __init__(self) -> None:
        self._table: pd.DataFrame | None = None

    def load_dataframe(self, table_name: str, layer: str = "silver") -> pd.DataFrame:
        assert table_name == "team_form_features"
        assert layer == "silver"
        if self._table is None:
            raise FileNotFoundError("team_form_features not yet written")
        return self._table.copy()

    def save_dataframe(self, df, table_name, layer="silver", **kwargs):
        assert table_name == "team_form_features"
        assert layer == "silver"
        # The upsert MUST use replace_mode=True (full preserved table is the write);
        # default append_mode would grow the table because team-form has no game_id key.
        assert kwargs.get("replace_mode") is True, (
            "upsert must call save_dataframe with replace_mode=True"
        )
        self._table = df.copy()


def _seed_multi_season_history() -> pd.DataFrame:
    """A multi-season team_form_features fixture spanning 2022, 2023, and 2024 W1-W2."""
    rows = []
    for season, week in [
        (2022, 1),
        (2022, 2),
        (2023, 1),
        (2023, 2),
        (2024, 1),
        (2024, 2),
    ]:
        for team in ("BUF", "MIA"):
            rows.append(
                {
                    "target_season": season,
                    "target_week": week,
                    "team": team,
                    "side": "offense",
                    "rolling_epa_per_play": 0.05 + week * 0.001,
                }
            )
    return pd.DataFrame(rows)


def _fresh_target_rows(season: int, week: int) -> pd.DataFrame:
    """Freshly-built rolling rows for one (target_season, target_week)."""
    return pd.DataFrame(
        [
            {
                "target_season": season,
                "target_week": week,
                "team": team,
                "side": "offense",
                "rolling_epa_per_play": 0.999,  # distinct sentinel value
            }
            for team in ("BUF", "MIA")
        ]
    )


class TestUpsertCurrentWeekForm:
    """_upsert_current_week_form preserves history and is idempotent (WR-06)."""

    def _make_builder_with_store(
        self, monkeypatch, store: _FakeSilverStore
    ) -> TeamFormBuilder:
        # Redirect the module-level storage seams used by the upsert. Constructing
        # TeamFormBuilder() touches get_settings() / TeamFormCalculator() only; no
        # storage is read at init, so a real builder is safe to instantiate.
        monkeypatch.setattr(
            "scripts.build_team_form.load_dataframe", store.load_dataframe
        )
        monkeypatch.setattr(
            "scripts.build_team_form.save_dataframe", store.save_dataframe
        )
        return TeamFormBuilder()

    def test_target_rows_replaced_and_history_preserved(self, monkeypatch):
        """Upserting one (season, week) replaces ONLY that target; all else survives."""
        store = _FakeSilverStore()
        store._table = _seed_multi_season_history()
        builder = self._make_builder_with_store(monkeypatch, store)

        original = store._table.copy()
        fresh = _fresh_target_rows(2024, 2)

        builder._upsert_current_week_form(fresh, target_season=2024, target_week=2)

        result = store._table

        # (b) Every OTHER (season, week) pair survives with its original count.
        for season, week in [(2022, 1), (2022, 2), (2023, 1), (2023, 2), (2024, 1)]:
            before = original[
                (original["target_season"] == season)
                & (original["target_week"] == week)
            ]
            after = result[
                (result["target_season"] == season) & (result["target_week"] == week)
            ]
            assert len(after) == len(before) == 2, (
                f"history for ({season}, W{week}) was not preserved"
            )

        # (a) The target (2024, W2) rows were REPLACED with the fresh sentinel values.
        target_after = result[
            (result["target_season"] == 2024) & (result["target_week"] == 2)
        ]
        assert len(target_after) == 2
        assert (target_after["rolling_epa_per_play"] == 0.999).all(), (
            "target rows were not replaced with the freshly-built values"
        )

        # The full season span is intact (no shrink): 2022, 2023, 2024 all present.
        assert set(result["target_season"].unique()) == {2022, 2023, 2024}

    def test_repeat_upsert_is_idempotent(self, monkeypatch):
        """Running the same upsert twice does NOT grow the row count (idempotency, c)."""
        store = _FakeSilverStore()
        store._table = _seed_multi_season_history()
        builder = self._make_builder_with_store(monkeypatch, store)

        fresh = _fresh_target_rows(2024, 2)

        builder._upsert_current_week_form(fresh, target_season=2024, target_week=2)
        count_after_first = len(store._table)

        builder._upsert_current_week_form(fresh, target_season=2024, target_week=2)
        count_after_second = len(store._table)

        assert count_after_first == count_after_second, (
            "repeat upsert grew the table -- not idempotent (append-merge leak)"
        )
        # Sanity: still exactly 2 rows for the target, full span preserved.
        target = store._table[
            (store._table["target_season"] == 2024) & (store._table["target_week"] == 2)
        ]
        assert len(target) == 2
        assert set(store._table["target_season"].unique()) == {2022, 2023, 2024}

    def test_upsert_into_empty_table_creates_target_only(self, monkeypatch):
        """When no history exists yet, the upsert seeds the table with the target rows."""
        store = _FakeSilverStore()  # empty -> load raises FileNotFoundError
        builder = self._make_builder_with_store(monkeypatch, store)

        fresh = _fresh_target_rows(2024, 5)
        builder._upsert_current_week_form(fresh, target_season=2024, target_week=5)

        assert store._table is not None
        assert len(store._table) == 2
        assert set(store._table["target_week"].unique()) == {5}
