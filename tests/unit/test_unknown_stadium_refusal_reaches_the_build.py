"""A stadium-resolution refusal stops the gold build; it does not degrade it.

OWNER RULING, 2026-09-13. Plan 33.1-03 FOUND this, did not fix it, and the
owner assigned it to Plan 33.1-04 because 33.1-04 already declares
`scripts/build_features.py` in its files_modified. It is a hand-off, not an
open defect left behind.

THE DEFECT
----------
`features.contextual.UnknownStadiumError` subclasses `ValueError`, and
`scripts.build_features._SOURCE_LOAD_ERRORS` contains `ValueError`. So inside
`load_all_feature_sources` a stadium-resolution REFUSAL -- the loudest thing
R1's resolver can say -- was caught by the contextual block's
graceful-degradation handler, logged as a WARNING, and converted into an EMPTY
contextual frame. Gold would then be built without the whole
venue / travel / rest / situational family, and the run would exit 0.

That is exactly the shape this milestone's invariant names: "a green run is
precisely what the known silent defect produces". It is also the same swallow
D33.1-07 removes one layer up, in `features/weather.py`.

This is MEASURED, not predicted. Building contextual features over today's
15-column silver raises
`UnknownStadiumError: game '2018_W01_ATL@PHI' names stadium_id None` -- because
Phase 33 Wave 12, which backfills `stadium_id` into `data/silver/games.parquet`,
has not run. Plan 33.1-07 owns that gap; this module owns the swallow.

THE SHAPE OF THE FIX, AND WHY
------------------------------
The SURGICAL one: re-raise `UnknownStadiumError` explicitly BEFORE the generic
`_SOURCE_LOAD_ERRORS` handler on the contextual block. The two alternatives
both move behaviour nobody asked to move:

* removing `ValueError` from `_SOURCE_LOAD_ERRORS` changes ELEVEN other
  optional-source guards, whose graceful-degradation contract is deliberate and
  separately tested;
* re-basing `UnknownStadiumError` off `ValueError` changes what every existing
  `except ValueError` around the resolver catches -- including
  `features/contextual.py:221` and `scripts/ingest_weather.py:782`, which catch
  it BY NAME today and would be unaffected, and any caller that catches the
  base type, which would not.

THE CONTROL IS THE POINT. Without it, "the refusal propagates" could be true
simply because no handler exists. It exists, it fires, and an ordinary
`ValueError` still degrades to an empty contextual frame -- which is today's
contract and stays.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import pytest

from tests.fixtures.elo_sandbox import (
    make_season_games,
    redirect_storage_to_sandbox,
)

LIVE_SEASON = 2026


def _builder_with_raising_contextual(monkeypatch, tmp_path, error: Exception):
    from data.storage import save_dataframe
    from scripts.build_features import FeatureMatrixBuilder

    redirect_storage_to_sandbox(monkeypatch, tmp_path)
    save_dataframe(
        make_season_games(LIVE_SEASON, weeks=1),
        "games",
        layer="silver",
        replace_mode=True,
    )

    builder = FeatureMatrixBuilder()

    def _raise(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(builder.contextual_calc, "build_features", _raise)
    return builder


class TestTheStadiumRefusalCannotBecomeAnEmptyFrame:
    def test_an_ordinary_value_error_IS_swallowed_into_an_empty_frame(
        self, tmp_path, monkeypatch
    ) -> None:
        """THE CONTROL. This is the behaviour the refusal must escape."""
        builder = _builder_with_raising_contextual(
            monkeypatch, tmp_path, ValueError("an ordinary source-load failure")
        )

        sources = builder.load_all_feature_sources(target_season=LIVE_SEASON)

        assert "contextual" in sources
        assert len(sources["contextual"]) == 0, (
            "a _SOURCE_LOAD_ERRORS member is converted into an empty contextual "
            "frame -- that is today's graceful-degradation contract, and the "
            "reason a stadium REFUSAL must not be treated as one of them."
        )

    def test_the_unknown_stadium_refusal_reaches_the_caller_instead(
        self, tmp_path, monkeypatch
    ) -> None:
        from features.contextual import UnknownStadiumError

        builder = _builder_with_raising_contextual(
            monkeypatch,
            tmp_path,
            UnknownStadiumError("game '2018_W01_ATL@PHI' names stadium_id None"),
        )

        with pytest.raises(UnknownStadiumError) as excinfo:
            builder.load_all_feature_sources(target_season=LIVE_SEASON)

        assert "2018_W01_ATL@PHI" in str(excinfo.value), (
            "the refusal must reach the caller with the game it names intact; "
            "a refusal the operator cannot act on is barely better than a "
            "warning nobody reads"
        )

    def test_the_refusal_is_still_a_value_error_so_the_fix_is_the_handler(
        self,
    ) -> None:
        """Records WHY an explicit handler is needed at all.

        If this ever stops being true, the explicit handler becomes redundant
        rather than wrong -- and this test says so rather than leaving the next
        reader to work out which of the two mechanisms is load-bearing.
        """
        from features.contextual import UnknownStadiumError
        from scripts.build_features import _SOURCE_LOAD_ERRORS

        assert issubclass(UnknownStadiumError, ValueError)
        assert ValueError in _SOURCE_LOAD_ERRORS
