"""The postseason partition, on a HELD-OUT FIXTURE -- BACKSTOP evidence, not live.

WHY THIS HALF IS FIXTURE-ONLY, STATED BEFORE ANYTHING IS ASSERTED (T-33-54)
---------------------------------------------------------------------------
All 272 rows of the single live 2026 capture are ``REG``, weeks 1 to 18. Weeks 19-22 are
unseeded at capture time -- the bracket does not exist until the regular season ends --
so there is no postseason row in 2026 to measure. The SPEC's own edge table marks this
half a HELD-OUT FIXTURE TEST for exactly that reason, and
``tests/fixtures/season_2026.WEEK_19_POSTSEASON_FIXTURE`` is the constructed wild-card
frame the whole phase shares.

THIS IS THEREFORE A BACKSTOP, AND CALLING IT ANYTHING ELSE WOULD BE A MISREPRESENTATION.
A constructed row proves that the partition WOULD place a postseason game correctly; it
does not prove that a real 2026 postseason game was ever partitioned, because none has
existed yet.

WHAT WOULD UPGRADE IT TO LIVE EVIDENCE
---------------------------------------
A future capture carrying 2026 weeks 19-22. When one exists, the fixture stops being the
only source: ``load_captured_schedule`` will return rows whose ``game_type`` is ``WC``,
``DIV``, ``CON`` or ``SB``, and this module's assertions can be re-pointed at them with
the constructed frame demoted to a shape check. Nothing else has to change.

THE season_type VALUE IS PRODUCED, NEVER TYPED
-----------------------------------------------
The fixture is built in the FEED's column shape, and the feed has no ``season_type`` --
it carries ``game_type`` and nothing else. ``season_type`` is a SILVER column, derived by
``scripts/ingest_games._derive_season_type``. So the fixture is run through the real
``GameDataIngester.transform_schedule_data`` and the derived value is compared against
``_derive_season_type`` applied to the same rows. A hand-typed ``"Postseason"`` literal
would make this module agree with itself forever, including on the day the derivation
changes -- which is precisely the failure that let every postseason game in this project
be labelled a regular-season one in the first place.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.ingest_games import GameDataIngester, _derive_season_type
from tests.fixtures.season_2026 import WEEK_19_POSTSEASON_FIXTURE
from utils.similar_games import SimilarGamesEngine

POSTSEASON = "Postseason"
REGULAR = "Regular"

# The value the SPEC's held-out case is about. Recorded as a constant so a reader can
# see that the expectation below is the postseason bucket and not a spelling of it.
EXPECTED_POSTSEASON_WEEK = 19

UPGRADE_CONDITION = (
    "a future capture carrying 2026 weeks 19-22 would replace this constructed frame "
    "with real rows; until one exists this module is a BACKSTOP, not live evidence."
)


# ---------------------------------------------------------------------------
# The partition under test.
# ---------------------------------------------------------------------------


def partition_by_season_type(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a silver-shaped frame into ``season_type`` buckets.

    Both buckets always exist, empty or not. A partition that omits the bucket it found
    nothing in forces every caller to guess whether an absent key means "no postseason
    games" or "this frame was never partitioned", and those are different facts.

    Args:
        frame: A frame carrying a ``season_type`` column.

    Returns:
        ``{Regular: frame, Postseason: frame}``, each possibly empty.
    """
    return {
        bucket: frame[frame["season_type"] == bucket].copy()
        for bucket in (REGULAR, POSTSEASON)
    }


# ---------------------------------------------------------------------------
# Fixtures: the held-out postseason frame, and a regular-season counterpart.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def postseason_silver() -> pd.DataFrame:
    """The week-19 fixture, run through the REAL production transformer."""
    return GameDataIngester().transform_schedule_data(WEEK_19_POSTSEASON_FIXTURE)


@pytest.fixture(scope="module")
def regular_season_silver() -> pd.DataFrame:
    """The same two games relabelled ``REG`` in week 5, through the same transformer.

    CONSTRUCTED from the held-out fixture rather than sliced out of the capture, so this
    module stays in the unit tier: it opens no parquet and touches no lake.
    """
    feed = WEEK_19_POSTSEASON_FIXTURE.copy()
    feed["game_type"] = "REG"
    feed["week"] = 5
    feed["game_id"] = [
        game_id.replace("_19_", "_05_") for game_id in feed["game_id"].astype(str)
    ]
    return GameDataIngester().transform_schedule_data(feed)


# ---------------------------------------------------------------------------
# The derivation is exercised, not restated.
# ---------------------------------------------------------------------------


def test_the_raw_fixture_carries_no_season_type_so_the_value_must_be_derived() -> None:
    """The premise of this whole module, asserted rather than assumed.

    If the fixture ever grew its own ``season_type`` column the test below would
    compare a hand-typed literal against a derivation and pass, while proving nothing
    about the derivation. Asserting the absence is what keeps the comparison honest.
    """
    assert "season_type" not in WEEK_19_POSTSEASON_FIXTURE.columns, (
        "WEEK_19_POSTSEASON_FIXTURE now carries its own season_type column. It is "
        "built in the FEED's column shape, and the feed has no such column -- "
        "season_type is derived at ingest. A fixture that supplies the answer cannot "
        "be used to check the code that computes it."
    )
    assert "game_type" in WEEK_19_POSTSEASON_FIXTURE.columns


def test_the_stored_season_type_equals_what_the_real_derivation_returns(
    postseason_silver: pd.DataFrame,
) -> None:
    """The held-out row's ``season_type`` comes from ``_derive_season_type``.

    Applied row by row so a frame that got the right value for one game and the wrong
    value for another cannot hide behind a set comparison.
    """
    feed_rows = WEEK_19_POSTSEASON_FIXTURE.to_dict("records")
    derived = [_derive_season_type(row) for row in feed_rows]
    stored = list(postseason_silver["season_type"])

    assert derived == stored, (
        f"derived {derived} but the transformer stored {stored}. The two can only "
        "disagree if something other than _derive_season_type wrote the column."
    )
    assert set(derived) == {POSTSEASON}, derived
    assert set(postseason_silver["week"]) == {EXPECTED_POSTSEASON_WEEK}, sorted(
        set(postseason_silver["week"])
    )
    assert set(postseason_silver["game_type"]) == {"WC"}


def test_the_regular_season_counterpart_derives_the_other_value(
    regular_season_silver: pd.DataFrame,
) -> None:
    """Non-vacuity: the same derivation answers ``Regular`` for a ``REG`` row.

    A derivation that returned ``Postseason`` for everything would satisfy the test
    above completely.
    """
    assert set(regular_season_silver["season_type"]) == {REGULAR}, sorted(
        set(regular_season_silver["season_type"])
    )


# ---------------------------------------------------------------------------
# The partition itself.
# ---------------------------------------------------------------------------


def test_the_held_out_week_19_row_lands_in_the_postseason_bucket(
    postseason_silver: pd.DataFrame,
) -> None:
    """A partition over ``season_type`` places the wild-card rows on the right side."""
    buckets = partition_by_season_type(postseason_silver)

    assert len(buckets[POSTSEASON]) == len(postseason_silver), (
        f"{len(buckets[POSTSEASON])} of {len(postseason_silver)} postseason rows "
        "landed in the postseason bucket."
    )
    assert buckets[REGULAR].empty, sorted(buckets[REGULAR]["game_id"])
    assert set(buckets[POSTSEASON]["week"]) == {EXPECTED_POSTSEASON_WEEK}


def test_a_frame_with_no_postseason_rows_yields_an_empty_bucket_and_raises_nothing(
    regular_season_silver: pd.DataFrame,
) -> None:
    """The empty case is a legitimate answer, not an error -- the R12 sibling of NF-11.

    Weeks 19-22 do not exist for most of a season, and every regular-season Friday run
    partitions a frame with zero postseason rows. That has to be an empty bucket rather
    than a raise, or the pipeline would fail every week until January.
    """
    buckets = partition_by_season_type(regular_season_silver)

    assert buckets[POSTSEASON].empty, sorted(buckets[POSTSEASON]["game_id"])
    assert list(buckets[POSTSEASON].columns) == list(regular_season_silver.columns), (
        "the empty postseason bucket lost its columns; an empty frame with no schema "
        "breaks a caller that concatenates the buckets back together."
    )
    assert len(buckets[REGULAR]) == len(regular_season_silver)
    assert set(buckets) == {REGULAR, POSTSEASON}, (
        "the partition dropped the bucket it found nothing in. An absent key and a "
        "measured zero are different facts."
    )


# ---------------------------------------------------------------------------
# The live consumer: utils/similar_games.py's fallback read.
# ---------------------------------------------------------------------------


def test_the_similar_games_fallback_read_resolves_to_the_real_value(
    postseason_silver: pd.DataFrame,
    regular_season_silver: pd.DataFrame,
) -> None:
    """``target.get("season_type", target.get("game_type"))`` returns the stored value.

    ``utils/similar_games.py`` reads ``season_type`` with ``game_type`` as a fallback.
    The fallback exists for frames predating the derivation, and the risk is that it is
    silently ALWAYS taken -- in which case the similarity engine would be comparing
    ``WC`` against ``REG`` rather than ``Postseason`` against ``Regular``, and would
    still look like it worked.
    """
    postseason_row = postseason_silver.iloc[0]
    regular_row = regular_season_silver.iloc[0]

    assert postseason_row.get("season_type", postseason_row.get("game_type")) == (
        POSTSEASON
    ), "the postseason row's fallback read resolved to the feed's game_type."
    assert regular_row.get("season_type", regular_row.get("game_type")) == REGULAR


def test_the_season_type_term_is_worth_exactly_its_documented_weight(
    tmp_path, postseason_silver: pd.DataFrame
) -> None:
    """The read is not merely resolvable -- it CHANGES the engine's score.

    Two pairs identical in every other respect (same week, same day-of-week flags) so
    the only moving part is ``season_type``. The engine adds 0.4 when the types match,
    which is the whole contribution the partition buys.

    The engine is constructed against a path under ``tmp_path``; it opens no connection
    at construction and this test calls no method that would.
    """
    engine = SimilarGamesEngine(db_path=str(tmp_path / "unused.duckdb"))

    target = postseason_silver.iloc[0]
    same_type = postseason_silver.iloc[1]
    other_type = postseason_silver.iloc[1].copy()
    other_type["season_type"] = REGULAR

    matched = engine._calculate_context_similarity(target, same_type)
    mismatched = engine._calculate_context_similarity(target, other_type)

    assert matched - mismatched == pytest.approx(0.4), (
        f"the season_type term contributed {matched - mismatched} rather than 0.4 "
        f"(matched={matched}, mismatched={mismatched})."
    )


# ---------------------------------------------------------------------------
# The limitation, asserted so it cannot be quietly forgotten.
# ---------------------------------------------------------------------------


def test_this_module_declares_itself_a_backstop_and_names_its_upgrade_condition() -> (
    None
):
    """T-33-54: fixture evidence is labelled as fixture evidence, in the record.

    A constructed frame presented as live evidence is how an unproven property comes to
    look proven. The module docstring carries the label and the upgrade condition, and
    this test is what stops either from being edited away silently.
    """
    # Whitespace-normalised, so a phrase that happens to straddle a line wrap in the
    # docstring is still found. The claim is about the WORDS, not about the wrapping.
    docstring = " ".join((__doc__ or "").lower().split())
    for phrase in (
        "backstop",
        "held-out fixture",
        "weeks 19-22 are unseeded",
        "what would upgrade it to live evidence",
    ):
        assert phrase in docstring, (
            f"the module docstring no longer states {phrase!r}; the limitation and the "
            "route out of it are the two things it exists to carry."
        )
    assert "weeks 19-22" in UPGRADE_CONDITION
