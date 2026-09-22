"""Every playing-surface spelling in data/venues.json is classified, and nothing defaults.

Plan 33.2-10, orchestrator-assigned extra step 3b (the deferred-items entry
"`surface_mismatch` treats natural and hybrid grass spelled `Grass`, `Hybrid Grass`,
`Desso GrassMaster` or `RealGrass` as synthetic").

THE DEFECT. ``features.contextual.GRASS_SURFACES`` held two spellings, ``Bermuda Grass`` and
``Kentucky Bluegrass``, and ``_is_grass_surface`` answered "synthetic" for EVERY other string
-- so Wembley, Tottenham, Frankfurt, Mexico City, Croke Park and the Olympiastadion
(``Grass``), Twickenham and Arena Corinthians (``Desso GrassMaster``, natural grass stitched
with synthetic fibres) and the Allianz Arena (``Hybrid Grass``) were all treated as
artificial turf, and ``surface_mismatch`` was wrong at every game played there.

THE FIX. An explicit, COMPLETE mapping of every distinct spelling to grass or synthetic
(hybrid grass is grass: a natural-grass pitch reinforced with fibres), and an unmapped
spelling RAISES by name -- at calculator construction for the venue file, and at
classification for any other string -- rather than silently defaulting.

``RealGrass`` IS SYNTHETIC. The deferred-items entry listed it among the natural pitches.
It is not: RealGrass is an infilled artificial-turf product, and Texas Stadium (DAL99, its
only record here) never had natural grass in its whole life -- Texas Turf, then AstroTurf,
then RealGrass from 2002. Its classification therefore does not change.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from features import contextual
from features.contextual import ContextualFeaturesCalculator

VENUES_PATH = Path(__file__).resolve().parents[2] / "data" / "venues.json"

# The spellings the defect misclassified, and the answer each must now give.
NOW_GRASS = ("Grass", "Hybrid Grass", "Desso GrassMaster")
STILL_GRASS = ("Bermuda Grass", "Kentucky Bluegrass")
SYNTHETIC = (
    "AstroPlay",
    "AstroTurf",
    "FieldTurf",
    "Matrix Turf",
    "NexTurf",
    "RealGrass",
)

# Was: ``"Sport Turf"`` sat in SYNTHETIC above, and step 3b classified it by what the
# spelling says. Plan 33.2-20 corrected its ONE record -- the Stade de France is a hybrid
# SIS Grass pitch, not synthetic turf -- which left the spelling used by no venue at all.
# The map mirrors the file in BOTH directions, so the entry was removed with the record,
# and the spelling now behaves like any other unrecorded string: it RAISES.
#
# The intent the old row carried is kept, not dropped: it asserted that a spelling is
# classified by what it says rather than by a guess about the venue. That is now asserted
# by ``Matrix Turf`` (still recorded, for SoFi and AT&T Stadium), whose two MISTAKEN
# copies at MEL00 and RIO00 are exactly what Plan 33.2-20 corrected.
RETIRED_WITH_ITS_ONLY_RECORD = ("Sport Turf",)


def _venue_surfaces() -> set[str]:
    venues = json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]
    return {str(venue["surface"]) for venue in venues}


class TestTheMappingIsComplete:
    def test_every_distinct_surface_in_venues_json_is_mapped(self) -> None:
        assert (
            sorted(_venue_surfaces() - set(contextual.SURFACE_CLASS_BY_SPELLING)) == []
        )

    def test_the_mapping_names_no_spelling_the_file_does_not_use(self) -> None:
        """A dead entry is a guess about a venue nobody recorded; the map mirrors the file."""
        assert (
            sorted(set(contextual.SURFACE_CLASS_BY_SPELLING) - _venue_surfaces()) == []
        )

    def test_every_class_is_grass_or_synthetic(self) -> None:
        assert set(contextual.SURFACE_CLASS_BY_SPELLING.values()) == {
            "grass",
            "synthetic",
        }

    def test_the_venue_file_validates(self) -> None:
        contextual.validate_surface_vocabulary()


class TestTheKnownSpellingsClassifyCorrectly:
    @pytest.mark.parametrize("spelling", NOW_GRASS + STILL_GRASS)
    def test_natural_and_hybrid_grass_is_grass(self, spelling: str) -> None:
        assert contextual.surface_is_grass(spelling) is True

    @pytest.mark.parametrize("spelling", SYNTHETIC)
    def test_artificial_turf_is_synthetic(self, spelling: str) -> None:
        assert contextual.surface_is_grass(spelling) is False

    def test_the_calculator_agrees_with_the_module_classifier(self) -> None:
        calc = ContextualFeaturesCalculator()
        for spelling in contextual.SURFACE_CLASS_BY_SPELLING:
            assert calc._is_grass_surface(spelling) is contextual.surface_is_grass(
                spelling
            )


class TestASpellingRetiredWithItsOnlyRecord:
    """Plan 33.2-20: the map mirrors the file, so a spelling nothing uses leaves it."""

    @pytest.mark.parametrize("spelling", RETIRED_WITH_ITS_ONLY_RECORD)
    def test_no_venue_record_uses_it_any_more(self, spelling: str) -> None:
        assert spelling not in _venue_surfaces()

    @pytest.mark.parametrize("spelling", RETIRED_WITH_ITS_ONLY_RECORD)
    def test_it_is_absent_from_the_map(self, spelling: str) -> None:
        assert spelling not in contextual.SURFACE_CLASS_BY_SPELLING

    @pytest.mark.parametrize("spelling", RETIRED_WITH_ITS_ONLY_RECORD)
    def test_it_now_raises_rather_than_defaulting(self, spelling: str) -> None:
        """Fail-closed: a returning record must be classified deliberately, not guessed."""
        with pytest.raises(
            contextual.UnknownSurfaceError, match="SURFACE_CLASS_BY_SPELLING"
        ):
            contextual.surface_is_grass(spelling)


class TestTheThreeCorrected2026Venues:
    """The corrected records classify as grass, which is what surface_mismatch reads."""

    @pytest.mark.parametrize("stadium_id", ("MEL00", "RIO00", "PAR00"))
    def test_each_corrected_venue_is_now_grass(self, stadium_id: str) -> None:
        venues = json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]
        record = next(v for v in venues if v["stadium_id"] == stadium_id)
        assert contextual.surface_is_grass(record["surface"]) is True, (
            f"{stadium_id} reads {record['surface']!r}, which classifies as synthetic. "
            "All three are natural or hybrid grass pitches "
            "(tests.phase33_state.P332_20_VENUE_SURFACE_SOURCES)."
        )

    def test_matrix_turf_still_has_the_records_it_was_copied_from(self) -> None:
        """Non-vacuity: the correction removed two COPIES, not the spelling itself."""
        venues = json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]
        matrix = sorted(
            str(v["stadium_id"]) for v in venues if v["surface"] == "Matrix Turf"
        )
        assert matrix == ["DAL00", "LAX01"], (
            f"Matrix Turf is now recorded for {matrix!r}. It was MEL00 and RIO00's value "
            "too, copied from SoFi Stadium (LAX01); the correction removed those two "
            "copies and must not have touched the venues that really use it."
        )


class TestAnUnknownSpellingRaises:
    @pytest.mark.parametrize("spelling", ["Astro Turf", "grass", "Natural Grass", ""])
    def test_an_unmapped_spelling_raises_by_name(self, spelling: str) -> None:
        with pytest.raises(
            contextual.UnknownSurfaceError, match="SURFACE_CLASS_BY_SPELLING"
        ):
            contextual.surface_is_grass(spelling)

    def test_an_unmapped_spelling_in_the_venue_file_refuses_validation(self) -> None:
        venues = json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]
        planted = [*venues, {**venues[0], "stadium_id": "ZZZ00", "surface": "Moss"}]
        with pytest.raises(contextual.UnknownSurfaceError, match="ZZZ00"):
            contextual.validate_surface_vocabulary(planted)

    def test_the_refusal_is_not_a_value_error(self) -> None:
        """It must not be swallowed by the build's optional-source handler."""
        assert not issubclass(contextual.UnknownSurfaceError, ValueError)


class TestTheMismatchFeatureUsesTheFix:
    def test_a_grass_team_at_wembley_is_no_longer_a_mismatch(self) -> None:
        """Miami (Bermuda Grass) at Wembley (Grass): same class, no mismatch."""
        calc = ContextualFeaturesCalculator()
        assert calc._compute_surface_mismatch("MIA", "wembley_stadium") == 0.0

    def test_a_turf_team_at_wembley_is_now_a_mismatch(self) -> None:
        """Buffalo (FieldTurf) at Wembley (Grass): different classes, a mismatch."""
        calc = ContextualFeaturesCalculator()
        assert calc._compute_surface_mismatch("BUF", "wembley_stadium") == 1.0
