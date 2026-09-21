"""FeatureBuilder Protocol definition.

Defines the structural type contract that all feature builders must satisfy.
The key requirement is the `as_of_datetime` parameter, which serves as the
time-fence: builders MUST NOT use any data with timestamps after this cutoff.

Uses typing.Protocol for structural subtyping -- existing classes satisfy
the protocol if they have matching method signatures, without needing to
change their inheritance hierarchy. Pyright catches violations at static
analysis time.

TWO INDEPENDENT PROTOCOLS LIVE HERE (Phase 33.2, Plan 33.2-01). ``FeatureBuilder``
is the build contract; ``InformationTimeProvider`` is the provenance contract the
gold information-time gate reads. They are deliberately separate: both are
``@runtime_checkable``, and a runtime ``isinstance`` check tests METHOD PRESENCE,
so adding a required member to ``FeatureBuilder`` would flip every existing
``isinstance(_, FeatureBuilder)`` assertion on a two-method builder. Protocols
compose by structural subtyping, so one class satisfies both with no base-class
change -- ``features.elo_features.EloFeatureBuilder`` is the first that does.
"""

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FeatureBuilder(Protocol):
    """Protocol that all feature builders must satisfy.

    The as_of_datetime parameter is the time-fence: builders MUST NOT
    use any data with timestamps after this cutoff. This enforces
    temporal correctness and prevents data leakage.
    """

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """Build features using only data available before as_of_datetime.

        Args:
            games_df: DataFrame of games to build features for.
            as_of_datetime: Time-fence cutoff. Only data before this
                timestamp may be used.
            target_season: Optional season to filter features for.
            target_week: Optional week to filter features for.

        Returns:
            DataFrame with computed features, one row per team per game.
        """
        ...

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get features for a single game using data before as_of_datetime.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Time-fence cutoff.

        Returns:
            Dictionary mapping feature names to values.
        """
        ...


@runtime_checkable
class InformationTimeProvider(Protocol):
    """The provenance contract: WHEN each game's information was known.

    Owned by Plan 33.2-01 (``<owned_contract>``); every later source-registering plan
    implements it verbatim. Declared here and checked structurally, in the same
    declare-in-docstring / check-structurally style as ``FeatureBuilder`` above.
    ``isinstance(builder, InformationTimeProvider)`` IS the gold build's admission test
    for the information-time gate, so the set of checked sources is a structural fact
    rather than a hand-kept list.

    INDEPENDENT OF ``FeatureBuilder``. A builder satisfies both Protocols by structural
    subtyping alone; neither Protocol appears in any builder's bases.

    THE CONTRACT
    ------------
    ``information_times`` returns a PER-ROW frame whose columns are exactly
    ``features.provenance.PROVENANCE_COLUMNS`` -- ``("game_id", "basis",
    "information_time")``:

    * exactly one row per ``game_id`` present in the source frame (a duplicate, a missing
      game or a row for a game the source lacks is refused);
    * ``basis`` is ``"per_row"`` or ``"no_information"`` -- there is no third;
    * ``per_row`` rows carry a tz-aware ``information_time``, compared to that game's own
      lock (18:00 ET on the ET calendar day before kickoff, ``utils.game_lock``);
    * ``no_information`` rows carry a NULL ``information_time`` and are NOT lock-compared;
      instead their VALUES in the source frame are checked against
      ``no_information_signature()``.

    ``no_information`` IS NOT AN EXEMPTION. A source with any ``no_information`` row and
    an EMPTY signature is refused: a source may not declare rows uncheckable and supply
    nothing to check them against.
    """

    def information_times(
        self,
        games_df: pd.DataFrame,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """The per-row provenance frame for the games this builder emits rows for.

        Args:
            games_df: The games the source frame was built from.
            target_season: Optional season filter, applied exactly as the builder
                applies it in ``build_features``.
            target_week: Optional week filter, likewise.

        Returns:
            A frame with exactly the ``PROVENANCE_COLUMNS`` columns.
        """
        ...

    def no_information_signature(self) -> Mapping[str, float | None]:
        """What an undatable row's VALUES must be, per source-frame column.

        A float entry means "must equal this value"; ``None`` means "must be null".

        Returns:
            Column name -> declared neutral value. Empty only for a source that never
            emits a ``no_information`` row.
        """
        ...
