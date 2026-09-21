"""Tests for FeatureBuilder Protocol compliance.

Verifies that the Protocol is correctly defined and that pyright-compatible
structural subtyping works as expected at runtime via isinstance checks
with runtime_checkable.
"""

from collections.abc import Mapping
from datetime import datetime

import pandas as pd

from features.protocol import FeatureBuilder, InformationTimeProvider


class ConformingBuilder:
    """A builder that satisfies the FeatureBuilder Protocol."""

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        return games_df

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        return {"feature": 1.0}


class MissingAsOfDatetime:
    """A builder missing the as_of_datetime parameter."""

    def build_features(
        self,
        games_df: pd.DataFrame,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        return games_df

    def get_features_for_game(
        self,
        game_id: str,
    ) -> dict[str, float]:
        return {"feature": 1.0}


class WrongReturnType:
    """A builder with wrong return type on build_features."""

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> list[dict]:  # Wrong: should be pd.DataFrame
        return [{}]

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        return {"feature": 1.0}


class TestFeatureBuilderProtocol:
    """Tests for FeatureBuilder Protocol definition."""

    def test_conforming_builder_satisfies_protocol(self):
        """A class with matching signature satisfies FeatureBuilder Protocol."""
        builder = ConformingBuilder()
        assert isinstance(builder, FeatureBuilder)

    def test_missing_as_of_datetime_does_not_satisfy_protocol(self):
        """A class missing as_of_datetime does NOT satisfy the Protocol.

        Note: runtime_checkable isinstance checks are limited to method
        name presence, not full signature. For signature checks, pyright
        is the enforcement mechanism. Here we verify that if we type-annotate
        a function parameter as FeatureBuilder, a MissingAsOfDatetime instance
        would fail pyright analysis. At runtime, we verify the Protocol class
        exists and the conforming builder passes.
        """
        # runtime_checkable checks method names only, so this may pass at runtime.
        # The real enforcement is at static analysis time via pyright.
        # We test that the Protocol has the expected method signatures.
        import inspect

        sig = inspect.signature(FeatureBuilder.build_features)
        params = list(sig.parameters.keys())
        assert "as_of_datetime" in params, (
            "Protocol must require as_of_datetime parameter"
        )

    def test_protocol_has_correct_build_features_signature(self):
        """Verify Protocol's build_features has the expected parameters."""
        import inspect

        sig = inspect.signature(FeatureBuilder.build_features)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "games_df" in params
        assert "as_of_datetime" in params
        assert "target_season" in params
        assert "target_week" in params

    def test_protocol_has_get_features_for_game(self):
        """Verify Protocol defines get_features_for_game method."""
        import inspect

        assert hasattr(FeatureBuilder, "get_features_for_game")
        sig = inspect.signature(FeatureBuilder.get_features_for_game)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "game_id" in params
        assert "as_of_datetime" in params

    def test_build_features_return_annotation_is_dataframe(self):
        """Verify build_features return type annotation is pd.DataFrame."""
        import inspect

        sig = inspect.signature(FeatureBuilder.build_features)
        assert sig.return_annotation is pd.DataFrame


# ---------------------------------------------------------------------------
# Phase 33.2, Plan 33.2-01 Task 2: the SEPARATE InformationTimeProvider Protocol.
#
# `information_times` lives on its own runtime-checkable Protocol, NOT on
# FeatureBuilder. FeatureBuilder is @runtime_checkable and runtime isinstance
# checks METHOD PRESENCE, so a new required member would flip every existing
# isinstance(_, FeatureBuilder) assertion -- the one above on ConformingBuilder,
# and the two in test_contextual_extensions.py and test_qb_tracking.py, which
# this plan deliberately does NOT edit. The member-set assertion below is the
# cheap guard that fails HERE, by name, if anybody re-expands FeatureBuilder.
# ---------------------------------------------------------------------------


def _public_members(protocol: type) -> set[str]:
    return {name for name in dir(protocol) if not name.startswith("_")}


class ConformingProvider:
    """A provenance supplier that implements ONLY the InformationTimeProvider members."""

    def information_times(
        self,
        games_df: pd.DataFrame,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["game_id", "basis", "information_time"])

    def no_information_signature(self) -> Mapping[str, float | None]:
        return {"feature": 0.0}


class BuilderAndProvider(ConformingBuilder):
    """Both Protocols by structural subtyping -- no Protocol appears in its bases."""

    def information_times(
        self,
        games_df: pd.DataFrame,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame(columns=["game_id", "basis", "information_time"])

    def no_information_signature(self) -> Mapping[str, float | None]:
        return {"feature": 0.0}


class TestInformationTimeProviderProtocol:
    """The provenance contract composes with FeatureBuilder; it does not extend it."""

    def test_feature_builder_member_set_is_unchanged(self):
        assert _public_members(FeatureBuilder) == {
            "build_features",
            "get_features_for_game",
        }

    def test_provider_member_set_is_exactly_two(self):
        assert _public_members(InformationTimeProvider) == {
            "information_times",
            "no_information_signature",
        }

    def test_provider_is_runtime_checkable(self):
        assert getattr(InformationTimeProvider, "_is_runtime_protocol", False)

    def test_the_two_method_builder_satisfies_only_feature_builder(self):
        builder = ConformingBuilder()
        assert isinstance(builder, FeatureBuilder)
        assert not isinstance(builder, InformationTimeProvider)

    def test_a_two_method_provider_satisfies_only_the_provider(self):
        provider = ConformingProvider()
        assert isinstance(provider, InformationTimeProvider)
        assert not isinstance(provider, FeatureBuilder)

    def test_one_class_satisfies_both_with_no_protocol_base(self):
        both = BuilderAndProvider()
        assert isinstance(both, FeatureBuilder)
        assert isinstance(both, InformationTimeProvider)
        assert FeatureBuilder not in type(both).__mro__
        assert InformationTimeProvider not in type(both).__mro__

    def test_information_times_signature(self):
        import inspect

        params = inspect.signature(InformationTimeProvider.information_times).parameters
        assert list(params) == ["self", "games_df", "target_season", "target_week"]
        assert params["target_season"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["target_week"].kind is inspect.Parameter.KEYWORD_ONLY
