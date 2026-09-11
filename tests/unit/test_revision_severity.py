"""The live-capture revision ruling: what a capture-to-capture difference MEANS.

TEST CLASS: plain unit tests. Nothing here reads a real manifest, a real bet list, a real
parquet or the network. Every fixture is a plain dict in the shape
``data.upstream_live.build_capture_entry`` produces, so the ruling is exercised offline and
a failure here is a failure of the RULE rather than of the environment.

WHAT THESE TESTS ARE FOR
------------------------
`.planning/research/PITFALLS.md` B3 says the calibration below is what decides whether the
whole detector is useful or ignored, and it names BOTH failure modes:

* A revision alert every single week -- the detector is miscalibrated and is wallpaper by
  week three. An in-season revision of the CURRENT week is NORMAL, and a NEW week appearing
  in a later capture is not a revision at all; it is what a live season does every Friday.
* No revision alert all season -- the detector is dead (F2). A downgraded CRITICAL and a
  genuinely calm week look identical in a log, so every ambiguity has to resolve LOUD.

The assertions that actually earn their keep are therefore these four:

1. ``is_revision`` is False for a capture whose only difference is a NEW week.
2. The graded escalation is a NUMBER -- ``severity_rank`` of the graded verdict strictly
   exceeds the ungraded one over the SAME underlying diff.
3. An UNRESOLVABLE graded set yields ``unknown`` with ``correction_owed`` exactly ``None``
   (asserted with ``is None``, never by falsiness), because ``False`` there is the silent
   downgrade the seam exists to prevent.
4. An EXPLICITLY-RECORDED empty graded set is a different observable from an unresolvable
   one, and every verdict names where it looked.

ASCII only, no emoji (CLAUDE.md hard constraint).

Run:  .venv/Scripts/python.exe -m pytest tests/unit/test_revision_severity.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data import live_revision
from data.graded_weeks import GRADED_WEEKS_SOURCE, GradedWeeksUnavailable
from data.live_revision import (
    LIVE_REVISION_SCHEMA_VERSION,
    VERDICT_KEYS,
    WeekDiff,
    as_record,
    compare_week_digests,
    detect_live_revision,
    verdict_severity_rank,
)
from data.revision_events import (
    CORRECTION_OWED,
    CORRECTION_OWED_SCOPE,
    VERDICT_SCHEMA_VERSION,
    RevisionEventClass,
    RevisionSeverity,
    severity_rank,
)
from data.upstream_live import (
    NO_WEEK_COLUMN_BUCKET,
    WEEK_DIGEST_SCHEMA_VERSION,
    WEEK_LABEL_MEANS,
    WEEK_LABEL_SEMANTICS_VERSION,
    WEEK_PARTITION_PER_WEEK,
)

# The module's own source, for the assertions that no hand-written severity string and no
# reader import has drifted into it. Taken from ``__file__`` rather than from a re-typed
# relative path, so the assertion follows the module if it ever moves.
LIVE_REVISION_SOURCE = Path(live_revision.__file__)


def bucket(rows: int, frame: str, **columns: str) -> dict:
    """One ``week_digests`` bucket in the shape 32-04 froze."""
    return {"rows": rows, "frame_sha256": frame, "columns": dict(columns)}


def digests_one_week() -> dict:
    """A single-week map used as the unchanged side of most comparisons."""
    return {"1": bucket(2, "frame-1", epa="epa-1", week="week-1")}


class TestAnIdenticalPairIsNotARevision:
    """Two captures of the same bytes must produce an entirely empty diff."""

    def test_identical_maps_report_no_change_at_all(self) -> None:
        prior = digests_one_week()
        diff = compare_week_digests(prior, digests_one_week())

        assert diff.weeks_changed == ()
        assert diff.weeks_added == ()
        assert diff.weeks_removed == ()
        assert diff.columns_changed == {}
        assert diff.rows_changed == {}
        assert diff.is_revision is False

    def test_two_empty_maps_are_not_a_revision(self) -> None:
        assert compare_week_digests({}, {}).is_revision is False


class TestAMovedWeekReportsItsColumnsAndRows:
    """D32-06: a moved week reports its changed column set and its row counts."""

    def test_a_moved_frame_digest_puts_the_week_in_weeks_changed(self) -> None:
        prior = digests_one_week()
        current = {"1": bucket(2, "frame-2", epa="epa-2", week="week-1")}

        diff = compare_week_digests(prior, current)

        assert diff.weeks_changed == ("1",)
        assert diff.is_revision is True

    def test_only_the_columns_that_moved_are_reported_and_they_are_sorted(self) -> None:
        prior = {"1": bucket(2, "frame-1", epa="e1", wp="w1", week="k1", air="a1")}
        current = {"1": bucket(2, "frame-2", epa="e2", wp="w1", week="k1", air="a2")}

        diff = compare_week_digests(prior, current)

        assert diff.columns_changed == {"1": ("air", "epa")}

    def test_a_moved_row_count_is_reported_as_a_prior_current_pair(self) -> None:
        prior = {"1": bucket(2, "frame-1", epa="e1")}
        current = {"1": bucket(7, "frame-2", epa="e1")}

        diff = compare_week_digests(prior, current)

        assert diff.rows_changed == {"1": (2, 7)}
        assert diff.is_revision is True

    def test_a_row_count_moves_even_when_no_column_digest_is_comparable(self) -> None:
        prior = {"1": bucket(2, "frame-1")}
        current = {"1": bucket(9, "frame-1")}

        diff = compare_week_digests(prior, current)

        assert diff.rows_changed == {"1": (2, 9)}
        assert diff.columns_changed == {}
        assert diff.is_revision is True

    def test_a_column_that_appeared_is_a_change_not_a_non_comparison(self) -> None:
        prior = {"1": bucket(2, "frame-1", epa="e1")}
        current = {"1": bucket(2, "frame-2", epa="e1", cpoe="c1")}

        diff = compare_week_digests(prior, current)

        assert diff.columns_changed == {"1": ("cpoe",)}
        assert diff.is_revision is True

    def test_a_column_that_vanished_is_a_change_not_a_non_comparison(self) -> None:
        prior = {"1": bucket(2, "frame-1", epa="e1", cpoe="c1")}
        current = {"1": bucket(2, "frame-2", epa="e1")}

        diff = compare_week_digests(prior, current)

        assert diff.columns_changed == {"1": ("cpoe",)}
        assert diff.is_revision is True


class TestInSeasonGrowthIsNotARevision:
    """PITFALLS B3, the single predicate that decides whether this detector is read."""

    def test_a_new_week_appears_in_weeks_added_and_never_in_weeks_changed(self) -> None:
        prior = digests_one_week()
        current = dict(prior)
        current["2"] = bucket(5, "frame-2", epa="epa-2", week="week-2")

        diff = compare_week_digests(prior, current)

        assert diff.weeks_added == ("2",)
        assert diff.weeks_changed == ()
        assert diff.columns_changed == {}
        assert diff.rows_changed == {}

    def test_growth_alone_is_not_a_revision(self) -> None:
        prior = digests_one_week()
        current = dict(prior)
        current["2"] = bucket(5, "frame-2", epa="epa-2", week="week-2")

        assert compare_week_digests(prior, current).is_revision is False

    def test_growth_beside_a_moved_week_is_still_a_revision(self) -> None:
        prior = digests_one_week()
        current = {
            "1": bucket(2, "frame-CHANGED", epa="epa-2", week="week-1"),
            "2": bucket(5, "frame-2", epa="epa-2", week="week-2"),
        }

        diff = compare_week_digests(prior, current)

        assert diff.weeks_added == ("2",)
        assert diff.weeks_changed == ("1",)
        assert diff.is_revision is True


class TestARemovedWeekIsARevisionAndALoudOne:
    """Upstream dropping a week it previously published is a bigger claim than moving one."""

    def test_a_week_present_only_in_prior_is_reported_removed(self) -> None:
        prior = {
            "1": bucket(2, "frame-1", epa="e1"),
            "2": bucket(5, "frame-2", epa="e2"),
        }
        current = {"1": bucket(2, "frame-1", epa="e1")}

        diff = compare_week_digests(prior, current)

        assert diff.weeks_removed == ("2",)
        assert diff.weeks_changed == ()

    def test_a_removed_week_makes_it_a_revision(self) -> None:
        prior = {
            "1": bucket(2, "frame-1", epa="e1"),
            "2": bucket(5, "frame-2", epa="e2"),
        }
        current = {"1": bucket(2, "frame-1", epa="e1")}

        assert compare_week_digests(prior, current).is_revision is True


class TestTheWholeFrameBucketComparesLikeAnyOther:
    """A dataset with no week column still reports a moved frame."""

    def test_a_moved_whole_frame_bucket_is_a_revision(self) -> None:
        prior = {NO_WEEK_COLUMN_BUCKET: bucket(40, "frame-1", pos="p1")}
        current = {NO_WEEK_COLUMN_BUCKET: bucket(40, "frame-2", pos="p2")}

        diff = compare_week_digests(prior, current)

        assert diff.weeks_changed == (NO_WEEK_COLUMN_BUCKET,)
        assert diff.columns_changed == {NO_WEEK_COLUMN_BUCKET: ("pos",)}
        assert diff.is_revision is True

    def test_a_first_whole_frame_bucket_is_growth_not_a_revision(self) -> None:
        current = {NO_WEEK_COLUMN_BUCKET: bucket(40, "frame-1", pos="p1")}

        diff = compare_week_digests({}, current)

        assert diff.weeks_added == (NO_WEEK_COLUMN_BUCKET,)
        assert diff.is_revision is False


class TestTheDiffIsSortedAndSerialisable:
    """The verdict rides in a committed file; an unstable ordering would read as a diff."""

    def test_every_collection_is_sorted(self) -> None:
        prior = {
            "3": bucket(1, "f3", b="b1", a="a1"),
            "10": bucket(1, "f10", z="z1"),
            "2": bucket(1, "f2", y="y1"),
        }
        current = {
            "3": bucket(1, "f3-moved", b="b2", a="a2"),
            "10": bucket(1, "f10-moved", z="z2"),
            "9": bucket(1, "f9"),
            "1": bucket(1, "f1"),
        }

        diff = compare_week_digests(prior, current)

        assert diff.weeks_changed == tuple(sorted(diff.weeks_changed))
        assert diff.weeks_added == tuple(sorted(diff.weeks_added))
        assert diff.weeks_removed == tuple(sorted(diff.weeks_removed))
        assert diff.columns_changed["3"] == ("a", "b")
        assert list(diff.columns_changed) == sorted(diff.columns_changed)

    def test_as_record_renders_lists_and_is_byte_identical_across_runs(self) -> None:
        import json

        prior = {"1": bucket(2, "f1", b="b1", a="a1")}
        current = {
            "1": bucket(3, "f1-moved", b="b2", a="a2"),
            "2": bucket(5, "f2"),
        }

        first = as_record(compare_week_digests(prior, current))
        second = as_record(compare_week_digests(prior, current))

        assert json.dumps(first) == json.dumps(second)
        assert first["weeks_changed"] == ["1"]
        assert first["weeks_added"] == ["2"]
        assert first["weeks_removed"] == []
        assert first["columns_changed"] == {"1": ["a", "b"]}
        assert first["rows_changed"] == {"1": [2, 3]}
        assert first["is_revision"] is True
        assert first["live_revision_schema_version"] == LIVE_REVISION_SCHEMA_VERSION

    def test_the_diff_is_frozen(self) -> None:
        import dataclasses

        assert dataclasses.is_dataclass(WeekDiff)
        diff = compare_week_digests(digests_one_week(), digests_one_week())
        try:
            diff.weeks_changed = ("1",)
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("WeekDiff must be a frozen dataclass")


class TestTheDiffNeverMutatesItsInputs:
    """The ruling is pure: the two maps it was handed come back untouched."""

    def test_neither_map_is_mutated(self) -> None:
        import copy

        prior = {"1": bucket(2, "f1", a="a1")}
        current = {"1": bucket(3, "f2", a="a2"), "2": bucket(1, "f3")}
        prior_before = copy.deepcopy(prior)
        current_before = copy.deepcopy(current)

        compare_week_digests(prior, current)

        assert prior == prior_before
        assert current == current_before


# ---------------------------------------------------------------------------
# Task 2: the severity ruling, the graded escalation, and the OWED correction.
# ---------------------------------------------------------------------------

# The FOURTEEN keys every verdict carries. Written out as literals rather than imported
# from the module, so a key SILENTLY renamed there fails here. A verdict rides inside a
# committed capture entry for the whole season; a shape that depends on which branch
# produced it is not a shape a reader can diff a season later.
EXPECTED_VERDICT_KEYS: frozenset[str] = frozenset(
    {
        "verdict_schema_version",
        "live_revision_schema_version",
        "dataset",
        "season",
        "week",
        "sequence",
        "event_class",
        "severity",
        "diff",
        "correction_owed",
        "correction_owed_scope",
        "graded_weeks",
        "graded_weeks_source",
        "reason",
    }
)


def three_weeks(**overrides: dict) -> dict:
    """A three-week digest map; ``overrides`` replaces whole buckets by week key."""
    base = {
        "1": bucket(10, "f1", epa="e1", wp="w1"),
        "2": bucket(11, "f2", epa="e2", wp="w2"),
        "3": bucket(12, "f3", epa="e3", wp="w3"),
    }
    base.update(overrides)
    return base


def capture_entry(
    week: int = 6, sequence: int = 1, digests: dict | None = None
) -> dict:
    """One capture entry in the shape ``upstream_live.build_capture_entry`` produces."""
    return {
        "week": week,
        "sequence": sequence,
        "captured_at_utc": "2026-09-11T18:00:00+00:00",
        "path": "bronze/upstream_live/pbp_2026_w6.parquet",
        "sha256": "0" * 64,
        "bytes": 1234,
        "rows": 2,
        "columns": ["game_id", "week", "epa"],
        "upstream_width": 372,
        "week_label_means": WEEK_LABEL_MEANS,
        "week_label_semantics_version": WEEK_LABEL_SEMANTICS_VERSION,
        "content_through_week": 6,
        "weeks_present": [1, 2, 3],
        "week_digest_schema_version": WEEK_DIGEST_SCHEMA_VERSION,
        "week_partition": WEEK_PARTITION_PER_WEEK,
        "week_digests": digests if digests is not None else three_weeks(),
    }


def graded(weeks: list[int], reason: str | None = None) -> dict:
    """The record ``data.graded_weeks.graded_weeks_record`` returns."""
    return {
        "season": 2026,
        "weeks": sorted(weeks),
        "source": GRADED_WEEKS_SOURCE,
        "resolved": True,
        "reason": reason,
    }


def rule(
    *,
    current: dict | None = None,
    prior: dict | None = None,
    graded_record: object = None,
) -> dict:
    """Call the detector with this module's fixtures, keeping each test to one call."""
    return detect_live_revision(
        dataset="pbp",
        season=2026,
        current_entry=capture_entry(digests=current),
        prior_entry=None if prior is None else capture_entry(digests=prior),
        graded=graded_record,
    )


def moved_week(key: str) -> dict:
    """``three_weeks`` with one week's bucket restated -- the ordinary revision shape."""
    restated = {
        "1": bucket(10, "f1-moved", epa="e1x", wp="w1"),
        "2": bucket(11, "f2-moved", epa="e2x", wp="w2"),
        "3": bucket(12, "f3-moved", epa="e3x", wp="w3"),
    }
    return three_weeks(**{key: restated[key]})


class TestTheFirstCaptureSaysSoRatherThanSayingClean:
    """CONTEXT's open discretion item: an explicit no_prior_capture, never a clean one."""

    def test_no_prior_capture_is_its_own_event_class(self) -> None:
        verdict = rule(current=three_weeks(), prior=None, graded_record=graded([1]))

        assert verdict["event_class"] == RevisionEventClass.NO_PRIOR_CAPTURE
        assert verdict["event_class"] != RevisionEventClass.CLEAN

    def test_no_prior_capture_is_informational_and_carries_a_reason(self) -> None:
        verdict = rule(current=three_weeks(), prior=None, graded_record=graded([1]))

        assert verdict["severity"] == RevisionSeverity.INFORMATIONAL
        assert "no previous capture" in verdict["reason"].lower()

    def test_no_prior_capture_records_no_diff_and_owes_nothing(self) -> None:
        verdict = rule(current=three_weeks(), prior=None, graded_record=graded([1]))

        assert verdict["diff"] is None
        assert verdict["correction_owed"] is False
        assert verdict["correction_owed_scope"] is None


class TestAComparedAndUnchangedPairIsClean:
    def test_an_identical_pair_is_clean_and_owes_nothing(self) -> None:
        verdict = rule(
            current=three_weeks(), prior=three_weeks(), graded_record=graded([1, 2])
        )

        assert verdict["event_class"] == RevisionEventClass.CLEAN
        assert verdict["severity"] == RevisionSeverity.INFORMATIONAL
        assert verdict["correction_owed"] is False
        assert verdict["correction_owed_scope"] is None

    def test_a_clean_verdict_still_says_where_it_looked(self) -> None:
        verdict = rule(
            current=three_weeks(), prior=three_weeks(), graded_record=graded([1, 2])
        )

        assert verdict["graded_weeks"] == [1, 2]
        assert verdict["graded_weeks_source"] == GRADED_WEEKS_SOURCE

    def test_growth_only_is_clean_not_a_live_revision(self) -> None:
        current = three_weeks()
        current["4"] = bucket(13, "f4", epa="e4", wp="w4")

        verdict = rule(
            current=current, prior=three_weeks(), graded_record=graded([1, 2])
        )

        assert verdict["event_class"] == RevisionEventClass.CLEAN
        assert verdict["diff"]["weeks_added"] == ["4"]


class TestAnUngradedRevisionIsTheOrdinaryCase:
    def test_a_changed_ungraded_week_is_live_revision_and_owes_nothing(self) -> None:
        verdict = rule(
            current=moved_week("3"), prior=three_weeks(), graded_record=graded([1, 2])
        )

        assert verdict["event_class"] == RevisionEventClass.LIVE_REVISION
        assert verdict["correction_owed"] is False
        assert verdict["correction_owed_scope"] is None

    def test_the_ungraded_case_is_informational_so_it_can_fire_every_week(self) -> None:
        verdict = rule(
            current=moved_week("3"), prior=three_weeks(), graded_record=graded([1, 2])
        )

        assert verdict["severity"] == RevisionSeverity.INFORMATIONAL


class TestAGradedRevisionEscalates:
    """PIN-03's whole point, and D32-12's 'the graded case escalates' as a NUMBER."""

    def test_a_changed_graded_week_is_live_revision_graded(self) -> None:
        verdict = rule(
            current=moved_week("2"), prior=three_weeks(), graded_record=graded([1, 2])
        )

        assert verdict["event_class"] == RevisionEventClass.LIVE_REVISION_GRADED
        assert verdict["correction_owed"] is True

    def test_the_owed_scope_is_queryable_data_not_prose(self) -> None:
        current = three_weeks(
            **{
                "1": bucket(10, "f1-moved", epa="e1x", wp="w1"),
                "2": bucket(11, "f2-moved", epa="e2x", wp="w2"),
            }
        )

        verdict = rule(
            current=current, prior=three_weeks(), graded_record=graded([2, 1])
        )

        scope = verdict["correction_owed_scope"]
        assert scope["season"] == 2026
        assert scope["weeks"] == [1, 2]
        assert all(isinstance(week, int) for week in scope["weeks"])
        assert scope["discharged_by"] == "phase-34 ledger correction block"
        assert isinstance(scope["reason"], str)
        assert scope["reason"]

    def test_the_escalation_is_a_number_over_the_same_underlying_diff(self) -> None:
        graded_verdict = rule(
            current=moved_week("2"), prior=three_weeks(), graded_record=graded([2])
        )
        ungraded_verdict = rule(
            current=moved_week("2"), prior=three_weeks(), graded_record=graded([1])
        )

        assert graded_verdict["diff"] == ungraded_verdict["diff"]
        assert severity_rank(
            RevisionSeverity(graded_verdict["severity"])
        ) > severity_rank(RevisionSeverity(ungraded_verdict["severity"]))

    def test_a_removed_graded_week_escalates_exactly_as_a_changed_one_does(
        self,
    ) -> None:
        prior = three_weeks()
        current = {"1": prior["1"], "3": prior["3"]}

        verdict = rule(current=current, prior=prior, graded_record=graded([2]))

        assert verdict["event_class"] == RevisionEventClass.LIVE_REVISION_GRADED
        assert verdict["correction_owed"] is True
        assert verdict["correction_owed_scope"]["weeks"] == [2]

    def test_the_correction_owed_keys_are_the_committed_ones(self) -> None:
        verdict = rule(
            current=moved_week("2"), prior=three_weeks(), graded_record=graded([2])
        )

        assert CORRECTION_OWED in verdict
        assert CORRECTION_OWED_SCOPE in verdict


class TestAnEmptyGradedSetAndAnUnresolvableOneAreDifferentObservables:
    """The D32-11 seam's whole reason for existing, asserted rather than assumed."""

    def test_an_explicitly_recorded_empty_set_yields_the_ordinary_live_revision(
        self,
    ) -> None:
        verdict = rule(
            current=moved_week("2"),
            prior=three_weeks(),
            graded_record=graded([], reason="no bet list at 'outputs/bet_list/...'"),
        )

        assert verdict["event_class"] == RevisionEventClass.LIVE_REVISION
        assert verdict["correction_owed"] is False
        assert verdict["graded_weeks"] == []
        assert verdict["graded_weeks_source"] == GRADED_WEEKS_SOURCE

    def test_an_unresolvable_set_is_unknown_and_owes_exactly_none(self) -> None:
        failure = GradedWeeksUnavailable(
            "the bet list at 'x.parquet' could not be read"
        )

        verdict = rule(
            current=moved_week("2"), prior=three_weeks(), graded_record=failure
        )

        assert verdict["event_class"] == RevisionEventClass.UNKNOWN
        assert verdict["correction_owed"] is None

    def test_the_unknown_verdict_is_louder_than_informational(self) -> None:
        verdict = rule(
            current=three_weeks(),
            prior=three_weeks(),
            graded_record=GradedWeeksUnavailable("unreadable"),
        )

        assert severity_rank(RevisionSeverity(verdict["severity"])) > severity_rank(
            RevisionSeverity.INFORMATIONAL
        )

    def test_the_unknown_verdict_carries_the_underlying_failure_text(self) -> None:
        failure = GradedWeeksUnavailable(
            "the bet list at 'x.parquet' could not be read"
        )

        verdict = rule(
            current=three_weeks(), prior=three_weeks(), graded_record=failure
        )

        assert "could not be read" in verdict["reason"]
        assert verdict["graded_weeks"] is None

    def test_an_unresolved_set_is_ruled_before_the_diff_so_a_clean_pair_is_not_quiet(
        self,
    ) -> None:
        verdict = rule(
            current=three_weeks(),
            prior=three_weeks(),
            graded_record=GradedWeeksUnavailable("unreadable"),
        )

        assert verdict["event_class"] == RevisionEventClass.UNKNOWN

    def test_a_missing_graded_argument_is_unresolved_not_an_empty_set(self) -> None:
        verdict = rule(current=moved_week("2"), prior=three_weeks(), graded_record=None)

        assert verdict["event_class"] == RevisionEventClass.UNKNOWN
        assert verdict["correction_owed"] is None


class TestTwoEntriesDigestedUnderDifferentRulesAreNotComparable:
    """WR-06. ``week_digest_schema_version`` was written for readers and never read.

    The stamp exists precisely because "changing it mid-season would mean earlier revision
    verdicts were computed under a different rule, so the season's revision history would
    stop being comparable end to end" -- yet ``detect_live_revision`` read both entries'
    ``week_digests`` and compared them without ever looking at either version.

    The consequence of the omission is not a missed finding, it is a FABRICATED one: on the
    first capture after any digest-shape bump, every shared week's digest differs because
    the RULE changed rather than because the bytes did. The detector would report a
    whole-season revision and escalate to a CRITICAL carrying ``correction_owed`` True for
    every graded week -- an obligation Phase 34 would then be asked to discharge against
    data that never moved.
    """

    @staticmethod
    def _ruled(prior_version: object, current_version: object) -> dict:
        """Rule an OTHERWISE-IDENTICAL pair whose two digest-schema stamps differ."""
        prior = capture_entry(digests=three_weeks())
        current = capture_entry(digests=three_weeks())
        prior["week_digest_schema_version"] = prior_version
        current["week_digest_schema_version"] = current_version
        return detect_live_revision(
            dataset="pbp",
            season=2026,
            current_entry=current,
            prior_entry=prior,
            graded=graded([1, 2, 3]),
        )

    def test_a_version_mismatch_is_unknown_and_owes_exactly_none(self) -> None:
        verdict = self._ruled(1, 2)

        assert verdict["event_class"] == RevisionEventClass.UNKNOWN
        assert verdict["correction_owed"] is None, (
            "a ruling that cannot compare its two inputs asserted something about the "
            "obligation. None is the claim 'undecided'; False would assert nothing is "
            "owed, which this branch cannot know."
        )
        assert verdict["correction_owed_scope"] is None

    def test_the_reason_names_both_versions_so_a_reader_can_see_the_bump(self) -> None:
        reason = self._ruled(1, 2)["reason"]

        assert "week-digest schema" in reason, (
            f"the reason does not name the cause:\n{reason}"
        )
        assert "1" in reason and "2" in reason, (
            f"the reason does not name BOTH versions:\n{reason}"
        )

    def test_the_diff_is_recorded_but_not_ruled_on(self) -> None:
        """Recording it costs nothing and a later reader may want it.

        What the branch refuses to do is draw a severity from it -- the same shape the
        unresolved-graded branch above uses, for the same reason.
        """
        verdict = self._ruled(1, 2)

        assert verdict["diff"] is not None, (
            "the diff was discarded; it cost nothing to compute and is the only evidence "
            "a later reader has of what the incomparable maps actually contained"
        )
        assert verdict["diff"]["live_revision_schema_version"] is not None

    def test_a_moved_graded_week_does_NOT_escalate_across_a_version_bump(self) -> None:
        """The whole point: the bump must not manufacture a CRITICAL.

        Under the pre-fix code this identical input reported ``live_revision_graded`` with
        ``correction_owed`` True, because the two maps disagree on every week.
        """
        prior = capture_entry(digests=three_weeks())
        current = capture_entry(digests=moved_week("2"))
        prior["week_digest_schema_version"] = 1
        current["week_digest_schema_version"] = 2

        verdict = detect_live_revision(
            dataset="pbp",
            season=2026,
            current_entry=current,
            prior_entry=prior,
            graded=graded([1, 2, 3]),
        )

        assert verdict["event_class"] != RevisionEventClass.LIVE_REVISION_GRADED, (
            "a digest-shape bump manufactured a graded revision and an obligation on a "
            "week whose bytes may not have moved at all"
        )
        assert verdict["event_class"] == RevisionEventClass.UNKNOWN
        assert verdict["correction_owed"] is None

    def test_matching_versions_still_rule_normally(self) -> None:
        """The guard must not swallow the ordinary case it sits in front of."""
        assert self._ruled(1, 1)["event_class"] == RevisionEventClass.CLEAN

    def test_two_entries_that_both_predate_the_stamp_still_compare(self) -> None:
        """Two absent stamps are EQUAL, so a pre-stamp pair is not made incomparable."""
        assert self._ruled(None, None)["event_class"] == RevisionEventClass.CLEAN

    def test_an_entry_gaining_the_stamp_is_a_mismatch(self) -> None:
        """Absent on one side and present on the other is exactly the bump shape."""
        assert self._ruled(None, 1)["event_class"] == RevisionEventClass.UNKNOWN

    def test_the_verdict_keeps_the_frozen_key_set(self) -> None:
        assert set(self._ruled(1, 2)) == EXPECTED_VERDICT_KEYS


class TestAnUnattributableBucketIsNeverTheQuietAnswer:
    """A whole-frame dataset cannot say WHICH week moved, so it must not claim none did."""

    def test_a_moved_whole_frame_bucket_with_graded_weeks_is_unknown(self) -> None:
        prior = {NO_WEEK_COLUMN_BUCKET: bucket(40, "f1", pos="p1")}
        current = {NO_WEEK_COLUMN_BUCKET: bucket(40, "f2", pos="p2")}

        verdict = rule(current=current, prior=prior, graded_record=graded([1, 2]))

        assert verdict["event_class"] == RevisionEventClass.UNKNOWN
        assert verdict["correction_owed"] is None
        assert NO_WEEK_COLUMN_BUCKET in verdict["reason"]

    def test_a_moved_whole_frame_bucket_with_nothing_graded_is_the_ordinary_case(
        self,
    ) -> None:
        prior = {NO_WEEK_COLUMN_BUCKET: bucket(40, "f1", pos="p1")}
        current = {NO_WEEK_COLUMN_BUCKET: bucket(40, "f2", pos="p2")}

        verdict = rule(current=current, prior=prior, graded_record=graded([]))

        assert verdict["event_class"] == RevisionEventClass.LIVE_REVISION
        assert verdict["correction_owed"] is False

    def test_a_definite_graded_week_outranks_the_unattributable_bucket(self) -> None:
        prior = {
            NO_WEEK_COLUMN_BUCKET: bucket(40, "f1", pos="p1"),
            "2": bucket(11, "f2", epa="e2"),
        }
        current = {
            NO_WEEK_COLUMN_BUCKET: bucket(40, "f2", pos="p2"),
            "2": bucket(11, "f2-moved", epa="e2x"),
        }

        verdict = rule(current=current, prior=prior, graded_record=graded([2]))

        assert verdict["event_class"] == RevisionEventClass.LIVE_REVISION_GRADED
        assert verdict["correction_owed"] is True


class TestTheVerdictShapeIsFrozenAndTheRulingStaysPure:
    def test_every_verdict_carries_exactly_the_fourteen_keys(self) -> None:
        cases = [
            rule(current=three_weeks(), prior=None, graded_record=graded([1])),
            rule(current=three_weeks(), prior=three_weeks(), graded_record=graded([1])),
            rule(
                current=moved_week("3"), prior=three_weeks(), graded_record=graded([1])
            ),
            rule(
                current=moved_week("1"), prior=three_weeks(), graded_record=graded([1])
            ),
            rule(
                current=three_weeks(),
                prior=three_weeks(),
                graded_record=GradedWeeksUnavailable("unreadable"),
            ),
        ]

        for verdict in cases:
            assert set(verdict) == EXPECTED_VERDICT_KEYS

    def test_the_module_publishes_the_key_set_it_writes(self) -> None:
        assert set(VERDICT_KEYS) == EXPECTED_VERDICT_KEYS
        assert len(VERDICT_KEYS) == len(EXPECTED_VERDICT_KEYS)

    def test_the_verdict_stamps_both_schema_versions_and_its_own_identity(self) -> None:
        verdict = rule(
            current=three_weeks(), prior=three_weeks(), graded_record=graded([1])
        )

        assert verdict["verdict_schema_version"] == VERDICT_SCHEMA_VERSION
        assert verdict["live_revision_schema_version"] == LIVE_REVISION_SCHEMA_VERSION
        assert verdict["dataset"] == "pbp"
        assert verdict["season"] == 2026
        assert verdict["week"] == 6
        assert verdict["sequence"] == 1

    def test_the_verdict_is_json_serialisable(self) -> None:
        import json

        verdict = rule(
            current=moved_week("1"), prior=three_weeks(), graded_record=graded([1])
        )

        assert json.loads(json.dumps(verdict))["event_class"] == "live_revision_graded"

    def test_neither_entry_is_mutated(self) -> None:
        import copy

        current_entry = capture_entry(digests=moved_week("1"))
        prior_entry = capture_entry(digests=three_weeks())
        current_before = copy.deepcopy(current_entry)
        prior_before = copy.deepcopy(prior_entry)

        detect_live_revision(
            dataset="pbp",
            season=2026,
            current_entry=current_entry,
            prior_entry=prior_entry,
            graded=graded([1]),
        )

        assert current_entry == current_before
        assert prior_entry == prior_before


class TestTheSeverityComesOnlyFromTheCommittedTable:
    """T-32-34: a hand-assigned severity drifting out of the frozen vocabulary."""

    @staticmethod
    def _source() -> str:
        return LIVE_REVISION_SOURCE.read_text(encoding="utf-8")

    def test_no_literal_severity_string_appears_in_the_module(self) -> None:
        import ast

        tree = ast.parse(self._source())
        literals = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in {"critical", "warning", "informational"}
        ]

        assert literals == []

    def test_the_module_uses_the_committed_table_and_the_rank_function(self) -> None:
        source = self._source()

        assert "DEFAULT_SEVERITY" in source
        assert "severity_rank" in source

    def test_the_module_imports_no_reader_and_no_higher_layer(self) -> None:
        import ast

        tree = ast.parse(self._source())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        assert not any(name.startswith("backtest") for name in imported)
        assert not any(name.startswith("api") for name in imported)
        assert not any(name.startswith("data.graded_weeks") for name in imported)


class TestTheOnlySanctionedSeverityComparison:
    """Why a helper exists at all: alphabetical order is not loudness order."""

    def test_a_naive_string_comparison_would_get_the_order_exactly_backwards(
        self,
    ) -> None:
        # RevisionSeverity is a StrEnum, so comparing two MEMBERS with < is legal and
        # silently alphabetical: the loudest severity sorts below the quietest one.
        assert RevisionSeverity.CRITICAL < RevisionSeverity.INFORMATIONAL
        assert severity_rank(RevisionSeverity.CRITICAL) > severity_rank(
            RevisionSeverity.INFORMATIONAL
        )

    def test_the_helper_ranks_the_graded_case_above_the_ungraded_one(self) -> None:
        graded_verdict = rule(
            current=moved_week("2"), prior=three_weeks(), graded_record=graded([2])
        )
        ungraded_verdict = rule(
            current=moved_week("2"), prior=three_weeks(), graded_record=graded([1])
        )

        assert verdict_severity_rank(graded_verdict) > verdict_severity_rank(
            ungraded_verdict
        )

    def test_the_helper_ranks_unknown_above_the_ordinary_case(self) -> None:
        unknown_verdict = rule(
            current=three_weeks(),
            prior=three_weeks(),
            graded_record=GradedWeeksUnavailable("unreadable"),
        )
        ordinary_verdict = rule(
            current=moved_week("3"), prior=three_weeks(), graded_record=graded([1])
        )

        assert verdict_severity_rank(unknown_verdict) > verdict_severity_rank(
            ordinary_verdict
        )

    def test_a_severity_outside_the_frozen_vocabulary_raises_rather_than_sorts(
        self,
    ) -> None:
        with pytest.raises(ValueError):
            verdict_severity_rank({"severity": "loud"})
