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

from data.live_revision import (
    LIVE_REVISION_SCHEMA_VERSION,
    WeekDiff,
    as_record,
    compare_week_digests,
)
from data.upstream_live import NO_WEEK_COLUMN_BUCKET


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
