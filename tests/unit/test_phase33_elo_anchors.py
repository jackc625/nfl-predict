"""The five re-derived Elo artifacts are pinned to committed content digests.

WHY THE WITNESS LIVES OUTSIDE THE WITNESSED FILES
-------------------------------------------------
``tests/phase33_state.ELO_ARTIFACT_DIGESTS`` holds the sha256 of each artifact,
in a git-TRACKED module that is not one of them. That separation is the reason the
anchor means anything: a file required to contain its own whole-file hash is
self-referential -- writing the hash changes the bytes it was taken over -- and the
only way out is a canonical exclusion rule invented under pressure to make a
failing assertion pass. ``tests/unit/test_preregistration_ancestry.py`` records the
same reasoning for the Phase-31 pre-registration; this module is the Elo instance
of it.

NO NEWLINE NORMALIZATION HERE, and that is not an oversight. The ancestry module
normalizes because it hashes TRACKED TEXT files and this repository has
``core.autocrlf=true``, so a raw-byte digest of a text file pins a value that holds
only on the machine that measured it. Four of these five artifacts are binary
parquet and the fifth is a gitignored JSON under ``data/``: none is ever
line-ending-translated by git, because none of them is in git at all. Normalizing
them would corrupt the parquet bytes.

AN UNDECIDED COMPARISON IS NOT A PASS AND NOT A FAILURE-TO-MATCH
----------------------------------------------------------------
``tests.data_boundary.digest_file`` falls back to a self-declaring
``stat-size-mtime:`` signature when a file cannot be opened -- which is exactly
what happens to ``data/nfl_predictions.duckdb`` while a read-write handle is held.
A signature on one side of a comparison and a content hash on the other disagree as
raw strings while saying NOTHING about the bytes. Every comparison below therefore
consults ``is_stat_signature`` first and fails BY NAME on a mixed pair, rather than
reporting a data move that may not have happened or, worse, passing.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.data_boundary import digest_file, is_stat_signature
from tests.phase33_state import (
    ELO_ARTIFACT_DIGESTS,
    ELO_FORENSIC_COPY_DIGESTS,
    ELO_FORENSIC_COPY_ROOT,
    ELO_PRE_PHASE_STATE,
    ELO_RATING_BAND_FROZEN,
    ELO_RATING_BAND_OBSERVED,
    ELO_RATING_BAND_PRE_RUN_POOLED,
    ELO_REDERIVATION_EXPECTED_CHANGED_FILES,
)

# The Elo generation is five artifacts, and the anchor set is judged as a SET:
# anchoring four of five would leave the fifth free to move inside a "verified"
# generation.
EXPECTED_ARTIFACT_COUNT = 5

_MISSING_ARTIFACT_REASON = (
    "the artifact is not present at {path} -- data/ is gitignored, so a checkout "
    "that has not built the lake has no bytes to anchor. This is a fact about the "
    "checkout, not about the data."
)


def _skip_if_absent(path: Path) -> None:
    if not path.is_file():
        pytest.skip(_MISSING_ARTIFACT_REASON.format(path=path.as_posix()))


@pytest.mark.parametrize(
    ("relative_path", "expected_digest"),
    ELO_ARTIFACT_DIGESTS,
    ids=[Path(name).name for name, _digest in ELO_ARTIFACT_DIGESTS],
)
def test_each_elo_artifact_matches_its_committed_anchor(relative_path, expected_digest):
    """The artifact on disk hashes to the value committed after the re-derivation."""
    path = Path(relative_path)
    _skip_if_absent(path)

    actual = digest_file(path)

    assert not is_stat_signature(actual) and not is_stat_signature(expected_digest), (
        f"UNDECIDED comparison for {relative_path}: one side is the locked-file "
        f"stat signature and the other a content hash (anchor={expected_digest}, "
        f"observed={actual}). These two values cannot be compared -- they say "
        "nothing about the bytes -- so this is reported by name rather than as a "
        "digest move and never as a pass. Close whatever holds the file open "
        "(usually a DuckDB connection from an earlier run) and re-run."
    )
    assert actual == expected_digest, (
        f"{relative_path} no longer matches its committed anchor.\n"
        f"  anchor   sha256 {expected_digest}\n"
        f"  on disk  sha256 {actual}\n"
        "The five Elo artifacts were re-derived once, under an owner ruling, with "
        "the changed-file set declared beforehand. A move since then was not part "
        "of that ruling: find what wrote it before re-anchoring anything."
    )


def test_all_five_artifacts_are_anchored():
    """Four of five is not a pinned generation."""
    assert len(ELO_ARTIFACT_DIGESTS) == EXPECTED_ARTIFACT_COUNT, (
        f"{len(ELO_ARTIFACT_DIGESTS)} artifact(s) are anchored, not "
        f"{EXPECTED_ARTIFACT_COUNT}. The publisher stages and validates five "
        "together; anchoring a subset leaves the rest free to move inside a "
        "generation that still reads as verified."
    )
    names = [name for name, _digest in ELO_ARTIFACT_DIGESTS]
    assert len(set(names)) == EXPECTED_ARTIFACT_COUNT, (
        f"the anchor set names a path twice: {names}"
    )
    for name in names:
        assert name in ELO_REDERIVATION_EXPECTED_CHANGED_FILES, (
            f"{name} is anchored but was never declared as a file the "
            "re-derivation would move, so one of the two is wrong."
        )


def test_every_anchor_is_a_content_hash_and_not_a_stat_signature():
    """A committed anchor that is a stat signature would pin metadata, not bytes."""
    degraded = [
        name for name, digest in ELO_ARTIFACT_DIGESTS if is_stat_signature(digest)
    ]
    assert not degraded, (
        f"these anchors were recorded as stat signatures rather than content "
        f"hashes: {degraded}. Size and mtime cannot show that content is "
        "unchanged (D33-32); an anchor taken that way pins nothing."
    )
    for name, digest in ELO_ARTIFACT_DIGESTS:
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef"), (
            f"the anchor for {name} is not a lowercase hex sha256: {digest!r}"
        )


def test_a_one_byte_change_moves_the_digest(tmp_path):
    """The CONTROL: prove the anchor is capable of catching a change at all.

    Without this, a green anchor test is consistent with a digest function that
    returns a constant. The flip is done on a COPY under ``tmp_path``; the
    production artifact is never written.
    """
    source = Path(ELO_ARTIFACT_DIGESTS[0][0])
    _skip_if_absent(source)

    original = source.read_bytes()
    assert len(original) > 0, f"{source.as_posix()} is empty"

    intact = tmp_path / "intact.bin"
    intact.write_bytes(original)
    intact_digest = digest_file(intact)
    assert intact_digest == ELO_ARTIFACT_DIGESTS[0][1], (
        "an untouched copy of the artifact does not reproduce its anchor, so the "
        "control cannot distinguish a flipped byte from a broken instrument."
    )

    flipped_bytes = bytearray(original)
    index = len(flipped_bytes) // 2
    flipped_bytes[index] ^= 0x01
    flipped = tmp_path / "flipped.bin"
    flipped.write_bytes(bytes(flipped_bytes))

    flipped_digest = digest_file(flipped)
    assert flipped_digest != intact_digest, (
        f"flipping one bit at offset {index} of a {len(original)}-byte copy did "
        "NOT move the digest. The anchors above would then be incapable of "
        "catching any change, and every one of them is decoration."
    )
    assert len(flipped_bytes) == len(original), (
        "the control changed the file's SIZE as well as its content, so it would "
        "also have been caught by a stat signature and proves less than intended."
    )


# ---------------------------------------------------------------------------
# The frozen band is not the observed band. This module may read both; the
# integration module deliberately reads only the frozen one.
# ---------------------------------------------------------------------------


def test_the_frozen_band_was_not_taken_from_the_run_it_judges():
    """Frozen and observed must be DIFFERENT pairs, or the freeze did not happen.

    If the two were identical, the "frozen" band would in fact have been recorded
    from the run's own output, and asserting a run's values fall inside a band
    derived from that same run is true by construction for any output whatever.
    """
    assert ELO_RATING_BAND_FROZEN != ELO_RATING_BAND_OBSERVED, (
        f"the frozen band {ELO_RATING_BAND_FROZEN} is identical to the band the "
        "run itself produced. That would mean the band was measured AFTER the "
        "re-derivation, in which case it asserts only that a maximum is not below "
        "a minimum."
    )


def test_the_frozen_band_strictly_contains_both_the_pre_run_and_post_run_ranges():
    """The freeze widened the PRE-run range, and the run landed inside it."""
    low, high = ELO_RATING_BAND_FROZEN
    pre_low, pre_high = ELO_RATING_BAND_PRE_RUN_POOLED
    post_low, post_high = ELO_RATING_BAND_OBSERVED

    assert low < pre_low and high > pre_high, (
        f"the frozen band {ELO_RATING_BAND_FROZEN} does not strictly contain the "
        f"pre-run pooled range {ELO_RATING_BAND_PRE_RUN_POOLED} it was widened "
        "from."
    )
    assert low <= post_low and high >= post_high, (
        f"the re-derived chain's observed range {ELO_RATING_BAND_OBSERVED} falls "
        f"outside the band {ELO_RATING_BAND_FROZEN} frozen before the run. That "
        "is a FINDING about the re-derivation; it is not a reason to widen the "
        "band."
    )


# ---------------------------------------------------------------------------
# The rollback artifact -- taken before the run, outside data/, and still intact.
# ---------------------------------------------------------------------------


def test_the_forensic_copy_sits_outside_the_guarded_data_root():
    """A copy inside data/ would be a guarded store and a second source of truth."""
    root = Path(ELO_FORENSIC_COPY_ROOT).as_posix().rstrip("/") + "/"
    assert not root.startswith("data/") and "/data/" not in root, (
        f"the forensic copy root {ELO_FORENSIC_COPY_ROOT} is inside the guarded "
        "data root. It was placed outside deliberately: a copy under data/ is "
        "itself a boundary-tracked production store and becomes a second source "
        "of truth inside the lake."
    )


def test_the_forensic_copy_records_all_five_prior_artifacts():
    """Five of five, matching the pre-run state this plan recorded."""
    assert len(ELO_FORENSIC_COPY_DIGESTS) == EXPECTED_ARTIFACT_COUNT, (
        f"{len(ELO_FORENSIC_COPY_DIGESTS)} forensic digest(s) recorded, not "
        f"{EXPECTED_ARTIFACT_COUNT}. A copy of some of a mixed generation could "
        "not reconstruct it."
    )
    recorded = {name for name, _digest in ELO_FORENSIC_COPY_DIGESTS}
    from_pre_state = {
        Path(name).name
        for name in ELO_REDERIVATION_EXPECTED_CHANGED_FILES
        if name.startswith("data/silver/elo") or name.endswith("games_with_elo.parquet")
    }
    assert recorded == from_pre_state - {"elo_generation.json"}, (
        "the forensic copy does not cover exactly the five artifacts the "
        f"re-derivation replaced: copied {sorted(recorded)}, replaced "
        f"{sorted(from_pre_state)}"
    )


@pytest.mark.parametrize(
    ("filename", "expected_digest"),
    ELO_FORENSIC_COPY_DIGESTS,
    ids=[name for name, _digest in ELO_FORENSIC_COPY_DIGESTS],
)
def test_the_forensic_copy_still_holds_the_pre_run_bytes(filename, expected_digest):
    """The rollback artifact is only an undo while its bytes are the OLD ones."""
    path = Path(ELO_FORENSIC_COPY_ROOT) / filename
    if not path.is_file():
        pytest.skip(
            f"the forensic copy is not present at {path.as_posix()} -- outputs/ "
            "is gitignored, so it does not travel with a fresh checkout."
        )

    actual = digest_file(path)
    assert not is_stat_signature(actual), (
        f"UNDECIDED comparison for {path.as_posix()}: the copy could not be read "
        "as content and degraded to a stat signature."
    )
    assert actual == expected_digest, (
        f"{path.as_posix()} no longer holds the bytes it was copied with.\n"
        f"  recorded sha256 {expected_digest}\n"
        f"  on disk  sha256 {actual}\n"
        "The copy is read-only and nothing downstream reads it, so a move here "
        "means something outside this plan wrote into the forensic directory."
    )


def test_the_forensic_digests_match_the_recorded_pre_run_state():
    """The rollback copy and the pre-run record are the same five files.

    Two independent records of the same instant; if they disagreed, one of them
    would be describing a store that never existed.
    """
    copied = dict(ELO_FORENSIC_COPY_DIGESTS)
    for key, recorded in ELO_PRE_PHASE_STATE.items():
        if not isinstance(recorded, dict) or "sha256" not in recorded:
            continue
        filename = f"{key}.json" if key == "elo_ratings" else f"{key}.parquet"
        assert copied.get(filename) == recorded["sha256"], (
            f"the forensic copy of {filename} carries "
            f"{copied.get(filename)} while ELO_PRE_PHASE_STATE records "
            f"{recorded['sha256']} for the same file at the same instant."
        )
