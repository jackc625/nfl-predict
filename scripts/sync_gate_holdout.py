"""Regenerate EXACTLY ``config/gate.toml``'s ``gate.seasons.holdout`` line, and nothing else.

WHY THIS SCRIPT EXISTS (Plan 33.1-09, Ruling S2 -- Codex 33.1-09 MEDIUM)
------------------------------------------------------------------------
Ten of the eleven sites that define the season partition can ``import conf.season_partition``
and compute. ``config/gate.toml`` cannot: it is TOML, its ``gate.seasons.holdout`` is a literal
list parsed by ``tomllib``, and no mechanism exists by which a TOML file could derive anything
from a Python rule.

So the honest description is that this site is a **GENERATED / PINNED MIRROR**, not a
derivation -- and ``tests/phase33_state.SEASON_PARTITION_SITES`` records that per-site
``mechanism`` so the overstatement is not repeated somewhere else. Calling all eleven sites
"deriving" would have been an overstated mechanism in a milestone whose whole point is not
overstating mechanisms.

A mirror needs two things to stay honest: something that REGENERATES it (this script) and
something that CATCHES it when it is stale (``tests/unit/test_season_partition_agreement.py``,
which EVALUATES the parsed TOML value against the rule -- so a stale mirror fails whether or
not anybody ever runs this command).

WHY A LINE EDIT AND NOT A PARSE-AND-DUMP -- THIS IS LOAD-BEARING
-----------------------------------------------------------------
The obvious implementation is ``tomllib.load`` / ``tomli_w.dump``. It would destroy the file.

``config/gate.toml``'s ``[baseline.*]`` block must stay BYTE-IDENTICAL (D33-11, D33.1-05). Its
own header requires it to be a verbatim generator paste, it carries an UNDISCHARGED disclosure
-- the frozen values diverge from a re-score in 47 of 68 fields -- and two DELIBERATE tripwires
are RED because of that divergence. A TOML serializer round-trip reformats the whole file:
comment placement, float formatting, key ordering, inline-table style. Every one of those is a
byte, and the block's digest (``tests/phase33_state.GATE_TOML_BASELINE_SHA256``) is taken over
bytes. Re-serializing would turn two deliberate tripwires green by accident -- clearing a
disclosure by reformatting it.

So this edits ONE LINE, in place, on raw bytes, and REFUSES if any other byte moved. Line
endings are preserved exactly (the file is CRLF in this Windows working tree and LF in the git
blob), which is why it works on bytes rather than on decoded text.

HOW THE REFUSAL IS ACTUALLY VERIFIED, stated precisely because it used to be stated wrongly
(code review WR-05). This paragraph claimed the guarantee was "a digest over the whole file
with that one line masked out: identical before and after". That check existed and COULD NOT
FAIL: it masked the same index on both sides of a one-element replacement, so the comparison
was between a value and itself. It read as a guarantee and behaved as a comment. There are now
two checks, and the difference between them is stated rather than blurred:

  * BEFORE the write, the reconstructed bytes are digested against the bytes as read. This
    catches a broken split/replace/join round trip. It proves nothing about the filesystem.
  * AFTER the write, the file is READ BACK from disk and digested again. THIS is the
    guarantee: two independently-produced byte strings, so it can fail. If it does, the
    ORIGINAL bytes are restored before raising.

WHAT IT DOES NOT TOUCH
----------------------
Everything else. Including, explicitly, the ``[baseline.*]`` block, ``[gate.secondary]``, and
the ``per_season_must_pass`` comment on line 39 that mentions "(2021-2024)" -- that comment
describes the frozen baseline's seasons, and rewriting prose is not this script's job.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tomllib
from pathlib import Path

from conf.season_partition import default_season_partition

__all__ = [
    "GATE_CONFIG_PATH",
    "MIRRORED_KEY",
    "MIRRORED_TABLE",
    "main",
    "sync_gate_holdout",
]

#: The file this script mirrors into. One constant so a caller and a test name it once.
GATE_CONFIG_PATH = Path("config/gate.toml")

#: The TOML table and key that ARE the mirror. Scoped to the table on purpose: a bare search
#: for a line starting with "holdout" would also be willing to match a key of that name under
#: some future table, and silently rewrite the wrong one.
MIRRORED_TABLE = "[gate.seasons]"
MIRRORED_KEY = "holdout"

#: ``MIRRORED_KEY`` as bytes, compared by EQUALITY against a line's key half.
_MIRRORED_KEY_BYTES = MIRRORED_KEY.encode()

#: What a TOML table header looks like, so an array continuation line that merely
#: starts with "[" and ends with "]" cannot be mistaken for one (code review WR-05).
#: A header is a dotted key in brackets: no commas, no digits-only elements, and
#: `[[x]]` array-of-table headers are accepted too.
_TABLE_HEADER_RE = re.compile(rb"\[{1,2}\s*[A-Za-z_][A-Za-z0-9_.\-\"' ]*\s*\]{1,2}")


def _target_line_index(lines: list[bytes]) -> int:
    """Return the index of the ``holdout`` assignment inside ``[gate.seasons]``.

    Scans forward from the table header and stops at the NEXT table header, so the match
    cannot escape its own table.

    Args:
        lines: The file's bytes split on ``b"\\n"`` (each element may still carry a trailing
            ``b"\\r"``, which is exactly what preserves the file's line endings).

    Returns:
        The index into *lines*.

    Raises:
        ValueError: If the table is absent, if the key is absent within it, or if the table
            carries more than one such assignment. All three are refusals rather than guesses:
            a mirror that writes to a line it is not sure about is worse than one that does
            not write at all.
    """
    header = MIRRORED_TABLE.encode()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == header)
    except StopIteration:
        msg = (
            f"{GATE_CONFIG_PATH} has no {MIRRORED_TABLE} table, so there is nothing to "
            "mirror into. REFUSING rather than guessing where the holdout belongs."
        )
        raise ValueError(msg) from None

    # THE KEY MATCH IS AN EQUALITY, NOT A PREFIX (code review WR-05). It was
    # `stripped.startswith(b"holdout")`, which would also claim a future
    # `holdout_seasons = [...]` in this same table -- and the mirror would then
    # rewrite the wrong line while reporting success.
    #
    # THE TABLE BOUNDARY IS A HEADER PATTERN, NOT "starts with [ and ends with ]"
    # (also WR-05). A multi-line array value whose continuation line reads
    # `[2021, 2022]` matched the old test and ENDED the scan early, so the mirror
    # would report "no holdout assignment" for a key that is plainly there.
    matches = []
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if _TABLE_HEADER_RE.fullmatch(stripped):
            break
        if (
            b"=" in stripped
            and stripped.split(b"=", 1)[0].strip() == _MIRRORED_KEY_BYTES
        ):
            matches.append(index)

    if not matches:
        msg = (
            f"{GATE_CONFIG_PATH}'s {MIRRORED_TABLE} table has no '{MIRRORED_KEY}' "
            "assignment. REFUSING: the mirror has no target."
        )
        raise ValueError(msg)
    if len(matches) > 1:
        msg = (
            f"{GATE_CONFIG_PATH}'s {MIRRORED_TABLE} table has {len(matches)} "
            f"'{MIRRORED_KEY}' assignments at lines "
            f"{[i + 1 for i in matches]}. REFUSING: a mirror that cannot tell which line "
            "it owns must not write."
        )
        raise ValueError(msg)
    return matches[0]


def _masked_digest(raw: bytes, index: int) -> str:
    """sha256 over FILE BYTES with line *index* replaced by a fixed placeholder.

    The instrument behind the other-bytes-unchanged check: the one line the mirror owns
    is masked out, so the digest answers "did anything ELSE move?" and nothing else.

    IT TAKES RAW BYTES, NOT A LIST (code review WR-05). It used to take the same
    ``lines`` list the replacement was applied to, which made the check it powered
    UNFALSIFIABLE: ``updated = list(lines)`` then ``updated[index] = replacement``
    changes exactly one element, and both sides masked that same index, so
    ``before != after`` was provably always False and ``len(updated) != len(lines)``
    provably always equal. A refusal the module docstring presents as THE guarantee
    protecting a byte-identical ``[baseline.*]`` block was dead code -- the same
    "reads as a guarantee, behaves as a comment" failure mode
    ``build_features._preserved_weather_columns`` warns about.

    Digesting BYTES lets the same function run over the file as READ and over the file
    as WRITTEN AND READ BACK, which are genuinely different objects, so it CAN fail.

    Args:
        raw: Whole-file bytes.
        index: Zero-based index of the line to mask.

    Returns:
        The hex digest.
    """
    masked = raw.split(b"\n")
    masked[index] = b"<MIRRORED-HOLDOUT-LINE>"
    return hashlib.sha256(b"\n".join(masked)).hexdigest()


def sync_gate_holdout(dry_run: bool = False) -> str:
    """Rewrite ``gate.seasons.holdout`` from the committed partition rule.

    Args:
        dry_run: When True, compute and report the change but write nothing.

    Returns:
        A human-readable report naming the old line, the new line and whether the file was
        written. When the mirror is already current the report says so and nothing is written
        in either mode.

    Raises:
        ValueError: If the target line cannot be identified unambiguously, if the edit would
            move any byte outside that line, or if the rewritten file no longer parses as TOML
            or no longer carries the rule's holdout.
    """
    holdout = default_season_partition().holdout

    raw = GATE_CONFIG_PATH.read_bytes()
    lines = raw.split(b"\n")
    index = _target_line_index(lines)

    original = lines[index]
    # Preserve the line's exact leading whitespace and its trailing CR (if the working tree is
    # CRLF). Only the bytes between them are rewritten.
    body = original[:-1] if original.endswith(b"\r") else original
    line_ending = b"\r" if original.endswith(b"\r") else b""
    indent = body[: len(body) - len(body.lstrip())]
    rendered = ", ".join(str(season) for season in holdout)
    replacement = indent + f"{MIRRORED_KEY} = [{rendered}]".encode() + line_ending

    report_lines = [
        f"  file:  {GATE_CONFIG_PATH}",
        f"  table: {MIRRORED_TABLE}",
        f"  line:  {index + 1}",
        f"  old:   {body.decode('utf-8')}",
        f"  new:   {(indent + f'{MIRRORED_KEY} = [{rendered}]'.encode()).decode('utf-8')}",
    ]

    if original == replacement:
        return "\n".join(
            [
                f"{MIRRORED_TABLE}.{MIRRORED_KEY} is already current; nothing to write.",
                *report_lines,
            ]
        )

    # The masked digest of the file AS READ. The post-write read-back below compares
    # against this, and that comparison is the only one here that can actually fail.
    before = _masked_digest(raw, index)
    updated = list(lines)
    updated[index] = replacement
    new_raw = b"\n".join(updated)

    # A pre-write check over the RECONSTRUCTED bytes. Weaker than the read-back, and
    # stated as such: it catches a defect in the split/replace/join round trip -- a
    # lost line, a mangled separator -- rather than proving anything about the
    # filesystem. Kept because it is free, and a corrupted buffer should never reach
    # write_bytes.
    rebuilt = _masked_digest(new_raw, index)
    if rebuilt != before or len(updated) != len(lines):
        msg = (
            "REFUSING to write: rebuilding the file from its lines moved bytes "
            f"OUTSIDE the {MIRRORED_TABLE}.{MIRRORED_KEY} line.\n"
            f"  masked digest as read:    {before}\n"
            f"  masked digest as rebuilt: {rebuilt}\n"
            f"  line count before/after: {len(lines)}/{len(updated)}\n"
            "config/gate.toml's [baseline.*] block must stay byte-identical (D33-11, "
            "D33.1-05) -- it carries an undischarged 47-of-68-field divergence and two "
            "deliberate tripwires depend on those bytes."
        )
        raise ValueError(msg)

    parsed = tomllib.loads(new_raw.decode("utf-8"))
    parsed_holdout = tuple(int(s) for s in parsed["gate"]["seasons"]["holdout"])
    if parsed_holdout != tuple(holdout):
        msg = (
            "REFUSING to write: the rewritten file parses to holdout "
            f"{list(parsed_holdout)}, which is not the rule's {list(holdout)}."
        )
        raise ValueError(msg)

    if dry_run:
        return "\n".join(
            [
                "DRY RUN -- one line WOULD change; nothing was written.",
                *report_lines,
                "  other bytes: unchanged (masked digest identical)",
            ]
        )

    GATE_CONFIG_PATH.write_bytes(new_raw)

    # THE REAL GUARANTEE (code review WR-05): read the file BACK and digest it with
    # the mirrored line masked. This compares two independently-produced byte strings
    # -- the file as it was read, and the file as it now exists on disk -- so it CAN
    # fail, which is the point. On a mismatch the ORIGINAL bytes are RESTORED before
    # raising: leaving a half-correct config/gate.toml behind would be strictly worse
    # than the stale mirror this command exists to fix.
    read_back = _masked_digest(GATE_CONFIG_PATH.read_bytes(), index)
    if read_back != before:
        GATE_CONFIG_PATH.write_bytes(raw)
        msg = (
            "REFUSING the write, and the ORIGINAL bytes have been RESTORED: the file "
            f"on disk differs outside the {MIRRORED_TABLE}.{MIRRORED_KEY} line.\n"
            f"  masked digest as read:      {before}\n"
            f"  masked digest as read back: {read_back}\n"
            "config/gate.toml's [baseline.*] block must stay byte-identical (D33-11, "
            "D33.1-05) -- it carries an undischarged 47-of-68-field divergence and two "
            "deliberate tripwires depend on those bytes."
        )
        raise ValueError(msg)
    return "\n".join(
        [
            "WROTE one line.",
            *report_lines,
            "  other bytes: unchanged (read back from disk, masked digest identical)",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code: 0 on success, 1 on a refusal.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.sync_gate_holdout",
        description=(
            "Regenerate config/gate.toml's gate.seasons.holdout from "
            "conf.season_partition, as a single-line edit that refuses if any other "
            "byte would move."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the one line that would change and write nothing.",
    )
    args = parser.parse_args(argv)

    try:
        print(sync_gate_holdout(dry_run=args.dry_run))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
