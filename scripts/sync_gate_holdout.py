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

So this edits ONE LINE, in place, on raw bytes, and REFUSES if its edit would move any other
byte. The refusal is verified by a digest over the whole file with that one line masked out:
identical before and after, or nothing is written. Line endings are preserved exactly (the file
is CRLF in this Windows working tree and LF in the git blob), which is why it works on bytes
rather than on decoded text.

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

    matches = []
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith(b"[") and stripped.endswith(b"]"):
            break
        if stripped.startswith(MIRRORED_KEY.encode()) and b"=" in stripped:
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


def _masked_digest(lines: list[bytes], index: int) -> str:
    """sha256 over the whole file with line *index* replaced by a fixed placeholder.

    This is the instrument behind the other-bytes-unchanged refusal: the one line the mirror
    owns is masked out, so the digest answers "did anything ELSE move?" and nothing else.
    """
    masked = list(lines)
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

    before = _masked_digest(lines, index)
    updated = list(lines)
    updated[index] = replacement
    after = _masked_digest(updated, index)

    if before != after or len(updated) != len(lines):
        msg = (
            "REFUSING to write: the edit would move bytes OUTSIDE the "
            f"{MIRRORED_TABLE}.{MIRRORED_KEY} line.\n"
            f"  masked digest before: {before}\n"
            f"  masked digest after:  {after}\n"
            f"  line count before/after: {len(lines)}/{len(updated)}\n"
            "config/gate.toml's [baseline.*] block must stay byte-identical (D33-11, "
            "D33.1-05) -- it carries an undischarged 47-of-68-field divergence and two "
            "deliberate tripwires depend on those bytes."
        )
        raise ValueError(msg)

    new_raw = b"\n".join(updated)
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
    return "\n".join(
        [
            "WROTE one line.",
            *report_lines,
            "  other bytes: unchanged (masked digest identical)",
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
