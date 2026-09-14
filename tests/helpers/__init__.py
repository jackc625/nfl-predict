"""Pure, tier-neutral assertion helpers shared by more than one test tier.

WHY A PACKAGE AND NOT A MODULE INSIDE ONE TIER
----------------------------------------------
This repo runs its tests in three tiers and a helper that lives inside one of them can
only be reached by the others through an upward import. A unit test importing from the
integration tier is reversed layering: it drags that tier's collection -- its conftest,
its fixtures and its lake-backed session setup -- into a process that was meant to run
alone. The helpers here belong to NO tier, so any tier may import them.

WHAT MAY LIVE HERE
------------------
Pure assertion and observation helpers only: no fixtures, no I/O, no project imports.
A helper that reads the data lake is not a helper, it is a fixture, and it belongs in
the tier that owns the lake.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations
