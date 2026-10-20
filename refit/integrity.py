"""The pre-registered integrity checks a re-fit must pass before it goes live (RFIT-06, SPEC R5).

GATE-FREE BY OWNER RULING, NEVER CHECK-FREE
-------------------------------------------
The record (``ROLLING-REFIT-PREREGISTRATION.md`` section 10) rules the re-fit gate-free: "the
shadow arm is the control and the season close is the verdict", so no performance test stands
between a re-fit and the live arm. A broken model must still never go live, so every check here is
an ENGINEERING fact -- the record was not edited, the rows were the rule's rows, the inputs and
settings are the frozen ones, the calibration never saw its own rows, the re-measure finished, the
models score, the next decision run can stamp them -- and none reads a result.

THE ORDER IS THE RECORD'S
-------------------------
The checks run in ``backtest.refit_preregistration_2026.INTEGRITY_CHECKS`` order and the first
failure refuses the attempt by name (:class:`IntegrityRefusal`). The last two,
``clean_code_commit`` and ``bundle_validation``, belong to the swap itself (``refit.swap``): one
must hold before the permanent install intent is written, the other is the swap's own validation.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from backtest.refit_preregistration_2026 import INTEGRITY_CHECKS

__all__ = [
    "IntegrityRefusal",
]


class IntegrityRefusal(Exception):
    """A re-fit refused by one pre-registered integrity check, named in ``check``.

    Inherits bare ``Exception``: several call sites in this repository catch ``ValueError`` /
    ``KeyError`` and degrade quietly, and a refused re-fit degraded into "carry on" would put a
    model the record forbids on the live arm. The hook records the attempt ``refused`` with this
    check, and the FAILED push names it.

    Attributes:
        check: One of ``INTEGRITY_CHECKS``.
    """

    def __init__(self, check: str, message: str) -> None:
        if check not in INTEGRITY_CHECKS:
            msg = (
                f"{check!r} is not a pre-registered integrity check {INTEGRITY_CHECKS}"
            )
            raise ValueError(msg)
        super().__init__(f"{check}: {message}")
        self.check = check
