"""The switch week the website reads is the switch week the record holds.

``api/`` may not import ``backtest`` (tests/api/test_import_guard*.py), so ``api/cache_schema.py``
carries its OWN copy of ``backtest.midseason_switch_2026.DAY_BEFORE_RETIRED_FROM`` for /bets to
decide, from that week on, that the day-before system is retired. A copy can drift; this pins it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import api.cache_schema
import backtest.midseason_switch_2026


def test_the_page_copy_equals_the_record() -> None:
    assert (
        api.cache_schema.DAY_BEFORE_RETIRED_FROM
        == backtest.midseason_switch_2026.DAY_BEFORE_RETIRED_FROM
    )
