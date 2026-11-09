"""Fit the key-number tables once and write ``config/key_number_tables.json``.

Reads the pinned nflverse schedules for the seasons named in ``backtest/early_arm_2026.py``, keeps
the regular-season games with a final and both closing lines, and fits one table per target: the
home margin against the closing spread, and the total against the closing total. Writes nothing
else. Do not run it between 17:00 and 17:20 ET.

Usage:
    uv run python -m scripts.fit_key_numbers

ASCII only, no emoji.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from backtest.early_arm_2026 import KEY_NUMBER_FIT_SEASONS, KEY_NUMBER_TABLES_PATH
from data.upstream_pin import load_schedules
from models.key_numbers import KeyNumberTable, fit_key_number_table

_REQUIRED_COLUMNS: list[str] = ["result", "total", "spread_line", "total_line"]


def fit_tables(schedules: pd.DataFrame) -> dict[str, KeyNumberTable]:
    """Fit the spread and total tables on the complete regular-season games."""
    complete = schedules.dropna(subset=_REQUIRED_COLUMNS)
    games = complete[complete["game_type"] == "REG"]
    return {
        "ats": fit_key_number_table(
            games["result"], games["spread_line"], low=-70, high=70, mirror=True
        ),
        "ou": fit_key_number_table(
            games["total"], games["total_line"], low=0, high=120, mirror=False
        ),
    }


def main() -> int:
    """Fit the recorded seasons, write the committed file and print each table's sd."""
    first, last = KEY_NUMBER_FIT_SEASONS
    tables = fit_tables(load_schedules(list(range(first, last + 1))))

    payload = {
        "fit_seasons": [first, last],
        **{key: asdict(table) for key, table in tables.items()},
    }
    Path(KEY_NUMBER_TABLES_PATH).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    for key, table in tables.items():
        print(f"{key} sd={table.sd:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
