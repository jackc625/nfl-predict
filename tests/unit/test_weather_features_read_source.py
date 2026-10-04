"""The gold-feeding reads of silver ``weather_features`` pin ``source="parquet"``.

Review fix WR-02 (commit 935bb97). ``data.storage.load_dataframe`` defaults to ``source="auto"``,
which tries DuckDB first. A stale 14-row DuckDB ``weather_features`` table left over from the
``weather_forecast`` era would win that race and silently feed gold 14 games of weather. The two
reads that feed gold therefore name the parquet file explicitly:

* ``scripts/build_features.py`` -- the gold build's weather join;
* ``features/weather.py`` -- the weather feature builder's own read.

THE SCAN. The source of those two modules is parsed, every ``load_dataframe(...)`` call whose
FIRST argument is the string ``"weather_features"`` is found, and its source (the 3rd positional
argument OR the ``source=`` keyword) must be the string ``"parquet"``. A call that omits the source
is flagged, because omitting it means "auto".

THREE CONTROLS
1. NON-VACUITY: exactly the two expected calls are found across the two files, and both files are
   scanned, so a scanner that matches nothing cannot report a clean tree.
2. PLANTED VIOLATIONS: a call with no source and a call with ``source="auto"`` are both flagged.
3. NO FALSE POSITIVE: a call on a different table, and a call with ``source="parquet"`` as a
   keyword, are not flagged.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCANNED_FILES: tuple[str, ...] = ("scripts/build_features.py", "features/weather.py")

#: One gold-feeding read in each scanned file, measured when the guard was written.
EXPECTED_CALLS_PER_FILE: int = 1

TABLE = "weather_features"
REQUIRED_SOURCE = "parquet"


def _string_value(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def weather_features_reads(source: str) -> list[tuple[int, str | None]]:
    """``(line, source_argument)`` for every ``load_dataframe("weather_features", ...)`` call.

    ``source_argument`` is the string given as the 3rd positional or the ``source=`` keyword, or
    ``None`` when it is absent (which ``load_dataframe`` reads as ``"auto"``).
    """
    reads: list[tuple[int, str | None]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or _call_name(node) != "load_dataframe":
            continue
        if not node.args or _string_value(node.args[0]) != TABLE:
            continue
        given: str | None = None
        if len(node.args) >= 3:
            given = _string_value(node.args[2])
        for keyword in node.keywords:
            if keyword.arg == "source":
                given = _string_value(keyword.value)
        reads.append((node.lineno, given))
    return reads


def violations(source: str) -> list[tuple[int, str | None]]:
    """The reads whose source is anything other than the string ``"parquet"``."""
    return [
        (line, given)
        for line, given in weather_features_reads(source)
        if given != REQUIRED_SOURCE
    ]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


class TestGoldFeedingWeatherReadsPinParquet:
    """The assertion, with its non-vacuity control."""

    def test_both_files_are_scanned_and_each_holds_the_expected_read(self) -> None:
        """NON-VACUITY: a scanner matching nothing cannot report a clean tree."""
        assert len(SCANNED_FILES) == 2
        counts = {
            path: len(weather_features_reads(_read(path))) for path in SCANNED_FILES
        }
        assert counts == dict.fromkeys(SCANNED_FILES, EXPECTED_CALLS_PER_FILE), counts

    def test_every_weather_features_read_names_the_parquet_file(self) -> None:
        found = {path: violations(_read(path)) for path in SCANNED_FILES}
        assert {path: bad for path, bad in found.items() if bad} == {}, (
            'a read of silver weather_features that does not pin source="parquet" falls back to '
            "the DuckDB-first default, where a stale weather_forecast-era table could win and "
            f"feed gold the wrong weather. Offending (line, source) per file: {found}"
        )


class TestTheScannerFlagsWhatItShould:
    """PLANTED VIOLATIONS and NO FALSE POSITIVE."""

    def test_a_read_with_no_source_is_flagged(self) -> None:
        planted = 'df = load_dataframe("weather_features", "silver")\n'
        assert violations(planted) == [(1, None)]

    def test_a_read_with_source_auto_is_flagged(self) -> None:
        planted = 'df = load_dataframe("weather_features", "silver", source="auto")\n'
        assert violations(planted) == [(1, "auto")]

    def test_a_positional_auto_is_flagged(self) -> None:
        planted = 'df = load_dataframe("weather_features", "silver", "auto")\n'
        assert violations(planted) == [(1, "auto")]

    def test_a_read_of_another_table_is_not_flagged(self) -> None:
        assert violations('df = load_dataframe("weather", "silver")\n') == []

    def test_the_keyword_and_positional_parquet_spellings_are_not_flagged(self) -> None:
        keyword = (
            'df = load_dataframe("weather_features", "silver", source="parquet")\n'
        )
        positional = 'df = load_dataframe("weather_features", "silver", "parquet")\n'
        assert violations(keyword) == []
        assert violations(positional) == []
        # and both are still SEEN, so the empty result above is not blindness
        assert len(weather_features_reads(keyword)) == 1
        assert len(weather_features_reads(positional)) == 1
