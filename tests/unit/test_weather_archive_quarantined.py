"""The archive endpoint is unreachable from the path that serves an unplayed game.

WHAT IS BEING GUARDED
---------------------
`https://archive-api.open-meteo.com/v1/archive` is a REANALYSIS product. It can describe
weather that has already occurred and nothing else. Pointing the live path at it is not a
performance problem or a data-quality problem; it is a category error, and the way it
FAILS is the dangerous part -- it returns a well-formed response with null hours, which is
exactly the shape a "fall back to the archive" repair would turn into a plausible number
for a game that has not been played.

D33-26 QUARANTINES the endpoint rather than deleting it. Deleting it would be wrong: the
Historical Forecast API reaches back only to about 2021, so the archive endpoint is the
only source for 2002-2020, and the historical weather backfill is the NAMED flip condition
for the 2026 gold-default switch. So it keeps working -- from ONE module, whose name says
what it is.

WHY A SOURCE SCAN, AND WHAT A SOURCE SCAN IS NOT
-------------------------------------------------
A structural scan proves a SHAPE is impossible. It cannot prove a behaviour is correct,
and it is not acceptance evidence for one. The `weather_source` vocabulary and the
beyond-horizon refusal are proven by raising and reading, in
`tests/unit/test_weather_pipeline.py` and `tests/unit/test_weather_forecast_horizon.py`.
What THIS module proves is narrower and structural: no amount of future editing inside the
live ingest module can reach the archive endpoint without turning this red first.

The scan is used here rather than a runtime check because a passing test suite is exactly
what the failure mode produces -- an archive fallback returns a value, so nothing raises.

FOUR CONTROLS, COPIED FROM `tests/unit/test_p31_constants_isolation.py`
------------------------------------------------------------------------
1. NON-VACUITY: the scan visits a non-empty module list. A scan over nothing is a green
   test that asserts nothing, which is worse than no test.
2. THE ASSERTION itself.
3. A PLANTED VIOLATION: a temp copy of a module containing an archive reference, proving
   the scan FIRES rather than only ever finding none.
4. NO FALSE POSITIVE: the legitimate backfill module is deliberately outside the scanned
   set, and a clean module is not flagged.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The archive endpoint, written out here so the scan has something concrete to look for.
# This module is DELIBERATELY not in the scanned set -- see
# `test_the_scan_does_not_flag_the_legitimate_backfill_module`.
ARCHIVE_URL_FRAGMENT = "archive-api.open-meteo.com"
ARCHIVE_CONSTANT_NAMES = frozenset({"ARCHIVE_ENDPOINT_URL", "OPEN_METEO_URL"})

# THE LIVE INGEST PATH. Every module through which a 2026 week's weather reaches silver.
# `required` marks the modules whose absence makes the scan meaningless.
LIVE_INGEST_MODULES: tuple[tuple[str, bool], ...] = (
    ("scripts/ingest_weather.py", True),
    ("features/weather.py", True),
)

# The ONE module the archive endpoint is reachable from, and the reason it is excluded.
# Named here rather than merely omitted, so the exclusion is a decision on the record
# instead of a gap somebody has to notice.
QUARANTINE_MODULE = "scripts/backfill_historical_weather.py"


def scan_module_for_archive_reference(path: Path) -> list[str]:
    """Return every archive-endpoint reference in the Python source at *path*.

    Two shapes are reported, because blocking one leaves the other:

    * A STRING literal anywhere containing the archive host. This catches the constant's
      value however it is named, an inline URL, and an f-string prefix.
    * A NAME or ATTRIBUTE binding one of the archive constant names, which catches an
      import of the constant from the quarantine module even though the URL text itself
      never appears in the importing file.

    Args:
        path: A Python source file.

    Returns:
        Human-readable violation strings, each naming the line. Empty when clean.
    """
    source = Path(path).read_text(encoding="utf-8")
    label = Path(path).as_posix()
    found: list[str] = []

    try:
        tree = ast.parse(source, filename=label)
    except SyntaxError as exc:  # pragma: no cover - a syntax error is its own failure
        return [f"{label}: could not be parsed: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if ARCHIVE_URL_FRAGMENT in node.value:
                found.append(
                    f"{label}:{node.lineno}: a string literal names the ARCHIVE host "
                    f"{ARCHIVE_URL_FRAGMENT!r}. That endpoint is a reanalysis product "
                    "and cannot answer for a game that has not been played; it is "
                    f"reachable only from {QUARANTINE_MODULE} (D33-26)."
                )
        if isinstance(node, ast.Name) and node.id in ARCHIVE_CONSTANT_NAMES:
            found.append(
                f"{label}:{node.lineno}: references the archive endpoint constant "
                f"{node.id}. Importing the constant reaches the archive endpoint just "
                "as surely as writing the URL out."
            )
        elif isinstance(node, ast.Attribute) and node.attr in ARCHIVE_CONSTANT_NAMES:
            found.append(
                f"{label}:{node.lineno}: references {node.attr} through a module "
                "reference, which reaches the archive endpoint."
            )
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in ARCHIVE_CONSTANT_NAMES:
                    found.append(
                        f"{label}:{node.lineno}: imports {alias.name} from "
                        f"{node.module}, which is the archive endpoint constant."
                    )

    return found


def _existing_live_modules() -> list[str]:
    present: list[str] = []
    for relative_path, required in LIVE_INGEST_MODULES:
        if (REPO_ROOT / relative_path).is_file():
            present.append(relative_path)
        elif required:
            pytest.fail(
                f"the REQUIRED live-ingest module {relative_path} is missing from this "
                "checkout. The quarantine scan would then visit a shorter list and "
                "could pass while asserting less than it claims."
            )
    return present


class TestTheScanIsLive:
    """Controls 1, 3 and 4 -- the scan visits something, fires, and does not over-fire."""

    def test_the_scan_visits_a_non_empty_module_list(self) -> None:
        present = _existing_live_modules()
        assert present, (
            "the archive quarantine scan visited ZERO modules. Every no-violation "
            "assertion below would then pass while proving nothing."
        )
        assert "scripts/ingest_weather.py" in present

    def test_the_scan_catches_a_planted_archive_reference(self) -> None:
        """Fail-closed: the scan REPORTS a violation rather than only ever finding none.

        Both shapes are planted -- the URL written out, and the constant imported from
        the quarantine module -- because blocking one and not the other would leave the
        obvious workaround open.
        """
        planted = Path(tempfile.mkdtemp()) / "planted_live_module.py"
        planted.write_text(
            'URL = "https://archive-api.open-meteo.com/v1/archive"\n'
            "from scripts.backfill_historical_weather import ARCHIVE_ENDPOINT_URL\n"
            "OTHER = ARCHIVE_ENDPOINT_URL\n",
            encoding="utf-8",
        )
        hits = scan_module_for_archive_reference(planted)
        joined = "\n".join(hits)

        assert hits, (
            "the scan found nothing in a module that names the archive endpoint"
        )
        assert "ARCHIVE host" in joined
        assert "ARCHIVE_ENDPOINT_URL" in joined

    def test_the_scan_does_not_flag_a_clean_module(self) -> None:
        clean = Path(tempfile.mkdtemp()) / "clean_live_module.py"
        clean.write_text(
            'FORECAST = "https://api.open-meteo.com/v1/forecast"\nHOURS = 24\n',
            encoding="utf-8",
        )
        assert scan_module_for_archive_reference(clean) == []

    def test_the_scan_does_not_flag_the_legitimate_backfill_module(self) -> None:
        """Control 4: the quarantine module IS excluded, and the reason is in its docstring.

        Excluding it silently would leave a reader unable to tell a deliberate boundary
        from an oversight, so the exclusion is asserted AND the module is required to say
        why it holds the endpoint.
        """
        scanned = {path for path, _ in LIVE_INGEST_MODULES}
        assert QUARANTINE_MODULE not in scanned

        module_path = REPO_ROOT / QUARANTINE_MODULE
        assert module_path.is_file(), (
            f"{QUARANTINE_MODULE} does not exist, so the archive endpoint has nowhere "
            "legitimate to live and the quarantine is a deletion in disguise."
        )
        docstring = ast.get_docstring(
            ast.parse(module_path.read_text(encoding="utf-8"))
        )
        assert docstring, f"{QUARANTINE_MODULE} has no module docstring"
        lowered = docstring.lower()
        assert "quarantin" in lowered, (
            "the quarantine module's docstring must SAY the endpoint is quarantined "
            "rather than deleted -- that sentence is the only thing telling a future "
            "reader why this module exists at all."
        )
        assert "2021" in docstring, (
            "the docstring must record WHY the endpoint is kept: the Historical "
            "Forecast API reaches back only to about 2021, so the archive endpoint is "
            "the only source for 2002-2020."
        )


class TestTheLiveIngestPathCannotReachTheArchiveEndpoint:
    """Control 2 -- the assertion this module exists for."""

    def test_no_live_module_references_the_archive_endpoint(self) -> None:
        violations: list[str] = []
        for relative_path in _existing_live_modules():
            violations.extend(
                scan_module_for_archive_reference(REPO_ROOT / relative_path)
            )

        assert not violations, (
            "archive-endpoint references found on the LIVE ingest path (D33-26, "
            "T-33-43).\n"
            + "\n".join(f"  - {line}" for line in violations)
            + "\n\nThe archive endpoint is a reanalysis product. It cannot answer for a "
            "game that has not been played, and a fallback to it is the "
            "fabricated-data class this project has already disclosed once. It is "
            f"reachable from {QUARANTINE_MODULE} and nowhere else."
        )

    def test_the_live_module_exposes_no_archive_constant_at_runtime(self) -> None:
        """The import-time view, beside the source-text view.

        A name assembled at run time would not appear as a literal in the AST, so this
        asserts the same fact through a different instrument.
        """
        import scripts.ingest_weather as ingest

        archive_names = [name for name in dir(ingest) if "ARCHIVE" in name.upper()]
        assert archive_names == [], (
            f"scripts/ingest_weather.py exposes {archive_names}. The archive endpoint "
            f"constant belongs in {QUARANTINE_MODULE}."
        )

    def test_the_backfill_module_still_holds_a_working_archive_endpoint(self) -> None:
        """QUARANTINED, NOT DELETED. Without this, the quarantine could be satisfied by
        removing the capability, which would strand 2002-2020."""
        import scripts.backfill_historical_weather as backfill

        assert hasattr(backfill, "ARCHIVE_ENDPOINT_URL")
        assert ARCHIVE_URL_FRAGMENT in backfill.ARCHIVE_ENDPOINT_URL
        assert backfill.ARCHIVE_ENDPOINT_URL.startswith("https://")
