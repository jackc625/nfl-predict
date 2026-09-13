"""No fabricated-weather shape can return to `features/weather.py`.

WHAT IS BEING GUARDED
---------------------
SPEC R4: "the 65.0 literal is deleted as a reachable path in
`features/weather.py`", and SPEC prohibition 1: "MUST NOT replace the 65.0
constant with any other numeric stand-in for a missing observation, under any
name -- a seasonal average or a venue mean is the same defect wearing a better
label."

WHY A SCAN FOR THE NUMBER ALONE WOULD PASS WHILE THE DEFECT SURVIVED
---------------------------------------------------------------------
FOUR shapes, because each of the first three passes a scan written for the one
before it (RESEARCH 8.2, and Codex 33.1-04 HIGH for the fourth):

1. THE LITERAL. `65.0`, the mild-temperature default.
2. THE FACTORY. `_default_temperature_features` also returned `temp_warm: 1.0`,
   `cold_impact_score: 0.0`, `scoring_multiplier: 1.0` and
   `ball_handling_difficulty: 1.0`. Those are the same fabrication in one-hot
   clothing and they survive deleting the number. `_default_wind_features` and
   `_default_precipitation_features` are named here too, even though their
   VALUES were true indoors: while a function is called `_default_*` the next
   reader wires the next `except` branch to it by analogy, which is precisely
   what had happened. They are renamed `_indoor_*`, and the old names must stay
   gone.
3. THE NO-WEATHER-ROW BLOCK. Two dict literals set `is_outdoor: False` AND
   `condition: "Clear"` for a game with no record, which made a missing record
   indistinguishable from a dome BY CONSTRUCTION rather than by accident. The
   shape is the PAIR: `is_outdoor: False` alone is an ordinary dome.
4. THE FABRICATING HANDLER. An `except` branch that RETURNS a default family
   rather than raising. A scan for the three shapes above passes while the wind
   and precipitation handlers still invent a perfectly calm dry day out of a
   calculation error -- in the two families the deployed O/U model reads most
   heavily. D33.1-07's third state is about the CALCULATION, not about
   temperature.

WHY AN AST WALK AND NOT A TEXT MATCH
-------------------------------------
Task 1 leaves COMMENTS behind that name what was deleted, including the number
and the two dict keys. A text scan would flag its own explanation and the only
way to make it pass would be to delete the explanation. Walking the AST means a
mention inside a comment or a docstring is a string, or nothing at all, and is
never a float constant or a dict literal. Do not "fix" this scan by making it
read the source text.

A structural scan proves a SHAPE is impossible. It is not acceptance evidence
for a BEHAVIOUR: the three D33.1-07 states are proven by building and reading,
in `tests/unit/test_weather_null_observation.py`.

FOUR CONTROLS, copied from `tests/unit/test_weather_archive_quarantined.py:31-42`
---------------------------------------------------------------------------------
1. NON-VACUITY: the scan visits a non-empty module list, and a REQUIRED module
   absent from the checkout fails BY NAME rather than shortening the list.
2. THE ASSERTION: `features/weather.py` carries none of the four shapes.
3. A PLANTED VIOLATION: each of the four shapes, planted into a temporary
   module, produces a hit -- so a green scan means something.
4. NO FALSE POSITIVE: a clean module, a comment-and-docstring mention, a dome
   dict without `condition: "Clear"`, and a RAISING `except` handler all
   produce zero hits.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The mild-temperature default, written out here so the scan has something
# concrete to look for. This module is deliberately NOT in the scanned set.
MILD_TEMPERATURE_DEFAULT = 65.0

# The three factory names that must not come back. The two renamed ones are
# included because the RENAME is the guard -- see shape 2 in the docstring.
FORBIDDEN_FACTORY_NAMES = frozenset(
    {
        "_default_temperature_features",
        "_default_wind_features",
        "_default_precipitation_features",
    }
)

# THE MODULES SCANNED. `required` marks the ones whose absence makes the scan
# meaningless rather than merely shorter.
#
# `scripts/backfill_historical_weather.py` and `scripts/ingest_weather.py` are
# NOT scanned, and the exclusion is a decision on the record rather than a gap:
# they WRITE weather records, and an indoor record legitimately carries fixed
# values. What this scan guards is the FEATURE builder, which is where an
# absent observation became a number for 6,485 of 6,499 gold rows.
WEATHER_FEATURE_MODULES: tuple[tuple[str, bool], ...] = (("features/weather.py", True),)

RECORD_WRITING_MODULES_NOT_SCANNED: tuple[str, ...] = (
    "scripts/ingest_weather.py",
    "scripts/backfill_historical_weather.py",
)

SPEC_PROHIBITION_1 = (
    "MUST NOT replace the 65.0 constant with any other numeric stand-in for a "
    "missing observation, under any name -- a seasonal average or a venue mean "
    "is the same defect wearing a better label (SPEC R4/R5, prohibition 1)."
)


def _is_indoor_factory_name(name: str | None) -> bool:
    return bool(name) and name.startswith("_indoor_") and name.endswith("_features")


def _dict_is_no_weather_row(node: ast.Dict) -> bool:
    """True for the PAIR that made a missing record look like a dome.

    The pair is the shape, not either key alone: `is_outdoor: False` on its own
    is an ordinary dome record, and flagging it would make this scan unusable.
    """
    outdoor_false = False
    condition_clear = False
    for key, value in zip(node.keys, node.values, strict=False):
        if not isinstance(key, ast.Constant):
            continue
        if key.value == "is_outdoor" and isinstance(value, ast.Constant):
            outdoor_false = value.value is False
        if key.value == "condition" and isinstance(value, ast.Constant):
            condition_clear = value.value == "Clear"
    return outdoor_false and condition_clear


def _enclosing_function(tree: ast.AST, target: ast.AST) -> str:
    """The name of the function *target* sits in, so a partial removal is named."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for child in ast.walk(node):
                if child is target:
                    return node.name
    return "<module>"


def scan_module_for_fabricated_weather(path: Path) -> list[str]:
    """Return every fabricated-weather shape in the Python source at *path*.

    Args:
        path: A Python source file.

    Returns:
        Human-readable violation strings, each of the form
        ``path:lineno: <what was found>``. ALL hits are reported, not the first,
        so a partial deletion is visible in one run. Empty when clean.
    """
    source = Path(path).read_text(encoding="utf-8")
    label = Path(path).as_posix()
    found: list[str] = []

    try:
        tree = ast.parse(source, filename=label)
    except SyntaxError as exc:  # pragma: no cover - a syntax error is its own failure
        return [f"{label}: could not be parsed: {exc}"]

    for node in ast.walk(tree):
        # SHAPE 1 -- the literal.
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, float)
            and node.value == MILD_TEMPERATURE_DEFAULT
        ):
            found.append(
                f"{label}:{node.lineno}: the mild-temperature default "
                f"{MILD_TEMPERATURE_DEFAULT} appears as a float constant. "
                f"{SPEC_PROHIBITION_1}"
            )

        # SHAPE 2 -- the factory, by any of its three names.
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name in FORBIDDEN_FACTORY_NAMES
        ):
            found.append(
                f"{label}:{node.lineno}: defines {node.name}. A `_default_*` "
                "weather factory is the mild default in one-hot clothing, and "
                "the NAME is what invites the next `except` branch to be wired "
                "to it by analogy. The indoor states are `_indoor_*_features`."
            )

        # SHAPE 3 -- the no-weather-row block.
        if isinstance(node, ast.Dict) and _dict_is_no_weather_row(node):
            found.append(
                f"{label}:{node.lineno}: a dict pairs `is_outdoor: False` with "
                '`condition: "Clear"`. That is the no-weather-row block: it '
                "routes a game with NO record down the indoor branch, which "
                "makes a missing record indistinguishable from a dome BY "
                "CONSTRUCTION rather than by accident."
            )

        # SHAPE 4 -- the fabricating `except` handler.
        if isinstance(node, ast.ExceptHandler):
            family = _enclosing_function(tree, node)
            for descendant in ast.walk(node):
                if not isinstance(descendant, ast.Return):
                    continue
                value = descendant.value
                if isinstance(value, ast.Dict):
                    found.append(
                        f"{label}:{descendant.lineno}: an `except` handler in "
                        f"{family} RETURNS a dict literal. A caught exception "
                        "is a BUG, not a state (D33.1-07): an `except` branch "
                        "here must raise."
                    )
                elif isinstance(value, ast.Call) and _is_indoor_factory_name(
                    getattr(value.func, "attr", None) or getattr(value.func, "id", None)
                ):
                    found.append(
                        f"{label}:{descendant.lineno}: an `except` handler in "
                        f"{family} RETURNS an indoor factory. The indoor values "
                        "are TRUE for a covered game and FABRICATED for a "
                        "calculation error -- which is the whole reason those "
                        "factories were renamed."
                    )

    return found


def _existing_weather_feature_modules() -> list[str]:
    present: list[str] = []
    for relative_path, required in WEATHER_FEATURE_MODULES:
        if (REPO_ROOT / relative_path).is_file():
            present.append(relative_path)
        elif required:
            pytest.fail(
                f"the REQUIRED weather feature module {relative_path} is missing "
                "from this checkout. The fabricated-weather scan would then "
                "visit a shorter list and could pass while asserting less than "
                "it claims."
            )
    return present


def _temporary_module(body: str) -> Path:
    planted = Path(tempfile.mkdtemp()) / "planted_weather_module.py"
    planted.write_text(body, encoding="utf-8")
    return planted


class TestTheScanIsLive:
    """Controls 1, 3 and 4."""

    def test_non_vacuity_the_scan_visits_a_non_empty_module_list(self) -> None:
        present = _existing_weather_feature_modules()
        assert present, (
            "the fabricated-weather scan visited ZERO modules. Every "
            "no-violation assertion below would then pass while proving nothing."
        )
        assert "features/weather.py" in present

    def test_non_vacuity_a_missing_required_module_fails_by_name(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "tests.unit.test_weather_no_imputed_default.WEATHER_FEATURE_MODULES",
            (("features/a_module_that_does_not_exist.py", True),),
        )
        # `pytest.fail` raises an OutcomeException, which is a BaseException and
        # NOT an Exception -- `pytest.raises(Exception)` would let it through
        # and this control would report a false failure.
        with pytest.raises(pytest.fail.Exception) as excinfo:
            _existing_weather_feature_modules()
        message = str(excinfo.value)
        assert "a_module_that_does_not_exist.py" in message
        assert "shorter list" in message, (
            "the failure must say WHY an absent required module matters: the "
            "scan would otherwise visit a shorter list and pass while "
            "asserting less than it claims"
        )

    def test_a_planted_mild_temperature_literal_is_flagged(self) -> None:
        planted = _temporary_module('DEFAULTS = {"temp_f": 65.0}\n')
        hits = scan_module_for_fabricated_weather(planted)
        assert len(hits) == 1, f"expected exactly one hit, got {hits}"
        assert planted.name in hits[0]
        assert ":1:" in hits[0]
        assert "65.0" in hits[0]
        assert "venue mean" in hits[0], (
            "the failure must quote SPEC prohibition 1, so a reader cannot "
            "satisfy the scan by swapping the number for a seasonal average"
        )

    @pytest.mark.parametrize("factory", sorted(FORBIDDEN_FACTORY_NAMES))
    def test_a_planted_default_factory_definition_is_flagged(
        self, factory: str
    ) -> None:
        planted = _temporary_module(f"def {factory}(self):\n    return dict(a=1.0)\n")
        hits = scan_module_for_fabricated_weather(planted)
        assert len(hits) == 1, f"expected exactly one hit, got {hits}"
        assert factory in hits[0]
        assert planted.name in hits[0]

    def test_a_planted_no_weather_row_dict_is_flagged(self) -> None:
        planted = _temporary_module(
            "ROW = {\n"
            '    "is_outdoor": False,\n'
            '    "wind_mph": 0.0,\n'
            '    "condition": "Clear",\n'
            "}\n"
        )
        hits = scan_module_for_fabricated_weather(planted)
        assert len(hits) == 1, f"expected exactly one hit, got {hits}"
        assert "is_outdoor" in hits[0]
        assert "BY CONSTRUCTION" in hits[0]

    def test_a_planted_fabricating_except_handler_is_flagged(self) -> None:
        """The fourth shape, and the one Codex's HIGH is about."""
        planted = _temporary_module(
            "class C:\n"
            "    def calculate_wind_features(self, payload):\n"
            "        try:\n"
            "            return float(payload)\n"
            "        except ValueError:\n"
            "            return self._indoor_wind_features()\n"
        )
        hits = scan_module_for_fabricated_weather(planted)
        assert len(hits) == 1, f"expected exactly one hit, got {hits}"
        assert "indoor factory" in hits[0]
        assert "calculate_wind_features" in hits[0], (
            "a partial removal must name WHICH of the three families survived"
        )

    def test_a_planted_dict_returning_except_handler_is_flagged(self) -> None:
        planted = _temporary_module(
            "def calculate_precipitation_features(payload):\n"
            "    try:\n"
            "        return float(payload)\n"
            "    except ValueError:\n"
            '        return {"is_dry": 1.0}\n'
        )
        hits = scan_module_for_fabricated_weather(planted)
        assert len(hits) == 1, f"expected exactly one hit, got {hits}"
        assert "calculate_precipitation_features" in hits[0]

    def test_no_false_positive_on_a_clean_module(self) -> None:
        planted = _temporary_module(
            "MILD = 64.0\n"
            'ROW = {"is_outdoor": False, "condition": "indoor"}\n'
            "def _indoor_wind_features():\n"
            '    return {"wind_mph": 0.0}\n'
        )
        assert scan_module_for_fabricated_weather(planted) == []

    def test_no_false_positive_on_a_comment_and_docstring_mention(self) -> None:
        """CONTROL 4, and the reason this scan walks the AST.

        Task 1 leaves comments behind that name what was deleted. A text scan
        would flag its own explanation, and the only way to make it pass would
        be to delete the explanation.
        """
        planted = _temporary_module(
            '"""The 65.0 mild-temperature default was deleted here."""\n'
            "# It used to be {'is_outdoor': False, 'condition': 'Clear'}.\n"
            "VALUE = 41.0\n"
        )
        assert scan_module_for_fabricated_weather(planted) == []

    def test_no_false_positive_on_a_dome_dict_without_condition_clear(self) -> None:
        """The shape is the PAIR. `is_outdoor: False` alone is a dome."""
        planted = _temporary_module('DOME = {"is_outdoor": False, "wind_mph": 0.0}\n')
        assert scan_module_for_fabricated_weather(planted) == []

    def test_no_false_positive_on_a_raising_except_handler(self) -> None:
        """Shape 4 flags a FABRICATING handler, not every handler."""
        planted = _temporary_module(
            "def calculate_wind_features(payload):\n"
            "    try:\n"
            "        return float(payload)\n"
            "    except ValueError as exc:\n"
            "        raise RuntimeError('wind') from exc\n"
        )
        assert scan_module_for_fabricated_weather(planted) == []


class TestTheWeatherBuilderCarriesNoFabricatedShape:
    """Control 2 -- the assertion this module exists for."""

    def test_no_weather_feature_module_carries_a_fabricated_shape(self) -> None:
        violations: list[str] = []
        for relative_path in _existing_weather_feature_modules():
            violations.extend(
                scan_module_for_fabricated_weather(REPO_ROOT / relative_path)
            )

        assert not violations, (
            "fabricated-weather shapes found in the weather FEATURE builder "
            "(SPEC R4, D33.1-07).\n"
            + "\n".join(f"  - {line}" for line in violations)
            + "\n\n"
            + SPEC_PROHIBITION_1
        )

    def test_the_record_writing_modules_are_excluded_on_the_record(self) -> None:
        """Control 4: the exclusion is a DECISION, not an omission."""
        scanned = {path for path, _ in WEATHER_FEATURE_MODULES}
        for module in RECORD_WRITING_MODULES_NOT_SCANNED:
            assert module not in scanned
            assert (REPO_ROOT / module).is_file(), (
                f"{module} does not exist, so this exclusion is describing a "
                "boundary that is not there"
            )

    def test_the_renamed_indoor_factories_are_present_and_reachable(self) -> None:
        """QUARANTINED, NOT DELETED -- the same shape the archive scan uses.

        Without this, the scan above could be satisfied by deleting the indoor
        states outright, which would make a dome NULL in wind and precipitation
        and contradict D33.1-07 in the other direction.
        """
        from features.weather import WeatherFeaturesCalculator

        for factory in (
            "_indoor_wind_features",
            "_indoor_precipitation_features",
            "_indoor_temperature_features",
        ):
            assert hasattr(WeatherFeaturesCalculator, factory)

        calculator = WeatherFeaturesCalculator()
        assert calculator._indoor_wind_features()["wind_calm"] == 1.0
        assert calculator._indoor_precipitation_features()["is_dry"] == 1.0
