# Coding Conventions

**Analysis Date:** 2026-03-18

## Naming Patterns

**Files:**
- Snake case for all Python files: `train_wp.py`, `elo_features.py`, `test_elo_and_probabilities.py`
- Module docstring at top of every file with description of purpose
- Test files: `test_*.py` or `*_test.py` naming convention
- Config files: `settings.py`, `config.py` in `conf/` and `api/` directories
- Feature builders: `*_features.py` (e.g., `elo_features.py`, `team_form.py`)

**Functions:**
- Snake case: `get_elo_features_for_game()`, `validate_game_data()`, `calculate_k_factor()`
- Private functions prefixed with underscore: `_expected_score()`, `_calculate_k_factor()`
- Descriptive names indicating purpose and return type
- Async functions prefixed with `async def`: `async def initialize_models()`

**Variables:**
- Snake case for all variables: `base_k`, `home_team`, `rolling_epa_per_play`
- Constants in UPPER_SNAKE_CASE: `NFL_TEAMS`, `GAME_ID_PATTERN`, `BASE_K_FACTOR`
- Meaningful names reflecting data type/content: `mock_games_data`, `feature_matrix`, `calibration_data`
- Enum values lowercase: `GameStatus.SCHEDULED`, `BetType.MONEYLINE`

**Types:**
- PascalCase for classes: `EloRatingSystem`, `FeatureMatrixBuilder`, `TestEloRatingSystem`
- Dataclasses: `@dataclass` decorator with descriptive class names: `EloRating`, `WPModelPrediction`, `WPModelResults`
- Enums: `class BetType(str, Enum)`, `class GameStatus(str, Enum)` for API response consistency
- Type hints everywhere using `typing` module: `Dict[str, float]`, `Optional[str]`, `List[Dict[str, Any]]`

## Code Style

**Formatting:**
- Black formatter with line length of 88 characters (`pyproject.toml`: `line-length = 88`)
- Target Python 3.11+ (`target-version = ['py311']`)
- Exclude: `.eggs`, `.git`, `.venv`, `.tox`, `build`, `dist`, `data`, `outputs`, `artifacts`

**Linting:**
- isort for import organization with Black profile compatibility (`profile = "black"`, `line_length = 88`)
- MyPy for strict type checking enabled:
  - `disallow_untyped_defs = true`
  - `disallow_incomplete_defs = true`
  - `check_untyped_defs = true`
  - `disallow_untyped_decorators = true`
- Flake8 available but not enforced in config
- Some third-party modules ignored by MyPy: `nfl_data_py.*`, `meteostat.*`, `duckdb.*`, `polars.*`, `lightgbm.*`

## Import Organization

**Order:**
1. Standard library imports: `import os`, `import sys`, `from pathlib import Path`, `from datetime import datetime`
2. Third-party imports: `import pandas as pd`, `import numpy as np`, `from fastapi import FastAPI`
3. Project imports: `from utils import get_logger`, `from conf.settings import get_settings`, `from api.schemas import GameStatus`

**Path Aliases:**
- Known first-party packages in `pyproject.toml`: `scripts`, `ratings`, `models`, `backtest`, `api`, `features`, `utils`, `conf`, `data`, `deployment`
- Imports use full paths from project root: `from ratings.elo import EloRatingSystem`, not relative imports

## Error Handling

**Patterns:**
- Custom exception hierarchy with base class `NFLPredictionAPIException`:
  - `DataNotFoundError` for 404 scenarios
  - `ValidationError` for 400 validation failures
  - `ModelUnavailableError` for 503 service unavailable
  - `DataStaleError` for 202 stale data
  - `RateLimitError` for 429 rate limiting
  - `ConfigurationError` for 500 config issues
- Exception handler utilities: `raise_not_found()`, `raise_validation_error()`, `raise_model_unavailable()`
- Try-except blocks with specific exception types (not bare `except`)
- Logging at error level with traceback: `logger.error(..., exc_info=True)`
- HTTP exceptions raised with `HTTPException(status_code=422, detail="message")`

Example from `api/exceptions.py`:
```python
class NFLPredictionAPIException(Exception):
    def __init__(self, message: str, status_code: int = 500, error_code: str = "INTERNAL_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
```

Example from `api/main.py` (line 221-243):
```python
def validate_team_param(team: Optional[str]) -> Optional[str]:
    if team is None:
        return None
    team = team.upper().strip()
    if not (2 <= len(team) <= 4):
        raise HTTPException(status_code=422, detail="Team abbreviation must be 2-4 characters")
    if not re.match(r'^[A-Z]+$', team):
        raise HTTPException(status_code=422, detail="Team abbreviation must contain only letters")
    return team
```

## Logging

**Framework:** structlog with setup via `utils.logging_config.setup_logging()` - provides structured logging with context
- Access via `get_logger(__name__)` helper function
- Returns module-scoped logger instances

**Patterns:**
- Import at module level: `logger = get_logger(__name__)`
- Log at appropriate level: `logger.info()` for operations, `logger.warning()` for issues, `logger.error()` for failures
- Include context in logs: `logger.error("Failed to get Elo features for game", home_team=home_team, away_team=away_team, error=str(e))`
- Use `extra={}` dict for structured logging in HTTP handlers: `extra={'request_id': request_id, 'path': request.url.path}`
- Never log sensitive data (API keys, passwords)

Example from `features/elo_features.py` (lines 67-70):
```python
logger.error("Failed to get Elo features for game",
            home_team=home_team, away_team=away_team,
            season=season, week=week, error=str(e))
```

## Comments

**When to Comment:**
- Module docstrings required for all files with triple-quoted description
- Class docstrings with purpose and key attributes
- Function/method docstrings with Args and Returns sections
- Inline comments for non-obvious logic or important calculations
- Do NOT comment obvious code (e.g., `x = 1  # set x to 1`)

**JSDoc/TSDoc Style:**
- Use Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections
- Type hints in function signatures, descriptions in docstring

Example from `ratings/elo.py` (lines 114-127):
```python
def _expected_score(self, rating_a: float, rating_b: float, hfa: float = 0.0) -> float:
    """
    Calculate expected score for team A vs team B.

    Args:
        rating_a: Team A's Elo rating
        rating_b: Team B's Elo rating
        hfa: Home field advantage for team A (if home)

    Returns:
        Expected score (0-1) for team A
    """
```

## Function Design

**Size:**
- Keep functions under 50 lines where practical
- Complex algorithms may extend to 100-150 lines (e.g., walk-forward validation)
- Extract helper functions for repeated logic
- Example complex function: `get_week_recommendations()` in `api/main.py` is 141 lines but handles full recommendation pipeline

**Parameters:**
- Use type hints for all parameters
- Optional parameters should have default values
- For multiple related parameters, consider dataclass: `WalkForwardValidator(train_size=0.8, test_size=0.2, ...)`
- Path-based parameters use `Path` or `str` from pathlib: `filepath: str = None`

**Return Values:**
- Always specify return type hint
- Return dataclasses for complex multi-field returns: `WPModelPrediction(game_id, home_team, ...)`
- Return tuples for multiple related values: `Tuple[pd.DataFrame, Dict[str, float]]`
- Return None explicitly for optional returns: `-> Optional[str]`
- Dict[str, Any] for flexible returns with varying keys

Example from `models/train_wp.py` (lines 78-87):
```python
@dataclass
class WPModelPrediction:
    game_id: str
    home_team: str
    away_team: str
    raw_win_probability: float
    calibrated_win_probability: Optional[float] = None
    prediction_confidence: Optional[float] = None
```

## Module Design

**Exports:**
- Use `__all__` list in `__init__.py` for public API (see `utils/__init__.py` lines 89-162)
- Example: `__all__ = ["get_logger", "setup_logging", "validate_game_data", ...]`
- Import and re-export convenience imports in init files

**Barrel Files:**
- Convention used: `__init__.py` in feature modules re-exports key classes and functions
- Example `utils/__init__.py`: imports from submodules and exposes in `__all__`
- Enables `from utils import get_logger` instead of `from utils.logging_config import get_logger`

**Module Organization:**
- Data models in separate `schemas.py` or `models.py`
- Business logic in main module files
- Tests in mirrored `tests/` directory structure
- Example: `api/schemas.py` contains all Pydantic models; `api/main.py` contains endpoints

---

*Convention analysis: 2026-03-18*
