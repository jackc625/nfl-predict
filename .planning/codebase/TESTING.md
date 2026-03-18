# Testing Patterns

**Analysis Date:** 2026-03-18

## Test Framework

**Runner:**
- pytest 7.4.0+ (`pyproject.toml`: `pytest>=7.4.0`)
- Config: `[tool.pytest.ini_options]` in `pyproject.toml` (lines 172-189)
- Test discovery paths: `testpaths = ["tests"]`
- File patterns: `python_files = ["test_*.py", "*_test.py"]`
- Class patterns: `python_classes = ["Test*"]`
- Function patterns: `python_functions = ["test_*"]`

**Assertion Library:**
- pytest built-in assertions (no separate assertion library)
- Custom assertion helpers in `tests/conftest.py` (lines 336-357):
  - `assert_dataframe_structure()` - validates DataFrame structure and columns
  - `assert_probability_range()` - validates values are in [0, 1]
  - `assert_no_data_leakage()` - ensures no future data in features

**Run Commands:**
```bash
pytest                           # Run all tests
pytest -v                        # Verbose output
pytest --cov=scripts,ratings,models,backtest,api --cov-report=html  # Generate HTML coverage
pytest tests/unit/               # Run only unit tests
pytest tests/integration/        # Run only integration tests
pytest -k "test_elo"             # Run tests matching pattern
pytest --cov-fail-under=80       # Fail if coverage < 80%
```

Async support: `asyncio_mode = "auto"` enabled for pytest-asyncio

## Test File Organization

**Location:**
- Co-located with source code is NOT used
- Separate `tests/` directory mirrors package structure
- Example structure:
  ```
  tests/
  ├── unit/
  │   ├── test_elo_and_probabilities.py
  │   ├── test_feature_builders.py
  │   └── __init__.py
  ├── integration/
  │   ├── test_end_to_end_pipeline.py
  │   ├── test_historical_week_pipeline.py
  │   └── __init__.py
  ├── api/
  │   ├── test_endpoints.py
  │   └── __init__.py
  ├── ui/
  │   ├── test_html_snapshots.py
  │   └── __init__.py
  ├── conftest.py
  ├── test_runner.py
  └── __init__.py
  ```

**Naming:**
- Test files: `test_*.py` (e.g., `test_elo_and_probabilities.py`)
- Test classes: `Test*` (e.g., `class TestEloRatingSystem`)
- Test functions: `test_*` (e.g., `def test_initial_ratings()`)

## Test Structure

**Suite Organization:**
```python
class TestEloRatingSystem:
    """Test the Elo rating system implementation."""

    def test_initial_ratings(self):
        """Test initial Elo ratings are set correctly."""
        elo_system = EloRatingSystem()
        initial_rating = elo_system.get_rating('BUF')
        assert initial_rating == 1500

    def test_elo_update_win(self):
        """Test Elo update for a win."""
        elo_system = EloRatingSystem()
        elo_system.ratings['TEAM_A'] = 1500
        elo_system.ratings['TEAM_B'] = 1500

        elo_system.update_ratings(
            home_team='TEAM_A',
            away_team='TEAM_B',
            home_score=28,
            away_score=21,
            season=2024,
            week=1
        )

        team_a_new = elo_system.get_rating('TEAM_A')
        team_b_new = elo_system.get_rating('TEAM_B')

        assert team_a_new > 1500  # Winner gains points
        assert team_b_new < 1500  # Loser loses points
```

**Patterns:**

1. **Arrange-Act-Assert**: Tests follow clear setup → action → verification
   ```python
   def test_elo_update_win(self):
       # Arrange
       elo_system = EloRatingSystem()
       elo_system.ratings['TEAM_A'] = 1500

       # Act
       elo_system.update_ratings(home_team='TEAM_A', away_team='TEAM_B', ...)

       # Assert
       assert elo_system.get_rating('TEAM_A') > 1500
   ```

2. **Setup not teardown**: Fixtures handle setup, rarely need teardown
   - Example `mock_games_data` fixture (conftest.py:44-74) creates fresh data each test
   - Temporary directories cleaned via context manager

3. **Assertion patterns**: Use specific assertions, not generic truthy checks
   ```python
   # Good
   assert team_a_new > team_a_initial, "Winner should gain Elo points"
   assert 1 <= data["current_week"] <= 22, "Week should be valid NFL week"

   # Avoid
   assert team_a_new  # Too vague
   assert data  # Doesn't specify what to check
   ```

## Mocking

**Framework:** unittest.mock built-in
- `from unittest.mock import Mock, patch`
- Accessed via pytest fixtures

**Patterns:**

1. **Service mocking for API tests** (`tests/api/test_endpoints.py` lines 31-56):
   ```python
   def test_current_week_endpoint(self, client, mock_predictions_data):
       with patch('api.services.get_current_week_info') as mock_service:
           mock_service.return_value = {
               'current_week': mock_predictions_data['current_week'],
               'current_season': mock_predictions_data['current_season'],
           }

           response = client.get("/current-week")
           assert response.status_code == 200
   ```

2. **Patch external dependencies**: Patch at import point where used
   ```python
   with patch('scripts.health_check.HealthChecker') as mock_checker:
       checker = mock_checker()
       # ... test code
   ```

3. **Fixture-based mocking**: Create reusable mock data as pytest fixtures
   ```python
   @pytest.fixture
   def mock_predictions_data():
       return {
           'current_week': 3,
           'current_season': 2024,
           'games': [...]
       }
   ```

**What to Mock:**
- External API calls (nfl_data_py, meteostat, odds APIs)
- File system operations for unit tests
- Database connections
- System time (for date-dependent tests)

**What NOT to Mock:**
- Core business logic (Elo calculations, feature engineering)
- Probability transforms and statistical functions
- Data validation logic
- Local utility functions

## Fixtures and Factories

**Test Data:**
```python
@pytest.fixture
def mock_games_data():
    """Create mock games data for testing."""
    et_tz = pytz.timezone('America/New_York')

    mock_games = []
    for week in range(1, 4):
        week_games = [
            {
                'game_id': f'MOCK_2024_W{week:02d}_BUF@MIA',
                'season': 2024,
                'week': week,
                'home_team': 'MIA',
                'away_team': 'BUF',
                'home_score': 24 if week <= 2 else None,
                'away_score': 21 if week <= 2 else None
            },
            # ... more games
        ]
        mock_games.extend(week_games)

    return pd.DataFrame(mock_games)

@pytest.fixture
def sample_feature_matrix():
    """Create a sample feature matrix with realistic data."""
    np.random.seed(42)  # For reproducible tests

    games = [
        {'game_id': 'TEST_2024_W01_BUF@MIA', 'season': 2024, 'week': 1},
        # ...
    ]

    features = []
    for game in games:
        feature_row = game.copy()
        feature_row.update({
            'home_elo_rating': np.random.uniform(1400, 1600),
            'away_elo_rating': np.random.uniform(1400, 1600),
            # ... more features
        })
        features.append(feature_row)

    return pd.DataFrame(features)
```

**Location:**
- Central `tests/conftest.py` contains all shared fixtures (lines 20-330)
- Session-scoped fixtures for expensive setup: `temp_data_dir`, `project_root_path`
- Function-scoped fixtures for test data: `mock_games_data`, `mock_predictions_data`
- Auto-used fixtures: `setup_test_environment` (line 324) runs for every test

**Fixture Scope:**
- `scope="session"` - one per test session (e.g., temp directories)
- `scope="function"` - default, one per test function (e.g., mock data)
- `autouse=True` - automatically used without explicit parameter (e.g., environment setup)

## Coverage

**Requirements:**
- Enforced minimum: 80% (`--cov-fail-under=80` in `pyproject.toml`)
- Measured modules: `scripts`, `ratings`, `models`, `backtest`, `api`

**View Coverage:**
```bash
pytest --cov=scripts,ratings,models,backtest,api --cov-report=term-missing
pytest --cov=scripts,ratings,models,backtest,api --cov-report=html
# HTML output available in htmlcov/index.html
```

**Configuration** (`pyproject.toml` lines 191-213):
```toml
[tool.coverage.run]
source = ["scripts", "ratings", "models", "backtest", "api"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__init__.py",
    "*/conftest.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

## Test Types

**Unit Tests** (`tests/unit/`):
- Scope: Individual functions and classes in isolation
- Approach: Mock dependencies, test single responsibility
- Examples in `test_elo_and_probabilities.py`:
  - `test_initial_ratings()` - tests default initialization
  - `test_elo_update_win()` - tests rating updates
  - `test_elo_uncertainty_decay()` - tests uncertainty convergence
- Location: `tests/unit/test_*.py`

**Integration Tests** (`tests/integration/`):
- Scope: End-to-end pipeline workflows
- Approach: Real data, minimal mocking, test data flow
- Examples:
  - `test_end_to_end_pipeline.py` - tests full ingestion → features → model → predictions
  - `test_historical_week_pipeline.py` - tests walk-forward on historical data
- Location: `tests/integration/test_*.py`

**API/Contract Tests** (`tests/api/`):
- Scope: Endpoint contracts and response schemas
- Approach: Mock service layer, test response structure
- Example `test_endpoints.py`:
  ```python
  def test_games_endpoint(self, client, mock_predictions_data):
      with patch('api.services.get_games_with_predictions') as mock_service:
          mock_service.return_value = mock_predictions_data['games']

          response = client.get("/games")
          assert response.status_code == 200
          data = response.json()

          if len(data) > 0:
              game = data[0]
              self._validate_game_prediction_schema(game)
  ```

**UI/Snapshot Tests** (`tests/ui/`):
- Scope: HTML template rendering
- Approach: BeautifulSoup parsing, snapshot comparisons
- Location: `tests/ui/test_html_snapshots.py`

## Common Patterns

**Async Testing:**
- pytest-asyncio handles async functions automatically (`asyncio_mode = "auto"`)
- Test async functions normally: `async def test_async_feature()`
- Example pattern:
  ```python
  @pytest.mark.asyncio
  async def test_async_initialization(self):
      from api.main import initialize_models
      await initialize_models()
      # assertions
  ```

**Error Testing:**
```python
def test_validation_error(self):
    """Test that invalid input raises validation error."""
    with pytest.raises(HTTPException) as exc_info:
        validate_team_param("INVALID_TEAM")

    assert exc_info.value.status_code == 422
    assert "Team abbreviation" in str(exc_info.value.detail)

def test_not_found_error(self):
    """Test that missing resource raises 404."""
    with patch('api.services.get_game_by_id') as mock_service:
        mock_service.return_value = None

        response = client.get("/games/NONEXISTENT_GAME")
        assert response.status_code == 404
```

**DataFrame Testing:**
```python
def test_feature_matrix_structure(self):
    """Test feature matrix has required columns."""
    builder = FeatureMatrixBuilder()
    result = builder.build_features(mock_games_data)

    # Use assertion helper
    assert_dataframe_structure(
        result,
        expected_columns=['game_id', 'season', 'week', 'home_team', 'away_team'],
        min_rows=1
    )

    # Check specific feature columns
    assert 'home_elo_rating' in result.columns
    assert 'elo_diff' in result.columns

def test_probability_range(self):
    """Test probability predictions are in valid range."""
    predictions = model.predict(feature_matrix)

    # Use assertion helper
    assert_probability_range(predictions['wp_prob_home'])
    assert_probability_range(predictions['ats_prob_home'])
```

**Data Leakage Testing:**
```python
def test_no_future_data_leakage(self, sample_feature_matrix):
    """Test that features don't use future information."""
    # Ensure game_date is before feature timestamp
    assert_no_data_leakage(
        sample_feature_matrix,
        prediction_date=pd.Timestamp('2024-09-07')
    )
```

## TestClient for FastAPI

**Setup:**
```python
@pytest.fixture
def api_client():
    """Create a test client for the FastAPI application."""
    from api.main import app
    return TestClient(app)
```

**Usage:**
```python
def test_root_endpoint(self, api_client):
    response = api_client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "name" in data
    assert data["name"] == "NFL Prediction API"

def test_with_query_params(self, api_client):
    response = api_client.get("/games?week=3&season=2024")
    assert response.status_code == 200
```

---

*Testing analysis: 2026-03-18*
