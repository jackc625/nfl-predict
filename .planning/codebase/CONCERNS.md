# Codebase Concerns

**Analysis Date:** 2026-03-18

## Tech Debt

**Broad Exception Handling:**
- Issue: Excessive use of `except Exception as e` catches without specific exception types across the codebase
- Files: `api/services.py`, `api/main.py`, `models/train_wp.py`, `models/train_ats.py`, `models/train_ou.py`, `backtest/metrics.py`, `backtest/walkforward.py`
- Impact: Masks specific failures, makes debugging harder, can hide critical issues as warnings rather than errors. Makes it difficult to handle recoverable errors differently from fatal ones.
- Fix approach: Replace broad catches with specific exception types (ValueError, FileNotFoundError, etc.). Create custom exception hierarchy for domain-specific errors already defined in `api/exceptions.py` and use them consistently throughout.

**In-Memory Fallback Database:**
- Issue: When DuckDB file is missing or connection fails, code silently falls back to in-memory database
- Files: `api/services.py` lines 47-57 (_get_db_connection method)
- Impact: Queries on in-memory database won't have actual data, returning empty results and masking infrastructure failures. Users see no data instead of clear error message about missing database.
- Fix approach: Instead of silent fallback, raise `DataNotFoundError` or `ModelUnavailableError` to fail fast and clearly. Let API return 503 Service Unavailable rather than misleading empty results.

**Fallback HTML Report Generation:**
- Issue: Multiple fallback HTML report generators return hardcoded placeholder content
- Files: `api/services.py` lines 1361-1427 (_generate_fallback_html_report method)
- Impact: Users see fake data when backtest results are missing, no clear indication that results are unavailable. Masks data pipeline failures.
- Fix approach: Return structured error response instead of HTML fallback. Let API clearly communicate missing backtest data via HTTP 404 or 503.

## Known Bugs

**Broad Exception Catch in Recommendations:**
- Symptoms: Missing recommendations don't raise errors, silently return None or empty list
- Files: `api/main.py` line 857, `api/services.py` line 468
- Trigger: When recommendation engine is unavailable or encounters error
- Workaround: Check if recommendations is None before using. Currently no validation.

**Datetime Normalization Assumption:**
- Symptoms: Timezone-naive datetime columns are assumed to be UTC already
- Files: `data/storage.py` lines 118-133 (_normalize_datetime_columns method)
- Trigger: When loading data with timezone-naive datetime from non-UTC sources
- Workaround: Ensure all data sources provide UTC timestamps explicitly. Add validation to detect timezone assumptions.

**Season/Week Inference Hardcoded:**
- Symptoms: Current season/week calculation uses fixed date offsets that may not match actual NFL schedule
- Files: `api/services.py` lines 75-128 (_infer_current_season_week method)
- Trigger: When using dates that don't align with standard NFL season (preseason, schedule changes)
- Workaround: For production, should query actual game schedule from database instead of inferring from date alone.

## Security Considerations

**Secrets Configuration:**
- Risk: No validation that required secrets are present at startup
- Files: `api/config.py`, `conf/settings.py`
- Current mitigation: Settings use Pydantic for validation, but only if explicitly required
- Recommendations: Add startup validation that checks for all required environment variables (ODDS_API_KEY, SECRET_KEY) and fails fast with clear error if missing. Never proceed with default/empty secrets.

**SQL Injection Prevention:**
- Risk: Manual SQL query construction in multiple places
- Files: `api/services.py` lines 134-143 (manual WHERE clause construction), `data/storage.py` (table name sanitization)
- Current mitigation: Table names are validated. Column values are parameterized in some places.
- Recommendations: Use parameterized queries throughout. Replace string concatenation with proper parameterization for all WHERE/ORDER BY clauses.

**Debug Mode in Production:**
- Risk: DEBUG flag and verbose error responses could leak sensitive information
- Files: `api/config.py` line 22, `conf/settings.py` line 251, `deployment/uvicorn.conf.py` line 98
- Current mitigation: Controlled via environment variable
- Recommendations: Ensure DEBUG is False in production. Audit error responses to remove sensitive details (file paths, internal data structures).

## Performance Bottlenecks

**Synchronous Database Queries on Each Request:**
- Problem: Every API request opens fresh DuckDB connection and reads data from disk
- Files: `api/services.py` (_get_db_connection method, _load_games_data, _load_predictions_data, etc.)
- Cause: No connection pooling, no caching of frequently-accessed data (games list, team info, etc.)
- Improvement path: Implement connection pooling with reusable connections. Cache static data (teams, venues) at startup. Use read-only mode consistently to avoid locking. Consider read replicas for high-concurrency scenarios.

**Full DataFrame Load for Filtering:**
- Problem: Code loads entire dataset into memory then filters, instead of filtering at database level
- Files: `api/services.py` lines 130-155 (loads all games then filters in Python)
- Cause: DuckDB queries built with string concatenation, unclear if indexes exist
- Improvement path: Move filtering to SQL WHERE clauses. Verify DuckDB indexes on season, week, game_id columns. Add query explain plans to understand performance.

**Large Model Artifacts:**
- Problem: Multiple large XGBoost/sklearn models loaded into memory for each prediction
- Files: `models/train_wp.py`, `models/train_ats.py`, `models/train_ou.py`
- Cause: Models persist to disk as joblib files, loaded fresh each time
- Improvement path: Implement model caching with LRU eviction. Consider model quantization for XGBoost. Lazy load models only when predictions requested.

**Backtest Report Generation:**
- Problem: Generating full HTML reports with charts happens synchronously during request
- Files: `api/services.py` line 1340 (BacktestReporter.generate_full_report), `backtest/reporting.py`
- Cause: No pre-generation, no caching of report
- Improvement path: Pre-generate reports as part of backtest pipeline. Cache generated reports. Serve pre-built HTML instead of generating on-demand.

## Fragile Areas

**Feature Leakage Validation:**
- Files: `features/validation.py` (check_data_leakage method), `models/train_wp.py`, `models/train_ats.py`, `models/train_ou.py`
- Why fragile: Complex column name pattern matching (keywords like 'closing_line', 'final', 'actual') is brittle. Easy to introduce leakage through subtle naming (e.g., 'final_spread_vs_opening' vs 'opening_spread_vs_closing'). Manual validation catches some cases but not all.
- Safe modification: Add deterministic whitelist of allowed columns and their expected ranges. Run feature validation with warnings promoted to errors during backtest. Add unit tests for each known leakage pattern.
- Test coverage: Limited test coverage for edge cases in `features/validation.py`. No tests for new feature types (like derived features).

**Walk-Forward Backtesting Temporal Ordering:**
- Files: `backtest/walkforward.py` (validates temporal order), `models/train_wp.py` (relies on it)
- Why fragile: Leakage detection depends on date columns being present and properly sorted. Scattered validation across multiple methods. If date column is missing, validation skips silently with warning.
- Safe modification: Make date columns required in all feature matrices. Fail loudly if validation columns missing. Add explicit assertion statements at training/prediction boundaries.
- Test coverage: `tests/integration/test_historical_week_pipeline.py` tests overall pipeline but limited edge case coverage for unusual season structures.

**Model Serialization/Deserialization:**
- Files: `models/utils.py` (ModelManager), `models/train_wp.py`, `models/train_ats.py`, `models/train_ou.py`
- Why fragile: joblib/pickle serialization of sklearn/XGBoost models can break if library versions change. No version validation when loading. Metadata stored separately as JSON, risk of getting out of sync.
- Safe modification: Pin sklearn/xgboost versions tightly. Add model version checks and compatibility layer. Consider ONNX format for cross-library compatibility. Store model version in artifact and validate on load.
- Test coverage: No tests for model loading with mismatched versions or corrupted artifacts.

**Recommendation Generation Logic:**
- Files: `utils/bet_recommender.py`, `utils/unit_sizing.py`, `api/services.py`
- Why fragile: Multiple recommendation tiers (high/medium/low) with complex Kelly Criterion calculations. If edge threshold or confidence thresholds change, behavior changes unexpectedly. Kelly fraction hardcoded in multiple places.
- Safe modification: Centralize all threshold constants in configuration. Add logging of threshold decisions. Unit test each confidence tier with known inputs. Validate Kelly calculations against reference implementations.
- Test coverage: `scripts/test_api_todo_fixes.py` tests basic flow but not edge cases of Kelly sizing.

## Scaling Limits

**Single DuckDB File Bottleneck:**
- Current capacity: Single read-only connection per request. Read-only mode prevents write conflicts but serializes writes.
- Limit: Will hit file locking issues with >10 concurrent readers writing (during backtest). Backup/migration operations require downtime.
- Scaling path: Migrate to networked database (PostgreSQL) or implement write queue with batch updates. Use DuckDB MotherDuck for cloud scaling if needed.

**Backtest Computation:**
- Current capacity: 7 seasons × 18 weeks ÷ 5 weeks per split = ~25 model trainings. Takes hours with current XGBoost configs.
- Limit: Adding more data sources or longer backtest history will exceed reasonable run time. Walk-forward with many splits becomes prohibitively slow.
- Scaling path: Implement parallel split processing. Consider incremental training. Move to distributed framework (Ray, Spark) for very large backtests. Cache intermediate features.

**Model Inference Latency:**
- Current capacity: ~100ms per game with 3 models (WP, ATS, O/U). Can handle <10 games/second.
- Limit: Web UI with 100+ games in one week will be slow. Adding more complex models (ensemble) will compound latency.
- Scaling path: Batch inference across games. Cache predictions. Consider model quantization. Deploy models to lightweight inference server (ONNXRuntime).

## Dependencies at Risk

**XGBoost Model Training:**
- Risk: XGBoost binary format not guaranteed stable across versions. Hyperparameter interfaces change.
- Impact: Backtest results become unreproducible with version upgrades. Need to retrain all models.
- Migration plan: Lock version to `xgboost==2.0.3`. Document any hyperparameter assumptions. Plan upgrade cycle with full retraining.

**nfl_data_py Data Source:**
- Risk: External library maintained by third party, no guarantees on data accuracy or API changes.
- Impact: Ingestion pipeline breaks if source changes schema or deprecates endpoints.
- Migration plan: Add validation layer checking data schema. Pin version. Monitor GitHub for deprecation warnings. Have fallback data source (ESPN API).

**scikit-learn Logistic Regression:**
- Risk: Library interfaces stable but serialization format can break.
- Impact: WP model artifacts won't load with different sklearn versions.
- Migration plan: Version lock sklearn to `1.3.2`. Use ONNX export for long-term portability.

**meteostat Weather Data:**
- Risk: Free weather API, no SLA, data coverage varies by location and time period.
- Impact: Weather features unavailable for some games, model degrades gracefully or fails.
- Migration plan: Add fallback to generic weather model (temperature only by month). Cache historical weather. Add monitoring for data availability.

## Missing Critical Features

**Data Backup and Recovery:**
- Problem: No backup mechanism for DuckDB database or Parquet artifacts
- Blocks: Unable to recover from data corruption, accidental deletion, or drive failure
- Recommendation: Implement daily backup to cloud storage (S3). Test recovery procedure monthly.

**Model Explainability:**
- Problem: Feature importance tracked but not easily accessible via API
- Blocks: Users can't understand why a model made a specific prediction
- Recommendation: Add SHAP values calculation. Create `/games/{game_id}/explanation` endpoint showing top contributing features.

**Configuration Validation:**
- Problem: Settings loaded but no validation that required keys exist or have sensible values
- Blocks: Configuration errors discovered at runtime rather than startup
- Recommendation: Add pydantic validators to check API keys, paths exist, numbers in reasonable ranges. Fail fast on startup if config invalid.

**Monitoring and Alerting:**
- Problem: No health checks for data freshness, model drift, prediction validity
- Blocks: Can't detect data pipeline failures until manually checked
- Recommendation: Implement data quality checks as part of ingestion. Track model calibration drift. Alert if prediction distributions change significantly.

**Bet Recommendation Explainability:**
- Problem: Edge calculation and Kelly sizing not transparent to user
- Blocks: Users can't validate or override recommendations
- Recommendation: Return calculation details with recommendations (edge_pct, kelly_fraction_used, confidence). Allow custom edge threshold in request.

## Test Coverage Gaps

**API Error Handling:**
- What's not tested: 503 responses when models unavailable, 404 when game not found, validation errors for invalid team names
- Files: `api/main.py`, `api/services.py` (exception handling paths)
- Risk: Silent failures returning wrong status codes or malformed error responses
- Priority: High

**Feature Leakage Edge Cases:**
- What's not tested: Derived features with subtle leakage patterns, new feature types not in validation keyword list
- Files: `features/validation.py`, custom feature builders
- Risk: Leakage introduced unknowingly, backtest results overestimated
- Priority: High

**Datetime and Timezone Handling:**
- What's not tested: Timezone-aware vs naive datetime mixing, DST transitions, leap years, offseason date calculation
- Files: `api/services.py`, `data/storage.py`, `utils/date_utils.py`
- Risk: Wrong season/week inference during edge periods (offseason, preseason)
- Priority: Medium

**Database Failure Modes:**
- What's not tested: DuckDB file corruption, missing files, read-only filesystem, disk space exhaustion
- Files: `data/storage.py`, `api/services.py`
- Risk: Unhandled exceptions crash API, no graceful degradation
- Priority: Medium

**Model Serialization Compatibility:**
- What's not tested: Loading models trained with different sklearn/xgboost versions, corrupted model files, missing metadata
- Files: `models/utils.py`, artifact loading code
- Risk: Models fail to load, predictions unavailable with unclear error
- Priority: Medium

**Concurrent Request Handling:**
- What's not tested: Multiple simultaneous API requests during backtest pipeline running
- Files: `api/main.py`, `data/storage.py` (read-only DuckDB connection)
- Risk: File locking issues, inconsistent results if data being written
- Priority: Low (current architecture read-only, but important for future writes)

---

*Concerns audit: 2026-03-18*
