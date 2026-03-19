# Operational Runbooks

> **Note:** This document was written during a prior development effort and predates the current
> roadmap. Some information may be outdated or incorrect. See `.planning/` for the authoritative
> project documentation, architecture decisions, and current development state.

This document provides comprehensive operational procedures for running and maintaining the NFL Prediction System in production. It covers regular operations, troubleshooting, maintenance, and emergency procedures.

## Table of Contents

1. [System Overview](#system-overview)
2. [Weekly Operations](#weekly-operations)
3. [Model Training & Updates](#model-training--updates)
4. [Data Pipeline Operations](#data-pipeline-operations)
5. [API & Web Service Operations](#api--web-service-operations)
6. [Monitoring & Alerting](#monitoring--alerting)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Maintenance Procedures](#maintenance-procedures)
9. [Emergency Procedures](#emergency-procedures)
10. [Performance Optimization](#performance-optimization)

## System Overview

### Architecture Components
- **Data Pipeline**: Bronze → Silver → Gold data transformation
- **Model Training**: WP, ATS, O/U model training and validation
- **Prediction API**: FastAPI service for real-time predictions
- **Web Interface**: User-facing prediction dashboard
- **Monitoring**: Health checks, alerts, and performance tracking

### Key Dependencies
- **Python 3.9+**: Core runtime environment
- **DuckDB**: Data storage and querying
- **FastAPI**: Web service framework
- **ML Libraries**: scikit-learn, XGBoost (LightGBM removed in Phase 1)
- **External APIs**: TheOddsAPI, weather services

### File Structure
```
nfl-predict/
├── data/           # Bronze/Silver/Gold data storage
├── models/         # Trained model artifacts
├── outputs/        # Prediction outputs and reports
├── scripts/        # Operational scripts
├── api/           # FastAPI application
├── web/           # Web interface
└── conf/          # Configuration files
```

## Weekly Operations

### Standard Weekly Schedule

#### Tuesday: Data Validation
**Time**: 9:00 AM ET
**Purpose**: Validate previous week's data integrity and outcomes

```bash
# Validate game results and update records
python scripts/validate_game_results.py --week current-1

# Check for data quality issues
python scripts/operational_monitoring.py --check-data-quality

# Generate data quality report
python scripts/generate_weekly_report.py --type data-quality
```

**Expected Outputs**:
- Data validation report in `outputs/reports/`
- Updated silver tables with verified results
- Alert notifications for any data discrepancies

#### Wednesday: Feature Refresh
**Time**: 10:00 AM ET
**Purpose**: Update all feature calculations with latest data

```bash
# Refresh team form metrics
python features/team_form.py --update-weekly

# Update Elo ratings
python features/elo_features.py --update-weekly

# Rebuild contextual features
python features/contextual.py --update-weekly

# Validate feature consistency
python scripts/validate_features.py --week current
```

**Validation Checks**:
- Feature value ranges within expected bounds
- No missing values for critical features
- Feature correlation matrix stability
- Historical consistency checks

#### Thursday: Model Updates
**Time**: 11:00 AM ET
**Purpose**: Retrain models with latest data (if needed)

```bash
# Check model performance drift
python scripts/model_drift_detection.py --weeks 4

# Retrain models if drift detected
python models/train_wp.py --update-incremental
python models/train_ats.py --update-incremental
python models/train_ou.py --update-incremental

# Validate model performance
python scripts/validate_models.py --type performance
```

**Thresholds for Retraining**:
- Accuracy drop > 3% over 4 weeks
- Calibration error increase > 0.05
- Feature importance drift > 20%

#### Friday 6:00 PM ET: Prediction Generation
**Time**: 6:00 PM ET (Market Close)
**Purpose**: Generate predictions for upcoming games

```bash
# Capture market snapshot
python scripts/capture_market_snapshot.py --timestamp "$(date '+%Y-%m-%dT18:00:00-04:00')"

# Update weather forecasts
python scripts/update_weather_forecasts.py --week current+1

# Generate predictions
python scripts/generate_predictions.py --week current+1

# Deploy to API
python scripts/deploy_predictions.py --week current+1

# Generate prediction reports
python scripts/generate_prediction_report.py --week current+1
```

**Critical Path Items**:
- Market data snapshot must complete before prediction generation
- All prediction models must be available and validated
- API deployment must complete before Saturday morning

#### Saturday: Pre-Game Validation
**Time**: 8:00 AM ET
**Purpose**: Final validation before games begin

```bash
# Validate prediction completeness
python scripts/validate_predictions.py --week current+1

# Check API health
python scripts/health_check.py --comprehensive

# Verify web interface
python scripts/test_web_interface.py

# Send readiness notification
python scripts/send_notification.py --type "system-ready"
```

### Weekend Monitoring

#### Game Day Operations
**Frequency**: Every 2 hours during games
**Purpose**: Monitor system performance and prediction accuracy

```bash
# Check API response times
curl -w "@curl-format.txt" -s -o /dev/null "http://localhost:8000/health"

# Monitor prediction usage
python scripts/monitor_api_usage.py --real-time

# Check for any system alerts
python scripts/check_system_alerts.py
```

**Key Metrics to Monitor**:
- API response time < 500ms
- Prediction API uptime > 99.5%
- Web interface availability
- Database connection health

## Model Training & Updates

### Seasonal Model Retraining

#### Full Retraining Schedule
**When**: End of each NFL season (February)
**Duration**: 4-6 hours
**Purpose**: Complete model refresh with full season data

```bash
# Backup current models
python scripts/backup_models.py --season $(date +%Y)

# Run full backtesting
python scripts/run_full_backtest.py --start-season 2018 --end-season $(date +%Y)

# Train new models
python models/train_wp.py --full-training --season $(date +%Y)
python models/train_ats.py --full-training --season $(date +%Y)
python models/train_ou.py --full-training --season $(date +%Y)

# Validate new models
python scripts/validate_new_models.py --compare-previous

# Deploy if validation passes
python scripts/deploy_new_models.py --season $(date +%Y)
```

#### Incremental Updates
**When**: Weekly (Thursday) if drift detected
**Duration**: 30-60 minutes
**Purpose**: Update models with recent data

```bash
# Incremental training with last 4 weeks
python models/train_wp.py --incremental --weeks 4
python models/train_ats.py --incremental --weeks 4
python models/train_ou.py --incremental --weeks 4

# A/B test new models
python scripts/ab_test_models.py --duration 1-week

# Deploy if A/B test shows improvement
python scripts/deploy_incremental_update.py
```

### Model Validation Procedures

#### Performance Validation
```bash
# Compare against baseline models
python scripts/compare_model_performance.py --baseline market-odds

# Validate calibration quality
python scripts/validate_calibration.py --weeks 8

# Check feature importance stability
python scripts/validate_feature_importance.py --compare-historical

# Generate model comparison report
python scripts/generate_model_report.py --type validation
```

**Validation Criteria**:
- **Accuracy**: Must exceed previous model by ≥1% or stay within 1%
- **Calibration**: ECE must be ≤0.05 for all models
- **Stability**: Feature importance changes ≤20% from previous version
- **Performance**: Prediction time ≤100ms per game

#### Rollback Procedures
If new models fail validation:

```bash
# Immediate rollback to previous version
python scripts/rollback_models.py --to-previous

# Investigate issues
python scripts/diagnose_model_issues.py --verbose

# Fix issues and re-validate
python scripts/fix_and_revalidate.py

# Document incident
python scripts/create_incident_report.py --type model-rollback
```

## Data Pipeline Operations

### Daily Data Ingestion

#### Morning Data Update (8:00 AM ET)
```bash
# Check for new game results
python scripts/check_new_results.py --date $(date -d "yesterday" +%Y-%m-%d)

# Ingest updated data
python scripts/ingest_daily_updates.py --date $(date +%Y-%m-%d)

# Validate data consistency
python scripts/validate_daily_data.py --date $(date +%Y-%m-%d)

# Update bronze tables
python scripts/update_bronze_tables.py --incremental
```

#### Data Transformation Pipeline
```bash
# Transform bronze to silver
python scripts/bronze_to_silver.py --date $(date +%Y-%m-%d)

# Build gold features
python scripts/silver_to_gold.py --features all --date $(date +%Y-%m-%d)

# Validate transformations
python scripts/validate_transformations.py --date $(date +%Y-%m-%d)

# Update feature store
python scripts/update_feature_store.py --date $(date +%Y-%m-%d)
```

### Data Quality Monitoring

#### Automated Quality Checks
```bash
# Check for missing data
python scripts/check_missing_data.py --tolerance 5%

# Validate data ranges
python scripts/validate_data_ranges.py --strict

# Check for duplicates
python scripts/check_duplicates.py --tables all

# Verify referential integrity
python scripts/check_referential_integrity.py
```

**Quality Thresholds**:
- Missing data: ≤5% for any critical field
- Outliers: ≤1% of records flagged as extreme outliers
- Duplicates: 0 duplicates allowed in primary keys
- Consistency: Cross-table joins must have ≥95% match rate

#### Data Recovery Procedures

**Missing Data Recovery**:
```bash
# Identify missing data sources
python scripts/identify_missing_sources.py --date $(date +%Y-%m-%d)

# Attempt automated recovery
python scripts/auto_recover_data.py --sources api,backup

# Manual data entry for critical missing data
python scripts/manual_data_entry.py --interactive

# Validate recovered data
python scripts/validate_recovered_data.py
```

**Corruption Recovery**:
```bash
# Detect data corruption
python scripts/detect_corruption.py --validate-checksums

# Restore from backup
python scripts/restore_from_backup.py --date $(date -d "1 day ago" +%Y-%m-%d)

# Re-run transformations
python scripts/rerun_transformations.py --from-backup

# Verify data integrity
python scripts/verify_data_integrity.py --full-check
```

## API & Web Service Operations

### API Service Management

#### Startup Procedures
```bash
# Start API service
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Verify health endpoint
curl http://localhost:8000/health

# Run smoke tests
python scripts/api_smoke_tests.py --comprehensive

# Monitor initial performance
python scripts/monitor_api_startup.py --duration 300
```

#### Production Deployment
```bash
# Pre-deployment validation
python scripts/validate_api_deployment.py --environment production

# Deploy with zero downtime
python scripts/rolling_deployment.py --api --zero-downtime

# Validate deployment
python scripts/validate_deployment.py --checks all

# Monitor post-deployment
python scripts/monitor_deployment.py --duration 3600
```

#### Service Monitoring
```bash
# Check service health every 5 minutes
*/5 * * * * python scripts/health_check.py --alert-on-failure

# Monitor API performance
python scripts/monitor_api_performance.py --real-time

# Check prediction quality
python scripts/monitor_prediction_quality.py --window 24h

# Generate service reports
python scripts/generate_service_report.py --daily
```

### Web Interface Operations

#### Frontend Deployment
```bash
# Build production assets
cd web && npm run build

# Deploy to web server
python scripts/deploy_frontend.py --environment production

# Run integration tests
python scripts/test_web_integration.py --comprehensive

# Validate user flows
python scripts/validate_user_flows.py
```

#### Performance Monitoring
```bash
# Monitor page load times
python scripts/monitor_web_performance.py --metrics all

# Check mobile responsiveness
python scripts/test_mobile_interface.py

# Validate accessibility
python scripts/test_accessibility.py --standards wcag2.1

# Monitor user engagement
python scripts/monitor_user_engagement.py --real-time
```

## Monitoring & Alerting

### Health Check System

#### System Health Monitoring
```bash
# Comprehensive health check
python scripts/health_check.py --comprehensive

# Database connectivity
python scripts/check_database_health.py

# Model availability
python scripts/check_model_health.py

# External dependencies
python scripts/check_external_dependencies.py
```

#### Alert Configuration

**Critical Alerts** (Immediate Response Required):
- API downtime > 5 minutes
- Database connection failure
- Model prediction errors > 10%
- Data pipeline failure

**Warning Alerts** (Response within 2 hours):
- API response time > 1 second
- Model performance drift > 2%
- Data quality issues
- Disk space > 80%

**Info Alerts** (Daily review):
- Weekly performance summaries
- Feature drift reports
- Usage statistics
- Capacity planning metrics

#### Alert Response Procedures

**Critical Alert Response**:
1. **Immediate Assessment** (within 5 minutes)
   ```bash
   # Quick system status check
   python scripts/emergency_status_check.py --brief
   ```

2. **Root Cause Analysis** (within 15 minutes)
   ```bash
   # Detailed diagnostics
   python scripts/comprehensive_diagnostics.py --emergency
   ```

3. **Recovery Actions** (within 30 minutes)
   ```bash
   # Automated recovery if possible
   python scripts/auto_recovery.py --emergency

   # Manual intervention if needed
   python scripts/manual_recovery_guide.py --issue $ISSUE_TYPE
   ```

4. **Post-Incident Review** (within 24 hours)
   ```bash
   # Generate incident report
   python scripts/generate_incident_report.py --alert-id $ALERT_ID
   ```

## Troubleshooting Guide

### Common Issues

#### API Performance Issues

**Symptoms**: Slow response times, timeout errors
**Diagnosis**:
```bash
# Check API performance metrics
python scripts/diagnose_api_performance.py --detailed

# Analyze response time distribution
python scripts/analyze_response_times.py --percentiles 50,90,95,99

# Check database query performance
python scripts/analyze_db_performance.py --slow-queries
```

**Solutions**:
1. **Scale API Workers**:
   ```bash
   # Increase worker count
   uvicorn api.main:app --workers 8
   ```

2. **Optimize Database Queries**:
   ```bash
   # Analyze and optimize slow queries
   python scripts/optimize_db_queries.py --auto-fix
   ```

3. **Enable Caching**:
   ```bash
   # Enable prediction caching
   python scripts/enable_prediction_cache.py --ttl 3600
   ```

#### Model Prediction Errors

**Symptoms**: Prediction failures, unexpected results
**Diagnosis**:
```bash
# Check model health
python scripts/diagnose_model_health.py --all-models

# Validate input data
python scripts/validate_prediction_inputs.py --sample 100

# Check feature pipeline
python scripts/diagnose_feature_pipeline.py --verbose
```

**Solutions**:
1. **Model Rollback**:
   ```bash
   # Rollback to previous working version
   python scripts/rollback_models.py --to-last-known-good
   ```

2. **Feature Validation**:
   ```bash
   # Fix feature pipeline issues
   python scripts/fix_feature_pipeline.py --auto-repair
   ```

3. **Model Retraining**:
   ```bash
   # Emergency model retraining
   python scripts/emergency_retrain.py --fast-mode
   ```

#### Data Pipeline Failures

**Symptoms**: Stale data, transformation errors
**Diagnosis**:
```bash
# Check pipeline status
python scripts/diagnose_pipeline_status.py --full-trace

# Validate data sources
python scripts/validate_data_sources.py --connectivity

# Check transformation logic
python scripts/validate_transformations.py --debug
```

**Solutions**:
1. **Restart Pipeline**:
   ```bash
   # Restart failed pipeline stage
   python scripts/restart_pipeline_stage.py --stage $FAILED_STAGE
   ```

2. **Data Recovery**:
   ```bash
   # Recover from backup
   python scripts/recover_pipeline_data.py --from-backup
   ```

3. **Manual Override**:
   ```bash
   # Manual data correction
   python scripts/manual_data_correction.py --interactive
   ```

### Diagnostic Tools

#### Performance Profiling
```bash
# Profile API endpoints
python scripts/profile_api_endpoints.py --duration 300

# Profile model inference
python scripts/profile_model_inference.py --iterations 1000

# Profile database operations
python scripts/profile_database_ops.py --queries all
```

#### System Resource Monitoring
```bash
# Monitor CPU usage
python scripts/monitor_cpu_usage.py --threshold 80

# Monitor memory usage
python scripts/monitor_memory_usage.py --alert-on-pressure

# Monitor disk I/O
python scripts/monitor_disk_io.py --detailed
```

#### Log Analysis
```bash
# Analyze error logs
python scripts/analyze_error_logs.py --last 24h

# Check warning patterns
python scripts/analyze_warning_patterns.py --frequency

# Generate log summary
python scripts/generate_log_summary.py --daily
```

## Maintenance Procedures

### Regular Maintenance Tasks

#### Weekly Maintenance (Sunday 2:00 AM ET)
```bash
# Database optimization
python scripts/optimize_database.py --vacuum --analyze

# Log rotation
python scripts/rotate_logs.py --keep-days 30

# Temporary file cleanup
python scripts/cleanup_temp_files.py --older-than 7-days

# Update dependencies (non-breaking)
python scripts/update_dependencies.py --patch-only
```

#### Monthly Maintenance (First Sunday)
```bash
# Full database maintenance
python scripts/full_database_maintenance.py

# Model performance review
python scripts/monthly_model_review.py --generate-report

# Capacity planning analysis
python scripts/analyze_capacity_needs.py --forecast 3-months

# Security updates
python scripts/apply_security_updates.py --test-first
```

#### Quarterly Maintenance
```bash
# System security audit
python scripts/security_audit.py --comprehensive

# Performance benchmarking
python scripts/performance_benchmark.py --compare-historical

# Disaster recovery testing
python scripts/test_disaster_recovery.py --full-simulation

# Documentation review
python scripts/review_documentation.py --update-needed
```

### Backup Procedures

#### Daily Backups
```bash
# Backup database
python scripts/backup_database.py --incremental --encrypt

# Backup model artifacts
python scripts/backup_models.py --daily

# Backup configuration
python scripts/backup_configuration.py --version-control

# Verify backup integrity
python scripts/verify_backup_integrity.py --daily
```

#### Weekly Full Backups
```bash
# Full system backup
python scripts/full_system_backup.py --compress --offsite

# Backup validation
python scripts/validate_full_backup.py --restore-test

# Backup rotation
python scripts/rotate_backups.py --keep-weeks 12
```

### Update Procedures

#### Dependency Updates
```bash
# Check for updates
python scripts/check_dependency_updates.py --security-only

# Test updates in staging
python scripts/test_dependency_updates.py --environment staging

# Apply updates
python scripts/apply_dependency_updates.py --backup-first

# Validate system after updates
python scripts/validate_post_update.py --comprehensive
```

#### Code Deployment
```bash
# Pre-deployment checks
python scripts/pre_deployment_checks.py --all

# Deploy with rollback capability
python scripts/deploy_with_rollback.py --version $NEW_VERSION

# Post-deployment validation
python scripts/post_deployment_validation.py --timeout 300

# Monitor post-deployment
python scripts/monitor_post_deployment.py --duration 3600
```

## Emergency Procedures

### System Recovery

#### Complete System Failure
1. **Immediate Response**:
   ```bash
   # Activate emergency mode
   python scripts/activate_emergency_mode.py

   # Assess system status
   python scripts/emergency_system_assessment.py
   ```

2. **Recovery Strategy**:
   ```bash
   # Determine recovery approach
   python scripts/determine_recovery_strategy.py --auto

   # Execute recovery plan
   python scripts/execute_recovery_plan.py --strategy $STRATEGY
   ```

3. **Service Restoration**:
   ```bash
   # Restore core services
   python scripts/restore_core_services.py --priority-order

   # Validate restoration
   python scripts/validate_service_restoration.py
   ```

#### Data Loss Recovery
1. **Assess Data Loss**:
   ```bash
   # Identify lost data
   python scripts/assess_data_loss.py --comprehensive

   # Determine recovery options
   python scripts/data_recovery_options.py --prioritize
   ```

2. **Execute Recovery**:
   ```bash
   # Restore from backups
   python scripts/restore_from_backup.py --target-date $LAST_GOOD_DATE

   # Reconstruct missing data
   python scripts/reconstruct_missing_data.py --auto-methods
   ```

3. **Validate Recovery**:
   ```bash
   # Validate data integrity
   python scripts/validate_data_integrity.py --post-recovery

   # Test system functionality
   python scripts/test_system_functionality.py --comprehensive
   ```

### Communication Procedures

#### Incident Communication
```bash
# Send initial alert
python scripts/send_incident_alert.py --severity $SEVERITY --stakeholders all

# Update stakeholders
python scripts/update_incident_status.py --incident-id $ID --status "$STATUS"

# Send resolution notification
python scripts/send_resolution_notification.py --incident-id $ID
```

#### Status Page Updates
```bash
# Update public status page
python scripts/update_status_page.py --status "$STATUS" --message "$MESSAGE"

# Schedule maintenance notifications
python scripts/schedule_maintenance_notification.py --advance-notice 24h
```

## Performance Optimization

### API Optimization

#### Response Time Optimization
```bash
# Analyze slow endpoints
python scripts/analyze_slow_endpoints.py --threshold 500ms

# Optimize database queries
python scripts/optimize_api_queries.py --auto-index

# Implement caching
python scripts/implement_smart_caching.py --strategy adaptive

# Load balancing optimization
python scripts/optimize_load_balancing.py --auto-tune
```

#### Resource Optimization
```bash
# Memory usage optimization
python scripts/optimize_memory_usage.py --profile-based

# CPU usage optimization
python scripts/optimize_cpu_usage.py --async-where-possible

# I/O optimization
python scripts/optimize_io_operations.py --batch-where-applicable
```

### Model Optimization

#### Inference Speed Optimization
```bash
# Profile model inference
python scripts/profile_model_inference.py --detailed

# Optimize feature computation
python scripts/optimize_feature_computation.py --cache-expensive

# Model quantization
python scripts/quantize_models.py --preserve-accuracy

# Batch prediction optimization
python scripts/optimize_batch_predictions.py --auto-tune
```

#### Memory Optimization
```bash
# Model size optimization
python scripts/optimize_model_size.py --compress-weights

# Feature memory optimization
python scripts/optimize_feature_memory.py --sparse-matrices

# Garbage collection tuning
python scripts/tune_garbage_collection.py --model-workload
```

### Database Optimization

#### Query Optimization
```bash
# Analyze query performance
python scripts/analyze_query_performance.py --identify-bottlenecks

# Auto-create missing indexes
python scripts/create_missing_indexes.py --analyze-usage

# Query plan optimization
python scripts/optimize_query_plans.py --cost-based

# Partition optimization
python scripts/optimize_partitions.py --time-based
```

#### Storage Optimization
```bash
# Data compression optimization
python scripts/optimize_data_compression.py --auto-select

# Archive old data
python scripts/archive_old_data.py --keep-months 24

# Storage cleanup
python scripts/cleanup_storage.py --reclaim-space

# Capacity planning
python scripts/storage_capacity_planning.py --forecast 12-months
```

---

*This operational runbook should be reviewed and updated quarterly to ensure procedures remain current with system changes and operational learnings.*