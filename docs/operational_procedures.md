# NFL Prediction System - Operational Procedures

> **Note:** This document was written during a prior development effort and predates the current
> roadmap. Some information may be outdated or incorrect. See `.planning/` for the authoritative
> project documentation, architecture decisions, and current development state.

This document outlines the operational procedures for monitoring, alerting, and recovery of the NFL Prediction System in production environments.

## Table of Contents

1. [System Health Monitoring](#system-health-monitoring)
2. [Alert Management](#alert-management)
3. [Recovery Procedures](#recovery-procedures)
4. [Performance Monitoring](#performance-monitoring)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Emergency Contacts](#emergency-contacts)

---

## System Health Monitoring

### Automated Monitoring

The system runs automated monitoring checks every 5 minutes via `operational_monitoring.py`:

```bash
# Run full monitoring cycle
python scripts/operational_monitoring.py --mode full

# Run in daemon mode (continuous monitoring)
python scripts/operational_monitoring.py --daemon --interval 300

# Monitor specific components
python scripts/operational_monitoring.py --mode data    # Data quality only
python scripts/operational_monitoring.py --mode models # Model drift only
```

### Key Metrics Monitored

#### Data Quality Metrics
- **Data Freshness**: Maximum 6 hours for critical tables
- **Data Completeness**: Minimum 14 games per week
- **Odds Coverage**: Minimum 90% odds availability
- **Business Rules**: No duplicate games, valid week numbers
- **Data Consistency**: Cross-table referential integrity

#### Model Performance Metrics
- **Prediction Accuracy**: Track accuracy degradation >3%
- **Model Calibration**: Minimum reliability score of 0.8
- **Feature Drift**: Monitor for >20% distribution changes
- **Prediction Distributions**: Flag extreme predictions >10%
- **Processing Time**: Maximum 60 seconds per prediction cycle

#### System Health Metrics
- **API Response Times**: Sub-second response targets
- **Database Connectivity**: Connection pool health
- **File System**: Disk space and I/O performance
- **Memory Usage**: Monitor for memory leaks

---

## Alert Management

### Alert Severity Levels

#### 🔴 **CRITICAL** - Immediate Action Required
- System completely down
- Zero predictions generated
- Database corruption
- Security breach

**Response Time**: < 15 minutes
**Escalation**: Immediate page/call

#### 🟠 **ERROR** - Action Required Within Hours
- Data pipeline failures
- Model accuracy drop >5%
- Missing critical data
- API returning errors

**Response Time**: < 2 hours
**Escalation**: Email + Slack

#### 🟡 **WARNING** - Action Required Within Day
- Data quality issues
- Model accuracy drop 3-5%
- Performance degradation
- Non-critical component failures

**Response Time**: < 24 hours
**Escalation**: Slack notification

#### 🟢 **INFO** - For Awareness Only
- Successful deployments
- Routine maintenance
- Performance improvements

**Response Time**: No immediate action
**Escalation**: Log entry only

### Alert Routing

Alerts are routed based on severity and component:

```yaml
# Critical: All channels
critical: ["email", "slack", "console"]

# Error: Electronic notification
error: ["slack", "console"]

# Warning: Console only
warning: ["console"]

# Info: Log only
info: ["console"]
```

### Alert Rate Limiting

- **Maximum**: 20 alerts per hour per component
- **Cooldown**: 30 minutes for repeat alerts
- **Suppression**: Identical alerts within cooldown period

---

## Recovery Procedures

### 1. Data Pipeline Failures

#### **Scenario**: Missing or Stale Data

**Symptoms**:
- Data age >6 hours
- Missing games for current week
- Empty data tables

**Recovery Steps**:

1. **Diagnose the Issue**:
   ```bash
   # Check data freshness
   python scripts/data_qa.py --check-freshness

   # Verify data sources
   python scripts/ingest_games.py --dry-run --season 2024 --week current
   ```

2. **Manual Data Refresh**:
   ```bash
   # Re-ingest games data
   python scripts/ingest_games.py --season 2024 --week current --force

   # Re-ingest odds data
   python scripts/ingest_odds.py --snapshot "$(date -Iseconds)" --force

   # Re-ingest weather data
   python scripts/ingest_weather.py --season 2024 --week current --force
   ```

3. **Rebuild Features**:
   ```bash
   # Rebuild feature matrices
   python scripts/build_features.py --season 2024 --week current --force
   ```

4. **Verify Recovery**:
   ```bash
   # Run data quality checks
   python scripts/operational_monitoring.py --mode data
   ```

**Prevention**:
- Monitor external API status
- Implement exponential backoff for API calls
- Set up data source redundancy

---

### 2. Model Performance Degradation

#### **Scenario**: Model Accuracy Drop >5%

**Symptoms**:
- Sustained accuracy decline
- Poor calibration scores
- Unusual prediction distributions

**Recovery Steps**:

1. **Assess the Degradation**:
   ```bash
   # Run model drift analysis
   python scripts/operational_monitoring.py --mode models

   # Check recent backtest results
   python backtest/walkforward.py --analyze-recent --weeks 4
   ```

2. **Identify Root Cause**:
   ```bash
   # Check feature drift
   python scripts/validate_features.py --compare-recent

   # Analyze prediction patterns
   python models/evaluation.py --analyze-predictions
   ```

3. **Model Recovery Options**:

   **Option A: Retrain Models**:
   ```bash
   # Retrain with recent data
   python models/train_wp.py --season 2024 --week current
   python models/train_ats.py --season 2024 --week current
   python models/train_ou.py --season 2024 --week current
   ```

   **Option B: Rollback to Previous Models**:
   ```bash
   # Restore previous model artifacts
   cp artifacts/models/wp_model_backup.pkl artifacts/models/wp_model.pkl
   cp artifacts/models/ats_model_backup.pkl artifacts/models/ats_model.pkl
   cp artifacts/models/ou_model_backup.pkl artifacts/models/ou_model.pkl
   ```

   **Option C: Adjust Model Parameters**:
   ```bash
   # Recalibrate models
   python models/calibrate.py --recalibrate --target all
   ```

4. **Validate Recovery**:
   ```bash
   # Run prediction pipeline test
   python models/prediction_pipeline.py --test --week current

   # Verify accuracy improvement
   python scripts/operational_monitoring.py --mode models
   ```

**Prevention**:
- Weekly model retraining schedule
- A/B testing for model updates
- Gradual rollout of model changes

---

### 3. API/System Failures

#### **Scenario**: API Unresponsive or Returning Errors

**Symptoms**:
- HTTP 500/503 errors
- Timeout responses
- Connection refused

**Recovery Steps**:

1. **Check System Health**:
   ```bash
   # Check API health
   curl http://localhost:8000/health

   # Check process status
   ps aux | grep uvicorn

   # Check system resources
   df -h  # Disk space
   free -h  # Memory usage
   ```

2. **Restart Services**:
   ```bash
   # Stop current API
   pkill -f uvicorn

   # Restart API server
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Database Recovery**:
   ```bash
   # Check database connection
   python -c "from data.storage import get_db_connection; print(get_db_connection().execute('SELECT 1').fetchone())"

   # Restart database if needed
   # (Commands depend on database system)
   ```

4. **Verify Recovery**:
   ```bash
   # Test API endpoints
   python scripts/test_api_endpoints.py

   # Check system monitoring
   python scripts/operational_monitoring.py --mode full
   ```

**Prevention**:
- Health check endpoints
- Load balancing
- Automatic restart on failure
- Resource monitoring and alerts

---

### 4. Prediction Pipeline Failures

#### **Scenario**: No Predictions Generated

**Symptoms**:
- Empty prediction files
- Pipeline crashes
- Feature engineering failures

**Recovery Steps**:

1. **Diagnose Pipeline State**:
   ```bash
   # Check pipeline logs
   tail -f logs/prediction_pipeline.log

   # Verify input data
   python scripts/validate_features.py --check-inputs
   ```

2. **Run Pipeline Components Individually**:
   ```bash
   # Test feature building
   python scripts/build_features.py --season 2024 --week current --debug

   # Test model loading
   python models/prediction_pipeline.py --test-models

   # Test prediction generation
   python models/prediction_pipeline.py --predict --week current --debug
   ```

3. **Full Pipeline Recovery**:
   ```bash
   # Clean and rebuild
   rm -rf outputs/predictions/current_*
   make predict  # Or run full prediction pipeline
   ```

4. **Verify Predictions**:
   ```bash
   # Check prediction outputs
   python scripts/validate_backtest_reproducibility.py --check-current

   # Test API integration
   curl http://localhost:8000/games | jq '.games[0].predictions'
   ```

**Prevention**:
- Input validation at each stage
- Graceful error handling
- Pipeline checkpoints
- Automated testing of pipeline components

---

## Performance Monitoring

### Key Performance Indicators (KPIs)

#### System Performance
- **API Response Time**: < 500ms (95th percentile)
- **Prediction Generation**: < 60 seconds per full cycle
- **Data Ingestion**: < 10 minutes per source
- **Model Training**: < 30 minutes per model

#### Prediction Quality
- **WP Accuracy**: > 65% (historical baseline)
- **ATS Accuracy**: > 52% (better than random)
- **Calibration Score**: > 0.8 (well-calibrated)
- **Log Loss**: Monitor for increases >10%

#### Business Metrics
- **Prediction Coverage**: 100% of NFL games
- **Data Freshness**: < 6 hours for all sources
- **Uptime**: > 99.5% availability
- **Edge Detection**: Identify >2% edge opportunities

### Performance Monitoring Commands

```bash
# Check current performance
python scripts/operational_monitoring.py --mode full --output perf_report.json

# Analyze recent trends
python backtest/reporting.py --performance-trends --weeks 4

# Generate performance dashboard
python scripts/create_performance_dashboard.py --output-dir dashboards/
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. "Data table 'games' is stale"
**Cause**: External API failure or network issues
**Solution**: Re-run data ingestion with `--force` flag

#### 2. "Model accuracy dropped by X%"
**Cause**: Model drift, data quality issues, or external factors
**Solution**: Analyze recent changes, retrain models, or rollback

#### 3. "API returning 500 errors"
**Cause**: Database connection issues, model loading failures
**Solution**: Check logs, restart services, verify dependencies

#### 4. "Prediction distributions are extreme"
**Cause**: Model artifacts corrupted, feature scaling issues
**Solution**: Regenerate models, check feature preprocessing

#### 5. "High missing odds percentage"
**Cause**: Odds provider API issues, timing problems
**Solution**: Check odds API status, adjust timing, use backup sources

### Log Analysis

#### Key Log Locations
- **API Logs**: `logs/api.log`
- **Pipeline Logs**: `logs/prediction_pipeline.log`
- **Data Ingestion**: `logs/data_ingestion.log`
- **Alert Logs**: `outputs/alerts.jsonl`

#### Log Analysis Commands
```bash
# Find errors in last hour
grep -E "(ERROR|CRITICAL)" logs/*.log | grep "$(date -d '1 hour ago' '+%Y-%m-%d %H')"

# Monitor real-time logs
tail -f logs/api.log | grep -E "(ERROR|WARNING)"

# Analyze alert patterns
jq '.level' outputs/alerts.jsonl | sort | uniq -c
```

---

## Emergency Contacts

### Escalation Matrix

#### Primary On-Call (0-2 hours)
- **System Admin**: [Phone], [Email]
- **Lead Developer**: [Phone], [Email]

#### Secondary On-Call (2-24 hours)
- **Data Engineer**: [Phone], [Email]
- **ML Engineer**: [Phone], [Email]

#### Management (24+ hours)
- **Technical Lead**: [Email]
- **Product Owner**: [Email]

### External Contacts

#### Data Providers
- **NFL Data API**: [Support Email], [Documentation]
- **Odds API**: [Support Email], [Documentation]
- **Weather API**: [Support Email], [Documentation]

#### Infrastructure
- **Cloud Provider**: [Support Portal], [Account Manager]
- **Database**: [Support Channel], [Documentation]

---

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily
- [ ] Review alert dashboard
- [ ] Check data freshness
- [ ] Verify prediction generation
- [ ] Monitor API performance

#### Weekly
- [ ] Analyze model performance trends
- [ ] Review and clear old log files
- [ ] Update team on system status
- [ ] Plan any necessary model retraining

#### Monthly
- [ ] Full system health review
- [ ] Update operational procedures
- [ ] Review and test recovery procedures
- [ ] Capacity planning assessment

#### Quarterly
- [ ] Disaster recovery testing
- [ ] Security audit
- [ ] Performance optimization review
- [ ] Infrastructure cost review

### Maintenance Windows

**Preferred Schedule**: Sunday 2:00-4:00 AM ET
**Backup Schedule**: Wednesday 1:00-3:00 AM ET

**Pre-Maintenance Checklist**:
- [ ] Notify stakeholders 48 hours in advance
- [ ] Backup all critical data and models
- [ ] Prepare rollback procedures
- [ ] Test maintenance procedures in staging

**Post-Maintenance Checklist**:
- [ ] Verify all systems operational
- [ ] Run full monitoring cycle
- [ ] Confirm prediction generation
- [ ] Update maintenance log

---

## Documentation Updates

This document should be reviewed and updated:
- After any system changes
- Following incident resolution
- Quarterly operational review
- When new team members join

**Last Updated**: 2024-09-15
**Version**: 1.0
**Next Review**: 2024-12-15