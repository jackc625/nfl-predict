# NFL Prediction System - Deployment Guide

> **Note:** This document was written during a prior development effort and predates the current
> roadmap. Some information may be outdated or incorrect. See `.planning/` for the authoritative
> project documentation, architecture decisions, and current development state.

This directory contains deployment configurations for the NFL Prediction System API, supporting development, Docker, and manual production setups.

## Quick Start

### Development
```bash
# Start development server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or using the production runner in development mode
python deployment/run_production.py --config .env.example
```

### Docker (Recommended)
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f nfl-predict-api

# Health check
curl http://localhost/health
```

### Manual Production Setup
```bash
# Copy and configure environment
cp deployment/production.env .env
# Edit .env with your configuration

# Start production server
python deployment/run_production.py --config .env
```

## File Overview

### Core Configuration
- **`production.env`** - Production environment variables template
- **`gunicorn.conf.py`** - Gunicorn WSGI server configuration
- **`uvicorn.conf.py`** - Uvicorn ASGI server configuration

### Security & Performance
- **`security.py`** - Security middleware and configurations
- **`static_config.py`** - Static file serving with caching
- **`nginx.conf`** - Nginx reverse proxy configuration

### Containerization
- **`Dockerfile`** - Multi-stage Docker build
- **`docker-compose.yml`** - Complete stack deployment
- **`docker-entrypoint.sh`** - Container initialization script

### Scheduling & Automation
- **`setup_scheduling.py`** - Cross-platform scheduling setup utility
- **`crontab.txt`** - Linux/macOS cron configuration template
- **`windows_scheduler.xml`** - Windows Task Scheduler XML definitions
- **`../.github/workflows/friday-production.yml`** - GitHub Actions workflow

## Scheduling & Automation

The system operates on a two-stage Friday schedule:

### Friday 5:00 PM ET - Data Update Pipeline

**Purpose**: Prepare data for the 6 PM odds snapshot

**Steps**:
1. Ingest current week games data
2. Ingest weather forecasts
3. Data quality validation
4. Update Elo ratings
5. Build team form metrics
6. Build contextual features
7. Build weather features
8. Prepare for odds snapshot

**Duration**: ~5-15 minutes (must complete before 6 PM)

### Friday 6:00 PM ET - Odds Snapshot & Predictions

**Purpose**: Main production run -- generate predictions

**Steps**:
1. Capture odds snapshot at exactly 6:00 PM ET
2. Build market anchor features
3. Create feature matrices
4. Validate features for data leakage
5. Validate prediction models
6. Generate predictions (WP, ATS, O/U)
7. Generate bet recommendations
8. Export prediction artifacts
9. Validate outputs
10. System health check

**Duration**: ~10-30 minutes

### Setting Up Scheduling

#### Windows (Task Scheduler)
```bash
# Dry run to see what would be created
python deployment/setup_scheduling.py --platform windows --dry-run

# Install the scheduled tasks (requires administrator privileges)
python deployment/setup_scheduling.py --platform windows --install
```

#### Linux/macOS (Cron)
```bash
# Generate customized crontab
python deployment/setup_scheduling.py --platform linux --dry-run

# Install crontab (manual step)
crontab deployment/crontab_custom.txt
```

#### GitHub Actions (Optional)
1. Configure repository secrets: `ODDS_API_KEY`, `WEATHER_API_KEY`
2. Enable the workflow in `.github/workflows/friday-production.yml`
3. Test with manual dispatch

### Testing Automation Scripts
```bash
# Test automation scripts
python deployment/setup_scheduling.py --test --dry-run

# Test individual scripts
python scripts/friday_data_update.py --dry-run
python scripts/friday_predictions_run.py --dry-run
```

## Configuration

### Environment Variables

```bash
# Core Application
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-secure-secret-key-here
API_HOST=0.0.0.0
API_PORT=8000

# Timezone (for scheduling)
TZ=America/New_York

# External APIs
ODDS_API_KEY=your-odds-api-key
WEATHER_API_KEY=your-weather-api-key

# Security
CORS_ORIGINS=https://yourdomain.com
RATE_LIMIT_ENABLED=true
API_RATE_LIMIT_PER_MINUTE=60

# Performance
GUNICORN_WORKERS=4
REDIS_URL=redis://localhost:6379/0

# Monitoring
ENABLE_EMAIL_NOTIFICATIONS=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Paths
- **Project Home**: `/path/to/nfl-predict`
- **Python Path**: `/path/to/nfl-predict/.venv/bin/python`
- **Log Directory**: `logs/`
- **Output Directory**: `outputs/predictions/`

## Security

The deployment includes comprehensive security:

- Content Security Policy (CSP)
- Strict Transport Security (HSTS)
- Rate limiting per IP and endpoint
- API key authentication for protected endpoints
- Security headers and CORS configuration

### Security Checklist

- [ ] Change default SECRET_KEY
- [ ] Configure CORS origins for production
- [ ] Enable HTTPS with valid certificates
- [ ] Set up rate limiting
- [ ] Configure firewall rules
- [ ] Enable audit logging
- [ ] Store API keys in environment variables, never in code
- [ ] Use least-privilege user accounts for scheduled tasks
- [ ] Regularly rotate API keys

## Health Checks

Built-in health check endpoints:

- **`/health`** - Basic application health
- **`/ready`** - Readiness check for load balancers
- **`/live`** - Liveness check for orchestrators

## Monitoring

### Log Files
- `logs/friday_data_update.json` - Data update execution log
- `logs/friday_predictions_run.json` - Predictions execution log
- `logs/cron_*.log` - Cron/scheduler logs

### Health Checks
```bash
# Manual health check
python scripts/health_check.py --mode comprehensive

# Operational monitoring
python scripts/operational_monitoring.py --comprehensive

# Check scheduling status
python deployment/setup_scheduling.py --status
```

### Expected Prediction Outputs
- `outputs/predictions/current_week_predictions.parquet`
- `outputs/predictions/current_week_predictions.json`
- `outputs/predictions/current_week_recommendations.json`
- `outputs/predictions/current_week_summary.csv`

## Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find process using port
sudo lsof -i :8000
```

**SSL Certificate Issues**
```bash
# Check certificate validity
openssl x509 -in certificate.crt -text -noout
```

**Script fails with import errors**
- Ensure virtual environment is activated
- Check Python path in scheduling configuration
- Verify all dependencies are installed

**Timezone issues**
- Set `TZ=America/New_York` environment variable
- Verify system timezone configuration
- Check cron daemon timezone settings

**Permission errors**
- Ensure script files are executable
- Check file/directory permissions
- Verify scheduler runs with appropriate user

**Network timeouts**
- Check internet connectivity
- Verify API keys are valid and not rate-limited
- Increase timeout values if needed

### Manual Execution
```bash
# Run manually with full logging
python scripts/friday_data_update.py --log-level DEBUG
python scripts/friday_predictions_run.py --log-level DEBUG

# Check execution logs
cat logs/friday_data_update.json | jq
cat logs/friday_predictions_run.json | jq
```

For additional help, see the project documentation in `.planning/`.
