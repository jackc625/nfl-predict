# NFL Prediction System - API Documentation

> **Note:** This document was written during a prior development effort and predates the current
> roadmap. Some information may be outdated or incorrect. See `.planning/` for the authoritative
> project documentation, architecture decisions, and current development state.

This document provides comprehensive documentation for the NFL Prediction System API, including all endpoints, request/response formats, authentication, rate limiting, and usage examples.

## Table of Contents

1. [Overview](#overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [Base URLs & Environments](#base-urls--environments)
4. [API Endpoints](#api-endpoints)
5. [Request/Response Formats](#requestresponse-formats)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Code Examples](#code-examples)
9. [SDK & Integrations](#sdk--integrations)

---

## Overview

The NFL Prediction System API provides access to:
- **Game Predictions**: Win probability, spread, and over/under predictions
- **Historical Data**: Past games, odds, and weather information
- **Model Analytics**: Backtest results, calibration data, and performance metrics
- **Team Statistics**: Team performance metrics and historical trends
- **Betting Recommendations**: Edge-based betting suggestions with risk analysis

### API Features
- **RESTful Design**: Standard HTTP methods and status codes
- **JSON Responses**: All data returned in JSON format with optional CSV downloads
- **Real-time Data**: Updated predictions and odds within 6 hours
- **Historical Coverage**: 6+ seasons of backtest data (2018-2024)
- **Comprehensive Filtering**: Filter by teams, weeks, seasons, and performance metrics

---

## Authentication & Authorization

### API Keys (Future Implementation)
```http
Authorization: Bearer your-api-key-here
```

**Current Status**: No authentication required (development phase)
**Production**: Will implement API key authentication with rate limiting per key

### Access Levels
- **Public**: Basic game data and predictions
- **Premium**: Detailed analytics, historical data, betting recommendations
- **Enterprise**: Custom integrations, bulk data access, real-time webhooks

---

## Base URLs & Environments

### Development
```
http://localhost:8000
```

### Production (Future)
```
https://api.nfl-predictions.com/v1
```

### Health Check
```http
GET /health
```
Always check this endpoint before making other requests to ensure API availability.

---

## API Endpoints

### 🏥 Health & Status

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-09-15T18:00:00Z",
  "database": "connected",
  "models": "loaded",
  "prediction_pipeline": "operational"
}
```

#### Service Information
```http
GET /info
```

**Response:**
```json
{
  "service": "NFL Prediction System",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "last_prediction_update": "2024-09-15T18:00:00Z",
  "data_coverage": {
    "seasons": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    "total_games": 2847,
    "total_predictions": 2847
  }
}
```

---

### 📅 Schedule & Timing

#### Current Week Metadata
```http
GET /current-week
```

**Response:**
```json
{
  "current_season": 2024,
  "current_week": 3,
  "week_type": "regular",
  "predictions_available": true,
  "last_updated": "2024-09-15T18:00:00Z",
  "games_this_week": 16,
  "prediction_cutoff": "2024-09-22T18:00:00Z"
}
```

---

### 🏈 Games & Predictions

#### List Games
```http
GET /games?season=2024&week=3&team=KC&limit=50&offset=0
```

**Query Parameters:**
- `season` (int): Filter by season year
- `week` (int): Filter by week number (1-22)
- `team` (str): Filter by team abbreviation (e.g., "KC", "BUF")
- `limit` (int): Number of results to return (default: 50, max: 100)
- `offset` (int): Number of results to skip (default: 0)

**Response:**
```json
{
  "games": [
    {
      "game_id": "2024_03_KC_BUF",
      "season": 2024,
      "week": 3,
      "home_team": "BUF",
      "away_team": "KC",
      "game_time": "2024-09-22T13:00:00Z",
      "venue": "Highmark Stadium",
      "weather": {
        "temperature": 72,
        "wind_speed": 8,
        "conditions": "Partly Cloudy"
      },
      "predictions": {
        "win_probability": {
          "home": 0.58,
          "away": 0.42,
          "confidence": 0.16
        },
        "spread": {
          "predicted_line": -2.5,
          "cover_probability": 0.52,
          "market_line": -3.0,
          "edge": 0.04
        },
        "total": {
          "predicted_total": 47.5,
          "over_probability": 0.48,
          "market_total": 49.0,
          "edge": -0.02
        }
      },
      "betting_recommendations": [
        {
          "bet_type": "spread",
          "recommendation": "BUF -3.0",
          "edge": 4.2,
          "kelly_fraction": 0.021,
          "confidence": "medium"
        }
      ]
    }
  ],
  "metadata": {
    "total_games": 16,
    "returned": 1,
    "has_more": false,
    "filters_applied": ["season=2024", "week=3"]
  }
}
```

#### Game Detail
```http
GET /games/{game_id}
```

**Path Parameters:**
- `game_id` (str): Unique game identifier (format: YYYY_WW_AWAY_HOME)

**Response:**
```json
{
  "game": {
    "game_id": "2024_03_KC_BUF",
    "season": 2024,
    "week": 3,
    "home_team": "BUF",
    "away_team": "KC",
    "game_time": "2024-09-22T13:00:00Z",
    "venue": "Highmark Stadium",
    "weather": {
      "temperature": 72,
      "wind_speed": 8,
      "humidity": 65,
      "conditions": "Partly Cloudy",
      "precipitation_chance": 10
    },
    "team_stats": {
      "home": {
        "elo_rating": 1520,
        "form_4_week": {
          "wins": 3,
          "avg_margin": 8.5,
          "offensive_epa": 0.12,
          "defensive_epa": -0.08
        }
      },
      "away": {
        "elo_rating": 1580,
        "form_4_week": {
          "wins": 4,
          "avg_margin": 12.3,
          "offensive_epa": 0.18,
          "defensive_epa": -0.15
        }
      }
    },
    "predictions": {
      "win_probability": {
        "home": 0.58,
        "away": 0.42,
        "confidence": 0.16,
        "model_version": "wp_v1.0"
      },
      "spread": {
        "predicted_line": -2.5,
        "cover_probability": 0.52,
        "market_line": -3.0,
        "edge": 0.04,
        "confidence": 0.08
      },
      "total": {
        "predicted_total": 47.5,
        "over_probability": 0.48,
        "under_probability": 0.52,
        "market_total": 49.0,
        "edge_over": -0.02,
        "edge_under": 0.02
      }
    },
    "fair_odds": {
      "moneyline_home": -138,
      "moneyline_away": 117,
      "spread_home": -110,
      "spread_away": -110,
      "over": 104,
      "under": -104
    },
    "betting_recommendations": [
      {
        "bet_type": "spread",
        "recommendation": "BUF -3.0",
        "edge": 4.2,
        "kelly_fraction": 0.021,
        "confidence": "medium",
        "risk_level": "low"
      }
    ]
  }
}
```

#### Week Games
```http
GET /weeks/{season}/{week}/games?min_edge=0.02
```

**Path Parameters:**
- `season` (int): Season year
- `week` (int): Week number (1-22)

**Query Parameters:**
- `min_edge` (float): Minimum edge threshold for filtering (0.0-1.0)

**Response:** Same format as `/games` endpoint

---

### 📊 Analytics & Backtesting

#### Backtest Summary
```http
GET /backtest?start_season=2020&end_season=2023&model_type=wp
```

**Query Parameters:**
- `start_season` (int): Starting season for analysis
- `end_season` (int): Ending season for analysis
- `model_type` (str): Filter by model type ("wp", "ats", "ou")

**Response:**
```json
{
  "summary": {
    "total_predictions": 1442,
    "accuracy_by_model": {
      "wp": {
        "accuracy": 0.653,
        "log_loss": 0.623,
        "calibration_slope": 0.98
      },
      "ats": {
        "accuracy": 0.517,
        "roi": 0.034,
        "kelly_roi": 0.089
      },
      "ou": {
        "accuracy": 0.502,
        "roi": 0.012,
        "over_accuracy": 0.485
      }
    },
    "betting_performance": {
      "total_bets": 287,
      "winning_bets": 156,
      "win_rate": 0.543,
      "total_roi": 0.089,
      "max_drawdown": -0.12,
      "sharpe_ratio": 1.23
    },
    "seasonal_breakdown": [
      {
        "season": 2020,
        "wp_accuracy": 0.641,
        "ats_accuracy": 0.523,
        "ou_accuracy": 0.489
      }
    ]
  },
  "metadata": {
    "seasons_analyzed": [2020, 2021, 2022, 2023],
    "total_games": 1088,
    "filters_applied": ["start_season=2020", "end_season=2023"]
  }
}
```

#### Calibration Data
```http
GET /calibration?model_type=wp&bins=10
```

**Query Parameters:**
- `model_type` (str): Model to analyze ("wp", "ats", "ou")
- `bins` (int): Number of probability bins for analysis (default: 10)

**Response:**
```json
{
  "calibration": {
    "model_type": "wp",
    "overall_metrics": {
      "reliability": 0.977,
      "resolution": 0.089,
      "uncertainty": 0.249,
      "expected_calibration_error": 0.023
    },
    "probability_bins": [
      {
        "bin_center": 0.05,
        "predicted_prob": 0.05,
        "actual_freq": 0.048,
        "count": 23,
        "confidence_interval": [0.031, 0.065]
      }
    ],
    "reliability_diagram": {
      "x_values": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
      "y_values": [0.048, 0.142, 0.237, 0.341, 0.456, 0.548, 0.658, 0.742, 0.863, 0.952]
    }
  }
}
```

---

### 🏟️ Teams

#### List Teams
```http
GET /teams
```

**Response:**
```json
{
  "teams": [
    {
      "team_id": "KC",
      "full_name": "Kansas City Chiefs",
      "city": "Kansas City",
      "conference": "AFC",
      "division": "West",
      "logo_url": "https://logos.nfl.com/KC.png"
    }
  ]
}
```

#### Team Statistics
```http
GET /teams/{team_id}/stats?season=2024&weeks=4
```

**Path Parameters:**
- `team_id` (str): Team abbreviation

**Query Parameters:**
- `season` (int): Season year
- `weeks` (int): Number of recent weeks to analyze

**Response:**
```json
{
  "team": {
    "team_id": "KC",
    "full_name": "Kansas City Chiefs",
    "current_record": {"wins": 2, "losses": 1},
    "elo_rating": 1580,
    "form_metrics": {
      "offensive_epa_per_play": 0.18,
      "defensive_epa_per_play": -0.15,
      "third_down_conversion": 0.42,
      "red_zone_efficiency": 0.67
    },
    "historical_performance": {
      "home_record": {"wins": 8, "losses": 1},
      "away_record": {"wins": 6, "losses": 3},
      "vs_spread": {"covers": 9, "losses": 8, "pushes": 0}
    }
  }
}
```

---

### 🎯 Predictions & Recommendations

#### Week Predictions
```http
GET /predictions/week/{season}/{week}
```

**Response:**
```json
{
  "predictions": [
    {
      "game_id": "2024_03_KC_BUF",
      "predictions": {
        "win_probability": {"home": 0.58, "away": 0.42},
        "spread": {"line": -2.5, "confidence": 0.08},
        "total": {"points": 47.5, "confidence": 0.12}
      }
    }
  ],
  "metadata": {
    "prediction_time": "2024-09-15T18:00:00Z",
    "model_versions": {
      "wp": "1.0",
      "ats": "1.0",
      "ou": "1.0"
    }
  }
}
```

#### Betting Recommendations
```http
GET /recommendations/week/{season}/{week}?min_edge=0.03
```

**Query Parameters:**
- `min_edge` (float): Minimum edge threshold for recommendations

**Response:**
```json
{
  "recommendations": [
    {
      "game_id": "2024_03_KC_BUF",
      "bet_type": "spread",
      "recommendation": "BUF -3.0",
      "edge": 0.042,
      "kelly_fraction": 0.021,
      "confidence": "medium",
      "risk_assessment": {
        "risk_level": "low",
        "max_loss": 0.021,
        "expected_return": 0.042
      }
    }
  ],
  "portfolio_summary": {
    "total_recommendations": 3,
    "total_kelly_fraction": 0.048,
    "expected_roi": 0.034,
    "risk_level": "low"
  }
}
```

---

### 🌐 Web UI Endpoints

#### Home Page
```http
GET /
```
Returns HTML page with current week games and predictions.

#### Game Detail View
```http
GET /games/{game_id}/view
```
Returns HTML page with detailed game analysis.

#### Backtest Results View
```http
GET /backtest/view
```
Returns HTML page with interactive backtest results.

---

### 📥 Data Downloads

#### Download Games CSV
```http
GET /downloads/games?season=2024&week=3&format=csv
```

**Query Parameters:**
- Standard filtering parameters
- `format` (str): "csv" or "json"

**Response:** Streaming CSV/JSON download

#### Download Backtest Data
```http
GET /downloads/backtest?start_season=2020&format=csv
```

**Response:** Streaming CSV download with backtest results

---

## Request/Response Formats

### Standard Response Structure
```json
{
  "data": {},           // Main response data
  "metadata": {         // Response metadata
    "timestamp": "2024-09-15T18:00:00Z",
    "total_results": 16,
    "returned_results": 16,
    "has_more": false,
    "filters_applied": [],
    "api_version": "1.0"
  },
  "status": "success"   // Response status
}
```

### Pagination
```json
{
  "data": [...],
  "pagination": {
    "limit": 50,
    "offset": 0,
    "total": 256,
    "has_next": true,
    "has_previous": false,
    "next_url": "/games?limit=50&offset=50",
    "previous_url": null
  }
}
```

### Date/Time Formats
- **ISO 8601**: `2024-09-15T18:00:00Z`
- **Timezone**: All times in UTC
- **Game Times**: Local venue time with UTC conversion available

---

## Error Handling

### HTTP Status Codes
- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters or request format
- `401 Unauthorized`: Missing or invalid API key
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Valid format but invalid data
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Temporary service outage

### Error Response Format
```json
{
  "error": {
    "code": "INVALID_PARAMETERS",
    "message": "Week parameter must be between 1 and 22",
    "details": {
      "parameter": "week",
      "provided_value": 25,
      "valid_range": [1, 22]
    },
    "request_id": "req_123456789"
  },
  "status": "error",
  "timestamp": "2024-09-15T18:00:00Z"
}
```

### Common Error Codes
- `INVALID_PARAMETERS`: Invalid request parameters
- `RESOURCE_NOT_FOUND`: Requested resource doesn't exist
- `DATA_NOT_AVAILABLE`: Data not yet available for requested period
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `MODEL_NOT_READY`: Prediction models not loaded
- `MAINTENANCE_MODE`: System under maintenance

---

## Rate Limiting

### Current Limits (Development)
- **No limits**: Development phase

### Production Limits (Future)
- **Free Tier**: 100 requests/hour
- **Premium**: 1,000 requests/hour
- **Enterprise**: 10,000 requests/hour

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1694808000
Retry-After: 3600
```

---

## Code Examples

### Python
```python
import requests

# Get current week games
response = requests.get('http://localhost:8000/games?season=2024&week=3')
games = response.json()

# Get game detail
game_id = "2024_03_KC_BUF"
detail = requests.get(f'http://localhost:8000/games/{game_id}')
game_detail = detail.json()

# Get betting recommendations
recommendations = requests.get(
    'http://localhost:8000/recommendations/week/2024/3?min_edge=0.02'
)
bets = recommendations.json()
```

### JavaScript
```javascript
// Fetch current week games
const games = await fetch('http://localhost:8000/games?season=2024&week=3')
  .then(response => response.json());

// Get backtest data
const backtest = await fetch('http://localhost:8000/backtest?start_season=2020')
  .then(response => response.json());

// Download CSV data
const csvData = await fetch('http://localhost:8000/downloads/games?format=csv')
  .then(response => response.text());
```

### cURL
```bash
# Get current week metadata
curl -X GET "http://localhost:8000/current-week"

# Get games with filtering
curl -X GET "http://localhost:8000/games?season=2024&week=3&team=KC"

# Get calibration data
curl -X GET "http://localhost:8000/calibration?model_type=wp&bins=10"

# Download backtest CSV
curl -X GET "http://localhost:8000/downloads/backtest?format=csv" \
     -o backtest_results.csv
```

---

## SDK & Integrations

### Official Python SDK (Future)
```python
from nfl_predictions import NFLClient

client = NFLClient(api_key="your-key-here")

# Get games
games = client.games.list(season=2024, week=3)

# Get predictions
predictions = client.predictions.get_week(season=2024, week=3)

# Get recommendations
recommendations = client.recommendations.get_week(
    season=2024, week=3, min_edge=0.02
)
```

### Webhook Integration (Future)
```json
{
  "webhook_url": "https://your-app.com/webhooks/nfl-predictions",
  "events": ["predictions.updated", "recommendations.generated"],
  "filters": {
    "teams": ["KC", "BUF"],
    "min_edge": 0.03
  }
}
```

---

## Versioning

### API Versioning
- **Current**: v1.0 (implied in base URL)
- **Future**: Explicit versioning (`/v2/games`)

### Backward Compatibility
- **Deprecation**: 6-month notice for breaking changes
- **Legacy Support**: 12-month support for deprecated endpoints

---

## Support & Resources

### Documentation
- **API Docs**: This document
- **OpenAPI Spec**: `/docs` (Swagger UI)
- **Postman Collection**: Available in `/docs/postman/`

### Support Channels
- **GitHub Issues**: Bug reports and feature requests
- **Email**: api-support@nfl-predictions.com
- **Discord**: Community support and discussions

### Rate Updates
- **Model Updates**: Weekly retraining
- **Data Refresh**: Every 6 hours during season
- **API Updates**: Announced 48 hours in advance

---

**Last Updated**: 2024-09-15
**API Version**: 1.0
**Documentation Version**: 1.0