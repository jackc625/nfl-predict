# Model Methodology Documentation

> **Note:** This document was written during a prior development effort and predates the current
> roadmap. Some information may be outdated or incorrect. See `.planning/` for the authoritative
> project documentation, architecture decisions, and current development state.

This document provides a comprehensive overview of the machine learning methodologies employed in the NFL Prediction System. The system implements three distinct but interconnected prediction models: Win Probability (WP), Against the Spread (ATS), and Over/Under (O/U) models.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Win Probability (WP) Model](#win-probability-wp-model)
3. [Against the Spread (ATS) Model](#against-the-spread-ats-model)
4. [Over/Under (O/U) Model](#over-under-ou-model)
5. [Supporting Components](#supporting-components)
6. [Validation Framework](#validation-framework)
7. [Performance Metrics](#performance-metrics)
8. [Implementation Details](#implementation-details)

## System Architecture

### Modeling Philosophy

The NFL Prediction System employs a multi-model approach designed to predict different aspects of NFL game outcomes while maintaining statistical rigor and preventing data leakage. The architecture follows these core principles:

- **Temporal Ordering**: All models strictly respect chronological order using walk-forward validation
- **Feature Consistency**: Shared feature engineering pipeline ensures consistency across models
- **Probabilistic Output**: All models output calibrated probabilities rather than binary predictions
- **Modular Design**: Each model can be trained and evaluated independently
- **Reproducibility**: Fixed random seeds and deterministic algorithms ensure consistent results

### Model Interconnections

```
Raw Data → Feature Engineering → [WP Model, ATS Model, O/U Model] → Predictions API
                ↓
         Feature Validation → Calibration System → Performance Evaluation
```

## Win Probability (WP) Model

### Overview

The Win Probability model predicts the likelihood of the home team winning the game. It serves as the foundation model and uses logistic regression with advanced feature selection and probability calibration.

**Location**: `models/train_wp.py`
**Model Type**: Logistic Regression
**Output**: Binary probability (home team wins)

### Methodology

#### Algorithm Selection
- **Primary**: Logistic Regression with Elastic Net regularization
- **Rationale**: Provides interpretable coefficients, handles multicollinearity, and offers built-in feature selection through L1 regularization

#### Feature Selection Process
The model implements multiple feature selection strategies:

1. **Variance Threshold**: Removes features with variance < 0.01
2. **Univariate Selection**: Uses F-statistics to select top K features
3. **Model-based Selection**: Uses L1-regularized logistic regression for feature importance
4. **Recursive Feature Elimination (RFE)**: Iteratively removes least important features
5. **No Selection**: Uses all available features (for comparison)

**Default Configuration**: Recursive Feature Elimination with 15 features maximum

#### Hyperparameter Optimization
The model supports multiple hyperparameter tuning strategies:

**Grid Search Parameters**:
- **L1 Regularization**: `C=[0.001, 0.01, 0.1, 1, 10, 100]`, `solver=['liblinear', 'saga']`
- **L2 Regularization**: `C=[0.001, 0.01, 0.1, 1, 10, 100]`, `solver=['liblinear', 'lbfgs', 'saga']`
- **Elastic Net**: `C=[0.001, 0.01, 0.1, 1, 10]`, `l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9]`, `solver=['saga']`

**Evaluation Metric**: Negative log-loss with 3-fold stratified cross-validation

#### Training Process
1. **Data Preparation**: Feature cleaning, missing value imputation, constant feature removal
2. **Feature Scaling**: StandardScaler normalization for all numeric features
3. **Feature Selection**: Apply configured selection method
4. **Hyperparameter Tuning**: Grid/random search with cross-validation
5. **Model Training**: Fit final logistic regression model
6. **Probability Calibration**: Isotonic regression calibration on training data
7. **Performance Evaluation**: Calculate comprehensive metrics

#### Probability Calibration
- **Method**: Isotonic Regression (primary), Platt Scaling (fallback)
- **Purpose**: Ensures predicted probabilities match actual outcome frequencies
- **Validation**: Within-season cross-validation to prevent overfitting

### Key Features Used
- **Elo Ratings**: Team strength differential
- **Form Metrics**: Rolling 4-week EPA and success rates
- **Contextual Features**: Rest days, travel distance, venue type
- **Weather Conditions**: Temperature, wind speed, precipitation
- **Market Information**: Opening lines, line movement (when available)

### Performance Characteristics
- **Typical Accuracy**: 65-70% on out-of-sample data
- **Log Loss**: ~0.6-0.65 (well-calibrated models)
- **Brier Score**: ~0.23-0.25 (lower is better)

## Against the Spread (ATS) Model

### Overview

The ATS model predicts point margins and cover probabilities for spread betting. It employs a hybrid approach combining margin regression with residual distribution modeling.

**Location**: `models/train_ats.py`
**Model Type**: XGBoost/LightGBM Regression + Residual Distribution
**Output**: Point margin prediction + cover probability

### Methodology

#### Dual Approach Architecture
The ATS model implements three complementary approaches:

1. **Regression Approach**: Predicts point margin directly
2. **Classification Approach**: Predicts cover probability directly
3. **Hybrid Approach**: Combines both methods

#### Regression Component

**Algorithm Options**:
- **XGBoost Regressor** (default): Gradient boosting with built-in regularization
- **LightGBM Regressor**: Memory-efficient gradient boosting
- **Random Forest**: Ensemble method for robust predictions
- **Gradient Boosting**: Traditional gradient boosting implementation

**Feature Selection**: Model-based selection using base model feature importances

**Hyperparameter Grid (XGBoost)**:
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}
```

#### Residual Distribution Converter

**Purpose**: Converts point margin predictions to cover probabilities using learned residual distributions

**Supported Distributions**:
- **Normal Distribution**: `P(cover) = 1 - Φ((margin_needed - predicted - μ) / σ)`
- **t-Distribution**: Heavy-tailed distribution for outlier robustness
- **Skewed Normal**: Asymmetric distribution for modeling scoring biases

**Process**:
1. Train regression model to predict point margins
2. Calculate residuals: `residuals = actual_margin - predicted_margin`
3. Fit probability distribution to residuals
4. For new predictions: `cover_prob = P(actual_margin > -spread | predicted_margin)`

#### Classification Component (Optional)
- **Direct Classification**: Train binary classifier for cover/no-cover
- **Same Base Models**: XGBoost, LightGBM, Random Forest classifiers
- **Ensemble Weighting**: Combine with regression approach for final prediction

### Training Process
1. **Feature Preparation**: Clean features, handle missing values with median imputation
2. **Feature Scaling**: RobustScaler (less sensitive to outliers than StandardScaler)
3. **Regression Training**: Train margin prediction model with hyperparameter tuning
4. **Residual Analysis**: Fit distribution to prediction residuals
5. **Classification Training**: Train cover classifier (if hybrid approach)
6. **Probability Calibration**: Calibrate final cover probabilities
7. **Performance Evaluation**: Calculate regression and classification metrics

### Key Features Used
- **All WP Features**: Plus additional spread-specific features
- **Recent Performance**: Team offensive/defensive efficiency trends
- **Matchup Factors**: Strength of schedule adjustments
- **Market Context**: Spread movement, betting percentages
- **Situational Factors**: Divisional games, primetime effects

### Performance Characteristics
- **Margin MAE**: ~10-12 points (typical)
- **Margin RMSE**: ~14-16 points
- **Cover Accuracy**: ~52-54% (vs. 50% random)
- **R²**: ~0.15-0.25 (NFL margin prediction inherently noisy)

## Over/Under (O/U) Model

### Overview

The O/U model predicts total points scored and over/under probabilities. It employs multiple methodologies including regression, Poisson simulation, and weather impact modeling.

**Location**: `models/train_ou.py`
**Model Type**: Multi-approach (Regression + Poisson + Weather)
**Output**: Total points prediction + over/under probabilities

### Methodology

#### Multi-Model Architecture

**Primary Model**: XGBoost/LightGBM regression for total points
**Supporting Models**:
- **Poisson Score Model**: Individual team score prediction using Poisson regression
- **Weather Impact Model**: Specialized weather effect modeling
- **Total Distribution Converter**: Converts total predictions to over/under probabilities

#### Total Points Regression

**Algorithm**: XGBoost Regressor (primary choice for total points)
**Target**: Total points scored in game
**Feature Selection**: Model-based using XGBoost feature importances

**Hyperparameter Optimization**: Same grid as ATS model

#### Poisson Score Model

**Purpose**: Model individual team scores using Poisson distributions
**Implementation**:
```python
home_score ~ Poisson(λ_home)
away_score ~ Poisson(λ_away)
total_score = home_score + away_score
```

**Correlation Adjustment**: Accounts for score correlation between teams
- Calculate residual correlation from training data
- Apply correlation adjustment in simulations
- Typically small but measurable effect

**Simulation Process**:
1. Predict λ_home and λ_away using PoissonRegressor
2. Run Monte Carlo simulations (10,000 iterations)
3. Calculate over/under probabilities from simulation results
4. Compare with regression-based approach

#### Weather Impact Model

**Rationale**: Weather significantly affects total scoring, particularly:
- **Wind Speed**: Reduces passing efficiency and kicking accuracy
- **Temperature**: Extreme cold/heat affects player performance
- **Precipitation**: Reduces ball handling and passing accuracy
- **Humidity**: Affects player endurance and ball trajectory

**Implementation**:
- Linear regression: `weather_impact = β₀ + β₁·wind + β₂·temp + β₃·precip + β₄·humidity`
- Fit on residuals from base total model
- Apply as additive adjustment to total predictions

#### Total Distribution Converter

**Same Framework as ATS**: Uses residual distribution fitting
**Distributions Supported**: Normal, t-distribution, skewed normal
**Conversion Formula**: `P(over) = P(actual_total > market_total | predicted_total)`

### Training Process
1. **Multi-Target Preparation**: Extract total points, individual scores, weather features
2. **Total Model Training**: Primary XGBoost regression for total points
3. **Poisson Model Training**: Fit separate models for home/away scores
4. **Weather Model Training**: Fit weather impact on total residuals
5. **Distribution Fitting**: Learn residual distribution for probability conversion
6. **Calibration**: Isotonic regression on over/under probabilities
7. **Ensemble Integration**: Combine all approaches for final predictions

### Key Features Used
- **Pace Factors**: Team possession counts, play rates
- **Offensive Efficiency**: Points per drive, red zone efficiency
- **Defensive Efficiency**: Points allowed per drive, takeaway rates
- **Weather Conditions**: Comprehensive weather feature set
- **Venue Effects**: Dome vs. outdoor, altitude adjustments
- **Rest and Travel**: Days of rest, travel distance effects

### Performance Characteristics
- **Total MAE**: ~8-10 points
- **Total RMSE**: ~12-14 points
- **Over/Under Accuracy**: ~53-55% (vs. 50% random)
- **Weather Model Impact**: ~1-3 point adjustments in extreme conditions

## Supporting Components

### Probability Calibration System

**Location**: `models/calibrate.py`

#### Isotonic Regression Calibration
- **Method**: Non-parametric, monotonic calibration
- **Advantages**: No distributional assumptions, preserves ranking
- **Implementation**: `sklearn.isotonic.IsotonicRegression`
- **Validation**: Cross-validation within training season

#### Platt Scaling Calibration
- **Method**: Sigmoid function fitting
- **Use Case**: Fallback when insufficient data for isotonic regression
- **Implementation**: Logistic regression on raw probabilities
- **Parameters**: Fit sigmoid: `P_calibrated = 1 / (1 + exp(A·P_raw + B))`

#### Calibration Metrics
- **Reliability Curve**: Plots predicted vs. actual probabilities in bins
- **Expected Calibration Error (ECE)**: Average difference between confidence and accuracy
- **Brier Score**: Mean squared difference between probabilities and outcomes
- **Calibration Slope**: Slope of reliability curve (1.0 = perfect calibration)

### Walk-Forward Validation Framework

**Location**: `models/utils.py`

#### Temporal Validation Protocol
```
Training: 2018-2020 → Test: 2021
Training: 2018-2021 → Test: 2022
Training: 2018-2022 → Test: 2023
Training: 2018-2023 → Test: 2024
```

#### Key Features
- **Strict Temporal Ordering**: No future data used in training
- **Expanding Window**: Use all available historical data
- **Season-Level Splits**: Prevent within-season leakage
- **Metadata Tracking**: Track training seasons, sample sizes, split dates

#### Validation Metrics
- **Overall Performance**: Aggregate metrics across all test seasons
- **Stability Analysis**: Season-to-season performance variation
- **Feature Evolution**: Track feature importance changes over time

### Model Management System

**Location**: `models/utils.py`

#### Model Serialization
- **Format**: Joblib pickle format for sklearn compatibility
- **Components Saved**: Model, scaler, feature selector, calibrator, metadata
- **Versioning**: Model version tracking with training date stamps
- **Configuration**: Save all hyperparameters and training settings

#### Metadata Tracking
```python
@dataclass
class ModelMetadata:
    model_name: str
    model_type: str
    target_type: str
    train_seasons: List[int]
    features_used: List[str]
    training_date: datetime
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
```

## Validation Framework

### Data Leakage Prevention

#### Temporal Constraints
- **Feature Cutoff**: Only use data available before game start
- **Market Data**: Use Friday 6 PM ET snapshot for odds
- **Opponent Stats**: No post-game statistics for current week
- **Form Metrics**: Rolling windows with strict temporal boundaries

#### Cross-Validation Protocol
- **Within-Season**: Stratified K-fold for hyperparameter tuning only
- **Across-Season**: Walk-forward validation for final evaluation
- **Validation Sets**: Hold out final 20% of each training season

### Statistical Testing

#### Model Significance
- **Baseline Comparison**: Compare against market odds and simple models
- **Statistical Tests**: Paired t-tests for performance differences
- **Confidence Intervals**: Bootstrap confidence intervals for key metrics
- **Effect Sizes**: Cohen's d for practical significance assessment

#### Calibration Testing
- **Hosmer-Lemeshow Test**: Goodness-of-fit test for calibration
- **Reliability Diagram**: Visual assessment of calibration quality
- **ECE Computation**: Quantitative calibration error measurement

## Performance Metrics

### Classification Metrics (WP, ATS Cover, O/U Over)

#### Primary Metrics
- **Accuracy**: Percentage of correct predictions
- **Log Loss**: Penalizes confident wrong predictions
- **Brier Score**: Mean squared error for probabilities
- **ROC AUC**: Area under receiver operating characteristic curve

#### Calibration Metrics
- **Expected Calibration Error (ECE)**: Average |confidence - accuracy|
- **Reliability Curve**: Predicted vs. actual probability by bins
- **Calibration Slope**: Linear relationship between predicted and actual
- **Calibration Intercept**: Systematic over/under-confidence

### Regression Metrics (ATS Margin, O/U Total)

#### Error Metrics
- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **Root Mean Square Error (RMSE)**: Penalizes large errors more heavily
- **Mean Absolute Percentage Error (MAPE)**: Relative error measurement
- **Median Absolute Error**: Robust to outliers

#### Correlation Metrics
- **R-squared (R²)**: Proportion of variance explained
- **Pearson Correlation**: Linear relationship strength
- **Spearman Correlation**: Monotonic relationship strength

### Betting Performance Metrics

#### Profitability Analysis
- **Return on Investment (ROI)**: Profit percentage on capital risked
- **Betting Units**: Standardized profit measurement
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Maximum Drawdown**: Largest peak-to-trough decline

#### Market Efficiency
- **Closing Line Value (CLV)**: Difference between model and closing odds
- **Market Beat Rate**: Frequency of outperforming market
- **Edge Detection**: Identifying profitable betting opportunities

## Implementation Details

### Computational Requirements

#### Training Time
- **WP Model**: ~5-10 minutes per season (depends on feature selection)
- **ATS Model**: ~15-30 minutes per season (XGBoost hyperparameter tuning)
- **O/U Model**: ~20-40 minutes per season (multiple sub-models)

#### Memory Usage
- **Feature Matrices**: ~100-500 MB per season depending on feature count
- **Model Storage**: ~10-50 MB per trained model
- **Prediction Batch**: ~1-10 MB per week of games

#### Scalability
- **Parallel Training**: All models support multi-core training (n_jobs=-1)
- **Feature Caching**: Pre-computed features stored in Parquet format
- **Incremental Updates**: Models can be retrained with new data weekly

### Error Handling

#### Data Quality Checks
- **Missing Features**: Default value imputation with logging
- **Outlier Detection**: IQR-based outlier flagging and capping
- **Feature Drift**: Monitor feature distribution changes over time
- **Target Validation**: Ensure target variables are within expected ranges

#### Model Robustness
- **Convergence Monitoring**: Track training loss and early stopping
- **Cross-Validation Stability**: Monitor CV score variance
- **Feature Selection Stability**: Track feature selection consistency
- **Calibration Quality**: Monitor calibration metrics during training

### Configuration Management

#### Model Parameters
```yaml
wp_model:
  feature_selection_method: "recursive"
  max_features: 15
  regularization_type: "elasticnet"
  use_calibration: true
  hyperparameter_tuning: "grid_search"

ats_model:
  model_type: "xgboost"
  approach: "hybrid"
  distribution_type: "normal"
  use_calibration: true

ou_model:
  model_type: "xgboost"
  use_poisson: true
  use_weather_model: true
  distribution_type: "normal"
```

#### Validation Settings
```yaml
validation:
  start_season: 2018
  end_season: 2024
  min_train_seasons: 3
  cv_folds: 3
  calibration_method: "isotonic"
```

### Future Enhancements

#### Model Improvements
- **Deep Learning**: Explore neural network architectures for non-linear patterns
- **Ensemble Methods**: Combine multiple base models for improved performance
- **Feature Learning**: Automated feature engineering and selection
- **Transfer Learning**: Leverage models from other sports or betting markets

#### Validation Enhancements
- **Online Learning**: Continuous model updates as new data arrives
- **Adaptive Calibration**: Dynamic calibration adjustments based on recent performance
- **Multi-Objective Optimization**: Balance accuracy, calibration, and profitability
- **Uncertainty Quantification**: Confidence intervals for individual predictions

---

*This documentation reflects the current state of the model methodology as implemented in the NFL Prediction System. For the most up-to-date implementation details, refer to the source code in the `models/` directory.*