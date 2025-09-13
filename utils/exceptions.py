"""Custom exception classes for NFL Prediction System."""


class NFLPredictException(Exception):
    """Base exception class for NFL Prediction System."""
    pass


class DataIngestionError(NFLPredictException):
    """Raised when data ingestion fails."""
    pass


class DataValidationError(NFLPredictException):
    """Raised when data validation fails."""
    pass


class ModelTrainingError(NFLPredictException):
    """Raised when model training fails."""
    pass


class ModelPredictionError(NFLPredictException):
    """Raised when model prediction fails."""
    pass


class ConfigurationError(NFLPredictException):
    """Raised when configuration is invalid."""
    pass


class ExternalAPIError(NFLPredictException):
    """Raised when external API calls fail."""
    pass


class BacktestError(NFLPredictException):
    """Raised when backtesting fails."""
    pass


class FeatureEngineeringError(NFLPredictException):
    """Raised when feature engineering fails."""
    pass