"""
API Exception Handling for NFL Prediction System.

This module is the canonical home for custom exceptions used across the
FastAPI application. Other API modules import their exception classes from
here. Plan 15-03 may trim genuinely unused classes after grep verification,
but ``ModelUnavailableError`` and ``NFLPredictionAPIException`` MUST remain
because they are imported by ``api.dependencies``.
"""

from __future__ import annotations

from typing import Any


class NFLPredictionAPIException(Exception):
    """Base exception for NFL Prediction API."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class DataNotFoundError(NFLPredictionAPIException):
    """Raised when requested data is not found (404 Not Found)."""

    def __init__(
        self,
        message: str = "Requested data not found",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=404,
            error_code="DATA_NOT_FOUND",
            details=details,
        )


class ValidationError(NFLPredictionAPIException):
    """Raised when input validation fails (400 Bad Request)."""

    def __init__(
        self,
        message: str = "Validation failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class ModelUnavailableError(NFLPredictionAPIException):
    """Raised when prediction data is unavailable (503 Service Unavailable)."""

    def __init__(
        self,
        message: str = "Prediction data is currently unavailable",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=503,
            error_code="MODEL_UNAVAILABLE",
            details=details,
        )
