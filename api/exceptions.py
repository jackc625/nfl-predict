"""API Exception Handling for NFL Prediction System.

Canonical home for custom API exceptions used across the FastAPI application.
Kept intentionally minimal -- only classes with active external usage.

Active usage (grep-verified by plan 15-03):
    - ``ModelUnavailableError`` -- raised by ``api.dependencies.get_db`` when the
      DuckDB web cache is missing or unreadable, mapped to 503 by FastAPI.
    - ``NFLPredictionAPIException`` -- base class for ``ModelUnavailableError``.

Previously-unused classes ``DataNotFoundError`` and ``ValidationError`` were
removed in plan 15-03 after grep confirmed zero importers. Any new exception
type added here must first be imported from at least one non-test module.
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
