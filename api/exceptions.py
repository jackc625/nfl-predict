"""API Exception Handling for NFL Prediction System.

Canonical home for custom API exceptions used across the FastAPI application.
Kept intentionally minimal -- only classes with active external usage.

Active usage (grep-verified by plan 15-03):
    - ``ModelUnavailableError`` -- raised by ``api.dependencies.get_db`` when the
      DuckDB web cache is missing or unreadable, mapped to 503 by FastAPI.
    - ``NFLPredictionAPIException`` -- base class for ``ModelUnavailableError`` and
      ``DataFormatMismatchError``.
    - ``DataFormatMismatchError`` -- raised by ``api.data_format_gate`` when the cache's
      data-format version differs from the code's (Phase 36, SPEC R9); its handler there
      answers 503 with the standalone notice.

Previously-unused classes ``DataNotFoundError`` and ``ValidationError`` were
removed in plan 15-03 after grep confirmed zero importers. Any new exception
type added here must first be imported from at least one non-test module.
"""

from __future__ import annotations

from typing import Any

from api.cache_schema import DataFormatState


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


class DataFormatMismatchError(NFLPredictionAPIException):
    """Raised when the cache was built for a different data format (503 Service Unavailable).

    Attributes:
        state: The cache's stored version judged against the code's.
        kind: Which kind of route refused (``api.data_format_gate.PAGE``, ``FRAGMENT`` or
            ``EXPORT``), so the handler answers a page with the notice page and the others with
            plain text.
    """

    def __init__(self, state: DataFormatState, kind: str) -> None:
        super().__init__(
            message="The cache's data format does not match the code's",
            status_code=503,
            error_code="DATA_FORMAT_MISMATCH",
            details={"expected": state.expected, "found": state.found_label},
        )
        self.state = state
        self.kind = kind
