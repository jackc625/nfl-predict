"""
Middleware for NFL Prediction API.

This module provides middleware for request/response logging, performance monitoring,
and request ID tracking for better observability.
"""

import json
import logging
import re
import time
import uuid
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging."""

    def __init__(self, app, log_bodies: bool = False, max_body_size: int = 1024):
        super().__init__(app)
        self.log_bodies = log_bodies
        self.max_body_size = max_body_size

        # Common secret field patterns to redact
        self.secret_patterns = [
            # Headers
            re.compile(r"(authorization|cookie|x-api-key)", re.IGNORECASE),
            # JSON field names
            re.compile(
                r"(password|token|secret|key|auth|api_key|access_token|refresh_token)",
                re.IGNORECASE,
            ),
        ]

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Record start time
        start_time = time.time()

        # Extract request information
        request_info = await self._extract_request_info(request)

        # Redact sensitive headers
        redacted_headers = self._redact_headers(dict(request.headers))

        # Log incoming request
        logger.info(
            "Incoming request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": self._get_client_ip(request),
                "user_agent": redacted_headers.get("user-agent"),
                "content_type": redacted_headers.get("content-type"),
                "content_length": redacted_headers.get("content-length"),
                "sensitive_headers_present": self._has_sensitive_headers(
                    request.headers
                ),
                **request_info,
            },
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Extract response information
            response_info = self._extract_response_info(response)

            # Log response
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time_ms": round(process_time * 1000, 2),
                    "response_size": response.headers.get("content-length"),
                    **response_info,
                },
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))

            return response

        except (
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            KeyError,
        ) as e:
            # Calculate processing time for failed requests
            process_time = time.time() - start_time

            # Log error
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "process_time_ms": round(process_time * 1000, 2),
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                },
            )

            # Re-raise the exception to be handled by exception handlers
            raise

    async def _extract_request_info(self, request: Request) -> dict[str, Any]:
        """Extract additional request information."""
        info = {}

        # Log request body if enabled and within size limit
        if self.log_bodies and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) <= self.max_body_size:
                    content_type = request.headers.get("content-type", "")
                    if "application/json" in content_type:
                        try:
                            parsed_body = json.loads(body.decode())
                            info["request_body"] = self._redact_json_secrets(
                                parsed_body
                            )
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            # For non-JSON content, don't log raw body to avoid exposing secrets
                            info["request_body"] = "<non-json content redacted>"
                    else:
                        # For non-JSON content types, don't log the body to avoid exposing secrets
                        info["request_body"] = f"<{content_type} content redacted>"
                else:
                    info["request_body"] = f"<body too large: {len(body)} bytes>"
            except (ValueError, TypeError, RuntimeError, OSError) as e:
                info["request_body_error"] = str(e)

        return info

    def _extract_response_info(self, response: Response) -> dict[str, Any]:
        """Extract response information."""
        info = {
            "content_type": response.headers.get("content-type"),
            "cache_control": response.headers.get("cache-control"),
        }

        return info

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address, handling proxies."""
        # Check for forwarded headers first (for load balancers/proxies)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP if there are multiple
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"

    def _redact_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Redact sensitive information from headers."""
        redacted = {}
        for key, value in headers.items():
            if any(pattern.match(key) for pattern in self.secret_patterns):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = value
        return redacted

    def _has_sensitive_headers(self, headers) -> bool:
        """Check if request contains sensitive headers."""
        return any(
            any(pattern.match(key) for pattern in self.secret_patterns)
            for key in headers
        )

    def _redact_json_secrets(self, obj: Any) -> Any:
        """Recursively redact sensitive fields from JSON objects."""
        if isinstance(obj, dict):
            redacted = {}
            for key, value in obj.items():
                if any(pattern.match(key) for pattern in self.secret_patterns):
                    redacted[key] = "<redacted>"
                else:
                    redacted[key] = self._redact_json_secrets(value)
            return redacted
        if isinstance(obj, list):
            return [self._redact_json_secrets(item) for item in obj]
        return obj


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and metrics."""

    def __init__(self, app, slow_request_threshold: float = 1.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log slow requests
            if process_time > self.slow_request_threshold:
                logger.warning(
                    "Slow request detected",
                    extra={
                        "request_id": getattr(request.state, "request_id", None),
                        "method": request.method,
                        "path": request.url.path,
                        "process_time_ms": round(process_time * 1000, 2),
                        "threshold_ms": self.slow_request_threshold * 1000,
                        "status_code": response.status_code,
                    },
                )

            # Add performance headers
            response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))

            return response

        except (
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
            OSError,
            KeyError,
        ) as e:
            process_time = time.time() - start_time

            # Log performance data for failed requests
            logger.error(
                "Request failed with performance data",
                extra={
                    "request_id": getattr(request.state, "request_id", None),
                    "method": request.method,
                    "path": request.url.path,
                    "process_time_ms": round(process_time * 1000, 2),
                    "exception": str(e),
                },
            )

            raise


class SecurityMiddleware(BaseHTTPMiddleware):
    """Basic security middleware."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Add API-specific headers
        response.headers["X-API-Version"] = "1.0"
        response.headers["X-Service"] = "nfl-prediction-api"

        return response


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware for API."""

    def __init__(
        self,
        app,
        allowed_origins: list | None = None,
        allowed_methods: list | None = None,
    ):
        super().__init__(app)
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or [
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "OPTIONS",
        ]

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = self._get_allowed_origin(
                request
            )
            response.headers["Access-Control-Allow-Methods"] = ", ".join(
                self.allowed_methods
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-Request-ID"
            )
            response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours
            return response

        response = await call_next(request)

        # Add CORS headers to response
        response.headers["Access-Control-Allow-Origin"] = self._get_allowed_origin(
            request
        )
        response.headers["Access-Control-Allow-Credentials"] = "false"
        response.headers["Access-Control-Expose-Headers"] = (
            "X-Request-ID, X-Process-Time"
        )

        return response

    def _get_allowed_origin(self, request: Request) -> str:
        """Get allowed origin for the request."""
        if "*" in self.allowed_origins:
            return "*"

        origin = request.headers.get("origin")
        if origin in self.allowed_origins:
            return origin

        # Default to first allowed origin if specific origin not found
        return self.allowed_origins[0] if self.allowed_origins else "*"


def setup_middleware(app, config: dict[str, Any] | None = None):
    """Set up all middleware for the FastAPI app."""

    config = config or {}

    # Security middleware (first)
    app.add_middleware(SecurityMiddleware)

    # CORS middleware
    cors_config = config.get("cors", {})
    app.add_middleware(
        CORSMiddleware,
        allowed_origins=cors_config.get("allowed_origins"),
        allowed_methods=cors_config.get("allowed_methods"),
    )

    # Performance monitoring
    performance_config = config.get("performance", {})
    app.add_middleware(
        PerformanceMiddleware,
        slow_request_threshold=performance_config.get("slow_request_threshold", 1.0),
    )

    # Request logging (last, so it captures all processing)
    logging_config = config.get("logging", {})
    app.add_middleware(
        RequestLoggingMiddleware,
        log_bodies=logging_config.get("log_bodies", False),
        max_body_size=logging_config.get("max_body_size", 1024),
    )
