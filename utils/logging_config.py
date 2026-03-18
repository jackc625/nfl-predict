"""Structured logging configuration using structlog."""

import logging
import logging.config
import sys
import uuid
from pathlib import Path
from typing import Any

import structlog
from structlog.typing import EventDict, Processor

from conf.settings import get_settings


def add_request_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add request ID to log entries."""
    if "request_id" not in event_dict:
        event_dict["request_id"] = str(uuid.uuid4())[:8]
    return event_dict


def add_module_info(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add module and function information to log entries."""
    try:
        # Skip if logger doesn't have _context (for compatibility)
        if hasattr(logger, "_context"):
            frame = logger._context.get("frame")
            if frame:
                event_dict["module"] = frame.f_globals.get("__name__", "unknown")
                event_dict["function"] = frame.f_code.co_name
    except:
        # Silently skip if frame info is not available
        pass
    return event_dict


def filter_sensitive_data(
    logger: Any, method_name: str, event_dict: EventDict
) -> EventDict:
    """Filter out sensitive information from logs."""
    sensitive_keys = {
        "password",
        "secret",
        "token",
        "key",
        "credential",
        "api_key",
        "auth",
        "authorization",
        "private",
    }

    def _filter_dict(data: dict[str, Any]) -> dict[str, Any]:
        filtered = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                filtered[key] = "***REDACTED***"
            elif isinstance(value, dict):
                filtered[key] = _filter_dict(value)
            elif isinstance(value, list):
                filtered[key] = [
                    _filter_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value
        return filtered

    # Filter the entire event dict
    return _filter_dict(event_dict)


def setup_logging(
    level: str | None = None,
    format_type: str | None = None,
    log_file: str | None = None,
    console: bool | None = None,
) -> None:
    """
    Setup structured logging with structlog.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format ('json' or 'text')
        log_file: Log file path
        console: Whether to log to console
    """
    settings = get_settings()

    # Use provided values or fall back to config
    log_level = level or settings.config.logging.level
    log_format = format_type or settings.config.logging.format
    enable_console = console if console is not None else settings.config.logging.console
    enable_file = log_file is not None or settings.config.logging.file

    # Ensure log directory exists
    if enable_file:
        file_path = log_file or settings.config.logging.file_path
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure processors
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_request_id,
        filter_sensitive_data,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    if settings.config.logging.include_timestamp:
        processors.append(structlog.processors.TimeStamper(fmt="iso"))

    if settings.config.logging.include_module:
        processors.append(add_module_info)

    # Add final processor based on format
    if log_format.lower() == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    handlers = []

    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(console_handler)

    if enable_file:
        file_path = log_file or settings.config.logging.file_path
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a configured structlog logger.

    Args:
        name: Logger name (defaults to calling module)

    Returns:
        Configured structlog logger
    """
    if name is None:
        # Get the calling module name
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get("__name__", "unknown")
        else:
            name = "unknown"

    return structlog.get_logger(name)


def with_request_id(request_id: str) -> structlog.BoundLogger:
    """
    Get a logger bound with a specific request ID.

    Args:
        request_id: Request identifier

    Returns:
        Logger bound with request ID
    """
    return structlog.get_logger().bind(request_id=request_id)


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log a function call with parameters.

    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger = get_logger()
    filtered_kwargs = {
        k: v
        for k, v in kwargs.items()
        if not any(
            sensitive in k.lower()
            for sensitive in ["password", "key", "token", "secret"]
        )
    }
    logger.info("Function called", function=func_name, parameters=filtered_kwargs)


def log_data_operation(operation: str, table: str, rows: int, **metadata) -> None:
    """
    Log a data operation.

    Args:
        operation: Type of operation (read, write, update, delete)
        table: Table or dataset name
        rows: Number of rows affected
        **metadata: Additional metadata
    """
    logger = get_logger()
    logger.info(
        "Data operation", operation=operation, table=table, rows=rows, **metadata
    )


def log_model_operation(
    operation: str, model_type: str, duration: float | None = None, **metadata
) -> None:
    """
    Log a model operation.

    Args:
        operation: Type of operation (train, predict, evaluate)
        model_type: Type of model (wp, ats, ou)
        duration: Operation duration in seconds
        **metadata: Additional metadata
    """
    logger = get_logger()
    log_data = {"operation": operation, "model_type": model_type, **metadata}
    if duration is not None:
        log_data["duration_seconds"] = duration

    logger.info("Model operation", **log_data)


def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration: float,
    request_id: str | None = None,
    **metadata,
) -> None:
    """
    Log an API request.

    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration: Request duration in seconds
        request_id: Request identifier
        **metadata: Additional metadata
    """
    logger = get_logger()
    if request_id:
        logger = logger.bind(request_id=request_id)

    logger.info(
        "API request",
        method=method,
        path=path,
        status_code=status_code,
        duration_seconds=duration,
        **metadata,
    )
