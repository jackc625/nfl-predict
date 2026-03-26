"""Shared dependencies for the FastAPI application.

Provides the Jinja2Blocks template engine and DataService factory.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2_fragments.fastapi import Jinja2Blocks

from .services import DataService

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "web_cache.duckdb"

templates = Jinja2Blocks(directory=str(TEMPLATES_DIR))


def format_datetime(value: str | datetime | None) -> str:
    """Format an ISO timestamp or datetime to human-readable string."""
    if value is None:
        return "Unknown"
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return str(value)
    return value.strftime("%b %d, %Y %I:%M %p")


templates.env.filters["format_datetime"] = format_datetime


def get_data_service() -> DataService:
    """Create a DataService instance pointing to the web cache."""
    return DataService(db_path=DB_PATH)
