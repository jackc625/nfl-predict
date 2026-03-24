"""Health check endpoint."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from api.dependencies import DB_PATH
from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return API health status including cache availability."""
    cache_ready = DB_PATH.exists()
    last_updated: datetime | None = None

    if cache_ready:
        try:
            import duckdb

            with duckdb.connect(str(DB_PATH), read_only=True) as conn:
                row = conn.execute(
                    "SELECT value FROM cache_meta WHERE key = 'last_updated'"
                ).fetchone()
                if row and row[0]:
                    last_updated = datetime.fromisoformat(row[0])
        except Exception:  # noqa: BLE001
            # Cache exists but may be empty/corrupt -- still report ready
            pass

    return HealthResponse(
        status="ok" if cache_ready else "degraded",
        cache_ready=cache_ready,
        last_updated=last_updated,
    )
