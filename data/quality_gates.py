"""Hard-fail data quality gates for Bronze-to-Silver and Silver-to-Gold boundaries.

These gates enforce strict data integrity rules. A single bad row kills the
entire batch -- this is intentional. Silent row-skipping causes downstream
data corruption that is far harder to debug than a hard failure at ingestion.
"""

from __future__ import annotations

import pandas as pd

from data.schemas import GameSchema
from utils.exceptions import DataValidationError

# Range bounds for Silver-to-Gold feature validation.
# Values outside these ranges indicate data corruption or calculation errors.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "elo_rating": (800.0, 2200.0),
    "elo_diff": (-600.0, 600.0),
    "epa_per_play": (-1.0, 1.0),
    "success_rate": (0.0, 1.0),
    "spread": (-50.0, 50.0),
    "total": (10.0, 100.0),
    "temp_f": (-50.0, 120.0),
    "wind_mph": (0.0, 100.0),
    "humidity_pct": (0.0, 100.0),
}


def validate_bronze_to_silver(
    df: pd.DataFrame,
    schema_class: type = GameSchema,
) -> pd.DataFrame:
    """Hard-fail validation gate for Bronze-to-Silver transition.

    Validates every row against the Pydantic schema. ONE bad row kills the
    entire batch -- this prevents partial/corrupt data from reaching Silver.

    Args:
        df: Raw DataFrame from Bronze layer
        schema_class: Pydantic model class to validate against

    Returns:
        Validated DataFrame with schema-conformant rows

    Raises:
        DataValidationError: If ANY row fails validation, with error details
    """
    errors: list[str] = []
    validated_rows: list[dict] = []

    for idx, row in df.iterrows():
        try:
            validated = schema_class(**row.to_dict())
            validated_rows.append(validated.model_dump())
        except (ValueError, TypeError) as e:
            errors.append(f"Row {idx} (game_id={row.get('game_id', 'UNKNOWN')}): {e}")

    if errors:
        error_report = "\n".join(errors[:20])
        total = len(errors)
        raise DataValidationError(
            f"Bronze-to-Silver validation failed with {total} error(s):\n"
            f"{error_report}" + (f"\n... and {total - 20} more" if total > 20 else "")
        )

    return pd.DataFrame(validated_rows)


def validate_silver_to_gold(
    df: pd.DataFrame,
    feature_ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Hard-fail range validation for Silver-to-Gold transition.

    Checks all numeric columns against expected ranges. Produces a detailed
    violation report and raises DataValidationError if any violations found.

    Args:
        df: Silver layer DataFrame
        feature_ranges: Dict of column -> (min, max) bounds. Uses FEATURE_RANGES
            default if not provided.

    Returns:
        The same DataFrame if all checks pass

    Raises:
        DataValidationError: With detailed range violation report
    """
    ranges = feature_ranges or FEATURE_RANGES
    violations: list[str] = []

    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue
        col_data = df[col].dropna()
        below = col_data[col_data < low]
        above = col_data[col_data > high]

        if len(below) > 0:
            violations.append(
                f"  {col}: {len(below)} value(s) below {low} (min={below.min():.4f})"
            )
        if len(above) > 0:
            violations.append(
                f"  {col}: {len(above)} value(s) above {high} (max={above.max():.4f})"
            )

    if violations:
        report = "\n".join(violations)
        raise DataValidationError(f"Silver-to-Gold range validation failed:\n{report}")

    return df
