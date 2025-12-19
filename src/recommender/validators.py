from __future__ import annotations

import logging
import pandas as pd


class ValidationError(Exception):
    """Raised when input data fails validation."""


REQUIRED_COLUMNS = {"userId", "movieId", "rating"}


def validate_required_columns(df: pd.DataFrame, required: set[str]) -> None:
    """
    Validate that DataFrame contains all required columns.
    """
    missing = required - set(df.columns)
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")


def validate_ratings_schema(
    df: pd.DataFrame,
    logger: logging.Logger = None,
    step_name: str = "ratings_schema_validation"
) -> None:
    """
    Validate the schema of the ratings DataFrame.

    Checks:
        - required columns exist
        - rating column is numeric
    """
    if logger:
        logger.info(
            "Validating ratings schema",
            extra={"event": f"validate_schema_{step_name}"}
        )

    # Check required columns
    validate_required_columns(df, REQUIRED_COLUMNS)

    # Ensure rating is numeric
    if not pd.api.types.is_numeric_dtype(df["rating"]):
        raise ValidationError("Column 'rating' must be numeric.")

    if logger:
        logger.info(
            "Ratings schema validated successfully",
            extra={"event": f"validate_schema_{step_name}_success"}
        )
