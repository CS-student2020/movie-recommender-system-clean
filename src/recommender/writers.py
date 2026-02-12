from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .logging_utils import configure_logger


def save_similarity_matrix_to_csv(
    similarity_df: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger | None = None,
) -> None:
    """
    Save the similarity matrix to a CSV file.
    Ensures output folder exists and logs the operation using JSON logs.
    """
    _logger = logger or configure_logger()

    _logger.info(
        "Saving similarity matrix to CSV",
        extra={
            "event": "save_similarity_csv",
            "shape": similarity_df.shape,
            "output_path": str(output_path),
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    similarity_df.to_csv(output_path, encoding="utf-8-sig")

    _logger.info(
        "Similarity matrix saved",
        extra={"event": "save_similarity_csv_success", "output_path": str(output_path)},
    )

