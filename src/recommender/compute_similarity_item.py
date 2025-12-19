"""
Orchestrator for computing the item-item cosine similarity matrix.

This module follows FAANG-level engineering standards:
- Centralized configuration management
- Structured JSON logging suitable for production environments
- Clear separation of concerns (no business logic here)
- Reusable compute engines
- Input validation
- Testable design
"""

from __future__ import annotations

import logging

from .config import load_app_config
from .logging_utils import configure_logger
from .mongo_loader import load_movies_and_ratings
from .similarity_engine import ItemItemCosineSimilarityEngine
from .validators import validate_ratings_schema
from .writers import save_similarity_matrix_to_csv


def main() -> None:
    """
    Entry point for the item-item similarity computation pipeline.

    Steps:
        1. Initialize structured logger.
        2. Load configuration (Mongo settings, output paths).
        3. Load movies and ratings from MongoDB.
        4. Validate ratings schema.
        5. Compute item-item similarity using engine.
        6. Save the resulting similarity matrix to CSV.
    """
    logger = configure_logger(
        name="recommender.similarity.item_item",
        level=logging.INFO
    )

    logger.info(
        "Starting item-item similarity computation pipeline",
        extra={"event": "pipeline_start"},
    )

    # ----------------------------------------------------------
    # Step 1 — Load configuration
    # ----------------------------------------------------------
    app_config = load_app_config()

    # ----------------------------------------------------------
    # Step 2 — Load data from MongoDB
    # ----------------------------------------------------------
    movies_df, ratings_df = load_movies_and_ratings(
        config=app_config.mongo,
        logger=logger,
    )

    # ----------------------------------------------------------
    # Step 3 — Validate ratings DataFrame
    # ----------------------------------------------------------
    validate_ratings_schema(
        df=ratings_df,
        logger=logger,
        step_name="item_item_similarity",
    )

    # ----------------------------------------------------------
    # Step 4 — Compute item-item cosine similarity using Engine
    # ----------------------------------------------------------
    engine = ItemItemCosineSimilarityEngine(logger=logger)
    similarity_df = engine.run_full_pipeline(ratings_df)

    # ----------------------------------------------------------
    # Step 5 — Save output to CSV
    # ----------------------------------------------------------
    save_similarity_matrix_to_csv(
        similarity_df=similarity_df,
        output_path=app_config.similarity.item_item_output_csv_path,
        logger=logger,
    )

    logger.info(
        "Item-item similarity computation pipeline completed successfully",
        extra={"event": "pipeline_end"},
    )

    print(f"✅ Item-item similarity matrix saved to: {app_config.similarity.item_item_output_csv_path}")


if __name__ == "__main__":
    main()
