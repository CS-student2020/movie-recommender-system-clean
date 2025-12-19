from __future__ import annotations

import logging

from .config import load_app_config
from .logging_utils import configure_logger
from .mongo_loader import load_movies_and_ratings
from .similarity_engine import UserUserCosineSimilarityEngine
from .writers import save_similarity_matrix_to_csv


def main() -> None:
    """
    Entry point for computing the user-user cosine similarity matrix.

    Steps:
        1. Load configuration (Mongo, output paths).
        2. Initialize logger.
        3. Load movies and ratings from MongoDB.
        4. Compute user-user similarity matrix from ratings.
        5. Save the similarity matrix to CSV.
    """
    logger = configure_logger(name="recommender.similarity", level=logging.INFO)

    logger.info(
        "Starting user-user similarity computation pipeline",
        extra={"event": "pipeline_start"},
    )

    # Load configuration
    app_config = load_app_config()

    # Load data
    movies_df, ratings_df = load_movies_and_ratings(
        config=app_config.mongo,
        logger=logger,
    )

    # Compute similarity
    engine = UserUserCosineSimilarityEngine(logger=logger)
    similarity_df = engine.run_full_pipeline(ratings_df)

    # Save CSV
    save_similarity_matrix_to_csv(
        similarity_df=similarity_df,
        output_path=app_config.similarity.user_user_output_csv_path,
        logger=logger,
    )


    logger.info(
        "User-user similarity computation pipeline finished successfully",
        extra={"event": "pipeline_end"},
    )
    print(f"✅ Similarity matrix created and saved to {app_config.similarity.user_user_output_csv_path}")


if __name__ == "__main__":
    main()
