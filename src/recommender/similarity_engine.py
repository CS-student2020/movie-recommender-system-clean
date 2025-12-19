from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .logging_utils import configure_logger
from .validators import validate_ratings_schema


class UserUserCosineSimilarityEngine:
    """
    Engine for computing user-user cosine similarity.

    High-level workflow:
        1. Validate and pivot ratings DataFrame.
        2. Compute cosine similarity.
        3. Return similarity matrix as DataFrame.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or configure_logger()

    def prepare_user_item_matrix(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert ratings DataFrame into a user-item pivot matrix.

        Rows: userId
        Columns: movieId
        Values: rating
        """
        self.logger.info(
            "Validating ratings schema",
            extra={"event": "validate_ratings_schema"},
        )
        validate_ratings_schema(ratings_df)

        self.logger.info(
            "Preparing user-item matrix",
            extra={"event": "prepare_user_item_matrix"},
        )

        pivot_df = ratings_df.pivot(
            index="userId",
            columns="movieId",
            values="rating",
        ).fillna(0.0)

        pivot_df = pivot_df.astype(float)

        self.logger.info(
            "User-item matrix prepared",
            extra={"event": "prepare_user_item_matrix_success", "shape": pivot_df.shape},
        )
        return pivot_df

    def compute_similarity(self, user_item_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cosine similarity between users.
        """
        self.logger.info(
            "Computing user-user cosine similarity",
            extra={"event": "compute_user_user_cosine", "shape": user_item_matrix.shape},
        )

        similarity_matrix = cosine_similarity(user_item_matrix.values)

        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=user_item_matrix.index,
            columns=user_item_matrix.index,
        )

        self.logger.info(
            "Cosine similarity computed",
            extra={"event": "compute_user_user_cosine_success", "shape": similarity_df.shape},
        )

        return similarity_df

    def run_full_pipeline(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline: validate → pivot → compute similarity.
        """
        pivot_df = self.prepare_user_item_matrix(ratings_df)
        return self.compute_similarity(pivot_df)


class ItemItemCosineSimilarityEngine:
    """
    Engine for computing item-item (movie-movie) cosine similarity.

    High-level workflow:
        1. Validate and pivot ratings DataFrame (movies as rows).
        2. Compute cosine similarity between items.
        3. Return similarity matrix as DataFrame.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or configure_logger()

    def prepare_item_user_matrix(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert ratings DataFrame into an item-user pivot matrix.

        Rows: movieId
        Columns: userId
        Values: rating
        """
        self.logger.info(
            "Validating ratings schema for item-item similarity",
            extra={"event": "validate_ratings_schema_item_item"},
        )
        validate_ratings_schema(ratings_df)

        self.logger.info(
            "Preparing item-user matrix",
            extra={"event": "prepare_item_user_matrix"},
        )

        pivot_df = ratings_df.pivot(
            index="movieId",
            columns="userId",
            values="rating",
        ).fillna(0.0)

        pivot_df = pivot_df.astype(float)

        self.logger.info(
            "Item-user matrix prepared",
            extra={"event": "prepare_item_user_matrix_success", "shape": pivot_df.shape},
        )
        return pivot_df

    def compute_similarity(self, item_user_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cosine similarity between items (movies).
        """
        self.logger.info(
            "Computing item-item cosine similarity",
            extra={"event": "compute_item_item_cosine", "shape": item_user_matrix.shape},
        )

        similarity_matrix = cosine_similarity(item_user_matrix.values)

        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=item_user_matrix.index,
            columns=item_user_matrix.index,
        )

        self.logger.info(
            "Item-item cosine similarity computed",
            extra={"event": "compute_item_item_cosine_success", "shape": similarity_df.shape},
        )

        return similarity_df

    def run_full_pipeline(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline for item-item similarity.
        """
        item_user_matrix = self.prepare_item_user_matrix(ratings_df)
        return self.compute_similarity(item_user_matrix)
