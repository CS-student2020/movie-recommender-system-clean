"""
Service layer for the movie recommender system.

This module defines the high-level service interface that will be consumed
by upper layers such as FastAPI, CLIs, or batch pipelines.

Important:
    - This module does NOT load data (no DB, no CSV, no config).
    - This module does NOT own ML/CF logic.
    - This module ONLY orchestrates domain/core functionality and exposes
      type-safe contracts for recommendation-related use cases.

Data and algorithm execution dependencies will be injected from outside
(e.g., in application "bootstrap" code).
"""

from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------
# Typed Return Models
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Recommendation:
    """
    Typed model representing a recommended movie for a user.

    Attributes
    ----------
    movie_id:
        Internal movie identifier (as used in the dataset/core).
    score:
        Relevance score for the recommendation. Higher means better.
    """
    movie_id: int
    score: float


@dataclass(frozen=True)
class SimilarMovie:
    """
    Typed model representing a movie similar to another reference movie.

    Attributes
    ----------
    movie_id:
        Internal movie identifier of the similar movie.
    similarity:
        Numeric similarity coefficient (e.g., cosine similarity).
        Range and interpretation depend on the similarity method.
    """
    movie_id: int
    similarity: float


# ---------------------------------------------------------------------
# Service Layer - Contract Only (No Implementation Yet)
# ---------------------------------------------------------------------

class RecommenderService:
    """
    High-level service for user-facing recommendation use cases.

    This service is intentionally thin at this stage. It exposes contracts
    for retrieving recommendations and similarity results, while delegating
    the actual collaborative filtering logic to the domain/core layer.

    Core/Domain logic (e.g., recommend_for_user, ranking, similarity) will
    be called from here in later stages, once dependency injection is wired.

    NOTE:
        This file currently defines the "shape" of the service.
        Implementation will be added once domain functions are integrated.
    """

    def __init__(
        self,
        # Example future dependencies:
        # ratings_data: "RatingsRepository",
        # similarity_engine: "SimilarityEngine",
        # svd_model: "MatrixFactorization",
    ) -> None:
        """
        Construct a RecommenderService instance.

        For now we do not store dependencies. They will be added once
        the application bootstrap layer is defined.
        """
        pass

    def get_recommendations_for_user(
        self,
        user_id: int,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[Recommendation]:
        """
        Retrieve recommendations for a given user.

        Parameters
        ----------
        user_id:
            User identifier in the dataset/core.
        limit:
            Maximum number of recommendations to return.
        min_score:
            Optional threshold for filtering low-score results.

        Returns
        -------
        List[Recommendation]
            List of typed Recommendation objects. Length <= limit.

        Implementation Note
        -------------------
        In the next stage, this will call into core/domain functions
        such as recommend_for_user(...) and ranking logic.
        """
        raise NotImplementedError(
            "get_recommendations_for_user is not implemented yet. "
            "It will later call into domain/core collaborative filtering."
        )

    def get_similar_movies(
        self,
        movie_id: int,
        limit: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[SimilarMovie]:
        """
        Retrieve movies similar to a given reference movie.

        Parameters
        ----------
        movie_id:
            Movie identifier to compute similarity against.
        limit:
            Maximum number of similar movies to return.
        min_similarity:
            Optional threshold to filter out weak similarities.

        Returns
        -------
        List[SimilarMovie]
            List of typed SimilarMovie objects. Length <= limit.

        Implementation Note
        -------------------
        In the next stage, this will call into similarity logic
        (e.g., similarity engine, cosine similarity, etc.).
        """
        raise NotImplementedError(
            "get_similar_movies is not implemented yet. "
            "It will later call into domain/core similarity computations."
        )
