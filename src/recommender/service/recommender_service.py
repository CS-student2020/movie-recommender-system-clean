"""
Service layer for the movie recommender system.

This module defines the high-level service interface that will be consumed
by upper layers such as FastAPI, CLIs, or batch pipelines.

Important:
    - This module does NOT perform any I/O (no DB, no CSV, no config loading).
    - This module does NOT own ML/CF logic.
    - This module ONLY orchestrates domain functionality and exposes
      type-safe contracts for recommendation-related use cases.

Data and algorithm execution dependencies are injected from the outside
(e.g., in application bootstrap code or API wiring).
"""

from dataclasses import dataclass, replace
from typing import List, Optional

import pandas as pd

from ..domain.recommend_for_user import (
    recommend_for_user,
    RecommendParams,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _rank_scores(n: int) -> List[float]:
    """
    Convert rank positions into a decreasing score in (0, 1].

    This is a pragmatic product-facing fallback when upstream model scores
    are degenerate (e.g., all equal). It preserves ordering while providing
    a meaningful relative score for UI/clients.

    Example (n=5): [1.0, 0.9, 0.8, 0.7, 0.6]
    """
    if n <= 0:
        return []
    if n == 1:
        return [1.0]

    # Linear decay from 1.0 down to 0.6 across n items.
    # (Keeps values "human-feeling" and avoids near-zero tails for small n.)
    return [1.0 - (i / (n - 1)) * 0.4 for i in range(n)]


def _scores_are_degenerate(series: pd.Series, eps: float = 1e-9) -> bool:
    """
    Return True if a numeric series has effectively no variance.
    """
    if series.empty:
        return True
    s = series.astype(float)
    return (float(s.max()) - float(s.min())) < eps


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
        Internal movie identifier (as used in the dataset).
    score:
        Relevance score for the recommendation. Higher means better.
        Note: If upstream model scores are degenerate, this may be a
        rank-derived relative score in (0, 1] to remain meaningful.
    title:
        Optional movie title, if available from movies metadata.
    """

    movie_id: int
    score: float
    title: Optional[str] = None


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
# Service Layer
# ---------------------------------------------------------------------


class RecommenderService:
    """
    High-level service for user-facing recommendation use cases.

    This service is intentionally thin. It exposes contracts for retrieving
    recommendations and similarity results, while delegating the actual
    collaborative filtering and ranking logic to the domain layer.

    Design principles:
        - No I/O in this layer.
        - No direct database or CSV access.
        - No model training or heavy computation.
        - Pure orchestration of already-prepared data and domain functions.
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        user_user_sim: pd.DataFrame,
        item_item_sim: pd.DataFrame,
        default_params: Optional[RecommendParams] = None,
    ) -> None:
        """
        Construct a RecommenderService instance.

        Parameters
        ----------
        ratings_df:
            Pre-loaded ratings data as a pandas DataFrame.
        movies_df:
            Pre-loaded movies metadata as a pandas DataFrame.
        user_user_sim:
            Pre-loaded user-user similarity matrix as a pandas DataFrame.
        item_item_sim:
            Pre-loaded item-item similarity matrix as a pandas DataFrame.
        default_params:
            Optional default recommendation hyperparameters. If None,
            the domain-level defaults will be used.
        """
        self._ratings_df = ratings_df
        self._movies_df = movies_df
        self._user_user_sim = user_user_sim
        self._item_item_sim = item_item_sim
        self._default_params = default_params

    # ------------------------------------------------------------------
    # Public API - Recommendations
    # ------------------------------------------------------------------

    def get_recommendations_for_user(
        self,
        user_id: int,
        limit: int = 10,
        min_score: Optional[float] = None,
        params: Optional[RecommendParams] = None,
    ) -> List[Recommendation]:
        """
        Retrieve recommendations for a given user.

        This method orchestrates the call to the domain layer
        (e.g., collaborative filtering + ranking) and maps the result
        into a type-safe list of Recommendation objects.

        Parameters
        ----------
        user_id:
            User identifier in the dataset.
        limit:
            Maximum number of recommendations to return.
        min_score:
            Optional score threshold. If provided, recommendations
            with scores below this value will be filtered out.
            Note: If upstream scores are degenerate and we fall back to
            rank-derived scores, `min_score` applies to those derived scores.
        params:
            Optional per-call recommendation parameters. If not provided,
            the service's default_params are used (if any), otherwise
            domain defaults are used.

        Returns
        -------
        List[Recommendation]
            List of typed Recommendation objects. Length is at most `limit`.
        """
        # Resolve effective params: call-level > default > domain defaults
        effective_params = params or self._default_params or RecommendParams()

        # Ensure top_k is aligned with the service-level limit
        if effective_params.top_k != limit:
            effective_params = replace(effective_params, top_k=limit)

        domain_df = recommend_for_user(
            user_id=user_id,
            movies_df=self._movies_df,
            ratings_df=self._ratings_df,
            user_user_sim=self._user_user_sim,
            item_item_sim=self._item_item_sim,
            params=effective_params,
        )

        # Defensive: handle empty output early
        if domain_df is None or getattr(domain_df, "empty", True):
            return []

        # Optionally filter by minimum score, if the score column exists.
        # (We may re-apply a similar filter after rank-score fallback.)
        if min_score is not None and "score" in domain_df.columns:
            domain_df = domain_df[domain_df["score"] >= min_score]

        if domain_df.empty:
            return []

        # Decide whether to use rank-derived score fallback.
        use_rank_score = False
        if "score" not in domain_df.columns:
            use_rank_score = True
        else:
            use_rank_score = _scores_are_degenerate(domain_df["score"])

        # Reset index for stable rank scoring and iteration
        domain_df = domain_df.reset_index(drop=True)

        rank_score_list: Optional[List[float]] = None
        if use_rank_score:
            rank_score_list = _rank_scores(len(domain_df))

            # If min_score was requested but upstream scores were degenerate,
            # apply the filter on rank-derived scores too.
            if min_score is not None:
                keep_idx = [i for i, s in enumerate(rank_score_list) if s >= min_score]
                domain_df = domain_df.loc[keep_idx].reset_index(drop=True)
                rank_score_list = [rank_score_list[i] for i in keep_idx]

            if domain_df.empty:
                return []

        # Map the domain DataFrame to a list of typed Recommendation objects.
        # Expected columns: movieId, score, support, reasons, optional title.
        has_title = "title" in domain_df.columns

        recommendations: List[Recommendation] = []
        for i, (_, row) in enumerate(domain_df.iterrows()):
            title_val: Optional[str] = None
            if has_title:
                val = row["title"]
                if isinstance(val, str):
                    title_val = val

            if use_rank_score and rank_score_list is not None:
                score_val = float(rank_score_list[i])
            else:
                score_val = float(row["score"])

            recommendations.append(
                Recommendation(
                    movie_id=int(row["movieId"]),
                    score=score_val,
                    title=title_val,
                )
            )

        # Enforce the `limit` at the service boundary as well.
        return recommendations[:limit]

    # ------------------------------------------------------------------
    # Public API - Similarity (placeholder, to be wired later)
    # ------------------------------------------------------------------

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
            List of typed SimilarMovie objects. Length is at most `limit`.

        Notes
        -----
        This is a placeholder. In a later iteration, this method will invoke
        a similarity engine from the domain layer (e.g., cosine similarity
        over a precomputed similarity matrix).
        """
        raise NotImplementedError(
            "get_similar_movies is not implemented yet. "
            "It will be wired to domain similarity computations "
            "in a later iteration."
        )
