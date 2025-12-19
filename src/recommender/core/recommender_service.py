from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..config import AppConfig, load_app_config
from ..mongo_loader import load_movies_and_ratings
from .recommend_for_user import RecommendParams, recommend_for_user


@dataclass
class RecommenderService:
    """
    High-level service exposing recommendation use cases.

    This class is designed to be used by:
        - FastAPI endpoints
        - CLI tools
        - offline batch processors
        - Jupyter / notebooks

    Responsibilities:
        - Load ratings and similarity matrices.
        - Provide user-facing recommendation utilities.
        - Act as the middle layer between algorithms & API.
    """

    ratings_df: pd.DataFrame
    user_user_similarity: Optional[pd.DataFrame] = None
    item_item_similarity: Optional[pd.DataFrame] = None
    logger: logging.Logger = logging.getLogger("recommender.service")

    # ------------------------------------------------------------
    # Factory Constructor
    # ------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        app_config: Optional[AppConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "RecommenderService":
        """
        Build a RecommenderService instance from application config.

        Loads:
            - ratings from MongoDB
            - precomputed similarity matrices from disk (if present)
        """
        if app_config is None:
            app_config = load_app_config()

        logger = logger or logging.getLogger("recommender.service")

        # Load data from MongoDB
        movies_df, ratings_df = load_movies_and_ratings(
            config=app_config.mongo,
            logger=logger,
        )

        logger.info(
            "Ratings loaded for recommender service",
            extra={
                "event": "service_ratings_loaded",
                "shape": ratings_df.shape,
            },
        )

        # Load similarity matrices if available
        user_user_sim = None
        item_item_sim = None

        try:
            user_user_sim = pd.read_csv(
                app_config.similarity.user_user_output_csv_path,
                index_col=0,
            )
            
            # 🔧 NORMALIZE IDS
            user_user_sim.index = user_user_sim.index.astype(int)
            user_user_sim.columns = user_user_sim.columns.astype(int)
            
            logger.info(
                "User-user similarity matrix loaded",
                extra={
                    "event": "service_user_user_similarity_loaded",
                    "shape": user_user_sim.shape,
                },
            )
        except FileNotFoundError:
            logger.warning(
                "User-user similarity matrix not found",
                extra={"event": "service_user_user_similarity_missing"},
            )

        try:
            item_item_sim = pd.read_csv(
                app_config.similarity.item_item_output_csv_path,
                index_col=0,
            )
            logger.info(
                "Item-item similarity matrix loaded",
                extra={
                    "event": "service_item_item_similarity_loaded",
                    "shape": item_item_sim.shape,
                },
            )
        except FileNotFoundError:
            logger.warning(
                "Item-item similarity matrix not found",
                extra={"event": "service_item_item_similarity_missing"},
            )

        return cls(
            ratings_df=ratings_df,
            user_user_similarity=user_user_sim,
            item_item_similarity=item_item_sim,
            logger=logger,
        )

    # ------------------------------------------------------------
    # Public API Methods (to be used by API / CLI)
    # ------------------------------------------------------------
    def similar_items(self, movie_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Return the top_k most similar movies to the given movie_id
        using the item-item similarity matrix.
        """
        if self.item_item_similarity is None:
            raise RuntimeError("Item-item similarity matrix is not loaded.")

        if movie_id not in self.item_item_similarity.index:
            raise ValueError(f"movie_id {movie_id} not found in similarity matrix.")

        sims = self.item_item_similarity.loc[movie_id]
        sims = sims.drop(labels=[movie_id], errors="ignore")
        top = sims.sort_values(ascending=False).head(top_k)

        result = top.reset_index()
        result.columns = ["movieId", "similarity"]

        self.logger.info(
            "Similar items fetched",
            extra={
                "event": "service_similar_items",
                "movie_id": movie_id,
                "top_k": top_k,
            },
        )

        return result

    def similar_users(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Return the top_k most similar users to the specified user_id
        using user-user similarity matrix.
        """
        if self.user_user_similarity is None:
            raise RuntimeError("User-user similarity matrix is not loaded.")

        if user_id not in self.user_user_similarity.index:
            raise ValueError(f"user_id {user_id} not found in similarity matrix.")

        sims = self.user_user_similarity.loc[user_id]
        sims = sims.drop(labels=[user_id], errors="ignore")
        top = sims.sort_values(ascending=False).head(top_k)

        result = top.reset_index()
        result.columns = ["userId", "similarity"]

        self.logger.info(
            "Similar users fetched",
            extra={
                "event": "service_similar_users",
                "user_id": user_id,
                "top_k": top_k,
            },
        )

        return result

    # ------------------------------------------------------------
    # Core Recommendation API
    # ------------------------------------------------------------
    def recommend_for_user(
        self,
        user_id: int,
        *,
        top_k: int = 10,
        neighbors_k: int = 50,
        min_similarity: float = 0.0,
        min_raters: int = 1,
        alpha: float = 1.0,
        explain: bool = False,
        max_reasons: int = 3,
    ) -> pd.DataFrame:

        """
        Recommend movies for a given user using collaborative filtering.

        Supports:
            - User-User CF (alpha = 1.0)
            - Item-Item CF (alpha = 0.0)
            - Hybrid CF     (0.0 < alpha < 1.0)

        Returns:
            DataFrame with columns:
                - movieId
                - score
                - support
                - (optional metadata if added later)
        """
        if self.user_user_similarity is None:
            raise RuntimeError("User-user similarity matrix is not loaded.")

        if alpha < 1.0 and self.item_item_similarity is None:
            raise RuntimeError(
                "Item-item similarity matrix is required when alpha < 1.0"
            )

        params = RecommendParams(
            top_k=top_k,
            neighbors_k=neighbors_k,
            min_similarity=min_similarity,
            min_raters=min_raters,
            alpha=alpha,
            explain=explain,
            max_reasons=max_reasons,
        )


        self.logger.info(
            "Generating recommendations for user",
            extra={
                "event": "service_recommend_for_user_start",
                "user_id": user_id,
                "top_k": top_k,
                "alpha": alpha,
            },
        )

        recommendations = recommend_for_user(
            user_id=user_id,
            movies_df=None,  # enrichment layer can be added later
            ratings_df=self.ratings_df,
            user_user_sim=self.user_user_similarity,
            item_item_sim=self.item_item_similarity,
            params=params,
        )

        self.logger.info(
            "Recommendations generated",
            extra={
                "event": "service_recommend_for_user_done",
                "user_id": user_id,
                "num_recommendations": len(recommendations),
            },
        )

        return recommendations
