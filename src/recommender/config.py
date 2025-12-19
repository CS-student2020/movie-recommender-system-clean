from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class MongoConfig:
    """
    Configuration for connecting to MongoDB.

    Attributes:
        uri: Full MongoDB connection string (from environment).
        db_name: Name of the database.
        movies_collection: Collection name for movies.
        ratings_collection: Collection name for ratings.
    """
    uri: str
    db_name: str = "movie_recommender_db"
    movies_collection: str = "movies"
    ratings_collection: str = "ratings"


@dataclass(frozen=True)
class SimilarityConfig:
    """
    Configuration for similarity matrix computation and persistence.

    Attributes:
        user_user_output_csv_path: CSV path for user-user similarity matrix.
        item_item_output_csv_path: CSV path for item-item similarity matrix.
    """
    user_user_output_csv_path: Path = Path("data/similarity_matrix_user_user.csv")
    item_item_output_csv_path: Path = Path("data/similarity_matrix_item_item.csv")


@dataclass(frozen=True)
class AppConfig:
    """
    Top-level configuration object for the similarity pipelines.
    """
    mongo: MongoConfig
    similarity: SimilarityConfig


def load_mongo_config_from_env(env_var_name: str = "MONGO_URI_DEV") -> MongoConfig:
    """
    Load MongoDB configuration from environment variables.

    Args:
        env_var_name: Name of the environment variable that stores the Mongo URI.

    Raises:
        ConfigError: If the environment variable is missing or empty.
    """
    load_dotenv()
    uri = os.getenv(env_var_name)

    if not uri:
        raise ConfigError(f"Environment variable {env_var_name!r} is not set or empty.")

    return MongoConfig(uri=uri)


def load_app_config(env_var_name: str = "MONGO_URI_DEV") -> AppConfig:
    """
    Construct and return the full application configuration.

    Args:
        env_var_name: Environment variable from which to load the Mongo URI.

    Returns:
        AppConfig
    """
    mongo_cfg = load_mongo_config_from_env(env_var_name)
    similarity_cfg = SimilarityConfig()
    return AppConfig(mongo=mongo_cfg, similarity=similarity_cfg)
