from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Tuple

import pandas as pd
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

from .config import MongoConfig
from .logging_utils import configure_logger


@contextmanager
def mongo_client(config: MongoConfig, logger: logging.Logger | None = None) -> Iterator[MongoClient]:
    """
    Context manager that safely opens and closes a MongoClient.

    Ensures proper cleanup and production-grade error handling.
    """
    _logger = logger or configure_logger()
    _logger.info("Opening MongoDB connection", extra={"event": "mongo_connect"})

    client = MongoClient(config.uri, serverSelectionTimeoutMS=10_000)

    try:
        client.admin.command("ping")  # Fail fast if unreachable
        _logger.info("MongoDB connection established", extra={"event": "mongo_connect_success"})
        yield client
    except (ServerSelectionTimeoutError, PyMongoError) as exc:
        _logger.error(
            "Failed to connect to MongoDB",
            extra={"event": "mongo_connect_failure", "exception_type": type(exc).__name__},
            exc_info=True,
        )
        raise
    finally:
        client.close()
        _logger.info("MongoDB connection closed", extra={"event": "mongo_disconnect"})


def get_collections(db: Database, config: MongoConfig) -> Tuple[Collection, Collection]:
    """
    Retrieve movies and ratings collections from the database.
    """
    return db[config.movies_collection], db[config.ratings_collection]


def load_movies_and_ratings(
    config: MongoConfig,
    logger: logging.Logger | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load movies and ratings from MongoDB into pandas DataFrames.

    Includes JSON-structured logging for production observability.
    """
    _logger = logger or configure_logger()

    with mongo_client(config=config, logger=_logger) as client:
        db = client[config.db_name]
        movies_collection, ratings_collection = get_collections(db, config)

        # Load movies
        _logger.info("Loading movies from MongoDB", extra={"event": "load_movies"})
        movies_data = list(movies_collection.find({}, {"_id": 0}))
        movies_df = pd.DataFrame(movies_data)
        _logger.info(
            "Movies loaded",
            extra={"event": "load_movies_success", "shape": movies_df.shape},
        )

        # Load ratings
        _logger.info("Loading ratings from MongoDB", extra={"event": "load_ratings"})
        ratings_data = list(ratings_collection.find({}, {"_id": 0}))
        ratings_df = pd.DataFrame(ratings_data)
        _logger.info(
            "Ratings loaded",
            extra={"event": "load_ratings_success", "shape": ratings_df.shape},
        )

    return movies_df, ratings_df

