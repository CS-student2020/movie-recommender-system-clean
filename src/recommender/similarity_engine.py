from __future__ import annotations

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .validators import validate_ratings_schema


def prepare_user_item_matrix(
    ratings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a dense user–item matrix from a long-form ratings DataFrame.

    This function belongs to the pure *Domain / Code Layer*:
    it performs in-memory computations only and has no side effects
    (no file I/O, no database access, no logging, no printing).

    Parameters
    ----------
    ratings_df:
        Long-form DataFrame with at least the following columns:
        - 'userId': user identifier
        - 'movieId': movie/item identifier
        - 'rating': numeric rating value

    Returns
    -------
    pd.DataFrame
        Pivoted user–item matrix:

        - index: userId
        - columns: movieId
        - values: rating (float)

        Missing entries are filled with 0.0 to produce a dense matrix
        suitable for cosine similarity.
    """
    validate_ratings_schema(ratings_df)

    pivot_df = ratings_df.pivot(
        index="userId",
        columns="movieId",
        values="rating",
    )

    # For cosine similarity we use a dense matrix (no NaN)
    pivot_df = pivot_df.fillna(0.0).astype(float)

    return pivot_df


def compute_user_user_cosine_similarity(
    user_item_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cosine similarity between users.

    Parameters
    ----------
    user_item_matrix:
        Dense user–item matrix with users as rows and items as columns.

    Returns
    -------
    pd.DataFrame
        User–user similarity matrix:

        - index: userId
        - columns: userId
        - values: cosine similarity in [0, 1].
    """
    similarity_matrix = cosine_similarity(user_item_matrix.values)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_item_matrix.index,
        columns=user_item_matrix.index,
    )

    return similarity_df


def build_user_user_similarity_from_ratings(
    ratings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convenience pipeline: ratings → user–item matrix → user–user similarity.

    Parameters
    ----------
    ratings_df:
        Long-form ratings DataFrame.

    Returns
    -------
    pd.DataFrame
        User–user cosine similarity matrix.
    """
    user_item_matrix = prepare_user_item_matrix(ratings_df)
    return compute_user_user_cosine_similarity(user_item_matrix)


def prepare_item_user_matrix(
    ratings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a dense item–user matrix from a long-form ratings DataFrame.

    This function belongs to the pure *Domain / Code Layer*:
    it performs in-memory computations only and has no side effects
    (no file I/O, no database access, no logging, no printing).

    Parameters
    ----------
    ratings_df:
        Long-form DataFrame with at least the following columns:
        - 'userId': user identifier
        - 'movieId': movie/item identifier
        - 'rating': numeric rating value

    Returns
    -------
    pd.DataFrame
        Pivoted item–user matrix:

        - index: movieId
        - columns: userId
        - values: rating (float)

        Missing entries are filled with 0.0 to produce a dense matrix
        suitable for cosine similarity.
    """
    validate_ratings_schema(ratings_df)

    pivot_df = ratings_df.pivot(
        index="movieId",
        columns="userId",
        values="rating",
    )

    pivot_df = pivot_df.fillna(0.0).astype(float)

    return pivot_df


def compute_item_item_cosine_similarity(
    item_user_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cosine similarity between items (movies).

    Parameters
    ----------
    item_user_matrix:
        Dense item–user matrix with items as rows and users as columns.

    Returns
    -------
    pd.DataFrame
        Item–item similarity matrix:

        - index: movieId
        - columns: movieId
        - values: cosine similarity in [0, 1].
    """
    similarity_matrix = cosine_similarity(item_user_matrix.values)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=item_user_matrix.index,
        columns=item_user_matrix.index,
    )

    return similarity_df


def build_item_item_similarity_from_ratings(
    ratings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convenience pipeline: ratings → item–user matrix → item–item similarity.

    Parameters
    ----------
    ratings_df:
        Long-form ratings DataFrame.

    Returns
    -------
    pd.DataFrame
        Item–item cosine similarity matrix.
    """
    item_user_matrix = prepare_item_user_matrix(ratings_df)
    return compute_item_item_cosine_similarity(item_user_matrix)
