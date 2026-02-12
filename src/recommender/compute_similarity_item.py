"""
Item–Item similarity computation (pure Domain / Code Layer).

This module exposes a single public function:

    compute_item_item_similarity(user_item_matrix, ...)

which:
- Takes an in-memory user–item rating matrix (pandas.DataFrame),
- Computes an item–item similarity matrix (pandas.DataFrame),
- Has *no* side effects (no file I/O, no DB access, no logging, no printing).

Typical usage from a higher layer (e.g. a service or script):

    from src.recommender.compute_similarity_item import compute_item_item_similarity

    similarity_df = compute_item_item_similarity(user_movie_matrix)
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


SimilarityMetric = Literal["cosine"]


def _cosine_similarity_1d(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two 1D vectors.

    Both x and y are assumed to be 1D NumPy arrays with the same length.
    If either vector has zero norm, this function returns 0.0.
    """
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def compute_item_item_similarity(
    user_item_matrix: pd.DataFrame,
    min_common_users: int = 1,
    metric: SimilarityMetric = "cosine",
    fill_diagonal: Optional[float] = 1.0,
) -> pd.DataFrame:
    """
    Compute an item–item similarity matrix from a user–item rating matrix.

    Parameters
    ----------
    user_item_matrix:
        DataFrame of shape (n_users, n_items).
        - Rows: users
        - Columns: items (e.g. movie IDs)
        - Values: numeric ratings (floats or ints); may contain NaN.

    min_common_users:
        Minimum number of users who must have rated *both* items in order
        for their similarity to be considered. If the number of common
        users is below this threshold, the similarity is set to 0.0.

    metric:
        Similarity metric to use. Currently only "cosine" is supported.
        The parameter is kept as a Literal to make future extensions
        (e.g. "pearson") easier without changing the public API.

    fill_diagonal:
        Value to fill on the diagonal of the similarity matrix
        (similarity of an item with itself). If None, the diagonal is left
        as computed (which in practice would be 1.0 for cosine).

    Returns
    -------
    pandas.DataFrame
        DataFrame of shape (n_items, n_items) with:
        - index: item IDs (same as user_item_matrix.columns)
        - columns: item IDs
        - values: similarity scores in [−1.0, 1.0] for cosine.

    Notes
    -----
    - This function is pure Domain / Code Layer:
      it performs in-memory computations only and has no side effects.
    - It does not perform any file or database I/O and does not log
      or print anything.
    """
    if metric != "cosine":
        raise ValueError(f"Unsupported similarity metric: {metric!r}")

    # If there are no items, return an empty similarity matrix.
    if user_item_matrix.shape[1] == 0:
        return pd.DataFrame(
            data=np.empty((0, 0), dtype=np.float64),
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns,
        )

    # Convert to NumPy for efficient numeric operations.
    # We work on a copy to avoid mutating the caller's DataFrame.
    values = user_item_matrix.to_numpy(dtype=float, copy=True)

    # Mask indicating which entries are *not* NaN.
    mask = ~np.isnan(values)

    n_users, n_items = values.shape
    similarities = np.zeros((n_items, n_items), dtype=np.float64)

    # Compute upper triangle (including diagonal), then mirror.
    for i in range(n_items):
        # An item is always maximally similar to itself;
        # we may overwrite this later with fill_diagonal.
        similarities[i, i] = 1.0

        for j in range(i + 1, n_items):
            # Users who rated both item i and item j
            common = mask[:, i] & mask[:, j]
            n_common = int(common.sum())

            if n_common < min_common_users or n_common == 0:
                sim = 0.0
            else:
                v_i = values[common, i]
                v_j = values[common, j]
                sim = _cosine_similarity_1d(v_i, v_j)

            similarities[i, j] = sim
            similarities[j, i] = sim  # symmetry

    if fill_diagonal is not None:
        np.fill_diagonal(similarities, fill_diagonal)

    item_labels = list(user_item_matrix.columns)
    similarity_df = pd.DataFrame(
        data=similarities,
        index=item_labels,
        columns=item_labels,
    )

    return similarity_df
