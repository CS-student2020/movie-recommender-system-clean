import pytest

# Mark the entire module as an integration test.
# This test depends on real CSV data files and must NOT run in CI.
pytestmark = pytest.mark.integration

import os
import pandas as pd

from recommender.evaluate_model import evaluate_cf_model


def test_evaluate_cf_model():
    """
    Integration test for collaborative filtering evaluation.

    This test validates that the evaluation pipeline runs correctly
    on real precomputed CSV artifacts (ratings matrix, similarity matrix).

    Expected behavior:
    - Metrics are computed without crashing
    - MAE and RMSE are non-negative
    - RMSE >= MAE (basic sanity check)
    """

    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data")
    )

    ratings_df = pd.read_csv(
        os.path.join(base_path, "ratings.csv")
    )

    user_movie_matrix = pd.read_csv(
        os.path.join(base_path, "user_movie_matrix.csv"),
        index_col=0,
    )

    similarity_df = pd.read_csv(
        os.path.join(base_path, "similarity_matrix.csv"),
        index_col=0,
    )

    mae, rmse = evaluate_cf_model(
        ratings_df,
        user_movie_matrix,
        similarity_df,
    )

    assert mae >= 0
    assert rmse >= 0
    assert rmse >= mae

