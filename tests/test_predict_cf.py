import os
import sys

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

# Add the project's src directory to sys.path (relative to the tests/ directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from recommender.domain.predict_cf import predict_rating


@pytest.fixture(scope="session")
def data():
    """Load required CSV fixtures once for all tests in this module."""
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

    user_movie_matrix = pd.read_csv(
        os.path.join(base_path, "user_movie_matrix.csv"),
        index_col=0,
    )
    similarity_df = pd.read_csv(
        os.path.join(base_path, "similarity_matrix.csv"),
        index_col=0,
    )
    users_df = pd.read_csv(
        os.path.join(base_path, "users.csv"),
        encoding="utf-8-sig",
    )

    user_dict = dict(zip(users_df.userId, users_df.username))
    return user_movie_matrix, similarity_df, user_dict


@pytest.mark.parametrize(
    "user_id,movie_id",
    [
        (1, 10),
        (1, 50),
        (1, 100),
        (3, 10),
        (3, 50),
        (5, 100),
    ],
)
def test_predict_rating_range(data, user_id, movie_id):
    """Ensure predicted ratings are in a valid range (0 <= rating <= 5)."""
    user_movie_matrix, similarity_df, user_dict = data

    rating = predict_rating(
        user_id=user_id,
        movie_id=movie_id,
        user_movie_matrix=user_movie_matrix,
        similarity_df=similarity_df,
    )

    username = user_dict.get(user_id, f"user_id={user_id}")
    assert 0 <= rating <= 5, (
        f"Invalid rating {rating:.4f}. "
        f"Expected 0 <= rating <= 5. Context: user={username}, movie_id={movie_id}"
    )

