import pytest
pytestmark = pytest.mark.integration

import pandas as pd

from recommender.evaluation.runner import evaluate_baselines


def test_evaluation_quality_gate_baselines():
    ratings_raw = pd.read_csv("data/ratings.csv")
    ratings = ratings_raw.rename(columns={"userId": "user_id", "movieId": "movie_id"})

    metrics = evaluate_baselines(ratings=ratings, top_k=10, seed=42)

    users = metrics["meta"]["users_evaluated"]
    assert users > 0, "No users evaluated; evaluation pipeline is broken."

    pop_p = metrics["popularity"]["precision@k"]
    pop_r = metrics["popularity"]["recall@k"]
    rnd_p = metrics["random"]["precision@k"]
    rnd_r = metrics["random"]["recall@k"]

    # Gate 1: Popularity should not be worse than Random (basic sanity)
    assert pop_p >= rnd_p
    assert pop_r >= rnd_r

    # Gate 2: Popularity must be non-zero (otherwise something is wrong with split/candidates)
    assert pop_r > 0.0
