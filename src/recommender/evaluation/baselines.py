from typing import Dict, List
import random
import pandas as pd


def popularity_baseline(
    train_ratings: pd.DataFrame,
    top_k: int,
) -> List[int]:
    """
    Return top-K most popular item_ids based on train interactions.
    """
    popularity = (
        train_ratings["movie_id"]
        .value_counts()
        .sort_values(ascending=False)
    )
    return popularity.head(top_k).index.tolist()


def random_baseline(
    candidate_item_ids: List[int],
    top_k: int,
    seed: int = 42,
) -> List[int]:
    """
    Return top-K random items from candidate set (deterministic with seed).
    """
    random.seed(seed)
    return random.sample(candidate_item_ids, k=top_k)

