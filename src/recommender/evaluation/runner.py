import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from .baselines import random_baseline
from .protocol import precision_at_k, recall_at_k


def _validate_ratings_schema(ratings: pd.DataFrame) -> None:
    required = {"user_id", "movie_id"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(
            f"ratings is missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(ratings.columns)}"
        )


def _build_per_user_holdout(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leave-one-out per user:
    - test: last interaction row per user (by row order)
    - train: all other rows
    Users with <2 interactions are excluded from evaluation (both train/test for that user).
    """
    counts = ratings.groupby("user_id").size()
    eligible_users = counts[counts >= 2].index

    eligible = ratings[ratings["user_id"].isin(eligible_users)].copy()

    last_idx = eligible.groupby("user_id").tail(1).index
    test = eligible.loc[last_idx].copy()
    train = eligible.drop(index=last_idx).copy()

    return train, test


def _global_popularity_ranking(train_ratings: pd.DataFrame) -> List[int]:
    """
    Global popularity ranking over the entire training set.
    """
    popularity = train_ratings["movie_id"].value_counts().sort_values(ascending=False)
    return popularity.index.tolist()


def evaluate_baselines(
    ratings: pd.DataFrame,
    top_k: int = 10,
    seed: int = 42,
) -> Dict:
    _validate_ratings_schema(ratings)

    train, test = _build_per_user_holdout(ratings)

    global_ranked_items = _global_popularity_ranking(train)
    candidate_item_ids: List[int] = global_ranked_items[:]  # candidates = items seen in train universe

    results = {
        "random": {"precision@k": 0.0, "recall@k": 0.0},
        "popularity": {"precision@k": 0.0, "recall@k": 0.0},
        "meta": {"users_evaluated": 0, "k": top_k},
    }

    precision_random: List[float] = []
    recall_random: List[float] = []
    precision_pop: List[float] = []
    recall_pop: List[float] = []

    # group train by user for "seen" set
    train_by_user = train.groupby("user_id")
    test_by_user = test.groupby("user_id")

    for user_id, test_df in test_by_user:
        test_item = int(test_df.iloc[0]["movie_id"])
        relevant: Set[int] = {test_item}

        user_train = train_by_user.get_group(user_id)
        seen: Set[int] = set(user_train["movie_id"].astype(int).tolist())

        # candidate items excluding what user has already seen in TRAIN
        available = [mid for mid in candidate_item_ids if mid not in seen]

        # Popularity baseline: take top-K from global ranking excluding seen
        pop_rec = available[:top_k]

        # Random baseline: sample from the same available set (deterministic per user)
        if len(available) >= top_k:
            rand_rec = random_baseline(available, top_k, seed=seed + int(user_id))
        else:
            # edge case: not enough candidates
            rand_rec = available[:]

        precision_pop.append(precision_at_k(pop_rec, relevant, top_k))
        recall_pop.append(recall_at_k(pop_rec, relevant, top_k))

        precision_random.append(precision_at_k(rand_rec, relevant, top_k))
        recall_random.append(recall_at_k(rand_rec, relevant, top_k))

    if precision_random:
        results["random"]["precision@k"] = sum(precision_random) / len(precision_random)
        results["random"]["recall@k"] = sum(recall_random) / len(recall_random)
        results["popularity"]["precision@k"] = sum(precision_pop) / len(precision_pop)
        results["popularity"]["recall@k"] = sum(recall_pop) / len(recall_pop)
        results["meta"]["users_evaluated"] = len(precision_random)

    return results


def run_and_save(
    ratings: pd.DataFrame,
    output_path: str = "metrics.json",
    top_k: int = 10,
    seed: int = 42,
) -> Dict:
    metrics = evaluate_baselines(
        ratings=ratings,
        top_k=top_k,
        seed=seed,
    )

    path = Path(output_path)
    path.write_text(json.dumps(metrics, indent=2))
    return metrics
