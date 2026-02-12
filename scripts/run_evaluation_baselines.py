import pandas as pd

from recommender.evaluation.runner import run_and_save


def main() -> None:
    ratings_raw = pd.read_csv("data/ratings.csv")

    # Normalize to internal contract: user_id / movie_id
    ratings = ratings_raw.rename(columns={"userId": "user_id", "movieId": "movie_id"})

    metrics = run_and_save(
        ratings=ratings,
        output_path="metrics.json",
        top_k=10,
        seed=42,
    )

    print("Wrote metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
