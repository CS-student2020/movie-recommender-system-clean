# scripts/manual_recommender_demo.py

from pathlib import Path
import pandas as pd

from src.recommender.service.recommender_service import RecommenderService
from src.recommender.core.recommend_for_user import RecommendParams


BASE_DIR = Path(__file__).resolve().parents[1]
RATINGS_PATH = BASE_DIR / "data" / "ratings.csv"
MOVIES_PATH = BASE_DIR / "data" / "movies.csv"
USER_USER_SIM_PATH = BASE_DIR / "data" / "similarity_matrix_user_user.csv"
ITEM_ITEM_SIM_PATH = BASE_DIR / "data" / "similarity_matrix_item_item.csv"


def load_and_normalize_ratings(path: Path) -> pd.DataFrame:
    """
    Load the ratings CSV and normalize column names and dtypes
    to the schema expected by the core/domain layer:
        - columns: userId, movieId, rating
        - dtypes: userId -> int, movieId -> int, rating -> float
    """
    df = pd.read_csv(path)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Rename common variants
    rename_map = {
        "user_id": "userId",
        "UserId": "userId",
        "USER_ID": "userId",
        "movie_id": "movieId",
        "MovieId": "movieId",
        "MOVIE_ID": "movieId",
        "Rating": "rating",
        "rating_value": "rating",
        "RATING": "rating",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    required = {"userId", "movieId", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"ratings_df missing required columns {sorted(missing)}; "
            f"available={list(df.columns)}"
        )

    # Force canonical dtypes
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)

    return df


def load_and_normalize_movies(path: Path) -> pd.DataFrame:
    """
    Load the movies CSV and normalize the movieId column:
        - ensure movieId exists
        - movieId -> int
    """
    df = pd.read_csv(path)

    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "movie_id": "movieId",
        "MovieId": "movieId",
        "MOVIE_ID": "movieId",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    if "movieId" not in df.columns:
        raise ValueError(
            f"movies_df missing 'movieId'; available={list(df.columns)}"
        )

    df["movieId"] = df["movieId"].astype(int)
    return df


def main() -> None:
    print("📥 Loading ratings...")
    ratings_df = load_and_normalize_ratings(RATINGS_PATH)
    print("✅ ratings_df columns:", list(ratings_df.columns))
    print("✅ ratings_df dtypes:", ratings_df.dtypes.to_dict())

    print("📥 Loading movies...")
    movies_df = load_and_normalize_movies(MOVIES_PATH)
    print("✅ movies_df columns:", list(movies_df.columns))
    print("✅ movies_df dtypes:", movies_df.dtypes.to_dict())

    print("📥 Loading user-user similarity matrix...")
    user_user_sim = pd.read_csv(USER_USER_SIM_PATH, index_col=0)

    # Normalize user-user similarity index/columns to int to match userId
    user_user_sim.index = user_user_sim.index.astype(int)
    user_user_sim.columns = user_user_sim.columns.astype(int)

    print("✅ user_user_sim shape:", user_user_sim.shape)
    print("✅ user_user_sim index dtype:", user_user_sim.index.dtype)
    print("✅ user_user_sim columns dtype:", user_user_sim.columns.dtype)

    print("📥 Loading item-item similarity matrix...")
    item_item_sim = pd.read_csv(ITEM_ITEM_SIM_PATH, index_col=0)

    # Normalize item-item similarity index/columns to int to match movieId
    item_item_sim.index = item_item_sim.index.astype(int)
    item_item_sim.columns = item_item_sim.columns.astype(int)

    print("✅ item_item_sim shape:", item_item_sim.shape)
    print("✅ item_item_sim index dtype:", item_item_sim.index.dtype)
    print("✅ item_item_sim columns dtype:", item_item_sim.columns.dtype)

    # Pick a "strong" user automatically: the one with most ratings
    value_counts = ratings_df["userId"].value_counts()
    top_user_id = int(value_counts.index[0])
    top_user_count = int(value_counts.iloc[0])

    print(f"\n📊 Top user by rating count: userId={top_user_id} with {top_user_count} ratings")

    user_id = top_user_id

    # Small debug: check overlap of this user's movies with item_item_sim columns
    user_movies = ratings_df.loc[ratings_df["userId"] == user_id, "movieId"].unique()
    overlap_item_item = len(set(user_movies) & set(item_item_sim.columns.tolist()))
    print(f"📊 Overlap of user {user_id}'s movies with item_item_sim columns: {overlap_item_item}")

    # Instantiate service with injected in-memory dependencies
    service = RecommenderService(
        ratings_df=ratings_df,
        movies_df=movies_df,
        user_user_sim=user_user_sim,
        item_item_sim=item_item_sim,
    )

    # For demo: hybrid/item-item config.
    # alpha = 1.0 → pure user-user CF
    # alpha = 0.0 → pure item-item CF
    demo_params = RecommendParams(
        top_k=10,
        neighbors_k=50,
        min_similarity=0.0,
        min_raters=1,
        alpha=0.0,  # start with pure item-item
        explain=False,
    )

    print(f"\n🎬 Getting recommendations for user {user_id} with alpha={demo_params.alpha}...")
    recommendations = service.get_recommendations_for_user(
        user_id=user_id,
        limit=10,
        params=demo_params,
    )

    print(f"\n🔮 Recommendations for user {user_id}:")
    if not recommendations:
        print("⚠️ No recommendations returned.")
    else:
        for rec in recommendations:
            if rec.title:
                print(f"  movie_id={rec.movie_id}, score={rec.score:.4f}, title={rec.title}")
            else:
                print(f"  movie_id={rec.movie_id}, score={rec.score:.4f}")


if __name__ == "__main__":
    main()
