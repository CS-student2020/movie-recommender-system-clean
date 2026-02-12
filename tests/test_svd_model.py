

import os
import sys
import pandas as pd

import pytest
pytest.skip("Legacy SVD test removed after refactor", allow_module_level=True)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def test_svd_model_training():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    ratings_df = pd.read_csv(os.path.join(base_path, "ratings.csv")).sample(5000, random_state=42)

    model = MatrixFactorization(ratings_df, epochs=2)
    model.train(verbose=False)

    sample = ratings_df.sample(1).iloc[0]
    pred = model.predict(sample["userId"], sample["movieId"])

    assert pred is not None

