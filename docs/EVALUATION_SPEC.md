# Evaluation Spec (Movie Recommender)

## Goal
Measure recommendation quality with reproducible, leakage-safe offline evaluation.

## Data Split (Leakage-safe)
- Split strategy: per-user holdout
- For each user with >= 2 interactions:
  - test item = latest interaction (by timestamp if available; otherwise last row order)
  - train = remaining interactions
- Users with < 2 interactions are excluded from evaluation set.

## Candidate Set
- Recommend from movies that exist in training universe.
- Exclude items the user has already seen in training when generating top-K.

## Metrics
- Precision@K
- Recall@K
- (Optional later) MAP@K / NDCG@K

## Baselines
- Random baseline (sanity)
- Popularity baseline (top global popular items from train)

## Outputs
- metrics.json (machine-readable)
- metrics.md (human-readable summary)

## Reproducibility
- Fixed random seed
- Deterministic evaluation subset for tests
