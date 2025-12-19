from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankParams:
    top_k: int = 10
    # Stable tie-breakers
    # score desc, support desc, movieId asc
    diversity: bool = False
    mmr_lambda: float = 0.8  # closer to 1 => relevance, closer to 0 => diversity
    mmr_candidates: int = 200  # only rerank top-N by score for speed


def _stable_rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = df.columns
    if "support" not in cols:
        df = df.assign(support=0)
    return df.sort_values(["score", "support", "movieId"], ascending=[False, False, True]).reset_index(drop=True)


def _mmr_rerank(
    df: pd.DataFrame,
    item_item_sim: pd.DataFrame,
    params: RankParams,
) -> pd.DataFrame:
    """
    MMR: iteratively select items balancing relevance and diversity:
      argmax λ * score(i) - (1-λ) * max_{j in selected} sim(i,j)
    Requires item-item similarity matrix with movieId indices/columns.
    """
    if df.empty or not params.diversity:
        return df

    candidates = df.head(params.mmr_candidates).copy()
    rest = df.iloc[params.mmr_candidates:].copy()

    # Only keep items present in sim matrix for diversity computation
    present_mask = candidates["movieId"].isin(item_item_sim.index)
    candidates_present = candidates[present_mask].reset_index(drop=True)
    candidates_missing = candidates[~present_mask].reset_index(drop=True)

    if len(candidates_present) <= 1:
        # nothing to diversify
        out = pd.concat([candidates_present, candidates_missing, rest], ignore_index=True)
        return _stable_rank(out)

    scores = candidates_present.set_index("movieId")["score"].to_dict()

    selected = []
    selected_set = set()

    # pick best by score first
    first = candidates_present.iloc[0]["movieId"]
    selected.append(first)
    selected_set.add(first)

    remaining = [mid for mid in candidates_present["movieId"].tolist() if mid != first]

    lam = float(params.mmr_lambda)
    lam = max(0.0, min(1.0, lam))

    while remaining and len(selected) < params.top_k:
        best_mid: Optional[int] = None
        best_val = -np.inf

        for mid in remaining:
            rel = scores.get(mid, 0.0)
            # diversity penalty: similarity to already-selected set
            sims = []
            for sid in selected:
                if mid in item_item_sim.index and sid in item_item_sim.columns:
                    sims.append(float(item_item_sim.loc[mid, sid]))
            penalty = max(sims) if sims else 0.0
            val = lam * rel - (1.0 - lam) * penalty
            if val > best_val:
                best_val = val
                best_mid = mid

        if best_mid is None:
            break

        selected.append(best_mid)
        selected_set.add(best_mid)
        remaining.remove(best_mid)

    # Build reranked list: selected first, then remaining by stable ranking
    selected_df = candidates_present[candidates_present["movieId"].isin(selected)].copy()
    selected_df["__order"] = selected_df["movieId"].apply(lambda x: selected.index(x))
    selected_df = selected_df.sort_values("__order").drop(columns="__order")

    leftover_df = candidates_present[~candidates_present["movieId"].isin(selected_set)].copy()
    leftover_df = _stable_rank(leftover_df)

    out = pd.concat([selected_df, leftover_df, candidates_missing, rest], ignore_index=True)
    out = _stable_rank(out)
    return out


def rank_recommendations(
    scored_df: pd.DataFrame,
    *,
    item_item_sim: Optional[pd.DataFrame],
    params: RankParams,
) -> pd.DataFrame:
    """
    Takes a scored candidates df with at least [movieId, score] and returns ranked top_k.
    """
    ranked = _stable_rank(scored_df)

    if params.diversity and item_item_sim is not None:
        ranked = _mmr_rerank(ranked, item_item_sim, params)

    return ranked.head(params.top_k).reset_index(drop=True)

