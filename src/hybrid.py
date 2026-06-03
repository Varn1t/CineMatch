import pandas as pd
import numpy as np
from src.content_based import get_content_recommendations
from src.collaborative import get_collab_recommendations


def normalize(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series * 0
    return (series - min_val) / (max_val - min_val)


def get_hybrid_recommendations(
    user_id,
    liked_movie_ids,           # list of movie_ids the user has rated highly
    model,
    trainset,
    all_movie_ids,
    similarity_matrix,
    id_to_index,
    movies_df,
    top_n=10,
    alpha=0.7                  # weight for collaborative score
):
    # --- Collaborative scores for this user ---
    collab_df = get_collab_recommendations(
        user_id, model, trainset, all_movie_ids, movies_df, top_n=100
    )

    # --- Content scores: aggregate across all liked movies ---
    content_frames = []
    for movie_id in liked_movie_ids:
        recs = get_content_recommendations(
            movie_id, similarity_matrix, id_to_index, movies_df, top_n=100
        )
        content_frames.append(recs[["movie_id", "content_score"]])

    if content_frames:
        content_df = pd.concat(content_frames)
        # Average content score across all seed movies
        content_df = content_df.groupby("movie_id", as_index=False)["content_score"].mean()
    else:
        content_df = pd.DataFrame(columns=["movie_id", "content_score"])

    # --- Merge both ---
    collab_df_clean = collab_df[["movie_id", "collab_score"]]
    merged = pd.merge(collab_df_clean, content_df, on="movie_id", how="outer")
    merged = pd.merge(merged, movies_df[["movie_id", "title"]], on="movie_id", how="left")

    # Fill missing scores with 0 before normalizing
    merged["collab_score"] = merged["collab_score"].fillna(0)
    merged["content_score"] = merged["content_score"].fillna(0)

    # --- Normalize both to [0, 1] ---
    merged["collab_norm"] = normalize(merged["collab_score"])
    merged["content_norm"] = normalize(merged["content_score"])

    # --- Weighted hybrid score ---
    merged["hybrid_score"] = (
        alpha * merged["collab_norm"] +
        (1 - alpha) * merged["content_norm"]
    )

    result = merged[["movie_id", "title", "hybrid_score", "collab_norm", "content_norm"]]
    return result.sort_values("hybrid_score", ascending=False).head(top_n).reset_index(drop=True)