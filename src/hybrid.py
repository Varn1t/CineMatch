import pandas as pd
import numpy as np


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
    # Score EVERY movie on both signals. Truncating each signal to its own top-100
    # before merging gave everything outside the SVD's top-100 a collab score of 0,
    # so the user's all-time-classics canon always won regardless of the seeds.
    # movies_df rows are aligned with similarity_matrix rows (see build_content_model).
    result = movies_df[["movie_id", "title"]].copy()

    # --- Collaborative: SVD prediction for this user, per item ---
    # Rank by the personalised part (item bias + user·item factors). Equivalent
    # ordering to model.predict, minus the rating-scale clipping that flattens the
    # top end into ties. Unknown items/users fall back to 0 (= global mean).
    try:
        inner_uid = trainset.to_inner_uid(user_id)
    except ValueError:
        inner_uid = None
    collab = []
    for mid in result["movie_id"]:
        try:
            iid = trainset.to_inner_iid(mid)
        except ValueError:          # movie has no ratings
            collab.append(0.0)
            continue
        score = model.bi[iid]
        if inner_uid is not None:
            score += np.dot(model.qi[iid], model.pu[inner_uid])
        collab.append(score)
    result["collab_score"] = collab

    # --- Content: mean genre similarity to the seeds ---
    seed_rows = [id_to_index[m] for m in liked_movie_ids if m in id_to_index]
    result["content_score"] = (
        np.asarray(similarity_matrix[seed_rows]).mean(axis=0) if seed_rows else 0.0
    )

    # Never recommend the seeds or anything the user has already rated
    exclude = set(liked_movie_ids)
    if inner_uid is not None:
        exclude |= {trainset.to_raw_iid(i) for i, _ in trainset.ur[inner_uid]}
    result = result[~result["movie_id"].isin(exclude)]

    # --- Normalize both to [0, 1] ---
    result["collab_norm"] = normalize(result["collab_score"])
    result["content_norm"] = normalize(result["content_score"])

    # --- Weighted hybrid score ---
    result["hybrid_score"] = (
        alpha * result["collab_norm"] +
        (1 - alpha) * result["content_norm"]
    )

    result = result[["movie_id", "title", "hybrid_score", "collab_norm", "content_norm"]]
    result = result.sort_values("hybrid_score", ascending=False, kind="stable")
    if top_n is not None:
        result = result.head(top_n)
    return result.reset_index(drop=True)
