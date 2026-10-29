from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

def build_collab_model(ratings_df):
    """
    Trains SVD on the full ratings dataframe.
    Returns the trained model and the full trainset.
    """
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(
        ratings_df[["user_id", "movie_id", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()
    model = SVD(n_factors=100, n_epochs=30, lr_all=0.007, reg_all=0.05, random_state=42)
    model.fit(trainset)

    return model, trainset

def get_collab_recommendations(user_id, model, trainset, all_movie_ids, movies_df, top_n=10):
    """
    For a given user, predict ratings for all unrated movies.
    Returns top_n with highest predicted rating.
    """
    rated_movie_ids = set([
        trainset.to_raw_iid(item_inner_id)
        for (item_inner_id, _) in trainset.ur[trainset.to_inner_uid(user_id)]
    ])

    # Predict on all unrated movies
    unrated = [mid for mid in all_movie_ids if mid not in rated_movie_ids]
    predictions = [model.predict(user_id, mid) for mid in unrated]

    # Sort by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:top_n]

    # Build result dataframe
    top_movie_ids = [p.iid for p in top_predictions]
    top_scores = [p.est for p in top_predictions]

    result = movies_df[movies_df["movie_id"].isin(top_movie_ids)][["movie_id", "title"]].copy()
    score_map = dict(zip(top_movie_ids, top_scores))
    result["collab_score"] = result["movie_id"].map(score_map)

    return result.sort_values("collab_score", ascending=False).reset_index(drop=True)


def get_similar_users(user_id, ratings_df, movies_df, top_n=5, loved_threshold=4.5):
    """
    Find the top_n most similar users to user_id using cosine similarity
    on the user-movie ratings matrix.

    Returns a list of dicts: [{user_id, similarity, loved_movies: [title, ...]}]
    """
    # Sparse user-movie matrix (missing = 0); a dense pivot is ~700 MB at 10k users
    users  = ratings_df["user_id"].astype("category")
    movies = ratings_df["movie_id"].astype("category")
    all_ids = users.cat.categories.tolist()

    if user_id not in all_ids:
        return []

    all_vecs = csr_matrix((ratings_df["rating"].values, (users.cat.codes, movies.cat.codes)))
    target_vec = all_vecs[all_ids.index(user_id)]

    sims = cosine_similarity(target_vec, all_vecs)[0]  # shape (n_users,)

    # Sort descending, skip the user themselves
    sim_series = pd.Series(sims, index=all_ids)
    sim_series = sim_series.drop(index=user_id, errors="ignore")
    top_users  = sim_series.nlargest(top_n)

    movie_title_map = dict(zip(movies_df["movie_id"], movies_df["title"]))

    results = []
    for uid, sim in top_users.items():
        loved = ratings_df[
            (ratings_df["user_id"] == uid) &
            (ratings_df["rating"] >= loved_threshold)
        ]["movie_id"].tolist()
        loved_titles = [movie_title_map.get(mid, f"#{mid}") for mid in loved[:5]]
        results.append({
            "user_id":    uid,
            "similarity": float(sim),
            "loved":      loved_titles,
        })

    return results