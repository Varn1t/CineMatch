"""Regression check for the hybrid recommender. Run: python test_hybrid.py (needs data/ml-latest-small)."""
from src.data_loader import load_ratings, load_movies, get_movie_genre_string
from src.content_based import build_content_model
from src.collaborative import build_collab_model
from src.hybrid import get_hybrid_recommendations

ratings, movies = load_ratings(), load_movies()
sim, id_to_index, movies_df = build_content_model(get_movie_genre_string(movies))
model, trainset = build_collab_model(ratings)
title_to_id = dict(zip(movies["title"], movies["movie_id"]))

seeds = [title_to_id["Inception (2010)"], title_to_id["Interstellar (2014)"]]
recs = get_hybrid_recommendations(1, seeds, model, trainset, None, sim, id_to_index, movies_df, top_n=10)

# Seeds and already-rated movies are never recommended
rated = set(ratings.loc[ratings["user_id"] == 1, "movie_id"])
assert not set(recs["movie_id"]) & (set(seeds) | rated), recs

# Recent seeds must not be drowned out by user 1's 1990s-heavy history / SVD's classics bias
years = recs["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
assert years.median() >= 2000, recs["title"].tolist()

print("ok:", recs["title"].tolist())
