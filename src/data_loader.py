import os
import pandas as pd

# MovieLens 32M subset (ratings to 2023, see make_ml32m_subset.py); set CINEMATCH_DATA=data/ml-latest-small for the old 2018 set
DATA_DIR = os.environ.get("CINEMATCH_DATA", "data/ml-32m-subset")

def load_ratings(path=None):
    ratings = pd.read_csv(path or os.path.join(DATA_DIR, "ratings.csv"))
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    return ratings[["user_id", "movie_id", "rating"]]

def load_movies(path=None):
    movies = pd.read_csv(path or os.path.join(DATA_DIR, "movies.csv"))
    movies = movies.rename(columns={"movieId": "movie_id"})
    return movies

def get_movie_genre_string(movies_df):
    movies_df = movies_df.copy()
    movies_df["genre_str"] = movies_df["genres"].str.replace("|", " ", regex=False)
    return movies_df[["movie_id", "title", "genre_str"]]

def load_links(path=None):
    links = pd.read_csv(path or os.path.join(DATA_DIR, "links.csv"))
    links = links.rename(columns={"movieId": "movie_id"})
    # Ensure IMDb ID has at least 7 digits (filled with leading zeros)
    links["imdb_id"] = links["imdbId"].astype(str).str.zfill(7)
    return links[["movie_id", "imdb_id"]]
