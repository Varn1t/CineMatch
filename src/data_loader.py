import pandas as pd

def load_ratings(path="data/ml-latest-small/ratings.csv"):
    ratings = pd.read_csv(path)
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    return ratings[["user_id", "movie_id", "rating"]]

def load_movies(path="data/ml-latest-small/movies.csv"):
    movies = pd.read_csv(path)
    movies = movies.rename(columns={"movieId": "movie_id"})
    return movies

def get_movie_genre_string(movies_df):
    movies_df = movies_df.copy()
    movies_df["genre_str"] = movies_df["genres"].str.replace("|", " ", regex=False)
    return movies_df[["movie_id", "title", "genre_str"]]

def load_links(path="data/ml-latest-small/links.csv"):
    links = pd.read_csv(path)
    links = links.rename(columns={"movieId": "movie_id"})
    # Ensure IMDb ID has at least 7 digits (filled with leading zeros)
    links["imdb_id"] = links["imdbId"].astype(str).str.zfill(7)
    return links[["movie_id", "imdb_id"]]
