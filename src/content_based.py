from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_content_model(movie_genres_df):
    """
    Takes the output of get_movie_genre_string().
    Returns the similarity matrix and the movie index mapping.
    """
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movie_genres_df["genre_str"])

    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Map movie_id -> index position in the matrix
    movie_genres_df = movie_genres_df.reset_index(drop=True)
    id_to_index = pd.Series(
        movie_genres_df.index,
        index=movie_genres_df["movie_id"]
    )

    return similarity_matrix, id_to_index, movie_genres_df


def get_content_recommendations(movie_id, similarity_matrix, id_to_index, movies_df, top_n=10):
    """
    Given a movie_id, return top_n similar movies based on genre similarity.
    """
    if movie_id not in id_to_index:
        return pd.DataFrame()

    idx = id_to_index[movie_id]
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort by similarity score descending, skip index 0 (itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    result = movies_df.iloc[movie_indices][["movie_id", "title", "genre_str"]].copy()
    result["content_score"] = scores

    return result.reset_index(drop=True)