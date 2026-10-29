"""Build a laptop-sized, recent subset of MovieLens 32M (ratings up to Oct 2023).

The full set (32M ratings, 87k movies) is too big for the app: the content model is a dense
movie x movie matrix. This keeps users active since SINCE, then the MAX_MOVIES most-rated movies
among them, and writes the same CSV layout the app reads.

    curl -O https://files.grouplens.org/datasets/movielens/ml-32m.zip   # then unzip
    python make_ml32m_subset.py path/to/ml-32m data/ml-32m-subset
"""
import sys
from pathlib import Path
import pandas as pd

SINCE = "2018-01-01"   # users must have rated something after this
N_USERS = 10000
MAX_MOVIES = 9000      # similarity matrix is MAX_MOVIES^2 float64 (~650 MB at 9000)
MIN_RATINGS = 5

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

ratings = pd.read_csv(src / "ratings.csv",
                      dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"})
recent = ratings.loc[ratings["timestamp"] >= pd.Timestamp(SINCE).timestamp(), "userId"].unique()
users = pd.Series(recent).sample(min(N_USERS, len(recent)), random_state=42)
ratings = ratings[ratings["userId"].isin(users)]

counts = ratings["movieId"].value_counts()
keep = counts[counts >= MIN_RATINGS].head(MAX_MOVIES).index
ratings = ratings[ratings["movieId"].isin(keep)]
# Renumber users 1..N — the app's User ID picker assumes contiguous IDs
new_uid = {u: i for i, u in enumerate(sorted(ratings["userId"].unique()), 1)}
ratings["userId"] = ratings["userId"].map(new_uid)
ratings.to_csv(dst / "ratings.csv", index=False)

for name in ["movies.csv", "links.csv", "tags.csv"]:
    df = pd.read_csv(src / name)
    df = df[df["movieId"].isin(keep)]
    if "userId" in df:
        df = df[df["userId"].isin(new_uid)].assign(userId=lambda d: d["userId"].map(new_uid))
    df.to_csv(dst / name, index=False)

print(f"{len(ratings):,} ratings · {ratings['userId'].nunique():,} users · {len(keep):,} movies -> {dst}")
