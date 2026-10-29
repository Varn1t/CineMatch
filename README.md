# 🎬 CineMatch — Hybrid Movie Recommender System

A full-featured movie recommendation system built with **Python + Streamlit** that combines collaborative filtering (SVD matrix factorization) and content-based filtering (TF-IDF genre similarity) into a polished, interactive web app.

---

## ✨ Features

| Tab | What it does |
|-----|-------------|
| **🎬 Recommend** | Pick 1–5 seed movies → get hybrid recommendations with score breakdowns, Surprise Me mode, genre filter, CSV export, watchlist, and recommendation history |
| **🔍 Explore Movies** | Search any movie to see its average rating, total ratings, IMDb link, and a full rating distribution histogram |
| **👥 User Insights** | Find your 5 nearest-neighbour users (who rates similarly to you), plus a side-by-side A/B comparison mode for two users |
| **📊 Evaluation** | Three sub-tabs: RMSE/Precision@K/Recall@K metrics, a PCA scatter of SVD latent factors coloured by genre, and an α sensitivity heatmap |

### Hybrid scoring
```
hybrid_score = α × collab_norm + (1 − α) × content_norm
```
- **α → 1**: trusts collaborative signals (what similar users liked)
- **α → 0**: trusts content signals (genre similarity to your seeds)
- Adjustable live in the sidebar

---

## 🗂️ Project structure

```
.
├── app.py                   # Streamlit UI — all tabs and visualisations
├── movie_recommender.py     # CLI entry point (optional standalone use)
├── src/
│   ├── data_loader.py       # Load ratings, movies, links CSVs
│   ├── collaborative.py     # SVD model (Surprise) + similar-user finder
│   ├── content_based.py     # TF-IDF genre similarity matrix
│   ├── hybrid.py            # Weighted hybrid fusion
│   └── evaluate.py          # 5-fold cross-validation: RMSE, Precision@K, Recall@K
└── data/
    └── ml-32m-subset/       # built by make_ml32m_subset.py (not committed)
        ├── movies.csv
        ├── ratings.csv
        ├── links.csv
        └── tags.csv
```

---

## 🚀 Quick start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/cinematch.git
cd cinematch
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download the dataset

The app defaults to a recent subset of [MovieLens 32M](https://grouplens.org/datasets/movielens/) (ratings up to Oct 2023):

```bash
curl -O https://files.grouplens.org/datasets/movielens/ml-32m.zip   # 239 MB, unzip it
python make_ml32m_subset.py ml-32m data/ml-32m-subset             # 10,000 users active since 2018, 9,000 movies
```

To use the older 2018 [MovieLens Latest Small](https://grouplens.org/datasets/movielens/latest/) instead, unzip it to `data/ml-latest-small/` and set `CINEMATCH_DATA=data/ml-latest-small`.

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## 🧠 How it works

### Collaborative filtering (`src/collaborative.py`)
- Uses the **Surprise** library's SVD implementation (Simon Funk matrix factorization)
- Trained on all user–movie ratings in `ratings.csv`
- Predicts how strongly the selected user would enjoy each unrated movie

### Content-based filtering (`src/content_based.py`)
- Builds a **TF-IDF** matrix over genre strings
- Computes **cosine similarity** between every pair of movies
- Given seed movies, returns movies most similar in genre profile

### Hybrid fusion (`src/hybrid.py`)
- Both score vectors are min-max normalised to [0, 1]
- Combined as a weighted sum controlled by **α** (sidebar slider)

### Evaluation (`src/evaluate.py`)
- **RMSE** via Surprise's 5-fold `cross_validate`
- **Precision@K / Recall@K** via a manual leave-one-out split at a 4★ relevance threshold

---

## 📦 Dependencies

See `requirements.txt`. Key libraries:

| Library | Role |
|---------|------|
| `streamlit` | Web UI |
| `scikit-surprise` | SVD collaborative filtering |
| `scikit-learn` | TF-IDF, cosine similarity, PCA |
| `pandas` / `numpy` | Data wrangling |
| `plotly` | Interactive charts |

---

## 🔮 Roadmap / known limits

> **Verified metrics — ml-32m subset (2,378,597 ratings, 10,000 users; SVD 100 factors; RMSE 5-fold CV, P/R@10 on an 80/20 split, 4★ threshold)**
> | Metric | Value |
> |--------|-------|
> | RMSE | **0.7452** |
> | Precision@10 | **77.3%** |
> | Recall@10 | **59.3%** |

- Dataset is a **MovieLens 32M subset** (9,000 movies, 2.4M ratings). The full set needs a sparse content model (the dense similarity matrix won't fit); the α sweep is already optimised to scale (1× collab pass, not 11×).
- Watchlist is **session-only** — no persistence between browser refreshes.
- IMDb links depend on `links.csv` being up to date.

---

## 📄 License

MIT — do whatever you like, attribution appreciated.
