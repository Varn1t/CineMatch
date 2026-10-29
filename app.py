import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from src.data_loader import load_ratings, load_movies, get_movie_genre_string, load_links
from src.content_based import build_content_model
from src.collaborative import build_collab_model, get_similar_users
from src.hybrid import get_hybrid_recommendations
from src.evaluate import evaluate_rmse, evaluate_precision_recall_at_k

st.set_page_config(page_title="CineMatch – Movie Recommender", layout="wide", page_icon="🎬")

def clean_html(html_str):
    return "".join(line.strip() for line in html_str.split("\n"))

# ═══════════════════════════════════════════════════════════════════════════════
# STYLES
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { font-family:'Inter',sans-serif; }

/* ── Hero ── */
.hero { padding:28px 0 20px 0; border-bottom:1px solid rgba(255,255,255,.08); margin-bottom:20px; }
.hero h1 { font-size:2.2rem; font-weight:700; margin:0 0 4px 0; color:#fff; }
.hero p  { font-size:.95rem; color:rgba(255,255,255,.5); margin:0; }

/* ── Section label ── */
.section-label { font-size:.72rem; font-weight:600; letter-spacing:.1em;
    text-transform:uppercase; color:rgba(255,255,255,.4); margin-bottom:8px; }

/* ── Seed cards ── */
.seed-grid { display:flex; flex-wrap:wrap; gap:10px; margin:8px 0 22px 0; }
.seed-card { background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12);
    border-radius:12px; padding:12px 16px; min-width:180px; max-width:300px; }
.seed-card-title { font-size:.95rem; font-weight:600; color:#fff; margin-bottom:8px; line-height:1.3; }
.seed-badges { display:flex; flex-wrap:wrap; gap:5px; }

/* ── Badges ── */
.badge { font-size:.68rem; font-weight:500; padding:2px 8px; border-radius:20px;
    background:rgba(255,255,255,.12); color:rgba(255,255,255,.75);
    border:1px solid rgba(255,255,255,.07); display:inline-block; }
.badge-surprise { background:rgba(255,140,0,.25); color:#ffa94d;
    border-color:rgba(255,140,0,.3); font-weight:700; }

/* ── Rec cards ── */
.rec-card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09);
    border-radius:14px; padding:20px 22px 14px 22px; margin-bottom:4px;
    transition:transform .15s ease, box-shadow .15s ease, background .15s ease; }
.rec-card:hover { transform:translateY(-2px); background:rgba(255,255,255,.07);
    border-color:rgba(255,255,255,.16); box-shadow:0 8px 24px rgba(0,0,0,.25); }
.rec-header { display:flex; justify-content:space-between; align-items:flex-start;
    margin-bottom:10px; gap:12px; }
.rec-rank { font-size:.8rem; font-weight:700; color:rgba(255,255,255,.3); min-width:28px; }
.rec-title { font-size:1.1rem; font-weight:700; color:#fff; flex:1; line-height:1.3; }
.imdb-btn { background:#f5c518; color:#000!important; font-weight:700; font-size:.75rem;
    padding:4px 10px; border-radius:8px; text-decoration:none!important;
    white-space:nowrap; flex-shrink:0; display:inline-block; transition:background .15s; }
.imdb-btn:hover { background:#d4a910; }
.rec-genres { display:flex; flex-wrap:wrap; gap:5px; margin:8px 0 14px 0; }

/* ── Score bars ── */
.score-block { margin-bottom:9px; }
.score-meta { display:flex; justify-content:space-between; align-items:center;
    font-size:.8rem; color:rgba(255,255,255,.55); margin-bottom:3px; }
.score-meta strong { color:rgba(255,255,255,.9); font-weight:600; }
.score-meta .hint { font-size:.7rem; color:rgba(255,255,255,.3); font-style:italic; }
.bar-bg { height:5px; border-radius:4px; background:rgba(255,255,255,.09); overflow:hidden; }
.bar-fill { height:100%; border-radius:4px; }
.bar-hybrid  { background:linear-gradient(90deg,#ff6b6b,#feca57); }
.bar-collab  { background:#4ecdc4; }
.bar-content { background:#a29bfe; }

/* ── Legend ── */
.legend-box { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08);
    border-radius:10px; padding:12px 16px; margin-bottom:20px; font-size:.8rem;
    color:rgba(255,255,255,.5); line-height:1.8; }
.legend-box strong { color:rgba(255,255,255,.85); }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%;
    margin-right:4px; vertical-align:middle; }
.dot-hybrid  { background:linear-gradient(90deg,#ff6b6b,#feca57); }
.dot-collab  { background:#4ecdc4; }
.dot-content { background:#a29bfe; }

/* ── User similarity cards ── */
.user-card { background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.1);
    border-radius:12px; padding:16px 18px; margin-bottom:12px; }
.user-card-header { display:flex; justify-content:space-between; align-items:center;
    margin-bottom:10px; }
.user-card-id { font-size:1rem; font-weight:700; color:#fff; }
.user-card-sim { font-size:.85rem; font-weight:600; color:#4ecdc4; }
.user-card-movies { font-size:.82rem; color:rgba(255,255,255,.55); }
.user-card-movies li { margin:3px 0; }

/* ── Eval metric ── */
.eval-card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09);
    border-radius:14px; padding:22px 24px; margin-bottom:16px; }
.eval-title { font-size:1rem; font-weight:700; color:#fff; margin-bottom:4px; }
.eval-subtitle { font-size:.8rem; color:rgba(255,255,255,.4); margin-bottom:14px; }
.eval-value { font-size:1.8rem; font-weight:700; margin-bottom:12px; }

/* ── Movie info card (Explore tab) ── */
.movie-info-card { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09);
    border-radius:14px; padding:22px 24px; margin-bottom:16px; }
.movie-info-title { font-size:1.4rem; font-weight:700; color:#fff; margin-bottom:8px; }
.movie-info-stats { display:flex; gap:24px; flex-wrap:wrap; margin-top:12px; }
.movie-stat { text-align:center; }
.movie-stat .val { font-size:1.4rem; font-weight:700; color:#fff; }
.movie-stat .lbl { font-size:.72rem; color:rgba(255,255,255,.4); text-transform:uppercase;
    letter-spacing:.07em; margin-top:2px; }

/* ── History ── */
.history-run { background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07);
    border-radius:10px; padding:12px 16px; margin-bottom:10px; }
.history-label { font-size:.78rem; color:rgba(255,255,255,.4);
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }
.history-seeds { font-size:.9rem; color:rgba(255,255,255,.8); margin-bottom:4px; }
.history-movies { font-size:.8rem; color:rgba(255,255,255,.5); }

/* ── Light mode overrides ── */
@media (prefers-color-scheme:light) {
    .hero h1 { color:#111; }
    .hero p,.section-label,.legend-box,.score-meta,.user-card-movies,
    .history-label,.history-movies,.eval-subtitle,.movie-stat .lbl { color:rgba(0,0,0,.5); }
    .seed-card,.rec-card,.user-card,.eval-card,.movie-info-card,.history-run {
        background:rgba(0,0,0,.03); border-color:rgba(0,0,0,.08); }
    .seed-card:hover,.rec-card:hover { background:rgba(0,0,0,.06); border-color:rgba(0,0,0,.14); }
    .seed-card-title,.rec-title,.rec-rank,.user-card-id,.eval-title,.movie-info-title,
    .movie-stat .val,.history-seeds,.legend-box strong,.score-meta strong { color:#111; }
    .badge { background:rgba(0,0,0,.07); color:#444; border-color:transparent; }
    .bar-bg { background:rgba(0,0,0,.07); }
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_all():
    ratings     = load_ratings()
    movies      = load_movies()
    movie_genres = get_movie_genre_string(movies)
    sim_matrix, id_to_index, movies_df = build_content_model(movie_genres)
    model, trainset = build_collab_model(ratings)
    all_movie_ids   = movies["movie_id"].tolist()
    links           = load_links()
    return ratings, movies, movies_df, sim_matrix, id_to_index, model, trainset, all_movie_ids, links

@st.cache_data
def run_rmse_eval(_ratings):
    return evaluate_rmse(_ratings)

@st.cache_data
def run_pr_eval(_ratings, k=10, threshold=4.0):
    return evaluate_precision_recall_at_k(_ratings, k=k, threshold=threshold)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for _key, _default in [
    ("watchlist", {}), ("history", []), ("why_cache", {}), ("seed_titles", [])
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(clean_html("""
<div class="hero">
  <h1>🎬 CineMatch</h1>
  <p>A hybrid movie recommender blending what you like with what people like you enjoy.</p>
</div>
"""), unsafe_allow_html=True)

with st.spinner("Training models on first load — usually under 30 seconds…"):
    (ratings, movies, movies_df, sim_matrix,
     id_to_index, model, trainset, all_movie_ids, links) = load_all()

st.success("✅ Models ready! Jump into any tab below.")

# ── Shared helpers ────────────────────────────────────────────────────────────
movie_title_to_id = dict(zip(movies_df["title"], movies_df["movie_id"]))
movie_id_to_title = dict(zip(movies_df["movie_id"], movies_df["title"]))
max_user_id = int(ratings["user_id"].max())

def get_seed_genres(liked_ids, movies_df_):
    g = set()
    for mid in liked_ids:
        row = movies_df_[movies_df_["movie_id"] == mid]
        if not row.empty and pd.notna(row.iloc[0].get("genres", None)):
            g.update(row.iloc[0]["genres"].split("|"))
    return g

def is_outside_comfort(genres_str, seed_genres):
    if pd.isna(genres_str) or not seed_genres:
        return False
    return len(set(genres_str.split("|")) & seed_genres) == 0

def get_why_this(rec_mid, liked_ids, sim_matrix_, id_to_index_, movies_df_):
    cache_key = (rec_mid, tuple(sorted(liked_ids)))
    if cache_key in st.session_state["why_cache"]:
        return st.session_state["why_cache"][cache_key]
    best_seed, best_sim = None, -1.0
    for sid in liked_ids:
        if rec_mid in id_to_index_ and sid in id_to_index_:
            s = float(sim_matrix_[id_to_index_[rec_mid], id_to_index_[sid]])
            if s > best_sim:
                best_sim, best_seed = s, sid
    if best_seed is None:
        result = ("your seed movies", 0.0)
    else:
        t_row = movies_df_[movies_df_["movie_id"] == best_seed]
        result = (t_row["title"].iloc[0] if not t_row.empty else f"#{best_seed}", best_sim)
    st.session_state["why_cache"][cache_key] = result
    return result

def render_rec_card(row, idx, liked_movie_ids, show_surprise_badge=False, show_actions=True):
    """Render one recommendation card (HTML) + Streamlit action widgets."""
    title        = row["title"]
    hybrid_score = row.get("hybrid_score", 0.0)
    collab_norm  = row.get("collab_norm", 0.0)
    content_norm = row.get("content_norm", 0.0)
    imdb_id      = row.get("imdb_id", None)
    genres_str   = row.get("genres", "")

    genres_list = genres_str.split("|") if pd.notna(genres_str) and genres_str else []
    badges_html = "".join(f'<span class="badge">{g}</span>' for g in genres_list)
    if show_surprise_badge:
        badges_html = '<span class="badge badge-surprise">🌶️ Outside comfort zone</span> ' + badges_html

    imdb_btn = (f'<a href="https://www.imdb.com/title/tt{int(imdb_id):07d}/" '
                f'target="_blank" class="imdb-btn">IMDb ↗</a>'
                if pd.notna(imdb_id) else "")

    h_pct = int(hybrid_score  * 100)
    c_pct = int(collab_norm   * 100)
    n_pct = int(content_norm  * 100)

    st.markdown(clean_html(f"""
    <div class="rec-card">
      <div class="rec-header">
        <span class="rec-rank">#{idx+1}</span>
        <span class="rec-title">{title}</span>
        {imdb_btn}
      </div>
      <div class="rec-genres">{badges_html}</div>
      <div class="score-block">
        <div class="score-meta"><span>⭐ Overall Match</span><strong>{hybrid_score:.1%}</strong></div>
        <div class="bar-bg"><div class="bar-fill bar-hybrid" style="width:{h_pct}%"></div></div>
      </div>
      <div class="score-block">
        <div class="score-meta">
          <span>👥 People Like You <span class="hint">(collaborative)</span></span>
          <strong>{collab_norm:.1%}</strong>
        </div>
        <div class="bar-bg"><div class="bar-fill bar-collab" style="width:{c_pct}%"></div></div>
      </div>
      <div class="score-block">
        <div class="score-meta">
          <span>🎭 Genre Match <span class="hint">(content-based)</span></span>
          <strong>{content_norm:.1%}</strong>
        </div>
        <div class="bar-bg"><div class="bar-fill bar-content" style="width:{n_pct}%"></div></div>
      </div>
    </div>
    """), unsafe_allow_html=True)

    if show_actions:
        movie_id = int(row["movie_id"])
        act1, act2, _ = st.columns([1, 2, 3])
        with act1:
            already_saved = movie_id in st.session_state["watchlist"]
            btn_label = "✅ Saved" if already_saved else "🔖 Save"
            if st.button(btn_label, key=f"wl_{movie_id}_{idx}", disabled=already_saved):
                st.session_state["watchlist"][movie_id] = title
                st.rerun()
        with act2:
            with st.expander("💡 Why recommended?"):
                seed_title, content_sim = get_why_this(
                    movie_id, liked_movie_ids, sim_matrix, id_to_index, movies_df
                )
                st.write(f"**Genre similarity** — most similar to **{seed_title}** "
                         f"from your seeds ({content_sim:.1%} genre overlap).")
                st.write("**People** — users who share your rating history also rated "
                         "this highly (collaborative filtering).")
        st.write("")  # small gap between cards


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Your Profile")
    user_id = st.number_input(
        "User ID", min_value=1, max_value=max_user_id, value=1, step=1,
        help=f"Any ID between 1 and {max_user_id}."
    )

    st.divider()
    st.subheader("Recommendation Mode")
    surprise_mode = st.toggle(
        "🎲 Surprise Me!", value=False,
        help="Forces α=0.2 and shows only movies outside your usual genres."
    )
    if surprise_mode:
        alpha = 0.2
        st.info("🌶️ Showing movies outside your comfort zone!")
    else:
        alpha = st.slider("People vs Genres balance", 0.0, 1.0, 0.7, 0.05,
                          help="Right = trust people more. Left = trust genre similarity more.")
        c1, c2 = st.columns(2)
        c1.metric("👥 People", f"{int(alpha*100)}%")
        c2.metric("🎭 Genres", f"{int((1-alpha)*100)}%")

    st.divider()
    top_n = st.slider("Number of recommendations", 5, 20, 10)

    st.divider()
    st.subheader("Filter by Genre")
    all_genres_set = set()
    for _g in movies["genres"].dropna():
        all_genres_set.update(_g.split("|"))
    all_genres_set.discard("(no genres listed)")
    selected_genres = st.multiselect("Only show these genres:",
                                     options=sorted(all_genres_set))

    st.divider()
    wl_count = len(st.session_state["watchlist"])
    st.metric("🔖 Watchlist", f"{wl_count} saved")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["🎬 Recommend", "🔍 Explore Movies", "👥 User Insights", "📊 Evaluation"]
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — RECOMMEND
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.subheader("🎯 Pick movies you liked")
    st.caption("Type to search, pick a result, then click ➕ Add. Up to 5 seeds.")

    watchlisted_ids = set(st.session_state["watchlist"].keys())

    # Seed defaults on first load
    if not st.session_state["seed_titles"]:
        st.session_state["seed_titles"] = [
            t for t in ["Toy Story (1995)", "Fargo (1996)"]
            if t in movie_title_to_id
        ]

    # ── Search-filtered add widget ─────────────────────────────────────────────
    seed_search = st.text_input(
        "Search movies:", placeholder="e.g. Matrix, Inception, Toy Story…",
        key="seed_search_box"
    )
    all_titles_sorted = sorted(movie_title_to_id.keys())
    if seed_search.strip():
        matches = [t for t in all_titles_sorted
                   if seed_search.lower() in t.lower()][:50]
    else:
        matches = []

    if matches:
        pick_col, add_col = st.columns([4, 1])
        with pick_col:
            chosen = st.selectbox("Results:", ["— pick one —"] + matches,
                                  label_visibility="collapsed", key="seed_pick")
        with add_col:
            if st.button("➕ Add", key="seed_add_btn"):
                if (chosen != "— pick one —"
                        and chosen not in st.session_state["seed_titles"]
                        and len(st.session_state["seed_titles"]) < 5):
                    st.session_state["seed_titles"].append(chosen)
                    st.rerun()
    elif seed_search.strip():
        st.caption("No matches — try a different search term.")

    # ── Current seed list with remove buttons ──────────────────────────────────
    if st.session_state["seed_titles"]:
        st.markdown('<p class="section-label" style="margin-top:6px">Current seeds</p>',
                    unsafe_allow_html=True)
        to_remove = []
        for t in list(st.session_state["seed_titles"]):
            rc1, rc2 = st.columns([8, 1])
            rc1.markdown(f"🎬 **{t}**")
            if rc2.button("✕", key=f"rm_seed_{t}"):
                to_remove.append(t)
        for t in to_remove:
            st.session_state["seed_titles"].remove(t)
        if to_remove:
            st.rerun()

    selected_titles = st.session_state["seed_titles"]
    liked_movie_ids = [movie_title_to_id[t] for t in selected_titles if t in movie_title_to_id]

    # ── Seed preview ──────────────────────────────────────────────────────────
    if liked_movie_ids:
        st.markdown('<p class="section-label">Selected seeds</p>', unsafe_allow_html=True)
        seed_html = '<div class="seed-grid">'
        for mid in liked_movie_ids:
            r = movies[movies["movie_id"] == mid]
            if r.empty:
                continue
            r = r.iloc[0]
            genre_badges = "".join(
                f'<span class="badge">{g}</span>'
                for g in (r["genres"].split("|") if pd.notna(r.get("genres")) else [])
            )
            seed_html += clean_html(f"""
            <div class="seed-card">
              <div class="seed-card-title">{r["title"]}</div>
              <div class="seed-badges">{genre_badges}</div>
            </div>""")
        seed_html += "</div>"
        st.markdown(seed_html, unsafe_allow_html=True)

    # ── Generate ──────────────────────────────────────────────────────────────
    if st.button("🎬 Generate Recommendations", type="primary") and liked_movie_ids:
        with st.spinner("Finding your perfect movies…"):
            recs = get_hybrid_recommendations(
                user_id=user_id, liked_movie_ids=liked_movie_ids,
                model=model, trainset=trainset, all_movie_ids=all_movie_ids,
                similarity_matrix=sim_matrix, id_to_index=id_to_index,
                movies_df=movies_df, top_n=None, alpha=alpha  # filter first, cut to top_n below
            )
            recs = pd.merge(recs, links, on="movie_id", how="left")
            recs = pd.merge(recs, movies[["movie_id", "genres"]], on="movie_id", how="left")

            # Remove watchlisted movies
            recs = recs[~recs["movie_id"].isin(watchlisted_ids)]

            # Genre filter
            if selected_genres:
                mask = recs["genres"].apply(
                    lambda x: any(g in str(x).split("|") for g in selected_genres)
                )
                recs = recs[mask]

            # Surprise mode: keep only movies outside seed genres
            if surprise_mode:
                seed_genres = get_seed_genres(liked_movie_ids, movies)
                recs["_outside"] = recs["genres"].apply(
                    lambda g: is_outside_comfort(g, seed_genres)
                )
                recs = recs[recs["_outside"]]

            recs = recs.head(top_n).reset_index(drop=True)

        # Save to history — store only what we need to display, not the full df
        st.session_state["history"].insert(0, {
            "seeds":      list(selected_titles),  # copy: the live list is mutated on remove
            "user_id":    user_id,
            "top_titles": recs["title"].head(3).tolist() if "title" in recs.columns else [],
        })
        if len(st.session_state["history"]) > 10:
            st.session_state["history"] = st.session_state["history"][:10]

        # Persist the run: widget clicks (Save) rerun the script, and without this
        # the cards — and the clicked button — vanish before the click is handled.
        st.session_state["last_run"] = dict(
            recs=recs, alpha=alpha, surprise=surprise_mode,
            seeds=liked_movie_ids, user_id=user_id,
        )

    run = st.session_state.get("last_run")
    if run:
        recs = run["recs"]
        if recs.empty:
            st.warning("No matches found. Try removing genre filters or turning off Surprise Me.")
        else:
            st.subheader(f"🏆 Top {len(recs)} Picks for User {run['user_id']}")

            # ── Legend ────────────────────────────────────────────────────────
            st.markdown(clean_html(f"""
            <div class="legend-box">
              <strong>How scores work:</strong><br>
              <span class="dot dot-hybrid"></span><strong>Overall Match</strong> — Weighted blend of both signals.<br>
              <span class="dot dot-collab"></span><strong>People Like You</strong> — How much similar users enjoyed this ({int(run['alpha']*100)}% weight).<br>
              <span class="dot dot-content"></span><strong>Genre Match</strong> — How similar the genres are to your seeds ({int((1-run['alpha'])*100)}% weight).
            </div>"""), unsafe_allow_html=True)

            # ── Cards ─────────────────────────────────────────────────────────
            seed_genres_set = get_seed_genres(run["seeds"], movies) if run["surprise"] else set()
            for rank, (_, row) in enumerate(recs.iterrows()):
                outside = run["surprise"] and is_outside_comfort(row.get("genres", ""), seed_genres_set)
                render_rec_card(row, rank, run["seeds"], show_surprise_badge=outside)

            # ── Genre donut ───────────────────────────────────────────────────
            with st.expander("📊 Genre breakdown of these results"):
                all_g = []
                for g_str in recs["genres"].dropna():
                    all_g.extend([g for g in g_str.split("|") if g != "(no genres listed)"])
                if all_g:
                    g_counts = pd.Series(all_g).value_counts().reset_index()
                    g_counts.columns = ["Genre", "Count"]
                    fig = px.pie(g_counts, names="Genre", values="Count", hole=0.48,
                                 title="Genre distribution across recommendations",
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      font_color="rgba(255,255,255,0.7)", margin=dict(t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)

            # ── Export CSV ────────────────────────────────────────────────────
            csv_cols = ["title", "genres", "hybrid_score", "collab_norm", "content_norm"]
            csv_buf  = io.StringIO()
            recs[[c for c in csv_cols if c in recs.columns]].to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download recommendations as CSV",
                data=csv_buf.getvalue(),
                file_name=f"cinematch_user{run['user_id']}_recs.csv",
                mime="text/csv"
            )

    elif not liked_movie_ids:
        st.info("👆 Select at least one seed movie above, then click Generate Recommendations.")

    # ── Watchlist ─────────────────────────────────────────────────────────────
    if st.session_state["watchlist"]:
        with st.expander(f"🔖 Watchlist ({len(st.session_state['watchlist'])} movies)"):
            to_remove = []
            for mid, title in list(st.session_state["watchlist"].items()):
                c1, c2 = st.columns([5, 1])
                c1.write(f"🎬 {title}")
                if c2.button("✕", key=f"rm_{mid}"):
                    to_remove.append(mid)
            for mid in to_remove:
                del st.session_state["watchlist"][mid]
            if to_remove:
                st.rerun()

    # ── History ───────────────────────────────────────────────────────────────
    if st.session_state["history"]:
        with st.expander(f"📜 Recommendation history ({len(st.session_state['history'])} runs)"):
            for i, run in enumerate(st.session_state["history"]):
                seed_str  = " · ".join(run.get("seeds", []))
                top_titles = run.get("top_titles", [])
                top_str    = "  ·  ".join(top_titles)
                st.markdown(clean_html(f"""
                <div class="history-run">
                  <div class="history-label">Run #{i+1} — User {run['user_id']}</div>
                  <div class="history-seeds">Seeds: {seed_str}</div>
                  <div class="history-movies">Top picks: {top_str}…</div>
                </div>"""), unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — EXPLORE MOVIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.subheader("🔍 Explore a Movie")
    st.caption("Type a movie title below to see its stats and rating breakdown.")

    explore_query = st.text_input(
        "Search:", placeholder="e.g. Pulp Fiction, Titanic…", key="explore_search"
    )
    explore_matches = (
        [t for t in sorted(movie_title_to_id.keys())
         if explore_query.lower() in t.lower()][:50]
        if explore_query.strip() else []
    )
    if explore_matches:
        search_title = st.selectbox("Results:", explore_matches,
                                    label_visibility="collapsed", key="explore_pick")
    else:
        search_title = None
        if explore_query.strip():
            st.caption("No matches found.")

    if search_title and search_title in movie_title_to_id:
        selected_mid = movie_title_to_id[search_title]

        # ── Movie info ────────────────────────────────────────────────────────
        movie_row   = movies[movies["movie_id"] == selected_mid].iloc[0]
        movie_ratings_df = ratings[ratings["movie_id"] == selected_mid]["rating"]

        avg_rating   = movie_ratings_df.mean() if not movie_ratings_df.empty else 0
        num_ratings  = len(movie_ratings_df)
        genres_list  = movie_row["genres"].split("|") if pd.notna(movie_row.get("genres")) else []
        genre_badges = "".join(f'<span class="badge">{g}</span>' for g in genres_list)

        link_row = links[links["movie_id"] == selected_mid]
        imdb_link = ""
        if not link_row.empty and pd.notna(link_row.iloc[0].get("imdb_id")):
            iid = int(link_row.iloc[0]["imdb_id"])
            imdb_link = f'<a href="https://www.imdb.com/title/tt{iid:07d}/" target="_blank" class="imdb-btn" style="margin-left:10px">IMDb ↗</a>'

        st.markdown(clean_html(f"""
        <div class="movie-info-card">
          <div class="movie-info-title">{movie_row["title"]} {imdb_link}</div>
          <div class="seed-badges" style="margin-bottom:14px">{genre_badges}</div>
          <div class="movie-info-stats">
            <div class="movie-stat">
              <div class="val">⭐ {avg_rating:.2f}</div>
              <div class="lbl">Avg Rating</div>
            </div>
            <div class="movie-stat">
              <div class="val">{num_ratings:,}</div>
              <div class="lbl">Total Ratings</div>
            </div>
            <div class="movie-stat">
              <div class="val">{movie_ratings_df[movie_ratings_df >= 4].count():,}</div>
              <div class="lbl">Loved (≥4★)</div>
            </div>
          </div>
        </div>"""), unsafe_allow_html=True)

        # ── Rating histogram ──────────────────────────────────────────────────
        if not movie_ratings_df.empty:
            st.subheader("⭐ Rating Distribution")
            st.caption("A bimodal (two-peak) chart means polarizing opinions. A single peak means consensus.")
            hist_df = movie_ratings_df.value_counts().sort_index().reset_index()
            hist_df.columns = ["Rating", "Count"]
            fig = px.bar(
                hist_df, x="Rating", y="Count",
                color="Count",
                color_continuous_scale="Plasma",
                labels={"Rating": "Star Rating (1–5)", "Count": "Number of Users"},
                title=f"How {num_ratings:,} users rated \"{movie_row['title']}\""
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="rgba(255,255,255,0.7)",
                coloraxis_showscale=False,
                margin=dict(t=50, b=20),
                xaxis=dict(tickvals=[1, 2, 3, 4, 5],
                           ticktext=["1 ★", "2 ★", "3 ★", "4 ★", "5 ★"])
            )
            fig.update_traces(marker_line_width=0)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No ratings found for this movie in the dataset.")
    else:
        st.info("👆 Type a movie name above to explore it.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — USER INSIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    insight_tab_a, insight_tab_b = st.tabs(["👤 Similar Users", "⚔️ Comparison Mode"])

    # ── Similar Users ─────────────────────────────────────────────────────────
    with insight_tab_a:
        st.subheader("👥 Who thinks like you?")
        st.caption(
            "Collaborative filtering works by finding users with similar taste and borrowing "
            "their ratings. Here you can see your 5 nearest neighbours."
        )
        sim_user_id = st.number_input(
            "User ID to analyse", min_value=1, max_value=max_user_id, value=int(user_id),
            step=1, key="sim_uid"
        )
        if st.button("Find Similar Users", type="primary"):
            with st.spinner("Computing user similarities…"):
                sim_users = get_similar_users(sim_user_id, ratings, movies_df, top_n=5)
            if not sim_users:
                st.warning("User not found in the dataset.")
            else:
                st.markdown(f"**Top 5 users most similar to User {sim_user_id}:**")
                for rank, u in enumerate(sim_users, 1):
                    movie_bullets = "".join(f"<li>{t}</li>" for t in u["loved"])
                    sim_bar_pct   = int(u["similarity"] * 100)
                    st.markdown(clean_html(f"""
                    <div class="user-card">
                      <div class="user-card-header">
                        <span class="user-card-id">#{rank} · User {u['user_id']}</span>
                        <span class="user-card-sim">{u['similarity']:.1%} similar</span>
                      </div>
                      <div class="bar-bg" style="margin-bottom:10px">
                        <div class="bar-fill bar-collab" style="width:{sim_bar_pct}%"></div>
                      </div>
                      <div class="user-card-movies">
                        Movies they loved (≥4.5★):<br>
                        <ul style="margin:4px 0 0 16px; padding:0">{movie_bullets}</ul>
                      </div>
                    </div>"""), unsafe_allow_html=True)

    # ── Comparison Mode ───────────────────────────────────────────────────────
    with insight_tab_b:
        st.subheader("⚔️ Side-by-Side Comparison")
        st.caption("Run recommendations for two different users to see how personalised the system is.")

        col_a, col_b = st.columns(2)
        _cmp_defaults = [t for t in ["Toy Story (1995)", "Fargo (1996)"]
                         if t in movie_title_to_id]
        with col_a:
            uid_a = st.number_input("User A", 1, max_user_id, 1, key="cmp_a")
            cmp_q_a = st.text_input("Search seeds A:", key="cmp_q_a",
                                    placeholder="e.g. Titanic")
            cmp_opts_a = ([t for t in all_titles_sorted if cmp_q_a.lower() in t.lower()][:40]
                          if cmp_q_a.strip() else _cmp_defaults)
            seeds_a = st.multiselect("Seeds A:", options=cmp_opts_a,
                                     default=[t for t in _cmp_defaults if t in cmp_opts_a],
                                     label_visibility="collapsed", key="seeds_a")
        with col_b:
            uid_b = st.number_input("User B", 1, max_user_id, 2, key="cmp_b")
            cmp_q_b = st.text_input("Search seeds B:", key="cmp_q_b",
                                    placeholder="e.g. Matrix")
            cmp_opts_b = ([t for t in all_titles_sorted if cmp_q_b.lower() in t.lower()][:40]
                          if cmp_q_b.strip() else _cmp_defaults)
            seeds_b = st.multiselect("Seeds B:", options=cmp_opts_b,
                                     default=[t for t in _cmp_defaults if t in cmp_opts_b],
                                     label_visibility="collapsed", key="seeds_b")

        if st.button("⚔️ Compare", type="primary"):
            ids_a = [movie_title_to_id[t] for t in seeds_a if t in movie_title_to_id]
            ids_b = [movie_title_to_id[t] for t in seeds_b if t in movie_title_to_id]
            if not ids_a or not ids_b:
                st.warning("Please pick at least one seed for each user.")
            else:
                with st.spinner("Computing both recommendation sets…"):
                    recs_a = get_hybrid_recommendations(uid_a, ids_a, model, trainset,
                                all_movie_ids, sim_matrix, id_to_index, movies_df,
                                top_n=8, alpha=alpha)
                    recs_b = get_hybrid_recommendations(uid_b, ids_b, model, trainset,
                                all_movie_ids, sim_matrix, id_to_index, movies_df,
                                top_n=8, alpha=alpha)

                shared = set(recs_a["title"]) & set(recs_b["title"])
                ca, cb = st.columns(2)
                with ca:
                    st.markdown(f"**User {uid_a}'s Recommendations**")
                    for rank, (_, r) in enumerate(recs_a.iterrows()):
                        marker = "🤝 " if r["title"] in shared else ""
                        st.markdown(f"{rank+1}. {marker}{r['title']} — {r['hybrid_score']:.1%}")
                with cb:
                    st.markdown(f"**User {uid_b}'s Recommendations**")
                    for rank, (_, r) in enumerate(recs_b.iterrows()):
                        marker = "🤝 " if r["title"] in shared else ""
                        st.markdown(f"{rank+1}. {marker}{r['title']} — {r['hybrid_score']:.1%}")

                st.info(f"🤝 **{len(shared)} movie(s) in common** — "
                        f"{'Very similar taste!' if len(shared) >= 4 else 'Quite different tastes!'}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — EVALUATION DASHBOARD  (three sub-tabs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    eval_sub_a, eval_sub_b, eval_sub_c = st.tabs(
        ["📏 Metrics", "🔬 Latent Factor Visualizer", "⚖️ α Sensitivity"]
    )

    # ══════════════════════════════════════════════════════════════════
    # SUB-TAB A — RMSE / Precision / Recall
    # ══════════════════════════════════════════════════════════════════
    with eval_sub_a:
        st.subheader("📊 Model Metrics")
        st.caption(
            "How well does the SVD collaborative model perform on held-out data? "
            "Runs 5-fold cross-validation — slow once, cached forever after."
        )
        k_val = st.slider("K for Precision@K and Recall@K", 5, 20, 10, key="eval_k")

        if st.button("▶️ Run Evaluation", type="primary", key="run_eval_btn"):
            with st.spinner("Running 5-fold cross-validation… (~30–60 seconds)"):
                rmse = run_rmse_eval(ratings)
                prec, recall = run_pr_eval(ratings, k=k_val)

            rmse_pct = max(0, min(100, int((1 - (rmse - 0.5) / 1.0) * 100)))
            st.markdown(clean_html(f"""
            <div class="eval-card">
              <div class="eval-title">📏 RMSE — Root Mean Squared Error</div>
              <div class="eval-subtitle">How far off predictions are on average (lower = better).
                Below 0.9 on a 1–5 scale is good.</div>
              <div class="eval-value" style="color:#ff6b6b">{rmse:.4f}</div>
              <div class="score-meta" style="margin-bottom:4px">
                <span>Accuracy (inverted)</span><strong>{rmse_pct}%</strong>
              </div>
              <div class="bar-bg"><div class="bar-fill bar-hybrid" style="width:{rmse_pct}%"></div></div>
            </div>"""), unsafe_allow_html=True)

            prec_pct = int(prec * 100)
            st.markdown(clean_html(f"""
            <div class="eval-card">
              <div class="eval-title">🎯 Precision@{k_val}</div>
              <div class="eval-subtitle">Of the top {k_val} picks, what fraction did the user actually like (≥4★)?</div>
              <div class="eval-value" style="color:#4ecdc4">{prec:.4f}</div>
              <div class="score-meta" style="margin-bottom:4px">
                <span>{prec_pct}% of top-{k_val} picks were hits</span><strong>{prec_pct}%</strong>
              </div>
              <div class="bar-bg"><div class="bar-fill bar-collab" style="width:{prec_pct}%"></div></div>
            </div>"""), unsafe_allow_html=True)

            rec_pct = int(recall * 100)
            st.markdown(clean_html(f"""
            <div class="eval-card">
              <div class="eval-title">🔎 Recall@{k_val}</div>
              <div class="eval-subtitle">Of all movies the user liked, what fraction appeared in the top {k_val}?</div>
              <div class="eval-value" style="color:#a29bfe">{recall:.4f}</div>
              <div class="score-meta" style="margin-bottom:4px">
                <span>{rec_pct}% of liked movies surfaced</span><strong>{rec_pct}%</strong>
              </div>
              <div class="bar-bg"><div class="bar-fill bar-content" style="width:{rec_pct}%"></div></div>
            </div>"""), unsafe_allow_html=True)

            st.success(
                f"✅ RMSE {rmse:.3f} · Precision@{k_val} {prec:.1%} · Recall@{k_val} {recall:.1%}"
            )
        else:
            st.info("👆 Click **Run Evaluation** above. Results are cached after the first run.")

    # ══════════════════════════════════════════════════════════════════
    # SUB-TAB B — LATENT FACTOR VISUALIZER
    # ══════════════════════════════════════════════════════════════════
    with eval_sub_b:
        st.subheader("🔬 Latent Factor Space (PCA of SVD Item Vectors)")
        st.caption(
            "SVD learns a hidden vector for every movie. PCA compresses those vectors "
            "into 2D so we can see which movies cluster together — and whether the "
            "clusters align with genres."
        )

        @st.cache_data
        def build_pca_df(_model, _trainset, _movies, title_map, n_sample=2000):
            """Project SVD item factors → 2D via PCA, return a plot-ready DataFrame.

            title_map is passed explicitly so this function has no hidden
            dependency on module-level state (safe to move to any file).
            """
            qi = _model.qi                          # shape: (n_items, n_factors)
            n_items = qi.shape[0]

            # Sample to keep the scatter readable
            sample_idx = np.random.RandomState(42).choice(
                n_items, size=min(n_sample, n_items), replace=False
            )

            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(qi[sample_idx])

            rows = []
            for local_i, inner_id in enumerate(sample_idx):
                try:
                    raw_id = _trainset.to_raw_iid(int(inner_id))
                    title  = title_map.get(raw_id, f"#{raw_id}")   # explicit param, no globals
                    mrow   = _movies[_movies["movie_id"] == raw_id]
                    genre  = (
                        mrow.iloc[0]["genres"].split("|")[0]
                        if not mrow.empty and pd.notna(mrow.iloc[0]["genres"])
                        else "Unknown"
                    )
                    rows.append({
                        "PC1": round(float(coords[local_i, 0]), 4),
                        "PC2": round(float(coords[local_i, 1]), 4),
                        "title": title,
                        "genre": genre,
                    })
                except Exception:
                    pass

            var_explained = pca.explained_variance_ratio_
            return pd.DataFrame(rows), var_explained

        if st.button("🔬 Build Visualizer", type="primary", key="pca_btn"):
            with st.spinner("Running PCA on SVD item factors…"):
                pca_df, var = build_pca_df(model, trainset, movies, movie_id_to_title)

            n_sample_label = len(pca_df)
            top_genres = pca_df["genre"].value_counts().head(12).index.tolist()
            plot_df = pca_df[pca_df["genre"].isin(top_genres)].copy()

            fig = px.scatter(
                plot_df, x="PC1", y="PC2",
                color="genre",
                hover_name="title",
                hover_data={"PC1": ":.3f", "PC2": ":.3f", "genre": True},
                title=(
                    f"Movie Latent Space — {n_sample_label:,} movies · "
                    f"PC1 {var[0]:.1%} var · PC2 {var[1]:.1%} var"
                ),
                color_discrete_sequence=px.colors.qualitative.Pastel,
                opacity=0.75,
                height=580,
            )
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="rgba(255,255,255,0.75)",
                legend_title_text="Primary Genre",
                margin=dict(t=60, b=20),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Showing top-12 genres by count out of {n_sample_label:,} sampled movies. "
                "Movies that cluster close together share similar latent traits learned from "
                "co-rating patterns — not just genre labels."
            )
        else:
            st.info(
                "👆 Click **Build Visualizer** to project SVD item vectors into 2D. "
                "The chart is cached after the first render."
            )

    # ══════════════════════════════════════════════════════════════════
    # SUB-TAB C — α SENSITIVITY CHART
    # ══════════════════════════════════════════════════════════════════
    with eval_sub_c:
        st.subheader("⚖️ α Sensitivity — How Does the Mix Change Your Results?")
        st.caption(
            "Sweeps α from 0.0 (pure genre similarity) to 1.0 (pure collaborative) "
            "in steps of 0.1 and records the top-5 recommendations at each step. "
            "Uses the current user ID and seed movies from the Recommend tab."
        )

        sweep_seeds = st.session_state.get("seed_titles", [])
        if sweep_seeds:
            st.info(
                f"**Current seeds:** {' · '.join(sweep_seeds)}  ·  "
                f"**User:** {user_id}"
            )
        else:
            st.warning("No seed movies selected yet — go to the Recommend tab and add some seeds first.")

        if st.button("⚖️ Run α Sweep", type="primary", key="alpha_sweep_btn") and sweep_seeds:
            sweep_ids = [movie_title_to_id[t] for t in sweep_seeds if t in movie_title_to_id]
            if not sweep_ids:
                st.warning("Seed movies not found in dataset.")
            else:
                alphas = [round(a * 0.1, 1) for a in range(11)]
                sweep_rows = []

                # ── Compute collab + content scores ONCE, then reweight per α ────────
                # This avoids 11 × O(n_unrated) model.predict calls — scales to
                # large datasets (e.g. MovieLens 1M) without modification.
                with st.spinner("Computing scores (1× collab + 1× content pass)…"):
                    merged = get_hybrid_recommendations(
                        user_id, sweep_ids, model, trainset, all_movie_ids,
                        sim_matrix, id_to_index, movies_df, top_n=None
                    ).rename(columns={"collab_norm": "cn", "content_norm": "tn"})

                # ── Sweep: pure numpy reweighting, no model calls ────────────────
                progress = st.progress(0, text="Building sweep table…")
                for step, a in enumerate(alphas):
                    merged["_h"] = a * merged["cn"] + (1 - a) * merged["tn"]
                    top5_titles  = merged.nlargest(5, "_h")["title"].tolist()
                    top5_titles += [""] * (5 - len(top5_titles))
                    sweep_rows.append({
                        "α": a,
                        "#1": top5_titles[0], "#2": top5_titles[1], "#3": top5_titles[2],
                        "#4": top5_titles[3], "#5": top5_titles[4],
                    })
                    progress.progress((step + 1) / len(alphas),
                                      text=f"α = {a:.1f} — top pick: {top5_titles[0][:40]}")
                progress.empty()

                sweep_df = pd.DataFrame(sweep_rows)

                # ── Heatmap: which titles appear across alphas ──────────────────
                # Collect all unique titles (preserving order of first appearance)
                all_titles_in_sweep = []
                seen = set()
                for row in sweep_rows:
                    for k in ["#1","#2","#3","#4","#5"]:
                        t = row[k]
                        if t and t not in seen:
                            all_titles_in_sweep.append(t)
                            seen.add(t)

                # Build presence matrix: rows=titles, cols=alphas
                heat_data = []
                for title in all_titles_in_sweep:
                    row_vals = []
                    for row in sweep_rows:
                        rank_val = 0
                        for rank_k, k in enumerate(["#1","#2","#3","#4","#5"], 1):
                            if row[k] == title:
                                rank_val = 6 - rank_k   # #1 → 5, #5 → 1
                                break
                        row_vals.append(rank_val)
                    heat_data.append(row_vals)

                heat_arr  = np.array(heat_data, dtype=float)
                alpha_lbls = [f"α={a:.1f}" for a in alphas]

                fig_heat = go.Figure(go.Heatmap(
                    z=heat_arr,
                    x=alpha_lbls,
                    y=all_titles_in_sweep,
                    colorscale=[
                        [0.0, "rgba(0,0,0,0)"],
                        [0.2, "#2d3561"],
                        [0.6, "#4ecdc4"],
                        [1.0, "#feca57"],
                    ],
                    showscale=True,
                    colorbar=dict(
                        title="Rank strength",
                        tickvals=[0,1,2,3,4,5],
                        ticktext=["not ranked","#5","#4","#3","#2","#1"],
                        tickfont=dict(color="rgba(255,255,255,0.6)"),
                    ),
                    hoverongaps=False,
                    hovertemplate="%{y}<br>%{x}<br>Rank strength: %{z}<extra></extra>",
                ))
                fig_heat.update_layout(
                    title="Top-5 recommendation presence as α shifts (yellow = #1 pick)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="rgba(255,255,255,0.75)",
                    height=max(300, 28 * len(all_titles_in_sweep) + 100),
                    margin=dict(t=60, b=20, l=20, r=20),
                    xaxis=dict(side="top"),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_heat, use_container_width=True)

                # ── Plain table view ───────────────────────────────────────────
                with st.expander("📋 See as table"):
                    display_df = sweep_df[["α","#1","#2","#3","#4","#5"]].copy()
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                st.caption(
                    "**Reading the heatmap:** each row is a movie. Yellow = ranked #1 at that α. "
                    "Grey = not in the top 5. Titles that appear across the full width are "
                    "recommended regardless of the mix — they're the safest bets."
                )
        elif not sweep_seeds:
            st.info("Add seed movies in the Recommend tab first, then come back here.")