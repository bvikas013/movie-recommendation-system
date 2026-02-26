"""
Movie Recommendation System — Streamlit App
Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import sys

# use current folder (no src folder)
sys.path.insert(0, os.path.dirname(__file__))
from recommender import MovieRecommender


# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------
# Styling
# ---------------------------------------------------
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    font-weight: 800;
    color: #E50914;
    text-align: center;
}
.sub-header {
    font-size: 1rem;
    color: #999;
    text-align: center;
    margin-bottom: 2rem;
}
.movie-card {
    background: #1a1a1a;
    border-radius: 12px;
    padding: 10px;
    text-align: center;
}
.movie-title {
    color: white;
    font-weight: 700;
    font-size: 0.9rem;
}
.movie-meta {
    color: #aaa;
    font-size: 0.75rem;
}
.score-badge {
    background:#E50914;
    color:white;
    border-radius:6px;
    padding:2px 6px;
    font-size:0.7rem;
}
</style>
""", unsafe_allow_html=True)

TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w300"
PLACEHOLDER_IMG = "https://via.placeholder.com/200x300/1a1a1a/E50914?text=No+Image"


# ---------------------------------------------------
# Load Model (FIXED VERSION)
# ---------------------------------------------------
@st.cache_resource(show_spinner="Loading recommendation engine...")
def load_model():

    rec = MovieRecommender()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    artifacts_path = os.path.join(BASE_DIR, "artifacts")

    movies_file = os.path.join(artifacts_path, "movies.pkl")
    sim_file = os.path.join(artifacts_path, "similarity.pkl")

    if os.path.exists(movies_file) and os.path.exists(sim_file):
        rec.load(artifacts_path)
        return rec

    return None


# ---------------------------------------------------
# Movie Card UI
# ---------------------------------------------------
def render_movie_card(col, title, vote_avg, release_date, poster_path, sim_score=None):

    img_url = f"{TMDB_IMG_BASE}{poster_path}" if poster_path else PLACEHOLDER_IMG
    year = str(release_date)[:4] if pd.notna(release_date) else "N/A"

    score_html = (
        f'<span class="score-badge">★ {vote_avg:.1f}</span>'
        if pd.notna(vote_avg) else ""
    )

    sim_html = (
        f'<p class="movie-meta">Similarity: {sim_score:.2%}</p>'
        if sim_score else ""
    )

    with col:
        st.markdown(f"""
        <div class="movie-card">
            <img src="{img_url}" width="100%" style="border-radius:8px;"
                 onerror="this.src='{PLACEHOLDER_IMG}'"/>
            <p class="movie-title">{title}</p>
            <p class="movie-meta">{year} {score_html}</p>
            {sim_html}
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------
# Main App
# ---------------------------------------------------
def main():

    st.markdown('<div class="main-header">🎬 CineMatch</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Content-based movie recommendations using cosine similarity</div>',
        unsafe_allow_html=True,
    )

    rec = load_model()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        n_recs = st.slider("Number of recommendations", 5, 20, 10)

        if rec:
            st.success(f"✅ {len(rec.df):,} movies loaded")

    if rec is None:
        st.error("Model artifacts not found. Run: python train.py")
        return

    tab1, tab2, tab3 = st.tabs(
        ["🔍 Recommendations", "🏆 Top Rated", "📊 About Model"]
    )

    # ---------------- TAB 1 ----------------
    with tab1:

        all_titles = sorted(rec.df["title"].tolist())

        selected = st.selectbox(
            "🎥 Choose a movie:",
            all_titles,
            index=all_titles.index("The Dark Knight")
            if "The Dark Knight" in all_titles else 0,
        )

        if st.button("✨ Get Recommendations", width="stretch", type="primary"):

            with st.spinner("Finding similar movies..."):

                results = rec.recommend(selected, n=n_recs)

                st.subheader(f"Because you liked **{selected}**")

                cols = st.columns(5)

                for i, row in results.iterrows():

                    render_movie_card(
                        cols[i % 5],
                        row["title"],
                        row["vote_average"],
                        row["release_date"],
                        row["poster_path"],
                        row["similarity_score"],
                    )

                    if (i + 1) % 5 == 0 and i + 1 < len(results):
                        cols = st.columns(5)

    # ---------------- TAB 2 ----------------
    with tab2:

        top = rec.top_rated(15)

        st.dataframe(
            top.style.background_gradient(subset=["vote_average"], cmap="RdYlGn"),
            width="stretch",
        )

    # ---------------- TAB 3 ----------------
    with tab3:

        st.markdown("""
        ### 🧠 How It Works

        - Combine overview + genres + cast + director → tags
        - Convert text into vectors using CountVectorizer
        - Compute cosine similarity
        - Return most similar movies
        """)

        c1, c2, c3 = st.columns(3)
        c1.metric("Movies", f"{len(rec.df):,}")
        c2.metric("Features", "5000")
        c3.metric("Algorithm", "Cosine Similarity")


if __name__ == "__main__":
    main()