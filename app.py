"""
Movie Recommendation System — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import sys

# FIX: no src folder → use current directory
sys.path.insert(0, os.path.dirname(__file__))
from recommender import MovieRecommender


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
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
    font-size: 0.9rem;
    font-weight: 700;
    color: white;
}
.movie-meta {
    font-size: 0.75rem;
    color: #aaa;
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


# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading recommendation engine...")
def load_model():
    rec = MovieRecommender()
    artifacts_path = os.path.join(os.path.dirname(__file__), "artifacts")

    if os.path.exists(artifacts_path):
        rec.load(artifacts_path)
        return rec
    return None


def render_movie_card(col, title, vote_avg, release_date, poster_path, sim_score=None):

    img_url = f"{TMDB_IMG_BASE}{poster_path}" if poster_path else PLACEHOLDER_IMG
    year = str(release_date)[:4] if pd.notna(release_date) else "N/A"

    score_html = f'<span class="score-badge">★ {vote_avg:.1f}</span>' if pd.notna(vote_avg) else ""
    sim_html = f'<p class="movie-meta">Similarity: {sim_score:.2%}</p>' if sim_score else ""

    with col:
        st.markdown(f"""
        <div class="movie-card">
            <img src="{img_url}" width="100%" style="border-radius:8px;"
                 onerror="this.src='{PLACEHOLDER_IMG}'"/>
            <p class="movie-title">{title}</p>
            <p class="movie-meta">{year} &nbsp; {score_html}</p>
            {sim_html}
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────
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
        ["🔍 Find Recommendations", "🏆 Top Rated", "📊 About Model"]
    )

    # ---------- Tab 1 ----------
    with tab1:

        all_titles = sorted(rec.df["title"].tolist())

        selected = st.selectbox(
            "🎥 Choose a movie:",
            all_titles,
            index=all_titles.index("The Dark Knight")
            if "The Dark Knight" in all_titles else 0,
        )

        # FIX: width instead of use_container_width
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

    # ---------- Tab 2 ----------
    with tab2:

        top = rec.top_rated(n=15)

        st.dataframe(
            top.style.background_gradient(subset=["vote_average"], cmap="RdYlGn"),
            width="stretch",  # FIXED
        )

    # ---------- Tab 3 ----------
    with tab3:
        st.markdown("""
        ### 🧠 How it works

        - Overview + genres + keywords + cast + director → tags  
        - CountVectorizer converts text to vectors  
        - Cosine similarity finds similar movies
        """)

        c1, c2, c3 = st.columns(3)
        c1.metric("Movies", f"{len(rec.df):,}")
        c2.metric("Features", "5000")
        c3.metric("Algorithm", "Cosine Similarity")


if __name__ == "__main__":
    main()