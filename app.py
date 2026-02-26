"""
Movie Recommendation System — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
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
        margin-bottom: 0.2rem;
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
        padding: 12px;
        text-align: center;
        transition: transform 0.2s;
        height: 100%;
    }
    .movie-card:hover { transform: scale(1.03); }
    .movie-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #fff;
        margin-top: 8px;
    }
    .movie-meta {
        font-size: 0.75rem;
        color: #aaa;
    }
    .score-badge {
        background: #E50914;
        color: white;
        border-radius: 6px;
        padding: 2px 6px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    .stSelectbox label { color: #fff !important; }
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
    """Render a single movie card in a Streamlit column."""
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

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Settings")
        n_recs = st.slider("Number of recommendations", 5, 20, 10)
        st.markdown("---")
        st.markdown("**About**")
        st.markdown(
            "Built by **B. Vikas** | AIML Student\n\n"
            "Uses TMDB metadata with cosine similarity on genres, keywords, cast & director."
        )
        st.markdown("---")
        if rec:
            st.success(f"✅ {len(rec.df):,} movies loaded")

    if rec is None:
        st.error(
            "⚠️ Model artifacts not found!\n\n"
            "**Setup Instructions:**\n"
            "1. Download the TMDB dataset from Kaggle\n"
            "2. Place CSVs in the `data/` folder\n"
            "3. Run: `python src/train.py`\n"
            "4. Restart the app"
        )
        st.code(
            "# Quick setup\n"
            "pip install -r requirements.txt\n"
            "# Download data from Kaggle (see README)\n"
            "python src/train.py",
            language="bash",
        )
        return

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["🔍 Find Recommendations", "🏆 Top Rated", "📊 About the Model"])

    # ── Tab 1: Recommendations ──
    with tab1:
        all_titles = sorted(rec.df["title"].tolist())
        selected   = st.selectbox("🎥 Choose a movie you like:", all_titles, index=all_titles.index("The Dark Knight") if "The Dark Knight" in all_titles else 0)

        if st.button("✨ Get Recommendations", use_container_width=True, type="primary"):
            with st.spinner("Finding similar movies..."):
                try:
                    results = rec.recommend(selected, n=n_recs)
                    st.subheader(f"Because you liked **{selected}**:")

                    cols = st.columns(5)
                    for i, row in results.iterrows():
                        render_movie_card(
                            cols[i % 5],
                            row["title"],
                            row["vote_average"],
                            row["release_date"],
                            row["poster_path"],
                            sim_score=row["similarity_score"],
                        )
                        if (i + 1) % 5 == 0 and i + 1 < len(results):
                            cols = st.columns(5)
                            st.markdown("<br>", unsafe_allow_html=True)

                except ValueError as e:
                    st.error(str(e))

    # ── Tab 2: Top Rated ──
    with tab2:
        st.subheader("🏆 Top Rated Movies in Dataset")
        top = rec.top_rated(n=15)
        st.dataframe(
            top.style.background_gradient(subset=["vote_average"], cmap="RdYlGn"),
            use_container_width=True,
        )

    # ── Tab 3: Model Info ──
    with tab3:
        st.subheader("📊 How the Recommendation Engine Works")
        st.markdown("""
        ### 🧠 Algorithm: Content-Based Filtering

        **Step 1 — Feature Engineering**  
        Each movie is described by combining:
        - 📝 **Overview** (plot keywords)
        - 🎭 **Genres** (e.g., Action, Drama)
        - 🔑 **Keywords** (TMDB tags)
        - 🎬 **Top 3 Cast members**
        - 🎥 **Director**

        All text is lowercased and concatenated into a single **"tags"** string.

        **Step 2 — Bag-of-Words Vectorization**  
        `CountVectorizer` (max 5,000 features, English stop-words removed) converts each movie's tags into a sparse vector.

        **Step 3 — Cosine Similarity**  
        The angle between two movie vectors determines similarity.  
        A score of **1.0** = identical features; **0.0** = no overlap.

        **Step 4 — Ranking**  
        For a query movie, all cosine scores are sorted descending and the top-N movies are returned.

        ```
        similarity(A, B) = (A · B) / (||A|| × ||B||)
        ```
        """)

        col1, col2, col3 = st.columns(3)
        col1.metric("📽️ Total Movies", f"{len(rec.df):,}")
        col2.metric("🔑 Max Features", "5,000")
        col3.metric("📐 Algorithm", "Cosine Similarity")


if __name__ == "__main__":
    main()
