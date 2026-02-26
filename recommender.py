"""
Movie Recommendation Engine
----------------------------
Content-based filtering using cosine similarity on movie metadata.
Dataset: TMDB 5000 Movies (from Kaggle)
"""

import pandas as pd
import numpy as np
import ast
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────
# Helper functions for parsing TMDB JSON columns
# ──────────────────────────────────────────────

def parse_list_column(obj, key="name", limit=None):
    """Extract 'name' values from a JSON-like list of dicts."""
    try:
        items = ast.literal_eval(obj)
        names = [i[key].replace(" ", "") for i in items]
        return names[:limit] if limit else names
    except (ValueError, TypeError):
        return []


def get_director(crew_json):
    """Extract the director's name from the crew column."""
    try:
        crew = ast.literal_eval(crew_json)
        for member in crew:
            if member.get("job") == "Director":
                return [member["name"].replace(" ", "")]
        return []
    except (ValueError, TypeError):
        return []


def clean_overview(text):
    """Tokenize overview into a list of words."""
    if isinstance(text, str):
        return text.split()
    return []


# ──────────────────────────────────────────────
# Data Loading & Preprocessing
# ──────────────────────────────────────────────

def load_and_merge(movies_path: str, credits_path: str) -> pd.DataFrame:
    """Load the two TMDB CSVs and merge them on title."""
    movies  = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # credits CSV has a 'movie_id' column; rename to match
    credits.rename(columns={"movie_id": "id"}, inplace=True)
    df = movies.merge(credits, on="id")

    # Keep only the columns we need
    df = df[["id", "title", "overview", "genres", "keywords",
             "cast", "crew", "vote_average", "vote_count",
             "release_date", "poster_path"]]
    return df


def build_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering: combine genres, keywords, top 3 cast members,
    director, and overview words into a single 'tags' string.
    """
    df = df.copy()
    df.dropna(inplace=True)

    df["genres"]   = df["genres"].apply(lambda x: parse_list_column(x))
    df["keywords"] = df["keywords"].apply(lambda x: parse_list_column(x))
    df["cast"]     = df["cast"].apply(lambda x: parse_list_column(x, limit=3))
    df["crew"]     = df["crew"].apply(get_director)
    df["overview"] = df["overview"].apply(clean_overview)

    # Concatenate all features → single list → join to string
    df["tags"] = (
        df["overview"] + df["genres"] + df["keywords"]
        + df["cast"] + df["crew"]
    )
    df["tags"] = df["tags"].apply(lambda x: " ".join(x).lower())

    return df[["id", "title", "tags", "vote_average",
               "vote_count", "release_date", "poster_path"]]


# ──────────────────────────────────────────────
# Similarity Matrix
# ──────────────────────────────────────────────

def build_similarity_matrix(df: pd.DataFrame):
    """
    Vectorize the tags column using CountVectorizer (bag-of-words)
    and compute pairwise cosine similarity.
    Returns the vectorizer and the similarity matrix.
    """
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return cv, similarity


# ──────────────────────────────────────────────
# Recommendation
# ──────────────────────────────────────────────

class MovieRecommender:
    """
    Content-based movie recommender using cosine similarity.

    Usage
    -----
    >>> rec = MovieRecommender()
    >>> rec.fit("data/tmdb_5000_movies.csv", "data/tmdb_5000_credits.csv")
    >>> rec.recommend("The Dark Knight")
    """

    def __init__(self):
        self.df         = None
        self.similarity = None
        self._index_map = {}   # title → row index

    # ── Training ──────────────────────────────
    def fit(self, movies_csv: str, credits_csv: str):
        raw = load_and_merge(movies_csv, credits_csv)
        self.df = build_tags(raw).reset_index(drop=True)
        _, self.similarity = build_similarity_matrix(self.df)
        self._index_map = {
            title: idx for idx, title in enumerate(self.df["title"])
        }
        print(f"[✓] Recommender fitted on {len(self.df)} movies.")

    # ── Saving / Loading ──────────────────────
    def save(self, path: str = "artifacts"):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/similarity.pkl", "wb") as f:
            pickle.dump(self.similarity, f)
        self.df.to_pickle(f"{path}/movies.pkl")
        print(f"[✓] Artifacts saved to '{path}/'")

    def load(self, path: str = "artifacts"):
        self.df = pd.read_pickle(f"{path}/movies.pkl")
        with open(f"{path}/similarity.pkl", "rb") as f:
            self.similarity = pickle.load(f)
        self._index_map = {
            title: idx for idx, title in enumerate(self.df["title"])
        }
        print(f"[✓] Recommender loaded — {len(self.df)} movies available.")

    # ── Recommending ──────────────────────────
    def recommend(self, title: str, n: int = 10) -> pd.DataFrame:
        """
        Return the top-n most similar movies.

        Parameters
        ----------
        title : str   Movie title (must exist in the dataset)
        n     : int   Number of recommendations (default 10)

        Returns
        -------
        pd.DataFrame  Columns: title, vote_average, release_date, poster_path
        """
        if title not in self._index_map:
            raise ValueError(
                f"'{title}' not found. Use search() to find valid titles."
            )

        idx = self._index_map[title]
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:n]

        rec_indices = [s[0] for s in scores]
        result = self.df.iloc[rec_indices][
            ["title", "vote_average", "vote_count", "release_date", "poster_path"]
        ].copy()
        result["similarity_score"] = [round(s[1], 4) for s in scores]
        result.reset_index(drop=True, inplace=True)
        return result

    def search(self, query: str, limit: int = 10) -> list:
        """Return movie titles that contain the query string (case-insensitive)."""
        q = query.lower()
        return [t for t in self._index_map if q in t.lower()][:limit]

    def top_rated(self, n: int = 10) -> pd.DataFrame:
        """Return top-rated movies (by vote_average, minimum 500 votes)."""
        return (
            self.df[self.df["vote_count"] > 500]
            .sort_values("vote_average", ascending=False)
            .head(n)[["title", "vote_average", "vote_count"]]
        )
