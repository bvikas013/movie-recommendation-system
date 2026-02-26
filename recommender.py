"""
Movie Recommendation Engine
----------------------------
Content-based movie recommender using cosine similarity.
Dataset: TMDB 5000 Movies (Kaggle)
"""

import pandas as pd
import ast
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# Helper Functions
# ======================================================

def parse_list_column(obj, key="name", limit=None):
    """Extract names from JSON-like column."""
    try:
        items = ast.literal_eval(obj)
        names = [i[key].replace(" ", "") for i in items]
        return names[:limit] if limit else names
    except Exception:
        return []


def get_director(crew_json):
    """Extract director name from crew."""
    try:
        crew = ast.literal_eval(crew_json)
        for member in crew:
            if member.get("job") == "Director":
                return [member["name"].replace(" ", "")]
        return []
    except Exception:
        return []


def clean_overview(text):
    """Split overview into words."""
    if isinstance(text, str):
        return text.split()
    return []


# ======================================================
# Load & Merge Data
# ======================================================

def load_and_merge(movies_path, credits_path):

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    if "movie_id" in credits.columns:
        credits.rename(columns={"movie_id": "id"}, inplace=True)

    df = movies.merge(credits, on="id")

    # Fix duplicate title columns
    if "title_x" in df.columns:
        df["title"] = df["title_x"]
    elif "title_y" in df.columns:
        df["title"] = df["title_y"]

    # Add poster column if missing
    if "poster_path" not in df.columns:
        df["poster_path"] = None

    df = df[
        [
            "id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew",
            "vote_average",
            "vote_count",
            "release_date",
            "poster_path",
        ]
    ]

    return df


# ======================================================
# Feature Engineering  (FIXED VERSION)
# ======================================================

def build_tags(df):

    df = df.copy()

    # ✅ fill missing values (DO NOT drop rows)
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("[]")
    df["keywords"] = df["keywords"].fillna("[]")
    df["cast"] = df["cast"].fillna("[]")
    df["crew"] = df["crew"].fillna("[]")

    df["genres"] = df["genres"].apply(parse_list_column)
    df["keywords"] = df["keywords"].apply(parse_list_column)
    df["cast"] = df["cast"].apply(lambda x: parse_list_column(x, limit=3))
    df["crew"] = df["crew"].apply(get_director)
    df["overview"] = df["overview"].apply(clean_overview)

    df["tags"] = (
        df["overview"]
        + df["genres"]
        + df["keywords"]
        + df["cast"]
        + df["crew"]
    )

    df["tags"] = df["tags"].apply(lambda x: " ".join(x).lower())

    # remove empty tags safely
    df = df[df["tags"].str.strip() != ""]

    return df[
        [
            "id",
            "title",
            "tags",
            "vote_average",
            "vote_count",
            "release_date",
            "poster_path",
        ]
    ]


# ======================================================
# Similarity Matrix
# ======================================================

def build_similarity_matrix(df):

    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()

    similarity = cosine_similarity(vectors)

    return cv, similarity


# ======================================================
# Movie Recommender Class
# ======================================================

class MovieRecommender:

    def __init__(self):
        self.df = None
        self.similarity = None
        self._index_map = {}

    def fit(self, movies_csv, credits_csv):

        raw = load_and_merge(movies_csv, credits_csv)

        self.df = build_tags(raw).reset_index(drop=True)

        _, self.similarity = build_similarity_matrix(self.df)

        self._index_map = {
            title: idx for idx, title in enumerate(self.df["title"])
        }

        print(f"[✓] Recommender trained on {len(self.df)} movies.")

    def save(self, path="artifacts"):

        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "similarity.pkl"), "wb") as f:
            pickle.dump(self.similarity, f)

        self.df.to_pickle(os.path.join(path, "movies.pkl"))

        print(f"[✓] Artifacts saved to {path}/")

    def load(self, path="artifacts"):

        self.df = pd.read_pickle(os.path.join(path, "movies.pkl"))

        with open(os.path.join(path, "similarity.pkl"), "rb") as f:
            self.similarity = pickle.load(f)

        self._index_map = {
            title: idx for idx, title in enumerate(self.df["title"])
        }

        print(f"[✓] Loaded {len(self.df)} movies.")

    def recommend(self, title, n=10):

        if title not in self._index_map:
            raise ValueError(f"{title} not found. Use search().")

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

    def search(self, query, limit=10):
        q = query.lower()
        return [t for t in self._index_map if q in t.lower()][:limit]

    def top_rated(self, n=10):
        return (
            self.df[self.df["vote_count"] > 500]
            .sort_values("vote_average", ascending=False)
            .head(n)[["title", "vote_average", "vote_count"]]
        )