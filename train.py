"""
train.py — Preprocess data and save model artifacts

Run once before launching the app:
    python train.py
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from recommender import MovieRecommender

# ---------- PATH SETTINGS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MOVIES_CSV = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
# -----------------------------------


def main():
    # Verify data files exist
    for path in [MOVIES_CSV, CREDITS_CSV]:
        if not os.path.exists(path):
            print(f"\n[✗] File not found: {path}")
            print(
                "\nPlease download the TMDB 5000 dataset from Kaggle:\n"
                "  https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata\n"
                "and place both CSVs inside the 'data/' folder.\n"
            )
            sys.exit(1)

    print("=" * 50)
    print("  Movie Recommendation System — Training")
    print("=" * 50)

    rec = MovieRecommender()

    print("\n[1/3] Loading & merging datasets...")
    print("[2/3] Building feature tags & similarity matrix...")
    rec.fit(MOVIES_CSV, CREDITS_CSV)

    # Create artifacts folder if missing
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("[3/3] Saving artifacts...")
    rec.save(ARTIFACTS_DIR)

    print("\n✅ Training complete!")
    print(f"   Artifacts saved to: {ARTIFACTS_DIR}/")
    print("\n   Now launch the app:")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    main()