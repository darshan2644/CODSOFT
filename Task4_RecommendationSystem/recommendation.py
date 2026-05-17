"""
Movie Recommendation System
===========================
Author  : Antigravity AI
Project : CODSOFT Internship - Task 4

INSTALLATION (run once in your terminal before using this script):
    pip install pandas scikit-learn

HOW THE TWO RECOMMENDATION APPROACHES WORK
-------------------------------------------

1. CONTENT-BASED FILTERING
   "Recommend movies similar to a movie the user already likes."
   - Each movie is described by its genre and plot keywords ("tags").
   - TF-IDF (Term Frequency–Inverse Document Frequency) converts those
     text tags into numeric vectors. Words that appear in many movies
     (e.g. "the") get downweighted; rare, distinctive words score higher.
   - Cosine Similarity measures the angle between two vectors.
     Score = 1  → identical content
     Score = 0  → completely unrelated
   - We rank all movies by their cosine similarity to the query movie
     and return the top N closest ones.

2. COLLABORATIVE FILTERING (user-based)
   "Recommend movies liked by users who share your taste."
   - We maintain a User × Movie ratings matrix (1–5 stars, 0 = not seen).
   - Cosine Similarity is computed between every pair of user rating vectors.
   - For the target user we find their most similar "neighbours".
   - Movies the neighbours rated highly but the target user hasn't seen yet
     are surfaced as recommendations — ranked by weighted average score.

HOW COSINE SIMILARITY WORKS
   Given two vectors A and B:
       cos(θ) = (A · B) / (||A|| × ||B||)
   It measures the cosine of the angle between them in n-dimensional space.
   Unlike Euclidean distance it is scale-invariant: a user who rates
   everything 5 is still "similar" to one who rates the same films 4.
"""

# ---------------------------------------------------------------------------
# INSTALLATION REMINDER
# pip install pandas scikit-learn
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# DATASET  (built-in — no external CSV required)
# ---------------------------------------------------------------------------

MOVIES_DATA = {
    "movie_id": list(range(1, 21)),
    "title": [
        "The Dark Knight",
        "Inception",
        "Interstellar",
        "The Matrix",
        "Avengers: Endgame",
        "The Shawshank Redemption",
        "Forrest Gump",
        "The Godfather",
        "Schindler's List",
        "Parasite",
        "Get Out",
        "A Quiet Place",
        "The Notebook",
        "Crazy Rich Asians",
        "La La Land",
        "Joker",
        "Knives Out",
        "Gone Girl",
        "Hereditary",
        "Everything Everywhere All at Once",
    ],
    "genre": [
        "Action",
        "Sci-Fi",
        "Sci-Fi",
        "Sci-Fi",
        "Action",
        "Drama",
        "Drama",
        "Drama",
        "Drama",
        "Thriller",
        "Horror",
        "Horror",
        "Romance",
        "Romance",
        "Romance",
        "Thriller",
        "Thriller",
        "Thriller",
        "Horror",
        "Sci-Fi",
    ],
    "description": [
        "Batman faces the Joker a criminal mastermind who plunges Gotham into chaos and anarchy",
        "A thief enters dreams to steal secrets and is given the task of planting an idea instead",
        "Astronauts travel through a wormhole near Saturn to find a new home for humanity",
        "A hacker discovers the world is a simulation and joins rebels to fight the machines",
        "The Avengers assemble one final time to reverse Thanos snap and save the universe",
        "Two prisoners develop a deep friendship over decades inside a corrupt Maine penitentiary",
        "A kind-hearted man from Alabama witnesses and influences decades of American history",
        "An aging patriarch transfers control of his crime empire to his reluctant son",
        "A German businessman risks everything to save Jewish refugees during the Holocaust",
        "A poor family schemes their way into working for a wealthy Korean household",
        "A young Black man visits his white girlfriend's parents and uncovers a disturbing secret",
        "A family is forced to live in near silence while hiding from creatures that hunt by sound",
        "A poor but passionate young man falls in love with a rich young woman in 1940s South Carolina",
        "A Chinese-American woman learns her boyfriend is incredibly wealthy and meets his family",
        "A jazz musician and an aspiring actress fall in love while pursuing their dreams in Los Angeles",
        "A failed comedian descends into madness and becomes the criminal clown of Gotham City",
        "A detective investigates the death of a crime novelist surrounded by his dysfunctional family",
        "A woman investigates the mysterious disappearance of her husband and a dark conspiracy unfolds",
        "A grieving family is haunted by a dark and terrifying ancestral secret after a grandmother dies",
        "A laundromat owner discovers she can access parallel universes and must save the multiverse",
    ],
    "rating": [9.0, 8.8, 8.6, 8.7, 8.4, 9.3, 8.8, 9.2, 8.9, 8.5,
               7.7, 7.5, 7.9, 7.0, 8.0, 8.4, 7.9, 8.1, 7.3, 7.8],
    "year":   [2008, 2010, 2014, 1999, 2019, 1994, 1994, 1972, 1993, 2019,
               2017, 2018, 2004, 2018, 2016, 2019, 2019, 2014, 2018, 2022],
}

# ---------------------------------------------------------------------------
# USER-MOVIE RATINGS MATRIX  (collaborative filtering data)
# Rows = users (User1 … User6), Columns = movie titles
# 0 = not watched, 1-5 = star rating
# ---------------------------------------------------------------------------

RATINGS_DATA = {
    #                          DK    Inc   Int   Mat   Avg   Shaw  Forr  God   Sch   Par
    "User1": [5, 4, 4, 5, 3,  0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   4, 0, 0, 0, 0],
    "User2": [0, 5, 5, 4, 0,  0, 0, 0, 0, 4,   0, 0, 0, 0, 0,   0, 0, 0, 0, 5],
    "User3": [0, 0, 0, 0, 0,  5, 5, 5, 5, 0,   0, 0, 3, 4, 0,   0, 0, 0, 0, 0],
    "User4": [0, 0, 0, 0, 0,  0, 0, 0, 0, 4,   5, 4, 0, 0, 0,   5, 4, 5, 4, 0],
    "User5": [3, 0, 0, 0, 4,  0, 4, 0, 0, 0,   0, 0, 5, 5, 5,   0, 0, 0, 0, 0],
    "User6": [4, 3, 0, 3, 5,  0, 0, 4, 0, 0,   0, 0, 0, 0, 0,   3, 5, 4, 0, 0],
}

# ---------------------------------------------------------------------------
# BUILD DATAFRAMES
# ---------------------------------------------------------------------------

def build_dataframes():
    """Construct the movies DataFrame and the ratings DataFrame."""
    movies_df = pd.DataFrame(MOVIES_DATA)

    # "tags" merges genre + description into one text blob for TF-IDF
    movies_df["tags"] = (
        movies_df["genre"].str.lower() + " " + movies_df["description"].str.lower()
    )

    ratings_df = pd.DataFrame(
        RATINGS_DATA,
        index=movies_df["title"]   # movie titles as row index
    ).T                             # transpose → users as rows, movies as columns

    return movies_df, ratings_df


# ---------------------------------------------------------------------------
# CONTENT-BASED FILTERING
# ---------------------------------------------------------------------------

def content_based_recommend(movie_title, df, num=5):
    """
    Find movies whose content (genre + description) is most similar
    to the given movie using TF-IDF vectors and cosine similarity.

    Parameters
    ----------
    movie_title : str  – title to base recommendations on
    df          : pd.DataFrame – the movies dataframe
    num         : int  – number of recommendations to return

    Returns
    -------
    pd.DataFrame of recommended movies, or None if title not found.
    """
    # ── Step 1: locate the movie (case-insensitive) ────────────────────────
    titles_lower = df["title"].str.lower()
    query        = movie_title.strip().lower()

    if query not in titles_lower.values:
        return None

    movie_index = titles_lower[titles_lower == query].index[0]

    # ── Step 2: build TF-IDF matrix from "tags" column ────────────────────
    # TfidfVectorizer converts each movie's tag string into a sparse vector.
    # stop_words="english" removes filler words like "the", "a", "is".
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["tags"])      # shape: (n_movies, vocab)

    # ── Step 3: compute cosine similarity of query vs all movies ──────────
    # We only need one row (the query movie), not the full n×n matrix.
    query_vec     = tfidf_matrix[movie_index]
    sim_scores    = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # ── Step 4: rank, exclude the query movie, return top N ───────────────
    ranked_indices = np.argsort(sim_scores)[::-1]       # descending order
    top_indices    = [i for i in ranked_indices if i != movie_index][:num]

    return df.iloc[top_indices][["title", "genre", "rating", "year"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# COLLABORATIVE FILTERING  (user-based)
# ---------------------------------------------------------------------------

def collaborative_recommend(user_id, ratings_df, movies_df, num=5):
    """
    Recommend movies for a user based on preferences of similar users.

    Algorithm
    ---------
    1. Compute cosine similarity between the target user's rating vector
       and every other user's rating vector.
    2. Identify the most similar neighbour(s).
    3. Collect movies the neighbours rated highly (≥ 4 stars) that the
       target user has NOT yet watched (rating == 0).
    4. Rank candidates by weighted-average score across neighbours.

    Parameters
    ----------
    user_id     : str  – e.g. "User1"
    ratings_df  : pd.DataFrame – users × movies ratings matrix
    movies_df   : pd.DataFrame – full movie info
    num         : int  – number of recommendations

    Returns
    -------
    pd.DataFrame of recommended movies, or None if user_id invalid.
    """
    if user_id not in ratings_df.index:
        return None

    # ── Step 1: user–user cosine similarity ───────────────────────────────
    sim_matrix = cosine_similarity(ratings_df.values)
    sim_df     = pd.DataFrame(
        sim_matrix,
        index   = ratings_df.index,
        columns = ratings_df.index,
    )

    # Similarities of ALL other users to the target user (excluding self)
    user_sims = sim_df[user_id].drop(user_id).sort_values(ascending=False)

    # ── Step 2: movies target user has NOT yet watched ────────────────────
    user_ratings  = ratings_df.loc[user_id]
    unwatched     = user_ratings[user_ratings == 0].index.tolist()

    if not unwatched:
        return pd.DataFrame()   # user has seen everything

    # ── Step 3: weighted score for each unwatched movie ───────────────────
    # score(movie) = Σ (similarity(neighbour) × neighbour_rating(movie))
    #                / Σ similarity(neighbour)   [only neighbours who watched]
    scores = {}
    for movie in unwatched:
        numerator   = 0.0
        denominator = 0.0
        for neighbour, sim in user_sims.items():
            neighbour_rating = ratings_df.loc[neighbour, movie]
            if neighbour_rating > 0 and sim > 0:   # neighbour watched it
                numerator   += sim * neighbour_rating
                denominator += sim
        if denominator > 0:
            scores[movie] = numerator / denominator

    if not scores:
        return pd.DataFrame()

    # ── Step 4: pick top N by score ───────────────────────────────────────
    top_movies = sorted(scores, key=scores.get, reverse=True)[:num]

    result = movies_df[movies_df["title"].isin(top_movies)][
        ["title", "genre", "rating", "year"]
    ].copy()

    # Add the predicted score column for transparency
    result["predicted_score"] = result["title"].map(
        lambda t: round(scores.get(t, 0), 2)
    )
    return result.sort_values("predicted_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------------------------

def print_banner():
    """Print the application header."""
    print("\n" + "=" * 54)
    print("   🎬  MOVIE RECOMMENDATION SYSTEM")
    print("        CodSoft Task 4")
    print("=" * 54)
    print("  Two approaches: Content-Based & Collaborative")
    print("=" * 54)


def print_menu(user_ids):
    """Display the main option menu."""
    users = ", ".join(user_ids)
    print("\n  ┌──────────────────────────────────────────────┐")
    print("  │  Options:                                    │")
    print("  │  1. Content-based recommendation (by title) │")
    print("  │  2. Collaborative filtering  (by user ID)   │")
    print("  │  3. Show all available movies               │")
    print("  │  4. Exit                                    │")
    print("  └──────────────────────────────────────────────┘")
    print(f"  Available users : {users}")


def print_recommendations(recs_df, heading="  🎬  Recommendations"):
    """Pretty-print a recommendations DataFrame."""
    if recs_df is None or recs_df.empty:
        print("  ℹ   No recommendations could be generated.")
        return

    print(f"\n{heading}")
    print("  " + "─" * 60)

    for i, row in recs_df.iterrows():
        score_str = ""
        if "predicted_score" in recs_df.columns:
            score_str = f"  [pred: {row['predicted_score']}]"
        print(
            f"  {i + 1}. {row['title']:<40}"
            f"  {row['genre']:<10}"
            f"  ⭐ {row['rating']}"
            f"  ({int(row['year'])})"
            f"{score_str}"
        )

    print("  " + "─" * 60)


def print_all_movies(df):
    """Print the complete catalogue."""
    print("\n  🎞   All Available Movies")
    print("  " + "─" * 66)
    print(f"  {'#':<4} {'Title':<44} {'Genre':<10} {'⭐':<5} {'Year'}")
    print("  " + "─" * 66)
    for _, row in df.iterrows():
        print(
            f"  {row['movie_id']:<4} {row['title']:<44}"
            f" {row['genre']:<10} {row['rating']:<5} {int(row['year'])}"
        )
    print("  " + "─" * 66)


# ---------------------------------------------------------------------------
# MAIN APPLICATION LOOP
# ---------------------------------------------------------------------------

def main():
    print_banner()

    # Build dataframes once at startup
    movies_df, ratings_df = build_dataframes()
    user_ids = list(ratings_df.index)

    while True:
        print_menu(user_ids)
        choice = input("\n  Enter option (1-4): ").strip()

        # ── Option 1: Content-Based ────────────────────────────────────────
        if choice == "1":
            print("\n  Enter a movie title you enjoy.")
            print("  Tip: type '3' first to see the full movie list.\n")
            movie_title = input("  Movie title : ").strip()

            if not movie_title:
                print("  ⚠   No title entered.")
                continue

            recs = content_based_recommend(movie_title, movies_df, num=5)

            if recs is None:
                print(f"\n  ❌  Movie \"{movie_title}\" not found in the database.")
                print("  ℹ   Use option 3 to see all available titles.")
            else:
                print(f"\n  Because you liked  →  \"{movie_title.title()}\"")
                print_recommendations(recs, "  🎬  Content-Based Recommendations")

        # ── Option 2: Collaborative Filtering ─────────────────────────────
        elif choice == "2":
            user_id = input(f"  Enter user ID {user_ids} : ").strip()

            if user_id not in user_ids:
                print(f"  ❌  User \"{user_id}\" not found.")
                print(f"  ℹ   Valid IDs: {', '.join(user_ids)}")
                continue

            recs = collaborative_recommend(user_id, ratings_df, movies_df, num=5)

            if recs is None:
                print(f"  ❌  User \"{user_id}\" not found.")
            elif recs.empty:
                print(f"  ℹ   No new recommendations for {user_id} — already seen everything!")
            else:
                print(f"\n  Recommendations for  →  {user_id}")
                print_recommendations(recs, "  🤝  Collaborative Recommendations")

        # ── Option 3: Show catalogue ───────────────────────────────────────
        elif choice == "3":
            print_all_movies(movies_df)

        # ── Option 4: Exit ─────────────────────────────────────────────────
        elif choice == "4":
            print("\n  👋  Goodbye! Enjoy your movies. 🎬\n")
            break

        else:
            print("  ⚠   Invalid option. Please enter 1, 2, 3, or 4.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
