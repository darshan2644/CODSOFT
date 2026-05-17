# Task 4 - Movie Recommendation System

Recommends movies to a user using two different approaches: **content-based filtering** (finds movies similar to one you like) and **collaborative filtering** (finds movies liked by similar users). All movie and user data is hardcoded — no external dataset needed.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Organizes movie and user data into tables (DataFrames) |
| `scikit-learn` | Computes cosine similarity to find similar movies or users |
| `numpy` | Handles numerical operations for similarity calculations |

---

## How to Run

**Step 1 — Install dependencies:**

```bash
pip install pandas scikit-learn numpy
```

**Step 2 — Run the script:**

```bash
python recommendation.py
```

---

## Sample Output

```
============================================================
        Movie Recommendation System
============================================================

Available movies:
 1. The Dark Knight
 2. Inception
 3. Interstellar
 4. The Matrix
 5. Avengers: Endgame
 6. Parasite
 7. The Godfather
 8. Pulp Fiction
... and more

------------------------------------------------------------

Enter a movie you like: Inception

--- Content-Based Recommendations (Similar Movies) ---

 1. Interstellar        | Sci-Fi, Drama    | ⭐ 8.6 | 2014
 2. The Matrix          | Sci-Fi, Action   | ⭐ 8.7 | 1999
 3. Shutter Island      | Thriller, Mystery| ⭐ 8.1 | 2010
 4. The Prestige        | Drama, Mystery   | ⭐ 8.5 | 2006
 5. Memento             | Thriller, Mystery| ⭐ 8.4 | 2000

------------------------------------------------------------

Enter your user ID to get collaborative recommendations (1-5): 3

--- Collaborative Filtering Recommendations (Based on Users Like You) ---

 1. The Godfather       | Crime, Drama     | ⭐ 9.2 | 1972
 2. Parasite            | Thriller, Drama  | ⭐ 8.5 | 2019
 3. Pulp Fiction        | Crime, Drama     | ⭐ 8.9 | 1994
 4. Goodfellas          | Crime, Drama     | ⭐ 8.7 | 1990
 5. Fight Club          | Drama, Thriller  | ⭐ 8.8 | 1999

============================================================
```
