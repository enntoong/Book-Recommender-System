import random

import pandas as pd

from collaborative_filtering import (
    build_collaborative_model,
    get_collaborative_recommendations
)


# ==================================================
# Load data
# ==================================================

books = pd.read_csv(
    "books_data/books.csv",
    sep=";",
    encoding="latin-1",
    low_memory=False,
    on_bad_lines="skip"
)

ratings = pd.read_csv(
    "books_data/ratings.csv",
    sep=";",
    encoding="latin-1",
    low_memory=False,
    on_bad_lines="skip"
)


# ==================================================
# Clean columns
# ==================================================

books.columns = (
    books.columns
    .str.replace('"', "", regex=False)
    .str.strip()
)

ratings.columns = (
    ratings.columns
    .str.replace('"', "", regex=False)
    .str.strip()
)

books["ISBN"] = (
    books["ISBN"]
    .astype(str)
    .str.replace('"', "", regex=False)
    .str.strip()
)

ratings["ISBN"] = (
    ratings["ISBN"]
    .astype(str)
    .str.replace('"', "", regex=False)
    .str.strip()
)

books["Book-Title"] = (
    books["Book-Title"]
    .fillna("Unknown Title")
    .astype(str)
    .str.strip()
)

books["Book-Author"] = (
    books["Book-Author"]
    .fillna("Unknown Author")
    .astype(str)
    .str.strip()
)

ratings["Book-Rating"] = pd.to_numeric(
    ratings["Book-Rating"],
    errors="coerce"
)

ratings = ratings.dropna(
    subset=["Book-Rating"]
)


# ==================================================
# Create normalized Book-Key
# ==================================================

books["Normalized-Title"] = (
    books["Book-Title"]
    .str.casefold()
    .str.replace(
        r"[^a-z0-9]+",
        "",
        regex=True
    )
)

books["Normalized-Author"] = (
    books["Book-Author"]
    .str.casefold()
    .str.replace(
        r"[^a-z0-9]+",
        "",
        regex=True
    )
)

books["Book-Key"] = (
    books["Normalized-Title"]
    + "|"
    + books["Normalized-Author"]
)

display_book_data = books.drop_duplicates(
    subset=["Book-Key"]
).copy()


# ==================================================
# Build collaborative model
# ==================================================

book_user_matrix, sparse_matrix = (
    build_collaborative_model(
        books_data=books,
        ratings_data=ratings,
        minimum_user_ratings=20,
        minimum_book_ratings=10
    )
)


# ==================================================
# Prepare explicit user-book ratings
# ==================================================

explicit_ratings = ratings[
    ratings["Book-Rating"] > 0
].copy()

rating_with_books = explicit_ratings.merge(
    books[
        [
            "ISBN",
            "Book-Key",
            "Book-Title",
            "Book-Author"
        ]
    ],
    on="ISBN",
    how="inner"
)

# Combine ratings for different ISBN editions
rating_with_books = (
    rating_with_books
    .groupby(
        [
            "User-ID",
            "Book-Key"
        ],
        as_index=False
    )["Book-Rating"]
    .mean()
)

# Keep only books that are inside the model
rating_with_books = rating_with_books[
    rating_with_books["Book-Key"].isin(
        book_user_matrix.index
    )
].copy()


# ==================================================
# Keep positive ratings
# ==================================================

# Ratings of 7 or above are treated as liked books
liked_books = rating_with_books[
    rating_with_books["Book-Rating"] >= 7
].copy()

user_liked_books = (
    liked_books
    .groupby("User-ID")["Book-Key"]
    .apply(list)
)

# A user needs at least two liked books:
# one query book and one hidden target book
eligible_users = user_liked_books[
    user_liked_books.apply(len) >= 2
]


# ==================================================
# Leave-One-Out evaluation
# ==================================================

random.seed(42)

number_of_users_to_test = min(
    500,
    len(eligible_users)
)

sampled_users = random.sample(
    list(eligible_users.index),
    number_of_users_to_test
)

hits = 0
tested_cases = 0
top_k = 10

test_details = []

for user_id in sampled_users:

    user_books = eligible_users.loc[
        user_id
    ]

    # Use one liked book as the selected/query book
    selected_book_key = random.choice(
        user_books
    )

    # Hide another liked book as the expected result
    remaining_books = [
        book_key
        for book_key in user_books
        if book_key != selected_book_key
    ]

    hidden_book_key = random.choice(
        remaining_books
    )

    recommendations = (
        get_collaborative_recommendations(
            selected_book_key=selected_book_key,
            book_user_matrix=book_user_matrix,
            sparse_matrix=sparse_matrix,
            display_book_data=display_book_data,
            number_of_recommendations=top_k
        )
    )

    if recommendations.empty:
        continue

    recommended_keys = recommendations[
        "Book-Key"
    ].tolist()

    is_hit = (
        hidden_book_key in recommended_keys
    )

    if is_hit:
        hits += 1

    tested_cases += 1

    test_details.append(
        {
            "User-ID": user_id,
            "Selected-Book": selected_book_key,
            "Hidden-Book": hidden_book_key,
            "Hit": is_hit
        }
    )


# ==================================================
# Results
# ==================================================

if tested_cases == 0:

    print(
        "No valid cases were available for testing."
    )

else:

    hit_rate = hits / tested_cases

    print("\nCOLLABORATIVE FILTERING EVALUATION")
    print("----------------------------------")
    print(f"Tested cases: {tested_cases}")
    print(f"Successful hits: {hits}")
    print(f"Top-K: {top_k}")
    print(f"Hit Rate@{top_k}: {hit_rate:.4f}")
    print(
        f"Success percentage: "
        f"{hit_rate * 100:.2f}%"
    )


details = pd.DataFrame(
    test_details
)

details.to_csv(
    "collaborative_evaluation_results.csv",
    index=False
)

print(
    "\nDetailed results saved to "
    "collaborative_evaluation_results.csv"
)