import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_collaborative_model(
    books_data,
    ratings_data,
    minimum_user_ratings=20,
    minimum_book_ratings=10
):
    """
    Build the data required for item-based
    collaborative filtering.
    """

    # Only use explicit ratings from 1 to 10
    explicit_ratings = ratings_data[
        ratings_data["Book-Rating"] > 0
    ].copy()

    # Add book title, author and Book-Key
    # to the rating records
    rating_with_titles = explicit_ratings.merge(
        books_data[
            [
                "ISBN",
                "Book-Title",
                "Book-Author",
                "Book-Key"
            ]
        ],
        on="ISBN",
        how="inner"
    )

    # Keep active users
    user_rating_counts = (
        rating_with_titles
        .groupby("User-ID")["Book-Rating"]
        .count()
    )

    active_users = user_rating_counts[
        user_rating_counts
        >= minimum_user_ratings
    ].index

    filtered_ratings = rating_with_titles[
        rating_with_titles["User-ID"].isin(
            active_users
        )
    ].copy()

    # Keep books with enough ratings
    book_rating_counts = (
        filtered_ratings
        .groupby("Book-Key")["Book-Rating"]
        .count()
    )

    eligible_books = book_rating_counts[
        book_rating_counts
        >= minimum_book_ratings
    ].index

    filtered_ratings = filtered_ratings[
        filtered_ratings["Book-Key"].isin(
            eligible_books
        )
    ].copy()

    # Combine ratings for different ISBN editions
    # only when the title and author are the same
    filtered_ratings = (
        filtered_ratings
        .groupby(
            [
                "Book-Key",
                "User-ID"
            ],
            as_index=False
        )["Book-Rating"]
        .mean()
    )

    # Create the book-user rating matrix
    book_user_matrix = filtered_ratings.pivot(
        index="Book-Key",
        columns="User-ID",
        values="Book-Rating"
    ).fillna(0)

    sparse_matrix = csr_matrix(
        book_user_matrix.values
    )

    return book_user_matrix, sparse_matrix


def get_collaborative_recommendations(
    selected_book_key,
    book_user_matrix,
    sparse_matrix,
    display_book_data,
    number_of_recommendations=10
):
    """
    Recommend books with similar user rating patterns.
    """

    if selected_book_key not in book_user_matrix.index:
        return pd.DataFrame()

    selected_index = book_user_matrix.index.get_loc(
        selected_book_key
    )

    selected_vector = sparse_matrix[
        selected_index
    ]

    similarity_scores = cosine_similarity(
        selected_vector,
        sparse_matrix
    ).flatten()

    sorted_indices = similarity_scores.argsort()[
        ::-1
    ]

    recommendation_keys = []
    recommendation_scores = []

    for book_index in sorted_indices:

        book_key = book_user_matrix.index[
            book_index
        ]

        similarity_score = similarity_scores[
            book_index
        ]

        if book_key == selected_book_key:
            continue

        if similarity_score <= 0:
            continue

        recommendation_keys.append(
            book_key
        )

        recommendation_scores.append(
            similarity_score
        )

        if (
            len(recommendation_keys)
            >= number_of_recommendations
        ):
            break

    if not recommendation_keys:
        return pd.DataFrame()

    results = pd.DataFrame(
        {
            "Book-Key": recommendation_keys,
            "collaborative_score":
                recommendation_scores
        }
    )

    results = results.merge(
        display_book_data,
        on="Book-Key",
        how="left"
    )

    return results