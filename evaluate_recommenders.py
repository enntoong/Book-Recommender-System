"""Reproducible held-out-user evaluation for the Lumen recommendation approaches.

Evaluation design
-----------------
The evaluation selects active readers who have several books rated 8/10 or above.
All ratings from the selected evaluation readers are removed before the
collaborative model and training-derived popularity statistics are built.

For each held-out reader:
1. One known positive book is used only as the recommendation seed.
2. The reader's other highly rated books form the unseen relevance set.
3. Collaborative, content-based, and hybrid Top-10 lists are compared against
   that relevance set.

This held-out-user protocol avoids direct leakage from the evaluated readers'
ratings into collaborative model construction. It is still an offline proxy:
historical high ratings are treated as relevance labels, not proof that a reader
would accept a recommendation in a live product.

Metrics
-------
Precision@10, Recall@10, F1@10, Hit Rate@10, Recommendation Availability,
Catalog Coverage@10, and mean recommendation calculation time.
"""
from __future__ import annotations

from collections import defaultdict
import time

import numpy as np
import pandas as pd

from collaborative_filtering import build_collaborative_model, get_collaborative_recommendations
from content_based_filtering import build_content_model, get_content_recommendations
from hybrid_filtering import get_hybrid_recommendations
from data_service import display_books, load_catalogue, load_explicit_ratings

K = 10
SAMPLE_USERS = 50
RANDOM_STATE = 42
MINIMUM_USER_RATINGS = 20
MINIMUM_BOOK_RATINGS = 10
MINIMUM_CONTENT_RATINGS = 2
POSITIVE_THRESHOLD = 8


def score_at_k(recommended_keys, relevant_keys, k=K):
    recommended = list(recommended_keys)[:k]
    relevant = set(relevant_keys)
    hits = len(set(recommended) & relevant)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, 1.0 if hits else 0.0


def _book_stats_from_ratings(ratings: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    enriched = ratings.merge(books[["ISBN", "Book-Key"]], on="ISBN", how="inner")
    stats = (
        enriched.groupby("Book-Key")
        .agg(
            num_ratings=("Book-Rating", "count"),
            avg_rating=("Book-Rating", "mean"),
        )
        .reset_index()
    )
    stats["num_ratings"] = stats["num_ratings"].astype(int)
    return stats


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    books = load_catalogue()
    ratings = load_explicit_ratings()
    display = display_books()

    enriched = ratings.merge(books[["ISBN", "Book-Key"]], on="ISBN", how="inner")
    user_rating_count = enriched.groupby("User-ID").size()

    positive = enriched[
        (enriched["Book-Rating"] >= POSITIVE_THRESHOLD)
        & enriched["User-ID"].isin(user_rating_count[user_rating_count >= MINIMUM_USER_RATINGS].index)
    ].drop_duplicates(["User-ID", "Book-Key"])

    user_positive = positive.groupby("User-ID")["Book-Key"].apply(list)
    eligible_users = user_positive[user_positive.map(len) >= 6].index.to_numpy()
    if len(eligible_users) == 0:
        raise RuntimeError("No suitable evaluation users found.")

    chosen_users = rng.choice(
        eligible_users,
        size=min(SAMPLE_USERS, len(eligible_users)),
        replace=False,
    )
    chosen_users = np.asarray(chosen_users)

    # Strictly remove every explicit rating from the sampled readers before
    # constructing collaborative training data and training-derived statistics.
    training_ratings = ratings[~ratings["User-ID"].isin(chosen_users)].copy()
    training_stats = _book_stats_from_ratings(training_ratings, books)
    training_popularity = dict(zip(training_stats["Book-Key"], training_stats["num_ratings"]))

    content_candidates = display.merge(
        training_stats[["Book-Key", "num_ratings"]],
        on="Book-Key",
        how="left",
    )
    content_candidates["num_ratings"] = content_candidates["num_ratings"].fillna(0).astype(int)
    content_candidates = content_candidates[
        content_candidates["num_ratings"] >= MINIMUM_CONTENT_RATINGS
    ].copy()
    if content_candidates.empty:
        content_candidates = display.copy()

    print(f"Evaluation users held out: {len(chosen_users)}")
    print(f"Training explicit ratings: {len(training_ratings):,}")
    print("Building held-out evaluation models...")

    book_user_matrix, sparse_matrix = build_collaborative_model(
        books_data=books,
        ratings_data=training_ratings,
        minimum_user_ratings=MINIMUM_USER_RATINGS,
        minimum_book_ratings=MINIMUM_BOOK_RATINGS,
    )
    content_index, vectorizer, content_matrix, key_to_position = build_content_model(
        content_candidates,
        max_features=40000,
    )
    display_lookup = display.set_index("Book-Key")

    catalogue_denominators = {
        "Collaborative Filtering": max(1, len(book_user_matrix.index)),
        "Content-Based Filtering": max(1, len(content_index)),
        "Hybrid Filtering": max(
            1, len(set(book_user_matrix.index).union(set(content_index["Book-Key"])))
        ),
    }

    accum = defaultdict(
        lambda: {
            "precision": [],
            "recall": [],
            "f1": [],
            "hit": [],
            "time_ms": [],
            "returned": [],
            "unique_recommended": set(),
        }
    )

    evaluated_cases = 0
    for user_id in chosen_users:
        positive_keys = list(dict.fromkeys(user_positive.loc[user_id]))
        # Select a seed with stronger evidence in the training population when possible.
        positive_keys.sort(key=lambda key: training_popularity.get(key, 0), reverse=True)
        seed = positive_keys[0]
        relevant = set(positive_keys[1:])

        if seed not in display_lookup.index or not relevant:
            continue

        selected_row = display_lookup.loc[seed].to_dict()
        evaluated_cases += 1

        start = time.perf_counter()
        collab_wide = get_collaborative_recommendations(
            selected_book_key=seed,
            book_user_matrix=book_user_matrix,
            sparse_matrix=sparse_matrix,
            display_book_data=display,
            number_of_recommendations=60,
        )
        collab_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        content_wide = get_content_recommendations(
            selected_book_key=seed,
            content_index=content_index,
            content_matrix=content_matrix,
            key_to_position=key_to_position,
            number_of_recommendations=60,
            vectorizer=vectorizer,
            selected_book_row=selected_row,
        )
        content_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        hybrid = get_hybrid_recommendations(
            collaborative_results=collab_wide,
            content_results=content_wide,
            rating_statistics=training_stats,
            collaborative_weight=0.55,
            number_of_recommendations=K,
        )
        hybrid_fusion_ms = (time.perf_counter() - start) * 1000

        model_outputs = {
            "Collaborative Filtering": (collab_wide.head(K), collab_ms),
            "Content-Based Filtering": (content_wide.head(K), content_ms),
            "Hybrid Filtering": (hybrid, collab_ms + content_ms + hybrid_fusion_ms),
        }

        for model_name, (result, elapsed) in model_outputs.items():
            rec_keys = result["Book-Key"].tolist() if result is not None and not result.empty else []
            precision, recall, f1, hit = score_at_k(rec_keys, relevant, K)
            accum[model_name]["precision"].append(precision)
            accum[model_name]["recall"].append(recall)
            accum[model_name]["f1"].append(f1)
            accum[model_name]["hit"].append(hit)
            accum[model_name]["time_ms"].append(elapsed)
            accum[model_name]["returned"].append(1.0 if rec_keys else 0.0)
            accum[model_name]["unique_recommended"].update(rec_keys)

    if evaluated_cases == 0:
        raise RuntimeError("No evaluation cases could be scored.")

    rows = []
    for model_name, values in accum.items():
        denominator = catalogue_denominators[model_name]
        rows.append(
            {
                "Model": model_name,
                "Precision@10": np.mean(values["precision"]),
                "Recall@10": np.mean(values["recall"]),
                "F1@10": np.mean(values["f1"]),
                "Hit Rate@10": np.mean(values["hit"]),
                "Recommendation Availability": np.mean(values["returned"]),
                "Catalog Coverage@10": len(values["unique_recommended"]) / denominator,
                "Unique Recommended Books": len(values["unique_recommended"]),
                "Recommendable Catalog Size": denominator,
                "Mean Recommendation Time (ms)": np.mean(values["time_ms"]),
                "Test Cases": len(values["precision"]),
            }
        )

    results = pd.DataFrame(rows).sort_values("F1@10", ascending=False)
    results.to_csv("recommender_evaluation_results.csv", index=False)

    print("\nHeld-out-user offline evaluation")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved: recommender_evaluation_results.csv")


if __name__ == "__main__":
    main()
