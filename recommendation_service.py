from __future__ import annotations

from functools import lru_cache
import html
import re
from difflib import SequenceMatcher

import pandas as pd

from collaborative_filtering import build_collaborative_model, get_collaborative_recommendations
from content_based_filtering import build_content_model, get_content_recommendations
from hybrid_filtering import get_hybrid_recommendations
from user_store import MAX_TASTE_SIGNALS
from data_service import (
    book_by_key,
    catalogue_with_stats,
    content_candidate_catalogue,
    display_books,
    load_catalogue,
    load_explicit_ratings,
    rating_statistics,
)


@lru_cache(maxsize=1)
def collaborative_model():
    return build_collaborative_model(
        books_data=load_catalogue(),
        ratings_data=load_explicit_ratings(),
        minimum_user_ratings=20,
        minimum_book_ratings=10,
    )


@lru_cache(maxsize=1)
def content_model():
    candidates = content_candidate_catalogue()
    return build_content_model(candidates, max_features=40000)



def _canonical_title(value: str) -> str:
    text = html.unescape(str(value or "")).casefold()
    text = re.sub(r"\([^)]*\)", " ", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    removable = {"a", "an", "the", "and", "novel", "paperback", "hardcover", "edition", "book"}
    return "".join(token for token in tokens if token not in removable)


def _diversify_personalised(ranked: list[dict], limit: int) -> list[dict]:
    """Avoid showing near-duplicate editions of the same title in one shelf."""
    selected = []
    signatures = []
    for item in ranked:
        row = item["row"]
        canon = _canonical_title(row.get("Book-Title", ""))
        author = str(row.get("Book-Author", "")).casefold()
        duplicate = False
        for existing_canon, existing_author in signatures:
            if author == existing_author and canon and existing_canon:
                similar = (
                    canon in existing_canon
                    or existing_canon in canon
                    or SequenceMatcher(None, canon, existing_canon).ratio() >= 0.88
                )
                if similar:
                    duplicate = True
                    break
        if duplicate:
            continue
        selected.append(item)
        signatures.append((canon, author))
        if len(selected) >= limit:
            break
    return selected

def _reader_reason(row: pd.Series) -> str:
    c = float(row.get("collaborative_score", 0) or 0)
    t = float(row.get("content_score", 0) or 0)
    raw = str(row.get("recommendation_reason", "")).casefold()
    if c > 0 and t > 0:
        return "Readers with similar tastes also enjoyed books like this"
    if c > 0:
        return "Popular with readers who enjoyed similar books"
    if "same author" in raw:
        return "More from an author connected to this read"
    if "same publisher" in raw:
        return "A similar publishing profile and reading style"
    if "publication period" in raw:
        return "A similar read from the same publication period"
    return "Similar in title, author and publishing profile"


def smart_recommendations(selected_book_key: str, limit: int = 16) -> list[dict]:
    selected = book_by_key(selected_book_key)
    if not selected:
        return []

    book_user_matrix, sparse_matrix = collaborative_model()
    collaborative = get_collaborative_recommendations(
        selected_book_key=selected_book_key,
        book_user_matrix=book_user_matrix,
        sparse_matrix=sparse_matrix,
        display_book_data=display_books(),
        number_of_recommendations=max(50, limit * 4),
    )

    content_index, vectorizer, matrix, key_to_position = content_model()
    content = get_content_recommendations(
        selected_book_key=selected_book_key,
        content_index=content_index,
        content_matrix=matrix,
        key_to_position=key_to_position,
        number_of_recommendations=max(50, limit * 4),
        vectorizer=vectorizer,
        selected_book_row=selected,
    )

    stats_for_hybrid = catalogue_with_stats()[
        ["Book-Key", "Book-Title", "num_ratings", "avg_rating"]
    ].drop_duplicates("Book-Key")

    hybrid = get_hybrid_recommendations(
        collaborative_results=collaborative,
        content_results=content,
        rating_statistics=stats_for_hybrid,
        collaborative_weight=0.55,
        number_of_recommendations=limit,
    )
    if hybrid.empty:
        return []

    hybrid = hybrid.merge(
        catalogue_with_stats()[["Book-Key", "num_ratings", "avg_rating", "ISBN"]],
        on="Book-Key",
        how="left",
        suffixes=("", "_catalogue"),
    )
    if "ISBN_catalogue" in hybrid.columns:
        hybrid["ISBN"] = hybrid["ISBN"].fillna(hybrid["ISBN_catalogue"])
    hybrid["recommendation_reason"] = hybrid.apply(_reader_reason, axis=1)
    hybrid = hybrid.drop_duplicates("Book-Key").head(limit).copy()

    # Reader-facing Match is always a relative shelf score: the strongest
    # recommendation in the current shelf is 99 and the rest are scaled to it.
    # It is a ranking aid, not a probability or accuracy percentage.
    max_score = float(hybrid["hybrid_score"].fillna(0).max()) if not hybrid.empty else 0.0
    if max_score > 0:
        hybrid["match_score"] = (
            (hybrid["hybrid_score"].fillna(0).clip(lower=0) / max_score) * 99
        ).round().astype(int).clip(lower=1, upper=99)
    else:
        hybrid["match_score"] = 1
    return hybrid.to_dict("records")


def personalised_recommendations(seed_keys: list[str], limit: int = 20) -> list[dict]:
    seeds = list(dict.fromkeys(k for k in seed_keys if k))[:MAX_TASTE_SIGNALS]
    if not seeds:
        return []

    combined = {}
    for seed_key in seeds:
        seed = book_by_key(seed_key)
        if not seed:
            continue
        recs = smart_recommendations(seed_key, limit=30)
        for rank, row in enumerate(recs, start=1):
            key = str(row.get("Book-Key", ""))
            if not key or key in seeds:
                continue
            base = float(row.get("hybrid_score", 0) or 0)
            rank_discount = 1.0 / (1.0 + 0.08 * (rank - 1))
            score = base * rank_discount
            current = combined.setdefault(key, {"row": row.copy(), "score": 0.0, "seeds": []})
            current["score"] += score
            title = seed["Book-Title"]
            if title not in current["seeds"]:
                current["seeds"].append(title)

    ranked = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
    selected = _diversify_personalised(ranked, limit)
    if not selected:
        return []

    max_personal_score = max(float(item["score"]) for item in selected) or 1.0
    output = []
    previous_match_score = 99
    for item in selected:
        row = item["row"].copy()
        row["personal_score"] = float(item["score"])
        seed_titles = item["seeds"][:2]
        if len(seed_titles) == 1:
            row["recommendation_reason"] = f"Inspired by your interest in {seed_titles[0]}"
        elif seed_titles:
            row["recommendation_reason"] = f"Matches your taste from {seed_titles[0]} and {seed_titles[1]}"

        # Same definition used by related-book shelves: relative to the strongest
        # recommendation in this current shelf, never a probability.
        relative = int(round((row["personal_score"] / max_personal_score) * 99))
        row["match_score"] = max(1, min(previous_match_score, relative))
        previous_match_score = row["match_score"]
        output.append(row)
    return output
