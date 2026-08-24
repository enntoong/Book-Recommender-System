import re
import html
from difflib import SequenceMatcher
import numpy as np
import pandas as pd


def _canonical_title(value):
    text = html.unescape(str(value or "")).casefold()
    text = re.sub(r"\([^)]*\)", " ", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    removable = {"a", "an", "the", "and", "novel", "paperback", "hardcover", "edition", "book"}
    return "".join(t for t in tokens if t not in removable)


def _diversify_ranked_books(data, limit):
    kept = []
    signatures = []
    for _, row in data.iterrows():
        title = str(row.get("Book-Title", ""))
        author = str(row.get("Book-Author", "")).casefold()
        canon = _canonical_title(title)
        duplicate = False
        for existing_canon, existing_author in signatures:
            if author == existing_author and canon and existing_canon:
                if canon in existing_canon or existing_canon in canon or SequenceMatcher(None, canon, existing_canon).ratio() >= 0.88:
                    duplicate = True
                    break
        if duplicate:
            continue
        kept.append(row)
        signatures.append((canon, author))
        if len(kept) >= limit:
            break
    return pd.DataFrame(kept).reset_index(drop=True) if kept else pd.DataFrame(columns=data.columns)


def _normalize(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if s.empty or float(s.max()) == float(s.min()):
        return s.clip(0, 1)
    return (s - s.min()) / (s.max() - s.min())


def get_hybrid_recommendations(
    collaborative_results,
    content_results,
    rating_statistics=None,
    collaborative_weight=0.55,
    number_of_recommendations=20,
):
    """Fuse collaborative and content recommendations into one ranked list."""
    collaborative_weight = float(np.clip(collaborative_weight, 0.0, 1.0))
    content_weight = 1.0 - collaborative_weight

    collab = collaborative_results.copy() if collaborative_results is not None else pd.DataFrame()
    content = content_results.copy() if content_results is not None else pd.DataFrame()

    if collab.empty and content.empty:
        return pd.DataFrame()
    if collab.empty:
        result = content.copy()
        result["collaborative_score"] = 0.0
        result["hybrid_score"] = _normalize(result.get("content_score", 0.0))
        result["recommendation_reason"] = result.get("recommendation_reason", "Strong content match")
        return _diversify_ranked_books(result.sort_values("hybrid_score", ascending=False), number_of_recommendations)
    if content.empty:
        result = collab.copy()
        result["content_score"] = 0.0
        result["hybrid_score"] = _normalize(result.get("collaborative_score", 0.0))
        result["recommendation_reason"] = "Readers with similar tastes also liked this"
        return _diversify_ranked_books(result.sort_values("hybrid_score", ascending=False), number_of_recommendations)

    wanted_meta = ["Book-Key", "ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-M"]
    collab_meta = collab[[c for c in wanted_meta if c in collab.columns]].copy()
    content_meta = content[[c for c in wanted_meta if c in content.columns]].copy()
    meta = pd.concat([collab_meta, content_meta], ignore_index=True).drop_duplicates("Book-Key")

    c1 = collab[["Book-Key", "collaborative_score"]].copy()
    c2 = content[["Book-Key", "content_score", "recommendation_reason"]].copy()
    result = c1.merge(c2, on="Book-Key", how="outer")
    result["collaborative_score"] = result["collaborative_score"].fillna(0.0)
    result["content_score"] = result["content_score"].fillna(0.0)
    result["collaborative_component"] = _normalize(result["collaborative_score"])
    result["content_component"] = _normalize(result["content_score"])
    result["hybrid_score"] = (
        collaborative_weight * result["collaborative_component"]
        + content_weight * result["content_component"]
    )

    # Popularity is used only as a tie-breaker, not as a third recommendation algorithm.
    if rating_statistics is not None and not rating_statistics.empty:
        if "Book-Key" in rating_statistics.columns:
            stats = rating_statistics[["Book-Key", "num_ratings", "avg_rating"]].drop_duplicates("Book-Key")
            result = result.merge(stats, on="Book-Key", how="left")
        elif "Book-Title" in rating_statistics.columns:
            title_lookup = pd.concat([
                collab[[c for c in collab.columns if c in ["Book-Key", "Book-Title"]]],
                meta[[c for c in meta.columns if c in ["Book-Key", "Book-Title"]]],
            ]).drop_duplicates("Book-Key")
            result = result.merge(title_lookup, on="Book-Key", how="left")
            stats = rating_statistics[["Book-Title", "num_ratings", "avg_rating"]].drop_duplicates("Book-Title")
            result = result.merge(stats, on="Book-Title", how="left")
        else:
            result["num_ratings"] = 0
            result["avg_rating"] = 0.0
        result["popularity_tiebreak"] = np.log1p(result["num_ratings"].fillna(0))
    else:
        result["popularity_tiebreak"] = 0.0

    result = result.merge(meta.drop(columns=["Book-Title"], errors="ignore"), on="Book-Key", how="left") if "Book-Title" in result else result.merge(meta, on="Book-Key", how="left")

    def reason(row):
        has_c = row.get("collaborative_score", 0) > 0
        has_t = row.get("content_score", 0) > 0
        if has_c and has_t:
            return "Strong reader-behaviour and content match"
        if has_c:
            return "Readers with similar tastes also liked this"
        return row.get("recommendation_reason") or "Strong content match"

    result["recommendation_reason"] = result.apply(reason, axis=1)
    ranked = result.sort_values(["hybrid_score", "popularity_tiebreak"], ascending=[False, False])
    return _diversify_ranked_books(ranked, number_of_recommendations)
