from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import html

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "books_data"
VALID_PUBLICATION_YEAR_MIN = 1900
# The catalogue's year distribution ends overwhelmingly by 2005; the tiny
# number of later values are treated as metadata outliers for this historical dataset.
VALID_PUBLICATION_YEAR_MAX = 2005


def _clean_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data.columns = data.columns.str.replace('"', '', regex=False).str.strip()
    return data


def _normalise_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.casefold()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )


@lru_cache(maxsize=1)
def load_catalogue() -> pd.DataFrame:
    """Load and clean the book catalogue once per application process."""
    books = pd.read_csv(
        DATA_DIR / "books.csv",
        sep=";",
        encoding="latin-1",
        low_memory=False,
        on_bad_lines="skip",
    )
    books = _clean_columns(books)

    keep = [
        "ISBN",
        "Book-Title",
        "Book-Author",
        "Year-Of-Publication",
        "Publisher",
        "Image-URL-S",
        "Image-URL-M",
        "Image-URL-L",
    ]
    books = books[[c for c in keep if c in books.columns]].copy()

    books["ISBN"] = books["ISBN"].astype(str).str.replace('"', '', regex=False).str.strip()
    books["Book-Title"] = books["Book-Title"].fillna("Unknown title").astype(str).str.strip().map(html.unescape)
    books["Book-Author"] = books["Book-Author"].fillna("Unknown author").astype(str).str.strip().map(html.unescape)
    books["Publisher"] = books["Publisher"].fillna("Unknown publisher").astype(str).str.strip().map(html.unescape)
    books["Image-URL-M"] = books.get("Image-URL-M", "").fillna("").astype(str).str.strip()

    books = books[(books["ISBN"] != "") & (books["Book-Title"] != "")].copy()
    books["Normalized-Title"] = _normalise_text(books["Book-Title"])
    books["Normalized-Author"] = _normalise_text(books["Book-Author"])
    books["Book-Key"] = books["Normalized-Title"] + "|" + books["Normalized-Author"]

    year = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
    books["year_numeric"] = year.where(
        year.between(VALID_PUBLICATION_YEAR_MIN, VALID_PUBLICATION_YEAR_MAX)
    )
    return books.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_all_ratings() -> pd.DataFrame:
    """Load and clean all rating events, including zero-valued implicit interactions."""
    ratings = pd.read_csv(
        DATA_DIR / "ratings.csv",
        sep=";",
        encoding="latin-1",
        low_memory=False,
        on_bad_lines="skip",
        usecols=lambda c: c.replace('"', '').strip() in {"User-ID", "ISBN", "Book-Rating"},
    )
    ratings = _clean_columns(ratings)
    ratings["ISBN"] = ratings["ISBN"].astype(str).str.replace('"', '', regex=False).str.strip()
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    ratings["User-ID"] = pd.to_numeric(ratings["User-ID"], errors="coerce")
    ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    ratings["User-ID"] = ratings["User-ID"].astype("int32")
    ratings["Book-Rating"] = ratings["Book-Rating"].astype("int8")
    return ratings.reset_index(drop=True)


@lru_cache(maxsize=1)
def load_explicit_ratings() -> pd.DataFrame:
    """Ratings used by the recommender. Zero-valued implicit events are excluded."""
    ratings = load_all_ratings()
    return ratings[ratings["Book-Rating"] > 0].reset_index(drop=True).copy()


@lru_cache(maxsize=1)
def display_books() -> pd.DataFrame:
    """One representative ISBN for each title+author book identity."""
    return load_catalogue().drop_duplicates("Book-Key").reset_index(drop=True).copy()


@lru_cache(maxsize=1)
def explicit_ratings_with_book_keys() -> pd.DataFrame:
    """Explicit rating events joined to Lumen's title+author book identity."""
    books = load_catalogue()[["ISBN", "Book-Key"]]
    ratings = load_explicit_ratings()[["ISBN", "Book-Rating"]]
    return ratings.merge(books, on="ISBN", how="inner")[["Book-Key", "Book-Rating"]]


@lru_cache(maxsize=1)
def rating_statistics() -> pd.DataFrame:
    joined = explicit_ratings_with_book_keys()
    stats = (
        joined.groupby("Book-Key")
        .agg(
            num_ratings=("Book-Rating", "count"),
            avg_rating=("Book-Rating", "mean"),
        )
        .reset_index()
    )
    stats["num_ratings"] = stats["num_ratings"].astype(int)
    return stats


@lru_cache(maxsize=1)
def rating_bucket_statistics() -> pd.DataFrame:
    """Aggregate real Book-Crossing ratings without losing the 1-10 scale.

    Book-Crossing explicit ratings are integers from 1 to 10. Lumen keeps
    this original scale unchanged in both the recommender logic and the
    reader-facing interface. Matching editions of the same title and author
    are combined, consistent with ``rating_statistics``.
    """
    ratings = explicit_ratings_with_book_keys().copy()
    if ratings.empty:
        return pd.DataFrame(columns=["Book-Key", "score10", "count"])

    ratings["score10"] = ratings["Book-Rating"].astype(int).clip(1, 10).astype("int8")
    return (
        ratings.groupby(["Book-Key", "score10"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )


def _rating_label(score10: int) -> str:
    return f"{score10}/10"


@lru_cache(maxsize=512)
def book_rating_distribution(book_key: str) -> list[dict]:
    """Return the ten original source rating rows (10/10 down to 1/10)."""
    grouped = rating_bucket_statistics()
    rows = grouped[grouped["Book-Key"] == str(book_key)]
    counts = dict(zip(rows["score10"].astype(int), rows["count"].astype(int)))
    total = sum(counts.values())
    return [
        {
            "score10": score10,
            "label": _rating_label(score10),
            "count": counts.get(score10, 0),
            "percent": (counts.get(score10, 0) / total * 100.0) if total else 0.0,
        }
        for score10 in range(10, 0, -1)
    ]


@lru_cache(maxsize=1)
def catalogue_with_stats() -> pd.DataFrame:
    catalogue = display_books().merge(rating_statistics(), on="Book-Key", how="left")
    catalogue["num_ratings"] = catalogue["num_ratings"].fillna(0).astype(int)
    catalogue["avg_rating"] = catalogue["avg_rating"].fillna(0.0).astype(float)
    catalogue["search_text"] = (
        catalogue["Book-Title"].fillna("").astype(str)
        + " "
        + catalogue["Book-Author"].fillna("").astype(str)
        + " "
        + catalogue["Publisher"].fillna("").astype(str)
        + " "
        + catalogue["ISBN"].fillna("").astype(str)
    ).str.casefold()
    return catalogue


@lru_cache(maxsize=1)
def content_candidate_catalogue() -> pd.DataFrame:
    """Candidate set for fast content retrieval.

    Books with at least two explicit ratings remain broad enough for discovery while
    reducing the live TF-IDF model from ~245k titles to ~47k candidates.
    """
    catalogue = catalogue_with_stats()
    candidates = catalogue[catalogue["num_ratings"] >= 2].copy()
    if candidates.empty:
        candidates = catalogue.copy()
    return candidates.reset_index(drop=True)



@lru_cache(maxsize=1)
def load_users() -> pd.DataFrame:
    """Load the user IDs used by the Book-Crossing dataset."""
    users = pd.read_csv(
        DATA_DIR / "users.csv",
        sep=";",
        encoding="latin-1",
        low_memory=False,
        on_bad_lines="skip",
        usecols=lambda c: c.replace('"', '').strip() in {"User-ID"},
    )
    users = _clean_columns(users)
    users["User-ID"] = pd.to_numeric(users["User-ID"], errors="coerce")
    return users.dropna(subset=["User-ID"]).drop_duplicates("User-ID").reset_index(drop=True)


def _estimated_category_summary(catalogue: pd.DataFrame) -> dict:
    """Estimate broad book categories from title keywords.

    The source dataset has no genre field. The chart therefore reports only
    keyword-classified titles and separately discloses how many titles could
    not be classified by these broad rules.
    """
    titles = catalogue["Book-Title"].fillna("").astype(str).str.casefold()
    remaining = pd.Series(True, index=catalogue.index)
    rules = [
        ("Romance", r"\b(?:love|romance|bride|wedding|heart|kiss|husband|wife|lover|passion)\b"),
        ("History & Biography", r"\b(?:history|biography|memoir|life of|war|president|queen|king|historical)\b"),
        ("Mystery & Crime", r"\b(?:mystery|murder|detective|crime|killer|thriller|sherlock|police|investigation)\b"),
        ("Children & YA", r"\b(?:children|kids|child|baby|teen|young adult|school|nursery|bedtime)\b"),
        ("Fantasy", r"\b(?:fantasy|magic|dragon|wizard|witch|kingdom|sword|fairy|vampire|elf|myth)\b"),
        ("Cooking & Lifestyle", r"\b(?:cook|cooking|recipe|kitchen|food|garden|home|craft|diet|health|fitness)\b"),
        ("Science Fiction", r"\b(?:science fiction|sci[- ]?fi|space|alien|robot|galaxy|future|cyber|planet)\b"),
        ("Business & Self-help", r"\b(?:business|success|leadership|management|money|investing|career|self-help|habits|motivation)\b"),
    ]
    output = []
    for label, pattern in rules:
        mask = remaining & titles.str.contains(pattern, regex=True, na=False)
        output.append({"label": label, "value": int(mask.sum())})
        remaining &= ~mask

    categories = sorted(output, key=lambda item: item["value"], reverse=True)
    classified = int(sum(item["value"] for item in categories))
    unclassified = int(remaining.sum())
    return {"categories": categories, "classified": classified, "unclassified": unclassified}


@lru_cache(maxsize=1)
def dataset_overview() -> dict:
    """Aggregate dataset statistics for the About page."""
    catalogue = display_books()
    ratings = load_explicit_ratings()
    users = load_users()

    average_rating_10 = float(ratings["Book-Rating"].mean()) if not ratings.empty else 0.0
    rated_isbns = set(ratings["ISBN"].astype(str))
    raw = load_catalogue()
    rated_keys = raw.loc[raw["ISBN"].isin(rated_isbns), "Book-Key"].nunique()

    rating_counts = ratings["Book-Rating"].astype(int).value_counts().sort_index()
    rating_distribution = [
        {
            "label": _rating_label(score10),
            "score10": score10,
            "value": int(rating_counts.get(score10, 0)),
        }
        for score10 in range(1, 11)
    ]

    publisher_counts = (
        catalogue["Publisher"]
        .replace("", "Unknown publisher")
        .fillna("Unknown publisher")
        .value_counts()
        .head(7)
    )
    top_publishers = [
        {"label": str(label), "value": int(value)}
        for label, value in publisher_counts.items()
    ]

    years = pd.to_numeric(catalogue["year_numeric"], errors="coerce")
    publication_periods = [
        {"label": "Before 1980", "value": int((years < 1980).sum())},
        {"label": "1980s", "value": int(years.between(1980, 1989).sum())},
        {"label": "1990s", "value": int(years.between(1990, 1999).sum())},
        {"label": "2000–2005", "value": int(years.between(2000, VALID_PUBLICATION_YEAR_MAX).sum())},
    ]

    valid_years = years.dropna()
    category_summary = _estimated_category_summary(catalogue)
    category_coverage = (category_summary["classified"] / len(catalogue) * 100.0) if len(catalogue) else 0.0
    year_min = int(valid_years.min()) if not valid_years.empty else None
    year_max = int(valid_years.max()) if not valid_years.empty else None

    evaluation_path = BASE_DIR / "recommender_evaluation_results.csv"
    evaluation_results = []
    if evaluation_path.exists():
        try:
            evaluation = pd.read_csv(evaluation_path)
            for _, row in evaluation.iterrows():
                evaluation_results.append({
                    "model": str(row.get("Model", "")),
                    "precision": float(row.get("Precision@10", 0.0) or 0.0),
                    "recall": float(row.get("Recall@10", 0.0) or 0.0),
                    "f1": float(row.get("F1@10", 0.0) or 0.0),
                    "hit_rate": float(row.get("Hit Rate@10", 0.0) or 0.0),
                    "availability": float(row.get("Recommendation Availability", 0.0) or 0.0),
                    "coverage": float(row.get("Catalog Coverage@10", 0.0) or 0.0),
                    "mean_time_ms": float(row.get("Mean Recommendation Time (ms)", 0.0) or 0.0),
                    "test_cases": int(row.get("Test Cases", 0) or 0),
                })
        except (OSError, ValueError, TypeError):
            evaluation_results = []

    return {
        "total_books": int(len(catalogue)),
        "total_authors": int(catalogue["Book-Author"].replace("", pd.NA).nunique()),
        "total_publishers": int(catalogue["Publisher"].replace("", pd.NA).nunique()),
        "total_users": int(len(users)),
        "total_ratings": int(len(ratings)),
        "average_rating_10": round(average_rating_10, 2),
        "rated_books": int(rated_keys),
        "year_min": year_min,
        "year_max": year_max,
        "rating_distribution": rating_distribution,
        "top_publishers": top_publishers,
        "publication_periods": publication_periods,
        "estimated_categories": category_summary["categories"],
        "estimated_category_classified": category_summary["classified"],
        "estimated_category_unclassified": category_summary["unclassified"],
        "estimated_category_coverage": round(category_coverage, 1),
        "evaluation_results": evaluation_results,
    }

def book_by_isbn(isbn: str) -> dict | None:
    isbn = str(isbn or "").strip()
    if not isbn:
        return None
    raw = load_catalogue()
    match = raw[raw["ISBN"] == isbn]
    if match.empty:
        return None
    key = str(match.iloc[0]["Book-Key"])
    display = catalogue_with_stats()
    item = display[display["Book-Key"] == key]
    if item.empty:
        return None
    return item.iloc[0].to_dict()


def book_by_key(book_key: str) -> dict | None:
    match = catalogue_with_stats()[catalogue_with_stats()["Book-Key"] == str(book_key)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def books_for_keys(book_keys: list[str]) -> list[dict]:
    keys = list(dict.fromkeys(str(k) for k in book_keys if k))
    if not keys:
        return []
    order = {key: i for i, key in enumerate(keys)}
    rows = catalogue_with_stats()[catalogue_with_stats()["Book-Key"].isin(keys)].copy()
    rows["_order"] = rows["Book-Key"].map(order)
    rows = rows.sort_values("_order")
    return rows.drop(columns=["_order", "search_text"], errors="ignore").to_dict("records")


def popular_books(limit: int = 12) -> list[dict]:
    rows = catalogue_with_stats().sort_values(["num_ratings", "avg_rating"], ascending=False).head(limit)
    return rows.drop(columns=["search_text"], errors="ignore").to_dict("records")


def highly_rated_books(limit: int = 12, minimum_ratings: int = 100) -> list[dict]:
    rows = catalogue_with_stats()
    rows = rows[rows["num_ratings"] >= minimum_ratings]
    rows = rows.sort_values(["avg_rating", "num_ratings"], ascending=False).head(limit)
    return rows.drop(columns=["search_text"], errors="ignore").to_dict("records")


def hidden_gems(limit: int = 12) -> list[dict]:
    rows = catalogue_with_stats()
    rows = rows[(rows["num_ratings"].between(20, 120)) & (rows["avg_rating"] >= 8.2)]
    rows = rows.sort_values(["avg_rating", "num_ratings"], ascending=False).head(limit)
    return rows.drop(columns=["search_text"], errors="ignore").to_dict("records")


def books_by_author(author: str, exclude_key: str | None = None, limit: int = 8) -> list[dict]:
    rows = catalogue_with_stats()
    rows = rows[rows["Book-Author"].astype(str).str.casefold() == str(author).casefold()]
    if exclude_key:
        rows = rows[rows["Book-Key"] != exclude_key]
    rows = rows.sort_values(["num_ratings", "avg_rating"], ascending=False).head(limit)
    return rows.drop(columns=["search_text"], errors="ignore").to_dict("records")


def discover_books(
    query: str = "",
    minimum_rating: float = 0.0,
    year_from: int | None = None,
    year_to: int | None = None,
    sort: str = "popular",
    page: int = 1,
    per_page: int = 24,
) -> tuple[list[dict], int]:
    rows = catalogue_with_stats()
    query = str(query or "").strip().casefold()
    if query:
        rows = rows[rows["search_text"].str.contains(query, regex=False, na=False)]
    if minimum_rating > 0:
        rows = rows[(rows["avg_rating"] >= minimum_rating) & (rows["num_ratings"] > 0)]
    if year_from:
        rows = rows[rows["year_numeric"] >= int(year_from)]
    if year_to:
        rows = rows[rows["year_numeric"] <= int(year_to)]

    if sort == "rating":
        # Bayesian-style weighted rating prevents a book with only one or two
        # perfect scores from outranking a strongly rated book with substantial
        # reader evidence. The displayed rating remains the original average;
        # the weighted score is used only for ranking.
        rows = rows[rows["num_ratings"] > 0].copy()
        if not rows.empty:
            global_mean = float(load_explicit_ratings()["Book-Rating"].mean())
            evidence_prior = 20.0
            v = rows["num_ratings"].astype(float)
            rows["_weighted_rating"] = (
                (v / (v + evidence_prior)) * rows["avg_rating"]
                + (evidence_prior / (v + evidence_prior)) * global_mean
            )
            rows = rows.sort_values(
                ["_weighted_rating", "num_ratings", "avg_rating"],
                ascending=False,
            )
    elif sort == "newest":
        rows = rows.sort_values(["year_numeric", "num_ratings"], ascending=False, na_position="last")
    elif sort == "title":
        rows = rows.sort_values("Book-Title", key=lambda s: s.astype(str).str.casefold())
    else:
        rows = rows.sort_values(["num_ratings", "avg_rating"], ascending=False)

    total = len(rows)
    page = max(int(page or 1), 1)
    start = (page - 1) * per_page
    selected = rows.iloc[start:start + per_page]
    return selected.drop(columns=["search_text", "_weighted_rating"], errors="ignore").to_dict("records"), total


def search_suggestions(query: str, limit: int = 10) -> list[dict]:
    q = str(query or "").strip().casefold()
    if len(q) < 2:
        return []
    rows = catalogue_with_stats()
    mask = rows["search_text"].str.contains(q, regex=False, na=False)
    rows = rows[mask].copy()
    if rows.empty:
        return []
    title_lower = rows["Book-Title"].str.casefold()
    author_lower = rows["Book-Author"].str.casefold()
    rows["_priority"] = (
        title_lower.eq(q).astype(int) * 8
        + title_lower.str.startswith(q).astype(int) * 5
        + author_lower.str.startswith(q).astype(int) * 3
        + np.log1p(rows["num_ratings"]) / 10
    )
    rows = rows.sort_values(["_priority", "num_ratings"], ascending=False).head(limit)
    return rows.drop(columns=["search_text", "_priority"], errors="ignore").to_dict("records")


def safe_year(value) -> str:
    try:
        year = int(float(value))
        if VALID_PUBLICATION_YEAR_MIN <= year <= VALID_PUBLICATION_YEAR_MAX:
            return str(year)
    except (TypeError, ValueError):
        pass
    return "Year not listed"


def clean_isbn(value: str) -> str:
    return re.sub(r"[^0-9Xx]", "", str(value or "")).upper()
