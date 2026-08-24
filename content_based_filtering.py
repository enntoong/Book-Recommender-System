import re
import html
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_service import VALID_PUBLICATION_YEAR_MIN, VALID_PUBLICATION_YEAR_MAX


def _slug(value):
    value = str(value or "").casefold()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "unknown"



def _canonical_title(value):
    text = html.unescape(str(value or "")).casefold()
    text = re.sub(r"\([^)]*\)", " ", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    removable = {"a", "an", "the", "and", "novel", "paperback", "hardcover", "edition", "book"}
    tokens = [t for t in tokens if t not in removable]
    return "".join(tokens)


def _content_text(row):
    title = str(row.get("Book-Title", ""))
    author = _slug(row.get("Book-Author", ""))
    publisher = _slug(row.get("Publisher", ""))
    year = pd.to_numeric(row.get("year_numeric", row.get("Year-Of-Publication", np.nan)), errors="coerce")
    if pd.isna(year) or year < VALID_PUBLICATION_YEAR_MIN or year > VALID_PUBLICATION_YEAR_MAX:
        year_token = "year_unknown decade_unknown"
    else:
        year = int(year)
        year_token = f"year_{year} decade_{(year // 10) * 10}"
    # Repeating author/publisher intentionally gives these metadata fields
    # more influence than a single title token.
    return (
        f"{title} "
        f"author_{author} author_{author} author_{author} "
        f"publisher_{publisher} publisher_{publisher} "
        f"{year_token}"
    )


def build_content_model(display_book_data, max_features=40000):
    """Build a metadata-based TF-IDF content model."""
    content_index = display_book_data.drop_duplicates("Book-Key").reset_index(drop=True).copy()
    content_index["Content-Text"] = content_index.apply(_content_text, axis=1)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(content_index["Content-Text"])
    key_to_position = pd.Series(content_index.index, index=content_index["Book-Key"]).to_dict()
    return content_index, vectorizer, matrix, key_to_position


def get_content_recommendations(
    selected_book_key,
    content_index,
    content_matrix,
    key_to_position,
    number_of_recommendations=20,
    vectorizer=None,
    selected_book_row=None,
):
    """Recommend books with similar title/author/publisher/year metadata.

    ``selected_book_row`` + ``vectorizer`` allow the selected book to act as a
    query even when it is not part of the smaller live candidate catalogue.
    This keeps the web application fast without losing cold-start support.
    """
    selected_position = key_to_position.get(selected_book_key)
    if selected_position is not None:
        selected = content_index.iloc[selected_position]
        selected_vector = content_matrix[selected_position]
    elif selected_book_row is not None and vectorizer is not None:
        selected = pd.Series(selected_book_row)
        selected_vector = vectorizer.transform([_content_text(selected)])
    else:
        return pd.DataFrame()

    scores = cosine_similarity(selected_vector, content_matrix).ravel()
    candidate_count = min(len(scores), max(number_of_recommendations * 8, 80))
    if candidate_count <= 1:
        return pd.DataFrame()

    top_idx = np.argpartition(scores, -candidate_count)[-candidate_count:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    selected_title = str(selected.get("Book-Title", "")).casefold()
    selected_title_norm = _canonical_title(selected.get("Book-Title", ""))
    selected_author = str(selected.get("Book-Author", "")).casefold()
    accepted_titles = []
    rows = []
    for idx in top_idx:
        row = content_index.iloc[idx].copy()
        candidate_key = str(row.get("Book-Key", ""))
        if candidate_key == str(selected_book_key) or scores[idx] <= 0:
            continue

        candidate_title = str(row.get("Book-Title", "")).casefold()
        candidate_title_norm = _canonical_title(row.get("Book-Title", ""))
        candidate_author = str(row.get("Book-Author", "")).casefold()

        same_book_text = (
            selected_title_norm and candidate_title_norm and
            (selected_title_norm in candidate_title_norm or candidate_title_norm in selected_title_norm)
        )
        if candidate_author == selected_author and (
            same_book_text or SequenceMatcher(None, candidate_title, selected_title).ratio() >= 0.82
        ):
            continue
        if any(
            candidate_author == a and (
                SequenceMatcher(None, candidate_title, t).ratio() >= 0.86 or
                (tn and candidate_title_norm and (tn in candidate_title_norm or candidate_title_norm in tn))
            )
            for t, tn, a in accepted_titles
        ):
            continue

        reasons = []
        if candidate_author == selected_author:
            reasons.append("same author")
        if str(row.get("Publisher", "")).casefold() == str(selected.get("Publisher", "")).casefold():
            reasons.append("same publisher")
        sy = pd.to_numeric(selected.get("Year-Of-Publication"), errors="coerce")
        ry = pd.to_numeric(row.get("Year-Of-Publication"), errors="coerce")
        if pd.notna(sy) and pd.notna(ry) and abs(float(sy) - float(ry)) <= 5:
            reasons.append("similar publication period")
        if not reasons:
            reasons.append("similar book metadata")

        row["content_score"] = float(scores[idx])
        row["recommendation_reason"] = ", ".join(reasons[:2]).capitalize()
        rows.append(row)
        accepted_titles.append((candidate_title, candidate_title_norm, candidate_author))
        if len(rows) >= number_of_recommendations:
            break

    return pd.DataFrame(rows)
