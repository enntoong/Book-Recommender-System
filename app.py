from __future__ import annotations

import math
import os
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, session, url_for

from data_service import (
    book_by_isbn,
    book_by_key,
    book_rating_distribution,
    books_by_author,
    books_for_keys,
    clean_isbn,
    discover_books,
    dataset_overview,
    hidden_gems,
    highly_rated_books,
    popular_books,
    safe_year,
    search_suggestions,
    VALID_PUBLICATION_YEAR_MIN,
    VALID_PUBLICATION_YEAR_MAX,
)
from recommendation_service import personalised_recommendations, smart_recommendations
from user_store import (
    get_entry,
    get_library,
    init_db,
    library_counts,
    MAX_TASTE_SIGNALS,
    reset_reader_profile,
    set_rating,
    taste_seed_keys,
    toggle_loved,
    toggle_saved,
)


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("LUMEN_SECRET_KEY", "lumen-local-demo-secret")
app.config["TEMPLATES_AUTO_RELOAD"] = True
init_db()


def _safe_local_target(target, fallback):
    """Return only a local application URL; otherwise use the fallback."""
    if target:
        parsed = urlparse(target)
        if not parsed.netloc and target.startswith("/"):
            return target
    return fallback


def _safe_next(default_endpoint="home"):
    target = request.form.get("next") or request.args.get("next")
    return _safe_local_target(target, url_for(default_endpoint))


def _current_url():
    """Current path + query string, safe to pass through forms and back links."""
    return request.full_path[:-1] if request.full_path.endswith("?") else request.full_path


def _book_details_url(isbn):
    """Open a book while remembering exactly which page the reader came from."""
    return url_for("book_details", isbn=isbn, back=_current_url())


def _remember_recent(book_key: str):
    recent = [str(k) for k in session.get("recent_books", []) if k]
    recent = [k for k in recent if k != book_key]
    recent.insert(0, book_key)
    session["recent_books"] = recent[:12]


def _cover_url(book):
    isbn = clean_isbn(book.get("ISBN", "")) if book else ""
    if isbn:
        local_path = BASE_DIR / "static" / "images" / "covers" / f"{isbn}.jpg"
        if local_path.exists():
            return url_for("static", filename=f"images/covers/{isbn}.jpg")
    dataset_url = str((book or {}).get("Image-URL-M", "") or "").strip()
    if dataset_url.startswith("http://"):
        dataset_url = "https://" + dataset_url[len("http://"):]
    if dataset_url.startswith("https://"):
        return dataset_url
    if isbn:
        return f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg?default=false"
    return url_for("static", filename="images/default_book.png")


@app.context_processor
def inject_helpers():
    return {
        "cover_url": _cover_url,
        "safe_year": safe_year,
        "default_cover": url_for("static", filename="images/default_book.png"),
        "library_counts": library_counts(),
        "valid_year_min": VALID_PUBLICATION_YEAR_MIN,
        "valid_year_max": VALID_PUBLICATION_YEAR_MAX,
        "current_url": _current_url(),
        "book_details_url": _book_details_url,
    }


@app.template_filter("compact")
def compact_number(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return "0"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


@app.get("/")
def home():
    seeds = taste_seed_keys(limit=MAX_TASTE_SIGNALS)
    personalised = personalised_recommendations(seeds, limit=10) if seeds else []
    seed_books = books_for_keys(seeds)
    recent = books_for_keys(session.get("recent_books", []))
    return render_template(
        "home.html",
        personalised=personalised,
        seed_books=seed_books,
        popular=popular_books(10),
        highly_rated=highly_rated_books(10),
        gems=hidden_gems(10),
        recent=recent[:10],
    )


@app.get("/about")
def about():
    return render_template("about.html", stats=dataset_overview())


@app.get("/discover")
def discover():
    query = request.args.get("q", "").strip()
    try:
        minimum_rating = float(request.args.get("rating", 0) or 0)
    except ValueError:
        minimum_rating = 0.0
    if not 0 <= minimum_rating <= 10:
        minimum_rating = 0.0
    try:
        year_from = int(request.args.get("from", "")) if request.args.get("from") else None
    except ValueError:
        year_from = None
    try:
        year_to = int(request.args.get("to", "")) if request.args.get("to") else None
    except ValueError:
        year_to = None

    if year_from is not None and not (VALID_PUBLICATION_YEAR_MIN <= year_from <= VALID_PUBLICATION_YEAR_MAX):
        year_from = None
    if year_to is not None and not (VALID_PUBLICATION_YEAR_MIN <= year_to <= VALID_PUBLICATION_YEAR_MAX):
        year_to = None
    if year_from is not None and year_to is not None and year_from > year_to:
        year_from, year_to = year_to, year_from
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    sort = request.args.get("sort", "popular")
    if sort not in {"popular", "rating", "newest", "title"}:
        sort = "popular"

    per_page = 24
    books, total = discover_books(
        query=query,
        minimum_rating=minimum_rating,
        year_from=year_from,
        year_to=year_to,
        sort=sort,
        page=page,
        per_page=per_page,
    )
    pages = max(1, math.ceil(total / per_page))
    if page > pages and total:
        page = pages
        books, total = discover_books(
            query=query,
            minimum_rating=minimum_rating,
            year_from=year_from,
            year_to=year_to,
            sort=sort,
            page=page,
            per_page=per_page,
        )
    result_start = ((page - 1) * per_page + 1) if total else 0
    result_end = min(page * per_page, total) if total else 0
    return render_template(
        "discover.html",
        books=books,
        query=query,
        minimum_rating=minimum_rating,
        year_from=year_from,
        year_to=year_to,
        sort=sort,
        page=page,
        pages=pages,
        total=total,
        result_start=result_start,
        result_end=result_end,
    )


@app.get("/for-you")
def for_you():
    query = request.args.get("q", "").strip()
    seeds = taste_seed_keys(limit=MAX_TASTE_SIGNALS)
    seed_books = books_for_keys(seeds)
    recommendations = personalised_recommendations(seeds, limit=20) if seeds else []
    search_results = search_suggestions(query, limit=12) if query else []
    return render_template(
        "for_you.html",
        seed_books=seed_books,
        recommendations=recommendations,
        query=query,
        search_results=search_results,
    )


@app.get("/my-books")
def my_books():
    view = request.args.get("view", "saved")
    if view not in {"saved", "loved", "rated"}:
        view = "saved"
    entries = get_library(view)
    books = books_for_keys([entry["book_key"] for entry in entries])
    entry_map = {entry["book_key"]: entry for entry in entries}
    for book in books:
        book["reader"] = entry_map.get(book["Book-Key"], {})
    return render_template("my_books.html", books=books, view=view)


@app.get("/book/<path:isbn>")
def book_details(isbn):
    book = book_by_isbn(isbn)
    if not book:
        abort(404)
    _remember_recent(str(book["Book-Key"]))
    reader = get_entry(str(book["Book-Key"]))
    rating_breakdown = book_rating_distribution(str(book["Book-Key"]))
    related = smart_recommendations(str(book["Book-Key"]), limit=10)
    more_by_author = books_by_author(str(book["Book-Author"]), exclude_key=str(book["Book-Key"]), limit=8)
    return render_template(
        "book_details.html",
        book=book,
        reader=reader,
        rating_breakdown=rating_breakdown,
        related=related,
        more_by_author=more_by_author,
        back_url=_safe_local_target(request.args.get("back"), url_for("discover")),
    )


@app.post("/action/save/<path:isbn>")
def action_save(isbn):
    book = book_by_isbn(isbn)
    if not book:
        abort(404)
    value = toggle_saved(str(book["Book-Key"]))
    flash("Saved to My Books." if value else "Removed from saved books.", "success")
    return redirect(_safe_next("home"))


@app.post("/action/love/<path:isbn>")
def action_love(isbn):
    book = book_by_isbn(isbn)
    if not book:
        abort(404)
    book_key = str(book["Book-Key"])
    current = get_entry(book_key)
    if not current["loved"] and current.get("rating") is not None and current["rating"] < 8:
        flash(
            f"This book is currently rated {current['rating']}/10. "
            "Raise the rating to 8–10/10 or remove the rating before marking it Loved.",
            "error",
        )
        return redirect(_safe_next("home"))

    value = toggle_loved(book_key)
    flash(
        "Added to your taste profile. Your recommendations will adapt." if value
        else "Removed from your taste profile.",
        "success",
    )
    return redirect(_safe_next("home"))


@app.post("/action/rate/<path:isbn>")
def action_rate(isbn):
    book = book_by_isbn(isbn)
    if not book:
        abort(404)
    raw = request.form.get("rating", "")
    try:
        rating = int(raw)
        if rating < 1 or rating > 10:
            raise ValueError
    except ValueError:
        flash("Choose a rating from 1 to 10.", "error")
        return redirect(_safe_next("home"))
    removed_loved = set_rating(str(book["Book-Key"]), rating)
    if removed_loved:
        flash(
            f"Your {rating}/10 rating was saved. Because Loved is a positive preference, "
            "this book was also removed from Loved.",
            "success",
        )
    elif rating >= 8:
        flash("Your rating was saved and added as a positive taste signal.", "success")
    else:
        flash("Your rating was saved. Ratings of 8–10/10 are used as positive taste signals.", "success")
    return redirect(_safe_next("home"))


@app.post("/action/clear-rating/<path:isbn>")
def action_clear_rating(isbn):
    book = book_by_isbn(isbn)
    if not book:
        abort(404)
    set_rating(str(book["Book-Key"]), None)
    flash("Rating removed.", "success")
    return redirect(_safe_next("home"))


@app.post("/action/reset-profile")
def action_reset_profile():
    reset_reader_profile()
    session.pop("recent_books", None)
    flash("Your local demo reading profile has been reset.", "success")
    return redirect(_safe_next("home"))


@app.get("/api/search")
def api_search():
    query = request.args.get("q", "")
    books = search_suggestions(query, limit=8)
    return jsonify([
        {
            "title": b["Book-Title"],
            "author": b["Book-Author"],
            "isbn": b["ISBN"],
            "url": url_for("book_details", isbn=b["ISBN"]),
        }
        for b in books
    ])


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)