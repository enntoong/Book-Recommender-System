from __future__ import annotations

from pathlib import Path
import sqlite3
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "lumen.db"
MAX_TASTE_SIGNALS = 10


def _connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with _connect() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS reader_books (
                book_key TEXT PRIMARY KEY,
                saved INTEGER NOT NULL DEFAULT 0,
                loved INTEGER NOT NULL DEFAULT 0,
                rating INTEGER,
                updated_at TEXT NOT NULL
            )
            """
        )

        # Version 2 stores personal ratings on the same 1-10 integer scale as
        # Book-Crossing. If an older Lumen database is present, its legacy
        # 1-5 values are migrated once to equivalent even values on the 1-10 scale.
        version = int(db.execute("PRAGMA user_version").fetchone()[0])
        if version < 2:
            db.execute(
                "UPDATE reader_books SET rating = rating * 2 "
                "WHERE rating IS NOT NULL AND rating BETWEEN 1 AND 5"
            )
            db.execute("PRAGMA user_version = 2")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_entry(book_key: str) -> dict:
    with _connect() as db:
        row = db.execute(
            "SELECT book_key, saved, loved, rating, updated_at FROM reader_books WHERE book_key = ?",
            (book_key,),
        ).fetchone()
    if row is None:
        return {"book_key": book_key, "saved": False, "loved": False, "rating": None}
    return {
        "book_key": row["book_key"],
        "saved": bool(row["saved"]),
        "loved": bool(row["loved"]),
        "rating": row["rating"],
        "updated_at": row["updated_at"],
    }


def _upsert(book_key: str, saved=None, loved=None, rating_marker="KEEP"):
    current = get_entry(book_key)
    new_saved = current["saved"] if saved is None else bool(saved)
    new_loved = current["loved"] if loved is None else bool(loved)
    new_rating = current["rating"] if rating_marker == "KEEP" else rating_marker

    with _connect() as db:
        if not new_saved and not new_loved and new_rating is None:
            db.execute("DELETE FROM reader_books WHERE book_key = ?", (book_key,))
        else:
            db.execute(
                """
                INSERT INTO reader_books (book_key, saved, loved, rating, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(book_key) DO UPDATE SET
                    saved = excluded.saved,
                    loved = excluded.loved,
                    rating = excluded.rating,
                    updated_at = excluded.updated_at
                """,
                (book_key, int(new_saved), int(new_loved), new_rating, _now()),
            )


def toggle_saved(book_key: str) -> bool:
    current = get_entry(book_key)
    value = not current["saved"]
    _upsert(book_key, saved=value)
    return value


def toggle_loved(book_key: str) -> bool:
    current = get_entry(book_key)
    value = not current["loved"]
    _upsert(book_key, loved=value)
    return value


def set_rating(book_key: str, rating: int | None) -> bool:
    """Store a 1-10 personal rating and keep positive-preference signals consistent.

    Returns True when a new rating below 8/10 automatically removes an existing
    Loved flag. Clearing a rating does not change Loved.
    """
    if rating is not None:
        rating = max(1, min(int(rating), 10))

    current = get_entry(book_key)
    remove_loved = bool(rating is not None and rating < 8 and current["loved"])
    _upsert(
        book_key,
        loved=False if remove_loved else None,
        rating_marker=rating,
    )
    return remove_loved


def get_library(view: str = "saved") -> list[dict]:
    condition = "saved = 1"
    if view == "loved":
        condition = "loved = 1"
    elif view == "rated":
        condition = "rating IS NOT NULL"
    with _connect() as db:
        rows = db.execute(
            f"SELECT book_key, saved, loved, rating, updated_at FROM reader_books WHERE {condition} ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def taste_seed_keys(limit: int = MAX_TASTE_SIGNALS) -> list[str]:
    """Loved books and personal ratings of at least 8/10 act as positive taste signals."""
    with _connect() as db:
        rows = db.execute(
            """
            SELECT book_key,
                   loved,
                   rating,
                   updated_at,
                   (CASE WHEN loved = 1 THEN 10 ELSE 0 END) + COALESCE(rating, 0) AS taste_strength
            FROM reader_books
            WHERE loved = 1 OR rating >= 8
            ORDER BY taste_strength DESC, updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [row["book_key"] for row in rows]


def library_counts() -> dict:
    with _connect() as db:
        row = db.execute(
            """
            SELECT
                SUM(CASE WHEN saved = 1 THEN 1 ELSE 0 END) AS saved_count,
                SUM(CASE WHEN loved = 1 THEN 1 ELSE 0 END) AS loved_count,
                SUM(CASE WHEN rating IS NOT NULL THEN 1 ELSE 0 END) AS rated_count
            FROM reader_books
            """
        ).fetchone()
    return {
        "saved": int(row["saved_count"] or 0),
        "loved": int(row["loved_count"] or 0),
        "rated": int(row["rated_count"] or 0),
    }


def reset_reader_profile():
    with _connect() as db:
        db.execute("DELETE FROM reader_books")
