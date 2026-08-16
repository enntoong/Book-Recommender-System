import os
import re

import pandas as pd
import requests
import streamlit as st
from streamlit_searchbox import st_searchbox

from collaborative_filtering import (
    build_collaborative_model,
    get_collaborative_recommendations
)


# ==================================================
# Page configuration
# ==================================================

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# CSS design
# ==================================================

st.markdown(
    """
    <style>

    /* Main page width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Space between columns */
    [data-testid="stHorizontalBlock"] {
        gap: 18px;
        align-items: stretch;
    }

    /* Book card */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
        height: 100%;
        background-color: white;
    }

    /* Book card hover */
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 9px 22px rgba(0, 0, 0, 0.15);
    }

    /* Book-cover image */
    [data-testid="stImage"] img {
        width: 100%;
        height: 230px;
        object-fit: contain;
        border-radius: 9px;
        background-color: white;
        padding: 5px;
    }

    /* Book title */
    .book-title {
        font-size: 17px;
        font-weight: 700;
        line-height: 1.35;
        min-height: 48px;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    /* Book information */
    .book-info {
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 4px;
    }

    /* Rating badge */
    .rating-badge {
        display: inline-block;
        background-color: #fff3cd;
        color: #7a5a00;
        border-radius: 20px;
        padding: 5px 10px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 6px;
        margin-bottom: 6px;
    }

    /* Page number */
    .page-information {
        text-align: center;
        font-size: 15px;
        font-weight: 600;
        padding-top: 10px;
    }

    /* Book-details information box */
    .details-information {
        font-size: 16px;
        line-height: 1.8;
        margin-bottom: 7px;
    }

    /* Description text */
    .book-description {
        font-size: 16px;
        line-height: 1.8;
        text-align: justify;
    }

    /* Hidden marker used to identify clickable book cards */
    .book-card-click-marker {
        display: none;
    }

    /* Only book cards containing the marker become clickable */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.book-card-click-marker) {
        position: relative;
        cursor: pointer;
        overflow: hidden;
    }

    /* Stretch the invisible button across the whole book card */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.book-card-click-marker)
    div[data-testid="stButton"] {
        position: absolute;
        inset: 0;
        z-index: 10;
        width: 100%;
        height: 100%;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:has(.book-card-click-marker)
    div[data-testid="stButton"] button {
        width: 100% !important;
        height: 100% !important;
        min-height: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        border-radius: 14px !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Hide the invisible button text */
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.book-card-click-marker)
    div[data-testid="stButton"] button p {
        opacity: 0;
        margin: 0 !important;
    }

    /* Search icon button only */
    .st-key-show_search_button button {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
        padding: 0 !important;
        margin: 0 !important;
        min-width: 36px !important;
        width: 36px !important;
        min-height: 36px !important;
        height: 36px !important;
        border-radius: 0 !important;
    }

    .st-key-show_search_button button p {
        font-size: 22px !important;
        margin: 0 !important;
    }

    .st-key-show_search_button button:hover {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        transform: scale(1.1);
    }

    .st-key-show_search_button button:focus,
    .st-key-show_search_button button:active {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# Default values
# ==================================================

DEFAULT_IMAGE = "images/default_book.png"


# ==================================================
# Session state
# ==================================================

if "current_view" not in st.session_state:
    st.session_state.current_view = "Book List"

if "selected_book_isbn" not in st.session_state:
    st.session_state.selected_book_isbn = None

if "show_book_search" not in st.session_state:
    st.session_state.show_book_search = False

if "home_page" not in st.session_state:
    st.session_state.home_page = 1

if "popular_page" not in st.session_state:
    st.session_state.popular_page = 1

if "recommendation_page" not in st.session_state:
    st.session_state.recommendation_page = 1

if "previous_search" not in st.session_state:
    st.session_state.previous_search = ""

if "previous_sort" not in st.session_state:
    st.session_state.previous_sort = "Default"


# ==================================================
# ISBN helper
# ==================================================

def clean_isbn(isbn):
    """
    Remove quotation marks, spaces and hyphens.
    """

    if pd.isna(isbn):
        return ""

    return (
        str(isbn)
        .replace('"', "")
        .replace("-", "")
        .replace(" ", "")
        .strip()
    )


# ==================================================
# Image functions
# ==================================================

def get_openlibrary_image(isbn, size="M"):
    """
    Generate an Open Library cover URL.
    """

    isbn = clean_isbn(isbn)

    if not isbn:
        return None

    return (
        "https://covers.openlibrary.org/"
        f"b/isbn/{isbn}-{size}.jpg?default=false"
    )


def download_book_cover(image_url, isbn):
    """
    Download one book cover and save it locally.

    The image is downloaded only when its local file
    does not already exist.
    """

    if not image_url:
        return None

    cleaned_isbn = clean_isbn(isbn)

    if not cleaned_isbn:
        return None

    cover_folder = os.path.join(
        "images",
        "covers"
    )

    os.makedirs(
        cover_folder,
        exist_ok=True
    )

    local_cover_path = os.path.join(
        cover_folder,
        f"{cleaned_isbn}.jpg"
    )

    # Do not download the same cover again.
    if os.path.exists(local_cover_path):
        return local_cover_path

    try:

        response = requests.get(
            image_url,
            timeout=8,
            allow_redirects=True
        )

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if (
            response.status_code == 200
            and "image" in content_type
            and len(response.content) > 0
        ):

            with open(
                local_cover_path,
                "wb"
            ) as image_file:

                image_file.write(
                    response.content
                )

            return local_cover_path

    except (
        requests.RequestException,
        OSError
    ):

        return None

    return None


def get_book_image(isbn, dataset_url):
    """
    Book-cover priority:

    1. Existing local cover
    2. Dataset image, then save locally
    3. Open Library image, then save locally
    4. Default local cover
    """

    cleaned_isbn = clean_isbn(isbn)

    local_cover_path = os.path.join(
        "images",
        "covers",
        f"{cleaned_isbn}.jpg"
    )

    # 1. Use the local cover when it already exists.
    if os.path.exists(local_cover_path):
        return local_cover_path

    # 2. Try the image URL from the dataset.
    if not pd.isna(dataset_url):

        dataset_url = str(
            dataset_url
        ).strip()

        if dataset_url:

            dataset_url = dataset_url.replace(
                "http://",
                "https://"
            )

            downloaded_cover = (
                download_book_cover(
                    dataset_url,
                    isbn
                )
            )

            if downloaded_cover:
                return downloaded_cover

    # 3. Try Open Library when the dataset image fails.
    openlibrary_url = get_openlibrary_image(
        isbn,
        size="M"
    )

    downloaded_cover = download_book_cover(
        openlibrary_url,
        isbn
    )

    if downloaded_cover:
        return downloaded_cover

    # 4. Use the default image when no cover is available.
    return DEFAULT_IMAGE


# ==================================================
# Open Library description
# ==================================================

@st.cache_data(show_spinner=False)
def get_openlibrary_book_information(isbn):
    """
    Retrieve book description and subjects
    from Open Library.
    """

    cleaned_isbn = clean_isbn(isbn)

    default_result = {
        "description": (
            "No description is available for this book."
        ),
        "subjects": []
    }

    if not cleaned_isbn:
        return default_result

    api_url = (
        "https://openlibrary.org/api/books"
        f"?bibkeys=ISBN:{cleaned_isbn}"
        "&jscmd=data"
        "&format=json"
    )

    try:

        response = requests.get(
            api_url,
            timeout=8
        )

        if response.status_code != 200:
            return default_result

        response_data = response.json()

        book_key = f"ISBN:{cleaned_isbn}"

        if book_key not in response_data:
            return default_result

        book_information = response_data[
            book_key
        ]

        description = book_information.get(
            "description",
            ""
        )

        if isinstance(description, dict):

            description = description.get(
                "value",
                ""
            )

        description = str(description).strip()

        if not description:

            description = (
                "No description is available "
                "for this book."
            )

        subjects_data = book_information.get(
            "subjects",
            []
        )

        subjects = []

        for subject in subjects_data[:8]:

            if isinstance(subject, dict):

                subject_name = subject.get(
                    "name",
                    ""
                )

                if subject_name:
                    subjects.append(subject_name)

        return {
            "description": description,
            "subjects": subjects
        }

    except (
        requests.RequestException,
        ValueError,
        TypeError
    ):

        return default_result


# ==================================================
# Text helper
# ==================================================

def shorten_text(text, maximum_length):
    """
    Shorten long text and add ...
    """

    if pd.isna(text):
        return "Unknown"

    text = str(text).strip()

    if not text:
        return "Unknown"

    if len(text) > maximum_length:
        return text[:maximum_length] + "..."

    return text


# ==================================================
# Open book-details page
# ==================================================

def open_book_details(isbn):
    """
    Save the selected ISBN and open details.
    """

    st.session_state.selected_book_isbn = str(
        isbn
    )

    st.session_state.current_view = (
        "Book Details"
    )

    st.rerun()


# ==================================================
# Book card
# ==================================================

def display_book_card(
    row,
    show_rating=False,
    key_prefix="book"
):
    """
    Display one book inside a card.
    """

    isbn = str(
        row["ISBN"]
    )

    unique_key = (
        f"{key_prefix}_"
        f"{clean_isbn(isbn)}_"
        f"{row.name}"
    )

    with st.container(border=True):

        # Hidden marker: CSS uses this to target only book cards.
        st.markdown(
            '<span class="book-card-click-marker"></span>',
            unsafe_allow_html=True
        )

        image_url = get_book_image(
            isbn=isbn,
            dataset_url=row["Image-URL-M"]
        )

        st.image(
            image_url,
            width="stretch"
        )

        book_title = shorten_text(
            row["Book-Title"],
            30
        )

        author = shorten_text(
            row["Book-Author"],
            28
        )

        publisher = shorten_text(
            row["Publisher"],
            28
        )

        year = row[
            "Year-Of-Publication"
        ]

        st.markdown(
            f"""
            <div class="book-title">
                {book_title}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="book-info">
                <strong>Author:</strong>
                {author}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="book-info">
                <strong>Year:</strong>
                {year}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="book-info">
                <strong>Publisher:</strong>
                {publisher}
            </div>
            """,
            unsafe_allow_html=True
        )

        if show_rating:

            if "avg_rating" in row.index:

                st.markdown(
                    f"""
                    <div class="rating-badge">
                        ⭐ {row['avg_rating']:.2f}/10
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if "num_ratings" in row.index:

                st.markdown(
                    f"""
                    <div class="book-info">
                        <strong>Total Ratings:</strong>
                        {int(row['num_ratings'])}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.caption(
            f"ISBN: {isbn}"
        )

        # This button is transparent and covers only this book card.
        # Clicking anywhere on the card opens the Book Details page.
        if st.button(
            "View book details",
            key=f"view_details_{unique_key}",
            help=f"View details for {book_title}"
        ):
            open_book_details(isbn)


# ==================================================
# Book grid
# ==================================================

def display_book_grid(
    book_data,
    show_rating=False,
    key_prefix="grid"
):
    """
    Display books using five columns per row.
    """

    columns_per_row = 5

    book_data = book_data.reset_index(
        drop=True
    )

    for row_start in range(
        0,
        len(book_data),
        columns_per_row
    ):

        columns = st.columns(
            columns_per_row
        )

        row_books = book_data.iloc[
            row_start:
            row_start + columns_per_row
        ]

        for column_index, (_, book_row) in enumerate(
            row_books.iterrows()
        ):

            with columns[column_index]:

                display_book_card(
                    book_row,
                    show_rating=show_rating,
                    key_prefix=(
                        f"{key_prefix}_"
                        f"{row_start}_"
                        f"{column_index}"
                    )
                )

        st.write("")


# ==================================================
# Load datasets
# ==================================================

@st.cache_data
def load_data():

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

    # Clean column names
    books.columns = (
        books.columns
        .str.replace(
            '"',
            "",
            regex=False
        )
        .str.strip()
    )

    ratings.columns = (
        ratings.columns
        .str.replace(
            '"',
            "",
            regex=False
        )
        .str.strip()
    )

    # Clean ISBN
    books["ISBN"] = (
        books["ISBN"]
        .astype(str)
        .str.replace(
            '"',
            "",
            regex=False
        )
        .str.strip()
    )

    ratings["ISBN"] = (
        ratings["ISBN"]
        .astype(str)
        .str.replace(
            '"',
            "",
            regex=False
        )
        .str.strip()
    )

    # Clean titles
    books["Book-Title"] = (
        books["Book-Title"]
        .fillna("Unknown Title")
        .astype(str)
        .str.strip()
    )

    # Clean authors
    books["Book-Author"] = (
        books["Book-Author"]
        .fillna("Unknown Author")
        .astype(str)
        .str.strip()
    )

    # Clean publishers
    books["Publisher"] = (
        books["Publisher"]
        .fillna("Unknown Publisher")
        .astype(str)
        .str.strip()
    )

    # Clean image URLs
    books["Image-URL-M"] = (
        books["Image-URL-M"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Convert rating to number
    ratings["Book-Rating"] = pd.to_numeric(
        ratings["Book-Rating"],
        errors="coerce"
    )

    ratings = ratings.dropna(
        subset=["Book-Rating"]
    )

    # Remove invalid books
    books = books.dropna(
        subset=[
            "ISBN",
            "Book-Title"
        ]
    )

    books = books[
        books["ISBN"].str.strip() != ""
    ]

    books = books[
        books["Book-Title"].str.strip() != ""
    ]

    return books, ratings


books, ratings = load_data()

# Create a standardized title for matching
books["Normalized-Title"] = (
    books["Book-Title"]
    .astype(str)
    .str.casefold()
    .str.replace(
        r"[^a-z0-9]+",
        "",
        regex=True
    )
)

# Create a standardized author for matching
books["Normalized-Author"] = (
    books["Book-Author"]
    .astype(str)
    .str.casefold()
    .str.replace(
        r"[^a-z0-9]+",
        "",
        regex=True
    )
)

# Create a unique key using standardized title and author
books["Book-Key"] = (
    books["Normalized-Title"]
    + "|"
    + books["Normalized-Author"]
)


# Keep one record for each title and author combination
display_book_data = books.drop_duplicates(
    subset=["Book-Key"]
).copy()


# ==================================================
# Book-rating statistics
# ==================================================

@st.cache_data
def build_book_rating_statistics(
    books_data,
    ratings_data
):
    """
    Calculate rating count and average
    for each book title.
    """

    explicit_ratings = ratings_data[
        ratings_data["Book-Rating"] > 0
    ].copy()

    rating_with_titles = explicit_ratings.merge(
        books_data[
            [
                "ISBN",
                "Book-Title"
            ]
        ],
        on="ISBN",
        how="inner"
    )

    statistics = (
        rating_with_titles
        .groupby("Book-Title")
        .agg(
            num_ratings=(
                "Book-Rating",
                "count"
            ),
            avg_rating=(
                "Book-Rating",
                "mean"
            )
        )
        .reset_index()
    )

    return statistics


book_rating_statistics = (
    build_book_rating_statistics(
        books,
        ratings
    )
)



# ==================================================
# Collaborative model
# ==================================================

@st.cache_resource(show_spinner=False)
def load_collaborative_model():
    """
    Build and cache collaborative model.
    """

    return build_collaborative_model(
        books_data=books,
        ratings_data=ratings,
        minimum_user_ratings=20,
        minimum_book_ratings=10
    )


book_user_matrix, collaborative_sparse_matrix = (
    load_collaborative_model()
)


# ==================================================
# Recommendation search
# ==================================================

@st.cache_data(show_spinner=False)
def build_recommendation_title_index(book_data):
    """
    Prepare the unique book list for recommendation search.

    Book-Key is used internally by the model.
    Display-Name is shown to the user.
    """

    title_data = (
        book_data[
            [
                "Book-Key",
                "Book-Title",
                "Book-Author"
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=["Book-Key"]
        )
        .copy()
    )

    title_data["Book-Key"] = (
        title_data["Book-Key"]
        .astype(str)
        .str.strip()
    )

    title_data = title_data[
        title_data["Book-Key"] != ""
    ]

    title_data["Display-Name"] = (
        title_data["Book-Title"]
        .astype(str)
        .str.strip()
        + " | "
        + title_data["Book-Author"]
        .astype(str)
        .str.strip()
    )

    title_data["search_text"] = (
        title_data["Display-Name"]
        .str.casefold()
    )

    return title_data.reset_index(drop=True)


recommendation_title_index = (
    build_recommendation_title_index(
        display_book_data
    )
)


def search_recommendation_books(search_text):
    """
    Return at most 20 matching books.

    Users can search using the title or author because
    Book-Key contains both values.
    """

    if not search_text:
        return []

    cleaned_search = str(
        search_text
    ).strip().casefold()

    if not cleaned_search:
        return []

    search_values = (
        recommendation_title_index[
            "search_text"
        ]
    )

    exact_match = (
        search_values == cleaned_search
    )

    starts_with_match = (
        search_values.str.startswith(
            cleaned_search,
            na=False
        )
    )

    word_starts_with_match = (
        search_values.str.contains(
            rf"(^|\s){re.escape(cleaned_search)}",
            regex=True,
            na=False
        )
    )

    contains_match = (
        search_values.str.contains(
            cleaned_search,
            regex=False,
            na=False
        )
    )

    matching_books = (
        recommendation_title_index[
            contains_match
        ]
        .assign(
            search_priority=(
                exact_match[contains_match]
                .astype(int) * 4
                + starts_with_match[contains_match]
                .astype(int) * 3
                + word_starts_with_match[contains_match]
                .astype(int) * 2
                + contains_match[contains_match]
                .astype(int)
            )
        )
        .sort_values(
            by=[
                "search_priority",
                "Display-Name"
            ],
            ascending=[
                False,
                True
            ]
        )
        .head(20)
    )

    return matching_books[
        "Display-Name"
    ].tolist()


# ==================================================
# Sidebar
# ==================================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Popular Books",
        "Recommendation"
    ],
    key="navigation_page"
)


# ==================================================
# Book-details page
# ==================================================

if st.session_state.current_view == "Book Details":

    selected_isbn = str(
        st.session_state.selected_book_isbn
    )

    selected_book_data = books[
        books["ISBN"].astype(str)
        == selected_isbn
    ]

    if selected_book_data.empty:

        st.warning(
            "Book information was not found."
        )

        if st.button(
            "← Back to Books",
            key="missing_book_back"
        ):

            st.session_state.current_view = (
                "Book List"
            )

            st.rerun()

    else:

        selected_book = (
            selected_book_data.iloc[0]
        )

        if st.button(
            "← Back to Books",
            key="details_back_button"
        ):

            st.session_state.current_view = (
                "Book List"
            )

            st.rerun()

        st.caption(
            f"{page}  ›  Book Details"
        )

        image_column, detail_column = st.columns(
            [1, 2.2]
        )

        with image_column:

            details_image = get_book_image(
                isbn=selected_book["ISBN"],
                dataset_url=selected_book[
                    "Image-URL-M"
                ]
            )

            st.image(
                details_image,
                width="stretch"
            )

        with detail_column:

            st.title(
                selected_book[
                    "Book-Title"
                ]
            )

            selected_title = selected_book[
                "Book-Title"
            ]

            rating_information = (
                book_rating_statistics[
                    book_rating_statistics[
                        "Book-Title"
                    ] == selected_title
                ]
            )

            if not rating_information.empty:

                average_rating = (
                    rating_information.iloc[0][
                        "avg_rating"
                    ]
                )

                number_of_ratings = int(
                    rating_information.iloc[0][
                        "num_ratings"
                    ]
                )

                st.markdown(
                    f"""
                    <div class="rating-badge">
                        ⭐ {average_rating:.2f}/10
                        &nbsp; | &nbsp;
                        {number_of_ratings} ratings
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
                <div class="details-information">
                    <strong>Author:</strong>
                    {selected_book['Book-Author']}
                </div>

                <div class="details-information">
                    <strong>Publication Year:</strong>
                    {selected_book['Year-Of-Publication']}
                </div>

                <div class="details-information">
                    <strong>Publisher:</strong>
                    {selected_book['Publisher']}
                </div>

                <div class="details-information">
                    <strong>ISBN:</strong>
                    {selected_book['ISBN']}
                </div>
                """,
                unsafe_allow_html=True
            )

            openlibrary_information = (
                get_openlibrary_book_information(
                    selected_book["ISBN"]
                )
            )

            subjects = openlibrary_information[
                "subjects"
            ]

            if subjects:

                st.markdown(
                    "**Categories:** "
                    + ", ".join(subjects)
                )

        st.divider()

        st.subheader(
            "Overview"
        )

        book_description = (
            openlibrary_information[
                "description"
            ]
        )

        st.markdown(
            f"""
            <div class="book-description">
                {book_description}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        st.subheader(
            "Product Details"
        )

        detail_label_column, detail_value_column = (
            st.columns(
                [1, 4]
            )
        )

        with detail_label_column:

            st.write("**Author**")
            st.write("**Publisher**")
            st.write("**Publication Year**")
            st.write("**ISBN**")

        with detail_value_column:

            st.write(
                selected_book[
                    "Book-Author"
                ]
            )

            st.write(
                selected_book[
                    "Publisher"
                ]
            )

            st.write(
                selected_book[
                    "Year-Of-Publication"
                ]
            )

            st.write(
                selected_book[
                    "ISBN"
                ]
            )


# ==================================================
# Home page
# ==================================================

elif page == "Home":

    # Title, sorting and search icon
    title_column, sort_column, search_icon_column = st.columns(
        [8, 1, 1],
        vertical_alignment="center"
    )

    with title_column:

        st.header(
            "All Books"
        )

    with sort_column:

        sort_option = st.selectbox(
            "Sort books",
            [
                "Default",
                "Title: A-Z",
                "Title: Z-A",
                "Author: A-Z",
                "Author: Z-A"
               
            ],
            label_visibility="collapsed",
            key="home_sort_option"
        )

    with search_icon_column:

        search_icon_clicked = st.button(
            "🔍",
            key="show_search_button",
            help="Search books"
        )

    if search_icon_clicked:

        st.session_state.show_book_search = (
            not st.session_state.show_book_search
        )

    search = ""

    if st.session_state.show_book_search:

        search = st.text_input(
            "Search book title or author",
            placeholder=(
                "Enter a title or author name"
            ),
            label_visibility="collapsed",
            key="home_all_books_search"
        )

    display_books = display_book_data.copy()

    # Search filtering
    if search.strip():

        search_text = search.strip()

        search_condition = (
            display_books["Book-Title"]
            .str.contains(
                search_text,
                case=False,
                na=False,
                regex=False
            )
            |
            display_books["Book-Author"]
            .str.contains(
                search_text,
                case=False,
                na=False,
                regex=False
            )
        )

        display_books = display_books[
            search_condition
        ]

    # Sorting
    if sort_option == "Title: A-Z":

        display_books = display_books.sort_values(
            by="Book-Title",
            ascending=True,
            key=lambda column: (
                column.astype(str).str.casefold()
            )
        )

    elif sort_option == "Title: Z-A":

        display_books = display_books.sort_values(
            by="Book-Title",
            ascending=False,
            key=lambda column: (
                column.astype(str).str.casefold()
            )
        )

    elif sort_option == "Author: A-Z":

        display_books = display_books.sort_values(
            by="Book-Author",
            ascending=True,
            key=lambda column: (
                column.astype(str).str.casefold()
            )
        )

    elif sort_option == "Author: Z-A":

        display_books = display_books.sort_values(
            by="Book-Author",
            ascending=False,
            key=lambda column: (
                column.astype(str).str.casefold()
            )
        )

    
    # Reset page when search or sorting changes
    if (
        search != st.session_state.previous_search
        or sort_option != st.session_state.previous_sort
    ):

        st.session_state.home_page = 1
        st.session_state.previous_search = search
        st.session_state.previous_sort = sort_option

    # Pagination
    books_per_page = 10

    total_books = len(
        display_books
    )

    total_pages = max(
        1,
        (
            total_books
            + books_per_page
            - 1
        )
        // books_per_page
    )

    if st.session_state.home_page > total_pages:
        st.session_state.home_page = total_pages

    if st.session_state.home_page < 1:
        st.session_state.home_page = 1

    current_page = (
        st.session_state.home_page
    )

    start_index = (
        current_page - 1
    ) * books_per_page

    end_index = (
        start_index
        + books_per_page
    )

    page_books = display_books.iloc[
        start_index:end_index
    ]

    # Results information
    if total_books > 0:

        st.write(
            f"Showing **{start_index + 1}** to "
            f"**{min(end_index, total_books)}** "
            f"of **{total_books}** books"
        )

    # Display books
    if page_books.empty:

        st.warning(
            "No books found."
        )

    else:

        display_book_grid(
            page_books,
            show_rating=False,
            key_prefix=(
                f"home_page_{current_page}_"
                f"{sort_option}"
            )
        )

    # Bottom pagination
    if total_books > 0:

        st.divider()

        previous_column, page_column, next_column = (
            st.columns(
                [1, 2, 1]
            )
        )

        with previous_column:

            if st.button(
                "⬅ Previous Page",
                disabled=current_page <= 1,
                width="stretch",
                key="bottom_previous"
            ):

                st.session_state.home_page -= 1
                st.rerun()

        with page_column:

            st.markdown(
                f"""
                <div class="page-information">
                    Page {current_page}
                    of {total_pages}
                </div>
                """,
                unsafe_allow_html=True
            )

        with next_column:

            if st.button(
                "Next Page ➡",
                disabled=(
                    current_page >= total_pages
                ),
                width="stretch",
                key="bottom_next"
            ):

                st.session_state.home_page += 1
                st.rerun()


# ==================================================
# Popular Books page
# ==================================================

elif page == "Popular Books":

    st.header(
        "Popular Books"
    )


    st.write(
        "Books with at least 50 user ratings."
    )

    # Get all books with at least 50 ratings
    popular_books = (
        book_rating_statistics[
            book_rating_statistics[
                "num_ratings"
            ] >= 50
        ]
        .sort_values(
            by=[
                "avg_rating",
                "num_ratings"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    # Keep one book information record per title
    book_information = books.drop_duplicates(
        subset=["Book-Title"]
    )

    # Add ISBN, author, publisher and image information
    popular_books = popular_books.merge(
        book_information,
        on="Book-Title",
        how="left"
    )

    # Pagination settings
    books_per_page = 10

    total_books = len(
        popular_books
    )

    total_pages = max(
        1,
        (
            total_books
            + books_per_page
            - 1
        )
        // books_per_page
    )

    # Make sure page number is valid
    if st.session_state.popular_page > total_pages:

        st.session_state.popular_page = (
            total_pages
        )

    if st.session_state.popular_page < 1:

        st.session_state.popular_page = 1

    current_page = (
        st.session_state.popular_page
    )

    start_index = (
        current_page - 1
    ) * books_per_page

    end_index = (
        start_index
        + books_per_page
    )

    page_books = popular_books.iloc[
        start_index:end_index
    ]

    if popular_books.empty:

        st.warning(
            "No popular books match the "
            "minimum rating requirement."
        )

    else:

        # Display result count
        st.write(
            f"Showing **{start_index + 1}** to "
            f"**{min(end_index, total_books)}** "
            f"of **{total_books}** popular books"
        )

        # Display current page books
        display_book_grid(
            page_books,
            show_rating=True,
            key_prefix=(
                f"popular_page_{current_page}"
            )
        )

        # Bottom pagination
        st.divider()

        previous_column, page_column, next_column = (
            st.columns(
                [1, 2, 1]
            )
        )

        with previous_column:

            if st.button(
                "⬅ Previous Page",
                disabled=current_page <= 1,
                width="stretch",
                key="popular_previous"
            ):

                st.session_state.popular_page -= 1
                st.rerun()

        with page_column:

            st.markdown(
                f"""
                <div class="page-information">
                    Page {current_page}
                    of {total_pages}
                </div>
                """,
                unsafe_allow_html=True
            )

        with next_column:

            if st.button(
                "Next Page ➡",
                disabled=(
                    current_page >= total_pages
                ),
                width="stretch",
                key="popular_next"
            ):

                st.session_state.popular_page += 1
                st.rerun()

# ==================================================
# Recommendation page
# ==================================================

elif page == "Recommendation":

    st.header(
        "Book Recommendation"
    )

    st.write(
        "Search and select a book to receive "
        "collaborative recommendations."
    )

    recommendation_selected_book = st_searchbox(
        search_recommendation_books,
        key="recommendation_selected_book",
        label="Search and select a book",
        placeholder=(
            "Type a book title, "
            "for example: Harry Potter"
        ),
        clear_on_submit=False
    )

    if st.button(
        "Recommend",
        type="primary",
        key="recommend_button"
    ):

        if not recommendation_selected_book:

            st.warning(
                "Please search and select a book first."
            )

        else:

            selected_match = recommendation_title_index[
                recommendation_title_index["Display-Name"]
                == recommendation_selected_book
            ]

            if selected_match.empty:

                st.warning(
                    "The selected book could not be found."
                )

            else:

                selected_book_key = selected_match.iloc[0][
                    "Book-Key"
                ]

                st.session_state[
                    "recommendation_result_key"
                ] = selected_book_key

                st.session_state.recommendation_page = 1
    # Display recommendation results
    if (
        "recommendation_result_key"
        in st.session_state
    ):

        selected_recommendation_key = (
            st.session_state[
                "recommendation_result_key"
            ]
        )

        selected_book_data = display_book_data[
            display_book_data["Book-Key"]
            == selected_recommendation_key
        ]

        if not selected_book_data.empty:

            st.subheader(
                "Selected Book"
            )

            selected_row = (
                selected_book_data.iloc[0]
            )

            selected_book_column, empty_column = st.columns(
                [1, 3]
            )

            with selected_book_column:

                display_book_card(
                    selected_row,
                    show_rating=False,
                    key_prefix="selected_recommendation"
                )

            with st.spinner(
                "Finding books liked by "
                "similar readers..."
            ):

                recommended_books = (
                    get_collaborative_recommendations(
                        selected_book_key=(
                            selected_recommendation_key
                        ),
                        book_user_matrix=(
                            book_user_matrix
                        ),
                        sparse_matrix=(
                            collaborative_sparse_matrix
                        ),
                        display_book_data=(
                            display_book_data
                        ),
                        number_of_recommendations=20
                    )
                )

            if recommended_books.empty:

                st.warning(
                    "This book does not have enough "
                    "rating data for collaborative "
                    "recommendations. Please try "
                    "another book."
                )

            else:

                st.subheader(
                    "Recommended Books"
                )

                st.caption(
                    "These books have similar "
                    "user-rating patterns to the "
                    "selected book."
                )

                # Pagination settings
                books_per_page = 10

                total_books = len(
                    recommended_books
                )

                total_pages = max(
                    1,
                    (
                        total_books
                        + books_per_page
                        - 1
                    )
                    // books_per_page
                )

                # Make sure page number is valid
                if (
                    st.session_state.recommendation_page
                    > total_pages
                ):

                    st.session_state.recommendation_page = (
                        total_pages
                    )

                if (
                    st.session_state.recommendation_page
                    < 1
                ):

                    st.session_state.recommendation_page = 1

                current_page = (
                    st.session_state.recommendation_page
                )

                start_index = (
                    current_page - 1
                ) * books_per_page

                end_index = (
                    start_index
                    + books_per_page
                )

                page_books = recommended_books.iloc[
                    start_index:end_index
                ]

                # Display result count
                st.write(
                    f"Showing **{start_index + 1}** to "
                    f"**{min(end_index, total_books)}** "
                    f"of **{total_books}** recommended books"
                )

                # Display similarity scores for checking
                st.subheader(
                    "Similarity Scores"
                )

                st.dataframe(
                    recommended_books[
                        [
                            "Book-Title",
                            "Book-Author",
                            "collaborative_score"
                        ]
                    ],
                    use_container_width=True
                )


                # Display current page books
                display_book_grid(
                    page_books,
                    show_rating=False,
                    key_prefix=(
                        f"collaborative_recommendations_"
                        f"page_{current_page}"
                    )
                )

                # Bottom pagination
                st.divider()

                previous_column, page_column, next_column = (
                    st.columns(
                        [1, 2, 1]
                    )
                )

                with previous_column:

                    if st.button(
                        "⬅ Previous Page",
                        disabled=current_page <= 1,
                        width="stretch",
                        key="recommendation_previous"
                    ):

                        st.session_state.recommendation_page -= 1
                        st.rerun()

                with page_column:

                    st.markdown(
                        f"""
                        <div class="page-information">
                            Page {current_page}
                            of {total_pages}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with next_column:

                    if st.button(
                        "Next Page ➡",
                        disabled=(
                            current_page >= total_pages
                        ),
                        width="stretch",
                        key="recommendation_next"
                    ):

                        st.session_state.recommendation_page += 1
                        st.rerun()