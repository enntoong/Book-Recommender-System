import streamlit as st
import pandas as pd

# Page setting
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# Load dataset
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

    books.columns = books.columns.str.replace('"', '').str.strip()
    ratings.columns = ratings.columns.str.replace('"', '').str.strip()

    # Only keep books with image URL
    books = books.dropna(subset=["Image-URL-M"])
    books = books[books["Image-URL-M"].astype(str).str.strip() != ""]
    books = books[books["Image-URL-M"].astype(str).str.startswith("http", na=False)]

    # Remove duplicate book titles
    books = books.drop_duplicates("Book-Title")

    return books, ratings

books, ratings = load_data()

# App title
st.title("📚 Book Recommendation System")
st.write("Welcome! Explore books and get recommendations based on ratings and similarity.")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Popular Books", "Recommendation"])

# Home page
if page == "Home":
    st.header("All Books")

    # Search box
    search = st.text_input("Search book title")

    # Filter books
    display_books = books.copy()

    if search:
        display_books = display_books[
            display_books["Book-Title"].str.contains(search, case=False, na=False)
        ]

    # Show only first 30 books for home page
    display_books = display_books.head(30)

    # Display books in grid
    cols = st.columns(5)

    for i, row in display_books.reset_index(drop=True).iterrows():
        with cols[i % 5]:
            st.image(row["Image-URL-M"], width=120)
            st.markdown(f"**{row['Book-Title']}**")
            st.write(f"Author: {row['Book-Author']}")
            st.write(f"Year: {row['Year-Of-Publication']}")
            st.write(f"Publisher: {row['Publisher']}")

# Popular books page
elif page == "Popular Books":
    st.header("Popular Books")

    book_ratings = ratings.merge(books, on="ISBN")

    popular_books = book_ratings.groupby("Book-Title").agg(
        num_ratings=("Book-Rating", "count"),
        avg_rating=("Book-Rating", "mean")
    ).reset_index()

    popular_books = popular_books[popular_books["num_ratings"] >= 100]
    popular_books = popular_books.sort_values("avg_rating", ascending=False).head(20)

    popular_books = popular_books.merge(
        books,
        on="Book-Title"
    )

    cols = st.columns(5)

    for i, row in popular_books.reset_index(drop=True).iterrows():
        with cols[i % 5]:
            st.image(row["Image-URL-M"], width=120)
            st.markdown(f"**{row['Book-Title']}**")
            st.write(f"Author: {row['Book-Author']}")
            st.write(f"Ratings: {row['num_ratings']}")
            st.write(f"Average Rating: {row['avg_rating']:.2f}")

# Recommendation page
elif page == "Recommendation":
    st.header("Book Recommendation")

    st.write("Recommendation function will be added later.")