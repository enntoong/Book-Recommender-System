# Lumen — Flask Book Recommender System

Lumen is a reader-facing Flask prototype built for an Artificial Intelligence recommender-system assignment. The interface behaves like a normal book discovery product; recommendation algorithm names are kept in the backend and analysis notebook rather than shown as controls to readers.

## Reader experience

### Home
- Large catalogue search
- Personalised shelf when a reading taste profile exists
- Popular reader favourites
- Highly rated books with sufficient rating evidence
- Hidden gems
- Recently viewed books

### Discover
- Search by title, author, publisher, or ISBN
- Normal reader-facing refinements: minimum rating, publication year, and sorting
- “Highest rated” uses a Bayesian-style weighted ranking so tiny rating samples do not dominate; displayed averages remain unchanged
- Pagination for the large catalogue

### For You
- Add books you genuinely loved
- Personal ratings use the original 1–10 scale; ratings of 8–10/10 also become positive taste signals
- Loved is treated as a strong positive preference: a rating below 8/10 removes Loved, and a low-rated book cannot be newly marked Loved until the conflict is resolved
- The system automatically builds a personalised recommendation shelf from several reference books
- Readers never choose a recommendation algorithm

### Book Details
- Book cover and metadata
- Reader rating activity
- Save for later
- Add/remove from taste profile
- Personal 1–10 rating using the same scale as the Book-Crossing explicit ratings
- Automatic “You might also enjoy” recommendations
- More books from the same author

### My Books
- Saved books
- Loved books
- Rated books
- Persistent local demo profile stored in SQLite
- Reset control for a clean classroom demonstration

## Recommendation engine

The source code contains three technical approaches for coursework analysis:

1. **Collaborative Filtering** — item-to-item cosine similarity based on reader rating behaviour.
2. **Content-Based Filtering** — TF-IDF similarity using title, author, publisher, and publication-period metadata.
3. **Hybrid Filtering** — combines collaborative and content evidence, with automatic fallback when behavioural data is sparse. The prototype uses a 55% collaborative / 45% content blend as a documented design choice; it is not presented as a universally optimal weight.

The live application uses the available signals automatically. Technical scores and model names are not exposed as reader-facing choices.

## Data quality and preprocessing

The Flask application and analysis notebook share the same preprocessing functions in `data_service.py`. Publication years are treated as valid from **1900 to 2005** because the dataset distribution ends overwhelmingly by 2005 and the very small number of later values are handled as metadata outliers. HTML entities in titles/authors/publishers are decoded consistently, and zero-valued rating events are kept for descriptive analysis but excluded from model training. Book-Crossing explicit ratings remain on their original 1–10 scale in both the backend and the reader-facing interface. Personal ratings use the same 1–10 integer scale, so dataset ratings and user ratings are directly comparable without any display conversion or merged source levels.

## Performance design

The complete catalogue contains more than 245,000 unique title-author combinations. For faster live content matching, the TF-IDF candidate index uses books with at least two explicit reader ratings. A selected query book can still come from the full catalogue and is transformed into the same feature space at request time. This keeps cold-start support while reducing model construction time and memory usage.

## Evaluation

Run:

```bash
python evaluate_recommenders.py
```

This creates `recommender_evaluation_results.csv` with:

- Precision@10
- Recall@10
- F1@10
- Hit Rate@10
- Recommendation Availability
- Catalog Coverage@10
- Mean recommendation calculation time

The included evaluation samples up to **50 eligible readers** and uses a **held-out-user protocol**. Evaluation readers are removed from collaborative model construction; one known positive book is used as the seed and their other highly rated books form the unseen relevance set. This avoids direct user-rating leakage during model construction. The About Us page also loads the generated CSV and shows Precision@10, Recall@10, F1@10, Hit Rate@10, recommendation availability, and mean recommendation time so both accuracy and efficiency are visible during the prototype demonstration.

## Final prototype flow

The implemented reader flow is intentionally simpler than an algorithm-selection demo:

1. The reader browses or searches the Book-Crossing catalogue.
2. The reader can save books, mark genuinely loved books, or rate books on the original 1–10 scale.
3. Loved books and ratings of 8–10/10 form the positive local taste profile.
4. Lumen automatically combines collaborative and content evidence through the hybrid recommender, with fallback when one evidence source is sparse.
5. Related and personalised shelves show a relative **Match 1–99** ranking score; it is not a probability.
6. The three technical approaches are compared separately in the offline evaluation.

For the assignment report and presentation, the system-flow diagram should match this implemented flow. The report should also state clearly which group member owns each preferred recommender solution, as required by the assignment, without exposing an algorithm selector to end users.

## Main files

```text
Book-Recommender-System/
├── app.py
├── data_service.py
├── user_store.py
├── recommendation_service.py
├── collaborative_filtering.py
├── content_based_filtering.py
├── hybrid_filtering.py
├── evaluate_recommenders.py
├── recommender_evaluation_results.csv
├── Book_Recommender_Analysis.ipynb
├── requirements.txt
├── books_data/
├── templates/
└── static/
```

## Windows setup using uv

From the project folder:

```powershell
uv venv .venv --python 3.12
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Then open:

```text
http://127.0.0.1:5000
```

You do **not** need to activate the virtual environment, so this works even when PowerShell blocks `Activate.ps1`.

After the first setup, future runs only need:

```powershell
.\.venv\Scripts\python.exe app.py
```

You can also double-click `run_lumen.bat` after the environment has been created.

## Analysis notebook

`Book_Recommender_Analysis.ipynb` is the technical/academic companion to the user-facing Flask application. It contains:

- dataset overview
- implicit vs explicit rating activity
- rating distribution
- most frequently rated books
- high-confidence top-rated books
- publication-decade analysis
- rating volume vs average rating scatter plot
- collaborative recommendations
- content-based recommendations
- hybrid recommendations
- hybrid score contribution graph
- offline evaluation metrics
- accuracy comparison graph
- efficiency comparison graph
- interpretation and limitations

The notebook and UI deliberately serve different audiences: the Flask pages are for readers, while the notebook makes the AI techniques and evaluation visible for coursework assessment and presentation.
