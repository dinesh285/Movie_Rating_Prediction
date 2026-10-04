import pandas as pd
import os

DATASET_DIR = "dataset"

MOVIES_FILE = os.path.join(
    DATASET_DIR,
    "movies_2005_2025_with_cast.csv"
)

AKAS_FILE = os.path.join(
    DATASET_DIR,
    "title.akas.tsv.gz"
)

OUTPUT_FILE = os.path.join(
    DATASET_DIR,
    "movies_2005_2025.csv"
)

print("Loading movie dataset...")

movies = pd.read_csv(MOVIES_FILE)

print("Total movies:", len(movies))

movie_ids = set(movies["tconst"])

print("\nReading IMDb title language data...")

language_data = {}

for chunk in pd.read_csv(
    AKAS_FILE,
    sep="\t",
    compression="gzip",
    na_values="\\N",
    usecols=[
        "titleId",
        "region",
        "language"
    ],
    chunksize=200000
):

    chunk = chunk[
        chunk["titleId"].isin(movie_ids)
    ]

    for _, row in chunk.iterrows():

        movie_id = row["titleId"]
        language = row["language"]

        if pd.isna(language):
            continue

        if movie_id not in language_data:
            language_data[movie_id] = set()

        language_data[movie_id].add(
            language
        )

    print(
        "Processed language chunk:",
        len(chunk)
    )

print("\nAssigning languages...")

def identify_language(movie_id):

    languages = language_data.get(
        movie_id,
        set()
    )

    # IMDb language codes
    if "te" in languages:
        return "Telugu"

    if "hi" in languages:
        return "Hindi"

    if "en" in languages:
        return "English"

    return "Other"


movies["Language"] = movies["tconst"].apply(
    identify_language
)

print("\nLanguage distribution:")

print(
    movies["Language"].value_counts()
)

# Keep only our three languages
movies = movies[
    movies["Language"].isin(
        ["Telugu", "Hindi", "English"]
    )
].copy()

# Reorder columns
movies = movies[
    [
        "tconst",
        "Movie_Name",
        "Language",
        "Year",
        "Genre",
        "Director",
        "Hero",
        "Heroine",
        "Runtime",
        "Votes",
        "Rating"
    ]
]

movies.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("FINAL DATASET CREATED!")
print("========================================")

print("File:", OUTPUT_FILE)

print("Total movies:", len(movies))

print("\nMovies by language:")

print(
    movies["Language"].value_counts()
)

print("\nSample data:")

print(
    movies.head(10).to_string(index=False)
)