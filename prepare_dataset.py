import pandas as pd
import os

DATASET_DIR = "dataset"

basics_file = os.path.join(DATASET_DIR, "title.basics.tsv.gz")
ratings_file = os.path.join(DATASET_DIR, "title.ratings.tsv.gz")

print("Reading IMDb title basics in chunks...")

selected_chunks = []

for chunk in pd.read_csv(
    basics_file,
    sep="\t",
    compression="gzip",
    na_values="\\N",
    usecols=[
        "tconst",
        "titleType",
        "primaryTitle",
        "startYear",
        "runtimeMinutes",
        "genres"
    ],
    dtype={
        "tconst": "string",
        "titleType": "string",
        "primaryTitle": "string",
        "startYear": "string",
        "runtimeMinutes": "string",
        "genres": "string"
    },
    chunksize=100000
):
    # Keep only movies
    chunk = chunk[chunk["titleType"] == "movie"]

    # Convert year
    chunk["startYear"] = pd.to_numeric(
        chunk["startYear"],
        errors="coerce"
    )

    # Keep 2005-2025
    chunk = chunk[
        chunk["startYear"].between(2005, 2025)
    ]

    if not chunk.empty:
        selected_chunks.append(chunk)

    print(
        "Processed chunk - selected rows:",
        len(chunk)
    )

print("\nCombining selected movies...")

basics = pd.concat(
    selected_chunks,
    ignore_index=True
)

print(
    "Movies from 2005-2025:",
    len(basics)
)

print("\nLoading ratings...")

ratings = pd.read_csv(
    ratings_file,
    sep="\t",
    compression="gzip"
)

print(
    "Ratings loaded:",
    len(ratings)
)

print("\nMerging movie information with ratings...")

movies = basics.merge(
    ratings,
    on="tconst",
    how="inner"
)

movies.rename(
    columns={
        "primaryTitle": "Movie_Name",
        "startYear": "Year",
        "runtimeMinutes": "Runtime",
        "genres": "Genre",
        "averageRating": "Rating",
        "numVotes": "Votes"
    },
    inplace=True
)

# Convert numeric columns
movies["Year"] = movies["Year"].astype(int)
movies["Rating"] = pd.to_numeric(
    movies["Rating"],
    errors="coerce"
)
movies["Votes"] = pd.to_numeric(
    movies["Votes"],
    errors="coerce"
)

# Remove missing ratings
movies = movies.dropna(
    subset=["Movie_Name", "Year", "Genre", "Rating"]
)

# Save
output_file = os.path.join(
    DATASET_DIR,
    "movies_2005_2025_basic.csv"
)

movies.to_csv(
    output_file,
    index=False
)

print("\n===================================")
print("DATASET CREATED SUCCESSFULLY!")
print("===================================")

print(
    "File:",
    output_file
)

print(
    "Total movies:",
    len(movies)
)

print("\nColumns:")

print(
    movies.columns.tolist()
)

print("\nFirst 10 records:")

print(
    movies.head(10).to_string(index=False)
)