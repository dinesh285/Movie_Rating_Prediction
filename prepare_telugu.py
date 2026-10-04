import pandas as pd
import os

# ========================================
# TELUGU MOVIE DATA PREPARATION
# ========================================

INPUT_FILE = "dataset/TeluguMovies_dataset.txt"
OUTPUT_FILE = "dataset/telugu_movies_2005_2025.csv"

print("========================================")
print("TELUGU MOVIE DATA PREPARATION")
print("========================================")

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("\nOriginal records:", len(df))

# ----------------------------------------
# Select required columns
# ----------------------------------------

df = df[
    [
        "Movie",
        "Year",
        "Genre",
        "Runtime",
        "Rating",
        "No.of.Ratings"
    ]
].copy()

# Rename columns
df.rename(
    columns={
        "Movie": "Movie_Name",
        "No.of.Ratings": "Votes"
    },
    inplace=True
)

# ----------------------------------------
# Convert numerical columns
# ----------------------------------------

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df["Runtime"] = pd.to_numeric(
    df["Runtime"],
    errors="coerce"
)

df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)

df["Votes"] = pd.to_numeric(
    df["Votes"],
    errors="coerce"
)

# ----------------------------------------
# Keep 2005–2025
# ----------------------------------------

df = df[
    (df["Year"] >= 2005) &
    (df["Year"] <= 2025)
].copy()

print(
    "Movies from 2005-2025:",
    len(df)
)

# ----------------------------------------
# Remove missing values
# ----------------------------------------

df["Movie_Name"] = (
    df["Movie_Name"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

df["Genre"] = (
    df["Genre"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

df["Runtime"] = df["Runtime"].fillna(
    df["Runtime"].median()
)

df["Votes"] = df["Votes"].fillna(0)

df["Rating"] = df["Rating"].fillna(
    df["Rating"].median()
)

# ----------------------------------------
# Remove duplicate movie names
# ----------------------------------------

df = df.drop_duplicates(
    subset=["Movie_Name", "Year"]
)

# ----------------------------------------
# Add language
# ----------------------------------------

df["Language"] = "Telugu"

# ----------------------------------------
# Add empty columns that match
# IMDb dataset structure
# ----------------------------------------

df["tconst"] = "TELUGU_REFERENCE"

df["Director"] = "Unknown"
df["Hero"] = "Unknown"
df["Heroine"] = "Unknown"

# ----------------------------------------
# Arrange columns
# ----------------------------------------

df = df[
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

# ----------------------------------------
# Save
# ----------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("TELUGU DATASET CREATED!")
print("========================================")

print(
    "File:",
    OUTPUT_FILE
)

print(
    "Total Telugu movies:",
    len(df)
)

print("\nYear distribution:")

print(
    df["Year"]
    .value_counts()
    .sort_index()
)

print("\nSample Telugu movies:")

print(
    df.head(20).to_string(index=False)
)

print("\n========================================")
print("DONE!")
print("========================================")