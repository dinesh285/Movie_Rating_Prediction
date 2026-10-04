import pandas as pd
import re
import unicodedata

IMDb_FILE = "dataset/movies_2005_2025_clean.csv"
TELUGU_FILE = "dataset/telugu_movies_2005_2025.csv"
OUTPUT_FILE = "dataset/movies_final_2005_2025.csv"


def normalize_title(title):
    if pd.isna(title):
        return ""

    title = str(title).lower()

    title = unicodedata.normalize(
        "NFKD", title
    ).encode(
        "ascii", "ignore"
    ).decode()

    title = re.sub(
        r"[^a-z0-9]+", " ", title
    )

    title = re.sub(
        r"\s+", " ", title
    ).strip()

    return title


print("========================================")
print("CREATING FINAL MOVIE DATASET")
print("========================================")


# ========================================
# 1. LOAD IMDb DATA
# ========================================

print("\nLoading IMDb dataset...")

imdb = pd.read_csv(IMDb_FILE)

print("IMDb records:", len(imdb))


# ========================================
# 2. LOAD TELUGU DATA
# ========================================

print("\nLoading Telugu reference dataset...")

telugu = pd.read_csv(TELUGU_FILE)

print("Telugu reference records:", len(telugu))


# ========================================
# 3. PREPARE IMDb DATA
# ========================================

imdb["title_key"] = (
    imdb["Movie_Name"]
    .apply(normalize_title)
)

# Keep original IMDb dataset
final_imdb = imdb.copy()


# ========================================
# 4. PREPARE TELUGU DATA
# ========================================

telugu["title_key"] = (
    telugu["Movie_Name"]
    .apply(normalize_title)
)

telugu["Year"] = pd.to_numeric(
    telugu["Year"],
    errors="coerce"
)

telugu["Rating"] = pd.to_numeric(
    telugu["Rating"],
    errors="coerce"
)

telugu["Runtime"] = pd.to_numeric(
    telugu["Runtime"],
    errors="coerce"
)

telugu["Votes"] = pd.to_numeric(
    telugu["Votes"],
    errors="coerce"
)


# Keep only 2005–2025

telugu = telugu[
    (telugu["Year"] >= 2005) &
    (telugu["Year"] <= 2025)
].copy()


# ========================================
# 5. FIND EXISTING IMDb MOVIES
# ========================================

print("\nChecking duplicate Telugu movies...")

existing_keys = set(
    zip(
        final_imdb["title_key"],
        final_imdb["Year"]
    )
)


# ========================================
# 6. FIND NEW TELUGU MOVIES
# ========================================

new_telugu = telugu[
    ~telugu.apply(
        lambda row:
        (row["title_key"], row["Year"])
        in existing_keys,
        axis=1
    )
].copy()

print(
    "New Telugu movies:",
    len(new_telugu)
)


# ========================================
# 7. PREPARE NEW TELUGU RECORDS
# ========================================

new_telugu["Language"] = "Telugu"

new_telugu["Director"] = "Unknown"
new_telugu["Hero"] = "Unknown"
new_telugu["Heroine"] = "Unknown"

new_telugu["tconst"] = (
    "TELUGU_" +
    new_telugu.index.astype(str)
)


# ========================================
# 8. SELECT REQUIRED COLUMNS
# ========================================

new_telugu = new_telugu[
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


# ========================================
# 9. PREPARE IMDb DATA
# ========================================

final_imdb = final_imdb[
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


# ========================================
# 10. COMBINE DATASETS
# ========================================

print("\nCombining datasets...")

final_df = pd.concat(
    [
        final_imdb,
        new_telugu
    ],
    ignore_index=True
)


# ========================================
# 11. REMOVE DUPLICATES
# ========================================

final_df["title_key"] = (
    final_df["Movie_Name"]
    .apply(normalize_title)
)

final_df = final_df.drop_duplicates(
    subset=["title_key", "Year"]
)


# ========================================
# 12. CLEAN
# ========================================

final_df = final_df[
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

final_df = final_df[
    final_df["Language"].isin(
        [
            "Telugu",
            "Hindi",
            "English"
        ]
    )
]

final_df = final_df[
    (final_df["Year"] >= 2005) &
    (final_df["Year"] <= 2025)
]

final_df = final_df[
    (final_df["Rating"] >= 0) &
    (final_df["Rating"] <= 10)
]


# ========================================
# 13. SAVE FINAL DATASET
# ========================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ========================================
# RESULTS
# ========================================

print("\n========================================")
print("FINAL DATASET CREATED SUCCESSFULLY!")
print("========================================")

print(
    "File:",
    OUTPUT_FILE
)

print(
    "Total movies:",
    len(final_df)
)

print("\nMovies by language:")

print(
    final_df["Language"].value_counts()
)

print("\nMovies by year:")

print(
    final_df["Year"]
    .value_counts()
    .sort_index()
)

print("\nSample records:")

print(
    final_df.head(20).to_string(index=False)
)

print("\n========================================")
print("DONE!")
print("========================================")