import pandas as pd
import re
import unicodedata

IMDb_FILE = "dataset/movies_2005_2025_clean.csv"
TELUGU_FILE = "dataset/telugu_movies_2005_2025.csv"
OUTPUT_FILE = "dataset/telugu_matched_movies.csv"


def normalize_title(title):
    if pd.isna(title):
        return ""

    title = str(title).lower()

    title = unicodedata.normalize(
        "NFKD",
        title
    ).encode(
        "ascii",
        "ignore"
    ).decode()

    title = re.sub(
        r"[^a-z0-9]+",
        " ",
        title
    )

    title = re.sub(
        r"\s+",
        " ",
        title
    ).strip()

    return title


print("========================================")
print("MATCHING TELUGU MOVIES WITH IMDb")
print("========================================")


# ========================================
# 1. LOAD IMDb DATASET
# ========================================

print("\nLoading IMDb dataset...")

imdb = pd.read_csv( IMDb_FILE )

print("IMDb movies:", len(imdb))


# ========================================
# 2. LOAD TELUGU REFERENCE DATASET
# ========================================

print("\nLoading Telugu reference dataset...")

telugu = pd.read_csv(TELUGU_FILE)

print("Telugu reference movies:", len(telugu))


# ========================================
# 3. NORMALIZE TITLES
# ========================================

print("\nNormalizing movie titles...")

imdb["title_key"] = imdb["Movie_Name"].apply(
    normalize_title
)

telugu["title_key"] = telugu["Movie_Name"].apply(
    normalize_title
)


# ========================================
# 4. PREPARE IMDb DATA
# ========================================

imdb_match = imdb[
    [
        "tconst",
        "Movie_Name",
        "Year",
        "Language",
        "Genre",
        "Director",
        "Hero",
        "Heroine",
        "Runtime",
        "Votes",
        "Rating",
        "title_key"
    ]
].copy()


imdb_match.rename(
    columns={
        "Movie_Name": "IMDb_Movie_Name",
        "Language": "IMDb_Language",
        "Genre": "IMDb_Genre",
        "Director": "IMDb_Director",
        "Hero": "IMDb_Hero",
        "Heroine": "IMDb_Heroine",
        "Runtime": "IMDb_Runtime",
        "Votes": "IMDb_Votes",
        "Rating": "IMDb_Rating"
    },
    inplace=True
)


# ========================================
# 5. REMOVE REFERENCE tconst
# ========================================

# The Telugu reference dataset has its own
# artificial tconst value. We don't need it.

if "tconst" in telugu.columns:
    telugu = telugu.drop(
        columns=["tconst"]
    )


# ========================================
# 6. MATCH TITLE + YEAR
# ========================================

print("\nMatching movies using Movie Name + Year...")

matched = telugu.merge(
    imdb_match,
    on=[
        "title_key",
        "Year"
    ],
    how="inner"
)

print(
    "Matched movies:",
    len(matched)
)


# ========================================
# 7. KEEP IMDb-CONFIRMED TELUGU
# ========================================

matched = matched[
    matched["IMDb_Language"] == "Telugu"
].copy()

print(
    "IMDb-confirmed Telugu movies:",
    len(matched)
)


# ========================================
# 8. CREATE FINAL DATASET
# ========================================

matched["Language"] = "Telugu"


matched = matched[
    [
        "tconst",
        "IMDb_Movie_Name",
        "Language",
        "Year",
        "IMDb_Genre",
        "IMDb_Director",
        "IMDb_Hero",
        "IMDb_Heroine",
        "IMDb_Runtime",
        "IMDb_Votes",
        "IMDb_Rating"
    ]
]


# ========================================
# 9. RENAME COLUMNS
# ========================================

matched.rename(
    columns={
        "IMDb_Movie_Name": "Movie_Name",
        "IMDb_Genre": "Genre",
        "IMDb_Director": "Director",
        "IMDb_Hero": "Hero",
        "IMDb_Heroine": "Heroine",
        "IMDb_Runtime": "Runtime",
        "IMDb_Votes": "Votes",
        "IMDb_Rating": "Rating"
    },
    inplace=True
)


# ========================================
# 10. REMOVE DUPLICATES
# ========================================

matched = matched.drop_duplicates(
    subset=["tconst"]
)


# ========================================
# 11. SAVE
# ========================================

matched.to_csv(
    OUTPUT_FILE,
    index=False
)


# ========================================
# 12. RESULTS
# ========================================

print("\n========================================")
print("TELUGU MATCHING COMPLETED!")
print("========================================")

print(
    "Output file:",
    OUTPUT_FILE
)

print(
    "Final Telugu movies:",
    len(matched)
)

print("\nLanguage distribution:")

print(
    matched["Language"].value_counts()
)

print("\nSample Telugu movies:")

print(
    matched.head(20).to_string(
        index=False
    )
)

print("\n========================================")
print("DONE!")
print("========================================")