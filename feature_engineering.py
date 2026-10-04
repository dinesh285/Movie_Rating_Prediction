import pandas as pd
import os

INPUT_FILE = "dataset/movies_final_2005_2025.csv"
OUTPUT_FILE = "dataset/movies_ml_ready.csv"

print("========================================")
print("FEATURE ENGINEERING")
print("========================================")

# ========================================
# 1. LOAD DATASET
# ========================================

print("\nLoading final dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total movies:", len(df))


# ========================================
# 2. REMOVE UNUSED ID
# ========================================

# IMDb ID is useful for reference,
# but it is not a prediction feature.

df = df.drop(
    columns=["tconst"],
    errors="ignore"
)


# ========================================
# 3. CLEAN TEXT FEATURES
# ========================================

text_columns = [
    "Movie_Name",
    "Language",
    "Genre",
    "Director",
    "Hero",
    "Heroine"
]

for col in text_columns:

    df[col] = (
        df[col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# ========================================
# 4. CLEAN NUMERICAL FEATURES
# ========================================

numeric_columns = [
    "Year",
    "Runtime",
    "Votes",
    "Rating"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ========================================
# 5. HANDLE MISSING VALUES
# ========================================

df["Runtime"] = df["Runtime"].fillna(
    df["Runtime"].median()
)

df["Votes"] = df["Votes"].fillna(0)


# ========================================
# 6. REMOVE INVALID RECORDS
# ========================================

df = df[
    df["Year"].between(
        2005,
        2025
    )
]

df = df[
    df["Rating"].between(
        0,
        10
    )
]


# ========================================
# 7. CREATE GENRE FEATURES
# ========================================

# Convert multiple genres into one text field

df["Genre"] = (
    df["Genre"]
    .str.replace(
        ",",
        " ",
        regex=False
    )
)


# ========================================
# 8. CREATE CAST/DIRECTOR TEXT FEATURE
# ========================================

df["People"] = (
    df["Director"] + " " +
    df["Hero"] + " " +
    df["Heroine"]
)


# ========================================
# 9. REMOVE MOVIE NAME FROM ML FEATURES
# ========================================

# Movie name is kept for display,
# but we don't use it to predict rating.

df["Movie_Name_Display"] = df["Movie_Name"]


# ========================================
# 10. SELECT ML DATA
# ========================================

ml_df = df[
    [
        "Movie_Name",
        "Language",
        "Year",
        "Genre",
        "People",
        "Runtime",
        "Votes",
        "Rating"
    ]
].copy()


# ========================================
# 11. SAVE
# ========================================

ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ========================================
# RESULTS
# ========================================

print("\n========================================")
print("FEATURE ENGINEERING COMPLETED!")
print("========================================")

print(
    "Output file:",
    OUTPUT_FILE
)

print(
    "Total records:",
    len(ml_df)
)

print("\nFeatures:")

print(
    ml_df.columns.tolist()
)

print("\nMissing values:")

print(
    ml_df.isnull().sum()
)

print("\nDataset preview:")

print(
    ml_df.head(10).to_string(
        index=False
    )
)

print("\n========================================")
print("DONE!")
print("========================================")