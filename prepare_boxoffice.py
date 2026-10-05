import pandas as pd
import numpy as np
import os
import re


# =========================================================
# MOVIE BOX OFFICE DATA PREPARATION
# =========================================================

INPUT_FILE = "dataset/boxoffice_raw.csv"
OUTPUT_FILE = "dataset/boxoffice_clean.csv"


print()
print("================================================")
print("       MOVIE BOX OFFICE DATA PREPARATION")
print("================================================")
print()


# =========================================================
# 1. CHECK INPUT FILE
# =========================================================

if not os.path.exists(INPUT_FILE):

    print("ERROR: Input dataset not found.")
    print()
    print("Expected file:")
    print(INPUT_FILE)
    print()
    print("First run:")
    print("python download_boxoffice.py")
    exit()


# =========================================================
# 2. LOAD DATASET
# =========================================================

print("Loading dataset...")
print()

try:

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

except Exception as error:

    print("Could not read dataset.")
    print(error)
    exit()


print("Dataset loaded successfully.")
print()
print("Number of rows:", len(df))
print()


# =========================================================
# 3. SHOW ORIGINAL COLUMNS
# =========================================================

print("Original columns:")
print()

for column in df.columns:
    print("-", column)

print()


# =========================================================
# 4. RENAME COLUMNS
# =========================================================

rename_map = {

    "Title": "Movie_Name",

    "Worldwide Gross_scraped": "Worldwide_Gross",

    "Worldwide Gross": "Worldwide_Gross",

    "India Gross": "India_Gross",

    "India Gross_scraped": "India_Gross",

    "Budget": "Budget"

}

df.rename(
    columns=rename_map,
    inplace=True
)


# =========================================================
# 5. FIND MOVIE TITLE COLUMN
# =========================================================

if "Movie_Name" not in df.columns:

    possible_title_columns = [
        "Film",
        "film",
        "Movie",
        "movie",
        "Name",
        "name"
    ]

    found_title = None

    for column in possible_title_columns:

        if column in df.columns:
            found_title = column
            break

    if found_title:

        df.rename(
            columns={
                found_title: "Movie_Name"
            },
            inplace=True
        )

    else:

        print("ERROR: Movie title column not found.")
        print()
        print("Available columns:")
        print(df.columns.tolist())
        exit()


# =========================================================
# 6. MAKE REQUIRED COLUMNS
# =========================================================

required_columns = [

    "Movie_Name",
    "Year",
    "Genre",
    "Director",
    "Cast",
    "Budget",
    "Worldwide_Gross"

]

for column in required_columns:

    if column not in df.columns:

        print(
            "WARNING:",
            column,
            "not found. Creating empty column."
        )

        df[column] = np.nan


# =========================================================
# 7. CLEAN MONEY VALUES
# =========================================================

def clean_money(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    value = value.replace(
        ",",
        ""
    )

    value = value.replace(
        "₹",
        ""
    )

    value = value.replace(
        "$",
        ""
    )

    value = value.replace(
        "Rs.",
        ""
    )

    value = value.replace(
        "Rs",
        ""
    )

    value = value.strip()

    # Find numeric value
    match = re.search(
        r"\d+(?:\.\d+)?",
        value
    )

    if not match:
        return np.nan

    try:

        number = float(
            match.group()
        )

        return number

    except:

        return np.nan


print("Cleaning Budget...")

df["Budget"] = df["Budget"].apply(
    clean_money
)


print("Cleaning Worldwide Gross...")

df["Worldwide_Gross"] = (
    df["Worldwide_Gross"]
    .apply(clean_money)
)


# =========================================================
# 8. CLEAN YEAR
# =========================================================

print("Cleaning Year...")

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)


# =========================================================
# 9. FILTER YEAR
# =========================================================

df = df[
    (df["Year"] >= 2005)
    &
    (df["Year"] <= 2025)
]


# =========================================================
# 10. REMOVE INVALID BUDGET/GROSS
# =========================================================

df = df[
    (df["Budget"] > 0)
    &
    (df["Worldwide_Gross"] > 0)
]


# =========================================================
# 11. CALCULATE RECOVERY RATIO
# =========================================================

print("Calculating Recovery Ratio...")

df["Recovery_Ratio"] = (
    df["Worldwide_Gross"]
    /
    df["Budget"]
)


# =========================================================
# 12. REMOVE EXTREME DATA ERRORS
# =========================================================

# Extremely large ratios are usually caused by
# inconsistent budget/gross units.

df = df[
    (df["Recovery_Ratio"] >= 0)
    &
    (df["Recovery_Ratio"] <= 100)
]


# =========================================================
# 13. CALCULATE ROI
# =========================================================

df["ROI_Percentage"] = (

    (
        df["Worldwide_Gross"]
        -
        df["Budget"]
    )

    /

    df["Budget"]

) * 100


# =========================================================
# 14. BOX OFFICE RATING
# =========================================================

def calculate_boxoffice_rating(ratio):

    if ratio >= 5:

        return 5.0

    elif ratio >= 4:

        return 4.8

    elif ratio >= 3:

        return 4.3

    elif ratio >= 2:

        return 3.8

    elif ratio >= 1.5:

        return 3.2

    elif ratio >= 1:

        return 2.5

    elif ratio >= 0.75:

        return 1.8

    else:

        return 1.0


df["BoxOffice_Rating"] = (
    df["Recovery_Ratio"]
    .apply(calculate_boxoffice_rating)
)


# =========================================================
# 15. BOX OFFICE VERDICT
# =========================================================

def calculate_verdict(rating):

    if rating >= 4.5:

        return "BLOCKBUSTER"

    elif rating >= 4.0:

        return "SUPER HIT"

    elif rating >= 3.5:

        return "HIT"

    elif rating >= 2.5:

        return "AVERAGE"

    else:

        return "FLOP"


df["BoxOffice_Verdict"] = (
    df["BoxOffice_Rating"]
    .apply(calculate_verdict)
)


# =========================================================
# 16. CLEAN TEXT COLUMNS
# =========================================================

text_columns = [

    "Movie_Name",
    "Genre",
    "Director",
    "Cast"

]


for column in text_columns:

    df[column] = (

        df[column]

        .fillna("Unknown")

        .astype(str)

        .str.strip()

    )


# =========================================================
# 17. REMOVE DUPLICATES
# =========================================================

df = df.drop_duplicates(

    subset=[
        "Movie_Name",
        "Year"
    ]

)


# =========================================================
# 18. SELECT FINAL COLUMNS
# =========================================================

final_columns = [

    "Movie_Name",

    "Year",

    "Genre",

    "Director",

    "Cast",

    "Budget",

    "Worldwide_Gross",

    "Recovery_Ratio",

    "ROI_Percentage",

    "BoxOffice_Rating",

    "BoxOffice_Verdict"

]


df = df[final_columns]


# =========================================================
# 19. SAVE CLEAN DATASET
# =========================================================

os.makedirs(
    "dataset",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# 20. DISPLAY RESULTS
# =========================================================

print()
print("================================================")
print("       BOX OFFICE DATASET CREATED")
print("================================================")
print()

print(
    "Total movies:",
    len(df)
)

print()

print(
    "Saved file:"
)

print(
    os.path.abspath(
        OUTPUT_FILE
    )
)

print()

print("Columns:")
print(
    df.columns.tolist()
)

print()

print("Box Office Verdict Distribution:")
print()

print(
    df["BoxOffice_Verdict"]
    .value_counts()
)

print()

print("Sample data:")
print()

print(
    df[
        [
            "Movie_Name",
            "Year",
            "Budget",
            "Worldwide_Gross",
            "Recovery_Ratio",
            "BoxOffice_Rating",
            "BoxOffice_Verdict"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print()
print("================================================")
print("              STEP 2 COMPLETE")
print("================================================")