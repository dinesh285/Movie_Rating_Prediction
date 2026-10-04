import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========================================
# MOVIE RATING PREDICTION - EDA
# ========================================

INPUT_FILE = "dataset/movies_2005_2025_clean.csv"

print("========================================")
print("MOVIE RATING DATASET - EDA")
print("========================================")

# ----------------------------------------
# Load Dataset
# ----------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully!")
print("Total Movies:", len(df))

# Create EDA output folder
os.makedirs("eda_results", exist_ok=True)


# ========================================
# 1. LANGUAGE DISTRIBUTION
# ========================================

print("\n========================================")
print("1. MOVIES BY LANGUAGE")
print("========================================")

language_counts = df["Language"].value_counts()

print(language_counts)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Language"
)

plt.title("Number of Movies by Language")
plt.xlabel("Language")
plt.ylabel("Number of Movies")

plt.tight_layout()

plt.savefig(
    "eda_results/language_distribution.png",
    dpi=300
)

plt.close()


# ========================================
# 2. MOVIES BY YEAR
# ========================================

print("\n========================================")
print("2. MOVIES BY YEAR")
print("========================================")

year_counts = (
    df["Year"]
    .value_counts()
    .sort_index()
)

print(year_counts)

plt.figure(figsize=(12, 5))

year_counts.plot(
    kind="bar"
)

plt.title("Number of Movies by Year")
plt.xlabel("Year")
plt.ylabel("Number of Movies")

plt.tight_layout()

plt.savefig(
    "eda_results/movies_by_year.png",
    dpi=300
)

plt.close()


# ========================================
# 3. RATING DISTRIBUTION
# ========================================

print("\n========================================")
print("3. RATING STATISTICS")
print("========================================")

print(
    df["Rating"].describe()
)

plt.figure(figsize=(10, 5))

sns.histplot(
    df["Rating"],
    bins=20,
    kde=True
)

plt.title("Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Movies")

plt.tight_layout()

plt.savefig(
    "eda_results/rating_distribution.png",
    dpi=300
)

plt.close()


# ========================================
# 4. AVERAGE RATING BY LANGUAGE
# ========================================

print("\n========================================")
print("4. AVERAGE RATING BY LANGUAGE")
print("========================================")

language_rating = (
    df.groupby("Language")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

print(
    language_rating
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="Language",
    y="Rating"
)

plt.title("Average Rating by Language")
plt.xlabel("Language")
plt.ylabel("Average Rating")

plt.tight_layout()

plt.savefig(
    "eda_results/average_rating_language.png",
    dpi=300
)

plt.close()


# ========================================
# 5. VOTES VS RATING
# ========================================

print("\n========================================")
print("5. VOTES VS RATING")
print("========================================")

# Use maximum 10,000 records for faster plotting
sample_size = min(
    10000,
    len(df)
)

sample_df = df.sample(
    sample_size,
    random_state=42
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sample_df,
    x="Votes",
    y="Rating",
    alpha=0.4
)

plt.title("Votes vs Movie Rating")
plt.xlabel("Number of Votes")
plt.ylabel("Rating")

plt.tight_layout()

plt.savefig(
    "eda_results/votes_vs_rating.png",
    dpi=300
)

plt.close()


# ========================================
# 6. TOP MOVIE GENRES
# ========================================

print("\n========================================")
print("6. TOP 15 MOVIE GENRES")
print("========================================")

genres = (
    df["Genre"]
    .str.split(",")
    .explode()
    .str.strip()
)

genre_counts = (
    genres
    .value_counts()
    .head(15)
)

print(
    genre_counts
)

plt.figure(figsize=(10, 6))

genre_counts.sort_values().plot(
    kind="barh"
)

plt.title("Top 15 Movie Genres")
plt.xlabel("Number of Movies")
plt.ylabel("Genre")

plt.tight_layout()

plt.savefig(
    "eda_results/top_genres.png",
    dpi=300
)

plt.close()


# ========================================
# 7. LANGUAGE + RATING SUMMARY
# ========================================

print("\n========================================")
print("7. LANGUAGE SUMMARY")
print("========================================")

language_summary = (
    df.groupby("Language")
    .agg(
        Movies=("Movie_Name", "count"),
        Average_Rating=("Rating", "mean"),
        Average_Votes=("Votes", "mean")
    )
    .round(2)
)

print(
    language_summary
)


# ========================================
# 8. TOP 10 HIGHEST RATED MOVIES
# ========================================

print("\n========================================")
print("8. TOP 10 HIGHEST RATED MOVIES")
print("========================================")

top_movies = (
    df[
        [
            "Movie_Name",
            "Language",
            "Year",
            "Rating",
            "Votes"
        ]
    ]
    .sort_values(
        by="Rating",
        ascending=False
    )
    .head(10)
)

print(
    top_movies.to_string(index=False)
)


# ========================================
# 9. DATASET INFORMATION
# ========================================

print("\n========================================")
print("9. DATASET INFORMATION")
print("========================================")

print("\nColumns:")

print(
    df.columns.tolist()
)

print("\nData Types:")

print(
    df.dtypes
)

print("\nMissing Values:")

print(
    df.isnull().sum()
)


# ========================================
# FINAL MESSAGE
# ========================================

print("\n========================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("========================================")

print("\nEDA graphs saved in:")

print(
    "eda_results/"
)

print("\nGenerated files:")

print(
    "1. language_distribution.png"
)

print(
    "2. movies_by_year.png"
)

print(
    "3. rating_distribution.png"
)

print(
    "4. average_rating_language.png"
)

print(
    "5. votes_vs_rating.png"
)

print(
    "6. top_genres.png"
)

print("\nNo graph windows will open.")
print("========================================")