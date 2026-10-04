import pandas as pd
import os

INPUT_FILE = "dataset/movies_2005_2025.csv"
OUTPUT_FILE = "dataset/movies_2005_2025_clean.csv"

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original records:", len(df))

# Remove duplicate movies
df = df.drop_duplicates(subset=["tconst"])

# Clean text columns
text_columns = [
    "Movie_Name",
    "Language",
    "Genre",
    "Director",
    "Hero",
    "Heroine"
]

for col in text_columns:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()

# Convert numerical columns
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Runtime"] = pd.to_numeric(df["Runtime"], errors="coerce")
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

# Remove invalid ratings
df = df[
    (df["Rating"] >= 0) &
    (df["Rating"] <= 10)
]

# Remove invalid years
df = df[
    (df["Year"] >= 2005) &
    (df["Year"] <= 2025)
]

# Fill missing numerical values
df["Runtime"] = df["Runtime"].fillna(df["Runtime"].median())
df["Votes"] = df["Votes"].fillna(0)

# Keep only required languages
df = df[
    df["Language"].isin(
        ["Telugu", "Hindi", "English"]
    )
]

# Reset index
df = df.reset_index(drop=True)

# Save cleaned dataset
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("DATASET CLEANED SUCCESSFULLY!")
print("========================================")

print("Output file:", OUTPUT_FILE)
print("Total records:", len(df))

print("\nLanguage distribution:")
print(df["Language"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset preview:")
print(df.head(10).to_string(index=False))