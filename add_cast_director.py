import pandas as pd
import os

DATASET_DIR = "dataset"

BASIC_FILE = os.path.join(
    DATASET_DIR,
    "movies_2005_2025_basic.csv"
)

CREW_FILE = os.path.join(
    DATASET_DIR,
    "title.crew.tsv.gz"
)

PRINCIPALS_FILE = os.path.join(
    DATASET_DIR,
    "title.principals.tsv.gz"
)

NAMES_FILE = os.path.join(
    DATASET_DIR,
    "name.basics.tsv.gz"
)

print("Loading basic movie dataset...")

movies = pd.read_csv(BASIC_FILE)

print("Movies:", len(movies))

movie_ids = set(movies["tconst"])

# --------------------------------------------------
# STEP 1: DIRECTORS
# --------------------------------------------------

print("\nReading directors...")

director_data = {}

for chunk in pd.read_csv(
    CREW_FILE,
    sep="\t",
    compression="gzip",
    na_values="\\N",
    chunksize=100000
):
    chunk = chunk[chunk["tconst"].isin(movie_ids)]

    for _, row in chunk.iterrows():

        directors = row["directors"]

        if pd.notna(directors):
            director_data[row["tconst"]] = directors

    print(
        "Processed crew chunk:",
        len(chunk)
    )

movies["Director_ID"] = movies["tconst"].map(
    director_data
)

print(
    "Movies with director information:",
    movies["Director_ID"].notna().sum()
)

# --------------------------------------------------
# STEP 2: PRINCIPAL CAST
# --------------------------------------------------

print("\nReading principal cast...")

cast_data = {}

for chunk in pd.read_csv(
    PRINCIPALS_FILE,
    sep="\t",
    compression="gzip",
    na_values="\\N",
    chunksize=200000
):

    chunk = chunk[
        chunk["tconst"].isin(movie_ids)
    ]

    # Keep actors and actresses only
    chunk = chunk[
        chunk["category"].isin(
            ["actor", "actress"]
        )
    ]

    for movie_id, group in chunk.groupby("tconst"):

        cast = []

        for _, row in group.iterrows():

            cast.append(
                (
                    row["nconst"],
                    row["category"],
                    row["ordering"]
                )
            )

        cast_data[movie_id] = cast

    print(
        "Processed cast chunk:",
        len(chunk)
    )

# --------------------------------------------------
# STEP 3: LOAD NAMES
# --------------------------------------------------

print("\nLoading names...")

names = pd.read_csv(
    NAMES_FILE,
    sep="\t",
    compression="gzip",
    na_values="\\N",
    usecols=[
        "nconst",
        "primaryName"
    ]
)

name_map = dict(
    zip(
        names["nconst"],
        names["primaryName"]
    )
)

print(
    "Names loaded:",
    len(name_map)
)

# --------------------------------------------------
# STEP 4: EXTRACT HERO / HEROINE
# --------------------------------------------------

print("\nExtracting Hero and Heroine...")

hero_data = {}
heroine_data = {}

for movie_id, cast_list in cast_data.items():

    # Sort by IMDb ordering
    cast_list = sorted(
        cast_list,
        key=lambda x: x[2]
    )

    hero = None
    heroine = None

    for nconst, category, ordering in cast_list:

        name = name_map.get(nconst)

        if not name:
            continue

        if category == "actor" and hero is None:
            hero = name

        elif category == "actress" and heroine is None:
            heroine = name

    hero_data[movie_id] = hero
    heroine_data[movie_id] = heroine

movies["Hero"] = movies["tconst"].map(
    hero_data
)

movies["Heroine"] = movies["tconst"].map(
    heroine_data
)

# --------------------------------------------------
# STEP 5: DIRECTOR NAME
# --------------------------------------------------

def get_first_person(value):

    if pd.isna(value):
        return None

    first_id = str(value).split(",")[0]

    return name_map.get(first_id)


movies["Director"] = movies["Director_ID"].apply(
    get_first_person
)

# --------------------------------------------------
# STEP 6: CLEAN
# --------------------------------------------------

movies.drop(
    columns=["Director_ID"],
    inplace=True
)

movies["Hero"] = movies["Hero"].fillna("Unknown")
movies["Heroine"] = movies["Heroine"].fillna("Unknown")
movies["Director"] = movies["Director"].fillna("Unknown")

# --------------------------------------------------
# STEP 7: SAVE
# --------------------------------------------------

output_file = os.path.join(
    DATASET_DIR,
    "movies_2005_2025_with_cast.csv"
)

movies.to_csv(
    output_file,
    index=False
)

print("\n========================================")
print("CAST AND DIRECTOR DATASET CREATED!")
print("========================================")

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

print("\nSample records:")

print(
    movies[
        [
            "Movie_Name",
            "Year",
            "Genre",
            "Director",
            "Hero",
            "Heroine",
            "Rating"
        ]
    ].head(10).to_string(index=False)
)