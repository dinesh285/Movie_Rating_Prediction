import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# =========================================================
# SETTINGS
# =========================================================

DATA_FILE = "dataset/boxoffice_clean.csv"

MODEL_FILE = "model/boxoffice_rating_model.pkl"

RESULT_FILE = "model/boxoffice_model_comparison.csv"


# =========================================================
# START
# =========================================================

print()
print("================================================")
print("       BOX OFFICE ML MODEL TRAINING")
print("================================================")
print()


# =========================================================
# 1. CHECK DATASET
# =========================================================

if not os.path.exists(DATA_FILE):

    print("ERROR: Dataset not found.")

    print()
    print("Expected:")
    print(DATA_FILE)

    print()
    print("Run this first:")
    print("python prepare_boxoffice.py")

    exit()


# =========================================================
# 2. LOAD DATASET
# =========================================================

print("Loading box-office dataset...")

df = pd.read_csv(
    DATA_FILE
)

print()

print(
    "Total rows:",
    len(df)
)

print()


# =========================================================
# 3. CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [

    "Year",
    "Genre",
    "Director",
    "Cast",
    "Budget",
    "BoxOffice_Rating"

]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    print("ERROR: Missing columns:")

    for column in missing_columns:
        print("-", column)

    exit()


# =========================================================
# 4. CLEAN NUMERIC DATA
# =========================================================

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df["Budget"] = pd.to_numeric(
    df["Budget"],
    errors="coerce"
)

df["BoxOffice_Rating"] = pd.to_numeric(
    df["BoxOffice_Rating"],
    errors="coerce"
)


# =========================================================
# 5. CLEAN TEXT DATA
# =========================================================

df["Genre"] = (

    df["Genre"]

    .fillna("Unknown")

    .astype(str)

)


df["Director"] = (

    df["Director"]

    .fillna("Unknown")

    .astype(str)

)


df["Cast"] = (

    df["Cast"]

    .fillna("Unknown")

    .astype(str)

)


# =========================================================
# 6. CREATE PEOPLE FEATURE
# =========================================================

print("Creating People feature...")

df["People"] = (

    df["Director"]

    + " "

    + df["Cast"]

)


# =========================================================
# 7. REMOVE INVALID ROWS
# =========================================================

df = df.dropna(

    subset=[

        "Year",

        "Budget",

        "BoxOffice_Rating"

    ]

)


# =========================================================
# 8. REMOVE INVALID BUDGET
# =========================================================

df = df[
    df["Budget"] > 0
]


# =========================================================
# 9. DISPLAY TRAINING SIZE
# =========================================================

print()

print(
    "Rows available for training:",
    len(df)
)

print()


# =========================================================
# 10. FEATURES
# =========================================================

X = df[

    [

        "Year",

        "Genre",

        "People",

        "Budget"

    ]

]


# =========================================================
# 11. TARGET
# =========================================================

# IMPORTANT:
#
# This is NOT IMDb Rating.
#
# This is the box-office rating calculated from
# Worldwide Gross / Budget.

y = df[
    "BoxOffice_Rating"
]


# =========================================================
# 12. TRAIN / TEST SPLIT
# =========================================================

print(
    "Splitting dataset..."
)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42

)


print()

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)

print()


# =========================================================
# 13. FEATURE TYPES
# =========================================================

numeric_features = [

    "Year",

    "Budget"

]


categorical_features = [

    "Genre"

]


people_feature = "People"


# =========================================================
# 14. NUMERIC PIPELINE
# =========================================================

numeric_transformer = Pipeline(

    steps=[

        (

            "imputer",

            SimpleImputer(

                strategy="median"

            )

        )

    ]

)


# =========================================================
# 15. CATEGORY PIPELINE
# =========================================================

categorical_transformer = Pipeline(

    steps=[

        (

            "imputer",

            SimpleImputer(

                strategy="most_frequent"

            )

        ),

        (

            "onehot",

            OneHotEncoder(

                handle_unknown="ignore"

            )

        )

    ]

)


# =========================================================
# 16. PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (

            "numeric",

            numeric_transformer,

            numeric_features

        ),

        (

            "category",

            categorical_transformer,

            categorical_features

        ),

        (

            "people",

            TfidfVectorizer(

                max_features=3000,

                ngram_range=(1, 2)

            ),

            people_feature

        )

    ]

)


# =========================================================
# 17. MACHINE LEARNING MODELS
# =========================================================

models = {

    "Ridge Regression":

        Ridge(

            alpha=10.0

        ),


    "Decision Tree":

        DecisionTreeRegressor(

            max_depth=12,

            random_state=42

        ),


    "Random Forest":

        RandomForestRegressor(

            n_estimators=200,

            max_depth=15,

            random_state=42,

            n_jobs=-1

        ),


    "Gradient Boosting":

        GradientBoostingRegressor(

            n_estimators=200,

            learning_rate=0.05,

            max_depth=3,

            random_state=42

        )

}


# =========================================================
# 18. TRAIN MODELS
# =========================================================

results = []


best_model = None

best_model_name = None

best_mae = float("inf")


print()
print("================================================")
print("              MODEL TRAINING")
print("================================================")
print()


for model_name, model in models.items():

    print(
        "Training:",
        model_name
    )

    print(
        "Please wait..."
    )


    # -----------------------------------------------------
    # PIPELINE
    # -----------------------------------------------------

    pipeline = Pipeline(

        steps=[

            (

                "preprocessor",

                preprocessor

            ),

            (

                "model",

                model

            )

        ]

    )


    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    pipeline.fit(

        X_train,

        y_train

    )


    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    predictions = pipeline.predict(

        X_test

    )


    # Keep prediction between 0 and 5

    predictions = np.clip(

        predictions,

        0,

        5

    )


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    mae = mean_absolute_error(

        y_test,

        predictions

    )


    rmse = np.sqrt(

        mean_squared_error(

            y_test,

            predictions

        )

    )


    r2 = r2_score(

        y_test,

        predictions

    )


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    print()

    print(
        "MAE :",
        round(mae, 4)
    )

    print(
        "RMSE:",
        round(rmse, 4)
    )

    print(
        "R2  :",
        round(r2, 4)
    )

    print()

    print("----------------------------------------")


    # -----------------------------------------------------
    # SAVE RESULT
    # -----------------------------------------------------

    results.append(

        {

            "Model":
            model_name,

            "MAE":
            mae,

            "RMSE":
            rmse,

            "R2":
            r2

        }

    )


    # -----------------------------------------------------
    # SELECT BEST MODEL
    # -----------------------------------------------------

    if mae < best_mae:

        best_mae = mae

        best_model = pipeline

        best_model_name = model_name


# =========================================================
# 19. CREATE MODEL DIRECTORY
# =========================================================

os.makedirs(

    "model",

    exist_ok=True

)


# =========================================================
# 20. SAVE MODEL COMPARISON
# =========================================================

results_df = pd.DataFrame(

    results

)


results_df = results_df.sort_values(

    by="MAE"

)


results_df.to_csv(

    RESULT_FILE,

    index=False

)


# =========================================================
# 21. SAVE BEST MODEL
# =========================================================

joblib.dump(

    best_model,

    MODEL_FILE

)


# =========================================================
# 22. FINAL RESULT
# =========================================================

print()
print("================================================")
print("       MODEL TRAINING COMPLETED")
print("================================================")
print()

print(
    results_df.to_string(
        index=False
    )
)

print()

print(
    "BEST MODEL:"
)

print(
    best_model_name
)

print()

print(
    "Best MAE:",
    round(
        best_mae,
        4
    )
)

print()

print(
    "Model saved:"
)

print(
    os.path.abspath(
        MODEL_FILE
    )
)

print()

print(
    "Comparison saved:"
)

print(
    os.path.abspath(
        RESULT_FILE
    )
)

print()
print("================================================")
print("              STEP 4 COMPLETE")
print("================================================")