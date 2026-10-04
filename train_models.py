import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ========================================
# SETTINGS
# ========================================

INPUT_FILE = "dataset/movies_ml_ready.csv"

MODEL_DIR = "model"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


print("========================================")
print("MOVIE RATING PREDICTION - ML TRAINING")
print("========================================")


# ========================================
# 1. LOAD DATASET
# ========================================

print("\nLoading dataset...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    "Total records:",
    len(df)
)


# ========================================
# 2. REMOVE INVALID RECORDS
# ========================================

df = df.dropna(
    subset=[
        "Rating",
        "Year"
    ]
)

df = df[
    df["Rating"].between(
        0,
        10
    )
]


# ========================================
# 3. TRAIN / TEST SPLIT
# ========================================

train_df = df[
    df["Year"] <= 2023
].copy()

test_df = df[
    df["Year"] >= 2024
].copy()


print("\nTraining records:", len(train_df))
print("Testing records:", len(test_df))


# ========================================
# 4. FEATURES
# ========================================

features = [
    "Language",
    "Year",
    "Genre",
    "People",
    "Runtime",
    "Votes"
]

target = "Rating"


X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


# ========================================
# 5. PREPROCESSING
# ========================================

categorical_features = [
    "Language",
    "Genre"
]

numeric_features = [
    "Year",
    "Runtime",
    "Votes"
]

people_features = "People"


# Categorical pipeline

categorical_pipeline = Pipeline(
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


# Numeric pipeline

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ========================================
# 6. COLUMN TRANSFORMER
# ========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),

        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "people",
            TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2)
            ),
            people_features
        )
    ]
)


# ========================================
# 7. MODELS
# ========================================

models = {

    "Ridge Regression":
        Ridge(
            alpha=10
        ),

    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=20,
            min_samples_leaf=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=150,
            max_depth=25,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
}


# ========================================
# 8. TRAIN MODELS
# ========================================

results = []

trained_pipelines = {}


for name, model in models.items():

    print("\n========================================")
    print("Training:", name)
    print("========================================")

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

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    # Keep predictions within rating range
    predictions = np.clip(
        predictions,
        0,
        10
    )

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

    results.append(
        {
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        }
    )

    trained_pipelines[name] = pipeline

    print(
        "MAE:",
        round(mae, 4)
    )

    print(
        "RMSE:",
        round(rmse, 4)
    )

    print(
        "R²:",
        round(r2, 4)
    )


# ========================================
# 9. MODEL COMPARISON
# ========================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="RMSE"
)

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    results_df.to_string(
        index=False
    )
)


# ========================================
# 10. SAVE RESULTS
# ========================================

results_df.to_csv(
    "model/model_comparison.csv",
    index=False
)


# ========================================
# 11. SELECT BEST MODEL
# ========================================

best_model_name = results_df.iloc[0]["Model"]

best_pipeline = trained_pipelines[
    best_model_name
]

print("\n========================================")
print("BEST MODEL")
print("========================================")

print(
    "Best model:",
    best_model_name
)


# ========================================
# 12. SAVE BEST MODEL
# ========================================

joblib.dump(
    best_pipeline,
    "model/movie_rating_model.pkl"
)

print(
    "\nModel saved:"
)

print(
    "model/movie_rating_model.pkl"
)


print("\n========================================")
print("ML TRAINING COMPLETED!")
print("========================================")