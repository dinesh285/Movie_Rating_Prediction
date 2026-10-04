import streamlit as st
import pandas as pd
import joblib
import os
import urllib.parse


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Movie Rating Predictor",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 35px;
}

/* Section headings */
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 25px;
}

/* Prediction result box */
.result-box {
    border: 2px solid #475569;
    border-radius: 18px;
    padding: 30px;
    margin-top: 20px;
    text-align: center;
}

/* Movie name */
.result-movie {
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Rating */
.result-rating {
    font-size: 48px;
    font-weight: 800;
    margin: 10px 0;
}

/* Status */
.result-status {
    font-size: 23px;
    font-weight: 700;
    margin-top: 10px;
}

/* Watch button */
.watch-link {
    display: block;
    text-align: center;
    padding: 14px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 700;
    text-decoration: none;
    margin-top: 20px;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    padding: 20px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "movie_rating_model.pkl"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "movies_ml_ready.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)

    return pd.DataFrame()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

    if model is None:
        st.error(
            "❌ Model file not found: "
            "model/movie_rating_model.pkl"
        )
        st.stop()

except Exception as e:

    st.error("❌ Unable to load the machine learning model.")
    st.code(str(e))
    st.stop()


# Load dataset
df = load_dataset()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎬 AI MOVIE RATING PREDICTOR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict movie ratings using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ABOUT PROJECT
# ============================================================

with st.expander("ℹ️ About This Project"):

    st.markdown("""
### 🎬 AI Movie Rating Predictor

This application predicts the expected rating of a movie
using Machine Learning.

**Dataset:** IMDb movie data

**Languages:**
- English
- Hindi
- Telugu

**Machine Learning Models evaluated:**
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

**Best Model:** Gradient Boosting Regressor

The model uses:

- Language
- Release Year
- Genre
- Director
- Hero
- Heroine
- Runtime
- Expected Votes
""")


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🎥 Enter Movie Details</div>',
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# TWO-COLUMN INPUT LAYOUT
# ============================================================

left_col, right_col = st.columns(2)


# ------------------------------------------------------------
# LEFT COLUMN
# ------------------------------------------------------------

with left_col:

    movie_name = st.text_input(
        "🎬 Movie Name",
        placeholder="Enter movie name"
    )

    language = st.selectbox(
        "🌐 Language",
        [
            "Telugu",
            "Hindi",
            "English"
        ]
    )

    year = st.number_input(
        "📅 Release Year",
        min_value=2005,
        max_value=2026,
        value=2026,
        step=1
    )

    genre = st.selectbox(
        "🎭 Genre",
        [
            "Action",
            "Adventure",
            "Animation",
            "Comedy",
            "Crime",
            "Drama",
            "Family",
            "Fantasy",
            "Horror",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Other"
        ]
    )


# ------------------------------------------------------------
# RIGHT COLUMN
# ------------------------------------------------------------

with right_col:

    director = st.text_input(
        "🎬 Director",
        placeholder="Enter director name"
    )

    hero = st.text_input(
        "⭐ Hero",
        placeholder="Optional"
    )

    heroine = st.text_input(
        "⭐ Heroine",
        placeholder="Optional"
    )

    runtime = st.number_input(
        "⏱️ Runtime (minutes)",
        min_value=30,
        max_value=300,
        value=140,
        step=1
    )


# ============================================================
# EXPECTED VOTES
# ============================================================

votes = st.number_input(
    "👥 Expected Number of Votes",
    min_value=0,
    value=1000,
    step=100
)

st.caption(
    "💡 Expected votes are included because the trained "
    "machine-learning model uses vote count as a feature."
)


# ============================================================
# PREDICT BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "⭐ PREDICT RATING",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CHECK MOVIE NAME
    # --------------------------------------------------------

    if not movie_name.strip():

        st.warning(
            "⚠️ Please enter a movie name."
        )

    else:

        # ----------------------------------------------------
        # HANDLE OPTIONAL FIELDS
        # ----------------------------------------------------

        director_value = (
            director.strip()
            if director.strip()
            else "Unknown"
        )

        hero_value = (
            hero.strip()
            if hero.strip()
            else "Unknown"
        )

        heroine_value = (
            heroine.strip()
            if heroine.strip()
            else "Unknown"
        )

        # ----------------------------------------------------
        # CREATE PEOPLE FEATURE
        # ----------------------------------------------------

        people = (
            director_value
            + " "
            + hero_value
            + " "
            + heroine_value
        )

        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "Language": [language],

            "Year": [year],

            "Genre": [genre],

            "People": [people],

            "Runtime": [runtime],

            "Votes": [votes]

        })

        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        try:

            prediction = model.predict(
                input_data
            )[0]

            # Keep rating between 0 and 10
            prediction = max(
                0,
                min(10, prediction)
            )

            prediction = round(
                prediction,
                1
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.code(str(e))

            st.stop()


        # ====================================================
        # RATING CATEGORY
        # ====================================================

        if prediction >= 8:

            category = "🟢 Highly Rated"

        elif prediction >= 7:

            category = "🔵 Good Rating"

        elif prediction >= 5:

            category = "🟡 Average Rating"

        else:

            category = "🔴 Low Rating"


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '🎬 PREDICTION RESULT'
            '</div>',
            unsafe_allow_html=True
        )

        st.write("")

        # ----------------------------------------------------
        # RESULT CARD
        # ----------------------------------------------------

        st.markdown(
            f"""
<div class="result-box">

<div class="result-movie">
🎬 {movie_name}
</div>

<div class="result-rating">
⭐ {prediction} / 10
</div>

<div class="result-status">
{category}
</div>

</div>
""",
            unsafe_allow_html=True
        )


        # ====================================================
        # WATCH MOVIE
        # ====================================================

        st.write("")

        st.markdown(
            "### 🎥 Watch Movie"
        )

        st.info(
            "Search for an official/legal streaming or "
            "rental source. Availability may depend on "
            "your country and the movie."
        )

        encoded_movie = urllib.parse.quote(
            movie_name
        )

        watch_url = (
            "https://www.google.com/search?q="
            + encoded_movie
            + "+watch+online+official+streaming"
        )

        st.markdown(
            f"""
<a href="{watch_url}" target="_blank"
class="watch-link">
🎥 WATCH {movie_name.upper()}
</a>
""",
            unsafe_allow_html=True
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '🤖 Machine Learning Model'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "Best Model",
        "Gradient Boosting"
    )


with metric2:

    st.metric(
        "MAE",
        "1.0444"
    )


with metric3:

    st.metric(
        "RMSE",
        "1.3460"
    )


with metric4:

    st.metric(
        "R² Score",
        "0.2565"
    )


st.caption(
    "Evaluation performed using a time-based test split "
    "(training: 2005–2023, testing: 2024–2025)."
)


# ============================================================
# PROJECT TECHNOLOGY
# ============================================================

with st.expander("🛠️ Technologies Used"):

    st.markdown("""
### Programming
- Python

### Machine Learning
- Scikit-learn
- Gradient Boosting Regressor
- Random Forest Regressor
- Decision Tree Regressor
- Ridge Regression

### Data Processing
- Pandas
- NumPy

### Model Saving
- Joblib

### Web Application
- Streamlit

### Dataset
- IMDb datasets
""")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

🎬 <b>AI Movie Rating Predictor</b><br>

Built using Python • Machine Learning • Scikit-learn • Streamlit

</div>
""",
    unsafe_allow_html=True
)