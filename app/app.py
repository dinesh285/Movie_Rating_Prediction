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

.main {
    background-color: #0b1220;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 35px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #334155;
    margin-top: 25px;
}

.rating {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
}

.movie-name {
    font-size: 25px;
    font-weight: 700;
    text-align: center;
}

.status {
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    margin-top: 10px;
}

.watch-button {
    display: inline-block;
    padding: 12px 25px;
    border-radius: 10px;
    text-decoration: none;
    font-weight: 700;
    font-size: 17px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)

    return pd.DataFrame()


try:
    model = load_model()
except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.code(str(e))
    st.stop()

df = load_dataset()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🎬 AI MOVIE RATING PREDICTOR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict movie ratings using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# INFORMATION
# ============================================================

with st.expander("ℹ️ About this Project"):

    st.write("""
    This project predicts movie ratings using Machine Learning.

    **Dataset:** IMDb movie data

    **Languages:** English, Hindi and Telugu

    **Prediction Model:** Gradient Boosting Regressor

    **Other models evaluated:**
    - Ridge Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting

    The model uses movie information such as language, year,
    genre, director, hero, heroine, runtime and expected votes.
    """)

# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("🎥 Enter Movie Details")

col1, col2 = st.columns(2)

with col1:

    movie_name = st.text_input(
        "🎬 Movie Name",
        placeholder="Enter movie name"
    )

    language = st.selectbox(
        "🌐 Language",
        [
            "English",
            "Hindi",
            "Telugu"
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

with col2:

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
# VOTES
# ============================================================

votes = st.number_input(
    "👥 Expected Number of Votes",
    min_value=0,
    value=1000,
    step=100
)

st.caption(
    "💡 Votes are used because the trained ML model includes "
    "movie popularity information."
)

# ============================================================
# PREDICTION
# ============================================================

predict_button = st.button(
    "⭐ PREDICT RATING",
    use_container_width=True
)

if predict_button:

    if not movie_name.strip():

        st.warning("⚠️ Please enter the movie name.")

    else:

        # ----------------------------------------------------
        # CREATE PEOPLE FEATURE
        # ----------------------------------------------------

        director_value = director.strip() if director.strip() else "Unknown"
        hero_value = hero.strip() if hero.strip() else "Unknown"
        heroine_value = heroine.strip() if heroine.strip() else "Unknown"

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
        # PREDICT
        # ----------------------------------------------------

        try:

            prediction = model.predict(input_data)[0]

            # Keep rating between 0 and 10
            prediction = max(0, min(10, prediction))

            prediction = round(prediction, 1)

        except Exception as e:

            st.error("❌ Prediction failed.")
            st.code(str(e))
            st.stop()

        # ----------------------------------------------------
        # RATING CATEGORY
        # ----------------------------------------------------

        if prediction >= 8:

            category = "🟢 Highly Rated"

        elif prediction >= 7:

            category = "🔵 Good Rating"

        elif prediction >= 5:

            category = "🟡 Average Rating"

        else:

            category = "🔴 Low Rating"

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader("🎬 PREDICTION RESULT")

        st.markdown(
            f"""
            <div class="result-box">

                <div class="movie-name">
                    🎬 {movie_name}
                </div>

                <div class="rating">
                    ⭐ {prediction} / 10
                </div>

                <div class="status">
                    {category}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ====================================================
        # WATCH MOVIE
        # ====================================================

        st.markdown("### 🎥 Watch Movie")

        st.info(
            "The button below searches for an official/legal "
            "streaming or rental source. Availability depends "
            "on the movie and your region."
        )

        encoded_movie = urllib.parse.quote(movie_name)

        watch_url = (
            "https://www.google.com/search?q="
            + encoded_movie
            + "+watch+online+official+streaming"
        )

        st.markdown(
            f"""
            <a href="{watch_url}" target="_blank">
                <button style="
                    width:100%;
                    padding:14px;
                    border:none;
                    border-radius:10px;
                    font-size:18px;
                    font-weight:bold;
                    cursor:pointer;
                ">
                    🎥 WATCH {movie_name.upper()}
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.subheader("🤖 Machine Learning Model")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "Best Model",
        "Gradient Boosting"
    )

with info2:
    st.metric(
        "MAE",
        "1.0444"
    )

with info3:
    st.metric(
        "R² Score",
        "0.2565"
    )

st.caption(
    "The displayed metrics are from the project's "
    "time-based test evaluation (2024–2025)."
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        🎬 AI Movie Rating Predictor<br>
        Built using Python • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)