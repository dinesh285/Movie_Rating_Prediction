import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from urllib.parse import quote_plus


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Movie Box Office Predictor",
    page_icon="🎬",
    layout="centered"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        margin-top: 10px;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result-card {
        border: 2px solid #444444;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .movie-title {
        font-size: 28px;
        font-weight: 700;
    }

    .rating {
        font-size: 52px;
        font-weight: 800;
        margin: 15px 0;
    }

    .verdict {
        font-size: 30px;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "model",
        "boxoffice_rating_model.pkl"
    )
)


try:

    model = joblib.load(MODEL_PATH)

except Exception as error:

    st.error("❌ Box-office model could not be loaded.")

    st.write("Expected model:")

    st.code(MODEL_PATH)

    st.write("Error:")

    st.code(str(error))

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎬 MOVIE BOX OFFICE PREDICTOR</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">🤖 AI-Powered Box Office Success Prediction</div>',
    unsafe_allow_html=True
)


# =========================================================
# INPUTS
# =========================================================

st.subheader("🎥 Enter Movie Details")


movie_name = st.text_input(
    "🎬 Movie Name",
    placeholder="Enter movie name"
)


language = st.selectbox(
    "🌐 Language",
    [
        "Telugu",
        "Hindi",
        "English",
        "Kannada"
    ]
)


year = st.number_input(
    "📅 Release Year",
    min_value=2005,
    max_value=2035,
    value=2026,
    step=1
)


genre = st.selectbox(
    "🎭 Genre",
    [
        "Action",
        "Comedy",
        "Drama",
        "Romance",
        "Thriller",
        "Horror",
        "Adventure",
        "Fantasy",
        "Crime",
        "Biography",
        "Family",
        "Sports",
        "Musical"
    ]
)


director = st.text_input(
    "🎬 Director",
    placeholder="Enter director name"
)


hero = st.text_input(
    "⭐ Hero",
    placeholder="Enter hero name (optional)"
)


heroine = st.text_input(
    "⭐ Heroine",
    placeholder="Enter heroine name (optional)"
)


budget = st.number_input(
    "💰 Budget (₹ Crore)",
    min_value=1.0,
    max_value=2000.0,
    value=50.0,
    step=1.0
)


# =========================================================
# PREDICT BUTTON
# =========================================================

predict = st.button(
    "⭐ PREDICT BOX OFFICE RATING",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict:

    if movie_name.strip() == "":

        st.warning(
            "⚠️ Please enter the movie name."
        )

        st.stop()


    # -----------------------------------------------------
    # DEFAULT VALUES
    # -----------------------------------------------------

    if director.strip() == "":
        director = "Unknown"

    if hero.strip() == "":
        hero = "Unknown"

    if heroine.strip() == "":
        heroine = "Unknown"


    # -----------------------------------------------------
    # PEOPLE
    # -----------------------------------------------------

    people = (
        director
        + " "
        + hero
        + " "
        + heroine
    )


    # -----------------------------------------------------
    # MODEL INPUT
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        {
            "Year": [year],
            "Genre": [genre],
            "People": [people],
            "Budget": [budget]
        }
    )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

    except Exception as error:

        st.error("❌ Prediction failed.")

        st.code(str(error))

        st.stop()


    prediction = float(
        np.clip(
            prediction,
            0,
            5
        )
    )

    prediction = round(
        prediction,
        1
    )


    # =====================================================
    # VERDICT
    # =====================================================

    if prediction >= 4.5:

        verdict = "🔥 BLOCKBUSTER"

        st.balloons()

        st.success(
            "🎉🎉 BLOCKBUSTER ALERT! 🎉🎉"
        )


    elif prediction >= 4.0:

        verdict = "🚀 SUPER HIT"

        st.balloons()

        st.success(
            "🎉 SUPER HIT ALERT! 🎉"
        )


    elif prediction >= 3.5:

        verdict = "🟢 HIT"

        st.success(
            "👏 HIT ALERT!"
        )


    elif prediction >= 2.5:

        verdict = "🟡 AVERAGE"

        st.info(
            "🎬 AVERAGE BOX OFFICE"
        )


    else:

        verdict = "🔴 FLOP"

        st.error(
            "⚠️ FLOP ALERT"
        )


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("---")

    st.subheader(
        "🎬 PREDICTION RESULT"
    )


    # -----------------------------------------------------
    # USE STREAMLIT NATIVE COMPONENTS
    # -----------------------------------------------------
    # This avoids the raw HTML problem visible in your
    # screenshot.

    with st.container(border=True):

        st.markdown(
            f"### 🎬 {movie_name}"
        )

        st.markdown(
            f"# ⭐ {prediction} / 5"
        )

        st.markdown(
            f"## {verdict}"
        )


    # =====================================================
    # 🎥 MOVIE LINK
    # =====================================================

    st.subheader(
        "🎥 Watch / Find Movie"
    )


    # Google search for legal/official availability
    search_query = quote_plus(
        f"{movie_name} {year} watch online official streaming"
    )


    movie_link = (
        "https://www.google.com/search?q="
        + search_query
    )


    st.link_button(
        "🎥 WATCH / FIND MOVIE",
        movie_link,
        type="primary",
        width="stretch"
    )


    st.caption(
        "The button searches for official/legal streaming "
        "availability. Availability depends on your region."
    )


    # =====================================================
    # MOVIE INFORMATION
    # =====================================================

    st.subheader(
        "📋 Movie Information"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            f"🌐 **Language:** {language}"
        )

        st.write(
            f"📅 **Year:** {year}"
        )

        st.write(
            f"🎭 **Genre:** {genre}"
        )

        st.write(
            f"💰 **Budget:** ₹{budget:.1f} Cr"
        )


    with col2:

        st.write(
            f"🎬 **Director:** {director}"
        )

        st.write(
            f"⭐ **Hero:** {hero}"
        )

        st.write(
            f"⭐ **Heroine:** {heroine}"
        )

        st.write(
            f"⭐ **Rating:** {prediction}/5"
        )


    # =====================================================
    # RESULT MESSAGE
    # =====================================================

    if prediction >= 4.5:

        st.markdown(
            """
            ### 🔥 BLOCKBUSTER!

            🎊 Excellent predicted commercial performance.
            """
        )

    elif prediction >= 4.0:

        st.markdown(
            """
            ### 🚀 SUPER HIT!

            🎊 Strong predicted commercial performance.
            """
        )

    elif prediction >= 3.5:

        st.markdown(
            """
            ### 🟢 HIT!

            👏 Good predicted commercial performance.
            """
        )

    elif prediction >= 2.5:

        st.markdown(
            """
            ### 🟡 AVERAGE

            🎬 Moderate predicted commercial performance.
            """
        )

    else:

        st.markdown(
            """
            ### 🔴 FLOP

            ⚠️ Weak predicted commercial performance.
            """
        )


# =========================================================
# RATING SCALE
# =========================================================

st.markdown("---")

st.subheader(
    "📊 Box Office Rating Scale"
)


st.write(
    "🔥 **4.5 – 5.0** → BLOCKBUSTER"
)

st.write(
    "🚀 **4.0 – 4.4** → SUPER HIT"
)

st.write(
    "🟢 **3.5 – 3.9** → HIT"
)

st.write(
    "🟡 **2.5 – 3.4** → AVERAGE"
)

st.write(
    "🔴 **0.0 – 2.4** → FLOP"
)


# =========================================================
# ABOUT
# =========================================================

st.markdown("---")

st.subheader(
    "ℹ️ About This Project"
)

st.write(
    """
    This project predicts movie box-office success using
    Machine Learning and historical financial performance.
    """
)

st.write(
    """
    The box-office target is derived from the
    Worldwide Gross / Budget recovery ratio.
    """
)

st.write(
    """
    IMDb audience rating is NOT used as the prediction target.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🎬 Movie Box Office Prediction Using Machine Learning • "
    "Python • Scikit-learn • Streamlit"
)