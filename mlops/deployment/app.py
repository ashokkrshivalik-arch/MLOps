
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="🌴",
    layout="wide"
)


# ============================================================
# 1. Load Model from Hugging Face Hub
# ============================================================

REPO_ID = "SagarAtHf/tourismpackagepredict-model"
FILENAME = "productionmodel.joblib"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME
    )
    return joblib.load(model_path)


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# ============================================================
# 2. Application Header
# ============================================================

st.title("🌴 Wellness Tourism Package Predictor")

st.write(
    "Enter the customer details below to predict the likelihood "
    "of purchasing the tourism package."
)


# ============================================================
# 3. Customer Input Form
# ============================================================

with st.form("prediction_form"):

    c1, c2, c3, c4 = st.columns(4)

    # ---------------- COLUMN 1 ----------------
    with c1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        type_of_contact = st.selectbox(
            "Type of Contact",
            ["Self Enquiry", "Company Invited"]
        )

        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        duration_pitch = st.number_input(
            "Duration of Pitch (mins)",
            min_value=0,
            max_value=120,
            value=15
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Salaried",
                "Small Business",
                "Large Business",
                "Free Lancer"
            ]
        )

    # ---------------- COLUMN 2 ----------------
    with c2:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        num_person = st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            max_value=10,
            value=2
        )

        num_followups = st.number_input(
            "Number of Follow-ups",
            min_value=0,
            max_value=10,
            value=3
        )

        product_pitched = st.selectbox(
            "Product Pitched",
            [
                "Basic",
                "Deluxe",
                "Standard",
                "Super Deluxe",
                "King"
            ]
        )

        prop_stars = st.slider(
            "Preferred Property Star",
            min_value=3,
            max_value=5,
            value=3
        )

    # ---------------- COLUMN 3 ----------------
    with c3:

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Married",
                "Unmarried",
                "Divorced"
            ]
        )

        num_trips = st.number_input(
            "Number of Trips",
            min_value=0,
            max_value=20,
            value=1
        )

        passport = st.selectbox(
            "Has Passport?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        pitch_satisfaction = st.slider(
            "Pitch Satisfaction Score",
            min_value=1,
            max_value=5,
            value=3
        )

        own_car = st.selectbox(
            "Owns a Car?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    # ---------------- COLUMN 4 ----------------
    with c4:

        num_children = st.number_input(
            "Number of Children Visiting",
            min_value=0,
            max_value=5,
            value=0
        )

        designation = st.selectbox(
            "Designation",
            [
                "Executive",
                "Manager",
                "Senior Manager",
                "AVP",
                "VP"
            ]
        )

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0,
            value=25000,
            step=1000
        )

    submit = st.form_submit_button(
        "Generate Prediction",
        use_container_width=True
    )


# ============================================================
# 4. Prediction Logic
# ============================================================

if submit:

    # Column names MUST exactly match training data

    data = {

        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_person,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": prop_stars,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }

    input_df = pd.DataFrame([data])

    try:

        # Pipeline automatically performs preprocessing
        probability = model.predict_proba(input_df)[0, 1]

        # Same threshold used during model training
        classification_threshold = 0.45

        st.divider()

        st.subheader("Prediction Result")

        if probability >= classification_threshold:

            st.success(
                f"🎯 High Potential Customer\n\n"
                f"Probability of purchasing package: "
                f"{probability:.2%}"
            )

            st.balloons()

        else:

            st.warning(
                f"⏳ Low Likelihood Customer\n\n"
                f"Probability of purchasing package: "
                f"{probability:.2%}"
            )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

        st.info(
            "Check that the model was trained using the same "
            "feature names and categorical values."
        )
