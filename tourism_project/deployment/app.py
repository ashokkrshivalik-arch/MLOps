
import os
import joblib
import pandas as pd
import streamlit as st

# =====================================================
# 1. Page configuration
# =====================================================

st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="🌴",
    layout="wide"
)

# =====================================================
# 2. Load trained model
# =====================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_tourism_model_v1.joblib"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()

# =====================================================
# 3. Application heading
# =====================================================

st.title("🌴 Wellness Tourism Package Predictor")

st.markdown(
    """
    Predict whether a customer is likely to purchase
    a wellness tourism package based on demographic,
    travel and sales interaction information.
    """
)

st.divider()

# =====================================================
# 4. Customer input form
# =====================================================

with st.form("tourism_prediction_form"):

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Married", "Unmarried", "Divorced"]
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

    with col2:
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
            min_value=0.0,
            max_value=1000000.0,
            value=25000.0,
            step=1000.0
        )

        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3]
        )

        own_car = st.selectbox(
            "Own Car",
            ["No", "Yes"]
        )

    with col3:
        passport = st.selectbox(
            "Passport",
            ["No", "Yes"]
        )

        number_of_trips = st.number_input(
            "Number of Trips per Year",
            min_value=0,
            max_value=30,
            value=3
        )

        number_of_person_visiting = st.number_input(
            "Number of Persons Visiting",
            min_value=1,
            max_value=20,
            value=3
        )

        number_of_children_visiting = st.number_input(
            "Number of Children Visiting",
            min_value=0,
            max_value=10,
            value=1
        )

    st.divider()

    st.subheader("🏖️ Tourism Package Information")

    col4, col5 = st.columns(2)

    with col4:
        product_pitched = st.selectbox(
            "Product Pitched",
            [
                "Basic",
                "Standard",
                "Deluxe",
                "Super Deluxe",
                "King"
            ]
        )

        preferred_property_star = st.selectbox(
            "Preferred Property Star",
            [3, 4, 5]
        )

        type_of_contact = st.selectbox(
            "Type of Contact",
            [
                "Self Enquiry",
                "Company Invited"
            ]
        )

    with col5:
        duration_of_pitch = st.number_input(
            "Duration of Pitch (Minutes)",
            min_value=0,
            max_value=120,
            value=15
        )

        number_of_followups = st.number_input(
            "Number of Follow-ups",
            min_value=0,
            max_value=20,
            value=3
        )

        pitch_satisfaction_score = st.slider(
            "Pitch Satisfaction Score",
            min_value=1,
            max_value=5,
            value=3
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔮 Predict Purchase",
        type="primary",
        use_container_width=True
    )

# =====================================================
# 5. Prepare prediction data
# =====================================================

if submitted:

    input_data = pd.DataFrame([{

        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income

    }])

    # Convert numeric columns to numeric types
    numeric_columns = [
        "Age",
        "CityTier",
        "DurationOfPitch",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "PreferredPropertyStar",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "MonthlyIncome"
    ]

    input_data[numeric_columns] = input_data[
        numeric_columns
    ].apply(pd.to_numeric)

    # =================================================
    # 6. Make prediction
    # =================================================

    try:

        probability = model.predict_proba(input_data)[0][1]

        # Same threshold used during model evaluation
        threshold = 0.45

        prediction = int(probability >= threshold)

        st.divider()
        st.subheader("📊 Prediction Results")

        col_result1, col_result2 = st.columns(2)

        with col_result1:
            st.metric(
                "Purchase Probability",
                f"{probability * 100:.2f}%"
            )

        with col_result2:
            st.metric(
                "Prediction Threshold",
                f"{threshold * 100:.0f}%"
            )

        st.progress(float(probability))

        if prediction == 1:

            st.success(
                "✅ Customer is predicted to PURCHASE "
                "the tourism package."
            )

            st.info(
                "Recommendation: Prioritize this customer "
                "for package booking and follow-up."
            )

        else:

            st.warning(
                "⚠️ Customer is predicted NOT TO PURCHASE "
                "the tourism package."
            )

            st.info(
                "Recommendation: Consider additional "
                "engagement or a more suitable package."
            )

        with st.expander("View Customer Input Data"):
            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as e:

        st.error("Prediction failed.")
        st.exception(e)

# =====================================================
# 7. Footer
# =====================================================

st.divider()

st.caption(
    "Wellness Tourism Prediction | "
    "Machine Learning Deployment using Streamlit"
)
