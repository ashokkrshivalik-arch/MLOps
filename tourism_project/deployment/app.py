
import os
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Tourism Prediction App",
    layout="wide"
)


# Model location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_tourism_model_v1.joblib"
)

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.exception(e)
    st.stop()

# Streamlit UI for Tourism Package Prediction
st.title("Wellness Tourism Package Prediction App")

st.write("""
This application predicts the **probability of a customer purchasing
a Wellness Tourism Package** based on customer information and
sales interaction details.

Please enter the required information below to get a prediction.
""")

# User input
age = st.number_input(
    "Age", min_value=18, max_value=100, value=35, step=1
)

contact_type = st.selectbox(
    "Type of Contact", ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox(
    "City Tier", [1, 2, 3]
)

duration_pitch = st.number_input(
    "Duration of Pitch (Minutes)",
    min_value=1, max_value=120, value=15
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox(
    "Gender", ["Male", "Female"]
)

persons_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1, max_value=20, value=2
)

followups = st.number_input(
    "Number of Follow-ups",
    min_value=0, max_value=10, value=3
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

property_star = st.selectbox(
    "Preferred Property Star", [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

number_trips = st.number_input(
    "Number of Trips",
    min_value=0, max_value=30, value=3
)

passport = st.selectbox(
    "Passport", [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

satisfaction = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
)

own_car = st.selectbox(
    "Own Car", [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0, max_value=10, value=1
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000, max_value=1000000,
    value=25000, step=1000
)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contact_type,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons_visiting,
    "NumberOfFollowups": followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_trips,
    "Passport": passport,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

# Add index column only if required by the trained pipeline
input_data["__index_level_0__"] = 0

# Prediction
if st.button("Predict Tourism Package Purchase"):

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result:")

        if int(prediction) == 1:
            st.success(
                "Customer is likely to purchase the "
                "Wellness Tourism Package."
            )
        else:
            st.warning(
                "Customer is unlikely to purchase the "
                "Wellness Tourism Package."
            )

        # Purchase probability
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)

            positive_class = (
                1 if 1 in classes else "1"
            )

            positive_index = classes.index(positive_class)

            probability = model.predict_proba(input_data)[0][positive_index]

            st.write(
                f"**Purchase Probability: {probability:.2%}**"
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
