import os
import streamlit as st
import pandas as pd
import joblib

# Ensure st.set_page_config() is the very first Streamlit command
st.set_page_config(page_title="Tourism Predictor", layout="wide")

# 1. Load Model
# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_medical_insurance_model_v1.joblib")
model = joblib.load(model_path)

st.title("🌴 Full Feature Wellness Tourism Predictor")
st.write("Please fill in all 19 parameters to get an accurate prediction.")

# 2. Complete Form with All 19 Features
with st.form("prediction_form"):
    # Using 4 columns to fit all 19 features neatly
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        age = st.number_input("Age", 18, 100, 30)
        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        duration_pitch = st.number_input("Duration of Pitch (mins)", 0, 120, 15)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])

    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        num_person = st.number_input("Number of Persons Visiting", 1, 10, 2)
        num_followups = st.number_input("Number of Follow-ups", 1, 10, 3)
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        prop_stars = st.slider("Preferred Property Star", 3, 5, 3)

    with c3:
        marital_status = st.selectbox("Marital Status", ["Married", "Unmarried", "Divorced"])
        num_trips = st.number_input("Number of Trips", 1, 20, 1)
        passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    with c4:
        num_children = st.number_input("Number of Children", 0, 5, 0)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income", value=25000)

    submit = st.form_submit_button("Generate Prediction")

# 3. Prediction Logic
if submit:
    # IMPORTANT: Dictionary keys must match the EXACT column names used during training
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

    # Get the probability
    try:
        # Note: Pipeline applies ColumnTransformer automatically
        prob = model.predict_proba(input_df)[0][1]

        st.divider()
        if prob >= 0.45:
            st.success(f"### Result: 🎯 High Potential (Prob: {prob:.2%})")
            st.balloons()
        else:
            st.warning(f"### Result: ⏳ Low Likelihood (Prob: {prob:.2%})")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Ensure the column names in app.py match your training data exactly.")
