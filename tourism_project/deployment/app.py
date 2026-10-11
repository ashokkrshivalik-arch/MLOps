import os
import streamlit as st
import joblib

st.set_page_config(
    page_title="Tourism Predictor",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "best_tourism_model_v1.joblib"
    )

    if not os.path.isfile(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

st.title("🌴 Wellness Tourism Predictor")
st.success("Model loaded successfully!")
