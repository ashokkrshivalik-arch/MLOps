
import os
import sys
import streamlit as st
import joblib

st.set_page_config(
    page_title="Tourism Predictor",
    layout="wide"
)

st.title("🌴 Wellness Tourism Predictor")

st.success("Streamlit application started successfully!")

st.write("Python version:", sys.version.split()[0])

# Model path
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_tourism_model_v1.joblib"
)

# Check model availability
if not os.path.isfile(MODEL_PATH):
    st.error("Trained model file not found!")
    st.code(MODEL_PATH)
    st.stop()

st.success("Trained model file found!")
st.write(
    "Model size:",
    round(os.path.getsize(MODEL_PATH) / (1024 * 1024), 2),
    "MiB"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    with st.spinner("Loading trained model..."):
        model = load_model()

    st.success("Model loaded successfully!")
    st.write("Model type:", type(model).__name__)

except Exception as e:
    st.error("Model loading failed!")
    st.exception(e)
    st.stop()
