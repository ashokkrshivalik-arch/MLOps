
import os
import sys
import streamlit as st
import joblib
import sklearn

st.set_page_config(page_title="Tourism Model Test")

st.title("Tourism Prediction - Deployment Test")

st.write("Python version:", sys.version)
st.write("Scikit-learn version:", sklearn.__version__)

model_path = os.path.join(
    os.path.dirname(__file__),
    "best_tourism_model_v1.joblib"
)

st.write("Model path:", model_path)
st.write("Model exists:", os.path.isfile(model_path))

if os.path.isfile(model_path):
    st.write(
        "Model size:",
        os.path.getsize(model_path),
        "bytes"
    )

    try:
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
        st.write("Model type:", str(type(model)))

    except Exception as e:
        st.error("Model loading failed")
        st.exception(e)
else:
    st.error("Model file is missing from deployment folder")
