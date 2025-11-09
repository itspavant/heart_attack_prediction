# app.py (Streamlit)
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")

st.title("Heart Attack Risk Predictor")
st.write("Enter patient details and get risk prediction (0 = no, 1 = yes).")

# Load pipeline once
@st.cache_resource
def load_model():
    return pickle.load(open("model_pipeline.pkl", "rb"))

pipe = load_model()

# Input form (use the ORIGINAL feature names used in notebook)
with st.form("input_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=52)
    sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1], index=1)
    cp = st.selectbox("Chest pain type (cp)", [0,1,2,3], index=2)
    trtbps = st.number_input("Resting blood pressure (trtbps)", value=130)
    chol = st.number_input("Cholesterol (chol)", value=250)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0,1], index=0)
    restecg = st.selectbox("Resting ECG (restecg)", [0,1,2], index=1)
    thalachh = st.number_input("Maximum heart rate achieved (thalachh)", value=160)
    exng = st.selectbox("Exercise induced angina (exng)", [0,1], index=0)
    oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", value=1.0, format="%.2f")
    slp = st.selectbox("Slope (slp)", [0,1,2], index=1)
    caa = st.selectbox("Number of major vessels (caa)", [0,1,2,3,4], index=0)
    thall = st.selectbox("Thal (thall)", [0,1,2,3], index=2)
    submit = st.form_submit_button("Predict")

if submit:
    sample = {
        "age": int(age),
        "sex": int(sex),
        "cp": int(cp),
        "trtbps": int(trtbps),
        "chol": int(chol),
        "fbs": int(fbs),
        "restecg": int(restecg),
        "thalachh": int(thalachh),
        "exng": int(exng),
        "oldpeak": float(oldpeak),
        "slp": int(slp),
        "caa": int(caa),
        "thall": int(thall),
    }
    sample_df = pd.DataFrame([sample])
    pred = pipe.predict(sample_df)[0]
    proba = pipe.predict_proba(sample_df)[0][1] if hasattr(pipe, "predict_proba") else None

    st.write("### Result")
    st.write(f"**Prediction (0 = no, 1 = yes):** {int(pred)}")
    if proba is not None:
        st.write(f"**Probability of risk:** {proba:.3f}")
    st.info("Note: This is a demo model — use clinical judgment for real decisions.")
