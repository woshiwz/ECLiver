

import streamlit as st
import numpy as np
import xgboost as xgb
import os

st.set_page_config(
    page_title="Esophageal Cancer Liver Metastasis Prediction",
    layout="centered"
)

st.title("Esophageal Cancer Liver Metastasis Prediction")

st.sidebar.header("Clinical Variables")

Age = st.sidebar.selectbox("Age", ["<50", "≥50"])
Sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
First = st.sidebar.selectbox("First", ["No", "Yes"])
Grade = st.sidebar.selectbox(
    "Grade",
    ["Grade I", "Grade II", "Grade III", "Grade IV"]
)
Radiology = st.sidebar.selectbox("Radiology", ["No", "Yes"])
N_stage = st.sidebar.selectbox("N stage", ["N0-1", "N2-3"])
Surgery = st.sidebar.selectbox("Surgery", ["No", "Yes"])

Age_map = {"<50": 0, "≥50": 1}
Sex_map = {"Male": 0, "Female": 1}
First_map = {"No": 0, "Yes": 1}
Grade_map = {
    "Grade I": 0,
    "Grade II": 1,
    "Grade III": 2,
    "Grade IV": 3
}
Radiology_map = {"No": 0, "Yes": 1}
N_stage_map = {"N0-1": 0, "N2-3": 1}
Surgery_map = {"No": 0, "Yes": 1}

features = np.array([[
    Age_map[Age],
    Sex_map[Sex],
    First_map[First],
    Grade_map[Grade],
    Radiology_map[Radiology],
    N_stage_map[N_stage],
    Surgery_map[Surgery]
]], dtype=float)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "modelECLiver.json"
)

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model

if st.button("Predict", type="primary"):

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found: modelECLiver.json")

    else:
        try:
            model = load_model()
            probability = float(
                model.predict_proba(features)[0, 1]
            )

            st.subheader("Prediction Result")
            st.metric(
                "Estimated probability of liver metastasis",
                f"{probability * 100:.2f}%"
            )

            st.caption(
                "This prediction is intended for research use "
                "and requires further clinical validation."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
