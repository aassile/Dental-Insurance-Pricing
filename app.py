import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
from src.preprocess import load_and_preprocess

st.title("Dental Insurance Premium Predictor")
st.write("Production-ready pricing model used in underwriting workflow")

model = joblib.load("models/xgb_dental_pricing.pkl")
X_train, _, _, _ = load_and_preprocess()

age = st.slider("Age", 18, 90, 45)
bmi = st.slider("BMI", 15.0, 50.0, 27.0)
children = st.slider("Number of Children", 0, 5, 1)
smoker = st.selectbox("Smoker?", ["No", "Yes"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

input_data = {
    'age': age, 'bmi': bmi, 'children': children,
    'smoker_yes': 1 if smoker == "Yes" else 0,
    'region_northwest': 1 if region == "northwest" else 0,
    'region_southeast': 1 if region == "southeast" else 0,
    'region_southwest': 1 if region == "southwest" else 0,
    'region_northeast': 1 if region == "northeast" else 0
}

import pandas as pd
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]

st.metric("Predicted Annual Premium", f"${prediction:,.0f}")

if st.button("Show SHAP Explanation"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=input_df.iloc[0]), show=False)
    st.pyplot(fig)