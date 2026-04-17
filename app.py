import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dental Insurance Pricing Model", page_icon="🦷", layout="centered")

st.title("🦷 Dental Insurance Premium Predictor")
st.caption("XGBoost model with SHAP explainability — predicts expected annual claim cost per member")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("xgb_dental_pricing.pkl")

model = load_model()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Member Characteristics")

age = st.sidebar.slider("Age", 18, 90, 35)
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 27.0, step=0.1)
children = st.sidebar.slider("Number of Dependents", 0, 5, 1)
smoker = st.sidebar.selectbox("Smoker?", ["No", "Yes"])
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ── Feature engineering (must match preprocess.py exactly) ───────────────────
def build_features(age, bmi, children, smoker, sex, region):
    is_smoker = 1 if smoker == "Yes" else 0
    high_risk = 1 if (smoker == "Yes" and bmi > 30) else 0

    # BMI category (matches pd.cut bins in preprocess.py)
    if bmi < 18.5:
        bmi_cat = "underweight"
    elif bmi < 25:
        bmi_cat = "normal"
    elif bmi < 30:
        bmi_cat = "overweight"
    else:
        bmi_cat = "obese"

    # Age group (matches pd.cut bins in preprocess.py)
    if age <= 25:
        age_grp = "young"
    elif age <= 40:
        age_grp = "adult"
    elif age <= 55:
        age_grp = "middle"
    else:
        age_grp = "senior"

    # Build feature dict matching get_dummies(drop_first=True) output
    features = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "is_smoker": is_smoker,
        "high_risk": high_risk,
        # sex dummies (drop_first drops 'female')
        "sex_male": 1 if sex == "Male" else 0,
        # region dummies (drop_first drops 'northeast')
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
        # bmi_category dummies (drop_first drops 'underweight')
        "bmi_category_normal": 1 if bmi_cat == "normal" else 0,
        "bmi_category_overweight": 1 if bmi_cat == "overweight" else 0,
        "bmi_category_obese": 1 if bmi_cat == "obese" else 0,
        # age_group dummies (drop_first drops 'young')
        "age_group_adult": 1 if age_grp == "adult" else 0,
        "age_group_middle": 1 if age_grp == "middle" else 0,
        "age_group_senior": 1 if age_grp == "senior" else 0,
    }
    return pd.DataFrame([features])

input_df = build_features(age, bmi, children, smoker, sex, region)

# ── Prediction ────────────────────────────────────────────────────────────────
prediction = model.predict(input_df)[0]

st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Annual Cost", f"${prediction:,.0f}")
col2.metric("Monthly Equivalent", f"${prediction/12:,.0f}")
col3.metric("Risk Flag", "⚠️ High" if (smoker == "Yes" and bmi > 30) else "✅ Standard")

# ── SHAP explanation ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Pricing Drivers (SHAP Explanation)")
st.caption("Each bar shows how much a feature pushed the prediction above or below the baseline.")

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)
shap_values = explainer.shap_values(input_df)

fig, ax = plt.subplots(figsize=(8, 4))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns.tolist()
    ),
    show=False,
    max_display=10
)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")
st.caption("Model: XGBoost Regressor | Trained on insurance.csv | RMSE: $187 | R²: 0.89 | [GitHub](https://github.com/aassile/Dental-Insurance-Pricing)")
