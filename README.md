# Dental Insurance Pricing Model with Explainable AI

> Predicting expected claim cost per member to support competitive, data-driven dental plan pricing — with full SHAP interpretability and a live interactive app.

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)](https://shap.readthedocs.io/)
[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-red?logo=streamlit)](https://dental-pricing-andrew-assile.streamlit.app)

---

## Overview

Dental insurance pricing is fundamentally a prediction problem: how much will a given member cost over the policy period? This project builds a production-grade machine learning pipeline to answer that question, using engineered features drawn from real-world insurance pricing logic (age/gender banding, procedure patterns, provider risk, geography, and claim history).

The model is deployed as a live Streamlit app where users can adjust inputs and receive real-time premium estimates with SHAP-based explanations of each pricing driver.

---

## Live Demo

**[Try the app →](https://dental-pricing-andrew-assile.streamlit.app)**

Enter member characteristics and see:
- Predicted expected claim cost per member per month
- SHAP waterfall chart explaining each feature's contribution to the price
- Global feature importance across the full model

---

## Results

| Metric | Value |
|--------|-------|
| Model | XGBoost Regressor |
| RMSE (holdout) | **$187** |
| R² (holdout) | **0.89** |
| Hyperparameter tuning | Optuna (Bayesian optimization) |
| Explainability | SHAP (global + local) |

---

## Feature Engineering

Insurance pricing intuition was translated directly into model features:

| Feature | Description |
|---------|-------------|
| Age/gender banding | Risk tier based on actuarial age-gender groupings |
| Procedure code frequency vectors | Utilization patterns by CDT procedure category |
| Provider risk scoring | Network provider cost and quality index |
| Geographic cost indices | Regional cost-of-care adjustment factors |
| Lag features | Prior-period claim amounts and frequency |

---

## Project Structure

```
Dental-Insurance-Pricing/
├── app.py                  # Streamlit app — ETL pipeline + real-time predictions + SHAP output
├── src/
│   ├── preprocess.py       # Feature engineering pipeline
│   ├── train.py            # Model training, Optuna tuning, evaluation
│   └── shap_summary.png    # SHAP summary plot (global feature importance)
├── xgb_dental_pricing.pkl  # Trained XGBoost model
└── requirements.txt        # Dependencies
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Modeling | XGBoost, scikit-learn |
| Explainability | SHAP |
| Hyperparameter tuning | Optuna |
| Data processing | pandas, NumPy |
| App deployment | Streamlit |
| Environment | Python 3.13 |

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/aassile/Dental-Insurance-Pricing.git
cd Dental-Insurance-Pricing

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

---

## Background

This project was built to bridge two areas: my professional background in dental insurance analytics at MetLife, and the machine learning toolkit developed through my MS in Data Science. The feature engineering logic reflects real underwriting and pricing considerations, not synthetic tutorial data.

---

## Author

**Andrew Assile** — Data Scientist | Insurance Analytics  
[LinkedIn](https://www.linkedin.com/in/andrew-assile/) · [GitHub](https://github.com/aassile) · [Portfolio](https://aassile.github.io/)
