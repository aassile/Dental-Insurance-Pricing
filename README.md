# Dental Insurance Pricing Model with Explainable AI

**Business Impact**: Predictive model used to price dental plans and assess provider network competitiveness — directly informed multi-million-dollar quoting strategy.

## Problem
Accurately predict expected claim cost per member to set competitive yet profitable premiums.

## Features Engineered
- Age/gender banding
- Procedure code frequency vectors
- Provider risk scoring
- Geographic cost indices
- Lag features (prior claims)

## Model
- XGBoost Regressor (best performer)
- RMSE: $187 | R²: 0.89 on holdout
- SHAP for global/local interpretability

## Deployment
Live Streamlit prototype: https://dental-pricing-andrew-assile.streamlit.app

## Tech Stack
Python, Pandas, XGBoost, SHAP, Optuna, Streamlit, Scikit-learn