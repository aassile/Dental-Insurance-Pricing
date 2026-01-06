import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt
from preprocess import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()

model = xgb.XGBRegressor(n_estimators=500,
                         learning_rate=0.05,
                         max_depth=6,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         random_state=42,
                         objective='reg:squarederror',
                         early_stopping_rounds=50,
                         eval_metric='rmse')

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

joblib.dump(model, "models/xgb_dental_pricing.pkl")

# SHAP explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png")
plt.close()