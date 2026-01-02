# -------------------------------
# Load required libraries
# -------------------------------
import pandas as pd
import numpy as np

# -------------------------------
# Load the final prepared dataset
# -------------------------------
data = pd.read_csv("asansol_crime_final.csv")

# Separate features (X) and target (y)
# ps_name is an identifier, crime_y is the target
X = data.drop(columns=["ps_name", "crime_y"])
y = data["crime_y"]

# -------------------------------
# Feature normalization
# Min–Max scaling ensures all features
# contribute on a comparable scale
# -------------------------------
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

# -------------------------------
# Random Forest Regression
# Baseline ensemble model
# -------------------------------
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,      # number of trees
    random_state=42        # reproducibility
)

# 5-fold cross-validation for stable evaluation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate using RMSE (lower is better)
rmse_scores = -cross_val_score(
    rf,
    X,
    y,
    scoring="neg_root_mean_squared_error",
    cv=cv
)

print("Random Forest RMSE (CV):", rmse_scores.mean())

# -------------------------------
# XGBoost Regression
# Boosting-based ensemble for comparison
# -------------------------------
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=200,      # number of boosting rounds
    learning_rate=0.05,    # step size shrinkage
    max_depth=3,           # shallow trees to prevent overfitting
    subsample=0.8,         # row sampling
    colsample_bytree=0.8,  # feature sampling
    random_state=42
)

# Cross-validated RMSE for XGBoost
rmse_xgb = -cross_val_score(
    xgb,
    X,
    y,
    scoring="neg_root_mean_squared_error",
    cv=cv
)

print("XGBoost RMSE (CV):", rmse_xgb.mean())

# -------------------------------
# Feature importance analysis
# Helps interpret which variables
# influence crime prediction most
# -------------------------------
rf.fit(X, y)

importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importances (Random Forest):")
print(importances)