import pandas as pd

# Load dataset
data = pd.read_csv("asansol_crime3.csv")

print(data)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Normalize both columns
data_scaled = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns
)

print(data_scaled)

#Train-Test split(4:1)
from sklearn.model_selection import train_test_split

X = data_scaled[["literacy", "industrial_intensity"]]   # feature
y = data_scaled["crime"]        # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

#Model training
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=100,      # number of trees
    learning_rate=0.1,     # how fast model learns
    max_depth=3,           # tree depth (keep small for small data)
    subsample=0.8,         # use 80% of data per tree
    colsample_bytree=0.8,  # use 80% of features per tree
    random_state=42
)

xgb.fit(X_train, y_train)

#Evaluation metrics(rmse & r^2)
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred_xgb = xgb.predict(X_test)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost RMSE:", rmse_xgb)
print("XGBoost R²:", r2_xgb)