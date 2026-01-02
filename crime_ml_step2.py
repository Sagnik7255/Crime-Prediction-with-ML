import pandas as pd

# Load dataset
data = pd.read_csv("asansol_crime2.csv")

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

X = data_scaled[["literacy", "crime_index", "safety_index"]]       #feature
y = data_scaled["crime"]                                           #target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

#Random_Forest_Regressor model-training
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

#Evaluation metrics(rmse & r^2)
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Random Forest RMSE:", rmse)
print("Random Forest R²:", r2)