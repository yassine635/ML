import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

df = pd.read_csv("city_temperature.csv", low_memory=False)

df = df[(df["Country"]=="US") & (df["Year"]==2013)]
df=df.iloc[:10000,:]
# ----- cyclic month encoding -----
df["month_sin"] = np.sin(2*np.pi*df["Month"]/12)
df["month_cos"] = np.cos(2*np.pi*df["Month"]/12)

# ----- one hot encode city -----
encoder = OneHotEncoder(sparse_output=False)

city_encoded = encoder.fit_transform(df[["City"]])

city_df = pd.DataFrame(
    city_encoded,
    columns=encoder.get_feature_names_out(["City"]),
    index=df.index
)

# ----- features -----
X = pd.concat([df[["month_sin","month_cos"]], city_df], axis=1)

y = df["AvgTemperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("TRAINING SET METRICS:")
print(f"R² Score: {train_r2:.4f}")
print(f"RMSE: {train_rmse:.4f}")

print("\nTEST SET METRICS:")
print(f"R² Score: {test_r2:.4f}")
print(f"RMSE: {test_rmse:.4f}")