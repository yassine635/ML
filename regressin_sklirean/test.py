import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


df=pd.read_csv("city_temperature.csv")
df=df[(df["Country"]=="US") & (df["Year"]==2013) ]

city=df["City"].unique()

city_id={}
j=1
for i in city:
    city_id[i]=j
    j+=1
df["city_id"] = df["City"].map(city_id)
X = df[['city_id', 'Month']]  # Features (2 columns)
y = df['AvgTemperature']        # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features object
poly = PolynomialFeatures(degree=3, include_bias=False)

# Transform the training data
X_train_poly = poly.fit_transform(X_train)
model = LinearRegression()
model.fit(X_train_poly, y_train)

y_train_pred = model.predict(X_train_poly)

X_test_poly = poly.transform(X_test)

# Make predictions on test data
y_test_pred = model.predict(X_test_poly)

train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Calculate metrics for testing
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("TRAINING SET METRICS:")
print(f"R² Score: {train_r2:.4f}")
print(f"RMSE: {train_rmse:.4f}")