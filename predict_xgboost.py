import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Enable the future behavior to handle downcasting properly
pd.set_option('future.no_silent_downcasting', True)

# Sample DataFrame
weather_ds = pd.read_csv(r"C:\Users\Surface Go\OneDrive\Apps\PythonProjects\GHCND_sample_csv.csv")

# Convert object columns to category type and then to integer codes
weather_ds['STATION'] = weather_ds['STATION'].astype('category').cat.codes
weather_ds['STATION_NAME'] = weather_ds['STATION_NAME'].astype('category').cat.codes

# Handle erroneous data in PRCP column
weather_ds['PRCP'] = weather_ds['PRCP'].replace(9999, pd.NA)
weather_ds['PRCP'] = weather_ds['PRCP'].fillna(weather_ds['PRCP'].median())

# Infer object types to handle the warning
weather_ds = weather_ds.infer_objects(copy=False)

# Define features and target
X = weather_ds.drop(columns=['DATE', 'PRCP'])
y = weather_ds['PRCP']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

# Train the model
model.fit(X_train, y_train)

# Predict using the model
predictions = model.predict(X_test)

print(predictions)

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, predictions)

# Calculate Root Mean Squared Error manually
rmse = np.sqrt(mse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
