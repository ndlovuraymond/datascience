import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Assuming you have a pandas DataFrame with a 'date' column and a 'value' column
# Make sure the 'date' column is of datetime type and sorted in ascending order

# Read the data into a DataFrame and set 'date' as the index
data = pd.read_csv("Lebron_data.csv")
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)

# Split the data into training and testing sets
train_data = data.iloc[
    -400:-164
]  # Use data from 2015 onwards except the last 2 seasons for training
test_data = data.iloc[-164:]  # Last 2 years for testing

# Fit the ARIMA model
model = ARIMA(train_data["pts"], order=(50, 1, 1))  # Example order, adjust as needed
model_fit = model.fit()

# Forecast 2 years ahead
forecast = model_fit.forecast(steps=164)  # Forecasting 164 games which is (2 years)
df = forecast.to_frame(name=None)

# Create a new DataFrame for the forecasted values
forecast_dates = pd.date_range(start=test_data.index[163], periods=164, freq="5D")
forecast_df = pd.DataFrame(
    df["predicted_mean"], index=forecast_dates, columns=["forecasted_value"]
)

# Plot the actual and forecasted values
plt.plot(data.index, data["pts"], label="Actual")
plt.plot(forecast_df.index, df["predicted_mean"], label="Forecast")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series Forecast")
plt.legend()
plt.show()
