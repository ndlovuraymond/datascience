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

yearly_ppg = data.groupby(data.index.year).mean()
# Split the data into training and testing sets
train_data = yearly_ppg.iloc[
    :-2
]  # Use data from 2015 onwards except the last 2 seasons for training
test_data = yearly_ppg.iloc[-2:]  # Last 2 years for testing

# Fit the ARIMA model
model = ARIMA(train_data["pts"], order=(11, 1, 1))  # Example order, adjust as needed
model_fit = model.fit()

# Forecast 2 years ahead
forecast = model_fit.forecast(steps=5)  # Forecasting 82 games which is (1 season)
df = forecast.to_frame(name="pts")

# Creating yearly forecast thing for the next year
forecast_dates = pd.Series([2016,2017,2018,2019,2020])
forecast_pts = pd.Series(forecast.array,index=range(0,5))
forecast_df = pd.concat([forecast_pts.rename("pts"),forecast_dates.rename("Date")],axis=1)
forecast_df.set_index("Date",inplace=True)

# Plot the actual and forecasted values
plt.plot(yearly_ppg.index, yearly_ppg["pts"], label="Actual")
plt.plot(forecast_df.index, forecast_df["pts"], label="Forecast")
plt.xlabel("Year")
plt.ylabel("Points Per Game")
plt.title("Time Series Forecast For Lebron's Scoring")
plt.xticks([2005,2010,2015,2020],["2005","2010","2015","2020"])
plt.legend()
plt.show()
