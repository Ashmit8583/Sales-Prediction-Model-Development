import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('sales_data.csv', parse_dates=['date'], index_col='date')

# Trend and seasonality analysis
df['sales'].plot(figsize=(10, 6))
plt.title('Sales Data')
plt.show()

# Decompose time series
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['sales'], model='additive')
fig = decomposition.plot()
plt.show()

# Train-test split
train = df.iloc[:-12]
test = df.iloc[-12:]

# Train SARIMAX model
model = SARIMAX(train['sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)

# Predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
test['predictions'] = predictions

# Evaluate model
mse = mean_squared_error(test['sales'], test['predictions'])
print(f'Mean Squared Error: {mse}')

# Plot predictions
test[['sales', 'predictions']].plot(figsize=(10, 6))
plt.title('Sales Predictions vs Actual')
plt.show()

# Forecast future sales
forecast = model_fit.get_forecast(steps=12)
forecast_df = forecast.conf_int()
forecast_df['forecast'] = model_fit.predict(start=forecast_df.index[0], end=forecast_df.index[-1])
forecast_df[['forecast']].plot(figsize=(10, 6))
plt.title('12-Month Sales Forecast')
plt.show()
