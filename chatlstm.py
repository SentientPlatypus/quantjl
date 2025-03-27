import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import requests
import json
import platform

dsp = "/"
if platform.system() == "Windows":
    dsp = "\\"
FMP_API_KEY = open(f".{dsp}apikey", "r").readline().strip()

def fetch_intraday(ticker: str, interval: str = "5min"):
    """
    Fetch intraday stock data from Financial Modeling Prep (FMP) API.
    Args:
        ticker (str): Stock ticker symbol.
        interval (str): Time interval for intraday data (e.g., "1min", "5min", "15min").
    Returns:
        pd.DataFrame: DataFrame containing intraday stock prices.
    """
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{ticker}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if not data or not isinstance(data, list):  # Ensure we received a list of records
        raise ValueError(f"Failed to fetch intraday data for {ticker}. Response: {data}")
    
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df

# 1. Get the data from FMP intraday
ticker = 'AAPL'
df = fetch_intraday(ticker)

# Plot the full "close" series
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['close'], label='Close Price')
plt.title(f"{ticker} Intraday Close Prices")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# Preprocessing: Create sequences for training/testing
def create_dataset(df: pd.DataFrame, n_lookback=100, n_forecast=10):
    """
    Creates dataset with sliding windows.
    Each input sample is a sequence of [open, high, low, close] values of length n_lookback,
    and each target is the subsequent n_forecast close values.
    """
    X, y = [], []
    for i in range(len(df) - n_lookback - n_forecast):
        # Use all four features for the lookback window
        X.append(df[['open','high','low','close']].iloc[i:i+n_lookback].values)
        # Target: next n_forecast close values (only the 'close' column)
        y.append(df['close'].iloc[i+n_lookback:i+n_lookback+n_forecast].values)
    return np.array(X), np.array(y)

# Scale the data.
# We'll use a single scaler for all features for the input.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df[['open', 'high', 'low', 'close']])
df_scaled = pd.DataFrame(scaled_features, columns=['open','high','low','close'], index=df.index)

# Create the dataset with lookback=100 and forecast=10
n_lookback = 100
n_forecast = 2
X, y = create_dataset(df_scaled, n_lookback, n_forecast)

# Split the dataset into training and testing sets (for example, 80% training)
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# 2. Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_lookback, 4)))
model.add(Dense(n_forecast))
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model (adjust epochs and batch_size as needed)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 3. Test the model against an earlier time step

# For testing, pick one sample from the test set.
# You could also choose a specific index from X (e.g., an earlier time slice)
test_sample_idx = 0
X_sample = X_test[test_sample_idx:test_sample_idx+1]
y_true_scaled = y_test[test_sample_idx]  # these are scaled values

# Predict the next 10 close values
y_pred_scaled = model.predict(X_sample)

# Since we scaled all four columns together, we need to invert the scaling for the close values only.
# The "close" column is the 4th feature in our original DataFrame.
close_min = scaler.data_min_[3]
close_max = scaler.data_max_[3]

# Inverse transform the scaled predictions and true values for the close column.
y_pred = y_pred_scaled * (close_max - close_min) + close_min
y_true = y_true_scaled * (close_max - close_min) + close_min

# Create a range for plotting (e.g., minute indices relative to the start of the forecast)
forecast_range = np.arange(n_lookback, n_lookback+n_forecast)

plt.figure(figsize=(10, 5))
plt.plot(forecast_range, y_true, marker='o', label="Actual Close")
plt.plot(forecast_range, y_pred.flatten(), marker='o', label="Predicted Close", linestyle='--')
plt.xlabel("Time step (relative to forecast start)")
plt.ylabel("Close Price")
plt.title("Next 10 Minute Close Predictions vs Actual")
plt.legend()
plt.grid(True)
plt.show()