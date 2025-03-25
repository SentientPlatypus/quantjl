import pandas as pd
import numpy as np


from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import platform
import requests
import matplotlib.pyplot as plt

dsp = "/"
if platform.system() == "Windows":
    dsp = "\\"
FMP_API_KEY = open(f".{dsp}apikey", "r").readline().strip()

def fetch_stock_data_fmp(ticker: str, period: str = "6mo"):
    """
    Fetch historical stock data from Financial Modeling Prep (FMP) API.
    Args:
        ticker (str): Stock ticker symbol.
        period (str): The period of historical data to fetch (e.g., "6mo", "max").
    Returns:
        pd.DataFrame: DataFrame containing historical stock prices.
    """
    # Map period to actual date ranges
    if period == "max":
        years = 20  # Get 20 years of data
    elif period == "6mo":
        years = 1   # Get 1 year of data (as FMP does not support exact months)
    else:
        years = 5   # Default to 5 years if period is not recognized

    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries={years * 365}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "historical" not in data:
        raise ValueError(f"Failed to fetch data for {ticker}. Response: {data}")

    df = pd.DataFrame(data["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    return df

def trainmodel(ticker:str):
    toDisplay = fetch_stock_data_fmp(ticker, period="max")
    df = fetch_stock_data_fmp(ticker, period="6mo")

    print(df)
    y = df['close'].fillna(method='ffill')
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 60  # length of input sequences (lookback period)
    n_forecast = 30  # length of output sequences (forecast period)
    X = []
    Y = []
    
    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])
    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=75, batch_size=32, verbose=0)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    

    # organize the results in a data frame
    df_past:pd.Dataframe = df[['close']].reset_index()
    df_past.rename(columns={'index': 'date', 'close': 'actual'}, inplace=True)
    df_past['date'] = pd.to_datetime(df_past['date'])
    df_past['forecast'] = np.nan
    df_past.loc[df_past.index[-1], 'forecast'] = df_past.loc[df_past.index[-1], 'actual']


    df_future = pd.DataFrame(columns=['date', 'actual', 'forecast'])
    df_future['date'] = pd.date_range(start=df_past['date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['forecast'] = Y_.flatten()
    df_future['actual'] = np.nan


    print(df_future)
    print(df_past)


    results: pd.DataFrame = pd.concat([df_past, df_future]).set_index('date')
    
    results = results[results["forecast"].notna()]
    results["open"] = results["forecast"]
    results["high"] = results["forecast"]
    results["low"] = results["forecast"]
    resultsShifted = results.shift(-1)
    results["close"] = resultsShifted["open"]
    final = pd.concat([toDisplay, results], sort=False, join="inner")
    return model, final
    

def testmodel(model, ticker: str, test_start_date: str = None, n_lookback=60, n_forecast=30):
    """
    Tests the provided model by comparing a 30-day forecast (starting from a chosen test date)
    with the actual historical data.
    
    Parameters:
    - model: A trained Keras model for forecasting.
    - ticker: Stock ticker symbol to fetch historical data.
    - test_start_date: Optional; a string date (e.g., "2023-01-01") indicating the forecast start date.
                       If None, a default date is chosen to ensure sufficient data.
    """
    
    # Fetch the full historical data (assumed to have a DateTime index)
    df_full = fetch_stock_data_fmp(ticker, period="max")
    
    # Ensure the index is datetime and sorted in ascending order
    if not np.issubdtype(df_full.index.dtype, np.datetime64):
        df_full.index = pd.to_datetime(df_full.index)
    df_full.sort_index(inplace=True)
    
    # If no test start date is provided, choose one that allows for a full 60-day lookback and 30-day forecast
    if test_start_date is None:
        # For instance, choose 90 days before the last available date
        last_date = df_full.index[-1]
        test_start_date = last_date - pd.Timedelta(days=90)
    else:
        test_start_date = pd.to_datetime(test_start_date)
    
    # Locate the nearest date in the index to our test_start_date
    if test_start_date not in df_full.index:
        # Find the closest date
        test_start_date = df_full.index[df_full.index.get_loc(test_start_date, method='nearest')]
    
    # Find the integer location of the test_start_date
    test_start_idx = df_full.index.get_loc(test_start_date)
    
    # Check there is enough data before and after the test_start_date
    if test_start_idx < n_lookback or test_start_idx + n_forecast > len(df_full):
        raise ValueError("Not enough data available for testing at the specified test_start_date.")
    
    # Prepare the input: get the previous n_lookback days of 'close' prices
    input_window = df_full.iloc[test_start_idx - n_lookback : test_start_idx]
    y_input = input_window['close'].fillna(method='ffill').values.reshape(-1, 1)
    
    # Fit a scaler on the input window (using the same scaling method as in training)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(y_input)
    input_scaled = scaler.transform(y_input)
    
    # Reshape for the model: (samples, time steps, features)
    X_input = np.array(input_scaled).reshape(1, n_lookback, 1)
    
    # Use the trained model to forecast the next n_forecast days
    forecast_scaled = model.predict(X_input)
    forecast_scaled = forecast_scaled.reshape(-1, 1)
    
    # Inverse transform the forecast to get actual price values
    forecast = scaler.inverse_transform(forecast_scaled)
    
    # Get the actual prices for the forecast period from the historical data
    actual_window = df_full.iloc[test_start_idx : test_start_idx + n_forecast]
    
    # Plot the forecast vs actual prices
    plt.figure(figsize=(10, 6))
    plt.plot(actual_window.index, actual_window['close'], label='Actual Price', marker='o')
    plt.plot(actual_window.index, forecast.flatten(), label='Forecast Price', marker='x')
    plt.title(f"30-Day Forecast vs. Actual from {test_start_date.date()}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_forecast(df: pd.DataFrame):
    """
    Plots historical stock data along with the forecast.
    Highlights the last 30 forecasted open values.
    Also tests on an earlier section of data for validation.
    """
    df = df.copy().sort_index()  # Ensure the data is sorted by date

    # Extract historical and forecasted data
    historical_data = df[df["close"].notna()].iloc[:-30]  # Past actual prices
    forecast_data = df[df["open"].notna()].iloc[-30:]  # Last 30 forecasted open values

    # Plot historical data
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data.index, historical_data["close"], label="Historical Data (Close)", color="blue")

    # Plot forecasted data
    plt.plot(forecast_data.index, forecast_data["open"], label="Forecast (Open)", color="orange")


    # Test model on an earlier segment (last 60 days, validating last 30 days)
    test_start = historical_data.index[-60]  # Start from 60 days before last known data
    test_end = historical_data.index[-30]    # Take the last 30 days before forecast
    test_segment = historical_data.loc[test_start:test_end]

    # Labels and legend
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Forecast vs Historical Data")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

if __name__ == "__main__":
    model, res = trainmodel("MSFT")
    plot_forecast(res)
    testmodel(model, "MSFT", test_start_date="2023-01-01", n_lookback=60, n_forecast=30)