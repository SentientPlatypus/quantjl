import pandas as pd
import numpy as np


from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import platform
import requests

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

def getNumbers(ticker:str):
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

    results: pd.DataFrame = pd.concat([df_past, df_future]).set_index('date')
    
    results = results[results["forecast"].notna()]
    results["open"] = results["forecast"]
    results["high"] = results["forecast"]
    results["low"] = results["forecast"]
    resultsShifted = results.shift(-1)
    results["close"] = resultsShifted["open"]
    final = pd.concat([toDisplay, results], sort=False, join="inner")
    return final
    


if __name__ == "__main__":
    print(getNumbers("MSFT"))