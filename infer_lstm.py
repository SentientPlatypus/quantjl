import sys
import json
import pandas as pd
import certifi
from urllib.request import urlopen
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotext as plt

APIKEY = open(f"./apikey", "r").readline().strip()

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

def load_data(ticker:str):

    
    # Build the URL for downloading high-frequency data
    url = "https://financialmodelingprep.com/api/v3/historical-chart/1min"
    if ticker.endswith("=X"):
        url += "/{}?apikey={}".format("USD" + ticker[:-2], APIKEY)
    else:
        url += "/{}?apikey={}".format(ticker, APIKEY)

    json_data = get_jsonparsed_data(url)

    df = pd.DataFrame(json_data)
    df_reversed = df.iloc[::-1].reset_index(drop=True)
    return df_reversed

def load_model(filename:str):
    # Load the pre-trained LSTM model from the specified filename
    try:
        model = tf.keras.models.load_model(filename)
        print(f"Model loaded from {filename}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    return model


def infer(ticker:str, model_filename:str = "msft_keras_model.keras", future_days: int = 30):
    data = load_data(ticker)
    model = load_model(model_filename)  # Load the pre-trained model
    scaler = MinMaxScaler(feature_range=(0,1))

    close_values = data.iloc[:, 4:5].values


    past_100_days = close_values[-100:].reshape(-1, 1)  # Get the last 100 days of closing prices

    input_data = scaler.fit_transform(past_100_days)
    predictions = []

    next_day_scaled = model.predict(np.array([input_data]))
    next_day = scaler.inverse_transform(next_day_scaled)
    predictions.append(float(next_day[0][0]))
    return predictions, past_100_days

def plot_movement(past, future, ticker):
    plt.plot_size(100, 100)  # Set the size of the plot

    x_future = list(range(1, len(future) + 1)) 
    x_past = list(range(len(future) + 1, len(future) + len(past) + 1))

    plt.plot(x_future, future, label="Predicted", color="blue")
    plt.plot(x_past, past, label="Past", color="orange")

    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Days")
    plt.ylabel("Price")

    plt.grid(True)
    plt.theme("dark")  # Use a dark theme for the plot

    plt.show()



if __name__ == "__main__":

    argv = sys.argv

    ticker = sys.argv[1]

    predictions, past_100 = infer(ticker, "AAPL_keras_model.keras", future_days=30)
    plot_movement(predictions, past_100.flatten().tolist(), ticker)


