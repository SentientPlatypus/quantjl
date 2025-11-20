#!/usr/bin/env python3

import sys
import certifi
import warnings
import json
import pandas as pd
from urllib.request import urlopen
import os
import platform
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

dsp = "/"
if platform.system() == "Windows":
    dsp = "\\"

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


def read_historical(tickers:list, apikey):
    df = pd.DataFrame()
    for ticker in tickers:
        # Build the URL for downloading historical data from 2000-01-01
        url = "https://financialmodelingprep.com/api/v3/historical-price-full"
        if ticker.endswith("=X"):
            url += "/{}?apikey={}&from=2000-01-01".format("USD" + ticker[:-2], apikey)
        else:
            url += "/{}?apikey={}&from=2000-01-01".format(ticker, apikey)
        
        json_data = get_jsonparsed_data(url)
        
        # Convert the "historical" data into a DataFrame
        hist_df = pd.DataFrame(json_data["historical"])
        # Sort in ascending order so that pct_change calculates correctly
        hist_df.sort_values("date", inplace=True)
        
        # Compute changeClosePercent as the percentage change in close from the previous day
        # This is calculated as: (today_close - previous_close) / previous_close * 100
        hist_df["changeClosePercent"] = hist_df["close"].pct_change() * 100
        
        # If you prefer to have the rows in descending order (most recent first),
        # you can reverse the DataFrame:
        hist_df = hist_df.iloc[::-1]
        
        # Write the modified DataFrame (which now includes the changeClosePercent column) to CSV.
        hist_df.to_csv(f".{dsp}data{dsp}{ticker}.csv", index=False)
        
        # Merge historical data of all tickers into one CSV file (using adjClose)
        hist = hist_df[["date", "adjClose"]].copy()
        hist = hist.rename(columns={"adjClose": ticker})
        if df.empty:
            df = hist
        else:
            df = df.merge(hist, on="date")
        
    df = df.drop(columns=["date"])




def read_high_frequency(tickers: list, apikey: str, start_date: str, day_index: int = 1):
    """
    Downloads 1-min high-frequency data for the given tickers from start_date to start_date+1 day.
    Saves each ticker's data to TICKER_day{day_index}.csv and returns the merged close-only dataframe.
    
    Args:
        tickers (list): List of stock tickers.
        apikey (str): Your FMP API key.
        start_date (str): Start date in YYYY-MM-DD format.
        day_index (int): Index to append to file name for saving multiple days.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(f".{dsp}data{dsp}{current_date}", exist_ok=True)


    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=1)
    end_date = end_dt.strftime("%Y-%m-%d")
    
    df = pd.DataFrame()
    for ticker in tickers:

        url = "https://financialmodelingprep.com/api/v3/historical-chart/1min"
        if ticker.endswith("=X"):
            url += f"/USD{ticker[:-2]}?from={start_date}&to={end_date}&apikey={apikey}"
        else:
            url += f"/{ticker}?from={start_date}&to={end_date}&apikey={apikey}"
        
        print(url)
        

        json_data = get_jsonparsed_data(url)
        
        hist_df = pd.DataFrame(json_data)[::-1]
        print(hist_df[-2:])

        if hist_df.empty:
            print(f"No data found for {ticker} on {start_date}. Skipping.")
            continue
        hist_df["changeClosePercent"] = hist_df["close"].pct_change().fillna(0) * 100
        hist_df = hist_df[::-1]
        print(hist_df[:2])
        
        save_path = f".{dsp}data{dsp}{current_date}{dsp}{ticker}_day{day_index}.csv"
        hist_df.to_csv(save_path, index=False)
        
        # Create merged close-only frame
        hist = hist_df[["date", "close"]].rename(columns={"close": ticker})
        df = hist if df.empty else df.merge(hist, on="date")

    if df.empty:
        print(f"No data found for any ticker on {start_date}.")
        return pd.DataFrame()
    df.iloc[-1, -1] = 0.0
    return df

def read_past_month(tickers: list, apikey: str, days: int = 30):
    """
    Calls `read_high_frequency` for the past `days` business days (weekdays only).
    
    Args:
        tickers (list): List of stock tickers.
        apikey (str): Your FMP API key.
        days (int): Number of past days to fetch.
    """
    today = datetime.today()
    count = 0
    day_offset = 1

    while count < days:
        candidate_date = today - timedelta(days=day_offset)
        if candidate_date.weekday() < 5:  # 0 = Monday, ..., 4 = Friday
            formatted_date = candidate_date.strftime("%Y-%m-%d")
            print(f"Fetching for day {count + 1}: {formatted_date}")
            read_high_frequency(tickers, apikey, formatted_date, count + 1)
            count += 1
        day_offset += 1



def main():
    tickers = sys.argv[1:]
    apikey = open(f".{dsp}apikey", "r").readline().strip()
    
    # read_historical(tickers, apikey)
    read_past_month(tickers, apikey, days=30)

if __name__ == "__main__":
    # Usage: ./download.py <ticker1> <ticker2> ...
    main()