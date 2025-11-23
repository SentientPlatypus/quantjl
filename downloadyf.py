import yfinance as yf
import pandas as pd

def download_minutes(ticker, interval="1m", period="7d", outfile=None):
    """
    Downloads minute-level OHLCV data from Yahoo Finance
    and saves it in the exact CSV format you need.

    interval: "1m", "2m", "5m", "15m", "30m", "60m"
    period: up to "30d" for minute data
    """
    df = yf.download(ticker, interval=interval, period=period)

    # Clean index to string timestamps
    df = df.reset_index()
    df.rename(columns={
        "Datetime": "date",
        "Open": "open",
        "Low": "low",
        "High": "high",
        "Close": "close",
        "Volume": "volume",
    }, inplace=True)

    # Compute % change in close price
    df["changeClosePercent"] = df["close"].pct_change() * 100

    # Convert to the exact format
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Reorder columns
    df = df[["date", "open", "low", "high", "close", "volume", "changeClosePercent"]]

    if outfile:
        df.to_csv(outfile, index=False)
        print(f"Saved to {outfile}")

    return df


# Example usage
df = download_minutes("MSFT", interval="1m", period="7d",
                      outfile="MSFT.csv")

df.to_csv("SPY.csv", index=False)
print(df.head())
