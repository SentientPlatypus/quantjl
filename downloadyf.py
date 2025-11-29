import yfinance as yf
import pandas as pd

def download_minutes(ticker, interval="1m", period="7d", outfile=None, prepost=False):
    """
    Downloads minute-level OHLCV data from Yahoo Finance
    and saves it in the exact CSV format you need.

    interval: "1m", "2m", "5m", "15m", "30m", "60m"
    period: up to "30d" for minute data
    """
    df = yf.download(ticker, interval=interval, period=period, prepost=prepost)
    df.columns = df.columns.get_level_values(0)
    print(df.columns)
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
if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    df = download_minutes(ticker, interval="1m", period="7d",
                        outfile=f"data/{ticker}.csv", prepost=True)
    print(df.head())
