import os
import pandas as pd
import yfinance as yf
from datetime import datetime

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
PRICE_OUTPUT = os.path.join(DATA_DIR, "prices.csv")

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "UNH",
    "HD", "PG", "MA", "DIS", "BAC",
    "XOM", "PFE", "KO", "PEP", "CSCO",
]

START_DATE = "2018-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

def fetch_prices():
    """Download daily OHLCV price data for all tickers.

    The function stores a single CSV with the columns:
    ticker, date, open, high, low, close, volume
    """
    all_frames = []
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if df.empty:
                continue
            df = df.reset_index()
            df["ticker"] = ticker
            all_frames.append(df[["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]])
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)
        result.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        result.to_csv(PRICE_OUTPUT, index=False)
        print(f"Saved price data to {PRICE_OUTPUT}")
    else:
        print("No price data fetched.")

if __name__ == "__main__":
    fetch_prices()
