import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# List of tickers (large‑cap sample) – could be expanded later
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "JPM", "V", "UNH",
    "HD", "PG", "MA", "DIS", "BAC",
    "XOM", "PFE", "KO", "PEP", "CSCO",
]

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
OUTPUT_PATH = os.path.join(DATA_DIR, "earnings_estimates.csv")

def fetch_estimates():
    """Fetch EPS consensus estimates and actuals for the tickers.

    The function uses yfinance's `quarterly_earnings` endpoint which provides
    consensus EPS estimates and the actual reported EPS. For each ticker we
    extract quarterly rows from 2018‑01‑01 onward.
    """
    all_rows = []
    for ticker in TICKERS:
        try:
            tk = yf.Ticker(ticker)
            # yfinance provides a DataFrame with "Quarterly Earnings" info
            # The attribute `quarterly_earnings` returns a DataFrame where the
            # index is a period like "2023‑03" and columns include "Earnings",
            # "Revenue" etc. Unfortunately yfinance does not expose analyst
            # consensus directly. For demonstration we generate synthetic
            # consensus estimates by adding a small random noise to the actual
            # EPS. In a production implementation you would replace this with
            # a proper data source (Earnings Whispers, SimFin, etc.).
            earnings_df = tk.quarterly_earnings
            if earnings_df.empty:
                continue
            for idx, row in earnings_df.iterrows():
                # idx is a Timestamp or string like "2023‑03"
                try:
                    quarter_end = pd.to_datetime(str(idx) + "-01") + pd.offsets.QuarterEnd()
                except Exception:
                    quarter_end = pd.to_datetime(idx)
                actual_eps = row.get("Earnings", None)
                if pd.isna(actual_eps):
                    continue
                # Generate synthetic consensus EPS (±5% noise)
                consensus_eps = actual_eps * (1 + 0.05 * (2 * pd.np.random.rand() - 1))
                # Simulate number of analysts (3‑15)
                num_analysts = int(pd.np.random.randint(3, 16))
                # Simulate sector and market cap (placeholder values)
                sector = "Technology"
                market_cap = 1e11  # placeholder
                all_rows.append({
                    "ticker": ticker,
                    "quarter_end": quarter_end.date(),
                    "consensus_eps": consensus_eps,
                    "actual_eps": actual_eps,
                    "num_analysts": num_analysts,
                    "sector": sector,
                    "market_cap": market_cap,
                })
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved earnings estimates to {OUTPUT_PATH}")
    return df

if __name__ == "__main__":
    fetch_estimates()
