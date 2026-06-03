import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_best.pkl")

def load_model():
    import joblib
    return joblib.load(MODEL_PATH)

def load_features():
    df = pd.read_csv(FEATURES_PATH)
    # Ensure dates are proper datetime
    df["announcement_date"] = pd.to_datetime(df["announcement_date"])
    X = df.drop(columns=["ticker", "announcement_date", "beat"]).fillna(0)
    y = df["beat"].values
    meta = df[["ticker", "announcement_date"]]
    return X, y, meta

def backtest():
    model = load_model()
    X, y, meta = load_features()
    preds_proba = model.predict_proba(X)[:, 1]
    # Convert probability to binary signal using 0.5 threshold (could be optimized)
    signals = (preds_proba >= 0.5).astype(int)
    # Build trade log
    trades = []
    equity = 1.0  # start with $1 capital
    equity_curve = []
    position = None  # holds dict with entry price, ticker, entry_date
    trade_cost = 0.001  # 10 bps round‑trip
    max_simultaneous = 10

    # Load price data for each ticker
    price_path = os.path.join(DATA_DIR, "prices.csv")
    price_df = pd.read_csv(price_path)
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
    price_by_ticker = {t: df.set_index('date').sort_index() for t, df in price_df.groupby('ticker')}

    # Iterate chronological order of announcements
    events = meta.copy()
    events["signal"] = signals
    events = events.sort_values("announcement_date")

    for idx, row in events.iterrows():
        ticker = row["ticker"]
        ann_date = row["announcement_date"].date()
        signal = row["signal"]
        # Determine entry date (5 trading days before announcement)
        price_series = price_by_ticker.get(ticker)
        if price_series is None:
            continue
        # Find the closest prior trading day to ann_date
        dates = list(price_series.index)
        if ann_date not in dates:
            # find prior date
            prior = [d for d in dates if d < ann_date]
            if not prior:
                continue
            ann_date = prior[-1]
        # Entry 5 days before
        entry_idx = dates.index(ann_date) - 5
        if entry_idx < 0:
            continue
        entry_date = dates[entry_idx]
        entry_price = price_series.loc[entry_date, "close"]
        # Exit next day after announcement (use close price of that day)
        exit_idx = dates.index(ann_date) + 1
        if exit_idx >= len(dates):
            continue
        exit_date = dates[exit_idx]
        exit_price = price_series.loc[exit_date, "open"]  # sell at open next day
        # Simulate trade if signal == 1 (predict beat) and we have capacity
        if signal == 1 and len([t for t in trades if t["exit_date"] > entry_date]) < max_simultaneous:
            pnl = (exit_price - entry_price) / entry_price - trade_cost
            equity *= (1 + pnl)
            trades.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
            })
            equity_curve.append({"date": exit_date, "equity": equity})
    # Convert to DataFrame
    equity_df = pd.DataFrame(equity_curve)
    equity_df.to_csv(os.path.join(REPORTS_DIR, "equity_curve.csv"), index=False)
    # Save trade log
    pd.DataFrame(trades).to_csv(os.path.join(REPORTS_DIR, "trade_log.csv"), index=False)
    print(f"Backtest completed. Final equity: {equity:.4f}")

if __name__ == "__main__":
    backtest()
