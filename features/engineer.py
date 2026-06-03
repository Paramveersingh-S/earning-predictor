import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta

# Helper to compute rolling returns
def _rolling_return(series, window):
    return series.pct_change(periods=window).shift(-window)

def engineer_features(earnings_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the 10 features described in the spec.

    Parameters
    ----------
    earnings_df : pd.DataFrame
        Columns: ticker, quarter_end (date), consensus_eps, actual_eps, num_analysts, sector, market_cap
    price_df : pd.DataFrame
        Columns: ticker, date, open, high, low, close, volume

    Returns
    -------
    pd.DataFrame
        One row per earnings event with engineered features and target ``beat``.
    """
    # Ensure date types
    earnings_df = earnings_df.copy()
    earnings_df["quarter_end"] = pd.to_datetime(earnings_df["quarter_end"]).dt.date
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.date

    # Helper: get price series for a ticker
    price_by_ticker = {t: df.set_index('date').sort_index() for t, df in price_df.groupby('ticker')}

    rows = []
    for _, row in earnings_df.iterrows():
        ticker = row['ticker']
        q_end = row['quarter_end']
        # Find announcement date approximated as quarter_end (real data may differ)
        ann_date = q_end
        # Filter price series up to announcement date
        price_series = price_by_ticker.get(ticker)
        if price_series is None:
            continue
        # Ensure we have enough history
        if ann_date not in price_series.index:
            # Find the nearest prior date
            prior_dates = price_series.index[price_series.index < ann_date]
            if prior_dates.empty:
                continue
            ann_date = prior_dates.max()
        # Feature 1: surprise_last_q (previous quarter)
        # Find previous quarter actual vs consensus
        prev_mask = (earnings_df['ticker'] == ticker) & (earnings_df['quarter_end'] < q_end)
        prev_rows = earnings_df[prev_mask].sort_values('quarter_end')
        if not prev_rows.empty:
            last_q = prev_rows.iloc[-1]
            surprise_last_q = (last_q['actual_eps'] - last_q['consensus_eps']) / abs(last_q['consensus_eps']) * 100
        else:
            surprise_last_q = np.nan
        # Feature 2: surprise_last_4q_mean
        last_4 = prev_rows.tail(4)
        if not last_4.empty:
            surprise_last_4q_mean = ((last_4['actual_eps'] - last_4['consensus_eps']).abs() / last_4['consensus_eps']).mean() * 100
        else:
            surprise_last_4q_mean = np.nan
        # Feature 3 & 4: estimate revision 30d / 7d – synthetic using consensus_eps history
        # Build synthetic history of consensus estimates by adding small random walk
        # Here we approximate revisions by looking at change in consensus_eps over windows
        past_estimates = earnings_df[(earnings_df['ticker'] == ticker) & (earnings_df['quarter_end'] <= q_end)].sort_values('quarter_end')
        # Estimate revision 30d: compare consensus of most recent quarter to that 30 days earlier (approx.)
        # Since we only have quarterly data, we approximate using previous quarter
        if len(past_estimates) >= 2:
            rev_30d = (past_estimates.iloc[-1]['consensus_eps'] - past_estimates.iloc[-2]['consensus_eps']) / abs(past_estimates.iloc[-2]['consensus_eps']) * 100
            rev_7d = rev_30d  # placeholder due to data granularity
        else:
            rev_30d = rev_7d = np.nan
        # Feature 5: analyst_dispersion
        analyst_dispersion = row['num_analysts'] / (abs(row['consensus_eps']) + 1e-9)  # placeholder; real dispersion needs individual analyst estimates
        # Feature 6 & 7: price momentum 20d, 5d before earnings
        price_window = price_series.loc[:ann_date]
        if len(price_window) >= 21:
            price_mom_20d = price_window['close'].iloc[-1] / price_window['close'].iloc[-21] - 1
        else:
            price_mom_20d = np.nan
        if len(price_window) >= 6:
            price_mom_5d = price_window['close'].iloc[-1] / price_window['close'].iloc[-6] - 1
        else:
            price_mom_5d = np.nan
        # Feature 8: vol_ratio (5d realized vol / 20d historical vol)
        if len(price_window) >= 21:
            returns_5d = price_window['close'].pct_change().dropna().iloc[-5:]
            vol_5d = returns_5d.std()
            returns_20d = price_window['close'].pct_change().dropna().iloc[-20:]
            vol_20d = returns_20d.std()
            vol_ratio = vol_5d / (vol_20d + 1e-9)
        else:
            vol_ratio = np.nan
        # Feature 9: sector one-hot (store as categorical, will be one-hot later)
        sector = row['sector']
        # Feature 10: log market cap
        market_cap_log = np.log(row['market_cap'] + 1e-9)
        # Target
        beat = int(row['actual_eps'] > row['consensus_eps'])
        # Build row dict
        feature_row = {
            'ticker': ticker,
            'announcement_date': ann_date,
            'surprise_last_q': surprise_last_q,
            'surprise_last_4q_mean': surprise_last_4q_mean,
            'estimate_revision_30d': rev_30d,
            'estimate_revision_7d': rev_7d,
            'analyst_dispersion': analyst_dispersion,
            'price_mom_20d': price_mom_20d,
            'price_mom_5d': price_mom_5d,
            'vol_ratio': vol_ratio,
            'sector': sector,
            'market_cap_log': market_cap_log,
            'beat': beat,
        }
        rows.append(feature_row)
    feature_df = pd.DataFrame(rows)
    # One-hot encode sector (optional later in model pipeline)
    return feature_df

if __name__ == "__main__":
    # Simple demo – load CSVs generated by fetch scripts
    earnings_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "earnings_estimates.csv"))
    prices_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "prices.csv"))
    earnings_df = pd.read_csv(earnings_path)
    prices_df = pd.read_csv(prices_path)
    features = engineer_features(earnings_df, prices_df)
    print(features.head())
