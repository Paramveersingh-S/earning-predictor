"""
Standalone pipeline runner for Earnings Surprise Predictor.
Run from the earnings_predictor/ directory:
    python run_pipeline.py
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, date

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR= os.path.join(BASE_DIR, "reports")
SHAP_DIR  = os.path.join(BASE_DIR, "web", "static", "shap")

for d in [DATA_DIR, MODEL_DIR, REPORT_DIR, SHAP_DIR]:
    os.makedirs(d, exist_ok=True)

# ── TICKERS & SECTOR MAP ──────────────────────────────────────────────────────
TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META",
    "NVDA","TSLA","JPM","V","UNH",
    "HD","PG","MA","DIS","BAC",
    "XOM","PFE","KO","PEP","CSCO",
]
SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology",
    "AMZN":"Consumer Discretionary","META":"Technology",
    "NVDA":"Technology","TSLA":"Consumer Discretionary",
    "JPM":"Financials","V":"Financials","UNH":"Healthcare",
    "HD":"Consumer Discretionary","PG":"Consumer Staples",
    "MA":"Financials","DIS":"Communication Services","BAC":"Financials",
    "XOM":"Energy","PFE":"Healthcare","KO":"Consumer Staples",
    "PEP":"Consumer Staples","CSCO":"Technology",
}
MCAP_MAP = {
    "AAPL":3e12,"MSFT":3e12,"GOOGL":2e12,"AMZN":1.8e12,"META":1.3e12,
    "NVDA":2.5e12,"TSLA":8e11,"JPM":5e11,"V":5e11,"UNH":4.5e11,
    "HD":3.5e11,"PG":3.5e11,"MA":4e11,"DIS":1.8e11,"BAC":3.5e11,
    "XOM":5e11,"PFE":1.5e11,"KO":2.5e11,"PEP":2.3e11,"CSCO":2e11,
}

# ── STEP 1: Generate synthetic earnings estimates ─────────────────────────────
print("\n[1/5] Generating synthetic earnings estimates...")
QUARTERS = pd.date_range(end=datetime.now(), periods=20, freq='QE')
rows = []
for ticker in TICKERS:
    base_eps = np.random.uniform(1.5, 6.0)
    for q in QUARTERS:
        actual_eps = base_eps + np.random.normal(0, 0.3)
        consensus_eps = actual_eps * (1 + np.random.uniform(-0.07, 0.07))
        rows.append({
            "ticker": ticker,
            "quarter_end": q.date(),
            "consensus_eps": round(consensus_eps, 4),
            "actual_eps": round(actual_eps, 4),
            "num_analysts": int(np.random.randint(5, 20)),
            "sector": SECTOR_MAP[ticker],
            "market_cap": MCAP_MAP[ticker],
        })
    base_eps *= (1 + np.random.uniform(0.0, 0.15))  # growth trend

earnings_df = pd.DataFrame(rows)
earnings_path = os.path.join(DATA_DIR, "earnings_estimates.csv")
earnings_df.to_csv(earnings_path, index=False)
print(f"  Saved {len(earnings_df)} rows -> {earnings_path}")

# ── STEP 2: Generate synthetic price data ────────────────────────────────────
print("\n[2/5] Generating synthetic price data...")
price_rows = []
START = date(2018, 1, 1)
END   = date.today()
dates = pd.bdate_range(START, END)

for ticker in TICKERS:
    price = np.random.uniform(50, 200)
    for d in dates:
        ret = np.random.normal(0.0003, 0.015)
        price = price * (1 + ret)
        vol = abs(np.random.normal(0, 0.01)) * price
        price_rows.append({
            "ticker": ticker, "date": d.date(),
            "open": round(price * (1 + np.random.uniform(-0.005, 0.005)), 2),
            "high": round(price + abs(vol), 2),
            "low":  round(price - abs(vol), 2),
            "close": round(price, 2),
            "volume": int(np.random.randint(5e6, 5e7)),
        })

prices_df = pd.DataFrame(price_rows)
prices_path = os.path.join(DATA_DIR, "prices.csv")
prices_df.to_csv(prices_path, index=False)
print(f"  Saved {len(prices_df)} rows -> {prices_path}")

# ── STEP 3: Feature engineering ───────────────────────────────────────────────
print("\n[3/5] Engineering features...")
earnings_df["quarter_end"] = pd.to_datetime(earnings_df["quarter_end"]).dt.date
prices_df["date"]          = pd.to_datetime(prices_df["date"]).dt.date
price_by_ticker = {t: grp.set_index("date").sort_index() for t, grp in prices_df.groupby("ticker")}

feat_rows = []
for _, row in earnings_df.iterrows():
    ticker  = row["ticker"]
    q_end   = row["quarter_end"]
    ps      = price_by_ticker.get(ticker)
    if ps is None:
        continue
    # Find nearest prior trading date
    prior = ps.index[ps.index <= q_end]
    if prior.empty:
        continue
    ann_date = prior.max()

    # Previous surprise history
    prev = earnings_df[(earnings_df["ticker"] == ticker) & (earnings_df["quarter_end"] < q_end)].sort_values("quarter_end")
    if not prev.empty:
        lq = prev.iloc[-1]
        surp_last = (lq["actual_eps"] - lq["consensus_eps"]) / (abs(lq["consensus_eps"]) + 1e-9) * 100
    else:
        surp_last = 0.0
    last4 = prev.tail(4)
    if not last4.empty:
        surp_4q = ((last4["actual_eps"] - last4["consensus_eps"]) / (last4["consensus_eps"].abs() + 1e-9)).mean() * 100
    else:
        surp_4q = 0.0

    # Estimate revisions (using prior consensus as proxy)
    if len(prev) >= 2:
        rev30 = (prev.iloc[-1]["consensus_eps"] - prev.iloc[-2]["consensus_eps"]) / (abs(prev.iloc[-2]["consensus_eps"]) + 1e-9) * 100
        rev7  = rev30
    else:
        rev30 = rev7 = 0.0

    analyst_disp = row["num_analysts"] / (abs(row["consensus_eps"]) + 1e-9)

    pw = ps.loc[:ann_date, "close"]
    mom20 = (pw.iloc[-1] / pw.iloc[-21] - 1) if len(pw) >= 21 else 0.0
    mom5  = (pw.iloc[-1] / pw.iloc[-6]  - 1) if len(pw) >= 6  else 0.0
    if len(pw) >= 21:
        r = pw.pct_change().dropna()
        vol5  = r.iloc[-5:].std()  if len(r) >= 5  else 0.0
        vol20 = r.iloc[-20:].std() if len(r) >= 20 else 1e-9
        vol_ratio = vol5 / (vol20 + 1e-9)
    else:
        vol_ratio = 1.0

    feat_rows.append({
        "ticker": ticker,
        "announcement_date": ann_date,
        "surprise_last_q": surp_last,
        "surprise_last_4q_mean": surp_4q,
        "estimate_revision_30d": rev30,
        "estimate_revision_7d":  rev7,
        "analyst_dispersion": analyst_disp,
        "price_mom_20d": mom20,
        "price_mom_5d":  mom5,
        "vol_ratio": vol_ratio,
        "sector": row["sector"],
        "market_cap_log": np.log(row["market_cap"] + 1),
        "beat": int(row["actual_eps"] > row["consensus_eps"]),
    })

features_df = pd.DataFrame(feat_rows)
features_path = os.path.join(DATA_DIR, "features.csv")
features_df.to_csv(features_path, index=False)
print(f"  Saved {len(features_df)} rows -> {features_path}")

# ── STEP 4: Train XGBoost ─────────────────────────────────────────────────────
print("\n[4/5] Training XGBoost model...")
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import xgboost as xgb

df = features_df.copy().dropna()
le = LabelEncoder()
df["sector_enc"] = le.fit_transform(df["sector"])

FEATURE_COLS = [
    "surprise_last_q","surprise_last_4q_mean","estimate_revision_30d",
    "estimate_revision_7d","analyst_dispersion","price_mom_20d",
    "price_mom_5d","vol_ratio","sector_enc","market_cap_log",
]
X = df[FEATURE_COLS]
y = df["beat"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight, use_label_encoder=False,
    eval_metric="logloss", random_state=42, verbosity=0,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"  Model AUC: {auc:.4f}")

# Save model & label encoder
with open(os.path.join(MODEL_DIR, "xgb_best.pkl"), "wb") as f:
    pickle.dump({"model": model, "le": le, "feature_cols": FEATURE_COLS}, f)

# Save predictions
df["prob_beat"] = model.predict_proba(X[FEATURE_COLS])[:, 1]
df["prediction"] = (df["prob_beat"] >= 0.5).astype(int)
pred_cols = ["ticker","announcement_date","beat","prob_beat","prediction"]
df[pred_cols].to_csv(os.path.join(DATA_DIR, "predictions.csv"), index=False)
print(f"  Saved predictions -> {os.path.join(DATA_DIR, 'predictions.csv')}")

# ── STEP 4b: SHAP explanations ────────────────────────────────────────────────
try:
    import shap, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("  Generating SHAP plots...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, "shap_summary.png"), dpi=120, bbox_inches="tight")
    plt.close()

    exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0].values,
        feature_names=FEATURE_COLS,
    )
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, "shap_waterfall_0.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  SHAP plots saved -> {SHAP_DIR}")
except Exception as e:
    print(f"  SHAP skipped: {e}")

# ── STEP 5: Backtest & reports ────────────────────────────────────────────────
print("\n[5/5] Running backtest & generating reports...")
df_bt = df.copy().sort_values("announcement_date")
portfolio_val = 100_000.0
equity_curve = []
wins, losses = 0, 0

for _, bt_row in df_bt.iterrows():
    signal = bt_row["prediction"]
    actual = bt_row["beat"]
    ret = np.random.normal(0.02, 0.03) if signal == 1 else 0.0
    ret *= (1 if actual == signal else -1)
    portfolio_val *= (1 + ret * 0.05)   # 5% position size
    equity_curve.append({"date": str(bt_row["announcement_date"]), "equity": round(portfolio_val, 2)})
    if actual == signal:
        wins += 1
    else:
        losses += 1

ec_df = pd.DataFrame(equity_curve)
ec_df.to_csv(os.path.join(REPORT_DIR, "equity_curve.csv"), index=False)

total_return = (portfolio_val - 100_000) / 100_000
win_rate = wins / max(wins + losses, 1)
returns_series = ec_df["equity"].pct_change().dropna()
sharpe = returns_series.mean() / (returns_series.std() + 1e-9) * (252 ** 0.5)

summary = {
    "model_auc": round(auc, 4),
    "sharpe": round(float(sharpe), 4),
    "total_return": round(total_return * 100, 2),
    "win_rate": round(win_rate * 100, 2),
}
with open(os.path.join(REPORT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"  Summary: {summary}")
print(f"\n✅ Pipeline complete! Refresh http://127.0.0.1:5000 to see the dashboard.")
