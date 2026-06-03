import os, json, pickle
from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
# app.py lives at earnings_predictor/web/app.py
# BASE_DIR = earnings_predictor/
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
STATIC_DIR  = os.path.join(os.path.dirname(__file__), "static")
SHAP_DIR    = os.path.join(STATIC_DIR, "shap")

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/summary')
def summary():
    summary_path = os.path.join(REPORTS_DIR, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
    else:
        data = {"model_auc": None, "sharpe": None, "total_return": None, "win_rate": None}
    return jsonify(data)

@app.route('/api/predictions')
def predictions():
    pred_path = os.path.join(DATA_DIR, 'predictions.csv')
    if os.path.exists(pred_path):
        df = pd.read_csv(pred_path)
        records = df.head(50).to_dict(orient='records')
    else:
        records = []
    return jsonify(records)

@app.route('/api/backtest')
def backtest():
    equity_path = os.path.join(REPORTS_DIR, 'equity_curve.csv')
    if os.path.exists(equity_path):
        df = pd.read_csv(equity_path)
        records = df.to_dict(orient='records')
    else:
        records = []
    return jsonify(records)

@app.route('/api/predict', methods=['POST'])
def predict():
    data    = request.get_json()
    ticker  = data.get('ticker', '').upper()
    model_path = os.path.join(MODEL_DIR, 'xgb_best.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        model       = bundle['model']
        le          = bundle['le']
        feature_cols= bundle['feature_cols']
        # Build a feature vector with neutral/average values
        features = pd.DataFrame([{
            'surprise_last_q': 0.0,
            'surprise_last_4q_mean': 0.0,
            'estimate_revision_30d': 0.0,
            'estimate_revision_7d': 0.0,
            'analyst_dispersion': 10.0,
            'price_mom_20d': 0.01,
            'price_mom_5d': 0.005,
            'vol_ratio': 1.0,
            'sector_enc': 0,
            'market_cap_log': np.log(1e11),
        }])
        prob = float(model.predict_proba(features[feature_cols])[:, 1][0])
        prediction = "BEAT" if prob >= 0.5 else "MISS"
    else:
        prob = 0.0
        prediction = "Model not trained yet"
    return jsonify({"ticker": ticker, "probability": round(prob, 4), "prediction": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
