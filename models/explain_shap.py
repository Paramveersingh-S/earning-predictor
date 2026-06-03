import os
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_best.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features.csv")
SHAP_DIR = os.path.join(BASE_DIR, "models", "shap_plots")
os.makedirs(SHAP_DIR, exist_ok=True)

def load_model():
    return joblib.load(MODEL_PATH)

def load_features():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["ticker", "announcement_date", "beat"]).fillna(0)
    return X, df

def compute_shap():
    model = load_model()
    X, df = load_features()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    summary_path = os.path.join(SHAP_DIR, "shap_summary.png")
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()
    # Individual waterfall for first sample
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X.iloc[0]), show=False)
    waterfall_path = os.path.join(SHAP_DIR, "shap_waterfall_0.png")
    plt.savefig(waterfall_path, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP plots to {SHAP_DIR}")
    return {
        "summary": summary_path,
        "waterfall_0": waterfall_path,
    }

if __name__ == "__main__":
    compute_shap()
