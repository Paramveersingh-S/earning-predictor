import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import joblib
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_best.pkl")

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["ticker", "announcement_date", "beat"]).fillna(0)
    y = df["beat"].values
    return X, y, df

def load_model():
    return joblib.load(MODEL_PATH)

def plot_roc(y_true, y_scores, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_precision_recall(y_true, y_scores, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="Precision‑Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision‑Recall Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_equity_curve(equity_path, out_path):
    equity_df = pd.read_csv(equity_path, parse_dates=["date"])
    plt.figure(figsize=(10, 6))
    plt.plot(equity_df["date"], equity_df["equity"], label="Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.title("Backtest Equity Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def generate_report():
    X, y, df = load_data()
    model = load_model()
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # ROC
    plot_roc(y, y_proba, os.path.join(REPORTS_DIR, "roc_curve.png"))
    # Confusion
    plot_confusion(y, y_pred, os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    # PR Curve
    plot_precision_recall(y, y_proba, os.path.join(REPORTS_DIR, "pr_curve.png"))
    # Equity Curve (generated previously by backtest)
    equity_csv = os.path.join(REPORTS_DIR, "equity_curve.csv")
    if os.path.exists(equity_csv):
        plot_equity_curve(equity_csv, os.path.join(REPORTS_DIR, "equity_curve_plot.png"))
    print("Report images generated in reports/ directory.")

if __name__ == "__main__":
    generate_report()
