import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")  # will be saved by engineer
TARGET = "beat"

def load_data():
    df = pd.read_csv(FEATURES_PATH)
    # Drop rows with NaNs (simplify for demo) – in production you may impute
    df = df.dropna().reset_index(drop=True)
    X = df.drop(columns=["ticker", "announcement_date", TARGET])
    y = df[TARGET]
    # Ensure proper ordering by announcement date for time split
    X["announcement_date"] = pd.to_datetime(df["announcement_date"]).astype(int)  # for ordering only
    X = X.sort_values("announcement_date")
    X = X.drop(columns=["announcement_date"])
    return X, y

def objective(trial):
    X, y = load_data()
    # TimeSeriesSplit (5 folds) preserving order
    tscv = TimeSeriesSplit(n_splits=5)
    # Sample a split for evaluation (last fold) – Optuna will evaluate on validation set
    train_index, valid_index = list(tscv.split(X))[-1]
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    # Compute class imbalance weight
    pos = sum(y_train == 1)
    neg = sum(y_train == 0)
    scale_pos_weight = neg / (pos + 1e-9)

    # Hyperparameters to tune
    param = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": scale_pos_weight,
        "verbosity": 0,
        "seed": 42,
    }
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=30, verbose=False)
    preds = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, preds)
    return auc

def tune_hyperparameters():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print("Best trial:")
    print(study.best_trial)
    return study.best_trial.params

def train_final_model(best_params):
    X, y = load_data()
    pos = sum(y == 1)
    neg = sum(y == 0)
    best_params["scale_pos_weight"] = neg / (pos + 1e-9)
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "auc"
    best_params["verbosity"] = 0
    best_params["seed"] = 42
    model = xgb.XGBClassifier(**best_params)
    # Use full data with early stopping via a validation split (last 20%)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
    model_path = os.path.join(MODEL_DIR, "xgb_best.pkl")
    joblib.dump(model, model_path)
    print(f"Saved trained model to {model_path}")
    return model_path

if __name__ == "__main__":
    best_params = tune_hyperparameters()
    train_final_model(best_params)
