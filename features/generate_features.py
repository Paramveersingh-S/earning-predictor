import os
import pandas as pd
from earnings_predictor.data.fetch_estimates import fetch_estimates
from earnings_predictor.data.fetch_prices import fetch_prices
from earnings_predictor.features.engineer import engineer_features

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

def generate_features():
    # Step 1: fetch raw data if not present
    estimates_path = os.path.join(DATA_DIR, "earnings_estimates.csv")
    prices_path = os.path.join(DATA_DIR, "prices.csv")
    if not os.path.exists(estimates_path):
        print("Fetching earnings estimates...")
        fetch_estimates()
    if not os.path.exists(prices_path):
        print("Fetching price data...")
        fetch_prices()
    # Load data
    earnings_df = pd.read_csv(estimates_path)
    prices_df = pd.read_csv(prices_path)
    # Engineer features
    features_df = engineer_features(earnings_df, prices_df)
    # Save features for model training
    features_path = os.path.join(DATA_DIR, "features.csv")
    features_df.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")

if __name__ == "__main__":
    generate_features()
