"""
Simple SVM-based stock direction predictor using data from Stooq.

- Downloads historical prices via pandas-datareader
- Builds a few simple features
- Predicts whether tomorrow's close will be UP (1) or DOWN (0)

Educational only. Not financial advice.
"""

import argparse
from datetime import datetime
import sys

import numpy as np
import pandas as pd
from pandas_datareader import data as web
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def load_data_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load daily OHLCV data from Stooq via pandas-datareader.

    Note: Stooq tickers are often like:
      - US stocks: 'AAPL.US', 'MSFT.US'
      - Indexes: '^SPX', etc.
    We'll auto-append '.US' if no suffix is given.
    """
    if "." not in ticker:
        stooq_ticker = ticker + ".US"
    else:
        stooq_ticker = ticker

    print(f"Downloading data for {stooq_ticker} from Stooq ({start} to {end})...")

    try:
        df = web.DataReader(stooq_ticker, "stooq", start=start, end=end)
    except Exception as e:
        print(f"Error fetching data from Stooq: {e}")
        sys.exit(1)

    if df.empty:
        print("Error: No data returned from Stooq. Check ticker or date range.")
        sys.exit(1)

    # Stooq returns most-recent-first; sort by date ascending
    df = df.sort_index()

    if "Close" not in df.columns:
        print("Error: 'Close' column not found in downloaded data.")
        sys.exit(1)

    # Keep only Close
    return df[["Close"]].copy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with a 'Close' column, add simple features
    and a binary target 'target_up'.
    """
    df = df.copy()

    # 1-day and 5-day returns
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)

    # Simple moving averages
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_10"] = df["Close"].rolling(window=10).mean()

    # Distance from 5-day MA (normalised)
    df["dist_ma_5"] = (df["Close"] - df["ma_5"]) / df["ma_5"]

    # 5-day volatility (std of daily returns)
    df["vol_5d"] = df["return_1d"].rolling(window=5).std()

    # Target: 1 if tomorrow's close > today's close
    df["tomorrow_close"] = df["Close"].shift(-1)
    df["target_up"] = (df["tomorrow_close"] > df["Close"]).astype(int)

    # Drop rows with NaN (from rolling & shift)
    df = df.dropna()

    return df


def train_and_evaluate(df: pd.DataFrame):
    df_feat = build_features(df)

    feature_cols = [
        "return_1d",
        "return_5d",
        "ma_5",
        "ma_10",
        "dist_ma_5",
        "vol_5d",
    ]

    X = df_feat[feature_cols].values
    y = df_feat["target_up"].values

    if len(df_feat) < 50:
        print("Warning: Very few samples after feature engineering. "
              "Results may be unreliable.")

    # Simple time-based split (80% train, 20% test)
    split_idx = int(len(df_feat) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Total samples: {len(df_feat)}, "
          f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Model pipeline: scaler + SVM classifier
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
    ])

    # Train
    print("Training SVM model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    baseline_acc = max(np.mean(y_test), 1 - np.mean(y_test))  # always-majority-class

    print("\n=== Evaluation ===")
    print(f"Test accuracy (SVM):         {acc:.3f}")
    print(f"Baseline (always majority):  {baseline_acc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

    # Simple "next day" prediction using the last available row
    last_features = X[-1].reshape(1, -1)
    next_up = model.predict(last_features)[0]

    print("=== Next-day direction prediction ===")
    if next_up == 1:
        print("Model says: Tomorrow will CLOSE UP 📈")
    else:
        print("Model says: Tomorrow will CLOSE DOWN 📉")

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple SVM stock up/down predictor using Stooq (educational)."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Ticker symbol (without .US suffix, default: AAPL)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2018-01-01",
        help="Start date YYYY-MM-DD (default: 2018-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Running SVM predictor for {args.ticker} "
          f"from {args.start} to {args.end}")

    df = load_data_stooq(args.ticker, args.start, args.end)
    train_and_evaluate(df)


if __name__ == "__main__":
    main()
