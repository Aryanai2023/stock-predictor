"""
Ensemble Stock Predictor (Stacking)

- Downloads OHLCV data from Yahoo Finance
- Builds simple technical indicators
- Creates binary target: 1 if next-day return > 0, else 0
- Trains base models: RandomForest, XGBoost, LogisticRegression
- Trains meta-model on out-of-fold predictions (stacking)
- Evaluates on chronological train/test split
- Provides a function to predict on latest days

Requirements:
    pip install yfinance pandas numpy scikit-learn xgboost
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


# ============================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================

def download_stock_data(
    ticker: str = "AAPL",
    period: str = "5y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        period: Lookback period (e.g., "1y", "5y", "max")
        interval: Data interval (e.g., "1d", "1h")

    Returns:
        DataFrame with columns: [Date, Open, High, Low, Close, Volume]
    """
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df = df.reset_index()
    df.rename(columns=str.lower, inplace=True)
    # Ensure standard column names
    df.rename(columns={"adj close": "adj_close"}, inplace=True)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators to the OHLCV DataFrame.

    Indicators:
        - Daily return
        - SMA_5, SMA_10, SMA_20
        - EMA_5, EMA_10
        - Rolling volatility (20-day std of returns)
        - RSI_14
        - MACD, MACD_signal
        - Bollinger band width (20 days)

    Returns:
        DataFrame with extra feature columns.
    """

    df = df.copy()

    # --- Returns ---
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # --- Simple Moving Averages ---
    df["sma_5"] = df["close"].rolling(window=5).mean()
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()

    # --- Exponential Moving Averages ---
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()

    # --- Volatility (Rolling std of returns) ---
    df["volatility_20"] = df["return"].rolling(window=20).std()

    # --- Relative Strength Index (RSI-14) ---
    window = 14
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df["rsi_14"] = rsi.values

    # --- MACD (12-26) ---
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # --- Bollinger Bands (20 days, 2 std) ---
    bb_window = 20
    rolling_mean = df["close"].rolling(window=bb_window).mean()
    rolling_std = df["close"].rolling(window=bb_window).std()
    df["bb_upper"] = rolling_mean + 2 * rolling_std
    df["bb_lower"] = rolling_mean - 2 * rolling_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (rolling_mean + 1e-9)

    # --- Volume features ---
    df["volume_change"] = df["volume"].pct_change()
    df["volume_sma_10"] = df["volume"].rolling(window=10).mean()

    return df


def create_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Create binary target: 1 if next-day return > 0, else 0.

    Args:
        df: DataFrame with 'close'
        horizon: number of days ahead (default 1)

    Returns:
        DataFrame with 'target' column.
    """
    df = df.copy()
    future_return = df["close"].shift(-horizon) / df["close"] - 1.0
    df["target"] = (future_return > 0).astype(int)
    return df


def prepare_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix X and target vector y.

    Drops rows with NaNs and non-feature columns.

    Returns:
        X: np.ndarray
        y: np.ndarray
        feature_cols: list of feature column names
        df_clean: cleaned DataFrame aligned with X and y (for dates etc.)
    """
    df = df.copy()

    # Drop rows that cannot have signals (NaNs due to indicators)
    df = df.dropna().reset_index(drop=True)

    # Columns not to use as features
    non_features = [
        "date", "target", "open", "high", "low",
        "close", "adj_close"
    ]
    feature_cols = [c for c in df.columns if c not in non_features and df[c].dtype != "O"]

    X = df[feature_cols].values
    y = df["target"].values

    return X, y, feature_cols, df


# ============================================================
# 2. STACKED ENSEMBLE TRAINING
# ============================================================

def train_stacked_ensemble(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    Train a stacked ensemble with base models and a meta-model.

    Base models:
        - RandomForestClassifier
        - XGBClassifier
        - LogisticRegression

    Meta-model:
        - LogisticRegression (on out-of-fold predictions of base models)

    Args:
        X: feature matrix
        y: labels (0/1)
        n_splits: number of time-series folds

    Returns:
        base_models: list of (name, fitted_model)
        meta_model: fitted meta-model
        oof_preds: out-of-fold predictions used to train meta-model
    """

    # Define base models
    base_models = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss"
            )
        ),
        (
            "logit",
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs"
            )
        )
    ]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    n_models = len(base_models)

    # Out-of-fold predictions: (n_samples, n_models)
    oof_preds = np.zeros((len(X), n_models))

    print("=== Training base models with TimeSeriesSplit ===")
    for model_idx, (name, model) in enumerate(base_models):
        print(f"\nBase model: {name}")
        fold_num = 0

        for train_idx, val_idx in tscv.split(X):
            fold_num += 1
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            val_proba = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, model_idx] = val_proba

            auc = roc_auc_score(y_val, val_proba)
            acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))
            print(f"  Fold {fold_num}: AUC = {auc:.4f} | ACC = {acc:.4f}")

    # Train meta-model on all out-of-fold predictions
    meta_model = LogisticRegression(max_iter=2000)
    meta_model.fit(oof_preds, y)

    meta_proba = meta_model.predict_proba(oof_preds)[:, 1]
    meta_auc = roc_auc_score(y, meta_proba)
    meta_acc = accuracy_score(y, (meta_proba > 0.5).astype(int))

    print("\n=== Meta-model performance on OOF data ===")
    print(f"AUC  = {meta_auc:.4f}")
    print(f"ACC  = {meta_acc:.4f}")

    # Retrain each base model on full dataset for future inference
    print("\n=== Retraining base models on full data ===")
    for name, model in base_models:
        print(f"Retraining {name} on full X")
        model.fit(X, y)

    return base_models, meta_model, oof_preds


# ============================================================
# 3. TRAIN / TEST SPLIT & EVALUATION
# ============================================================

def train_test_split_time_series(
    X: np.ndarray,
    y: np.ndarray,
    test_size_ratio: float = 0.2
):
    """
    Simple chronological split: last test_size_ratio as test.

    Returns:
        X_train, X_test, y_train, y_test, train_idx, test_idx
    """
    n = len(X)
    test_size = int(n * test_size_ratio)
    train_size = n - test_size

    train_idx = np.arange(0, train_size)
    test_idx = np.arange(train_size, n)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test, train_idx, test_idx


def fit_ensemble_with_train_test(X: np.ndarray, y: np.ndarray):
    """
    Train ensemble using only training set to generate OOF predictions,
    then evaluate on test set.

    Returns:
        base_models: fitted base models
        meta_model: fitted meta model
        metrics: dict with metrics
        test_predictions: dict of model_name -> proba on test
    """
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split_time_series(
        X, y, test_size_ratio=0.2
    )

    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples : {len(X_test)}")

    # Train stacked ensemble on training set only
    base_models, meta_model, oof_preds_train = train_stacked_ensemble(X_train, y_train)

    # Build test predictions
    n_models = len(base_models)
    base_test_preds = np.zeros((len(X_test), n_models))
    test_predictions = {}

    print("\n=== Evaluating base models on test set ===")
    for i, (name, model) in enumerate(base_models):
        proba = model.predict_proba(X_test)[:, 1]
        base_test_preds[:, i] = proba

        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, (proba > 0.5).astype(int))
        test_predictions[name] = proba

        print(f"{name} -> Test AUC: {auc:.4f} | ACC: {acc:.4f}")

    # Meta-model test performance
    meta_test_proba = meta_model.predict_proba(base_test_preds)[:, 1]
    meta_auc = roc_auc_score(y_test, meta_test_proba)
    meta_acc = accuracy_score(y_test, (meta_test_proba > 0.5).astype(int))
    test_predictions["meta"] = meta_test_proba

    print("\n=== Meta-model (ensemble) performance on TEST set ===")
    print(f"AUC: {meta_auc:.4f}")
    print(f"ACC: {meta_acc:.4f}")
    print("\nClassification report (meta-model, threshold=0.5):")
    print(classification_report(y_test, (meta_test_proba > 0.5).astype(int)))

    metrics = {
        "meta_auc": meta_auc,
        "meta_acc": meta_acc
    }

    return base_models, meta_model, metrics, test_predictions


# ============================================================
# 4. PREDICTION / INFERENCE
# ============================================================

def predict_ensemble_latest(
    df: pd.DataFrame,
    feature_cols,
    base_models,
    meta_model,
    window: int = 10
) -> pd.DataFrame:
    """
    Predict ensemble probability for the last 'window' rows of the DataFrame.

    Args:
        df: cleaned dataframe after feature creation and dropping NaNs
        feature_cols: list of feature column names
        base_models: list of (name, model)
        meta_model: trained meta-model
        window: number of latest rows to predict

    Returns:
        DataFrame with date, close, target, each base model proba and meta proba
    """
    df = df.copy()
    df_latest = df.tail(window).reset_index(drop=True)

    X_latest = df_latest[feature_cols].values
    n_models = len(base_models)

    base_preds = np.zeros((len(X_latest), n_models))
    base_columns = []

    for i, (name, model) in enumerate(base_models):
        base_columns.append(f"proba_{name}")
        base_preds[:, i] = model.predict_proba(X_latest)[:, 1]

    meta_proba = meta_model.predict_proba(base_preds)[:, 1]

    result = df_latest[["date", "close", "target"]].copy()
    for i, col_name in enumerate(base_columns):
        result[col_name] = base_preds[:, i]
    result["proba_meta"] = meta_proba

    return result


# ============================================================
# 5. MAIN SCRIPT
# ============================================================

def main():
    # -----------------------------------
    # CONFIG
    # -----------------------------------
    ticker = "AAPL"     # change this to any stock you like, e.g. "MSFT", "TSLA"
    period = "5y"
    interval = "1d"

    print(f"Downloading data for {ticker} ({period}, {interval})...")
    df_raw = download_stock_data(ticker=ticker, period=period, interval=interval)

    print("Adding technical indicators...")
    df_feat = add_technical_indicators(df_raw)

    print("Creating target variable...")
    df_target = create_target(df_feat, horizon=1)

    print("Preparing feature matrix...")
    X, y, feature_cols, df_clean = prepare_feature_matrix(df_target)

    print(f"Number of samples after cleaning: {len(df_clean)}")
    print(f"Number of features: {len(feature_cols)}")

    # -----------------------------------
    # TRAIN & EVALUATE ENSEMBLE
    # -----------------------------------
    base_models, meta_model, metrics, test_predictions = fit_ensemble_with_train_test(X, y)

    # -----------------------------------
    # PREDICT ON LATEST DAYS
    # -----------------------------------
    print("\nPredicting on latest 10 days...")
    latest_preds = predict_ensemble_latest(
        df_clean, feature_cols, base_models, meta_model, window=10
    )
    print(latest_preds)


if __name__ == "__main__":
    main()
