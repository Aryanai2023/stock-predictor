# simple_svm_stock_predictor.py

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# 1. SETTINGS
TICKER = "AAPL"  # change to any symbol, e.g. "MSFT", "TSLA"
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"

# 2. DOWNLOAD DATA
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

# Keep only what we need
df = df[["Close"]].copy()

# 3. CREATE FEATURES
# Daily return
df["return_1d"] = df["Close"].pct_change()

# Simple moving averages
df["ma_5"] = df["Close"].rolling(window=5).mean()
df["ma_10"] = df["Close"].rolling(window=10).mean()

# 4. CREATE LABEL (TARGET)
# 1 if next day's close is higher, else 0
df["tomorrow_close"] = df["Close"].shift(-1)
df["target_up"] = (df["tomorrow_close"] > df["Close"]).astype(int)

# Drop rows with NaN (from rolling & shift)
df = df.dropna()

# 5. FEATURES (X) AND TARGET (y)
feature_cols = ["return_1d", "ma_5", "ma_10"]
X = df[feature_cols].values
y = df["target_up"].values

# 6. TRAIN/TEST SPLIT (simple time-based split)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 7. BUILD SVM PIPELINE (SCALER + SVM)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

# 8. TRAIN
model.fit(X_train, y_train)

# 9. EVALUATE
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Ticker: {TICKER}")
print(f"Test accuracy (up/down next day): {acc:.3f}")

# 10. SIMPLE "NEXT DAY" PREDICTION (from latest row)
last_features = X[-1].reshape(1, -1)
next_up = model.predict(last_features)[0]

if next_up == 1:
    print("Model says: Tomorrow will CLOSE UP 📈")
else:
    print("Model says: Tomorrow will CLOSE DOWN 📉")
