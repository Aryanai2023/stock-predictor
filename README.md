# Stock Predictor & Analysis Toolkit 🧠📈

Educational playground for experimenting with **stock price prediction**, **technical analysis**, and **simple portfolio backtesting** in Python.

> ⚠️ **Disclaimer:**  
> This project is for **learning and experimentation only**.  
> It is **not** financial advice. Do **not** use it to make real trading decisions.

---

## 🚀 What’s inside?

This repository currently includes:

1. **`simple_svm_stock_predictor.py`**  
   A very simple **SVM-based “up or down” classifier** for individual stocks.
   - Fetches daily price data from a free data source (e.g. Stooq via `pandas-datareader` or Yahoo via `yfinance`, depending on your version).
   - Builds basic features (returns, moving averages, etc.).
   - Tries to predict whether **tomorrow’s close will be UP (1) or DOWN (0)**.
   - Prints:
     - Train/test split size  
     - Test accuracy  
     - Baseline accuracy (always predicting majority class)  
     - A “tomorrow up/down” prediction based on the latest data

2. **`stock_analysis_extended.py`**  
   A **detailed multi-ticker analysis tool**.
   - Supports **multiple tickers in one run**, e.g. `AAPL,MSFT,TSLA`
   - For each ticker, computes:
     - Daily return, log return, cumulative return
     - Annualised return & volatility
     - Sharpe ratio (rf ≈ 0)
     - Maximum drawdown
     - % up-days / down-days
     - Average up-day / down-day move
     - Best & worst days
     - Moving averages: 20 / 50 / 200
     - RSI (14), MACD (12, 26, 9)
   - Generates:
     - **Short natural-language summary** per ticker  
     - **Combined metrics table** across all tickers  
     - **Daily return correlation matrix**
   - Optional plots:
     - Price + moving averages
     - Drawdown curve
     - RSI
     - MACD
     - Correlation heatmap

3. **`portfolio_backtester.py`**  
   A simple **portfolio backtester**.
   - Input:
     - List of tickers
     - Optional weights (otherwise equal-weight)
     - Start & end dates
   - Builds a rebalanced daily portfolio (weights normalised to sum to 1)
   - Computes:
     - Daily & cumulative returns
     - Annualised return & volatility
     - Sharpe ratio (rf ≈ 0)
     - Maximum drawdown
     - Best and worst portfolio days
   - Optional plot:
     - Portfolio equity curve vs each individual asset (all normalised to 1.0)

---

## 🧩 Project structure

```text
stock-predictor/
├─ simple_svm_stock_predictor.py     # SVM-based up/down classifier for a single stock
├─ stock_analysis_extended.py        # Multi-ticker technical & risk analysis
├─ portfolio_backtester.py           # Simple portfolio backtester
├─ .gitignore                        # Ignore caches, envs, token files, etc.
└─ README.md                         # You are here 🙂
