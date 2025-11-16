"""
stock_analysis.py

Detailed stock analysis using data from Stooq via pandas-datareader.

Features:
- Downloads OHLCV data
- Computes returns, volatility, drawdowns
- Computes moving averages, RSI, MACD
- Prints a textual summary
- Optional plots (price + MAs, drawdown, RSI, MACD)

Educational only. Not financial advice.
"""

import argparse
from datetime import datetime
import sys
import math

import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt


# ---------------------- Data Loading ---------------------- #

def load_data_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load daily OHLCV data from Stooq via pandas-datareader.

    Stooq tickers:
      - US stocks: 'AAPL.US', 'MSFT.US', etc.
    If user passes 'AAPL', we auto-convert to 'AAPL.US'.
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

    # Ensure typical columns exist
    expected_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = expected_cols.difference(df.columns)
    if missing:
        print(f"Warning: missing columns from Stooq data: {missing}")

    return df


# ---------------------- Indicator Computation ---------------------- #

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given OHLCV data, compute extra columns:
    - daily_return
    - log_return
    - cum_return
    - rolling_vol_20
    - max_drawdown
    - ma_20, ma_50, ma_200
    - RSI_14
    - MACD, MACD_signal, MACD_hist
    """
    df = df.copy()

    if "Close" not in df.columns:
        print("Error: 'Close' column is required for indicator computation.")
        sys.exit(1)

    close = df["Close"]

    # Returns
    df["daily_return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))

    # Cumulative return (from first valid day)
    df["cum_return"] = (1 + df["daily_return"]).cumprod() - 1

    # Rolling volatility (20-day)
    df["rolling_vol_20"] = df["daily_return"].rolling(window=20).std() * np.sqrt(252)

    # Moving averages
    df["ma_20"] = close.rolling(window=20).mean()
    df["ma_50"] = close.rolling(window=50).mean()
    df["ma_200"] = close.rolling(window=200).mean()

    # Drawdowns
    running_max = close.cummax()
    drawdown = close / running_max - 1.0
    df["drawdown"] = drawdown
    df["max_drawdown"] = drawdown.cummin()

    # RSI (14)
    df["RSI_14"] = compute_rsi(close, window=14)

    # MACD (12, 26, 9)
    macd, macd_signal, macd_hist = compute_macd(close)
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist

    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal_window: int = 9):
    """
    Compute MACD line, signal line, and histogram.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    hist = macd - signal

    return macd, signal, hist


# ---------------------- Summary Stats ---------------------- #

def print_summary(df: pd.DataFrame, ticker: str):
    """
    Print a textual summary of performance and risk.
    """
    close = df["Close"].dropna()

    if len(close) < 2:
        print("Not enough data for summary.")
        return

    start_date = close.index[0].date()
    end_date = close.index[-1].date()
    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])

    total_return = end_price / start_price - 1

    # Annualised stats using log returns
    log_ret = df["log_return"].dropna()
    if len(log_ret) == 0:
        print("No log returns available; cannot compute annualised stats.")
        return

    mean_log_daily = log_ret.mean()
    vol_log_daily = log_ret.std()

    trading_days = 252
    ann_return = math.exp(mean_log_daily * trading_days) - 1
    ann_vol = vol_log_daily * math.sqrt(trading_days)

    # Simple Sharpe (rf ~ 0)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    max_dd = df["max_drawdown"].min() if "max_drawdown" in df.columns else np.nan

    # Best/worst days
    daily_ret = df["daily_return"].dropna()
    best_day_ret = daily_ret.max()
    worst_day_ret = daily_ret.min()
    best_day = daily_ret.idxmax().date() if not daily_ret.empty else None
    worst_day = daily_ret.idxmin().date() if not daily_ret.empty else None

    print("\n==================== SUMMARY ====================")
    print(f"Ticker:         {ticker}")
    print(f"Period:         {start_date} -> {end_date}")
    print(f"Start price:    {start_price:.2f}")
    print(f"End price:      {end_price:.2f}")
    print(f"Total return:   {total_return * 100:.2f}%")
    print(f"Ann. return:    {ann_return * 100:.2f}%")
    print(f"Ann. volatility:{ann_vol * 100:.2f}%")
    print(f"Sharpe (rf≈0):  {sharpe:.2f}")
    print(f"Max drawdown:   {max_dd * 100:.2f}%")
    if best_day and worst_day:
        print(f"Best day:       {best_day}  ({best_day_ret * 100:.2f}%)")
        print(f"Worst day:      {worst_day} ({worst_day_ret * 100:.2f}%)")
    print("=================================================\n")


# ---------------------- Plotting ---------------------- #

def plot_price_and_ma(df: pd.DataFrame, ticker: str):
    plt.figure()
    plt.plot(df.index, df["Close"], label="Close")
    if "ma_20" in df.columns:
        plt.plot(df.index, df["ma_20"], label="MA 20")
    if "ma_50" in df.columns:
        plt.plot(df.index, df["ma_50"], label="MA 50")
    if "ma_200" in df.columns:
        plt.plot(df.index, df["ma_200"], label="MA 200")
    plt.title(f"{ticker} Price & Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)


def plot_drawdown(df: pd.DataFrame, ticker: str):
    if "drawdown" not in df.columns:
        return
    plt.figure()
    plt.plot(df.index, df["drawdown"], label="Drawdown")
    plt.title(f"{ticker} Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)


def plot_rsi_macd(df: pd.DataFrame, ticker: str):
    # RSI
    if "RSI_14" in df.columns:
        plt.figure()
        plt.plot(df.index, df["RSI_14"], label="RSI 14")
        plt.axhline(70, linestyle="--")
        plt.axhline(30, linestyle="--")
        plt.title(f"{ticker} RSI (14)")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.grid(True)

    # MACD
    if {"MACD", "MACD_signal", "MACD_hist"}.issubset(df.columns):
        plt.figure()
        plt.plot(df.index, df["MACD"], label="MACD")
        plt.plot(df.index, df["MACD_signal"], label="Signal")
        plt.bar(df.index, df["MACD_hist"], alpha=0.5, label="Hist")
        plt.title(f"{ticker} MACD")
        plt.xlabel("Date")
        plt.ylabel("MACD")
        plt.legend()
        plt.grid(True)


# ---------------------- CLI & Main ---------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detailed stock analysis using Stooq data (educational)."
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
        default="2015-01-01",
        help="Start date YYYY-MM-DD (default: 2015-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, do not show plots (summary only).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Running detailed analysis for {args.ticker} "
          f"from {args.start} to {args.end}")

    df_raw = load_data_stooq(args.ticker, args.start, args.end)
    df = compute_indicators(df_raw)

    print_summary(df, args.ticker)

    if not args.no_plots:
        plot_price_and_ma(df, args.ticker)
        plot_drawdown(df, args.ticker)
        plot_rsi_macd(df, args.ticker)
        plt.show()


if __name__ == "__main__":
    main()
