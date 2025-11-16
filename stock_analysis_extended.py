"""
stock_analysis_extended.py

Detailed multi-ticker stock analysis using data from Stooq via pandas-datareader.

Features:
- Supports multiple tickers in one run (--tickers AAPL,MSFT,TSLA)
- Downloads OHLCV data
- Computes returns, volatility, drawdowns
- Computes moving averages, RSI, MACD
- Per-ticker metrics & short natural-language summary
- Combined metrics table
- Correlation matrix of daily returns
- Optional plots:
    * Price + moving averages
    * Drawdown
    * RSI & MACD
    * Correlation heatmap

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

    print(f"\nDownloading data for {stooq_ticker} from Stooq ({start} to {end})...")

    try:
        df = web.DataReader(stooq_ticker, "stooq", start=start, end=end)
    except Exception as e:
        print(f"Error fetching data from Stooq for {ticker}: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"Error: No data returned from Stooq for {ticker}.")
        return pd.DataFrame()

    # Stooq returns most-recent-first; sort by date ascending
    df = df.sort_index()

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
        raise ValueError("'Close' column is required for indicator computation.")

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


# ---------------------- Metrics & Summaries ---------------------- #

def compute_metrics(df: pd.DataFrame, ticker: str) -> dict:
    """
    Compute summary metrics for a single ticker.
    Returns a dict suitable for building a DataFrame later.
    """
    metrics = {"ticker": ticker}

    close = df["Close"].dropna()
    if len(close) < 2:
        metrics.update({
            "start_date": None,
            "end_date": None,
            "start_price": np.nan,
            "end_price": np.nan,
            "total_return_pct": np.nan,
            "ann_return_pct": np.nan,
            "ann_vol_pct": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
            "up_days_pct": np.nan,
            "down_days_pct": np.nan,
            "avg_up_day_pct": np.nan,
            "avg_down_day_pct": np.nan,
            "best_day_pct": np.nan,
            "worst_day_pct": np.nan,
        })
        return metrics

    start_date = close.index[0].date()
    end_date = close.index[-1].date()
    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])

    total_return = end_price / start_price - 1

    log_ret = df["log_return"].dropna()
    trading_days = 252

    if len(log_ret) > 0:
        mean_log_daily = log_ret.mean()
        vol_log_daily = log_ret.std()

        ann_return = math.exp(mean_log_daily * trading_days) - 1
        ann_vol = vol_log_daily * math.sqrt(trading_days)
        sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    else:
        ann_return = np.nan
        ann_vol = np.nan
        sharpe = np.nan

    max_dd = df["max_drawdown"].min() if "max_drawdown" in df.columns else np.nan

    daily_ret = df["daily_return"].dropna()
    if len(daily_ret) > 0:
        up_days = (daily_ret > 0).sum()
        down_days = (daily_ret < 0).sum()
        total_days = len(daily_ret)
        up_days_pct = up_days / total_days * 100
        down_days_pct = down_days / total_days * 100

        avg_up_day = daily_ret[daily_ret > 0].mean() * 100 if up_days > 0 else np.nan
        avg_down_day = daily_ret[daily_ret < 0].mean() * 100 if down_days > 0 else np.nan

        best_day_ret = daily_ret.max() * 100
        worst_day_ret = daily_ret.min() * 100
    else:
        up_days_pct = down_days_pct = np.nan
        avg_up_day = avg_down_day = np.nan
        best_day_ret = worst_day_ret = np.nan

    metrics.update({
        "start_date": start_date,
        "end_date": end_date,
        "start_price": start_price,
        "end_price": end_price,
        "total_return_pct": total_return * 100,
        "ann_return_pct": ann_return * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100 if not pd.isna(max_dd) else np.nan,
        "up_days_pct": up_days_pct,
        "down_days_pct": down_days_pct,
        "avg_up_day_pct": avg_up_day,
        "avg_down_day_pct": avg_down_day,
        "best_day_pct": best_day_ret,
        "worst_day_pct": worst_day_ret,
    })

    return metrics


def generate_short_summary(df: pd.DataFrame, ticker: str) -> str:
    """
    Generate a short, human-readable summary for the stock.
    Assumes compute_indicators() has already been run.
    """
    close = df["Close"].dropna()

    if len(close) < 2:
        return f"{ticker}: Not enough data to analyse."

    start_date = close.index[0].date()
    end_date = close.index[-1].date()
    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])

    total_return = end_price / start_price - 1

    log_ret = df["log_return"].dropna()
    if len(log_ret) == 0:
        return (f"{ticker}: Price moved from {start_price:.2f} to {end_price:.2f} "
                f"({total_return * 100:.1f}% total), but no return stats available.")

    mean_log_daily = log_ret.mean()
    vol_log_daily = log_ret.std()
    trading_days = 252

    ann_return = math.exp(mean_log_daily * trading_days) - 1
    ann_vol = vol_log_daily * math.sqrt(trading_days)
    sharpe = ann_return / ann_vol if ann_vol != 0 else float("nan")
    max_dd = df["max_drawdown"].min() if "max_drawdown" in df.columns else float("nan")

    summary = (
        f"{ticker}: From {start_date} to {end_date}, price moved "
        f"from {start_price:.2f} to {end_price:.2f} "
        f"({total_return * 100:.1f}% total, {ann_return * 100:.1f}% annualised). "
        f"Annualised volatility was {ann_vol * 100:.1f}% with a Sharpe of {sharpe:.2f} "
        f"and a maximum drawdown of {max_dd * 100:.1f}%."
    )

    return summary


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


def plot_correlation_heatmap(corr: pd.DataFrame):
    plt.figure()
    plt.imshow(corr, interpolation="nearest")
    plt.title("Daily Return Correlation")
    plt.colorbar()
    tick_marks = range(len(corr.columns))
    plt.xticks(tick_marks, corr.columns, rotation=45)
    plt.yticks(tick_marks, corr.index)
    plt.tight_layout()


# ---------------------- CLI & Main ---------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extended multi-ticker stock analysis using Stooq data (educational)."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL",
        help="Comma-separated ticker symbols (without .US suffix), e.g. 'AAPL,MSFT,TSLA'",
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
    parser.add_argument(
        "--save-metrics",
        type=str,
        default=None,
        help="Optional path to save metrics table as CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ticker_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not ticker_list:
        print("No valid tickers provided.")
        sys.exit(1)

    print(f"Running analysis for: {', '.join(ticker_list)}")
    print(f"Period: {args.start} to {args.end}")

    all_metrics = []
    daily_return_dict = {}  # for correlation
    dfs_by_ticker = {}

    for ticker in ticker_list:
        df_raw = load_data_stooq(ticker, args.start, args.end)
        if df_raw.empty:
            print(f"Skipping {ticker} due to missing data.")
            continue

        df = compute_indicators(df_raw)
        dfs_by_ticker[ticker] = df

        metrics = compute_metrics(df, ticker)
        all_metrics.append(metrics)

        # Store daily returns for correlation matrix
        daily_return_dict[ticker] = df["daily_return"]

        # Print short summary
        print("\nShort summary:")
        print(generate_short_summary(df, ticker))

        # Plot individual ticker stuff
        if not args.no_plots:
            plot_price_and_ma(df, ticker)
            plot_drawdown(df, ticker)
            plot_rsi_macd(df, ticker)

    # If we have at least 1 ticker with data, show combined metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        print("\n==================== METRICS TABLE ====================")
        print(metrics_df.to_string(index=False))
        print("=======================================================\n")

        # Optionally save metrics to CSV
        if args.save_metrics:
            metrics_df.to_csv(args.save_metrics, index=False)
            print(f"Saved metrics to {args.save_metrics}")

        # Correlation matrix
        if len(daily_return_dict) >= 2:
            returns_df = pd.DataFrame(daily_return_dict)
            corr = returns_df.corr()
            print("Daily return correlation matrix:")
            print(corr)

            if not args.no_plots:
                plot_correlation_heatmap(corr)

    else:
        print("No valid data for any ticker. Exiting.")

    if not args.no_plots and all_metrics:
        plt.show()


if __name__ == "__main__":
    main()
