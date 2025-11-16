"""
portfolio_backtester.py

Simple portfolio backtester using Stooq data via pandas-datareader.

Features:
- Fetches daily prices for multiple tickers
- Builds an equally-weighted or custom-weighted portfolio
- Computes:
    * Daily & cumulative returns
    * Annualised return & volatility
    * Sharpe ratio (rf ≈ 0)
    * Max drawdown
    * Best/worst day
- Optional plot of portfolio equity curve vs individual assets

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

def load_prices_stooq(tickers, start, end):
    """
    Load adjusted close prices (using 'Close') for multiple tickers from Stooq.

    Stooq tickers for US stocks: 'AAPL.US', 'MSFT.US', etc.
    We auto-append '.US' if no suffix is given.
    """
    price_data = {}

    for t in tickers:
        if "." not in t:
            stooq_ticker = t + ".US"
        else:
            stooq_ticker = t

        print(f"Downloading data for {stooq_ticker} ({start} -> {end})...")
        try:
            df = web.DataReader(stooq_ticker, "stooq", start=start, end=end)
        except Exception as e:
            print(f"Error fetching data for {t}: {e}")
            continue

        if df.empty:
            print(f"No data returned for {t}, skipping.")
            continue

        df = df.sort_index()
        if "Close" not in df.columns:
            print(f"'Close' column missing for {t}, skipping.")
            continue

        price_data[t] = df["Close"]

    if not price_data:
        print("Error: no valid data for any ticker.")
        sys.exit(1)

    # Align on common dates (inner join)
    prices = pd.DataFrame(price_data)
    prices = prices.dropna()

    if prices.empty:
        print("Error: no overlapping dates across tickers.")
        sys.exit(1)

    return prices


# ---------------------- Portfolio Logic ---------------------- #

def normalise_weights(raw_weights, n_assets):
    """
    Convert raw weights (list or None) into a proper weight vector that sums to 1.
    """
    if raw_weights is None:
        # equal weights
        w = np.ones(n_assets) / n_assets
    else:
        w = np.array(raw_weights, dtype=float)
        if len(w) != n_assets:
            print("Error: number of weights must match number of tickers.")
            sys.exit(1)
        if np.allclose(w.sum(), 0):
            print("Error: sum of weights is zero.")
            sys.exit(1)
        w = w / w.sum()
    return w


def compute_portfolio_stats(prices: pd.DataFrame, weights: np.ndarray):
    """
    Given price DataFrame (Date x Asset) and weights, compute portfolio stats.

    Returns:
        portfolio_df: DataFrame with 'portfolio_value' and 'portfolio_return'
        stats: dict with summary metrics
    """
    # Start with initial value 1.0
    initial_value = 1.0

    # Compute daily returns for each asset
    asset_returns = prices.pct_change()
    asset_returns = asset_returns.dropna()

    # Portfolio daily returns: weighted sum
    port_ret = (asset_returns * weights).sum(axis=1)

    # Portfolio value over time
    port_value = (1 + port_ret).cumprod() * initial_value

    portfolio_df = pd.DataFrame({
        "portfolio_value": port_value,
        "portfolio_return": port_ret
    })

    # Stats
    start_date = port_value.index[0].date()
    end_date = port_value.index[-1].date()
    start_val = float(port_value.iloc[0])
    end_val = float(port_value.iloc[-1])

    total_return = end_val / start_val - 1

    # Use log returns for annualised stats
    log_ret = np.log(1 + port_ret)
    mean_log_daily = log_ret.mean()
    vol_log_daily = log_ret.std()
    trading_days = 252

    ann_return = math.exp(mean_log_daily * trading_days) - 1
    ann_vol = vol_log_daily * math.sqrt(trading_days)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    # Max drawdown
    running_max = port_value.cummax()
    drawdown = port_value / running_max - 1
    max_dd = drawdown.min()

    # Best / worst day
    best_day_ret = port_ret.max()
    worst_day_ret = port_ret.min()
    best_day = port_ret.idxmax().date()
    worst_day = port_ret.idxmin().date()

    stats = {
        "start_date": start_date,
        "end_date": end_date,
        "total_return_pct": total_return * 100,
        "ann_return_pct": ann_return * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "best_day": best_day,
        "best_day_pct": best_day_ret * 100,
        "worst_day": worst_day,
        "worst_day_pct": worst_day_ret * 100,
    }

    return portfolio_df, stats


def print_portfolio_summary(tickers, weights, stats):
    print("\n================ PORTFOLIO SUMMARY ================")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Weights: {', '.join(f'{w:.2f}' for w in weights)} (normalised)")
    print(f"Period:  {stats['start_date']} -> {stats['end_date']}")
    print(f"Total return:      {stats['total_return_pct']:.2f}%")
    print(f"Annualised return: {stats['ann_return_pct']:.2f}%")
    print(f"Annualised vol:    {stats['ann_vol_pct']:.2f}%")
    print(f"Sharpe (rf≈0):     {stats['sharpe']:.2f}")
    print(f"Max drawdown:      {stats['max_drawdown_pct']:.2f}%")
    print(f"Best day:          {stats['best_day']}  ({stats['best_day_pct']:.2f}%)")
    print(f"Worst day:         {stats['worst_day']} ({stats['worst_day_pct']:.2f}%)")
    print("===================================================\n")


# ---------------------- Plotting ---------------------- #

def plot_portfolio_and_assets(prices: pd.DataFrame, portfolio_df: pd.DataFrame):
    """
    Plot portfolio value vs each individual asset (normalised to 1.0).
    """
    plt.figure()
    # Normalise asset prices to start at 1.0
    norm_prices = prices / prices.iloc[0]
    for col in norm_prices.columns:
        plt.plot(norm_prices.index, norm_prices[col], label=col)
    plt.plot(portfolio_df.index, portfolio_df["portfolio_value"], label="PORTFOLIO", linewidth=2)
    plt.title("Portfolio vs Individual Assets (normalised)")
    plt.xlabel("Date")
    plt.ylabel("Normalised value")
    plt.legend()
    plt.grid(True)


# ---------------------- CLI ---------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple portfolio backtester using Stooq data (educational)."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated ticker symbols (without .US suffix), e.g. 'AAPL,MSFT,TSLA'",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional comma-separated weights, e.g. '0.5,0.3,0.2'. "
             "If omitted, equal weights are used.",
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

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("Error: no valid tickers provided.")
        sys.exit(1)

    if args.weights:
        raw_weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    else:
        raw_weights = None

    print(f"Running portfolio backtest for: {', '.join(tickers)}")
    print(f"Period: {args.start} -> {args.end}")

    prices = load_prices_stooq(tickers, args.start, args.end)

    weights = normalise_weights(raw_weights, n_assets=len(prices.columns))

    portfolio_df, stats = compute_portfolio_stats(prices, weights)
    print_portfolio_summary(list(prices.columns), weights, stats)

    if not args.no_plots:
        plot_portfolio_and_assets(prices, portfolio_df)
        plt.show()


if __name__ == "__main__":
    main()
