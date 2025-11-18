"""
portfolio_backtester.py

Simple portfolio backtester using Stooq data via pandas-datareader.

Improvements:
- Configurable annual risk-free rate for Sharpe Ratio calculation.
- Optional benchmark stats.
- Cleaned argument parsing and error handling.

Dependencies:
    pip install pandas numpy pandas-datareader matplotlib

Features:
- Fetches daily prices for multiple tickers
- Builds an equally-weighted or custom-weighted portfolio
- Computes:
    * Daily & cumulative returns
    * Annualised return & volatility
    * Sharpe ratio (configurable with risk-free rate)
    * Max drawdown
    * Best/worst day
- Per-asset stats table (return, volatility, Sharpe)
- Optional benchmark comparison
- Optional CSV export of portfolio equity curve
- Optional plot of portfolio equity curve vs individual assets (and benchmark)

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


TRADING_DAYS = 252


# ---------------------- Date helpers ---------------------- #


def parse_date(s: str) -> datetime:
    """Parse YYYY-MM-DD date string into datetime."""
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {s}. Use YYYY-MM-DD."
        ) from exc


# ---------------------- Data Loading ---------------------- #


def _to_stooq_ticker(t: str) -> str:
    """
    Convert a ticker to Stooq format.
    For US stocks: 'AAPL' -> 'AAPL.US'
    Leaves indices / already-qualified tickers unchanged, e.g. '^SPX', 'AAPL.US'.
    """
    t = t.strip()
    if not t:
        return t
    if "." in t or "^" in t:
        return t
    return t + ".US"


def load_prices_stooq(tickers: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    """
    Load close prices for multiple tickers from Stooq.

    Returns DataFrame indexed by date with one column per ticker.
    Exits if no valid data or no overlapping dates.
    """
    price_data: dict[str, pd.Series] = {}

    for t in tickers:
        stooq_ticker = _to_stooq_ticker(t)
        print(f"Downloading data for {stooq_ticker} ({start.date()} -> {end.date()})...")
        try:
            # web.DataReader returns prices in descending order, we sort later
            df = web.DataReader(stooq_ticker, "stooq", start=start, end=end)
        except Exception as e:  # noqa: BLE001
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
    prices = pd.DataFrame(price_data).dropna()

    if prices.empty:
        print("Error: no overlapping dates across tickers.")
        sys.exit(1)

    return prices


def load_single_price_stooq(ticker: str, start: datetime, end: datetime) -> pd.Series:
    """Load a single benchmark series (Close) from Stooq."""
    stooq_ticker = _to_stooq_ticker(ticker)
    print(f"Downloading benchmark {stooq_ticker} ({start.date()} -> {end.date()})...")
    try:
        df = web.DataReader(stooq_ticker, "stooq", start=start, end=end)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not fetch benchmark {ticker}: {e}")
        return pd.Series(dtype=float, name=ticker)

    if df.empty or "Close" not in df.columns:
        print(f"Warning: empty or invalid benchmark data for {ticker}")
        return pd.Series(dtype=float, name=ticker)

    df = df.sort_index()
    return df["Close"].rename(ticker)


# ---------------------- Portfolio Logic ---------------------- #


def normalise_weights(raw_weights: list[float] | None, n_assets: int) -> np.ndarray:
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
        # Check if weights sum to near zero, which is likely an error
        if np.isclose(w.sum(), 0):
            print("Error: sum of weights is zero or near zero.")
            sys.exit(1)
        w = w / w.sum()
    return w


def compute_portfolio_stats(
    prices: pd.DataFrame,
    weights: np.ndarray,
    initial_value: float = 1_000.0,
    risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Given price DataFrame (Date x Asset) and weights, compute portfolio stats.

    Returns:
        portfolio_df: DataFrame with 'portfolio_value' and 'portfolio_return'
        stats: dict with summary metrics
        asset_returns: DataFrame with daily asset returns
    """
    # Compute daily returns for each asset
    asset_returns = prices.pct_change().dropna()

    # Portfolio daily returns: weighted sum
    port_ret = (asset_returns * weights).sum(axis=1)

    # Portfolio value over time
    port_value = (1 + port_ret).cumprod() * initial_value

    portfolio_df = pd.DataFrame(
        {
            "portfolio_value": port_value,
            "portfolio_return": port_ret,
        }
    )

    # --- Stats Calculation ---
    start_date = port_value.index[0].date()
    end_date = port_value.index[-1].date()
    start_val = float(port_value.iloc[0])
    end_val = float(port_value.iloc[-1])

    total_return = end_val / start_val - 1

    # Use log returns for more accurate annualised stats (compounding)
    log_ret = np.log(1 + port_ret)
    mean_log_daily = log_ret.mean()
    vol_log_daily = log_ret.std()

    ann_return = math.exp(mean_log_daily * TRADING_DAYS) - 1
    ann_vol = vol_log_daily * math.sqrt(TRADING_DAYS)

    # Sharpe Ratio: (Annualized Return - Risk-Free Rate) / Annualized Volatility
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else np.nan

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
        "initial_value": initial_value,
        "final_value": end_val,
    }

    return portfolio_df, stats, asset_returns


def compute_per_asset_stats(
    asset_returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Compute annualised return, volatility and Sharpe per asset.
    Returns a DataFrame (asset x metric).
    """
    log_ret = np.log(1 + asset_returns)

    mean_log_daily = log_ret.mean()
    vol_log_daily = log_ret.std()

    ann_return = np.exp(mean_log_daily * TRADING_DAYS) - 1
    ann_vol = vol_log_daily * np.sqrt(TRADING_DAYS)

    # Sharpe Ratio: (Annualized Return - Risk-Free Rate) / Annualized Volatility
    sharpe = (ann_return - risk_free_rate) / ann_vol.replace(0, np.nan)

    stats_df = pd.DataFrame(
        {
            "ann_return_pct": ann_return * 100,
            "ann_vol_pct": ann_vol * 100,
            "sharpe": sharpe,
        }
    )

    return stats_df


def compute_benchmark_stats(
    benchmark: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict | None:
    """
    Compute simple stats for a benchmark series (Close prices).
    """
    benchmark = benchmark.dropna()
    if benchmark.empty:
        return None

    ret = benchmark.pct_change().dropna()
    log_ret = np.log(1 + ret)

    mean_log_daily = log_ret.mean()
    vol_log_daily = log_ret.std()

    ann_return = math.exp(mean_log_daily * TRADING_DAYS) - 1
    ann_vol = vol_log_daily * math.sqrt(TRADING_DAYS)

    # Sharpe Ratio: (Annualized Return - Risk-Free Rate) / Annualized Volatility
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else np.nan

    start_price = float(benchmark.iloc[0])
    end_price = float(benchmark.iloc[-1])
    total_return = end_price / start_price - 1

    return {
        "start_date": benchmark.index[0].date(),
        "end_date": benchmark.index[-1].date(),
        "total_return_pct": total_return * 100,
        "ann_return_pct": ann_return * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": sharpe,
    }


def print_portfolio_summary(
    tickers: list[str],
    weights: np.ndarray,
    stats: dict,
    risk_free_rate: float,
    asset_stats: pd.DataFrame | None = None,
    benchmark_stats: dict | None = None,
) -> None:
    print("\n================ PORTFOLIO SUMMARY ================")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Weights: {', '.join(f'{w:.2f}' for w in weights)} (normalised)")
    print(f"Period:        {stats['start_date']} -> {stats['end_date']}")
    print(f"Initial value: {stats['initial_value']:.2f}")
    print(f"Final value:   {stats['final_value']:.2f}")
    print(f"Total return:  {stats['total_return_pct']:.2f}%")
    print(f"Ann. return:   {stats['ann_return_pct']:.2f}%")
    print(f"Ann. vol:      {stats['ann_vol_pct']:.2f}%")
    print(f"Sharpe (rf={risk_free_rate:.2f}): {stats['sharpe']:.2f}")
    print(f"Max drawdown:  {stats['max_drawdown_pct']:.2f}%")
    print(f"Best day:      {stats['best_day']}  ({stats['best_day_pct']:.2f}%)")
    print(f"Worst day:     {stats['worst_day']} ({stats['worst_day_pct']:.2f}%)")
    print("===================================================\n")

    if asset_stats is not None:
        print("Per-asset metrics (annualised):")
        print(asset_stats.round(2).to_string())
        print()

    if benchmark_stats is not None:
        print("Benchmark comparison:")
        print(f"Period:       {benchmark_stats['start_date']} -> {benchmark_stats['end_date']}")
        print(f"Total return: {benchmark_stats['total_return_pct']:.2f}%")
        print(f"Ann. return:  {benchmark_stats['ann_return_pct']:.2f}%")
        print(f"Ann. vol:     {benchmark_stats['ann_vol_pct']:.2f}%")
        print(f"Sharpe:       {benchmark_stats['sharpe']:.2f}")
        print("===================================================\n")


# ---------------------- Plotting ---------------------- #


def plot_portfolio_and_assets(
    prices: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    benchmark: pd.Series | None = None,
) -> None:
    """
    Plot portfolio value vs each individual asset (normalised to 1.0).
    If benchmark is provided, also plot it.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(12, 7))

    # Normalise asset prices to start at 1.0
    norm_prices = prices / prices.iloc[0]
    for col in norm_prices.columns:
        plt.plot(norm_prices.index, norm_prices[col], label=col, alpha=0.6, linewidth=1.5)

    # Normalise portfolio so it also starts at 1.0 for visual comparison
    port_norm = portfolio_df["portfolio_value"] / portfolio_df["portfolio_value"].iloc[0]
    plt.plot(port_norm.index, port_norm, label="PORTFOLIO", linewidth=3)

    if benchmark is not None and not benchmark.empty:
        bench_norm = benchmark / benchmark.iloc[0]
        bench_name = benchmark.name if benchmark.name else "BENCHMARK"
        plt.plot(bench_norm.index, bench_norm, label=bench_name, linestyle="--", linewidth=2)

    plt.title("Portfolio vs Assets and Benchmark (Normalised to 1.0)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalised Value", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()


# ---------------------- CLI ---------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple portfolio backtester using Stooq data (educational only, not financial advice)."
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
        help=(
            "Optional comma-separated weights, e.g. '0.5,0.3,0.2'. "
            "If omitted, equal weights are used."
        ),
    )
    parser.add_argument(
        "--start",
        type=parse_date,
        default=parse_date("2015-01-01"),
        help="Start date YYYY-MM-DD (default: 2015-01-01)",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=datetime.today(),
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--initial",
        type=float,
        default=1_000.0,
        help="Initial portfolio value (default: 1000.0)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Optional benchmark ticker (e.g. 'SPY').",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annual risk-free rate for Sharpe calculation (e.g. 0.03 for 3%). Default is 0.0.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, do not show plots (summary only).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to save portfolio equity curve as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start >= args.end:
        print("Error: start date must be before end date.")
        sys.exit(1)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("Error: no valid tickers provided.")
        sys.exit(1)

    if args.weights:
        try:
            raw_weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
        except ValueError:
            print("Error: could not parse weights as floats.")
            sys.exit(1)
    else:
        raw_weights = None

    print(f"Running portfolio backtest for: {', '.join(tickers)}")
    print(f"Period: {args.start.date()} -> {args.end.date()}")
    print(f"Initial value: {args.initial:.2f}")
    print(f"Risk-free rate (rf): {args.risk_free_rate * 100:.2f}%\n")

    prices = load_prices_stooq(tickers, args.start, args.end)
    weights = normalise_weights(raw_weights, n_assets=len(prices.columns))

    portfolio_df, stats, asset_returns = compute_portfolio_stats(
        prices,
        weights,
        initial_value=args.initial,
        risk_free_rate=args.risk_free_rate,
    )
    asset_stats = compute_per_asset_stats(asset_returns, risk_free_rate=args.risk_free_rate)

    benchmark_stats = None
    benchmark_series = pd.Series(dtype=float)

    if args.benchmark:
        benchmark_series = load_single_price_stooq(
            args.benchmark.upper(),
            args.start,
            args.end,
        )
        if not benchmark_series.empty:
            benchmark_stats = compute_benchmark_stats(
                benchmark_series,
                risk_free_rate=args.risk_free_rate,
            )

    print_portfolio_summary(
        list(prices.columns),
        weights,
        stats,
        args.risk_free_rate,
        asset_stats=asset_stats,
        benchmark_stats=benchmark_stats,
    )

    if args.out_csv:
        try:
            portfolio_df.to_csv(args.out_csv, index=True)
            print(f"Saved portfolio equity curve to {args.out_csv}")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not save CSV to {args.out_csv}: {e}")

    if not args.no_plots:
        plot_portfolio_and_assets(prices, portfolio_df, benchmark=benchmark_series)
        plt.show()


if __name__ == "__main__":
    main()
