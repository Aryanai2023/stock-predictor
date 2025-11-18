"""
portfolio_monte_carlo.py

Performs a Monte Carlo simulation for portfolio future projection.

IMPROVISATION:
- Added --target-annual-return argument so the user can specify a forecast
  for the portfolio's expected return (drift) instead of relying only on
  historical averages.
- Added --sims to control the number of Monte Carlo paths.
- Added --seed for reproducible simulations.
- Added --no-plots to skip plotting.
- Added --out-csv to export simulated paths.

Dependencies:
    pip install pandas numpy pandas-datareader matplotlib

Educational only. Not financial advice.
"""

import argparse
from datetime import datetime, timedelta
import sys
import math

import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt

# --- Configuration ---
TRADING_DAYS = 252


# ---------------------- Date Helpers ---------------------- #


def parse_date(s: str) -> datetime:
    """Parse YYYY-MM-DD date string into datetime."""
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {s}. Use YYYY-MM-DD."
        ) from exc


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


# ---------------------- Data Loading ---------------------- #


def load_prices_stooq(tickers: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    """
    Load close prices for multiple tickers from Stooq.
    Returns DataFrame indexed by date with one column per ticker.
    """
    price_data: dict[str, pd.Series] = {}
    print("Fetching historical data for Monte Carlo simulation...")

    for t in tickers:
        stooq_ticker = _to_stooq_ticker(t)
        print(f"  {stooq_ticker}: {start.date()} -> {end.date()}")
        try:
            df = web.DataReader(stooq_ticker, "stooq", start=start, end=end)
        except Exception as e:  # noqa: BLE001
            print(f"Error fetching data for {t}: {e}")
            continue

        if df.empty or "Close" not in df.columns:
            print(f"No valid data returned for {t}, skipping.")
            continue

        price_data[t] = df.sort_index()["Close"]

    if not price_data:
        print("Error: no valid data for any ticker.")
        sys.exit(1)

    # Align on common dates (inner join)
    prices = pd.DataFrame(price_data).dropna()

    if prices.empty:
        print("Error: no overlapping dates across tickers.")
        sys.exit(1)

    return prices


# ---------------------- Portfolio Logic ---------------------- #


def normalise_weights(raw_weights: list[float] | None, n_assets: int) -> np.ndarray:
    """
    Convert raw weights (list or None) into a proper weight vector that sums to 1.
    """
    if raw_weights is None:
        w = np.ones(n_assets) / n_assets
    else:
        w = np.array(raw_weights, dtype=float)
        if len(w) != n_assets:
            print("Error: number of weights must match number of tickers.")
            sys.exit(1)
        if np.isclose(w.sum(), 0):
            print("Error: sum of weights is zero or near zero.")
            sys.exit(1)
        w = w / w.sum()
    return w


# ---------------------- Monte Carlo Core Logic ---------------------- #


def run_monte_carlo(
    prices: pd.DataFrame,
    weights: np.ndarray,
    days_to_project: int,
    initial_value: float = 1_000.0,
    target_annual_return: float = 0.0,
    num_simulations: int = 1_000,
    rng_seed: int | None = None,
) -> pd.DataFrame:
    """
    Runs the Monte Carlo simulation to project future portfolio values.
    Uses the last price date as the starting point.

    Returns:
        simulation_df: DataFrame indexed by future dates, one column per simulation.
    """
    # 1. Calculate historical metrics to determine VOLATILITY
    daily_returns = prices.pct_change().dropna()
    cov_matrix = daily_returns.cov()

    # 2. Calculate portfolio standard deviation (volatility) from historical data
    port_stdev = float(np.sqrt(weights.T @ cov_matrix.values @ weights))

    # 3. Calculate daily drift (expected return) based on the target annual return
    #    Formula for daily drift (mu_daily): (1 + annual_return)^(1/TRADING_DAYS) - 1
    if target_annual_return <= -1.0:
        print("Error: Target Annual Return must be greater than -1 (i.e., loss less than 100%).")
        sys.exit(1)

    daily_drift_target = math.pow(1 + target_annual_return, 1 / TRADING_DAYS) - 1

    # 4. Prepare RNG and run simulation
    rng = np.random.default_rng(rng_seed)

    print(f"Running {num_simulations} simulations over {days_to_project} trading days...")
    print(f"  Implied daily drift: {daily_drift_target * 100:.4f}%")
    print(f"  Historical daily volatility: {port_stdev * 100:.4f}%\n")

    # 5. Generate random daily returns for all paths at once
    #    Daily Return ~ Normal(daily_drift_target, port_stdev)
    daily_returns_paths = rng.normal(
        loc=daily_drift_target,
        scale=port_stdev,
        size=(days_to_project, num_simulations),
    )

    # 6. Compute cumulative portfolio values for each path
    cumulative_returns_paths = np.cumprod(1 + daily_returns_paths, axis=0) * initial_value

    # 7. Create a date index for the projection horizon
    last_date = prices.index[-1]
    date_range = pd.to_datetime(
        [last_date + timedelta(days=d) for d in range(1, days_to_project + 1)]
    )

    # 8. Build the DataFrame
    columns = [f"Sim_{i+1}" for i in range(num_simulations)]
    simulation_df = pd.DataFrame(cumulative_returns_paths, index=date_range, columns=columns)

    return simulation_df


# ---------------------- Reporting and Plotting ---------------------- #


def print_simulation_summary(
    simulation_df: pd.DataFrame,
    target_annual_return: float,
    initial_value: float,
    num_simulations: int,
) -> None:
    """
    Calculate and print key percentile metrics from the simulation results.
    """
    final_values = simulation_df.iloc[-1]

    # Calculate key percentiles for the final value
    best_case = final_values.quantile(0.99)
    very_likely_best = final_values.quantile(0.75)
    median = final_values.quantile(0.50)
    very_likely_worst = final_values.quantile(0.25)
    worst_case = final_values.quantile(0.01)

    horizon_days = len(simulation_df)
    horizon_years = horizon_days / TRADING_DAYS

    print("\n=============== MONTE CARLO SUMMARY ===============")
    print(f"Simulations:         {num_simulations}")
    print(f"Projection Days:     {horizon_days} (~{horizon_years:.2f} years)")
    print(f"Target Annual Return: {target_annual_return * 100:,.2f}%")
    print(f"Initial Value:       ${initial_value:,.2f}")
    print("---------------------------------------------------")
    print(f"99th Percentile (Best Case): ${best_case:,.2f}")
    print(f"75th Percentile:             ${very_likely_best:,.2f}")
    print(f"50th Percentile (Median):    ${median:,.2f}")
    print(f"25th Percentile:             ${very_likely_worst:,.2f}")
    print(f"1st Percentile (Worst Case): ${worst_case:,.2f}")
    print("===================================================\n")


def plot_monte_carlo_results(simulation_df: pd.DataFrame) -> None:
    """
    Plot all simulated paths and highlight key percentile lines.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(12, 8))

    # Plot all individual simulation paths (faded)
    plt.plot(simulation_df.index, simulation_df, color="gray", alpha=0.1, linewidth=0.5)

    # Calculate daily percentile lines
    upper_99 = simulation_df.quantile(0.99, axis=1)
    upper_75 = simulation_df.quantile(0.75, axis=1)
    median = simulation_df.quantile(0.50, axis=1)
    lower_25 = simulation_df.quantile(0.25, axis=1)
    lower_01 = simulation_df.quantile(0.01, axis=1)

    # Plot percentile lines
    plt.plot(median.index, median, label="Median (50%)", linewidth=2.5)
    plt.plot(upper_75.index, upper_75, label="75th Percentile", linewidth=1.5)
    plt.plot(lower_25.index, lower_25, label="25th Percentile", linewidth=1.5)

    # Shade the range between 25th and 75th percentiles (the "likely" zone)
    plt.fill_between(
        simulation_df.index,
        lower_25,
        upper_75,
        alpha=0.2,
        label="50% Confidence Interval",
    )

    # Optionally show 1% and 99% as thin boundary lines
    plt.plot(upper_99.index, upper_99, linewidth=1.0, linestyle="--", label="99th Percentile")
    plt.plot(lower_01.index, lower_01, linewidth=1.0, linestyle="--", label="1st Percentile")

    plt.title("Portfolio Monte Carlo Simulation Projection", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Value ($)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()


# ---------------------- CLI ---------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portfolio Monte Carlo simulation using Stooq data (educational only, not financial advice)."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated ticker symbols (e.g. 'AAPL,MSFT,GOOG').",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional comma-separated weights. If omitted, equal weights are used.",
    )
    parser.add_argument(
        "--history-start",
        type=parse_date,
        default=parse_date("2020-01-01"),
        help="Start date for historical data (YYYY-MM-DD, default: 2020-01-01).",
    )
    parser.add_argument(
        "--history-end",
        type=parse_date,
        default=datetime.today(),
        help="End date for historical data (YYYY-MM-DD, default: today).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=TRADING_DAYS,
        help=f"Number of future trading days to simulate (default: {TRADING_DAYS} ≈ 1 year).",
    )
    parser.add_argument(
        "--initial",
        type=float,
        default=1_000.0,
        help="Initial portfolio value (default: 1000.0).",
    )
    parser.add_argument(
        "--target-annual-return",
        type=float,
        default=0.10,  # Default to a realistic 10% annual return
        help="Target annual return (drift) to use in the simulation (e.g., 0.10 for 10%).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=1_000,
        help="Number of Monte Carlo simulations (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible simulations.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, do not show plots (print summary only).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to save simulated paths as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.history_start >= args.history_end:
        print("Error: history start date must be before history end date.")
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

    print(f"Assets:             {', '.join(tickers)}")
    print(f"Historical period:  {args.history_start.date()} -> {args.history_end.date()}")
    print(f"Projection period:  {args.days} trading days")
    print(f"Initial investment: ${args.initial:,.2f}")
    print(f"Target annual return (drift): {args.target_annual_return * 100:.2f}%")
    print(f"Simulations:        {args.sims}")
    if args.seed is not None:
        print(f"Random seed:        {args.seed}")
    print()

    prices = load_prices_stooq(tickers, args.history_start, args.history_end)
    weights = normalise_weights(raw_weights, n_assets=len(prices.columns))
    print(f"Normalised weights: {', '.join(f'{w:.2f}' for w in weights)}\n")

    # Run the simulation, passing the new target return and simulation count
    simulation_df = run_monte_carlo(
        prices,
        weights,
        days_to_project=args.days,
        initial_value=args.initial,
        target_annual_return=args.target_annual_return,
        num_simulations=args.sims,
        rng_seed=args.seed,
    )

    # Output results
    print_simulation_summary(
        simulation_df,
        target_annual_return=args.target_annual_return,
        initial_value=args.initial,
        num_simulations=args.sims,
    )

    if args.out_csv:
        try:
            simulation_df.to_csv(args.out_csv, index=True)
            print(f"Saved Monte Carlo paths to {args.out_csv}")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not save CSV to {args.out_csv}: {e}")

    # Plot results
    if not args.no_plots:
        plot_monte_carlo_results(simulation_df)
        plt.show()


if __name__ == "__main__":
    main()
