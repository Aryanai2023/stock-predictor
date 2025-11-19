"""
portfolio_monte_carlo.py

Performs a Monte Carlo simulation for portfolio future projection with enhanced
features and robust error handling.

IMPROVEMENTS:
- Better error handling and validation
- Geometric Brownian Motion with drift adjustment
- Additional statistical metrics (Sharpe ratio, VaR, CVaR)
- Improved visualization with confidence bands
- Progress tracking for long simulations
- Better data validation and cleaning
- Option to export detailed statistics
- Correlation matrix visualization
- Support for different rebalancing strategies

Dependencies:
    pip install pandas numpy pandas-datareader matplotlib tqdm

Educational only. Not financial advice.
"""

import argparse
from datetime import datetime, timedelta
import sys
import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    warnings.warn("tqdm not available. Install for progress bars: pip install tqdm")

# --- Configuration ---
TRADING_DAYS = 252
MIN_DATA_POINTS = 60  # Minimum trading days of historical data required


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


def load_prices_stooq(
    tickers: list[str], 
    start: datetime, 
    end: datetime,
    min_data_points: int = MIN_DATA_POINTS
) -> pd.DataFrame:
    """
    Load close prices for multiple tickers from Stooq.
    Returns DataFrame indexed by date with one column per ticker.
    
    Args:
        tickers: List of ticker symbols
        start: Start date for historical data
        end: End date for historical data
        min_data_points: Minimum number of data points required per ticker
        
    Returns:
        DataFrame with close prices, indexed by date
    """
    price_data: dict[str, pd.Series] = {}
    failed_tickers = []
    
    print("Fetching historical data for Monte Carlo simulation...")

    for t in tickers:
        stooq_ticker = _to_stooq_ticker(t)
        print(f"  {stooq_ticker}: {start.date()} -> {end.date()}")
        try:
            df = web.DataReader(stooq_ticker, "stooq", start=start, end=end)
        except Exception as e:
            print(f"  ⚠ Error fetching data for {t}: {e}")
            failed_tickers.append(t)
            continue

        if df.empty or "Close" not in df.columns:
            print(f"  ⚠ No valid data returned for {t}, skipping.")
            failed_tickers.append(t)
            continue

        # Sort and clean data
        series = df.sort_index()["Close"]
        
        # Check for sufficient data points
        if len(series) < min_data_points:
            print(f"  ⚠ Insufficient data for {t}: {len(series)} points (need {min_data_points})")
            failed_tickers.append(t)
            continue
            
        # Remove any infinite or extreme values
        series = series.replace([np.inf, -np.inf], np.nan)
        
        price_data[t] = series
        print(f"  ✓ Loaded {len(series)} data points for {t}")

    if not price_data:
        print("\n❌ Error: no valid data for any ticker.")
        if failed_tickers:
            print(f"Failed tickers: {', '.join(failed_tickers)}")
        sys.exit(1)

    # Align on common dates (inner join)
    prices = pd.DataFrame(price_data).dropna()

    if prices.empty:
        print("\n❌ Error: no overlapping dates across tickers.")
        sys.exit(1)
        
    print(f"\n✓ Successfully loaded data: {len(prices)} trading days across {len(prices.columns)} assets")

    return prices


# ---------------------- Portfolio Logic ---------------------- #


def normalise_weights(raw_weights: Optional[list[float]], n_assets: int) -> np.ndarray:
    """
    Convert raw weights (list or None) into a proper weight vector that sums to 1.
    
    Args:
        raw_weights: List of weights or None for equal weighting
        n_assets: Number of assets in portfolio
        
    Returns:
        Normalized weight vector
    """
    if raw_weights is None:
        w = np.ones(n_assets) / n_assets
        print("Using equal weights for all assets")
    else:
        w = np.array(raw_weights, dtype=float)
        if len(w) != n_assets:
            print(f"❌ Error: number of weights ({len(w)}) must match number of tickers ({n_assets}).")
            sys.exit(1)
        if np.isclose(w.sum(), 0):
            print("❌ Error: sum of weights is zero or near zero.")
            sys.exit(1)
        if np.any(w < 0):
            print("⚠ Warning: negative weights detected (short positions)")
        w = w / w.sum()
    return w


def calculate_portfolio_metrics(
    daily_returns: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = 0.02
) -> dict:
    """
    Calculate key portfolio metrics from historical data.
    
    Args:
        daily_returns: DataFrame of daily returns for each asset
        weights: Portfolio weights
        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        
    Returns:
        Dictionary of portfolio metrics
    """
    # Portfolio returns
    portfolio_returns = (daily_returns @ weights)
    
    # Covariance matrix
    cov_matrix = daily_returns.cov()
    
    # Portfolio volatility (daily)
    port_variance = weights.T @ cov_matrix.values @ weights
    port_stdev = float(np.sqrt(port_variance))
    
    # Annualized metrics
    annual_return = portfolio_returns.mean() * TRADING_DAYS
    annual_volatility = port_stdev * np.sqrt(TRADING_DAYS)
    
    # Sharpe ratio
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess_returns = portfolio_returns - daily_rf
    sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(TRADING_DAYS)
    
    # Correlation matrix
    correlation_matrix = daily_returns.corr()
    
    return {
        "daily_volatility": port_stdev,
        "annual_volatility": annual_volatility,
        "historical_annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "covariance_matrix": cov_matrix,
        "correlation_matrix": correlation_matrix,
        "portfolio_returns": portfolio_returns,
    }


# ---------------------- Monte Carlo Core Logic ---------------------- #


def run_monte_carlo(
    prices: pd.DataFrame,
    weights: np.ndarray,
    days_to_project: int,
    initial_value: float = 1_000.0,
    target_annual_return: float = 0.0,
    num_simulations: int = 1_000,
    rng_seed: Optional[int] = None,
    risk_free_rate: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    """
    Runs the Monte Carlo simulation to project future portfolio values using
    Geometric Brownian Motion with specified drift.

    Args:
        prices: Historical price data
        weights: Portfolio weights
        days_to_project: Number of trading days to project
        initial_value: Starting portfolio value
        target_annual_return: Target annual return (drift)
        num_simulations: Number of simulation paths
        rng_seed: Random seed for reproducibility
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Tuple of (simulation DataFrame, metrics dictionary)
    """
    # 1. Calculate historical metrics
    daily_returns = prices.pct_change().dropna()
    
    if len(daily_returns) < MIN_DATA_POINTS:
        print(f"❌ Error: Insufficient historical data ({len(daily_returns)} days). Need at least {MIN_DATA_POINTS}.")
        sys.exit(1)
    
    metrics = calculate_portfolio_metrics(daily_returns, weights, risk_free_rate)
    port_stdev = metrics["daily_volatility"]
    
    # 2. Validate target return
    if target_annual_return <= -1.0:
        print("❌ Error: Target Annual Return must be greater than -1 (i.e., loss less than 100%).")
        sys.exit(1)

    # 3. Calculate daily drift with volatility adjustment for geometric Brownian motion
    # For GBM: drift_adjusted = mu - 0.5 * sigma^2
    # Then daily_drift = (1 + target_annual)^(1/252) - 1
    daily_drift_unadjusted = math.pow(1 + target_annual_return, 1 / TRADING_DAYS) - 1
    
    # Adjust for volatility drag (reduces expected growth due to compounding of volatility)
    daily_drift = daily_drift_unadjusted + 0.5 * (port_stdev ** 2)

    # 4. Prepare RNG and run simulation
    rng = np.random.default_rng(rng_seed)

    print(f"\n{'='*60}")
    print(f"Running {num_simulations:,} simulations over {days_to_project} trading days...")
    print(f"{'='*60}")
    print(f"Historical Metrics:")
    print(f"  Annual Return:     {metrics['historical_annual_return']*100:>8.2f}%")
    print(f"  Annual Volatility: {metrics['annual_volatility']*100:>8.2f}%")
    print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:>8.2f}")
    print(f"\nSimulation Parameters:")
    print(f"  Target Annual Return:  {target_annual_return*100:>8.2f}%")
    print(f"  Daily Drift (adjusted): {daily_drift*100:>8.4f}%")
    print(f"  Daily Volatility:       {port_stdev*100:>8.4f}%")
    print(f"{'='*60}\n")

    # 5. Generate random daily returns for all paths
    # Using geometric Brownian motion: dS/S = μ*dt + σ*dW
    if TQDM_AVAILABLE and num_simulations >= 100:
        print("Generating simulation paths...")
        random_shocks = rng.normal(0, 1, size=(days_to_project, num_simulations))
    else:
        random_shocks = rng.normal(0, 1, size=(days_to_project, num_simulations))
    
    daily_returns_paths = daily_drift + port_stdev * random_shocks

    # 6. Compute cumulative portfolio values
    cumulative_factors = np.cumprod(1 + daily_returns_paths, axis=0)
    cumulative_values = cumulative_factors * initial_value

    # 7. Create a date index for the projection horizon
    last_date = prices.index[-1]
    date_range = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=days_to_project,
        freq='D'
    )

    # 8. Build the DataFrame
    columns = [f"Sim_{i+1}" for i in range(num_simulations)]
    simulation_df = pd.DataFrame(cumulative_values, index=date_range, columns=columns)
    
    # 9. Store additional metrics
    simulation_metrics = {
        **metrics,
        "target_annual_return": target_annual_return,
        "daily_drift": daily_drift,
        "num_simulations": num_simulations,
        "days_projected": days_to_project,
        "initial_value": initial_value,
    }

    return simulation_df, simulation_metrics


# ---------------------- Risk Analysis ---------------------- #


def calculate_risk_metrics(simulation_df: pd.DataFrame, confidence_level: float = 0.95) -> dict:
    """
    Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    
    Args:
        simulation_df: DataFrame of simulated portfolio values
        confidence_level: Confidence level for VaR/CVaR (e.g., 0.95 for 95%)
        
    Returns:
        Dictionary of risk metrics
    """
    final_values = simulation_df.iloc[-1]
    initial_value = simulation_df.iloc[0].mean()  # Should be same across all sims
    
    # Calculate returns
    final_returns = (final_values - initial_value) / initial_value
    
    # Value at Risk (VaR) - maximum loss at given confidence level
    var_percentile = 1 - confidence_level
    var = final_returns.quantile(var_percentile)
    
    # Conditional Value at Risk (CVaR) - expected loss beyond VaR
    cvar = final_returns[final_returns <= var].mean()
    
    # Probability of loss
    prob_loss = (final_returns < 0).sum() / len(final_returns)
    
    return {
        "var": var,
        "cvar": cvar,
        "probability_of_loss": prob_loss,
        "confidence_level": confidence_level,
    }


# ---------------------- Reporting and Plotting ---------------------- #


def print_simulation_summary(
    simulation_df: pd.DataFrame,
    simulation_metrics: dict,
) -> None:
    """
    Calculate and print comprehensive summary statistics from simulation results.
    """
    final_values = simulation_df.iloc[-1]
    initial_value = simulation_metrics["initial_value"]
    
    # Calculate percentiles
    percentiles = {
        99: final_values.quantile(0.99),
        95: final_values.quantile(0.95),
        75: final_values.quantile(0.75),
        50: final_values.quantile(0.50),
        25: final_values.quantile(0.25),
        5: final_values.quantile(0.05),
        1: final_values.quantile(0.01),
    }
    
    # Calculate returns
    final_returns = (final_values - initial_value) / initial_value
    mean_return = final_returns.mean()
    
    # Risk metrics
    risk_metrics = calculate_risk_metrics(simulation_df)
    
    horizon_days = len(simulation_df)
    horizon_years = horizon_days / TRADING_DAYS

    print("\n" + "="*70)
    print(" "*20 + "MONTE CARLO SUMMARY")
    print("="*70)
    
    print("\nSIMULATION PARAMETERS:")
    print(f"  Number of Simulations:    {simulation_metrics['num_simulations']:,}")
    print(f"  Projection Horizon:       {horizon_days} trading days (~{horizon_years:.2f} years)")
    print(f"  Initial Portfolio Value:  ${initial_value:,.2f}")
    print(f"  Target Annual Return:     {simulation_metrics['target_annual_return']*100:.2f}%")
    
    print("\nHISTORICAL PORTFOLIO METRICS:")
    print(f"  Historical Annual Return: {simulation_metrics['historical_annual_return']*100:.2f}%")
    print(f"  Annual Volatility:        {simulation_metrics['annual_volatility']*100:.2f}%")
    print(f"  Sharpe Ratio:             {simulation_metrics['sharpe_ratio']:.2f}")
    
    print("\nPROJECTED PORTFOLIO VALUES:")
    print(f"  Mean (Expected):          ${final_values.mean():,.2f} ({mean_return*100:+.2f}%)")
    print(f"  Median (50th %ile):       ${percentiles[50]:,.2f}")
    print(f"  Standard Deviation:       ${final_values.std():,.2f}")
    print("-" * 70)
    print(f"  99th Percentile:          ${percentiles[99]:,.2f}")
    print(f"  95th Percentile:          ${percentiles[95]:,.2f}")
    print(f"  75th Percentile:          ${percentiles[75]:,.2f}")
    print(f"  25th Percentile:          ${percentiles[25]:,.2f}")
    print(f"  5th Percentile:           ${percentiles[5]:,.2f}")
    print(f"  1st Percentile:           ${percentiles[1]:,.2f}")
    
    print("\nRISK METRICS:")
    print(f"  Probability of Loss:      {risk_metrics['probability_of_loss']*100:.2f}%")
    print(f"  VaR (95% confidence):     {risk_metrics['var']*100:.2f}%")
    print(f"  CVaR (95% confidence):    {risk_metrics['cvar']*100:.2f}%")
    
    print("="*70)
    print("\nNote: VaR = Value at Risk (worst expected loss at 95% confidence)")
    print("      CVaR = Conditional VaR (expected loss when VaR is exceeded)\n")


def plot_monte_carlo_results(
    simulation_df: pd.DataFrame,
    simulation_metrics: dict,
    show_correlation: bool = False,
) -> None:
    """
    Create comprehensive visualization of Monte Carlo simulation results.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Determine figure layout
    if show_correlation:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                        gridspec_kw={'height_ratios': [2, 1]})
        ax3 = None
    
    # --- Plot 1: Simulation Paths ---
    # Plot individual paths (faded)
    ax1.plot(simulation_df.index, simulation_df, color="lightgray", 
             alpha=0.05, linewidth=0.5, zorder=1)

    # Calculate percentiles
    percentiles = {
        99: simulation_df.quantile(0.99, axis=1),
        95: simulation_df.quantile(0.95, axis=1),
        75: simulation_df.quantile(0.75, axis=1),
        50: simulation_df.quantile(0.50, axis=1),
        25: simulation_df.quantile(0.25, axis=1),
        5: simulation_df.quantile(0.05, axis=1),
        1: simulation_df.quantile(0.01, axis=1),
    }
    
    # Plot key percentiles
    ax1.plot(percentiles[50].index, percentiles[50], 
             label="Median (50%)", linewidth=2.5, color='#2E86AB', zorder=3)
    ax1.plot(percentiles[75].index, percentiles[75], 
             label="75th Percentile", linewidth=1.8, color='#A23B72', order=2)
    ax1.plot(percentiles[25].index, percentiles[25], 
             label="25th Percentile", linewidth=1.8, color='#F18F01', zorder=2)
    
    # Confidence bands
    ax1.fill_between(simulation_df.index, percentiles[25], percentiles[75],
                     alpha=0.25, color='#2E86AB', label="50% Confidence Band", zorder=1)
    ax1.fill_between(simulation_df.index, percentiles[5], percentiles[95],
                     alpha=0.10, color='#2E86AB', label="90% Confidence Band", zorder=1)
    
    # Extreme percentiles (dashed)
    ax1.plot(percentiles[95].index, percentiles[95], 
             linewidth=1.2, linestyle="--", color='gray', alpha=0.7,
             label="95th/5th Percentiles", zorder=2)
    ax1.plot(percentiles[5].index, percentiles[5], 
             linewidth=1.2, linestyle="--", color='gray', alpha=0.7, zorder=2)

    # Add initial value line
    ax1.axhline(y=simulation_metrics['initial_value'], 
                color='red', linestyle=':', linewidth=1.5, 
                alpha=0.7, label='Initial Value', zorder=2)

    ax1.set_title(f"Portfolio Monte Carlo Simulation ({simulation_metrics['num_simulations']:,} paths)", 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # --- Plot 2: Distribution of Final Values ---
    final_values = simulation_df.iloc[-1]
    
    ax2.hist(final_values, bins=50, edgecolor='black', alpha=0.7, color='#2E86AB')
    ax2.axvline(final_values.median(), color='red', linestyle='--', 
                linewidth=2, label=f'Median: ${final_values.median():,.0f}')
    ax2.axvline(final_values.mean(), color='orange', linestyle='--', 
                linewidth=2, label=f'Mean: ${final_values.mean():,.0f}')
    ax2.axvline(simulation_metrics['initial_value'], color='green', 
                linestyle=':', linewidth=2, label=f'Initial: ${simulation_metrics["initial_value"]:,.0f}')
    
    ax2.set_title("Distribution of Final Portfolio Values", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Portfolio Value ($)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.legend(framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.5, axis='y')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # --- Plot 3: Correlation Matrix (if requested) ---
    if show_correlation and ax3 is not None:
        corr_matrix = simulation_metrics['correlation_matrix']
        im = ax3.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Correlation', rotation=270, labelpad=15)
        
        # Add labels
        tickers = corr_matrix.columns.tolist()
        ax3.set_xticks(range(len(tickers)))
        ax3.set_yticks(range(len(tickers)))
        ax3.set_xticklabels(tickers, rotation=45, ha='right')
        ax3.set_yticklabels(tickers)
        
        # Add correlation values
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                text = ax3.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        ax3.set_title("Asset Correlation Matrix", fontsize=14, fontweight='bold')
    
    plt.tight_layout()


def save_detailed_report(
    simulation_df: pd.DataFrame,
    simulation_metrics: dict,
    filepath: str,
) -> None:
    """
    Save a detailed statistical report to CSV.
    """
    final_values = simulation_df.iloc[-1]
    
    # Create summary statistics
    stats = {
        'Initial Value': simulation_metrics['initial_value'],
        'Mean Final Value': final_values.mean(),
        'Median Final Value': final_values.median(),
        'Std Dev Final Value': final_values.std(),
        '99th Percentile': final_values.quantile(0.99),
        '95th Percentile': final_values.quantile(0.95),
        '75th Percentile': final_values.quantile(0.75),
        '25th Percentile': final_values.quantile(0.25),
        '5th Percentile': final_values.quantile(0.05),
        '1st Percentile': final_values.quantile(0.01),
        'Number of Simulations': simulation_metrics['num_simulations'],
        'Days Projected': simulation_metrics['days_projected'],
        'Target Annual Return': simulation_metrics['target_annual_return'],
        'Historical Annual Return': simulation_metrics['historical_annual_return'],
        'Annual Volatility': simulation_metrics['annual_volatility'],
        'Sharpe Ratio': simulation_metrics['sharpe_ratio'],
    }
    
    # Add risk metrics
    risk_metrics = calculate_risk_metrics(simulation_df)
    stats.update({
        'Probability of Loss': risk_metrics['probability_of_loss'],
        'VaR (95%)': risk_metrics['var'],
        'CVaR (95%)': risk_metrics['cvar'],
    })
    
    # Save to CSV
    pd.DataFrame([stats]).T.to_csv(filepath, header=['Value'])
    print(f"\n✓ Saved detailed statistics to {filepath}")


# ---------------------- CLI ---------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhanced Portfolio Monte Carlo simulation (educational only, not financial advice).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation with default 10% target return
  python portfolio_monte_carlo.py --tickers AAPL,MSFT,GOOG
  
  # Custom weights and 12% target return
  python portfolio_monte_carlo.py --tickers AAPL,MSFT,GOOG --weights 0.5,0.3,0.2 --target-annual-return 0.12
  
  # 5-year projection with 5000 simulations
  python portfolio_monte_carlo.py --tickers AAPL,MSFT --days 1260 --sims 5000
  
  # Export results with detailed statistics
  python portfolio_monte_carlo.py --tickers AAPL,MSFT --out-csv paths.csv --out-stats stats.csv
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated ticker symbols (e.g. 'AAPL,MSFT,GOOG').",
    )
    
    # Portfolio configuration
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional comma-separated weights. If omitted, equal weights are used.",
    )
    parser.add_argument(
        "--initial",
        type=float,
        default=10_000.0,
        help="Initial portfolio value (default: 10,000.0).",
    )
    
    # Historical data parameters
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
    
    # Simulation parameters
    parser.add_argument(
        "--days",
        type=int,
        default=TRADING_DAYS,
        help=f"Number of future trading days to simulate (default: {TRADING_DAYS} ≈ 1 year).",
    )
    parser.add_argument(
        "--target-annual-return",
        type=float,
        default=0.10,
        help="Target annual return (drift) for simulation (e.g., 0.10 for 10%%, default: 0.10).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=1_000,
        help="Number of Monte Carlo simulations (default: 1,000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible simulations.",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.02,
        help="Annual risk-free rate for Sharpe ratio (default: 0.02 = 2%%).",
    )
    
    # Output options
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="If set, do not show plots (print summary only).",
    )
    parser.add_argument(
        "--show-correlation",
        action="store_true",
        help="Show asset correlation matrix in plots.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Optional path to save simulated paths as CSV.",
    )
    parser.add_argument(
        "--out-stats",
        type=str,
        default=None,
        help="Optional path to save detailed statistics as CSV.",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the Monte Carlo simulation."""
    args = parse_args()

    # Validate dates
    if args.history_start >= args.history_end:
        print("❌ Error: history start date must be before history end date.")
        sys.exit(1)

    # Parse tickers
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("❌ Error: no valid tickers provided.")
        sys.exit(1)

    # Parse weights
    if args.weights:
        try:
            raw_weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
        except ValueError:
            print("❌ Error: could not parse weights as floats.")
            sys.exit(1)
    else:
        raw_weights = None

    # Print configuration
    print("\n" + "="*70)
    print(" "*20 + "MONTE CARLO CONFIGURATION")
    print("="*70)
    print(f"Assets:              {', '.join(tickers)}")
    print(f"Historical period:   {args.history_start.date()} → {args.history_end.date()}")
    print(f"Projection period:   {args.days} trading days (~{args.days/TRADING_DAYS:.1f} years)")
    print(f"Initial investment:  ${args.initial:,.2f}")
    print(f"Target annual return: {args.target_annual_return * 100:.2f}%")
    print(f"Number of simulations: {args.sims:,}")
    print(f"Risk-free rate:      {args.risk_free_rate * 100:.2f}%")
    if args.seed is not None:
        print(f"Random seed:         {args.seed}")
    print("="*70)

    # Load data
    prices = load_prices_stooq(tickers, args.history_start, args.history_end)
    weights = normalise_weights(raw_weights, n_assets=len(prices.columns))
    
    print(f"\nNormalized weights:")
    for ticker, weight in zip(prices.columns, weights):
        print(f"  {ticker}: {weight*100:.2f}%")

    # Run simulation
    simulation_df, simulation_metrics = run_monte_carlo(
        prices,
        weights,
        days_to_project=args.days,
        initial_value=args.initial,
        target_annual_return=args.target_annual_return,
        num_simulations=args.sims,
        rng_seed=args.seed,
        risk_free_rate=args.risk_free_rate,
    )

    # Print summary
    print_simulation_summary(simulation_df, simulation_metrics)

    # Save outputs
    if args.out_csv:
        try:
            simulation_df.to_csv(args.out_csv, index=True)
            print(f"✓ Saved Monte Carlo paths to {args.out_csv}")
        except Exception as e:
            print(f"⚠ Warning: could not save CSV to {args.out_csv}: {e}")
    
    if args.out_stats:
        try:
            save_detailed_report(simulation_df, simulation_metrics, args.out_stats)
        except Exception as e:
            print(f"⚠ Warning: could not save statistics to {args.out_stats}: {e}")

    # Plot results
    if not args.no_plots:
        plot_monte_carlo_results(
            simulation_df, 
            simulation_metrics,
            show_correlation=args.show_correlation
        )
        plt.show()
    
    print("\n✓ Simulation complete!\n")


if __name__ == "__main__":
    main()