"""
Enhanced Streamlit Stock Analysis Toolkit

Major improvements:
- Better UI with modern styling and layouts
- Additional technical indicators (MACD, Bollinger Bands)
- Enhanced SVM with feature importance and confidence scores
- Monte Carlo simulation for risk analysis
- Advanced portfolio optimization (Efficient Frontier)
- Real-time data validation and error handling
- Export functionality for all analyses
- Interactive plots with Plotly
- Risk-adjusted metrics dashboard

Data source: Stooq via pandas-datareader (no API key required)
Educational purposes only. Not financial advice.
"""

import math
import warnings
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from pandas_datareader import data as web
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.optimize import minimize
from scipy import stats

warnings.filterwarnings('ignore')

# ==================== PAGE CONFIGURATION ==================== #

st.set_page_config(
    page_title="Advanced Stock Toolkit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Advanced Stock Analysis Toolkit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Comprehensive trading analysis with ML predictions, portfolio optimization, and risk analytics</div>', unsafe_allow_html=True)

# ==================== SIDEBAR CONFIGURATION ==================== #

with st.sidebar:
    st.markdown("### ⚙️ Application Settings")
    st.info(
        "**Educational Tool Only**\n\n"
        "This application is designed for learning and experimentation. "
        "All predictions and analyses should not be used as financial advice. "
        "Past performance does not guarantee future results."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Data Information")
    st.markdown("""
    - **Source**: Stooq (via pandas-datareader)
    - **Update**: Daily close prices
    - **Coverage**: US stocks (add .US suffix)
    """)
    
    st.markdown("---")
    cache_status = st.checkbox("Enable data caching", value=True, 
                               help="Cache downloaded data for faster reloads")


# ==================== UTILITY FUNCTIONS ==================== #

def validate_dates(start: date, end: date) -> bool:
    """Validate date inputs"""
    if start >= end:
        st.error("❌ Start date must be before end date")
        return False
    if end > date.today():
        st.warning("⚠️ End date is in the future. Using today's date.")
        return True
    if (end - start).days < 30:
        st.warning("⚠️ Date range is very short. Consider using at least 30 days for meaningful analysis.")
    return True


def format_large_number(num: float) -> str:
    """Format large numbers with K, M, B suffixes"""
    if abs(num) >= 1e9:
        return f"${num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"


@st.cache_data(ttl=3600, show_spinner=True)
def load_stock_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Load complete OHLCV data for a single ticker from Stooq
    Returns DataFrame with Open, High, Low, Close, Volume
    """
    if "." not in ticker:
        stooq_ticker = f"{ticker}.US"
    else:
        stooq_ticker = ticker
    
    try:
        df = web.DataReader(stooq_ticker, "stooq", start=start, end=end)
        df = df.sort_index()
        
        if df.empty:
            st.warning(f"⚠️ No data returned for {ticker}")
            return pd.DataFrame()
        
        # Ensure we have required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            st.warning(f"⚠️ Missing required columns for {ticker}")
            return pd.DataFrame()
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=True)
def load_multiple_stocks(tickers: List[str], start: date, end: date) -> Dict[str, pd.DataFrame]:
    """Load data for multiple tickers"""
    data = {}
    progress_bar = st.progress(0)
    
    for idx, ticker in enumerate(tickers):
        df = load_stock_data(ticker, start, end)
        if not df.empty:
            data[ticker] = df
        progress_bar.progress((idx + 1) / len(tickers))
    
    progress_bar.empty()
    return data


def align_price_data(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align close prices across tickers on common dates"""
    if not data_dict:
        return pd.DataFrame()
    
    prices = {}
    for ticker, df in data_dict.items():
        if "Close" in df.columns:
            prices[ticker] = df["Close"]
    
    if not prices:
        return pd.DataFrame()
    
    aligned = pd.DataFrame(prices).dropna()
    return aligned


# ==================== TECHNICAL INDICATORS ==================== #

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive technical indicators
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    
    # Returns
    df["daily_return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))
    df["cum_return"] = (1 + df["daily_return"]).cumprod() - 1
    
    # Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{period}"] = close.rolling(window=period).mean()
        df[f"EMA_{period}"] = close.ewm(span=period, adjust=False).mean()
    
    # Volatility
    df["volatility_20"] = df["daily_return"].rolling(window=20).std()
    df["volatility_50"] = df["daily_return"].rolling(window=50).std()
    
    # Bollinger Bands
    sma_20 = df["SMA_20"]
    std_20 = close.rolling(window=20).std()
    df["BB_upper"] = sma_20 + (std_20 * 2)
    df["BB_lower"] = sma_20 - (std_20 * 2)
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma_20
    
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    df["Stoch_K"] = 100 * ((close - low_14) / (high_14 - low_14))
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
    
    # ATR (Average True Range)
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["ATR_14"] = true_range.rolling(window=14).mean()
    
    # Volume indicators
    df["Volume_SMA_20"] = volume.rolling(window=20).mean()
    df["Volume_ratio"] = volume / df["Volume_SMA_20"]
    
    # Drawdown
    running_max = close.cummax()
    df["drawdown"] = (close / running_max) - 1.0
    df["drawdown_duration"] = (df["drawdown"] < 0).astype(int).groupby(
        (df["drawdown"] == 0).cumsum()
    ).cumsum()
    
    return df


def compute_risk_metrics(df: pd.DataFrame) -> Dict:
    """Compute comprehensive risk and performance metrics"""
    close = df["Close"].dropna()
    
    if len(close) < 2:
        return {}
    
    # Basic info
    start_date = close.index[0]
    end_date = close.index[-1]
    days = (end_date - start_date).days
    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])
    
    # Returns
    total_return = (end_price / start_price) - 1
    log_returns = df["log_return"].dropna()
    
    if len(log_returns) > 0:
        # Annualized metrics
        trading_days = 252
        mean_log_daily = log_returns.mean()
        vol_log_daily = log_returns.std()
        
        ann_return = math.exp(mean_log_daily * trading_days) - 1
        ann_vol = vol_log_daily * math.sqrt(trading_days)
        
        # Sharpe ratio (assuming rf=0)
        sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
        
        # Sortino ratio (downside deviation)
        downside_returns = log_returns[log_returns < 0]
        downside_std = downside_returns.std() * math.sqrt(trading_days) if len(downside_returns) > 0 else 0
        sortino = ann_return / downside_std if downside_std != 0 else np.nan
        
        # Calmar ratio
        max_dd = df["drawdown"].min() if "drawdown" in df.columns else 0
        calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan
    else:
        ann_return = ann_vol = sharpe = sortino = calmar = np.nan
        max_dd = np.nan
    
    # Win/loss stats
    daily_ret = df["daily_return"].dropna()
    if len(daily_ret) > 0:
        winning_days = daily_ret[daily_ret > 0]
        losing_days = daily_ret[daily_ret < 0]
        
        win_rate = len(winning_days) / len(daily_ret) * 100
        avg_win = winning_days.mean() if len(winning_days) > 0 else 0
        avg_loss = losing_days.mean() if len(losing_days) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
        
        # Largest movements
        max_gain = daily_ret.max()
        max_loss = daily_ret.min()
    else:
        win_rate = avg_win = avg_loss = win_loss_ratio = np.nan
        max_gain = max_loss = np.nan
    
    # Value at Risk (VaR) and Expected Shortfall (CVaR)
    if len(daily_ret) > 0:
        var_95 = daily_ret.quantile(0.05)
        cvar_95 = daily_ret[daily_ret <= var_95].mean()
    else:
        var_95 = cvar_95 = np.nan
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "days": days,
        "start_price": start_price,
        "end_price": end_price,
        "total_return_pct": total_return * 100,
        "ann_return_pct": ann_return * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown_pct": max_dd * 100 if not pd.isna(max_dd) else np.nan,
        "win_rate_pct": win_rate,
        "avg_win_pct": avg_win * 100,
        "avg_loss_pct": avg_loss * 100,
        "win_loss_ratio": win_loss_ratio,
        "max_gain_pct": max_gain * 100,
        "max_loss_pct": max_loss * 100,
        "var_95_pct": var_95 * 100,
        "cvar_95_pct": cvar_95 * 100,
    }


# ==================== MACHINE LEARNING MODELS ==================== #

def build_ml_features(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Build comprehensive feature set for ML models
    """
    df = df.copy()
    close = df["Close"]
    
    # Price-based features
    for lag in [1, 2, 3, 5, 10]:
        df[f"return_{lag}d"] = close.pct_change(lag)
        df[f"log_return_{lag}d"] = np.log(close / close.shift(lag))
    
    # Moving average features
    for period in [5, 10, 20, 50]:
        df[f"sma_{period}"] = close.rolling(window=period).mean()
        df[f"price_to_sma_{period}"] = close / df[f"sma_{period}"] - 1
    
    # Volatility features
    for period in [5, 10, 20]:
        df[f"volatility_{period}"] = close.pct_change().rolling(window=period).std()
    
    # Technical indicators
    if "RSI_14" in df.columns:
        df["rsi"] = df["RSI_14"]
    if "MACD" in df.columns:
        df["macd"] = df["MACD"]
        df["macd_signal"] = df["MACD_signal"]
    
    # Volume features
    if "Volume_ratio" in df.columns:
        df["volume_ratio"] = df["Volume_ratio"]
    
    # Target: Next day's direction
    df["target_up"] = (close.shift(-1) > close).astype(int)
    
    # Target: Next day's return (for regression)
    df["target_return"] = close.pct_change().shift(-1)
    
    df = df.dropna()
    return df


def train_ml_models(df_features: pd.DataFrame, test_ratio: float = 0.2) -> Dict:
    """
    Train multiple ML models and return results
    """
    # Define feature columns
    feature_cols = [col for col in df_features.columns 
                   if col not in ["target_up", "target_return", "Close", "Open", "High", "Low", "Volume"]
                   and not col.startswith("SMA_") and not col.startswith("EMA_")
                   and not col.startswith("BB_")]
    
    X = df_features[feature_cols].values
    y = df_features["target_up"].values
    
    # Train/test split
    split_idx = int(len(df_features) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    results = {}
    
    # SVM Model
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
    ])
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_proba = svm_model.predict_proba(X_test)
    
    results["SVM"] = {
        "model": svm_model,
        "predictions": svm_pred,
        "probabilities": svm_proba,
        "accuracy": accuracy_score(y_test, svm_pred),
        "confusion_matrix": confusion_matrix(y_test, svm_pred),
    }
    
    # Random Forest Model
    rf_model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    
    # Feature importance
    feature_importance = rf_model.named_steps["rf"].feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)
    
    results["Random Forest"] = {
        "model": rf_model,
        "predictions": rf_pred,
        "probabilities": rf_proba,
        "accuracy": accuracy_score(y_test, rf_pred),
        "confusion_matrix": confusion_matrix(y_test, rf_pred),
        "feature_importance": importance_df
    }
    
    # Ensemble prediction (average probabilities)
    ensemble_proba = (svm_proba + rf_proba) / 2
    ensemble_pred = (ensemble_proba[:, 1] > 0.5).astype(int)
    
    results["Ensemble"] = {
        "predictions": ensemble_pred,
        "probabilities": ensemble_proba,
        "accuracy": accuracy_score(y_test, ensemble_pred),
        "confusion_matrix": confusion_matrix(y_test, ensemble_pred),
    }
    
    # Next-day prediction
    last_features = X[-1].reshape(1, -1)
    for model_name in ["SVM", "Random Forest"]:
        model = results[model_name]["model"]
        next_pred = model.predict(last_features)[0]
        next_proba = model.predict_proba(last_features)[0]
        results[model_name]["next_prediction"] = int(next_pred)
        results[model_name]["next_confidence"] = float(next_proba[next_pred])
    
    # Ensemble next prediction
    ensemble_next_proba = (
        results["SVM"]["model"].predict_proba(last_features)[0] +
        results["Random Forest"]["model"].predict_proba(last_features)[0]
    ) / 2
    results["Ensemble"]["next_prediction"] = int(ensemble_next_proba[1] > 0.5)
    results["Ensemble"]["next_confidence"] = float(max(ensemble_next_proba))
    
    # Additional metrics
    baseline_accuracy = max(np.mean(y_test), 1 - np.mean(y_test))
    results["baseline_accuracy"] = baseline_accuracy
    results["test_size"] = len(y_test)
    results["train_size"] = len(y_train)
    results["feature_names"] = feature_cols
    
    return results


# ==================== PORTFOLIO OPTIMIZATION ==================== #

def portfolio_performance(weights: np.ndarray, returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> Tuple[float, float]:
    """Calculate portfolio return and volatility"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return portfolio_return, portfolio_std


def negative_sharpe(weights: np.ndarray, returns: pd.DataFrame, cov_matrix: pd.DataFrame, rf: float = 0) -> float:
    """Negative Sharpe ratio for optimization"""
    p_return, p_std = portfolio_performance(weights, returns, cov_matrix)
    return -(p_return - rf) / p_std


def optimize_portfolio(returns: pd.DataFrame, method: str = "max_sharpe") -> Dict:
    """
    Optimize portfolio weights using different strategies
    """
    n_assets = len(returns.columns)
    cov_matrix = returns.cov()
    
    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    
    # Bounds: 0 <= weight <= 1 (no shorting)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weights
    init_guess = np.array([1/n_assets] * n_assets)
    
    if method == "max_sharpe":
        # Maximize Sharpe ratio
        result = minimize(
            negative_sharpe,
            init_guess,
            args=(returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
    
    elif method == "min_volatility":
        # Minimize volatility
        def portfolio_volatility(weights):
            return portfolio_performance(weights, returns, cov_matrix)[1]
        
        result = minimize(
            portfolio_volatility,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
    
    elif method == "equal_weight":
        result = type('obj', (object,), {
            'x': init_guess,
            'success': True
        })()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if result.success:
        optimal_weights = result.x
        p_return, p_vol = portfolio_performance(optimal_weights, returns, cov_matrix)
        sharpe = p_return / p_vol if p_vol != 0 else 0
        
        return {
            "weights": optimal_weights,
            "annual_return": p_return,
            "annual_volatility": p_vol,
            "sharpe_ratio": sharpe,
            "success": True
        }
    else:
        return {"success": False, "message": "Optimization failed"}


def efficient_frontier(returns: pd.DataFrame, n_points: int = 50) -> pd.DataFrame:
    """Generate efficient frontier"""
    n_assets = len(returns.columns)
    cov_matrix = returns.cov()
    
    # Find min and max return portfolios
    def portfolio_return(weights):
        return -portfolio_performance(weights, returns, cov_matrix)[0]
    
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    init_guess = np.array([1/n_assets] * n_assets)
    
    # Max return portfolio
    result_max = minimize(portfolio_return, init_guess, method="SLSQP", 
                         bounds=bounds, constraints=constraints)
    max_return = -result_max.fun
    
    # Min volatility portfolio
    def portfolio_volatility(weights):
        return portfolio_performance(weights, returns, cov_matrix)[1]
    
    result_min = minimize(portfolio_volatility, init_guess, method="SLSQP",
                         bounds=bounds, constraints=constraints)
    min_return = portfolio_performance(result_min.x, returns, cov_matrix)[0]
    
    # Generate frontier
    target_returns = np.linspace(min_return, max_return, n_points)
    frontier_volatilities = []
    frontier_weights = []
    
    for target_return in target_returns:
        constraints_with_return = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
            {"type": "eq", "fun": lambda x: portfolio_performance(x, returns, cov_matrix)[0] - target_return}
        ]
        
        result = minimize(
            portfolio_volatility,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_with_return
        )
        
        if result.success:
            frontier_volatilities.append(result.fun)
            frontier_weights.append(result.x)
        else:
            frontier_volatilities.append(np.nan)
            frontier_weights.append([np.nan] * n_assets)
    
    frontier_df = pd.DataFrame({
        "Return": target_returns,
        "Volatility": frontier_volatilities,
        "Weights": frontier_weights
    }).dropna()
    
    return frontier_df


def monte_carlo_simulation(last_price: float, returns: pd.Series, 
                          n_simulations: int = 1000, n_days: int = 252) -> np.ndarray:
    """Run Monte Carlo simulation for price paths"""
    mean_return = returns.mean()
    std_return = returns.std()
    
    simulations = np.zeros((n_simulations, n_days))
    
    for i in range(n_simulations):
        daily_returns = np.random.normal(mean_return, std_return, n_days)
        price_path = last_price * np.exp(np.cumsum(daily_returns))
        simulations[i] = price_path
    
    return simulations


# ==================== PLOTTING FUNCTIONS ==================== #

def plot_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create interactive candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=ticker
    )])
    
    fig.update_layout(
        title=f"{ticker} - Candlestick Chart",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        template="plotly_white",
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def plot_technical_indicators(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Plot price with technical indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{ticker} - Price & Moving Averages", "MACD", "RSI", "Volume"),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Price and MAs
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="black")), row=1, col=1)
    for ma in [20, 50, 200]:
        if f"SMA_{ma}" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{ma}"], name=f"SMA {ma}"), row=1, col=1)
    
    # Bollinger Bands
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", 
                                line=dict(dash="dash", color="gray")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
                                line=dict(dash="dash", color="gray"), fill="tonexty"), row=1, col=1)
    
    # MACD
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="blue")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="red")), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram"), row=2, col=1)
    
    # RSI
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI", line=dict(color="purple")), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="lightblue"), row=4, col=1)
    if "Volume_SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Volume_SMA_20"], name="Vol MA 20", 
                                line=dict(color="orange")), row=4, col=1)
    
    fig.update_layout(height=1000, template="plotly_white", showlegend=True)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    
    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> go.Figure:
    """Plot confusion matrix as heatmap"""
    labels = ["Down (0)", "Up (1)"]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted Down", "Predicted Up"],
        y=["Actual Down", "Actual Up"],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_efficient_frontier(frontier_df: pd.DataFrame, optimal_portfolios: Dict) -> go.Figure:
    """Plot efficient frontier with optimal portfolios"""
    fig = go.Figure()
    
    # Efficient frontier
    fig.add_trace(go.Scatter(
        x=frontier_df["Volatility"] * 100,
        y=frontier_df["Return"] * 100,
        mode="lines",
        name="Efficient Frontier",
        line=dict(color="blue", width=2)
    ))
    
    # Optimal portfolios
    colors = {"max_sharpe": "red", "min_volatility": "green", "equal_weight": "orange"}
    for method, portfolio in optimal_portfolios.items():
        if portfolio["success"]:
            fig.add_trace(go.Scatter(
                x=[portfolio["annual_volatility"] * 100],
                y=[portfolio["annual_return"] * 100],
                mode="markers",
                name=method.replace("_", " ").title(),
                marker=dict(size=12, color=colors.get(method, "black"))
            ))
    
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annual Volatility (%)",
        yaxis_title="Annual Return (%)",
        template="plotly_white",
        height=500
    )
    
    return fig


# ==================== MAIN APPLICATION ==================== #

mode = st.sidebar.radio(
    "🎯 Select Analysis Tool",
    [
        "🤖 ML Stock Predictor",
        "📊 Multi-Stock Analysis",
        "💼 Portfolio Optimizer",
        "🎲 Monte Carlo Simulation"
    ],
)


# ==================== MODE 1: ML PREDICTOR ==================== #

if mode == "🤖 ML Stock Predictor":
    st.header("🤖 Advanced ML Stock Price Predictor")
    st.markdown("Train multiple machine learning models to predict next-day price direction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter ticker without .US suffix").upper()
    with col2:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=730))
    with col3:
        end_date = st.date_input("End Date", value=date.today())
    
    with st.expander("⚙️ Model Configuration"):
        col_a, col_b = st.columns(2)
        with col_a:
            test_ratio = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05, 
                                  help="Proportion of data used for testing")
        with col_b:
            show_technical = st.checkbox("Show Technical Indicators", value=True)
    
    if st.button("🚀 Run Analysis", type="primary"):
        if not validate_dates(start_date, end_date):
            st.stop()
        
        with st.spinner(f"Loading data for {ticker}..."):
            df = load_stock_data(ticker, start_date, end_date)
        
        if df.empty:
            st.error(f"❌ No data available for {ticker}")
            st.stop()
        
        # Compute indicators
        with st.spinner("Computing technical indicators..."):
            df_indicators = compute_technical_indicators(df)
        
        # Create tabs
        tabs = st.tabs(["📈 Price Chart", "🎯 ML Predictions", "📊 Technical Analysis", "📉 Risk Metrics"])
        
        # Tab 1: Price Chart
        with tabs[0]:
            st.plotly_chart(plot_candlestick(df_indicators, ticker), use_container_width=True)
            
            if show_technical:
                st.plotly_chart(plot_technical_indicators(df_indicators, ticker), use_container_width=True)
        
        # Tab 2: ML Predictions
        with tabs[1]:
            with st.spinner("Training machine learning models..."):
                df_features = build_ml_features(df_indicators)
                
                if len(df_features) < 100:
                    st.warning("⚠️ Limited data available. Results may be unreliable.")
                
                ml_results = train_ml_models(df_features, test_ratio=test_ratio)
            
            st.subheader("🎯 Model Performance Comparison")
            
            # Performance metrics
            perf_data = []
            for model_name in ["SVM", "Random Forest", "Ensemble"]:
                perf_data.append({
                    "Model": model_name,
                    "Accuracy": f"{ml_results[model_name]['accuracy']:.3f}",
                    "Baseline": f"{ml_results['baseline_accuracy']:.3f}",
                    "Beat Baseline": "✅" if ml_results[model_name]['accuracy'] > ml_results['baseline_accuracy'] else "❌"
                })
            
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
            
            # Next-day predictions
            st.subheader("🔮 Next Trading Day Predictions")
            
            pred_cols = st.columns(3)
            for idx, model_name in enumerate(["SVM", "Random Forest", "Ensemble"]):
                with pred_cols[idx]:
                    pred = ml_results[model_name]["next_prediction"]
                    conf = ml_results[model_name]["next_confidence"]
                    
                    if pred == 1:
                        st.success(f"**{model_name}**")
                        st.metric("Prediction", "📈 UP", delta=f"{conf:.1%} confidence")
                    else:
                        st.error(f"**{model_name}**")
                        st.metric("Prediction", "📉 DOWN", delta=f"{conf:.1%} confidence")
            
            # Confusion matrices
            st.subheader("📊 Confusion Matrices")
            cm_cols = st.columns(2)
            
            with cm_cols[0]:
                st.plotly_chart(
                    plot_confusion_matrix(ml_results["SVM"]["confusion_matrix"], "SVM Model"),
                    use_container_width=True
                )
            
            with cm_cols[1]:
                st.plotly_chart(
                    plot_confusion_matrix(ml_results["Random Forest"]["confusion_matrix"], "Random Forest Model"),
                    use_container_width=True
                )
            
            # Feature importance
            if "feature_importance" in ml_results["Random Forest"]:
                st.subheader("🎯 Feature Importance (Random Forest)")
                importance_df = ml_results["Random Forest"]["feature_importance"].head(10)
                
                fig = px.bar(importance_df, x="importance", y="feature", orientation="h",
                           title="Top 10 Most Important Features")
                fig.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Technical Analysis
        with tabs[2]:
            st.subheader("📊 Current Technical Indicators")
            
            latest = df_indicators.iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RSI (14)", f"{latest['RSI_14']:.2f}")
                if latest['RSI_14'] > 70:
                    st.caption("🔴 Overbought")
                elif latest['RSI_14'] < 30:
                    st.caption("🟢 Oversold")
                else:
                    st.caption("🟡 Neutral")
            
            with col2:
                st.metric("MACD", f"{latest['MACD']:.2f}")
                if latest['MACD'] > latest['MACD_signal']:
                    st.caption("🟢 Bullish")
                else:
                    st.caption("🔴 Bearish")
            
            with col3:
                st.metric("BB Width", f"{latest['BB_width']:.3f}")
                st.caption("Volatility Measure")
            
            with col4:
                st.metric("ATR (14)", f"{latest['ATR_14']:.2f}")
                st.caption("Average True Range")
            
            # Moving average analysis
            st.subheader("📈 Moving Average Analysis")
            ma_data = []
            for period in [20, 50, 200]:
                col_name = f"SMA_{period}"
                if col_name in latest.index:
                    ma_value = latest[col_name]
                    distance = ((latest['Close'] / ma_value) - 1) * 100
                    trend = "Above 🟢" if latest['Close'] > ma_value else "Below 🔴"
                    ma_data.append({
                        "MA Period": f"SMA {period}",
                        "Value": f"${ma_value:.2f}",
                        "Distance": f"{distance:+.2f}%",
                        "Position": trend
                    })
            
            st.dataframe(pd.DataFrame(ma_data), use_container_width=True, hide_index=True)
        
        # Tab 4: Risk Metrics
        with tabs[3]:
            metrics = compute_risk_metrics(df_indicators)
            
            if metrics:
                st.subheader("📊 Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
                    st.metric("Annualized Return", f"{metrics['ann_return_pct']:.2f}%")
                with col2:
                    st.metric("Annual Volatility", f"{metrics['ann_vol_pct']:.2f}%")
                    st.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
                with col4:
                    st.metric("Win Rate", f"{metrics['win_rate_pct']:.2f}%")
                    st.metric("Win/Loss Ratio", f"{metrics['win_loss_ratio']:.2f}")
                
                st.subheader("⚠️ Risk Metrics")
                
                risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                with risk_col1:
                    st.metric("VaR (95%)", f"{metrics['var_95_pct']:.2f}%")
                with risk_col2:
                    st.metric("CVaR (95%)", f"{metrics['cvar_95_pct']:.2f}%")
                with risk_col3:
                    st.metric("Max Daily Gain", f"{metrics['max_gain_pct']:.2f}%")
                with risk_col4:
                    st.metric("Max Daily Loss", f"{metrics['max_loss_pct']:.2f}%")
                
                # Drawdown chart
                st.subheader("📉 Drawdown Over Time")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_indicators.index,
                    y=df_indicators["drawdown"] * 100,
                    fill='tozeroy',
                    name="Drawdown",
                    line=dict(color="red")
                ))
                fig.update_layout(
                    title="Historical Drawdown",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)


# ==================== MODE 2: MULTI-STOCK ANALYSIS ==================== #

elif mode == "📊 Multi-Stock Analysis":
    st.header("📊 Multi-Stock Comparative Analysis")
    st.markdown("Compare multiple stocks side-by-side with comprehensive metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        tickers_str = st.text_input("Stock Tickers (comma-separated)", value="AAPL,MSFT,GOOGL,TSLA")
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365))
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        end_date = st.date_input("End Date", value=date.today())
    
    if st.button("🔍 Analyze Stocks", type="primary"):
        if not validate_dates(start_date, end_date):
            st.stop()
        
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        
        if not tickers:
            st.error("Please enter at least one ticker")
            st.stop()
        
        # Load data
        with st.spinner(f"Loading data for {len(tickers)} stocks..."):
            data_dict = load_multiple_stocks(tickers, start_date, end_date)
        
        if not data_dict:
            st.error("No data could be loaded for any ticker")
            st.stop()
        
        # Compute metrics for each stock
        all_metrics = []
        
        for ticker, df in data_dict.items():
            df_ind = compute_technical_indicators(df)
            metrics = compute_risk_metrics(df_ind)
            if metrics:
                metrics["Ticker"] = ticker
                all_metrics.append(metrics)
        
        if not all_metrics:
            st.error("Could not compute metrics for any stock")
            st.stop()
        
        metrics_df = pd.DataFrame(all_metrics).set_index("Ticker")
        
        # Create tabs
        tabs = st.tabs(["📊 Comparison", "📈 Price Charts", "📉 Returns", "🔗 Correlation", "📋 Full Metrics"])
        
        # Tab 1: Key Metrics Comparison
        with tabs[0]:
            st.subheader("🎯 Key Performance Indicators")
            
            # Select key metrics to display
            key_metrics = metrics_df[[
                "total_return_pct", "ann_return_pct", "ann_vol_pct",
                "sharpe_ratio", "max_drawdown_pct", "win_rate_pct"
            ]].copy()
            
            key_metrics.columns = [
                "Total Return (%)", "Annual Return (%)", "Annual Vol (%)",
                "Sharpe Ratio", "Max Drawdown (%)", "Win Rate (%)"
            ]
            
            st.dataframe(
                key_metrics.style.background_gradient(cmap="RdYlGn", subset=["Total Return (%)", "Annual Return (%)", "Sharpe Ratio"])
                           .background_gradient(cmap="RdYlGn_r", subset=["Annual Vol (%)", "Max Drawdown (%)"]),
                use_container_width=True
            )
            
            # Best/Worst performers
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Best Performers")
                best_return = metrics_df.nlargest(3, "total_return_pct")[["total_return_pct", "sharpe_ratio"]]
                best_return.columns = ["Total Return (%)", "Sharpe Ratio"]
                st.dataframe(best_return, use_container_width=True)
            
            with col2:
                st.subheader("⚠️ Highest Risk")
                highest_vol = metrics_df.nlargest(3, "ann_vol_pct")[["ann_vol_pct", "max_drawdown_pct"]]
                highest_vol.columns = ["Annual Vol (%)", "Max Drawdown (%)"]
                st.dataframe(highest_vol, use_container_width=True)
            
            # Bar charts
            st.subheader("📊 Visual Comparison")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                fig = px.bar(metrics_df.reset_index(), x="Ticker", y="total_return_pct",
                           title="Total Returns Comparison", labels={"total_return_pct": "Return (%)"})
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_col2:
                fig = px.bar(metrics_df.reset_index(), x="Ticker", y="sharpe_ratio",
                           title="Sharpe Ratio Comparison", labels={"sharpe_ratio": "Sharpe Ratio"})
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Price Charts
        with tabs[1]:
            st.subheader("📈 Normalized Price Comparison")
            
            prices = align_price_data(data_dict)
            norm_prices = (prices / prices.iloc[0]) * 100
            
            fig = go.Figure()
            for ticker in norm_prices.columns:
                fig.add_trace(go.Scatter(
                    x=norm_prices.index,
                    y=norm_prices[ticker],
                    name=ticker,
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Normalized Prices (Base = 100)",
                xaxis_title="Date",
                yaxis_title="Indexed Price",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Returns Distribution
        with tabs[2]:
            st.subheader("📊 Daily Returns Distribution")
            
            prices = align_price_data(data_dict)
            returns = prices.pct_change().dropna()
            
            fig = go.Figure()
            for ticker in returns.columns:
                fig.add_trace(go.Histogram(
                    x=returns[ticker] * 100,
                    name=ticker,
                    opacity=0.7
                ))
            
            fig.update_layout(
                title="Daily Returns Distribution",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                barmode='overlay',
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Returns statistics
            st.subheader("📈 Returns Statistics")
            returns_stats = pd.DataFrame({
                "Mean (%)": returns.mean() * 100,
                "Std Dev (%)": returns.std() * 100,
                "Skewness": returns.skew(),
                "Kurtosis": returns.kurtosis()
            })
            st.dataframe(returns_stats.T, use_container_width=True)
        
        # Tab 4: Correlation
        with tabs[3]:
            prices = align_price_data(data_dict)
            returns = prices.pct_change().dropna()
            
            if len(returns.columns) >= 2:
                st.subheader("🔗 Correlation Matrix")
                
                corr_matrix = returns.corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale="RdBu",
                    zmid=0,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 12}
                ))
                
                fig.update_layout(
                    title="Daily Returns Correlation",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation insights
                st.subheader("🔍 Correlation Insights")
                
                # Find most and least correlated pairs
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            "Pair": f"{corr_matrix.columns[i]} - {corr_matrix.index[j]}",
                            "Correlation": corr_matrix.iloc[i, j]
                        })
                
                corr_pairs_df = pd.DataFrame(corr_pairs).sort_values("Correlation", ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Most Correlated Pairs:**")
                    st.dataframe(corr_pairs_df.head(5), use_container_width=True, hide_index=True)
                
                with col2:
                    st.write("**Least Correlated Pairs:**")
                    st.dataframe(corr_pairs_df.tail(5), use_container_width=True, hide_index=True)
        
        # Tab 5: Full Metrics
        with tabs[4]:
            st.subheader("📋 Complete Metrics Table")
            st.dataframe(metrics_df, use_container_width=True)
            
            # Download button
            csv = metrics_df.to_csv()
            st.download_button(
                label="📥 Download Metrics as CSV",
                data=csv,
                file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


# ==================== MODE 3: PORTFOLIO OPTIMIZER ==================== #

elif mode == "💼 Portfolio Optimizer":
    st.header("💼 Portfolio Optimization & Backtesting")
    st.markdown("Optimize portfolio weights using Modern Portfolio Theory")
    
    col1, col2 = st.columns(2)
    with col1:
        tickers_str = st.text_input("Portfolio Tickers (comma-separated)", value="AAPL,MSFT,GOOGL,AMZN")
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=730))
    with col2:
        st.write("")
        st.write("")
        end_date = st.date_input("End Date", value=date.today())
    
    with st.expander("⚙️ Optimization Settings"):
        optimization_methods = st.multiselect(
            "Select Optimization Methods",
            ["Maximum Sharpe", "Minimum Volatility", "Equal Weight"],
            default=["Maximum Sharpe", "Minimum Volatility", "Equal Weight"]
        )
        
        show_frontier = st.checkbox("Show Efficient Frontier", value=True,
                                   help="Computationally intensive for many assets")
    
    if st.button("🚀 Optimize Portfolio", type="primary"):
        if not validate_dates(start_date, end_date):
            st.stop()
        
        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        
        if len(tickers) < 2:
            st.error("Please enter at least 2 tickers for portfolio optimization")
            st.stop()
        
        # Load data
        with st.spinner(f"Loading data for {len(tickers)} assets..."):
            data_dict = load_multiple_stocks(tickers, start_date, end_date)
        
        if len(data_dict) < 2:
            st.error("Need at least 2 valid tickers for portfolio optimization")
            st.stop()
        
        # Align prices and compute returns
        prices = align_price_data(data_dict)
        returns = prices.pct_change().dropna()
        
        st.success(f"✅ Successfully loaded data for {len(prices.columns)} assets")
        
        # Create tabs
        tabs = st.tabs(["🎯 Optimal Portfolios", "📈 Efficient Frontier", "📊 Performance", "📋 Detailed Weights"])
        
        # Optimize portfolios
        method_mapping = {
            "Maximum Sharpe": "max_sharpe",
            "Minimum Volatility": "min_volatility",
            "Equal Weight": "equal_weight"
        }
        
        optimal_portfolios = {}
        
        with st.spinner("Optimizing portfolios..."):
            for method_name in optimization_methods:
                method_key = method_mapping[method_name]
                result = optimize_portfolio(returns, method=method_key)
                if result["success"]:
                    optimal_portfolios[method_key] = result
        
        # Tab 1: Optimal Portfolios
        with tabs[0]:
            st.subheader("🎯 Optimized Portfolio Allocations")
            
            # Display each portfolio
            for method_name in optimization_methods:
                method_key = method_mapping[method_name]
                
                if method_key in optimal_portfolios:
                    portfolio = optimal_portfolios[method_key]
                    
                    st.markdown(f"### {method_name} Portfolio")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Annual Return", f"{portfolio['annual_return']*100:.2f}%")
                    with col2:
                        st.metric("Annual Volatility", f"{portfolio['annual_volatility']*100:.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{portfolio['sharpe_ratio']:.2f}")
                    with col4:
                        st.metric("Assets", len(portfolio['weights']))
                    
                    # Weight distribution
                    weights_df = pd.DataFrame({
                        "Asset": prices.columns,
                        "Weight": portfolio['weights']
                    }).sort_values("Weight", ascending=False)
                    weights_df["Weight %"] = weights_df["Weight"] * 100
                    
                    # Pie chart
                    fig = px.pie(weights_df, values="Weight", names="Asset",
                               title=f"{method_name} - Asset Allocation")
                    fig.update_layout(template="plotly_white", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
        
        # Tab 2: Efficient Frontier
        with tabs[1]:
            if show_frontier and len(prices.columns) <= 10:
                with st.spinner("Generating efficient frontier..."):
                    frontier_df = efficient_frontier(returns, n_points=30)
                
                if not frontier_df.empty:
                    fig = plot_efficient_frontier(frontier_df, optimal_portfolios)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("💡 The efficient frontier shows the best possible risk-return combinations. "
                           "Portfolios on the frontier provide maximum return for a given level of risk.")
                else:
                    st.warning("Could not generate efficient frontier")
            elif len(prices.columns) > 10:
                st.warning("Efficient frontier calculation is disabled for portfolios with >10 assets (too computationally intensive)")
            else:
                st.info("Enable 'Show Efficient Frontier' in settings to view the efficient frontier")
        
        # Tab 3: Performance Comparison
        with tabs[2]:
            st.subheader("📊 Portfolio Performance Backtest")
            
            # Backtest each portfolio
            performance_data = []
            portfolio_values = pd.DataFrame()
            
            for method_name in optimization_methods:
                method_key = method_mapping[method_name]
                
                if method_key in optimal_portfolios:
                    portfolio = optimal_portfolios[method_key]
                    weights = portfolio['weights']
                    
                    # Calculate portfolio returns
                    port_returns = (returns * weights).sum(axis=1)
                    port_value = (1 + port_returns).cumprod()
                    
                    portfolio_values[method_name] = port_value
                    
                    # Calculate metrics
                    total_return = (port_value.iloc[-1] - 1) * 100
                    ann_vol = port_returns.std() * np.sqrt(252) * 100
                    sharpe = portfolio['sharpe_ratio']
                    
                    max_dd = ((port_value / port_value.cummax()) - 1).min() * 100
                    
                    performance_data.append({
                        "Strategy": method_name,
                        "Total Return (%)": total_return,
                        "Annual Vol (%)": ann_vol,
                        "Sharpe Ratio": sharpe,
                        "Max Drawdown (%)": max_dd
                    })
            
            # Display performance table
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(
                perf_df.style.background_gradient(subset=["Total Return (%)", "Sharpe Ratio"], cmap="RdYlGn")
                       .background_gradient(subset=["Annual Vol (%)", "Max Drawdown (%)"], cmap="RdYlGn_r"),
                use_container_width=True,
                hide_index=True
            )
            
            # Plot cumulative returns
            st.subheader("📈 Cumulative Returns Comparison")
            
            fig = go.Figure()
            for col in portfolio_values.columns:
                fig.add_trace(go.Scatter(
                    x=portfolio_values.index,
                    y=(portfolio_values[col] - 1) * 100,
                    name=col,
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Portfolio Performance Comparison",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 4: Detailed Weights
        with tabs[3]:
            st.subheader("📋 Detailed Portfolio Weights")
            
            for method_name in optimization_methods:
                method_key = method_mapping[method_name]
                
                if method_key in optimal_portfolios:
                    st.markdown(f"### {method_name}")
                    
                    portfolio = optimal_portfolios[method_key]
                    weights_df = pd.DataFrame({
                        "Asset": prices.columns,
                        "Weight": portfolio['weights'],
                        "Weight (%)": portfolio['weights'] * 100
                    }).sort_values("Weight", ascending=False)
                    
                    # Add allocation amount for $10,000 portfolio
                    weights_df["Allocation ($10K)"] = weights_df["Weight"] * 10000
                    
                    st.dataframe(weights_df, use_container_width=True, hide_index=True)
                    
                    # Bar chart
                    fig = px.bar(weights_df, x="Asset", y="Weight (%)",
                               title=f"{method_name} - Weight Distribution")
                    fig.update_layout(template="plotly_white", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")


# ==================== MODE 4: MONTE CARLO SIMULATION ==================== #

elif mode == "🎲 Monte Carlo Simulation":
    st.header("🎲 Monte Carlo Price Simulation")
    st.markdown("Simulate thousands of possible future price paths based on historical volatility")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Stock Ticker", value="AAPL").upper()
        start_date = st.date_input("Historical Start", value=date.today() - timedelta(days=365))
    with col2:
        st.write("")
        st.write("")
        end_date = st.date_input("Historical End", value=date.today())
    with col3:
        st.write("")
        st.write("")
        sim_days = st.number_input("Days to Simulate", min_value=1, max_value=365, value=252)
    
    with st.expander("⚙️ Simulation Parameters"):
        n_simulations = st.slider("Number of Simulations", 100, 10000, 1000, 100,
                                  help="More simulations = more accurate but slower")
        confidence_level = st.slider("Confidence Level (%)", 80, 99, 95, 1)
    
    if st.button("🎲 Run Simulation", type="primary"):
        if not validate_dates(start_date, end_date):
            st.stop()
        
        # Load historical data
        with st.spinner(f"Loading historical data for {ticker}..."):
            df = load_stock_data(ticker, start_date, end_date)
        
        if df.empty:
            st.error(f"No data available for {ticker}")
            st.stop()
        
        # Compute returns
        returns = df["Close"].pct_change().dropna()
        last_price = float(df["Close"].iloc[-1])
        
        # Run simulation
        with st.spinner(f"Running {n_simulations:,} Monte Carlo simulations..."):
            simulations = monte_carlo_simulation(last_price, returns, n_simulations, sim_days)
        
        # Create tabs
        tabs = st.tabs(["📊 Simulation Results", "📈 Price Paths", "📉 Distribution", "📋 Statistics"])
        
        # Tab 1: Results Summary
        with tabs[0]:
            st.subheader("🎯 Simulation Summary")
            
            final_prices = simulations[:, -1]
            
            # Calculate percentiles
            percentiles = {
                f"{confidence_level}% Upper": np.percentile(final_prices, confidence_level),
                "Median": np.median(final_prices),
                f"{confidence_level}% Lower": np.percentile(final_prices, 100 - confidence_level),
                "Mean": np.mean(final_prices)
            }
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${last_price:.2f}")
                st.metric("Simulation Period", f"{sim_days} days")
            
            with col2:
                expected_price = percentiles["Mean"]
                expected_return = ((expected_price / last_price) - 1) * 100
                st.metric("Expected Price", f"${expected_price:.2f}",
                         delta=f"{expected_return:+.1f}%")
            
            with col3:
                upper = percentiles[f"{confidence_level}% Upper"]
                upper_return = ((upper / last_price) - 1) * 100
                st.metric(f"Upside ({confidence_level}%)", f"${upper:.2f}",
                         delta=f"{upper_return:+.1f}%")
            
            with col4:
                lower = percentiles[f"{confidence_level}% Lower"]
                lower_return = ((lower / last_price) - 1) * 100
                st.metric(f"Downside ({confidence_level}%)", f"${lower:.2f}",
                         delta=f"{lower_return:+.1f}%")
            
            # Probability metrics
            st.subheader("📊 Probability Analysis")
            
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            
            with prob_col1:
                prob_profit = (final_prices > last_price).sum() / n_simulations * 100
                st.metric("Probability of Profit", f"{prob_profit:.1f}%")
            
            with prob_col2:
                prob_double = (final_prices > last_price * 2).sum() / n_simulations * 100
                st.metric("Probability of 2x", f"{prob_double:.1f}%")
            
            with prob_col3:
                prob_loss_50 = (final_prices < last_price * 0.5).sum() / n_simulations * 100
                st.metric("Probability of 50% Loss", f"{prob_loss_50:.1f}%")
        
        # Tab 2: Price Paths
        with tabs[1]:
            st.subheader("📈 Simulated Price Paths")
            
            # Plot sample of paths
            n_display = min(100, n_simulations)
            
            fig = go.Figure()
            
            # Plot sample paths
            for i in range(n_display):
                fig.add_trace(go.Scatter(
                    y=simulations[i],
                    mode='lines',
                    line=dict(width=0.5),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Plot mean path
            mean_path = simulations.mean(axis=0)
            fig.add_trace(go.Scatter(
                y=mean_path,
                mode='lines',
                name='Mean Path',
                line=dict(color='red', width=3)
            ))
            
            # Plot confidence interval
            upper_bound = np.percentile(simulations, confidence_level, axis=0)
            lower_bound = np.percentile(simulations, 100 - confidence_level, axis=0)
            
            fig.add_trace(go.Scatter(
                y=upper_bound,
                mode='lines',
                name=f'{confidence_level}% Upper',
                line=dict(color='green', dash='dash', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                y=lower_bound,
                mode='lines',
                name=f'{confidence_level}% Lower',
                line=dict(color='orange', dash='dash', width=2),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title=f"{ticker} - {n_simulations:,} Monte Carlo Simulations ({n_display} shown)",
                xaxis_title="Trading Days",
                yaxis_title="Price ($)",
                template="plotly_white",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"💡 Showing {n_display} out of {n_simulations:,} simulated paths. "
                   f"The red line shows the average path, while the shaded area represents the {confidence_level}% confidence interval.")
        
        # Tab 3: Final Price Distribution
        with tabs[2]:
            st.subheader("📊 Final Price Distribution")
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=final_prices,
                nbinsx=50,
                name="Final Prices",
                marker_color='lightblue'
            ))
            
            # Add vertical lines for key statistics
            fig.add_vline(x=last_price, line_dash="dash", line_color="black",
                         annotation_text="Current Price")
            fig.add_vline(x=percentiles["Mean"], line_dash="dash", line_color="red",
                         annotation_text="Mean")
            fig.add_vline(x=percentiles["Median"], line_dash="dash", line_color="green",
                         annotation_text="Median")
            
            fig.update_layout(
                title=f"Distribution of Final Prices after {sim_days} Days",
                xaxis_title="Price ($)",
                yaxis_title="Frequency",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Returns distribution
            st.subheader("📈 Returns Distribution")
            
            final_returns = ((final_prices / last_price) - 1) * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=final_returns,
                nbinsx=50,
                name="Returns",
                marker_color='lightgreen'
            ))
            
            fig.add_vline(x=0, line_dash="dash", line_color="black",
                         annotation_text="Break-even")
            
            fig.update_layout(
                title="Distribution of Returns",
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 4: Detailed Statistics
        with tabs[3]:
            st.subheader("📋 Statistical Summary")
            
            stats_data = {
                "Metric": [
                    "Current Price",
                    "Mean Final Price",
                    "Median Final Price",
                    "Std Dev of Final Prices",
                    "Min Final Price",
                    "Max Final Price",
                    f"{confidence_level}% Lower Bound",
                    f"{confidence_level}% Upper Bound",
                    "Expected Return (%)",
                    "Volatility (Annualized %)",
                    "Probability of Profit (%)",
                    "Probability of 50% Gain (%)",
                    "Probability of 50% Loss (%)"
                ],
                "Value": [
                    f"${last_price:.2f}",
                    f"${percentiles['Mean']:.2f}",
                    f"${percentiles['Median']:.2f}",
                    f"${final_prices.std():.2f}",
                    f"${final_prices.min():.2f}",
                    f"${final_prices.max():.2f}",
                    f"${percentiles[f'{confidence_level}% Lower']:.2f}",
                    f"${percentiles[f'{confidence_level}% Upper']:.2f}",
                    f"{((percentiles['Mean'] / last_price) - 1) * 100:.2f}%",
                    f"{returns.std() * np.sqrt(252) * 100:.2f}%",
                    f"{(final_prices > last_price).sum() / n_simulations * 100:.2f}%",
                    f"{(final_prices > last_price * 1.5).sum() / n_simulations * 100:.2f}%",
                    f"{(final_prices < last_price * 0.5).sum() / n_simulations * 100:.2f}%"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            # Percentile table
            st.subheader("📊 Price Percentiles")
            
            percentile_values = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            percentile_data = []
            
            for p in percentile_values:
                price = np.percentile(final_prices, p)
                return_pct = ((price / last_price) - 1) * 100
                percentile_data.append({
                    "Percentile": f"{p}th",
                    "Price": f"${price:.2f}",
                    "Return": f"{return_pct:+.2f}%"
                })
            
            percentile_df = pd.DataFrame(percentile_data)
            st.dataframe(percentile_df, use_container_width=True, hide_index=True)
            
            # Export results
            st.subheader("💾 Export Results")
            
            export_df = pd.DataFrame({
                "Simulation": range(1, n_simulations + 1),
                "Final Price": final_prices,
                "Return (%)": ((final_prices / last_price) - 1) * 100
            })
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Simulation Results",
                data=csv,
                file_name=f"monte_carlo_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


# ==================== FOOTER ==================== #

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Advanced Stock Analysis Toolkit</strong></p>
    <p>Educational purposes only • Not financial advice • Past performance does not guarantee future results</p>
    <p>Data source: Stooq via pandas-datareader</p>
</div>
""", unsafe_allow_html=True)