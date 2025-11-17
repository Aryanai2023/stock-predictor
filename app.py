"""
Streamlit front-end for stock tools:

Tabs:
1) Single-stock SVM up/down predictor
2) Multi-ticker analysis
3) Portfolio backtester

Data source: Stooq via pandas-datareader (no API key).
Educational only. Not financial advice.
"""

import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
from pandas_datareader import data as web
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# ---------------------- Global Page Config ---------------------- #

st.set_page_config(
    page_title="Stock Toolkit",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Prediction & Analysis Toolkit")
st.caption("Educational only. Not financial advice. Data source: Stooq via pandas-datareader.")

st.sidebar.markdown("### ⚙️ Global settings")
st.sidebar.info(
    "This app is for **learning and experimentation only**. "
    "Models are simple and ignore costs, slippage, etc."
)


# ---------------------- Utility helpers ---------------------- #

def validate_dates(start: date, end: date) -> bool:
    if start >= end:
        st.error("Start date must be before end date.")
        return False
    return True


@st.cache_data(show_spinner=True)
def load_stooq_close(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Load Close prices for a single ticker from Stooq."""
    if "." not in ticker:
        stooq_ticker = ticker + ".US"
    else:
        stooq_ticker = ticker

    try:
        df = web.DataReader(stooq_ticker, "stooq", start=start, end=end)
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

    df = df.sort_index()

    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    return df[["Close"]].copy()


@st.cache_data(show_spinner=True)
def load_stooq_closes_multi(tickers, start: date, end: date) -> pd.DataFrame:
    """Load Close prices for multiple tickers, aligned on common dates."""
    data = {}
    for t in tickers:
        df = load_stooq_close(t, start, end)
        if not df.empty:
            data[t] = df["Close"]
        else:
            st.warning(f"No data for {t}, skipping.")
    if not data:
        return pd.DataFrame()
    prices = pd.DataFrame(data).dropna()
    return prices


# ---------------------- SVM Predictor ---------------------- #

def build_svm_features(df_close: pd.DataFrame) -> pd.DataFrame:
    df = df_close.copy()
    df["return_1d"] = df["Close"].pct_change()
    df["return_5d"] = df["Close"].pct_change(5)
    df["ma_5"] = df["Close"].rolling(window=5).mean()
    df["ma_10"] = df["Close"].rolling(window=10).mean()
    df["dist_ma_5"] = (df["Close"] - df["ma_5"]) / df["ma_5"]
    df["vol_5d"] = df["return_1d"].rolling(window=5).std()

    # Target: 1 if tomorrow's close > today's
    df["tomorrow_close"] = df["Close"].shift(-1)
    df["target_up"] = (df["tomorrow_close"] > df["Close"]).astype(int)

    df = df.dropna()
    return df


def run_svm_model(df_feat: pd.DataFrame, kernel: str, C: float, test_ratio: float):
    feature_cols = [
        "return_1d",
        "return_5d",
        "ma_5",
        "ma_10",
        "dist_ma_5",
        "vol_5d",
    ]
    X = df_feat[feature_cols].values
    y = df_feat["target_up"].values

    if len(df_feat) < 50:
        st.warning("Very few samples after feature engineering. Results may be unreliable.")

    split_idx = int(len(df_feat) * (1 - test_ratio))
    if split_idx <= 0 or split_idx >= len(df_feat):
        st.error("Invalid train/test split. Try a smaller/larger test ratio.")
        return None

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=kernel, C=C, gamma="scale"))
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    baseline_acc = max(np.mean(y_test), 1 - np.mean(y_test))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Next-day prediction using last row
    last_features = X[-1].reshape(1, -1)
    next_up = model.predict(last_features)[0]

    # Class balance
    up_ratio = y.mean()
    down_ratio = 1 - up_ratio

    return {
        "model": model,
        "accuracy": acc,
        "baseline": baseline_acc,
        "next_up": int(next_up),
        "samples": len(df_feat),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "confusion_matrix": cm,
        "up_ratio": up_ratio,
        "down_ratio": down_ratio,
    }


# ---------------------- Analysis indicators ---------------------- #

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]

    df["daily_return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))
    df["cum_return"] = (1 + df["daily_return"]).cumprod() - 1
    df["ma_20"] = close.rolling(window=20).mean()
    df["ma_50"] = close.rolling(window=50).mean()
    df["ma_200"] = close.rolling(window=200).mean()

    running_max = close.cummax()
    drawdown = close / running_max - 1.0
    df["drawdown"] = drawdown
    df["max_drawdown"] = drawdown.cummin()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    close = df["Close"].dropna()
    if len(close) < 2:
        return {}

    start_date = close.index[0].date()
    end_date = close.index[-1].date()
    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])
    total_return = end_price / start_price - 1

    log_ret = df["log_return"].dropna()
    if len(log_ret) > 0:
        mean_log_daily = log_ret.mean()
        vol_log_daily = log_ret.std()
        trading_days = 252
        ann_return = math.exp(mean_log_daily * trading_days) - 1
        ann_vol = vol_log_daily * math.sqrt(trading_days)
        sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    else:
        ann_return = ann_vol = sharpe = np.nan

    max_dd = df["max_drawdown"].min() if "max_drawdown" in df.columns else np.nan

    daily_ret = df["daily_return"].dropna()
    if len(daily_ret) > 0:
        up_days = (daily_ret > 0).sum()
        down_days = (daily_ret < 0).sum()
        total_days = len(daily_ret)
        up_days_pct = up_days / total_days * 100
        down_days_pct = down_days / total_days * 100
    else:
        up_days_pct = down_days_pct = np.nan

    return {
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
    }


def short_text_summary(ticker: str, m: dict) -> str:
    if not m or "start_price" not in m or pd.isna(m["start_price"]):
        return f"{ticker}: not enough data."
    return (
        f"{ticker}: from {m['start_date']} to {m['end_date']}, price moved "
        f"from {m['start_price']:.2f} to {m['end_price']:.2f} "
        f"({m['total_return_pct']:.1f}% total, {m['ann_return_pct']:.1f}% annualised). "
        f"Ann. vol {m['ann_vol_pct']:.1f}%, Sharpe {m['sharpe']:.2f}, "
        f"max drawdown {m['max_drawdown_pct']:.1f}%."
    )


# ---------------------- Portfolio helpers ---------------------- #

def normalise_weights(raw_weights, n_assets):
    if raw_weights is None:
        w = np.ones(n_assets) / n_assets
    else:
        w = np.array(raw_weights, dtype=float)
        if len(w) != n_assets:
            st.error("Number of weights must match number of tickers.")
            return None
        if np.allclose(w.sum(), 0):
            st.error("Sum of weights is zero.")
            return None
        w = w / w.sum()
    return w


def compute_portfolio(prices: pd.DataFrame, weights: np.ndarray):
    asset_returns = prices.pct_change().dropna()
    port_ret = (asset_returns * weights).sum(axis=1)
    port_val = (1 + port_ret).cumprod()

    # Stats
    log_ret = np.log(1 + port_ret)
    mean_log_daily = log_ret.mean()
    vol_log_daily = log_ret.std()
    trading_days = 252
    ann_return = math.exp(mean_log_daily * trading_days) - 1
    ann_vol = vol_log_daily * math.sqrt(trading_days)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    running_max = port_val.cummax()
    drawdown = port_val / running_max - 1
    max_dd = drawdown.min()

    return {
        "returns": port_ret,
        "values": port_val,
        "ann_return_pct": ann_return * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
    }


# ---------------------- Main UI Mode Selector ---------------------- #

mode = st.sidebar.radio(
    "Select tool",
    ["🔮 SVM Stock Predictor", "📊 Multi-Ticker Analysis", "📦 Portfolio Backtester"],
)


# ====== MODE 1: SVM PREDICTOR ====== #

if mode == "🔮 SVM Stock Predictor":
    st.header("🔮 Single-Stock SVM Up/Down Predictor")

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Ticker (no .US)", value="AAPL").upper()
    with col2:
        start_date = st.date_input("Start date", value=date(2018, 1, 1))
    with col3:
        end_date = st.date_input("End date", value=date(2023, 1, 1))

    with st.expander("Advanced SVM settings"):
        kernel = st.selectbox("SVM kernel", options=["rbf", "linear", "poly"], index=0)
        C = st.slider("Regularisation C", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        test_ratio = st.slider("Test size (fraction)", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    if st.button("Run SVM"):
        if not validate_dates(start_date, end_date):
            st.stop()

        df_close = load_stooq_close(ticker, start_date, end_date)
        if df_close.empty:
            st.error("No data found for this ticker/date range.")
        else:
            tabs = st.tabs(["📉 Price", "🤖 Model performance", "📊 Confusion matrix"])
            with tabs[0]:
                st.subheader("Price history")
                st.line_chart(df_close["Close"])

            df_feat = build_svm_features(df_close)
            if df_feat.empty:
                st.error("Not enough data after feature engineering.")
            else:
                results = run_svm_model(df_feat, kernel=kernel, C=C, test_ratio=test_ratio)
                if results is None:
                    st.stop()

                with tabs[1]:
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Samples", results["samples"])
                    col_b.metric("Train size", results["train_size"])
                    col_c.metric("Test size", results["test_size"])
                    col_d.metric("Test accuracy", f"{results['accuracy']:.3f}")

                    st.write(f"Baseline accuracy (always majority): **{results['baseline']:.3f}**")
                    st.write(
                        f"Class balance (all data): **UP ~ {results['up_ratio']*100:.1f}%**, "
                        f"DOWN ~ {results['down_ratio']*100:.1f}%"
                    )

                    st.subheader("Next-day prediction")
                    if results["next_up"] == 1:
                        st.success("Model says: Tomorrow will **CLOSE UP** 📈")
                    else:
                        st.warning("Model says: Tomorrow will **CLOSE DOWN** 📉")

                with tabs[2]:
                    cm = results["confusion_matrix"]
                    cm_df = pd.DataFrame(
                        cm,
                        index=["Actual DOWN (0)", "Actual UP (1)"],
                        columns=["Pred DOWN (0)", "Pred UP (1)"],
                    )
                    st.write("Confusion matrix (test set):")
                    st.dataframe(cm_df)


# ====== MODE 2: MULTI-TICKER ANALYSIS ====== #

elif mode == "📊 Multi-Ticker Analysis":
    st.header("📊 Multi-Ticker Technical & Risk Analysis")

    tickers_str = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,TSLA")
    start_date = st.date_input("Start date", value=date(2018, 1, 1))
    end_date = st.date_input("End date", value=date(2023, 1, 1))

    show_plots = st.checkbox("Show plots", value=True)

    if st.button("Run analysis"):
        if not validate_dates(start_date, end_date):
            st.stop()

        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        prices = load_stooq_closes_multi(tickers, start_date, end_date)
        if prices.empty:
            st.error("No overlapping data for any ticker.")
        else:
            tabs = st.tabs(["📉 Prices", "📑 Metrics", "📈 Drawdowns & Correlations"])

            metrics_list = []
            daily_return_dict = {}

            for t in prices.columns:
                df = prices[[t]].rename(columns={t: "Close"})
                df_ind = compute_indicators(df)
                m = compute_metrics(df_ind)
                m["ticker"] = t
                metrics_list.append(m)
                daily_return_dict[t] = df_ind["daily_return"]

            metrics_df = pd.DataFrame(metrics_list).set_index("ticker")

            with tabs[0]:
                st.subheader("Price history (normalised)")
                norm_prices = prices / prices.iloc[0]
                st.line_chart(norm_prices)

            with tabs[1]:
                st.subheader("Metrics table")
                st.dataframe(metrics_df)

                # Short summaries
                st.subheader("Short summaries")
                for t in metrics_df.index:
                    m = metrics_df.loc[t].to_dict()
                    st.write("• " + short_text_summary(t, m))

                # Download metrics CSV
                csv_bytes = metrics_df.to_csv().encode("utf-8")
                st.download_button(
                    "Download metrics as CSV",
                    data=csv_bytes,
                    file_name="stock_metrics.csv",
                    mime="text/csv",
                )

            with tabs[2]:
                if len(daily_return_dict) >= 2:
                    ret_df = pd.DataFrame(daily_return_dict).dropna()
                    corr = ret_df.corr()
                    st.subheader("Daily return correlation")
                    st.dataframe(corr.style.background_gradient(cmap="coolwarm"))
                else:
                    st.info("Correlation needs at least two tickers with valid data.")

                if show_plots:
                    st.subheader("Drawdowns")
                    dd_df = pd.DataFrame(
                        {
                            t: (prices[t] / prices[t].cummax() - 1)
                            for t in prices.columns
                        }
                    )
                    st.line_chart(dd_df)


# ====== MODE 3: PORTFOLIO BACKTESTER ====== #

elif mode == "📦 Portfolio Backtester":
    st.header("📦 Simple Portfolio Backtester")

    tickers_str = st.text_input("Tickers (comma-separated)", value="AAPL,MSFT,TSLA")
    weights_str = st.text_input(
        "Weights (comma-separated, optional – will be normalised)",
        value="",
        placeholder="e.g. 0.5,0.3,0.2 or leave blank for equal weights",
    )
    start_date = st.date_input("Start date", value=date(2018, 1, 1))
    end_date = st.date_input("End date", value=date(2023, 1, 1))

    if st.button("Run backtest"):
        if not validate_dates(start_date, end_date):
            st.stop()

        tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
        if not tickers:
            st.error("Please provide at least one ticker.")
        else:
            prices = load_stooq_closes_multi(tickers, start_date, end_date)
            if prices.empty:
                st.error("No overlapping data for these tickers.")
            else:
                if weights_str.strip():
                    try:
                        raw_weights = [
                            float(x.strip()) for x in weights_str.split(",") if x.strip()
                        ]
                    except ValueError:
                        st.error("Could not parse weights as numbers.")
                        raw_weights = None
                else:
                    raw_weights = None

                weights = normalise_weights(raw_weights, n_assets=len(prices.columns))
                if weights is None:
                    st.stop()

                st.write("Using tickers:", list(prices.columns))
                st.write("Normalised weights:", [f"{w:.2f}" for w in weights])

                port = compute_portfolio(prices, weights)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Ann. return", f"{port['ann_return_pct']:.2f}%")
                col2.metric("Ann. vol", f"{port['ann_vol_pct']:.2f}%")
                col3.metric("Sharpe (rf≈0)", f"{port['sharpe']:.2f}")
                col4.metric("Max drawdown", f"{port['max_drawdown_pct']:.2f}%")

                st.subheader("Portfolio vs individual assets (normalised)")
                norm_prices = prices / prices.iloc[0]
                df_plot = norm_prices.copy()
                df_plot["PORTFOLIO"] = port["values"]
                st.line_chart(df_plot)
