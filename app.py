import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


st.set_page_config(page_title="Agentic AI Trading Dashboard", layout="wide")


# -----------------------------
# Session state initialization
# -----------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

if "chatgpt_prompt" not in st.session_state:
    st.session_state.chatgpt_prompt = ""

if "chatgpt_output" not in st.session_state:
    st.session_state.chatgpt_output = ""


# -----------------------------
# Utility functions
# -----------------------------
def safe_download_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()
    df = df.reset_index()

    # Handle both Date and Datetime index names safely
    date_col = None
    for col in df.columns:
        if str(col).lower() in ["date", "datetime"]:
            date_col = col
            break

    if date_col is None:
        raise ValueError("No Date/Datetime column found after download.")

    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates().reset_index(drop=True)
    df = df.ffill().dropna().reset_index(drop=True)
    return df


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_features(df: pd.DataFrame, sma_short: int, sma_long: int, rsi_window: int) -> pd.DataFrame:
    out = df.copy()
    out["Daily Return"] = out["Close"].pct_change()
    out["SMA_Short"] = out["Close"].rolling(window=sma_short).mean()
    out["SMA_Long"] = out["Close"].rolling(window=sma_long).mean()
    out["RSI"] = compute_rsi(out["Close"], window=rsi_window)

    ema_12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema_12 - ema_26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["Rolling Volatility"] = out["Daily Return"].rolling(20).std()
    return out


def make_baseline_signals(df: pd.DataFrame, rsi_buy_max: float, rsi_sell_min: float) -> pd.DataFrame:
    out = df.copy()
    out["Baseline_Signal"] = "Hold"

    buy_condition = (
        (out["SMA_Short"] > out["SMA_Long"]) &
        (out["RSI"] < rsi_buy_max) &
        (out["MACD"] > out["MACD_Signal"])
    )

    sell_condition = (
        (out["SMA_Short"] < out["SMA_Long"]) &
        (out["RSI"] > rsi_sell_min) &
        (out["MACD"] < out["MACD_Signal"])
    )

    out.loc[buy_condition, "Baseline_Signal"] = "Buy"
    out.loc[sell_condition, "Baseline_Signal"] = "Sell"
    return out


def get_yahoo_finance_news(ticker: str, max_items: int = 10) -> pd.DataFrame:
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(url)

    rows: List[Dict[str, Optional[str]]] = []
    for entry in feed.entries[:max_items]:
        rows.append(
            {
                "title": getattr(entry, "title", None),
                "published": getattr(entry, "published", None),
                "link": getattr(entry, "link", None),
            }
        )

    return pd.DataFrame(rows)


def simple_sentiment_label(text: str) -> str:
    positive_words = {
        "beat", "beats", "surge", "rally", "gain", "gains", "growth", "strong",
        "upgrade", "record", "profit", "profits", "optimism", "bullish",
        "expands", "outperform", "rebound", "boost", "positive"
    }
    negative_words = {
        "miss", "misses", "fall", "falls", "drop", "drops", "weak", "downgrade",
        "loss", "losses", "lawsuit", "risk", "risks", "bearish", "cut", "cuts",
        "decline", "slump", "warning", "negative"
    }

    tokens = {token.strip(".,:;!?()[]{}\"'").lower() for token in text.split()}
    pos_score = len(tokens & positive_words)
    neg_score = len(tokens & negative_words)

    if pos_score > neg_score:
        return "Positive"
    if neg_score > pos_score:
        return "Negative"
    return "Neutral"


def score_news_sentiment(news_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, float]:
    if news_df.empty:
        empty = pd.DataFrame(columns=["headline", "sentiment", "sentiment_value"])
        return empty, "Neutral", 0.0

    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    rows = []

    for title in news_df["title"].fillna(""):
        label = simple_sentiment_label(title)
        rows.append(
            {
                "headline": title,
                "sentiment": label,
                "sentiment_value": sentiment_map[label],
            }
        )

    sentiment_df = pd.DataFrame(rows)
    avg_score = float(sentiment_df["sentiment_value"].mean()) if not sentiment_df.empty else 0.0

    if avg_score > 0.2:
        overall = "Positive"
    elif avg_score < -0.2:
        overall = "Negative"
    else:
        overall = "Neutral"

    return sentiment_df, overall, avg_score


@dataclass
class AgentDecision:
    market_signal: str
    sentiment_signal: str
    initial_decision: str
    final_decision: str
    risk_message: str


def decision_agent(latest_row: pd.Series, overall_sentiment: str) -> str:
    bullish_trend = latest_row["SMA_Short"] > latest_row["SMA_Long"]
    bullish_momentum = latest_row["MACD"] > latest_row["MACD_Signal"]
    rsi_buy_ok = latest_row["RSI"] < 70

    bearish_trend = latest_row["SMA_Short"] < latest_row["SMA_Long"]
    bearish_momentum = latest_row["MACD"] < latest_row["MACD_Signal"]
    rsi_sell_ok = latest_row["RSI"] > 30

    if bullish_trend and bullish_momentum and rsi_buy_ok and overall_sentiment == "Positive":
        return "Buy"
    if bearish_trend and bearish_momentum and rsi_sell_ok and overall_sentiment == "Negative":
        return "Sell"
    return "Hold"


def risk_agent(decision: str, latest_row: pd.Series, volatility_threshold: float) -> Tuple[str, str]:
    vol = latest_row.get("Rolling Volatility", np.nan)

    if pd.notna(vol) and vol > volatility_threshold and decision == "Buy":
        return "Hold", "Risk Agent blocked Buy because rolling volatility exceeded the selected threshold."

    return decision, "Risk Agent approved the decision."


def apply_agentic_strategy(
    df: pd.DataFrame,
    sentiment_mode: str,
    static_sentiment: str,
    volatility_threshold: float,
) -> pd.DataFrame:
    out = df.copy()

    if sentiment_mode == "Manual sentiment":
        out["Overall_Sentiment"] = static_sentiment
    else:
        rolling_sentiment = []
        for _, row in out.iterrows():
            if row["MACD"] > row["MACD_Signal"] and row["RSI"] < 65:
                rolling_sentiment.append("Positive")
            elif row["MACD"] < row["MACD_Signal"] and row["RSI"] > 55:
                rolling_sentiment.append("Negative")
            else:
                rolling_sentiment.append("Neutral")
        out["Overall_Sentiment"] = rolling_sentiment

    out["Agentic_Signal"] = "Hold"

    for idx in range(len(out)):
        row = out.iloc[idx]
        initial_decision = decision_agent(row, row["Overall_Sentiment"])
        final_decision, _ = risk_agent(initial_decision, row, volatility_threshold)
        out.at[idx, "Agentic_Signal"] = final_decision

    return out


def make_positions(signal_series: pd.Series) -> pd.Series:
    position = pd.Series(np.nan, index=signal_series.index)
    position[signal_series == "Buy"] = 1
    position[signal_series == "Sell"] = 0
    return position.ffill().fillna(0)


def add_backtest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Baseline_Position"] = make_positions(out["Baseline_Signal"])
    out["Agentic_Position"] = make_positions(out["Agentic_Signal"])

    out["Baseline_Return"] = out["Daily Return"] * out["Baseline_Position"].shift(1)
    out["Agentic_Return"] = out["Daily Return"] * out["Agentic_Position"].shift(1)

    out["Cumulative_Market_Return"] = (1 + out["Daily Return"].fillna(0)).cumprod()
    out["Cumulative_Baseline_Return"] = (1 + out["Baseline_Return"].fillna(0)).cumprod()
    out["Cumulative_Agentic_Return"] = (1 + out["Agentic_Return"].fillna(0)).cumprod()

    return out


def annualized_volatility(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty or s.std() == 0:
        return 0.0
    return float(s.std() * np.sqrt(252))


def sharpe_ratio(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty or s.std() == 0:
        return 0.0
    return float((s.mean() / s.std()) * np.sqrt(252))


def max_drawdown(cumulative_series: pd.Series) -> float:
    rolling_max = cumulative_series.cummax()
    drawdown = (cumulative_series - rolling_max) / rolling_max
    return float(drawdown.min())


def metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Strategy": ["Buy-and-Hold", "Baseline Technical", "Agentic AI"],
            "Total Return": [
                df["Cumulative_Market_Return"].iloc[-1] - 1,
                df["Cumulative_Baseline_Return"].iloc[-1] - 1,
                df["Cumulative_Agentic_Return"].iloc[-1] - 1,
            ],
            "Sharpe Ratio": [
                sharpe_ratio(df["Daily Return"]),
                sharpe_ratio(df["Baseline_Return"]),
                sharpe_ratio(df["Agentic_Return"]),
            ],
            "Volatility": [
                annualized_volatility(df["Daily Return"]),
                annualized_volatility(df["Baseline_Return"]),
                annualized_volatility(df["Agentic_Return"]),
            ],
            "Max Drawdown": [
                max_drawdown(df["Cumulative_Market_Return"]),
                max_drawdown(df["Cumulative_Baseline_Return"]),
                max_drawdown(df["Cumulative_Agentic_Return"]),
            ],
        }
    )


def cumulative_return_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Cumulative_Market_Return"], mode="lines", name="Buy-and-Hold"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Cumulative_Baseline_Return"], mode="lines", name="Baseline Technical"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Cumulative_Agentic_Return"], mode="lines", name="Agentic AI"))
    fig.update_layout(
        title="Cumulative Return Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Return"
    )
    return fig


def price_indicator_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_Short"], mode="lines", name="SMA Short"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_Long"], mode="lines", name="SMA Long"))
    fig.update_layout(
        title="Price and Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], mode="lines", name="RSI"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(
        title="RSI",
        xaxis_title="Date",
        yaxis_title="RSI"
    )
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], mode="lines", name="MACD Signal"))
    fig.update_layout(
        title="MACD",
        xaxis_title="Date",
        yaxis_title="Value"
    )
    return fig


def get_openai_interpretation(prompt: str, model: str) -> str:
    if OpenAI is None:
        return "OpenAI package is not installed."

    api_key = None

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "OPENAI_API_KEY not found in Streamlit secrets or environment variables."

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=prompt,
        )
        return response.output_text.strip()
    except Exception as e:
        return f"Error generating interpretation: {e}"


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Controls")

with st.sidebar:
    ticker = st.text_input("Ticker", value="MSFT").upper().strip()
    period = st.selectbox("History period", ["1y", "2y", "5y", "10y"], index=2)
    interval = st.selectbox("Data interval", ["1d", "1wk", "1mo"], index=0)

    st.markdown("---")
    st.subheader("Indicator Settings")
    sma_short = st.slider("Short SMA", min_value=5, max_value=50, value=20)
    sma_long = st.slider("Long SMA", min_value=20, max_value=200, value=50)
    rsi_window = st.slider("RSI Window", min_value=7, max_value=30, value=14)
    rsi_buy_max = st.slider("Baseline Buy RSI Upper Limit", min_value=50, max_value=80, value=70)
    rsi_sell_min = st.slider("Baseline Sell RSI Lower Limit", min_value=20, max_value=50, value=30)

    st.markdown("---")
    st.subheader("Agentic AI Settings")
    sentiment_mode = st.radio("Sentiment mode", ["Manual sentiment", "Derived sentiment"], index=0)
    static_sentiment = st.selectbox("Manual overall sentiment", ["Positive", "Neutral", "Negative"], index=0)
    volatility_threshold = st.slider(
        "Risk Agent Volatility Threshold",
        min_value=0.005,
        max_value=0.080,
        value=0.030,
        step=0.001,
    )

    st.markdown("---")
    st.subheader("ChatGPT Interpretation")
    openai_model = st.text_input("OpenAI Model", value="gpt-4o-mini")

    run_button = st.button("Run Analysis", type="primary", key="run_analysis_button")


# -----------------------------
# Main app
# -----------------------------
st.title("Agentic AI Trading Dashboard")
st.caption("Interactive Streamlit app for baseline vs agentic AI stock strategy comparison")

st.markdown(
    """
This app lets you pick a stock ticker and compare three approaches:
- Buy-and-Hold
- Baseline technical strategy
- Agentic AI strategy with sentiment and risk-control layers

The agentic pipeline uses four conceptual agents:
1. Market Analysis Agent
2. News Retrieval / Sentiment Agent
3. Decision Agent
4. Risk Agent
"""
)

if run_button:
    try:
        data = safe_download_data(ticker, period, interval)
        data = add_features(data, sma_short=sma_short, sma_long=sma_long, rsi_window=rsi_window)
        data = make_baseline_signals(data, rsi_buy_max=rsi_buy_max, rsi_sell_min=rsi_sell_min)

        news_df = get_yahoo_finance_news(ticker)
        sentiment_df, overall_sentiment, avg_sentiment_score = score_news_sentiment(news_df)

        if sentiment_mode == "Manual sentiment":
            overall_sentiment_to_use = static_sentiment
        else:
            overall_sentiment_to_use = overall_sentiment

        data = apply_agentic_strategy(
            data,
            sentiment_mode=sentiment_mode,
            static_sentiment=overall_sentiment_to_use,
            volatility_threshold=volatility_threshold,
        )

        data = add_backtest(data)
        results = metrics_table(data)

        latest = data.iloc[-1]
        initial_decision = decision_agent(latest, overall_sentiment_to_use)
        final_decision, risk_message = risk_agent(initial_decision, latest, volatility_threshold)

        default_prompt = f"""
Interpretation of the strategy results.

Ticker: {ticker}
Period: {period}
Interval: {interval}

Performance summary:
{results.to_string(index=False)}

Latest technical state:
Close={latest['Close']:.2f}, SMA_Short={latest['SMA_Short']:.2f}, SMA_Long={latest['SMA_Long']:.2f}, RSI={latest['RSI']:.2f}, MACD={latest['MACD']:.4f}, MACD_Signal={latest['MACD_Signal']:.4f}

Sentiment used: {overall_sentiment_to_use}
Initial decision: {initial_decision}
Final decision: {final_decision}
Risk note: {risk_message}

Please explain:
1. Which strategy performed best and why.
2. What the agentic AI layer changed.
3. The trade-off between return and risk.
4. A concise academic interpretation suitable for a report.
""".strip()

        st.session_state.analysis_done = True
        st.session_state.chatgpt_prompt = default_prompt
        st.session_state.chatgpt_output = ""
        st.session_state.analysis_data = {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "sentiment_mode": sentiment_mode,
            "static_sentiment": static_sentiment,
            "volatility_threshold": volatility_threshold,
            "openai_model": openai_model,
            "data": data,
            "news_df": news_df,
            "sentiment_df": sentiment_df,
            "results": results,
            "latest": latest,
            "overall_sentiment_to_use": overall_sentiment_to_use,
            "avg_sentiment_score": avg_sentiment_score,
            "initial_decision": initial_decision,
            "final_decision": final_decision,
            "risk_message": risk_message,
        }

    except Exception as exc:
        st.error(f"Error: {exc}")


if st.session_state.analysis_done and st.session_state.analysis_data is not None:
    analysis = st.session_state.analysis_data
    data = analysis["data"]
    news_df = analysis["news_df"]
    sentiment_df = analysis["sentiment_df"]
    results = analysis["results"]
    latest = analysis["latest"]
    overall_sentiment_to_use = analysis["overall_sentiment_to_use"]
    avg_sentiment_score = analysis["avg_sentiment_score"]
    initial_decision = analysis["initial_decision"]
    final_decision = analysis["final_decision"]
    risk_message = analysis["risk_message"]
    ticker_display = analysis["ticker"]
    sentiment_mode_display = analysis["sentiment_mode"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ticker", ticker_display)
    col2.metric("Latest Close", f"{latest['Close']:.2f}")
    col3.metric("Baseline Signal", latest["Baseline_Signal"])
    col4.metric("Agentic Final Decision", final_decision)

    st.subheader("Agentic AI Summary")
    left, right = st.columns(2)
    with left:
        st.write(f"**Sentiment mode:** {sentiment_mode_display}")
        st.write(f"**Overall sentiment used:** {overall_sentiment_to_use}")
        st.write(f"**Sentiment score:** {avg_sentiment_score:.2f}")
    with right:
        st.write(f"**Initial decision:** {initial_decision}")
        st.write(f"**Final decision:** {final_decision}")
        st.write(f"**Risk agent note:** {risk_message}")

    st.subheader("Performance Metrics")
    styled_results = results.copy()
    for col in ["Total Return", "Volatility", "Max Drawdown"]:
        styled_results[col] = styled_results[col].map(lambda x: f"{x:.2%}")
    styled_results["Sharpe Ratio"] = styled_results["Sharpe Ratio"].map(lambda x: f"{x:.2f}")
    st.dataframe(styled_results, use_container_width=True)

    st.subheader("Return Comparison")
    st.plotly_chart(cumulative_return_chart(data), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(price_indicator_chart(data), use_container_width=True)
    with c2:
        st.plotly_chart(rsi_chart(data), use_container_width=True)

    st.plotly_chart(macd_chart(data), use_container_width=True)

    st.subheader("Recent News and Sentiment")
    if news_df.empty:
        st.info("No Yahoo Finance RSS headlines were returned for this ticker right now.")
    else:
        news_display = news_df.copy().rename(
            columns={"title": "Headline", "published": "Published", "link": "Link"}
        )
        if not sentiment_df.empty:
            news_display["Sentiment"] = sentiment_df["sentiment"].values[: len(news_display)]
        st.dataframe(news_display, use_container_width=True)

    st.subheader("Raw Backtest Data")
    preview_cols = [
        "Date", "Close", "SMA_Short", "SMA_Long", "RSI", "MACD", "MACD_Signal",
        "Baseline_Signal", "Agentic_Signal", "Baseline_Position", "Agentic_Position",
        "Cumulative_Market_Return", "Cumulative_Baseline_Return", "Cumulative_Agentic_Return"
    ]
    available_preview_cols = [c for c in preview_cols if c in data.columns]
    st.dataframe(data[available_preview_cols].tail(50), use_container_width=True)

    csv_bytes = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results CSV",
        data=csv_bytes,
        file_name=f"{ticker_display.lower()}_agentic_trading_results.csv",
        mime="text/csv",
        key="download_results_csv_button",
    )

    st.subheader("ChatGPT Results Interpretation")

    if st.button("Generate ChatGPT Interpretation", key="generate_chatgpt_interpretation_button"):
        with st.spinner("Generating interpretation..."):
            st.session_state.chatgpt_output = get_openai_interpretation(
                st.session_state.chatgpt_prompt,
                openai_model
            )

    if st.session_state.chatgpt_output:
        st.write(st.session_state.chatgpt_output)

else:
    st.info("Pick your settings in the sidebar and click 'Run Analysis'.")
