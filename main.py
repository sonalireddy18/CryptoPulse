import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from textblob import TextBlob

# --- CONFIGURATION ---
st.set_page_config(page_title="CryptoPulse AI (INR)", layout="wide")
st.title("📈 CryptoPulse: AI Prediction Dashboard")

# --- 1. TOP USER CONTROLS ---
st.markdown("### ⚙️ Dashboard Controls")
top_col1, top_col2 = st.columns(2)

with top_col1:
    crypto_choice = st.selectbox("Select Cryptocurrency", ("BTC-USD", "ETH-USD", "DOGE-USD", "XRP-USD"))
with top_col2:
    prediction_days = st.slider("Forecast Horizon (Days)", 1, 30, 7)

st.markdown("---")

# --- 2. DATA FETCHING (INR) ---
@st.cache_data(ttl=3600)
def get_conversion_rate():
    ticker = yf.Ticker("USDINR=X")
    data = ticker.history(period="1d")
    return float(data['Close'].iloc[-1])

@st.cache_data(ttl=3600)
def load_data(ticker, rate):
    df = yf.download(ticker, period="2y", auto_adjust=True)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col] * rate
    return df

# --- 3. EXECUTION LOGIC ---
try:
    usd_to_inr = get_conversion_rate()
    data = load_data(crypto_choice, usd_to_inr)
    
    # Simple Sentiment Logic
    ticker_obj = yf.Ticker(crypto_choice)
    news = ticker_obj.news
    titles = [n.get('title', '') for n in news[:5]] if news else []
    sentiment_score = sum([TextBlob(t).sentiment.polarity for t in titles]) / len(titles) if titles else 0.0

    # Indicator Math
    current_price_inr = float(data['Close'].iloc[-1])
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    last_sma = float(data['SMA_20'].iloc[-1]) if not pd.isna(data['SMA_20'].iloc[-1]) else current_price_inr

    def get_signal(price, sma, sent):
        if price > sma and sent > 0.05: return "🟢 STRONG BUY", "Upward Price Trend + Positive Sentiment"
        elif price < sma and sent < -0.05: return "🔴 STRONG SELL", "Downward Price Trend + Negative Sentiment"
        else: return "🟡 HOLD", "Neutral Market Trend"

    signal, reason = get_signal(current_price_inr, last_sma, sentiment_score)

    # --- 4. MAIN KPI DISPLAY ---
    col1, col2 = st.columns(2)
    col1.metric("Current Price (INR)", f"₹{current_price_inr:,.2f}")
    col2.subheader(f"Signal: {signal}")
    st.info(f"Analysis: {reason}")

    # --- 5. AI PREDICTION GRAPH ---
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    df_train['ds'] = df_train['ds'].dt.tz_localize(None)
    
    model = Prophet(daily_seasonality=True)
    model.fit(df_train)
    future = model.make_future_dataframe(periods=prediction_days)
    forecast = model.predict(future)

    st.subheader(f"Price Forecast (INR)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="History"))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted", line=dict(dash='dash', color='orange')))
    fig.update_layout(yaxis_title="Price (₹)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. PREDICTION BREAKDOWN ---
    st.subheader("Prediction Breakdown")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_days))

except Exception as e:
    st.error(f"Error: {e}")
