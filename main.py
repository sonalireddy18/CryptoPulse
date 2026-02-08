import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as goimport streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from textblob import TextBlob

# ---------------- CONFIG ----------------
st.set_page_config(page_title="CryptoPulse AI (INR)", layout="wide")
st.title("📈 CryptoPulse: Dashboard")
# ---------------- CONTROLS ----------------
st.markdown("### ⚙️ Dashboard Controls")
c1, c2 = st.columns(2)
with c1:
    crypto_choice = st.selectbox(
        "Select Cryptocurrency",
        ("BTC-USD", "ETH-USD", "DOGE-USD", "XRP-USD")
    )
with c2:
    prediction_days = st.slider("Forecast Horizon (Days)", 1, 30, 7)
st.markdown("---")
# ---------------- DATA FUNCTIONS ----------------
@st.cache_data(ttl=3600)
def get_conversion_rate():
    try:
        ticker = yf.Ticker("USDINR=X")
        data = ticker.history(period="1d")
        if data.empty:
            return 83.0
        return float(data["Close"].iloc[-1])
    except:
        return 83.0
@st.cache_data(ttl=3600)
def load_data(ticker, rate):
    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col] * rate
    return df
# ---------------- MAIN EXECUTION ----------------
with st.spinner("Loading market data..."):
    try:
        usd_to_inr = get_conversion_rate()
        data = load_data(crypto_choice, usd_to_inr)

        # -------- HARD SAFETY CHECK --------
        if data.empty or "Close" not in data.columns:
            st.error("No market data received. Please try again later or change coin.")
            st.stop()
        # ---------------- SENTIMENT ----------------
        try:
            ticker_obj = yf.Ticker(crypto_choice)
            news = ticker_obj.news
            titles = [n.get("title", "") for n in news[:5]] if news else []
            avg_polarity = (
                sum(TextBlob(t).sentiment.polarity for t in titles) / len(titles)
                if titles else 0.0
            )
        except:
            avg_polarity = 0.0
        fng_value = (avg_polarity + 1) * 50
        # ---------------- INDICATORS ----------------
        current_price = float(data["Close"].iloc[-1])

        data["SMA_20"] = data["Close"].rolling(window=20).mean()

        if data["SMA_20"].dropna().empty:
            last_sma = current_price
        else:
            last_sma = float(data["SMA_20"].iloc[-1])
        # ---------------- KPIs ----------------
        left, right = st.columns(2)
        with left:
            st.metric("Current Price (INR)", f"₹{current_price:,.2f}")
            st.metric("Sentiment Score", f"{avg_polarity:.2f}")
            if current_price > last_sma and avg_polarity > 0.05:
                st.markdown("### Signal: :green[**🟢 STRONG BUY**]")
                st.info("Positive sentiment + price above SMA")

            elif current_price < last_sma and avg_polarity < -0.05:
                st.markdown("### Signal: :red[**🔴 STRONG SELL**]")
                st.info("Negative sentiment + price below SMA")

            else:
                st.markdown("## Signal: :orange[**🟡 HOLD**]")
                st.markdown("#### Wait for clearer confirmation.")

        with right:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fng_value,
                title={"text": "Fear & Greed Index"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 25], "color": "red"},
                        {"range": [25, 50], "color": "orange"},
                        {"range": [50, 75], "color": "yellowgreen"},
                        {"range": [75, 100], "color": "green"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ---------------- PROPHET ----------------
        df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        df_train["ds"] = df_train["ds"].dt.tz_localize(None)

        model = Prophet(daily_seasonality=True)
        model.fit(df_train)

        future = model.make_future_dataframe(periods=prediction_days)
        forecast = model.predict(future)
        # ---------------- FORECAST GRAPH ----------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="History"))
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            name="Predicted",
            line=dict(dash="dash", color="orange")
        ))

        fig.update_layout(
            title=f"AI Forecast for {crypto_choice}",
            yaxis_title="Price (₹)",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- BREAKDOWN TABLE ----------------
        st.subheader("Prediction Breakdown")

        breakdown = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(prediction_days)
        breakdown = breakdown.rename(columns={
            "ds": "Date",
            "yhat": "Predicted Price",
            "yhat_lower": "Lower Bound",
            "yhat_upper": "Upper Bound",
        })

        breakdown.set_index("Date", inplace=True)
        st.dataframe(breakdown, use_container_width=True)

    except Exception as e:
        st.error(f"Unexpected error: {e}")

from textblob import TextBlob
# --- CONFIGURATION ---
st.set_page_config(page_title="CryptoPulse AI (INR)", layout="wide")
st.title("📈 CryptoPulse: Dashboard")

# --- 1. TOP USER CONTROLS ---
st.markdown("### ⚙️ Dashboard Controls")
top_col1, top_col2 = st.columns(2)

with top_col1:
    crypto_choice = st.selectbox("Select Cryptocurrency", ("BTC-USD", "ETH-USD", "DOGE-USD", "XRP-USD"))
with top_col2:
    prediction_days = st.slider("Forecast Horizon (Days)", 1, 30, 7)

st.markdown("---")

# --- 2. DATA FETCHING ---
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

# --- 3. EXECUTION ---
try:
    usd_to_inr = get_conversion_rate()
    data = load_data(crypto_choice, usd_to_inr)
    
    # Sentiment Logic
    ticker_obj = yf.Ticker(crypto_choice)
    news = ticker_obj.news
    titles = [n.get('title', '') for n in news[:5]] if news else []
    avg_polarity = sum([TextBlob(t).sentiment.polarity for t in titles]) / len(titles) if titles else 0.0
    fng_value = (avg_polarity + 1) * 50  # Scales -1...1 to 0...100

    # Indicators
    current_price_inr = float(data['Close'].iloc[-1])
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    last_sma = float(data['SMA_20'].iloc[-1]) if not pd.isna(data['SMA_20'].iloc[-1]) else current_price_inr

    # --- 4. MAIN KPI & FEAR/GREED METER ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.metric("Current Price (INR)", f"₹{current_price_inr:,.2f}")
        st.metric("Sentiment Score", f"{avg_polarity:.2f}")
        
        # --- BOLD SIGNAL LOGIC ---
        if current_price_inr > last_sma and avg_polarity > 0.05:
            st.markdown("### Signal: :green[**🟢 STRONG BUY**]")
            st.info("Positive Sentiment + Upward Trend")
        elif current_price_inr < last_sma and avg_polarity < -0.05:
            st.markdown("### Signal: :red[**🔴 STRONG SELL**]")
            st.info("Negative Sentiment + Downward Trend")
        else:
            # HERE IS THE BIG BOLD YELLOW HOLD SIGNAL
            st.markdown("## Signal: :orange[**🟡 HOLD**]")
            st.markdown("#### **Wait for a clearer market trend.**")

    with col_right:
        # FEAR & GREED GAUGE
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = fng_value,
            title = {'text': "Fear & Greed Index"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 25], 'color': "red"},
                    {'range': [25, 50], 'color': "orange"},
                    {'range': [50, 75], 'color': "yellowgreen"},
                    {'range': [75, 100], 'color': "green"}],
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- 5. AI PREDICTION ---
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    df_train['ds'] = df_train['ds'].dt.tz_localize(None)
    
    model = Prophet(daily_seasonality=True)
    model.fit(df_train)
    future = model.make_future_dataframe(periods=prediction_days)
    forecast = model.predict(future)
    # Prediction Graph
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="History"))
    fig_line.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted", line=dict(dash='dash', color='orange')))
    fig_line.update_layout(title=f"AI Forecast for {crypto_choice}", yaxis_title="Price (₹)", hovermode="x unified")
    st.plotly_chart(fig_line, use_container_width=True)

    # --- 6. PREDICTION BREAKDOWN ---
    st.subheader("Prediction Breakdown")
    # We select columns and set Date (ds) as the index to remove row numbers 731, 732
    breakdown_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_days)
    breakdown_df = breakdown_df.rename(columns={'ds': 'Date', 'yhat': 'Predicted Price', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})
    breakdown_df.set_index('Date', inplace=True)
    st.dataframe(breakdown_df, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
