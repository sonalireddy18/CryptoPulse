# CryptoPulse 

**CryptoPulse** is a real-time cryptocurrency dashboard that provides price tracking in INR, sentiment analysis from news headlines, and AI-driven price forecasting.

## Features
* **Live Conversion:** Fetches USD to INR exchange rates automatically.
* **Sentiment Analysis:** Uses `TextBlob` to analyze the latest news headlines for the selected crypto.
* **Fear & Greed Index:** A custom visual gauge based on market sentiment.
* **AI Forecasting:** Leverages Facebook's `Prophet` model to predict prices for the next 1–30 days.
* **Trading Signals:** Provides "Buy", "Hold", or "Sell" recommendations based on SMA and sentiment logic.

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/CryptoPulse.git](https://github.com/YOUR_USERNAME/CryptoPulse.git)
    cd CryptoPulse
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:**
    ```bash
    streamlit run main.py
    ```

## Tech Stack
* **Frontend:** Streamlit
* **Data:** yfinance (Yahoo Finance API)
* **Forecasting:** Prophet
* **Analysis:** TextBlob, Pandas
* **Visualization:** Plotly

