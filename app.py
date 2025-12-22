import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# Page config (MUST be first)
# -------------------------------
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# -------------------------------
# App title & description
# -------------------------------
st.title("📈 Stock Price Prediction")
st.info("👈 Enter a stock ticker and click **Run Prediction** to start")

# -------------------------------
# Sidebar inputs
# -------------------------------
st.sidebar.header("🔎 Stock Settings")

ticker = st.sidebar.text_input(
    "Stock Ticker (e.g. AAPL, NVDA, TSLA)",
    value="NVDA"
).upper().strip()

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2022-01-01")
)

run_button = st.sidebar.button("▶ Run Prediction")

# -------------------------------
# Main logic
# -------------------------------
if run_button:
    with st.spinner("Downloading and processing data..."):

        # 1. Download data safely
        try:
            df = yf.download(ticker, start=start_date, progress=False)
        except Exception:
            st.error("❌ Error downloading stock data.")
            st.stop()

        if df is None or df.empty:
            st.error("❌ No data found. Please check the ticker symbol.")
            st.stop()

        df = df[["Close"]].dropna()

        # 2. Feature engineering
        df["Close_lag_1"] = df["Close"].shift(1)
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)

        if len(df) < 50:
            st.error("❌ Not enough data to train the model.")
            st.stop()

        # 3. Train/Test split (time-series safe)
        split_ratio = 0.8
        split_index = int(len(df) * split_ratio)

        train = df.iloc[:split_index]
        test = df.iloc[split_index:]

        X_train = train[["Close_lag_1", "MA_5"]]
        y_train = train["Target"]
        X_test = test[["Close_lag_1", "MA_5"]]
        y_test = test["Target"]

        # 4. Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 5. Predictions (historical)
        y_pred = model.predict(X_test)

        # 6. Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # -------------------------------
        # 7. One-week future prediction
        # -------------------------------
        last_close = df["Close"].iloc[-1]
        last_ma5 = df["Close"].iloc[-5:].mean()

        future_predictions = []
        future_dates = pd.date_range(
            start=df.index[-1] + pd.Timedelta(days=1),
            periods=7,
            freq="B"  # business days only
        )

        for _ in range(7):
            pred = model.predict([[last_close, last_ma5]])[0]
            future_predictions.append(pred)
            last_close = pred
            last_ma5 = (last_ma5 * 4 + pred) / 5

        future_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Close": future_predictions
        })

    # -------------------------------
    # Display results
    # -------------------------------
    st.success("✅ Prediction completed successfully!")

    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")

    # -------------------------------
    # Historical prediction plot
    # -------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.index, y_test.values, label="Actual Price")
    ax.plot(y_test.index, y_pred, label="Predicted Price")
    ax.set_title(f"{ticker} – Historical Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # Historical results table
    # -------------------------------
    result_df = pd.DataFrame({
        "Date": y_test.index,
        "Actual Close": y_test.values,
        "Predicted Close": y_pred
    })

    st.subheader("📊 Historical Prediction (Last 20 Days)")
    st.dataframe(result_df.tail(20), use_container_width=True)

    # -------------------------------
    # Future prediction section
    # -------------------------------
    st.subheader("🔮 1-Week Future Price Prediction")

    st.warning(
        "⚠️ Forecasts are for educational purposes only and should not be used for trading."
    )

    st.dataframe(future_df, use_container_width=True)

    # -------------------------------
    # Future prediction plot
    # -------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(df.index[-60:], df["Close"].iloc[-60:], label="Recent Actual Price")
    ax2.plot(
        future_df["Date"],
        future_df["Predicted Close"],
        marker="o",
        linestyle="--",
        label="1-Week Forecast"
    )
    ax2.set_title(f"{ticker} – 1 Week Future Forecast")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2)

    # -------------------------------
    # Download CSV (historical)
    # -------------------------------
    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Historical Predictions CSV",
        data=csv,
        file_name=f"predictions_{ticker}.csv",
        mime="text/csv"
    )
