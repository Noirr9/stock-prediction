# Beginner-friendly stock price prediction (Streamlit Version)

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("📈 Stock Price Prediction")
# st.write("Predict next day's closing price using a simple ML model")

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", "NVDA")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))

run_button = st.sidebar.button("Run Prediction")


# -------------------------------
# Run model when button is clicked
# -------------------------------
if run_button:

    st.info("Downloading stock data...")
    df = yf.download(ticker, start=start_date)

    if df.empty:
        st.error("No data found. Please check the ticker symbol.")
        st.stop()

    df = df[["Close"]]
    df.dropna(inplace=True)

    # -------------------------------
    # Feature engineering
    # -------------------------------
    df["Close_lag_1"] = df["Close"].shift(1)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)

    # -------------------------------
    # Train / Test split
    # -------------------------------
    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = train[["Close_lag_1", "MA_5"]]
    y_train = train["Target"]

    X_test = test[["Close_lag_1", "MA_5"]]
    y_test = test["Target"]

    # -------------------------------
    # Train model
    # -------------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)

    # -------------------------------
    # Predictions
    # -------------------------------
    y_pred = model.predict(X_test)

    # -------------------------------
    # Evaluation
    # -------------------------------
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("📊 Model Performance")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")

    # -------------------------------
    # Prediction DataFrame
    # -------------------------------
    pred_df = pd.DataFrame({
        "Date": y_test.index,
        "Actual Close": y_test.values,
        "Predicted Close": y_pred
    })

    st.subheader("📋 Prediction Table")
    st.dataframe(pred_df.tail(20))

    # -------------------------------
    # Plot
    # -------------------------------
    st.subheader("📉 Actual vs Predicted Prices")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.index, y_test.values, label="Actual Price")
    ax.plot(y_test.index, y_pred, label="Predicted Price")
    ax.set_title(f"{ticker} Stock Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

    # -------------------------------
    # Download CSV
    # -------------------------------
    st.subheader("⬇ Download Predictions")
    csv = pred_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"predictions_{ticker}.csv",
        mime="text/csv"
    )

else:
    st.warning("👈 Set parameters and click **Run Prediction**")
