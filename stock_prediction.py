# Beginner-friendly stock price prediction
# Predict next day's Close price using Linear Regression

# Requirements:
# pip install yfinance pandas numpy scikit-learn matplotlib

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -------------------------------
# 1. Download stock data
# -------------------------------
ticker = "NVDA"
start_date = "2018-01-01"
end_date = None   # None = today

print("Downloading data...")
df = yf.download(ticker, start=start_date, end=end_date)

df = df[["Close"]]
df.dropna(inplace=True)


# -------------------------------
# 2. Create simple features
# -------------------------------

# Yesterday's close
df["Close_lag_1"] = df["Close"].shift(1)

# 5-day moving average
df["MA_5"] = df["Close"].rolling(5).mean()

# Target: tomorrow's close
df["Target"] = df["Close"].shift(-1)

df.dropna(inplace=True)


# -------------------------------
# 3. Train / Test split
# -------------------------------
split_ratio = 0.8
split_index = int(len(df) * split_ratio)

train = df.iloc[:split_index]
test = df.iloc[split_index:]

X_train = train[["Close_lag_1", "MA_5"]]
y_train = train["Target"]

X_test = test[["Close_lag_1", "MA_5"]]
y_test = test["Target"]


# -------------------------------
# 4. Train model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)


# -------------------------------
# 5. Make predictions
# -------------------------------
y_pred = model.predict(X_test)


# -------------------------------
# 6. Evaluate model
# -------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


# -------------------------------
# 7. Save predictions to CSV
# -------------------------------
os.makedirs("outputs", exist_ok=True)

pred_df = pd.DataFrame({
    "Date": y_test.index,
    "Actual_Close": y_test.values,
    "Predicted_Close": y_pred
})

csv_path = f"outputs/predictions_{ticker}.csv"
pred_df.to_csv(csv_path, index=False)

print(f"\nPrediction CSV saved to: {csv_path}")


# -------------------------------
# 8. Plot & save graph
# -------------------------------
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label="Actual Price")
plt.plot(y_test.index, y_pred, label="Predicted Price")
plt.title(f"{ticker} Stock Price Prediction (Beginner Model)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()

plot_path = f"outputs/prediction_plot_{ticker}.png"
plt.savefig(plot_path)
plt.show()

print(f"Prediction plot saved to: {plot_path}")