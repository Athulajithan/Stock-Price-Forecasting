import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Stock Price Forecasting", layout="wide")

st.title("📊 Stock Price Forecasting using LSTM & XGBoost")
st.markdown("""
This project demonstrates **end-to-end time-series forecasting**
using historical stock price data, feature engineering,
deep learning (LSTM), and machine learning (XGBoost).

📌 **Accuracy is evaluated on both log returns (primary) and prices (secondary).**
""")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AAPL.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df.set_index("Date", inplace=True)
    return df

df = load_data()

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    lstm_model = load_model("models/lstm_model.h5", compile=False)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    return lstm_model, scaler, xgb_model

lstm_model, scaler, xgb_model = load_models()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Settings")

forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

show_eda = st.sidebar.checkbox("Show EDA", True)
show_lstm_forecast = st.sidebar.checkbox("Show LSTM Forecast", True)
show_xgb_forecast = st.sidebar.checkbox("Show XGBoost Forecast", True)

st.sidebar.markdown("---")
st.sidebar.info(
    "ℹ️ Comparison graphs are shown **only when both models are enabled**."
)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df_feat = df.copy()
df_feat["log_return"] = np.log(df_feat["Adj Close"]).diff()
df_feat["rolling_std_30"] = df_feat["Adj Close"].rolling(30).std()
df_feat.dropna(inplace=True)

df_2019 = df_feat[df_feat.index >= "2019-01-01"]

# ==========================================================
# 📊 EDA (UNCHANGED)
# ==========================================================
if show_eda:
    st.header("📈 Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(df_2019["Adj Close"])
        ax.set_title("Closing Price (From 2019)")
        ax.grid(True)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(df_2019["Volume"], color="orange")
        ax.set_title("Trading Volume")
        ax.grid(True)
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots()
        ax.plot(df_2019["log_return"])
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title("Daily Log Returns")
        ax.grid(True)
        st.pyplot(fig)

    with col4:
        fig, ax = plt.subplots()
        ax.plot(df_2019["rolling_std_30"], color="purple")
        ax.set_title("30-Day Rolling Volatility")
        ax.grid(True)
        st.pyplot(fig)

# ==========================================================
# 🤖 LSTM MODEL
# ==========================================================
LOOKBACK = 120
scaled = scaler.transform(df_feat[["log_return"]])

X_lstm, y_lstm = [], []
for i in range(LOOKBACK, len(scaled)):
    X_lstm.append(scaled[i-LOOKBACK:i, 0])
    y_lstm.append(scaled[i, 0])

X_lstm = np.array(X_lstm).reshape(-1, LOOKBACK, 1)
y_lstm = np.array(y_lstm)

split = int(len(X_lstm) * 0.8)
X_test = X_lstm[split:]
y_test = y_lstm[split:]

y_pred = lstm_model.predict(X_test, verbose=0)

y_test_ret = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_ret = scaler.inverse_transform(y_pred)

lstm_rmse_ret = np.sqrt(mean_squared_error(y_test_ret, y_pred_ret))
lstm_mae_ret = mean_absolute_error(y_test_ret, y_pred_ret)

actual_price = df_feat["Adj Close"].iloc[-len(y_test_ret):].values
pred_price = actual_price[0] * np.exp(np.cumsum(y_pred_ret.flatten()))

lstm_rmse_price = np.sqrt(mean_squared_error(actual_price, pred_price))
lstm_mae_price = mean_absolute_error(actual_price, pred_price)

if show_lstm_forecast:
    st.markdown("---")
    st.header("🤖 LSTM Model")

    st.subheader("📊 LSTM Accuracy")
    st.metric("RMSE (Returns)", round(lstm_rmse_ret, 6))
    st.metric("MAE (Returns)", round(lstm_mae_ret, 6))
    st.metric("RMSE (Price)", round(lstm_rmse_price, 2))
    st.metric("MAE (Price)", round(lstm_mae_price, 2))

    last_seq = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    future_scaled = []

    for _ in range(forecast_days):
        p = lstm_model.predict(last_seq, verbose=0)[0][0]
        future_scaled.append(p)
        last_seq = np.append(last_seq[:, 1:, :], [[[p]]], axis=1)

    future_returns = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    )

    last_price = df_feat["Adj Close"].iloc[-1]
    future_price_lstm = last_price * np.exp(np.cumsum(future_returns))

    dates = pd.bdate_range(
        start=df_feat.index[-1] + timedelta(days=1),
        periods=forecast_days
    )

    lstm_df = pd.DataFrame({
        "LSTM_Forecast_USD": future_price_lstm.flatten(),
        "LSTM_Forecast_INR": future_price_lstm.flatten() * 83
    }, index=dates)

    st.subheader("📈 LSTM Forecast Graph")
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df_2019["Adj Close"], label="Historical")
    ax.plot(lstm_df.index, lstm_df["LSTM_Forecast_USD"], "--", label="LSTM Forecast")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("📋 LSTM Forecast Matrix")
    st.dataframe(lstm_df)

# ==========================================================
# 🌲 XGBOOST MODEL
# ==========================================================
df_xgb = df_feat.copy()
df_xgb["lag_1"] = df_xgb["log_return"].shift(1)
df_xgb["lag_5"] = df_xgb["log_return"].shift(5)
df_xgb["lag_10"] = df_xgb["log_return"].shift(10)
df_xgb["roll_mean_5"] = df_xgb["log_return"].rolling(5).mean()
df_xgb["roll_std_5"] = df_xgb["log_return"].rolling(5).std()
df_xgb.dropna(inplace=True)

features = ["lag_1","lag_5","lag_10","roll_mean_5","roll_std_5"]
X = df_xgb[features]
y = df_xgb["log_return"]

split = int(len(X) * 0.8)
X_test = X.iloc[split:]
y_test = y.iloc[split:]

y_pred = xgb_model.predict(X_test)

xgb_rmse_ret = np.sqrt(mean_squared_error(y_test, y_pred))
xgb_mae_ret = mean_absolute_error(y_test, y_pred)

actual_price = df_feat["Adj Close"].iloc[-len(y_test):].values
pred_price = actual_price[0] * np.exp(np.cumsum(y_pred))

xgb_rmse_price = np.sqrt(mean_squared_error(actual_price, pred_price))
xgb_mae_price = mean_absolute_error(actual_price, pred_price)

if show_xgb_forecast:
    st.markdown("---")
    st.header("🌲 XGBoost Model")

    st.subheader("📊 XGBoost Accuracy")
    st.metric("RMSE (Returns)", round(xgb_rmse_ret, 6))
    st.metric("MAE (Returns)", round(xgb_mae_ret, 6))
    st.metric("RMSE (Price)", round(xgb_rmse_price, 2))
    st.metric("MAE (Price)", round(xgb_mae_price, 2))

    history = df_xgb.copy()
    future_returns = []

    for _ in range(forecast_days):
        row = history.iloc[-1]
        X_next = np.array([[row[f] for f in features]])
        p = xgb_model.predict(X_next)[0]
        future_returns.append(p)
        history = pd.concat(
            [history, pd.DataFrame({"log_return":[p]})],
            ignore_index=True
        )

    future_price_xgb = last_price * np.exp(np.cumsum(future_returns))

    xgb_df = pd.DataFrame({
        "XGBoost_Forecast_USD": future_price_xgb,
        "XGBoost_Forecast_INR": future_price_xgb * 83
    }, index=dates)

    st.subheader("📈 XGBoost Forecast Graph")
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(df_2019["Adj Close"], label="Historical")
    ax.plot(xgb_df.index, xgb_df["XGBoost_Forecast_USD"], "--", label="XGBoost Forecast")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("📋 XGBoost Forecast Matrix")
    st.dataframe(xgb_df)

# ==========================================================
# 🔀 COMPARISON (AUTO-DISABLED)
# ==========================================================
st.markdown("---")
st.header("🔀 Model Comparison")

# --------------------------------------------------
# Show comparison ONLY if both models are enabled
# --------------------------------------------------
if show_lstm_forecast and show_xgb_forecast:

    # ==============================
    # 1️⃣ Historical + Forecast Comparison
    # ==============================
    st.subheader("📊 Historical + Forecast Comparison")

    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(
        df_2019["Adj Close"],
        label="Historical Price",
        color="blue"
    )
    ax.plot(
        lstm_df.index,
        lstm_df["LSTM_Forecast_USD"],
        "--",
        label="LSTM Forecast",
        color="green"
    )
    ax.plot(
        xgb_df.index,
        xgb_df["XGBoost_Forecast_USD"],
        "--",
        label="XGBoost Forecast",
        color="orange"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ==============================
    # 2️⃣ Forecast-Only Comparison
    # ==============================
    st.subheader("📈 Forecast-Only Comparison (No Historical Prices)")

    fig, ax = plt.subplots(figsize=(14,6))

    ax.plot(
        lstm_df.index,
        lstm_df["LSTM_Forecast_USD"],
        linestyle="--",
        marker="o",
        label="LSTM Forecast",
        color="green"
    )

    ax.plot(
        xgb_df.index,
        xgb_df["XGBoost_Forecast_USD"],
        linestyle="--",
        marker="s",
        label="XGBoost Forecast",
        color="orange"
    )

    ax.set_title("LSTM vs XGBoost – Forecast Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

# --------------------------------------------------
# If only one model is enabled
# --------------------------------------------------
elif show_lstm_forecast or show_xgb_forecast:
    st.info(
        "ℹ️ Enable **both LSTM and XGBoost forecasts** from the sidebar "
        "to view comparison graphs."
    )

# --------------------------------------------------
# If both models are disabled
# --------------------------------------------------
else:
    st.warning(
        "⚠️ Both models are disabled. Enable at least one model to view results."
    )


# -----------------------------
# SMART MODEL EVALUATION FOOTER
# -----------------------------
st.markdown("---")
st.header("🏁 Model Evaluation Summary")

# Show footer ONLY when both models are enabled
if show_lstm_forecast and show_xgb_forecast:

    # Decide best model based on RETURN RMSE (primary metric)
    if lstm_rmse_ret < xgb_rmse_ret:
        best_model = "LSTM"
        reason = (
            "LSTM captures long-term temporal dependencies in time-series data more effectively. "
            "It shows lower error on log returns, indicating better generalization for sequential patterns."
        )
    else:
        best_model = "XGBoost"
        reason = (
            "XGBoost performs better due to strong feature engineering with lagged variables and rolling statistics. "
            "It achieves lower return prediction error, making it more reliable for short-term forecasting."
        )

    st.success(f"✅ **Best Performing Model: {best_model}**")

    st.markdown("### 📌 Why this model performed better:")
    st.markdown(f"- {reason}")

    st.markdown("### 📊 Evaluation Criteria Used:")
    st.markdown("""
    - **Primary Metric:** RMSE & MAE on **log returns**  
    - **Secondary Metric:** RMSE & MAE on **reconstructed prices**  
    - Log returns are preferred because they stabilize variance and improve stationarity.
    """)

    st.markdown("""

    """)

# If only one model is enabled
elif show_lstm_forecast or show_xgb_forecast:
    st.info(
        "ℹ️ Enable **both LSTM and XGBoost models** to view the comparative evaluation summary."
    )

# If both models are disabled
else:
    st.warning(
        "⚠️ Model evaluation summary is unavailable because both models are disabled."
    )
