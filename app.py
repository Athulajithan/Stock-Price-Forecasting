import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
import time

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import LSTM, Dropout

warnings.filterwarnings("ignore")

# ======================================================
# STREAMLIT CONFIG
# ======================================================
st.set_page_config(page_title="AAPL Forecasting System", layout="wide")
st.title("📈 Apple Stock Price Forecasting System")

MIN_PRICE = 291

# ======================================================
# PIPELINE CONTROLLER
# ======================================================
if "step" not in st.session_state:
    st.session_state.step = 1

def next_step():
    if st.session_state.step < 6:
        st.session_state.step += 1

def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1

# ======================================================
# SIDEBAR (GLOBAL CONTROLS)
# ======================================================
st.sidebar.header("⚙️ Control Panel")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

st.session_state.forecast_days = st.sidebar.selectbox(
    "Forecast Horizon (Days)",
    [7, 15, 30],
    index=2
)

st.session_state.model_choice = st.sidebar.selectbox(
    "Select Model",
    ["SARIMA", "Random Forest", "XGBoost", "GRU", "LSTM"]
)

view = st.sidebar.radio(
    "Forecast View",
    ["Graph", "Table", "Both"]
)

# ======================================================
# TOP NAVIGATION BAR (EVERY STEP)
# ======================================================
def navigation_bar():
    c1, c2, c3 = st.columns([1, 6, 1])
    with c1:
        if st.button("⬅ Back"):
            prev_step()
    with c3:
        if st.button("Next ➡"):
            next_step()

# ======================================================
# PIPELINE VISUAL
# ======================================================
steps = [
    "1️⃣ Upload",
    "2️⃣ EDA",
    "3️⃣ Diagnostics",
    "4️⃣ Model",
    "5️⃣ Training",
    "6️⃣ Forecast"
]

current = st.session_state.step - 1
progress_value = min(1.0, max(0.0, current / (len(steps) - 1)))
st.progress(progress_value)

cols = st.columns(len(steps))
for i, col in enumerate(cols):
    if i == current:
        col.markdown(f"**➡ {steps[i]}**")
    else:
        col.markdown(steps[i])

st.markdown("1 Upload → 2 EDA → 3 Diagnostics → 4 Model → 5 Training → 6 Forecast")

# ======================================================
# DATA LOADER
# ======================================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(
        df["Date"], dayfirst=True, format="mixed", errors="coerce"
    )
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    df = df[["Close"]]
    df.dropna(inplace=True)
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

def mae_rmse(y_true, y_pred):
    return (
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
    )

def fit_status(train_mae, test_mae):
    if train_mae < test_mae * 0.5:
        return "Overfitting"
    elif train_mae > test_mae:
        return "Underfitting"
    else:
        return "Balanced"

# ======================================================
# MODELS (UNCHANGED)
# ======================================================
def sarima_model():
    model = SARIMAX(train["Return"], order=(1, 0, 1))
    fit = model.fit(disp=False)
    return fit.fittedvalues, fit.forecast(len(test)), fit.forecast(forecast_days)

def rf_model(tuned=False):
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train["Return"].values
    X_test = np.arange(len(train), len(df)).reshape(-1, 1)

    if tuned:
        params = {"n_estimators": [100, 300], "max_depth": [5, 10]}
        grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            params, cv=3,
            scoring="neg_mean_squared_error",
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=10)

    model.fit(X_train, y_train)
    future_idx = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    return model.predict(X_train), model.predict(X_test), model.predict(future_idx)

def xgb_model(tuned=False):
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train["Return"].values
    X_test = np.arange(len(train), len(df)).reshape(-1, 1)

    if tuned:
        params = {
            "n_estimators": [200, 400],
            "max_depth": [4, 6],
            "learning_rate": [0.03, 0.05],
        }
        grid = GridSearchCV(
            XGBRegressor(objective="reg:squarederror"),
            params, cv=3,
            scoring="neg_mean_squared_error",
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05
        )

    model.fit(X_train, y_train)
    future_idx = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    return model.predict(X_train), model.predict(X_test), model.predict(future_idx)

def gru_model():
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Return"]])
    X, y = [], []
    lookback = 20

    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)
    X_train, X_test = X[:train_size-lookback], X[train_size-lookback:]
    y_train, y_test = y[:train_size-lookback], y[train_size-lookback:]

    model = Sequential([
        GRU(32, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        GRU(16),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train, y_train,
        epochs=20, batch_size=16,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=3)],
        verbose=0
    )

    train_pred = scaler.inverse_transform(model.predict(X_train)).flatten()
    test_pred = scaler.inverse_transform(model.predict(X_test)).flatten()

    last_seq = X[-1]
    future = []
    for _ in range(forecast_days):
        r = model.predict(last_seq.reshape(1, lookback, 1))[0, 0]
        future.append(r)
        last_seq = np.roll(last_seq, -1)
        last_seq[-1] = r

    future = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()
    return train_pred, test_pred, future

def lstm_model():
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Return"]])
    X, y = [], []
    lookback = 20

    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)
    X_train, X_test = X[:train_size-lookback], X[train_size-lookback:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train, y[:train_size-lookback],
        epochs=20, batch_size=16,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=3)],
        verbose=0
    )

    train_pred = scaler.inverse_transform(model.predict(X_train)).flatten()
    test_pred = scaler.inverse_transform(model.predict(X_test)).flatten()

    last_seq = X[-1]
    future = []
    for _ in range(forecast_days):
        r = model.predict(last_seq.reshape(1, lookback, 1))[0, 0]
        future.append(r)
        last_seq = np.roll(last_seq, -1)
        last_seq[-1] = r

    future = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()
    return train_pred, test_pred, future

# ======================================================
# STEP 1 – UPLOAD CHECK
# ======================================================
if st.session_state.step == 1:
    navigation_bar()
    st.header("STEP 1 → Upload Data")

    if not uploaded_file:
        st.warning("Please upload a CSV file from the sidebar.")
        st.stop()

    st.session_state.df = load_data(uploaded_file)
    st.success("Data uploaded successfully")

# ======================================================
# STEP 2 – EDA
# ======================================================
elif st.session_state.step == 2:
    navigation_bar()
    df = st.session_state.df

    st.header("🔍 Exploratory Data Analysis")
    st.plotly_chart(px.line(df, y="Close"), use_container_width=True)
    st.plotly_chart(px.histogram(df, x="Return", nbins=100), use_container_width=True)

# ======================================================
# STEP 3 – DIAGNOSTICS
# ======================================================
elif st.session_state.step == 3:
    navigation_bar()
    df = st.session_state.df

    st.header("📊 Statistical Hypothesis Testing")

    adf_p = adfuller(df["Return"])[1]
    jb_p = jarque_bera(df["Return"])[1]
    lb_p = acorr_ljungbox(df["Return"], lags=[10], return_df=True)["lb_pvalue"].iloc[0]

    st.write(f"ADF p-value: {adf_p:.4f}")
    st.write(f"Jarque-Bera p-value: {jb_p:.4f}")
    st.write(f"Ljung-Box p-value: {lb_p:.4f}")

# ======================================================
# STEP 4 – MODEL CONFIRM
# ======================================================
elif st.session_state.step == 4:
    navigation_bar()
    st.header("STEP 4 → Model Selection")
    st.success(f"Selected Model: {st.session_state.model_choice}")

# ======================================================
# STEP 5 – TRAINING
# ======================================================
elif st.session_state.step == 5:
    navigation_bar()
    df = st.session_state.df
    forecast_days = st.session_state.forecast_days

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    globals()["train"] = train
    globals()["test"] = test
    globals()["df"] = df

    model_choice = st.session_state.model_choice

    if model_choice == "SARIMA":
        train_pred, test_pred, future_returns = sarima_model()
    elif model_choice == "Random Forest":
        train_pred, test_pred, future_returns = rf_model()
    elif model_choice == "XGBoost":
        train_pred, test_pred, future_returns = xgb_model()
    elif model_choice == "LSTM":
        train_pred, test_pred, future_returns = lstm_model()
    else:
        train_pred, test_pred, future_returns = gru_model()

    st.session_state.future_returns = future_returns
    st.success("Model training completed")

# ======================================================
# STEP 6 – FORECAST
# ======================================================
elif st.session_state.step == 6:
    navigation_bar()
    df = st.session_state.df
    future_returns = st.session_state.future_returns
    forecast_days = st.session_state.forecast_days

    last_price = df["Close"].iloc[-1]
    future_prices = [last_price]

    for r in future_returns:
        future_prices.append(future_prices[-1] * (1 + r))

    future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1)[1:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Actual"))
    fig.add_trace(go.Scatter(x=future_dates, y=future_prices[1:], name="Forecast"))
    st.plotly_chart(fig, use_container_width=True)

    st.success("✅ Full pipeline completed successfully")
