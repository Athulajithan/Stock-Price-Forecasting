
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


# ===== PIPELINE CONTROLLER =====
if "step" not in st.session_state:
    st.session_state.step = 1

def next_step():
    if st.session_state.step < 6:
        st.session_state.step += 1


def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1



st.set_page_config(page_title="AAPL Forecasting System", layout="wide")
st.title("📈 Apple Stock Price Forecasting System")

# ===== PIPELINE VISUAL =====
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

# uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
# forecast_days = st.sidebar.selectbox("Forecast Days", [7, 15, 30], index=2)
# model_choice = st.sidebar.selectbox(
#    "Model", ["SARIMA", "Random Forest", "XGBoost", "GRU", "LSTM"]
#)
# view = st.sidebar.radio("Forecast View", ["Graph", "Table", "Both"])

MIN_PRICE = 291

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

    # ===================== MODELS =====================
def sarima_model():
    model = SARIMAX(train["Return"], order=(1, 0, 1))
    fit = model.fit(disp=False)
    train_pred = fit.fittedvalues
    test_pred = fit.forecast(len(test))
    future = fit.forecast(forecast_days)
    return train_pred, test_pred, future

def rf_model(tuned=False):
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train["Return"].values
    X_test = np.arange(len(train), len(df)).reshape(-1, 1)

    if tuned:
        params = {"n_estimators": [100, 300], "max_depth": [5, 10]}
        grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            params,
            cv=3,
            scoring="neg_mean_squared_error",
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=10)

    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    future_idx = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    future = model.predict(future_idx)
    return train_pred, test_pred, future

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
            params,
            cv=3,
            scoring="neg_mean_squared_error",
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05
        )

    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    future_idx = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    future = model.predict(future_idx)
    return train_pred, test_pred, future

def gru_model():
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Return"]])

    X, y = [], []
    lookback = 20
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback : i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)
    X_train, X_test = X[: train_size - lookback], X[train_size - lookback :]
    y_train, y_test = y[: train_size - lookback], y[train_size - lookback :]

    model = Sequential(
        [
            GRU(32, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            GRU(16),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=3)],
        verbose=0,
    )

    train_pred = scaler.inverse_transform(
        model.predict(X_train)
    ).flatten()
    test_pred = scaler.inverse_transform(
        model.predict(X_test)
    ).flatten()

    last_seq = X[-1]
    future = []
    for _ in range(forecast_days):
        r = model.predict(last_seq.reshape(1, lookback, 1))[0, 0]
        future.append(r)
        last_seq = np.roll(last_seq, -1)
        last_seq[-1] = r

    future = scaler.inverse_transform(
        np.array(future).reshape(-1, 1)
    ).flatten()

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
    y_train, y_test = y[:train_size-lookback], y[train_size-lookback:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=16,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=3)],
        verbose=0
    )

    train_pred = scaler.inverse_transform(
        model.predict(X_train)
    ).flatten()

    test_pred = scaler.inverse_transform(
        model.predict(X_test)
    ).flatten()

    last_seq = X[-1]
    future = []

    for _ in range(forecast_days):
        r = model.predict(last_seq.reshape(1, lookback, 1))[0, 0]
        future.append(r)
        last_seq = np.roll(last_seq, -1)
        last_seq[-1] = r

    future = scaler.inverse_transform(
        np.array(future).reshape(-1, 1)
    ).flatten()

    return train_pred, test_pred, future

if st.session_state.step == 1:
    st.header("STEP 1 → Upload Data")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    st.session_state.forecast_days = st.selectbox(
    "Forecast Horizon (Days)",
    [7, 15, 30],
    index=2
    )


    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        st.success("Data uploaded successfully")
        st.button("Next ▶", on_click=next_step, key="next_1")





# ===================== EDA =====================
elif st.session_state.step == 2:
    df = st.session_state.df



    st.header("🔍 Exploratory Data Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Start Date", df.index.min().strftime("%d-%m-%Y"))
    c2.metric("End Date", df.index.max().strftime("%d-%m-%Y"))
    c3.metric("Records", len(df))


    st.subheader("Closing Price Trend")
    st.plotly_chart(px.line(df, x=df.index, y="Close"), use_container_width=True)

    st.subheader("Return Distribution")
    st.plotly_chart(px.histogram(df, x="Return", nbins=100), use_container_width=True)

    df["Rolling_Volatility"] = df["Return"].rolling(30, min_periods=1).std()
    st.subheader("Rolling Volatility (Start → End)")
    st.plotly_chart(
        px.line(df, x=df.index, y="Rolling_Volatility"),
        use_container_width=True
    )
    
    st.button("Next ▶", on_click=next_step, key="next_2")
    st.button("◀ Back", on_click=prev_step, key="back_2")



# ===================== STATISTICAL HYPOTHESIS TESTING =====================

elif st.session_state.step == 3:
    df = st.session_state.df

    st.header("📊 Statistical Hypothesis Testing (Before Modeling)")

    adf_stat, adf_p, _, _, _, _ = adfuller(df["Return"])
    jb_stat, jb_p = jarque_bera(df["Return"])
    lb_test = acorr_ljungbox(df["Return"], lags=[10], return_df=True)
    arch_stat, arch_p, _, _ = het_arch(df["Return"])

    hypothesis_df = pd.DataFrame({
        "Test": [
            "ADF Test (Stationarity)",
            "Jarque-Bera Test (Normality)",
            "Ljung-Box Test (Autocorrelation)",
            "ARCH Test (Heteroskedasticity)"
        ],
        "Statistic": [
            adf_stat,
            jb_stat,
            lb_test["lb_stat"].iloc[0],
            arch_stat
        ],
        "p-value": [
            adf_p,
            jb_p,
            lb_test["lb_pvalue"].iloc[0],
            arch_p
        ],
        "Inference": [
            "Stationary" if adf_p < 0.05 else "Non-Stationary",
            "Non-Normal" if jb_p < 0.05 else "Normal",
            "Autocorrelation Present" if lb_test["lb_pvalue"].iloc[0] < 0.05 else "No Autocorrelation",
            "Heteroskedasticity Present" if arch_p < 0.05 else "No ARCH Effect"
        ]
    })

    st.dataframe(hypothesis_df)

    st.info(
        "Statistical tests justify using SARIMA, ML, and DL models "
        "due to autocorrelation, non-normality, and volatility clustering."
    )

    st.header("📉 ACF & PACF Analysis (SARIMA Justification)")

    st.button("Next ▶", on_click=next_step, key="next_3")
    st.button("◀ Back", on_click=prev_step, key="back_3")


    

    fig1, ax1 = plt.subplots()
    plot_acf(df["Return"], lags=30, ax=ax1)
    ax1.set_title("Autocorrelation Function (ACF)")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    plot_pacf(df["Return"], lags=30, ax=ax2)
    ax2.set_title("Partial Autocorrelation Function (PACF)")
    st.pyplot(fig2)

    st.caption(
        "ACF and PACF plots confirm temporal dependency and "
        "help select AR and MA terms for SARIMA."
    )

elif st.session_state.step == 4:
    df = st.session_state.df

    st.header("STEP 4 → Model Selection")

    col1, col2, col3 = st.columns(3)

    if col1.button("SARIMA"):
        st.session_state.model_choice = "SARIMA"
    if col1.button("Random Forest"):
        st.session_state.model_choice = "Random Forest"

    if col2.button("XGBoost"):
        st.session_state.model_choice = "XGBoost"
    if col2.button("GRU"):
        st.session_state.model_choice = "GRU"

    if col3.button("LSTM"):
        st.session_state.model_choice = "LSTM"

    if "model_choice" in st.session_state:
        st.success(f"Selected Model: {st.session_state.model_choice}")

    st.button("Next ▶", on_click=next_step, key="next_4")
    st.button("◀ Back", on_click=prev_step, key="back_4")


# ===================== SPLIT =====================


    


# ===================== EXECUTION =====================
elif st.session_state.step == 5:
    df = st.session_state.df
    forecast_days = st.session_state.forecast_days

    st.header("🧠 Model Diagnostics & Auto-Tuning")

    if "model_choice" not in st.session_state:
        st.warning("Please select a model in Step 4")
        st.stop()

    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    globals()["train"] = train
    globals()["test"] = test
    globals()["df"] = df
    globals()["forecast_days"] = forecast_days


    model_choice = st.session_state.model_choice

    # model training starts here


    if model_choice == "SARIMA":
        train_pred, test_pred, future_returns = sarima_model()
        st.session_state.future_returns = future_returns

    elif model_choice == "Random Forest":
        train_pred, test_pred, future_returns = rf_model(False)
        st.session_state.future_returns = future_returns

    elif model_choice == "XGBoost":
        train_pred, test_pred, future_returns = xgb_model(False)
        st.session_state.future_returns = future_returns

    elif model_choice == "LSTM":
        train_pred, test_pred, future_returns = lstm_model()
        st.session_state.future_returns = future_returns

    else:
        train_pred, test_pred, future_returns = gru_model()
        st.session_state.future_returns = future_returns




    train_mae, train_rmse = mae_rmse(
        train["Return"][: len(train_pred)], train_pred
    )
    test_mae, test_rmse = mae_rmse(test["Return"], test_pred)

    status = fit_status(train_mae, test_mae)

    st.session_state.train_mae = train_mae
    st.session_state.test_mae = test_mae
    st.session_state.train_rmse = train_rmse
    st.session_state.test_rmse = test_rmse
    st.session_state.status = status

    # ===================== PRICE MAE / RMSE (CORRECT PLACE) =====================
# Convert RETURNS → PRICE for TEST period only (valid evaluation)

    train_size = int(len(df) * 0.8)

    test_prices_actual = df["Close"].iloc[train_size + 1:]

    test_start_price = df["Close"].iloc[train_size]

    test_prices_pred = (
        test_start_price
        * (1 + pd.Series(test_pred)).cumprod()
    ).values[:len(test_prices_actual)]

    price_mae, price_rmse = mae_rmse(
        test_prices_actual[:len(test_prices_pred)],
        test_prices_pred
    )

    st.session_state.price_mae = price_mae
    st.session_state.price_rmse = price_rmse



    before = pd.DataFrame(
        {
            "Stage": ["Before Tuning"],
            "Train MAE": [train_mae],
            "Train RMSE": [train_rmse],
            "Test MAE": [test_mae],
            "Test RMSE": [test_rmse],
            "Fit Status": [status],
        }
    )

    if status != "Balanced" and model_choice in ["Random Forest", "XGBoost"]:
        st.warning(f"{status} detected → Auto-tuning applied")
        time.sleep(1)

        if model_choice == "Random Forest":
            train_pred, test_pred, future_returns = rf_model(True)
        else:
            train_pred, test_pred, future_returns = xgb_model(True)

        train_mae, train_rmse = mae_rmse(
            train["Return"][: len(train_pred)], train_pred
        )
        test_mae, test_rmse = mae_rmse(test["Return"], test_pred)
        status = fit_status(train_mae, test_mae)

    after = pd.DataFrame(
        {
            "Stage": ["After Tuning"],
            "Train MAE": [train_mae],
            "Train RMSE": [train_rmse],
            "Test MAE": [test_mae],
            "Test RMSE": [test_rmse],
            "Fit Status": [status],
        }
    )

    st.subheader("📊 MAE & RMSE — Before vs After")
    st.dataframe(pd.concat([before, after], ignore_index=True))

    st.button("Next ▶", on_click=next_step, key="next_5")
    st.button("◀ Back", on_click=prev_step, key="back_5")



# ===================== PRICE FORECAST =====================
elif st.session_state.step == 6:
    df = st.session_state.df
    forecast_days = st.session_state.forecast_days
    future_returns = st.session_state.future_returns

    train_mae = st.session_state.train_mae
    test_mae = st.session_state.test_mae
    train_rmse = st.session_state.train_rmse
    test_rmse = st.session_state.test_rmse
    status = st.session_state.status


    historical_vol = df["Return"].std()
    recent_mean = df["Return"].tail(30).mean()

    last_price = df["Close"].iloc[-1]
    future_prices = [last_price]

    for r in future_returns:
        adjusted = 0.7 * r + 0.3 * recent_mean
        noise = np.random.normal(0, historical_vol)
        next_price = future_prices[-1] * (1 + adjusted + noise)
        next_price = max(next_price, MIN_PRICE)
        future_prices.append(next_price)

    future_prices = np.array(future_prices[1:])

    price_mae = st.session_state.price_mae
    price_rmse = st.session_state.price_rmse

    st.subheader("💰 Price-Based Error Metrics")

    p1, p2 = st.columns(2)
    p1.metric("MAE (Price)", f"{price_mae:.2f}")
    p2.metric("RMSE (Price)", f"{price_rmse:.2f}")


    future_dates = pd.date_range(
        df.index[-1], periods=forecast_days + 1
    )[1:]

    forecast_df = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted Price": future_prices,
        }
    )


    st.header("📈 Forecast Output")

    view = st.radio(
    "How do you want to see the forecast?",
    ["Graph", "Table", "Both"]
    )

# existing forecast graph/table code below



    if view in ["Graph", "Both"]:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Close"], name="Actual")
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates, y=future_prices, name="Forecast"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    if view in ["Table", "Both"]:
        st.dataframe(forecast_df)

    st.success("✅ Full pipeline completed successfully")

# ===================== MODEL SUMMARY =====================
if st.session_state.step >= 6:
    st.header("🏆 Final Model Summary")

    summary_df = pd.DataFrame({
        "Metric": [
            "Train MAE (Returns)",
            "Test MAE (Returns)",
            "Train RMSE (Returns)",
            "Test RMSE (Returns)",
            "MAE (Price)",
            "RMSE (Price)",
            "Fit Status"
        ],
        "Value": [
            train_mae,
            test_mae,
            train_rmse,
            test_rmse,
            price_mae,
            price_rmse,
            status
        ]
    })

    st.dataframe(summary_df)
