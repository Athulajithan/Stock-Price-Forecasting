📈 Apple Stock Price Forecasting System

An end-to-end interactive time series forecasting application built using Streamlit to forecast Apple Inc. (AAPL) stock prices.

The project demonstrates a complete data science workflow, combining statistical models, machine learning, and deep learning, along with EDA, hypothesis testing, auto-tuning, model diagnostics, and price forecasting.

🔍 Project Overview

This project implements a step-by-step forecasting pipeline:

📤 Data upload and preprocessing

📊 Exploratory Data Analysis (EDA)

📉 Statistical hypothesis testing

🤖 Model selection (Statistical, ML & DL)

🧠 Model training with diagnostics & auto-tuning

📈 Future stock price forecasting

The application is fully interactive and suitable for:

Academic projects

Interviews & viva

Data science / data analyst portfolios

🔁 Forecasting Pipeline

The system follows a controlled 6-step pipeline:

Upload stock price data

Exploratory Data Analysis

Statistical diagnostics

Model selection

Model training & evaluation

Forecast generation

Each step must be completed sequentially to ensure correctness.

📂 Dataset Requirements

The uploaded CSV file must contain:

Column	Description
Date	Trading date
Close	Closing stock price

The system automatically:

Parses and cleans dates

Sorts data chronologically

Computes daily percentage returns

📊 Exploratory Data Analysis (EDA)

EDA includes:

Closing price trend visualization

Return distribution analysis

30-day rolling volatility

Dataset duration and record count

These insights help understand trend, volatility, and risk behavior.

📉 Statistical Hypothesis Testing

Before modeling, the following tests are performed:

ADF Test – Stationarity check

Jarque–Bera Test – Normality test

Ljung–Box Test – Autocorrelation detection

ARCH Test – Volatility clustering

ACF and PACF plots are used to justify SARIMA and learning-based models.

🤖 Models Used

The system supports five forecasting models:

🔹 SARIMA

Statistical time series model

Captures autocorrelation structure

🔹 Random Forest

Ensemble machine learning model

Captures nonlinear patterns

Supports auto hyperparameter tuning

🔹 XGBoost

Gradient boosting model

Strong performance on structured data

Auto-tuned using GridSearchCV

🔹 GRU

Deep learning recurrent neural network

Efficient for sequential time-series data

🔹 LSTM

Advanced recurrent neural network

Captures long-term temporal dependencies

Includes dropout for regularization

All models are trained on returns to ensure stationarity.

🧠 Model Training & Auto-Tuning

80% training / 20% testing split

Evaluation metrics:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Automatic detection of:

Overfitting

Underfitting

Auto hyperparameter tuning applied to:

Random Forest

XGBoost (only when required)

💰 Price Forecasting Logic

Forecasted returns are converted back into prices

Uses:

Last known closing price

Recent mean returns

Historical volatility

Noise stabilization

Minimum price constraint applied

Both return-based and price-based error metrics are reported.

📈 Forecast Output

Users can view forecasts as:

📊 Interactive line chart

📋 Tabular forecast values

📊 + 📋 Combined view

A final summary displays:

Train/Test MAE & RMSE (Returns)

MAE & RMSE (Prices)

Model fit status

🛠️ Tech Stack

Python

Streamlit

Pandas, NumPy

Plotly, Matplotlib

Scikit-learn

Statsmodels

XGBoost

TensorFlow / Keras
