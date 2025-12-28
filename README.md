# 📈 Stock Price Forecasting using LSTM & XGBoost

An **end-to-end time series forecasting project** that predicts future stock prices using **LSTM (Deep Learning)** and **XGBoost (Machine Learning)**.  
The project includes **EDA, feature engineering, model evaluation, forecasting, comparison, and deployment** using **Streamlit**.

---

## 🔍 Project Overview

This project demonstrates a **complete data science workflow** applied to stock price forecasting:

- 📊 Exploratory Data Analysis (EDA)
- 🔧 Feature Engineering
- 🤖 Model Building (LSTM & XGBoost)
- 📈 Future Price Forecasting
- 📊 Model Evaluation (Returns & Prices)
- ⚖️ Model Comparison
- 🌐 Deployment using Streamlit & GitHub

The application is **interactive** and designed for:
- Academic projects
- Interviews & viva
- Data science portfolios

---

## 🧠 Models Used

### 🔹 LSTM (Long Short-Term Memory)
- Captures **long-term temporal dependencies**
- Uses **sliding windows of log returns**
- Suitable for sequential time-series data

### 🔹 XGBoost
- Tree-based ensemble model
- Uses **engineered lag features and rolling statistics**
- Effective for short-term and structured patterns

---

## 🔧 Feature Engineering

### Global Features
- Log returns (to stabilize variance)
- 30-day rolling volatility

### LSTM-Specific Features
- Sequential windows of log returns
- Lookback window of 120 days

### XGBoost-Specific Features
- Lagged returns (1, 5, 10 days)
- Rolling mean and rolling standard deviation

---

## 📊 Model Evaluation

### Metrics Used
- **Primary Metrics:** RMSE & MAE on **log returns**
- **Secondary Metrics:** RMSE & MAE on **reconstructed prices**

📌 Log returns are used as the primary evaluation metric to reduce scale bias and avoid error compounding.

The application dynamically identifies and explains **which model performs better** based on return-based accuracy.

---

## 📈 Visualizations Included

- Closing price trend (from 2019)
- Trading volume analysis
- Log return behavior
- Volatility analysis
- LSTM forecast visualization
- XGBoost forecast visualization
- Historical + forecast comparison
- Forecast-only model comparison
- Feature importance (XGBoost)
- Forecast value matrices (USD & INR)

---
