# 📈 Apple Stock Price Forecasting System

An end-to-end **interactive time series forecasting application** built using **Streamlit** to forecast **Apple Inc. (AAPL)** stock prices.

This project demonstrates a **complete data science workflow** using **statistical models, machine learning, and deep learning**, along with Exploratory Data Analysis (EDA), hypothesis testing, auto-tuning, evaluation, price forecasting, and deployment.

---

## 🔗 Live Deployment

🚀 **Live Application (Streamlit Cloud):**  
👉 https://athul-stock-price-forecasting.streamlit.app/

The application is publicly accessible and allows users to upload datasets, train forecasting models, and generate future stock price predictions interactively without any local setup.

---

## 🔍 Project Overview

This project implements a structured, real-world forecasting pipeline:

- 📤 Data upload and preprocessing  
- 📊 Exploratory Data Analysis (EDA)  
- 📉 Statistical hypothesis testing  
- 🤖 Model selection  
- 🧠 Model training with diagnostics & auto-tuning  
- 📈 Future stock price forecasting  

The application is **interactive** and suitable for:

- Academic projects  
- Interviews & viva  
- Data science / data analyst portfolios  

---

## 🔁 Forecasting Pipeline

1. Upload stock price data  
2. Perform Exploratory Data Analysis  
3. Run statistical diagnostics  
4. Select forecasting model  
5. Train model with evaluation & auto-tuning  
6. Generate future forecasts  

⚠️ Each step must be completed sequentially to ensure valid modeling and reliable results.

---

## 📂 Dataset Requirements

The uploaded CSV file must contain the following columns:

| Column | Description |
|------|-------------|
| Date | Trading date |
| Close | Closing stock price |

### Automatic preprocessing:
- Cleans and parses dates  
- Sorts data chronologically  
- Computes daily percentage returns  
- Handles missing values  

---

## 📊 Exploratory Data Analysis (EDA)

EDA includes:
- Closing price trend visualization  
- Return distribution analysis  
- 30-day rolling volatility  
- Dataset duration and record count  

These analyses help understand trend behavior, risk, and market dynamics before modeling.

---

## 📉 Statistical Hypothesis Testing

The following tests are performed before modeling:

- **ADF Test** – Stationarity  
- **Jarque–Bera Test** – Normality  
- **Ljung–Box Test** – Autocorrelation  
- **ARCH Test** – Volatility clustering  

ACF and PACF diagnostics justify the use of **SARIMA, Machine Learning, and Deep Learning models**.

---

## 🤖 Forecasting Models

### 📌 SARIMA
- Classical statistical time-series model  
- Captures trend, seasonality, and autocorrelation  

### 📌 Random Forest
- Ensemble machine learning model  
- Detects nonlinear relationships  
- Supports automatic hyperparameter tuning  

### 📌 XGBoost
- Gradient boosting model  
- High performance on structured data  
- Auto-tuned using GridSearchCV  

### 📌 GRU
- Recurrent neural network  
- Efficient for sequential data  
- Faster training than LSTM  

### 📌 LSTM
- Advanced recurrent neural network  
- Captures long-term dependencies  
- Uses dropout for regularization  

📌 **All models are trained on returns** to ensure stationarity.

---

## 🧠 Model Training & Evaluation

- **Train/Test Split:** 80% / 20%  
- **Evaluation Metrics:**
  - MAE (Mean Absolute Error)  
  - RMSE (Root Mean Squared Error)  

The system automatically detects:
- Overfitting  
- Underfitting  

Auto hyperparameter tuning is applied to:
- Random Forest  
- XGBoost  

---

## 💰 Price Forecasting Logic

- Forecasted returns are converted back to prices using:
  - Last known closing price  
  - Mean returns  
  - Historical volatility  
- Noise stabilization applied  
- Minimum price constraint enforced  

Both **return-based** and **price-based** error metrics are reported.

---

## 📈 Forecast Output

Users can view results as:
- Interactive forecast line chart  
- Tabular forecast values  
- Combined visualization and table  

Final summary includes:
- Train/Test MAE & RMSE (Returns)  
- MAE & RMSE (Prices)  
- Model fit status  

---

## 🛠️ Tech Stack

- **Programming:** Python  
- **Web Application:** Streamlit  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Plotly, Matplotlib  
- **Statistical Modeling:** Statsmodels  
- **Machine Learning:** Scikit-learn  
- **Boosting:** XGBoost  
- **Deep Learning:** TensorFlow / Keras  

---

## 🚀 Deployment

- **Platform:** Streamlit Community Cloud  
- **Deployment Type:** Public Web Application  
- **Entry Point:** `app.py`  
- **Automatic Rebuilds:** Enabled on GitHub updates  

### Deployment Workflow
1. Push project to GitHub repository  
2. Define dependencies in `requirements.txt`  
3. Connect GitHub repository to Streamlit Cloud  
4. Deploy application using `app.py`  
5. Automatic redeployment on every code update  

---

## 📁 Project Structure

```text
apple-stock-forecasting/
│
├── app.py
├── requirements.txt
├── README.md

