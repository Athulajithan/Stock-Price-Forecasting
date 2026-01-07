# 📈 Apple Stock Price Forecasting System

An end-to-end **interactive time series forecasting application** built using **Streamlit** to forecast **Apple Inc. (AAPL)** stock prices.

This project demonstrates a **complete data science workflow** using **statistical models, machine learning, and deep learning**, along with EDA, hypothesis testing, auto-tuning, evaluation, and price forecasting.

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

Each step must be completed sequentially.

---

## 📂 Dataset Requirements

The uploaded CSV file must contain:

| Column | Description |
|------|-------------|
| Date | Trading date |
| Close | Closing stock price |

The system automatically:
- Cleans and parses dates
- Sorts data chronologically
- Computes daily percentage returns

---

## 📊 Exploratory Data Analysis (EDA)

EDA includes:
- Closing price trend visualization  
- Return distribution analysis  
- 30-day rolling volatility  
- Dataset duration and record count  

---

## 📉 Statistical Hypothesis Testing

The following tests are performed before modeling:

- ADF Test – Stationarity  
- Jarque–Bera Test – Normality  
- Ljung–Box Test – Autocorrelation  
- ARCH Test – Volatility clustering  

ACF and PACF plots justify the use of **SARIMA, ML, and DL models**.

---

## 🤖 Models Used

### SARIMA
- Statistical time-series model
- Captures autocorrelation structure

### Random Forest
- Ensemble machine learning model
- Detects nonlinear patterns
- Supports auto hyperparameter tuning

### XGBoost
- Gradient boosting model
- High-performance structured learning
- Auto-tuned using GridSearchCV

### GRU
- Recurrent neural network
- Efficient for sequential data

### LSTM
- Advanced recurrent neural network
- Captures long-term dependencies
- Uses dropout for regularization

All models are trained on **returns** to ensure stationarity.

---

## 🧠 Model Training & Evaluation

- 80% training / 20% testing split  
- Metrics used:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
- Automatic detection of:
  - Overfitting
  - Underfitting
- Auto hyperparameter tuning applied to:
  - Random Forest
  - XGBoost

---

## 💰 Price Forecasting Logic

- Forecasted returns are converted back to prices
- Uses:
  - Last known closing price
  - Recent mean returns
  - Historical volatility
  - Noise stabilization
- Minimum price constraint applied

Both **return-based** and **price-based** error metrics are reported.

---

## 📈 Forecast Output

Users can view forecasts as:
- Interactive line chart  
- Tabular forecast values  
- Combined chart and table  

Final summary includes:
- Train/Test MAE & RMSE (Returns)
- MAE & RMSE (Prices)
- Model fit status

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Pandas, NumPy  
- Plotly, Matplotlib  
- Scikit-learn  
- Statsmodels  
- XGBoost  
- TensorFlow / Keras  

---

## 📁 Project Structure

```
apple-stock-forecasting/
│
├── app.py
├── requirements.txt
├── README.md
└── data.csv (uploaded by user)
```

---

## ⚙️ Installation & Run Locally

```
git clone https://github.com/your-username/apple-stock-forecasting.git
cd apple-stock-forecasting
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎯 Use Cases

- Data Analyst portfolio project  
- Data Science interview demonstration  
- Time series forecasting showcase  
- Streamlit deployment example  

---

## ⚠️ Disclaimer

This project is for **educational and analytical purposes only**.  
Forecasts are **not financial advice**.

---

## 👤 Author

Athul N A  
Thrissur, Kerala, India  
Email: athulajithan039@gmail.com  
GitHub: https://github.com/Athulajithan  
