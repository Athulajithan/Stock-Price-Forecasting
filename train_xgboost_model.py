import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor

df = pd.read_csv("AAPL.csv")
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
df.set_index("Date", inplace=True)

df["log_return"] = np.log(df["Adj Close"]).diff()
df.dropna(inplace=True)

def create_features(df):
    X = pd.DataFrame(index=df.index)
    X["lag_1"] = df["log_return"].shift(1)
    X["lag_5"] = df["log_return"].shift(5)
    X["lag_10"] = df["log_return"].shift(10)
    X["roll_mean_5"] = df["log_return"].rolling(5).mean().shift(1)
    X["roll_std_5"] = df["log_return"].rolling(5).std().shift(1)
    y = df["log_return"]
    X.dropna(inplace=True)
    y = y.loc[X.index]
    return X, y

X, y = create_features(df)

train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    early_stopping_rounds=20,
    eval_metric="rmse"
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

with open("models/xgboost_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ XGBoost model saved")
