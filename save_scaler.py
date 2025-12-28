import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("AAPL.csv")
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
df.set_index("Date", inplace=True)

df["log_return"] = np.log(df["Adj Close"]).diff()
df.dropna(inplace=True)

scaler = StandardScaler()
scaler.fit(df[["log_return"]])

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ scaler.pkl saved")
