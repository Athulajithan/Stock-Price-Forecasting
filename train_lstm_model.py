import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("AAPL.csv")
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
df.set_index("Date", inplace=True)

df["log_return"] = np.log(df["Adj Close"]).diff()
df.dropna(inplace=True)

scaler = StandardScaler()
scaled = scaler.fit_transform(df[["log_return"]])

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

def create_sequences(data, lookback=120):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

LOOKBACK = 120
X, y = create_sequences(scaled, LOOKBACK)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = X_train.reshape(-1, LOOKBACK, 1)
X_test = X_test.reshape(-1, LOOKBACK, 1)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

model.save("models/lstm_model.h5")
print("✅ LSTM model saved")
