import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


# ================= SEQUENCE CREATION =================
def create_sequences(X, y, window=24):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)


# ================= TRAIN GRU =================
def train_gru(X, y):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled)

    model = Sequential([
        GRU(128, return_sequences=True),
        Dropout(0.2),
        GRU(64),
        Dense(y.shape[1])
    ])

    model.compile(loss="mse", optimizer="adam")

    model.fit(
        X_seq,
        y_seq,
        epochs=80,
        batch_size=8,
        verbose=0
    )

    return model, scaler_X, scaler_y


# ================= FORECAST GRU =================
def forecast_gru(model, last_seq, scaler_y, steps=24, last_timestamp=None, lat=None):

    preds = []
    seq = last_seq.copy()

    current_time = last_timestamp if last_timestamp is not None else pd.Timestamp.now()
    start_time = current_time.floor("H") + pd.Timedelta(hours=1)

    for _ in range(steps):

        hour = current_time.hour

        # ================= MODEL =================
        pred_scaled = model.predict(seq[np.newaxis,:,:], verbose=0)[0]
        pred = scaler_y.inverse_transform([pred_scaled])[0]
        pred = np.array(pred)

        # 🔥 inject time awareness
        time_factor = np.sin(2*np.pi*hour/24)
        pred[0] += 0.3 * time_factor

        # ================= PHYSICS =================
        physics = pred.copy()

        # 🌞 solar
        if lat is not None:
            solar_strength = 800 * max(0.4, np.cos(np.radians(lat)))
        else:
            solar_strength = 700

        if 6 <= hour <= 18:
            physics[5] = solar_strength * np.exp(-0.1 * (hour - 12)**2)
        else:
            physics[5] = 0

        # 🌡 temperature trend
        if 6 <= hour <= 14:
            physics[0] += 0.5
        elif 15 <= hour <= 22:
            physics[0] -= 0.4

        # 💧 humidity inverse
        physics[1] -= 0.2 * (physics[0] - pred[0])

        # ================= HYBRID =================
        final_pred = 0.7 * pred + 0.3 * physics

        # ================= RANDOMNESS =================
        final_pred += np.random.normal(0, 0.25, size=final_pred.shape)

        # 🌬 wind smoothing
        final_pred[2] = 0.7 * pred[2] + 0.3 * final_pred[2]

        # 🔥 strict solar rule
        if hour < 6 or hour > 18:
            final_pred[5] = 0

        # ================= STABILITY =================
        final_pred[0] = np.clip(final_pred[0], 5, 50)
        final_pred[1] = np.clip(final_pred[1], 0, 100)
        final_pred[2] = max(0, final_pred[2])
        final_pred[3] = final_pred[3] % 360
        final_pred[4] = np.clip(final_pred[4], 0, 100)
        final_pred[5] = max(0, final_pred[5])

        preds.append(final_pred.copy())

        # ================= UPDATE SEQUENCE =================
        seq = np.roll(seq, -1, axis=0)

        new_row = seq[-1].copy()

        alpha = 0.85
        momentum = new_row[-6:] - seq[-2][-6:]

        new_row[-6:] = (
            alpha * final_pred +
            (1 - alpha) * new_row[-6:] +
            0.2 * momentum
        )

        seq[-1] = new_row

        current_time += pd.Timedelta(hours=1)

    return np.array(preds), start_time