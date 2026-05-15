import pandas as pd
import numpy as np

def create_features(df):
    df = df.copy()

    # ================= TIME FEATURES =================
    df["hour"] = df.index.hour

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_sin"] = np.sin(2 * np.pi * df.index.day / 31)
    df["day_cos"] = np.cos(2 * np.pi * df.index.day / 31)

    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    df["is_day"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)

    # ================= LAGS =================
    for col in ["temp","humidity","wind","wind_dir","cloud","solar"]:
        for lag in range(1,7):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # ================= ROLLING =================
    for col in ["temp","humidity","wind"]:
        df[f"{col}_roll_mean_6"] = df[col].rolling(6).mean()
        df[f"{col}_roll_std_6"] = df[col].rolling(6).std()

    # ================= PHYSICS FEATURES =================
    df["temp_trend"] = df["temp"].diff()
    df["humidity_trend"] = df["humidity"].diff()

    df["temp_humidity"] = df["temp"] * df["humidity"]
    df["solar_cloud"] = df["solar"] * (100 - df["cloud"])

    df.dropna(inplace=True)

    return df