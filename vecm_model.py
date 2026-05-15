import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from sklearn.preprocessing import StandardScaler


def train_vecm(df):

    target_cols = ["temp","humidity","wind","wind_dir","cloud","solar"]

    data = df[target_cols].copy()

    # ================= SCALING =================
    scaler = StandardScaler()

    data_scaled = pd.DataFrame(
        scaler.fit_transform(data),
        columns=target_cols,
        index=data.index
    )

    # ================= SAFE LAG SELECTION =================
    try:
        lag_order = select_order(data_scaled, maxlags=12, deterministic="ci")

        if lag_order.aic is not None:
            optimal_lag = lag_order.aic.idxmin()
        else:
            optimal_lag = 3

    except:
        optimal_lag = 3

    # fallback safety
    if optimal_lag is None or optimal_lag < 1:
        optimal_lag = 3

    # ================= COINTEGRATION =================
    try:
        coint_rank = select_coint_rank(
            data_scaled,
            det_order=0,
            k_ar_diff=optimal_lag,
            method="trace",
            signif=0.05
        )

        rank = coint_rank.rank
        if rank == 0:
            rank = 1

    except:
        rank = 1

    # ================= MODEL =================
    model = VECM(
        data_scaled,
        k_ar_diff=optimal_lag,
        coint_rank=rank,
        deterministic="ci"
    )

    fitted_model = model.fit()

    return fitted_model, scaler


def forecast_vecm(model, scaler, steps=24, last_timestamp=None):

    forecast_scaled = model.predict(steps=steps)

    forecast = scaler.inverse_transform(forecast_scaled)

    forecast = np.array(forecast)

    # ================= STABILIZATION =================
    forecast[:,0] = np.clip(forecast[:,0], 5, 50)
    forecast[:,1] = np.clip(forecast[:,1], 0, 100)
    forecast[:,2] = np.maximum(forecast[:,2], 0)
    forecast[:,3] = forecast[:,3] % 360
    forecast[:,4] = np.clip(forecast[:,4], 0, 100)
    forecast[:,5] = np.maximum(forecast[:,5], 0)

    # ================= ADD SMALL VARIATION =================
    noise = np.random.normal(0, 0.2, size=forecast.shape)
    forecast += noise

    # ================= TIME =================
    if last_timestamp is not None:
        start_time = last_timestamp.floor("H") + pd.Timedelta(hours=1)
    else:
        start_time = pd.Timestamp.now().ceil("H")

    return forecast, start_time