import numpy as np
import pandas as pd

def forecast(model, last_row, steps=24, feature_names=None,
             scaler_X=None, scaler_y=None, df=None, lat=None):

    preds = []
    base = np.array(last_row, dtype=float)

    target_vars = ["temp","humidity","wind","wind_dir","cloud","solar"]
    feature_index = {col: i for i, col in enumerate(feature_names)}

    now = pd.Timestamp.now()
    current_time = now.floor("H") + pd.Timedelta(hours=1)

    for step in range(steps):
        hour = current_time.hour

        # ================= TIME FEATURES =================
        if "hour_sin" in feature_index:
            base[feature_index["hour_sin"]] = np.sin(2*np.pi*hour/24)
            base[feature_index["hour_cos"]] = np.cos(2*np.pi*hour/24)

        if "is_day" in feature_index:
            base[feature_index["is_day"]] = 1 if 6 <= hour <= 18 else 0

        # ================= MODEL =================
        X_input = base.reshape(1,-1)

        X_scaled = scaler_X.transform(X_input) if scaler_X is not None else X_input
        pred_scaled = model.predict(X_scaled)[0]

        ml_pred = scaler_y.inverse_transform([pred_scaled])[0] if scaler_y is not None else pred_scaled
        ml_pred = np.array(ml_pred, dtype=float)

        # ================= HOUR PATTERN =================
        # ================= HOUR PATTERN (UPGRADED) =================
        if df is not None:
            hour_data = df[df.index.hour == hour].tail(60)

            if len(hour_data) > 0:
                hour_pattern = hour_data[target_vars].median().values

                # 🔥 ADD VARIABILITY (VERY IMPORTANT)
                hour_pattern += np.random.normal(0, 0.25, size=hour_pattern.shape)
            else:
                hour_pattern = ml_pred
        else:
            hour_pattern = ml_pred

        # ================= FULL PHYSICS MODEL =================
        physics = ml_pred.copy()

        # -------- SOLAR (already handled later) --------

        # -------- TEMPERATURE DYNAMICS --------
        if 6 <= hour <= 14:
            physics[0] += 0.8
        elif 15 <= hour <= 22:
            physics[0] -= 0.5
        else:
            physics[0] -= 0.3  # night cooling

        # -------- HUMIDITY (inverse relation) --------
        if "temp_lag_1" in feature_index:
            prev_temp = base[feature_index["temp_lag_1"]]
            physics[1] -= 0.2 * (physics[0] - prev_temp)

        # wind increases during daytime heating
        if 10 <= hour <= 17:
            physics[2] += 0.5
        else:
            physics[2] -= 0.2

        # clouds increase with humidity
        physics[4] += 0.1 * (physics[1] - 60)

        # -------- WIND DIRECTION DRIFT --------
        physics[3] += np.random.normal(0, 5)


        # ================= DYNAMIC WEIGHTS =================
        if 6 <= hour <= 18:
        # daytime → ML strong
            w_ml, w_hour, w_phy = 0.60, 0.25, 0.15
        else:
            # night → physics strong
            w_ml, w_hour, w_phy = 0.45, 0.20, 0.35

        # ================= FINAL HYBRID =================
        final_pred = (
            w_ml * ml_pred +
            w_hour * hour_pattern +
            w_phy * physics
        )
        final_pred[1] += np.random.normal(0, 0.4)  # humidity
        final_pred[4] += np.random.normal(0, 0.3)  # cloud
        # ================= CONTROLLED RANDOMNESS =================
        final_pred[0] += np.random.normal(0, 0.3)  # temp
        final_pred[2] += np.random.normal(0, 0.2)  # wind

        # ================= SOLAR FIX =================
        # -------- LATITUDE BASED DAYLIGHT --------
        # ================= LATITUDE-AWARE SOLAR =================

        day_of_year = current_time.dayofyear

        # solar declination
        decl = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))

        # sunrise hour calculation
        lat_rad = np.deg2rad(lat if lat is not None else 20)  # replace later with real lat
        decl_rad = np.deg2rad(decl)

        cos_omega = -np.tan(lat_rad) * np.tan(decl_rad)
        cos_omega = np.clip(cos_omega, -1, 1)

        day_length = 2 * np.degrees(np.arccos(cos_omega)) / 15

        sunrise = 12 - day_length / 2
        sunset = 12 + day_length / 2

        if hour < sunrise or hour > sunset:
            final_pred[5] = 0
        else:
            peak = (sunrise + sunset) / 2
            final_pred[5] = 800 * np.exp(-((hour - peak) ** 2) / 10)

        

        # ================= CLIPPING =================
        final_pred[0] = np.clip(final_pred[0], 5, 50)
        final_pred[1] = np.clip(final_pred[1], 0, 100)
        final_pred[2] = max(0, final_pred[2])
        final_pred[3] = final_pred[3] % 360
        final_pred[4] = np.clip(final_pred[4], 0, 100)
        final_pred[5] = max(0, final_pred[5])

        preds.append(final_pred.copy())

        # ================= SHIFT LAGS (CRITICAL FIX) =================
        for var in target_vars:

            lag_cols = [col for col in feature_names if col.startswith(var) and "_lag_" in col]
            if not lag_cols:
                continue

            max_lag = max(int(col.split("_")[-1]) for col in lag_cols)

            for i in range(max_lag, 1, -1):
                base[feature_index[f"{var}_lag_{i}"]] = base[feature_index[f"{var}_lag_{i-1}"]]

            base[feature_index[f"{var}_lag_1"]] = final_pred[target_vars.index(var)]

        current_time += pd.Timedelta(hours=1)

    return np.array(preds), current_time - pd.Timedelta(hours=steps)