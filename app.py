import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_fetch import fetch_weather
from features import create_features
from models import get_lightgbm, get_catboost
from gru_model import train_gru, forecast_gru, create_sequences
from vecm_model import train_vecm, forecast_vecm
from metrics import evaluate
from forecast import forecast
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide", page_title="Weather Forecast AI")

# ---------------- HEADER ----------------
st.title("🌦️ Weather Forecast AI Dashboard")
st.markdown("### Multi-Model Time Series Forecasting System")

# ---------------- SIDEBAR ----------------
lat = st.sidebar.number_input("Latitude", value=18.52)
lon = st.sidebar.number_input("Longitude", value=73.85)

run_prediction = st.button("🚀 Get Prediction")

models_selected = st.sidebar.multiselect(
    "Select Models",
    ["LightGBM", "CatBoost", "GRU", "VECM"],
    default=["LightGBM", "CatBoost", "GRU", "VECM"]
)

forecast_steps = st.sidebar.slider("Forecast Hours", 24, 48)

# ---------------- EXECUTION ----------------
if run_prediction:
    df = fetch_weather(lat, lon)
    df = df.tail(60 * 24)

    df_feat = create_features(df)

    X = df_feat.drop(["temp","humidity","wind","wind_dir","cloud","solar"], axis=1)
    y = df_feat[["temp","humidity","wind","wind_dir","cloud","solar"]]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    split = int(len(X)*0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler_X.fit(X_train)
    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y.fit(y_train)
    y_train_scaled = scaler_y.transform(y_train)

    predictions = {}
    forecasts = {}

    # ================= MODELS =================

    # -------- LIGHTGBM --------
    if "LightGBM" in models_selected:
        lgbm = get_lightgbm()
        lgbm.fit(X_train_scaled, y_train_scaled)

        pred_scaled = lgbm.predict(X_test_scaled)
        pred = scaler_y.inverse_transform(pred_scaled)
        predictions["LightGBM"] = pred

        preds_lgbm, start_time = forecast(
            lgbm,
            X.iloc[-1].values,
            forecast_steps,
            X.columns,
            scaler_X,
            scaler_y,
            df,
            lat
        )
        forecasts["LightGBM"] = preds_lgbm

    # -------- CATBOOST --------
    if "CatBoost" in models_selected:
        cat = get_catboost()
        cat.fit(X_train_scaled, y_train_scaled)

        pred_scaled = cat.predict(X_test_scaled)
        pred = scaler_y.inverse_transform(pred_scaled)
        predictions["CatBoost"] = pred

        preds_cat, start_time = forecast(
            cat,
            X.iloc[-1].values,
            forecast_steps,
            X.columns,
            scaler_X,
            scaler_y,
            df,
            lat
        )
        forecasts["CatBoost"] = preds_cat

    # -------- GRU --------
    # ---------------- GRU ----------------
    if "GRU" in models_selected:

        X_np = X.values
        y_np = y.values

        gru_model, sx, sy = train_gru(X_np, y_np)

        # -------- TRAIN PREDICTION (FOR METRICS) --------
        X_scaled = sx.transform(X_np)
        y_scaled = sy.transform(y_np)

        X_seq, y_seq = create_sequences(X_scaled, y_scaled)

        pred_seq = gru_model.predict(X_seq)
        pred = sy.inverse_transform(pred_seq)

        # Align shapes
        actual = y_np[-len(pred):]

        predictions["GRU"] = pred

        # -------- FORECAST --------
        preds_gru, start_time = forecast_gru(
            gru_model,
            X_seq[-1],
            sy,
            steps=forecast_steps,
            last_timestamp=df.index[-1],
            lat = lat
        )
        forecasts["GRU"] = preds_gru

    # -------- VECM --------
    # ---------------- VECM ----------------
    if "VECM" in models_selected:

        vecm, scaler_vecm= train_vecm(df[["temp","humidity","wind","wind_dir","cloud","solar"]])

        # -------- TEST PREDICTION --------
        steps_test = len(y_test)

        pred_scaled = vecm.predict(steps=steps_test)
        pred = scaler_vecm.inverse_transform(pred_scaled)

        predictions["VECM"] = pred

        # -------- FORECAST --------
        preds_vecm, start_time = forecast_vecm(
            vecm,
            scaler_vecm,
            forecast_steps,
            last_timestamp=df.index[-1]
        )
        forecasts["VECM"] = preds_vecm
        start_time = df.index[-1].floor("H") + pd.Timedelta(hours=1)

    # ================= METRICS (TAB1 FIX) =================
    metrics_list = []

    for name, pred in predictions.items():
        try:
            actual = y_test.values
            pred = pred[:len(actual)]

            mae, rmse, mape = evaluate(actual[:,0], pred[:,0])

            metrics_list.append({
                "Model": name,
                "MAE": round(mae,2),
                "RMSE": round(rmse,2),
                "MAPE (%)": round(mape,2),
                "Accuracy (%)": round(max(0,100-mape),2)
            })
        except:
            pass
    # ================= ENSEMBLE MODEL =================

    if "LightGBM" in forecasts and "GRU" in forecasts:
        forecasts["Ensemble"] = (
            0.6 * forecasts["LightGBM"] +
            0.4 * forecasts["GRU"]
        )
    if "Ensemble" in forecasts:
        predictions["Ensemble"] = forecasts["Ensemble"]    

    metrics_df = pd.DataFrame(metrics_list).sort_values("RMSE").reset_index(drop=True)

    best_model = metrics_df.iloc[0]["Model"]

    # ================= TIME =================
    start_time = pd.Timestamp.now().ceil("H")
    first_key = list(forecasts.keys())[0]
    time_index = pd.date_range(
        start=start_time,
        periods=len(forecasts[first_key]),
        freq="H"
    )

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Leaderboard","📈 Predictions","🔮 Forecast","📉 Errors"])

    # ================= TAB1 =================
    with tab1:
        st.subheader("Model Accuracy Comparison")
        def highlight_accuracy(val, max_val, min_val):
            if val == max_val:
                return 'background-color: #00FF00; color: black'  # bright green (best)
            elif val > (max_val + min_val) / 2:
                return 'background-color: #90EE90; color: black'  # light green (good)
            else:
                return 'background-color: #FF7F7F; color: black'  # light red (low)


        max_acc = metrics_df["Accuracy (%)"].max()
        min_acc = metrics_df["Accuracy (%)"].min()

        styled_df = metrics_df.style.applymap(
            lambda val: highlight_accuracy(val, max_acc, min_acc),
            subset=["Accuracy (%)"]
        )

        st.dataframe(styled_df, use_container_width=True)
        st.success(f"🏆 Best Model: {best_model}")

    # ================= TAB2 =================
    # ================= TAB2 =================
    with tab2:

        # -------- ENSEMBLE (KEEP THIS) --------
        available = []

        if "LightGBM" in forecasts:
            available.append(0.5 * forecasts["LightGBM"])

        if "GRU" in forecasts:
            available.append(0.3 * forecasts["GRU"])

        if "CatBoost" in forecasts:
            available.append(0.2 * forecasts["CatBoost"])

        best_future = sum(available)

        # -------- EXTRACT VARIABLES --------
        temp = best_future[:,0]
        hum = best_future[:,1]
        wind = best_future[:,2]
        wind_dir = best_future[:,3]
        cloud = best_future[:,4]
        solar = best_future[:,5]

        # -------- TIME FORMAT --------
        def format_time_range(t):
            start = t.strftime("%I %p")
            end = (t + pd.Timedelta(hours=1)).strftime("%I %p")
            return f"{start} - {end}"
        
        temp = np.clip(temp, 5, 45)
        # -------- DATAFRAME --------
        df_display = pd.DataFrame({
            "Time": [format_time_range(t) for t in time_index],
            "Temp (°C)": np.round(temp).astype(int),
            "Humidity (%)": hum.round(2),
            "Wind (m/s)": wind.round(2),
            "Wind Dir (°)": wind_dir.round(2),
            "Cloud (%)": cloud.round(2),
            "Solar (W/m²)": solar.round(2),
        })

        # -------- EMOJI --------
        def weather_emoji(row):
            if row["Solar (W/m²)"] < 10:
                return "🌙"
            elif row["Cloud (%)"] > 60:
                return "☁️"
            elif row["Solar (W/m²)"] > 200:
                return "☀️"
            else:
                return "🌤"

        df_display["Weather"] = df_display.apply(weather_emoji, axis=1)

        # ================= HEATMAP =================
        st.subheader("🌡 Heatmap Overview")

        heat_df = pd.DataFrame({
            "Solar (W/m²)": df_display["Solar (W/m²)"]
        }).T

        fig_heat = go.Figure(data=go.Heatmap(
            z=heat_df.values,
            x=df_display["Time"],
            y=heat_df.index,
            colorscale='YlOrRd'
        ))

        st.plotly_chart(fig_heat, use_container_width=True)
        
        #heatmap
        st.subheader("🕒 Hourly Weather Cards")

        cols = st.columns(4)

        for i in range(len(df_display)):

            row = df_display.iloc[i]

            emoji = row["Weather"]

            card = f"""
            **{row['Time']}**

            {emoji}

            🌡 {row['Temp (°C)']}°C  
            💧 {row['Humidity (%)']}%  
            🌬 {row['Wind (m/s)']} m/s  
            ☁️ {row['Cloud (%)']}%  
            ☀️ {row['Solar (W/m²)']} W/m²
            """

            cols[i % 4].markdown(card)

        ## statement block    
        st.subheader("🧠 Weather Intelligence Summary")
        # ===== Extract arrays =====
        temp_arr = df_display["Temp (°C)"].values
        hum_arr = df_display["Humidity (%)"].values
        wind_arr = df_display["Wind (m/s)"].values
        wind_dir_arr = df_display["Wind Dir (°)"].values
        cloud_arr = df_display["Cloud (%)"].values
        solar_arr = df_display["Solar (W/m²)"].values
        time_arr = df_display["Time"].values

        summary = []
        # ===== RAIN PROBABILITY =====
        rain_score = (
            0.5 * (hum_arr / 100) +
            0.3 * (cloud_arr / 100) +
            0.2 * (1 - solar_arr / (solar_arr.max() + 1e-5))
        )

        rain_percent = int(np.mean(rain_score) * 100)

        if rain_percent > 70:
            summary.append(f"🌧 High probability of rainfall (~{rain_percent}%).")
        elif rain_percent > 40:
            summary.append(f"🌦 Moderate chance of rain (~{rain_percent}%).")
        else:
            summary.append(f"🌤 Low probability of rain (~{rain_percent}%).")

        # ===== SEVERE WEATHER DETECTION =====

        # 🔥 Heatwave
        if temp_arr.max() > 40:
            peak_time = time_arr[np.argmax(temp_arr)]
            summary.append(f"🔥 Severe heatwave expected around {peak_time}.")

        # 🌪 Strong wind / storm
        if wind_arr.max() > 12:
            peak_time = time_arr[np.argmax(wind_arr)]
            summary.append(f"🌪 Strong wind / storm conditions likely near {peak_time}.")

        # ⛈ Storm condition (combined logic)
        storm_idx = np.where(
            (hum_arr > 80) &
            (cloud_arr > 70) &
            (wind_arr > 8)
        )[0]

        if len(storm_idx) > 0:
            storm_time = time_arr[storm_idx[0]]
            summary.append(f"⛈ Storm conditions possible around {storm_time}.")    

        # ===== Temperature Trend with Time =====
        temp_diff = np.diff(temp_arr)

        rise_hours = np.where(temp_diff > 0)[0]
        fall_hours = np.where(temp_diff < 0)[0]

        if len(rise_hours) > 3:
            start = time_arr[rise_hours[0]]
            end = time_arr[rise_hours[-1]]
            summary.append(f"🌡 From {start} to {end}, temperature is expected to rise.")

        if temp_arr.max() > 38:
            peak_time = time_arr[np.argmax(temp_arr)]
            summary.append(f"🔥 Heatwave conditions likely around {peak_time}.")

        # ===== Solar =====
        if solar_arr.max() > 600:
            peak_time = time_arr[np.argmax(solar_arr)]
            summary.append(f"☀️ Strong sunlight expected near {peak_time}.")

        # ===== Rain Detection =====
        rain_idx = np.where((hum_arr > 70) & (cloud_arr > 60) & (solar_arr < 200))[0]

        if len(rain_idx) > 0:
            start = time_arr[rain_idx[0]]
            summary.append(f"🌧 Rainfall chances increase around {start} due to high humidity and cloud cover.")

        # ===== Cloud =====
        if cloud_arr.mean() < 30:
            summary.append("☁️ Mostly clear skies expected.")
        elif cloud_arr.mean() > 70:
            summary.append("☁️ Overcast conditions likely.")

        # ===== Wind =====
        avg_wind = np.mean(wind_arr)
        if avg_wind > 8:
            summary.append("🌬 Strong winds expected.")
        else:
            summary.append("🌬 Moderate wind conditions.")

        # ===== Wind Direction =====
        def get_direction(angle):
            if 45 <= angle < 135:
                return "East"
            elif 135 <= angle < 225:
                return "South"
            elif 225 <= angle < 315:
                return "West"
            else:
                return "North"

        direction = get_direction(np.mean(wind_dir_arr))
        summary.append(f"🧭 Winds predominantly from the {direction}.")

        # ===== FINAL OUTPUT =====
        for line in summary:
            st.markdown(line)

    # ================= TAB3 =================
    with tab3:

        for model_name, data in forecasts.items():

            st.subheader(f"{model_name} Forecast")

            df_m = pd.DataFrame({
                "Time": [format_time_range(t) for t in time_index],
                "Temp (°C)": data[:,0].round(0).astype(int),
                "Humidity (%)": data[:,1].round(2),
                "Wind (m/s)": data[:,2].round(2),
                "Wind Dir (°)": data[:,3].round(2),
                "Cloud (%)": data[:,4].round(2),
                "Solar (W/m²)": data[:,5].round(2),
            })

            st.dataframe(df_m, use_container_width=True)

    # ================= TAB4 =================
    with tab4:

        st.subheader("📊 Model Performance Analysis")

        # ================= MAE GRAPH =================
        st.markdown("### 🔹 MAE Comparison")

        fig_mae = go.Figure()

        fig_mae.add_trace(go.Bar(
            x=metrics_df["Model"],
            y=metrics_df["MAE"],
            name="MAE"
        ))

        fig_mae.update_layout(
            title="Mean Absolute Error (MAE)",
            xaxis_title="Models",
            yaxis_title="Error"
        )

        st.plotly_chart(fig_mae, use_container_width=True)

        # ================= RMSE GRAPH =================
        st.markdown("### 🔹 RMSE Comparison")

        fig_rmse = go.Figure()

        fig_rmse.add_trace(go.Bar(
            x=metrics_df["Model"],
            y=metrics_df["RMSE"],
            name="RMSE"
        ))

        fig_rmse.update_layout(
            title="Root Mean Squared Error (RMSE)",
            xaxis_title="Models",
            yaxis_title="Error"
        )

        st.plotly_chart(fig_rmse, use_container_width=True)

        # ================= MAPE GRAPH =================
        st.markdown("### 🔹 MAPE Comparison")

        fig_mape = go.Figure()

        fig_mape.add_trace(go.Bar(
            x=metrics_df["Model"],
            y=metrics_df["MAPE (%)"],
            name="MAPE"
        ))

        fig_mape.update_layout(
            title="Mean Absolute Percentage Error (MAPE)",
            xaxis_title="Models",
            yaxis_title="Percentage"
        )

        st.plotly_chart(fig_mape, use_container_width=True)

        st.subheader("📢 Model Insight")

        # Best model (lowest RMSE)
        best_model = metrics_df.loc[metrics_df["RMSE"].idxmin()]

        # Worst model (highest RMSE)
        worst_model = metrics_df.loc[metrics_df["RMSE"].idxmax()]

        # Best MAPE
        best_mape_model = metrics_df.loc[metrics_df["MAPE (%)"].idxmin()]

        # Generate statement
        statement = f"""
        ✔ Best Model: **{best_model['Model']}** (Lowest RMSE: {best_model['RMSE']})

        ⚠️ Highest Error Model: **{worst_model['Model']}** (RMSE: {worst_model['RMSE']})

        📊 Most Accurate Percentage Prediction: **{best_mape_model['Model']}** (MAPE: {best_mape_model['MAPE (%)']}%)

        👉 Recommendation: Use **{best_model['Model']}** for reliable weather forecasting.
        """

        st.success(statement)