import requests
import pandas as pd

def fetch_weather(lat, lon):

    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,cloudcover,shortwave_radiation&past_days=90"

    data = requests.get(url).json()
    hourly = data["hourly"]

    df = pd.DataFrame({
        "time": hourly["time"],
        "temp": hourly["temperature_2m"],
        "humidity": hourly["relativehumidity_2m"],
        "wind": hourly["windspeed_10m"],
        "wind_dir": hourly["winddirection_10m"],
        "cloud": hourly["cloudcover"],
        "solar": hourly["shortwave_radiation"]
    })

    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    return df