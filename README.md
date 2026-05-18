# Hybrid AI-Based Hourly Weather Forecasting

An intelligent weather forecasting system that predicts hourly weather conditions using Machine Learning, Deep Learning, Statistical Forecasting, Hybrid Intelligence, and Ensemble Learning.

## Features

- Hourly weather forecasting using geographical coordinates
- Multi-model prediction system:
  - LightGBM
  - CatBoost
  - GRU
  - VECM
- Hybrid physics-informed weather correction
- Ensemble forecasting engine
- Interactive Streamlit dashboard
- Model performance evaluation (MAE, RMSE, MAPE)
- Weather intelligence summary
- Solar radiation heatmap
- KPI weather cards
- Forecast comparison tables

## Forecasted Parameters

- Temperature
- Humidity
- Wind Speed
- Wind Direction
- Cloud Cover
- Solar Radiation

## Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn
- LightGBM
- CatBoost
- TensorFlow / Keras
- Statsmodels
- Plotly
- Open-Meteo API
- Optuna

## Project Workflow

User Input (Latitude / Longitude)  
→ Weather Data Fetching  
→ Data Preprocessing  
→ Feature Engineering  
→ Model Training  
→ Hybrid Intelligence Layer  
→ Ensemble Forecasting  
→ Performance Evaluation  
→ Interactive Dashboard  

## Model Contribution

| Model | Role |
|------|------|
| LightGBM | Nonlinear structured forecasting |
| CatBoost | Feature interaction learning |
| GRU | Sequential time-series forecasting |
| VECM | Statistical dependency modeling |

## Installation

Clone repository:

```bash
git clone https://github.com/your-username/Hybrid-AI-Based-Hourly-Weather-Forecasting.git
cd Hybrid-AI-Based-Hourly-Weather-Forecasting
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```

## Applications

- Weather decision support
- Climate analysis
- Agriculture planning
- Transportation weather monitoring
- Renewable energy forecasting
- Environmental intelligence

## Future Scope

- Real-time live forecasting
- Transformer-based models
- Satellite weather integration
- Weather alert notifications
- Cloud deployment
