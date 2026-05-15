
import numpy as np

def evaluate(y_true, y_pred):
    mae = np.mean(abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(abs((y_true - y_pred)/(y_true+1e-5)))*100
    return mae, rmse, mape
