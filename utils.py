import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = y_true.align(y_pred, join='inner')
    return {
        "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }
