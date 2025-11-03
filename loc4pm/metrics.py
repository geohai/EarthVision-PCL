import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def regression_metrics(truth, pred):
    t = np.asarray(truth).ravel()
    p = np.asarray(pred).ravel()
    return {
        "R2":   float(r2_score(t, p)),
        "MAE":  float(mean_absolute_error(t, p)),
        "RMSE": float(np.sqrt(mean_squared_error(t, p))),
        "MBE":  float(np.mean(p - t)),
    }