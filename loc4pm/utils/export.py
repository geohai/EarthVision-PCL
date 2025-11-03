import os, json, joblib
import numpy as np
import pandas as pd
from ..metrics import regression_metrics

def save_predictions_and_metrics(y_true, y_pred, y_scaler_joblib_path: str, out_csv: str, out_metrics_json: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_metrics_json), exist_ok=True)

    ys = joblib.load(y_scaler_joblib_path)
    inv_true = np.array([ys.inverse_y(v) for v in y_true.reshape(-1)])
    inv_pred = np.array([ys.inverse_y(v) for v in y_pred.reshape(-1)])

    df = pd.DataFrame({ 'truth': inv_true, 'pred': inv_pred })
    df.to_csv(out_csv, index=False)

    mets = regression_metrics(inv_true, inv_pred)
    with open(out_metrics_json, 'w') as f:
        json.dump(mets, f, indent=2)

    return mets