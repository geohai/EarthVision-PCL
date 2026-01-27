import os
import json
import joblib
import numpy as np
import pandas as pd
from ..metrics import regression_metrics


def save_predictions_and_metrics(
    y_true,
    y_pred,
    y_scaler_joblib_path: str,
    out_csv: str,
    out_metrics_json: str,
    lat=None,
    lon=None,
    dates=None,
):
    """Save inverse‑scaled predictions/targets to CSV and metrics to JSON.

    This helper will inverse‑transform the scaled predictions and targets using
    the provided ``y_scaler_joblib_path``.  It always writes a CSV with
    ``truth`` and ``pred`` columns and, when provided, also writes the
    corresponding latitude, longitude and date for each sample.  Regression
    metrics are computed using the inverse‑scaled values and written to
    ``out_metrics_json``.

    Parameters
    ----------
    y_true : np.ndarray
        Scaled ground‑truth values (1‑D or 2‑D).  Values will be flattened.
    y_pred : np.ndarray
        Scaled model predictions (1‑D or 2‑D).  Values will be flattened.
    y_scaler_joblib_path : str
        Path to a joblib dump of the y_scaler used during training.  This scaler
        must define an ``inverse_y`` method.
    out_csv : str
        Path to the CSV file where predictions will be saved.  Parent
        directories will be created as needed.
    out_metrics_json : str
        Path to the JSON file where regression metrics will be written.
    lat : array‑like, optional
        Latitude values corresponding to each prediction/target.  If
        provided, these are included as a ``lat`` column in the output CSV.
    lon : array‑like, optional
        Longitude values corresponding to each prediction/target.  If
        provided, these are included as a ``lon`` column in the output CSV.
    dates : array‑like of str, optional
        Date strings corresponding to each prediction/target.  If provided,
        these are included as a ``date`` column in the output CSV.

    Returns
    -------
    dict
        Regression metrics computed on the inverse‑scaled values.
    """
    # Ensure output directories exist
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_metrics_json), exist_ok=True)

    # Load the scaler and inverse‑transform the flattened arrays
    ys = joblib.load(y_scaler_joblib_path)
    # Flatten inputs to 1‑D
    inv_true = np.array([ys.inverse_y(v) for v in y_true.reshape(-1)])
    inv_pred = np.array([ys.inverse_y(v) for v in y_pred.reshape(-1)])

    # Build a dict for the DataFrame
    data = {
        'truth': inv_true,
        'pred': inv_pred,
    }
    if lat is not None:
        data['lat'] = list(lat)
    if lon is not None:
        data['lon'] = list(lon)
    if dates is not None:
        data['date'] = list(dates)

    df = pd.DataFrame(data)
    df.to_csv(out_csv, index=False)

    mets = regression_metrics(inv_true, inv_pred)
    with open(out_metrics_json, 'w') as f:
        json.dump(mets, f, indent=2)

    return mets