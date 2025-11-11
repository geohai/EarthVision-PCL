"""Daily time-series dataset with optional geographic and physical targets.

This dataset loads daily time-series samples from NumPy files, applies
feature scaling, and supports returning geographic coordinates, month
indices, and additional physical simulation variables. It is an
enhancement of the original ``DailyTSDataset`` to support dual targets
(observation and physical simulation) used in the dual-decoder model.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


DATE_FMT = "%Y-%m-%d"


class MinMaxScalerLite:
    """A lightweight MinMax scaler for 1-D or multi-dimensional arrays.

    This scaler computes per-feature minima and maxima and applies a linear
    transformation to map values into a specified range. It can operate on
    both time-series features (shape [N, T, F]) and single targets (shape
    [N, F] or [N, 1]).
    """

    def __init__(self, feature_range: Tuple[float, float] = (-1.0, 1.0)) -> None:
        self.min: Optional[np.ndarray | float] = None  # shape [F] for multi-dim, scalar for single-dim
        self.max: Optional[np.ndarray | float] = None
        self.lo, self.hi = feature_range

    def fit_x_chunk(self, X: np.ndarray) -> None:
        """Update scaler min/max using a chunk of features.

        Args:
            X: Array of shape [N, T, F] containing feature values.
        """
        n, t, f = X.shape
        X2 = X.reshape(n * t, f)
        mn = np.nanmin(X2, axis=0)
        mx = np.nanmax(X2, axis=0)
        if self.min is None:
            self.min, self.max = mn, mx
        else:
            self.min = np.minimum(self.min, mn)
            self.max = np.maximum(self.max, mx)

    def fit_y_chunk(self, y: np.ndarray) -> None:
        """Update scaler min/max using a chunk of target values.

        Args:
            y: Array of shape [N] or [N,1] containing target values.
        """
        vmin = float(np.nanmin(y))
        vmax = float(np.nanmax(y))
        if self.min is None:
            self.min, self.max = vmin, vmax
        else:
            self.min = min(self.min, vmin)
            self.max = max(self.max, vmax)

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        den = (self.max - self.min)
        # Avoid division by zero
        den = np.where(den == 0, 1.0, den)
        Z = (X - self.min) / den
        return Z * (self.hi - self.lo) + self.lo

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        den = (self.max - self.min) or 1.0
        z = (y - self.min) / den
        return z * (self.hi - self.lo) + self.lo

    def inverse_y(self, z: np.ndarray) -> np.ndarray:
        den = (self.max - self.min) or 1.0
        y = (z - self.lo) / (self.hi - self.lo) * den + self.min
        return y


def iter_dates(start_date: str, end_date: str) -> List[str]:
    d0 = pd.to_datetime(start_date)
    d1 = pd.to_datetime(end_date)
    return pd.date_range(d0, d1, freq='D').strftime(DATE_FMT).tolist()


def discover_pairs(root: str, dates: List[str], x_prefix: str, y_prefix: str) -> List[Tuple[str, str]]:
    pairs = []
    for ds in dates:
        yyyy = ds[:4]
        x = os.path.join(root, yyyy, f"{x_prefix}{ds}.npy")
        y = os.path.join(root, yyyy, f"{y_prefix}{ds}.npy")
        if os.path.exists(x) and os.path.exists(y):
            pairs.append((x, y))
    if not pairs:
        raise FileNotFoundError(f"No daily pairs found in {root} for {dates[0]}..{dates[-1]}")
    return pairs


class DailyTSDataset(Dataset):
    """Daily time-series dataset for pollution modeling.

    This dataset loads daily samples of time-series features and observation
    targets from NumPy files. Optionally, it can return geographic
    coordinates, month indices, and additional physical simulation targets.
    """

    def __init__(self,
                 root: str,
                 start_date: str,
                 end_date: str,
                 x_prefix: str = 'TS_X_',
                 y_prefix: str = 'TS_y_',
                 drop_feature_indices: Optional[List[int]] = None,
                 filter_y_positive: bool = True,
                 remove_nan_rows: bool = True,
                 scaler_range: Tuple[float, float] = (-1.0, 1.0),
                 prefetch_files: int = 4,
                 lat_feature_index: Optional[int] = None,
                 lon_feature_index: Optional[int] = None,
                 *,
                 return_coords: bool = False,
                 return_month: bool = False,
                 z_feature_indices: Optional[List[int]] = None,
                 z_scaler_range: Optional[Tuple[float, float]] = None) -> None:
        super().__init__()
        self.root = root
        self.dates = iter_dates(start_date, end_date)
        self.pairs = discover_pairs(root, self.dates, x_prefix, y_prefix)
        self.drop_idxs = set(drop_feature_indices or [])
        self.filter_y_positive = filter_y_positive
        self.remove_nan_rows = remove_nan_rows
        self.prefetch_files = max(1, int(prefetch_files))
        self.lat_idx = lat_feature_index
        self.lon_idx = lon_feature_index

        # Flags controlling auxiliary outputs
        self.return_coords = bool(return_coords)
        self.return_month = bool(return_month)
        self.z_feature_indices = list(z_feature_indices or [])
        self.return_z = len(self.z_feature_indices) > 0

        self.keep_feat_idx: Optional[List[int]] = None
        self.file_meta: List[Dict] = []
        x_scaler = MinMaxScalerLite(feature_range=scaler_range)
        y_scaler = MinMaxScalerLite(feature_range=scaler_range)
        # Use separate scaler for z if provided; fallback to same range
        z_range = z_scaler_range if z_scaler_range is not None else scaler_range
        z_scaler = MinMaxScalerLite(feature_range=z_range) if self.return_z else None

        index_map: List[Tuple[int, int]] = []
        coords_list: List[Tuple[float, float]] = []
        months_list: List[int] = []
        # Iterate through daily files and accumulate scaling statistics and index mapping
        for fid, (xp, yp) in enumerate(self.pairs):
            X = np.load(xp)  # shape [N, T, F]
            y = np.load(yp)  # shape [N] or [N,1]
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            N, T, F = X.shape
            # Extract raw coordinate vectors if lat/lon indices are provided
            if self.lat_idx is not None and self.lon_idx is not None and self.lat_idx < F and self.lon_idx < F:
                latvec = X[:, 0, self.lat_idx]
                lonvec = X[:, 0, self.lon_idx]
            else:
                latvec = np.full((N,), np.nan, dtype=float)
                lonvec = np.full((N,), np.nan, dtype=float)
            # Initialize keep indices (features to retain) on first file
            if self.keep_feat_idx is None:
                self.keep_feat_idx = [i for i in range(F) if i not in self.drop_idxs]
            # Subset X to kept features
            Xk = X[:, :, self.keep_feat_idx]
            # Build validity mask for rows
            m = np.ones((N,), dtype=bool)
            if self.filter_y_positive:
                m &= (y.reshape(-1) > 0)
            if self.remove_nan_rows:
                m &= ~np.isnan(Xk).any(axis=(1, 2))
            valid_idx = np.where(m)[0]
            if valid_idx.size:
                x_scaler.fit_x_chunk(Xk[valid_idx])
                y_scaler.fit_y_chunk(y[valid_idx])
                # Fit z scaler on valid rows if needed
                if self.return_z and z_scaler is not None:
                    # Extract z_chunk from raw X for all time steps to capture variability, shape [N,T,len(z)]
                    z_chunk = X[valid_idx][:, -1, self.z_feature_indices]
                    # For scaling, treat z as features with a dummy time dimension
                    z_scaler.fit_x_chunk(z_chunk[:, None, :])
            # Populate index map and auxiliary metadata
            for rid in valid_idx:
                index_map.append((fid, int(rid)))
                coords_list.append((float(latvec[rid]), float(lonvec[rid])))
                # compute the month (1-12) corresponding to this file/date
                try:
                    date_str = self.dates[fid]
                    mth = int(date_str.split("-")[1])
                except Exception:
                    mth = 1
                months_list.append(mth)
            self.file_meta.append({
                'x_path': xp,
                'y_path': yp,
                'n_rows': N,
                'valid_idx': valid_idx,
            })
        if not index_map:
            raise RuntimeError("No valid samples after filtering; relax filters or check data.")
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.z_scaler = z_scaler
        self.index_map = index_map
        self.coords = np.asarray(coords_list, dtype=float)  # shape [M,2]
        self.months = np.asarray(months_list, dtype=int)

        # Cache loaded arrays to reduce I/O overhead
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []

    def __len__(self) -> int:
        return len(self.index_map)

    def _get_from_cache(self, path: str) -> np.ndarray:
        if path in self._cache:
            return self._cache[path]
        arr = np.load(path, mmap_mode=None)
        self._cache[path] = arr
        self._cache_order.append(path)
        if len(self._cache_order) > self.prefetch_files:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return arr

    def __getitem__(self, idx: int):
        fid, rid = self.index_map[idx]
        meta = self.file_meta[fid]
        X = self._get_from_cache(meta['x_path'])
        y = self._get_from_cache(meta['y_path'])
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        # Subset features and scale them
        x_row = X[rid][:, self.keep_feat_idx]
        y_row = y[rid].astype(np.float32)
        x_row = self.x_scaler.transform_x(x_row).astype(np.float32)
        y_row = self.y_scaler.transform_y(y_row).astype(np.float32)
        # Build output tuple dynamically
        out: List[np.ndarray | float | int] = [x_row]
        if self.return_coords:
            out.append(self.coords[idx].astype(np.float32))
        if self.return_month:
            out.append(self.months[idx].astype(np.int64))
        # Append scaled physical variables if requested
        if self.return_z:
            # Extract raw z values from the first time step of the raw X array
            z_raw = X[rid, -1, self.z_feature_indices].astype(np.float32)
            # Scale z using z_scaler (element-wise). z_scaler.min/max may be arrays or scalars
            z_row = self.z_scaler.transform_x(z_raw[None, None, :]).reshape(-1)  # [Z]
            z_row = z_row.astype(np.float32)
            out.append(z_row.astype(np.float32))
        out.append(y_row)
        return tuple(out)