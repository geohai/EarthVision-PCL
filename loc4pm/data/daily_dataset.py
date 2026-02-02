"""
Daily time-series dataset with support for auxiliary physical targets and
z‑score standardisation.

This module extends the original ``DailyTSDataset`` used in LOC4PM to
support an auxiliary physical head in the model and to normalise the
observation target ``y`` via z‑score.  In addition to the base
time‑series features and optional coordinate/time indices, the dataset
returns primary physical variables ``z`` (e.g. PM2.5), auxiliary
physical variables ``z_aux`` (e.g. meteorological conditions), and the
standardised observation ``y``.  When precomputed statistics are
provided for any of these variables, they are used for z‑score
standardisation; otherwise the dataset falls back to the configured
min/max or robust scalers.
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
    transformation to map values into a specified range.  It can operate on
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


class RobustScalerLite:
    """A robust scaler using the median and IQR.

    This scaler scales targets based on their median and inter‑quartile range (IQR), which makes it
    more robust to outliers and right‑skewed distributions compared to a simple min/max scaler.  The scaled
    values are linearly mapped into a specified range.  For example, with ``feature_range=(-1, 1)``,
    the first quartile (25th percentile) of the data will map to approximately ``-0.5`` and the third
    quartile (75th percentile) will map to ``0.5``.
    """

    def __init__(self, feature_range: Tuple[float, float] = (-1.0, 1.0)) -> None:
        self.median: Optional[float] = None
        self.iqr: Optional[float] = None
        self.lo, self.hi = feature_range

    def fit_y_chunk(self, y: np.ndarray) -> None:
        """Update the scaler's median and IQR using a chunk of target values."""
        y_flat = y.reshape(-1).astype(float)
        med = float(np.nanmedian(y_flat))
        q1 = float(np.nanpercentile(y_flat, 25))
        q3 = float(np.nanpercentile(y_flat, 75))
        iqr = q3 - q1
        if iqr == 0:
            iqr = 1.0
        if self.median is None:
            self.median = med
            self.iqr = iqr
        else:
            # simple running average to accumulate across chunks
            self.median = (self.median + med) / 2.0
            self.iqr = (self.iqr + iqr) / 2.0

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.median is None or self.iqr is None:
            raise RuntimeError("RobustScalerLite must be fitted before calling transform_y")
        z = (y - self.median) / self.iqr
        return z * (self.hi - self.lo) + (self.hi + self.lo) / 2.0

    def inverse_y(self, z: np.ndarray) -> np.ndarray:
        if self.median is None or self.iqr is None:
            raise RuntimeError("RobustScalerLite must be fitted before calling inverse_y")
        z0 = (z - (self.hi + self.lo) / 2.0) / (self.hi - self.lo)
        return z0 * self.iqr + self.median


class ZScoreScalerLite:
    """Simple z‑score scaler for targets.

    This scaler applies ``(y - mean) / std`` to normalise the target variable.  When ``std`` is zero,
    it defaults to one to avoid division by zero.  The scaler also provides an ``inverse_y`` method
    to recover the original units.
    """
    def __init__(self, mean: float, std: float) -> None:
        self.mean = float(mean)
        self.std = float(std) if float(std) != 0.0 else 1.0

    def fit_y_chunk(self, y: np.ndarray) -> None:
        # no fitting needed for z‑score with precomputed stats
        pass

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / self.std

    def inverse_y(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean


def iter_dates(start_date: str, end_date: str) -> List[str]:
    d0 = pd.to_datetime(start_date)
    d1 = pd.to_datetime(end_date)
    return pd.date_range(d0, d1, freq='D').strftime(DATE_FMT).tolist()


def discover_pairs(root: str, dates: List[str], x_prefix: str, y_prefix: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
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
    """Daily time-series dataset for pollution modelling with auxiliary physical targets.

    In addition to the base time-series and observation target ``y``, this
    dataset can return two sets of physical variables: a primary vector
    ``z`` (e.g. PM2.5) and an auxiliary vector ``z_aux``.  These vectors are
    extracted from the last time step of the raw input and can be
    standardised using precomputed statistics (mean and std).  When
    statistics are not provided, the primary physical variables ``z`` are
    optionally scaled using a MinMax scaler, while the auxiliary variables
    ``z_aux`` are returned without scaling.  The observation target ``y``
    can be scaled via min/max, robust, or z‑score depending on
    configuration.
    """

    def __init__(
        self,
        root: str,
        start_date: str,
        end_date: str,
        x_prefix: str = 'TS_X_',
        y_prefix: str = 'TS_y_',
        drop_feature_indices: Optional[List[int]] = None,
        filter_y_positive: bool = True,
        remove_nan_rows: bool = True,
        scaler_range: Tuple[float, float] = (-1.0, 1.0),
        scaler_type: str = 'minmax',
        prefetch_files: int = 4,
        lat_feature_index: Optional[int] = None,
        lon_feature_index: Optional[int] = None,
        *,
        return_coords: bool = False,
        return_month: bool = False,
        time_variant: str = 'monthly',
        # indices for primary physical variables (e.g. PM25_TOT)
        z_feature_indices: Optional[List[int]] = None,
        # indices for auxiliary physical variables
        z_aux_feature_indices: Optional[List[int]] = None,
        # optional names for primary and auxiliary physical variables; used with z_stats
        z_feature_names: Optional[List[str]] = None,
        z_aux_feature_names: Optional[List[str]] = None,
        # dictionary of statistics {var_name: {"mean": x, "std": y}}
        z_stats: Optional[Dict[str, Dict[str, float]]] = None,
        # optional range for minmax scaling of z if stats not provided
        z_scaler_range: Optional[Tuple[float, float]] = None,
        # optional statistics for y (mean and std) for z‑score scaling
        y_stats: Optional[Dict[str, float]] = None,
        ncar_path: Optional[str] = None,
    ) -> None:
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
        # Determine temporal variant (monthly vs day-of-year)
        tv = str(time_variant or 'monthly').lower()
        self.time_variant = tv
        if 'doy' in tv:
            self.use_doy = True
            self.use_month = False
        elif 'month' in tv:
            self.use_month = True
            self.use_doy = False
        else:
            self.use_month = True
            self.use_doy = False

        # Primary and auxiliary physical variable indices
        self.z_feature_indices = list(z_feature_indices or [])
        self.z_aux_feature_indices = list(z_aux_feature_indices or [])
        self.return_z = len(self.z_feature_indices) > 0
        self.return_z_aux = len(self.z_aux_feature_indices) > 0

        # Names and stats for physical variables
        self.z_feature_names = list(z_feature_names or [])
        self.z_aux_feature_names = list(z_aux_feature_names or [])
        self.z_stats: Dict[str, Dict[str, float]] = z_stats or {}

        # Statistics for y (for z‑score scaling)
        self.y_stats = y_stats or {}

        # Mapping for NCAR sampling (set by dataset init)
        global _DEFAULT_TIME_VARIANT
        _DEFAULT_TIME_VARIANT = self.time_variant

        # Placeholder lists for mapping indices to physical stats arrays
        self.z_mean: Optional[np.ndarray] = None
        self.z_std: Optional[np.ndarray] = None
        self.z_aux_mean: Optional[np.ndarray] = None
        self.z_aux_std: Optional[np.ndarray] = None

        self.keep_feat_idx: Optional[List[int]] = None
        self.file_meta: List[Dict] = []

        # Scalers for features and targets
        x_scaler = MinMaxScalerLite(feature_range=scaler_range)
        # Choose target scaler based on requested type
        scaler_type_lower = (scaler_type or 'minmax').lower()
        if scaler_type_lower == 'robust':
            y_scaler = RobustScalerLite(feature_range=scaler_range)
        elif scaler_type_lower == 'zscore' or self.y_stats:
            # Use z‑score scaler for y when requested or when explicit stats provided
            mean = float(self.y_stats.get('mean', 0.0))
            std = float(self.y_stats.get('std', 1.0))
            if std == 0:
                std = 1.0
            y_scaler = ZScoreScalerLite(mean=mean, std=std)
        else:
            y_scaler = MinMaxScalerLite(feature_range=scaler_range)
        # Use separate scaler for z if stats are not provided; otherwise scaling will be via z_stats
        z_range = z_scaler_range if z_scaler_range is not None else scaler_range
        z_scaler = MinMaxScalerLite(feature_range=z_range) if (self.return_z and not self.z_stats) else None

        # Optional pre-fit on NCAR for z_scaler (only when z_stats not provided)
        self._z_prefit_on_ncar = False
        if self.return_z and z_scaler is not None and ncar_path:
            try:
                # load_ncar_grid and _NCAR_CACHE are defined later in this module
                load_ncar_grid(ncar_path)
                if 'z' in _NCAR_CACHE:
                    z_scaler.fit_y_chunk(_NCAR_CACHE['z'])
                    self._z_prefit_on_ncar = True
            except Exception:
                self._z_prefit_on_ncar = False

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
                # For z‑score scaling of y, no fitting needed; otherwise fit minmax/robust
                if not isinstance(y_scaler, ZScoreScalerLite):
                    y_scaler.fit_y_chunk(y[valid_idx])
                # Fit z scaler on valid rows if using MinMax and not pre-fitted on NCAR
                if self.return_z and z_scaler is not None and not self._z_prefit_on_ncar:
                    z_chunk = X[valid_idx][:, :, self.z_feature_indices]
                    z_scaler.fit_x_chunk(z_chunk)
            # Populate index map and auxiliary metadata
            for rid in valid_idx:
                index_map.append((fid, int(rid)))
                coords_list.append((float(latvec[rid]), float(lonvec[rid])))
                try:
                    date_str = self.dates[fid]
                    if self.use_doy:
                        t_idx = int(pd.to_datetime(date_str, format=DATE_FMT, errors='coerce').dayofyear)
                        if np.isnan(t_idx):
                            raise ValueError("Invalid date")
                    else:
                        t_idx = int(date_str.split("-")[1])
                except Exception:
                    t_idx = 1
                months_list.append(t_idx)
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

        # Precompute mean/std arrays for physical variables when statistics are provided
        if self.z_stats:
            if self.return_z and (self.z_feature_names or list(self.z_stats.keys())):
                means, stds = [], []
                # Use provided names to lookup means/stds.  Fall back to all stats in order.
                names = self.z_feature_names or list(self.z_stats.keys())[: len(self.z_feature_indices)]
                for nm in names:
                    st = self.z_stats.get(nm, None)
                    if st is None:
                        continue
                    mu = float(st.get('mean', 0.0))
                    sd = float(st.get('std', 1.0))
                    if sd == 0:
                        sd = 1.0
                    means.append(mu)
                    stds.append(sd)
                if means:
                    self.z_mean = np.array(means, dtype=np.float32)
                    self.z_std = np.array(stds, dtype=np.float32)
            if self.return_z_aux and (self.z_aux_feature_names or list(self.z_stats.keys())):
                a_means, a_stds = [], []
                names = self.z_aux_feature_names or list(self.z_stats.keys())[-len(self.z_aux_feature_indices):]
                for nm in names:
                    st = self.z_stats.get(nm, None)
                    if st is None:
                        continue
                    mu = float(st.get('mean', 0.0))
                    sd = float(st.get('std', 1.0))
                    if sd == 0:
                        sd = 1.0
                    a_means.append(mu)
                    a_stds.append(sd)
                if a_means:
                    self.z_aux_mean = np.array(a_means, dtype=np.float32)
                    self.z_aux_std = np.array(a_stds, dtype=np.float32)

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
        # y_scaler may be ZScoreScalerLite or MinMax/Robust
        y_row = self.y_scaler.transform_y(y_row).astype(np.float32)
        # Build output tuple dynamically
        out: List[np.ndarray | float | int] = [x_row]
        if self.return_coords:
            out.append(self.coords[idx].astype(np.float32))
        if self.return_month:
            out.append(self.months[idx].astype(np.int64))
        # Append primary physical variables (z)
        if self.return_z:
            z_raw = X[rid, -1, self.z_feature_indices].astype(np.float32)
            if self.z_mean is not None and self.z_std is not None:
                z_row = (z_raw - self.z_mean) / self.z_std
            elif self.z_scaler is not None:
                z_row = self.z_scaler.transform_x(z_raw[None, None, :]).reshape(-1)
            else:
                z_row = z_raw
            out.append(z_row.astype(np.float32))
        # Append auxiliary physical variables (z_aux)
        if self.return_z_aux:
            z_aux_raw = X[rid, -1, self.z_aux_feature_indices].astype(np.float32)
            if self.z_aux_mean is not None and self.z_aux_std is not None:
                z_aux_row = (z_aux_raw - self.z_aux_mean) / self.z_aux_std
            else:
                z_aux_row = z_aux_raw
            out.append(z_aux_row.astype(np.float32))
        out.append(y_row)
        return tuple(out)


# ---------------------------------------------------------------------------
# NCAR grid loading and random sampling (unchanged from original)
# These functions are reproduced here to make the dataset self‑contained.
_NCAR_CACHE: Dict[str, np.ndarray] = {}
_DEFAULT_TIME_VARIANT: str = 'monthly'


def load_ncar_grid(ncar_path: str) -> None:
    if 'lat_min' in _NCAR_CACHE:
        return
    files = sorted([os.path.join(ncar_path, f) for f in os.listdir(ncar_path) if f.endswith('.parquet')])
    dfs: List[pd.DataFrame] = []
    required = {
        'lat_nw', 'lon_nw', 'lat_ne', 'lon_ne', 'lat_se', 'lon_se',
        'lat_sw', 'lon_sw', 'PM25_TOT', 'date'
    }
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if df.empty or not required.issubset(df.columns):
            continue
        dfs.append(df[list(required)])
    if not dfs:
        return
    df_all = pd.concat(dfs, ignore_index=True)
    corners_lat = df_all[['lat_nw', 'lat_ne', 'lat_se', 'lat_sw']].to_numpy(dtype=float)
    corners_lon = df_all[['lon_nw', 'lon_ne', 'lon_se', 'lon_sw']].to_numpy(dtype=float)
    _NCAR_CACHE['lat_min'] = corners_lat.min(axis=1)
    _NCAR_CACHE['lat_max'] = corners_lat.max(axis=1)
    _NCAR_CACHE['lon_min'] = corners_lon.min(axis=1)
    _NCAR_CACHE['lon_max'] = corners_lon.max(axis=1)
    _NCAR_CACHE['z'] = df_all['PM25_TOT'].to_numpy(dtype=float)
    try:
        dt = pd.to_datetime(df_all['date'])
        months = dt.dt.month.to_numpy(dtype=int)
        doys = dt.dt.dayofyear.to_numpy(dtype=int)
    except Exception:
        months = np.ones(len(df_all), dtype=int)
        doys = np.ones(len(df_all), dtype=int)
    _NCAR_CACHE['month'] = months
    _NCAR_CACHE['doy'] = doys


def sample_ncar_points(num_samples: int, time_variant: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_samples <= 0 or 'lat_min' not in _NCAR_CACHE:
        return (
            np.zeros((0, 2), dtype=float),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
        )
    lat_min = _NCAR_CACHE['lat_min']
    lat_max = _NCAR_CACHE['lat_max']
    lon_min = _NCAR_CACHE['lon_min']
    lon_max = _NCAR_CACHE['lon_max']
    z_vals_full = _NCAR_CACHE['z']
    months_full = _NCAR_CACHE['month']
    doys_full = _NCAR_CACHE.get('doy', months_full)
    idxs = np.random.randint(0, len(lat_min), size=int(num_samples))
    lat_low = lat_min[idxs]
    lat_hi = lat_max[idxs]
    lon_low = lon_min[idxs]
    lon_hi = lon_max[idxs]
    lats = lat_low + np.random.rand(int(num_samples)) * (lat_hi - lat_low)
    lons = lon_low + np.random.rand(int(num_samples)) * (lon_hi - lon_low)
    coords = np.stack([lats, lons], axis=1)
    variant = str(time_variant or _DEFAULT_TIME_VARIANT or 'monthly').lower()
    if 'doy' in variant:
        time_full = doys_full
    else:
        time_full = months_full
    time_vals = time_full[idxs]
    z_vals = z_vals_full[idxs]
    return coords.astype(float), time_vals.astype(int), z_vals.astype(float)