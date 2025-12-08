"""Daily time-series dataset with optional geographic and physical targets.

This dataset loads daily time-series samples from NumPy files, applies
feature scaling, and supports returning geographic coordinates, temporal indices
such as month or day-of-year, and additional physical simulation variables. It
is an enhancement of the original ``DailyTSDataset`` to support dual targets
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


# ---------------------------------------------------------------------------
# Additional scaler: RobustScalerLite
#
# This scaler uses the median and inter‑quartile range (IQR) to perform
# robust scaling on targets.  It is particularly useful for variables like
# PM2.5 concentrations that exhibit strong right‑skew and outliers.  Values
# are centred around the median and scaled by the IQR, then mapped into a
# specified range.  See ``RobustScalerLite.transform_y`` for details.
# ---------------------------------------------------------------------------

class RobustScalerLite:
    """A robust scaler using the median and IQR.

    This scaler scales targets based on their median and inter‑quartile range (IQR), which makes it more
    robust to outliers and right‑skewed distributions compared to a simple min/max scaler.  The scaled
    values are linearly mapped into a specified range.  For example, with ``feature_range=(-1, 1)``,
    the first quartile (25th percentile) of the data will map to approximately ``-0.5`` and the third
    quartile (75th percentile) will map to ``0.5``.

    Parameters
    ----------
    feature_range: tuple of (low, high)
        Desired range of transformed data.  Defaults to (-1.0, 1.0).
    """
    def __init__(self, feature_range: Tuple[float, float] = (-1.0, 1.0)) -> None:
        self.median: Optional[float] = None
        self.iqr: Optional[float] = None
        self.lo, self.hi = feature_range

    def fit_y_chunk(self, y: np.ndarray) -> None:
        """Update the scaler's median and IQR using a chunk of target values.

        Args
        ----
        y: Array of shape [N] or [N,1] containing target values.
        """
        # flatten and convert to float
        y_flat = y.reshape(-1).astype(float)
        med = float(np.nanmedian(y_flat))
        q1 = float(np.nanpercentile(y_flat, 25))
        q3 = float(np.nanpercentile(y_flat, 75))
        iqr = q3 - q1
        # avoid zero IQR
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
        """Scale target values using the robust statistics.

        Values are first centered around the median and scaled by the IQR, then mapped into the
        specified ``feature_range``.  Note that the lower and upper quartiles map to around
        ``0.25*(hi - lo) + lo`` and ``0.75*(hi - lo) + lo``, respectively.
        """
        if self.median is None or self.iqr is None:
            raise RuntimeError("RobustScalerLite must be fitted before calling transform_y")
        # center and scale by IQR: q1→-0.5, q3→0.5
        z = (y - self.median) / self.iqr
        # map from [-0.5, 0.5] to [lo, hi]
        return z * (self.hi - self.lo) + (self.hi + self.lo) / 2.0

    def inverse_y(self, z: np.ndarray) -> np.ndarray:
        if self.median is None or self.iqr is None:
            raise RuntimeError("RobustScalerLite must be fitted before calling inverse_y")
        # map back to centred and scaled values
        z0 = (z - (self.hi + self.lo) / 2.0) / (self.hi - self.lo)
        return z0 * self.iqr + self.median


# New robust scaler will be appended later via patch


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
    coordinates, temporal indices (month or day-of-year), and additional physical simulation targets.
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
                 scaler_type: str = 'minmax',
                 prefetch_files: int = 4,
                 lat_feature_index: Optional[int] = None,
                 lon_feature_index: Optional[int] = None,
                 *,
                 return_coords: bool = False,
                 return_month: bool = False,
                 time_variant: str = 'monthly',
                 z_feature_indices: Optional[List[int]] = None,
                 z_scaler_range: Optional[Tuple[float, float]] = None,
                 ncar_path: Optional[str] = None) -> None:
        """Instantiate the daily dataset.

        Parameters
        ----------
        return_coords: bool
            Whether to return (lat, lon) coordinates for each sample.
        return_month: bool
            If True, return a temporal index (month or day-of-year) for each sample.
        time_variant: str
            Determines how the temporal index is interpreted.  ``'monthly'`` returns
            month indices (1-12); ``'doy'`` returns day-of-year indices (1-365/366).
        """
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
        # ``return_month`` indicates whether a temporal index (month or day-of-year)
        # should be returned.  The interpretation of the index is controlled by
        # ``time_variant`` (see below).  We keep the original name for backwards
        # compatibility.
        self.return_month = bool(return_month)
        # Determine how to interpret temporal indices.  If ``time_variant`` is
        # 'doy', day-of-year indices (1-365/366) will be used; otherwise month
        # indices (1-12) are used.
        # Interpret the requested temporal variant.  Any string containing
        # ``"doy"`` (case insensitive) will enable day‑of‑year indices, while
        # strings containing ``"month"`` or ``"monthly"`` enable month
        # indices.  Otherwise, default to months.
        tv = str(time_variant or 'monthly').lower()
        self.time_variant = tv
        if 'doy' in tv:
            self.use_doy = True
            self.use_month = False
        elif 'month' in tv:
            self.use_month = True
            self.use_doy = False
        else:
            # default to month if unspecified
            self.use_month = True
            self.use_doy = False
        self.z_feature_indices = list(z_feature_indices or [])
        self.return_z = len(self.z_feature_indices) > 0

        # Update the module-level default variant for NCAR sampling.  This allows
        # ``sample_ncar_points`` to return day-of-year indices when the dataset
        # was created with ``time_variant='doy'`` but the caller does not
        # explicitly pass a ``time_variant``.  Without this, NCAR sampling
        # during training would always default to months.
        global _DEFAULT_TIME_VARIANT
        _DEFAULT_TIME_VARIANT = self.time_variant

        self.keep_feat_idx: Optional[List[int]] = None
        self.file_meta: List[Dict] = []
        x_scaler = MinMaxScalerLite(feature_range=scaler_range)
        # Choose target scaler based on requested type; robust scaling can better handle right-skewed PM2.5
        scaler_type_lower = (scaler_type or 'minmax').lower()
        if scaler_type_lower == 'robust':
            y_scaler = RobustScalerLite(feature_range=scaler_range)
        else:
            y_scaler = MinMaxScalerLite(feature_range=scaler_range)
        # Use separate scaler for z if provided; fallback to same range
        z_range = z_scaler_range if z_scaler_range is not None else scaler_range
        z_scaler = MinMaxScalerLite(feature_range=z_range) if self.return_z else None

        # Optionally pre-fit the z scaler on the full NCAR PM25_TOT distribution.
        # This uses all values from the parquet files (via ``load_ncar_grid``)
        # instead of only the EO rows present in this dataset.
        self._z_prefit_on_ncar = False
        if self.return_z and z_scaler is not None and ncar_path:
            try:
                # load_ncar_grid and _NCAR_CACHE are defined later in this module
                load_ncar_grid(ncar_path)
                if 'z' in _NCAR_CACHE:
                    # _NCAR_CACHE['z'] is df_all['PM25_TOT'].to_numpy(...)
                    z_scaler.fit_y_chunk(_NCAR_CACHE['z'])
                    self._z_prefit_on_ncar = True
            except Exception:
                # If anything goes wrong, we fall back to fitting on EO samples below.
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
                y_scaler.fit_y_chunk(y[valid_idx])
                # Fit z scaler on valid rows only if we did NOT already pre-fit on NCAR.
                if self.return_z and z_scaler is not None and not self._z_prefit_on_ncar:
                    # Extract z_chunk from raw X for all time steps to capture variability, shape [N,T,len(z)]
                    z_chunk = X[valid_idx][:, :, self.z_feature_indices]
                    # For scaling, treat z as features with a dummy time dimension
                    z_scaler.fit_x_chunk(z_chunk)
            # Populate index map and auxiliary metadata
            for rid in valid_idx:
                index_map.append((fid, int(rid)))
                coords_list.append((float(latvec[rid]), float(lonvec[rid])))
                # compute the temporal index corresponding to this file/date
                try:
                    date_str = self.dates[fid]
                    if self.use_doy:
                        # Use pandas to compute day-of-year; fallback to 1 if parsing fails
                        t_idx = int(pd.to_datetime(date_str, format=DATE_FMT, errors='coerce').dayofyear)
                        # pandas returns NaN if parsing fails; treat as invalid
                        if np.isnan(t_idx):
                            raise ValueError("Invalid date")
                    else:
                        # month (1-12)
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

# ---------------------------------------------------------------------------
# NCAR grid loading and random sampling
#
# The following functions provide a lightweight interface for loading NCAR
# reanalysis parquet files and sampling random latitude/longitude points
# uniformly within each grid cell.  They are intended to be used in
# ``train.py`` when augmenting training batches with synthetic samples.

_NCAR_CACHE: Dict[str, np.ndarray] = {}

# Global default temporal variant used by ``sample_ncar_points`` when no explicit
# ``time_variant`` argument is provided.  This value is updated in
# ``DailyTSDataset.__init__`` based on the ``time_variant`` passed to the
# dataset constructor.  If no dataset has been instantiated, it defaults
# to 'monthly'.
_DEFAULT_TIME_VARIANT: str = 'monthly'


def load_ncar_grid(ncar_path: str) -> None:
    """Load NCAR parquet files and cache arrays for random sampling.

    This function reads all ``*.parquet`` files under ``ncar_path`` and
    extracts the minimum and maximum latitudes and longitudes for each
    grid cell, the PM25_TOT target and both month and day-of-year indices
    derived from the ``date`` column.  The resulting arrays are stored in a
    module-level cache so that subsequent calls to ``sample_ncar_points``
    can reuse them without reloading the parquet files.
    """
    if 'lat_min' in _NCAR_CACHE:
        return  # already loaded
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
    """Sample random NCAR points uniformly within grid cells.

    Parameters
    ----------
    num_samples: int
        Number of random points to sample.  If zero or the NCAR cache is
        empty, returns empty arrays.
    time_variant: str
        Either 'monthly' or 'doy'.  Determines whether month (1-12) or day-of-year
        (1-365/366) indices are returned for the sampled points.

    Returns
    -------
    coords : ndarray of shape [N,2]
        Sampled latitude/longitude coordinates (lat, lon).
    time_vals : ndarray of shape [N]
        Temporal indices corresponding to the sampled points.
    z_vals : ndarray of shape [N]
        Physical target values (PM25_TOT) for the sampled points.
    """
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
    # Randomly select rows
    idxs = np.random.randint(0, len(lat_min), size=int(num_samples))
    lat_low = lat_min[idxs]
    lat_hi = lat_max[idxs]
    lon_low = lon_min[idxs]
    lon_hi = lon_max[idxs]
    lats = lat_low + np.random.rand(int(num_samples)) * (lat_hi - lat_low)
    lons = lon_low + np.random.rand(int(num_samples)) * (lon_hi - lon_low)
    coords = np.stack([lats, lons], axis=1)
    # Choose the appropriate temporal indices based on time_variant.  If none
    # provided, fall back to the module-level default set by
    # ``DailyTSDataset.__init__``.
    variant = str(time_variant or _DEFAULT_TIME_VARIANT or 'monthly').lower()
    # Interpret any variant containing "doy" as a day‑of‑year selection; otherwise
    # default to months.  This accommodates variants like "doy-hadamard".
    if 'doy' in variant:
        time_full = doys_full
    else:
        time_full = months_full
    time_vals = time_full[idxs]
    z_vals = z_vals_full[idxs]
    return coords.astype(float), time_vals.astype(int), z_vals.astype(float)