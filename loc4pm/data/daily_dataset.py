import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from torch.utils.data import Dataset

DATE_FMT = "%Y-%m-%d"

class MinMaxScalerLite:
    def __init__(self, feature_range=(-1.0, 1.0)):
        self.min = None  # shape [F] for X, scalar for y
        self.max = None
        self.lo, self.hi = feature_range
    def fit_x_chunk(self, X):
        n,t,f = X.shape
        X2 = X.reshape(n*t, f)
        mn = np.nanmin(X2, axis=0)
        mx = np.nanmax(X2, axis=0)
        if self.min is None:
            self.min, self.max = mn, mx
        else:
            self.min = np.minimum(self.min, mn)
            self.max = np.maximum(self.max, mx)
    def fit_y_chunk(self, y):
        vmin = float(np.nanmin(y))
        vmax = float(np.nanmax(y))
        if self.min is None:
            self.min, self.max = vmin, vmax
        else:
            self.min = min(self.min, vmin)
            self.max = max(self.max, vmax)
    def transform_x(self, X):
        den = (self.max - self.min)
        den[den == 0] = 1.0
        Z = (X - self.min) / den
        return Z * (self.hi - self.lo) + self.lo
    def transform_y(self, y):
        den = (self.max - self.min) or 1.0
        z = (y - self.min) / den
        return z * (self.hi - self.lo) + self.lo
    def inverse_y(self, z):
        den = (self.max - self.min) or 1.0
        y = (z - self.lo) / (self.hi - self.lo) * den + self.min
        return y

def iter_dates(start_date: str, end_date: str) -> List[str]:
    d0 = pd.to_datetime(start_date)
    d1 = pd.to_datetime(end_date)
    return pd.date_range(d0, d1, freq='D').strftime(DATE_FMT).tolist()

def discover_pairs(root: str, dates: List[str], x_prefix: str, y_prefix: str) -> List[Tuple[str,str]]:
    pairs = []
    for ds in dates:
        yyyy = ds[:4]
        x = os.path.join(root, yyyy, f"{x_prefix}{ds}.npy")
        y = os.path.join(root, yyyy, f"{y_prefix}{ds}.npy")
        if os.path.exists(x) and os.path.exists(y):
            pairs.append((x,y))
    if not pairs:
        raise FileNotFoundError(f"No daily pairs found in {root} for {dates[0]}..{dates[-1]}")
    return pairs

class DailyTSDataset(Dataset):
    def __init__(self,
                 root: str,
                 start_date: str,
                 end_date: str,
                 x_prefix: str = 'TS_X_',
                 y_prefix: str = 'TS_y_',
                 drop_feature_indices: List[int] = None,
                 filter_y_positive: bool = True,
                 remove_nan_rows: bool = True,
                 scaler_range = (-1.0, 1.0),
                 prefetch_files: int = 4,
                 lat_feature_index: int = None,
                 lon_feature_index: int = None,
                 *,
                 return_coords: bool = False,
                 return_month: bool = False):
        self.root = root
        self.dates = iter_dates(start_date, end_date)
        self.pairs = discover_pairs(root, self.dates, x_prefix, y_prefix)
        self.drop_idxs = set(drop_feature_indices or [])
        self.filter_y_positive = filter_y_positive
        self.remove_nan_rows = remove_nan_rows
        self.prefetch_files = max(1, int(prefetch_files))
        self.lat_idx = lat_feature_index
        self.lon_idx = lon_feature_index

        # Flags to control whether geographic coordinates and month indices are returned from
        # __getitem__. When enabled, the __getitem__ signature changes to include these
        # auxiliary fields before the target value.
        self.return_coords = bool(return_coords)
        self.return_month = bool(return_month)

        self.keep_feat_idx = None
        self.file_meta: List[Dict] = []
        x_scaler = MinMaxScalerLite(feature_range=scaler_range)
        y_scaler = MinMaxScalerLite(feature_range=scaler_range)

        index_map = []
        coords = []
        months = []
        for fid, (xp, yp) in enumerate(self.pairs):
            X = np.load(xp)  # [N,T,F]
            y = np.load(yp)  # [N] or [N,1]
            if y.ndim == 1:
                y = y.reshape(-1,1)
            N, T, F = X.shape

            # record mapping + coords (if indices provided)
            if self.lat_idx is not None and self.lon_idx is not None and self.lat_idx < F and self.lon_idx < F:
                latvec = X[:, 0, self.lat_idx]
                lonvec = X[:, 0, self.lon_idx]
            else:
                latvec = np.full((N,), np.nan, dtype=float)
                lonvec = np.full((N,), np.nan, dtype=float)

            if self.keep_feat_idx is None:
                self.keep_feat_idx = [i for i in range(F) if i not in self.drop_idxs]
            Xk = X[:,:, self.keep_feat_idx]

            m = np.ones((N,), dtype=bool)
            if self.filter_y_positive:
                m &= (y.reshape(-1) > 0)
            if self.remove_nan_rows:
                m &= ~np.isnan(Xk).any(axis=(1,2))

            valid_idx = np.where(m)[0]
            if valid_idx.size:
                x_scaler.fit_x_chunk(Xk[valid_idx])
                y_scaler.fit_y_chunk(y[valid_idx])

            for rid in valid_idx:
                index_map.append((fid, int(rid)))
                coords.append((float(latvec[rid]), float(lonvec[rid])))
                # compute the month (1-12) corresponding to this file/date
                try:
                    date_str = self.dates[fid]
                    m = int(date_str.split("-")[1])
                except Exception:
                    m = 1
                months.append(m)

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
        self.index_map = index_map
        self.coords = np.asarray(coords, dtype=float)  # shape [M,2]
        self.months = np.asarray(months, dtype=int)

        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []

    def __len__(self):
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

    def __getitem__(self, idx):
        fid, rid = self.index_map[idx]
        meta = self.file_meta[fid]
        X = self._get_from_cache(meta['x_path'])
        y = self._get_from_cache(meta['y_path'])
        if y.ndim == 1:
            y = y.reshape(-1,1)
        x_row = X[rid][:, self.keep_feat_idx]
        y_row = y[rid].astype(np.float32)
        # scale inputs/target
        x_row = self.x_scaler.transform_x(x_row).astype(np.float32)
        y_row = self.y_scaler.transform_y(y_row).astype(np.float32)
        # When neither coords nor month are requested, preserve the original API
        if not self.return_coords and not self.return_month:
            return x_row, y_row
        # build output tuple dynamically
        out = [x_row]
        if self.return_coords:
            out.append(self.coords[idx].astype(np.float32))
        if self.return_month:
            out.append(self.months[idx].astype(np.int64))
        out.append(y_row)
        return tuple(out)
