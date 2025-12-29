"""
Utility functions for generating train/test splits.

This module extends the existing random and spatial fold split utilities by
introducing a checkerboard based spatial split.  Checkerboard splits divide
the spatial extent into a regular grid of a specified degree (in degrees
latitude/longitude) and assign fold identifiers to each grid cell in a
checkerboard pattern.  All samples within the same grid cell belong to the
same fold.  A single fold is then held out for testing while the remaining
folds form the training pool.  This approach provides a stricter spatial
separation than the point‑based split used previously and follows the
methodology described in recent GeoAI evaluation papers.

Functions
---------
random_holdout_indices(n_total, test_frac, seed)
    Randomly assigns a fraction of indices to the test set.

spatial_fold_indices(coords, n_splits, fold_index, seed)
    Assigns samples to spatial folds based on unique coordinates.

checkerboard_deg_fold_indices(coords, grid_deg, n_splits, fold_index, scale)
    Assigns samples to folds using a checkerboard grid with cell size
    ``grid_deg`` degrees.  The spatial extent can be either global or
    constrained to the bounding box of the input coordinates.

Notes
-----
The checkerboard pattern cycles through ``n_splits`` fold identifiers by
computing ``(row + column) % n_splits`` for each grid cell.  Rows and
columns are derived from zero‑based indices of the latitude and longitude
coordinates within the defined spatial extent.  A global extent uses the
full range of latitudes (‑90 to 90) and longitudes (‑180 to 180); a
regional extent uses the min/max bounds of the provided coordinates.
"""

import numpy as np
from functools import lru_cache

__all__ = [
    "random_holdout_indices",
    "spatial_fold_indices",
    "checkerboard_deg_fold_indices",
    "create_spatial_folds_legacy",
]

_CONUS_GEOJSON_URL_DEFAULT = (
    "https://github.com/mapbox/mapboxgl-jupyter/raw/refs/heads/master/examples/data/us-states.geojson"
)
_CONUS_EXCLUDE_IDS_DEFAULT = ("02", "15", "72")  # AK, HI, PR


def random_holdout_indices(n_total: int, test_frac: float, seed: int):
    """Randomly partition indices into train and test sets.

    Parameters
    ----------
    n_total : int
        Total number of samples.
    test_frac : float
        Fraction of samples to assign to the test set.
    seed : int
        Random seed used for reproducibility.

    Returns
    -------
    tuple of (train_indices, test_indices)
        Arrays of indices corresponding to the training and test sets.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    n_test = int(round(n_total * float(test_frac)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def spatial_fold_indices(coords, n_splits: int, fold_index: int, seed: int):
    """Assign unique (lat, lon) pairs to folds reproducibly and return train/test indices.

    Parameters
    ----------
    coords : array‑like of shape (N, 2)
        Latitude and longitude coordinates for each sample.
    n_splits : int
        Number of spatial folds.
    fold_index : int
        Index of the fold to be used as the test set (0‑based).
    seed : int
        Random seed used to shuffle unique locations.

    Returns
    -------
    tuple of (train_indices, test_indices)
        Arrays of indices for training and testing, respectively.
    """
    coords = np.asarray(coords)
    uniq, inv = np.unique(coords, axis=0, return_inverse=True)
    rng = np.random.RandomState(seed)
    order = np.arange(len(uniq))
    rng.shuffle(order)
    fold_id_for_loc = np.zeros(len(uniq), dtype=int)
    fold_id_for_loc[order] = np.arange(len(uniq)) % int(n_splits)
    sample_fold = fold_id_for_loc[inv]
    test_mask = (sample_fold == int(fold_index))
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx


@lru_cache(maxsize=4)
def _get_conus_bbox_from_geojson(
    geojson_url: str = _CONUS_GEOJSON_URL_DEFAULT,
    exclude_ids: tuple[str, ...] = _CONUS_EXCLUDE_IDS_DEFAULT,
) -> tuple[float, float, float, float]:
    """
    Returns (min_lon, min_lat, max_lon, max_lat) from the CONUS boundary GeoJSON.
    Cached so the URL is fetched/read only once per process.
    """
    import geopandas as gpd  # lazy import to avoid hard dependency at import time

    gdf = gpd.read_file(geojson_url)
    if "id" in gdf.columns and exclude_ids:
        gdf = gdf[~gdf["id"].isin(list(exclude_ids))]

    gdf = gdf.to_crs(epsg=4326)
    minx, miny, maxx, maxy = gdf.total_bounds  # (min_lon, min_lat, max_lon, max_lat)
    return float(minx), float(miny), float(maxx), float(maxy)


def checkerboard_deg_fold_indices(
    coords: np.ndarray,
    grid_deg: float,
    n_splits: int,
    fold_index: int,
    scale: str = "global",
    *,
    conus_geojson_url: str = _CONUS_GEOJSON_URL_DEFAULT,
    conus_exclude_ids: tuple[str, ...] = _CONUS_EXCLUDE_IDS_DEFAULT,
    conus_bbox: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign samples to folds using a checkerboard grid of fixed degree.

    scale:
      - 'global': use (-90..90, -180..180)
      - 'conus':  use CONUS bbox from GeoJSON (or conus_bbox if provided)
      - other:    fallback to data bbox
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be of shape (N,2)")
    lat = coords[:, 0]
    lon = coords[:, 1]

    scale_lower = str(scale or "global").lower()

    def _data_bbox():
        valid_lat = lat[~np.isnan(lat)]
        valid_lon = lon[~np.isnan(lon)]
        min_lat = valid_lat.min() if valid_lat.size else -90.0
        max_lat = valid_lat.max() if valid_lat.size else 90.0
        min_lon = valid_lon.min() if valid_lon.size else -180.0
        max_lon = valid_lon.max() if valid_lon.size else 180.0
        return min_lon, min_lat, max_lon, max_lat

    if scale_lower == "global":
        min_lat, max_lat = -90.0, 90.0
        min_lon, max_lon = -180.0, 180.0

    elif scale_lower in ("conus", "conus_geojson", "conus-geojson"):
        if conus_bbox is not None:
            # expects (min_lon, min_lat, max_lon, max_lat)
            min_lon, min_lat, max_lon, max_lat = map(float, conus_bbox)
        else:
            try:
                min_lon, min_lat, max_lon, max_lat = _get_conus_bbox_from_geojson(
                    geojson_url=conus_geojson_url,
                    exclude_ids=conus_exclude_ids,
                )
            except Exception:
                # geopandas missing, no network, etc. -> fallback to data bbox
                min_lon, min_lat, max_lon, max_lat = _data_bbox()

    else:
        min_lon, min_lat, max_lon, max_lat = _data_bbox()

    with np.errstate(invalid="ignore"):
        row = np.floor((lat - min_lat) / float(grid_deg)).astype(int)
        col = np.floor((lon - min_lon) / float(grid_deg)).astype(int)

    row[np.isnan(lat) | np.isnan(row)] = -1
    col[np.isnan(lon) | np.isnan(col)] = -1

    fold_ids = np.full(len(coords), -1, dtype=int)
    valid_mask = (row >= 0) & (col >= 0)
    fold_ids[valid_mask] = (row[valid_mask] + col[valid_mask]) % int(n_splits)

    fold_index = int(fold_index)
    if fold_index < 0 or fold_index >= int(n_splits):
        raise ValueError(f"fold_index must be within [0, {n_splits - 1}]")

    test_mask = (fold_ids == fold_index)
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx


def create_spatial_folds_legacy(lat, lon, n_splits=10, random_state=42):
    """Backward‑compatible helper based on the user's legacy function.

    This function assigns samples to folds based on unique (lat, lon) pairs,
    shuffling the unique locations using a fixed random seed.  It is included
    here for backward compatibility and should be replaced by
    ``spatial_fold_indices`` in new code.
    """
    np.random.seed(random_state)
    unique_locs = np.unique(np.stack([lat, lon], axis=1), axis=0)
    np.random.shuffle(unique_locs)
    fold_assignments = np.arange(len(unique_locs)) % n_splits
    fold_mapping = {tuple(loc): fold for loc, fold in zip(unique_locs, fold_assignments)}
    folds = np.array([fold_mapping[tuple(loc)] for loc in np.stack([lat, lon], axis=1)])
    return folds