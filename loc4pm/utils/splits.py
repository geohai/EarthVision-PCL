import numpy as np

__all__ = ["random_holdout_indices", "spatial_fold_indices", "create_spatial_folds_legacy"]

def random_holdout_indices(n_total: int, test_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    n_test = int(round(n_total * float(test_frac)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx

def spatial_fold_indices(coords, n_splits: int, fold_index: int, seed: int):
    """Assign unique (lat,lon) to folds reproducibly, return indices for the requested fold as test.
    coords: array-like [N,2]
    """
    coords = np.asarray(coords)
    # build unique locations map
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

# Backward-compatible helper based on the user's legacy function.
def create_spatial_folds_legacy(lat, lon, n_splits=10, random_state=42):
    np.random.seed(random_state)
    unique_locs = np.unique(np.stack([lat, lon], axis=1), axis=0)
    np.random.shuffle(unique_locs)
    fold_assignments = np.arange(len(unique_locs)) % n_splits
    fold_mapping = { tuple(loc): fold for loc, fold in zip(unique_locs, fold_assignments) }
    folds = np.array([fold_mapping[tuple(loc)] for loc in np.stack([lat, lon], axis=1)])
    return folds