import numpy as np
from loc4pm.utils.splits import random_holdout_indices, spatial_fold_indices

def test_random_reproducible():
    n = 100
    tr1, te1 = random_holdout_indices(n, 0.2, seed=123)
    tr2, te2 = random_holdout_indices(n, 0.2, seed=123)
    assert np.array_equal(te1, te2)

def test_spatial_fold_mapping():
    # 4 unique coords, 20 samples (5 per coord)
    coords = np.repeat(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float), 5, axis=0)
    tr, te = spatial_fold_indices(coords, n_splits=2, fold_index=1, seed=42)
    # every sample from a given coord should be together (no leakage)
    # check that TE contains only whole groups of 5
    assert len(te) % 5 == 0