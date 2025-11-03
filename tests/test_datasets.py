import os, numpy as np
from loc4pm.data.daily_dataset import DailyTSDataset

def _mk(tmp, date, n=5, t=3, f=10):
    y = date[:4]
    ddir = os.path.join(tmp, y)
    os.makedirs(ddir, exist_ok=True)
    X = np.random.rand(n, t, f).astype('float32')
    yv = np.abs(np.random.randn(n).astype('float32')) + 1.0
    np.save(os.path.join(ddir, f'TS_X_{date}.npy'), X)
    np.save(os.path.join(ddir, f'TS_y_{date}.npy'), yv)

def test_date_window(tmp_path):
    root = tmp_path.as_posix()
    _mk(root, '2018-01-01'); _mk(root, '2018-01-02'); _mk(root, '2018-01-03')
    ds = DailyTSDataset(root=root, start_date='2018-01-02', end_date='2018-01-03')
    assert len(ds) > 0
    used = {os.path.basename(m['x_path']) for m in ds.file_meta}
    assert 'TS_X_2018-01-01.npy' not in used

def test_scaling(tmp_path):
    root = tmp_path.as_posix()
    _mk(root, '2018-01-01')
    ds = DailyTSDataset(root=root, start_date='2018-01-01', end_date='2018-01-01')
    x, y = ds[0]
    assert x.dtype == np.float32 and y.dtype == np.float32
    assert np.all(x <= 1.0 + 1e-6) and np.all(x >= -1.0 - 1e-6)