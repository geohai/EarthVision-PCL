"""Training script for LOC4PM with optional location encoders and NCAR augmentation.

This script orchestrates the training of BiLSTM‑based pollution models with
optional attention, location fusion, and physical simulation heads.  In
addition to the original observation targets, it supports augmenting the
training batches with randomly sampled NCAR reanalysis points to better
exploit the gapless simulation data.  The number of random NCAR points
sampled per batch is controlled via ``train.ncar_random_ratio`` in the
configuration.  When enabled, the physical loss is normalized by the
combined number of NCAR and observation samples to balance the
contributions from supervised and unsupervised data.
"""

from __future__ import annotations

import os
import yaml
import json
import joblib
import time
import datetime
import hashlib
import logging
import re
import contextlib
import glob

import pandas as pd
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from .data.daily_dataset import DailyTSDataset, load_ncar_grid as _load_ncar_grid_ds, sample_ncar_points as _sample_ncar_points_ds
from .models.bilstm_attn import BiLSTMAttnRegressor
from .models.bilstm_attn_fusion import BiLSTMAttnLocRegressor
from .utils.export import save_predictions_and_metrics
from .utils.splits import random_holdout_indices, spatial_fold_indices


_PAT = re.compile(r"\$\{([^}]+)\}")

# -----------------------------------------------------------------------------
# NCAR reanalysis augmentation helpers
#
# When ``train.ncar_random_ratio`` in the configuration is positive, the
# training loop will sample additional coordinate/target pairs from the NCAR
# reanalysis dataset.  These points are drawn uniformly from within each grid
# cell (defined by four corner coordinates) and use the PM25_TOT value as
# the physical target.  The augmentation is optional and disabled by default.

_NCAR_DF: Optional[pd.DataFrame] = None
_NCAR_LAT_MIN: Optional[np.ndarray] = None
_NCAR_LAT_MAX: Optional[np.ndarray] = None
_NCAR_LON_MIN: Optional[np.ndarray] = None
_NCAR_LON_MAX: Optional[np.ndarray] = None
_NCAR_Z: Optional[np.ndarray] = None
_NCAR_MONTH: Optional[np.ndarray] = None


def _load_ncar_grid(ncar_path: str) -> None:
    """Load NCAR parquet files and prepare arrays for random sampling.

    This function populates module‑level arrays used by
    ``_sample_ncar_points``.  It expects parquet files with columns
    ``['lat_nw','lon_nw','lat_ne','lon_ne','lat_se','lon_se','lat_sw','lon_sw',
    'PM25_TOT','date']``.  The month index is extracted from the ``date``
    column (assumed to be parseable via ``pandas.to_datetime``).

    Parameters
    ----------
    ncar_path: str
        Directory containing parquet files for each month.  All files
        matching ``*.parquet`` will be loaded.
    """
    global _NCAR_DF, _NCAR_LAT_MIN, _NCAR_LAT_MAX, _NCAR_LON_MIN, _NCAR_LON_MAX, _NCAR_Z, _NCAR_MONTH

    if _NCAR_DF is not None:
        # Already loaded
        return
    files = sorted(glob.glob(os.path.join(ncar_path, "*.parquet")))
    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            logging.warning("Failed to load NCAR parquet %s: %s", f, e)
            continue
        if df.empty:
            continue
        # Ensure required columns exist
        required = {
            'lat_nw', 'lon_nw', 'lat_ne', 'lon_ne', 'lat_se', 'lon_se', 'lat_sw', 'lon_sw',
            'PM25_TOT', 'date'
        }
        if not required.issubset(df.columns):
            missing = required.difference(set(df.columns))
            logging.warning("Missing NCAR columns %s in file %s", missing, f)
            continue
        dfs.append(df[list(required)])
    if not dfs:
        logging.warning("No NCAR parquet files loaded from %s", ncar_path)
        return
    _NCAR_DF = pd.concat(dfs, ignore_index=True)
    # Compute bounding boxes and physical targets
    corners_lat = _NCAR_DF[['lat_nw', 'lat_ne', 'lat_se', 'lat_sw']].to_numpy(dtype=float)
    corners_lon = _NCAR_DF[['lon_nw', 'lon_ne', 'lon_se', 'lon_sw']].to_numpy(dtype=float)
    _NCAR_LAT_MIN = corners_lat.min(axis=1)
    _NCAR_LAT_MAX = corners_lat.max(axis=1)
    _NCAR_LON_MIN = corners_lon.min(axis=1)
    _NCAR_LON_MAX = corners_lon.max(axis=1)
    _NCAR_Z = _NCAR_DF['PM25_TOT'].to_numpy(dtype=float)
    try:
        _NCAR_MONTH = pd.to_datetime(_NCAR_DF['date']).dt.month.to_numpy(dtype=int)
    except Exception:
        # Fallback: parse month from the parquet filename or default to 1
        _NCAR_MONTH = np.ones(len(_NCAR_DF), dtype=int)


def _sample_ncar_points(num_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample random NCAR points uniformly within grid cells.

    Parameters
    ----------
    num_samples: int
        Number of random points to sample.

    Returns
    -------
    tuple of (coords, month, z):
        ``coords`` is an array of shape [N,2] with latitude and longitude;
        ``month`` is an array of shape [N] with month indices (1‑12);
        ``z`` is an array of shape [N] containing the physical target PM25_TOT.
    """
    if num_samples <= 0 or _NCAR_LAT_MIN is None:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)
    # Randomly select rows from the NCAR data
    idxs = np.random.randint(0, len(_NCAR_LAT_MIN), size=num_samples)
    lat_min = _NCAR_LAT_MIN[idxs]
    lat_max = _NCAR_LAT_MAX[idxs]
    lon_min = _NCAR_LON_MIN[idxs]
    lon_max = _NCAR_LON_MAX[idxs]
    # Uniformly sample within the bounding box
    lats = lat_min + np.random.rand(num_samples) * (lat_max - lat_min)
    lons = lon_min + np.random.rand(num_samples) * (lon_max - lon_min)
    z_vals = _NCAR_Z[idxs]
    months = _NCAR_MONTH[idxs] if _NCAR_MONTH is not None else np.ones(num_samples, dtype=int)
    coords = np.stack([lats, lons], axis=1)
    return coords.astype(float), months.astype(int), z_vals.astype(float)


# ------------------------------- helpers ---------------------------------- #

def _deep_get(dic, dotted):
    cur = dic
    for k in dotted.split('.'):
        cur = cur[k]
    return cur


def _expand_all_strings(obj, cfg):
    if isinstance(obj, dict):
        return {k: _expand_all_strings(v, cfg) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_all_strings(v, cfg) for v in obj]
    if isinstance(obj, str):
        def repl(m):
            key = m.group(1)
            try:
                return str(_deep_get(cfg, key))
            except Exception:
                return m.group(0)  # leave as‑is if missing
        return _PAT.sub(repl, obj)
    return obj


def _expand_placeholders(s: str, cfg: dict) -> str:
    if not isinstance(s, str):
        return s
    pat = re.compile(r"\$\{([^}]+)\}")
    def repl(m):
        key = m.group(1)
        try:
            return str(_deep_get(cfg, key))
        except Exception:
            return m.group(0)
    return pat.sub(repl, s)


def _expand_cfg_strings(cfg: dict) -> None:
    if 'logging' in cfg and 'run_name' in cfg['logging']:
        cfg['logging']['run_name'] = _expand_placeholders(cfg['logging']['run_name'], cfg)
    if 'logging' in cfg:
        for k in ('epoch_csv', 'step_csv'):
            if k in cfg['logging']:
                cfg['logging'][k] = _expand_placeholders(cfg['logging'][k], cfg)
    if 'eval' in cfg:
        for k in ('out_csv_val', 'out_csv_test'):
            if k in cfg['eval']:
                cfg['eval'][k] = _expand_placeholders(cfg['eval'][k], cfg)


def _parse_overrides(pairs):
    def cast(v):
        if isinstance(v, str):
            lv = v.lower()
            if lv in ("true", "false"):
                return lv == "true"
            try:
                if "." in v:
                    return float(v)
                return int(v)
            except ValueError:
                return v
        return v
    kv = {}
    if not pairs:
        return kv
    if len(pairs) % 2 != 0:
        raise ValueError("Overrides must be KEY VALUE pairs")
    for i in range(0, len(pairs), 2):
        kv[pairs[i]] = cast(pairs[i + 1])
    return kv


def _nested_set(dic, keys, value):
    ks = keys.split('.')
    cur = dic
    for k in ks[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[ks[-1]] = value


def _apply_overrides(cfg, updates):
    for k, v in updates.items():
        _nested_set(cfg, k, v)


# -------------------------- data + scheduler ------------------------------ #

def make_dataset(cfg):
    """Instantiate DailyTSDataset and persist scalers."""
    loc_cfg = cfg.get('model', {}).get('location', {}) or {}
    loc_name = str(loc_cfg.get('name', 'none') or 'none').lower()
    return_coords = bool(loc_name and loc_name not in ('none', ''))
    loc_variant = str(loc_cfg.get('variant', '') or '').lower()
    # Return time index (month or day of year) whenever a variant is specified
    return_month = return_coords and loc_variant in ('monthly', 'doy')

    data_cfg = cfg['data']
    # Optional z targets
    z_feature_indices = data_cfg.get('z_feature_indices', [])
    z_scaler_range = tuple(data_cfg.get('z_scaler_range', data_cfg.get('scaler_range', (-1.0, 1.0))))
    ncar_path = data_cfg.get('ncar_grid_path', None)

    ds = DailyTSDataset(
        root=data_cfg['root'],
        start_date=data_cfg['start_date'],
        end_date=data_cfg['end_date'],
        x_prefix=data_cfg['x_prefix'],
        y_prefix=data_cfg['y_prefix'],
        drop_feature_indices=data_cfg['drop_feature_indices'],
        filter_y_positive=data_cfg['filter_y_positive'],
        remove_nan_rows=data_cfg['remove_nan_rows'],
        scaler_range=tuple(data_cfg['scaler_range']),
        prefetch_files=data_cfg.get('prefetch_files', 4),
        lat_feature_index=data_cfg.get('lat_feature_index', None),
        lon_feature_index=data_cfg.get('lon_feature_index', None),
        return_coords=return_coords,
        return_month=return_month,
        # NEW for physical head
        z_feature_indices=z_feature_indices,
        z_scaler_range=z_scaler_range,
        # NEW: allow choosing target scaler type (minmax or robust)
        scaler_type=data_cfg.get('scaler_type', 'minmax'),
        ncar_path=ncar_path,
        # NEW: propagate variant to dataset for temporal encoding
        time_variant=loc_variant,
    )

    os.makedirs(data_cfg['save_scalers_to'], exist_ok=True)
    joblib.dump(ds.x_scaler, os.path.join(data_cfg['save_scalers_to'], 'X_scaler.joblib'))
    joblib.dump(ds.y_scaler, os.path.join(data_cfg['save_scalers_to'], 'y_scaler.joblib'))
    if getattr(ds, 'return_z', False) and getattr(ds, 'z_scaler', None) is not None:
        joblib.dump(ds.z_scaler, os.path.join(data_cfg['save_scalers_to'], 'z_scaler.joblib'))
    return ds


def make_splits(cfg, ds):
    n = len(ds)
    split = cfg['split']
    name = split['name']
    seed = int(split['seed'])
    test_frac = float(split['test_split'])
    val_frac = float(split['val_split'])

    if name == 'random':
        train_pool, test_idx = random_holdout_indices(n, test_frac, seed)
    elif name == 'spatial':
        train_pool, test_idx = spatial_fold_indices(ds.coords, split['spatial']['n_splits'], split['spatial']['fold_index'], seed)
    else:
        raise ValueError(f"Unknown split.name: {name}")

    if val_frac > 0:
        tr_idx, val_idx = random_holdout_indices(len(train_pool), val_frac, seed + 1)
        train_idx = np.asarray(train_pool)[tr_idx]
        val_idx = np.asarray(train_pool)[val_idx]
    else:
        train_idx = np.asarray(train_pool)
        val_idx = np.array([], dtype=int)

    return train_idx, val_idx, np.asarray(test_idx)


def make_loaders(cfg, ds, train_idx, val_idx, device):
    num_workers = int(cfg['data']['num_workers'])
    train_set = Subset(ds, train_idx)
    train_loader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=(device.type == 'cuda'), drop_last=False)
    if val_idx.size > 0:
        val_set = Subset(ds, val_idx)
        val_loader = DataLoader(val_set, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=(device.type == 'cuda'), drop_last=False)
    else:
        val_loader = None

    first_batch = next(iter(train_loader))
    x0 = first_batch[0] if isinstance(first_batch, (list, tuple)) else first_batch
    input_size = x0.shape[-1]
    return train_loader, val_loader, input_size


def make_scheduler(opt, cfg):
    sc = cfg['train'].get('scheduler', {})
    name = sc.get('name', 'none').lower()
    if name == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(opt, gamma=float(sc.get('gamma', 0.98)))
    if name == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=sc.get('mode', 'min'),
            factor=float(sc.get('factor', 0.5)),
            patience=int(sc.get('patience', 5)),
            threshold=float(sc.get('threshold', 1e-6)),
            threshold_mode=sc.get('threshold_mode', 'abs'),
            cooldown=int(sc.get('cooldown', 0)),
            min_lr=float(sc.get('min_lr', 4e-5)),
        )
    return None


# -------------------------- batch / output utils -------------------------- #

def _split_model_outputs(out):
    # Accept (yhat, attn) or (yhat, zhat, attn)
    if isinstance(out, (list, tuple)):
        if len(out) == 3:
            yhat, zhat, attn = out
        elif len(out) == 2:
            yhat, attn = out
            zhat = None
        else:
            yhat, zhat, attn = out, None, None
    else:
        yhat, zhat, attn = out, None, None
    return yhat, zhat, attn


def _parse_batch(batch, use_location: bool, return_month: bool, expect_z: bool):
    """
    Supports tuples: (x, y), (x,z,y), (x,coords,y), (x,coords,z,y),
                     (x,coords,month,y), (x,coords,month,z,y)
    """
    if not isinstance(batch, (list, tuple)):
        raise RuntimeError("Unexpected batch format")
    x = batch[0]
    coords = month_t = z = y = None
    if not use_location:
        if expect_z:
            if len(batch) != 3:
                raise RuntimeError(f"Expected (x,z,y), got len={len(batch)}")
            _, z, y = batch
        else:
            if len(batch) != 2:
                raise RuntimeError(f"Expected (x,y), got len={len(batch)}")
            _, y = batch
    else:
        if return_month:
            if expect_z:
                if len(batch) != 5:
                    raise RuntimeError(f"Expected (x,coords,month,z,y), got len={len(batch)}")
                x, coords, month_t, z, y = batch
            else:
                if len(batch) != 4:
                    raise RuntimeError(f"Expected (x,coords,month,y), got len={len(batch)}")
                x, coords, month_t, y = batch
        else:
            if expect_z:
                if len(batch) != 4:
                    raise RuntimeError(f"Expected (x,coords,z,y), got len={len(batch)}")
                x, coords, z, y = batch
            else:
                if len(batch) != 3:
                    raise RuntimeError(f"Expected (x,coords,y), got len={len(batch)}")
                x, coords, y = batch
    return x, coords, month_t, z, y


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def _count_params_by_top(model):
    # aggregate by top‑level module name from named_parameters()
    buckets = {}
    for name, p in model.named_parameters():
        top = name.split('.', 1)[0]
        if top not in buckets:
            buckets[top] = {'total': 0, 'trainable': 0}
        n = p.numel()
        buckets[top]['total'] += n
        if p.requires_grad:
            buckets[top]['trainable'] += n
    return buckets


# --------------------------------- eval ----------------------------------- #

def evaluate(
    loader,
    model,
    device,
    loss_fn,
    *,
    use_location: bool = False,
    return_month: bool = False,
    return_z: bool = False,
    loss_weight_physical: float = 0.0,
):
    """Evaluate with component losses: returns (mean_total, mean_pred, mean_phys, y_pred, y_true)."""
    model.eval()
    tot_losses, pred_losses, phys_losses = [], [], []
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for batch in loader:
            x, coords, month_t, z, y = _parse_batch(batch, use_location, return_month, expect_z=return_z)
            x = x.to(device)
            y = y.to(device).squeeze(-1)
            coords = coords.to(device) if coords is not None else None
            month_t = month_t.to(device) if month_t is not None else None
            z = z.to(device).squeeze(-1) if z is not None else None

            out = model(x, coords, month_t) if use_location else model(x)
            yhat, zhat, _ = _split_model_outputs(out)

            pred_loss = loss_fn(yhat, y)
            phys_loss = torch.tensor(0.0, device=device)
            if return_z and (zhat is not None):
                zpred = zhat.squeeze(-1)
                phys_loss = loss_fn(zpred, z)

            loss = pred_loss + loss_weight_physical * phys_loss

            tot_losses.append(loss.item())
            pred_losses.append(pred_loss.item())
            phys_losses.append(float(phys_loss.item() if torch.is_tensor(phys_loss) else phys_loss))
            y_pred_list.append(yhat.detach().cpu().numpy())
            y_true_list.append(y.detach().cpu().numpy())

    mean_total = float(np.mean(tot_losses)) if tot_losses else float('nan')
    mean_pred = float(np.mean(pred_losses)) if pred_losses else float('nan')
    mean_phys = float(np.mean(phys_losses)) if phys_losses else 0.0
    return mean_total, mean_pred, mean_phys, np.concatenate(y_pred_list), np.concatenate(y_true_list)


# --------------------------------- run ------------------------------------ #

def run(cfg_path: str, overrides=None):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if overrides:
        _apply_overrides(cfg, _parse_overrides(overrides))

    cfg = _expand_all_strings(cfg, cfg)
    _expand_cfg_strings(cfg)

    # From‑scratch override to ensure location backbones aren't pretrained/frozen.
    from_scratch = bool(cfg.get('train', {}).get('from_scratch', False))
    if from_scratch:
        loc_section = cfg.get('model', {}).get('location', {}) or {}
        if 'pretrained' in loc_section:
            loc_section['pretrained'] = False
        if 'freeze' in loc_section:
            loc_section['freeze'] = False
        cfg.setdefault('model', {}).setdefault('location', {})
        cfg['model']['location'].update(loc_section)

    # Run ID / dirs
    job_id = os.getenv("SLURM_JOB_ID", "local")
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    keyparts = {k: cfg[k] for k in ('model', 'train', 'data', 'split') if k in cfg}
    h = hashlib.md5(json.dumps(keyparts, sort_keys=True, default=str).encode()).hexdigest()[:8]
    run_id = f"{cfg['project']['name']}_{cfg['data']['start_date']}_{cfg['data']['end_date']}_{cfg['split']['name']}_{ts}_j{job_id}_{h}"

    base_results = cfg['project']['results_dir']
    base_logs = cfg['project']['log_dir']
    base_ckpt = cfg['project']['ckpt_dir']
    run_results_dir = os.path.join(base_results, run_id)
    run_logs_dir = os.path.join(base_logs, run_id)
    run_tb_dir = os.path.join(run_logs_dir, "tb")
    run_ckpt_dir = os.path.join(base_ckpt, run_id)

    cfg['eval']['out_csv_val'] = os.path.join(run_results_dir, "val.csv")
    cfg['eval']['out_csv_test'] = os.path.join(run_results_dir, "test.csv")

    os.makedirs(run_results_dir, exist_ok=True)
    os.makedirs(run_logs_dir, exist_ok=True)
    os.makedirs(run_tb_dir, exist_ok=True)
    os.makedirs(run_ckpt_dir, exist_ok=True)

    # Logging
    log_path = os.path.join(run_logs_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger("loc4pm.train")
    log.info("Run ID: %s", run_id)
    log.info("Job ID: %s", job_id)
    log.info("Saving results to: %s", run_results_dir)

    # Snapshot cfg
    with open(os.path.join(run_results_dir, "config.resolved.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    if overrides:
        with open(os.path.join(run_results_dir, "overrides.json"), "w") as f:
            json.dump(_parse_overrides(overrides), f, indent=2)

    # Device
    req_dev = str(cfg['train']['device']).lower()
    device = torch.device('cuda' if (req_dev == 'cuda' and torch.cuda.is_available()) else 'cpu')

    # Data + splits + loaders
    ds = make_dataset(cfg)
    z_scaler = ds.z_scaler

    train_idx, val_idx, test_idx = make_splits(cfg, ds)
    train_loader, val_loader, input_size = make_loaders(cfg, ds, train_idx, val_idx, device)

    # Physical config
    data_cfg = cfg.get('data', {})
    z_feature_indices = data_cfg.get('z_feature_indices', [])
    physical_out_dim = len(z_feature_indices)
    # Base weight for physical loss (may be dynamically updated)
    loss_weight_physical = float(cfg['train'].get('loss_weight_physical', 0.0))

    # Location / model config
    loc_cfg = cfg.get('model', {}).get('location', {}) or {}
    loc_name = str(loc_cfg.get('name', 'none') or 'none').lower()
    use_location = bool(loc_name and loc_name not in ('none', ''))
    loc_variant = str(loc_cfg.get('variant', '') or '').lower()
    # Return month/day-of-year when location is used and variant is specified
    return_month = use_location and loc_variant in ('monthly', 'doy')
    physical_head_hidden_dim = cfg['model'].get('physical_head_hidden_dim', cfg['model']['attention']['attn_dim'])

    # Build model with optional head dropout
    head_dropout = cfg['model'].get('head_dropout', None)
    if use_location:
        model = BiLSTMAttnLocRegressor(
            input_size=input_size,
            hidden_size=cfg['model']['hidden_size'],
            num_layers=cfg['model']['num_layers'],
            bidirectional=cfg['model']['bidirectional'],
            dropout=cfg['model']['dropout'],
            layer_norm=cfg['model']['layer_norm'],
            attn_type=cfg['model']['attention']['type'],
            attn_dim=cfg['model']['attention']['attn_dim'],
            loc_name=loc_cfg.get('name'),
            loc_variant=loc_cfg.get('variant'),
            loc_emb_dim=loc_cfg.get('emb_dim'),
            loc_pretrained=bool(loc_cfg.get('pretrained', True)),
            loc_freeze=bool(loc_cfg.get('freeze', True)),
            loc_proj_dim=loc_cfg.get('proj_dim'),
            fusion_method=loc_cfg.get('fusion', 'concat'),
            fusion_hidden_dim=loc_cfg.get('head_hidden_dim', cfg['model']['attention']['attn_dim']),
            # NEW: physical head
            physical_head_hidden_dim=physical_head_hidden_dim if physical_out_dim > 0 else None,
            physical_out_dim=physical_out_dim,
            head_dropout=head_dropout,
        ).to(device)
    else:
        model = BiLSTMAttnRegressor(
            input_size=input_size,
            hidden_size=cfg['model']['hidden_size'],
            num_layers=cfg['model']['num_layers'],
            bidirectional=cfg['model']['bidirectional'],
            dropout=cfg['model']['dropout'],
            layer_norm=cfg['model']['layer_norm'],
            attn_type=cfg['model']['attention']['type'],
            attn_dim=cfg['model']['attention']['attn_dim'],
            head_dropout=head_dropout,
        ).to(device)

    # --- Setup banner ---
    log.info("---- RUN SETUP ----")
    log.info("Device: %s | Mixed Precision: %s", device, bool(cfg['train'].get('mixed_precision') and device.type == 'cuda'))
    log.info("Model: BiLSTM+Attn%s", " + LocFusion" if use_location else "")
    if use_location:
        log.info(
            "  LocEnc name=%s, variant=%s, emb_dim=%s, proj_dim=%s",
            loc_cfg.get('name'),
            loc_cfg.get('variant'),
            loc_cfg.get('emb_dim'),
            loc_cfg.get('proj_dim'),
        )
        log.info(
            "  pretrained=%s, freeze=%s, fusion=%s, head_hidden_dim=%s",
            bool(loc_cfg.get('pretrained', False)),
            bool(loc_cfg.get('freeze', False)),
            loc_cfg.get('fusion', 'concat'),
            loc_cfg.get('head_hidden_dim'),
        )
    log.info("Train from scratch: %s", from_scratch)
    log.info("Physical head: %s (out_dim=%d)", "ENABLED" if physical_out_dim > 0 else "DISABLED", physical_out_dim)
    log.info(
        "Loss weight c (physical): %.4f%s",
        loss_weight_physical,
        "  [diagnostic-only]" if loss_weight_physical == 0.0 and physical_out_dim > 0 else "",
    )

    # --- Parameter counts ---
    tot, trn, frz = _count_params(model)
    log.info(
        "Parameters: total=%s | trainable=%s | frozen=%s",
        f"{tot:,}",
        f"{trn:,}",
        f"{frz:,}",
    )
    if trn == 0:
        log.warning("WARNING: 0 trainable parameters detected — check 'pretrained'/'freeze' and from_scratch settings.")

    # Optional breakdown by top‑level submodule (useful to confirm loc encoder is trainable)
    _top = _count_params_by_top(model)
    for k, v in sorted(_top.items(), key=lambda kv: kv[0]):
        log.info("  [%s] trainable=%s / total=%s", k, f"{v['trainable']:,}", f"{v['total']:,}")

    # Optimizer / scheduler / loss
    opt_cfg = cfg['train'].get('optimizer', {})
    opt_name = str(opt_cfg.get('name', 'adam')).lower()
    lr = float(opt_cfg.get('lr', 1e-3))
    default_wd = float(opt_cfg.get('weight_decay', 0.0))
    branch_wd = opt_cfg.get('branch_weight_decay', {}) or {}
    # Build parameter groups with branch‑specific weight decay
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split('.')[0]
        wd = branch_wd.get(top, default_wd)
        param_groups.append({'params': [param], 'weight_decay': wd})
    if opt_name == 'adamw':
        opt = torch.optim.AdamW(param_groups, lr=lr, weight_decay=0.0)
    else:
        # fall back to Adam; use zero global weight decay since per‑group values are set
        opt = torch.optim.Adam(param_groups, lr=lr, weight_decay=0.0)
    sched = make_scheduler(opt, cfg)
    scfg = cfg['train'].get('scheduler', {})
    log.info(
        "Optimizer: %s(lr=%.3e, default_weight_decay=%.1e) with branch overrides: %s",
        opt_name.upper(), lr, default_wd, branch_wd,
    )
    log.info(
        "Scheduler: %s %s",
        scfg.get('name', 'none'),
        (
            f"(gamma={scfg.get('gamma')})"
            if scfg.get('name', '').lower() == 'exponential'
            else (
                f"(mode={scfg.get('mode')}, factor={scfg.get('factor')}, patience={scfg.get('patience')})"
                if scfg.get('name', '').lower() == 'reduce_on_plateau'
                else ""
            )
        ),
    )

    loss_fn = nn.HuberLoss()
    use_amp = bool(cfg['train'].get('mixed_precision') and device.type == 'cuda')
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        amp_autocast = (lambda: torch.amp.autocast('cuda')) if use_amp else (lambda: contextlib.nullcontext())
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # older API
        amp_autocast = (lambda: torch.cuda.amp.autocast()) if use_amp else (lambda: contextlib.nullcontext())

    writer = SummaryWriter(run_tb_dir) if cfg['logging']['tensorboard'] else None

    # CSV logs
    epoch_csv_path = os.path.join(run_results_dir, "train.epochs.csv")
    step_csv_path = os.path.join(run_results_dir, "train.steps.csv")
    save_epoch_csv = True
    save_step_csv = False
    step_every = int(cfg['logging'].get('step_csv_every_n', 50))

    def _append_csv_row(path: str, row: dict) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.exists(path)
        pd.DataFrame([row]).to_csv(path, mode='a', index=False, header=not exists)

    best_val = float('inf')
    best_epoch = -1
    patience = int(cfg['train']['early_stopping']['patience'])

    # ----------------------- NCAR augmentation setup ----------------------- #
    ncar_ratio = float(cfg['train'].get('ncar_random_ratio', 0.0))
    ncar_ratio = max(0.0, ncar_ratio)
    ncar_path = cfg['data'].get('ncar_grid_path', '/home/zhongying/Documents/loc4pm/dataset/ncar_grid_parquet/')
    # Load NCAR grid data via the dataset helper.  If loading fails or
    # sampling yields no points, disable augmentation.
    if ncar_ratio > 0.0 and physical_out_dim > 0:
        try:
            _load_ncar_grid_ds(ncar_path)
            # probe sampling to verify data is loaded
            test_coords, _, _ = _sample_ncar_points_ds(1)
            if test_coords.size == 0:
                log.warning("NCAR random ratio requested but no data loaded from %s", ncar_path)
                ncar_ratio = 0.0
            else:
                log.info("NCAR grid loaded from %s", ncar_path)
        except Exception as e:
            log.warning("Failed to load NCAR grid from %s: %s", ncar_path, e)
            ncar_ratio = 0.0

    # --------------------- dynamic weight update setup -------------------- #
    dwa_cfg = cfg['train'].get('loss_weight_physical_dynamic', {}) or {}
    dwa_enabled = bool(dwa_cfg.get('enabled', False))
    dwa_temp = float(dwa_cfg.get('temperature', 2.0)) if dwa_enabled else None
    dwa_cap = dwa_cfg.get('cap', None)
    prev_pred_loss_epoch = None
    prev_prev_pred_loss_epoch = None
    prev_phys_loss_epoch = None
    prev_prev_phys_loss_epoch = None

    # ----------------------------- training -------------------------------- #
    train_start = time.perf_counter()
    for epoch in range(1, cfg['train']['epochs'] + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_tot_losses, tr_pred_losses, tr_phys_losses = [], [], []
        seen = 0
        # Track how many random NCAR samples are drawn this epoch
        rand_total_samples = 0

        expect_z = (physical_out_dim > 0)
        for i, batch in enumerate(train_loader):
            x, coords, month_t, z, y = _parse_batch(batch, use_location, return_month, expect_z=expect_z)
            bs = x.shape[0]
            seen += bs
            x = x.to(device)
            y = y.to(device).squeeze(-1)
            coords = coords.to(device) if coords is not None else None
            month_t = month_t.to(device) if month_t is not None else None
            z = z.to(device).squeeze(-1) if z is not None else None

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with amp_autocast():
                    out = model(x, coords, month_t) if use_location else model(x)
                    yhat, zhat, _ = _split_model_outputs(out)
                    pred_loss = loss_fn(yhat, y)
                    # Compute physical loss for dataset samples
                    phys_loss_ds = torch.tensor(0.0, device=device)
                    if expect_z and (zhat is not None) and (z is not None):
                        zpred = zhat.squeeze(-1)
                        phys_loss_ds = loss_fn(zpred, z)
                    # Sample additional NCAR points if enabled
                    phys_loss_rand = torch.tensor(0.0, device=device)
                    n_rand = 0
                    if ncar_ratio > 0.0 and expect_z:
                        n_rand = int(round(ncar_ratio * bs))
                        rand_total_samples += n_rand
                        if n_rand > 0:
                            coords_rand_np, month_rand_np, z_rand_np = _sample_ncar_points_ds(n_rand)
                            # In the training loop, after sampling random NCAR points:
                            if z_scaler is not None:
                                # reshape to [N, 1, 1] to match scaler input
                                z_reshaped = z_rand_np.reshape(-1, 1, 1)
                                # apply the scaler’s transform_x method and flatten back
                                z_scaled = z_scaler.transform_x(z_reshaped).reshape(-1)
                                z_rand_np = z_scaled.astype(np.float32)
                            if coords_rand_np.size > 0:
                                coords_rand = torch.tensor(coords_rand_np, dtype=torch.float32, device=device)
                                month_rand = (
                                    torch.tensor(month_rand_np, dtype=torch.int64, device=device)
                                    if return_month
                                    else None
                                )
                                z_rand = torch.tensor(z_rand_np, dtype=torch.float32, device=device)
                                # Obtain location embeddings and z predictions directly via the physical head
                                loc_rand_emb = model.loc_encoder(coords_rand, month_rand) if use_location else None
                                if loc_rand_emb is not None and model.physical_head is not None:
                                    zhat_rand = model.physical_head(loc_rand_emb)
                                    zpred_rand = zhat_rand.squeeze(-1)
                                    phys_loss_rand = loss_fn(zpred_rand, z_rand)
                    # Combine physical losses by normalizing over sample counts
                    if expect_z and (bs + n_rand) > 0:
                        # Convert mean losses back to sums, then average
                        total_phys_loss = phys_loss_ds * bs + phys_loss_rand * max(n_rand, 0)
                        phys_loss = total_phys_loss / float(bs + n_rand)
                    else:
                        phys_loss = torch.tensor(0.0, device=device)
                    loss = pred_loss + loss_weight_physical * phys_loss
                scaler.scale(loss).backward()
                if cfg['train'].get('grad_clip_norm') is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip_norm'])
                scaler.step(opt)
                scaler.update()
            else:
                out = model(x, coords, month_t) if use_location else model(x)
                yhat, zhat, _ = _split_model_outputs(out)
                pred_loss = loss_fn(yhat, y)
                phys_loss_ds = torch.tensor(0.0, device=device)
                if expect_z and (zhat is not None) and (z is not None):
                    zpred = zhat.squeeze(-1)
                    phys_loss_ds = loss_fn(zpred, z)
                phys_loss_rand = torch.tensor(0.0, device=device)
                n_rand = 0
                if ncar_ratio > 0.0 and expect_z:
                    n_rand = int(round(ncar_ratio * bs))
                    rand_total_samples += n_rand
                    if n_rand > 0:
                        coords_rand_np, month_rand_np, z_rand_np = _sample_ncar_points_ds(n_rand)
                        if coords_rand_np.size > 0:
                            coords_rand = torch.tensor(coords_rand_np, dtype=torch.float32, device=device)
                            month_rand = (
                                torch.tensor(month_rand_np, dtype=torch.int64, device=device)
                                if return_month
                                else None
                            )
                            z_rand = torch.tensor(z_rand_np, dtype=torch.float32, device=device)
                            loc_rand_emb = model.loc_encoder(coords_rand, month_rand) if use_location else None
                            if loc_rand_emb is not None and model.physical_head is not None:
                                zhat_rand = model.physical_head(loc_rand_emb)
                                zpred_rand = zhat_rand.squeeze(-1)
                                phys_loss_rand = loss_fn(zpred_rand, z_rand)
                if expect_z and (bs + n_rand) > 0:
                    total_phys_loss = phys_loss_ds * bs + phys_loss_rand * max(n_rand, 0)
                    phys_loss = total_phys_loss / float(bs + n_rand)
                else:
                    phys_loss = torch.tensor(0.0, device=device)
                loss = pred_loss + loss_weight_physical * phys_loss
                loss.backward()
                if cfg['train'].get('grad_clip_norm') is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip_norm'])
                opt.step()

            tr_tot_losses.append(loss.item())
            tr_pred_losses.append(pred_loss.item())
            tr_phys_losses.append(float(phys_loss.item() if torch.is_tensor(phys_loss) else phys_loss))

            # TB step logs
            if writer and (i + 1) % cfg['logging']['log_every_steps'] == 0:
                writer.add_scalar('train/step_total_loss', loss.item(), i + (epoch - 1) * len(train_loader))
                writer.add_scalar('train/step_pred_loss', pred_loss.item(), i + (epoch - 1) * len(train_loader))
                writer.add_scalar('train/step_phys_loss', tr_phys_losses[-1], i + (epoch - 1) * len(train_loader))

            # Optional CSV per‑step
            if save_step_csv and ((i + 1) % step_every == 0):
                _append_csv_row(
                    step_csv_path,
                    {
                        'epoch': int(epoch),
                        'step_in_epoch': int(i + 1),
                        'train_total': float(tr_tot_losses[-1]),
                        'train_pred': float(tr_pred_losses[-1]),
                        'train_phys': float(tr_phys_losses[-1]),
                        'lr': float(opt.param_groups[0]['lr']),
                    },
                )

        # Compute epoch-level statistics
        tr_total = float(np.mean(tr_tot_losses)) if tr_tot_losses else float('nan')
        tr_pred = float(np.mean(tr_pred_losses)) if tr_pred_losses else float('nan')
        tr_phys = float(np.mean(tr_phys_losses)) if tr_phys_losses else 0.0

        ep_sec = time.perf_counter() - ep_t0
        samples_per_sec = seen / ep_sec if ep_sec > 0 else float('nan')
        # Include the number of random NCAR samples drawn in the epoch in the log
        log.info(
            f"Epoch {epoch:03d} | train_total={tr_total:.5f} | pred={tr_pred:.5f} | phys={tr_phys:.5f} | "
            f"lr={opt.param_groups[0]['lr']:.3e} | samples={seen} | rand_samples={rand_total_samples} | "
            f"{samples_per_sec:.1f} samples/s | duration={ep_sec:.1f}s"
        )

        if writer:
            writer.add_scalar('train/epoch_total_loss', tr_total, epoch)
            writer.add_scalar('train/epoch_pred_loss', tr_pred, epoch)
            writer.add_scalar('train/epoch_phys_loss', tr_phys, epoch)
            writer.add_scalar('opt/lr', opt.param_groups[0]['lr'], epoch)

        # ---------- dynamic loss weight update ----------
        if dwa_enabled:
            if prev_pred_loss_epoch is not None and prev_prev_pred_loss_epoch is not None and prev_phys_loss_epoch is not None and prev_prev_phys_loss_epoch is not None:
                ratio_pred = prev_pred_loss_epoch / max(prev_prev_pred_loss_epoch, 1e-8)
                ratio_phys = prev_phys_loss_epoch / max(prev_prev_phys_loss_epoch, 1e-8)
                # Compute weighting ratio using softmax across two tasks
                exp_pred = np.exp(ratio_pred / dwa_temp)
                exp_phys = np.exp(ratio_phys / dwa_temp)
                # weight for physical relative to prediction
                weight_ratio = exp_phys / exp_pred
                # Update the loss weight multiplicatively
                loss_weight_physical = loss_weight_physical * weight_ratio
                # Optionally cap the weight to prevent explosion
                if dwa_cap is not None:
                    loss_weight_physical = float(min(loss_weight_physical, dwa_cap))
                log.info(
                    "Dynamic weight update: ratio_pred=%.4f, ratio_phys=%.4f, weight_ratio=%.4f, new_loss_weight_physical=%.5f",
                    ratio_pred, ratio_phys, weight_ratio, loss_weight_physical,
                )
                if writer:
                    writer.add_scalar('train/loss_weight_physical', loss_weight_physical, epoch)
            # update historical losses for next epoch
            prev_prev_pred_loss_epoch = prev_pred_loss_epoch
            prev_pred_loss_epoch = tr_pred
            prev_prev_phys_loss_epoch = prev_phys_loss_epoch
            prev_phys_loss_epoch = tr_phys

        # --------------------------- validation ---------------------------- #
        if val_loader is not None:
            val_total, val_pred, val_phys, y_pred_val, y_true_val = evaluate(
                val_loader,
                model,
                device,
                nn.HuberLoss(),
                use_location=use_location,
                return_month=return_month,
                return_z=(physical_out_dim > 0),
                loss_weight_physical=loss_weight_physical,
            )
            log.info(f"Epoch {epoch:03d} | val_total={val_total:.5f} | pred={val_pred:.5f} | phys={val_phys:.5f}")

            if save_epoch_csv:
                _append_csv_row(
                    epoch_csv_path,
                    {
                        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                        'epoch': int(epoch),
                        'train_total': float(tr_total),
                        'train_pred': float(tr_pred),
                        'train_phys': float(tr_phys),
                        'val_total': float(val_total),
                        'val_pred': float(val_pred),
                        'val_phys': float(val_phys),
                        'lr': float(opt.param_groups[0]['lr']),
                        'epoch_seconds': float(ep_sec),
                        'samples_in_epoch': int(seen),
                        'samples_per_sec': float(samples_per_sec),
                        'loss_weight_physical': float(loss_weight_physical),
                    },
                )
            if writer:
                writer.add_scalar('val/epoch_total_loss', val_total, epoch)
                writer.add_scalar('val/epoch_pred_loss', val_pred, epoch)
                writer.add_scalar('val/epoch_phys_loss', val_phys, epoch)
            if val_total < best_val:
                best_val = val_total
                best_epoch = epoch
                os.makedirs(cfg['project']['ckpt_dir'], exist_ok=True)
                torch.save(
                    {'model': model.state_dict(), 'cfg': cfg, 'epoch': epoch},
                    os.path.join(run_ckpt_dir, 'best.pt'),
                )
            # Scheduler step
            if sched is not None:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(val_total)
                else:
                    sched.step()
            # Early stopping
            if epoch - best_epoch >= int(cfg['train']['early_stopping']['patience']):
                log.info(f"Early stopping at epoch {epoch} (best {best_epoch}, val {best_val:.4f})")
                break
        else:
            # No val loader; still step scheduler if non‑plateau
            if save_epoch_csv:
                _append_csv_row(
                    epoch_csv_path,
                    {
                        'epoch': int(epoch),
                        'train_total': float(tr_total),
                        'train_pred': float(tr_pred),
                        'train_phys': float(tr_phys),
                        'val_total': float('nan'),
                        'val_pred': float('nan'),
                        'val_phys': float('nan'),
                        'lr': float(opt.param_groups[0]['lr']),
                        'loss_weight_physical': float(loss_weight_physical),
                    },
                )
            if sched is not None and not isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step()

    # ------------------------------ export --------------------------------- #
    if val_loader is not None and cfg['eval']['save_val_predictions']:
        out_csv = cfg['eval']['out_csv_val']
        mets_json = os.path.splitext(out_csv)[0] + '.metrics.json'
        ysc_path = os.path.join(cfg['data']['save_scalers_to'], 'y_scaler.joblib')
        _vt, _vp, _vph, y_pred_val, y_true_val = evaluate(
            val_loader,
            model,
            device,
            nn.HuberLoss(),
            use_location=use_location,
            return_month=return_month,
            return_z=(physical_out_dim > 0),
            loss_weight_physical=loss_weight_physical,
        )
        mets = save_predictions_and_metrics(y_true_val, y_pred_val, ysc_path, out_csv, mets_json)
        logging.info("Validation metrics:\n%s", json.dumps(mets, indent=2))

    if test_idx.size > 0:
        test_loader = DataLoader(
            Subset(ds, test_idx),
            batch_size=cfg['train']['batch_size'],
            shuffle=False,
            num_workers=int(cfg['data']['num_workers']),
            pin_memory=(device.type == 'cuda'),
            drop_last=False,
        )
        _tt, _tp, _tph, y_pred_t, y_true_t = evaluate(
            test_loader,
            model,
            device,
            nn.HuberLoss(),
            use_location=use_location,
            return_month=return_month,
            return_z=(physical_out_dim > 0),
            loss_weight_physical=loss_weight_physical,
        )
        out_csv_t = cfg['eval']['out_csv_test']
        mets_json_t = os.path.splitext(out_csv_t)[0] + '.metrics.json'
        ysc_path = os.path.join(cfg['data']['save_scalers_to'], 'y_scaler.joblib')
        tmets = save_predictions_and_metrics(y_true_t, y_pred_t, ysc_path, out_csv_t, mets_json_t)
        logging.info("Test metrics:\n%s", json.dumps(tmets, indent=2))

    if writer:
        writer.close()

    total_sec = time.perf_counter() - (train_start)
    log.info("Training finished in %.1fs (%.1f minutes)", total_sec, total_sec / 60.0)


# --------------------------------- cli ------------------------------------ #

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    ap.add_argument('overrides', nargs='*')
    args = ap.parse_args()
    run(args.config, overrides=args.overrides)