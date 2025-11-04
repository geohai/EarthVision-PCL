import os, yaml, json, joblib, time, datetime, hashlib, logging, re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from .data.daily_dataset import DailyTSDataset
from .models.bilstm_attn import BiLSTMAttnRegressor
from .utils.export import save_predictions_and_metrics
from .utils.splits import random_holdout_indices, spatial_fold_indices
import contextlib

_PAT = re.compile(r"\$\{([^}]+)\}")

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
                return m.group(0)  # leave as-is if missing
        return _PAT.sub(repl, obj)
    return obj

def _expand_placeholders(s: str, cfg: dict) -> str:
  if not isinstance(s, str):
    return s
  pat = re.compile(r"\\$\\{([^}]+)\\}")
  def repl(m):
    key = m.group(1)
    try:
      return str(_deep_get(cfg, key))
    except Exception:
      return m.group(0)  # leave as-is if missing
  return pat.sub(repl, s)

def _expand_cfg_strings(cfg: dict):
  if 'logging' in cfg and 'run_name' in cfg['logging']:
    cfg['logging']['run_name'] = _expand_placeholders(cfg['logging']['run_name'], cfg)
  # expand any optional CSV log paths
  if 'logging' in cfg:
    for k in ('epoch_csv','step_csv'):
      if k in cfg['logging']:
        cfg['logging'][k] = _expand_placeholders(cfg['logging'][k], cfg)
  if 'eval' in cfg:
    for k in ('out_csv_val','out_csv_test'):
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
        kv[pairs[i]] = cast(pairs[i+1])
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


def make_dataset(cfg):
    ds = DailyTSDataset(
        root = cfg['data']['root'],
        start_date = cfg['data']['start_date'],
        end_date   = cfg['data']['end_date'],
        x_prefix   = cfg['data']['x_prefix'],
        y_prefix   = cfg['data']['y_prefix'],
        drop_feature_indices = cfg['data']['drop_feature_indices'],
        filter_y_positive    = cfg['data']['filter_y_positive'],
        remove_nan_rows      = cfg['data']['remove_nan_rows'],
        scaler_range         = tuple(cfg['data']['scaler_range']),
        prefetch_files       = cfg['data'].get('prefetch_files', 4),
        lat_feature_index    = cfg['data'].get('lat_feature_index', None),
        lon_feature_index    = cfg['data'].get('lon_feature_index', None),
    )

    # persist scalers for later inverse-transform
    os.makedirs(cfg['data']['save_scalers_to'], exist_ok=True)
    joblib.dump(ds.x_scaler, os.path.join(cfg['data']['save_scalers_to'], 'X_scaler.joblib'))
    joblib.dump(ds.y_scaler, os.path.join(cfg['data']['save_scalers_to'], 'y_scaler.joblib'))
    return ds


def make_splits(cfg, ds):
    n = len(ds)
    split = cfg['split']
    name = split['name']
    seed = int(split['seed'])
    test_frac = float(split['test_split'])
    val_frac  = float(split['val_split'])

    # first, carve out TEST indices
    if name == 'random':
        train_pool, test_idx = random_holdout_indices(n, test_frac, seed)
    elif name == 'spatial':
        train_pool, test_idx = spatial_fold_indices(ds.coords, split['spatial']['n_splits'],
                                                    split['spatial']['fold_index'], seed)
    else:
        raise ValueError(f"Unknown split.name: {name}")

    # then carve out VAL from the remaining train_pool deterministically
    if val_frac > 0:
        tr_idx, val_idx = random_holdout_indices(len(train_pool), val_frac, seed+1)
        train_idx = np.asarray(train_pool)[tr_idx]
        val_idx   = np.asarray(train_pool)[val_idx]
    else:
        train_idx = np.asarray(train_pool)
        val_idx   = np.array([], dtype=int)

    return train_idx, val_idx, np.asarray(test_idx)


def make_loaders(cfg, ds, train_idx, val_idx, device):
    num_workers = int(cfg['data']['num_workers'])
    train_set = Subset(ds, train_idx)
    train_loader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type=='cuda'), drop_last=False)

    if val_idx.size > 0:
        val_set = Subset(ds, val_idx)
        val_loader = DataLoader(val_set, batch_size=cfg['train']['batch_size'], shuffle=False,
                                num_workers=num_workers, pin_memory=(device.type=='cuda'), drop_last=False)
    else:
        val_loader = None

    # infer input size from a sample
    x0, _ = next(iter(train_loader))
    input_size = x0.shape[-1]
    return train_loader, val_loader, input_size


def evaluate(loader, model, device, loss_fn):
    model.eval()
    losses, y_pred, y_true = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).squeeze(-1)
            yhat, _ = model(x, mask=None)
            loss = loss_fn(yhat, y).item()
            losses.append(loss)
            y_pred.append(yhat.detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())
    return float(np.mean(losses)), np.concatenate(y_pred), np.concatenate(y_true)


def run(cfg_path: str, overrides=None):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if overrides:
        _apply_overrides(cfg, _parse_overrides(overrides))

    cfg = _expand_all_strings(cfg, cfg)

    # ---- Build a robust run_id ----
    job_id = os.getenv("SLURM_JOB_ID", "local")
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # hash the key parts of cfg so we can tell hyperparam variants apart
    keyparts = {k: cfg[k] for k in ('model', 'train', 'data', 'split') if k in cfg}
    h = hashlib.md5(json.dumps(keyparts, sort_keys=True).encode()).hexdigest()[:8]
    run_id = f"{cfg['project']['name']}_{cfg['data']['start_date']}_{cfg['data']['end_date']}_{cfg['split']['name']}_{ts}_j{job_id}_{h}"

    # ---- Derive run-scoped directories ----
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

    # ---- Configure a human-readable text logger with timestamps ----
    log_path = os.path.join(run_logs_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    log = logging.getLogger("loc4pm.train")
    log.info("Run ID: %s", run_id)
    log.info("Job ID: %s", job_id)
    log.info("Saving results to: %s", run_results_dir)

    # ---- Snapshot the resolved config for reproducibility ----
    with open(os.path.join(run_results_dir, "config.resolved.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    if overrides:
        with open(os.path.join(run_results_dir, "overrides.json"), "w") as f:
            json.dump(_parse_overrides(overrides), f, indent=2)

    req_dev = str(cfg['train']['device']).lower()
    device = torch.device('cuda' if (req_dev == 'cuda' and torch.cuda.is_available()) else 'cpu')

    ds = make_dataset(cfg)
    train_idx, val_idx, test_idx = make_splits(cfg, ds)
    train_loader, val_loader, input_size = make_loaders(cfg, ds, train_idx, val_idx, device)

    model = BiLSTMAttnRegressor(
        input_size=input_size,
        hidden_size=cfg['model']['hidden_size'],
        num_layers=cfg['model']['num_layers'],
        bidirectional=cfg['model']['bidirectional'],
        dropout=cfg['model']['dropout'],
        layer_norm=cfg['model']['layer_norm'],
        attn_type=cfg['model']['attention']['type'],
        attn_dim=cfg['model']['attention']['attn_dim']
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg['train']['optimizer']['lr'],
                           weight_decay=cfg['train']['optimizer']['weight_decay'])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=float(cfg['train']['scheduler']['gamma'])) \
            if cfg['train']['scheduler']['name'] == 'exponential' else None

    loss_fn = nn.HuberLoss()
    use_amp = bool(cfg['train']['mixed_precision'] and device.type=='cuda')
    # Create GradScaler with compatibility across torch versions
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        amp_autocast = (lambda: torch.amp.autocast('cuda')) if use_amp else (lambda: contextlib.nullcontext())
    except TypeError:
        # Older API (or without device arg)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        amp_autocast = (lambda: torch.cuda.amp.autocast()) if use_amp else (lambda: contextlib.nullcontext())

    writer = SummaryWriter(run_tb_dir) if cfg['logging']['tensorboard'] else None

    # CSV logging setup
    # CSV paths inside the run directory
    epoch_csv_path = os.path.join(run_results_dir, "train.epochs.csv")
    step_csv_path = os.path.join(run_results_dir, "train.steps.csv")  # if you use it
    save_epoch_csv = True
    save_step_csv = False
    step_every = int(cfg['logging'].get('step_csv_every_n', 50))

    def _append_csv_row(path: str, row: dict):
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.exists(path)
        pd.DataFrame([row]).to_csv(path, mode='a', index=False, header=not exists)

    best_val = float('inf')
    best_epoch = -1
    patience = int(cfg['train']['early_stopping']['patience'])
    global_step = 0

    train_start = time.perf_counter()
    total_seen = 0
    for epoch in range(1, cfg['train']['epochs']+1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_losses = []
        seen = 0
        for i, (x, y) in enumerate(train_loader):
            bs = x.shape[0]
            seen += bs
            total_seen += bs
            x = x.to(device)
            y = y.to(device).squeeze(-1)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with amp_autocast():
                    yhat, _ = model(x, mask=None)
                    loss = loss_fn(yhat, y)
                scaler.scale(loss).backward()
                if cfg['train']['grad_clip_norm'] is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip_norm'])
                scaler.step(opt)
                scaler.update()
            else:
                yhat, _ = model(x, mask=None)
                loss = loss_fn(yhat, y)
                loss.backward()
                if cfg['train']['grad_clip_norm'] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip_norm'])
                opt.step()
            tr_losses.append(loss.item())
            global_step += 1
            if writer and (i+1) % cfg['logging']['log_every_steps'] == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
            # optional step CSV log
            if save_step_csv and step_csv_path and ((i+1) % step_every == 0):
                _append_csv_row(step_csv_path, {
                    'global_step': int(global_step),
                    'epoch': int(epoch),
                    'step_in_epoch': int(i+1),
                    'train_loss': float(loss.item()),
                    'lr': float(opt.param_groups[0]['lr']),
                })

        tr_loss = float(np.mean(tr_losses))
        ep_sec = time.perf_counter() - ep_t0
        samples_per_sec = seen / ep_sec if ep_sec > 0 else float('nan')
        log.info(f"Epoch {epoch:03d} | train_loss={tr_loss:.5f} | "
                 f"lr={opt.param_groups[0]['lr']:.3e} | "
                 f"samples={seen} | {samples_per_sec:.1f} samples/s | "
                 f"duration={ep_sec:.1f}s")

        # validate (if we have a val split)
        if val_loader is not None:
            val_loss, y_pred_val, y_true_val = evaluate(val_loader, model, device, loss_fn)
            log.info(f"Epoch {epoch:03d} | val_loss={val_loss:.5f}")
            if save_epoch_csv:
                _append_csv_row(epoch_csv_path, {
                    'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                    'epoch': int(epoch),
                    'train_loss': float(tr_loss),
                    'val_loss': float(val_loss),
                    'lr': float(opt.param_groups[0]['lr']),
                    'epoch_seconds': float(ep_sec),
                    'samples_in_epoch': int(seen),
                    'samples_per_sec': float(samples_per_sec),
                })
            if writer:
                writer.add_scalar('train/epoch_loss', tr_loss, epoch)
                writer.add_scalar('val/loss', val_loss, epoch)
                if sched is not None:
                    writer.add_scalar('opt/lr', opt.param_groups[0]['lr'], epoch)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                os.makedirs(cfg['project']['ckpt_dir'], exist_ok=True)
                torch.save({'model': model.state_dict(), 'cfg': cfg, 'epoch': epoch},
                           os.path.join(run_ckpt_dir, 'best.pt'))
            if sched is not None:
                sched.step()
            if epoch - best_epoch >= patience:
                log.info(f"Early stopping at epoch {epoch} (best {best_epoch}, val {best_val:.4f})")
                break
        else:
            # no val loader; still save periodic checkpoints if desired
            if save_epoch_csv and epoch_csv_path:
                _append_csv_row(epoch_csv_path, {
                    'epoch': int(epoch),
                    'train_loss': float(tr_loss),
                    'val_loss': float('nan'),
                    'lr': float(opt.param_groups[0]['lr']),
                })
            if sched is not None:
                sched.step()

    # export VAL metrics if we had a val split
    if val_loader is not None and cfg['eval']['save_val_predictions']:
        out_csv = cfg['eval']['out_csv_val']
        mets_json = os.path.splitext(out_csv)[0] + '.metrics.json'
        ysc_path = os.path.join(cfg['data']['save_scalers_to'], 'y_scaler.joblib')
        _val_loss, y_pred_val, y_true_val = evaluate(val_loader, model, device, loss_fn)
        mets = save_predictions_and_metrics(y_true_val, y_pred_val, ysc_path, out_csv, mets_json)
        print('Validation metrics:', json.dumps(mets, indent=2))
        log.info("Validation metrics:\n%s", json.dumps(mets, indent=2))

    # export TEST metrics if test split > 0
    if test_idx.size > 0:
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=cfg['train']['batch_size'], shuffle=False,
                                 num_workers=int(cfg['data']['num_workers']), pin_memory=True, drop_last=False)
        _t_loss, y_pred_t, y_true_t = evaluate(test_loader, model, device, loss_fn)
        out_csv_t = cfg['eval']['out_csv_test']
        mets_json_t = os.path.splitext(out_csv_t)[0] + '.metrics.json'
        ysc_path = os.path.join(cfg['data']['save_scalers_to'], 'y_scaler.joblib')
        tmets = save_predictions_and_metrics(y_true_t, y_pred_t, ysc_path, out_csv_t, mets_json_t)
        print('Test metrics:', json.dumps(tmets, indent=2))
        log.info("Test metrics:\n%s", json.dumps(tmets, indent=2))

    if writer:
        writer.close()

    total_sec = time.perf_counter() - train_start
    log.info("Training finished in %.1fs (%.1f minutes)", total_sec, total_sec / 60.0)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/default.yaml')
    ap.add_argument('overrides', nargs='*')
    args = ap.parse_args()
    run(args.config, overrides=args.overrides)