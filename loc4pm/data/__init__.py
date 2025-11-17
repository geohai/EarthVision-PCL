"""Data loading utilities for LOC4PM.

This package contains dataset classes and helper functions for loading
time‑series features, observation targets, and additional physical
simulation targets.  It also includes utilities for loading and
randomly sampling from NCAR grid parquet files used to augment
training batches with synthetic coordinates and PM2.5 values.

"""

from .daily_dataset import DailyTSDataset