from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_provider.scaler import StandardScaler


def read_multivariate_csv(csv_path, time_column="date"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. "
            "Check data.root_path and data.files in configs/default.yaml."
        )

    frame = pd.read_csv(csv_path)
    if time_column not in frame.columns:
        raise ValueError(
            f"Timestamp column '{time_column}' not found in {csv_path}. "
            f"Available columns: {list(frame.columns)[:12]}"
        )

    feature_columns = []
    numeric_arrays = []
    for col in frame.columns:
        if col == time_column or str(col).lower().startswith("unnamed"):
            continue
        converted = pd.to_numeric(frame[col], errors="coerce")
        # Ignore genuinely non-numeric metadata columns.
        if converted.notna().sum() == 0:
            continue
        if converted.isna().any():
            raise ValueError(
                f"Numeric feature '{col}' contains missing/non-numeric values. "
                "Clean or impute the CSV explicitly before training."
            )
        feature_columns.append(col)
        numeric_arrays.append(converted.to_numpy(dtype=np.float32))

    if not numeric_arrays:
        raise ValueError(f"No numeric feature columns found in {csv_path}.")

    values = np.stack(numeric_arrays, axis=1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values found in {csv_path}.")

    return values, feature_columns


class TimeSeriesWindowDataset(Dataset):
    """
    Returns:
        x: [L,C] historical context
        y: [H,C] future target

    Target windows are strictly contained in their own chronological split.
    """

    def __init__(self, scaled_values, seq_len, pred_len, split, split_bounds):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        self.values = np.asarray(scaled_values, dtype=np.float32)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.split = split

        split_start, split_end = split_bounds[split]
        first_target = max(self.seq_len, split_start)
        last_target = split_end - self.pred_len

        if last_target < first_target:
            raise ValueError(
                f"Not enough rows for {split}: bounds={split_bounds[split]}, "
                f"L={seq_len}, H={pred_len}."
            )

        self.target_starts = np.arange(
            first_target, last_target + 1, dtype=np.int64
        )

    def __len__(self):
        return len(self.target_starts)

    def __getitem__(self, index):
        target_start = int(self.target_starts[index])
        x_start = target_start - self.seq_len
        x_end = target_start
        y_end = target_start + self.pred_len

        x = self.values[x_start:x_end]
        y = self.values[x_end:y_end]

        if x.shape[0] != self.seq_len or y.shape[0] != self.pred_len:
            raise RuntimeError("Window construction produced an invalid shape.")

        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
